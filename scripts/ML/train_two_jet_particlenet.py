#!/usr/bin/env python3
"""Train and evaluate TwoJetParticleNet on one or more Parquet shards.

Examples
--------
Use already prepared train/validation/test files (recommended)::

    python train_two_jet_particlenet_optimised_v2.py \
        --train-parquet 'cache/analysis_dataset/*train*.parquet' \
        --val-parquet 'cache/analysis_dataset/*val*.parquet' \
        --test-parquet 'cache/analysis_dataset/*test*.parquet'

Or make a new stratified event split from a common pool::

python train_two_jet_particlenet_optimised_v2.py \
    --parquet 'cache/*.parquet' \
    --output-dir results/particlenet \
    --epochs 30 --batch-size 64 --max-constituents 100
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import Sampler, Subset

from two_jet_particlenet import (
    TwoJetBatch,
    TwoJetParquetDataset,
    TwoJetParticleNet,
    infer_particle_type_vocabulary,
    make_loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a two-jet, event-level ParticleNet classifier."
    )
    parser.add_argument(
        "--parquet",
        nargs="+",
        help=("Parquet paths or quoted globs to split internally. Mutually "
              "exclusive with the three predefined-split arguments."),
    )
    parser.add_argument(
        "--train-parquet",
        nargs="+",
        help="Training Parquet paths or quoted globs.",
    )
    parser.add_argument(
        "--val-parquet",
        nargs="+",
        help="Validation Parquet paths or quoted globs.",
    )
    parser.add_argument(
        "--test-parquet",
        nargs="+",
        help="Test Parquet paths or quoted globs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/particlenet"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-constituents", type=int, default=100)
    parser.add_argument(
        "--shard-cache-size",
        type=int,
        default=2,
        help="Complete Parquet shards cached per DataLoader worker (default: 2).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched by each worker (used only when workers > 0).",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Pin host batches; defaults to enabled on CUDA.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DataLoader workers alive between passes/epochs.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatic mixed precision; defaults to enabled on CUDA.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--use-particle-type",
        action="store_true",
        help="Embed constituent_type codes (vocabulary is scanned from input shards).",
    )
    parser.add_argument(
        "--particle-types",
        nargs="+",
        type=int,
        default=None,
        metavar="CODE",
        help=("Explicit training-only particle-type vocabulary. Avoids a full "
              "constituent_type scan; implies --use-particle-type."),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--limit-events",
        type=int,
        default=None,
        help=("Optional class-balanced debugging limit. With predefined "
              "splits, this limit is applied separately to each split."),
    )
    return parser.parse_args()


def expand_paths(patterns: Sequence[str]) -> list[Path]:
    """Expand paths and simple globs without relying on shell expansion."""

    import glob

    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            candidate = Path(pattern)
            if candidate.is_file():
                paths.append(candidate)
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError(f"No Parquet files matched: {list(patterns)}")
    return unique


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def stratified_split(
    labels: np.ndarray,
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split each class independently, preserving class proportions."""

    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val/test fractions must be non-negative and sum to < 1")
    rng = np.random.default_rng(seed)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_test = round(test_fraction * len(indices))
        n_val = round(val_fraction * len(indices))
        test.extend(indices[:n_test].tolist())
        val.extend(indices[n_test : n_test + n_val].tolist())
        train.extend(indices[n_test + n_val :].tolist())
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    if not train or not val or not test:
        raise ValueError("Every split must contain events; increase the dataset size")
    return train, val, test


def read_labels(paths: Sequence[Path]) -> np.ndarray:
    import awkward as ak

    chunks = [np.asarray(ak.to_numpy(ak.from_parquet(path, columns=["label"])["label"]))
              for path in paths]
    labels = np.concatenate(chunks).astype(np.int8, copy=False)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("The label column must contain only 0 and 1")
    return labels


def select_indices(labels: np.ndarray, limit: int | None, seed: int) -> list[int]:
    """Select a reproducible, approximately class-balanced debugging sample."""

    if limit is None or limit >= len(labels):
        return list(range(len(labels)))
    if limit < 6:
        raise ValueError("--limit-events must be at least 6")
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    selected: list[int] = []
    remaining = limit
    for position, label in enumerate(classes):
        candidates = np.flatnonzero(labels == label)
        rng.shuffle(candidates)
        take = remaining if position == len(classes) - 1 else min(
            len(candidates), limit // len(classes)
        )
        selected.extend(candidates[:take].tolist())
        remaining -= take
    if remaining:
        unselected = np.setdiff1d(np.arange(len(labels)), np.asarray(selected))
        rng.shuffle(unselected)
        selected.extend(unselected[:remaining].tolist())
    rng.shuffle(selected)
    return selected


def class_weight(labels: np.ndarray, train_indices: Sequence[int]) -> float:
    train_labels = labels[np.asarray(train_indices)]
    positives = int(train_labels.sum())
    negatives = len(train_labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Training split must contain both signal and background")
    return negatives / positives


class ShardLocalBatchSampler(Sampler[list[int]]):
    """Shuffle training data while keeping most batches within one shard.

    The yielded values index a ``Subset``.  Full batches are produced from one
    Parquet shard; only shard remainders are combined.  This prevents global
    random sampling from repeatedly evicting and re-reading complete shards.
    """

    def __init__(
        self,
        dataset: TwoJetParquetDataset,
        subset_indices: Sequence[int],
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        groups: dict[int, list[int]] = {}
        for subset_position, dataset_index in enumerate(subset_indices):
            groups.setdefault(dataset.shard_index(dataset_index), []).append(
                subset_position
            )
        self.groups = tuple(np.asarray(group, dtype=np.int64) for group in groups.values())

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        group_order = rng.permutation(len(self.groups))
        remainder: list[int] = []
        for group_id in group_order:
            positions = self.groups[int(group_id)].copy()
            rng.shuffle(positions)
            full_stop = (len(positions) // self.batch_size) * self.batch_size
            for start in range(0, full_stop, self.batch_size):
                yield positions[start : start + self.batch_size].tolist()
            remainder.extend(positions[full_stop:].tolist())
            while len(remainder) >= self.batch_size:
                yield remainder[: self.batch_size]
                del remainder[: self.batch_size]
        if remainder:
            yield remainder

    def __len__(self) -> int:
        full = sum(len(group) // self.batch_size for group in self.groups)
        remainder = sum(len(group) % self.batch_size for group in self.groups)
        return full + (remainder + self.batch_size - 1) // self.batch_size


def run_epoch(
    model: nn.Module,
    loader: Iterable[TwoJetBatch],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    non_blocking: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct = torch.zeros((), device=device, dtype=torch.int64)
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device, non_blocking=non_blocking)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                logits = model(batch)
                loss = criterion(logits, batch.y)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            n = batch.y.numel()
            loss_sum += loss.detach().double() * n
            correct += ((logits >= 0) == (batch.y >= 0.5)).sum()
            count += n
    if not count:
        raise ValueError("DataLoader produced no events")
    return float((loss_sum / count).item()), float((correct / count).item())


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: Iterable[TwoJetBatch],
    device: torch.device,
    *,
    amp: bool = False,
    non_blocking: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=non_blocking)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            probabilities = torch.sigmoid(model(batch))
        labels.append(batch.y.cpu().numpy())
        scores.append(probabilities.cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    axes[0].set(xlabel="Epoch", ylabel="BCE loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)


def save_test_results(
    indices: Sequence[int], labels: np.ndarray, scores: np.ndarray, output_dir: Path
) -> dict[str, object]:
    predictions = (scores >= 0.5).astype(np.int8)
    auc = float(roc_auc_score(labels, scores))
    accuracy = float(accuracy_score(labels, predictions))
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics: dict[str, object] = {
        "test_events": int(len(labels)),
        "test_accuracy": accuracy,
        "test_roc_auc": auc,
        "confusion_matrix": matrix.tolist(),
    }
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset_index", "label", "signal_probability", "prediction"))
        writer.writerows(zip(indices, labels.astype(int), scores, predictions))

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc:.4f}")
    axis.plot((0, 1), (0, 1), "--", color="0.5")
    axis.set(xlabel="Background efficiency", ylabel="Signal efficiency", xlim=(0, 1), ylim=(0, 1))
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)
    return metrics


def main() -> None:
    args = parse_args()
    setup_start = time.perf_counter()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.num_workers < 0 or args.shard_cache_size < 1:
        raise ValueError("--num-workers must be non-negative and cache size positive")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    predefined_flags = (args.train_parquet, args.val_parquet, args.test_parquet)
    has_predefined = any(value is not None for value in predefined_flags)
    if args.parquet is not None and has_predefined:
        raise ValueError(
            "Use either --parquet or --train-parquet/--val-parquet/--test-parquet, not both"
        )
    if args.parquet is None and not has_predefined:
        raise ValueError(
            "Provide --parquet, or all of --train-parquet, --val-parquet and --test-parquet"
        )
    if has_predefined and not all(value is not None for value in predefined_flags):
        raise ValueError(
            "Predefined mode requires all of --train-parquet, --val-parquet and --test-parquet"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    amp_enabled = device.type == "cuda" if args.amp is None else args.amp
    if amp_enabled and device.type != "cuda":
        raise ValueError("--amp is currently supported only with CUDA")
    pin_memory = device.type == "cuda" if args.pin_memory is None else args.pin_memory
    non_blocking = bool(pin_memory and device.type == "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    def particle_types(paths: Sequence[Path]) -> tuple[int, ...]:
        if args.particle_types is not None:
            return tuple(sorted(set(args.particle_types)))
        if args.use_particle_type:
            return infer_particle_type_vocabulary(paths)
        return ()

    if has_predefined:
        train_paths = expand_paths(args.train_parquet)
        val_paths = expand_paths(args.val_parquet)
        test_paths = expand_paths(args.test_parquet)
        split_path_sets = {
            "train": set(train_paths),
            "validation": set(val_paths),
            "test": set(test_paths),
        }
        for left, right in (("train", "validation"),
                            ("train", "test"),
                            ("validation", "test")):
            overlap = split_path_sets[left] & split_path_sets[right]
            if overlap:
                names = ", ".join(str(path) for path in sorted(overlap))
                raise ValueError(
                    f"The {left} and {right} patterns matched the same file(s): {names}"
                )
        train_labels = read_labels(train_paths)
        val_labels = read_labels(val_paths)
        test_labels_all = read_labels(test_paths)
        train_indices = select_indices(train_labels, args.limit_events, args.seed)
        val_indices = select_indices(val_labels, args.limit_events, args.seed + 1)
        test_indices = select_indices(test_labels_all, args.limit_events, args.seed + 2)
        type_vocabulary = particle_types(train_paths)
        dataset_args = dict(
            type_vocabulary=type_vocabulary,
            max_constituents=args.max_constituents,
            shard_cache_size=args.shard_cache_size,
        )
        train_dataset = TwoJetParquetDataset(train_paths, **dataset_args)
        val_dataset = TwoJetParquetDataset(val_paths, **dataset_args)
        test_dataset = TwoJetParquetDataset(test_paths, **dataset_args)
    else:
        paths = expand_paths(args.parquet)
        all_labels = read_labels(paths)
        usable_indices = select_indices(all_labels, args.limit_events, args.seed)
        labels = all_labels[np.asarray(usable_indices)]
        type_vocabulary = particle_types(paths)
        dataset = TwoJetParquetDataset(
            paths,
            type_vocabulary=type_vocabulary,
            max_constituents=args.max_constituents,
            shard_cache_size=args.shard_cache_size,
        )
        train_rel, val_rel, test_rel = stratified_split(
            labels,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        train_indices = [usable_indices[i] for i in train_rel]
        val_indices = [usable_indices[i] for i in val_rel]
        test_indices = [usable_indices[i] for i in test_rel]
        train_labels = all_labels
        train_dataset = val_dataset = test_dataset = dataset

    # Sequential evaluation order is both shard-local and necessary for the
    # dataset indices written alongside predictions.
    val_indices = sorted(val_indices)
    test_indices = sorted(test_indices)
    mean, std = train_dataset.high_level_statistics(train_indices)

    common_loader_args = dict(
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
    )
    train_subset = Subset(train_dataset, train_indices)
    train_batch_sampler = ShardLocalBatchSampler(
        train_dataset, train_indices, args.batch_size, args.seed
    )
    train_loader = make_loader(
        train_subset,
        batch_sampler=train_batch_sampler,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        **common_loader_args,
    )
    val_loader = make_loader(
        Subset(val_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        persistent_workers=False,
        **common_loader_args,
    )
    test_loader = make_loader(
        Subset(test_dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        persistent_workers=False,
        **common_loader_args,
    )

    model = TwoJetParticleNet(
        high_level_dim=train_dataset.high_level_dim,
        num_particle_types=train_dataset.num_particle_types,
        high_level_mean=mean,
        high_level_std=std,
    ).to(device)
    positive_weight = class_weight(train_labels, train_indices)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    print(f"Device: {device}")
    print(f"Events: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    split_mode = "predefined files" if has_predefined else "internal stratified split"
    print(f"Split mode: {split_mode}")
    print(
        f"High-level features ({train_dataset.high_level_dim}): "
        f"{train_dataset.high_level_names}"
    )
    print(f"Particle-type vocabulary size: {train_dataset.num_particle_types}")
    if type_vocabulary:
        print(f"Particle-type codes: {type_vocabulary}")
    print(
        f"AMP: {amp_enabled} | pinned memory: {pin_memory} | "
        f"workers: {args.num_workers} | setup: {time.perf_counter() - setup_start:.1f}s"
    )
    with (args.output_dir / "configuration.json").open("w") as handle:
        json.dump(
            {
                "arguments": vars(args),
                "device": str(device),
                "amp_enabled": amp_enabled,
                "pin_memory": pin_memory,
                "high_level_names": train_dataset.high_level_names,
                "high_level_mean": mean.tolist(),
                "high_level_std": std.tolist(),
                "particle_type_vocabulary": type_vocabulary,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            },
            handle,
            indent=2,
            default=str,
        )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    stale_epochs = 0
    checkpoint_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            amp=amp_enabled,
            scaler=scaler,
            non_blocking=non_blocking,
        )
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            amp=amp_enabled,
            non_blocking=non_blocking,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "seconds": epoch_seconds,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train loss {train_loss:.5f}, acc {train_accuracy:.4f} "
            f"| val loss {val_loss:.5f}, acc {val_accuracy:.4f} "
            f"| {epoch_seconds:.1f}s"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "high_level_names": train_dataset.high_level_names,
                    "type_vocabulary": type_vocabulary,
                    "args": vars(args),
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    save_history(history, args.output_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_labels, test_scores = predict(
        model,
        test_loader,
        device,
        amp=amp_enabled,
        non_blocking=non_blocking,
    )
    metrics = save_test_results(test_indices, test_labels, test_scores, args.output_dir)
    print(json.dumps(metrics, indent=2))
    print(f"Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
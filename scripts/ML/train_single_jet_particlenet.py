#!/usr/bin/env python3
"""Train a single-jet ParticleNet quark/gluon classifier.

Class convention: quark=0, gluon=1.  The output sigmoid is therefore P(gluon).
The script accepts either per-jet truth or a scalar event label when both jets
are guaranteed by the dataset definition to have the event flavour.
"""

from __future__ import annotations

import argparse
import csv
import glob
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

from single_jet_particlenet import (
    SingleJetBatch,
    SingleJetParquetDataset,
    SingleJetParticleNet,
    infer_particle_type_vocabulary,
    make_loader,
    read_jet_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-jet quark/gluon ParticleNet tagger.")
    parser.add_argument("--train-parquet", nargs="+", required=True)
    parser.add_argument("--val-parquet", nargs="+", required=True)
    parser.add_argument("--test-parquet", nargs="+", required=True)
    parser.add_argument(
        "--label-field", default="jet_label",
        help="Label column: two-entry per-jet truth or scalar event truth, according to --label-source.",
    )
    parser.add_argument(
        "--label-source", choices=("per-jet", "event"), default="per-jet",
        help="Use two labels per event, or duplicate one scalar event label for both jets.",
    )
    parser.add_argument("--label-format", choices=("binary", "pdg"), default="binary")
    parser.add_argument("--quark-pdgs", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--unknown-label-policy", choices=("error", "drop"), default="error")
    parser.add_argument("--output-dir", type=Path, default=Path("results/single_jet_particlenet"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-constituents", type=int, default=100)
    parser.add_argument("--shard-cache-size", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-particle-type", action="store_true")
    parser.add_argument("--particle-types", nargs="+", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--limit-jets", type=int, default=None,
        help="Class-balanced debugging limit applied separately to each split.",
    )
    return parser.parse_args()


def expand_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        paths.extend(matches or ([Path(pattern)] if Path(pattern).is_file() else []))
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


def valid_indices(labels: np.ndarray, policy: str, split: str) -> list[int]:
    bad = np.flatnonzero(labels < 0)
    if bad.size and policy == "error":
        examples = bad[:10].tolist()
        raise ValueError(
            f"{split} contains {len(bad)} unsupported jet labels at flattened indices "
            f"{examples}. Use --unknown-label-policy drop only if this exclusion is intended."
        )
    return np.flatnonzero(labels >= 0).tolist()


def balanced_limit(labels: np.ndarray, indices: Sequence[int], limit: int | None, seed: int) -> list[int]:
    if limit is None or limit >= len(indices):
        return list(indices)
    if limit < 2:
        raise ValueError("--limit-jets must be at least 2")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in (0, 1):
        candidates = np.asarray([i for i in indices if labels[i] == label], dtype=np.int64)
        rng.shuffle(candidates)
        selected.extend(candidates[: limit // 2].tolist())
    remaining = limit - len(selected)
    if remaining:
        rest = np.setdiff1d(np.asarray(indices), np.asarray(selected))
        rng.shuffle(rest)
        selected.extend(rest[:remaining].tolist())
    rng.shuffle(selected)
    if len(selected) < 2 or len(np.unique(labels[selected])) != 2:
        raise ValueError("Selected sample must contain both quark and gluon jets")
    return selected


class ShardLocalBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: SingleJetParquetDataset, indices: Sequence[int], batch_size: int, seed: int) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        groups: dict[int, list[int]] = {}
        for subset_position, dataset_index in enumerate(indices):
            groups.setdefault(dataset.shard_index(dataset_index), []).append(subset_position)
        self.groups = tuple(np.asarray(group, dtype=np.int64) for group in groups.values())

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        remainder: list[int] = []
        for group_id in rng.permutation(len(self.groups)):
            group = self.groups[int(group_id)].copy()
            rng.shuffle(group)
            stop = (len(group) // self.batch_size) * self.batch_size
            for start in range(0, stop, self.batch_size):
                yield group[start:start + self.batch_size].tolist()
            remainder.extend(group[stop:].tolist())
        rng.shuffle(remainder)
        for start in range(0, len(remainder), self.batch_size):
            yield remainder[start:start + self.batch_size]

    def __len__(self) -> int:
        return (sum(len(group) for group in self.groups) + self.batch_size - 1) // self.batch_size


def run_epoch(
    model: nn.Module,
    loader: Iterable[SingleJetBatch],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *, amp: bool, scaler: torch.amp.GradScaler | None, non_blocking: bool,
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
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
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
            size = batch.y.numel()
            loss_sum += loss.detach().double() * size
            correct += ((logits >= 0) == (batch.y >= 0.5)).sum()
            count += size
    if not count:
        raise ValueError("DataLoader produced no jets")
    return float((loss_sum / count).item()), float((correct / count).item())


@torch.no_grad()
def predict(model: nn.Module, loader: Iterable[SingleJetBatch], device: torch.device, *, amp: bool, non_blocking: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    events: list[np.ndarray] = []
    jets: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=non_blocking)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            probability = torch.sigmoid(model(batch))
        labels.append(batch.y.cpu().numpy())
        scores.append(probability.cpu().numpy())
        events.append(batch.event_index.cpu().numpy())
        jets.append(batch.jet_index.cpu().numpy())
    return tuple(np.concatenate(items) for items in (labels, scores, events, jets))


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    axes[0].set(xlabel="Epoch", ylabel="BCE loss"); axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1)); axes[1].legend()
    fig.tight_layout(); fig.savefig(output_dir / "learning_curves.png", dpi=160); plt.close(fig)


def save_results(labels: np.ndarray, scores: np.ndarray, events: np.ndarray, jets: np.ndarray, output_dir: Path) -> dict[str, object]:
    predictions = (scores >= 0.5).astype(np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    metrics: dict[str, object] = {
        "class_convention": {"0": "quark", "1": "gluon"},
        "test_jets": int(len(labels)),
        "test_accuracy": float(accuracy_score(labels, predictions)),
        "test_roc_auc_gluon": float(roc_auc_score(labels, scores)),
        "confusion_matrix_rows_truth_columns_prediction": matrix.tolist(),
    }
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("event_index", "jet_index", "truth", "gluon_probability", "prediction"))
        writer.writerows(zip(events, jets, labels.astype(int), scores, predictions))
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot(fpr, tpr, label=f"AUC = {metrics['test_roc_auc_gluon']:.4f}")
    axis.plot((0, 1), (0, 1), "--", color="0.5")
    axis.set(xlabel="Quark mistag efficiency", ylabel="Gluon efficiency", xlim=(0, 1), ylim=(0, 1))
    axis.legend(loc="lower right"); fig.tight_layout(); fig.savefig(output_dir / "roc_curve.png", dpi=160); plt.close(fig)
    return metrics


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    if args.label_field == "label" and args.label_source != "event":
        raise ValueError("--label-field label requires --label-source event")
    if min(args.epochs, args.batch_size, args.max_constituents, args.shard_cache_size, args.prefetch_factor) < 1:
        raise ValueError("Epochs, batch/cache sizes, constituent limit and prefetch factor must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_paths = expand_paths(args.train_parquet)
    val_paths = expand_paths(args.val_parquet)
    test_paths = expand_paths(args.test_parquet)
    path_sets = [set(train_paths), set(val_paths), set(test_paths)]
    if any(path_sets[i] & path_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Train, validation and test file patterns must not overlap")
    label_args = dict(
        label_field=args.label_field, label_source=args.label_source,
        label_format=args.label_format, quark_pdgs=args.quark_pdgs,
    )
    train_labels = read_jet_labels(train_paths, **label_args)
    val_labels = read_jet_labels(val_paths, **label_args)
    test_labels = read_jet_labels(test_paths, **label_args)
    train_indices = balanced_limit(train_labels, valid_indices(train_labels, args.unknown_label_policy, "train"), args.limit_jets, args.seed)
    val_indices = sorted(balanced_limit(val_labels, valid_indices(val_labels, args.unknown_label_policy, "validation"), args.limit_jets, args.seed + 1))
    test_indices = sorted(balanced_limit(test_labels, valid_indices(test_labels, args.unknown_label_policy, "test"), args.limit_jets, args.seed + 2))

    if args.particle_types is not None:
        vocabulary = tuple(sorted(set(args.particle_types)))
    elif args.use_particle_type:
        vocabulary = infer_particle_type_vocabulary(train_paths)
    else:
        vocabulary = ()
    dataset_args = dict(
        **label_args,
        type_vocabulary=vocabulary,
        max_constituents=args.max_constituents,
        shard_cache_size=args.shard_cache_size,
    )
    train_dataset = SingleJetParquetDataset(train_paths, **dataset_args)
    val_dataset = SingleJetParquetDataset(val_paths, **dataset_args)
    test_dataset = SingleJetParquetDataset(test_paths, **dataset_args)
    device = choose_device(args.device)
    amp = device.type == "cuda" if args.amp is None else args.amp
    if amp and device.type != "cuda":
        raise ValueError("AMP is supported only on CUDA")
    pin_memory = device.type == "cuda" if args.pin_memory is None else args.pin_memory
    non_blocking = bool(pin_memory and device.type == "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    loader_args = dict(num_workers=args.num_workers, pin_memory=pin_memory, prefetch_factor=args.prefetch_factor)
    train_subset = Subset(train_dataset, train_indices)
    train_loader = make_loader(
        train_subset,
        batch_sampler=ShardLocalBatchSampler(train_dataset, train_indices, args.batch_size, args.seed),
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        **loader_args,
    )
    val_loader = make_loader(Subset(val_dataset, val_indices), batch_size=args.batch_size, shuffle=False, persistent_workers=False, **loader_args)
    test_loader = make_loader(Subset(test_dataset, test_indices), batch_size=args.batch_size, shuffle=False, persistent_workers=False, **loader_args)

    model = SingleJetParticleNet(num_particle_types=train_dataset.num_particle_types).to(device)
    selected_train_labels = train_labels[np.asarray(train_indices)]
    positives = int(selected_train_labels.sum())
    negatives = len(selected_train_labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Training sample must contain both quark and gluon jets")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "arguments": vars(args), "device": str(device), "amp_enabled": amp,
        "pin_memory": pin_memory, "class_convention": {"0": "quark", "1": "gluon"},
        "particle_type_vocabulary": vocabulary,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with (args.output_dir / "configuration.json").open("w") as handle:
        json.dump(configuration, handle, indent=2, default=str)
    print(f"Device: {device}")
    print(f"Jets: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    print(
        f"Truth: field={args.label_field!r}, source={args.label_source}, "
        f"format={args.label_format}, 0=quark, 1=gluon"
    )
    print(f"Particle-type vocabulary size: {train_dataset.num_particle_types}")
    print(f"AMP: {amp} | pinned memory: {pin_memory} | workers: {args.num_workers} | setup: {time.perf_counter() - start:.1f}s")

    history: list[dict[str, float]] = []
    best = float("inf")
    stale = 0
    checkpoint_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer, amp=amp, scaler=scaler, non_blocking=non_blocking)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device, amp=amp, scaler=None, non_blocking=non_blocking)
        row = {
            "epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss,
            "train_accuracy": train_accuracy, "val_accuracy": val_accuracy,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.5f}, acc {train_accuracy:.4f} | val loss {val_loss:.5f}, acc {val_accuracy:.4f} | {row['seconds']:.1f}s")
        if val_loss < best:
            best = val_loss; stale = 0
            torch.save({
                "model_state_dict": model.state_dict(), "type_vocabulary": vocabulary,
                "class_convention": {0: "quark", 1: "gluon"}, "args": vars(args),
                "best_val_loss": best,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after {epoch} epochs"); break
    save_history(history, args.output_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    labels, scores, events, jets = predict(model, test_loader, device, amp=amp, non_blocking=non_blocking)
    metrics = save_results(labels, scores, events, jets, args.output_dir)
    print(json.dumps(metrics, indent=2))
    print(f"Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
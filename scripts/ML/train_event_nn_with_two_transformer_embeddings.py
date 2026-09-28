#!/usr/bin/env python3
"""Train the dissertation wide event NN using two frozen transformer embeddings.

This is the latent-representation version of the existing stacked classifier.
Instead of reducing each jet to one Particle-Transformer gluon probability, it
removes/bypasses the transformer's final 128 -> 1 classification layer and
keeps the 128-dimensional representation for each jet:

    22 existing high-level observables
    + 128-d leading-jet transformer embedding
    + 128-d subleading-jet transformer embedding
    ------------------------------------------------
    278 inputs -> wide event MLP [256, 128, 64] -> event signal logit

By default ``--embedding-source prelogit`` captures the 128-vector immediately
before the final Linear(128, 1) layer.  This corresponds most literally to
"taking off the last layer": the original gluon logit is a linear projection
of this vector, so the event NN retains all information needed to reconstruct
the original jet score while also gaining access to the other latent features.

An alternative ``--embedding-source cls`` captures the transformer's normalized
CLS representation immediately before the whole classifier head.

The Particle Transformer remains completely frozen.  Embeddings are generated
once for all events and cached as float16 .npy arrays to limit disk use.  The
event MLP keeps the dissertation configuration:

    hidden layers = [256, 128, 64]
    activation    = ReLU
    batch norm    = enabled
    dropout       = 0.15
    loss          = BCEWithLogitsLoss
    optimiser     = AdamW
    learning rate = 1e-3
    weight decay  = 1e-4
    batch size    = 4096

Preprocessing is fitted on the event-level training sample only.  A memory-
conscious in-place implementation is used because 278 float32 inputs for the
full event sample are several GB.

Expected repository files
-------------------------
Place this script in scripts/ML/.  It reuses:

    scripts/ML/train_event_nn_architecture_sweep.py
    scripts/ML/train_event_nn_with_two_transformer_scores.py
    scripts/ML/train_single_jet_particle_transformer_v3_scheduler.py
    scripts/ML/single_jet_particlenet.py

Example
-------
python scripts/ML/train_event_nn_with_two_transformer_embeddings.py \\
    --dataset-root cache/analysis_dataset \\
    --transformer-checkpoint \\
      results/particle_transformer_250k_d128_h8_l4_pid_cosine/best_model.pt \\
    --output-dir \\
      outputs/ml/nn_wide_plus_transformer_embeddings_250k
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Subset, TensorDataset

# -----------------------------------------------------------------------------
# Reuse the exact dissertation event NN and the transformer/data helpers that
# have already been validated in the score-based stacked classifier.
# -----------------------------------------------------------------------------
try:
    from train_event_nn_architecture_sweep import (
        EventMLP,
        FEATURE_NAMES,
        combine_classes,
        fixed_signal_efficiencies,
        load_parquet_matrix,
        make_loaders,
        predict as predict_event_nn,
        resolve_device,
        set_seed,
        train_model,
    )
except ImportError:
    from scripts.ML.train_event_nn_architecture_sweep import (
        EventMLP,
        FEATURE_NAMES,
        combine_classes,
        fixed_signal_efficiencies,
        load_parquet_matrix,
        make_loaders,
        predict as predict_event_nn,
        resolve_device,
        set_seed,
        train_model,
    )

try:
    from train_event_nn_with_two_transformer_scores import (
        build_transformer_for_dataset,
        load_frozen_transformer,
        real_parquet_files,
        validation_directory,
    )
except ImportError:
    from scripts.ML.train_event_nn_with_two_transformer_scores import (
        build_transformer_for_dataset,
        load_frozen_transformer,
        real_parquet_files,
        validation_directory,
    )

try:
    from single_jet_particlenet import SingleJetParquetDataset, make_loader as make_jet_loader
except ImportError:
    from scripts.ML.single_jet_particlenet import (
        SingleJetParquetDataset,
        make_loader as make_jet_loader,
    )


# =============================================================================
# Frozen event-NN configuration from dissertation
# =============================================================================

HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.15
ACTIVATION = "relu"
BATCH_NORM = True

if len(FEATURE_NAMES) != 22:
    raise RuntimeError(f"Expected 22 dissertation inputs, found {len(FEATURE_NAMES)}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("cache/analysis_dataset"),
    )
    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        required=True,
        help="Frozen single-jet Particle Transformer best_model.pt checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ml/nn_wide_plus_transformer_embeddings"),
    )

    # Same event NN defaults as the dissertation wide model.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--parquet-batch-size", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    # Frozen transformer inference.
    parser.add_argument("--transformer-batch-size", type=int, default=256)
    parser.add_argument("--transformer-num-workers", type=int, default=2)
    parser.add_argument(
        "--transformer-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to the AMP setting stored in the transformer checkpoint.",
    )
    parser.add_argument(
        "--embedding-source",
        choices=("prelogit", "cls"),
        default="prelogit",
        help=(
            "prelogit = 128-vector immediately before final Linear(128,1); "
            "cls = normalized transformer CLS embedding before classifier head."
        ),
    )
    parser.add_argument(
        "--embedding-cache-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype for cached embeddings. float16 roughly halves disk usage.",
    )
    parser.add_argument(
        "--recompute-transformer-embeddings",
        action="store_true",
        help="Ignore compatible embedding caches and rerun transformer inference.",
    )

    # Debugging only. 0 means all available events per class per split.
    parser.add_argument(
        "--max-events-per-class",
        type=int,
        default=0,
        help="Optional per-class cap for quick debugging; 0 uses all events.",
    )

    return parser.parse_args()


# =============================================================================
# Embedding cache helpers
# =============================================================================


def embedding_cache_paths(
    cache_dir: Path,
    sample_name: str,
    split_name: str,
    source: str,
) -> tuple[Path, Path]:
    stem = f"{sample_name}_{split_name}_two_jet_{source}_embeddings"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def embedding_cache_is_compatible(
    data_path: Path,
    metadata_path: Path,
    checkpoint_path: Path,
    n_events: int,
    embedding_dim: int,
    embedding_source: str,
    cache_dtype: str,
) -> bool:
    if not data_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
        if not (
            metadata.get("checkpoint_path") == str(checkpoint_path.resolve())
            and int(metadata.get("checkpoint_mtime_ns", -1))
            == checkpoint_path.stat().st_mtime_ns
            and int(metadata.get("n_events", -1)) == n_events
            and int(metadata.get("embedding_dim", -1)) == embedding_dim
            and metadata.get("embedding_source") == embedding_source
            and metadata.get("cache_dtype") == cache_dtype
        ):
            return False

        array = np.load(data_path, mmap_mode="r")
        expected = (n_events, 2, embedding_dim)
        good = array.shape == expected
        del array
        return good
    except Exception:
        return False


@torch.no_grad()
def extract_two_jet_embeddings_for_events(
    *,
    files: list[Path],
    sample_name: str,
    split_name: str,
    checkpoint_path: Path,
    checkpoint: dict,
    train_args: dict,
    vocabulary: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    max_events: int,
    cache_dir: Path,
    recompute: bool,
    embedding_source: str,
    cache_dtype: str,
) -> np.ndarray:
    """Return memory-mapped [event, jet(0/1), embedding] frozen representations."""

    dataset_args = dict(
        label_field=str(train_args.get("label_field", "label")),
        label_source=str(train_args.get("label_source", "event")),
        label_format=str(train_args.get("label_format", "binary")),
        quark_pdgs=tuple(train_args.get("quark_pdgs", (1, 2, 3, 4, 5))),
        type_vocabulary=vocabulary,
        max_constituents=int(train_args.get("max_constituents", 100)),
        shard_cache_size=int(train_args.get("shard_cache_size", 2)),
    )

    dataset = SingleJetParquetDataset(files, **dataset_args)
    if len(dataset) % 2 != 0:
        raise RuntimeError(
            f"Single-jet dataset length is not even for {sample_name}/{split_name}"
        )

    total_events = len(dataset) // 2
    n_events = total_events if max_events <= 0 else min(total_events, max_events)
    embedding_dim = int(train_args.get("d_model", 0))
    if embedding_dim <= 0:
        raise ValueError("Could not determine transformer d_model from checkpoint")

    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, metadata_path = embedding_cache_paths(
        cache_dir, sample_name, split_name, embedding_source
    )

    if (
        not recompute
        and embedding_cache_is_compatible(
            data_path,
            metadata_path,
            checkpoint_path,
            n_events,
            embedding_dim,
            embedding_source,
            cache_dtype,
        )
    ):
        embeddings = np.load(data_path, mmap_mode="r")
        print(
            f"Loaded cached transformer embeddings: {data_path} | "
            f"shape={embeddings.shape} dtype={embeddings.dtype}"
        )
        del dataset
        return embeddings

    model = build_transformer_for_dataset(checkpoint, train_args, dataset, device)

    if n_events < total_events:
        inference_dataset = Subset(dataset, range(2 * n_events))
    else:
        inference_dataset = dataset

    pin_memory = device.type == "cuda"
    non_blocking = pin_memory and device.type == "cuda"
    prefetch_factor = int(train_args.get("prefetch_factor", 2))

    loader = make_jet_loader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
    )

    # Capture the desired 128-d representation without changing the trained
    # transformer class/state_dict.
    captured: dict[str, torch.Tensor] = {}

    if embedding_source == "prelogit":
        final_linear = model.classifier[-1]
        if not isinstance(final_linear, torch.nn.Linear):
            raise TypeError("Expected final transformer classifier module to be Linear")

        def capture_prelogit(module, inputs):
            captured["embedding"] = inputs[0].detach()

        hook_handle = final_linear.register_forward_pre_hook(capture_prelogit)
    elif embedding_source == "cls":
        def capture_cls(module, inputs, output):
            captured["embedding"] = output.detach()

        hook_handle = model.final_norm.register_forward_hook(capture_cls)
    else:
        raise ValueError(f"Unsupported embedding source: {embedding_source}")

    np_dtype = np.float16 if cache_dtype == "float16" else np.float32
    embeddings = np.lib.format.open_memmap(
        data_path,
        mode="w+",
        dtype=np_dtype,
        shape=(n_events, 2, embedding_dim),
    )
    embeddings[:] = np.nan

    score_sum = np.zeros(2, dtype=np.float64)
    score_count = np.zeros(2, dtype=np.int64)
    jets_written = 0

    print(
        f"Extracting {embedding_source} embeddings for {sample_name}/{split_name}: "
        f"{n_events:,} events ({2*n_events:,} jets), dim={embedding_dim}..."
    )
    start = time.perf_counter()

    try:
        model.eval()
        for batch in loader:
            batch = batch.to(device, non_blocking=non_blocking)
            captured.clear()

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                logits = model(batch)

            if "embedding" not in captured:
                raise RuntimeError("Transformer embedding hook did not fire")

            batch_embedding = (
                captured["embedding"].float().cpu().numpy().astype(np_dtype, copy=False)
            )
            event_indices = batch.event_index.cpu().numpy().astype(np.int64, copy=False)
            jet_indices = batch.jet_index.cpu().numpy().astype(np.int64, copy=False)
            probabilities = torch.sigmoid(logits).float().cpu().numpy()

            valid = event_indices < n_events
            event_indices = event_indices[valid]
            jet_indices = jet_indices[valid]
            batch_embedding = batch_embedding[valid]
            probabilities = probabilities[valid]

            if np.any((jet_indices < 0) | (jet_indices > 1)):
                bad = np.unique(jet_indices[(jet_indices < 0) | (jet_indices > 1)])
                raise RuntimeError(
                    f"Unexpected jet indices from transformer dataset: {bad.tolist()}"
                )

            embeddings[event_indices, jet_indices, :] = batch_embedding
            for jet in (0, 1):
                mask = jet_indices == jet
                if np.any(mask):
                    score_sum[jet] += float(np.sum(probabilities[mask], dtype=np.float64))
                    score_count[jet] += int(np.sum(mask))

            jets_written += len(event_indices)

    finally:
        hook_handle.remove()

    embeddings.flush()

    if jets_written != 2 * n_events:
        raise RuntimeError(
            f"Expected to write {2*n_events:,} jet embeddings but wrote {jets_written:,}"
        )

    # Check for missing/non-finite rows in manageable chunks.
    check_step = 100_000
    for start_event in range(0, n_events, check_step):
        stop_event = min(start_event + check_step, n_events)
        if not np.all(np.isfinite(embeddings[start_event:stop_event])):
            raise RuntimeError(
                f"Non-finite embedding detected in {sample_name}/{split_name} "
                f"events {start_event}:{stop_event}"
            )

    elapsed = time.perf_counter() - start
    mean_scores = score_sum / np.maximum(score_count, 1)

    metadata = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "n_events": int(n_events),
        "embedding_dim": int(embedding_dim),
        "embedding_source": embedding_source,
        "cache_dtype": cache_dtype,
        "mean_gluon_score_j1": float(mean_scores[0]),
        "mean_gluon_score_j2": float(mean_scores[1]),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(
        f"  done in {elapsed:.1f}s | mean original gluon scores "
        f"J1={mean_scores[0]:.4f}, J2={mean_scores[1]:.4f}"
    )
    print(
        f"  cached -> {data_path} "
        f"({data_path.stat().st_size / (1024**3):.2f} GiB)"
    )

    del model, dataset, inference_dataset, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Re-open read-only so downstream code does not accidentally modify cache.
    del embeddings
    return np.load(data_path, mmap_mode="r")


# =============================================================================
# Memory-conscious event data preparation / preprocessing
# =============================================================================


def embedding_feature_names(embedding_dim: int) -> list[str]:
    names = []
    for jet_name in ("leading", "subleading"):
        for index in range(embedding_dim):
            names.append(f"{jet_name}_transformer_embedding_{index:03d}")
    return names


def augment_class_split(
    *,
    files: list[Path],
    sample_name: str,
    split_name: str,
    args: argparse.Namespace,
    checkpoint: dict,
    train_args: dict,
    vocabulary: tuple[int, ...],
    transformer_amp: bool,
    device: torch.device,
    embedding_dim: int,
) -> np.ndarray:
    high_level = load_parquet_matrix(
        files,
        args.max_events_per_class,
        args.parquet_batch_size,
        f"{sample_name} {split_name} high-level features",
    )

    embeddings = extract_two_jet_embeddings_for_events(
        files=files,
        sample_name=sample_name,
        split_name=split_name,
        checkpoint_path=args.transformer_checkpoint,
        checkpoint=checkpoint,
        train_args=train_args,
        vocabulary=vocabulary,
        device=device,
        batch_size=args.transformer_batch_size,
        num_workers=args.transformer_num_workers,
        amp=transformer_amp,
        max_events=args.max_events_per_class,
        cache_dir=args.output_dir / "transformer_embedding_cache",
        recompute=args.recompute_transformer_embeddings,
        embedding_source=args.embedding_source,
        cache_dtype=args.embedding_cache_dtype,
    )

    if len(high_level) != len(embeddings):
        raise RuntimeError(
            f"Row mismatch for {sample_name}/{split_name}: "
            f"high-level={len(high_level):,}, embeddings={len(embeddings):,}"
        )

    input_dim = 22 + 2 * embedding_dim
    augmented = np.empty((len(high_level), input_dim), dtype=np.float32)
    augmented[:, :22] = high_level

    # Convert float16 embedding cache to float32 directly into the final matrix
    # in chunks, avoiding a second full-size temporary array.
    chunk = 100_000
    for start in range(0, len(high_level), chunk):
        stop = min(start + chunk, len(high_level))
        augmented[start:stop, 22:] = np.asarray(
            embeddings[start:stop], dtype=np.float32
        ).reshape(stop - start, 2 * embedding_dim)

    del high_level, embeddings
    return augmented


def fit_preprocessing_inplace(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same median-impute + z-score scheme, without a full float64 copy."""

    if X_train.dtype != np.float32:
        raise TypeError("Expected float32 training matrix")

    X_train[~np.isfinite(X_train)] = np.nan

    # 278 columns only, so column-wise median imputation is memory efficient.
    medians = np.empty(X_train.shape[1], dtype=np.float32)
    for column in range(X_train.shape[1]):
        values = X_train[:, column]
        median = np.nanmedian(values)
        if not np.isfinite(median):
            median = 0.0
        medians[column] = np.float32(median)
        missing = np.isnan(values)
        if np.any(missing):
            values[missing] = medians[column]

    # Accumulate moments in float64, matching the numerical intent of the
    # original preprocessing without duplicating the whole matrix in float64.
    means = np.mean(X_train, axis=0, dtype=np.float64).astype(np.float32)
    stds = np.std(X_train, axis=0, dtype=np.float64).astype(np.float32)
    stds = np.where(stds > 1.0e-12, stds, 1.0).astype(np.float32)

    X_train -= means
    X_train /= stds
    return medians, means, stds


def apply_preprocessing_inplace(
    X: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> None:
    X[~np.isfinite(X)] = np.nan
    for column in range(X.shape[1]):
        values = X[:, column]
        missing = np.isnan(values)
        if np.any(missing):
            values[missing] = medians[column]
    X -= means
    X /= stds


def load_combined_split(
    *,
    split: str,
    split_files: dict[tuple[str, str], list[Path]],
    args: argparse.Namespace,
    checkpoint: dict,
    train_args: dict,
    vocabulary: tuple[int, ...],
    transformer_amp: bool,
    device: torch.device,
    embedding_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    signal = augment_class_split(
        files=split_files[("signal", split)],
        sample_name="signal",
        split_name=split,
        args=args,
        checkpoint=checkpoint,
        train_args=train_args,
        vocabulary=vocabulary,
        transformer_amp=transformer_amp,
        device=device,
        embedding_dim=embedding_dim,
    )
    background = augment_class_split(
        files=split_files[("background", split)],
        sample_name="background",
        split_name=split,
        args=args,
        checkpoint=checkpoint,
        train_args=train_args,
        vocabulary=vocabulary,
        transformer_amp=transformer_amp,
        device=device,
        embedding_dim=embedding_dim,
    )

    X, y = combine_classes(signal, background, seed)
    del signal, background
    return X, y


# =============================================================================
# Evaluation / outputs
# =============================================================================


def make_eval_loader(X: np.ndarray, y: np.ndarray, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def operating_points_frame(y: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    points = fixed_signal_efficiencies(y, scores, targets=[0.5, 0.7, 0.8, 0.9])
    rows = []
    for target, point in points.items():
        rows.append(
            {
                "target_signal_efficiency": target,
                "signal_efficiency": point["signal_efficiency"],
                "background_efficiency": point["background_efficiency"],
                "background_rejection": point["background_rejection"],
                "threshold": point["threshold"],
            }
        )
    return pd.DataFrame(rows)


def save_roc(y: np.ndarray, scores: np.ndarray, auc: float, output: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y, scores)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.5f}")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_training_history(history: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Training")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="BCE loss", title="Training / validation loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["val_auc"], label="Validation ROC AUC")
    axes[1].set(xlabel="Epoch", ylabel="ROC AUC", title="Validation ROC AUC")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.random_seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    checkpoint, configuration, transformer_train_args, vocabulary, amp_stored = (
        load_frozen_transformer(args.transformer_checkpoint, device)
    )
    del configuration

    transformer_amp = amp_stored if args.transformer_amp is None else bool(args.transformer_amp)
    if device.type != "cuda":
        transformer_amp = False

    embedding_dim = int(transformer_train_args.get("d_model", 0))
    if embedding_dim <= 0:
        raise ValueError("Could not determine d_model from transformer checkpoint")

    embedding_names = embedding_feature_names(embedding_dim)
    augmented_feature_names = list(FEATURE_NAMES) + embedding_names
    input_dim = len(augmented_feature_names)

    print("=" * 92)
    print("WIDE EVENT NN + TWO FROZEN PARTICLE-TRANSFORMER EMBEDDINGS")
    print("=" * 92)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Event inputs: 22 high-level + 2 x {embedding_dim}-d transformer embeddings "
        f"= {input_dim}"
    )
    print(f"Embedding source: {args.embedding_source}")
    print(f"Embedding cache dtype: {args.embedding_cache_dtype}")
    print(f"Event MLP: {HIDDEN_DIMS}, ReLU, batch norm, dropout={DROPOUT}")
    print(
        f"AdamW: lr={args.learning_rate:.1e}, weight_decay={args.weight_decay:.1e}, "
        f"batch={args.batch_size}"
    )

    print("\nFrozen transformer")
    print(f"  checkpoint: {args.transformer_checkpoint}")
    print(f"  best epoch: {checkpoint.get('epoch', 'unknown')}")
    if "val_loss" in checkpoint:
        print(f"  validation loss: {float(checkpoint['val_loss']):.6f}")
    print(
        "  architecture: "
        f"d={transformer_train_args.get('d_model')}, "
        f"heads={transformer_train_args.get('num_heads')}, "
        f"layers={transformer_train_args.get('num_layers')}, "
        f"ffn={transformer_train_args.get('ffn_dim')}"
    )
    print(f"  particle types: {'enabled' if vocabulary else 'disabled'}")
    print(f"  AMP inference: {transformer_amp}")
    if args.embedding_source == "prelogit":
        print(
            "  extraction point: input to final Linear(128,1) classifier layer "
            "(literal last-layer removal)"
        )
    else:
        print("  extraction point: normalized CLS vector before classifier head")

    roots = {
        "signal": args.dataset_root / "signal",
        "background": args.dataset_root / "background",
    }
    split_files: dict[tuple[str, str], list[Path]] = {}
    for sample, root in roots.items():
        split_files[(sample, "train")] = real_parquet_files(root / "train")
        split_files[(sample, "validation")] = real_parquet_files(validation_directory(root))
        split_files[(sample, "test")] = real_parquet_files(root / "test")

    # ------------------------------------------------------------------
    # Train split first: fit preprocessing and standardise in-place before
    # loading validation/test to keep peak RAM usage under control.
    # ------------------------------------------------------------------
    X_train, y_train = load_combined_split(
        split="train",
        split_files=split_files,
        args=args,
        checkpoint=checkpoint,
        train_args=transformer_train_args,
        vocabulary=vocabulary,
        transformer_amp=transformer_amp,
        device=device,
        embedding_dim=embedding_dim,
        seed=args.random_seed,
    )

    print("\nFitting preprocessing on training data only...")
    medians, means, stds = fit_preprocessing_inplace(X_train)

    np.savez(
        args.output_dir / "preprocessing.npz",
        medians=medians,
        means=means,
        stds=stds,
        feature_names=np.asarray(augmented_feature_names),
    )

    X_val, y_val = load_combined_split(
        split="validation",
        split_files=split_files,
        args=args,
        checkpoint=checkpoint,
        train_args=transformer_train_args,
        vocabulary=vocabulary,
        transformer_amp=transformer_amp,
        device=device,
        embedding_dim=embedding_dim,
        seed=args.random_seed + 1,
    )
    apply_preprocessing_inplace(X_val, medians, means, stds)

    X_test, y_test = load_combined_split(
        split="test",
        split_files=split_files,
        args=args,
        checkpoint=checkpoint,
        train_args=transformer_train_args,
        vocabulary=vocabulary,
        transformer_amp=transformer_amp,
        device=device,
        embedding_dim=embedding_dim,
        seed=args.random_seed + 2,
    )
    apply_preprocessing_inplace(X_test, medians, means, stds)

    print("\nEvent samples")
    for name, y in (("train", y_train), ("validation", y_val), ("test", y_test)):
        print(
            f"  {name:10s}: total={len(y):,} | "
            f"signal={int(np.sum(y==1)):,} | background={int(np.sum(y==0)):,}"
        )

    n_signal = int(np.sum(y_train == 1))
    n_background = int(np.sum(y_train == 0))
    pos_weight = n_background / n_signal
    print(f"\nBCE positive-class weight: {pos_weight:.6f}")

    train_loader, val_loader = make_loaders(
        X_train,
        y_train,
        X_val,
        y_val,
        args.batch_size,
        args.num_workers,
        args.random_seed,
    )

    model = EventMLP(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        batch_norm=BATCH_NORM,
        activation=ACTIVATION,
    ).to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Event-NN trainable parameters: {n_parameters:,}")

    start = time.time()
    history, best_state, best_epoch, best_val_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        pos_weight=pos_weight,
    )
    training_seconds = time.time() - start

    model.load_state_dict(best_state)
    model.eval()

    val_scores, val_labels = predict_event_nn(model, val_loader, device)
    test_loader = make_eval_loader(X_test, y_test, args.batch_size, args.num_workers)
    test_scores, test_labels = predict_event_nn(model, test_loader, device)

    val_auc = float(roc_auc_score(val_labels, val_scores))
    val_ap = float(average_precision_score(val_labels, val_scores))
    test_auc = float(roc_auc_score(test_labels, test_scores))
    test_ap = float(average_precision_score(test_labels, test_scores))

    val_ops = operating_points_frame(val_labels, val_scores)
    test_ops = operating_points_frame(test_labels, test_scores)

    history.to_csv(args.output_dir / "training_history.csv", index=False)
    val_ops.to_csv(args.output_dir / "validation_operating_points.csv", index=False)
    test_ops.to_csv(args.output_dir / "test_operating_points.csv", index=False)

    save_training_history(history, args.output_dir / "training_history.png")
    save_roc(
        val_labels,
        val_scores,
        val_auc,
        args.output_dir / "validation_roc.png",
        "Wide NN + two transformer jet embeddings: validation ROC",
    )
    save_roc(
        test_labels,
        test_scores,
        test_auc,
        args.output_dir / "test_roc.png",
        "Wide NN + two transformer jet embeddings: test ROC",
    )

    pd.DataFrame(
        {
            "label": test_labels.astype(np.int8),
            "event_score": test_scores.astype(np.float32),
        }
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)

    checkpoint_out = {
        "architecture": "wide_plus_two_transformer_embeddings",
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "batch_norm": BATCH_NORM,
        "activation": ACTIVATION,
        "input_dim": input_dim,
        "feature_names": augmented_feature_names,
        "model_state_dict": best_state,
        "preprocessing": {
            "medians": medians,
            "means": means,
            "stds": stds,
        },
        "best_epoch": int(best_epoch),
        "validation_auc": val_auc,
        "test_auc": test_auc,
        "transformer": {
            "checkpoint": str(args.transformer_checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_val_loss": checkpoint.get("val_loss"),
            "training_arguments": transformer_train_args,
            "particle_type_vocabulary": vocabulary,
            "frozen": True,
            "embedding_source": args.embedding_source,
            "embedding_dim_per_jet": embedding_dim,
            "embedding_cache_dtype": args.embedding_cache_dtype,
        },
    }
    torch.save(checkpoint_out, args.output_dir / "best_model.pt")

    metrics = {
        "model": "wide_event_nn_plus_two_frozen_transformer_embeddings",
        "feature_count": input_dim,
        "high_level_feature_count": 22,
        "embedding_dim_per_jet": embedding_dim,
        "embedding_source": args.embedding_source,
        "features": augmented_feature_names,
        "hidden_dims": HIDDEN_DIMS,
        "activation": ACTIVATION,
        "batch_norm": BATCH_NORM,
        "dropout": DROPOUT,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "positive_class_weight": pos_weight,
        "best_epoch": int(best_epoch),
        "best_training_validation_auc": float(best_val_auc),
        "validation_auc": val_auc,
        "validation_ap": val_ap,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "training_seconds": training_seconds,
        "n_train": len(y_train),
        "n_validation": len(y_val),
        "n_test": len(y_test),
        "transformer_checkpoint": str(args.transformer_checkpoint),
        "transformer_checkpoint_epoch": checkpoint.get("epoch"),
        "transformer_checkpoint_val_loss": checkpoint.get("val_loss"),
        "transformer_frozen": True,
    }
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, default=str)

    print("\n" + "=" * 92)
    print("RESULT")
    print("=" * 92)
    print(f"Best event-NN epoch: {best_epoch}")
    print(f"Validation ROC AUC:  {val_auc:.6f}")
    print(f"Test ROC AUC:        {test_auc:.6f}")
    print(f"Test average precision: {test_ap:.6f}")
    print("\nTest operating points")
    for row in test_ops.itertuples(index=False):
        print(
            f"  eps_s={row.signal_efficiency:.3f} | "
            f"eps_b={row.background_efficiency:.6e} | "
            f"R_b={row.background_rejection:.1f}"
        )
    print(f"\nOutputs: {args.output_dir}")
    print(f"Event-NN training time: {training_seconds:.1f}s")
    print("Transformer embedding caches are retained for fast reruns.")


if __name__ == "__main__":
    main()
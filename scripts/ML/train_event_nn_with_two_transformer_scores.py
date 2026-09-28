#!/usr/bin/env python3
"""Train the dissertation wide event NN with two frozen transformer jet scores.

This is a hierarchical/stacked event classifier:

    22 existing high-level event/jet observables
    + leading-jet Particle Transformer P(gluon)
    + subleading-jet Particle Transformer P(gluon)
    ------------------------------------------------
    24 inputs -> wide event MLP [256, 128, 64] -> event signal logit

The Particle Transformer is frozen.  Its two scores are generated once for
all events in each split and cached to disk.  The event MLP then uses the same
architecture and training hyperparameters as the final dissertation wide NN:

    hidden layers = [256, 128, 64]
    activation    = ReLU
    batch norm    = enabled
    dropout       = 0.15
    loss          = BCEWithLogitsLoss
    optimiser     = AdamW
    learning rate = 1e-3
    weight decay  = 1e-4
    batch size    = 4096

The event MLP is selected using the validation split and is then evaluated on
the test split.  Preprocessing is fitted on the event-level training split only.

Expected repository files
-------------------------
Place this script in scripts/ML/.  It reuses:

    scripts/ML/train_event_nn_architecture_sweep.py
    scripts/ML/train_single_jet_particle_transformer_v3_scheduler.py
    scripts/ML/single_jet_particlenet.py

The transformer checkpoint should be a best_model.pt produced by the v3
training script (for example the planned 500k-jet transformer).

Example
-------
python scripts/ML/train_event_nn_with_two_transformer_scores.py \
    --dataset-root cache/analysis_dataset \
    --transformer-checkpoint results/particle_transformer_500k_d128_h8_l4_pid_cosine/best_model.pt \
    --output-dir outputs/ml/nn_wide_plus_transformer_500k
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
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
# Reuse the exact event-NN and transformer implementations already used in the
# dissertation analysis.  The fallback imports also allow package-style use.
# -----------------------------------------------------------------------------
try:
    from train_event_nn_architecture_sweep import (
        EventMLP,
        FEATURE_NAMES,
        apply_preprocessing,
        combine_classes,
        fit_preprocessing,
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
        apply_preprocessing,
        combine_classes,
        fit_preprocessing,
        fixed_signal_efficiencies,
        load_parquet_matrix,
        make_loaders,
        predict as predict_event_nn,
        resolve_device,
        set_seed,
        train_model,
    )

try:
    from train_single_jet_particle_transformer_v3_scheduler import (
        SingleJetParticleTransformer,
        predict as predict_transformer,
    )
except ImportError:
    from scripts.ML.train_single_jet_particle_transformer_v3_scheduler import (
        SingleJetParticleTransformer,
        predict as predict_transformer,
    )

try:
    from single_jet_particlenet import SingleJetParquetDataset, make_loader as make_jet_loader
except ImportError:
    from scripts.ML.single_jet_particlenet import (
        SingleJetParquetDataset,
        make_loader as make_jet_loader,
    )


# =============================================================================
# Frozen event-NN configuration from the dissertation
# =============================================================================

HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.15
ACTIVATION = "relu"
BATCH_NORM = True

TRANSFORMER_SCORE_FEATURES = [
    "leading_transformer_gluon_score",
    "subleading_transformer_gluon_score",
]

AUGMENTED_FEATURE_NAMES = list(FEATURE_NAMES) + TRANSFORMER_SCORE_FEATURES

if len(FEATURE_NAMES) != 22:
    raise RuntimeError(f"Expected 22 dissertation inputs, found {len(FEATURE_NAMES)}")
if len(AUGMENTED_FEATURE_NAMES) != 24:
    raise RuntimeError("Expected 24 augmented event inputs")


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
        default=Path("outputs/ml/nn_wide_plus_transformer_scores"),
    )

    # Event NN: same defaults as dissertation wide model.
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

    # Transformer inference.  A larger inference batch can be tried if GPU RAM permits.
    parser.add_argument("--transformer-batch-size", type=int, default=256)
    parser.add_argument("--transformer-num-workers", type=int, default=2)
    parser.add_argument(
        "--transformer-amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to the AMP setting stored in the transformer checkpoint.",
    )
    parser.add_argument(
        "--recompute-transformer-scores",
        action="store_true",
        help="Ignore cached two-jet scores and run transformer inference again.",
    )

    # Debugging only.  0 means all available events per class per split.
    parser.add_argument(
        "--max-events-per-class",
        type=int,
        default=0,
        help="Optional per-class cap for quick debugging; 0 uses all events.",
    )

    return parser.parse_args()


# =============================================================================
# File helpers
# =============================================================================


def real_parquet_files(directory: Path) -> list[Path]:
    """Return real Parquet shards while excluding EOS .sys version artefacts."""
    files = sorted(
        path
        for path in directory.rglob("*.parquet")
        if path.is_file() and not path.name.startswith(".sys.")
    )
    if not files:
        raise FileNotFoundError(f"No valid Parquet files found beneath {directory}")
    return files


def validation_directory(class_root: Path) -> Path:
    for name in ("validation", "val"):
        candidate = class_root / name
        if candidate.is_dir():
            files = [
                p
                for p in candidate.rglob("*.parquet")
                if p.is_file() and not p.name.startswith(".sys.")
            ]
            if files:
                return candidate
    raise FileNotFoundError(f"No validation/val directory found beneath {class_root}")


# =============================================================================
# Transformer reconstruction
# =============================================================================


def required(mapping: dict, key: str):
    if key not in mapping:
        raise KeyError(f"Transformer checkpoint is missing argument {key!r}")
    return mapping[key]


def load_frozen_transformer(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    SingleJetParticleTransformer,
    dict,
    dict,
    tuple[int, ...],
    bool,
]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Transformer checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint or "configuration" not in checkpoint:
        raise ValueError(
            "Transformer checkpoint must contain model_state_dict and configuration"
        )

    configuration = checkpoint["configuration"]
    train_args = configuration.get("arguments")
    if not isinstance(train_args, dict):
        raise ValueError("Transformer checkpoint has no training arguments dictionary")

    vocabulary = tuple(int(v) for v in configuration.get("particle_type_vocabulary", ()))

    # Build a tiny dataset later to obtain num_particle_types exactly, because the
    # dataset reserves the padding/unknown index consistently with training.
    # The model itself is constructed in build_transformer_for_dataset().
    amp_stored = bool(configuration.get("amp_enabled", False))

    return checkpoint, configuration, train_args, vocabulary, amp_stored


def build_transformer_for_dataset(
    checkpoint: dict,
    train_args: dict,
    dataset: SingleJetParquetDataset,
    device: torch.device,
) -> SingleJetParticleTransformer:
    model = SingleJetParticleTransformer(
        num_particle_types=dataset.num_particle_types,
        type_embedding_dim=int(train_args.get("type_embedding_dim", 8)),
        d_model=int(required(train_args, "d_model")),
        num_heads=int(required(train_args, "num_heads")),
        num_layers=int(required(train_args, "num_layers")),
        ffn_dim=int(required(train_args, "ffn_dim")),
        dropout=float(train_args.get("dropout", 0.10)),
        pair_hidden_dim=int(train_args.get("pair_hidden_dim", 32)),
        use_pairwise_bias=not bool(train_args.get("disable_pairwise_bias", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


# =============================================================================
# Transformer score caching
# =============================================================================


def cache_is_compatible(
    cache_path: Path,
    checkpoint_path: Path,
    requested_events: int,
) -> bool:
    if not cache_path.is_file():
        return False
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            cached_path = str(payload["checkpoint_path"].item())
            cached_mtime = int(payload["checkpoint_mtime_ns"].item())
            cached_events = int(payload["n_events"].item())
        return (
            cached_path == str(checkpoint_path.resolve())
            and cached_mtime == checkpoint_path.stat().st_mtime_ns
            and cached_events == requested_events
        )
    except Exception:
        return False


def score_two_jets_for_events(
    *,
    files: list[Path],
    sample_name: str,
    split_name: str,
    checkpoint_path: Path,
    checkpoint: dict,
    configuration: dict,
    train_args: dict,
    vocabulary: tuple[int, ...],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    max_events: int,
    cache_dir: Path,
    recompute: bool,
) -> np.ndarray:
    """Return [n_events, 2] = P(gluon) for stored jet 0/1 of every event."""

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
        raise RuntimeError(f"Single-jet dataset length is not even for {sample_name}/{split_name}")

    total_events = len(dataset) // 2
    n_events = total_events if max_events <= 0 else min(total_events, max_events)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sample_name}_{split_name}_two_jet_scores.npz"

    if not recompute and cache_is_compatible(cache_path, checkpoint_path, n_events):
        with np.load(cache_path, allow_pickle=False) as payload:
            scores = np.asarray(payload["scores"], dtype=np.float32)
        if scores.shape != (n_events, 2):
            raise RuntimeError(f"Bad cached score shape in {cache_path}: {scores.shape}")
        print(f"Loaded cached transformer scores: {cache_path} | {scores.shape}")
        return scores

    model = build_transformer_for_dataset(checkpoint, train_args, dataset, device)

    # load_parquet_matrix(max_events=N) takes the first N events in file order.
    # SingleJetParquetDataset exposes event0-jet0, event0-jet1, event1-jet0, ...,
    # therefore selecting the first 2*N jet indices preserves exact row alignment.
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

    print(
        f"Scoring {sample_name}/{split_name}: {n_events:,} events "
        f"({2*n_events:,} jets) with frozen transformer..."
    )
    start = time.perf_counter()

    labels, scores_flat, event_indices, jet_indices = predict_transformer(
        model,
        loader,
        device,
        amp=amp,
        non_blocking=non_blocking,
    )

    del labels

    scores = np.full((n_events, 2), np.nan, dtype=np.float32)
    event_indices = np.asarray(event_indices, dtype=np.int64)
    jet_indices = np.asarray(jet_indices, dtype=np.int64)
    scores_flat = np.asarray(scores_flat, dtype=np.float32)

    valid = event_indices < n_events
    event_indices = event_indices[valid]
    jet_indices = jet_indices[valid]
    scores_flat = scores_flat[valid]

    if np.any((jet_indices < 0) | (jet_indices > 1)):
        bad = np.unique(jet_indices[(jet_indices < 0) | (jet_indices > 1)])
        raise RuntimeError(f"Unexpected jet indices from transformer dataset: {bad.tolist()}")

    scores[event_indices, jet_indices] = scores_flat

    if not np.all(np.isfinite(scores)):
        missing = np.argwhere(~np.isfinite(scores))[:10]
        raise RuntimeError(
            f"Missing transformer scores for {sample_name}/{split_name}; examples {missing.tolist()}"
        )

    np.savez_compressed(
        cache_path,
        scores=scores,
        checkpoint_path=np.asarray(str(checkpoint_path.resolve())),
        checkpoint_mtime_ns=np.asarray(checkpoint_path.stat().st_mtime_ns, dtype=np.int64),
        checkpoint_epoch=np.asarray(int(checkpoint.get("epoch", -1)), dtype=np.int64),
        n_events=np.asarray(n_events, dtype=np.int64),
    )

    print(
        f"  done in {time.perf_counter()-start:.1f}s | "
        f"mean scores J1={scores[:,0].mean():.4f}, J2={scores[:,1].mean():.4f}"
    )
    print(f"  cached -> {cache_path}")

    del model, dataset, inference_dataset, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return scores


# =============================================================================
# Event data preparation
# =============================================================================


def augment_class_split(
    *,
    files: list[Path],
    sample_name: str,
    split_name: str,
    args: argparse.Namespace,
    checkpoint: dict,
    configuration: dict,
    train_args: dict,
    vocabulary: tuple[int, ...],
    transformer_amp: bool,
    device: torch.device,
) -> np.ndarray:
    high_level = load_parquet_matrix(
        files,
        args.max_events_per_class,
        args.parquet_batch_size,
        f"{sample_name} {split_name} high-level features",
    )

    jet_scores = score_two_jets_for_events(
        files=files,
        sample_name=sample_name,
        split_name=split_name,
        checkpoint_path=args.transformer_checkpoint,
        checkpoint=checkpoint,
        configuration=configuration,
        train_args=train_args,
        vocabulary=vocabulary,
        device=device,
        batch_size=args.transformer_batch_size,
        num_workers=args.transformer_num_workers,
        amp=transformer_amp,
        max_events=args.max_events_per_class,
        cache_dir=args.output_dir / "transformer_score_cache",
        recompute=args.recompute_transformer_scores,
    )

    if len(high_level) != len(jet_scores):
        raise RuntimeError(
            f"Row mismatch for {sample_name}/{split_name}: "
            f"high-level={len(high_level):,}, transformer={len(jet_scores):,}"
        )

    augmented = np.column_stack((high_level, jet_scores)).astype(np.float32, copy=False)
    if augmented.shape[1] != 24:
        raise RuntimeError(f"Expected 24 inputs, got {augmented.shape[1]}")

    return augmented


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

    print("=" * 88)
    print("WIDE EVENT NN + TWO FROZEN PARTICLE-TRANSFORMER JET SCORES")
    print("=" * 88)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Event inputs: 22 high-level + 2 transformer scores = 24")
    print(f"Event MLP: {HIDDEN_DIMS}, ReLU, batch norm, dropout={DROPOUT}")
    print(
        f"AdamW: lr={args.learning_rate:.1e}, weight_decay={args.weight_decay:.1e}, "
        f"batch={args.batch_size}"
    )

    checkpoint, configuration, transformer_train_args, vocabulary, amp_stored = (
        load_frozen_transformer(args.transformer_checkpoint, device)
    )
    transformer_amp = amp_stored if args.transformer_amp is None else bool(args.transformer_amp)
    if device.type != "cuda":
        transformer_amp = False

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

    # ------------------------------------------------------------------
    # Locate all splits.
    # ------------------------------------------------------------------
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
    # Generate/load 24-input arrays.  Transformer inference is the slow
    # stage, but scores are cached so the event NN can be retrained quickly.
    # ------------------------------------------------------------------
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for split in ("train", "validation", "test"):
        for sample in ("signal", "background"):
            arrays[(sample, split)] = augment_class_split(
                files=split_files[(sample, split)],
                sample_name=sample,
                split_name=split,
                args=args,
                checkpoint=checkpoint,
                configuration=configuration,
                train_args=transformer_train_args,
                vocabulary=vocabulary,
                transformer_amp=transformer_amp,
                device=device,
            )

    # ------------------------------------------------------------------
    # Build event-level train / validation / test samples.
    # Signal=1, background=0, as in the dissertation event NN.
    # ------------------------------------------------------------------
    X_train, y_train = combine_classes(
        arrays[("signal", "train")],
        arrays[("background", "train")],
        args.random_seed,
    )
    X_val, y_val = combine_classes(
        arrays[("signal", "validation")],
        arrays[("background", "validation")],
        args.random_seed + 1,
    )
    X_test, y_test = combine_classes(
        arrays[("signal", "test")],
        arrays[("background", "test")],
        args.random_seed + 2,
    )
    del arrays

    print("\nEvent samples")
    for name, y in (("train", y_train), ("validation", y_val), ("test", y_test)):
        print(
            f"  {name:10s}: total={len(y):,} | "
            f"signal={int(np.sum(y==1)):,} | background={int(np.sum(y==0)):,}"
        )

    # ------------------------------------------------------------------
    # Same preprocessing philosophy as current event NN, now for all 24
    # inputs.  Fit ONLY on training.
    # ------------------------------------------------------------------
    medians, means, stds = fit_preprocessing(X_train)
    X_train = apply_preprocessing(X_train, medians, means, stds)
    X_val = apply_preprocessing(X_val, medians, means, stds)
    X_test = apply_preprocessing(X_test, medians, means, stds)

    np.savez(
        args.output_dir / "preprocessing.npz",
        medians=medians,
        means=means,
        stds=stds,
        feature_names=np.asarray(AUGMENTED_FEATURE_NAMES),
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
        input_dim=24,
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

    save_roc(
        val_labels,
        val_scores,
        val_auc,
        args.output_dir / "validation_roc.png",
        "Wide NN + two transformer jet scores: validation ROC",
    )
    save_roc(
        test_labels,
        test_scores,
        test_auc,
        args.output_dir / "test_roc.png",
        "Wide NN + two transformer jet scores: test ROC",
    )

    pd.DataFrame(
        {
            "label": test_labels.astype(np.int8),
            "event_score": test_scores.astype(np.float32),
        }
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)

    checkpoint_out = {
        "architecture": "wide_plus_two_transformer_scores",
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "batch_norm": BATCH_NORM,
        "activation": ACTIVATION,
        "input_dim": 24,
        "feature_names": AUGMENTED_FEATURE_NAMES,
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
            "score_definition": "sigmoid(logit) = P(gluon)",
        },
    }
    torch.save(checkpoint_out, args.output_dir / "best_model.pt")

    metrics = {
        "model": "wide_event_nn_plus_two_frozen_transformer_scores",
        "feature_count": 24,
        "features": AUGMENTED_FEATURE_NAMES,
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
        "transformer_checkpoin t": str(args.transformer_checkpoint),
        "transformer_checkpoint_epoch": checkpoint.get("epoch"),
        "transformer_checkpoint_val_loss": checkpoint.get("val_loss"),
        "transformer_frozen": True,
        "transformer_scores": TRANSFORMER_SCORE_FEATURES,
    }
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, default=str)

    print("\n" + "=" * 88)
    print("RESULT")
    print("=" * 88)
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
    print("Transformer score caches are retained for fast reruns.")


if __name__ == "__main__":
    main()

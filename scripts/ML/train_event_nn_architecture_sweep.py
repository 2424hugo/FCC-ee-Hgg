#!/usr/bin/env python3
"""
Train and compare multiple event-level MLP architectures.

The program:

1. Loads the same 22 scalar event features used by the event-level BDT.
2. Uses only train and validation splits.
3. Fits preprocessing using the training data only.
4. Trains several neural-network architectures.
5. Uses early stopping based on validation ROC AUC.
6. Calculates background efficiency at fixed signal efficiencies.
7. Produces:
       - architecture_comparison.csv
       - architecture_comparison.png
       - validation_roc_comparison.png
       - training histories
       - individual model checkpoints
       - best_model.pt
       - best_model_metadata.json

The test set is deliberately NOT read. It should only be evaluated once the
architecture and hyperparameters have been fixed.
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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Neural-network architectures
# =============================================================================

ARCHITECTURES = {
    "tiny": {
        "hidden": [32],
        "dropout": 0.10,
    },
    "shallow": {
        "hidden": [64, 32],
        "dropout": 0.10,
    },
    "baseline": {
        "hidden": [128, 64, 32],
        "dropout": 0.15,
    },
    "wide": {
        "hidden": [256, 128, 64],
        "dropout": 0.15,
    },
    "deep": {
        "hidden": [256, 128, 64, 32],
        "dropout": 0.15,
    },
    "very_deep": {
        "hidden": [256, 256, 128, 64, 32],
        "dropout": 0.20,
    },
    "bottleneck": {
        "hidden": [128, 64, 16],
        "dropout": 0.10,
    },
    "large": {
        "hidden": [512, 256, 128, 64],
        "dropout": 0.20,
    },
}


# =============================================================================
# Features
# =============================================================================

PAIR_FEATURE_COLUMNS = (
    "jet_energy",
    "jet_mass",
    "constituent_multiplicity",
    "e2_beta_0p2",
    "e3_beta_0p2",
    "jet_pt",
    "jet_p",
    "c2_beta_0p2",
    "d2_beta_0p2",
    "jet_theta",
)


FEATURE_NAMES = [
    "event_invariant_mass",
    "n_jets_original",
]

for feature in PAIR_FEATURE_COLUMNS:
    FEATURE_NAMES.extend(
        [
            f"leading_{feature}",
            f"subleading_{feature}",
        ]
    )

EXPECTED_FEATURES = 22

if len(FEATURE_NAMES) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Expected {EXPECTED_FEATURES} features, "
        f"but constructed {len(FEATURE_NAMES)}."
    )


# =============================================================================
# Argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Compare multiple event-level neural-network architectures."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("cache/analysis_dataset"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ml/nn_architecture_sweep"),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help=(
            "Minimum increase in validation AUC required to count "
            "as an improvement."
        ),
    )

    parser.add_argument(
        "--max-events-per-class",
        type=int,
        default=0,
        help=(
            "Maximum number of events loaded per class and split. "
            "0 means use all available events."
        ),
    )

    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    parser.add_argument(
        "--architectures",
        nargs="+",
        default=None,
        help=(
            "Optional subset of architecture names. "
            "By default all architectures are trained."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Reproducibility
# =============================================================================


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Dataset handling
# =============================================================================


def parquet_files(directory: Path) -> list[Path]:
    """Return real Parquet files, excluding EOS metadata directories."""

    files = sorted(
        path
        for path in directory.rglob("*.parquet")
        if path.is_file()
    )

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found beneath {directory}"
        )

    return files


def resolve_validation_directory(class_root: Path) -> Path:

    for name in ("validation", "val"):

        candidate = class_root / name

        if candidate.is_dir() and any(candidate.rglob("*.parquet")):
            return candidate

    raise FileNotFoundError(
        f"Could not find validation split below {class_root}"
    )


def numeric_array(array: pa.Array) -> np.ndarray:

    converted = pc.cast(
        array,
        pa.float32(),
        safe=False,
    )

    return np.asarray(
        converted.to_numpy(
            zero_copy_only=False,
        ),
        dtype=np.float32,
    )


def two_jet_array(
    array: pa.Array,
    feature_name: str,
) -> np.ndarray:

    lengths = pc.list_value_length(array)

    too_short = pc.any(
        pc.less(lengths, 2)
    ).as_py()

    if too_short:
        raise ValueError(
            f"{feature_name} contains an event with fewer than two jets."
        )

    first = numeric_array(
        pc.list_element(array, 0)
    )

    second = numeric_array(
        pc.list_element(array, 1)
    )

    return np.column_stack(
        (first, second)
    )


def validate_file_schema(files: list[Path]) -> None:

    required = {
        "event_invariant_mass",
        "n_jets_original",
        *PAIR_FEATURE_COLUMNS,
    }

    print("\nChecking Parquet schemas...")

    for number, path in enumerate(files, start=1):

        schema = pq.ParquetFile(path).schema_arrow

        fields = {
            field.name
            for field in schema
        }

        missing = required - fields

        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        print(
            f"\r  checked {number}/{len(files)} files",
            end="",
            flush=True,
        )

    print("\nSchema check passed.")


def load_parquet_matrix(
    files: list[Path],
    max_events: int,
    batch_size: int,
    description: str,
) -> np.ndarray:

    columns = [
        "event_invariant_mass",
        "n_jets_original",
        *PAIR_FEATURE_COLUMNS,
    ]

    blocks = []

    events_loaded = 0

    print(
        f"\nLoading {description} "
        f"from {len(files)} shard(s)"
    )

    for file_number, path in enumerate(
        files,
        start=1,
    ):

        parquet_file = pq.ParquetFile(path)

        for batch in parquet_file.iter_batches(
            columns=columns,
            batch_size=batch_size,
            use_threads=True,
        ):

            arrays = {
                name: batch.column(
                    batch.schema.get_field_index(name)
                )
                for name in columns
            }

            event_mass = numeric_array(
                arrays["event_invariant_mass"]
            )

            n_jets = numeric_array(
                arrays["n_jets_original"]
            )

            jet_values = {
                name: two_jet_array(
                    arrays[name],
                    name,
                )
                for name in PAIR_FEATURE_COLUMNS
            }

            # -------------------------------------------------------------
            # Order jets consistently by energy.
            # Jet 0 = leading jet.
            # Jet 1 = subleading jet.
            # -------------------------------------------------------------

            energies = jet_values["jet_energy"]

            swap = (
                np.isfinite(energies[:, 0])
                & np.isfinite(energies[:, 1])
                & (energies[:, 1] > energies[:, 0])
            )

            if np.any(swap):

                for values in jet_values.values():

                    values[swap] = values[
                        swap
                    ][:, ::-1]

            feature_columns = [
                event_mass,
                n_jets,
            ]

            for name in PAIR_FEATURE_COLUMNS:

                feature_columns.append(
                    jet_values[name][:, 0]
                )

                feature_columns.append(
                    jet_values[name][:, 1]
                )

            block = np.column_stack(
                feature_columns
            ).astype(
                np.float32,
                copy=False,
            )

            if max_events > 0:

                remaining = (
                    max_events
                    - events_loaded
                )

                if remaining <= 0:
                    break

                block = block[:remaining]

            blocks.append(block)

            events_loaded += len(block)

            if (
                max_events > 0
                and events_loaded >= max_events
            ):
                break

        print(
            f"\r  shard {file_number:>4}/"
            f"{len(files)} | "
            f"events loaded: {events_loaded:,}",
            end="",
            flush=True,
        )

        if (
            max_events > 0
            and events_loaded >= max_events
        ):
            break

    print()

    if not blocks:
        raise RuntimeError(
            f"No events loaded for {description}"
        )

    matrix = np.concatenate(
        blocks,
        axis=0,
    )

    print(
        f"{description}: {matrix.shape[0]:,} events"
    )

    return matrix


def combine_classes(
    signal: np.ndarray,
    background: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:

    X = np.concatenate(
        [signal, background],
        axis=0,
    )

    y = np.concatenate(
        [
            np.ones(
                len(signal),
                dtype=np.float32,
            ),
            np.zeros(
                len(background),
                dtype=np.float32,
            ),
        ]
    )

    rng = np.random.default_rng(seed)

    order = rng.permutation(len(y))

    return (
        X[order],
        y[order],
    )


# =============================================================================
# Preprocessing
# =============================================================================


def fit_preprocessing(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    X = X_train.astype(
        np.float64,
        copy=True,
    )

    X[~np.isfinite(X)] = np.nan

    medians = np.nanmedian(
        X,
        axis=0,
    )

    # Protect against pathological completely-empty columns.
    medians = np.where(
        np.isfinite(medians),
        medians,
        0.0,
    )

    missing_rows, missing_cols = np.where(
        np.isnan(X)
    )

    X[
        missing_rows,
        missing_cols,
    ] = medians[missing_cols]

    means = np.mean(
        X,
        axis=0,
    )

    stds = np.std(
        X,
        axis=0,
    )

    stds = np.where(
        stds > 1e-12,
        stds,
        1.0,
    )

    return (
        medians.astype(np.float32),
        means.astype(np.float32),
        stds.astype(np.float32),
    )


def apply_preprocessing(
    X: np.ndarray,
    medians: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:

    X = X.astype(
        np.float32,
        copy=True,
    )

    X[~np.isfinite(X)] = np.nan

    rows, cols = np.where(
        np.isnan(X)
    )

    if len(rows) > 0:
        X[rows, cols] = medians[cols]

    X -= means

    X /= stds

    return X


# =============================================================================
# Model
# =============================================================================


class EventMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dims,
        dropout=0.1,
        batch_norm=True,
        activation="relu",
    ):

        super().__init__()

        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }

        if activation not in activations:
            raise ValueError(
                f"Unknown activation: {activation}"
            )

        activation_cls = activations[
            activation
        ]

        layers = []

        previous = input_dim

        for width in hidden_dims:

            layers.append(
                nn.Linear(
                    previous,
                    width,
                )
            )

            if batch_norm:

                layers.append(
                    nn.BatchNorm1d(width)
                )

            layers.append(
                activation_cls()
            )

            if dropout > 0:

                layers.append(
                    nn.Dropout(dropout)
                )

            previous = width

        # Output is a LOGIT.
        # Do not apply sigmoid here because
        # BCEWithLogitsLoss does that internally.
        layers.append(
            nn.Linear(
                previous,
                1,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.network(
            x
        ).squeeze(1)


# =============================================================================
# Data loaders
# =============================================================================


def make_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size,
    num_workers,
    seed,
):

    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )

    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    generator = torch.Generator()

    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
    )


# =============================================================================
# Prediction
# =============================================================================


@torch.no_grad()
def predict(
    model,
    loader,
    device,
) -> tuple[np.ndarray, np.ndarray]:

    model.eval()

    all_scores = []
    all_labels = []

    for features, labels in loader:

        features = features.to(
            device,
            non_blocking=True,
        )

        logits = model(features)

        scores = torch.sigmoid(
            logits
        )

        all_scores.append(
            scores.cpu().numpy()
        )

        all_labels.append(
            labels.numpy()
        )

    return (
        np.concatenate(all_scores),
        np.concatenate(all_labels),
    )


# =============================================================================
# Operating points
# =============================================================================


def fixed_signal_efficiencies(
    y_true,
    scores,
    targets,
):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
    )

    results = {}

    for target in targets:

        index = np.argmin(
            np.abs(
                tpr - target
            )
        )

        background_efficiency = float(
            fpr[index]
        )

        results[target] = {
            "signal_efficiency": float(
                tpr[index]
            ),
            "background_efficiency":
                background_efficiency,
            "background_rejection": (
                float(
                    1.0
                    / background_efficiency
                )
                if background_efficiency > 0
                else float("inf")
            ),
            "threshold": float(
                thresholds[index]
            ),
        }

    return results


# =============================================================================
# Training
# =============================================================================


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    learning_rate,
    weight_decay,
    patience,
    min_delta,
    pos_weight,
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [pos_weight],
            dtype=torch.float32,
            device=device,
        )
    )

    use_amp = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_ap": [],
    }

    best_auc = -np.inf
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    for epoch in range(
        1,
        epochs + 1,
    ):

        epoch_start = time.time()

        # -------------------------------------------------------------
        # Training
        # -------------------------------------------------------------

        model.train()

        train_loss_sum = 0.0
        train_events = 0

        for features, labels in train_loader:

            features = features.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):

                logits = model(features)

                loss = criterion(
                    logits,
                    labels,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            batch_events = len(labels)

            train_loss_sum += (
                loss.item()
                * batch_events
            )

            train_events += (
                batch_events
            )

        train_loss = (
            train_loss_sum
            / train_events
        )

        # -------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------

        model.eval()

        val_loss_sum = 0.0
        val_events = 0

        all_scores = []
        all_labels = []

        with torch.no_grad():

            for features, labels in val_loader:

                features = features.to(
                    device,
                    non_blocking=True,
                )

                labels_device = labels.to(
                    device,
                    non_blocking=True,
                )

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp,
                ):

                    logits = model(
                        features
                    )

                    loss = criterion(
                        logits,
                        labels_device,
                    )

                scores = torch.sigmoid(
                    logits
                )

                batch_events = len(
                    labels
                )

                val_loss_sum += (
                    loss.item()
                    * batch_events
                )

                val_events += (
                    batch_events
                )

                all_scores.append(
                    scores.cpu().numpy()
                )

                all_labels.append(
                    labels.numpy()
                )

        val_loss = (
            val_loss_sum
            / val_events
        )

        val_scores = np.concatenate(
            all_scores
        )

        val_labels = np.concatenate(
            all_labels
        )

        val_auc = roc_auc_score(
            val_labels,
            val_scores,
        )

        val_ap = average_precision_score(
            val_labels,
            val_scores,
        )

        history["epoch"].append(
            epoch
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_auc"].append(
            val_auc
        )

        history["val_ap"].append(
            val_ap
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.6f} | "
            f"val loss {val_loss:.6f} | "
            f"val AUC {val_auc:.6f} | "
            f"val AP {val_ap:.6f} | "
            f"{elapsed:.1f}s"
        )

        # -------------------------------------------------------------
        # Early stopping based on validation AUC
        # -------------------------------------------------------------

        if (
            val_auc
            > best_auc + min_delta
        ):

            best_auc = val_auc
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "Early stopping: "
                f"no validation-AUC improvement "
                f"for {patience} epochs."
            )

            break

    if best_state is None:
        raise RuntimeError(
            "Training finished without a valid model state."
        )

    print(
        f"Best epoch: {best_epoch} | "
        f"best validation AUC: {best_auc:.6f}"
    )

    return (
        pd.DataFrame(history),
        best_state,
        best_epoch,
        best_auc,
    )


# =============================================================================
# Plotting
# =============================================================================


def save_training_history(
    history,
    name,
    output_path,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        label="Training",
    )

    axes[0].plot(
        history["epoch"],
        history["val_loss"],
        label="Validation",
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "BCE loss"
    )

    axes[0].set_title(
        f"{name}: loss"
    )

    axes[0].legend()

    axes[0].grid(
        alpha=0.25
    )

    axes[1].plot(
        history["epoch"],
        history["val_auc"],
        label="Validation ROC AUC",
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "ROC AUC"
    )

    axes[1].set_title(
        f"{name}: validation AUC"
    )

    axes[1].grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def save_roc_comparison(
    roc_results,
    output_path,
):

    fig, axis = plt.subplots(
        figsize=(8, 7)
    )

    for name, data in roc_results.items():

        axis.plot(
            data["fpr"],
            data["tpr"],
            linewidth=2,
            label=(
                f"{name} "
                f"(AUC={data['auc']:.4f})"
            ),
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    axis.set_xlabel(
        "Background efficiency"
    )

    axis.set_ylabel(
        "Signal efficiency"
    )

    axis.set_title(
        "Validation ROC: NN architecture comparison"
    )

    axis.set_xlim(
        0,
        1,
    )

    axis.set_ylim(
        0,
        1.01,
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        loc="lower right",
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def save_architecture_comparison(
    results_df,
    output_path,
):

    ordered = results_df.sort_values(
        "validation_auc",
        ascending=True,
    )

    fig, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.barh(
        ordered["architecture"],
        ordered["validation_auc"],
    )

    axis.set_xlabel(
        "Validation ROC AUC"
    )

    axis.set_ylabel(
        "Architecture"
    )

    axis.set_title(
        "Event-level neural-network architecture comparison"
    )

    minimum = max(
        0.0,
        ordered["validation_auc"].min()
        - 0.02,
    )

    axis.set_xlim(
        minimum,
        1.0,
    )

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


# =============================================================================
# Device
# =============================================================================


def resolve_device(
    requested: str,
) -> torch.device:

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# =============================================================================
# Main
# =============================================================================


def main():

    args = parse_args()

    set_seed(
        args.random_seed
    )

    device = resolve_device(
        args.device
    )

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir = (
        output_dir
        / "models"
    )

    history_dir = (
        output_dir
        / "histories"
    )

    plot_dir = (
        output_dir
        / "plots"
    )

    for directory in (
        model_dir,
        history_dir,
        plot_dir,
    ):

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("=" * 80)

    print(
        "Event-level neural-network architecture sweep"
    )

    print("=" * 80)

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"Features: {len(FEATURE_NAMES)}"
    )

    for i, feature in enumerate(
        FEATURE_NAMES,
        start=1,
    ):

        print(
            f"  {i:>2}. {feature}"
        )

    # -----------------------------------------------------------------
    # Architectures to train
    # -----------------------------------------------------------------

    if args.architectures is None:

        selected_architectures = (
            ARCHITECTURES
        )

    else:

        unknown = [
            name
            for name in args.architectures
            if name not in ARCHITECTURES
        ]

        if unknown:

            raise ValueError(
                "Unknown architecture(s): "
                f"{unknown}. "
                f"Available: {list(ARCHITECTURES)}"
            )

        selected_architectures = {
            name: ARCHITECTURES[name]
            for name in args.architectures
        }

    print(
        "\nArchitectures:"
    )

    for name, config in (
        selected_architectures.items()
    ):

        print(
            f"  {name:>12}: "
            f"{config['hidden']} "
            f"dropout={config['dropout']}"
        )

    # -----------------------------------------------------------------
    # Locate datasets
    # -----------------------------------------------------------------

    signal_root = (
        args.dataset_root
        / "signal"
    )

    background_root = (
        args.dataset_root
        / "background"
    )

    signal_train_files = parquet_files(
        signal_root / "train"
    )

    background_train_files = parquet_files(
        background_root / "train"
    )

    signal_val_dir = (
        resolve_validation_directory(
            signal_root
        )
    )

    background_val_dir = (
        resolve_validation_directory(
            background_root
        )
    )

    signal_val_files = parquet_files(
        signal_val_dir
    )

    background_val_files = parquet_files(
        background_val_dir
    )

    all_files = (
        signal_train_files
        + background_train_files
        + signal_val_files
        + background_val_files
    )

    validate_file_schema(
        all_files
    )

    # -----------------------------------------------------------------
    # Load train and validation
    # -----------------------------------------------------------------

    signal_train = load_parquet_matrix(
        signal_train_files,
        args.max_events_per_class,
        args.parquet_batch_size,
        "signal training",
    )

    background_train = load_parquet_matrix(
        background_train_files,
        args.max_events_per_class,
        args.parquet_batch_size,
        "background training",
    )

    signal_val = load_parquet_matrix(
        signal_val_files,
        args.max_events_per_class,
        args.parquet_batch_size,
        "signal validation",
    )

    background_val = load_parquet_matrix(
        background_val_files,
        args.max_events_per_class,
        args.parquet_batch_size,
        "background validation",
    )

    X_train, y_train = combine_classes(
        signal_train,
        background_train,
        args.random_seed,
    )

    X_val, y_val = combine_classes(
        signal_val,
        background_val,
        args.random_seed + 1,
    )

    print(
        "\nDataset summary"
    )

    print(
        f"Training:   {len(y_train):,}"
    )

    print(
        f"  signal:   "
        f"{int(np.sum(y_train == 1)):,}"
    )

    print(
        f"  background: "
        f"{int(np.sum(y_train == 0)):,}"
    )

    print(
        f"Validation: {len(y_val):,}"
    )

    print(
        f"  signal:   "
        f"{int(np.sum(y_val == 1)):,}"
    )

    print(
        f"  background: "
        f"{int(np.sum(y_val == 0)):,}"
    )

    # -----------------------------------------------------------------
    # Fit preprocessing using TRAINING DATA ONLY
    # -----------------------------------------------------------------

    print(
        "\nFitting preprocessing..."
    )

    (
        medians,
        means,
        stds,
    ) = fit_preprocessing(
        X_train
    )

    X_train = apply_preprocessing(
        X_train,
        medians,
        means,
        stds,
    )

    X_val = apply_preprocessing(
        X_val,
        medians,
        means,
        stds,
    )

    np.savez(
        output_dir
        / "preprocessing.npz",
        medians=medians,
        means=means,
        stds=stds,
        feature_names=np.asarray(
            FEATURE_NAMES
        ),
    )

    print(
        "Preprocessing saved."
    )

    # -----------------------------------------------------------------
    # Class weighting
    # -----------------------------------------------------------------

    n_signal_train = np.sum(
        y_train == 1
    )

    n_background_train = np.sum(
        y_train == 0
    )

    pos_weight = float(
        n_background_train
        / n_signal_train
    )

    print(
        f"\nBCE positive-class weight: "
        f"{pos_weight:.4f}"
    )

    # -----------------------------------------------------------------
    # Architecture sweep
    # -----------------------------------------------------------------

    results = []

    roc_results = {}

    for architecture_number, (
        name,
        config,
    ) in enumerate(
        selected_architectures.items(),
        start=1,
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"Architecture "
            f"{architecture_number}/"
            f"{len(selected_architectures)}: "
            f"{name}"
        )

        print(
            f"Hidden layers: "
            f"{config['hidden']}"
        )

        print(
            f"Dropout: "
            f"{config['dropout']}"
        )

        print(
            "=" * 80
        )

        # Reset seeds so differences mainly reflect architecture.
        set_seed(
            args.random_seed
        )

        train_loader, val_loader = (
            make_loaders(
                X_train,
                y_train,
                X_val,
                y_val,
                args.batch_size,
                args.num_workers,
                args.random_seed,
            )
        )

        model = EventMLP(
            input_dim=X_train.shape[1],
            hidden_dims=config[
                "hidden"
            ],
            dropout=config[
                "dropout"
            ],
            batch_norm=True,
            activation="relu",
        ).to(device)

        n_parameters = sum(
            parameter.numel()
            for parameter
            in model.parameters()
            if parameter.requires_grad
        )

        print(
            f"Trainable parameters: "
            f"{n_parameters:,}"
        )

        start_time = time.time()

        (
            history,
            best_state,
            best_epoch,
            best_training_auc,
        ) = train_model(
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

        training_time = (
            time.time()
            - start_time
        )

        model.load_state_dict(
            best_state
        )

        val_scores, val_labels = predict(
            model,
            val_loader,
            device,
        )

        auc = roc_auc_score(
            val_labels,
            val_scores,
        )

        ap = average_precision_score(
            val_labels,
            val_scores,
        )

        operating_points = (
            fixed_signal_efficiencies(
                val_labels,
                val_scores,
                targets=[
                    0.5,
                    0.7,
                    0.8,
                    0.9,
                ],
            )
        )

        fpr, tpr, thresholds = (
            roc_curve(
                val_labels,
                val_scores,
            )
        )

        roc_results[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
        }

        # -------------------------------------------------------------
        # Save model
        # -------------------------------------------------------------

        checkpoint = {
            "architecture": name,
            "hidden_dims": config[
                "hidden"
            ],
            "dropout": config[
                "dropout"
            ],
            "batch_norm": True,
            "activation": "relu",
            "input_dim": int(
                X_train.shape[1]
            ),
            "feature_names":
                FEATURE_NAMES,
            "model_state_dict":
                best_state,
            "validation_auc":
                float(auc),
            "validation_ap":
                float(ap),
            "best_epoch":
                int(best_epoch),
            "preprocessing": {
                "medians": medians,
                "means": means,
                "stds": stds,
            },
        }

        checkpoint_path = (
            model_dir
            / f"{name}.pt"
        )

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        history.to_csv(
            history_dir
            / f"{name}_history.csv",
            index=False,
        )

        save_training_history(
            history,
            name,
            plot_dir
            / f"{name}_training.png",
        )

        # -------------------------------------------------------------
        # Result table
        # -------------------------------------------------------------

        row = {
            "architecture": name,
            "hidden_layers":
                "-".join(
                    str(value)
                    for value
                    in config["hidden"]
                ),
            "dropout":
                config["dropout"],
            "parameters":
                n_parameters,
            "best_epoch":
                best_epoch,
            "training_time_seconds":
                training_time,
            "validation_auc":
                auc,
            "validation_ap":
                ap,
        }

        for target in (
            0.5,
            0.7,
            0.8,
            0.9,
        ):

            suffix = int(
                target * 100
            )

            point = (
                operating_points[target]
            )

            row[
                f"eps_s_{suffix}"
            ] = point[
                "signal_efficiency"
            ]

            row[
                f"eps_b_at_eps_s_{suffix}"
            ] = point[
                "background_efficiency"
            ]

            row[
                f"background_rejection_at_eps_s_{suffix}"
            ] = point[
                "background_rejection"
            ]

            row[
                f"threshold_at_eps_s_{suffix}"
            ] = point[
                "threshold"
            ]

        results.append(row)

        print(
            "\nValidation results"
        )

        print(
            f"  ROC AUC: "
            f"{auc:.6f}"
        )

        print(
            f"  Average precision: "
            f"{ap:.6f}"
        )

        for target, point in (
            operating_points.items()
        ):

            print(
                f"  eps_s={target:.1f}: "
                f"eps_b="
                f"{point['background_efficiency']:.6f} "
                f"| rejection="
                f"{point['background_rejection']:.2f}"
            )

        print(
            f"  Training time: "
            f"{training_time:.1f}s"
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Rank architectures
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "validation_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    results_path = (
        output_dir
        / "architecture_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ARCHITECTURE RANKING"
    )

    print(
        "=" * 80
    )

    display_columns = [
        "architecture",
        "parameters",
        "best_epoch",
        "validation_auc",
        "validation_ap",
        "eps_b_at_eps_s_50",
        "eps_b_at_eps_s_70",
        "eps_b_at_eps_s_80",
        "eps_b_at_eps_s_90",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Plots comparing all models
    # -----------------------------------------------------------------

    save_roc_comparison(
        roc_results,
        plot_dir
        / "validation_roc_comparison.png",
    )

    save_architecture_comparison(
        results_df,
        plot_dir
        / "architecture_comparison.png",
    )

    # -----------------------------------------------------------------
    # Save best model separately
    # -----------------------------------------------------------------

    best_architecture = (
        results_df.iloc[0]
    )

    best_name = (
        best_architecture[
            "architecture"
        ]
    )

    source_checkpoint = torch.load(
        model_dir
        / f"{best_name}.pt",
        map_location="cpu",
        weights_only=False,
    )

    torch.save(
        source_checkpoint,
        output_dir
        / "best_model.pt",
    )

    best_metadata = {
        "selection_metric":
            "validation_roc_auc",
        "architecture":
            best_name,
        "hidden_layers":
            ARCHITECTURES[
                best_name
            ]["hidden"],
        "dropout":
            ARCHITECTURES[
                best_name
            ]["dropout"],
        "parameters":
            int(
                best_architecture[
                    "parameters"
                ]
            ),
        "best_epoch":
            int(
                best_architecture[
                    "best_epoch"
                ]
            ),
        "validation_auc":
            float(
                best_architecture[
                    "validation_auc"
                ]
            ),
        "validation_ap":
            float(
                best_architecture[
                    "validation_ap"
                ]
            ),
        "background_efficiencies": {
            "eps_s_50":
                float(
                    best_architecture[
                        "eps_b_at_eps_s_50"
                    ]
                ),
            "eps_s_70":
                float(
                    best_architecture[
                        "eps_b_at_eps_s_70"
                    ]
                ),
            "eps_s_80":
                float(
                    best_architecture[
                        "eps_b_at_eps_s_80"
                    ]
                ),
            "eps_s_90":
                float(
                    best_architecture[
                        "eps_b_at_eps_s_90"
                    ]
                ),
        },
        "feature_names":
            FEATURE_NAMES,
        "test_split_used":
            False,
    }

    with open(
        output_dir
        / "best_model_metadata.json",
        "w",
    ) as file:

        json.dump(
            best_metadata,
            file,
            indent=2,
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BEST ARCHITECTURE"
    )

    print(
        "=" * 80
    )

    print(
        f"Architecture: "
        f"{best_name}"
    )

    print(
        f"Hidden layers: "
        f"{ARCHITECTURES[best_name]['hidden']}"
    )

    print(
        f"Validation AUC: "
        f"{best_architecture['validation_auc']:.6f}"
    )

    print(
        f"Validation AP: "
        f"{best_architecture['validation_ap']:.6f}"
    )

    print(
        "\nOutputs written to:"
    )

    print(
        f"  {output_dir}"
    )

    print(
        "\nImportant: the test dataset has NOT been evaluated."
    )

    print(
        "Freeze the architecture/hyperparameters before "
        "running the final test evaluation."
    )


if __name__ == "__main__":
    main()
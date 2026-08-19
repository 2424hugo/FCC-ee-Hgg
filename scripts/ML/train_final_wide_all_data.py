#!/usr/bin/env python3
"""
Final refit of the frozen wide event-level MLP using ALL cached training and
validation events.

The architecture and hyperparameters are already frozen from the previous
multi-seed study:

    hidden layers = [256, 128, 64]
    dropout       = 0.15
    activation    = ReLU
    batch norm    = True
    optimiser     = AdamW
    learning rate = 1e-3
    weight decay  = 1e-4

The training epoch count is NOT tuned on the test set. By default it is taken
as the median best_epoch from the previous five wide-network validation runs.

The test split is deliberately NOT read here. Evaluate the saved checkpoint
after training with scripts.ML.evaluate_frozen_nn_test.

Important:
    Since the test set has already been inspected previously, this should be
    described as a final refit with a frozen analysis procedure, not as a new
    independent model-selection test.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.ML.train_event_nn_architecture_sweep import (
    EventMLP,
    FEATURE_NAMES,
    apply_preprocessing,
    combine_classes,
    fit_preprocessing,
    load_parquet_matrix,
    parquet_files,
    resolve_validation_directory,
)


# =============================================================================
# Frozen architecture
# =============================================================================

HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.15
ACTIVATION = "relu"
BATCH_NORM = True


# =============================================================================
# Arguments
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Final refit of the frozen wide MLP using all train + validation data."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("cache/analysis_dataset"),
    )

    parser.add_argument(
        "--multiseed-root",
        type=Path,
        default=Path(
            "outputs/ml/nn_multiseed_sweep/wide"
        ),
        help=(
            "Directory containing seed_*/architecture_comparison.csv "
            "from the previous wide-network seed study."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/ml/nn_final_wide_all_data"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help=(
            "Fixed number of training epochs. "
            "0 means use the median best_epoch from the previous wide seed runs."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=100_000,
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

    return parser.parse_args()


# =============================================================================
# Utility
# =============================================================================


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested, but torch.cuda.is_available() is False."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# =============================================================================
# Determine fixed epoch count from previous validation study
# =============================================================================


def determine_epochs(
    multiseed_root: Path,
) -> tuple[int, list[int]]:

    best_epochs = []

    result_files = sorted(
        multiseed_root.glob(
            "seed_*/architecture_comparison.csv"
        )
    )

    if not result_files:
        raise FileNotFoundError(
            f"No previous wide-network results found beneath {multiseed_root}"
        )

    for path in result_files:

        df = pd.read_csv(path)

        if len(df) != 1:
            raise ValueError(
                f"Expected one row in {path}, found {len(df)}"
            )

        architecture = str(
            df.iloc[0]["architecture"]
        )

        if architecture != "wide":
            raise ValueError(
                f"{path} contains architecture={architecture!r}, "
                "but this final refit expects 'wide'."
            )

        best_epochs.append(
            int(
                df.iloc[0]["best_epoch"]
            )
        )

    median_epoch = int(
        round(
            float(
                np.median(best_epochs)
            )
        )
    )

    return median_epoch, best_epochs


# =============================================================================
# Final training
# =============================================================================


def train_fixed_epochs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    pos_weight: float,
) -> pd.DataFrame:

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

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        start = time.time()

        model.train()

        loss_sum = 0.0
        events_seen = 0

        correct = 0

        for features, labels in loader:

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

            batch_size = len(labels)

            loss_sum += (
                loss.item()
                * batch_size
            )

            events_seen += batch_size

            predictions = (
                logits >= 0
            ).float()

            correct += int(
                (
                    predictions == labels
                )
                .sum()
                .item()
            )

        epoch_loss = (
            loss_sum
            / events_seen
        )

        accuracy = (
            correct
            / events_seen
        )

        elapsed = (
            time.time()
            - start
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_accuracy": accuracy,
                "seconds": elapsed,
            }
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"loss {epoch_loss:.6f} | "
            f"acc {accuracy:.6f} | "
            f"{elapsed:.1f}s"
        )

    return pd.DataFrame(history)


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    args = parse_args()

    set_seed(
        args.random_seed
    )

    device = resolve_device(
        args.device
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )

    print(
        "FINAL WIDE MLP REFIT — ALL TRAIN + VALIDATION DATA"
    )

    print(
        "=" * 80
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        f"Frozen architecture: "
        f"{HIDDEN_DIMS}"
    )

    print(
        f"Dropout: {DROPOUT}"
    )

    print(
        f"Learning rate: "
        f"{args.learning_rate}"
    )

    print(
        f"Weight decay: "
        f"{args.weight_decay}"
    )

    # -----------------------------------------------------------------
    # Fix number of epochs from PREVIOUS validation results.
    # -----------------------------------------------------------------

    if args.epochs > 0:

        epochs = (
            args.epochs
        )

        previous_best_epochs = []

        epoch_source = (
            "explicit command-line value"
        )

    else:

        (
            epochs,
            previous_best_epochs,
        ) = determine_epochs(
            args.multiseed_root
        )

        epoch_source = (
            "median best_epoch from previous wide multi-seed study"
        )

    print(
        "\nTraining epoch choice"
    )

    if previous_best_epochs:

        print(
            f"  previous best epochs: "
            f"{previous_best_epochs}"
        )

    print(
        f"  final epoch count: {epochs}"
    )

    print(
        f"  source: {epoch_source}"
    )

    # -----------------------------------------------------------------
    # Locate all train + validation shards.
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
        signal_root
        / "train"
    )

    signal_val_files = parquet_files(
        resolve_validation_directory(
            signal_root
        )
    )

    background_train_files = parquet_files(
        background_root
        / "train"
    )

    background_val_files = parquet_files(
        resolve_validation_directory(
            background_root
        )
    )

    print(
        "\nDevelopment shards"
    )

    print(
        f"  signal train:       "
        f"{len(signal_train_files)}"
    )

    print(
        f"  signal validation:  "
        f"{len(signal_val_files)}"
    )

    print(
        f"  background train:   "
        f"{len(background_train_files)}"
    )

    print(
        f"  background validation: "
        f"{len(background_val_files)}"
    )

    # -----------------------------------------------------------------
    # Load ALL events.
    # max_events=0 means unlimited.
    # -----------------------------------------------------------------

    signal_train = load_parquet_matrix(
        signal_train_files,
        max_events=0,
        batch_size=args.parquet_batch_size,
        description="signal training",
    )

    signal_val = load_parquet_matrix(
        signal_val_files,
        max_events=0,
        batch_size=args.parquet_batch_size,
        description="signal validation",
    )

    background_train = load_parquet_matrix(
        background_train_files,
        max_events=0,
        batch_size=args.parquet_batch_size,
        description="background training",
    )

    background_val = load_parquet_matrix(
        background_val_files,
        max_events=0,
        batch_size=args.parquet_batch_size,
        description="background validation",
    )

    signal_all = np.concatenate(
        [
            signal_train,
            signal_val,
        ],
        axis=0,
    )

    background_all = np.concatenate(
        [
            background_train,
            background_val,
        ],
        axis=0,
    )

    del (
        signal_train,
        signal_val,
        background_train,
        background_val,
    )

    print(
        "\nCombined development sample"
    )

    print(
        f"  signal:     "
        f"{len(signal_all):,}"
    )

    print(
        f"  background: "
        f"{len(background_all):,}"
    )

    print(
        f"  total:      "
        f"{len(signal_all) + len(background_all):,}"
    )

    # -----------------------------------------------------------------
    # Combine classes.
    #
    # Unlike the 100k/class architecture sweep, this intentionally uses
    # ALL cached development events.
    # -----------------------------------------------------------------

    X_all, y_all = combine_classes(
        signal_all,
        background_all,
        args.random_seed,
    )

    del (
        signal_all,
        background_all,
    )

    # -----------------------------------------------------------------
    # Preprocessing is now fitted on the entire development sample.
    # -----------------------------------------------------------------

    print(
        "\nFitting preprocessing on ALL train + validation data..."
    )

    (
        medians,
        means,
        stds,
    ) = fit_preprocessing(
        X_all
    )

    X_all = apply_preprocessing(
        X_all,
        medians,
        means,
        stds,
    )

    np.savez(
        args.output_dir
        / "preprocessing.npz",
        medians=medians,
        means=means,
        stds=stds,
        feature_names=np.asarray(
            FEATURE_NAMES
        ),
    )

    # -----------------------------------------------------------------
    # Class weighting.
    #
    # We are now using the natural cached class ratio, therefore retain
    # BCE positive-class weighting.
    # -----------------------------------------------------------------

    n_signal = int(
        np.sum(
            y_all == 1
        )
    )

    n_background = int(
        np.sum(
            y_all == 0
        )
    )

    pos_weight = (
        n_background
        / n_signal
    )

    print(
        "\nTraining sample"
    )

    print(
        f"  signal:       "
        f"{n_signal:,}"
    )

    print(
        f"  background:   "
        f"{n_background:,}"
    )

    print(
        f"  pos_weight:   "
        f"{pos_weight:.6f}"
    )

    # -----------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------

    dataset = TensorDataset(
        torch.from_numpy(
            X_all
        ),
        torch.from_numpy(
            y_all
        ),
    )

    generator = torch.Generator()

    generator.manual_seed(
        args.random_seed
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
        generator=generator,
    )

    # -----------------------------------------------------------------
    # Frozen model
    # -----------------------------------------------------------------

    model = EventMLP(
        input_dim=X_all.shape[1],
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        batch_norm=BATCH_NORM,
        activation=ACTIVATION,
    ).to(device)

    n_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{n_parameters:,}"
    )

    print(
        "\nStarting final training..."
    )

    start_time = time.time()

    history = train_fixed_epochs(
        model=model,
        loader=loader,
        device=device,
        epochs=epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
    )

    total_training_time = (
        time.time()
        - start_time
    )

    # -----------------------------------------------------------------
    # Save final model
    # -----------------------------------------------------------------

    checkpoint = {
        "architecture":
            "wide_final_all_data",
        "hidden_dims":
            HIDDEN_DIMS,
        "dropout":
            DROPOUT,
        "batch_norm":
            BATCH_NORM,
        "activation":
            ACTIVATION,
        "input_dim":
            int(
                X_all.shape[1]
            ),
        "feature_names":
            FEATURE_NAMES,
        "model_state_dict":
            model.state_dict(),
        "preprocessing": {
            "medians":
                medians,
            "means":
                means,
            "stds":
                stds,
        },
        "training": {
            "epochs":
                epochs,
            "epoch_selection":
                epoch_source,
            "previous_best_epochs":
                previous_best_epochs,
            "batch_size":
                args.batch_size,
            "learning_rate":
                args.learning_rate,
            "weight_decay":
                args.weight_decay,
            "random_seed":
                args.random_seed,
            "signal_events":
                n_signal,
            "background_events":
                n_background,
            "positive_class_weight":
                pos_weight,
            "total_training_time_seconds":
                total_training_time,
            "used_training_split":
                True,
            "used_validation_split":
                True,
            "used_test_split":
                False,
        },
    }

    checkpoint_path = (
        args.output_dir
        / "final_model.pt"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    history.to_csv(
        args.output_dir
        / "training_history.csv",
        index=False,
    )

    metadata = {
        "architecture":
            "wide",
        "hidden_dims":
            HIDDEN_DIMS,
        "dropout":
            DROPOUT,
        "parameters":
            n_parameters,
        "epochs":
            epochs,
        "epoch_selection":
            epoch_source,
        "previous_best_epochs":
            previous_best_epochs,
        "signal_events":
            n_signal,
        "background_events":
            n_background,
        "total_events":
            n_signal
            + n_background,
        "pos_weight":
            pos_weight,
        "test_used_during_training":
            False,
    }

    with open(
        args.output_dir
        / "training_metadata.json",
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL REFIT COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Signal events used:     "
        f"{n_signal:,}"
    )

    print(
        f"Background events used: "
        f"{n_background:,}"
    )

    print(
        f"Epochs:                 "
        f"{epochs}"
    )

    print(
        f"Training time:          "
        f"{total_training_time:.1f}s"
    )

    print(
        f"Checkpoint:             "
        f"{checkpoint_path}"
    )

    print(
        "\nThe test split has NOT been read by this training script."
    )


if __name__ == "__main__":
    main()
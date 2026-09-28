#!/usr/bin/env python3
"""
Compare activation functions for the frozen wide event-level MLP.

Activations tested:
    ReLU
    GELU
    SiLU

The architecture and all other training hyperparameters are held fixed:

    22 -> 256 -> 128 -> 64 -> 1

Each activation is trained over several random seeds. Performance is compared
using:

    - validation ROC AUC
    - validation average precision
    - background efficiency at fixed signal efficiencies

The TEST split is deliberately NOT used.

This is intended as a controlled hyperparameter study. If differences between
activations are comparable to seed-to-seed variation, the conclusion should be
that activation choice has negligible impact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from scripts.ML.train_event_nn_architecture_sweep import (
    EventMLP,
    FEATURE_NAMES,
    apply_preprocessing,
    combine_classes,
    fit_preprocessing,
    fixed_signal_efficiencies,
    load_parquet_matrix,
    make_loaders,
    parquet_files,
    predict,
    resolve_validation_directory,
    set_seed,
    train_model,
)


# =============================================================================
# Fixed architecture
# =============================================================================

HIDDEN_DIMS = [256, 128, 64]

DROPOUT = 0.15

BATCH_NORM = True

ACTIVATIONS = [
    "relu",
    "gelu",
    "silu",
]


# =============================================================================
# Arguments
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Compare ReLU, GELU and SiLU activations for the frozen wide MLP."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "cache/analysis_dataset"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/ml/nn_activation_study"
        ),
    )

    parser.add_argument(
        "--max-events-per-class",
        type=int,
        default=100_000,
        help=(
            "Maximum number of signal/background events used in each "
            "train and validation split. 0 means all available events."
        ),
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
        "--patience",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            42,
            43,
            44,
            45,
            46,
        ],
    )

    parser.add_argument(
        "--activations",
        nargs="+",
        choices=[
            "relu",
            "gelu",
            "silu",
        ],
        default=ACTIVATIONS,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    return parser.parse_args()


# =============================================================================
# Device
# =============================================================================


def resolve_device(
    requested: str,
) -> torch.device:

    if requested == "cpu":

        return torch.device(
            "cpu"
        )

    if requested == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but CUDA is not available."
            )

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# =============================================================================
# Single training run
# =============================================================================


def train_one_model(
    activation: str,
    seed: int,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict:

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"ACTIVATION: {activation.upper()} | SEED: {seed}"
    )

    print(
        "=" * 90
    )

    # -----------------------------------------------------------------
    # Copy raw matrices so each run starts identically.
    # -----------------------------------------------------------------

    X_train = X_train_raw.copy()

    X_val = X_val_raw.copy()

    # -----------------------------------------------------------------
    # Training-only preprocessing.
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Class weighting.
    # -----------------------------------------------------------------

    n_signal = int(
        np.sum(
            y_train == 1
        )
    )

    n_background = int(
        np.sum(
            y_train == 0
        )
    )

    pos_weight = (
        n_background
        / n_signal
    )

    # -----------------------------------------------------------------
    # Reset random state.
    # -----------------------------------------------------------------

    set_seed(
        seed
    )

    train_loader, val_loader = (
        make_loaders(
            X_train,
            y_train,
            X_val,
            y_val,
            args.batch_size,
            args.num_workers,
            seed,
        )
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    model = EventMLP(
        input_dim=X_train.shape[1],
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        batch_norm=BATCH_NORM,
        activation=activation,
    ).to(
        device
    )

    n_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Architecture: "
        f"{X_train.shape[1]} -> "
        f"{' -> '.join(str(x) for x in HIDDEN_DIMS)} -> 1"
    )

    print(
        f"Activation: {activation}"
    )

    print(
        f"Parameters: {n_parameters:,}"
    )

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------

    (
        history,
        best_state,
        best_epoch,
        best_auc,
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

    model.load_state_dict(
        best_state
    )

    # -----------------------------------------------------------------
    # Final validation inference
    # -----------------------------------------------------------------

    val_scores, val_labels = predict(
        model,
        val_loader,
        device,
    )

    validation_auc = roc_auc_score(
        val_labels,
        val_scores,
    )

    validation_ap = average_precision_score(
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

    # -----------------------------------------------------------------
    # Save run outputs
    # -----------------------------------------------------------------

    run_dir = (
        output_dir
        / activation
        / f"seed_{seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_csv(
        run_dir
        / "training_history.csv",
        index=False,
    )

    checkpoint = {
        "activation":
            activation,
        "seed":
            seed,
        "hidden_dims":
            HIDDEN_DIMS,
        "dropout":
            DROPOUT,
        "batch_norm":
            BATCH_NORM,
        "input_dim":
            int(
                X_train.shape[1]
            ),
        "feature_names":
            FEATURE_NAMES,
        "parameters":
            n_parameters,
        "best_epoch":
            int(
                best_epoch
            ),
        "validation_auc":
            float(
                validation_auc
            ),
        "validation_ap":
            float(
                validation_ap
            ),
        "model_state_dict":
            best_state,
        "preprocessing": {
            "medians":
                medians,
            "means":
                means,
            "stds":
                stds,
        },
    }

    torch.save(
        checkpoint,
        run_dir
        / "best_model.pt",
    )

    result = {
        "activation":
            activation,
        "seed":
            seed,
        "parameters":
            n_parameters,
        "best_epoch":
            int(
                best_epoch
            ),
        "validation_auc":
            float(
                validation_auc
            ),
        "validation_ap":
            float(
                validation_ap
            ),
        "eps_b_at_eps_s_50":
            operating_points[
                0.5
            ][
                "background_efficiency"
            ],
        "eps_b_at_eps_s_70":
            operating_points[
                0.7
            ][
                "background_efficiency"
            ],
        "eps_b_at_eps_s_80":
            operating_points[
                0.8
            ][
                "background_efficiency"
            ],
        "eps_b_at_eps_s_90":
            operating_points[
                0.9
            ][
                "background_efficiency"
            ],
    }

    print(
        "\nResult"
    )

    print(
        f"  validation AUC: "
        f"{validation_auc:.6f}"
    )

    print(
        f"  validation AP:  "
        f"{validation_ap:.6f}"
    )

    print(
        f"  eps_b @ eps_s=0.5: "
        f"{result['eps_b_at_eps_s_50']:.6f}"
    )

    print(
        f"  eps_b @ eps_s=0.7: "
        f"{result['eps_b_at_eps_s_70']:.6f}"
    )

    print(
        f"  eps_b @ eps_s=0.8: "
        f"{result['eps_b_at_eps_s_80']:.6f}"
    )

    print(
        f"  eps_b @ eps_s=0.9: "
        f"{result['eps_b_at_eps_s_90']:.6f}"
    )

    if device.type == "cuda":

        torch.cuda.empty_cache()

    return result


# =============================================================================
# Summary
# =============================================================================


def summarise_results(
    results: pd.DataFrame,
) -> pd.DataFrame:

    metrics = [
        "validation_auc",
        "validation_ap",
        "eps_b_at_eps_s_50",
        "eps_b_at_eps_s_70",
        "eps_b_at_eps_s_80",
        "eps_b_at_eps_s_90",
        "best_epoch",
    ]

    rows = []

    for (
        activation,
        group,
    ) in results.groupby(
        "activation"
    ):

        row = {
            "activation":
                activation,
            "n_seeds":
                len(group),
            "parameters":
                int(
                    group[
                        "parameters"
                    ].iloc[0]
                ),
        }

        for metric in metrics:

            values = (
                group[
                    metric
                ]
                .astype(
                    float
                )
            )

            row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                values.std(
                    ddof=1
                )
                if len(values) > 1
                else 0.0
            )

        rows.append(
            row
        )

    summary = pd.DataFrame(
        rows
    )

    summary = summary.sort_values(
        "validation_auc_mean",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return summary


# =============================================================================
# Plots
# =============================================================================


def save_auc_plot(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:

    activations = list(
        summary[
            "activation"
        ]
    )

    x = np.arange(
        len(
            activations
        )
    )

    means = []

    stds = []

    for activation in activations:

        row = summary[
            summary[
                "activation"
            ]
            == activation
        ].iloc[
            0
        ]

        means.append(
            row[
                "validation_auc_mean"
            ]
        )

        stds.append(
            row[
                "validation_auc_std"
            ]
        )

    fig, axis = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    axis.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        capsize=6,
        label="Mean ± seed std",
    )

    for index, activation in enumerate(
        activations
    ):

        values = (
            results[
                results[
                    "activation"
                ]
                == activation
            ][
                "validation_auc"
            ]
            .to_numpy()
        )

        offsets = np.linspace(
            -0.07,
            0.07,
            len(values),
        )

        axis.scatter(
            index
            + offsets,
            values,
            alpha=0.7,
        )

    axis.set_xticks(
        x,
        [
            activation.upper()
            for activation
            in activations
        ],
    )

    axis.set_ylabel(
        "Validation ROC AUC"
    )

    axis.set_xlabel(
        "Activation function"
    )

    axis.set_title(
        "Activation-function comparison"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        fig
    )


def save_background_efficiency_plot(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:

    activations = list(
        summary[
            "activation"
        ]
    )

    x = np.arange(
        len(
            activations
        )
    )

    fig, axis = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    for target in [
        50,
        70,
        80,
        90,
    ]:

        mean_column = (
            f"eps_b_at_eps_s_{target}_mean"
        )

        std_column = (
            f"eps_b_at_eps_s_{target}_std"
        )

        means = []

        stds = []

        for activation in activations:

            row = summary[
                summary[
                    "activation"
                ]
                == activation
            ].iloc[
                0
            ]

            means.append(
                row[
                    mean_column
                ]
            )

            stds.append(
                row[
                    std_column
                ]
            )

        axis.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=(
                rf"$\epsilon_s="
                f"{target / 100:.1f}$"
            ),
        )

    axis.set_xticks(
        x,
        [
            activation.upper()
            for activation
            in activations
        ],
    )

    axis.set_ylabel(
        "Background efficiency"
    )

    axis.set_xlabel(
        "Activation function"
    )

    axis.set_title(
        "Background rejection vs activation function"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        fig
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    args = parse_args()

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        args.device
    )

    print(
        "=" * 90
    )

    print(
        "NN ACTIVATION-FUNCTION STUDY"
    )

    print(
        "=" * 90
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    print(
        f"Architecture: "
        f"{len(FEATURE_NAMES)} -> "
        f"{' -> '.join(str(x) for x in HIDDEN_DIMS)} -> 1"
    )

    print(
        f"Dropout: {DROPOUT}"
    )

    print(
        f"Activations: "
        f"{args.activations}"
    )

    print(
        f"Seeds: "
        f"{args.seeds}"
    )

    # -----------------------------------------------------------------
    # Locate data
    # -----------------------------------------------------------------

    signal_root = (
        args.dataset_root
        / "signal"
    )

    background_root = (
        args.dataset_root
        / "background"
    )

    signal_train_files = (
        parquet_files(
            signal_root
            / "train"
        )
    )

    background_train_files = (
        parquet_files(
            background_root
            / "train"
        )
    )

    signal_val_files = (
        parquet_files(
            resolve_validation_directory(
                signal_root
            )
        )
    )

    background_val_files = (
        parquet_files(
            resolve_validation_directory(
                background_root
            )
        )
    )

    # -----------------------------------------------------------------
    # Load once.
    # -----------------------------------------------------------------

    signal_train = (
        load_parquet_matrix(
            signal_train_files,
            args.max_events_per_class,
            args.parquet_batch_size,
            "signal training",
        )
    )

    background_train = (
        load_parquet_matrix(
            background_train_files,
            args.max_events_per_class,
            args.parquet_batch_size,
            "background training",
        )
    )

    signal_val = (
        load_parquet_matrix(
            signal_val_files,
            args.max_events_per_class,
            args.parquet_batch_size,
            "signal validation",
        )
    )

    background_val = (
        load_parquet_matrix(
            background_val_files,
            args.max_events_per_class,
            args.parquet_batch_size,
            "background validation",
        )
    )

    X_train, y_train = (
        combine_classes(
            signal_train,
            background_train,
            42,
        )
    )

    X_val, y_val = (
        combine_classes(
            signal_val,
            background_val,
            43,
        )
    )

    del (
        signal_train,
        background_train,
        signal_val,
        background_val,
    )

    print(
        "\nDataset"
    )

    print(
        f"  train:      "
        f"{len(y_train):,}"
    )

    print(
        f"  validation: "
        f"{len(y_val):,}"
    )

    # -----------------------------------------------------------------
    # Train all activation/seed combinations
    # -----------------------------------------------------------------

    results = []

    total_runs = (
        len(
            args.activations
        )
        * len(
            args.seeds
        )
    )

    run_number = 0

    for activation in args.activations:

        for seed in args.seeds:

            run_number += 1

            print(
                "\n"
                + "#" * 90
            )

            print(
                f"RUN "
                f"{run_number}/"
                f"{total_runs}"
            )

            print(
                "#" * 90
            )

            result = train_one_model(
                activation=activation,
                seed=seed,
                X_train_raw=X_train,
                y_train=y_train,
                X_val_raw=X_val,
                y_val=y_val,
                args=args,
                device=device,
                output_dir=output_dir,
            )

            results.append(
                result
            )

    # -----------------------------------------------------------------
    # Save all individual results
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        output_dir
        / "activation_seed_results.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Summary across seeds
    # -----------------------------------------------------------------

    summary = summarise_results(
        results_df
    )

    summary.to_csv(
        output_dir
        / "activation_summary.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ACTIVATION COMPARISON"
    )

    print(
        "=" * 100
    )

    display_columns = [
        "activation",
        "n_seeds",
        "validation_auc_mean",
        "validation_auc_std",
        "validation_ap_mean",
        "validation_ap_std",
        "eps_b_at_eps_s_50_mean",
        "eps_b_at_eps_s_70_mean",
        "eps_b_at_eps_s_80_mean",
        "eps_b_at_eps_s_90_mean",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Differences relative to ReLU
    # -----------------------------------------------------------------

    if "relu" in summary[
        "activation"
    ].values:

        relu_row = summary[
            summary[
                "activation"
            ]
            == "relu"
        ].iloc[
            0
        ]

        relu_auc = (
            relu_row[
                "validation_auc_mean"
            ]
        )

        relu_eps_b_70 = (
            relu_row[
                "eps_b_at_eps_s_70_mean"
            ]
        )

        summary[
            "delta_auc_vs_relu"
        ] = (
            summary[
                "validation_auc_mean"
            ]
            - relu_auc
        )

        summary[
            "delta_eps_b_70_vs_relu"
        ] = (
            summary[
                "eps_b_at_eps_s_70_mean"
            ]
            - relu_eps_b_70
        )

        summary.to_csv(
            output_dir
            / "activation_summary.csv",
            index=False,
        )

        print(
            "\nDifferences relative to ReLU:"
        )

        print(
            summary[
                [
                    "activation",
                    "delta_auc_vs_relu",
                    "delta_eps_b_70_vs_relu",
                ]
            ].to_string(
                index=False
            )
        )

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------

    save_auc_plot(
        results_df,
        summary,
        output_dir
        / "activation_auc_comparison.png",
    )

    save_background_efficiency_plot(
        summary,
        output_dir
        / "activation_background_efficiency.png",
    )

    # -----------------------------------------------------------------
    # JSON
    # -----------------------------------------------------------------

    best = summary.iloc[
        0
    ]

    summary_json = {
        "architecture": {
            "input_features":
                len(
                    FEATURE_NAMES
                ),
            "hidden_dims":
                HIDDEN_DIMS,
            "dropout":
                DROPOUT,
            "batch_norm":
                BATCH_NORM,
        },
        "activations":
            args.activations,
        "seeds":
            args.seeds,
        "best_by_mean_auc": {
            "activation":
                best[
                    "activation"
                ],
            "mean_auc":
                float(
                    best[
                        "validation_auc_mean"
                    ]
                ),
            "std_auc":
                float(
                    best[
                        "validation_auc_std"
                    ]
                ),
        },
        "test_split_used":
            False,
    }

    with open(
        output_dir
        / "activation_study_summary.json",
        "w",
    ) as file:

        json.dump(
            summary_json,
            file,
            indent=2,
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "BEST ACTIVATION BY MEAN VALIDATION AUC"
    )

    print(
        "=" * 100
    )

    print(
        f"{best['activation'].upper()}: "
        f"{best['validation_auc_mean']:.6f} "
        f"± "
        f"{best['validation_auc_std']:.6f}"
    )

    print(
        "\nOutputs:"
    )

    for filename in [
        "activation_seed_results.csv",
        "activation_summary.csv",
        "activation_study_summary.json",
        "activation_auc_comparison.png",
        "activation_background_efficiency.png",
    ]:

        print(
            f"  "
            f"{output_dir / filename}"
        )

    print(
        "\nThe TEST split was NOT used."
    )


if __name__ == "__main__":
    main()
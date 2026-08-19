#!/usr/bin/env python3
"""
Leave-one-feature-out ablation study for the event-level NN.

Purpose
-------
Measure how strongly the frozen wide MLP depends on each of the 22 input
variables.

Procedure
---------
1. Load the same training and validation events once.
2. Train the frozen wide architecture using all 22 features.
3. Repeat training 22 times, removing one feature at a time.
4. Compare:
       - validation ROC AUC
       - validation AP
       - background efficiency at fixed signal efficiencies
5. Rank features by the degradation in performance when removed.

The TEST split is deliberately never read.

Interpretation
--------------
For feature i:

    Delta AUC_i = AUC_all_features - AUC_without_i

A large positive Delta AUC means that removing that feature hurts the model,
so the NN depends relatively strongly on that feature.

Because correlated variables may substitute for one another, a small Delta AUC
does NOT necessarily mean that the feature contains no physics information.
"""

from __future__ import annotations

import argparse
import copy
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
            "Measure NN dependence on each of the 22 high-level variables "
            "using leave-one-feature-out retraining."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("cache/analysis_dataset"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/ml/nn_feature_dependence"
        ),
    )

    parser.add_argument(
        "--max-events-per-class",
        type=int,
        default=100_000,
        help=(
            "Maximum events per class for train and validation. "
            "0 means use all available events."
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
        "--random-seed",
        type=int,
        default=42,
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

    parser.add_argument(
        "--only-features",
        nargs="+",
        default=None,
        help=(
            "Optional subset of feature names to ablate. "
            "The full 22-feature baseline is always trained."
        ),
    )

    return parser.parse_args()


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
                "CUDA requested but CUDA is unavailable."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# =============================================================================
# Train one feature configuration
# =============================================================================


def train_configuration(
    name: str,
    selected_indices: list[int],
    X_train_full: np.ndarray,
    y_train: np.ndarray,
    X_val_full: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict:

    selected_names = [
        FEATURE_NAMES[index]
        for index in selected_indices
    ]

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"CONFIGURATION: {name}"
    )

    print(
        "=" * 90
    )

    print(
        f"Input features: "
        f"{len(selected_indices)}"
    )

    for number, feature in enumerate(
        selected_names,
        start=1,
    ):

        print(
            f"  {number:>2}. "
            f"{feature}"
        )

    # -----------------------------------------------------------------
    # Select columns
    # -----------------------------------------------------------------

    X_train = X_train_full[
        :,
        selected_indices,
    ].copy()

    X_val = X_val_full[
        :,
        selected_indices,
    ].copy()

    # -----------------------------------------------------------------
    # Preprocessing must be fitted independently for this configuration,
    # using training data only.
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
    # Same class weighting for every configuration.
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
    # Reset seed so all configurations start comparably.
    # -----------------------------------------------------------------

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
        input_dim=len(
            selected_indices
        ),
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
        batch_norm=BATCH_NORM,
        activation=ACTIVATION,
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
        f"\nTrainable parameters: "
        f"{n_parameters:,}"
    )

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

    configuration_dir = (
        output_dir
        / name
    )

    configuration_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_csv(
        configuration_dir
        / "training_history.csv",
        index=False,
    )

    checkpoint = {
        "configuration":
            name,
        "selected_indices":
            selected_indices,
        "feature_names":
            selected_names,
        "hidden_dims":
            HIDDEN_DIMS,
        "dropout":
            DROPOUT,
        "best_epoch":
            int(best_epoch),
        "validation_auc":
            float(auc),
        "validation_ap":
            float(ap),
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
        configuration_dir
        / "best_model.pt",
    )

    result = {
        "configuration":
            name,
        "n_features":
            len(selected_indices),
        "parameters":
            n_parameters,
        "best_epoch":
            int(best_epoch),
        "validation_auc":
            float(auc),
        "validation_ap":
            float(ap),
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
        f"  AUC: "
        f"{auc:.6f}"
    )

    print(
        f"  AP:  "
        f"{ap:.6f}"
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
# Plotting
# =============================================================================


def save_auc_dependence_plot(
    results: pd.DataFrame,
    output_path: Path,
) -> None:

    feature_results = results[
        results["removed_feature"].notna()
    ].copy()

    feature_results = (
        feature_results
        .sort_values(
            "delta_auc",
            ascending=True,
        )
    )

    fig_height = max(
        8,
        0.38
        * len(feature_results),
    )

    fig, axis = plt.subplots(
        figsize=(
            10,
            fig_height,
        )
    )

    axis.barh(
        feature_results[
            "removed_feature"
        ],
        feature_results[
            "delta_auc"
        ],
    )

    axis.axvline(
        0.0,
        linewidth=1,
    )

    axis.set_xlabel(
        r"$\Delta$AUC = "
        r"AUC$_{22}$ - "
        r"AUC$_{\mathrm{without\ feature}}$"
    )

    axis.set_ylabel(
        "Removed feature"
    )

    axis.set_title(
        "NN leave-one-feature-out dependence"
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


def save_background_dependence_plot(
    results: pd.DataFrame,
    output_path: Path,
) -> None:

    feature_results = results[
        results["removed_feature"].notna()
    ].copy()

    feature_results = (
        feature_results
        .sort_values(
            "delta_eps_b_70",
            ascending=True,
        )
    )

    fig_height = max(
        8,
        0.38
        * len(feature_results),
    )

    fig, axis = plt.subplots(
        figsize=(
            10,
            fig_height,
        )
    )

    axis.barh(
        feature_results[
            "removed_feature"
        ],
        feature_results[
            "delta_eps_b_70"
        ],
    )

    axis.axvline(
        0.0,
        linewidth=1,
    )

    axis.set_xlabel(
        r"$\Delta\epsilon_b$ at "
        r"$\epsilon_s=0.7$"
    )

    axis.set_ylabel(
        "Removed feature"
    )

    axis.set_title(
        "Effect of removing each feature on background efficiency"
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
        "NN FEATURE DEPENDENCE STUDY"
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
        f"{HIDDEN_DIMS}"
    )

    print(
        f"Number of features: "
        f"{len(FEATURE_NAMES)}"
    )

    print(
        "\nFeatures:"
    )

    for number, feature in enumerate(
        FEATURE_NAMES,
        start=1,
    ):

        print(
            f"  {number:>2}. "
            f"{feature}"
        )

    # -----------------------------------------------------------------
    # Validate requested feature subset.
    # -----------------------------------------------------------------

    if args.only_features is None:

        features_to_remove = list(
            FEATURE_NAMES
        )

    else:

        unknown = [
            feature
            for feature
            in args.only_features
            if feature
            not in FEATURE_NAMES
        ]

        if unknown:

            raise ValueError(
                f"Unknown feature(s): "
                f"{unknown}"
            )

        features_to_remove = (
            args.only_features
        )

    # -----------------------------------------------------------------
    # Load development data ONCE.
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

    X_train_full, y_train = (
        combine_classes(
            signal_train,
            background_train,
            args.random_seed,
        )
    )

    X_val_full, y_val = (
        combine_classes(
            signal_val,
            background_val,
            args.random_seed + 1,
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
        f"  training:   "
        f"{len(y_train):,}"
    )

    print(
        f"  validation: "
        f"{len(y_val):,}"
    )

    # -----------------------------------------------------------------
    # Baseline with all 22 features
    # -----------------------------------------------------------------

    all_indices = list(
        range(
            len(
                FEATURE_NAMES
            )
        )
    )

    baseline = train_configuration(
        name="all_22_features",
        selected_indices=all_indices,
        X_train_full=X_train_full,
        y_train=y_train,
        X_val_full=X_val_full,
        y_val=y_val,
        args=args,
        device=device,
        output_dir=output_dir,
    )

    baseline[
        "removed_feature"
    ] = None

    baseline[
        "delta_auc"
    ] = 0.0

    baseline[
        "delta_ap"
    ] = 0.0

    baseline[
        "delta_eps_b_50"
    ] = 0.0

    baseline[
        "delta_eps_b_70"
    ] = 0.0

    baseline[
        "delta_eps_b_80"
    ] = 0.0

    baseline[
        "delta_eps_b_90"
    ] = 0.0

    results = [
        baseline
    ]

    baseline_auc = (
        baseline[
            "validation_auc"
        ]
    )

    baseline_ap = (
        baseline[
            "validation_ap"
        ]
    )

    # -----------------------------------------------------------------
    # Leave one feature out
    # -----------------------------------------------------------------

    for feature_number, feature in enumerate(
        features_to_remove,
        start=1,
    ):

        print(
            "\n"
            + "#" * 90
        )

        print(
            f"ABLATION "
            f"{feature_number}/"
            f"{len(features_to_remove)}"
        )

        print(
            f"Removing: {feature}"
        )

        print(
            "#" * 90
        )

        removed_index = (
            FEATURE_NAMES.index(
                feature
            )
        )

        selected_indices = [
            index
            for index
            in all_indices
            if index
            != removed_index
        ]

        safe_name = (
            feature
            .replace(
                "/",
                "_",
            )
            .replace(
                " ",
                "_",
            )
        )

        result = (
            train_configuration(
                name=(
                    f"without_"
                    f"{safe_name}"
                ),
                selected_indices=
                    selected_indices,
                X_train_full=
                    X_train_full,
                y_train=
                    y_train,
                X_val_full=
                    X_val_full,
                y_val=
                    y_val,
                args=
                    args,
                device=
                    device,
                output_dir=
                    output_dir,
            )
        )

        result[
            "removed_feature"
        ] = feature

        result[
            "delta_auc"
        ] = (
            baseline_auc
            - result[
                "validation_auc"
            ]
        )

        result[
            "delta_ap"
        ] = (
            baseline_ap
            - result[
                "validation_ap"
            ]
        )

        result[
            "delta_eps_b_50"
        ] = (
            result[
                "eps_b_at_eps_s_50"
            ]
            - baseline[
                "eps_b_at_eps_s_50"
            ]
        )

        result[
            "delta_eps_b_70"
        ] = (
            result[
                "eps_b_at_eps_s_70"
            ]
            - baseline[
                "eps_b_at_eps_s_70"
            ]
        )

        result[
            "delta_eps_b_80"
        ] = (
            result[
                "eps_b_at_eps_s_80"
            ]
            - baseline[
                "eps_b_at_eps_s_80"
            ]
        )

        result[
            "delta_eps_b_90"
        ] = (
            result[
                "eps_b_at_eps_s_90"
            ]
            - baseline[
                "eps_b_at_eps_s_90"
            ]
        )

        results.append(
            result
        )

    # -----------------------------------------------------------------
    # Final table
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        output_dir
        / "feature_ablation_results.csv",
        index=False,
    )

    ranking = (
        results_df[
            results_df[
                "removed_feature"
            ].notna()
        ]
        .sort_values(
            "delta_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    ranking.to_csv(
        output_dir
        / "feature_importance_ranking.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FEATURE DEPENDENCE RANKING"
    )

    print(
        "=" * 100
    )

    print(
        f"\nAll-feature validation AUC: "
        f"{baseline_auc:.6f}"
    )

    display_columns = [
        "removed_feature",
        "validation_auc",
        "delta_auc",
        "eps_b_at_eps_s_70",
        "delta_eps_b_70",
        "best_epoch",
    ]

    print(
        "\n"
        + ranking[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------

    save_auc_dependence_plot(
        results_df,
        output_dir
        / "feature_dependence_auc.png",
    )

    save_background_dependence_plot(
        results_df,
        output_dir
        / "feature_dependence_eps_b_70.png",
    )

    # -----------------------------------------------------------------
    # JSON summary
    # -----------------------------------------------------------------

    summary = {
        "architecture": {
            "hidden_dims":
                HIDDEN_DIMS,
            "dropout":
                DROPOUT,
        },
        "baseline": {
            "n_features":
                22,
            "validation_auc":
                baseline_auc,
            "validation_ap":
                baseline_ap,
        },
        "ranking_by_delta_auc": [
            {
                "rank":
                    rank + 1,
                "feature":
                    row[
                        "removed_feature"
                    ],
                "validation_auc_without":
                    float(
                        row[
                            "validation_auc"
                        ]
                    ),
                "delta_auc":
                    float(
                        row[
                            "delta_auc"
                        ]
                    ),
                "eps_b_70_without":
                    float(
                        row[
                            "eps_b_at_eps_s_70"
                        ]
                    ),
                "delta_eps_b_70":
                    float(
                        row[
                            "delta_eps_b_70"
                        ]
                    ),
            }
            for rank, (
                _,
                row,
            ) in enumerate(
                ranking.iterrows()
            )
        ],
        "test_split_used":
            False,
    }

    with open(
        output_dir
        / "feature_dependence_summary.json",
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "OUTPUTS"
    )

    print(
        "=" * 100
    )

    for filename in [
        "feature_ablation_results.csv",
        "feature_importance_ranking.csv",
        "feature_dependence_summary.json",
        "feature_dependence_auc.png",
        "feature_dependence_eps_b_70.png",
    ]:

        print(
            f"  "
            f"{output_dir / filename}"
        )

    print(
        "\nThe test split was NOT used."
    )


if __name__ == "__main__":
    main()
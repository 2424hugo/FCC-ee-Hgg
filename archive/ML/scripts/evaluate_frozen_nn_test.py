#!/usr/bin/env python3
"""
Evaluate the frozen event-level NN on the untouched test set and calculate
physically weighted signal significance.

Default physics normalisation
-----------------------------
Signal:
    e+e- -> H -> gg
    sigma = 23 ab

Background:
    e+e- -> qq
    sigma = 61 pb

Integrated luminosity:
    L = 10 ab^-1

Generator-level TEST counts:
    signal     = 200,000
    background = 1,200,000

The event weight for each selected test event is

    w = sigma * L / N_generated

where N_generated is the generator-level number of events in that test split.

The analysis dataset already contains the event-selection requirements, so the
weighted yield before applying the NN cut automatically includes the selection
efficiency.

IMPORTANT:
    This script reads the test set. Do not use its output to retune the model.
"""

from __future__ import annotations

import argparse
import json
import math
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
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

# Reuse the validated loader/model definitions from the architecture sweep.
from scripts.ML.train_event_nn_architecture_sweep import (
    EventMLP,
    apply_preprocessing,
    load_parquet_matrix,
    parquet_files,
)


# =============================================================================
# Arguments
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen event-level NN on the untouched test split "
            "and calculate physically weighted significance."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("cache/analysis_dataset"),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/ml/nn_multiseed_sweep/"
            "wide/seed_42/best_model.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/ml/nn_frozen_wide_test"
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
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    # -----------------------------------------------------------------
    # Physical normalisation
    #
    # Use fb and fb^-1 internally:
    #
    # 23 ab = 0.023 fb
    # 61 pb = 61,000 fb
    # 10 ab^-1 = 10,000 fb^-1
    # -----------------------------------------------------------------

    parser.add_argument(
        "--signal-cross-section-fb",
        type=float,
        default=0.023,
        help="H->gg signal cross section in fb.",
    )

    parser.add_argument(
        "--background-cross-section-fb",
        type=float,
        default=61_000.0,
        help="qq background cross section in fb.",
    )

    parser.add_argument(
        "--luminosity-fb",
        type=float,
        default=10_000.0,
        help="Integrated luminosity in fb^-1.",
    )

    parser.add_argument(
        "--signal-generated-events",
        type=int,
        default=200_000,
        help=(
            "Generator-level signal event count corresponding to "
            "the test split."
        ),
    )

    parser.add_argument(
        "--background-generated-events",
        type=int,
        default=1_200_000,
        help=(
            "Generator-level background event count corresponding to "
            "the test split."
        ),
    )

    parser.add_argument(
        "--systematics",
        type=float,
        nargs="+",
        default=[
            0.0,
            0.0001,
            0.001,
            0.01,
            0.05,
            0.10,
        ],
        help=(
            "Fractional background systematic uncertainties. "
            "For example 0.01 means 1%%."
        ),
    )

    parser.add_argument(
        "--min-background-mc",
        type=int,
        default=20,
        help=(
            "Minimum selected background MC events required for an "
            "operating point to be treated as statistically supported."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Device
# =============================================================================


def resolve_device(requested: str) -> torch.device:

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# =============================================================================
# Model loading
# =============================================================================


def load_model(
    checkpoint_path: Path,
    device: torch.device,
):

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required = [
        "hidden_dims",
        "dropout",
        "input_dim",
        "model_state_dict",
        "preprocessing",
    ]

    missing = [
        key
        for key in required
        if key not in checkpoint
    ]

    if missing:
        raise KeyError(
            f"Checkpoint is missing required fields: {missing}"
        )

    model = EventMLP(
        input_dim=int(
            checkpoint["input_dim"]
        ),
        hidden_dims=list(
            checkpoint["hidden_dims"]
        ),
        dropout=float(
            checkpoint["dropout"]
        ),
        batch_norm=bool(
            checkpoint.get(
                "batch_norm",
                True,
            )
        ),
        activation=str(
            checkpoint.get(
                "activation",
                "relu",
            )
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    preprocessing = checkpoint[
        "preprocessing"
    ]

    medians = np.asarray(
        preprocessing["medians"],
        dtype=np.float32,
    )

    means = np.asarray(
        preprocessing["means"],
        dtype=np.float32,
    )

    stds = np.asarray(
        preprocessing["stds"],
        dtype=np.float32,
    )

    return (
        model,
        checkpoint,
        medians,
        means,
        stds,
    )


# =============================================================================
# Prediction
# =============================================================================


@torch.no_grad()
def predict_scores(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:

    dataset = TensorDataset(
        torch.from_numpy(X)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    scores = []

    use_amp = (
        device.type == "cuda"
    )

    for (features,) in loader:

        features = features.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):

            logits = model(features)

        probabilities = torch.sigmoid(
            logits
        )

        scores.append(
            probabilities
            .cpu()
            .numpy()
        )

    return np.concatenate(scores)


# =============================================================================
# Significance
# =============================================================================


def asimov_significance_no_systematic(
    signal: float,
    background: float,
) -> float:
    """
    Standard counting-experiment Asimov significance:

        Z_A = sqrt(
            2 * [(S+B) ln(1 + S/B) - S]
        )
    """

    if signal <= 0:
        return 0.0

    if background <= 0:
        return float("inf")

    value = 2.0 * (
        (signal + background)
        * math.log1p(
            signal / background
        )
        - signal
    )

    return math.sqrt(
        max(value, 0.0)
    )


def asimov_significance_with_systematic(
    signal: float,
    background: float,
    fractional_background_uncertainty: float,
) -> float:
    """
    Asimov significance with a Gaussian background-normalisation uncertainty.

    Implements Eq. (71) of Cowan et al.,
    Eur. Phys. J. C 71 (2011) 1554.

    sigma_b = fractional_background_uncertainty * B
    """

    if signal <= 0:
        return 0.0

    if background <= 0:
        return float("inf")

    if fractional_background_uncertainty <= 0:

        return asimov_significance_no_systematic(
            signal,
            background,
        )

    sigma_b = (
        fractional_background_uncertainty
        * background
    )

    sigma_b2 = (
        sigma_b * sigma_b
    )

    numerator_1 = (
        (signal + background)
        * (
            background + sigma_b2
        )
    )

    denominator_1 = (
        background * background
        + (
            signal + background
        )
        * sigma_b2
    )

    term_1 = (
        (signal + background)
        * math.log(
            numerator_1
            / denominator_1
        )
    )

    numerator_2 = (
        sigma_b2
        * signal
    )

    denominator_2 = (
        background
        * (
            background + sigma_b2
        )
    )

    term_2 = (
        (background * background)
        / sigma_b2
        * math.log1p(
            numerator_2
            / denominator_2
        )
    )

    value = 2.0 * (
        term_1 - term_2
    )

    return math.sqrt(
        max(value, 0.0)
    )


# =============================================================================
# Threshold scan
# =============================================================================


def threshold_scan(
    signal_scores: np.ndarray,
    background_scores: np.ndarray,
    signal_weight: float,
    background_weight: float,
    systematics: list[float],
) -> pd.DataFrame:

    # -------------------------------------------------------------
    # Scan every unique model score.
    #
    # Sorting scores allows cumulative event counts without repeatedly
    # applying masks to the complete arrays.
    # -------------------------------------------------------------

    all_scores = np.concatenate(
        [
            signal_scores,
            background_scores,
        ]
    )

    # Add a no-cut point at threshold 0.
    thresholds = np.unique(
        np.concatenate(
            [
                np.asarray(
                    [0.0],
                    dtype=np.float64,
                ),
                all_scores.astype(
                    np.float64
                ),
            ]
        )
    )

    thresholds.sort()

    signal_sorted = np.sort(
        signal_scores
    )

    background_sorted = np.sort(
        background_scores
    )

    signal_total_mc = len(
        signal_scores
    )

    background_total_mc = len(
        background_scores
    )

    rows = []

    for threshold in thresholds:

        # Number of entries >= threshold.
        signal_index = np.searchsorted(
            signal_sorted,
            threshold,
            side="left",
        )

        background_index = np.searchsorted(
            background_sorted,
            threshold,
            side="left",
        )

        signal_mc = (
            signal_total_mc
            - signal_index
        )

        background_mc = (
            background_total_mc
            - background_index
        )

        signal_efficiency = (
            signal_mc
            / signal_total_mc
        )

        background_efficiency = (
            background_mc
            / background_total_mc
        )

        signal_yield = (
            signal_mc
            * signal_weight
        )

        background_yield = (
            background_mc
            * background_weight
        )

        if background_yield > 0:

            s_over_b = (
                signal_yield
                / background_yield
            )

            s_over_sqrt_b = (
                signal_yield
                / math.sqrt(
                    background_yield
                )
            )

        else:

            s_over_b = float("inf")
            s_over_sqrt_b = float("inf")

        row = {
            "threshold":
                float(threshold),
            "signal_mc_events":
                int(signal_mc),
            "background_mc_events":
                int(background_mc),
            "signal_efficiency":
                float(signal_efficiency),
            "background_efficiency":
                float(background_efficiency),
            "signal_yield":
                float(signal_yield),
            "background_yield":
                float(background_yield),
            "s_over_b":
                float(s_over_b),
            "s_over_sqrt_b":
                float(s_over_sqrt_b),
        }

        for systematic in systematics:

            percentage = (
                100.0
                * systematic
            )

            label = (
                f"asimov_z_bkg_syst_"
                f"{percentage:g}pct"
            )

            row[label] = (
                asimov_significance_with_systematic(
                    signal_yield,
                    background_yield,
                    systematic,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================


def save_roc_plot(
    y_true,
    scores,
    auc,
    output_path,
):

    fpr, tpr, _ = roc_curve(
        y_true,
        scores,
    )

    fig, axis = plt.subplots(
        figsize=(7, 6)
    )

    axis.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"Frozen NN test "
            f"(AUC={auc:.5f})"
        ),
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    axis.set_xlabel(
        "Background efficiency"
    )

    axis.set_ylabel(
        "Signal efficiency"
    )

    axis.set_title(
        "Frozen wide MLP: untouched test set"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend(
        loc="lower right"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def save_score_distribution(
    signal_scores,
    background_scores,
    output_path,
):

    fig, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.hist(
        signal_scores,
        bins=100,
        density=True,
        histtype="step",
        linewidth=2,
        label="Signal",
    )

    axis.hist(
        background_scores,
        bins=100,
        density=True,
        histtype="step",
        linewidth=2,
        label="Background",
    )

    axis.set_xlabel(
        "NN signal score"
    )

    axis.set_ylabel(
        "Density"
    )

    axis.set_title(
        "Frozen wide MLP: test-set score"
    )

    axis.set_yscale(
        "log"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def save_significance_plot(
    scan: pd.DataFrame,
    systematics: list[float],
    output_path: Path,
):

    fig, axis = plt.subplots(
        figsize=(8, 6)
    )

    for systematic in systematics:

        percentage = (
            100.0
            * systematic
        )

        column = (
            f"asimov_z_bkg_syst_"
            f"{percentage:g}pct"
        )

        axis.plot(
            scan["threshold"],
            scan[column],
            label=(
                f"{percentage:g}% "
                f"background syst."
            ),
        )

    axis.set_xlabel(
        "NN threshold"
    )

    axis.set_ylabel(
        r"Asimov significance $Z_A$"
    )

    axis.set_title(
        "Physically weighted NN significance"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

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

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(
        args.device
    )

    print(
        "=" * 80
    )

    print(
        "Frozen NN test evaluation"
    )

    print(
        "=" * 80
    )

    print(
        f"Checkpoint: {args.checkpoint}"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # -----------------------------------------------------------------
    # Load frozen checkpoint
    # -----------------------------------------------------------------

    (
        model,
        checkpoint,
        medians,
        means,
        stds,
    ) = load_model(
        args.checkpoint,
        device,
    )

    print(
        "\nFrozen model"
    )

    print(
        f"  architecture: "
        f"{checkpoint.get('architecture')}"
    )

    print(
        f"  hidden dims: "
        f"{checkpoint['hidden_dims']}"
    )

    print(
        f"  dropout: "
        f"{checkpoint['dropout']}"
    )

    print(
        f"  validation AUC stored in checkpoint: "
        f"{checkpoint.get('validation_auc')}"
    )

    # -----------------------------------------------------------------
    # Load untouched test data
    # -----------------------------------------------------------------

    signal_test_files = parquet_files(
        args.dataset_root
        / "signal"
        / "test"
    )

    background_test_files = parquet_files(
        args.dataset_root
        / "background"
        / "test"
    )

    print(
        "\nTest files"
    )

    print(
        f"  signal:     "
        f"{len(signal_test_files)}"
    )

    print(
        f"  background: "
        f"{len(background_test_files)}"
    )

    signal_test = load_parquet_matrix(
        signal_test_files,
        max_events=0,
        batch_size=args.parquet_batch_size,
        description="signal test",
    )

    background_test = load_parquet_matrix(
        background_test_files,
        max_events=0,
        batch_size=args.parquet_batch_size,
        description="background test",
    )

    # -----------------------------------------------------------------
    # Apply TRAINING preprocessing from checkpoint.
    # Nothing is fitted on the test data.
    # -----------------------------------------------------------------

    signal_test = apply_preprocessing(
        signal_test,
        medians,
        means,
        stds,
    )

    background_test = apply_preprocessing(
        background_test,
        medians,
        means,
        stds,
    )

    # -----------------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------------

    print(
        "\nRunning frozen NN inference..."
    )

    signal_scores = predict_scores(
        model,
        signal_test,
        args.batch_size,
        args.num_workers,
        device,
    )

    background_scores = predict_scores(
        model,
        background_test,
        args.batch_size,
        args.num_workers,
        device,
    )

    y_true = np.concatenate(
        [
            np.ones(
                len(signal_scores),
                dtype=np.int8,
            ),
            np.zeros(
                len(background_scores),
                dtype=np.int8,
            ),
        ]
    )

    scores = np.concatenate(
        [
            signal_scores,
            background_scores,
        ]
    )

    test_auc = roc_auc_score(
        y_true,
        scores,
    )

    test_ap = average_precision_score(
        y_true,
        scores,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "UNTOUCHED TEST PERFORMANCE"
    )

    print(
        "=" * 80
    )

    print(
        f"ROC AUC:           "
        f"{test_auc:.6f}"
    )

    print(
        f"Average precision: "
        f"{test_ap:.6f}"
    )

    # -----------------------------------------------------------------
    # Physical weights
    # -----------------------------------------------------------------

    signal_weight = (
        args.signal_cross_section_fb
        * args.luminosity_fb
        / args.signal_generated_events
    )

    background_weight = (
        args.background_cross_section_fb
        * args.luminosity_fb
        / args.background_generated_events
    )

    initial_signal_yield = (
        len(signal_scores)
        * signal_weight
    )

    initial_background_yield = (
        len(background_scores)
        * background_weight
    )

    signal_selection_efficiency = (
        len(signal_scores)
        / args.signal_generated_events
    )

    background_selection_efficiency = (
        len(background_scores)
        / args.background_generated_events
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PHYSICAL NORMALISATION"
    )

    print(
        "=" * 80
    )

    print(
        f"Signal cross section:     "
        f"{args.signal_cross_section_fb} fb"
    )

    print(
        f"Background cross section: "
        f"{args.background_cross_section_fb} fb"
    )

    print(
        f"Luminosity:               "
        f"{args.luminosity_fb} fb^-1"
    )

    print(
        f"Signal generated events:  "
        f"{args.signal_generated_events:,}"
    )

    print(
        f"Background generated:     "
        f"{args.background_generated_events:,}"
    )

    print(
        f"\nSignal event weight:      "
        f"{signal_weight:.8g}"
    )

    print(
        f"Background event weight:  "
        f"{background_weight:.8g}"
    )

    print(
        "\nAfter analysis preselection, before NN:"
    )

    print(
        f"  signal MC events:       "
        f"{len(signal_scores):,}"
    )

    print(
        f"  background MC events:   "
        f"{len(background_scores):,}"
    )

    print(
        f"  signal selection eff.:  "
        f"{signal_selection_efficiency:.6f}"
    )

    print(
        f"  background selection eff.: "
        f"{background_selection_efficiency:.6f}"
    )

    print(
        f"  physical signal yield:  "
        f"{initial_signal_yield:.6f}"
    )

    print(
        f"  physical background:    "
        f"{initial_background_yield:.6f}"
    )

    # -----------------------------------------------------------------
    # Threshold scan
    # -----------------------------------------------------------------

    print(
        "\nScanning NN threshold..."
    )

    scan = threshold_scan(
        signal_scores,
        background_scores,
        signal_weight,
        background_weight,
        args.systematics,
    )

    scan.to_csv(
        output_dir
        / "threshold_scan.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Statistically supported operating points.
    #
    # This avoids reporting an apparent optimum based on, for example,
    # only one surviving background MC event.
    # -------------------------------------------------------------

    supported = scan[
        scan["background_mc_events"]
        >= args.min_background_mc
    ].copy()

    if supported.empty:

        raise RuntimeError(
            "No threshold satisfies "
            f"--min-background-mc={args.min_background_mc}"
        )

    best_rows = []

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BEST STATISTICALLY SUPPORTED OPERATING POINTS"
    )

    print(
        "=" * 80
    )

    print(
        f"Minimum selected background MC events: "
        f"{args.min_background_mc}"
    )

    for systematic in args.systematics:

        percentage = (
            100.0
            * systematic
        )

        column = (
            f"asimov_z_bkg_syst_"
            f"{percentage:g}pct"
        )

        index = supported[
            column
        ].idxmax()

        best = supported.loc[
            index
        ]

        best_row = {
            "background_systematic":
                systematic,
            "threshold":
                float(
                    best["threshold"]
                ),
            "signal_mc_events":
                int(
                    best[
                        "signal_mc_events"
                    ]
                ),
            "background_mc_events":
                int(
                    best[
                        "background_mc_events"
                    ]
                ),
            "signal_efficiency":
                float(
                    best[
                        "signal_efficiency"
                    ]
                ),
            "background_efficiency":
                float(
                    best[
                        "background_efficiency"
                    ]
                ),
            "signal_yield":
                float(
                    best[
                        "signal_yield"
                    ]
                ),
            "background_yield":
                float(
                    best[
                        "background_yield"
                    ]
                ),
            "s_over_b":
                float(
                    best[
                        "s_over_b"
                    ]
                ),
            "s_over_sqrt_b":
                float(
                    best[
                        "s_over_sqrt_b"
                    ]
                ),
            "asimov_z":
                float(
                    best[column]
                ),
        }

        best_rows.append(
            best_row
        )

        print(
            f"\nBackground systematic: "
            f"{percentage:g}%"
        )

        print(
            f"  threshold:             "
            f"{best['threshold']:.8f}"
        )

        print(
            f"  signal efficiency:     "
            f"{best['signal_efficiency']:.8f}"
        )

        print(
            f"  background efficiency: "
            f"{best['background_efficiency']:.8e}"
        )

        print(
            f"  selected signal MC:    "
            f"{int(best['signal_mc_events']):,}"
        )

        print(
            f"  selected background MC:"
            f" {int(best['background_mc_events']):,}"
        )

        print(
            f"  signal yield:          "
            f"{best['signal_yield']:.6f}"
        )

        print(
            f"  background yield:      "
            f"{best['background_yield']:.6f}"
        )

        print(
            f"  S/B:                   "
            f"{best['s_over_b']:.8g}"
        )

        print(
            f"  S/sqrt(B):             "
            f"{best['s_over_sqrt_b']:.8g}"
        )

        print(
            f"  Asimov Z:              "
            f"{best[column]:.8g}"
        )

    best_df = pd.DataFrame(
        best_rows
    )

    best_df.to_csv(
        output_dir
        / "best_operating_points.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Predictions
    # -----------------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "label": y_true,
            "score": scores,
        }
    )

    prediction_df.to_csv(
        output_dir
        / "test_predictions.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Summary JSON
    # -----------------------------------------------------------------

    summary = {
        "checkpoint":
            str(args.checkpoint),
        "architecture":
            checkpoint.get(
                "architecture"
            ),
        "hidden_dims":
            checkpoint.get(
                "hidden_dims"
            ),
        "test_auc":
            float(test_auc),
        "test_average_precision":
            float(test_ap),
        "test_mc_events": {
            "signal":
                int(
                    len(signal_scores)
                ),
            "background":
                int(
                    len(background_scores)
                ),
        },
        "physics": {
            "signal_cross_section_fb":
                args.signal_cross_section_fb,
            "background_cross_section_fb":
                args.background_cross_section_fb,
            "luminosity_fb":
                args.luminosity_fb,
            "signal_generated_events":
                args.signal_generated_events,
            "background_generated_events":
                args.background_generated_events,
            "signal_event_weight":
                signal_weight,
            "background_event_weight":
                background_weight,
            "signal_selection_efficiency":
                signal_selection_efficiency,
            "background_selection_efficiency":
                background_selection_efficiency,
            "pre_nn_signal_yield":
                initial_signal_yield,
            "pre_nn_background_yield":
                initial_background_yield,
        },
        "minimum_background_mc":
            args.min_background_mc,
        "best_operating_points":
            best_rows,
    }

    with open(
        output_dir
        / "test_metrics.json",
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------

    save_roc_plot(
        y_true,
        scores,
        test_auc,
        output_dir
        / "test_roc.png",
    )

    save_score_distribution(
        signal_scores,
        background_scores,
        output_dir
        / "test_score_distribution.png",
    )

    save_significance_plot(
        supported,
        args.systematics,
        output_dir
        / "weighted_significance_vs_threshold.png",
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OUTPUTS"
    )

    print(
        "=" * 80
    )

    for filename in [
        "test_metrics.json",
        "test_predictions.csv",
        "threshold_scan.csv",
        "best_operating_points.csv",
        "test_roc.png",
        "test_score_distribution.png",
        "weighted_significance_vs_threshold.png",
    ]:

        print(
            f"  {output_dir / filename}"
        )

    print(
        "\nTEST SET HAS NOW BEEN EVALUATED."
    )

    print(
        "Do not use these test results to retune the architecture "
        "or hyperparameters."
    )


if __name__ == "__main__":
    main()
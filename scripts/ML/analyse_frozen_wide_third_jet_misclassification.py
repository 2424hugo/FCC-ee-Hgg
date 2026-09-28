#!/usr/bin/env python3
"""
Study whether misclassified events in the frozen wide event-level NN are
more likely to contain a third (or higher) reconstructed jet.

Default model
-------------
Frozen wide MLP selected from the validation study:

    outputs/ml/nn_multiseed_sweep/wide/seed_42/best_model.pt

The script evaluates the existing test split only. It does NOT retrain or
retune the network.

For each requested signal-efficiency operating point it reports, for

    TP : correctly identified signal
    FN : misidentified signal
    FP : misidentified background
    TN : correctly identified background

how many events have

    n_jets_original >= 3

and therefore contain at least one reconstructed jet that was not represented
explicitly by the two leading-jet inputs supplied to the NN.

Outputs
-------
    third_jet_summary.csv
    operating_points.csv
    event_level_diagnostics.csv      (optional; enabled by default)
    fraction_with_extra_jet.png
    n_jets_distribution.png

Typical usage from the FCC-ee-Hgg repository root:

    python scripts/ML/analyse_frozen_wide_third_jet_misclassification.py

Optional example:

    python scripts/ML/analyse_frozen_wide_third_jet_misclassification.py \
        --signal-efficiencies 0.5 0.7 0.8 0.9 \
        --fixed-thresholds 0.5 0.9419

IMPORTANT:
    This is post-hoc diagnostic analysis of the already-inspected test set.
    Do not use these results to retune the architecture or hyperparameters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# When this file is executed directly as
#
#     python scripts/ML/analyse_frozen_wide_third_jet_misclassification.py
#
# Python puts scripts/ML (rather than the repository root) on sys.path.
# Add the FCC-ee-Hgg repository root explicitly so the scripts.ML package
# can be imported reliably.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ML.train_event_nn_architecture_sweep import (
    EventMLP,
    FEATURE_NAMES,
    apply_preprocessing,
    load_parquet_matrix,
    parquet_files,
)


# =============================================================================
# Arguments
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
        help="Root of the cached signal/background train/validation/test dataset.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/ml/nn_multiseed_sweep/wide/seed_42/best_model.pt"
        ),
        help="Frozen wide NN checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ml/nn_frozen_wide_third_jet_study"),
    )

    parser.add_argument(
        "--signal-efficiencies",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.8, 0.9],
        help=(
            "Signal-efficiency operating points. Thresholds are obtained from "
            "the frozen NN signal-score distribution on the test sample."
        ),
    )

    parser.add_argument(
        "--fixed-thresholds",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Optional additional fixed NN score thresholds to study, e.g. "
            "--fixed-thresholds 0.5 0.9419"
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

    parser.add_argument(
        "--save-event-level",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write event-level labels, scores and n_jets_original to CSV.",
    )

    return parser.parse_args()


# =============================================================================
# Device and model loading
# =============================================================================


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested, but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    dict,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "hidden_dims",
        "dropout",
        "input_dim",
        "model_state_dict",
        "preprocessing",
    }
    missing = required - set(checkpoint)
    if missing:
        raise KeyError(
            f"Checkpoint is missing required fields: {sorted(missing)}"
        )

    checkpoint_features = checkpoint.get("feature_names")
    if checkpoint_features is not None:
        if list(checkpoint_features) != list(FEATURE_NAMES):
            raise ValueError(
                "Checkpoint feature ordering does not match FEATURE_NAMES in "
                "train_event_nn_architecture_sweep.py."
            )

    model = EventMLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
        batch_norm=bool(checkpoint.get("batch_norm", True)),
        activation=str(checkpoint.get("activation", "relu")),
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    preprocessing = checkpoint["preprocessing"]
    medians = np.asarray(preprocessing["medians"], dtype=np.float32)
    means = np.asarray(preprocessing["means"], dtype=np.float32)
    stds = np.asarray(preprocessing["stds"], dtype=np.float32)

    return model, checkpoint, medians, means, stds


@torch.no_grad()
def predict_scores(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    score_blocks: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for (features,) in loader:
        features = features.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(features)

        score_blocks.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(score_blocks).reshape(-1)


# =============================================================================
# Operating points and diagnostics
# =============================================================================


def threshold_for_signal_efficiency(
    signal_scores: np.ndarray,
    target_efficiency: float,
) -> float:
    """Return a score threshold giving approximately the requested signal efficiency.

    Events are accepted as signal when score >= threshold. Because scores can tie,
    the achieved efficiency can be slightly above the requested value.
    """

    if not (0.0 < target_efficiency <= 1.0):
        raise ValueError(
            f"Signal efficiency must lie in (0, 1], got {target_efficiency}."
        )

    n = len(signal_scores)
    keep = max(1, int(np.ceil(target_efficiency * n)))

    # kth largest value = element n-keep in ascending partition.
    index = n - keep
    threshold = np.partition(signal_scores, index)[index]
    return float(threshold)


def category_row(
    *,
    operating_point: str,
    target_signal_efficiency: float | None,
    threshold: float,
    category: str,
    label: str,
    scores: np.ndarray,
    n_jets: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | str | None]:
    selected_njets = n_jets[mask]
    total = int(mask.sum())

    if total == 0:
        return {
            "operating_point": operating_point,
            "target_signal_efficiency": target_signal_efficiency,
            "threshold": threshold,
            "category": category,
            "meaning": label,
            "events": 0,
            "n_exactly_2_jets": 0,
            "n_exactly_3_jets": 0,
            "n_4plus_jets": 0,
            "n_3plus_jets": 0,
            "fraction_3plus_jets": np.nan,
            "mean_n_jets_original": np.nan,
            "median_n_jets_original": np.nan,
            "mean_score": np.nan,
        }

    n_exactly_2 = int(np.sum(selected_njets == 2))
    n_exactly_3 = int(np.sum(selected_njets == 3))
    n_4plus = int(np.sum(selected_njets >= 4))
    n_3plus = int(np.sum(selected_njets >= 3))

    return {
        "operating_point": operating_point,
        "target_signal_efficiency": target_signal_efficiency,
        "threshold": threshold,
        "category": category,
        "meaning": label,
        "events": total,
        "n_exactly_2_jets": n_exactly_2,
        "n_exactly_3_jets": n_exactly_3,
        "n_4plus_jets": n_4plus,
        "n_3plus_jets": n_3plus,
        "fraction_3plus_jets": n_3plus / total,
        "mean_n_jets_original": float(np.mean(selected_njets)),
        "median_n_jets_original": float(np.median(selected_njets)),
        "mean_score": float(np.mean(scores[mask])),
    }


def analyse_threshold(
    *,
    operating_point: str,
    target_signal_efficiency: float | None,
    threshold: float,
    signal_scores: np.ndarray,
    background_scores: np.ndarray,
    signal_njets: np.ndarray,
    background_njets: np.ndarray,
) -> tuple[list[dict], dict]:
    # Signal is predicted when score >= threshold.
    tp = signal_scores >= threshold
    fn = ~tp
    fp = background_scores >= threshold
    tn = ~fp

    signal_eff = float(np.mean(tp))
    background_eff = float(np.mean(fp))

    rows = [
        category_row(
            operating_point=operating_point,
            target_signal_efficiency=target_signal_efficiency,
            threshold=threshold,
            category="TP",
            label="correct signal",
            scores=signal_scores,
            n_jets=signal_njets,
            mask=tp,
        ),
        category_row(
            operating_point=operating_point,
            target_signal_efficiency=target_signal_efficiency,
            threshold=threshold,
            category="FN",
            label="misidentified signal",
            scores=signal_scores,
            n_jets=signal_njets,
            mask=fn,
        ),
        category_row(
            operating_point=operating_point,
            target_signal_efficiency=target_signal_efficiency,
            threshold=threshold,
            category="FP",
            label="misidentified background",
            scores=background_scores,
            n_jets=background_njets,
            mask=fp,
        ),
        category_row(
            operating_point=operating_point,
            target_signal_efficiency=target_signal_efficiency,
            threshold=threshold,
            category="TN",
            label="correct background",
            scores=background_scores,
            n_jets=background_njets,
            mask=tn,
        ),
    ]

    op_row = {
        "operating_point": operating_point,
        "target_signal_efficiency": target_signal_efficiency,
        "threshold": threshold,
        "achieved_signal_efficiency": signal_eff,
        "background_efficiency": background_eff,
        "background_rejection": (
            float("inf") if background_eff == 0.0 else 1.0 / background_eff
        ),
        "true_positive_signal": int(tp.sum()),
        "false_negative_signal": int(fn.sum()),
        "false_positive_background": int(fp.sum()),
        "true_negative_background": int(tn.sum()),
        "fn_with_3plus_jets": int(np.sum(fn & (signal_njets >= 3))),
        "fp_with_3plus_jets": int(np.sum(fp & (background_njets >= 3))),
        "fn_fraction_with_3plus_jets": (
            float(np.mean(signal_njets[fn] >= 3)) if np.any(fn) else np.nan
        ),
        "fp_fraction_with_3plus_jets": (
            float(np.mean(background_njets[fp] >= 3)) if np.any(fp) else np.nan
        ),
    }

    return rows, op_row


# =============================================================================
# Plots
# =============================================================================


def save_fraction_plot(summary: pd.DataFrame, output_path: Path) -> None:
    categories = ["TP", "FN", "FP", "TN"]
    labels = {
        "TP": "Correct signal",
        "FN": "Misidentified signal",
        "FP": "Misidentified background",
        "TN": "Correct background",
    }

    fig, axis = plt.subplots(figsize=(9, 6))

    operating_points = list(summary["operating_point"].drop_duplicates())
    x = np.arange(len(operating_points), dtype=float)
    width = 0.19

    for i, category in enumerate(categories):
        values = []
        for op in operating_points:
            subset = summary[
                (summary["operating_point"] == op)
                & (summary["category"] == category)
            ]
            values.append(
                float(subset.iloc[0]["fraction_3plus_jets"])
                if len(subset)
                else np.nan
            )

        offset = (i - 1.5) * width
        axis.bar(x + offset, values, width=width, label=labels[category])

    axis.set_xticks(x)
    axis.set_xticklabels(operating_points, rotation=20, ha="right")
    axis.set_ylabel(r"Fraction with $N_{\rm jets}\geq 3$")
    axis.set_xlabel("NN operating point")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_njets_distribution(
    signal_scores: np.ndarray,
    background_scores: np.ndarray,
    signal_njets: np.ndarray,
    background_njets: np.ndarray,
    threshold: float,
    operating_point: str,
    output_path: Path,
) -> None:
    tp = signal_scores >= threshold
    fn = ~tp
    fp = background_scores >= threshold
    tn = ~fp

    max_n = int(
        max(
            np.max(signal_njets),
            np.max(background_njets),
            4,
        )
    )
    bins = np.arange(1.5, max_n + 1.5, 1.0)

    fig, axis = plt.subplots(figsize=(8, 6))

    for values, label in [
        (signal_njets[tp], "Correct signal (TP)"),
        (signal_njets[fn], "Misidentified signal (FN)"),
        (background_njets[fp], "Misidentified background (FP)"),
        (background_njets[tn], "Correct background (TN)"),
    ]:
        if len(values) == 0:
            continue
        axis.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
        )

    axis.set_xlabel(r"Original reconstructed jet multiplicity $N_{\rm jets}$")
    axis.set_ylabel("Density")
    axis.set_title(f"Frozen wide NN: {operating_point}")
    axis.set_xticks(np.arange(2, max_n + 1))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    for efficiency in args.signal_efficiencies:
        if not (0.0 < efficiency <= 1.0):
            raise ValueError(
                f"All --signal-efficiencies must lie in (0, 1], got {efficiency}."
            )

    for threshold in args.fixed_thresholds:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"All --fixed-thresholds must lie in [0, 1], got {threshold}."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print("=" * 80)
    print("FROZEN WIDE NN — THIRD-JET MISCLASSIFICATION STUDY")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model, checkpoint, medians, means, stds = load_model(
        args.checkpoint,
        device,
    )

    print("\nFrozen model")
    print(f"  architecture: {checkpoint.get('architecture')}")
    print(f"  hidden dims:  {checkpoint['hidden_dims']}")
    print(f"  dropout:      {checkpoint['dropout']}")

    signal_files = parquet_files(args.dataset_root / "signal" / "test")
    background_files = parquet_files(args.dataset_root / "background" / "test")

    # Keep the RAW matrix before preprocessing so n_jets_original remains in
    # physical integer units. The feature ordering is defined by FEATURE_NAMES.
    signal_raw = load_parquet_matrix(
        signal_files,
        max_events=0,
        batch_size=args.parquet_batch_size,
        description="signal test",
    )
    background_raw = load_parquet_matrix(
        background_files,
        max_events=0,
        batch_size=args.parquet_batch_size,
        description="background test",
    )

    njet_index = FEATURE_NAMES.index("n_jets_original")
    signal_njets = np.rint(signal_raw[:, njet_index]).astype(np.int16)
    background_njets = np.rint(background_raw[:, njet_index]).astype(np.int16)

    if np.any(signal_njets < 2) or np.any(background_njets < 2):
        raise ValueError(
            "Found a test event with n_jets_original < 2, inconsistent with the "
            "analysis selection."
        )

    signal_X = apply_preprocessing(
        signal_raw.copy(), medians, means, stds
    )
    background_X = apply_preprocessing(
        background_raw.copy(), medians, means, stds
    )

    print("\nRunning frozen NN inference...")
    signal_scores = predict_scores(
        model,
        signal_X,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    background_scores = predict_scores(
        model,
        background_X,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    del signal_X, background_X

    if len(signal_scores) != len(signal_njets):
        raise RuntimeError("Signal score/event alignment failed.")
    if len(background_scores) != len(background_njets):
        raise RuntimeError("Background score/event alignment failed.")

    print("\nTest sample")
    print(f"  signal events:     {len(signal_scores):,}")
    print(f"  background events: {len(background_scores):,}")
    print(
        "  signal with >=3 jets:     "
        f"{np.sum(signal_njets >= 3):,} "
        f"({np.mean(signal_njets >= 3):.3%})"
    )
    print(
        "  background with >=3 jets: "
        f"{np.sum(background_njets >= 3):,} "
        f"({np.mean(background_njets >= 3):.3%})"
    )

    all_summary_rows: list[dict] = []
    all_operating_rows: list[dict] = []

    # -------------------------------------------------------------
    # Fixed-signal-efficiency operating points.
    # -------------------------------------------------------------
    for target in args.signal_efficiencies:
        threshold = threshold_for_signal_efficiency(signal_scores, target)
        name = f"eps_sig={target:.2f}"

        rows, op_row = analyse_threshold(
            operating_point=name,
            target_signal_efficiency=target,
            threshold=threshold,
            signal_scores=signal_scores,
            background_scores=background_scores,
            signal_njets=signal_njets,
            background_njets=background_njets,
        )
        all_summary_rows.extend(rows)
        all_operating_rows.append(op_row)

    # -------------------------------------------------------------
    # Optional fixed score thresholds.
    # -------------------------------------------------------------
    for threshold in args.fixed_thresholds:
        name = f"threshold={threshold:.6g}"

        rows, op_row = analyse_threshold(
            operating_point=name,
            target_signal_efficiency=None,
            threshold=float(threshold),
            signal_scores=signal_scores,
            background_scores=background_scores,
            signal_njets=signal_njets,
            background_njets=background_njets,
        )
        all_summary_rows.extend(rows)
        all_operating_rows.append(op_row)

    summary = pd.DataFrame(all_summary_rows)
    operating_points = pd.DataFrame(all_operating_rows)

    summary.to_csv(args.output_dir / "third_jet_summary.csv", index=False)
    operating_points.to_csv(args.output_dir / "operating_points.csv", index=False)

    if args.save_event_level:
        event_level = pd.concat(
            [
                pd.DataFrame(
                    {
                        "label": 1,
                        "class_name": "signal",
                        "score": signal_scores,
                        "n_jets_original": signal_njets,
                        "has_3plus_jets": signal_njets >= 3,
                    }
                ),
                pd.DataFrame(
                    {
                        "label": 0,
                        "class_name": "background",
                        "score": background_scores,
                        "n_jets_original": background_njets,
                        "has_3plus_jets": background_njets >= 3,
                    }
                ),
            ],
            ignore_index=True,
        )
        event_level.to_csv(
            args.output_dir / "event_level_diagnostics.csv",
            index=False,
        )

    save_fraction_plot(
        summary,
        args.output_dir / "fraction_with_extra_jet.png",
    )

    # Use eps_sig=0.70 for the detailed multiplicity plot when available,
    # otherwise use the first operating point supplied.
    preferred_name = "eps_sig=0.70"
    if preferred_name not in set(operating_points["operating_point"]):
        preferred_name = str(operating_points.iloc[0]["operating_point"])

    preferred = operating_points[
        operating_points["operating_point"] == preferred_name
    ].iloc[0]

    save_njets_distribution(
        signal_scores,
        background_scores,
        signal_njets,
        background_njets,
        threshold=float(preferred["threshold"]),
        operating_point=preferred_name,
        output_path=args.output_dir / "n_jets_distribution.png",
    )

    # -------------------------------------------------------------
    # Console summary focused on the two misidentified categories.
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MISIDENTIFICATION SUMMARY")
    print("=" * 80)

    for _, op in operating_points.iterrows():
        name = str(op["operating_point"])
        threshold = float(op["threshold"])

        fn = summary[
            (summary["operating_point"] == name)
            & (summary["category"] == "FN")
        ].iloc[0]
        fp = summary[
            (summary["operating_point"] == name)
            & (summary["category"] == "FP")
        ].iloc[0]

        print(f"\n{name}")
        print(f"  threshold: {threshold:.8f}")
        print(
            "  achieved eps_sig / eps_bkg: "
            f"{op['achieved_signal_efficiency']:.6f} / "
            f"{op['background_efficiency']:.6e}"
        )
        print(
            "  FN signal with >=3 jets: "
            f"{int(fn['n_3plus_jets']):,} / {int(fn['events']):,} "
            f"({fn['fraction_3plus_jets']:.3%})"
        )
        print(
            "  FP background with >=3 jets: "
            f"{int(fp['n_3plus_jets']):,} / {int(fp['events']):,} "
            f"({fp['fraction_3plus_jets']:.3%})"
        )

    print("\nOutputs")
    for name in [
        "third_jet_summary.csv",
        "operating_points.csv",
        "fraction_with_extra_jet.png",
        "n_jets_distribution.png",
    ]:
        print(f"  {args.output_dir / name}")

    if args.save_event_level:
        print(f"  {args.output_dir / 'event_level_diagnostics.csv'}")

    print(
        "\nInterpretation: compare FN against TP and FP against TN. A larger "
        "fraction with n_jets_original >= 3 in the misidentified categories "
        "would indicate that additional reconstructed radiation is associated "
        "with classification failures when only the leading two jets are "
        "represented explicitly."
    )


if __name__ == "__main__":
    main()
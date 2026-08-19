#!/usr/bin/env python3
"""Evaluate a saved 22-variable BDT on every validation Parquet shard.

Place this file beside ``train_bdt_22_variables.py``.  It loads the saved
model without refitting it, leaves the test split untouched, and reports the
background efficiency at fixed signal efficiencies.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from train_bdt_22_variables import (
    combine_classes,
    load_parquet_matrix,
    parquet_files,
    resolve_feature_definitions,
    resolve_validation_directory,
    validate_feature_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved BDT on all signal/background validation shards."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("cache/analysis_dataset"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/ml/bdt_22_variables/bdt_model.joblib"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ml/bdt_22_variables/all_validation_metrics.json"),
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--target-signal-efficiencies",
        type=float,
        nargs="+",
        default=[0.90, 0.70, 0.50, 0.49],
    )
    parser.add_argument("--random-state", type=int, default=43)
    return parser.parse_args()


def wilson_interval(successes: int, trials: int, z: float) -> tuple[float, float]:
    """Wilson binomial proportion interval (valid even for very small counts)."""
    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return centre - half_width, centre + half_width


def operating_point(
    target: float,
    false_positive_rate: np.ndarray,
    true_positive_rate: np.ndarray,
    thresholds: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | int | list[float]]:
    # ROC thresholds descend.  The first point reaching the requested TPR is
    # the tightest available cut with signal efficiency >= the target.
    candidates = np.flatnonzero(true_positive_rate >= target)
    if len(candidates) == 0:
        raise RuntimeError(f"ROC curve never reaches signal efficiency {target}")
    index = int(candidates[0])
    threshold = float(thresholds[index])

    signal_mask = labels == 1
    background_mask = ~signal_mask
    signal_survivors = int(np.count_nonzero(scores[signal_mask] >= threshold))
    background_survivors = int(np.count_nonzero(scores[background_mask] >= threshold))
    n_signal = int(np.count_nonzero(signal_mask))
    n_background = int(np.count_nonzero(background_mask))
    background_efficiency = background_survivors / n_background
    interval_68 = wilson_interval(background_survivors, n_background, 1.0)
    interval_95 = wilson_interval(background_survivors, n_background, 1.959963984540054)

    return {
        "target_signal_efficiency": target,
        "threshold": threshold,
        "signal_efficiency": signal_survivors / n_signal,
        "background_efficiency": background_efficiency,
        "background_rejection_factor": (
            math.inf if background_efficiency == 0.0 else 1.0 / background_efficiency
        ),
        "signal_survivors": signal_survivors,
        "signal_events": n_signal,
        "background_survivors": background_survivors,
        "background_events": n_background,
        "background_efficiency_wilson_68": list(interval_68),
        "background_efficiency_wilson_95": list(interval_95),
        "roc_fpr": float(false_positive_rate[index]),
        "roc_tpr": float(true_positive_rate[index]),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if any(not 0.0 < target <= 1.0 for target in args.target_signal_efficiencies):
        raise ValueError("Target signal efficiencies must lie in (0, 1]")

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_names = bundle["features"]
    definitions = resolve_feature_definitions(feature_names)

    directories = {
        "signal": resolve_validation_directory(args.dataset_root / "signal"),
        "background": resolve_validation_directory(args.dataset_root / "background"),
    }
    files = {name: parquet_files(path) for name, path in directories.items()}
    validate_feature_schema(files["signal"] + files["background"], definitions)

    loaded: dict[str, np.ndarray] = {}
    missing: dict[str, dict[str, int]] = {}
    infinities: dict[str, dict[str, int]] = {}
    for class_name in ("signal", "background"):
        matrix, class_missing, class_infinities = load_parquet_matrix(
            files=files[class_name],
            definitions=definitions,
            batch_size=args.batch_size,
            max_events=0,  # Zero means every event in every validation shard.
            description=f"{class_name} validation (all shards)",
        )
        loaded[class_name] = matrix
        missing[class_name] = class_missing
        infinities[class_name] = class_infinities

    features, labels = combine_classes(
        loaded["signal"], loaded["background"], args.random_state
    )
    scores = model.predict_proba(features)[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    points = [
        operating_point(target, fpr, tpr, thresholds, scores, labels)
        for target in args.target_signal_efficiencies
    ]

    result = {
        "model": str(args.model),
        "validation_signal_events": int(np.count_nonzero(labels == 1)),
        "validation_background_events": int(np.count_nonzero(labels == 0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "operating_points": points,
        "missing_values": missing,
        "infinities_converted_to_nan": infinities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")

    print("\nAll-validation results")
    print(f"  Signal events:          {result['validation_signal_events']:,}")
    print(f"  Background events:      {result['validation_background_events']:,}")
    print(f"  ROC AUC:                {result['roc_auc']:.6f}")
    print(f"  Average precision:      {result['average_precision']:.6f}")
    print("\nFixed signal-efficiency operating points")
    print(" target    threshold    eps_signal      eps_background    B survivors    rejection")
    for point in points:
        rejection = point["background_rejection_factor"]
        rejection_text = "infinite" if math.isinf(rejection) else f"{rejection:,.1f}"
        print(
            f" {point['target_signal_efficiency']:>6.2f}"
            f"  {point['threshold']:>11.6f}"
            f"  {point['signal_efficiency']:>12.6f}"
            f"  {point['background_efficiency']:>18.6e}"
            f"  {point['background_survivors']:>13,}"
            f"  {rejection_text:>11}"
        )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Final dissertation ML evaluation pipeline.

Purpose
-------
1. Load the frozen event-level NN checkpoint.
2. Evaluate it once on the untouched test split.
3. Load optional BDT / ParticleNet test predictions and make a combined ROC plot.
4. Plot leave-one-feature-out dependence results.
5. Scan the NN score threshold on the TEST set for diagnostics.
6. Report physical S, B, S/B and Asimov significance using the normalization manifest.
7. Evaluate the PRE-FROZEN validation-selected operating point on the test set.
8. Save dissertation-ready tables and figures.

Important statistical convention
--------------------------------
The test set is NOT used to choose the final threshold.  The final operating point
is the validation threshold stored in the NN joblib bundle.  The test threshold
scan is saved only as a diagnostic / sensitivity curve.

Run this script from the repository root, with scripts/ML on PYTHONPATH or place
this file in scripts/ML beside train_event_bdt_22_variables.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

try:
    import train_event_bdt_22_variables as common
except ImportError as exc:
    raise RuntimeError(
        "Could not import train_event_bdt_22_variables.py. "
        "Place this script in scripts/ML/ or add scripts/ML to PYTHONPATH."
    ) from exc


FIXED_SIGNAL_EFFICIENCIES = (0.50, 0.70, 0.80, 0.90)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--normalization-manifest", type=Path, required=True)
    p.add_argument("--nn-model", type=Path, required=True,
                   help="Saved event_nn_22_variables.joblib.")
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help=(
            "Optional test-prediction CSV for another model. Repeat as needed, e.g. "
            "--comparison BDT=results/bdt/test_predictions.csv "
            "--comparison ParticleNet=results/particlenet/predictions.csv. "
            "CSV must contain a binary truth column and a signal-score column."
        ),
    )
    p.add_argument(
        "--feature-ablation-csv",
        type=Path,
        default=None,
        help=(
            "Leave-one-feature-out CSV containing columns such as "
            "removed_feature, validation_auc, delta_auc, "
            "eps_b_at_eps_s_70, delta_eps_b_70."
        ),
    )

    p.add_argument("--thresholds", type=int, default=4001)
    p.add_argument(
        "--background-systematics",
        type=float,
        nargs="+",
        default=[0.0, 0.0001, 0.001, 0.01],
        help="Fractional background uncertainties used for Asimov Z.",
    )
    p.add_argument(
        "--min-background-neff",
        type=float,
        default=100.0,
        help="Minimum background effective MC count for a statistically supported point.",
    )
    p.add_argument(
        "--diagnostic-optimize-systematic",
        type=float,
        default=0.0001,
        help="Systematic used only to mark the best statistically supported TEST diagnostic point.",
    )
    p.add_argument(
        "--test-split-name",
        default="test",
        help="Name of the untouched split in the analysis dataset.",
    )
    return p.parse_args()


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [safe_json(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def transform_nn_features(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
) -> np.ndarray:
    imputer = bundle["imputer"]
    scaler = bundle["scaler"]
    x = imputer.transform(frame).astype(np.float32, copy=False)
    return np.asarray(scaler.transform(x), dtype=np.float32)


def reconstruct_nn_sources(
    bundle: dict[str, Any],
    dataset_root: Path,
    manifest: dict[str, Any],
):
    """
    Re-resolve exactly the saved feature definitions against the current parquet schema.
    """
    event_features = bundle.get("event_features")
    jet_features = bundle.get("base_jet_features")

    if event_features is None or jet_features is None:
        raise KeyError(
            "NN bundle does not contain event_features/base_jet_features. "
            "Use the model saved by train_event_nn_22_variables.py."
        )

    first_pattern = str(manifest["samples"][0]["path_glob"])
    first_file = common.sample_files(dataset_root, first_pattern, "train")[0]

    sources = common.resolve_feature_sources(
        list(event_features),
        list(jet_features),
        common.parquet_columns(first_file),
        first_file,
    )

    resolved_names = [source.output_name for source in sources]
    saved_names = list(bundle.get("feature_names", []))
    if saved_names and resolved_names != saved_names:
        raise RuntimeError(
            "Resolved feature ordering differs from the saved NN feature ordering.\n"
            f"Saved:    {saved_names}\n"
            f"Resolved: {resolved_names}"
        )
    return sources


def weighted_fixed_efficiency_points(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    targets=FIXED_SIGNAL_EFFICIENCIES,
) -> pd.DataFrame:
    sig = y == 1
    bkg = y == 0

    total_s = float(weights[sig].sum())
    total_b = float(weights[bkg].sum())
    if total_s <= 0 or total_b <= 0:
        raise ValueError("Both signal and background must have positive total test weight.")

    signal_scores = scores[sig]
    signal_weights = weights[sig]
    order = np.argsort(signal_scores)[::-1]
    cumulative = np.cumsum(signal_weights[order]) / total_s

    rows = []
    for target in targets:
        pos = min(
            int(np.searchsorted(cumulative, target, side="left")),
            len(order) - 1,
        )
        threshold = float(signal_scores[order[pos]])
        selected = scores >= threshold

        sw_s = float(weights[selected & sig].sum())
        sw_b = float(weights[selected & bkg].sum())
        sw2_b = float(np.square(weights[selected & bkg]).sum())

        eps_s = sw_s / total_s
        eps_b = sw_b / total_b
        rows.append(
            {
                "target_signal_efficiency": target,
                "threshold": threshold,
                "signal_efficiency": eps_s,
                "background_efficiency": eps_b,
                "background_rejection": np.inf if eps_b <= 0 else 1.0 / eps_b,
                "signal_yield": sw_s,
                "background_yield": sw_b,
                "s_over_b": np.inf if sw_b <= 0 else sw_s / sw_b,
                "background_neff": (
                    sw_b * sw_b / sw2_b if sw2_b > 0 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_threshold(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    systematics: list[float],
) -> dict[str, float]:
    selected = scores >= threshold
    sig = y == 1
    bkg = y == 0

    total_s = float(weights[sig].sum())
    total_b = float(weights[bkg].sum())

    s = float(weights[selected & sig].sum())
    b = float(weights[selected & bkg].sum())
    b2 = float(np.square(weights[selected & bkg]).sum())

    result = {
        "threshold": float(threshold),
        "signal_efficiency": s / total_s if total_s > 0 else np.nan,
        "background_efficiency": b / total_b if total_b > 0 else np.nan,
        "signal_yield": s,
        "background_yield": b,
        "s_over_b": s / b if b > 0 else np.inf,
        "background_neff": b * b / b2 if b2 > 0 else 0.0,
    }
    for frac in sorted(set(systematics)):
        result[f"asimov_z_bkg_syst_{frac:g}"] = common.asimov_significance(
            s, b, frac
        )
    return result


def infer_prediction_columns(frame: pd.DataFrame) -> tuple[str, str]:
    truth_candidates = (
        "label", "y_true", "truth", "target", "class", "is_signal"
    )
    score_candidates = (
        "signal_probability",
        "signal_score",
        "score",
        "probability",
        "prediction_score",
        "gluon_probability",
    )

    truth = next((c for c in truth_candidates if c in frame.columns), None)
    score = next((c for c in score_candidates if c in frame.columns), None)

    if truth is None:
        raise ValueError(
            f"Could not infer truth column from {list(frame.columns)}"
        )
    if score is None:
        numeric = [
            c for c in frame.columns
            if c != truth and pd.api.types.is_numeric_dtype(frame[c])
        ]
        # Avoid obvious index / hard prediction columns.
        numeric = [
            c for c in numeric
            if c.lower() not in {
                "dataset_index", "index", "event", "event_index", "prediction"
            }
        ]
        if len(numeric) == 1:
            score = numeric[0]
        else:
            raise ValueError(
                "Could not infer score column. Expected one of "
                f"{score_candidates}; columns are {list(frame.columns)}"
            )

    return truth, score


def parse_comparison(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(
            f"Comparison {spec!r} must have form NAME=CSV"
        )
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name:
        raise ValueError(f"Empty model name in comparison {spec!r}")
    return name, path


def roc_data(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    auc = float(roc_auc_score(y, scores, sample_weight=weights))
    ap = float(average_precision_score(y, scores, sample_weight=weights))
    fpr, tpr, _ = roc_curve(y, scores, sample_weight=weights)
    return fpr, tpr, auc, ap


def plot_nn_roc(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    op: pd.DataFrame,
    output: Path,
    auc: float,
) -> None:
    fpr, tpr, _ = roc_curve(y, scores, sample_weight=weights)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(fpr, tpr, linewidth=2.0, label=f"NN (AUC = {auc:.4f})")

    for row in op.itertuples(index=False):
        ax.scatter(
            row.background_efficiency,
            row.signal_efficiency,
            s=38,
            zorder=3,
        )
        ax.annotate(
            f"{100*row.target_signal_efficiency:.0f}%",
            (row.background_efficiency, row.signal_efficiency),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=9,
        )

    ax.plot([0, 1], [0, 1], "--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_score_distribution(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    output: Path,
) -> None:
    # Normalise each class independently: this is a shape comparison, not a yield plot.
    sig = y == 1
    bkg = y == 0

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    bins = np.linspace(0.0, 1.0, 51)

    ax.hist(
        scores[bkg],
        bins=bins,
        weights=weights[bkg] / weights[bkg].sum(),
        histtype="step",
        linewidth=1.8,
        label="Background",
    )
    ax.hist(
        scores[sig],
        bins=bins,
        weights=weights[sig] / weights[sig].sum(),
        histtype="step",
        linewidth=1.8,
        label="Signal",
    )

    ax.set_xlabel("NN signal score")
    ax.set_ylabel("Normalised events")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_model_comparison(
    model_curves: list[dict[str, Any]],
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for item in model_curves:
        ax.plot(
            item["fpr"],
            item["tpr"],
            linewidth=2.0,
            label=f'{item["name"]} (AUC = {item["auc"]:.4f})',
        )
    ax.plot([0, 1], [0, 1], "--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def feature_ablation_plots(path: Path, output_dir: Path) -> None:
    frame = pd.read_csv(path)

    required_auc = {"removed_feature", "delta_auc"}
    if not required_auc.issubset(frame.columns):
        raise ValueError(
            f"{path} is missing {required_auc - set(frame.columns)}"
        )

    # AUC dependence
    auc_frame = frame.sort_values("delta_auc", ascending=True)
    height = max(4.5, 0.34 * len(auc_frame) + 1.2)
    fig, ax = plt.subplots(figsize=(8.0, height))
    ax.barh(auc_frame["removed_feature"], auc_frame["delta_auc"])
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel(r"$\Delta$AUC = AUC(all) - AUC(with feature removed)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_dependence_delta_auc.png", dpi=300)
    plt.close(fig)

    # Background-efficiency dependence
    if "delta_eps_b_70" in frame.columns:
        bkg_frame = frame.sort_values("delta_eps_b_70", ascending=True)
        fig, ax = plt.subplots(figsize=(8.0, height))
        ax.barh(
            bkg_frame["removed_feature"],
            bkg_frame["delta_eps_b_70"],
        )
        ax.axvline(0.0, linewidth=1.0)
        ax.set_xlabel(
            r"$\Delta\epsilon_b$ at $\epsilon_s=0.70$ "
            r"(removed - all features)"
        )
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(
            output_dir / "feature_dependence_delta_epsb70.png",
            dpi=300,
        )
        plt.close(fig)


def plot_significance_scan(
    scan: pd.DataFrame,
    frozen_threshold: float,
    output: Path,
    systematic: float,
    diagnostic_best: pd.Series | None,
) -> None:
    z_col = f"asimov_z_bkg_syst_{systematic:g}"
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(
        scan["signal_efficiency"],
        scan[z_col],
        linewidth=2.0,
        label=f"Asimov Z ({100*systematic:g}% bkg. syst.)",
    )

    # Frozen validation-selected point on the x-axis position implied by test set.
    nearest = scan.iloc[
        (scan["threshold"] - frozen_threshold).abs().argmin()
    ]
    ax.scatter(
        nearest["signal_efficiency"],
        nearest[z_col],
        s=55,
        zorder=4,
        label="Frozen validation-selected threshold",
    )

    if diagnostic_best is not None:
        ax.scatter(
            diagnostic_best["signal_efficiency"],
            diagnostic_best[z_col],
            marker="x",
            s=70,
            zorder=4,
            label="Best supported test point (diagnostic only)",
        )

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Asimov significance")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.thresholds < 2:
        raise ValueError("--thresholds must be at least 2")
    if args.diagnostic_optimize_systematic not in args.background_systematics:
        args.background_systematics.append(
            args.diagnostic_optimize_systematic
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)

    manifest = json.loads(args.normalization_manifest.read_text())
    nn_bundle = joblib.load(args.nn_model)

    required_bundle_keys = {"model", "imputer", "scaler", "validation_threshold"}
    missing = required_bundle_keys - set(nn_bundle)
    if missing:
        raise KeyError(
            f"NN bundle {args.nn_model} is missing keys: {sorted(missing)}"
        )

    print("=" * 88)
    print("FINAL NN TEST EVALUATION")
    print("=" * 88)

    sources = reconstruct_nn_sources(
        nn_bundle,
        args.dataset_root,
        manifest,
    )

    X_test, y_test, w_test, process_test = common.load_split(
        args.dataset_root,
        manifest,
        args.test_split_name,
        sources,
    )

    x_test_scaled = transform_nn_features(X_test, nn_bundle)
    nn_scores = np.asarray(
        nn_bundle["model"].predict_proba(x_test_scaled)[:, 1],
        dtype=float,
    )

    if not np.isfinite(nn_scores).all():
        raise FloatingPointError("NN produced non-finite test scores.")

    nn_auc = float(
        roc_auc_score(y_test, nn_scores, sample_weight=w_test)
    )
    nn_ap = float(
        average_precision_score(y_test, nn_scores, sample_weight=w_test)
    )

    fixed_ops = weighted_fixed_efficiency_points(
        y_test, nn_scores, w_test
    )
    fixed_ops.to_csv(
        tables / "nn_test_fixed_signal_efficiency_points.csv",
        index=False,
    )

    plot_nn_roc(
        y_test,
        nn_scores,
        w_test,
        fixed_ops,
        figures / "nn_test_roc.png",
        nn_auc,
    )
    plot_score_distribution(
        y_test,
        nn_scores,
        w_test,
        figures / "nn_test_score_distribution.png",
    )

    # Save NN predictions to make subsequent comparisons reproducible.
    pd.DataFrame(
        {
            "label": y_test.astype(int),
            "signal_probability": nn_scores,
            "physical_weight": w_test,
            "process": process_test,
        }
    ).to_csv(tables / "nn_test_predictions.csv", index=False)

    # Full TEST scan is diagnostic only.
    systematics = sorted(set(args.background_systematics))
    test_scan = common.threshold_scan(
        y_test,
        nn_scores,
        w_test,
        systematics,
        args.thresholds,
    )
    test_scan.to_csv(tables / "nn_test_significance_scan.csv", index=False)

    frozen_threshold = float(nn_bundle["validation_threshold"])
    frozen_result = evaluate_threshold(
        y_test,
        nn_scores,
        w_test,
        frozen_threshold,
        systematics,
    )
    frozen_result["selection_origin"] = "validation-selected threshold"
    pd.DataFrame([frozen_result]).to_csv(
        tables / "nn_final_frozen_operating_point_test.csv",
        index=False,
    )

    z_col = (
        f"asimov_z_bkg_syst_{args.diagnostic_optimize_systematic:g}"
    )
    supported = test_scan[
        (test_scan["background_neff"] >= args.min_background_neff)
        & np.isfinite(test_scan[z_col])
    ]
    diagnostic_best = None
    if not supported.empty:
        diagnostic_best = supported.loc[supported[z_col].idxmax()]
        diagnostic_best.to_frame().T.to_csv(
            tables / "nn_best_supported_test_point_DIAGNOSTIC_ONLY.csv",
            index=False,
        )

    plot_significance_scan(
        test_scan,
        frozen_threshold,
        figures / "nn_test_significance_vs_signal_efficiency.png",
        args.diagnostic_optimize_systematic,
        diagnostic_best,
    )

    # Model comparison
    model_curves: list[dict[str, Any]] = []
    fpr, tpr, auc, ap = roc_data(
        y_test, nn_scores, w_test
    )
    model_curves.append(
        {
            "name": "NN",
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc,
            "ap": ap,
            "events": len(y_test),
            "weighted": True,
        }
    )

    comparison_rows = [
        {
            "model": "NN",
            "test_auc": auc,
            "test_average_precision": ap,
            "events": len(y_test),
            "weighted": True,
        }
    ]

    for spec in args.comparison:
        name, path = parse_comparison(spec)
        frame = pd.read_csv(path)
        truth_col, score_col = infer_prediction_columns(frame)

        y = frame[truth_col].to_numpy(dtype=int)
        score = frame[score_col].to_numpy(dtype=float)

        if not np.isin(y, (0, 1)).all():
            raise ValueError(
                f"{name}: truth column {truth_col!r} is not binary."
            )
        if not np.isfinite(score).all():
            raise ValueError(f"{name}: score column contains non-finite values.")

        # External model prediction files generally do not carry physical weights.
        # ROC/AUC does not require equal sample size, but interpretation must be
        # apples-to-apples: same event-level test definition is strongly preferred.
        fpr_m, tpr_m, auc_m, ap_m = roc_data(y, score, None)
        model_curves.append(
            {
                "name": name,
                "fpr": fpr_m,
                "tpr": tpr_m,
                "auc": auc_m,
                "ap": ap_m,
                "events": len(y),
                "weighted": False,
            }
        )
        comparison_rows.append(
            {
                "model": name,
                "test_auc": auc_m,
                "test_average_precision": ap_m,
                "events": len(y),
                "weighted": False,
                "truth_column": truth_col,
                "score_column": score_col,
                "prediction_file": str(path),
            }
        )

    pd.DataFrame(comparison_rows).to_csv(
        tables / "model_comparison.csv",
        index=False,
    )
    plot_model_comparison(
        model_curves,
        figures / "model_comparison_roc.png",
    )

    # Feature dependence
    if args.feature_ablation_csv is not None:
        feature_ablation_plots(
            args.feature_ablation_csv,
            figures,
        )
        pd.read_csv(args.feature_ablation_csv).to_csv(
            tables / "feature_dependence.csv",
            index=False,
        )

    summary = {
        "test_split": args.test_split_name,
        "n_test_events": len(y_test),
        "nn_test_weighted_roc_auc": nn_auc,
        "nn_test_weighted_average_precision": nn_ap,
        "frozen_validation_threshold": frozen_threshold,
        "final_test_result_at_frozen_threshold": frozen_result,
        "fixed_signal_efficiency_points": fixed_ops.to_dict(orient="records"),
        "diagnostic_systematic": args.diagnostic_optimize_systematic,
        "minimum_background_neff": args.min_background_neff,
        "diagnostic_best_test_point": (
            None if diagnostic_best is None else diagnostic_best.to_dict()
        ),
        "important_note": (
            "The final threshold is the threshold selected on validation and stored "
            "in the frozen NN bundle. The best test-scan point is diagnostic only "
            "and must not be used to tune the analysis."
        ),
    }
    (args.output_dir / "final_summary.json").write_text(
        json.dumps(safe_json(summary), indent=2) + "\n"
    )

    print(f"Test events:                  {len(y_test):,}")
    print(f"Weighted ROC AUC:             {nn_auc:.6f}")
    print(f"Weighted average precision:   {nn_ap:.6g}")
    print()
    print("Fixed signal-efficiency points:")
    cols = [
        "target_signal_efficiency",
        "signal_efficiency",
        "background_efficiency",
        "background_rejection",
        "threshold",
    ]
    print(fixed_ops[cols].to_string(index=False))
    print()
    print("FINAL FROZEN OPERATING POINT (selected on validation, evaluated on test)")
    print(f"  threshold:                  {frozen_result['threshold']:.8f}")
    print(f"  signal efficiency:          {frozen_result['signal_efficiency']:.6g}")
    print(f"  background efficiency:      {frozen_result['background_efficiency']:.6e}")
    print(f"  signal yield:               {frozen_result['signal_yield']:.6g}")
    print(f"  background yield:           {frozen_result['background_yield']:.6g}")
    print(f"  S/B:                        {frozen_result['s_over_b']:.6g}")
    print(f"  background Neff:            {frozen_result['background_neff']:.3f}")
    for frac in systematics:
        print(
            f"  Asimov Z ({100*frac:g}% syst):      "
            f"{frozen_result[f'asimov_z_bkg_syst_{frac:g}']:.6g}"
        )

    if diagnostic_best is not None:
        print()
        print("Best statistically supported TEST point (DIAGNOSTIC ONLY — do not tune to it)")
        print(
            f"  threshold={diagnostic_best['threshold']:.8f}, "
            f"eps_s={diagnostic_best['signal_efficiency']:.6g}, "
            f"eps_b={diagnostic_best['background_efficiency']:.6e}, "
            f"Neff_b={diagnostic_best['background_neff']:.1f}, "
            f"Z={diagnostic_best[z_col]:.6g}"
        )
    else:
        print()
        print(
            "WARNING: no test threshold satisfies "
            f"background Neff >= {args.min_background_neff:g}."
        )

    print()
    print(f"Saved final analysis to: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

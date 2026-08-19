#!/usr/bin/env python3
"""Evaluate the saved 22-variable BDT on the untouched full test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import train_event_bdt_22_variables as common


FIXED_EPS_S = (0.50, 0.70, 0.80, 0.90)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--normalization-manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--training-metrics", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def fixed_operating_points(y, score, targets=FIXED_EPS_S):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)

    sig_scores = np.sort(score[y == 1])[::-1]
    n_sig = len(sig_scores)
    n_bkg = int(np.sum(y == 0))

    rows = []
    for target in targets:
        idx = min(int(np.ceil(target * n_sig)) - 1, n_sig - 1)
        threshold = float(sig_scores[idx])

        selected = score >= threshold
        ns = int(np.sum(selected & (y == 1)))
        nb = int(np.sum(selected & (y == 0)))

        eps_s = ns / n_sig
        eps_b = nb / n_bkg
        rows.append({
            "target_signal_efficiency": target,
            "threshold": threshold,
            "signal_efficiency": eps_s,
            "background_efficiency": eps_b,
            "background_rejection_factor": np.inf if eps_b == 0 else 1.0 / eps_b,
            "signal_survivors": ns,
            "signal_events": n_sig,
            "background_survivors": nb,
            "background_events": n_bkg,
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.normalization_manifest.read_text())
    training_metrics = json.loads(args.training_metrics.read_text())
    feature_names = list(training_metrics["features"])

    # The training script resolves event/jet sources internally. Reuse the same
    # default feature definitions if available, otherwise reconstruct them from
    # the saved 22 output feature names.
    event_features = ["event_invariant_mass", "n_jets_original"]

    jet_feature_bases = []
    for name in feature_names:
        if name.startswith("leading_"):
            base = name[len("leading_"):]
            if base not in jet_feature_bases:
                jet_feature_bases.append(base)
        elif name.startswith("subleading_"):
            base = name[len("subleading_"):]
            if base not in jet_feature_bases:
                jet_feature_bases.append(base)

    first_pattern = str(manifest["samples"][0]["path_glob"])
    first_file = common.sample_files(args.dataset_root, first_pattern, "test")[0]

    sources = common.resolve_feature_sources(
        event_features,
        jet_feature_bases,
        common.parquet_columns(first_file),
        first_file,
    )

    resolved = [s.output_name for s in sources]
    if resolved != feature_names:
        raise RuntimeError(
            "Resolved feature ordering does not match the BDT training order.\n"
            f"Saved:    {feature_names}\n"
            f"Resolved: {resolved}"
        )

    X_test, y_test, w_test, process_test = common.load_split(
        args.dataset_root,
        manifest,
        "test",
        sources,
    )

    bundle = joblib.load(args.model)
    model = bundle["model"]
    score = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
    
    if not np.isfinite(score).all():
        raise RuntimeError("Non-finite BDT scores found.")

    auc = float(roc_auc_score(y_test, score))
    ap = float(average_precision_score(y_test, score))
    fpr, tpr, _ = roc_curve(y_test, score)

    ops = fixed_operating_points(y_test, score)
    ops.to_csv(args.output_dir / "test_operating_points.csv", index=False)

    pred = pd.DataFrame({
        "label": np.asarray(y_test, dtype=int),
        "score": score,
        "physical_weight": np.asarray(w_test, dtype=float),
        "process": process_test,
    })
    pred.to_csv(args.output_dir / "test_predictions.csv", index=False)

    metrics = {
        "model": str(args.model),
        "test_signal_events": int(np.sum(np.asarray(y_test) == 1)),
        "test_background_events": int(np.sum(np.asarray(y_test) == 0)),
        "test_auc": auc,
        "test_average_precision": ap,
        "operating_points": ops.to_dict(orient="records"),
    }
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(fpr, tpr, lw=2, label=f"BDT (AUC = {auc:.4f})")
    for row in ops.itertuples(index=False):
        ax.scatter(row.background_efficiency, row.signal_efficiency, s=35)
        ax.annotate(
            f"{100*row.target_signal_efficiency:.0f}%",
            (row.background_efficiency, row.signal_efficiency),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=9,
        )
    ax.plot([0, 1], [0, 1], "--", lw=1, alpha=0.6)
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "test_roc.png", dpi=300)
    plt.close(fig)

    print("=" * 80)
    print("BDT FULL TEST EVALUATION")
    print("=" * 80)
    print(f"Signal events:      {metrics['test_signal_events']:,}")
    print(f"Background events:  {metrics['test_background_events']:,}")
    print(f"Test ROC AUC:       {auc:.9f}")
    print(f"Test AP:            {ap:.9f}")
    print()
    print(ops.to_string(index=False))
    print()
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create final NN vs BDT vs ParticleNet model-comparison outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


TARGETS = (0.50, 0.70, 0.80, 0.90)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nn", type=Path, required=True)
    p.add_argument("--bdt", type=Path, required=True)
    p.add_argument("--particlenet", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def infer_columns(df):
    truth_candidates = ("label", "y_true", "truth", "target")
    score_candidates = ("score", "signal_probability", "signal_score", "probability")
    truth = next((c for c in truth_candidates if c in df.columns), None)
    score = next((c for c in score_candidates if c in df.columns), None)
    if truth is None or score is None:
        raise ValueError(f"Could not infer label/score columns from: {list(df.columns)}")
    return truth, score


def operating_points(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)

    sig_scores = np.sort(s[y == 1])[::-1]
    ns = len(sig_scores)
    nb = int(np.sum(y == 0))

    rows = []
    for target in TARGETS:
        idx = min(int(np.ceil(target * ns)) - 1, ns - 1)
        threshold = float(sig_scores[idx])
        sel = s >= threshold
        sig_surv = int(np.sum(sel & (y == 1)))
        bkg_surv = int(np.sum(sel & (y == 0)))
        eps_s = sig_surv / ns
        eps_b = bkg_surv / nb
        rows.append({
            "target_signal_efficiency": target,
            "threshold": threshold,
            "signal_efficiency": eps_s,
            "background_efficiency": eps_b,
            "background_rejection_factor": np.inf if eps_b == 0 else 1.0 / eps_b,
        })
    return pd.DataFrame(rows)


def evaluate(name, path):
    df = pd.read_csv(path)
    truth_col, score_col = infer_columns(df)
    y = df[truth_col].to_numpy(dtype=int)
    s = df[score_col].to_numpy(dtype=float)

    if not np.isin(y, [0, 1]).all():
        raise ValueError(f"{name}: labels are not binary.")
    if not np.isfinite(s).all():
        raise ValueError(f"{name}: non-finite scores found.")

    fpr, tpr, _ = roc_curve(y, s)
    auc = float(roc_auc_score(y, s))
    ap = float(average_precision_score(y, s))
    ops = operating_points(y, s)

    return {
        "name": name,
        "path": str(path),
        "n_events": len(y),
        "n_signal": int(np.sum(y == 1)),
        "n_background": int(np.sum(y == 0)),
        "auc": auc,
        "ap": ap,
        "fpr": fpr,
        "tpr": tpr,
        "ops": ops,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        evaluate("Final wide NN", args.nn),
        evaluate("BDT", args.bdt),
        evaluate("ParticleNet", args.particlenet),
    ]

    # Summary table
    rows = []
    for m in models:
        row = {
            "model": m["name"],
            "test_events": m["n_events"],
            "signal_events": m["n_signal"],
            "background_events": m["n_background"],
            "roc_auc": m["auc"],
            "average_precision": m["ap"],
        }
        for op in m["ops"].itertuples(index=False):
            tag = int(round(100 * op.target_signal_efficiency))
            row[f"eps_b_at_eps_s_{tag}"] = op.background_efficiency
            row[f"rejection_at_eps_s_{tag}"] = op.background_rejection_factor
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "model_comparison_metrics.csv", index=False)

    # All operating points
    all_ops = []
    for m in models:
        tmp = m["ops"].copy()
        tmp.insert(0, "model", m["name"])
        all_ops.append(tmp)
    pd.concat(all_ops, ignore_index=True).to_csv(
        args.output_dir / "model_comparison_operating_points.csv", index=False
    )

    # ROC plot
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for m in models:
        ax.plot(m["fpr"], m["tpr"], lw=2, label=f'{m["name"]} (AUC = {m["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], "--", lw=1, alpha=0.6)
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "model_comparison_roc.png", dpi=300)
    plt.close(fig)

    # Low-FPR zoom, often more useful for the physics discussion.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for m in models:
        ax.plot(m["fpr"], m["tpr"], lw=2, label=m["name"])
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.set_xlim(0, 0.20)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / "model_comparison_roc_low_fpr.png", dpi=300)
    plt.close(fig)

    print("=" * 100)
    print("FINAL MODEL COMPARISON")
    print("=" * 100)
    print(summary.to_string(index=False))
    print()
    print("NOTE: ParticleNet was evaluated on its own smaller test subset; "
          "NN and BDT use the full common 576,570-event test split.")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()

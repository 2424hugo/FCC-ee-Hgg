#!/usr/bin/env python3
"""Create dissertation-ready NN feature-dependence plots and ranked tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot leave-one-feature-out dependence from feature_ablation_results.csv"
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--top-n",
        type=int,
        default=22,
        help="Number of removed-feature configurations to include in plots/ranking.",
    )
    return p.parse_args()


def clean_label(name: str) -> str:
    replacements = {
        "leading_": "Leading ",
        "subleading_": "Subleading ",
        "event_invariant_mass": "Event invariant mass",
        "n_jets_original": "Original jet multiplicity",
        "jet_energy": "jet energy",
        "jet_mass": "jet mass",
        "constituent_multiplicity": "constituent multiplicity",
        "e2_beta_0p2": r"$e_2^{(\beta=0.2)}$",
        "e3_beta_0p2": r"$e_3^{(\beta=0.2)}$",
        "c2_beta_0p2": r"$C_2^{(\beta=0.2)}$",
        "d2_beta_0p2": r"$D_2^{(\beta=0.2)}$",
        "jet_pt": r"jet $p_T$",
        "jet_p": r"jet $|\vec{p}|$",
        "jet_theta": r"jet $\theta$",
    }

    if name in replacements:
        return replacements[name]

    out = name
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = {
        "configuration",
        "removed_feature",
        "validation_auc",
        "delta_auc",
        "eps_b_at_eps_s_70",
        "delta_eps_b_70",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    base = df[df["removed_feature"].isna() | (df["removed_feature"].astype(str).str.len() == 0)]
    if len(base) != 1:
        raise ValueError("Expected exactly one all-features baseline row.")
    base = base.iloc[0]

    ab = df[df["removed_feature"].notna() & (df["removed_feature"].astype(str).str.len() > 0)].copy()

    # Rank by loss in AUC. Positive means performance worsened when the feature was removed.
    ab["abs_delta_auc"] = ab["delta_auc"].abs()
    ab["feature_label"] = ab["removed_feature"].map(clean_label)
    ab = ab.sort_values("delta_auc", ascending=False).head(args.top_n).copy()

    # Compact dissertation table.
    ranked = ab[
        [
            "removed_feature",
            "validation_auc",
            "delta_auc",
            "eps_b_at_eps_s_70",
            "delta_eps_b_70",
        ]
    ].copy()
    ranked.insert(1, "importance_rank_by_delta_auc", np.arange(1, len(ranked) + 1))
    ranked["background_rejection_at_eps_s_70"] = 1.0 / ranked["eps_b_at_eps_s_70"]
    ranked.to_csv(args.output_dir / "feature_dependence_ranked.csv", index=False)

    # A second compact "top features" table for easy inclusion in write-up.
    ranked.head(10).to_csv(args.output_dir / "feature_dependence_top10.csv", index=False)

    # Plot 1: delta AUC.
    p1 = ab.sort_values("delta_auc", ascending=True)

    fig_h = max(6.0, 0.34 * len(p1) + 1.5)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    ax.barh(p1["feature_label"], p1["delta_auc"])
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel(
        r"$\Delta$AUC = AUC$_{\rm all}$ $-$ AUC$_{\rm feature\ removed}$"
    )
    ax.set_ylabel("")
    ax.set_title("NN feature dependence: leave-one-feature-out AUC change")
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.99,
        0.01,
        f"All-feature validation AUC = {base['validation_auc']:.5f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "feature_dependence_delta_auc.png", dpi=300)
    plt.close(fig)

    # Plot 2: change in background efficiency at 70% signal efficiency.
    # Positive delta means background rejection worsened when feature was removed.
    p2 = ab.sort_values("delta_eps_b_70", ascending=True)

    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    ax.barh(p2["feature_label"], p2["delta_eps_b_70"])
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel(
        r"$\Delta\epsilon_b$ at $\epsilon_s=0.70$ "
        r"= $\epsilon_{b,\rm removed}-\epsilon_{b,\rm all}$"
    )
    ax.set_ylabel("")
    ax.set_title(
        r"NN feature dependence: background-efficiency change at $\epsilon_s=0.70$"
    )
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.99,
        0.01,
        (
            rf"All-feature $\epsilon_b$ = {base['eps_b_at_eps_s_70']:.4f} "
            rf"($1/\epsilon_b$ = {1.0/base['eps_b_at_eps_s_70']:.2f})"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "feature_dependence_delta_epsb70.png", dpi=300)
    plt.close(fig)

    # Plot 3: top-10 only, easier to read in a dissertation main text.
    top10 = ab.sort_values("delta_auc", ascending=False).head(10).sort_values("delta_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.barh(top10["feature_label"], top10["delta_auc"])
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel(r"$\Delta$AUC")
    ax.set_ylabel("")
    ax.set_title("Most influential NN inputs from feature ablation")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "feature_dependence_top10_delta_auc.png", dpi=300)
    plt.close(fig)

    print("=" * 92)
    print("FEATURE DEPENDENCE SUMMARY")
    print("=" * 92)
    print(f"All-feature validation AUC:              {base['validation_auc']:.6f}")
    print(f"All-feature eps_b at eps_s=0.70:        {base['eps_b_at_eps_s_70']:.6f}")
    print(f"All-feature rejection at eps_s=0.70:    {1.0/base['eps_b_at_eps_s_70']:.3f}")
    print()
    print("Top 10 features by leave-one-out delta AUC:")
    print(
        ranked.head(10)[
            [
                "importance_rank_by_delta_auc",
                "removed_feature",
                "delta_auc",
                "delta_eps_b_70",
            ]
        ].to_string(index=False)
    )
    print()
    n_negative = int((ab["delta_auc"] < 0).sum())
    print(
        f"{n_negative} removed-feature configurations have negative delta AUC; "
        "interpret these as negligible/redundant within training variation, not as "
        "evidence that the feature is intrinsically harmful."
    )
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

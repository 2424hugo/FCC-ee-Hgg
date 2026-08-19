#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_ARCHITECTURES = [
    "baseline",
    "wide",
    "very_deep",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple random-seed NN architecture runs "
            "into mean/std summary tables."
        )
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help=(
            "Directory containing architecture/seed_N/architecture_comparison.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def find_results(input_root: Path) -> pd.DataFrame:

    rows = []

    for architecture_dir in sorted(input_root.iterdir()):

        if not architecture_dir.is_dir():
            continue

        architecture = architecture_dir.name

        if architecture not in EXPECTED_ARCHITECTURES:
            continue

        for seed_dir in sorted(
            architecture_dir.glob("seed_*")
        ):

            try:
                seed = int(
                    seed_dir.name.replace(
                        "seed_",
                        "",
                    )
                )
            except ValueError:
                continue

            result_path = (
                seed_dir
                / "architecture_comparison.csv"
            )

            if not result_path.exists():
                print(
                    f"WARNING: missing {result_path}"
                )
                continue

            df = pd.read_csv(
                result_path
            )

            if len(df) != 1:
                raise ValueError(
                    f"Expected exactly one architecture in "
                    f"{result_path}, found {len(df)} rows."
                )

            row = df.iloc[0].to_dict()

            row["seed"] = seed
            row["architecture"] = architecture

            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No architecture comparison files found beneath {input_root}"
        )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        ["architecture", "seed"]
    ).reset_index(drop=True)

    return results


def summarise(results: pd.DataFrame) -> pd.DataFrame:

    metric_columns = [
        "validation_auc",
        "validation_ap",
        "eps_b_at_eps_s_50",
        "eps_b_at_eps_s_70",
        "eps_b_at_eps_s_80",
        "eps_b_at_eps_s_90",
        "background_rejection_at_eps_s_50",
        "background_rejection_at_eps_s_70",
        "background_rejection_at_eps_s_80",
        "background_rejection_at_eps_s_90",
        "best_epoch",
        "training_time_seconds",
    ]

    available_metrics = [
        column
        for column in metric_columns
        if column in results.columns
    ]

    grouped = results.groupby(
        "architecture"
    )

    summary_rows = []

    for architecture, group in grouped:

        row = {
            "architecture": architecture,
            "n_seeds": len(group),
        }

        if "parameters" in group.columns:
            row["parameters"] = int(
                group["parameters"].iloc[0]
            )

        for metric in available_metrics:

            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if len(values) == 0:
                continue

            row[f"{metric}_mean"] = (
                float(values.mean())
            )

            row[f"{metric}_std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0
            )

            row[f"{metric}_min"] = (
                float(values.min())
            )

            row[f"{metric}_max"] = (
                float(values.max())
            )

        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    )

    summary = summary.sort_values(
        "validation_auc_mean",
        ascending=False,
    ).reset_index(drop=True)

    return summary


def save_auc_plot(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:

    architectures = list(
        summary["architecture"]
    )

    means = [
        summary.loc[
            summary["architecture"] == arch,
            "validation_auc_mean",
        ].iloc[0]
        for arch in architectures
    ]

    stds = [
        summary.loc[
            summary["architecture"] == arch,
            "validation_auc_std",
        ].iloc[0]
        for arch in architectures
    ]

    x = np.arange(
        len(architectures)
    )

    fig, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        capsize=5,
        label="Mean ± seed std",
    )

    for index, architecture in enumerate(
        architectures
    ):

        architecture_values = results.loc[
            results["architecture"] == architecture,
            "validation_auc",
        ]

        jitter = np.linspace(
            -0.08,
            0.08,
            len(architecture_values),
        )

        axis.scatter(
            np.full(
                len(architecture_values),
                index,
            )
            + jitter,
            architecture_values,
            alpha=0.7,
        )

    axis.set_xticks(
        x,
        architectures,
    )

    axis.set_ylabel(
        "Validation ROC AUC"
    )

    axis.set_xlabel(
        "Architecture"
    )

    axis.set_title(
        "NN architecture stability across random seeds"
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

    plt.close(fig)


def save_background_efficiency_plot(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:

    architectures = list(
        summary["architecture"]
    )

    targets = [
        50,
        70,
        80,
        90,
    ]

    x = np.arange(
        len(architectures)
    )

    fig, axis = plt.subplots(
        figsize=(9, 6)
    )

    for target in targets:

        mean_column = (
            f"eps_b_at_eps_s_{target}_mean"
        )

        std_column = (
            f"eps_b_at_eps_s_{target}_std"
        )

        if mean_column not in summary.columns:
            continue

        means = summary[
            mean_column
        ].to_numpy()

        stds = summary[
            std_column
        ].to_numpy()

        axis.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=(
                rf"$\epsilon_s={target / 100:.1f}$"
            ),
        )

    axis.set_xticks(
        x,
        architectures,
    )

    axis.set_ylabel(
        "Background efficiency"
    )

    axis.set_xlabel(
        "Architecture"
    )

    axis.set_title(
        "Background efficiency stability across random seeds"
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

    plt.close(fig)


def main() -> None:

    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = find_results(
        args.input_root
    )

    summary = summarise(
        results
    )

    raw_output = (
        args.output_dir
        / "all_seed_results.csv"
    )

    summary_output = (
        args.output_dir
        / "multiseed_summary.csv"
    )

    results.to_csv(
        raw_output,
        index=False,
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    print(
        "\nIndividual seed results:"
    )

    display_individual = [
        column
        for column in [
            "architecture",
            "seed",
            "validation_auc",
            "validation_ap",
            "eps_b_at_eps_s_50",
            "eps_b_at_eps_s_70",
            "eps_b_at_eps_s_80",
            "eps_b_at_eps_s_90",
        ]
        if column in results.columns
    ]

    print(
        results[
            display_individual
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "MULTI-SEED SUMMARY"
    )

    print(
        "=" * 90
    )

    display_summary = [
        column
        for column in [
            "architecture",
            "n_seeds",
            "parameters",
            "validation_auc_mean",
            "validation_auc_std",
            "eps_b_at_eps_s_50_mean",
            "eps_b_at_eps_s_50_std",
            "eps_b_at_eps_s_70_mean",
            "eps_b_at_eps_s_70_std",
            "eps_b_at_eps_s_80_mean",
            "eps_b_at_eps_s_80_std",
            "eps_b_at_eps_s_90_mean",
            "eps_b_at_eps_s_90_std",
        ]
        if column in summary.columns
    ]

    print(
        summary[
            display_summary
        ].to_string(
            index=False
        )
    )

    save_auc_plot(
        results,
        summary,
        args.output_dir
        / "auc_multiseed_comparison.png",
    )

    save_background_efficiency_plot(
        summary,
        args.output_dir
        / "background_efficiency_multiseed.png",
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {raw_output}"
    )

    print(
        f"  {summary_output}"
    )

    print(
        f"  {args.output_dir / 'auc_multiseed_comparison.png'}"
    )

    print(
        f"  {args.output_dir / 'background_efficiency_multiseed.png'}"
    )

    print(
        "\nHighest mean validation AUC:"
    )

    best = summary.iloc[0]

    print(
        f"  {best['architecture']}: "
        f"{best['validation_auc_mean']:.6f} "
        f"± {best['validation_auc_std']:.6f}"
    )


if __name__ == "__main__":
    main()
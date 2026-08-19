#!/usr/bin/env python3
"""
Scan FCC-ee integrated luminosity for the frozen NN analysis.

Input:
    threshold_scan.csv

Expected columns include:
    threshold
    signal_mc_events
    background_mc_events
    signal_yield
    background_yield

The stored yields are assumed to correspond to a reference luminosity
(default 10 ab^-1).

For each luminosity L:

    S(L, t) = S_ref(t) * L / L_ref
    B(L, t) = B_ref(t) * L / L_ref

The program then calculates the Asimov significance at every NN threshold t,
optionally including a fractional background systematic uncertainty, and
selects the threshold that maximizes Z_A at that luminosity.

This is superior to a simple Z ~ sqrt(L) extrapolation because the optimal
classifier threshold can change with luminosity and systematic uncertainty.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan NN significance as a function of FCC-ee luminosity."
    )

    parser.add_argument(
        "--threshold-scan",
        type=Path,
        default=Path(
            "outputs/ml/nn_final_wide_all_data_test/threshold_scan.csv"
        ),
        help="Threshold scan from evaluate_frozen_nn_test.py.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/ml/nn_final_wide_all_data_test/luminosity_scan"
        ),
    )

    parser.add_argument(
        "--reference-luminosity-ab",
        type=float,
        default=10.0,
        help="Luminosity corresponding to yields in threshold_scan.csv.",
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
            "Fractional background uncertainties. "
            "Example: 0.0001 = 0.01%%."
        ),
    )

    parser.add_argument(
        "--luminosity-min-ab",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--luminosity-max-ab",
        type=float,
        default=1.0e6,
    )

    parser.add_argument(
        "--luminosity-points",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--min-background-mc",
        type=int,
        default=20,
        help=(
            "Require at least this many surviving background MC events "
            "in the original test sample."
        ),
    )

    parser.add_argument(
        "--target-significances",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 5.0],
    )

    parser.add_argument(
        "--luminosity-per-year-ab",
        type=float,
        default=10.0,
        help=(
            "Assumed integrated luminosity delivered per year. "
            "Default = 10 ab^-1/year."
        ),
    )

    parser.add_argument(
        "--interaction-points",
        type=int,
        default=1,
        help=(
            "Number of simultaneous interaction points whose luminosities "
            "are combined."
        ),
    )

    return parser.parse_args()


def asimov_no_systematic(
    signal: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """
    Standard Asimov counting significance:

        Z_A = sqrt(2 * [(S+B) ln(1 + S/B) - S])
    """

    result = np.zeros_like(
        signal,
        dtype=np.float64,
    )

    valid = (
        (signal > 0)
        & (background > 0)
    )

    s = signal[valid]
    b = background[valid]

    value = 2.0 * (
        (s + b)
        * np.log1p(s / b)
        - s
    )

    result[valid] = np.sqrt(
        np.maximum(
            value,
            0.0,
        )
    )

    return result


def asimov_with_systematic(
    signal: np.ndarray,
    background: np.ndarray,
    fractional_background_uncertainty: float,
) -> np.ndarray:
    """
    Asimov significance including Gaussian background uncertainty.

    Based on Cowan et al., EPJC 71 (2011) 1554.

    sigma_b = delta_b * B
    """

    if fractional_background_uncertainty <= 0:
        return asimov_no_systematic(
            signal,
            background,
        )

    result = np.zeros_like(
        signal,
        dtype=np.float64,
    )

    valid = (
        (signal > 0)
        & (background > 0)
    )

    s = signal[valid]
    b = background[valid]

    sigma_b = (
        fractional_background_uncertainty
        * b
    )

    sigma_b2 = sigma_b**2

    numerator_1 = (
        (s + b)
        * (b + sigma_b2)
    )

    denominator_1 = (
        b**2
        + (s + b) * sigma_b2
    )

    term_1 = (
        (s + b)
        * np.log(
            numerator_1
            / denominator_1
        )
    )

    numerator_2 = (
        sigma_b2
        * s
    )

    denominator_2 = (
        b
        * (b + sigma_b2)
    )

    term_2 = (
        b**2
        / sigma_b2
        * np.log1p(
            numerator_2
            / denominator_2
        )
    )

    value = 2.0 * (
        term_1 - term_2
    )

    result[valid] = np.sqrt(
        np.maximum(
            value,
            0.0,
        )
    )

    return result


def optimise_at_luminosity(
    scan: pd.DataFrame,
    luminosity_ab: float,
    reference_luminosity_ab: float,
    systematic: float,
) -> dict:
    scale = (
        luminosity_ab
        / reference_luminosity_ab
    )

    signal = (
        scan["signal_yield"].to_numpy(
            dtype=np.float64
        )
        * scale
    )

    background = (
        scan["background_yield"].to_numpy(
            dtype=np.float64
        )
        * scale
    )

    z = asimov_with_systematic(
        signal,
        background,
        systematic,
    )

    best_index = int(
        np.nanargmax(z)
    )

    row = scan.iloc[
        best_index
    ]

    return {
        "luminosity_ab":
            float(luminosity_ab),
        "background_systematic":
            float(systematic),
        "best_threshold":
            float(row["threshold"]),
        "signal_efficiency":
            float(row["signal_efficiency"]),
        "background_efficiency":
            float(row["background_efficiency"]),
        "signal_mc_events":
            int(row["signal_mc_events"]),
        "background_mc_events":
            int(row["background_mc_events"]),
        "signal_yield":
            float(signal[best_index]),
        "background_yield":
            float(background[best_index]),
        "s_over_b":
            (
                float(
                    signal[best_index]
                    / background[best_index]
                )
                if background[best_index] > 0
                else float("inf")
            ),
        "asimov_z":
            float(z[best_index]),
    }


def find_target_crossing(
    luminosities: np.ndarray,
    significances: np.ndarray,
    target: float,
) -> float | None:
    """
    Find approximate luminosity where the optimized significance first
    reaches the requested target.

    Interpolation is done linearly in log(L), which is appropriate for the
    logarithmically spaced luminosity grid.
    """

    indices = np.flatnonzero(
        significances >= target
    )

    if len(indices) == 0:
        return None

    index = int(
        indices[0]
    )

    if index == 0:
        return float(
            luminosities[0]
        )

    l1 = float(
        luminosities[index - 1]
    )

    l2 = float(
        luminosities[index]
    )

    z1 = float(
        significances[index - 1]
    )

    z2 = float(
        significances[index]
    )

    if z2 == z1:
        return l2

    fraction = (
        target - z1
    ) / (
        z2 - z1
    )

    log_l = (
        np.log(l1)
        + fraction
        * (
            np.log(l2)
            - np.log(l1)
        )
    )

    return float(
        np.exp(log_l)
    )


def main() -> None:
    args = parse_args()

    if not args.threshold_scan.exists():
        raise FileNotFoundError(
            f"Threshold scan not found: {args.threshold_scan}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scan = pd.read_csv(
        args.threshold_scan
    )

    required_columns = {
        "threshold",
        "signal_mc_events",
        "background_mc_events",
        "signal_efficiency",
        "background_efficiency",
        "signal_yield",
        "background_yield",
    }

    missing = (
        required_columns
        - set(scan.columns)
    )

    if missing:
        raise ValueError(
            f"Threshold scan is missing columns: {sorted(missing)}"
        )

    # -------------------------------------------------------------
    # Restrict to thresholds with sufficient background MC support.
    # -------------------------------------------------------------

    supported = scan[
        scan["background_mc_events"]
        >= args.min_background_mc
    ].copy()

    if supported.empty:
        raise RuntimeError(
            f"No operating point retains at least "
            f"{args.min_background_mc} background MC events."
        )

    print(
        "=" * 90
    )

    print(
        "FCC-ee LUMINOSITY SIGNIFICANCE SCAN"
    )

    print(
        "=" * 90
    )

    print(
        f"Threshold scan:          {args.threshold_scan}"
    )

    print(
        f"Reference luminosity:    "
        f"{args.reference_luminosity_ab:g} ab^-1"
    )

    print(
        f"Supported thresholds:    "
        f"{len(supported):,}"
    )

    print(
        f"Minimum background MC:   "
        f"{args.min_background_mc}"
    )

    print(
        f"Luminosity range:        "
        f"{args.luminosity_min_ab:g} -> "
        f"{args.luminosity_max_ab:g} ab^-1"
    )

    # -------------------------------------------------------------
    # Log-spaced luminosity grid.
    # -------------------------------------------------------------

    luminosities = np.logspace(
        np.log10(
            args.luminosity_min_ab
        ),
        np.log10(
            args.luminosity_max_ab
        ),
        args.luminosity_points,
    )

    all_rows = []

    target_rows = []

    for systematic in args.systematics:

        print(
            "\n"
            + "-" * 90
        )

        print(
            f"Background systematic: "
            f"{100 * systematic:g}%"
        )

        systematic_rows = []

        for luminosity in luminosities:

            result = optimise_at_luminosity(
                supported,
                luminosity,
                args.reference_luminosity_ab,
                systematic,
            )

            systematic_rows.append(
                result
            )

            all_rows.append(
                result
            )

        systematic_df = pd.DataFrame(
            systematic_rows
        )

        significance_values = (
            systematic_df[
                "asimov_z"
            ].to_numpy()
        )

        print(
            f"  max Z within scan: "
            f"{np.nanmax(significance_values):.6g}"
        )

        for target in args.target_significances:

            required_luminosity = (
                find_target_crossing(
                    luminosities,
                    significance_values,
                    target,
                )
            )

            if required_luminosity is None:

                print(
                    f"  {target:g} sigma: "
                    f"NOT REACHED below "
                    f"{args.luminosity_max_ab:g} ab^-1"
                )

                target_rows.append(
                    {
                        "background_systematic":
                            systematic,
                        "target_significance":
                            target,
                        "required_luminosity_ab":
                            np.nan,
                        "years_one_ip":
                            np.nan,
                        "years_combined_ips":
                            np.nan,
                    }
                )

                continue

            years_one_ip = (
                required_luminosity
                / args.luminosity_per_year_ab
            )

            combined_rate = (
                args.luminosity_per_year_ab
                * args.interaction_points
            )

            years_combined = (
                required_luminosity
                / combined_rate
            )

            print(
                f"  {target:g} sigma: "
                f"{required_luminosity:.3f} ab^-1 "
                f"| {years_one_ip:.2f} years/IP "
                f"| {years_combined:.2f} years "
                f"with {args.interaction_points} IP(s)"
            )

            target_rows.append(
                {
                    "background_systematic":
                        systematic,
                    "target_significance":
                        target,
                    "required_luminosity_ab":
                        required_luminosity,
                    "years_one_ip":
                        years_one_ip,
                    "years_combined_ips":
                        years_combined,
                }
            )

    results = pd.DataFrame(
        all_rows
    )

    targets = pd.DataFrame(
        target_rows
    )

    results.to_csv(
        args.output_dir
        / "luminosity_scan.csv",
        index=False,
    )

    targets.to_csv(
        args.output_dir
        / "target_luminosities.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Plot optimized significance vs luminosity.
    # -------------------------------------------------------------

    fig, axis = plt.subplots(
        figsize=(9, 7)
    )

    for systematic in args.systematics:

        subset = results[
            results[
                "background_systematic"
            ]
            == systematic
        ]

        axis.plot(
            subset["luminosity_ab"],
            subset["asimov_z"],
            label=(
                f"{100 * systematic:g}% "
                f"background syst."
            ),
        )

    for target in args.target_significances:

        axis.axhline(
            target,
            linestyle="--",
            linewidth=1,
        )

    axis.set_xscale(
        "log"
    )

    axis.set_yscale(
        "log"
    )

    axis.set_xlabel(
        r"Integrated luminosity [ab$^{-1}$]"
    )

    axis.set_ylabel(
        r"Optimized Asimov significance $Z_A$"
    )

    axis.set_title(
        "FCC-ee H→gg NN sensitivity vs luminosity"
    )

    axis.grid(
        alpha=0.25,
        which="both",
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        args.output_dir
        / "significance_vs_luminosity.png",
        dpi=200,
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # Plot how optimum threshold changes.
    # -------------------------------------------------------------

    fig, axis = plt.subplots(
        figsize=(9, 7)
    )

    for systematic in args.systematics:

        subset = results[
            results[
                "background_systematic"
            ]
            == systematic
        ]

        axis.plot(
            subset["luminosity_ab"],
            subset["best_threshold"],
            label=(
                f"{100 * systematic:g}% "
                f"background syst."
            ),
        )

    axis.set_xscale(
        "log"
    )

    axis.set_xlabel(
        r"Integrated luminosity [ab$^{-1}$]"
    )

    axis.set_ylabel(
        "Optimal NN threshold"
    )

    axis.set_title(
        "Optimal classifier threshold vs luminosity"
    )

    axis.grid(
        alpha=0.25,
        which="both",
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        args.output_dir
        / "optimal_threshold_vs_luminosity.png",
        dpi=200,
    )

    plt.close(fig)

    # -------------------------------------------------------------
    # Summary JSON.
    # -------------------------------------------------------------

    summary = {
        "threshold_scan":
            str(args.threshold_scan),
        "reference_luminosity_ab":
            args.reference_luminosity_ab,
        "luminosity_range_ab": [
            args.luminosity_min_ab,
            args.luminosity_max_ab,
        ],
        "luminosity_points":
            args.luminosity_points,
        "minimum_background_mc":
            args.min_background_mc,
        "systematics":
            args.systematics,
        "target_significances":
            args.target_significances,
        "luminosity_per_year_ab":
            args.luminosity_per_year_ab,
        "interaction_points":
            args.interaction_points,
        "targets":
            target_rows,
    }

    with open(
        args.output_dir
        / "luminosity_scan_summary.json",
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "OUTPUTS"
    )

    print(
        "=" * 90
    )

    for filename in [
        "luminosity_scan.csv",
        "target_luminosities.csv",
        "luminosity_scan_summary.json",
        "significance_vs_luminosity.png",
        "optimal_threshold_vs_luminosity.png",
    ]:

        print(
            f"  {args.output_dir / filename}"
        )


if __name__ == "__main__":
    main()
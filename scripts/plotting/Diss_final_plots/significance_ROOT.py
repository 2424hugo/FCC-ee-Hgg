#!/usr/bin/env python3

"""
Create final NN significance plots for the dissertation.

Produces
--------
1. Asimov significance versus classifier threshold at 10 ab^-1.
2. Optimised Asimov significance versus integrated luminosity.

The significance is calculated from the final held-out NN predictions.

A minimum of 20 surviving background MC events is required for an
operating point to be considered reliable.
"""

from pathlib import Path
import csv
import math

import numpy as np
import ROOT


# =============================================================================
# Configuration
# =============================================================================

PREDICTIONS_FILE = Path(
    "outputs/ml/nn_final_wide_all_data_test/test_predictions.csv"
)

OUTPUT_DIR = Path(
    "outputs/plots/dissertation/statistical_sensitivity"
)

THRESHOLD_OUTPUT = (
    OUTPUT_DIR / "significance_vs_threshold.pdf"
)

LUMINOSITY_OUTPUT = (
    OUTPUT_DIR / "significance_vs_luminosity.pdf"
)


# Physics inputs
SIGMA_SIGNAL_FB = 0.023
SIGMA_BACKGROUND_FB = 61_000.0

N_GENERATED_SIGNAL = 200_000
N_GENERATED_BACKGROUND = 1_200_000

REFERENCE_LUMINOSITY_AB = 10.0

MIN_BACKGROUND_MC = 20


# Fractional background systematic uncertainties
SYSTEMATICS = [
    0.0,
    0.0001,   # 0.01%
    0.001,    # 0.1%
    0.01,     # 1%
]


SYSTEMATIC_LABELS = {
    0.0: "Statistical only",
    0.0001: "#delta_{B} = 0.01%",
    0.001: "#delta_{B} = 0.1%",
    0.01: "#delta_{B} = 1%",
}


# ROOT colours
SYSTEMATIC_COLOURS = {
    0.0: ROOT.kBlack,
    0.0001: ROOT.kBlue + 1,
    0.001: ROOT.kRed + 1,
    0.01: ROOT.kGreen + 2,
}


SYSTEMATIC_STYLES = {
    0.0: 1,
    0.0001: 1,
    0.001: 1,
    0.01: 1,
}


# Luminosity range
LUMINOSITY_MIN_AB = 1.0
LUMINOSITY_MAX_AB = 1.0e6
N_LUMINOSITY_POINTS = 300


# =============================================================================
# ROOT style
# =============================================================================

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
ROOT.gStyle.SetPadColor(ROOT.kWhite)
ROOT.gStyle.SetFrameFillColor(ROOT.kWhite)

ROOT.gStyle.SetPadLeftMargin(0.15)
ROOT.gStyle.SetPadRightMargin(0.05)
ROOT.gStyle.SetPadBottomMargin(0.14)
ROOT.gStyle.SetPadTopMargin(0.14)

ROOT.gStyle.SetTitleFont(42, "XYZ")
ROOT.gStyle.SetLabelFont(42, "XYZ")

ROOT.gStyle.SetTitleSize(0.045, "XYZ")
ROOT.gStyle.SetLabelSize(0.040, "XYZ")

ROOT.gStyle.SetTitleOffset(1.10, "X")
ROOT.gStyle.SetTitleOffset(1.35, "Y")

ROOT.gStyle.SetLegendBorderSize(0)
ROOT.gStyle.SetLegendFillStyle(0)

ROOT.gStyle.SetLineWidth(2)
ROOT.gStyle.SetFrameLineWidth(2)


# =============================================================================
# Utilities
# =============================================================================

def draw_canvas_title(title_text):

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextAlign(22)
    latex.SetTextSize(0.040)

    latex.DrawLatex(
        0.50,
        0.965,
        title_text
    )

    return latex


def load_predictions(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{path}"
        )

    labels = []
    scores = []

    with path.open("r", newline="") as f:

        reader = csv.DictReader(f)

        required = {"label", "score"}

        missing = required.difference(
            reader.fieldnames
        )

        if missing:
            raise RuntimeError(
                f"Missing columns: {sorted(missing)}"
            )

        for row in reader:

            labels.append(
                int(row["label"])
            )

            scores.append(
                float(row["score"])
            )

    labels = np.asarray(
        labels,
        dtype=np.int8
    )

    scores = np.asarray(
        scores,
        dtype=np.float64
    )

    return labels, scores


# =============================================================================
# Physics normalization
# =============================================================================

def event_weights_per_ab():

    # 1 ab^-1 = 1000 fb^-1
    fb_per_ab = 1000.0

    signal_weight = (
        SIGMA_SIGNAL_FB
        * fb_per_ab
        / N_GENERATED_SIGNAL
    )

    background_weight = (
        SIGMA_BACKGROUND_FB
        * fb_per_ab
        / N_GENERATED_BACKGROUND
    )

    return (
        signal_weight,
        background_weight
    )


# =============================================================================
# Asimov significance
# =============================================================================

def asimov_significance(s, b, fractional_background_uncertainty=0.0):
    """
    Median discovery significance for a single-bin counting experiment.

    For zero systematic uncertainty:
        Z_A = sqrt[2 ((S+B) ln(1+S/B) - S)]

    For non-zero background uncertainty the Cowan et al. expression is used.
    """

    if s <= 0.0 or b <= 0.0:
        return 0.0

    delta = fractional_background_uncertainty

    if delta <= 0.0:

        inside = 2.0 * (
            (s + b) * math.log1p(s / b)
            - s
        )

        return math.sqrt(
            max(inside, 0.0)
        )

    sigma_b = delta * b
    sigma_b2 = sigma_b * sigma_b

    numerator_1 = (
        (s + b)
        * (b + sigma_b2)
    )

    denominator_1 = (
        b * b
        + (s + b) * sigma_b2
    )

    term_1 = (
        (s + b)
        * math.log(
            numerator_1 / denominator_1
        )
    )

    term_2_argument = (
        1.0
        + (
            sigma_b2 * s
            / (
                b * (b + sigma_b2)
            )
        )
    )

    term_2 = (
        (b * b / sigma_b2)
        * math.log(term_2_argument)
    )

    inside = 2.0 * (
        term_1 - term_2
    )

    return math.sqrt(
        max(inside, 0.0)
    )


# =============================================================================
# Threshold scan
# =============================================================================

def build_threshold_scan(labels, scores):

    order = np.argsort(
        -scores,
        kind="mergesort"
    )

    sorted_scores = scores[order]
    sorted_labels = labels[order]

    cumulative_signal = np.cumsum(
        sorted_labels == 1
    )

    cumulative_background = np.cumsum(
        sorted_labels == 0
    )

    # Keep only the end of each tied score group.
    indices = np.r_[
        np.where(
            np.diff(sorted_scores) != 0
        )[0],
        len(sorted_scores) - 1
    ]

    thresholds = sorted_scores[indices]

    signal_mc = (
        cumulative_signal[indices]
        .astype(np.int64)
    )

    background_mc = (
        cumulative_background[indices]
        .astype(np.int64)
    )

    # Require enough background MC events.
    valid = (
        background_mc
        >= MIN_BACKGROUND_MC
    )

    return {
        "threshold": thresholds[valid],
        "signal_mc": signal_mc[valid],
        "background_mc": background_mc[valid],
    }


# =============================================================================
# Significance values for one luminosity
# =============================================================================

def significance_scan(
    threshold_scan,
    luminosity_ab,
    fractional_background_uncertainty
):

    signal_weight_per_ab, background_weight_per_ab = (
        event_weights_per_ab()
    )

    signal_yields = (
        threshold_scan["signal_mc"]
        * signal_weight_per_ab
        * luminosity_ab
    )

    background_yields = (
        threshold_scan["background_mc"]
        * background_weight_per_ab
        * luminosity_ab
    )

    z_values = np.asarray(
        [
            asimov_significance(
                float(s),
                float(b),
                fractional_background_uncertainty
            )
            for s, b
            in zip(
                signal_yields,
                background_yields
            )
        ],
        dtype=np.float64
    )

    return (
        signal_yields,
        background_yields,
        z_values
    )


# =============================================================================
# Graph 5
# =============================================================================

def draw_significance_vs_threshold(
    threshold_scan
):

    canvas = ROOT.TCanvas(
        "c_significance_threshold",
        "Significance versus threshold",
        900,
        700
    )

    frame = canvas.DrawFrame(
        0.0,
        0.0,
        1.0,
        0.060
    )

    frame.GetXaxis().SetTitle(
        "Neural-network signal-score threshold"
    )

    frame.GetYaxis().SetTitle(
        "Asimov significance, Z_{A}"
    )

    graphs = []
    best_points = []

    legend = ROOT.TLegend(
        0.20,
        0.57,
        0.50,
        0.82
    )

    legend.SetTextFont(42)
    legend.SetTextSize(0.032)

    for systematic in SYSTEMATICS:

        (
            signal_yields,
            background_yields,
            z_values
        ) = significance_scan(
            threshold_scan,
            REFERENCE_LUMINOSITY_AB,
            systematic
        )

        x = np.asarray(
            threshold_scan["threshold"],
            dtype=np.float64
        )

        y = np.asarray(
            z_values,
            dtype=np.float64
        )

        # Reverse so threshold increases from left to right.
        # Make contiguous copies because PyROOT cannot use negative-stride arrays.
        x = np.ascontiguousarray(
            x[::-1],
            dtype=np.float64
        )

        y = np.ascontiguousarray(
            y[::-1],
            dtype=np.float64
        )

        graph = ROOT.TGraph(
            len(x),
            x,
            y
        )

        graph.SetLineColor(
            SYSTEMATIC_COLOURS[systematic]
        )

        graph.SetLineStyle(
            SYSTEMATIC_STYLES[systematic]
        )

        graph.SetLineWidth(3)

        graph.Draw("L SAME")

        graphs.append(graph)

        legend.AddEntry(
            graph,
            SYSTEMATIC_LABELS[systematic],
            "l"
        )

        best_index = int(
            np.argmax(z_values)
        )

        best_threshold = float(
            threshold_scan["threshold"][
                best_index
            ]
        )

        best_z = float(
            z_values[best_index]
        )

        best_signal = float(
            signal_yields[best_index]
        )

        best_background = float(
            background_yields[best_index]
        )

        best_points.append(
            {
                "systematic": systematic,
                "threshold": best_threshold,
                "z": best_z,
                "signal": best_signal,
                "background": best_background,
            }
        )

    legend.Draw()

    title = draw_canvas_title(
        "Significance versus classifier threshold"
    )

    canvas.RedrawAxis()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    canvas.SaveAs(
        str(THRESHOLD_OUTPUT)
    )

    return (
        canvas,
        frame,
        graphs,
        legend,
        title,
        best_points
    )


# =============================================================================
# Optimised significance at one luminosity
# =============================================================================

def best_significance_at_luminosity(
    threshold_scan,
    luminosity_ab,
    systematic
):

    (
        signal_yields,
        background_yields,
        z_values
    ) = significance_scan(
        threshold_scan,
        luminosity_ab,
        systematic
    )

    best_index = int(
        np.argmax(z_values)
    )

    return {
        "z": float(
            z_values[best_index]
        ),
        "threshold": float(
            threshold_scan["threshold"][
                best_index
            ]
        ),
        "signal": float(
            signal_yields[best_index]
        ),
        "background": float(
            background_yields[best_index]
        ),
    }


# =============================================================================
# Find luminosity crossing
# =============================================================================

def find_crossing(
    luminosities,
    significances,
    target
):

    for i in range(
        1,
        len(luminosities)
    ):

        z1 = significances[i - 1]
        z2 = significances[i]

        if (
            z1 < target
            and z2 >= target
        ):

            # Interpolate in log luminosity.
            log_l1 = math.log(
                luminosities[i - 1]
            )

            log_l2 = math.log(
                luminosities[i]
            )

            fraction = (
                (target - z1)
                / (z2 - z1)
            )

            log_crossing = (
                log_l1
                + fraction
                * (log_l2 - log_l1)
            )

            return math.exp(
                log_crossing
            )

    return None


# =============================================================================
# Graph 6
# =============================================================================

def draw_significance_vs_luminosity(
    threshold_scan
):

    luminosities = np.logspace(
        math.log10(LUMINOSITY_MIN_AB),
        math.log10(LUMINOSITY_MAX_AB),
        N_LUMINOSITY_POINTS
    )

    canvas = ROOT.TCanvas(
        "c_significance_luminosity",
        "Significance versus luminosity",
        900,
        700
    )

    canvas.SetLogx()

    frame = canvas.DrawFrame(
        LUMINOSITY_MIN_AB,
        0.0,
        LUMINOSITY_MAX_AB,
        6.0
    )

    frame.GetXaxis().SetTitle(
        "Integrated luminosity (ab^{-1})"
    )
    frame.GetXaxis().CenterTitle(True)
    frame.GetXaxis().SetTitleSize(0.043)
    
    frame.GetYaxis().SetTitle(
        "Maximum Asimov significance, Z_{A}"
    )

    graphs = []

    crossing_results = {}

    legend = ROOT.TLegend(
        0.18,
        0.58,
        0.48,
        0.82
    )

    legend.SetTextFont(42)
    legend.SetTextSize(0.032)

    for systematic in SYSTEMATICS:

        significance_values = []

        for luminosity in luminosities:

            result = (
                best_significance_at_luminosity(
                    threshold_scan,
                    float(luminosity),
                    systematic
                )
            )

            significance_values.append(
                result["z"]
            )

        significance_values = np.asarray(
            significance_values,
            dtype=np.float64
        )

        x_lumi = np.ascontiguousarray(
            luminosities,
            dtype=np.float64
        )

        y_significance = np.ascontiguousarray(
            significance_values,
            dtype=np.float64
        )

        graph = ROOT.TGraph(
            len(x_lumi),
            x_lumi,
            y_significance
        )

        graph.SetLineColor(
            SYSTEMATIC_COLOURS[systematic]
        )

        graph.SetLineStyle(
            SYSTEMATIC_STYLES[systematic]
        )

        graph.SetLineWidth(3)

        graph.Draw("L SAME")

        graphs.append(graph)

        legend.AddEntry(
            graph,
            SYSTEMATIC_LABELS[systematic],
            "l"
        )

        crossing_results[
            systematic
        ] = {
            "3sigma": find_crossing(
                luminosities,
                significance_values,
                3.0
            ),
            "5sigma": find_crossing(
                luminosities,
                significance_values,
                5.0
            ),
            "max_z": float(
                np.max(
                    significance_values
                )
            ),
        }

    # Evidence line
    evidence_line = ROOT.TLine(
        LUMINOSITY_MIN_AB,
        3.0,
        LUMINOSITY_MAX_AB,
        3.0
    )

    evidence_line.SetLineColor(
        ROOT.kGray + 2
    )
    evidence_line.SetLineStyle(2)
    evidence_line.SetLineWidth(2)
    evidence_line.Draw("SAME")

    # Discovery line
    discovery_line = ROOT.TLine(
        LUMINOSITY_MIN_AB,
        5.0,
        LUMINOSITY_MAX_AB,
        5.0
    )

    discovery_line.SetLineColor(
        ROOT.kGray + 2
    )
    discovery_line.SetLineStyle(3)
    discovery_line.SetLineWidth(2)
    discovery_line.Draw("SAME")

    legend.Draw()

    significance_labels = ROOT.TLatex()
    significance_labels.SetTextFont(42)
    significance_labels.SetTextSize(0.028)
    significance_labels.SetTextAlign(32)   # right-aligned

    significance_labels.DrawLatex(
        9.0e5,
        3.15,
        "3#sigma evidence"
    )

    significance_labels.DrawLatex(
        9.0e5,
        5.15,
        "5#sigma discovery"
    )

    title = draw_canvas_title(
        "Significance versus integrated luminosity"
    )

    canvas.RedrawAxis()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    canvas.SaveAs(
        str(LUMINOSITY_OUTPUT)
    )

    return (
        canvas,
        frame,
        graphs,
        evidence_line,
        discovery_line,
        legend,
        significance_labels,
        title,
        crossing_results
    )


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 78)
    print("FINAL NN STATISTICAL SENSITIVITY")
    print("=" * 78)

    labels, scores = load_predictions(
        PREDICTIONS_FILE
    )

    n_signal = int(
        np.sum(labels == 1)
    )

    n_background = int(
        np.sum(labels == 0)
    )

    print()
    print(f"Signal test events     : {n_signal:,}")
    print(f"Background test events : {n_background:,}")

    (
        signal_weight_per_ab,
        background_weight_per_ab
    ) = event_weights_per_ab()

    print()
    print("Physical normalization")
    print("----------------------")

    print(
        f"Signal weight / ab^-1     : "
        f"{signal_weight_per_ab:.8g}"
    )

    print(
        f"Background weight / ab^-1 : "
        f"{background_weight_per_ab:.8g}"
    )

    threshold_scan = build_threshold_scan(
        labels,
        scores
    )

    print()
    print(
        f"Valid threshold points: "
        f"{len(threshold_scan['threshold']):,}"
    )

    print(
        f"Minimum surviving background MC: "
        f"{MIN_BACKGROUND_MC}"
    )

    # -------------------------------------------------------------------------
    # Graph 5
    # -------------------------------------------------------------------------

    (
        _,
        _,
        _,
        _,
        _,
        best_points
    ) = draw_significance_vs_threshold(
        threshold_scan
    )

    print()
    print(
        f"Best operating points at "
        f"{REFERENCE_LUMINOSITY_AB:.1f} ab^-1"
    )

    print("-" * 72)

    for point in best_points:

        print(
            f"{SYSTEMATIC_LABELS[point['systematic']]:<24s}"
            f" threshold={point['threshold']:.5f}"
            f"  S={point['signal']:.4f}"
            f"  B={point['background']:.2f}"
            f"  Z={point['z']:.6f}"
        )

    # -------------------------------------------------------------------------
    # Graph 6
    # -------------------------------------------------------------------------

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        crossings
    ) = draw_significance_vs_luminosity(
        threshold_scan
    )

    print()
    print("Luminosity required")
    print("-------------------")

    for systematic in SYSTEMATICS:

        result = crossings[systematic]

        print()
        print(
            SYSTEMATIC_LABELS[systematic]
        )

        if result["3sigma"] is None:
            print(
                "  3 sigma: not reached "
                f"below {LUMINOSITY_MAX_AB:.1e} ab^-1"
            )
        else:
            print(
                f"  3 sigma: "
                f"{result['3sigma']:.3g} ab^-1"
            )

        if result["5sigma"] is None:
            print(
                "  5 sigma: not reached "
                f"below {LUMINOSITY_MAX_AB:.1e} ab^-1"
            )
        else:
            print(
                f"  5 sigma: "
                f"{result['5sigma']:.3g} ab^-1"
            )

    print()
    print("Plots written")
    print("-------------")
    print(THRESHOLD_OUTPUT)
    print(LUMINOSITY_OUTPUT)


if __name__ == "__main__":
    main()
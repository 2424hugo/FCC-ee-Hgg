#!/usr/bin/env python3

"""
Create dissertation-ready ROOT plots comparing the final 22-variable
BDT and 22-variable dense neural network.

Inputs
------
NN:
    outputs/ml/nn_final_wide_all_data_test/test_predictions.csv

BDT:
    outputs/ml/bdt_22_variables_test/test_predictions.csv

Expected CSV columns
--------------------
NN:
    label,score

BDT:
    label,score,physical_weight,process

Only label and score are used for these unweighted classifier-performance
plots.

Outputs
-------
outputs/plots/dissertation/classifier_performance/
    classifier_roc.pdf
    classifier_precision_recall.pdf
    classifier_output_nn.pdf
    classifier_output_bdt.pdf
"""

from pathlib import Path
import csv
import json
import math
import re

import numpy as np
import ROOT


# =============================================================================
# Configuration
# =============================================================================

NN_PREDICTIONS = Path(
    "outputs/ml/nn_final_wide_all_data_test/test_predictions.csv"
)

BDT_PREDICTIONS = Path(
    "outputs/ml/bdt_22_variables_test/test_predictions.csv"
)

# Candidate metrics files. The script will use the first one that exists.
NN_METRICS_CANDIDATES = [
    Path("outputs/ml/nn_final_wide_all_data_test/test_metrics.json"),
    Path("outputs/ml/nn_final_wide_all_data_test/metrics.json"),
]

BDT_METRICS_CANDIDATES = [
    Path("outputs/ml/bdt_22_variables_test/test_metrics.json"),
    Path("outputs/ml/bdt_22_variables_test/metrics.json"),
]

OUTPUT_DIR = Path(
    "outputs/plots/dissertation/classifier_performance"
)

# Known held-out test sample.
EXPECTED_SIGNAL = 185_017
EXPECTED_BACKGROUND = 391_553
EXPECTED_TOTAL = EXPECTED_SIGNAL + EXPECTED_BACKGROUND

N_SCORE_BINS = 50


# =============================================================================
# ROOT configuration
# =============================================================================

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)

ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
ROOT.gStyle.SetPadColor(ROOT.kWhite)
ROOT.gStyle.SetFrameFillColor(ROOT.kWhite)

ROOT.gStyle.SetPadLeftMargin(0.16)
ROOT.gStyle.SetPadRightMargin(0.04)
ROOT.gStyle.SetPadBottomMargin(0.14)
ROOT.gStyle.SetPadTopMargin(0.14)

ROOT.gStyle.SetTitleFont(42, "XYZ")
ROOT.gStyle.SetLabelFont(42, "XYZ")

ROOT.gStyle.SetTitleSize(0.045, "XYZ")
ROOT.gStyle.SetLabelSize(0.040, "XYZ")

ROOT.gStyle.SetTitleOffset(1.10, "X")
ROOT.gStyle.SetTitleOffset(1.45, "Y")

ROOT.gStyle.SetLegendBorderSize(0)
ROOT.gStyle.SetLegendFillStyle(0)

ROOT.gStyle.SetLineWidth(2)
ROOT.gStyle.SetFrameLineWidth(2)

ROOT.gStyle.SetEndErrorSize(0)


# Consistent colours.
NN_COLOUR = ROOT.kBlue + 1
BDT_COLOUR = ROOT.kRed + 1

SIGNAL_COLOUR = ROOT.kBlue + 1
BACKGROUND_COLOUR = ROOT.kRed + 1


# =============================================================================
# Prediction loading
# =============================================================================

def load_predictions(path):
    """
    Load label and score columns from a CSV file.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n  {path}"
        )

    labels = []
    scores = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"No CSV header found in:\n  {path}"
            )

        required = {"label", "score"}
        missing = required.difference(reader.fieldnames)

        if missing:
            raise RuntimeError(
                f"{path} is missing required columns: {sorted(missing)}\n"
                f"Available columns: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
                score = float(row["score"])
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"Could not parse row {row_number} in {path}"
                ) from exc

            labels.append(label)
            scores.append(score)

    return (
        np.asarray(labels, dtype=np.int8),
        np.asarray(scores, dtype=np.float64),
    )


# =============================================================================
# Dataset checks
# =============================================================================

def check_dataset(name, labels, scores):
    """
    Perform consistency checks on one prediction file.
    """

    if len(labels) != len(scores):
        raise RuntimeError(
            f"{name}: label and score arrays have different lengths."
        )

    if len(labels) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"{name}: expected {EXPECTED_TOTAL:,} events, "
            f"found {len(labels):,}."
        )

    if not np.all(np.isin(labels, [0, 1])):
        raise RuntimeError(
            f"{name}: labels must contain only 0 and 1."
        )

    if not np.all(np.isfinite(scores)):
        n_bad = np.sum(~np.isfinite(scores))
        raise RuntimeError(
            f"{name}: found {n_bad} non-finite scores."
        )

    n_signal = int(np.sum(labels == 1))
    n_background = int(np.sum(labels == 0))

    if n_signal != EXPECTED_SIGNAL:
        raise RuntimeError(
            f"{name}: expected {EXPECTED_SIGNAL:,} signal events, "
            f"found {n_signal:,}."
        )

    if n_background != EXPECTED_BACKGROUND:
        raise RuntimeError(
            f"{name}: expected {EXPECTED_BACKGROUND:,} background events, "
            f"found {n_background:,}."
        )

    signal_scores = scores[labels == 1]
    background_scores = scores[labels == 0]

    mean_signal = float(np.mean(signal_scores))
    mean_background = float(np.mean(background_scores))

    if mean_signal <= mean_background:
        raise RuntimeError(
            f"{name}: signal scores are not larger on average than "
            "background scores.\n"
            "The classifier-score direction may be reversed."
        )

    print()
    print(name)
    print("-" * len(name))
    print(f"Total events      : {len(labels):,}")
    print(f"Signal events     : {n_signal:,}")
    print(f"Background events : {n_background:,}")
    print(
        f"Score range       : "
        f"[{np.min(scores):.8g}, {np.max(scores):.8g}]"
    )
    print(f"Mean signal score : {mean_signal:.8f}")
    print(f"Mean bkg score    : {mean_background:.8f}")


def check_cross_model_labels(nn_labels, bdt_labels):
    """
    Verify that NN and BDT files contain the same label sequence.
    """

    if len(nn_labels) != len(bdt_labels):
        raise RuntimeError(
            "NN and BDT files contain different numbers of events."
        )

    if not np.array_equal(nn_labels, bdt_labels):
        mismatch = np.flatnonzero(nn_labels != bdt_labels)

        raise RuntimeError(
            "NN and BDT label sequences differ.\n"
            f"First mismatch at row index {mismatch[0]}."
        )

    print()
    print("Cross-model consistency")
    print("-----------------------")
    print("Event counts match.")
    print("Label sequences match exactly.")


# =============================================================================
# ROC and precision-recall calculations
# =============================================================================

def calculate_curves(labels, scores):
    """
    Calculate ROC and precision-recall points.

    Higher score is assumed to mean more signal-like.

    Equal classifier scores are handled as tied thresholds.
    """

    order = np.argsort(-scores, kind="mergesort")

    sorted_labels = labels[order]
    sorted_scores = scores[order]

    cumulative_signal = np.cumsum(sorted_labels == 1)
    cumulative_background = np.cumsum(sorted_labels == 0)

    # Keep only the final event for each distinct classifier score.
    threshold_indices = np.r_[
        np.where(np.diff(sorted_scores) != 0)[0],
        len(sorted_scores) - 1,
    ]

    tp = cumulative_signal[threshold_indices].astype(np.float64)
    fp = cumulative_background[threshold_indices].astype(np.float64)

    n_signal = float(np.sum(labels == 1))
    n_background = float(np.sum(labels == 0))

    signal_efficiency = tp / n_signal
    background_efficiency = fp / n_background

    precision = tp / (tp + fp)

    # ROC AUC.
    roc_x = np.r_[0.0, background_efficiency]
    roc_y = np.r_[0.0, signal_efficiency]

    auc = float(
        np.trapezoid(roc_y, roc_x)
    )

    # Average precision in the same discrete form used by sklearn:
    #
    # AP = sum_n (R_n - R_{n-1}) P_n
    previous_recall = np.r_[
        0.0,
        signal_efficiency[:-1],
    ]

    average_precision = float(
        np.sum(
            (signal_efficiency - previous_recall)
            * precision
        )
    )

    return {
        "signal_efficiency": signal_efficiency,
        "background_efficiency": background_efficiency,
        "precision": precision,
        "auc": auc,
        "ap": average_precision,
    }


# =============================================================================
# Optional metrics JSON loading
# =============================================================================

def normalise_key(key):
    """
    Convert JSON keys to a simple comparison form.
    """

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(key).lower()
    )


def find_metric(obj, candidate_names):
    """
    Recursively search a JSON object for a metric.
    """

    wanted = {
        normalise_key(name)
        for name in candidate_names
    }

    if isinstance(obj, dict):

        for key, value in obj.items():

            if normalise_key(key) in wanted:

                if isinstance(value, (int, float)):
                    return float(value)

        for value in obj.values():

            found = find_metric(
                value,
                candidate_names
            )

            if found is not None:
                return found

    elif isinstance(obj, list):

        for value in obj:

            found = find_metric(
                value,
                candidate_names
            )

            if found is not None:
                return found

    return None


def read_metrics(candidates):
    """
    Read reported AUC/AP from the first existing metrics JSON file.

    Returns
    -------
    path, auc, ap
    """

    existing_path = None

    for path in candidates:
        if path.exists():
            existing_path = path
            break

    if existing_path is None:
        return None, None, None

    with existing_path.open("r") as f:
        data = json.load(f)

    auc = find_metric(
        data,
        [
            "roc_auc",
            "test_roc_auc",
            "auc",
            "test_auc",
        ],
    )

    ap = find_metric(
        data,
        [
            "average_precision",
            "test_average_precision",
            "average_precision_score",
            "ap",
            "test_ap",
            "pr_auc",
            "test_pr_auc",
        ],
    )

    return existing_path, auc, ap


def choose_metric(reported, recomputed):
    """
    Prefer a metric from the original evaluation file where available.

    This is particularly useful for the NN because the saved prediction CSV
    may contain rounded classifier scores.
    """

    if reported is not None:
        return reported

    return recomputed


# =============================================================================
# ROOT utility functions
# =============================================================================

def make_graph(x, y, colour, line_style=1):
    """
    Convert numpy arrays to a ROOT TGraph.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    graph = ROOT.TGraph(
        len(x),
        x,
        y
    )

    graph.SetLineColor(colour)
    graph.SetLineWidth(3)
    graph.SetLineStyle(line_style)

    return graph


def draw_canvas_title(title_text):
    """
    Draw a title above the plotting frame, inside the top canvas margin.
    """

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextAlign(22)   # centred
    latex.SetTextSize(0.040)

    # y > 1 - top margin lower edge of frame
    latex.DrawLatex(0.50, 0.965, title_text)

    return latex

# =============================================================================
# ROC plot
# =============================================================================

def draw_roc(
    nn_curves,
    bdt_curves,
    nn_auc,
    bdt_auc,
    output_path
):
    """
    Draw HEP-style ROC curve:

        x = signal efficiency
        y = background efficiency

    Background efficiency is shown logarithmically.
    """

    canvas = ROOT.TCanvas(
        "c_roc",
        "Classifier ROC",
        800,
        700
    )

    canvas.SetLogy()

    y_min = 1.0e-6

    frame = canvas.DrawFrame(
        0.0,
        y_min,
        1.0,
        1.0
    )

    frame.GetXaxis().SetTitle(
        "Signal efficiency, #epsilon_{sig}"
    )

    frame.GetYaxis().SetTitle(
        "Background efficiency (mistag rate), #epsilon_{bkg}"
    )

    frame.GetXaxis().SetNdivisions(510)

    # Log plots cannot display y = 0, so remove zero-background points.
    nn_mask = (
        nn_curves["background_efficiency"] > 0
    )

    bdt_mask = (
        bdt_curves["background_efficiency"] > 0
    )

    graph_nn = make_graph(
        nn_curves["signal_efficiency"][nn_mask],
        nn_curves["background_efficiency"][nn_mask],
        NN_COLOUR
    )

    graph_bdt = make_graph(
        bdt_curves["signal_efficiency"][bdt_mask],
        bdt_curves["background_efficiency"][bdt_mask],
        BDT_COLOUR
    )

    graph_bdt.Draw("L SAME")
    graph_nn.Draw("L SAME")

    legend = ROOT.TLegend(
        0.44,
        0.24,
        0.86,
        0.37
    )

    legend.SetTextFont(42)
    legend.SetTextSize(0.032)

    legend.AddEntry(
        graph_nn,
        f"22-variable NN (AUC = {nn_auc:.4f})",
        "l"
    )

    legend.AddEntry(
        graph_bdt,
        f"22-variable BDT (AUC = {bdt_auc:.4f})",
        "l"
    )

    legend.Draw()

    title = draw_canvas_title(
        "Receiver operating characteristic (ROC) curve"
    )

    canvas.RedrawAxis()
    canvas.SaveAs(str(output_path))

    return (
        canvas,
        frame,
        graph_nn,
        graph_bdt,
        legend,
        title
    )


# =============================================================================
# Precision-recall plot
# =============================================================================

def draw_precision_recall(
    nn_curves,
    bdt_curves,
    nn_ap,
    bdt_ap,
    signal_fraction,
    output_path
):
    """
    Draw precision versus recall for both classifiers.
    """

    canvas = ROOT.TCanvas(
        "c_precision_recall",
        "Precision Recall",
        800,
        700
    )

    frame = canvas.DrawFrame(
        0.0,
        0.0,
        1.0,
        1.02
    )

    frame.GetXaxis().SetTitle(
        "Recall / signal efficiency"
    )

    frame.GetYaxis().SetTitle(
        "Precision"
    )

    # Add conventional PR starting point.
    nn_recall = np.r_[
        0.0,
        nn_curves["signal_efficiency"],
    ]

    nn_precision = np.r_[
        1.0,
        nn_curves["precision"],
    ]

    bdt_recall = np.r_[
        0.0,
        bdt_curves["signal_efficiency"],
    ]

    bdt_precision = np.r_[
        1.0,
        bdt_curves["precision"],
    ]

    graph_nn = make_graph(
        nn_recall,
        nn_precision,
        NN_COLOUR
    )

    graph_bdt = make_graph(
        bdt_recall,
        bdt_precision,
        BDT_COLOUR
    )

    graph_bdt.Draw("L SAME")
    graph_nn.Draw("L SAME")

    # Random classifier baseline for the unweighted test sample.
    baseline = ROOT.TLine(
        0.0,
        signal_fraction,
        1.0,
        signal_fraction
    )

    baseline.SetLineColor(
        ROOT.kGray + 2
    )

    baseline.SetLineWidth(2)
    baseline.SetLineStyle(2)
    baseline.Draw("SAME")

    legend = ROOT.TLegend(
        0.39,
        0.39,
        0.84,
        0.57
    )

    legend.SetTextFont(42)
    legend.SetTextSize(0.031)

    legend.AddEntry(
        graph_nn,
        f"22-variable NN (AP = {nn_ap:.4f})",
        "l"
    )

    legend.AddEntry(
        graph_bdt,
        f"22-variable BDT (AP = {bdt_ap:.4f})",
        "l"
    )

    legend.AddEntry(
        baseline,
        f"Random classifier ({signal_fraction:.3f})",
        "l"
    )

    legend.Draw()

    title = draw_canvas_title(
        "Precision-recall (PR) curve"
    )

    canvas.RedrawAxis()
    canvas.SaveAs(str(output_path))

    return (
        canvas,
        frame,
        graph_nn,
        graph_bdt,
        baseline,
        legend,
        title
    )


# =============================================================================
# Classifier-score distributions
# =============================================================================

def make_density_hist(
    name,
    scores,
    bins,
    x_min,
    x_max
):
    """
    Create a normalized histogram corresponding to

        (1/N) dN/d(score)

    The integral over the score range is unity.
    """

    histogram = ROOT.TH1D(
        name,
        "",
        bins,
        x_min,
        x_max
    )

    counts, edges = np.histogram(
        scores,
        bins=bins,
        range=(x_min, x_max)
    )

    bin_width = edges[1] - edges[0]

    normalisation = (
        len(scores) * bin_width
    )

    for i, count in enumerate(
        counts,
        start=1
    ):

        histogram.SetBinContent(
            i,
            count / normalisation
        )

        if count > 0:
            error = (
                math.sqrt(count)
                / normalisation
            )
        else:
            error = 0.0

        histogram.SetBinError(
            i,
            error
        )

    return histogram


def draw_score_distribution(
    labels,
    scores,
    model_name,
    x_axis_title,
    output_path
):
    """
    Draw normalized signal and background score distributions.
    """

    signal_scores = scores[
        labels == 1
    ]

    background_scores = scores[
        labels == 0
    ]

    score_min = float(
        np.min(scores)
    )

    score_max = float(
        np.max(scores)
    )

    # Use the natural [0,1] range for probability-like outputs.
    if (
        score_min >= -1.0e-9
        and score_max <= 1.0 + 1.0e-9
    ):
        x_min = 0.0
        x_max = 1.0

    else:
        score_range = (
            score_max - score_min
        )

        if score_range <= 0:
            raise RuntimeError(
                f"{model_name}: classifier score has zero range."
            )

        padding = (
            0.02 * score_range
        )

        x_min = (
            score_min - padding
        )

        x_max = (
            score_max + padding
        )

    safe_name = re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        model_name
    )

    hist_signal = make_density_hist(
        f"h_{safe_name}_signal",
        signal_scores,
        N_SCORE_BINS,
        x_min,
        x_max
    )

    hist_background = make_density_hist(
        f"h_{safe_name}_background",
        background_scores,
        N_SCORE_BINS,
        x_min,
        x_max
    )

    # Signal styling.
    hist_signal.SetLineColor(
        SIGNAL_COLOUR
    )

    hist_signal.SetLineWidth(3)

    hist_signal.SetFillColorAlpha(
        SIGNAL_COLOUR,
        0.22
    )

    # Background styling.
    hist_background.SetLineColor(
        BACKGROUND_COLOUR
    )

    hist_background.SetLineWidth(3)

    hist_background.SetFillColorAlpha(
        BACKGROUND_COLOUR,
        0.18
    )

    maximum = max(
        hist_signal.GetMaximum(),
        hist_background.GetMaximum()
    )

    hist_signal.SetMaximum(
        24.0
    )

    hist_signal.SetMinimum(0.0)

    hist_signal.GetXaxis().SetTitle(
        x_axis_title
    )

    hist_signal.GetYaxis().SetTitle("Normalised density")

    canvas = ROOT.TCanvas(
        f"c_{safe_name}",
        model_name,
        800,
        700
    )

    # Draw filled histograms.
    hist_signal.Draw(
        "HIST"
    )

    hist_background.Draw(
        "HIST SAME"
    )

    # Re-draw outlines so both remain visible in overlap regions.
    hist_signal.Draw(
        "HIST SAME"
    )

    hist_background.Draw(
        "HIST SAME"
    )

    legend = ROOT.TLegend(
        0.54,
        0.72,
        0.82,
        0.86
    )

    legend.SetTextFont(42)
    legend.SetTextSize(0.038)

    legend.AddEntry(
        hist_signal,
        "H #rightarrow gg",
        "lf"
    )

    legend.AddEntry(
        hist_background,
        "q#bar{q}",
        "lf"
    )

    legend.Draw()

    if "neural network" in model_name.lower():
        title = draw_canvas_title(
            "Neural-network classifier output distribution"
        )
    else:
        title = draw_canvas_title(
            "BDT classifier output distribution"
        )

    canvas.RedrawAxis()
    canvas.SaveAs(
        str(output_path)
    )

    return (
        canvas,
        hist_signal,
        hist_background,
        legend,
        title
    )


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 78)
    print("BDT + NN CLASSIFIER PERFORMANCE")
    print("=" * 78)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # Load saved predictions
    # -------------------------------------------------------------------------

    print()
    print("Loading prediction files...")

    nn_labels, nn_scores = load_predictions(
        NN_PREDICTIONS
    )

    bdt_labels, bdt_scores = load_predictions(
        BDT_PREDICTIONS
    )

    # -------------------------------------------------------------------------
    # Validate datasets
    # -------------------------------------------------------------------------

    check_dataset(
        "22-variable neural network",
        nn_labels,
        nn_scores
    )

    check_dataset(
        "22-variable BDT",
        bdt_labels,
        bdt_scores
    )

    check_cross_model_labels(
        nn_labels,
        bdt_labels
    )

    # -------------------------------------------------------------------------
    # Calculate ROC and PR curves
    # -------------------------------------------------------------------------

    print()
    print("Calculating ROC and precision-recall curves...")

    nn_curves = calculate_curves(
        nn_labels,
        nn_scores
    )

    bdt_curves = calculate_curves(
        bdt_labels,
        bdt_scores
    )

    # -------------------------------------------------------------------------
    # Read originally reported metrics if available
    # -------------------------------------------------------------------------

    (
        nn_metrics_path,
        nn_reported_auc,
        nn_reported_ap
    ) = read_metrics(
        NN_METRICS_CANDIDATES
    )

    (
        bdt_metrics_path,
        bdt_reported_auc,
        bdt_reported_ap
    ) = read_metrics(
        BDT_METRICS_CANDIDATES
    )

    nn_auc = choose_metric(
        nn_reported_auc,
        nn_curves["auc"]
    )

    nn_ap = choose_metric(
        nn_reported_ap,
        nn_curves["ap"]
    )

    bdt_auc = choose_metric(
        bdt_reported_auc,
        bdt_curves["auc"]
    )

    bdt_ap = choose_metric(
        bdt_reported_ap,
        bdt_curves["ap"]
    )

    # -------------------------------------------------------------------------
    # Print metrics
    # -------------------------------------------------------------------------

    print()
    print("Classifier performance")
    print("----------------------")

    print(
        f"NN ROC AUC from predictions : "
        f"{nn_curves['auc']:.8f}"
    )

    print(
        f"NN AP from predictions      : "
        f"{nn_curves['ap']:.8f}"
    )

    if nn_metrics_path is not None:
        print(
            f"NN metrics file             : "
            f"{nn_metrics_path}"
        )

    if nn_reported_auc is not None:
        print(
            f"NN reported ROC AUC         : "
            f"{nn_reported_auc:.8f}"
        )

    if nn_reported_ap is not None:
        print(
            f"NN reported AP              : "
            f"{nn_reported_ap:.8f}"
        )

    print()

    print(
        f"BDT ROC AUC from predictions: "
        f"{bdt_curves['auc']:.8f}"
    )

    print(
        f"BDT AP from predictions     : "
        f"{bdt_curves['ap']:.8f}"
    )

    if bdt_metrics_path is not None:
        print(
            f"BDT metrics file            : "
            f"{bdt_metrics_path}"
        )

    if bdt_reported_auc is not None:
        print(
            f"BDT reported ROC AUC        : "
            f"{bdt_reported_auc:.8f}"
        )

    if bdt_reported_ap is not None:
        print(
            f"BDT reported AP             : "
            f"{bdt_reported_ap:.8f}"
        )

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------

    roc_path = (
        OUTPUT_DIR
        / "classifier_roc.pdf"
    )

    pr_path = (
        OUTPUT_DIR
        / "classifier_precision_recall.pdf"
    )

    nn_score_path = (
        OUTPUT_DIR
        / "classifier_output_nn.pdf"
    )

    bdt_score_path = (
        OUTPUT_DIR
        / "classifier_output_bdt.pdf"
    )

    # -------------------------------------------------------------------------
    # Draw ROC
    # -------------------------------------------------------------------------

    print()
    print("Drawing ROC curve...")

    draw_roc(
        nn_curves,
        bdt_curves,
        nn_auc,
        bdt_auc,
        roc_path
    )

    # -------------------------------------------------------------------------
    # Draw precision-recall
    # -------------------------------------------------------------------------

    signal_fraction = (
        EXPECTED_SIGNAL
        / EXPECTED_TOTAL
    )

    print("Drawing precision-recall curve...")

    draw_precision_recall(
        nn_curves,
        bdt_curves,
        nn_ap,
        bdt_ap,
        signal_fraction,
        pr_path
    )

    # -------------------------------------------------------------------------
    # Draw NN score distribution
    # -------------------------------------------------------------------------

    print("Drawing neural-network score distribution...")

    draw_score_distribution(
        nn_labels,
        nn_scores,
        "22-variable neural network",
        "Neural-network signal score",
        nn_score_path
    )

    # -------------------------------------------------------------------------
    # Draw BDT score distribution
    # -------------------------------------------------------------------------

    print("Drawing BDT score distribution...")

    draw_score_distribution(
        bdt_labels,
        bdt_scores,
        "22-variable BDT",
        "BDT signal score",
        bdt_score_path
    )

    # -------------------------------------------------------------------------
    # Finished
    # -------------------------------------------------------------------------

    print()
    print("Plots written")
    print("-------------")
    print(roc_path)
    print(pr_path)
    print(nn_score_path)
    print(bdt_score_path)

    print()
    print("Done.")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
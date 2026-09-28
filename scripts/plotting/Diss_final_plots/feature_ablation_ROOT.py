#!/usr/bin/env python3

from pathlib import Path
import csv

import ROOT


# =============================================================================
# Configuration
# =============================================================================

INPUT_FILE = Path(
    "outputs/ml/final_feature_dependence/feature_dependence_top10.csv"
)

OUTPUT_DIR = Path(
    "outputs/plots/dissertation/feature_dependence"
)

OUTPUT_FILE = OUTPUT_DIR / "feature_ablation_top10.pdf"


# =============================================================================
# ROOT style
# =============================================================================

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
ROOT.gStyle.SetPadColor(ROOT.kWhite)
ROOT.gStyle.SetFrameFillColor(ROOT.kWhite)

# Large left margin for feature names.
ROOT.gStyle.SetPadLeftMargin(0.31)
ROOT.gStyle.SetPadRightMargin(0.05)
ROOT.gStyle.SetPadBottomMargin(0.14)
ROOT.gStyle.SetPadTopMargin(0.14)

ROOT.gStyle.SetTitleFont(42, "XYZ")
ROOT.gStyle.SetLabelFont(42, "XYZ")

ROOT.gStyle.SetTitleSize(0.045, "XYZ")
ROOT.gStyle.SetLabelSize(0.036, "XYZ")

ROOT.gStyle.SetTitleOffset(1.15, "X")
ROOT.gStyle.SetTitleOffset(1.35, "Y")

ROOT.gStyle.SetLineWidth(2)
ROOT.gStyle.SetFrameLineWidth(2)


# =============================================================================
# Human-readable feature labels
# =============================================================================

FEATURE_LABELS = {
    "leading_c2_beta_0p2":
        "Leading C_{2}^{#beta=0.2}",

    "subleading_c2_beta_0p2":
        "Subleading C_{2}^{#beta=0.2}",

    "leading_d2_beta_0p2":
        "Leading D_{2}^{#beta=0.2}",

    "subleading_d2_beta_0p2":
        "Subleading D_{2}^{#beta=0.2}",

    "leading_e2_beta_0p2":
        "Leading e_{2}^{#beta=0.2}",

    "subleading_e2_beta_0p2":
        "Subleading e_{2}^{#beta=0.2}",

    "leading_e3_beta_0p2":
        "Leading e_{3}^{#beta=0.2}",

    "subleading_e3_beta_0p2":
        "Subleading e_{3}^{#beta=0.2}",

    "leading_jet_theta":
        "Leading jet #theta",

    "subleading_jet_theta":
        "Subleading jet #theta",

    "leading_jet_energy":
        "Leading jet energy",

    "subleading_jet_energy":
        "Subleading jet energy",

    "leading_jet_mass":
        "Leading jet mass",

    "subleading_jet_mass":
        "Subleading jet mass",

    "leading_jet_pt":
        "Leading jet p_{T}",

    "subleading_jet_pt":
        "Subleading jet p_{T}",

    "leading_jet_p":
        "Leading jet |p|",

    "subleading_jet_p":
        "Subleading jet |p|",

    "leading_constituent_multiplicity":
        "Leading constituent multiplicity",

    "subleading_constituent_multiplicity":
        "Subleading constituent multiplicity",

    "event_invariant_mass":
        "Event invariant mass",

    "n_jets_original":
        "Jet multiplicity",
}


# =============================================================================
# Utilities
# =============================================================================

def load_data(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{path}"
        )

    rows = []

    with path.open("r", newline="") as f:

        reader = csv.DictReader(f)

        required = {
            "removed_feature",
            "importance_rank_by_delta_auc",
            "validation_auc",
            "delta_auc",
        }

        missing = required.difference(reader.fieldnames)

        if missing:
            raise RuntimeError(
                f"Missing required columns: {sorted(missing)}"
            )

        for row in reader:

            rows.append(
                {
                    "feature": row["removed_feature"],
                    "rank": int(row["importance_rank_by_delta_auc"]),
                    "validation_auc": float(row["validation_auc"]),
                    "delta_auc": float(row["delta_auc"]),
                }
            )

    # Best / most important feature first.
    rows.sort(key=lambda x: x["rank"])

    return rows


def feature_label(name):

    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]

    # Fallback for unexpected feature names.
    return name.replace("_", " ")


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


# =============================================================================
# Plot
# =============================================================================

def draw_feature_ablation(rows):

    n = len(rows)

    # Scale AUC differences by 1000 for readability.
    scale = 1000.0

    values = [
        row["delta_auc"] * scale
        for row in rows
    ]

    # Allow for negative delta-AUC values if present.
    maximum = max(values)

    if maximum <= 0:
        raise RuntimeError(
            "Top-10 feature ablation values are not positive."
        )

    x_min = 0.0
    x_max = 1.20 * maximum

    canvas = ROOT.TCanvas(
        "c_feature_ablation",
        "Feature ablation",
        1000,
        760
    )

    # Dummy 2D histogram provides axes and categorical y labels.
    frame = ROOT.TH2D(
        "frame_feature_ablation",
        "",
        100,
        x_min,
        x_max,
        n,
        0,
        n
    )

    frame.GetXaxis().SetTitle(
        "#DeltaAUC #times 10^{3}"
    )
    frame.GetXaxis().CenterTitle(True)

    frame.GetYaxis().SetTitle("")

    frame.GetXaxis().SetTitleSize(0.043)
    frame.GetXaxis().SetLabelSize(0.037)

    frame.GetYaxis().SetLabelSize(0.034)
    frame.GetYaxis().SetTickLength(0)

    # Rank 1 appears at the top.
    for i, row in enumerate(rows):

        y_bin = n - i

        frame.GetYaxis().SetBinLabel(
            y_bin,
            feature_label(row["feature"])
        )

    frame.Draw("AXIS")

    # Keep objects alive until canvas is written.
    boxes = []
    value_labels = []

    positive_colour = ROOT.kBlue + 1
    negative_colour = ROOT.kRed + 1

    for i, row in enumerate(rows):

        value = row["delta_auc"] * scale

        # Rank 1 at top.
        y_center = n - i - 0.5

        y_low = y_center - 0.30
        y_high = y_center + 0.30

        x1 = min(0.0, value)
        x2 = max(0.0, value)

        box = ROOT.TBox(
            x1,
            y_low,
            x2,
            y_high
        )

        colour = (
            positive_colour
            if value >= 0
            else negative_colour
        )

        box.SetFillColorAlpha(
            colour,
            0.65
        )

        box.SetLineColor(colour)
        box.SetLineWidth(2)

        box.Draw("SAME")
        boxes.append(box)

        # Numerical value at the end of each bar.
        text = ROOT.TLatex()
        text.SetTextFont(42)
        text.SetTextSize(0.027)
        text.SetTextAlign(
            12 if value >= 0 else 32
        )

        offset = 0.015 * (x_max - x_min)

        x_text = (
            value + offset
            if value >= 0
            else value - offset
        )

        text.DrawLatex(
            x_text,
            y_center,
            f"{value:.2f}"
        )

        value_labels.append(text)


    title = draw_canvas_title(
        "Neural-network feature-ablation importance"
    )

    canvas.RedrawAxis()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    canvas.SaveAs(
        str(OUTPUT_FILE)
    )

    return (
        canvas,
        frame,
        boxes,
        value_labels,
        title
    )


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 72)
    print("NEURAL-NETWORK FEATURE ABLATION")
    print("=" * 72)

    rows = load_data(INPUT_FILE)

    print()
    print(f"Loaded {len(rows)} features from:")
    print(INPUT_FILE)

    # Infer the full-model validation AUC.
    baselines = [
        row["validation_auc"] + row["delta_auc"]
        for row in rows
    ]

    baseline_auc = sum(baselines) / len(baselines)

    print()
    print(
        f"Inferred full-model validation AUC: "
        f"{baseline_auc:.8f}"
    )

    print()
    print("Feature ranking")
    print("---------------")

    for row in rows:

        print(
            f"{row['rank']:2d}. "
            f"{row['feature']:<40s} "
            f"delta AUC = {row['delta_auc']:+.8f}"
        )

    draw_feature_ablation(rows)

    print()
    print("Plot written:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
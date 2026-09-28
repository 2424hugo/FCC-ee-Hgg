from pathlib import Path

import awkward as ak
import numpy as np
import ROOT


# ============================================================
# Configuration
# ============================================================

BASE = Path("cache/analysis_dataset")
PLOT_DIR = Path("outputs/plots/diss_plots/correlation")

OUTPUT_PNG = PLOT_DIR / "observable_correlation_matrix.png"
OUTPUT_PDF = PLOT_DIR / "observable_correlation_matrix.pdf"

SPLITS = ["train", "validation", "test"]


# ============================================================
# Variables
#
# Same 22 scalar inputs used by the event-level BDT/NN:
#   2 event variables
#   10 variables for J1
#   10 variables for J2
# ============================================================

EVENT_FIELDS = [
]

JET_FIELDS = [
    "jet_energy",
    "jet_mass",
    "constituent_multiplicity",
    "e2_beta_0p2",
    "e3_beta_0p2",
    "c2_beta_0p2",
    "jet_theta",
]

ALL_FIELDS = EVENT_FIELDS + JET_FIELDS


# ============================================================
# Labels
#
# Ordering:
#   event quantities
#   all leading-jet quantities
#   all sub-leading-jet quantities
#
# This makes correlations within and between the jets easier
# to identify as blocks in the matrix.
# ============================================================

LABELS = [
    "E_{J1}",
    "m_{J1}",
    "N_{const,J1}",
    "e_{2,J1}",
    "e_{3,J1}",
    "C_{2,J1}",
    "#theta_{J1}",

    "E_{J2}",
    "m_{J2}",
    "N_{const,J2}",
    "e_{2,J2}",
    "e_{3,J2}",
    "C_{2,J2}",
    "#theta_{J2}",
]

# ============================================================
# ROOT style
# ============================================================

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetCanvasColor(0)
ROOT.gStyle.SetPadColor(0)
ROOT.gStyle.SetFrameFillColor(0)
ROOT.gStyle.SetNumberContours(255)


def set_diverging_palette():
    """
    Blue -> white -> red palette.

    Appropriate for a correlation matrix:
        -1 -> blue
         0 -> white
        +1 -> red
    """

    stops = np.array(
        [0.00, 0.50, 1.00],
        dtype=np.float64,
    )

    red = np.array(
        [0.20, 1.00, 0.80],
        dtype=np.float64,
    )

    green = np.array(
        [0.35, 1.00, 0.20],
        dtype=np.float64,
    )

    blue = np.array(
        [0.80, 1.00, 0.20],
        dtype=np.float64,
    )

    ROOT.TColor.CreateGradientColorTable(
        len(stops),
        stops,
        red,
        green,
        blue,
        255,
    )


set_diverging_palette()


# ============================================================
# Helpers
# ============================================================

def to_numpy_1d(array):
    """
    Convert a scalar Awkward array to NumPy.

    Missing values are represented by NaN.
    """

    array = ak.fill_none(
        array,
        np.nan,
    )

    values = ak.to_numpy(array)

    return np.asarray(
        values,
        dtype=np.float64,
    )


def extract_jet_column(array, jet_index):
    """
    Extract one ordered jet from an event -> jet array.

    jet_index = 0 : leading jet J1
    jet_index = 1 : sub-leading jet J2
    """

    padded = ak.pad_none(
        array,
        2,
        axis=1,
        clip=False,
    )

    values = padded[:, jet_index]

    return to_numpy_1d(values)


# ============================================================
# Construct feature matrix
# ============================================================

def extract_feature_matrix(data):
    """
    Convert one Parquet shard into

        shape = (n_events, 14)

    Ordering:
        7 leading-jet variables
        7 sub-leading-jet variables
    """

    columns = []

    # --------------------------------------------------------
    # Leading jet J1
    # --------------------------------------------------------

    for field in JET_FIELDS:

        values = extract_jet_column(
            data[field],
            jet_index=0,
        )

        columns.append(values)

    # --------------------------------------------------------
    # Sub-leading jet J2
    # --------------------------------------------------------

    for field in JET_FIELDS:

        values = extract_jet_column(
            data[field],
            jet_index=1,
        )

        columns.append(values)

    matrix = np.column_stack(columns)

    if matrix.shape[1] != len(LABELS):
        raise RuntimeError(
            f"Feature matrix contains {matrix.shape[1]} columns, "
            f"but {len(LABELS)} labels were defined."
        )

    return matrix


# ============================================================
# Read complete dataset
# ============================================================

def process_sample(sample):

    matrices = []

    n_events = 0
    n_files = 0

    print(f"\nProcessing {sample}")

    for split in SPLITS:

        folder = BASE / sample / split

        files = sorted(
            folder.glob("events_*_part_*.parquet")
        )

        print(
            f"  {split:12s}: "
            f"{len(files):4d} files"
        )

        if not files:
            print(
                f"    WARNING: no files found in {folder}"
            )
            continue

        for i, filepath in enumerate(
            files,
            start=1,
        ):

            # Read only the fields required for this plot.
            data = ak.from_parquet(
                filepath,
                columns=ALL_FIELDS,
            )

            matrix = extract_feature_matrix(
                data
            )

            matrices.append(matrix)

            n_events += len(data)
            n_files += 1

            if i % 20 == 0 or i == len(files):

                print(
                    f"      processed "
                    f"{i}/{len(files)} files"
                )

    if not matrices:
        raise RuntimeError(
            f"No Parquet data found for sample '{sample}'."
        )

    matrix = np.concatenate(
        matrices,
        axis=0,
    )

    print(
        f"  Total: {n_events:,} events "
        f"from {n_files} files"
    )

    print(
        f"  Matrix shape: "
        f"{matrix.shape}"
    )

    # --------------------------------------------------------
    # Report non-finite values
    # --------------------------------------------------------

    print("\n  Non-finite values by observable:")

    for i, label in enumerate(LABELS):

        n_bad = np.count_nonzero(
            ~np.isfinite(matrix[:, i])
        )

        if n_bad > 0:

            fraction = (
                100.0 * n_bad / len(matrix)
            )

            print(
                f"    {label:18s}: "
                f"{n_bad:8,d} "
                f"({fraction:.4f}%)"
            )

    return matrix


# ============================================================
# Pearson correlation
# ============================================================

def calculate_correlation_matrix(matrix):
    """
    Calculate Pearson rho independently for each variable pair.

    Only events where BOTH variables are finite are used for
    that individual correlation coefficient.

    This avoids removing a complete event because, for example,
    D2 is non-finite for one jet.
    """

    n_features = matrix.shape[1]

    correlation = np.full(
        (n_features, n_features),
        np.nan,
        dtype=np.float64,
    )

    pair_counts = np.zeros(
        (n_features, n_features),
        dtype=np.int64,
    )

    for i in range(n_features):

        x = matrix[:, i]

        for j in range(i, n_features):

            y = matrix[:, j]

            finite = (
                np.isfinite(x)
                & np.isfinite(y)
            )

            n_valid = np.count_nonzero(
                finite
            )

            pair_counts[i, j] = n_valid
            pair_counts[j, i] = n_valid

            if n_valid < 2:
                continue

            x_valid = x[finite]
            y_valid = y[finite]

            # A constant variable has undefined Pearson rho.
            if (
                np.std(x_valid) == 0
                or np.std(y_valid) == 0
            ):
                continue

            rho = np.corrcoef(
                x_valid,
                y_valid,
            )[0, 1]

            correlation[i, j] = rho
            correlation[j, i] = rho

    return correlation, pair_counts


# ============================================================
# Print strongest correlations
# ============================================================

def print_strongest_correlations(
    correlation,
    sample,
    n=20,
):
    """
    Print strongest off-diagonal correlations by |rho|.
    """

    pairs = []

    n_features = len(LABELS)

    for i in range(n_features):

        for j in range(i + 1, n_features):

            rho = correlation[i, j]

            if np.isfinite(rho):

                pairs.append(
                    (
                        abs(rho),
                        rho,
                        LABELS[i],
                        LABELS[j],
                    )
                )

    pairs.sort(
        reverse=True,
        key=lambda item: item[0],
    )

    print(
        f"\nStrongest correlations: {sample}"
    )

    for _, rho, label_i, label_j in pairs[:n]:

        print(
            f"  {label_i:18s} "
            f"vs "
            f"{label_j:18s} "
            f"rho = {rho:+.4f}"
        )


# ============================================================
# ROOT heatmap
# ============================================================

def make_root_hist(
    correlation,
    name,
):

    n_features = len(LABELS)

    hist = ROOT.TH2D(
        name,
        "",
        n_features,
        0,
        n_features,
        n_features,
        0,
        n_features,
    )

    hist.SetDirectory(0)

    # --------------------------------------------------------
    # Axis labels
    # --------------------------------------------------------

    for i, label in enumerate(LABELS):

        hist.GetXaxis().SetBinLabel(
            i + 1,
            label,
        )

        # Reverse y ordering so first feature appears at top.
        hist.GetYaxis().SetBinLabel(
            n_features - i,
            label,
        )

    # --------------------------------------------------------
    # Bin contents
    # --------------------------------------------------------

    for i in range(n_features):

        for j in range(n_features):

            value = correlation[i, j]

            if not np.isfinite(value):
                value = 0.0

            hist.SetBinContent(
                j + 1,
                n_features - i,
                value,
            )

    hist.SetMinimum(-1.0)
    hist.SetMaximum(+1.0)

    return hist


# ============================================================
# Draw one matrix
# ============================================================

def draw_single_matrix(
    correlation,
    sample_name,
    output_stem,
):

    canvas = ROOT.TCanvas(
        f"canvas_{output_stem}",
        "",
        1200,
        1100,
    )

    canvas.SetLeftMargin(0.18)
    canvas.SetRightMargin(0.15)
    canvas.SetBottomMargin(0.18)
    canvas.SetTopMargin(0.10)

    hist = make_root_hist(
        correlation,
        f"hist_{output_stem}",
    )

    hist.GetXaxis().SetLabelSize(0.025)
    hist.GetYaxis().SetLabelSize(0.025)

    hist.GetXaxis().LabelsOption("v")

    hist.GetZaxis().SetTitle(
        "Pearson correlation coefficient #rho"
    )

    hist.GetZaxis().SetTitleSize(0.035)
    hist.GetZaxis().SetLabelSize(0.030)
    hist.GetZaxis().SetTitleOffset(1.15)

    hist.Draw("COLZ")

    title = ROOT.TLatex()
    title.SetNDC()
    title.SetTextAlign(22)
    title.SetTextFont(62)
    title.SetTextSize(0.035)

    title.DrawLatex(
        0.50,
        0.965,
        sample_name,
    )

    canvas.Update()

    png = PLOT_DIR / f"{output_stem}.png"
    pdf = PLOT_DIR / f"{output_stem}.pdf"

    canvas.SaveAs(str(png))
    canvas.SaveAs(str(pdf))

    print("\nSaved:")
    print(f"  {png}")
    print(f"  {pdf}")

    canvas.Close()


# ============================================================
# Draw signal/background side-by-side
# ============================================================

def draw_combined_matrix(
    signal_corr,
    background_corr,
):

    canvas = ROOT.TCanvas(
        "canvas_correlations_combined",
        "",
        2000,
        1000,
    )

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    title = ROOT.TLatex()
    title.SetNDC()
    title.SetTextAlign(22)
    title.SetTextFont(62)
    title.SetTextSize(0.035)

    title.DrawLatex(
        0.50,
        0.975,
        "Observable Correlations",
    )

    signal_hist = make_root_hist(
        signal_corr,
        "signal_correlations",
    )

    background_hist = make_root_hist(
        background_corr,
        "background_correlations",
    )

    panels = [
        (
            signal_hist,
            "H #rightarrow gg signal",
        ),
        (
            background_hist,
            "e^{+}e^{-} #rightarrow q#bar{q} background",
        ),
    ]

    keep_alive = [
        title,
        signal_hist,
        background_hist,
    ]

    for index, (
        hist,
        panel_title,
    ) in enumerate(panels):

        x1 = (
            0.01
            if index == 0
            else 0.505
        )

        x2 = (
            0.495
            if index == 0
            else 0.99
        )

        pad = ROOT.TPad(
            f"correlation_pad_{index}",
            "",
            x1,
            0.02,
            x2,
            0.93,
        )

        pad.SetLeftMargin(0.20)
        pad.SetRightMargin(0.16)
        pad.SetBottomMargin(0.20)
        pad.SetTopMargin(0.07)

        pad.Draw()
        pad.cd()

        hist.GetXaxis().SetLabelSize(0.024)
        hist.GetYaxis().SetLabelSize(0.024)

        hist.GetXaxis().LabelsOption("v")

        hist.GetZaxis().SetTitle(
            "Pearson #rho"
        )

        hist.GetZaxis().SetTitleSize(0.032)
        hist.GetZaxis().SetLabelSize(0.028)
        hist.GetZaxis().SetTitleOffset(1.10)

        hist.Draw("COLZ")

        panel_text = ROOT.TLatex()
        panel_text.SetNDC()
        panel_text.SetTextFont(62)
        panel_text.SetTextSize(0.033)

        panel_text.DrawLatex(
            0.20,
            0.95,
            panel_title,
        )

        keep_alive.extend([
            pad,
            panel_text,
        ])

        canvas.cd()

    canvas.Update()

    canvas.SaveAs(
        str(OUTPUT_PNG)
    )

    canvas.SaveAs(
        str(OUTPUT_PDF)
    )

    print("\nSaved combined matrix:")
    print(f"  {OUTPUT_PNG}")
    print(f"  {OUTPUT_PDF}")

    canvas.Close()


# ============================================================
# Main
# ============================================================

def main():

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Read samples
    # --------------------------------------------------------

    signal = process_sample(
        "signal"
    )

    background = process_sample(
        "background"
    )

    # --------------------------------------------------------
    # Calculate correlations
    # --------------------------------------------------------

    print("\nCalculating signal correlations...")

    signal_corr, signal_counts = (
        calculate_correlation_matrix(
            signal
        )
    )

    print("\nCalculating background correlations...")

    background_corr, background_counts = (
        calculate_correlation_matrix(
            background
        )
    )

    # --------------------------------------------------------
    # Print strongest correlations
    # --------------------------------------------------------

    print_strongest_correlations(
        signal_corr,
        "signal",
    )

    print_strongest_correlations(
        background_corr,
        "background",
    )

    # --------------------------------------------------------
    # Individual matrices
    # --------------------------------------------------------

    draw_single_matrix(
        signal_corr,
        "H #rightarrow gg signal",
        "signal_correlation_matrix",
    )

    draw_single_matrix(
        background_corr,
        "e^{+}e^{-} #rightarrow q#bar{q} background",
        "background_correlation_matrix",
    )

    # --------------------------------------------------------
    # Combined dissertation figure
    # --------------------------------------------------------

    draw_combined_matrix(
        signal_corr,
        background_corr,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
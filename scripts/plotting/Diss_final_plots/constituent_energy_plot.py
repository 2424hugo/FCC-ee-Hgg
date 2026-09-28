from pathlib import Path

import awkward as ak
import numpy as np
import ROOT


# ============================================================
# Configuration
# ============================================================

BASE = Path("cache/analysis_dataset")
PLOT_DIR = Path("outputs/plots/diss_plots/two_graph")

OUTPUT_PNG = PLOT_DIR / "constituent_energy.png"
OUTPUT_PDF = PLOT_DIR / "constituent_energy.pdf"
ROOT_OUTPUT = PLOT_DIR / "constituent_energy_histograms.root"

SPLITS = ["train", "validation", "test"]

BINS = 100
XMIN = 0.0
XMAX = 20.0


# ============================================================
# ROOT style
# ============================================================

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetCanvasColor(0)
ROOT.gStyle.SetPadColor(0)
ROOT.gStyle.SetFrameFillColor(0)
ROOT.gStyle.SetLegendBorderSize(0)

ROOT.TH1.AddDirectory(False)


# ============================================================
# Helpers
# ============================================================

def finite_numpy(values):
    values = ak.drop_none(values)
    values = ak.to_numpy(values)
    values = values[np.isfinite(values)]
    return values


def extract_constituent_values(array, category):
    """
    constituent_energy structure:

        event
          -> jet
              -> constituent

    Jet ordering:
        index 0 -> leading jet
        index 1 -> subleading jet
    """

    if category == "leading":
        padded = ak.pad_none(array, 1, axis=1)
        values = padded[:, 0]

    elif category == "subleading":
        padded = ak.pad_none(array, 2, axis=1)
        values = padded[:, 1]

    else:
        raise ValueError(f"Unknown category: {category}")

    # Flatten over events and constituents
    values = ak.flatten(values, axis=None)

    return finite_numpy(values)


def make_histogram(sample, category):
    hist = ROOT.TH1D(
        f"{sample}_constituent_energy_{category}",
        "",
        BINS,
        XMIN,
        XMAX,
    )

    hist.Sumw2()
    hist.SetDirectory(0)

    return hist


def fill_histogram(hist, values):
    for value in values:
        hist.Fill(float(value))


def normalise_density(hist):
    integral = hist.Integral("width")

    if integral > 0:
        hist.Scale(1.0 / integral)


def style_hist(hist, is_signal=True):
    if is_signal:
        line_color = ROOT.kBlue + 1
        fill_color = ROOT.kBlue
    else:
        line_color = ROOT.kRed + 1
        fill_color = ROOT.kRed

    hist.SetLineColor(line_color)
    hist.SetLineWidth(2)
    hist.SetFillColorAlpha(fill_color, 0.35)


def configure_axes(hist, show_ylabel=True):
    hist.GetXaxis().SetTitle("Constituent energy [GeV]")

    if show_ylabel:
        hist.GetYaxis().SetTitle("Normalised density")
    else:
        hist.GetYaxis().SetTitle("")

    hist.GetXaxis().SetTitleSize(0.050)
    hist.GetYaxis().SetTitleSize(0.050)

    hist.GetXaxis().SetLabelSize(0.040)
    hist.GetYaxis().SetLabelSize(0.040)

    hist.GetXaxis().SetTitleOffset(1.10)
    hist.GetYaxis().SetTitleOffset(1.35)


# ============================================================
# Read cache
# ============================================================

def process_sample(sample):

    histograms = {
        "leading": make_histogram(sample, "leading"),
        "subleading": make_histogram(sample, "subleading"),
    }

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

        for i, filepath in enumerate(files, start=1):

            data = ak.from_parquet(
                filepath,
                columns=["constituent_energy"],
            )

            n_events += len(data)
            n_files += 1

            array = data["constituent_energy"]

            for category in ["leading", "subleading"]:

                values = extract_constituent_values(
                    array,
                    category,
                )

                fill_histogram(
                    histograms[category],
                    values,
                )

            if i % 20 == 0:
                print(
                    f"      processed "
                    f"{i}/{len(files)} files"
                )

    print(
        f"  Total: {n_events:,} events "
        f"from {n_files} files"
    )

    return histograms


# ============================================================
# Plotting
# ============================================================

def draw_plot(signal_hists, background_hists):

    canvas = ROOT.TCanvas(
        "canvas_constituent_energy",
        "",
        1300,
        900,
    )

    canvas.cd()

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    title = ROOT.TLatex()
    title.SetNDC()
    title.SetTextAlign(22)
    title.SetTextFont(62)
    title.SetTextSize(0.042)

    title.DrawLatex(
        0.50,
        0.97,
        "Jet Constituent Energy",
    )

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    signal_dummy = ROOT.TH1F(
        "signal_dummy_constituent_energy",
        "",
        1,
        0,
        1,
    )

    background_dummy = ROOT.TH1F(
        "background_dummy_constituent_energy",
        "",
        1,
        0,
        1,
    )

    style_hist(signal_dummy, True)
    style_hist(background_dummy, False)

    legend = ROOT.TLegend(
        0.78,
        0.91,
        0.96,
        0.97,
    )

    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.030)

    legend.AddEntry(
        signal_dummy,
        "Signal",
        "lf",
    )

    legend.AddEntry(
        background_dummy,
        "Background",
        "lf",
    )

    legend.Draw()

    # --------------------------------------------------------
    # Panel layout
    # --------------------------------------------------------

    panels = [
        ("leading", "Leading jet"),
        ("subleading", "Subleading jet"),
    ]

    left = 0.035
    right = 0.99
    gap = 0.02
    bottom = 0.04
    top = 0.90

    total_width = right - left - gap
    pad_width = total_width / 2.0

    keep_alive = [
        title,
        legend,
        signal_dummy,
        background_dummy,
    ]

    # --------------------------------------------------------
    # Draw panels
    # --------------------------------------------------------

    for i, (category, panel_title) in enumerate(panels):

        x1 = left + i * (pad_width + gap)
        x2 = x1 + pad_width

        canvas.cd()

        pad = ROOT.TPad(
            f"pad_constituent_energy_{category}",
            "",
            x1,
            bottom,
            x2,
            top,
        )

        pad.SetLeftMargin(
            0.13 if i == 0 else 0.10
        )
        pad.SetRightMargin(0.02)
        pad.SetBottomMargin(0.12)
        pad.SetTopMargin(0.05)

        pad.Draw()
        pad.cd()

        signal = signal_hists[category].Clone(
            f"signal_{category}_draw"
        )

        background = background_hists[category].Clone(
            f"background_{category}_draw"
        )

        signal.SetDirectory(0)
        background.SetDirectory(0)

        normalise_density(signal)
        normalise_density(background)

        style_hist(
            signal,
            is_signal=True,
        )

        style_hist(
            background,
            is_signal=False,
        )

        true_maximum = max(
            signal.GetMaximum(),
            background.GetMaximum(),
        )

        ymax = 1.25 * true_maximum

        frame = ROOT.TH1D(
            f"frame_constituent_energy_{category}",
            "",
            BINS,
            XMIN,
            XMAX,
        )

        frame.SetDirectory(0)
        frame.SetMinimum(0.0)
        frame.SetMaximum(ymax)

        configure_axes(
            frame,
            show_ylabel=(i == 0),
        )

        frame.Draw("AXIS")

        background.Draw("HIST SAME")
        signal.Draw("HIST SAME")

        ROOT.gPad.RedrawAxis()

        panel_text = ROOT.TLatex()
        panel_text.SetNDC()
        panel_text.SetTextFont(62)
        panel_text.SetTextSize(0.045)

        panel_text.DrawLatex(
            0.13 if i == 0 else 0.10,
            0.965,
            panel_title,
        )

        keep_alive.extend([
            pad,
            frame,
            signal,
            background,
            panel_text,
        ])

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    canvas.cd()
    canvas.Update()

    canvas.SaveAs(str(OUTPUT_PNG))
    canvas.SaveAs(str(OUTPUT_PDF))

    print(f"\nSaved:")
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

    signal_histograms = process_sample(
        "signal"
    )

    background_histograms = process_sample(
        "background"
    )

    # --------------------------------------------------------
    # Save raw ROOT histograms
    # --------------------------------------------------------

    root_file = ROOT.TFile(
        str(ROOT_OUTPUT),
        "RECREATE",
    )

    for hist in signal_histograms.values():
        hist.Write()

    for hist in background_histograms.values():
        hist.Write()

    root_file.Close()

    print(
        f"\nROOT histograms saved to:\n"
        f"  {ROOT_OUTPUT}"
    )

    draw_plot(
        signal_histograms,
        background_histograms,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
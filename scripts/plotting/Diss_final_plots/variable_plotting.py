from pathlib import Path

import awkward as ak
import numpy as np
import ROOT


# ============================================================
# Configuration
# ============================================================

BASE = Path("cache/analysis_dataset")
PLOT_DIR = Path("outputs/plots/diss_plots/two_graph")
ROOT_OUTPUT = PLOT_DIR / "observable_histograms.root"

SPLITS = ["train", "validation", "test"]

# kind:
#   per_jet   -> three panels: all selected jets, leading jet, subleading jet
#   per_event -> single panel
VARIABLES = {
    # --------------------------------------------------------
    # Jet kinematics
    # --------------------------------------------------------
    "jet_energy": {
        "column": "jet_energy",
        "title": "Jet Energy",
        "xlabel": "Jet energy [GeV]",
        "bins": 80,
        "xmin": 0.0,
        "xmax": 80.0,
        "kind": "per_jet",
    },

    "jet_mass": {
        "column": "jet_mass",
        "title": "Jet Mass",
        "xlabel": "Jet mass [GeV]",
        "bins": 80,
        "xmin": 0.0,
        "xmax": 55.0,
        "kind": "per_jet",
    },

    "jet_pt": {
        "column": "jet_pt",
        "title": "Jet Transverse Momentum",
        "xlabel": "Jet p_{T} [GeV]",
        "bins": 80,
        "xmin": 0.0,
        "xmax": 80.0,
        "kind": "per_jet",
    },

    "jet_p": {
        "column": "jet_p",
        "title": "Jet Momentum Magnitude",
        "xlabel": "Jet p [GeV]",
        "bins": 80,
        "xmin": 0.0,
        "xmax": 80.0,
        "kind": "per_jet",
    },

    "jet_theta": {
        "column": "jet_theta",
        "title": "Jet Polar Angle",
        "xlabel": "Jet #theta [rad]",
        "bins": 64,
        "xmin": 0.0,
        "xmax": np.pi,
        "kind": "per_jet",
    },

    # --------------------------------------------------------
    # Jet substructure
    # --------------------------------------------------------
    "constituent_multiplicity": {
        "column": "constituent_multiplicity",
        "title": "Constituent Multiplicity",
        "xlabel": "Constituent multiplicity",
        "bins": 70,
        "xmin": 0.5,
        "xmax": 70.5,
        "kind": "per_jet",
    },

    "e2_beta_0p2": {
        "column": "e2_beta_0p2",
        "title": "e_{2}^{(#beta=0.2)}",
        "xlabel": "e_{2}^{(#beta=0.2)}",
        "bins": 80,
        "xmin": 0.0,
        "xmax": 0.50,
        "kind": "per_jet",
    },

    "e3_beta_0p2": {
        "column": "e3_beta_0p2",
        "title": "e_{3}^{(#beta=0.2)}",
        "xlabel": "e_{3}^{(#beta=0.2)}",
        "bins": 100,
        "xmin": 0.0,
        "xmax": 0.15,
        "kind": "per_jet",
    },

    "c2_beta_0p2": {
        "column": "c2_beta_0p2",
        "title": "C_{2}^{(#beta=0.2)}",
        "xlabel": "C_{2}^{(#beta=0.2)}",
        "bins": 80,
        "xmin": 0.0,
        "xmax": 0.80,
        "kind": "per_jet",
    },

    "d2_beta_0p2": {
        "column": "d2_beta_0p2",
        "title": "D_{2}^{(#beta=0.2)}",
        "xlabel": "D_{2}^{(#beta=0.2)}",
        "bins": 70,
        "xmin": 1.0,
        "xmax": 2.2,
        "kind": "per_jet",
    },

    # --------------------------------------------------------
    # Event-level observable
    # --------------------------------------------------------
    "event_invariant_mass": {
        "column": "event_invariant_mass",
        "title": "Event Invariant Mass",
        "xlabel": "Event invariant mass [GeV]",
        "bins": 60,
        "xmin": 120.0,
        "xmax": 132.0,
        "kind": "per_event",
    },
}


# ============================================================
# ROOT style
# ============================================================

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTitleBorderSize(0)
ROOT.gStyle.SetTitleFillColor(0)
ROOT.gStyle.SetCanvasColor(0)
ROOT.gStyle.SetPadColor(0)
ROOT.gStyle.SetFrameFillColor(0)
ROOT.gStyle.SetLegendBorderSize(0)

ROOT.gStyle.SetPadLeftMargin(0.14)
ROOT.gStyle.SetPadBottomMargin(0.14)
ROOT.gStyle.SetPadRightMargin(0.05)
ROOT.gStyle.SetPadTopMargin(0.08)

ROOT.gStyle.SetTitleFont(42, "XYZ")
ROOT.gStyle.SetLabelFont(42, "XYZ")
ROOT.gStyle.SetTextFont(42)

ROOT.TH1.AddDirectory(False)


# ============================================================
# Helpers
# ============================================================

def make_histogram(sample, variable, category, config):
    hist = ROOT.TH1D(
        f"{sample}_{variable}_{category}",
        "",
        config["bins"],
        config["xmin"],
        config["xmax"],
    )
    hist.Sumw2()
    hist.SetDirectory(0)
    return hist


def finite_numpy(values):
    values = ak.drop_none(values)
    values = ak.to_numpy(values)
    values = values[np.isfinite(values)]
    return values


def extract_jet_values(array, category):
    """
    Jet arrays are ordered by energy:
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

    return finite_numpy(values)


def extract_event_values(array):
    return finite_numpy(array)


def fill_histogram(hist, values):
    for value in values:
        hist.Fill(float(value))

def draw_main_title(text):
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextAlign(22)   # centred
    latex.SetTextFont(62)
    latex.SetTextSize(0.040)
    latex.DrawLatex(0.50, 0.965, text)
    return latex


# ============================================================
# Read dataset
# ============================================================

def process_sample(sample):

    histograms = {}

    for variable, config in VARIABLES.items():

        if config["kind"] == "per_jet":
            histograms[variable] = {
                "leading": make_histogram(
                    sample, variable, "leading", config
                ),
                "subleading": make_histogram(
                    sample, variable, "subleading", config
                ),
            }
        else:
            histograms[variable] = {
                "event": make_histogram(sample, variable, "event", config),
            }

    n_events = 0
    n_files = 0

    print(f"\nProcessing {sample}")

    for split in SPLITS:
        folder = BASE / sample / split
        files = sorted(folder.glob("events_*_part_*.parquet"))

        print(f"  {split:12s}: {len(files):4d} files")

        for i, filepath in enumerate(files, start=1):

            data = ak.from_parquet(filepath)

            n_events += len(data)
            n_files += 1

            for variable, config in VARIABLES.items():

                array = data[config["column"]]

                if config["kind"] == "per_jet":

                    for category in ["leading", "subleading"]:
                        values = extract_jet_values(array, category)
                        fill_histogram(histograms[variable][category], values)

                else:
                    values = extract_event_values(array)
                    fill_histogram(histograms[variable]["event"], values)

            if i % 20 == 0:
                print(f"      processed {i}/{len(files)} files")

    print(f"  Total: {n_events:,} events from {n_files} files")

    return histograms


# ============================================================
# Histogram styling and normalisation
# ============================================================

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


def configure_axes(hist, xlabel, show_ylabel=True):
    hist.GetXaxis().SetTitle(xlabel)

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

    hist.GetXaxis().SetNdivisions(510)
    hist.GetYaxis().SetNdivisions(510)


# ============================================================
# Plotting
# ============================================================

def draw_two_panel_plot(variable, config, signal_hists, background_hists):

    canvas = ROOT.TCanvas(
        f"canvas_{variable}",
        "",
        1300,
        720,
    )

    # ========================================================
    # Main title
    # ========================================================

    canvas.cd()

    main_title = ROOT.TLatex()
    main_title.SetNDC()
    main_title.SetTextAlign(22)
    main_title.SetTextFont(62)
    main_title.SetTextSize(0.042)

    main_title.DrawLatex(
        0.50,
        0.965,
        config["title"],
    )

    # ========================================================
    # Shared legend
    # ========================================================

    signal_dummy = ROOT.TH1F(
        f"signal_dummy_{variable}",
        "",
        1,
        0,
        1,
    )

    background_dummy = ROOT.TH1F(
        f"background_dummy_{variable}",
        "",
        1,
        0,
        1,
    )

    style_hist(signal_dummy, True)
    style_hist(background_dummy, False)

    legend = ROOT.TLegend(
        0.76,
        0.915,
        0.95,
        0.975,
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

    # ========================================================
    # Two panels
    # ========================================================

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
        main_title,
        legend,
        signal_dummy,
        background_dummy,
    ]

    for i, (category, panel_title) in enumerate(panels):

        x1 = left + i * (pad_width + gap)
        x2 = x1 + pad_width

        # Always return to parent canvas
        canvas.cd()

        pad = ROOT.TPad(
            f"pad_{variable}_{category}",
            "",
            x1,
            bottom,
            x2,
            top,
        )

        pad.SetLeftMargin(
            0.15 if i == 0 else 0.12
        )
        pad.SetRightMargin(0.035)
        pad.SetBottomMargin(0.15)
        pad.SetTopMargin(0.10)

        pad.Draw()
        pad.cd()

        # ====================================================
        # Histograms
        # ====================================================

        signal = signal_hists[category].Clone(
            f"{variable}_{category}_signal_draw"
        )

        background = background_hists[category].Clone(
            f"{variable}_{category}_background_draw"
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

        # ====================================================
        # Y-axis range
        # ====================================================

        true_maximum = max(
            signal.GetMaximum(),
            background.GetMaximum(),
        )

        ymax = 1.25 * true_maximum

        # ====================================================
        # Independent frame
        # ====================================================

        frame = ROOT.TH1D(
            f"frame_{variable}_{category}",
            "",
            config["bins"],
            config["xmin"],
            config["xmax"],
        )

        frame.SetDirectory(0)
        frame.SetMinimum(0.0)
        frame.SetMaximum(ymax)

        configure_axes(
            frame,
            config["xlabel"],
            show_ylabel=(i == 0),
        )

        # ====================================================
        # Draw
        # ====================================================

        frame.Draw("AXIS")
        background.Draw("HIST SAME")
        signal.Draw("HIST SAME")

        ROOT.gPad.RedrawAxis()

        # ====================================================
        # Panel title
        # ====================================================

        panel_text = ROOT.TLatex()
        panel_text.SetNDC()
        panel_text.SetTextFont(62)
        panel_text.SetTextSize(0.045)

        panel_text.DrawLatex(
            0.15 if i == 0 else 0.12,
            0.94,
            panel_title,
        )

        keep_alive.extend([
            pad,
            frame,
            signal,
            background,
            panel_text,
        ])

    # ========================================================
    # Save
    # ========================================================

    canvas.cd()
    canvas.Update()

    png = PLOT_DIR / f"{variable}.png"
    pdf = PLOT_DIR / f"{variable}.pdf"

    canvas.SaveAs(str(png))
    canvas.SaveAs(str(pdf))

    print(f"  Saved {png}")

    canvas.Close()

def draw_single_panel_plot(variable, config, signal_hist, background_hist):

    canvas = ROOT.TCanvas(
        f"canvas_{variable}",
        "",
        900,
        720,
    )

    canvas.cd()

    # ========================================================
    # Main title
    # ========================================================

    main_title = ROOT.TLatex()
    main_title.SetNDC()
    main_title.SetTextAlign(22)
    main_title.SetTextFont(62)
    main_title.SetTextSize(0.040)

    main_title.DrawLatex(
        0.50,
        0.965,
        config["title"],
    )

    # ========================================================
    # Shared legend
    # ========================================================

    signal_dummy = ROOT.TH1F(
        f"signal_dummy_{variable}",
        "",
        1,
        0,
        1,
    )

    background_dummy = ROOT.TH1F(
        f"background_dummy_{variable}",
        "",
        1,
        0,
        1,
    )

    style_hist(signal_dummy, True)
    style_hist(background_dummy, False)

    legend = ROOT.TLegend(
        0.74,
        0.915,
        0.96,
        0.975,
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

    # ========================================================
    # Main plotting pad
    # ========================================================

    canvas.cd()

    pad = ROOT.TPad(
        f"pad_{variable}",
        "",
        0.10,
        0.10,
        0.95,
        0.88,
    )

    pad.SetLeftMargin(0.13)
    pad.SetRightMargin(0.02)
    pad.SetBottomMargin(0.12)
    pad.SetTopMargin(0.05)
    
    pad.Draw()
    pad.cd()

    # ========================================================
    # Histograms
    # ========================================================

    signal = signal_hist.Clone(
        f"{variable}_signal_draw"
    )

    background = background_hist.Clone(
        f"{variable}_background_draw"
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

    ymax = 1.20 * true_maximum

    # ========================================================
    # Independent frame
    # ========================================================

    frame = ROOT.TH1D(
        f"frame_{variable}",
        "",
        config["bins"],
        config["xmin"],
        config["xmax"],
    )

    frame.SetDirectory(0)
    frame.SetMinimum(0.0)
    frame.SetMaximum(ymax)

    configure_axes(
        frame,
        config["xlabel"],
        show_ylabel=True,
    )

    frame.Draw("AXIS")

    background.Draw("HIST SAME")
    signal.Draw("HIST SAME")

    ROOT.gPad.RedrawAxis()

    # ========================================================
    # Save
    # ========================================================

    canvas.cd()
    canvas.Update()

    png = PLOT_DIR / f"{variable}.png"
    pdf = PLOT_DIR / f"{variable}.pdf"

    canvas.SaveAs(str(png))
    canvas.SaveAs(str(pdf))

    print(f"  Saved {png}")

    canvas.Close()

# ============================================================
# Main
# ============================================================

def main():

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    signal_histograms = process_sample("signal")
    background_histograms = process_sample("background")

    # --------------------------------------------------------
    # Save raw histograms
    # --------------------------------------------------------
    root_file = ROOT.TFile(str(ROOT_OUTPUT), "RECREATE")

    for variable in VARIABLES:
        for hist in signal_histograms[variable].values():
            hist.Write()
        for hist in background_histograms[variable].values():
            hist.Write()

    root_file.Close()

    print(f"\nROOT histograms saved to:\n  {ROOT_OUTPUT}")

    # --------------------------------------------------------
    # Make plots
    # --------------------------------------------------------
    print("\nCreating plots")

    for variable, config in VARIABLES.items():

        if config["kind"] == "per_jet":
            draw_two_panel_plot(
                variable,
                config,
                signal_histograms[variable],
                background_histograms[variable],
            )
        else:
            draw_single_panel_plot(
                variable,
                config,
                signal_histograms[variable]["event"],
                background_histograms[variable]["event"],
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
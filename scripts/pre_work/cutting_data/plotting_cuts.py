import awkward as ak
import numpy as np
import ROOT
import uproot

from scripts.data_processing.build_analysis_data_refactored import load_splits


MASS_CUT = 120.0
SPLIT_NAMES = ("train", "validation", "test")

splits = load_splits("cache/dataset_splits.json")


def get_all_files(sample):
    """
    Combine train, validation and test ROOT files for one sample.
    """
    return [
        file_path
        for split_name in SPLIT_NAMES
        for file_path in splits[sample][split_name]
    ]


def calculate_event_masses(input_files):
    """
    Calculate reconstructed event invariant mass directly
    from the original ROOT files.
    """

    masses = []

    branches = [
        "ReconstructedParticles/ReconstructedParticles.energy",
        "ReconstructedParticles/ReconstructedParticles.momentum.x",
        "ReconstructedParticles/ReconstructedParticles.momentum.y",
        "ReconstructedParticles/ReconstructedParticles.momentum.z",
    ]

    for file_number, input_file in enumerate(input_files, start=1):

        print(
            f"Reading file {file_number}/{len(input_files)}: "
            f"{input_file}"
        )

        with uproot.open(input_file) as root_file:

            tree = root_file["events"]

            data = tree.arrays(
                branches,
                library="ak",
            )

            E = data[
                "ReconstructedParticles/ReconstructedParticles.energy"
            ]
            px = data[
                "ReconstructedParticles/ReconstructedParticles.momentum.x"
            ]
            py = data[
                "ReconstructedParticles/ReconstructedParticles.momentum.y"
            ]
            pz = data[
                "ReconstructedParticles/ReconstructedParticles.momentum.z"
            ]

            # Sum reconstructed four-momentum over all particles
            E_event = ak.sum(E, axis=1)
            px_event = ak.sum(px, axis=1)
            py_event = ak.sum(py, axis=1)
            pz_event = ak.sum(pz, axis=1)

            # Event invariant mass squared
            mass_squared = (
                E_event**2
                - px_event**2
                - py_event**2
                - pz_event**2
            )

            # Protect against tiny negative numerical values
            event_mass = np.sqrt(
                np.maximum(
                    ak.to_numpy(mass_squared),
                    0.0,
                )
            )

            event_mass = event_mass[np.isfinite(event_mass)]

            masses.append(event_mass)

    return np.concatenate(masses)


# ------------------------------------------------------------------
# Original ROOT files
# Only use the first 10 files from each sample
# ------------------------------------------------------------------

signal_files = get_all_files("signal")[:10]
background_files = get_all_files("background")[:10]

print(f"Signal ROOT files used: {len(signal_files)}")
print(f"Background ROOT files used: {len(background_files)}")


# ------------------------------------------------------------------
# Calculate event invariant masses
# ------------------------------------------------------------------

print("\nCalculating signal masses")
sig_mass = calculate_event_masses(signal_files)

print("\nCalculating background masses")
bkg_mass = calculate_event_masses(background_files)


# ------------------------------------------------------------------
# Apply mass selection
# ------------------------------------------------------------------

sig_mass_cut = sig_mass[sig_mass > MASS_CUT]
bkg_mass_cut = bkg_mass[bkg_mass > MASS_CUT]

print("\nEvent counts")
print(f"Signal before cut:     {len(sig_mass)}")
print(f"Signal after cut:      {len(sig_mass_cut)}")
print(f"Background before cut: {len(bkg_mass)}")
print(f"Background after cut:  {len(bkg_mass_cut)}")

print("\nSelection efficiencies")
print(
    f"Signal:     {len(sig_mass_cut) / len(sig_mass):.4f}"
)
print(
    f"Background: {len(bkg_mass_cut) / len(bkg_mass):.4f}"
)


# ------------------------------------------------------------------
# ROOT plotting
# ------------------------------------------------------------------

ROOT.gROOT.SetBatch(True)

# ROOT style
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTitleBorderSize(0)
ROOT.gStyle.SetLegendBorderSize(0)

ROOT.gStyle.SetPadLeftMargin(0.12)
ROOT.gStyle.SetPadRightMargin(0.04)
ROOT.gStyle.SetPadBottomMargin(0.12)
ROOT.gStyle.SetPadTopMargin(0.10)

ROOT.gStyle.SetTitleSize(0.045, "XYZ")
ROOT.gStyle.SetLabelSize(0.04, "XYZ")



# ------------------------------------------------------------------
# Canvas with two pads
# ------------------------------------------------------------------

canvas = ROOT.TCanvas(
    "canvas",
    "Event invariant mass",
    1400,
    550,
)

canvas.Divide(2, 1)


# ==================================================================
# Left panel: before selection
# ==================================================================

canvas.cd(1)

ROOT.gPad.SetTicks(1, 1)
ROOT.gPad.SetLeftMargin(0.13)
ROOT.gPad.SetRightMargin(0.03)
ROOT.gPad.SetBottomMargin(0.13)
ROOT.gPad.SetTopMargin(0.10)


# Histograms
h_sig_before = ROOT.TH1D(
    "h_sig_before",
    "",
    100,
    80.0,
    130.0,
)

h_bkg_before = ROOT.TH1D(
    "h_bkg_before",
    "",
    100,
    80.0,
    130.0,
)


# Fill histograms
for value in sig_mass:
    h_sig_before.Fill(float(value))

for value in bkg_mass:
    h_bkg_before.Fill(float(value))


# Normalise to unit area
if h_sig_before.Integral() > 0:
    h_sig_before.Scale(1.0 / h_sig_before.Integral())

if h_bkg_before.Integral() > 0:
    h_bkg_before.Scale(1.0 / h_bkg_before.Integral())


# Styling
h_sig_before.SetLineColor(ROOT.kBlue + 1)
h_sig_before.SetFillColorAlpha(ROOT.kBlue, 0.35)
h_sig_before.SetLineWidth(2)

h_bkg_before.SetLineColor(ROOT.kRed + 1)
h_bkg_before.SetFillColorAlpha(ROOT.kRed, 0.25)
h_bkg_before.SetLineWidth(2)


# Axis labels
h_sig_before.GetXaxis().SetTitle(
    "Event invariant mass [GeV]"
)

h_sig_before.GetYaxis().SetTitle(
    "Normalised density"
)

h_sig_before.GetXaxis().CenterTitle()
h_sig_before.GetYaxis().CenterTitle()


# Set y range to fit both histograms
max_before = max(
    h_sig_before.GetMaximum(),
    h_bkg_before.GetMaximum(),
)

h_sig_before.SetMaximum(
    max_before * 1.15
)


# Draw
h_sig_before.Draw("HIST")
h_bkg_before.Draw("HIST SAME")


# Mass cut line
cut_line = ROOT.TLine(
    MASS_CUT,
    0.0,
    MASS_CUT,
    max_before * 1.15,
)

cut_line.SetLineColor(ROOT.kGray + 2)
cut_line.SetLineStyle(7)
cut_line.SetLineWidth(2)

cut_line.Draw("SAME")


# Title
title_before = ROOT.TLatex()
title_before.SetNDC()
title_before.SetTextAlign(22)
title_before.SetTextSize(0.045)

title_before.DrawLatex(
    0.50,
    0.94,
    "Before event-mass selection",
)


# Legend
legend_before = ROOT.TLegend(
    0.16,
    0.72,
    0.48,
    0.88,
)

legend_before.SetFillStyle(0)
legend_before.SetBorderSize(0)
legend_before.SetTextSize(0.035)

legend_before.AddEntry(
    h_sig_before,
    "Signal",
    "f",
)

legend_before.AddEntry(
    h_bkg_before,
    "Background",
    "f",
)

legend_before.AddEntry(
    cut_line,
    "m_{event} = 120 GeV",
    "l",
)

legend_before.Draw()


# ==================================================================
# Right panel: after selection
# ==================================================================

canvas.cd(2)

ROOT.gPad.SetTicks(1, 1)
ROOT.gPad.SetLeftMargin(0.13)
ROOT.gPad.SetRightMargin(0.03)
ROOT.gPad.SetBottomMargin(0.13)
ROOT.gPad.SetTopMargin(0.10)


h_sig_after = ROOT.TH1D(
    "h_sig_after",
    "",
    100,
    120.0,
    130.0,
)

h_bkg_after = ROOT.TH1D(
    "h_bkg_after",
    "",
    100,
    120.0,
    130.0,
)


# Fill
for value in sig_mass_cut:
    h_sig_after.Fill(float(value))

for value in bkg_mass_cut:
    h_bkg_after.Fill(float(value))


# Normalise
if h_sig_after.Integral() > 0:
    h_sig_after.Scale(
        1.0 / h_sig_after.Integral()
    )

if h_bkg_after.Integral() > 0:
    h_bkg_after.Scale(
        1.0 / h_bkg_after.Integral()
    )


# Styling
h_sig_after.SetLineColor(ROOT.kBlue + 1)
h_sig_after.SetFillColorAlpha(ROOT.kBlue, 0.35)
h_sig_after.SetLineWidth(2)

h_bkg_after.SetLineColor(ROOT.kRed + 1)
h_bkg_after.SetFillColorAlpha(ROOT.kRed, 0.25)
h_bkg_after.SetLineWidth(2)


# Axes
h_sig_after.GetXaxis().SetTitle(
    "Event invariant mass [GeV]"
)

h_sig_after.GetYaxis().SetTitle(
    "Normalised density"
)

h_sig_after.GetXaxis().CenterTitle()
h_sig_after.GetYaxis().CenterTitle()


max_after = max(
    h_sig_after.GetMaximum(),
    h_bkg_after.GetMaximum(),
)

h_sig_after.SetMaximum(
    max_after * 1.15
)


# Draw
h_sig_after.Draw("HIST")
h_bkg_after.Draw("HIST SAME")


# Title
title_after = ROOT.TLatex()
title_after.SetNDC()
title_after.SetTextAlign(22)
title_after.SetTextSize(0.045)

title_after.DrawLatex(
    0.50,
    0.94,
    "After m_{event} > 120 GeV selection",
)


# Legend
legend_after = ROOT.TLegend(
    0.66,
    0.77,
    0.91,
    0.88,
)

legend_after.SetFillStyle(0)
legend_after.SetBorderSize(0)
legend_after.SetTextSize(0.035)

legend_after.AddEntry(
    h_sig_after,
    "Signal",
    "f",
)

legend_after.AddEntry(
    h_bkg_after,
    "Background",
    "f",
)

legend_after.Draw()


# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

canvas.Update()

canvas.SaveAs(
    "outputs/plots/diss_plots/"
    "event_mass_before_after_ROOT.png"
)

canvas.SaveAs(
    "outputs/plots/diss_plots/"
    "event_mass_before_after_ROOT.pdf"
)
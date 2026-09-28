import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import uproot
import ROOT
from array import array

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
# Event-mass cut optimisation
# ------------------------------------------------------------------

cuts = np.arange(80, 130, 0.5)

signal_eff = []
background_eff = []
significance_proxy = []

for cut in cuts:

    sig_pass = np.sum(sig_mass > cut)
    bkg_pass = np.sum(bkg_mass > cut)

    sig_eff = sig_pass / len(sig_mass)
    bkg_eff = bkg_pass / len(bkg_mass)

    signal_eff.append(sig_eff)
    background_eff.append(bkg_eff)

    if bkg_eff > 0:
        significance_proxy.append(
            sig_eff / np.sqrt(bkg_eff)
        )
    else:
        significance_proxy.append(0.0)


signal_eff = np.asarray(signal_eff)
background_eff = np.asarray(background_eff)
significance_proxy = np.asarray(significance_proxy)

# Best threshold according to epsilon_S / sqrt(epsilon_B)
best = np.argmax(significance_proxy)

best_cut = cuts[best]
best_sig_eff = signal_eff[best]
best_bkg_eff = background_eff[best]
best_proxy = significance_proxy[best]

print("\nMass-cut optimisation")
print(f"Best cut = {best_cut:.1f} GeV")
print(f"Signal efficiency = {best_sig_eff:.4f}")
print(f"Background efficiency = {best_bkg_eff:.4f}")
print(
    r"epsilon_S / sqrt(epsilon_B) = "
    f"{best_proxy:.4f}"
)


# ------------------------------------------------------------------
# ROOT / TMVA-style optimisation plot
# ------------------------------------------------------------------

ROOT.gROOT.SetBatch(True)

ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTitleBorderSize(0)
ROOT.gStyle.SetLegendBorderSize(0)

ROOT.gStyle.SetPadLeftMargin(0.12)
ROOT.gStyle.SetPadRightMargin(0.04)
ROOT.gStyle.SetPadBottomMargin(0.12)
ROOT.gStyle.SetPadTopMargin(0.08)

ROOT.gStyle.SetTitleSize(0.045, "XYZ")
ROOT.gStyle.SetLabelSize(0.04, "XYZ")

# Normalise optimisation proxy to its maximum
proxy_norm = significance_proxy / np.max(significance_proxy)

x_vals = array("d", cuts.astype(float))
sig_vals = array("d", signal_eff.astype(float))
bkg_vals = array("d", background_eff.astype(float))
proxy_vals = array("d", proxy_norm.astype(float))

n_points = len(cuts)

canvas = ROOT.TCanvas(
    "canvas",
    "Event-mass cut optimisation",
    900,
    700,
)

canvas.SetTicks(1, 1)

# Main frame
frame = canvas.DrawFrame(
    80.0,
    0.0,
    130.0,
    1.05,
)

frame.SetTitle("Event-mass cut optimisation")
frame.GetXaxis().SetTitle("Event-mass threshold [GeV]")
frame.GetYaxis().SetTitle("Efficiency / normalised metric")

frame.GetXaxis().CenterTitle()
frame.GetYaxis().CenterTitle()

# Signal efficiency
graph_sig = ROOT.TGraph(
    n_points,
    x_vals,
    sig_vals,
)

graph_sig.SetLineColor(ROOT.kBlue + 1)
graph_sig.SetLineWidth(3)
graph_sig.Draw("L SAME")

# Background efficiency
graph_bkg = ROOT.TGraph(
    n_points,
    x_vals,
    bkg_vals,
)

graph_bkg.SetLineColor(ROOT.kRed + 1)
graph_bkg.SetLineWidth(3)
graph_bkg.Draw("L SAME")

# Normalised optimisation metric
graph_proxy = ROOT.TGraph(
    n_points,
    x_vals,
    proxy_vals,
)

graph_proxy.SetLineColor(ROOT.kBlack)
graph_proxy.SetLineWidth(3)
graph_proxy.SetLineStyle(2)
graph_proxy.Draw("L SAME")

# Selected cut
selected_line = ROOT.TLine(
    MASS_CUT,
    0.0,
    MASS_CUT,
    1.05,
)

selected_line.SetLineColor(ROOT.kGray + 2)
selected_line.SetLineStyle(7)
selected_line.SetLineWidth(2)
selected_line.Draw("SAME")

# Optimum marker
best_proxy_norm = proxy_norm[best]

best_marker = ROOT.TMarker(
    best_cut,
    best_proxy_norm,
    20,
)

best_marker.SetMarkerColor(ROOT.kBlack)
best_marker.SetMarkerSize(1.2)
best_marker.Draw("SAME")

# Cleaner legend
legend = ROOT.TLegend(
    0.15,
    0.15,
    0.47,
    0.32,
)

legend.SetFillStyle(0)
legend.SetBorderSize(0)
legend.SetTextSize(0.035)

legend.AddEntry(
    graph_sig,
    "Signal efficiency",
    "l",
)

legend.AddEntry(
    graph_bkg,
    "Background efficiency",
    "l",
)

legend.AddEntry(
    graph_proxy,
    "Normalised optimisation metric",
    "l",
)

legend.AddEntry(
    selected_line,
    "Selected cut: 120 GeV",
    "l",
)

legend.AddEntry(
    best_marker,
    f"Optimum: {best_cut:.1f} GeV",
    "p",
)

legend.Draw()

canvas.RedrawAxis()

canvas.SaveAs(
    "outputs/plots/diss_plots/"
    "event_mass_cut_optimisation_ROOT.pdf"
)

canvas.SaveAs(
    "outputs/plots/diss_plots/"
    "event_mass_cut_optimisation_ROOT.png"
)
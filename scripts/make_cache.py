import uproot
import awkward as ak
import glob
from config import *

signal_path = SIGNAL_PATH
bkg_path = BACKGROUND_PATH

sig_files = glob.glob(signal_path)[:12]
bkg_files = glob.glob(bkg_path)[:12]

print("Signal files found:", len(sig_files))
print("Background files found:", len(bkg_files))


def make_cache(branches, signal_out, background_out):
    signal_data = uproot.concatenate(
        [f"{f}:events" for f in sig_files],
        expressions=branches,
        library="ak",
    )

    bkg_data = uproot.concatenate(
        [f"{f}:events" for f in bkg_files],
        expressions=branches,
        library="ak",
    )

    ak.to_parquet(signal_data, signal_out, compression=None)
    ak.to_parquet(bkg_data, background_out, compression=None)

    print(f"Saved {signal_out}")
    print(f"Saved {background_out}")

jet_branches = [
    "Jet/Jet.energy",
    "Jet/Jet.mass",
    "Jet/Jet.momentum.x",
    "Jet/Jet.momentum.y",
    "Jet/Jet.momentum.z",
    "Jet/Jet.charge",
    "Jet/Jet.particles_begin",
    "Jet/Jet.particles_end",
]

reco_branches = [
    "ReconstructedParticles/ReconstructedParticles.energy",
    "ReconstructedParticles/ReconstructedParticles.mass",
    "ReconstructedParticles/ReconstructedParticles.charge",
    "ReconstructedParticles/ReconstructedParticles.momentum.x",
    "ReconstructedParticles/ReconstructedParticles.momentum.y",
    "ReconstructedParticles/ReconstructedParticles.momentum.z",
]

track_branches = [
    "EFlowTrack_1/EFlowTrack_1.D0",
    "EFlowTrack_1/EFlowTrack_1.Z0",
    "EFlowTrack_1/EFlowTrack_1.phi",
    "EFlowTrack_1/EFlowTrack_1.omega",
    "EFlowTrack_1/EFlowTrack_1.tanLambda",
]

neutral_photon_met_branches = [
    "EFlowNeutralHadron/EFlowNeutralHadron.energy",
    "EFlowPhoton/EFlowPhoton.energy",
    "MissingET/MissingET.energy",
    "MissingET/MissingET.momentum.x",
    "MissingET/MissingET.momentum.y",
    "MissingET/MissingET.momentum.z",
]

energy_func = [
        "Jet/Jet.particles_begin",
        "Jet/Jet.particles_end",
        "ReconstructedParticles/ReconstructedParticles.momentum.x",
        "ReconstructedParticles/ReconstructedParticles.momentum.y",
        "ReconstructedParticles/ReconstructedParticles.momentum.z",
        "ReconstructedParticles/ReconstructedParticles.energy",
]

make_cache(energy_func,
        "cache/signal_jetParticles.parquet",
        "cache/background_jetParticles.parquet")


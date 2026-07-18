import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

mass_cut = 120

sig_data = ak.from_parquet("cache/signal_jetParticles.parquet")
bkg_data = ak.from_parquet("cache/background_jetParticles.parquet")

sig_mass_record = ak.from_parquet("cache/signal_event_mass.parquet")
bkg_mass_record = ak.from_parquet("cache/background_event_mass.parquet")

sig_mass = sig_mass_record["event_invariant_mass"]
bkg_mass = bkg_mass_record["event_invariant_mass"]


sig_mask = sig_mass > mass_cut
bkg_mask = bkg_mass > mass_cut

sig_cut = sig_data[sig_mask]
bkg_cut = bkg_data[bkg_mask]

ak.to_parquet(
        sig_cut,
        "cache/mass_cut/signal_mass_gt_120.parquet",
        compression=None
)

ak.to_parquet(
        bkg_cut,
        "cache/mass_cut/background_mass_gt_120.parquet",
        compression=None
)

print(f"Signal: {len(sig_data)} -> {len(sig_cut)}")
print(f"Background: {len(bkg_data)} -> {len(bkg_cut)}")

sig_mass_cut = sig_mass[sig_mask]
bkg_mass_cut = bkg_mass[bkg_mask]

print("Minimum cut signal mass:", ak.min(sig_mass_cut))
print("Minimum cut background mass:", ak.min(bkg_mass_cut))

print("Signal efficiency:", len(sig_cut) / len(sig_data))
print("Background efficiency:", len(bkg_cut) / len(bkg_data))


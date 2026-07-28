import awkward as ak
import numpy as np

sig = ak.from_parquet(
            "cache/mass_cut/signal_leading_two_jets_mass_gt_120.parquet"
            )

bkg = ak.from_parquet(
            "cache/mass_cut/background_leading_two_jets_mass_gt_120.parquet"
            )

sig_njets = ak.num(sig["Jet/Jet.energy"], axis=1)
bkg_njets = ak.num(bkg["Jet/Jet.energy"], axis=1)

print("Signal unique jet counts:", np.unique(ak.to_numpy(sig_njets)))
print("Background unique jet counts:", np.unique(ak.to_numpy(bkg_njets)))

print("Signal minimum mass:", ak.min(sig["event_invariant_mass"]))
print("Background minimum mass:", ak.min(bkg["event_invariant_mass"]))

sig_energy = sig["Jet/Jet.energy"]
bkg_energy = bkg["Jet/Jet.energy"]

print("Signal correctly ordered:", ak.all(sig_energy[:, 0] >= sig_energy[:, 1]))
print("Background correctly ordered:", ak.all(bkg_energy[:, 0] >= bkg_energy[:, 1]))

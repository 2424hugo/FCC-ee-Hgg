#imports
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

# loading in data
sig = ak.from_parquet("cache/signal_jet.parquet")
bkg = ak.from_parquet("cache/background_jet.parquet")

# function that takes in jet x and y momentum to calculate transverse momentum.
def jet_transverse_momentum(data):

    px = data["Jet/Jet.momentum.x"]
    py = data["Jet/Jet.momentum.y"]

    return np.sqrt(px**2 + py**2)

# run event_iver_mass for sig and bkg
sig_all_transverseMomentum = ak.to_numpy(ak.flatten(jet_transverse_momentum(sig)))
bkg_all_transverseMomentum = ak.to_numpy(ak.flatten(jet_transverse_momentum(bkg)))


# plot data and save png
plt.figure(figsize=(8,6))

plt.hist(sig_all_transverseMomentum, bins = 150, label = 'Signal', density=True, alpha = 0.5)
plt.hist(bkg_all_transverseMomentum, bins = 150, label = 'Background', density=True, alpha = 0.5)

plt.xlabel(r"All-Jet tranverse momentum [GeV]")
plt.ylabel("Normalized Density")
plt.title("Transverse momentum of All Reconstructed Jets per Event")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/hl_observables/Jet_transverseMomentum.png",dpi=300)
plt.close

sig_jet_pt = jet_transverse_momentum(sig)
bkg_jet_pt = jet_transverse_momentum(bkg)

ak.to_parquet(
    ak.Array({"jet_transverseMomentum": sig_jet_pt}),
    "cache/signal_jet_transverseMomentum.parquet",
    compression=None
)


ak.to_parquet(
    ak.Array({"jet_transverseMomentum": bkg_jet_pt}),
    "cache/background_jet_transverseMomentum.parquet",
    compression=None
)

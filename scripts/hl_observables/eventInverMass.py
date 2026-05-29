#imports
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

# loading in data
sig = ak.from_parquet("cache/signal_jet.parquet")
bkg = ak.from_parquet("cache/background_jet.parquet")

# function that takes in jet energy and momentums and finds the inveriant mass of the events
def event_inver_mass(data):

    E = ak.sum(data["Jet/Jet.energy"], axis=1)

    px = ak.sum(data["Jet/Jet.momentum.x"], axis=1)
    py = ak.sum(data["Jet/Jet.momentum.y"], axis=1)
    pz = ak.sum(data["Jet/Jet.momentum.z"], axis=1)

    mass2 = E**2 - px**2 - py**2 - pz**2

    # Numerical protection
    mass2 = ak.where(mass2 < 0, 0, mass2)

    return np.sqrt(mass2)


# run event_iver_mass for sig and bkg
sig_alljet_mass = event_inver_mass(sig)
bkg_alljet_mass = event_inver_mass(bkg)

sig_alljet_mass = np.asarray(event_inver_mass(sig))
bkg_alljet_mass = np.asarray(event_inver_mass(bkg))

# plot data and save png
plt.figure(figsize=(8,6))

plt.hist(sig_alljet_mass, bins = 150, label = 'Signal', density=True, alpha = 0.5)
plt.hist(bkg_alljet_mass, bins = 150, label = 'Background', density=True, alpha = 0.5)

plt.xlabel(r"All-Jet Invariant Mass [GeV]")
plt.ylabel("Normalized Density")
plt.title("Invariant Mass of All Reconstructed Jets per Event")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/hl_observables/event_invariant_mass.png",dpi=300)
plt.close

ak.to_parquet(
    ak.Array({"event_invariant_mass": sig_alljet_mass}),
    "cache/signal_event_mass.parquet",
    compression=None
)

ak.to_parquet(
    ak.Array({"event_invariant_mass": bkg_alljet_mass}),
    "cache/background_event_mass.parquet",
    compression=None
)

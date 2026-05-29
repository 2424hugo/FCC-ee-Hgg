# imports
import awkward as ak
import matplotlib.pyplot as plt

# loading in data
sig = ak.from_parquet("cache/signal_jet.parquet")
bkg = ak.from_parquet("cache/background_jet.parquet")

# function for comparing features for signal and background
def plotting_feature(feature, label,bins=100):
	plt.figure(figsize=(8, 6))
	plt.hist(ak.flatten(sig[feature]), bins = bins, label = 'Signal', density=True, alpha = 0.5)
	plt.hist(ak.flatten(bkg[feature]), bins = bins, label = 'Background', density=True, alpha = 0.5)

	plt.xlabel(label)
	plt.ylabel("Normalized Density")
	plt.title(feature)
	plt.legend()
	plt.tight_layout()

	safe_name = feature.replace("/", "_")
	plt.savefig(f"outputs/plots/{safe_name}.png", dpi=300)
	plt.close()

# jet features
features = {
    "Jet/Jet.energy": "Jet Energy [GeV]",
    "Jet/Jet.mass": "Jet Mass [GeV]",
    "Jet/Jet.momentum.x": r"Jet $p_x$ [GeV]",
    "Jet/Jet.momentum.y": r"Jet $p_y$ [GeV]",
    "Jet/Jet.momentum.z": r"Jet $p_z$ [GeV]",
    "Jet/Jet.charge": "Jet Charge",
}

# loop over features
for feature, label in features.items():
    plotting_feature(feature, label)

# plot the particles
sig_n_constituents = (
    sig["Jet/Jet.particles_end"]
    - sig["Jet/Jet.particles_begin"]
)

bkg_n_constituents = (
    bkg["Jet/Jet.particles_end"]
    - bkg["Jet/Jet.particles_begin"]
)

plt.figure(figsize=(8,6))

plt.hist(
    ak.flatten(sig_n_constituents),
    bins=95,
    density=True,
    alpha=0.5,
    label="Signal"
)

plt.hist(
    ak.flatten(bkg_n_constituents),
    bins=80,
    density=True,
    alpha=0.5,
    label="Background"
)

plt.xlabel("Jet Constituent Multiplicity")
plt.ylabel("Normalised Density")
plt.title("Jet Constituent Multiplicity")
plt.legend()

plt.savefig(
    "outputs/plots/jet_constituent_multiplicity.png",
    dpi=300
)

plt.close()

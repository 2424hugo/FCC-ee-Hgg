# imports
import awkward as ak
import matplotlib.pyplot as plt

# loading in data
sig = ak.from_parquet("cache/signal_reco.parquet")
bkg = ak.from_parquet("cache/background_reco.parquet")

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
	plt.savefig(f"outputs/plots/Recon_distributions/{safe_name}.png", dpi=300)
	plt.close()

# reconstructed particle features
features = {
    "ReconstructedParticles/ReconstructedParticles.energy":
        "Particle Energy [GeV]",

    "ReconstructedParticles/ReconstructedParticles.mass":
        "Particle Mass [GeV]",

    "ReconstructedParticles/ReconstructedParticles.charge":
        "Particle Charge",

    "ReconstructedParticles/ReconstructedParticles.momentum.x":
        r"Particle $p_x$ [GeV]",

    "ReconstructedParticles/ReconstructedParticles.momentum.y":
        r"Particle $p_y$ [GeV]",

    "ReconstructedParticles/ReconstructedParticles.momentum.z":
        r"Particle $p_z$ [GeV]",
}

# loop over features
for feature, label in features.items():
    plotting_feature(feature, label)

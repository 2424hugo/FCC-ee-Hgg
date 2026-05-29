# imports
import awkward as ak
import matplotlib.pyplot as plt

# loading in data
sig = ak.from_parquet("cache/signal_missingEnPhotoNeut.parquet")
bkg = ak.from_parquet("cache/background_missingEnPhotoNeut.parquet")

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
	plt.savefig(f"outputs/plots/MET_Photon_NeutralHadron_distributions/{safe_name}.png", dpi=300)
	plt.close()

# features
features = {
    "EFlowNeutralHadron/EFlowNeutralHadron.energy":
        "Neutral Hadron Energy [GeV]",

    "EFlowPhoton/EFlowPhoton.energy":
        "Photon Energy [GeV]",

    "MissingET/MissingET.energy":
        "Missing Energy [GeV]",

    "MissingET/MissingET.momentum.x":
        r"Missing Energy $p_x$ [GeV]",

    "MissingET/MissingET.momentum.y":
        r"Missing Energy $p_y$ [GeV]",

    "MissingET/MissingET.momentum.z":
        r"Missing Energy $p_z$ [GeV]",
}

# loop over features
for feature, label in features.items():
    plotting_feature(feature, label)

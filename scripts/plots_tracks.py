# imports
import awkward as ak
import matplotlib.pyplot as plt

# loading in data
sig = ak.from_parquet("cache/signal_tracks.parquet")
bkg = ak.from_parquet("cache/background_tracks.parquet")

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
	plt.savefig(f"outputs/plots/track_distributions/{safe_name}.png", dpi=300)
	plt.close()

# track features
features = {
    "EFlowTrack_1/EFlowTrack_1.D0": r"Track $D_0$ [mm]",
    "EFlowTrack_1/EFlowTrack_1.Z0": r"Track $Z_0$ [mm]",
    "EFlowTrack_1/EFlowTrack_1.phi": r"Track $\phi$ [rad]",
    "EFlowTrack_1/EFlowTrack_1.omega": r"Track $\omega$",
    "EFlowTrack_1/EFlowTrack_1.tanLambda": r"Track $\tan\lambda$",
}

# loop over features
for feature, label in features.items():
    plotting_feature(feature, label)

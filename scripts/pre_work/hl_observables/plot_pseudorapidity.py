# Finding a plotting psudo rap for both signal and background
#
# Using this formular:
# eta = (1/2)*ln((p+p_z)/(p-p_z)), where p is p=sqrt(p_y^2+p_x^2+p_z^2)

# imports
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

# loading data
sig = ak.from_parquet("cache/signal_jet.parquet")
bkg = ak.from_parquet("cache/background_jet.parquet")

# function to find the momentum of the jet
def jet_momentum(data):
	p_x = data["Jet/Jet.momentum.x"]
	p_y = data["Jet/Jet.momentum.y"]
	p_z = data["Jet/Jet.momentum.z"]

	p = np.sqrt(p_x**2 + p_y**2 + p_z**2)

	return p

# finding and plotting jet momentums
sig_momentum = jet_momentum(sig)
bkg_momentum = jet_momentum(bkg)

plt.figure(figsize = (8, 6))
plt.hist(ak.flatten(sig_momentum), bins = 100, label = 'Signal', density=True, alpha = 0.5)
plt.hist(ak.flatten(bkg_momentum), bins = 100, label = 'Background', density=True, alpha = 0.5)

plt.xlabel("Jet momentum")
plt.ylabel("Normalized Density")
plt.title("Momentum per a jet")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/plots/hl_observables/jet_momentums.png", dpi=300)
plt.close()


# finding and plotting eta
def eta(data):
	p_x = data["Jet/Jet.momentum.x"]
	p_y = data["Jet/Jet.momentum.y"]
	p_z = data["Jet/Jet.momentum.z"]

	p = np.sqrt(p_x**2 + p_y**2 + p_z**2)

	eta = (1/2)*np.log((p + p_z) / (p - p_z))
	return eta

# finding and plotting jet momentums
sig_eta = eta(sig)
bkg_eta = eta(bkg)

plt.figure(figsize = (8, 6))
plt.hist(ak.flatten(sig_eta), bins = 100, label = 'Signal', density=True, alpha = 0.5)
plt.hist(ak.flatten(bkg_eta), bins = 100, label = 'Background', density=True, alpha = 0.5)

plt.xlabel("Jet pseidorapidity")
plt.ylabel("Normalized Density")
plt.title("Pseidorapidity per a jet")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/plots/hl_observables/jet_pseidorapidity.png", dpi=300)
plt.close()

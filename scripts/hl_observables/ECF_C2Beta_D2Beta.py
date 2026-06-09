import matplotlib.pyplot as plt
import awkward as ak
import numpy as np

sig_e2 = ak.from_parquet("cache/signal_e2_beta_05_1000.parquet")["e2_beta_0.5"]
bkg_e2 = ak.from_parquet("cache/bkg_e2_beta_05_1000.parquet")["e2_beta_0.5"]

sig_e3 = ak.from_parquet("cache/signal_e3_beta_05_1000.parquet")["e3_beta_0.5"]
bkg_e3 = ak.from_parquet("cache/bkg_e3_beta_05_1000.parquet")["e3_beta_0.5"]

def energy_correlation_C(e2, e3):
    mask = e2 > 0
    return (e3[mask] / e2[mask]**2)

def energy_correlation_D(e2, e3, beta=1):
    mask = e2 > 0
    return (e3[mask] / e2[mask]**3)

sig_C = energy_correlation_C(sig_e2, sig_e3)
bkg_C = energy_correlation_C(bkg_e2, bkg_e3)

sig_C_flat = ak.to_numpy(ak.flatten(sig_C,axis=None))
bkg_C_flat = ak.to_numpy(ak.flatten(bkg_C,axis=None))

plt.hist(sig_C_flat, density=True, label='signal', alpha=0.5, bins=100)
plt.hist(bkg_C_flat, density=True, label='background', alpha=0.5, bins=100)

plt.legend()
plt.title(r"Energy correlation function $C_2$ (beta=0.5)")
plt.ylabel("Density")

plt.savefig("outputs/plots/hl_observables/energy_correlation_C2Beta05.png")
plt.close()

sig_D = energy_correlation_D(sig_e2, sig_e3)
bkg_D = energy_correlation_D(bkg_e2, bkg_e3)

sig_D_flat = ak.to_numpy(ak.flatten(sig_D,axis=None))
bkg_D_flat = ak.to_numpy(ak.flatten(bkg_D,axis=None))

plt.hist(sig_D_flat, density=True, label='signal', alpha=0.5, bins=100)
plt.hist(bkg_D_flat, density=True, label='background', alpha=0.5, bins=100)

plt.legend()
plt.title(r"Energy correlation function $D_2$ (beta=0.5)")
plt.ylabel("Density")

plt.savefig("outputs/plots/hl_observables/energy_correlation_D2Beta05.png")
plt.close()

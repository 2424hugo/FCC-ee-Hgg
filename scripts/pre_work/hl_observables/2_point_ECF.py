import matplotlib.pyplot as plt
import awkward as ak
import numpy as np

sig_z = ak.from_parquet("cache/signal_energy_fraction_z_10000.parquet")['energy_fraction_z']
bkg_z = ak.from_parquet("cache/bkg_energy_fraction_z_10000.parquet")['energy_fraction_z']
 
sig_theta = ak.from_parquet("cache/signal_theta2_ij_10000.parquet")['theta2_ij']
bkg_theta = ak.from_parquet("cache/bkg_theta2_ij_10000.parquet")['theta2_ij']

sig_z = sig_z[:len(sig_theta)]
bkg_z = bkg_z[:len(bkg_theta)]

def ecf2(z, theta2, beta=1):
        # all unique constituent pairs: (zi, zj)
        z_pairs = ak.combinations(z, 2, axis=-1, fields=["zi", "zj"])
        zi = z_pairs["zi"]
        zj = z_pairs["zj"]

        assert ak.all(ak.num(zi, axis=-1) == ak.num(theta2, axis=-1))

        return ak.sum(zi * zj * theta2**beta, axis=-1)

sig = ecf2(sig_z, sig_theta, beta = 2)
bkg = ecf2(bkg_z, bkg_theta, beta = 2)

sig_flat = ak.to_numpy(ak.flatten(sig, axis=None))
bkg_flat = ak.to_numpy(ak.flatten(bkg, axis=None))

plt.hist(sig_flat, density=True, label='signal', alpha=0.5, bins=100)
plt.hist(bkg_flat, density=True, label='background', alpha=0.5, bins=100)
plt.legend()
plt.title("Two point energy function (beta = 2)")
plt.ylabel("Density")

plt.savefig("outputs/plots/energy_func/two_point_energy_distribution_beta2.png")

ak.to_parquet(ak.Array({"e2_beta_2": sig}),"cache/signal_e2_beta_2_1000.parquet", compression=None)
ak.to_parquet(ak.Array({"e2_beta_2": bkg}),"cache/bkg_e2_beta_2_1000.parquet", compression=None)

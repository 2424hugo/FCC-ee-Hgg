# imports
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

# loading in data
sig = ak.from_parquet("cache/signal_jetParticles.parquet")
bkg = ak.from_parquet("cache/background_jetParticles.parquet")


def find_energyFraction_z(data):
    # find where each jets particles begin and end
    begin = data["Jet/Jet.particles_begin"]
    end   = data["Jet/Jet.particles_end"]
    # energy for each particle
    energy = data["ReconstructedParticles/ReconstructedParticles.energy"]
   
    all_z = []
    for event in range(len(energy)):
        particles_z = []

        for jet in range(len(begin[event])):

            b = begin[event][jet]
            e = end[event][jet]

            jet_energy = energy[event][b:e]

            z = jet_energy / ak.sum(jet_energy)

            particles_z.append(z)
        
        all_z.append(particles_z)
    return ak.Array(all_z)

print(sig.fields)

sig_z = find_energyFraction_z(sig[:10000])
bkg_z = find_energyFraction_z(bkg[:10000])

plt.figure(figsize=(8,6))

plt.hist(
        ak.to_numpy(ak.flatten(sig_z, axis=None)),
        bins=100,
        density=True,
        alpha = 0.5,
        label="Signal"
)

plt.hist(
        ak.to_numpy(ak.flatten(bkg_z, axis=None)),
        bins=100,
        density=True,
        alpha = 0.5,
        label="Background"    
)

plt.xlabel(r"$z_i = E_i/E_J$")
plt.ylabel("Density")
plt.legend()
plt.savefig("outputs/plots/energy_fraction_z.png", dpi=300)
plt.close()

def find_theta2_ij_jet(px, py, pz, E):
    theta2 = []

    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            dot = (
                     E[i] * E[j]
                     -px[i] * px[j]
                     -py[i] * py[j]
                    -pz[i] * pz[j]
            )
            theta2.append(np.sqrt(2 * dot / (E[i] * E[j])))
    return ak.Array(theta2)

def find_theta2_all_jet(data):
    # find where each jets particles begin and end
    begin = data["Jet/Jet.particles_begin"]
    end   = data["Jet/Jet.particles_end"]
    # momentum components for each particle
    px = data["ReconstructedParticles/ReconstructedParticles.momentum.x"]
    py = data["ReconstructedParticles/ReconstructedParticles.momentum.y"] 
    pz = data["ReconstructedParticles/ReconstructedParticles.momentum.z"]
    E = data["ReconstructedParticles/ReconstructedParticles.energy"]
   
   
    all_theta2  = []
    for event in range(len(E)):
        particles_theta2 = []

        for jet in range(len(begin[event])):

            b = begin[event][jet]
            e = end[event][jet]

            jet_theta2 = find_theta2_ij_jet(
                    px[event][b:e],
                    py[event][b:e],
                    pz[event][b:e],
                    E[event][b:e],
            )

            particles_theta2.append(jet_theta2)
        all_theta2.append(particles_theta2)
    return ak.Array(all_theta2)

sig_theta2 = find_theta2_all_jet(sig[:1000])
bkg_theta2 = find_theta2_all_jet(bkg[:1000])

plt.hist(
        ak.to_numpy(ak.flatten(sig_theta2, axis=None)),
        bins=100,
        label='Signal',
        density=True,
        alpha=0.5)
plt.hist(
        ak.to_numpy(ak.flatten(bkg_theta2, axis=None)),
        bins=100,
        label='Background',
        density=True,
        alpha=0.5)


plt.xlabel(r"$\theta_{ij}^2$")
plt.ylabel("Density")
plt.savefig("outputs/plots/theta2_ij.png", dpi=300)

ak.to_parquet(
        ak.Array({"theta2_ij": sig_theta2}),
        "cache/signal_theta2_ij_10000.parquet",
        compression=None
)
ak.to_parquet(
        ak.Array({"theta2_ij": bkg_theta2}),
        "cache/bkg_theta2_ij_10000.parquet",
        compression=None
)



ak.to_parquet(
        ak.Array({"energy_fraction_z": sig_z }),
        "cache/signal_energy_fraction_z_10000.parquet",
        compression=None
)
ak.to_parquet(
        ak.Array({"energy_fraction_z": bkg_z }),
        "cache/bkg_energy_fraction_z_10000.parquet",
        compression=None
)

import matplotlib.pyplot as plt
import awkward as ak
import numpy as np

sig = ak.from_parquet("cache/signal_jetParticles.parquet")
bkg = ak.from_parquet("cache/background_jetParticles.parquet")

print(sig.fields)


def theta_between(px1, py1, pz1, E1, px2, py2, pz2, E2):
    dot = E1 * E2 - px1 * px2 - py1 * py2 - pz1 * pz2
    value = 2 * dot / (E1 * E2)
    return np.sqrt(np.maximum(0, value))

def ecf3_jet(px, py, pz, E, beta=0.2):
    z = E / ak.sum(E)

    triples = ak.combinations(
            ak.zip({"z": z, "px": px, "py": py, "pz": pz, "E": E}),
            3,
            axis=0,
            fields=["a", "b", "c"]
    )

    a = triples["a"]
    b = triples["b"]
    c = triples["c"]

    theta_ab = theta_between(a.px, a.py, a.pz, a.E, b.px, b.py, b.pz, b.E)
    theta_ac = theta_between(a.px, a.py, a.pz, a.E, c.px, c.py, c.pz, c.E)
    theta_bc = theta_between(b.px, b.py, b.pz, b.E, c.px, c.py, c.pz, c.E)

    return ak.sum(
            a.z * b.z * c.z *
            theta_ab**beta *
            theta_ac**beta *
            theta_bc**beta)

def ecf3_all_jets(data, beta=1):
    begin = data["Jet/Jet.particles_begin"]
    end = data["Jet/Jet.particles_end"]

    px = data["ReconstructedParticles/ReconstructedParticles.momentum.x"]
    py = data["ReconstructedParticles/ReconstructedParticles.momentum.y"]
    pz = data["ReconstructedParticles/ReconstructedParticles.momentum.z"]
    E  = data["ReconstructedParticles/ReconstructedParticles.energy"]

    all_e3 = []

    for event in range(len(E)):
        event_e3 = []

        for jet in range(len(begin[event])):
            b = begin[event][jet]
            e = end[event][jet]

            value = ecf3_jet(px[event][b:e],py[event][b:e],pz[event][b:e],E[event][b:e],beta=beta)

            event_e3.append(value)

        all_e3.append(event_e3)

    return ak.Array(all_e3)

sig_e3 = ecf3_all_jets(sig[:1000], beta=0.5)
bkg_e3 = ecf3_all_jets(bkg[:1000], beta=0.5)

ak.to_parquet(ak.Array({"e3_beta_0.5": sig_e3}),"cache/signal_e3_beta_05_1000.parquet", compression=None)

ak.to_parquet(ak.Array({"e3_beta_0.5": bkg_e3}),"cache/bkg_e3_beta_05_1000.parquet", compression=None)

sig_flat = ak.to_numpy(ak.flatten(sig_e3, axis=None))
bkg_flat = ak.to_numpy(ak.flatten(bkg_e3, axis=None))

plt.hist(sig_flat, bins=100, density=True, label="signal", alpha=0.5)
plt.hist(bkg_flat, bins=100, density=True, label="background", alpha=0.5)
plt.legend()
plt.xlabel(r"$e_3^{(\beta=1)}$")
plt.ylabel("Density")
plt.savefig("outputs/plots/e3_beta_1_distribution.png")

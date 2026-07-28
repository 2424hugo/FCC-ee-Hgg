import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import gc

BETA = 0.1

sig = ak.from_parquet(
        "cache/mass_cut/signal_leading_two_jet_constituents.parquet")

bkg = ak.from_parquet(
        "cache/mass_cut/bkg_leading_two_jet_constituents.parquet")


def theta_between(px1, py1, pz1, E1, px2, py2, pz2, E2):
    """
    Angular distance used for the ee energy correlation function:

        theta_ij = sqrt(2 p_i.p_j / (E_i E_j)

    For massless particles this becomes sqrt(2(1-cos(theta))).
    """

    denominator = E1*E2

    dot = E1*E2 - px1 * px2 - py1 * py2 - pz1 * pz2

    value = ak.where(
        denominator > 0,
        2.0 * dot / denominator,
        0.0,
    )

    return np.maximum(0.0, value)

def ecf3_jet(px, py, pz, energy, beta):
    """
    Calculate e3^(beta) for one jet
    """

    # e3 requires at least three particles
    if len(energy) < 3:
        return 0.0

    jet_energy = ak.sum(energy)

    if jet_energy <= 0:
        return 0.0

    z = energy / jet_energy

    particles = ak.zip({
        "z": z,
        "px": px,
        "py": py,
        "pz": pz,
        "energy": energy,})

    triples = ak.combinations(
            particles,
            3,
            axis=0,
            fields=["a", "b", "c"],)

    a = triples["a"]
    b = triples["b"]
    c = triples["c"]

    theta_ab = theta_between(
        a.px, a.py, a.pz, a.energy,
        b.px, b.py, b.pz, b.energy,
    )

    theta_ac = theta_between(
        a.px, a.py, a.pz, a.energy,
        c.px, c.py, c.pz, c.energy,
    )

    theta_bc = theta_between(
        b.px, b.py, b.pz, b.energy,
        c.px, c.py, c.pz, c.energy,
    )

    angular_term = (
        theta_ab
        * theta_ac
        * theta_bc
    ) ** (beta / 2.0)

    terms = (
        a.z
        * b.z
        * c.z
        * angular_term
    )

    return float(ak.sum(terms))

def ecf3_all_jets(data, beta=2.0):
    """
    Return shape:

        events × 2 jets
    """

    all_e3 = []

    for event in range(len(data)):
        if event % 10_000 == 0:
            print(f"Processing event {event:,} / {len(data):,}")

        event_e3 = []

        for jet in range(2):
            value = ecf3_jet(
                data["px"][event][jet],
                data["py"][event][jet],
                data["pz"][event][jet],
                data["energy"][event][jet],
                beta=beta,
            )

            event_e3.append(value)

        all_e3.append(event_e3)

    return ak.Array(all_e3)

# Test on 1,000 events first
sig_e3 = ecf3_all_jets(sig[:10000], beta=BETA)
bkg_e3 = ecf3_all_jets(bkg[:10000], beta=BETA)

ak.to_parquet(
    ak.Array({"e3_beta_0p1": sig_e3}),
    "cache/mass_cut/hl_observables/signal_e3_beta_0p1.parquet",
    compression=None,
)

ak.to_parquet(
    ak.Array({"e3_beta_0p1": bkg_e3}),
    "cache/mass_cut/hl_observables/bkg_e3_beta_0p1.parquet",
    compression=None,
)

"""
print("Signal ECF3 type:", ak.type(sig_e3))
print("Background ECF3 type:", ak.type(bkg_e3))

print("Signal first events:", sig_e3[:5])
print("Background first events:", bkg_e3[:5])

# Combine leading and subleading jets into one distribution
sig_flat = ak.to_numpy(ak.flatten(sig_e3, axis=None))
bkg_flat = ak.to_numpy(ak.flatten(bkg_e3, axis=None))

# Remove non-finite values before plotting
sig_flat = sig_flat[np.isfinite(sig_flat)]
bkg_flat = bkg_flat[np.isfinite(bkg_flat)]

plt.figure()

plt.hist(
    sig_flat,
    bins=100,
    density=True,
    histtype="step",
    label="Signal",
)

plt.hist(
    bkg_flat,
    bins=100,
    density=True,
    histtype="step",
    label="Background",
)

plt.xlabel(r"$e_3^{(\beta=0.02)}$")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/plots/2_jet_selection/hl_observables/e3_beta_0.02_distribution.png",
    dpi=300,
)

plt.close()


labels = np.concatenate([
    np.ones(len(sig_flat)),
    np.zeros(len(bkg_flat))
])

scores = np.concatenate([
    sig_flat,
    bkg_flat
])

auc = roc_auc_score(labels, scores)

print(f"β = {BETA}: AUC = {auc:.4f}")"""



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
    Returns theta_ij^2:

        theta_ij^2 = 2 p_i.p_j / (E_i E_j)
    """

    denominator = E1 * E2

    dot = (
        E1 * E2
        - px1 * px2
        - py1 * py2
        - pz1 * pz2
    )

    value = ak.where(
        denominator > 0,
        2.0 * dot / denominator,
        0.0,
    )

    return np.maximum(0.0, value)

def ecf2_jet(px, py, pz, energy, beta):
    """
    Calculate e2^(beta) for one jet
    """

    # e2 requires at least two particles
    if len(energy) < 2:
        return 0.0

    jet_energy = ak.sum(energy)

    if jet_energy <= 0:
        return 0.0

    # Energy fractions
    z = energy / jet_energy

    particles = ak.zip({
        "z": z,
        "px": px,
        "py": py,
        "pz": pz,
        "energy": energy,
    })

    # All unique i < j particle pairs
    pairs = ak.combinations(
        particles,
        2,
        axis=0,
        fields=["a", "b"],
    )

    a = pairs["a"]
    b = pairs["b"]

    # This is theta_ij^2
    theta2_ab = theta_between(
        a.px, a.py, a.pz, a.energy,
        b.px, b.py, b.pz, b.energy,
    )

    # theta^beta = (theta^2)^(beta/2)
    angular_term = theta2_ab ** (beta / 2.0)

    terms = (
        a.z
        * b.z
        * angular_term
    )

    return float(ak.sum(terms))

def ecf2_all_jets(data, beta=2.0):
    """
    Return shape:

        events × 2 jets
    """

    all_e2 = []

    for event in range(len(data)):
        if event % 10_000 == 0:
            print(f"Processing event {event:,} / {len(data):,}")

        event_e2 = []

        for jet in range(2):
            value = ecf2_jet(
                data["px"][event][jet],
                data["py"][event][jet],
                data["pz"][event][jet],
                data["energy"][event][jet],
                beta=beta,
            )

            event_e2.append(value)

        all_e2.append(event_e2)

    return ak.Array(all_e2)
"""
sig_e2 = ecf2_all_jets(sig[:1000], beta=BETA)
bkg_e2 = ecf2_all_jets(bkg[:1000], beta=BETA)

print("Signal ECF2 type:", ak.type(sig_e2))
print("Background ECF2 type:", ak.type(bkg_e2))

print("Signal first events:", sig_e2[:5])
print("Background first events:", bkg_e2[:5])

sig_flat = ak.to_numpy(ak.flatten(sig_e2, axis=None))
bkg_flat = ak.to_numpy(ak.flatten(bkg_e2, axis=None))

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

plt.xlabel(r"$e_2^{(\beta=0.5)}$")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/plots/2_jet_selection/hl_observables/e2_beta_05_distribution.png",
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

print(f"β = {BETA}: AUC = {auc:.4f}")
print(f"Separation AUC = {max(auc, 1 - auc):.4f}")

def calculate_auc(sig_values, bkg_values):
    labels = np.concatenate([
        np.ones(len(sig_values)),
        np.zeros(len(bkg_values)),
    ])

    scores = np.concatenate([
        sig_values,
        bkg_values,
    ])

    auc = roc_auc_score(labels, scores)

    return max(auc, 1 - auc)

betas = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

for beta in betas:

    sig_e2 = ecf2_all_jets(sig[:10000], beta=beta)
    bkg_e2 = ecf2_all_jets(bkg[:10000], beta=beta)

    sig_lead = ak.to_numpy(sig_e2[:, 0])
    sig_sub  = ak.to_numpy(sig_e2[:, 1])

    bkg_lead = ak.to_numpy(bkg_e2[:, 0])
    bkg_sub  = ak.to_numpy(bkg_e2[:, 1])

    auc_lead = calculate_auc(sig_lead, bkg_lead)
    auc_sub  = calculate_auc(sig_sub, bkg_sub)

    print(
        f"β = {beta:<4} | "
        f"lead AUC = {auc_lead:.4f} | "
        f"sub AUC = {auc_sub:.4f}"
    )"""

sig_e2 = ecf2_all_jets(sig[:10000], beta=BETA)
bkg_e2 = ecf2_all_jets(bkg[:10000], beta=BETA)

sig_energy_func = ak.Array({
    "e2_beta_0p1": sig_e2})
bkg_energy_func = ak.Array({
    "e2_beta_0p1": bkg_e2})

ak.to_parquet(
    sig_energy_func,
    "cache/mass_cut/hl_observables/signal_ecf_beta_0p1.parquet",
    compression=None,
)

ak.to_parquet(
    bkg_energy_func,
    "cache/mass_cut/hl_observables/bkg_ecf_beta_0p1.parquet",
    compression=None,
)

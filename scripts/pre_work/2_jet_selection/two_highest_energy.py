import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

# Cut level
mass_cut = 120

# Loading data
sig_data = ak.from_parquet("cache/signal_jet.parquet")
bkg_data = ak.from_parquet("cache/background_jet.parquet")

sig_mass_record = ak.from_parquet("cache/signal_event_mass.parquet")
bkg_mass_record = ak.from_parquet("cache/background_event_mass.parquet")
sig_mass = sig_mass_record["event_invariant_mass"]
bkg_mass = bkg_mass_record["event_invariant_mass"]

# Create masks
sig_mask = sig_mass > mass_cut
bkg_mask = bkg_mass > mass_cut

# Appling mask
sig_mass_cut = sig_mass[sig_mask]
bkg_mass_cut = bkg_mass[bkg_mask]

sig_cut = sig_data[sig_mask]
bkg_cut = bkg_data[bkg_mask]


def select_two_highest_energy_jets(data):
    jet_energy = data["Jet/Jet.energy"]

    # Sort jet indices from highest to lowest energy
    order = ak.argsort(jet_energy, axis=1, ascending=False)

    # Select the first two indices in every event
    leading_two_indices = order[:, :2]

    # Retain events that contain at least two jets
    has_two_jets = ak.num(jet_energy, axis=1)>=2

    data = data[has_two_jets]
    leading_two_indices = leading_two_indices[has_two_jets]

    return data, leading_two_indices

sig_two_jet, sig_jet_indices = select_two_highest_energy_jets(sig_cut)
bkg_two_jet, bkg_jet_indices = select_two_highest_energy_jets(bkg_cut)

sig_selected_energy = sig_two_jet["Jet/Jet.energy"][sig_jet_indices]
bkg_selected_energy = bkg_two_jet["Jet/Jet.energy"][bkg_jet_indices]

sig_selected_mass = sig_two_jet["Jet/Jet.mass"][sig_jet_indices]
bkg_selected_mass = bkg_two_jet["Jet/Jet.mass"][bkg_jet_indices]
"""
plt.figure(figsize=(8, 6))

plt.hist(
        ak.to_numpy(sig_selected_mass[:, 0]),
        bins=100,
        density=True,
        histtype="step",
        linewidth=2,
        label="Signal leading jet")
plt.hist(
        ak.to_numpy(sig_selected_mass[:, 1]),
        bins=100,
        density=True,
        histtype="step",
        linewidth=2,
        label="Signal subleading jet")

plt.hist(
        ak.to_numpy(bkg_selected_mass[:, 0]),
        bins=100,
        density=True,
        histtype="step",
        linewidth=2,
        label="Background leading jet")
plt.hist(
        ak.to_numpy(bkg_selected_mass[:, 1]),
        bins=100,
        density=True,
        histtype="step",
        linewidth=2,
        label="Background subleading jet")

plt.xlabel("Selected jet masses [Gev]")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(
        "outputs/plots/2_jet_selection/subleading_mass_jets_.png",
        dpi=300)
plt.close()

sig_two_jet_eff = len(sig_two_jet) / len(sig_cut)
bkg_two_jet_eff = len(bkg_two_jet) / len(bkg_cut)

print(f"Signal events after mass cut: {len(sig_cut)}")
print(f"Signal events with >= 2 jets:  {len(sig_two_jet)}")
print(f"Signal >=2 jet efficiency:     {sig_two_jet_eff:.4f}")

print()

print(f"Background events after mass cut: {len(bkg_cut)}")
print(f"Background events with >= 2 jets:  {len(bkg_two_jet)}")
print(f"Background >=2 jet efficiency:     {bkg_two_jet_eff:.4f}")

H_sig, xedges, yedges = np.histogram2d(
    ak.to_numpy(sig_selected_mass[:,0]),
    ak.to_numpy(sig_selected_mass[:,1]),
    bins=75,
    density=True
)

H_bkg, _, _ = np.histogram2d(
    ak.to_numpy(bkg_selected_mass[:,0]),
    ak.to_numpy(bkg_selected_mass[:,1]),
    bins=[xedges, yedges],
    density=True
)

v = np.max(np.abs(H_sig - H_bkg))

plt.imshow(
    (H_sig - H_bkg).T,
    origin="lower",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    aspect="auto",
    cmap="RdBu_r",
    vmin=-v,
    vmax=v,
)

plt.colorbar(label="Signal density − Background density")
plt.xlabel("Leading jet mass [GeV]")
plt.ylabel("Subleading jet mass [GeV]")

plt.savefig("outputs/plots/2_jet_selection/mass_2d_comparison_weighted.png", dpi=300)"""

from sklearn.metrics import roc_curve, roc_auc_score

# Leading jet masses
sig_subleading_mass = ak.to_numpy(sig_selected_mass[:, 1])
bkg_subleading_mass = ak.to_numpy(bkg_selected_mass[:, 1])

# Labels: signal = 1, background = 0
y_true = np.concatenate([
    np.ones(len(sig_subleading_mass)),
    np.zeros(len(bkg_subleading_mass))
])

# Classifier score: leading jet mass
y_score = np.concatenate([
    sig_subleading_mass,
    bkg_subleading_mass
])

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
subleading_mass_auc = roc_auc_score(y_true, y_score)

print(f"Subleading jet mass AUC: {subleading_mass_auc:.4f}")

plt.figure(figsize=(7, 6))
plt.plot(
    fpr,
    tpr,
    label=f"subleading jet mass, AUC = {subleading_mass_auc:.3f}"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)
plt.xlabel("Background efficiency")
plt.ylabel("Signal efficiency")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(
    "outputs/plots/2_jet_selection/subleading_jet_mass_ROC.png",
    dpi=300
)
plt.close()

youden_j = tpr - fpr
best_index = np.argmax(youden_j)

best_threshold = thresholds[best_index]
best_sig_eff = tpr[best_index]
best_bkg_eff = fpr[best_index]

print(f"Best lower mass cut: m_lead > {best_threshold:.3f} GeV")
print(f"Signal efficiency:   {best_sig_eff:.4f}")
print(f"Background efficiency: {best_bkg_eff:.4f}")

print("Signal leading:",
      np.mean(sig_selected_mass[:,0]),
      np.std(sig_selected_mass[:,0]))

print("Signal subleading:",
      np.mean(sig_selected_mass[:,1]),
      np.std(sig_selected_mass[:,1]))

print("Background leading:",
      np.mean(bkg_selected_mass[:,0]),
      np.std(bkg_selected_mass[:,0]))

print("Background subleading:",
      np.mean(bkg_selected_mass[:,1]),
      np.std(bkg_selected_mass[:,1]))

print(np.all(
    ak.to_numpy(sig_selected_energy[:,0]) >=
    ak.to_numpy(sig_selected_energy[:,1])
))

print(np.all(
    ak.to_numpy(bkg_selected_energy[:,0]) >=
    ak.to_numpy(bkg_selected_energy[:,1])
))

# Sum of leading and subleading jet masses
sig_mass_sum = ak.to_numpy(
    sig_selected_mass[:, 0] * sig_selected_mass[:, 1]
)

bkg_mass_sum = ak.to_numpy(
    bkg_selected_mass[:, 0] * bkg_selected_mass[:, 1]
)

# Labels: signal = 1, background = 0
y_true = np.concatenate([
    np.ones(len(sig_mass_sum)),
    np.zeros(len(bkg_mass_sum))
])

# Classifier score: sum of jet masses
y_score = np.concatenate([
    sig_mass_sum,
    bkg_mass_sum
])

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_score)
mass_sum_auc = roc_auc_score(y_true, y_score)

print(f"Leading + subleading jet mass AUC: {mass_sum_auc:.4f}")
"""
plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Leading + subleading jet masses, AUC = {mass_sum_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("Background efficiency")
plt.ylabel("Signal efficiency")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/plots/2_jet_selection/leading_subleading_mass_sum_ROC.png",
    dpi=300
)

plt.close()
"""
# Best cut (Youden statistic)
youden_j = tpr - fpr
best_index = np.argmax(youden_j)

best_threshold = thresholds[best_index]
best_sig_eff = tpr[best_index]
best_bkg_eff = fpr[best_index]

print(f"Best lower cut: m₁ + m₂ > {best_threshold:.3f} GeV")
print(f"Signal efficiency:     {best_sig_eff:.4f}")
print(f"Background efficiency: {best_bkg_eff:.4f}")
print(f"Background rejection:  {1-best_bkg_eff:.4f}")

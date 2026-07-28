import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

# Build signal feature matrix:
# column 0 = leading jet mass
# column 1 = subleading jet mass
X_sig = np.column_stack([
    ak.to_numpy(sig_selected_mass[:, 0]),
    ak.to_numpy(sig_selected_mass[:, 1])
])

# Build background feature matrix
X_bkg = np.column_stack([
    ak.to_numpy(bkg_selected_mass[:, 0]),
    ak.to_numpy(bkg_selected_mass[:, 1])
])

# Remove any rows containing NaN or infinity
X_sig = X_sig[np.all(np.isfinite(X_sig), axis=1)]
X_bkg = X_bkg[np.all(np.isfinite(X_bkg), axis=1)]

# Combine signal and background
X = np.vstack([X_sig, X_bkg])

# Labels: signal = 1, background = 0
y = np.concatenate([
    np.ones(len(X_sig)),
    np.zeros(len(X_bkg))
])

# Split into training and testing samples
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Standardisation + logistic regression
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        random_state=42
    )
)

# Train
model.fit(X_train, y_train)

# Signal probabilities for the test sample
y_score = model.predict_proba(X_test)[:, 1]

# ROC and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_score)
combined_mass_auc = roc_auc_score(y_test, y_score)

print(
    f"Leading + subleading jet mass AUC: "
    f"{combined_mass_auc:.4f}"
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=(
        f"Leading + subleading masses, "
        f"AUC = {combined_mass_auc:.3f}"
    )
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
    "outputs/plots/2_jet_selection/"
    "combined_jet_mass_logistic_ROC.png",
    dpi=300
)

plt.close()

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=1000,
    random_state=42
)

tree.fit(X_train, y_train)

tree_score = tree.predict_proba(X_test)[:, 1]

tree_auc = roc_auc_score(
    y_test,
    tree_score
)

print(
    f"Decision tree AUC using both masses: "
    f"{tree_auc:.4f}"
)

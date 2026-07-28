# Will create a basic binary classifier that takes in the inital jet variables.
#
# Variables are: Multiplicity, energy, mass, z-momentum and transversal momentum.

# imports
import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

# loading in data
sig = ak.from_parquet("cache/signal_jet.parquet")
bkg = ak.from_parquet("cache/background_jet.parquet")

sig_transverseMomentum = ak.from_parquet("cache/signal_jet_transverseMomentum.parquet")['jet_transverseMomentum']
bkg_transverseMomentum = ak.from_parquet("cache/background_jet_transverseMomentum.parquet")['jet_transverseMomentum']

sig_jetmulti = (
	sig["Jet/Jet.particles_end"]
	- sig["Jet/Jet.particles_begin"]
)


bkg_jetmulti = (
	bkg["Jet/Jet.particles_end"]
	- bkg["Jet/Jet.particles_begin"]
)


X_sig = np.column_stack([
	ak.to_numpy(ak.flatten(sig_jetmulti)),
	ak.to_numpy(ak.flatten(sig_transverseMomentum)),
	ak.to_numpy(ak.flatten(sig["Jet/Jet.energy"])),
	ak.to_numpy(ak.flatten(sig["Jet/Jet.momentum.z"]))
])
y_sig = np.ones(len(X_sig))


X_bkg = np.column_stack([
	ak.to_numpy(ak.flatten(sig_jetmulti)),
	ak.to_numpy(ak.flatten(sig_transverseMomentum)),
	ak.to_numpy(ak.flatten(sig["Jet/Jet.energy"])),
	ak.to_numpy(ak.flatten(sig["Jet/Jet.momentum.z"]))
])
y_bkg = np.zeros(len(X_sig))


X = np.vstack([X_sig, X_bkg])
y = np.concatenate([y_sig, y_bkg])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(
    max_iter=1000,
    verbose=1
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_score))
print(classification_report(y_test, y_pred))

sig_scores = y_score[y_test == 1]
bkg_scores = y_score[y_test == 0]

plt.figure(figsize=(8,6))

plt.hist(
    sig_scores,
    bins=50,
    density=True,
    histtype="step",
    label="Signal"
)

plt.hist(
    bkg_scores,
    bins=50,
    density=True,
    histtype="step",
    label="Background"
)

plt.xlabel("Classifier Score")
plt.ylabel("Density")
plt.legend()

plt.savefig("outputs/plots/classifier_scores_per_jet.png", dpi=300)

plt.close()

print("Done")



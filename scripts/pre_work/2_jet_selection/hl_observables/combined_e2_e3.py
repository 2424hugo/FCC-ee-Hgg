import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)


bkg = ak.from_parquet("cache/mass_cut/hl_observables/bkg_ecf_combined_beta_0p1.parquet")
sig = ak.from_parquet("cache/mass_cut/hl_observables/signal_ecf_combined_beta_0p1.parquet")

print(ak.fields(bkg))
print(ak.fields(sig))

sig_e2 = sig['e2_beta_0p1']
sig_e3 = sig['e3_beta_0p1']
bkg_e2 = bkg['e2_beta_0p1']
bkg_e3 = bkg['e3_beta_0p1']

sig_e2 = ak.to_numpy(sig_e2)
bkg_e2 = ak.to_numpy(bkg_e2)
sig_e3 = ak.to_numpy(sig_e3)
bkg_e3 = ak.to_numpy(bkg_e3)

epsilon = 1e-12

sig_c2 = sig_e3 / (sig_e2**2 + epsilon)
bkg_c2 = bkg_e3 / (bkg_e2**2 + epsilon)

sig_d2 = sig_e3 / (sig_e2**3 + epsilon)
bkg_d2 = bkg_e3 / (bkg_e2**3 + epsilon)

X_sig = np.column_stack([
    sig_e2[:, 0],   # leading-jet e2
    sig_e2[:, 1],   # subleading-jet e2
    sig_e3[:, 0],   # leading-jet e3
    sig_e3[:, 1],   # subleading-jet e3
    sig_c2[:, 0],
    sig_c2[:, 1],
    sig_d2[:, 0],
    sig_d2[:, 1],
])

X_bkg = np.column_stack([
    bkg_e2[:, 0],
    bkg_e2[:, 1],
    bkg_e3[:, 0],
    bkg_e3[:, 1],
    bkg_c2[:, 0],
    bkg_c2[:, 1],
    bkg_d2[:, 0],
    bkg_d2[:, 1],
])

# Labels: signal = 1, background = 0
y_sig = np.ones(len(X_sig), dtype=int)
y_bkg = np.zeros(len(X_bkg), dtype=int)

X = np.concatenate([X_sig, X_bkg], axis=0)
y = np.concatenate([y_sig, y_bkg], axis=0)


feature_names = [
    "leading e2",
    "subleading e2",
    "leading e3",
    "subleading e3",
    "leading c2",
    "subleading c2",
    "leading d2",
    "subleading d2",
]

print("X shape:", X.shape)
print("y shape:", y.shape)

# --------------------------------------------------
# 3. Clean and transform features
# --------------------------------------------------

valid = np.all(np.isfinite(X), axis=1)
print("Invalid events removed:", np.sum(~valid))

X = X[valid]
y = y[valid]

# Protect against log10(0)
epsilon = 1e-12
X = np.log10(np.clip(X, epsilon, None))

print("Values <= epsilon:", np.sum(X == np.log10(epsilon)))

# --------------------------------------------------
# 4. Split the data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# 5. Train logistic regression
# --------------------------------------------------

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Evaluate classifier
# --------------------------------------------------

signal_score = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, signal_score)
fpr, tpr, thresholds = roc_curve(y_test, signal_score)

print(f"Test AUC: {auc:.4f}")

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=fr"Logistic regression, AUC = {auc:.4f}",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Random classifier",
)

plt.xlabel("Background efficiency")
plt.ylabel("Signal efficiency")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/plots/2_jet_selection/hl_observables/combined_e2_e3/ecf_logistic_regression_ROC.png",
    dpi=300,
)
plt.show()

youden_j = tpr - fpr
best_index = np.argmax(youden_j)

best_threshold = thresholds[best_index]
best_signal_efficiency = tpr[best_index]
best_background_efficiency = fpr[best_index]

print(f"Best threshold: {best_threshold:.6f}")
print(f"Signal efficiency: {best_signal_efficiency:.4f}")
print(f"Background efficiency: {best_background_efficiency:.4f}")
print(f"Background rejection: {1 - best_background_efficiency:.4f}")

logistic_model = model.named_steps["logisticregression"]

for name, coefficient in zip(
    feature_names,
    logistic_model.coef_[0],
):
    print(f"{name:20s}: {coefficient:+.4f}")

print(f"Intercept: {logistic_model.intercept_[0]:+.4f}")

logistic_model = model
logistic_scores = logistic_model.predict_proba(X_test)[:, 1]

logistic_auc = roc_auc_score(y_test, logistic_scores)
log_fpr, log_tpr, log_thresholds = roc_curve(
    y_test,
    logistic_scores,
)

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)

xgb_model.fit(X_train, y_train)

xgb_scores = xgb_model.predict_proba(X_test)[:, 1]

xgb_auc = roc_auc_score(y_test, xgb_scores)
xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(
    y_test,
    xgb_scores,
)

print(f"Logistic-regression AUC: {logistic_auc:.4f}")
print(f"XGBoost AUC:            {xgb_auc:.4f}")

plt.figure(figsize=(7, 6))

plt.plot(
    log_fpr,
    log_tpr,
    label=f"Logistic regression, AUC = {logistic_auc:.4f}",
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    label=f"XGBoost, AUC = {xgb_auc:.4f}",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Random classifier",
)

plt.xlabel("Background efficiency")
plt.ylabel("Signal efficiency")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/plots/2_jet_selection/hl_observables/"
    "combined_e2_e3/ecf_model_comparison_ROC.png",
    dpi=300,
)

plt.close()

# --------------------------------------------------
# 8. Compare performance at fixed background rates
# --------------------------------------------------

def signal_efficiency_at_background(fpr, tpr, target):
    return np.interp(target, fpr, tpr)


print("\nSignal efficiency at fixed background efficiency")

for target in [1e-1, 1e-2, 1e-3]:
    logistic_eff = signal_efficiency_at_background(
        log_fpr, log_tpr, target
    )
    xgb_eff = signal_efficiency_at_background(
        xgb_fpr, xgb_tpr, target
    )

    print(
        f"epsilon_B = {target:.0e}: "
        f"logistic epsilon_S = {logistic_eff:.4f}, "
        f"XGBoost epsilon_S = {xgb_eff:.4f}"
    )

def print_youden_point(name, fpr, tpr, thresholds):
    index = np.argmax(tpr - fpr)

    print(f"\n{name} Youden-J operating point")
    print(f"Threshold:            {thresholds[index]:.6f}")
    print(f"Signal efficiency:    {tpr[index]:.4f}")
    print(f"Background efficiency:{fpr[index]:.4f}")
    print(f"Background rejection: {1 - fpr[index]:.4f}")


print_youden_point(
    "Logistic regression",
    log_fpr,
    log_tpr,
    log_thresholds,
)

print_youden_point(
    "XGBoost",
    xgb_fpr,
    xgb_tpr,
    xgb_thresholds,
)

plt.figure(figsize=(7, 6))

plt.plot(
    log_fpr,
    log_tpr,
    label=f"Logistic regression, AUC = {logistic_auc:.4f}",
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    label=f"XGBoost, AUC = {xgb_auc:.4f}",
)

plt.xscale("log")
plt.xlim(1e-4, 1)
plt.ylim(0, 1)

plt.xlabel("Background efficiency")
plt.ylabel("Signal efficiency")
plt.grid(alpha=0.3, which="both")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/plots/2_jet_selection/hl_observables/"
    "combined_e2_e3/ecf_model_comparison_log_ROC.png",
    dpi=300,
)

plt.close()

n_test_background = np.sum(y_test == 0)

print(f"\nBackground test events: {n_test_background}")
print(
    "Minimum nonzero measurable background efficiency: "
    f"{1 / n_test_background:.3e}"
)
print(
    "Approximate 95% upper limit if zero events pass: "
    f"{3 / n_test_background:.3e}"
)

from sklearn.utils.class_weight import compute_sample_weight

train_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

xgb_model.fit(
    X_train,
    y_train,
    sample_weight=train_weights,
)

order = np.argsort(xgb_model.feature_importances_)[::-1]

print("\nXGBoost feature importances")

for index in order:
    print(
        f"{feature_names[index]:20s}: "
        f"{xgb_model.feature_importances_[index]:.4f}"
    )

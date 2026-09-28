import numpy as np
import pandas as pd


NN_FILE = (
    "outputs/ml/nn_final_wide_all_data_test/"
    "test_predictions.csv"
)

BDT_FILE = (
    "outputs/ml/bdt_22_variables_test/"
    "test_predictions.csv"
)

TARGET_SIGNAL_EFFICIENCIES = [
    0.30,
    0.50,
    0.70,
    0.80,
    0.90,
]


def calculate_operating_points(path, model_name):

    df = pd.read_csv(
        path,
        usecols=["label", "score"]
    )

    labels = df["label"].to_numpy()
    scores = df["score"].to_numpy()

    n_signal = np.sum(labels == 1)
    n_background = np.sum(labels == 0)

    # Sort thresholds from highest score to lowest.
    order = np.argsort(-scores)

    labels_sorted = labels[order]
    scores_sorted = scores[order]

    tp = np.cumsum(labels_sorted == 1)
    fp = np.cumsum(labels_sorted == 0)

    signal_eff = tp / n_signal
    background_eff = fp / n_background

    results = []

    for target in TARGET_SIGNAL_EFFICIENCIES:

        # Find point closest to requested signal efficiency.
        index = np.argmin(
            np.abs(signal_eff - target)
        )

        eps_sig = signal_eff[index]
        eps_bkg = background_eff[index]
        threshold = scores_sorted[index]

        rejection = (
            1.0 / eps_bkg
            if eps_bkg > 0
            else np.inf
        )

        results.append({
            "model": model_name,
            "target_signal_efficiency": target,
            "signal_efficiency": eps_sig,
            "threshold": threshold,
            "background_efficiency": eps_bkg,
            "background_rejection": rejection,
        })

    return pd.DataFrame(results)


nn = calculate_operating_points(
    NN_FILE,
    "NN"
)

bdt = calculate_operating_points(
    BDT_FILE,
    "BDT"
)

combined = pd.concat(
    [nn, bdt],
    ignore_index=True
)

combined = combined.sort_values(
    [
        "target_signal_efficiency",
        "model"
    ]
)

print()
print(combined.to_string(
    index=False,
    formatters={
        "target_signal_efficiency": "{:.2f}".format,
        "signal_efficiency": "{:.6f}".format,
        "threshold": "{:.6f}".format,
        "background_efficiency": "{:.6e}".format,
        "background_rejection": "{:.1f}".format,
    }
))

output = (
    "outputs/ml/"
    "final_classifier_operating_points.csv"
)

combined.to_csv(
    output,
    index=False
)

print()
print(f"Saved to: {output}")
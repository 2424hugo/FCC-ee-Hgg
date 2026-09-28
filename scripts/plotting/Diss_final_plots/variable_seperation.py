from pathlib import Path
import csv

import awkward as ak
import numpy as np

from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp


# ============================================================
# Configuration
# ============================================================

BASE = Path("cache/analysis_dataset")

OUTPUT_DIR = Path(
    "outputs/plots/diss_plots/discrimination"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "observable_discrimination.csv"
)

SPLITS = [
    "train",
    "validation",
    "test",
]

# Set to None to use every shard.
# Set to 2 for quick testing.
MAX_FILES_PER_SPLIT = None

# Number of bins used for overlap coefficient
OVERLAP_BINS = 200


# ============================================================
# Variables
# ============================================================

EVENT_FIELDS = [
    "event_invariant_mass",
    "n_jets_original",
]

JET_FIELDS = [
    "jet_energy",
    "jet_mass",
    "constituent_multiplicity",
    "e2_beta_0p2",
    "e3_beta_0p2",
    "jet_pt",
    "jet_p",
    "c2_beta_0p2",
    "d2_beta_0p2",
    "jet_theta",
]


# ============================================================
# Variable definitions
#
# key:
#   unique internal name
#
# field:
#   Parquet field
#
# jet:
#   None -> event-level
#   0    -> leading jet
#   1    -> sub-leading jet
# ============================================================

VARIABLES = [
    {
        "key": "event_invariant_mass",
        "label": "Event invariant mass",
        "field": "event_invariant_mass",
        "jet": None,
    },
    {
        "key": "n_jets_original",
        "label": "Number of jets",
        "field": "n_jets_original",
        "jet": None,
    },

    {
        "key": "J1_energy",
        "label": "Leading jet energy",
        "field": "jet_energy",
        "jet": 0,
    },
    {
        "key": "J1_mass",
        "label": "Leading jet mass",
        "field": "jet_mass",
        "jet": 0,
    },
    {
        "key": "J1_constituent_multiplicity",
        "label": "Leading constituent multiplicity",
        "field": "constituent_multiplicity",
        "jet": 0,
    },
    {
        "key": "J1_e2",
        "label": "Leading e2",
        "field": "e2_beta_0p2",
        "jet": 0,
    },
    {
        "key": "J1_e3",
        "label": "Leading e3",
        "field": "e3_beta_0p2",
        "jet": 0,
    },
    {
        "key": "J1_pt",
        "label": "Leading jet pT",
        "field": "jet_pt",
        "jet": 0,
    },
    {
        "key": "J1_p",
        "label": "Leading jet p",
        "field": "jet_p",
        "jet": 0,
    },
    {
        "key": "J1_c2",
        "label": "Leading C2",
        "field": "c2_beta_0p2",
        "jet": 0,
    },
    {
        "key": "J1_d2",
        "label": "Leading D2",
        "field": "d2_beta_0p2",
        "jet": 0,
    },
    {
        "key": "J1_theta",
        "label": "Leading jet theta",
        "field": "jet_theta",
        "jet": 0,
    },

    {
        "key": "J2_energy",
        "label": "Sub-leading jet energy",
        "field": "jet_energy",
        "jet": 1,
    },
    {
        "key": "J2_mass",
        "label": "Sub-leading jet mass",
        "field": "jet_mass",
        "jet": 1,
    },
    {
        "key": "J2_constituent_multiplicity",
        "label": "Sub-leading constituent multiplicity",
        "field": "constituent_multiplicity",
        "jet": 1,
    },
    {
        "key": "J2_e2",
        "label": "Sub-leading e2",
        "field": "e2_beta_0p2",
        "jet": 1,
    },
    {
        "key": "J2_e3",
        "label": "Sub-leading e3",
        "field": "e3_beta_0p2",
        "jet": 1,
    },
    {
        "key": "J2_pt",
        "label": "Sub-leading jet pT",
        "field": "jet_pt",
        "jet": 1,
    },
    {
        "key": "J2_p",
        "label": "Sub-leading jet p",
        "field": "jet_p",
        "jet": 1,
    },
    {
        "key": "J2_c2",
        "label": "Sub-leading C2",
        "field": "c2_beta_0p2",
        "jet": 1,
    },
    {
        "key": "J2_d2",
        "label": "Sub-leading D2",
        "field": "d2_beta_0p2",
        "jet": 1,
    },
    {
        "key": "J2_theta",
        "label": "Sub-leading jet theta",
        "field": "jet_theta",
        "jet": 1,
    },
]


ALL_FIELDS = sorted(
    set(
        EVENT_FIELDS
        + JET_FIELDS
    )
)


# ============================================================
# Helpers
# ============================================================

def to_numpy_finite(array):
    """
    Convert Awkward array to finite 1D NumPy array.
    """

    array = ak.drop_none(array)

    values = ak.to_numpy(array)

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return values[
        np.isfinite(values)
    ]


def extract_variable(data, variable):
    """
    Extract one scalar observable for every event.
    """

    field = variable["field"]
    jet_index = variable["jet"]

    array = data[field]

    # Event-level quantity
    if jet_index is None:
        return to_numpy_finite(array)

    # Jet-level quantity
    padded = ak.pad_none(
        array,
        2,
        axis=1,
        clip=False,
    )

    values = padded[:, jet_index]

    return to_numpy_finite(values)


# ============================================================
# Read one sample
# ============================================================

def process_sample(sample):
    """
    Return:
        {
            variable_key: numpy array
        }
    """

    collected = {
        variable["key"]: []
        for variable in VARIABLES
    }

    total_files = 0
    total_events = 0

    print(f"\nProcessing {sample}")

    for split in SPLITS:

        folder = (
            BASE
            / sample
            / split
        )

        files = sorted(
            folder.glob(
                "events_*_part_*.parquet"
            )
        )

        if MAX_FILES_PER_SPLIT is not None:
            files = files[
                :MAX_FILES_PER_SPLIT
            ]

        print(
            f"  {split:12s}: "
            f"{len(files):4d} files"
        )

        for i, filepath in enumerate(
            files,
            start=1,
        ):

            data = ak.from_parquet(
                filepath,
                columns=ALL_FIELDS,
            )

            total_events += len(data)
            total_files += 1

            for variable in VARIABLES:

                values = extract_variable(
                    data,
                    variable,
                )

                collected[
                    variable["key"]
                ].append(values)

            if (
                i % 20 == 0
                or i == len(files)
            ):
                print(
                    f"      processed "
                    f"{i}/{len(files)}"
                )

    # Concatenate shards
    for key in collected:

        if not collected[key]:
            collected[key] = np.array([])
            continue

        collected[key] = np.concatenate(
            collected[key]
        )

    print(
        f"  Total: {total_events:,} events "
        f"from {total_files} files"
    )

    return collected


# ============================================================
# Metrics
# ============================================================

def calculate_auc(
    signal,
    background,
):
    """
    Orientation-independent univariate ROC AUC.

    Returns:
        raw_auc
        discriminating_auc

    discriminating_auc is always >= 0.5.
    """

    y_true = np.concatenate(
        [
            np.ones(len(signal)),
            np.zeros(len(background)),
        ]
    )

    scores = np.concatenate(
        [
            signal,
            background,
        ]
    )

    raw_auc = roc_auc_score(
        y_true,
        scores,
    )

    discrimination_auc = max(
        raw_auc,
        1.0 - raw_auc,
    )

    return (
        raw_auc,
        discrimination_auc,
    )


def calculate_overlap(
    signal,
    background,
    bins=OVERLAP_BINS,
):
    """
    Histogram overlap coefficient.

    0 = no overlap
    1 = complete overlap
    """

    combined = np.concatenate(
        [
            signal,
            background,
        ]
    )

    xmin = np.min(combined)
    xmax = np.max(combined)

    if xmin == xmax:
        return 1.0

    bin_edges = np.linspace(
        xmin,
        xmax,
        bins + 1,
    )

    sig_hist, _ = np.histogram(
        signal,
        bins=bin_edges,
        density=True,
    )

    bkg_hist, _ = np.histogram(
        background,
        bins=bin_edges,
        density=True,
    )

    widths = np.diff(
        bin_edges
    )

    overlap = np.sum(
        np.minimum(
            sig_hist,
            bkg_hist,
        )
        * widths
    )

    return float(overlap)


def calculate_ks(
    signal,
    background,
):
    """
    Two-sample Kolmogorov-Smirnov test.

    Returns:
        D statistic
        p-value
    """

    result = ks_2samp(
        signal,
        background,
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


# ============================================================
# Analyse all variables
# ============================================================

def analyse_variables(
    signal_data,
    background_data,
):

    results = []

    print(
        "\n"
        + "=" * 100
    )

    print(
        "Observable discrimination"
    )

    print(
        "=" * 100
    )

    for variable in VARIABLES:

        key = variable["key"]
        label = variable["label"]

        signal = signal_data[key]
        background = background_data[key]

        if (
            len(signal) < 2
            or len(background) < 2
        ):
            print(
                f"Skipping {label}: "
                "insufficient entries"
            )
            continue

        raw_auc, auc = calculate_auc(
            signal,
            background,
        )

        overlap = calculate_overlap(
            signal,
            background,
        )

        ks_d, ks_p = calculate_ks(
            signal,
            background,
        )

        signal_mean = np.mean(signal)
        background_mean = np.mean(background)

        results.append(
            {
                "key": key,
                "label": label,

                "n_signal": len(signal),
                "n_background": len(background),

                "signal_mean": signal_mean,
                "background_mean": background_mean,

                "raw_auc": raw_auc,
                "auc": auc,

                "overlap": overlap,

                "ks_d": ks_d,
                "ks_p": ks_p,
            }
        )

    # Sort by standalone AUC
    results.sort(
        key=lambda x: x["auc"],
        reverse=True,
    )

    return results


# ============================================================
# Print table
# ============================================================

def print_results(results):

    print()

    print(
        f"{'Variable':38s}"
        f"{'AUC':>8s}"
        f"{'Overlap':>10s}"
        f"{'KS D':>10s}"
        f"{'Sig mean':>14s}"
        f"{'Bkg mean':>14s}"
    )

    print(
        "-" * 94
    )

    for result in results:

        print(
            f"{result['label'][:37]:38s}"
            f"{result['auc']:8.4f}"
            f"{result['overlap']:10.4f}"
            f"{result['ks_d']:10.4f}"
            f"{result['signal_mean']:14.4g}"
            f"{result['background_mean']:14.4g}"
        )


# ============================================================
# Save CSV
# ============================================================

def save_csv(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "rank",
        "key",
        "label",

        "n_signal",
        "n_background",

        "signal_mean",
        "background_mean",

        "raw_auc",
        "auc",

        "overlap",

        "ks_d",
        "ks_p",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
    ) as outfile:

        writer = csv.DictWriter(
            outfile,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for rank, result in enumerate(
            results,
            start=1,
        ):

            row = {
                "rank": rank,
                **result,
            }

            writer.writerow(row)

    print(
        f"\nSaved results to:\n"
        f"  {OUTPUT_CSV}"
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    signal = process_sample(
        "signal"
    )

    background = process_sample(
        "background"
    )

    results = analyse_variables(
        signal,
        background,
    )

    print_results(
        results
    )

    save_csv(
        results
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
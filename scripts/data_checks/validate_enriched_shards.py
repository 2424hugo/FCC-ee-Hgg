"""Validate all enriched analysis Parquet shards."""

from pathlib import Path

import awkward as ak
import numpy as np


DATASET_ROOT = Path("cache/analysis_dataset")
SAMPLES = ("signal", "background")
SPLITS = ("train", "validation", "test")

TWO_JET_FIELDS = (
    "constituent_multiplicity",
    "e2_beta_0p2",
    "e3_beta_0p2",
    "jet_pt",
    "jet_p",
    "c2_beta_0p2",
    "d2_beta_0p2",
    "jet_theta",
)

FINITE_FIELDS = (
    "constituent_multiplicity",
    "e2_beta_0p2",
    "e3_beta_0p2",
    "jet_pt",
    "jet_p",
    "jet_theta",
)

EXPECTED_LABELS = {
    "background": 0,
    "signal": 1,
}


def as_two_jet_array(data, field):
    values = np.asarray(ak.to_numpy(data[field]))

    expected_shape = (len(data), 2)
    if values.shape != expected_shape:
        raise ValueError(
            f"{field}: expected shape {expected_shape}, "
            f"found {values.shape}"
        )

    return values


def validate_shard(path, sample):
    data = ak.from_parquet(path)

    if len(data) == 0:
        raise ValueError("shard contains no events")

    required_fields = {
        "source_file",
        "source_event",
        "label",
        *TWO_JET_FIELDS,
    }

    missing = required_fields - set(ak.fields(data))
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")

    arrays = {
        field: as_two_jet_array(data, field)
        for field in TWO_JET_FIELDS
    }

    for field in FINITE_FIELDS:
        number_invalid = np.count_nonzero(
            ~np.isfinite(arrays[field])
        )
        if number_invalid:
            raise ValueError(
                f"{field}: {number_invalid} non-finite values"
            )

    multiplicity = arrays["constituent_multiplicity"]
    e2 = arrays["e2_beta_0p2"]
    e3 = arrays["e3_beta_0p2"]
    c2 = arrays["c2_beta_0p2"]
    d2 = arrays["d2_beta_0p2"]

    if np.any(multiplicity < 1):
        raise ValueError("found a jet with zero constituents")

    if np.any(e2 < 0) or np.any(e3 < 0):
        raise ValueError("found a negative ECF value")

    if np.any(arrays["jet_pt"] < 0):
        raise ValueError("found negative jet_pt")

    if np.any(arrays["jet_p"] < 0):
        raise ValueError("found negative jet_p")

    theta = arrays["jet_theta"]
    if np.any((theta < 0) | (theta > np.pi)):
        raise ValueError("jet_theta lies outside [0, pi]")

    # C2 and D2 must be undefined for the same jets.
    c2_nan = np.isnan(c2)
    d2_nan = np.isnan(d2)

    if np.any(np.isinf(c2)) or np.any(np.isinf(d2)):
        raise ValueError("C2 or D2 contains infinity")

    if not np.array_equal(c2_nan, d2_nan):
        raise ValueError(
            "C2 and D2 have different NaN locations"
        )

    # NaN is permitted only for a one-constituent jet with e2=e3=0.
    expected_nan = (
        (multiplicity == 1)
        & (e2 == 0)
        & (e3 == 0)
    )

    if not np.array_equal(c2_nan, expected_nan):
        unexpected = np.count_nonzero(c2_nan & ~expected_nan)
        missing_nan = np.count_nonzero(expected_nan & ~c2_nan)

        raise ValueError(
            "C2/D2 NaN pattern is inconsistent: "
            f"{unexpected} unexpected, "
            f"{missing_nan} expected but absent"
        )

    labels = np.asarray(ak.to_numpy(data["label"]))
    expected_label = EXPECTED_LABELS[sample]

    if not np.all(labels == expected_label):
        raise ValueError(
            f"incorrect labels for {sample}; "
            f"expected {expected_label}"
        )

    source_files = set(ak.to_list(data["source_file"]))

    return len(data), np.count_nonzero(c2_nan), source_files


def main():
    failures = []
    source_files_by_split = {}
    total_shards = 0
    total_events = 0
    total_nan_jets = 0

    for sample in SAMPLES:
        for split in SPLITS:
            directory = DATASET_ROOT / sample / split

            paths = sorted(
                path
                for path in directory.rglob("*part_*.parquet")
                if not path.name.startswith(".")
            )

            key = (sample, split)
            source_files_by_split[key] = set()

            if not paths:
                failures.append(
                    f"{directory}: no Parquet shards found"
                )
                continue

            split_events = 0

            for path in paths:
                try:
                    events, nan_jets, source_files = (
                        validate_shard(path, sample)
                    )

                    source_files_by_split[key].update(
                        source_files
                    )

                    split_events += events
                    total_events += events
                    total_nan_jets += nan_jets
                    total_shards += 1

                    print(
                        f"PASS  {path} | "
                        f"events={events:,} | "
                        f"undefined C2/D2 jets={nan_jets:,}"
                    )

                except Exception as error:
                    failures.append(f"{path}: {error}")
                    print(f"FAIL  {path} | {error}")

            print(
                f"\n{sample}/{split}: "
                f"{len(paths)} shards, "
                f"{split_events:,} events\n"
            )

    # Check file-level train/validation/test separation.
    for sample in SAMPLES:
        for index, split_a in enumerate(SPLITS):
            for split_b in SPLITS[index + 1:]:
                files_a = source_files_by_split[
                    (sample, split_a)
                ]
                files_b = source_files_by_split[
                    (sample, split_b)
                ]

                overlap = files_a & files_b

                if overlap:
                    failures.append(
                        f"{sample}: {len(overlap)} source files "
                        f"overlap between {split_a} and {split_b}"
                    )

    print("=" * 70)
    print(f"Validated shards: {total_shards:,}")
    print(f"Validated events: {total_events:,}")
    print(
        "One-constituent jets with undefined C2/D2: "
        f"{total_nan_jets:,}"
    )

    if failures:
        print(f"\nVALIDATION FAILED: {len(failures)} problem(s)")

        for failure in failures:
            print(f"  - {failure}")

        raise SystemExit(1)

    print("\nPASS: every shard and dataset split is valid.")


if __name__ == "__main__":
    main()
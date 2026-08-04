"""Add the selected high-level observables to analysis Parquet shards.

The expensive energy-correlation functions are evaluated in small event
chunks.  Each input shard is replaced only after the temporary output has
been reloaded and validated successfully.

Run from the repository root, for example:

    python -m scripts.data_processing.add_selected_hl_to_shards \
        --file cache/analysis_dataset/background/train/events_000_part_0000.parquet

    python -m scripts.data_processing.add_selected_hl_to_shards \
        --all \
        --dataset-root cache/analysis_dataset/background/train
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import awkward as ak
import numpy as np

from scripts.data_processing.add_hl_observables import add_hl_observables


BETA = 0.2
DEFAULT_CHUNK_SIZE = 500

# The helper uses short names internally.  The ECF names written to disk retain
# beta explicitly so that their definition remains unambiguous.
OUTPUT_FIELD_MAP = {
    "e2": "e2_beta_0p2",
    "e3": "e3_beta_0p2",
    "jet_pt": "jet_pt",
    "jet_p": "jet_p",
    "c2": "c2_beta_0p2",
    "d2": "d2_beta_0p2",
    "jet_theta": "jet_theta",
}
ML_FIELDS = tuple(OUTPUT_FIELD_MAP.values())


def calculate_observables_in_chunks(
    data: ak.Array,
    chunk_size: int,
) -> dict[str, ak.Array]:
    """Calculate the seven selected observables without full-shard triples."""

    if len(data) == 0:
        raise ValueError("Cannot enrich an empty shard")

    output: dict[str, list[np.ndarray]] = {
        stored_name: [] for stored_name in ML_FIELDS
    }

    number_of_events = len(data)

    for start in range(0, number_of_events, chunk_size):
        stop = min(start + chunk_size, number_of_events)
        result = add_hl_observables(data[start:stop], beta=BETA)

        for result_name, stored_name in OUTPUT_FIELD_MAP.items():
            values = np.asarray(
                ak.to_numpy(result[result_name]),
                dtype=np.float64,
            )

            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError(
                    f"{stored_name} has unexpected shape {values.shape}"
                )

            output[stored_name].append(values)

        print(
            f"    processed {stop:,}/{number_of_events:,} events",
            flush=True,
        )

    return {
        field: ak.Array(np.concatenate(chunks, axis=0))
        for field, chunks in output.items()
    }


def validate_enriched_shard(data: ak.Array) -> None:
    """Validate event counts, two-jet structure, and numerical values."""

    missing = [field for field in ML_FIELDS if field not in ak.fields(data)]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    for field in ML_FIELDS:
        if len(data[field]) != len(data):
            raise ValueError(f"{field} has the wrong event count")

        if not bool(ak.all(ak.num(data[field], axis=1) == 2)):
            raise ValueError(
                f"{field} does not contain exactly two jets per event"
            )

    always_finite = (
        "e2_beta_0p2",
        "e3_beta_0p2",
        "jet_pt",
        "jet_p",
        "jet_theta",
    )
    for field in always_finite:
        number_invalid = int(ak.sum(~np.isfinite(data[field])))
        if number_invalid:
            raise ValueError(
                f"{field} contains {number_invalid:,} non-finite values"
            )

    non_negative = (
        "e2_beta_0p2",
        "e3_beta_0p2",
        "jet_pt",
        "jet_p",
    )
    for field in non_negative:
        if not bool(ak.all(data[field] >= 0.0)):
            raise ValueError(f"{field} contains negative values")

    if not bool(
        ak.all(
            (data["jet_theta"] >= 0.0)
            & (data["jet_theta"] <= np.pi)
        )
    ):
        raise ValueError("jet_theta contains values outside [0, pi]")

    # C2 and D2 are undefined when e2 is too close to zero.  Report these
    # events rather than silently treating NaN as a physical value.
    for field in ("c2_beta_0p2", "d2_beta_0p2"):
        number_invalid = int(ak.sum(~np.isfinite(data[field])))
        print(f"    {field}: {number_invalid:,} non-finite values")


def enrich_shard(input_path: Path, chunk_size: int) -> None:
    """Enrich one shard and replace it only after a reload validation."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Shard does not exist: {input_path}")

    print(f"\nProcessing: {input_path}")

    data = ak.from_parquet(input_path)
    print(f"    events: {len(data):,}")

    observables = calculate_observables_in_chunks(data, chunk_size)
    for field, values in observables.items():
        data = ak.with_field(data, values, field)

    validate_enriched_shard(data)

    temporary_path = input_path.with_name(f".{input_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)

    try:
        ak.to_parquet(data, temporary_path, compression=None)

        written = ak.from_parquet(temporary_path)
        validate_enriched_shard(written)

        if len(written) != len(data):
            raise ValueError("Written shard has a different event count")

        os.replace(temporary_path, input_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"    updated: {input_path}")


def find_shards(dataset_root: Path) -> list[Path]:
    """Find both ``part_*.parquet`` and ``events_*_part_*.parquet`` shards."""

    if not dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root does not exist: {dataset_root}"
        )

    return sorted(
        path
        for path in dataset_root.rglob("*.parquet")
        if not path.name.startswith(".")
        and (
            path.name.startswith("part_")
            or "_part_" in path.stem
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--file",
        type=Path,
        help="Process one Parquet shard",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Process every shard below --dataset-root",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("cache/analysis_dataset"),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    paths = (
        [args.file]
        if args.file is not None
        else find_shards(args.dataset_root)
    )

    if not paths:
        raise FileNotFoundError("No Parquet shards found")

    print(f"beta: {BETA}")
    print(f"chunk size: {args.chunk_size:,}")
    print(f"shards to process: {len(paths):,}")

    for index, path in enumerate(paths, start=1):
        print(f"\nShard {index:,}/{len(paths):,}")
        enrich_shard(path, args.chunk_size)


if __name__ == "__main__":
    main()
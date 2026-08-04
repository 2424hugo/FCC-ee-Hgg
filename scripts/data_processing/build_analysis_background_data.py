from pathlib import Path

import pyarrow.parquet as pq

from build_analysis_data_refactored import (
    build_file_cache,
    load_splits,
)


MANIFEST_PATH = Path("cache/dataset_splits_50_12_12.json")

EXPECTED_BACKGROUND_FILES = {
    "train": 50,
    "validation": 12,
    "test": 12,
}

SPLIT_NAMES = ("train", "validation", "test")


def main():
    splits = load_splits(MANIFEST_PATH)
    background_splits = splits["background"]

    # Validate file counts.
    for split_name in SPLIT_NAMES:
        input_files = background_splits[split_name]
        expected = EXPECTED_BACKGROUND_FILES[split_name]

        if len(input_files) != expected:
            raise ValueError(
                f"Expected {expected} background files in {split_name}, "
                f"but found {len(input_files)}"
            )

    # Validate total count and disjointness.
    all_files = [
        file_path
        for split_name in SPLIT_NAMES
        for file_path in background_splits[split_name]
    ]

    if len(all_files) != 74:
        raise ValueError(
            f"Expected 74 background files, but found {len(all_files)}"
        )

    if len(set(all_files)) != len(all_files):
        raise ValueError(
            "The background splits are not disjoint: "
            "at least one ROOT file appears in multiple splits."
        )

    summaries = []

    for split_name in SPLIT_NAMES:
        input_files = background_splits[split_name]

        output_dir = (
            Path("cache/analysis_dataset")
            / "background"
            / split_name
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n{'=' * 70}\n"
            f"Building background {split_name}: "
            f"{len(input_files)} ROOT files\n"
            f"Output: {output_dir}\n"
            f"{'=' * 70}"
        )

        for file_number, input_file_string in enumerate(
            input_files,
            start=1,
        ):
            input_file = Path(input_file_string)
            source_prefix = input_file.stem

            print(
                f"\nBackground {split_name} file "
                f"{file_number}/{len(input_files)}"
            )
            print(f"Input: {input_file}")

            existing_shards = sorted(
                output_dir.glob(
                    f"{source_prefix}_part_*.parquet"
                )
            )

            if len(existing_shards) == 1:
                # Confirm that the Parquet footer is readable.
                rows = (
                    pq.ParquetFile(existing_shards[0])
                    .metadata.num_rows
                )

                print(
                    f"Skipping completed file: {input_file.name} "
                    f"(1 shard, {rows:,} selected events)"
                )
                continue

            if existing_shards:
                raise RuntimeError(
                    f"Unexpected output for {input_file.name}: "
                    f"found {len(existing_shards)} shards: "
                    f"{[path.name for path in existing_shards]}"
                )

            summary = build_file_cache(
                input_file=input_file,
                output_dir=output_dir,
                label=0,
                # A ROOT file contains 100,000 input events, so all
                # selected events will be placed in one output shard.
                events_per_shard=100_000,
                compression=None,
            )

            if len(summary.output_files) != 1:
                raise RuntimeError(
                    f"Expected one output shard for {input_file.name}, "
                    f"but created {len(summary.output_files)}"
                )

            summaries.append(
                {
                    "split": split_name,
                    "input_file": str(input_file),
                    "summary": summary,
                }
            )

            print(summary)

    print("\nCompleted all background files")
    print("Train ROOT files: 50")
    print("Validation ROOT files: 12")
    print("Test ROOT files: 12")
    print("Total ROOT files: 74")
    print(f"Newly processed this run: {len(summaries)}")


if __name__ == "__main__":
    main()

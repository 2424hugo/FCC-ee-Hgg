# imports to process ROOT file and read data split from json manifest
from build_analysis_data_refactored import build_file_cache, load_splits
# handle file paths safely
from pathlib import Path

# loading in the splits
splits = load_splits("cache/dataset_splits.json")

# expected signal-file allocation
EXPECTED_SIGNAL_FILES = {
    "train": 8,
    "validation": 2,
    "test": 2,
}
SPLIT_NAMES = ("train", "validation", "test")

"""
test_samples = {
    "signal": {
        "label": 1,
        "input_file": splits["signal"]["train"][0],
    },
    "background": {
        "label": 0,
        "input_file": splits["background"]["train"][0],
    },
}

for sample, config in test_samples.items():
    input_file = Path(config["input_file"])

    print(f"\nTesting {sample}")
    print(f"Input: {input_file}")

    summary = build_file_cache(
        input_file=input_file,
        output_dir=Path("cache/analysis_dataset") / sample / "train",
        label=config["label"],
    )

    print(summary)
"""

def main():
    # laoding in the splits
    splits = load_splits("cache/dataset_splits.json")
    # only looking at the signal files
    signal_splits = splits["signal"]

    # Validate the manifest before processing.
    for split_name in SPLIT_NAMES: # for train, val, test
        input_files = signal_splits[split_name]
        expected = EXPECTED_SIGNAL_FILES[split_name]

        if len(input_files) != expected: # check that input is same as expected
            raise ValueError(
                f"Expected {expected} signal files in {split_name}, "
                f"but found {len(input_files)}"
            )
    # combine three lists of ROOT paths into only one
    all_files = [
        file_path
        for split_name in SPLIT_NAMES
        for file_path in signal_splits[split_name]
    ]

    if len(all_files) != 12: # make sure that the number of files are 12
        raise ValueError(
            f"Expected 12 signal files, but found {len(all_files)}"
        )

    if len(set(all_files)) != len(all_files): # ensure each file is unqiue
        raise ValueError(
            "The signal splits are not disjoint: "
            "at least one ROOT file appears in multiple splits."
        )

    summaries = [] # stores record of each processed ROOT file

    for split_name in SPLIT_NAMES: # repeat for train, val, test
        input_files = signal_splits[split_name]
        output_dir = (
            Path("cache/analysis_dataset")
            / "signal"
            / split_name
        ) # create save dir location name
        output_dir.mkdir(parents=True, exist_ok=True) # make sure dir exists
        
        # prints number of files and where they will be saved after processing
        print(
            f"\n{'=' * 70}\n"
            f"Building signal {split_name}: "
            f"{len(input_files)} ROOT files\n"
            f"Output: {output_dir}\n"
            f"{'=' * 70}"
        )
        # repeat for every ROOT file inputed
        for file_number, input_file_string in enumerate(
            input_files,
            start=1,
        ):
            input_file = Path(input_file_string) # convert to Path object
            # print which file is currently processing
            print(
                f"\nSignal {split_name} file "
                f"{file_number}/{len(input_files)}"
            )
            print(f"Input: {input_file}") # print file name
            # used for filename
            source_prefix = input_file.stem
            
            existing_shards = sorted(
                output_dir.glob(f"{source_prefix}_part_*.parquet")
            )
            # after two runs file is complete
            if len(existing_shards) == 2:
                print(
                    f"Skipping completed file: {input_file.name} "
                    f"({len(existing_shards)} shards already exist)"
                )
                continue
            # make sure there are two files
            if existing_shards:
                raise RuntimeError(
                    f"Incomplete output for {input_file.name}: "
                    f"found {len(existing_shards)} shard(s): "
                    f"{[path.name for path in existing_shards]}"
                )
            # create parquet shards
            summary = build_file_cache(
                input_file=input_file,
                output_dir=output_dir,
                label=1,
            )
            # add one record per pocessed file
            summaries.append(
                {
                    "split": split_name,
                    "input_file": str(input_file),
                    "summary": summary,
                }
            )
            # displays number of events, selected events, output shards and elapsed time
            print(summary)
    # show that run is complete
    print("\nCompleted all signal files")
    print("Train ROOT files: 8")
    print("Validation ROOT files: 2")
    print("Test ROOT files: 2")

# run if file is called directly
if __name__ == "__main__":
    main()

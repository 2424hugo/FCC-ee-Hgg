import json
from pathlib import Path

from build_analysis_data_refactored import load_splits

# location of json files which store train/val/test split
SOURCE_MANIFEST = Path("cache/dataset_splits.json") # containing all sig and bkg files
OUTPUT_MANIFEST = Path("cache/dataset_splits_50_12_12.json") # containing all sig and 74 bkg files

# background subset size to be used
SUBSET_COUNTS = {
    "train": 50,
    "validation": 12,
    "test": 12,
}


def main():
    # loading in full split
    splits = load_splits(SOURCE_MANIFEST)
    
    # selecting background files
    # Looping for and taking the first:
    # train :50
    # val :12
    # test :12
    background_subset = {
        split_name: splits["background"][split_name][:number_of_files]
        for split_name, number_of_files in SUBSET_COUNTS.items()
    }
    
    # combine the 3 lists into 1 for checks
    all_background_files = [
        file_path
        for split_name in SUBSET_COUNTS
        for file_path in background_subset[split_name]
    ]
    
    # check that all 74 files are contained
    if len(all_background_files) != 74:
        raise ValueError(
            f"Expected 74 background files, found "
            f"{len(all_background_files)}"
        )
    
    # check that all files are unqiue without overlap
    if len(set(all_background_files)) != 74:
        raise ValueError(
            "Background train, validation and test subsets overlap"
        )
    
    # creating the new manifest
    # copying all signal
    # using new reduced background
    subset_manifest = {
        "signal": splits["signal"],
        "background": background_subset,
    }
    
    # checks cache exists
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    
    # writs JSON manifest
    with OUTPUT_MANIFEST.open("w") as output_file:
        json.dump(subset_manifest, output_file, indent=2) # store new dictionary
    print(f"Wrote: {OUTPUT_MANIFEST}")
    
    # reports num of selected files for background split
    for split_name in SUBSET_COUNTS:
        print(
            f"Background {split_name:10s}: "
            f"{len(background_subset[split_name])} files"
        )


if __name__ == "__main__":
    # run main
    main()
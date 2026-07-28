import json
from pathlib import Path
import random
import glob
import uproot

SIGNAL_PATH = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/wzp6_ee_Hgg_ecm125/*.root"

BACKGROUND_PATH = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/wzp6_ee_qq_ecm125/*.root"

# loading in the sorted data files
background_files = sorted(glob.glob(BACKGROUND_PATH))
signal_files = sorted(glob.glob(SIGNAL_PATH))

# random seed for split
rng = random.Random(42)

# copy of file list
signal_shuffled = signal_files.copy()
background_shuffled = background_files.copy()

# shuffle the lists
rng.shuffle(signal_shuffled)
rng.shuffle(background_shuffled)

# split the signal data into training, validation and test
signal_train = signal_shuffled[:8]
signal_val = signal_shuffled[8:10]
signal_test = signal_shuffled[10:]

# find the 70% and 85% split marks for bkg data
bkg_train_end = int(0.70 * len(background_shuffled))
bkg_val_end = int(0.85 * len(background_shuffled))
# create split of 70% 15% 15%
background_train = background_shuffled[:bkg_train_end]
background_val = background_shuffled[bkg_train_end:bkg_val_end]
background_test = background_shuffled[bkg_val_end:]

# check the split
print("Signal:", len(signal_train), len(signal_val), len(signal_test))
print("Background:", len(background_train), len(background_val), len(background_test))

# verify that no files are used twice or lost
signal_disjoint = (
    set(signal_train).isdisjoint(signal_val)
    and set(signal_train).isdisjoint(signal_test)
    and set(signal_val).isdisjoint(signal_test)
)
background_disjoint = (
    set(background_train).isdisjoint(background_val)
    and set(background_train).isdisjoint(background_test)
    and set(background_val).isdisjoint(background_test)
)

signal_complete = (
    set(signal_train + signal_val + signal_test)
    == set(signal_files)
)

background_complete = (
    set(background_train + background_val + background_test)
    == set(background_files)
)

print("Signal partitions disjoint:", signal_disjoint)
print("Background partitions disjoint:", background_disjoint)
print("All signal files retained:", signal_complete)
print("All background files retained:", background_complete)

# save splits using json file
# create manifest
splits = {
        "seed": 42,
        "signal": {
            "train": signal_train,
            "validation": signal_val,
            "test": signal_test,
        },
        "background": {
            "train": background_train,
            "validation": background_val,
            "test": background_test,
        },
}
# the output path
output_path = Path("cache/dataset_splits.json")
# saving locations
with output_path.open("w") as split_file:
    json.dump(splits, split_file, indent=2)
# check the save
print("Saved split manifest:", output_path)
print("File exists:", output_path.exists())
print("File size:", output_path.stat().st_size, "bytes")

with output_path.open("r") as split_file:
    saved_splits = json.load(split_file)

print(
    "Saved signal:",
    len(saved_splits["signal"]["train"]),
    len(saved_splits["signal"]["validation"]),
    len(saved_splits["signal"]["test"]),
)
print(
    "Saved background:",
    len(saved_splits["background"]["train"]),
    len(saved_splits["background"]["validation"]),
    len(saved_splits["background"]["test"]),
)

print(saved_splits["background"]["train"][0])
print(saved_splits["signal"]["train"][0])

import random
import glob
import uproot

SIGNAL_PATH = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/wzp6_ee_Hgg_ecm125/*.root"

BACKGROUND_PATH = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/wzp6_ee_qq_ecm125/*.root"

# loading in the sorted data files
signal_files = sorted(glob.glob(SIGNAL_PATH))
background_files = sorted(glob.glob(BACKGROUND_PATH))

# How many files are in this location
print("Signal files:", len(signal_files))
print("Background files:", len(background_files))

# what does each root file contain
f = uproot.open(signal_files[0])
print(f.keys())

# how many events in each files events
for path in signal_files:
    with uproot.open(path) as f:
        print(path, f["events"].num_entries)
"""
# random selection of 20 files in bkg
import random

rng = random.Random(42)
sample_paths = random.sample(background_files, 20)

sample_counts = []

for path in sample_paths:
    with uproot.open(path) as bkg_file:
        sample_counts.append(bkg_file["events"].num_entries)

print(set(sample_counts))
"""
# search for event weights
with uproot.open(background_files[0]) as bkg_file:
    bkg_tree = bkg_file["events"]
    weight_keys = [key for key in bkg_tree.keys() if "weight" in key.lower()]

print(weight_keys)

metadata_keys = [
    key for key in bkg_tree.keys()
    if any(word in key.lower() for word in ["eventheader", "eventweight", "crosssection"])
]

print(metadata_keys)

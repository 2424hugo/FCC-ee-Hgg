import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


BRANCHES = [
    "ReconstructedParticles/ReconstructedParticles.energy",
    "ReconstructedParticles/ReconstructedParticles.momentum.x",
    "ReconstructedParticles/ReconstructedParticles.momentum.y",
    "ReconstructedParticles/ReconstructedParticles.momentum.z",

    "Jet/Jet.energy",
    "Jet/Jet.momentum.x",
    "Jet/Jet.momentum.y",
    "Jet/Jet.momentum.z",

    "Jet/Jet.particles_begin",
    "Jet/Jet.particles_end",

    "Jet#2/Jet#2.index",
    "Jet#2/Jet#2.collectionID",
]


# Load the fixed dataset split
with Path("cache/dataset_splits.json").open() as f:
    splits = json.load(f)

test_file = splits["signal"]["test"][0]


# Load only 20 events from one test file
with uproot.open(test_file) as root_file:
    data = root_file["events"].arrays(
        BRANCHES,
        entry_start=0,
        entry_stop=20,
        library="ak",
    )

print(f"Loaded {len(data)} events from:")
print(test_file)


# Jet constituent relation ranges
begin = data["Jet/Jet.particles_begin"]
end = data["Jet/Jet.particles_end"]

# Relation collection
relation_index = data["Jet#2/Jet#2.index"]
relation_collection = data["Jet#2/Jet#2.collectionID"]

# Reconstructed-particle four-vectors
reco_E = data["ReconstructedParticles/ReconstructedParticles.energy"]
reco_px = data["ReconstructedParticles/ReconstructedParticles.momentum.x"]
reco_py = data["ReconstructedParticles/ReconstructedParticles.momentum.y"]
reco_pz = data["ReconstructedParticles/ReconstructedParticles.momentum.z"]

# Stored jet four-vectors
jet_E = data["Jet/Jet.energy"]
jet_px = data["Jet/Jet.momentum.x"]
jet_py = data["Jet/Jet.momentum.y"]
jet_pz = data["Jet/Jet.momentum.z"]


# Check which collection IDs are referenced
all_collection_ids = ak.to_numpy(
    ak.flatten(relation_collection, axis=None)
)

print(
    "\nUnique collection IDs:",
    np.unique(all_collection_ids),
)


# Check relation-array lengths
print("\nRelation length checks:")

for event in range(min(5, len(data))):
    required = int(ak.max(end[event])) if len(end[event]) > 0 else 0
    available = len(relation_index[event])

    print(
        f"Event {event}: "
        f"max end={required}, "
        f"relation length={available}, "
        f"exact match={required == available}"
    )


# Check constituent four-vector sums
print("\nFour-vector checks:")

for event in range(min(5, len(data))):
    print(f"\nEvent {event}")

    for jet in range(len(begin[event])):
        b = int(begin[event][jet])
        e = int(end[event][jet])

        # Select this jet's entries in the relation array
        constituent_indices = relation_index[event][b:e]
        collection_ids = relation_collection[event][b:e]

        number_of_reco_particles = len(reco_E[event])

        indices_valid = (
            len(constituent_indices) > 0
            and bool(
                ak.all(
                    (constituent_indices >= 0)
                    & (constituent_indices < number_of_reco_particles)
                )
            )
        )

        print(
            f"Jet {jet}: "
            f"range=[{b}:{e}], "
            f"n={len(constituent_indices)}, "
            f"indices valid={indices_valid}"
        )

        if not indices_valid:
            print("  Invalid or empty constituent indices")
            continue

        # Use the relation indices to access ReconstructedParticles
        constituent_E = reco_E[event][constituent_indices]
        constituent_px = reco_px[event][constituent_indices]
        constituent_py = reco_py[event][constituent_indices]
        constituent_pz = reco_pz[event][constituent_indices]

        sum_E = float(ak.sum(constituent_E))
        sum_px = float(ak.sum(constituent_px))
        sum_py = float(ak.sum(constituent_py))
        sum_pz = float(ak.sum(constituent_pz))

        stored_E = float(jet_E[event][jet])
        stored_px = float(jet_px[event][jet])
        stored_py = float(jet_py[event][jet])
        stored_pz = float(jet_pz[event][jet])

        matches = np.allclose(
            [stored_E, stored_px, stored_py, stored_pz],
            [sum_E, sum_px, sum_py, sum_pz],
            rtol=1e-5,
            atol=1e-5,
        )

        print("  collection IDs:", np.unique(ak.to_numpy(collection_ids)))
        print(f"  E : jet={stored_E:.6f}, sum={sum_E:.6f}")
        print(f"  px: jet={stored_px:.6f}, sum={sum_px:.6f}")
        print(f"  py: jet={stored_py:.6f}, sum={sum_py:.6f}")
        print(f"  pz: jet={stored_pz:.6f}, sum={sum_pz:.6f}")
        print("  Four-vector matches:", matches)
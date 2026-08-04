# Standard-library imports for reading the dataset split,
# handling file paths, and measuring runtime
import json
import time
from pathlib import Path
# imports for data handling
import awkward as ak
import numpy as np
import uproot

# configuration to control how much data is loaded
ROOT_BATCH_SIZE = 10_000 # Number of ROOT events read per batch
EVENTS_PER_SHARD = 50_000 # Number of selected events written per shard
EVENT_MASS_MIN = 120.0 # Keep events with jet-based invariant mass above 120 GeV

# output location for the cache
OUTPUT_FILE = Path(
    "cache/analysis_dataset/background/train/"
    "part_0000.parquet"
)

# ROOT branches required for event selection, jet storage,
# and constituent extraction
branches = [
    # Jet-level kinematic quantities
    "Jet/Jet.energy",
    "Jet/Jet.momentum.x",
    "Jet/Jet.momentum.y",
    "Jet/Jet.momentum.z",
    "Jet/Jet.mass",
    # Start and exclusive end positions of each jet's
    # entries in the Jet#2 relation array
    "Jet/Jet.particles_begin",
    "Jet/Jet.particles_end",
    "Jet#2/Jet#2.index", # ReconstructedParticles indices referenced by the jet-particle relation
    # reconstructed-particle quantities
    "ReconstructedParticles/ReconstructedParticles.energy",
    "ReconstructedParticles/ReconstructedParticles.momentum.x",
    "ReconstructedParticles/ReconstructedParticles.momentum.y",
    "ReconstructedParticles/ReconstructedParticles.momentum.z",
    "ReconstructedParticles/ReconstructedParticles.mass",
    "ReconstructedParticles/ReconstructedParticles.charge",
    "ReconstructedParticles/ReconstructedParticles.type",
]

def relation_indices_for_jet(
    relation_indices,
    begins,
    ends,
):
    """
    Extract the ReconstructedParticles indices belonging to one jet
    in every event.

    Parameters
    ----------
    relation_indices
        Per-event Jet#2 relation arrays.
        Shape: events * variable number of relation entries

    begins, ends
        Start and exclusive-end positions for one jet per event.
        Shape: events

    Returns
    -------
    Awkward Array
        Shape: events * variable number of constituent indices
    """

    relation_positions = ak.local_index(
        relation_indices,
        axis=1,
    )

    relation_mask = (
        (relation_positions >= begins[:, None])
        & (relation_positions < ends[:, None])
    )

    return relation_indices[relation_mask]


def extract_two_jet_constituents(
    reco_values,
    relation_indices,
    selected_begins,
    selected_ends,
):
    """
    Extract one reconstructed-particle field for the two selected jets.

    Returns an array with shape:

        events * 2 jets * variable number of constituents
    """

    leading_indices = relation_indices_for_jet(
        relation_indices,
        selected_begins[:, 0],
        selected_ends[:, 0],
    )

    subleading_indices = relation_indices_for_jet(
        relation_indices,
        selected_begins[:, 1],
        selected_ends[:, 1],
    )

    # Jagged indexing applies each event's indices to that event's
    # ReconstructedParticles collection.
    leading_values = reco_values[leading_indices]
    subleading_values = reco_values[subleading_indices]

    return ak.concatenate(
        [
            leading_values[:, None],
            subleading_values[:, None],
        ],
        axis=1,
    )

# Load the frozen train/validation/test file split
with Path("cache/dataset_splits.json").open() as f:
    splits = json.load(f)

test_file = splits["background"]["train"][0]

print(f"Opening: {test_file}")
print(f"Output:  {OUTPUT_FILE}")

# Store completed batch-level Awkward arrays
shard_batches = []
total_input_events = 0 # Number of ROOT events read
total_selected_events = 0 # Number of selected events stored

# initialise timing variables
total_start = time.perf_counter()
processing_time = 0.0
write_time = 0.0

# open the ROOT file
with uproot.open(test_file) as root_file:
    tree = root_file["events"]
    
    # iterate through the ROOT file in batches of 10,000 events.
    for batch_number, arrays in enumerate(
        tree.iterate(
            expressions=branches,
            step_size=ROOT_BATCH_SIZE,
            library="ak",
        )
    ):
        
        # Construct source event numbers
        batch_size = len(arrays)
        source_event_numbers = np.arange(
            total_input_events,
            total_input_events + batch_size,
        )

        total_input_events += batch_size

        processing_start = time.perf_counter()
        
        # build the reconstructed event four-vector
        event_energy = ak.sum(
            arrays["Jet/Jet.energy"],
            axis=1,
        )
        event_px = ak.sum(
            arrays["Jet/Jet.momentum.x"],
            axis=1,
        )
        event_py = ak.sum(
            arrays["Jet/Jet.momentum.y"],
            axis=1,
        )
        event_pz = ak.sum(
            arrays["Jet/Jet.momentum.z"],
            axis=1,
        )
        
        # create the invarent mass for the event
        event_mass_squared = (event_energy**2- event_px**2- event_py**2- event_pz**2)
        event_mass = np.sqrt(
            np.maximum(ak.to_numpy(event_mass_squared), 0.0)
        )
        # count jets in every event
        n_jets = ak.to_numpy(
            ak.num(arrays["Jet/Jet.energy"], axis=1)
        )
        # apply the event selection for 120 GeV
        selection = (
            (n_jets >= 2)
            & (event_mass > EVENT_MASS_MIN)
        ) # boolean array

        selected_indices = np.nonzero(selection)[0] # selected events using boolean array
        
        remaining_events = EVENTS_PER_SHARD - total_selected_events

        if remaining_events <= 0:
            break

        selected_indices = selected_indices[:remaining_events]

        if len(selected_indices) == 0:
            continue

        vector_start = time.perf_counter()

        # Select all passing events in this ROOT batch.
        selected = arrays[selected_indices]
        
        selected_event_mass = event_mass[selected_indices]
        selected_n_jets = n_jets[selected_indices]
        selected_source_events = source_event_numbers[selected_indices]

        # Find the leading and subleading jet in every event.
        jet_order = ak.argsort(
            selected["Jet/Jet.energy"],
            axis=1,
            ascending=False,
        )[:, :2]
        
        # Extract the selected jet fields.
        jet_energy = selected[
            "Jet/Jet.energy"
        ][jet_order]
        
        jet_px = selected[
            "Jet/Jet.momentum.x"
        ][jet_order]
        
        jet_py = selected[
            "Jet/Jet.momentum.y"
        ][jet_order]
        
        jet_pz = selected[
            "Jet/Jet.momentum.z"
        ][jet_order]
        
        jet_mass = selected[
            "Jet/Jet.mass"
        ][jet_order]
        
        # Select the relation-array boundaries for the chosen jets.
        selected_begins = selected[
            "Jet/Jet.particles_begin"
        ][jet_order]
        
        selected_ends = selected[
            "Jet/Jet.particles_end"
        ][jet_order]
        
        relation_indices = selected[
            "Jet#2/Jet#2.index"
        ]
        
        # Reconstructed-particle collections.
        reco_energy = selected[
            "ReconstructedParticles/"
            "ReconstructedParticles.energy"
            ]
            
        reco_px = selected[
            "ReconstructedParticles/"
            "ReconstructedParticles.momentum.x"
        ]
        
        reco_py = selected[
            "ReconstructedParticles/"
            "ReconstructedParticles.momentum.y"
        ]
        
        reco_pz = selected[
            "ReconstructedParticles/"
            "ReconstructedParticles.momentum.z"
        ]
        
        reco_mass = selected[
            "ReconstructedParticles/"
            "ReconstructedParticles.mass"
        ]
        
        reco_charge = selected[
            "ReconstructedParticles/"
            "ReconstructedParticles.charge"
        ]
        
        reco_type = selected[
            "ReconstructedParticles/"
            "ReconstructedParticles.type"
        ]

        # Apply the jet relation mapping to all events simultaneously.
        constituent_energy = extract_two_jet_constituents(
            reco_energy,
            relation_indices,
            selected_begins,
            selected_ends,
        )

        constituent_px = extract_two_jet_constituents(
            reco_px,
            relation_indices,
            selected_begins,
            selected_ends,
        )

        constituent_py = extract_two_jet_constituents(
            reco_py,
            relation_indices,
            selected_begins,
            selected_ends,
        )

        constituent_pz = extract_two_jet_constituents(
            reco_pz,
            relation_indices,
            selected_begins,
            selected_ends,
        )

        constituent_mass = extract_two_jet_constituents(
            reco_mass,
            relation_indices,
            selected_begins,
            selected_ends,
        )

        constituent_charge = extract_two_jet_constituents(
            reco_charge,
            relation_indices,
            selected_begins,
            selected_ends,
        )

        constituent_type = extract_two_jet_constituents(
            reco_type,
            relation_indices,
            selected_begins,
            selected_ends,
        )

        constituent_multiplicity = ak.num(
            constituent_energy,
            axis=2,
        )

        n_selected = len(selected_indices)

        # Create one event-level record array directly.
        batch_array = ak.zip(
            {
                "source_file": ak.Array(
                    [str(test_file)] * n_selected
                ),
                "source_event": selected_source_events,
                "label": np.ones(
                    n_selected,
                    dtype=np.int8,
                ),
                "event_invariant_mass": selected_event_mass,
                "n_jets_original": selected_n_jets,

                "jet_energy": jet_energy,
                "jet_px": jet_px,
                "jet_py": jet_py,
                "jet_pz": jet_pz,
                "jet_mass": jet_mass,

                "constituent_energy": constituent_energy,
                "constituent_px": constituent_px,
                "constituent_py": constituent_py,
                "constituent_pz": constituent_pz,
                "constituent_mass": constituent_mass,
                "constituent_charge": constituent_charge,
                "constituent_type": constituent_type,
                "constituent_multiplicity": constituent_multiplicity,
            },
            depth_limit=1,
        )

        vector_time = time.perf_counter() - vector_start

        shard_batches.append(batch_array)
        total_selected_events += len(batch_array)

        print(
            f"Vector construction: {vector_time:.3f} s"
        )
        
        processing_time += time.perf_counter() - processing_start

        elapsed = time.perf_counter() - total_start

        print(
            f"Batch {batch_number:02d} | "
            f"input: {total_input_events:,} | "
            f"selected: {total_selected_events:,} | "
            f"elapsed: {elapsed:.2f} s"
        )
        # Stop reading once this shard contains the requested number of events
        if total_selected_events >= EVENTS_PER_SHARD:
            break

print("\nCombining batch arrays...")
conversion_start = time.perf_counter()
# Join the small number of batch-level arrays into one shard
output_array = ak.concatenate(
    shard_batches,
    axis=0,
)

print("\nValidating vectorised output...")

assert len(output_array) <= EVENTS_PER_SHARD

if len(output_array) < EVENTS_PER_SHARD:
    print(
        f"Warning: only {len(output_array):,} events "
        f"were available for this shard."
    )

assert ak.all(
    ak.num(output_array.jet_energy, axis=1) == 2
)

assert ak.all(
    ak.num(output_array.constituent_energy, axis=1) == 2
)

assert ak.all(
    ak.num(output_array.constituent_energy, axis=2)
    == output_array.constituent_multiplicity
)

# Confirm all constituent fields have identical ragged structure.
reference_lengths = ak.num(
    output_array.constituent_energy,
    axis=2,
)

for field in [
    "constituent_px",
    "constituent_py",
    "constituent_pz",
    "constituent_mass",
    "constituent_charge",
    "constituent_type",
]:
    assert ak.all(
        ak.num(output_array[field], axis=2)
        == reference_lengths
    )

print("Vectorised output validation passed.")
print("First event jet energies:",
      output_array.jet_energy[0])
print("First event multiplicities:",
      output_array.constituent_multiplicity[0])

processing_time += time.perf_counter() - conversion_start

print("\nWriting Parquet shard...")

write_start = time.perf_counter()

# Write the Parquet file
ak.to_parquet(
    output_array,
    OUTPUT_FILE,
    compression=None,
)

write_time = time.perf_counter() - write_start
total_elapsed = time.perf_counter() - total_start

# Print benchmark results
print("\nFinished")
print(f"Input events read:     {total_input_events:,}")
print(f"Selected events saved: {len(output_array):,}")
print(f"Processing time:       {processing_time:.2f} s")
print(f"Parquet write time:    {write_time:.2f} s")
print(f"Total time:            {total_elapsed:.2f} s")
print(f"Output size:           {OUTPUT_FILE.stat().st_size / 1e6:.1f} MB")
print(f"Output type:\n{output_array.type}")

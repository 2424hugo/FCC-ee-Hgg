"""Build reusable, vectorised FCC-ee analysis caches from EDM4hep ROOT files."""
# imports

# tells python to store type annotations for later interpretations instead of when func is defined
# makes annotations safer and more flexable
from __future__ import annotations

# writing parquets
import os
import shutil
import tempfile
import pyarrow.parquet as pq

import json # reading dataset-split manifest
import time # measuring processing time

from dataclasses import dataclass # structed summary object
from pathlib import Path # path handling
from typing import Any # type annotations

# data handling
import awkward as ak
import numpy as np
import uproot

# list of branches I want in each shard
BRANCHES = [
    # Jet level
    "Jet/Jet.energy",
    "Jet/Jet.momentum.x",
    "Jet/Jet.momentum.y",
    "Jet/Jet.momentum.z",
    "Jet/Jet.mass",
    "Jet/Jet.particles_begin",
    "Jet/Jet.particles_end",
    "Jet#2/Jet#2.index",
    # Reconstucted particles
    "ReconstructedParticles/ReconstructedParticles.energy",
    "ReconstructedParticles/ReconstructedParticles.momentum.x",
    "ReconstructedParticles/ReconstructedParticles.momentum.y",
    "ReconstructedParticles/ReconstructedParticles.momentum.z",
    "ReconstructedParticles/ReconstructedParticles.mass",
    "ReconstructedParticles/ReconstructedParticles.charge",
    "ReconstructedParticles/ReconstructedParticles.type",
]

# object to process one ROOTS file
@dataclass(frozen=True) # after creating it is immutable
class BuildSummary:
    """Summary returned after one ROOT file has been processed."""
    # what is stored in the object
    input_file: Path
    output_files: tuple[Path, ...]
    input_events: int
    selected_events: int
    elapsed_seconds: float

# func that finds the particles beloning to a jet
def relation_indices_for_jet(
    relation_indices: ak.Array, # Jet#2
    begins: ak.Array, # particles_begin
    ends: ak.Array, # particles_end
) -> ak.Array:
    """Return the ReconstructedParticles indices belonging to one jet."""

    # awkward array of particle positions
    relation_positions = ak.local_index(relation_indices, axis=1)
    
    # mask for particles in selected jet
    relation_mask = (
        (relation_positions >= begins[:, None])
        & (relation_positions < ends[:, None])
    )

    # return the indices for particles in selected jet
    return relation_indices[relation_mask]

# extracting the reconstructed particle properties for the 2 jets
# called seperately for required variables: E, px, py, mass, ect
def extract_two_jet_constituents(
    reco_values: ak.Array, # variable that is targeted
    relation_indices: ak.Array, # energy, charge, ect
    selected_begins: ak.Array, # selected jet begin
    selected_ends: ak.Array, # selected jet end
) -> ak.Array:
    """Extract one particle field for the leading and subleading jets."""

    # particle indices for leading jet
    leading_indices = relation_indices_for_jet(
        relation_indices,
        selected_begins[:, 0],
        selected_ends[:, 0],
    )
    # particle indices for subleading jet
    subleading_indices = relation_indices_for_jet(
        relation_indices,
        selected_begins[:, 1],
        selected_ends[:, 1],
    )

    # returns a events leading and subleading particle values
    return ak.concatenate(
        [
            reco_values[leading_indices][:, None],
            reco_values[subleading_indices][:, None],
        ],
        axis=1,
    )

# selected events -> analysis records
# ROOTS events into event-level table
def vectorise_selected_events(
    arrays: ak.Array,
    selected_indices: np.ndarray,
    source_file: Path,
    source_event_numbers: np.ndarray,
    event_mass: np.ndarray,
    n_jets: np.ndarray,
    label: int,
) -> ak.Array:
    """Convert selected events from one ROOT batch into analysis records."""

    # only events that passed selection
    selected = arrays[selected_indices]
    # original number of events
    selected_source_events = source_event_numbers[selected_indices]
    # order jet by enery
    # jet energies = [22, 61, 45, 10]
    # jet order = [1, 2, 0, 3]
    # selected = [1, 2]
    jet_order = ak.argsort(
        selected["Jet/Jet.energy"],
        axis=1,
        ascending=False,
    )[:, :2]
    # helper for selecting jet fields
    # each field has shape (num of selected events, 2 jets)
    def selected_jet_field(branch: str) -> ak.Array:
        return selected[branch][jet_order]

    # get constituent relationship ranges
    jet_energy = selected_jet_field("Jet/Jet.energy")
    jet_px = selected_jet_field("Jet/Jet.momentum.x")
    jet_py = selected_jet_field("Jet/Jet.momentum.y")
    jet_pz = selected_jet_field("Jet/Jet.momentum.z")
    jet_mass = selected_jet_field("Jet/Jet.mass")

    selected_begins = selected_jet_field("Jet/Jet.particles_begin")
    selected_ends = selected_jet_field("Jet/Jet.particles_end")
    relation_indices = selected["Jet#2/Jet#2.index"]

    # map particle names to ROOT branches
    reco_branches = {
        "energy": "ReconstructedParticles/ReconstructedParticles.energy",
        "px": "ReconstructedParticles/ReconstructedParticles.momentum.x",
        "py": "ReconstructedParticles/ReconstructedParticles.momentum.y",
        "pz": "ReconstructedParticles/ReconstructedParticles.momentum.z",
        "mass": "ReconstructedParticles/ReconstructedParticles.mass",
        "charge": "ReconstructedParticles/ReconstructedParticles.charge",
        "type": "ReconstructedParticles/ReconstructedParticles.type",
    }

    # calculate constituents[] for all recon particle variables
    constituents = {
        name: extract_two_jet_constituents(
            selected[branch],
            relation_indices,
            selected_begins,
            selected_ends,
        )
        for name, branch in reco_branches.items()
    }

    n_selected = len(selected_indices)

    # build the event records
    return ak.zip(
        {
            "source_file": ak.Array([str(source_file)] * n_selected),
            "source_event": selected_source_events,
            "label": np.full(n_selected, label, dtype=np.int8),
            "event_invariant_mass": event_mass[selected_indices],
            "n_jets_original": n_jets[selected_indices],
            "jet_energy": jet_energy,
            "jet_px": jet_px,
            "jet_py": jet_py,
            "jet_pz": jet_pz,
            "jet_mass": jet_mass,
            "constituent_energy": constituents["energy"],
            "constituent_px": constituents["px"],
            "constituent_py": constituents["py"],
            "constituent_pz": constituents["pz"],
            "constituent_mass": constituents["mass"],
            "constituent_charge": constituents["charge"],
            "constituent_type": constituents["type"],
            "constituent_multiplicity": ak.num(
                constituents["energy"],
                axis=2,
            ),
        },
        depth_limit=1,
    )

# validating an output shard
# ran before writing it out, checks key conditions
def validate_shard(
    shard: ak.Array,
    *,
    label: int,
    event_mass_min: float,
) -> None:
    """Raise AssertionError if a completed output shard is inconsistent."""

    assert len(shard) > 0 # check if empty
    assert ak.all(shard.label == label) # check correct labels
    assert ak.all(shard.event_invariant_mass > event_mass_min) # events pass mass selection
    assert ak.all(ak.num(shard.jet_energy, axis=1) == 2) # exactly 2 jets stored
    assert ak.all(ak.num(shard.constituent_energy, axis=1) == 2) # both jects have recon particles

    reference_lengths = ak.num(shard.constituent_energy, axis=2) # find number of constituent particles
    assert ak.all(reference_lengths == shard.constituent_multiplicity) # equal to multiplicity

    # check length of all recon particles variables
    for field in [
        "constituent_px",
        "constituent_py",
        "constituent_pz",
        "constituent_mass",
        "constituent_charge",
        "constituent_type",
    ]:
        assert ak.all(ak.num(shard[field], axis=2) == reference_lengths)

# main file processing func
def build_file_cache(
    input_file: str | Path, # which file to process
    output_dir: str | Path, # dir to store processed file
    label: int, # 1 for sig, 0 for bkg
    *,
    root_batch_size: int = 10_000, # how many ROOT events to read at once
    events_per_shard: int = 50_000, # events per a shard
    event_mass_min: float = 120.0, # event mass threshold
    output_prefix: str | None = None, # beginning of output name
    compression: str | None = None, # make sure compression is None
    tree_name: str = "events", # select events in ROOT file
) -> BuildSummary:
    """
    Process one complete ROOT file and write all selected events to Parquet.

    Parameters
    ----------
    input_file
        EDM4hep ROOT file to process.
    output_dir
        Directory in which Parquet shards will be written.
    label
        Class label: normally 1 for signal and 0 for background.
    root_batch_size
        Number of ROOT events read at a time.
    events_per_shard
        Maximum selected events written to each Parquet shard.
    event_mass_min
        Require jet-based event invariant mass to exceed this value in GeV.
    output_prefix
        Prefix for shard names. Defaults to the ROOT file stem, preventing
        different source files from overwriting one another.
    compression
        Awkward Parquet compression setting. Use None where zstd is unavailable.
    tree_name
        Name of the ROOT TTree.
    """

    # checking for invalid class
    if label not in (0, 1):
        raise ValueError("label must be 0 (background) or 1 (signal)")
    # batch and shard size must be positive
    if root_batch_size <= 0 or events_per_shard <= 0:
        raise ValueError("batch and shard sizes must be positive")
    # convert into Path objests
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # create dir if required
    # if no prefix given use ROOT filename
    prefix = output_prefix or input_file.stem
    # check for overlap
    existing_outputs = sorted(output_dir.glob(f"{prefix}_part_*.parquet"))
    if existing_outputs:
        raise FileExistsError(
            f"{len(existing_outputs)} output shard(s) already use prefix "
            f"{prefix!r} in {output_dir}. Remove or rename them explicitly "
            "before rebuilding this source file."
        )
    # initialising counters and the buffer
    total_start = time.perf_counter()
    total_input_events = 0 # how many events read
    total_selected_events = 0 # how many events passed selection
    shard_number = 0 # how many shards have been written
    pending_batches: list[ak.Array] = [] # which output shard is being written
    pending_events = 0 # selected events waiting to be written
    output_files: list[Path] = [] # all output file names

    # writing events held in memory
    def write_pending_shard(n_events: int) -> None:
        """Write the first n_events in the pending buffer and retain overflow."""
        # modifies variables in surrounding build_file_cache
        nonlocal pending_batches, pending_events, shard_number

        # combine the pending batches into one awkward array
        combined = ak.concatenate(pending_batches, axis=0)
        shard = combined[:n_events] # what made it into the shard
        remainder = combined[n_events:] # what was overflow, for next shard

        # validate the shard
        validate_shard(
            shard,
            label=label,
            event_mass_min=event_mass_min,
        )
        
        output_file = output_dir / f"{prefix}_part_{shard_number:04d}.parquet" # construct file output name
        
        output_file = Path(output_file) # turn into Path object
        # temporary filename, to prevent incomplete or corrupted parquet files
        eos_temporary = output_file.with_name(
            f"{output_file.name}.incomplete.{os.getpid()}"
        )

        # write locally first
        with tempfile.TemporaryDirectory(
            prefix="fcc_parquet_",
            dir="/tmp",
        ) as temporary_directory:
            local_file = Path(temporary_directory) / output_file.name

            # Write the complete Parquet file to local LXPLUS storage.
            ak.to_parquet(
                shard,
                local_file,
                compression=compression,
            )

            # Check that the local Parquet footer is readable.
            pq.ParquetFile(local_file)

            try:
                # Transfer the completed file to a temporary EOS filename.
                shutil.copyfile(local_file, eos_temporary)

                # Confirm that the copy on EOS is complete.
                pq.ParquetFile(eos_temporary)

                # Publish it under the final name atomically.
                # only after validation so no other script should see corrupted file
                os.replace(eos_temporary, output_file)

                # Final validation using the published filename.
                pq.ParquetFile(output_file)

            except Exception:
                # Remove only this run's incomplete temporary copy.
                if eos_temporary.exists():
                    eos_temporary.unlink()
                raise
        # updata outputed file names
        output_files.append(output_file)
        # declare what has been saved and how many events
        print(
            f"Wrote {output_file} "
            f"({len(shard):,} selected events)"
        )

        shard_number += 1 # update shard number 
        pending_batches = [remainder] if len(remainder) else [] # update pending batches
        pending_events = len(remainder) # update pending events

    # Onto next file
    print(f"Opening: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # reading ROOT file
    with uproot.open(input_file) as root_file:
        tree = root_file[tree_name] # file opened and events selected
        # only read listed branches, 10_000 events at a time, output is ak array
        for batch_number, arrays in enumerate(
            tree.iterate(
                expressions=BRANCHES,
                step_size=root_batch_size,
                library="ak",
            )
        ):
            batch_size = len(arrays) # number of events in current batch
            source_event_numbers = np.arange(
                total_input_events,
                total_input_events + batch_size,
            ) # the original entry numbers in the ROOT event tree
            total_input_events += batch_size
            # calculating the event invariant mass
            event_energy = ak.sum(arrays["Jet/Jet.energy"], axis=1)
            event_px = ak.sum(arrays["Jet/Jet.momentum.x"], axis=1)
            event_py = ak.sum(arrays["Jet/Jet.momentum.y"], axis=1)
            event_pz = ak.sum(arrays["Jet/Jet.momentum.z"], axis=1)

            event_mass_squared = (
                event_energy**2
                - event_px**2
                - event_py**2
                - event_pz**2
            )
            event_mass = np.sqrt(
                np.maximum(ak.to_numpy(event_mass_squared), 0.0)
            )
            # counting number of jets
            n_jets = ak.to_numpy(
                ak.num(arrays["Jet/Jet.energy"], axis=1)
            )
            # apply num of jets and min energy selection
            selected_indices = np.flatnonzero(
                (n_jets >= 2) & (event_mass > event_mass_min)
            )
            # if at least one event passed
            if len(selected_indices):
                vector_start = time.perf_counter() # start timer
                # create event records
                batch_array = vectorise_selected_events(
                    arrays=arrays,
                    selected_indices=selected_indices,
                    source_file=input_file,
                    source_event_numbers=source_event_numbers,
                    event_mass=event_mass,
                    n_jets=n_jets,
                    label=label,
                )
                # add the event records to the pending buffer
                pending_batches.append(batch_array)
                pending_events += len(batch_array)
                total_selected_events += len(batch_array)
                # whenever 50_000 events or more are buffered, write one complete event shard
                while pending_events >= events_per_shard:
                    write_pending_shard(events_per_shard)

                vector_seconds = time.perf_counter() - vector_start
            else:
                vector_seconds = 0.0

            elapsed = time.perf_counter() - total_start
            print(
                f"Batch {batch_number:02d} | "
                f"input: {total_input_events:,} | "
                f"selected: {total_selected_events:,} | "
                f"vector: {vector_seconds:.3f} s | "
                f"elapsed: {elapsed:.2f} s"
            )
    # if final shard is less than 50_000, still write
    if pending_events:
        write_pending_shard(pending_events)
    # return a summary
    elapsed_seconds = time.perf_counter() - total_start
    print(
        f"Finished {input_file.name}: "
        f"{total_selected_events:,}/{total_input_events:,} events selected "
        f"into {len(output_files)} shard(s) in {elapsed_seconds:.2f} s"
    )

    return BuildSummary(
        input_file=input_file,
        output_files=tuple(output_files),
        input_events=total_input_events,
        selected_events=total_selected_events,
        elapsed_seconds=elapsed_seconds,
    )

# loading in the split data set
def load_splits(path: str | Path = "cache/dataset_splits.json") -> dict[str, Any]:
    """Load the frozen signal/background train/validation/test split."""
    # read in the json split manifest
    with Path(path).open() as split_file:
        return json.load(split_file)

# run if file is run directly
def main() -> None:
    """Run the current one-signal/one-background training-file test."""
    # load the split
    splits = load_splits()
    # only on the first training files
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
        build_file_cache(
            input_file=config["input_file"],
            output_dir=Path("cache/analysis_dataset") / sample / "train",
            label=config["label"],
        )

if __name__ == "__main__":
    main()

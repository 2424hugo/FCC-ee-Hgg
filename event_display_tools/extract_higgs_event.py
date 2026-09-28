#!/usr/bin/env python3
"""Export one FCC-ee EDM4hep/PODIO ROOT event for an event display.

Requires Python >=3.9, uproot 5 and awkward 2; CERN ROOT is not required.
Install: python -m pip install "uproot>=5,<6" "awkward>=2,<3"

Run on an ORIGINAL SIGNAL ROOT file, not the 22-feature analysis table:
    python extract_higgs_event.py signal.root
    python extract_higgs_event.py signal.root --event 42 --output event_42.json

Without --event, select the first event with mass >120 GeV and >=2 jets
among the first 1000 entries. Mass uses ALL ReconstructedParticles.
--event is a zero-based TTree entry number, not a generator event ID, and
bypasses selection. The input file is opened read-only.

The JSON retains original branch names, array order, PODIO ObjectID indices,
collection IDs, relation offsets, track states, hits and truth if readable.
No tracks, detector geometry or hits are invented. Values retain input units.
Unresolvable optional branches are reported; NaN/Inf become JSON null with
their locations recorded. A small sample of file metadata is also exported.

Jet.particles_begin/end index a RELATION array, not necessarily the
ReconstructedParticles array. This script deliberately preserves those
relations unchanged rather than guessing the collection mapping.

API reference: https://uproot.readthedocs.io/en/stable/basic.html
"""

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import awkward as ak
    import uproot
except ImportError as exc:
    raise SystemExit(
        'Missing dependency. Run: python -m pip install "uproot>=5,<6" "awkward>=2,<3"'
    ) from exc


RECO = "ReconstructedParticles/ReconstructedParticles."
JET = "Jet/Jet."
KINEMATIC_BRANCHES = [RECO + k for k in (
    "energy", "momentum.x", "momentum.y", "momentum.z"
)] + [JET + "energy"]


def to_builtin(value):
    """Handle arrays and model objects without converting numerical data to text."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, (ak.Array, ak.Record)):
        return to_builtin(ak.to_list(value))
    if hasattr(value, "tolist"):
        return to_builtin(value.tolist())
    if hasattr(value, "tojson"):
        return to_builtin(value.tojson())
    raise TypeError(f"Cannot encode {type(value).__name__}")


def read_entries(branch, start, stop):
    """Use Awkward first, then NumPy for otherwise unsupported object branches."""
    errors = []
    for library in ("ak", "np"):
        try:
            return to_builtin(branch.array(
                entry_start=start, entry_stop=stop, library=library
            ))
        except Exception as exc:
            errors.append(f"{library}: {type(exc).__name__}: {str(exc)[:500]}")
    raise RuntimeError(" | ".join(errors))


def leaf_branches(tree):
    """Skip split container branches, retaining every terminal data branch."""
    return {
        name: branch for name, branch in tree.items(recursive=True, full_paths=True)
        if not branch.keys(recursive=False)
    }


def event_summary(data):
    vectors = [data[name] for name in KINEMATIC_BRANCHES[:4]]
    if any(not isinstance(v, list) for v in vectors):
        raise ValueError("Expected per-event arrays for reconstructed-particle momenta.")
    if len({len(v) for v in vectors}) != 1:
        raise ValueError("Reconstructed-particle energy/momentum array lengths disagree.")
    finite = all(isinstance(x, (int, float)) and math.isfinite(x)
                 for v in vectors for x in v)
    mass = None
    totals = None
    mass2 = None
    if finite:
        totals = [math.fsum(v) for v in vectors]
        energy, px, py, pz = totals
        mass2 = energy**2 - px**2 - py**2 - pz**2
        # Only clamp tiny negative values attributable to rounding.
        tolerance = 1e-10 * max(energy**2, px**2 + py**2 + pz**2, 1.0)
        if mass2 >= -tolerance:
            mass = math.sqrt(max(mass2, 0.0))

    jet_energies = data[JET + "energy"]
    if not isinstance(jet_energies, list):
        raise ValueError("Expected a per-event array for Jet.energy.")
    finite_jets = all(isinstance(e, (int, float)) and math.isfinite(e)
                      for e in jet_energies)
    order = sorted(range(len(jet_energies)), key=lambda i: jet_energies[i],
                   reverse=True) if finite_jets else []
    leading = []
    for i in order[:2]:
        row = {"original_jet_index": i, "energy_GeV": jet_energies[i]}
        for field in ("mass", "momentum.x", "momentum.y", "momentum.z"):
            values = data.get(JET + field)
            if isinstance(values, list) and len(values) == len(jet_energies):
                row[field] = values[i]
        leading.append(row)

    return {
        "n_reconstructed_particles": len(vectors[0]),
        "n_jets": len(jet_energies),
        "finite_particle_kinematics": finite,
        "finite_jet_energies": finite_jets,
        "event_mass_GeV": mass,
        "event_mass_squared_GeV2": mass2,
        "summed_reconstructed_four_vector_E_px_py_pz_GeV": totals,
        "leading_two_jets": leading,
    }


def select_entry(tree, args):
    n_entries = int(tree.num_entries)
    if args.event is not None:
        if not 0 <= args.event < n_entries:
            raise ValueError(f"--event must be between 0 and {n_entries - 1}.")
        return args.event, {"mode": "explicit_entry", "cuts_applied": False}

    stop = min(n_entries, args.start + args.max_scan)
    if args.start >= n_entries:
        raise ValueError(f"--start is outside this tree ({n_entries} entries).")
    print(f"Looking for mass > {args.min_mass:g} GeV and >= {args.min_jets} jets "
          f"in entries {args.start} to {stop - 1}...", flush=True)
    for start in range(args.start, stop, 128):
        end = min(start + 128, stop)
        chunk = {name: read_entries(tree[name], start, end)
                 for name in KINEMATIC_BRANCHES}
        for offset in range(end - start):
            data = {name: rows[offset] for name, rows in chunk.items()}
            info = event_summary(data)
            mass = info["event_mass_GeV"]
            if (mass is not None and mass > args.min_mass
                    and info["n_jets"] >= args.min_jets
                    and info["finite_jet_energies"]):
                return start + offset, {
                    "mode": "first_passing_entry", "cuts_applied": True,
                    "event_mass_GeV_strictly_greater_than": args.min_mass,
                    "n_jets_at_least": args.min_jets,
                    "scan_start": args.start,
                    "entries_examined": start + offset - args.start + 1,
                    "mass_definition": "Invariant mass of all ReconstructedParticles",
                }
    raise ValueError(
        f"No passing event found in entries {args.start} to {stop - 1}. "
        "Increase --max-scan, change --start, or specify --event N."
    )


def extract_branches(tree, start, stop, single_event=False):
    data, errors, schema = {}, {}, {}
    for name, branch in leaf_branches(tree).items():
        schema[name] = str(getattr(branch, "typename", "unknown"))
        try:
            rows = read_entries(branch, start, stop)
            if len(rows) != stop - start:
                raise ValueError(f"Expected {stop - start} entries, got {len(rows)}")
            data[name] = rows[0] if single_event else rows
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {str(exc)[:1200]}"
    return data, errors, schema


def extract_metadata(root_file):
    result = {}
    for name in root_file.keys(cycle=False, recursive=False):
        if "metadata" not in name.lower():
            continue
        try:
            obj = root_file[name]
            if hasattr(obj, "num_entries") and hasattr(obj, "items"):
                count = int(obj.num_entries)
                stop = min(count, 10)
                data, errors, schema = extract_branches(obj, 0, stop)
                result[name] = {
                    "total_entries": count, "exported_entries": stop,
                    "truncated": stop < count, "branches": data,
                    "unreadable_branches": errors, "branch_types": schema,
                }
            else:
                result[name] = {"value": to_builtin(obj)}
        except Exception as exc:
            result[name] = {"error": f"{type(exc).__name__}: {str(exc)[:1200]}"}
    return result


def clean_nonfinite(value, replacements, path="$"):
    if isinstance(value, float) and not math.isfinite(value):
        replacements.append({"path": path, "original_value": str(value)})
        return None
    if isinstance(value, list):
        return [clean_nonfinite(v, replacements, f"{path}[{i}]")
                for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {k: clean_nonfinite(v, replacements, f"{path}[{json.dumps(k)}]")
                for k, v in value.items()}
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root_file", help="Local original signal ROOT file, or supported URL")
    parser.add_argument("--tree", default="events", help="Tree name (default: events)")
    parser.add_argument("--event", type=int, help="Explicit zero-based entry; bypasses cuts")
    parser.add_argument("--start", type=int, default=0, help="First entry to scan (default: 0)")
    parser.add_argument("--max-scan", type=int, default=1000, help="Maximum scan length")
    parser.add_argument("--min-mass", type=float, default=120.0, help="Strict mass cut in GeV")
    parser.add_argument("--min-jets", type=int, default=2, help="Minimum number of jets")
    parser.add_argument("--output", default="higgs_event.json", help="Output JSON filename")
    parser.add_argument("--force", action="store_true", help="Replace an existing output JSON")
    args = parser.parse_args(argv)
    if (args.start < 0 or args.max_scan <= 0 or args.min_jets < 0
            or not math.isfinite(args.min_mass) or args.min_mass < 0):
        parser.error("Scan bounds and selection thresholds must be valid non-negative numbers.")

    output = Path(args.output).expanduser()
    if output.suffix.lower() != ".json":
        parser.error("--output must have the .json extension.")
    if output.exists() and not args.force:
        parser.error(f"Output already exists: {output}. Choose another name or use --force.")
    source = args.root_file if "://" in args.root_file else Path(args.root_file).expanduser()
    if isinstance(source, Path) and source.resolve() == output.resolve():
        parser.error("The output must be a different file from the input.")

    with uproot.open(source) as root_file:
        if args.tree not in root_file:
            raise ValueError(f"Tree {args.tree!r} not found. File keys: {root_file.keys()}")
        tree = root_file[args.tree]
        missing = [name for name in KINEMATIC_BRANCHES if name not in tree]
        if missing:
            raise ValueError("This script expects the original EDM4hep event schema. "
                             "Missing branches: " + ", ".join(missing))
        entry, selection = select_entry(tree, args)
        print(f"Extracting entry {entry} (zero-based)...", flush=True)
        data, errors, schema = extract_branches(tree, entry, entry + 1, single_event=True)
        required_errors = [name for name in KINEMATIC_BRANCHES if name not in data]
        if required_errors:
            raise RuntimeError("Required branches could not be exported: " +
                               json.dumps({name: errors.get(name, "Missing")
                                           for name in required_errors}, indent=2))
        summary = event_summary(data)
        summary["passes_requested_mass_and_jet_cuts"] = (
            summary["event_mass_GeV"] is not None
            and summary["event_mass_GeV"] > args.min_mass
            and summary["n_jets"] >= args.min_jets
        )
        summary["collection_counts"] = {
            label: len(data[name]) if isinstance(data.get(name), list) else None
            for label, name in {
                "track_states": "EFlowTrack_1/EFlowTrack_1.phi",
                "photon_clusters": "EFlowPhoton/EFlowPhoton.energy",
                "neutral_hadron_clusters": "EFlowNeutralHadron/EFlowNeutralHadron.energy",
                "tracker_hits": "TrackerHits/TrackerHits.cellID",
                "calorimeter_hits": "CalorimeterHits/CalorimeterHits.cellID",
                "generator_particles": "Particle/Particle.PDG",
            }.items()
        }
        payload = {
            "format": "fcc-ee-event-display-extract-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "filename": args.root_file.replace("\\", "/").rsplit("/", 1)[-1].split("?")[0],
                "root_file_uuid": str(root_file.file.uuid),
                "tree": args.tree,
                "tree_object_path": tree.object_path,
                "entry_index_zero_based": entry,
                "tree_entries": int(tree.num_entries),
                "root_keys": root_file.keys(cycle=True, recursive=False),
            },
            "software": {"python": platform.python_version(),
                         "uproot": uproot.__version__, "awkward": ak.__version__},
            "selection": selection,
            "summary": summary,
            "interpretation_notes": [
                "Input values and ordering retained; no geometry or trajectories generated.",
                "Summary assumes energy/momentum in GeV, as in this FCC-ee analysis.",
                "Spatial, time and magnetic-field values retain source units without conversion.",
                "PODIO relation offsets index ObjectID arrays; resolve index AND collectionID.",
                "A listed empty hit collection is not evidence of simulated detector hits.",
                "Use a signal sample: this extractor does not classify or truth-select H->gg.",
            ],
            "event_branches": data,
            "branch_types": schema,
            "available_branch_names": tree.keys(recursive=True, full_paths=True),
            "unreadable_event_branches": errors,
            "file_metadata": extract_metadata(root_file),
        }

    replacements = []
    payload = clean_nonfinite(payload, replacements)
    payload["nonfinite_replacements"] = replacements
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if args.force else "x", encoding="utf-8") as handle:
        handle.write(text)
    mass = summary["event_mass_GeV"]
    mass_label = f"{mass:.4f} GeV" if mass is not None else "undefined (check kinematics)"
    print(f"Saved: {output.resolve()}")
    print(f"Entry: {entry}; event mass: {mass_label}; jets: {summary['n_jets']}; "
          f"reconstructed particles: {summary['n_reconstructed_particles']}")
    print(f"Branches exported: {len(data)}; unreadable: {len(errors)}; "
          f"JSON size: {output.stat().st_size / 1024:.1f} KiB")
    if errors:
        print("Optional branch read errors are recorded in the JSON; keep them in the upload.")
    print("Upload this JSON file to continue building the event display.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

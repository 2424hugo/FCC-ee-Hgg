#!/usr/bin/env python3
"""Build 10 signal and 10 background event displays on CERN EOS.

Keep extract_higgs_event.py and plot_higgs_event_3d.py beside this script.
Install: python -m pip install "uproot>=5,<6" "awkward>=2,<3" numpy "plotly>=6,<7" "matplotlib>=3.6"
Run on your CERN Jupyter/LXPLUS machine: python create_event_gallery.py

The defaults point to your Winter 2023 IDEA Hgg and qq samples. Both classes
must pass mass >120 GeV and >=2 reconstructed jets. Files are shuffled with a
fixed seed; take the first passing event in a randomly positioned window in
each file. Prefer a different file for each event, cycling only when needed.
These are reproducible illustrations, NOT a uniform sample of all accepted
events, a weighted physics sample, or a classifier-selected comparison.

Outputs in event_gallery/: index.html; signal/ and background/ each containing
10 HTML, PNG and JSON files; manifest.json with sources, entries and settings;
assets/plotly.min.js. The complete folder is also saved as event_gallery.zip.
Keep the folder structure intact: all HTML files share the local JS asset.

If interrupted, use the same command with --resume. Completed figures are
retained and incomplete ones are rebuilt from the exported JSON where possible.
"""

import argparse
import hashlib
import html
import json
import math
import platform
import sys
import types
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import uproot
    import awkward as ak
    import plotly
    from plotly.offline import get_plotlyjs
    import extract_higgs_event as extract
    import plot_higgs_event_3d as display
except ImportError as exc:
    raise SystemExit(
        'Keep all three scripts together and install dependencies:\n'
        'python -m pip install "uproot>=5,<6" "awkward>=2,<3" numpy "plotly>=6,<7" "matplotlib>=3.6"'
    ) from exc

EOS = Path("/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA")
PROCESSES = {"signal": "wzp6_ee_Hgg_ecm125", "background": "wzp6_ee_qq_ecm125"}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def script_hashes():
    folder = Path(__file__).resolve().parent
    return {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
            for name in ("create_event_gallery.py", "extract_higgs_event.py", "plot_higgs_event_3d.py")}


def root_files(directory):
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise ValueError(
            f"ROOT directory not accessible: {directory}\n"
            "Run on your CERN machine with EOS mounted, or set --signal-dir and --background-dir "
            "to directories containing downloaded original ROOT files."
        )
    files = sorted(p.resolve() for p in directory.glob("*.root") if p.is_file())
    if not files:
        raise ValueError(f"No .root files found directly inside {directory}.")
    return files


def export_selected(root_path, args, random_seed, already_used):
    """Read only kinematics while selecting; export all readable leaves for one event."""
    rng = np.random.default_rng(random_seed)
    with uproot.open(root_path) as root_file:
        if args.tree not in root_file:
            raise ValueError(f"No {args.tree!r} tree in {root_path.name}")
        tree = root_file[args.tree]
        n = int(tree.num_entries)
        if n == 0:
            raise ValueError("Empty event tree")
        missing = [key for key in extract.KINEMATIC_BRANCHES if key not in tree]
        if missing:
            raise ValueError("Missing selection branches: " + ", ".join(missing))
        uuid = str(root_file.file.uuid)
        # Multiple attempts in the same file allow recovery when a short window
        # contains no passing events or an earlier run selected the same entry.
        for _ in range(8):
            start = int(rng.integers(0, n))
            options = types.SimpleNamespace(event=None, start=start, max_scan=args.scan_window,
                                            min_mass=args.min_mass, min_jets=args.min_jets)
            try:
                entry, selection = extract.select_entry(tree, options)
            except ValueError as exc:
                if "No passing event found" in str(exc):
                    continue
                raise
            key = (uuid, entry)
            if key in already_used:
                continue
            data, errors, schema = extract.extract_branches(tree, entry, entry+1, single_event=True)
            absent = [name for name in extract.KINEMATIC_BRANCHES if name not in data]
            if absent:
                raise ValueError("Cannot export required branches: " + ", ".join(absent))
            info = extract.event_summary(data)
            if not (info["event_mass_GeV"] is not None and info["event_mass_GeV"] > args.min_mass
                    and info["n_jets"] >= args.min_jets):
                raise ValueError("Exported event failed the selection recheck")
            payload = {
                "format": "fcc-ee-event-display-extract-v1", "created_utc": now(),
                "source": {"filename": root_path.name, "root_path": str(root_path),
                           "root_file_uuid": uuid, "tree": args.tree, "tree_object_path": tree.object_path,
                           "entry_index_zero_based": entry, "tree_entries": n},
                "software": {"python": platform.python_version(), "uproot": uproot.__version__,
                             "awkward": ak.__version__, "plotly": plotly.__version__},
                "selection": dict(selection, random_window_seed=int(random_seed)),
                "summary": info, "event_branches": data, "branch_types": schema,
                "available_branch_names": tree.keys(recursive=True, full_paths=True),
                "unreadable_event_branches": errors, "file_metadata": extract.extract_metadata(root_file),
                "interpretation_notes": ["Original numerical values, ordering and ObjectID relations retained.",
                                         "Summary momenta and energies are in GeV; spatial values remain in source units."]}
            replacements = []
            payload = extract.clean_nonfinite(payload, replacements)
            payload["nonfinite_replacements"] = replacements
            return payload
    raise ValueError("No new passing event found in the attempted random windows")


def render_record(output, record, args):
    payload = json.loads((output/record["json"]).read_text(encoding="utf-8"))
    options = types.SimpleNamespace(
        bz=None, position_unit="mm", colour_by="auto", track_mode="states", min_energy=args.min_energy,
        tracker_radius=2.0, tracker_half_length=2.0, calo_radius=2.35, calo_half_length=2.5,
        no_clusters=False, hits=args.hits, max_points=10000, max_turns=1.0, max_path=12.0, roll=55.0)
    scene = display.make_scene(payload, options)
    process_title = "H → gg signal" if record["sample"] == "signal" else "q q̄ continuum background"
    title = f"FCC-ee simulation · {process_title} · √s = 125 GeV"
    figure = display.plotly_figure(scene, title)
    display.save_html(figure, scene, title, output/record["html"], plotly_js="../assets/plotly.min.js")
    display.save_png(scene, title, output/record["png"])
    record["rendering"] = {
        "saved_track_states_used": scene["states_used"], "momentum_fallback_paths": scene["charged_fallback"],
        "verified_jet_membership": scene["membership_verified"], "display_capped_paths": scene["limited_count"],
        "notes": scene["notes"]}
    record["status"] = "complete"


def gallery_html(manifest):
    """Only thumbnails are loaded in the gallery; 3D views open individually."""
    cards = []
    records = manifest["events"]
    complete = [record for record in records if record["status"] == "complete"]
    n_signal = sum(r["sample"] == "signal" for r in complete)
    n_background = sum(r["sample"] == "background" for r in complete)
    for record in complete:
        sample = record["sample"]
        summary = record["summary"]
        title = f"{sample.capitalize()} {record['number']:02d}"
        cards.append(
            f'<article class="card {sample}" data-sample="{sample}">'
            f'<a href="{record["html"]}" target="_blank" rel="noopener">'
            f'<img loading="lazy" src="{record["png"]}" alt="{title} event display"></a>'
            '<div class="body">'
            f'<span class="tag">{sample.upper()}</span><h2>{title}</h2>'
            f'<p>{summary["event_mass_GeV"]:.2f} GeV &nbsp; · &nbsp; {summary["n_jets"]} jets'
            f'<br>{summary["n_reconstructed_particles"]} reconstructed particles</p>'
            f'<div class="links"><a href="{record["html"]}" target="_blank" rel="noopener">Open 3D display ↗</a>'
            f'<a href="{record["png"]}" download>PNG</a><a href="{record["json"]}" download>JSON</a></div>'
            f'<details><summary>Event source</summary><p>{html.escape(record["source"]["filename"])}'
            f'<br>Entry {record["source"]["entry_index_zero_based"]} (zero-based)</p></details>'
            '</div></article>')
    config = manifest["config"]
    status = "Complete" if manifest.get("complete") else "In progress"
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FCC-ee event gallery</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#070d17;color:#eef3fa;font:16px system-ui,sans-serif}}
main{{max-width:1600px;margin:auto;padding:48px 36px}}h1{{font-size:38px;margin:0 0 12px}}header p{{color:#a8bbcf;line-height:1.65;max-width:900px}}
.status{{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:#80cedb}}nav{{display:flex;gap:12px;margin:28px 0}}
button{{border:1px solid #34425a;background:transparent;color:inherit;padding:10px 18px;border-radius:20px;cursor:pointer}}
button.active{{background:#e5eefb;color:#09101b;border-color:#e5eefb}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(265px,1fr));gap:22px}}
.card{{border:1px solid #243047;border-radius:14px;overflow:hidden;background:#0d1523}}.card[hidden]{{display:none}}
.card img{{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#050b15}}.body{{padding:18px}}
.tag{{font-size:11px;letter-spacing:.13em;color:#53d6e9}}.background .tag{{color:#ffaa77}}h2{{font-size:20px;margin:8px 0 12px}}
.body p{{color:#a8bbcf;font-size:14px;line-height:1.6}}a{{color:#c8eafa;text-decoration:none}}a:hover{{text-decoration:underline}}
.links{{display:flex;gap:16px;font-size:13px;margin:18px 0}}details{{font-size:12px;color:#91a8c4}}details p{{overflow-wrap:anywhere}}
footer{{border-top:1px solid #243047;margin-top:32px;padding-top:20px;color:#a8bbcf;font-size:13px;line-height:1.7}}
@media(max-width:600px){{main{{padding:28px 18px}}h1{{font-size:28px}}nav{{flex-wrap:wrap}}}}
</style></head><body><main><header><div class="status">{status} · {len(complete)} displays</div>
<h1>FCC-ee event gallery</h1><p>H → gg signal and q q̄ continuum background at √s = 125 GeV.
Both samples require reconstructed event mass &gt; {config['min_mass']:g} GeV and at least {config['min_jets']} jets.
Open an individual display to rotate, zoom and inspect particles.</p></header>
<nav aria-label="Filter events"><button class="active" data-filter="all" aria-pressed="true">All ({len(complete)})</button>
<button data-filter="signal" aria-pressed="false">Signal ({n_signal})</button><button data-filter="background" aria-pressed="false">Background ({n_background})</button></nav>
<section class="grid">{''.join(cards)}</section><footer>
Colours show verified reconstructed jet membership where available. The geometry is schematic; curves use saved track states where linked.
The displays do not add material interactions or detailed detector showers.<br>
Seed {config['seed']}. These are illustrative events from random file windows, not a weighted or uniform sample of all accepted events.
No event was selected using its appearance or a classifier score.<br><a href="manifest.json">Selection and source manifest</a>.
Keep the assets folder with these files for offline viewing.</footer></main>
<script>document.querySelectorAll('button[data-filter]').forEach(button=>button.addEventListener('click',()=>{{
 document.querySelectorAll('button[data-filter]').forEach(b=>{{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',b===button?'true':'false')}});
 document.querySelectorAll('.card').forEach(card=>{{card.hidden=button.dataset.filter!=='all'&&card.dataset.sample!==button.dataset.filter}});
}}));</script></body></html>'''


def checkpoint(output, manifest):
    manifest["updated_utc"] = now()
    write_json(output/"manifest.json", manifest)
    (output/"index.html").write_text(gallery_html(manifest), encoding="utf-8")


def make_archive(output):
    archive = output.parent/(output.name + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                bundle.write(path, arcname=str(Path(output.name)/path.relative_to(output)))
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal-dir", default=str(EOS/PROCESSES["signal"]))
    parser.add_argument("--background-dir", default=str(EOS/PROCESSES["background"]))
    parser.add_argument("--per-class", type=int, default=10, help="Number of distinct events per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tree", default="events")
    parser.add_argument("--min-mass", type=float, default=120.0)
    parser.add_argument("--min-jets", type=int, default=2)
    parser.add_argument("--scan-window", type=int, default=1024, help="Maximum entries examined per random window")
    parser.add_argument("--min-energy", type=float, default=0.0, help="Display-only particle energy cut [GeV]")
    parser.add_argument("--hits", action="store_true", help="Also show stored tracker reference points")
    parser.add_argument("--output", default="event_gallery", help="New output folder")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run with identical settings")
    parser.add_argument("--no-zip", action="store_true", help="Skip the final gallery ZIP archive")
    args = parser.parse_args(argv)
    if args.per_class < 1 or args.seed < 0 or args.scan_window < 1 or args.min_jets < 0:
        parser.error("Counts and scan length must be positive; seed/min-jets must be non-negative.")
    if any(not math.isfinite(x) or x < 0 for x in [args.min_mass, args.min_energy]):
        parser.error("Mass/energy thresholds must be finite and non-negative.")
    # Check Matplotlib before expensive ROOT reads.
    import matplotlib
    output = Path(args.output).expanduser().resolve()
    config = {key: getattr(args, key) for key in ["per_class", "seed", "tree", "min_mass", "min_jets", "scan_window", "min_energy", "hits"]}
    config.update(signal_dir=str(Path(args.signal_dir).expanduser().resolve()),
                  background_dir=str(Path(args.background_dir).expanduser().resolve()))
    if args.resume:
        manifest_path = output/"manifest.json"
        if not manifest_path.is_file():
            parser.error("--resume requires an existing manifest.json in the output folder.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != config or manifest.get("script_sha256") != script_hashes():
            parser.error("Settings or scripts differ from the saved run. Use the original settings/scripts or a new --output folder.")
    else:
        if output.exists() and any(output.iterdir()):
            parser.error("Output folder is not empty. Use --resume or choose a new --output folder.")
        files = {sample: root_files(config[sample+"_dir"]) for sample in PROCESSES}
        if set(files["signal"]) & set(files["background"]):
            parser.error("Signal and background input files overlap. Supply distinct process directories.")
        rng = np.random.default_rng(args.seed)
        manifest = {"format": "fcc-ee-event-gallery-v1", "created_utc": now(), "config": config,
                    "script_sha256": script_hashes(), "software": {"python": platform.python_version(),
                    "uproot": uproot.__version__, "awkward": ak.__version__, "numpy": np.__version__,
                    "plotly": plotly.__version__, "matplotlib": matplotlib.__version__},
                    "file_order": {sample: [str(files[sample][i]) for i in rng.permutation(len(files[sample]))]
                                   for sample in PROCESSES},
                    "attempts": {sample: 0 for sample in PROCESSES}, "events": [], "skipped_attempts": [],
                    "selection_method": "Seeded shuffled files; first passing event in a random contiguous window. Prefer one event per file.",
                    "complete": False}
        output.mkdir(parents=True, exist_ok=True)
    (output/"assets").mkdir(exist_ok=True)
    (output/"assets"/"plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    for sample in PROCESSES:
        (output/sample).mkdir(exist_ok=True)
    checkpoint(output, manifest)
    # Resume previously selected events before drawing more candidates.
    for record in manifest["events"]:
        paths = [output/record[k] for k in ("json", "html", "png")]
        if record["status"] != "complete" or not all(p.is_file() for p in paths):
            print(f"Resuming {record['sample']} {record['number']:02d}...", flush=True)
            render_record(output, record, args)
            checkpoint(output, manifest)
    used = {(r["source"]["root_file_uuid"], r["source"]["entry_index_zero_based"]) for r in manifest["events"]}
    for sample_number, sample in enumerate(PROCESSES):
        while sum(r["sample"] == sample for r in manifest["events"]) < args.per_class:
            attempt = manifest["attempts"][sample]
            if attempt >= max(100, args.per_class*20):
                raise RuntimeError(f"Too many unsuccessful {sample} attempts. See manifest.json; no replacement events were invented.")
            manifest["attempts"][sample] += 1
            file_order = manifest["file_order"][sample]
            root_path = Path(file_order[attempt % len(file_order)])
            number = 1 + sum(r["sample"] == sample for r in manifest["events"])
            print(f"\n[{sample} {number}/{args.per_class}] {root_path.name}", flush=True)
            random_seed = int(np.random.SeedSequence([args.seed, sample_number, attempt]).generate_state(1)[0])
            try:
                payload = export_selected(root_path, args, random_seed, used)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print("Skipping attempt: " + message, flush=True)
                manifest["skipped_attempts"].append({"sample": sample, "attempt": attempt,
                                                      "root_file": str(root_path), "reason": message})
                checkpoint(output, manifest)
                continue
            stem = f"{sample}/{sample}_{number:02d}"
            record = {"sample": sample, "process": PROCESSES[sample], "number": number,
                      "source": payload["source"], "summary": payload["summary"],
                      "json": stem + ".json", "html": stem + ".html", "png": stem + ".png",
                      "status": "selected"}
            write_json(output/record["json"], payload)
            manifest["events"].append(record)
            used.add((payload["source"]["root_file_uuid"], payload["source"]["entry_index_zero_based"]))
            checkpoint(output, manifest)
            render_record(output, record, args)
            checkpoint(output, manifest)
            info = record["summary"]
            print(f"Created {record['html']}: entry {record['source']['entry_index_zero_based']}, "
                  f"mass {info['event_mass_GeV']:.3f} GeV, {info['n_jets']} jets", flush=True)
    manifest["complete"] = True
    checkpoint(output, manifest)
    print(f"\nComplete: {args.per_class} signal + {args.per_class} background displays.")
    print(f"Gallery: {output/'index.html'}")
    if not args.no_zip:
        print(f"Downloadable gallery archive: {make_archive(output)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}\nIf a run was started, rerun the same command with --resume.", file=sys.stderr)
        sys.exit(1)

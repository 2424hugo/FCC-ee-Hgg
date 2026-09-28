#!/usr/bin/env python3
"""Re-render exported FCC-ee events as slide PNGs and portable interactive HTML.

Keep plot_higgs_event_3d.py beside this script. No ROOT/EOS access is needed.
Install: python -m pip install numpy "plotly>=6,<7" "matplotlib>=3.6"
Run: python repair_event_gallery.py event_gallery
One event: python repair_event_gallery.py event_42.json --sample signal

Existing JSON files and event choices are retained. Output goes to a new sibling
folder with suffix _fixed, plus a complete ZIP and a separate PNG-only ZIP.
PNG views are orthographic x-y projections looking from +z, with +x right and
+y up. Geometry is schematic; line length is not energy. All valid jet axes
are retained. The clean PNG has no labels; its assumptions and source are
stored in a same-stem caption.txt file and in the gallery manifest.
"""

import argparse
import base64
import copy
import hashlib
import html
import json
import math
import sys
import types
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch
    import plot_higgs_event_3d as display
except ImportError as exc:
    raise SystemExit(
        'Keep plot_higgs_event_3d.py beside this script and install dependencies:\n'
        'python -m pip install numpy "plotly>=6,<7" "matplotlib>=3.6"'
    ) from exc


def charge_color(path):
    q = path["charge"]
    key = ("Unknown charge" if not np.isfinite(q) else "Positive charge" if q > 0
           else "Negative charge" if q < 0 else "Neutral")
    return display.COLORS[key]


def save_beam_png(scene, output, dpi=240, centred=False):
    """Project the existing 3D trajectories, without changing their coordinates."""
    outer = max(radius for _, radius, _, _ in scene["geometry"])
    # Leave room for jet arrows, which end slightly beyond the calo outline.
    ymax = 1.22 * outer
    xspan = 2 * ymax * 16 / 9
    centre_fraction = 0.5 if centred else 0.69
    fig = plt.figure(figsize=(16, 9), dpi=dpi, facecolor=display.BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=display.BG)
    ax.set_xlim(-centre_fraction * xspan, (1-centre_fraction) * xspan)
    ax.set_ylim(-ymax, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    for radius in outer * np.array([0.05, 0.25, 0.50, 0.75, 0.90]):
        ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor="#223648", lw=0.8, alpha=0.42))
    for phi in np.linspace(0, 2*np.pi, 16, endpoint=False):
        ax.plot([0, outer*np.cos(phi)], [0, outer*np.sin(phi)],
                color="#23354a", lw=0.6, alpha=0.28, zorder=0)
    for _, radius, _, _ in scene["geometry"]:
        ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor="#31495e",
                            lw=1.65 if radius == outer else 1.05, alpha=0.68))
    for path in scene["paths"]:
        points = path["points"]
        ax.plot(points[:, 0], points[:, 1], color=charge_color(path),
                lw=0.82, alpha=0.64, linestyle=(0, (2.2, 2.2)) if path["dashed"] else "-",
                solid_capstyle="round", zorder=2)
        if path["limited"]:
            ax.plot(points[-1, 0], points[-1, 1], marker="o", markersize=2.7,
                    markeredgewidth=0.65, markerfacecolor="none",
                    color=charge_color(path), alpha=0.85, zorder=3)
    for jet in reversed(scene["jets"]):
        end = jet["end"][:2]
        if np.linalg.norm(end) < 1e-9:
            # A jet parallel to the beam has no transverse arrow.
            continue
        ax.add_patch(FancyArrowPatch((0, 0), tuple(end), arrowstyle="-|>",
                                    mutation_scale=19, linewidth=2.5, color=jet["color"],
                                    shrinkA=0, shrinkB=0, zorder=5))
    ax.scatter([0], [0], s=140, color="#57c6f0", alpha=0.12, edgecolors="none", zorder=7)
    ax.scatter([0], [0], s=70, color="#f7fbff", edgecolors="#80ccec", linewidths=0.9, zorder=8)
    fig.savefig(output, dpi=dpi, facecolor=display.BG, edgecolor=display.BG,
                metadata={"Title": scene["summary"], "Description":
                          "Orthographic x-y event projection; schematic geometry. "
                          "Positive charge yellow; negative green; neutral blue dashed. "
                          "All valid jet axes shown; open circles mark display-capped paths."})
    plt.close(fig)


def save_portable_html(scene, title, png_path, output):
    """Inline JS, data and PNG fallback: a page can be moved on its own."""
    figure = display.plotly_figure(scene, title)
    # Explicit sizing avoids reliance on a surrounding notebook/preview layout.
    figure.update_layout(autosize=True)
    post_script = """
      var gd = document.getElementById('{plot_id}');
      function showReady() {
        // Do not replace the static preview with a failed/blank WebGL scene.
        var s = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
        var gl = s && s.glplot && s.glplot.gl;
        if (!gl || (gl.isContextLost && gl.isContextLost())) return;
        document.body.classList.add('ready');
        requestAnimationFrame(function() { Plotly.Plots.resize(gd); });
      }
      gd.on('plotly_webglcontextlost', function() {
        document.body.classList.remove('ready');
        document.getElementById('status').textContent = '3D graphics are unavailable. The PNG preview is shown below.';
      });
      showReady();
    """
    chart = figure.to_html(full_html=False, include_plotlyjs=True, div_id="fcc-event",
                           default_width="100%", default_height="86vh", post_script=post_script,
                           config={"responsive": True, "displaylogo": False, "scrollZoom": True,
                                   "toImageButtonOptions": {"format": "png", "filename": output.stem,
                                                            "width": 2400, "height": 1600, "scale": 1}})
    image_uri = "data:image/png;base64," + base64.b64encode(png_path.read_bytes()).decode("ascii")
    notes = "".join("<li>" + html.escape(note) + "</li>" for note in scene["notes"])
    title_html = html.escape(title)
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title_html}</title><style>
:root{{color-scheme:dark}}body{{margin:0;background:{display.BG};color:{display.FG};font:15px Arial,sans-serif}}
a{{color:#9dddeb}}header,footer{{padding:16px 4vw;line-height:1.6}}header h1{{font-size:23px;margin:0 0 5px}}
header p{{margin:3px 0;color:{display.MUTED}}}#fallback img{{display:block;width:100%;height:auto}}
#interactive{{visibility:hidden;height:0;overflow:hidden}}.ready #interactive{{visibility:visible;height:auto;overflow:visible}}
.ready #fallback{{display:none}}.ready #status{{display:none}}footer{{color:{display.MUTED}}}
</style></head><body><header><h1>{title_html}</h1><p>{html.escape(scene['summary'])}</p>
<p id="status">Static preview below. Opening 3D… If it does not appear, download this HTML and open it in a browser outside the notebook preview.</p>
<a href="{image_uri}" download="{html.escape(png_path.name, quote=True)}">Download slide PNG</a></header>
<div id="fallback"><img src="{image_uri}" alt="End-on event projection"></div>
<div id="interactive">{chart}</div>
<footer>Drag to rotate · scroll to zoom · click legend entries to toggle objects.
<details><summary>Event display notes</summary><ul>{notes}</ul></details></footer>
<script>
setTimeout(function(){{if(!document.body.classList.contains('ready')){{
document.getElementById('status').textContent='The interactive view has not loaded. This PNG works without JavaScript. For 3D, download the HTML and open it in a browser with WebGL enabled.';
}}}},15000);
</script></body></html>'''
    output.write_text(document, encoding="utf-8")


def read_inputs(source, sample=None):
    if source.is_file():
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("format") != "fcc-ee-event-display-extract-v1":
            raise ValueError("The input file must be an event JSON; for a manifest, pass its containing folder.")
        sample = sample or "event"
        return {}, [(source, {"sample": sample, "number": 1})]
    if not source.is_dir():
        raise ValueError(f"No such file/folder: {source}. Pass the folder containing your exported event JSON files.")
    manifest_path = source / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = []
        for record in manifest.get("events", []):
            path = (source / record["json"]).resolve()
            if not path.is_relative_to(source):
                raise ValueError("A manifest JSON path points outside the gallery folder.")
            if not path.is_file():
                raise ValueError(f"Missing exported event: {path}. Restore that JSON before running the repair.")
            records.append((path, record))
        if not records:
            raise ValueError("The gallery manifest contains no events.")
        return manifest, records
    records = []
    counts = Counter()
    for path in sorted(source.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if payload.get("format") != "fcc-ee-event-display-extract-v1":
            continue
        parts = [p.lower() for p in path.relative_to(source).parts]
        label = ("signal" if "signal" in parts or path.stem.lower().startswith("signal_") else
                 "background" if "background" in parts or path.stem.lower().startswith("background_") else sample or "event")
        counts[label] += 1
        records.append((path, {"sample": label, "number": counts[label]}))
    if not records:
        raise ValueError("No exported event JSON files found. Keep the JSON files from create_event_gallery.py with the gallery.")
    return {}, records


def title_for(sample):
    process = {"signal": "H → gg signal", "background": "q q̄ continuum background"}.get(sample, "reconstructed event")
    return f"FCC-ee simulation · {process} · √s = 125 GeV"


def gallery_html(manifest):
    cards = []
    for r in manifest["events"]:
        name = f"{r['sample'].capitalize()} {r['number']:02d}"
        link = (f'<a href="{r["html"]}">Open interactive 3D</a> · ' if r.get("html") else "")
        cards.append(f'''<article><a href="{r['png']}"><img src="{r['png']}" loading="lazy" alt="{name}"></a>
<div><h2>{name}</h2><p>{html.escape(r['view_summary'])}</p>
<p>{link}<a href="{r['png']}" download>PNG</a> · <a href="{r['caption']}">Caption</a> · <a href="{r['json']}">JSON</a></p></div></article>''')
    counts = Counter(r["sample"] for r in manifest["events"])
    count_text = " · ".join(f"{n} {label}" for label, n in sorted(counts.items()))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FCC-ee event PNGs and 3D displays</title><style>
:root{{color-scheme:dark}}body{{background:#050b15;color:#edf3fa;font:16px system-ui,sans-serif;margin:0}}
main{{max-width:1500px;margin:auto;padding:36px}}h1{{font-size:32px}}p{{color:#a8bbcf;line-height:1.6}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:22px}}
article{{border:1px solid #26374a;border-radius:12px;overflow:hidden}}article img{{display:block;width:100%;aspect-ratio:16/9}}
article div{{padding:18px}}h2{{font-size:20px}}a{{color:#9dddeb}}footer{{margin-top:30px;line-height:1.6}}
</style></head><body><main><h1>FCC-ee event displays</h1><p>{count_text} · {manifest['export']['pixels'][0]} × {manifest['export']['pixels'][1]} PNGs.
End-on projections with charge colours. Each interactive page contains its own data, JavaScript and PNG preview.</p>
<section class="grid">{''.join(cards)}</section><footer>Yellow: positive charge · green: negative charge · dashed blue: neutral direction.
Jet arrows: cyan J1, orange J2, purple J3, then additional colours, ordered by energy. All valid axes are retained.<br>
Schematic geometry. Curves use saved track states where available, otherwise momentum propagation. No added material interactions.
Open endpoint circles mark display truncation at one turn or 12 m; line length is not energy.<br>
The clean PNGs omit labels and detector hit markers. Use the caption files when presenting them as scientific figures.
Event selection is unchanged; <a href="manifest.json">source and display manifest</a>.</footer></main></body></html>'''


def make_archives(output, manifest):
    full_zip = output.with_name(output.name + ".zip")
    png_zip = output.with_name(output.name + "_pngs.zip")
    with zipfile.ZipFile(full_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, str(Path(output.name)/path.relative_to(output)))
    with zipfile.ZipFile(png_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for r in manifest["events"]:
            for key in ["png", "caption"]:
                archive.write(output/r[key], r[key])
        archive.write(output/"manifest.json", "manifest.json")
    return full_zip, png_zip


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default="event_gallery", help="Exported gallery folder, or one event JSON")
    parser.add_argument("--output", help="New output folder (default: input name plus _fixed)")
    parser.add_argument("--sample", choices=["signal", "background"], help="Class for a single event or unlabelled JSONs")
    parser.add_argument("--dpi", type=int, default=240, help="16×9 inches at this DPI; default 3840×2160 pixels")
    parser.add_argument("--centred", action="store_true", help="Centre detector instead of leaving space for slide text")
    parser.add_argument("--png-only", action="store_true", help="Generate PNGs without interactive HTML")
    parser.add_argument("--no-zip", action="store_true", help="Skip archives")
    args = parser.parse_args(argv)
    if args.dpi < 50:
        parser.error("--dpi must be at least 50.")
    source = Path(args.source).expanduser().resolve()
    original, inputs = read_inputs(source, args.sample)
    output = (Path(args.output).expanduser().resolve() if args.output else
              source.with_name((source.stem if source.is_file() else source.name) + "_fixed"))
    if output == source or (source.is_dir() and output.is_relative_to(source)):
        parser.error("Choose an output folder outside the source gallery.")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output is not empty: {output}. Choose another --output folder.")
    # Validate the entire selection before writing. Do not silently replace a
    # missing, duplicate, mislabelled or incompatible event with another one.
    events = []
    seen = set()
    counters = Counter()
    for path, record in inputs:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get("format") != "fcc-ee-event-display-extract-v1":
            raise ValueError(f"Incompatible event JSON: {path}")
        src = payload.get("source", {})
        if record.get("source") and record["source"] != src:
            raise ValueError(f"Manifest source does not match event JSON: {path}")
        identity = (src.get("root_file_uuid") or src.get("root_path") or src.get("filename"),
                    src.get("entry_index_zero_based"))
        if None in identity:
            raise ValueError(f"Missing source identity in event JSON: {path}")
        if identity in seen:
            raise ValueError(f"Duplicate source event {identity}; no duplicate displays were created.")
        seen.add(identity)
        sample = record.get("sample", args.sample or "event")
        if sample not in {"signal", "background", "event"}:
            raise ValueError(f"Unsupported sample label: {sample}")
        counters[sample] += 1
        number = counters[sample]
        events.append((raw, payload, record, sample, number))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "fcc-ee-event-gallery-png-portable-v1", "events": [],
                "original_selection": copy.deepcopy({k: v for k, v in original.items() if k != "events"}),
                "export": {"created_utc": datetime.now(timezone.utc).isoformat(),
                           "pixels": [16*args.dpi, 9*args.dpi], "projection": "x-y, view from +z; +x right, +y up",
                           "layout": "centred" if args.centred else "detector right; slide text space left",
                           "colour_by": "charge", "self_contained_html": not args.png_only}}
    min_energy = float(original.get("config", {}).get("min_energy", 0))
    if not math.isfinite(min_energy) or min_energy < 0:
        raise ValueError("Saved display energy threshold is invalid.")
    options = types.SimpleNamespace(
        bz=None, position_unit="mm", colour_by="charge", track_mode="states", min_energy=min_energy,
        tracker_radius=2.0, tracker_half_length=2.0, calo_radius=2.35, calo_half_length=2.5,
        no_clusters=True, hits=False, max_points=10000, max_turns=1.0, max_path=12.0, roll=55.0)
    for i, (raw, payload, original_record, sample, number) in enumerate(events, 1):
        stem = f"{sample}/{sample}_{number:02d}"
        (output/sample).mkdir(exist_ok=True)
        scene = display.make_scene(payload, options)
        record = {"sample": sample, "number": number, "source": payload["source"],
                  "summary": payload.get("summary", {}), "view_summary": scene["summary"],
                  "json": stem + ".json", "png": stem + ".png", "caption": stem + "_caption.txt",
                  "html": None if args.png_only else stem + ".html", "status": "complete",
                  "source_json_sha256": hashlib.sha256(raw).hexdigest(),
                  "original_gallery_record": original_record,
                  "rendering": {"saved_track_states_used": scene["states_used"],
                                "momentum_fallback_paths": scene["charged_fallback"],
                                "display_capped_paths": scene["limited_count"], "notes": scene["notes"]}}
        (output/record["json"]).write_bytes(raw)
        title = title_for(sample)
        save_beam_png(scene, output/record["png"], args.dpi, args.centred)
        caption = (title + "\n" + scene["summary"] + "\n" + scene["detail"] + "\n\n"
                   "Orthographic x-y projection viewed from +z; +x right and +y up.\n"
                   "Positive charge: yellow; negative: green; neutral directions: blue dashed.\n"
                   "All valid jet axes shown in energy order: cyan, orange, purple, then further colours.\n"
                   "Jet arrow length encodes geometric extent, not energy. No hit/cluster markers shown.\n"
                   + "\n".join(scene["notes"]) + "\n\nSource: "
                   + (payload["source"].get("root_path") or payload["source"].get("filename", "unavailable"))
                   + "\nROOT file UUID: " + str(payload["source"].get("root_file_uuid", "unavailable")) + "\n")
        (output/record["caption"]).write_text(caption, encoding="utf-8")
        if not args.png_only:
            save_portable_html(scene, title, output/record["png"], output/record["html"])
        manifest["events"].append(record)
        (output/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output/"index.html").write_text(gallery_html(manifest), encoding="utf-8")
        print(f"[{i:02d}/{len(events):02d}] {sample} {number:02d}: PNG" + (" + portable HTML" if not args.png_only else ""), flush=True)
    print(f"\nSaved {len(events)} event PNGs: {output}")
    print("Counts: " + ", ".join(f"{n} {sample}" for sample, n in counters.items()))
    if not args.no_zip:
        for archive in make_archives(output, manifest):
            print(f"ZIP: {archive}")
    print("For slides, use the PNGs. For rotation, download and extract the ZIP, then open an individual HTML in a browser.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

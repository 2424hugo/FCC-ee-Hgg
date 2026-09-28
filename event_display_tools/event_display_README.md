# FCC-ee event display tools

Create 10 separate H → gg signal displays and 10 separate q q̄ background displays using the original Winter 2023 IDEA ROOT files.

## Run on your CERN Jupyter/LXPLUS machine

Extract `event_display_tools.zip` into a new folder. Keep its three Python scripts together.

```bash
python -m pip install "uproot>=5,<6" "awkward>=2,<3" numpy "plotly>=6,<7" "matplotlib>=3.6"
python create_event_gallery.py
```

The script's input defaults are:

| Sample | Directory |
|---|---|
| Signal | `/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/wzp6_ee_Hgg_ecm125/` |
| Background | `/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/wzp6_ee_qq_ecm125/` |

These directories must be accessible from the machine running the code. For downloaded data, pass `--signal-dir` and `--background-dir` with the actual local directories. Supply the correct process in each directory: sample labels come from these inputs, not a classifier or a truth-decay selection.

## Results

The script creates `event_gallery/` containing:

- `index.html`: a gallery with Signal and Background filters.
- `signal/signal_01.html` through `signal_10.html`: 10 interactive signal displays.
- `background/background_01.html` through `background_10.html`: 10 interactive background displays.
- A 300 dpi PNG and the extracted JSON alongside every HTML display.
- `manifest.json`: exact source filenames, ROOT UUIDs, entry numbers, cuts, random seeds, script hashes and rendering notes.
- `assets/plotly.min.js`: one shared copy of Plotly for offline viewing.

It also creates `event_gallery.zip`. Download that archive using Jupyter's file browser, extract it locally, and open `event_gallery/index.html` in your browser. Keep the folder structure intact; moving only an individual HTML file will omit its shared JavaScript dependency.

Only the gallery thumbnails load initially. Each interactive model opens in its own tab when selected, avoiding 20 simultaneous WebGL displays.

## Selection

Both samples require:

- invariant mass of **all** reconstructed particles strictly greater than 120 GeV;
- at least two reconstructed jets.

Seed 42 shuffles the source files. The script starts at a random entry in each file and takes the first event passing the cuts within a window of up to 1,024 entries. It prefers one selected event per file. If fewer files are available, it cycles through them while excluding previously selected `(ROOT UUID, entry)` pairs. It never chooses events based on appearance or classifier output.

These are reproducible illustrations, **not** a uniform sample of every passing event, a weighted physics sample, or evidence of typical classifier performance. The same seed reproduces the selection when the source files, settings and software remain unchanged.

## Useful commands

Continue an interrupted default run:

```bash
python create_event_gallery.py --resume
```

For a run with custom settings, retain those arguments when adding `--resume`. The script checks the saved settings and script hashes, keeps completed displays, and regenerates an unfinished figure from its extracted JSON when possible.

Show the stored tracker reference points as well:

```bash
python create_event_gallery.py --hits --output event_gallery_with_points
```

Generate a different set:

```bash
python create_event_gallery.py --seed 73 --output event_gallery_seed73
```

The script refuses to silently replace a non-empty output folder. Use `--resume` for the same run, or a new output folder for a new selection.

## Rendering assumptions

The updated renderer uses:

- every reconstructed jet axis, ordered by energy;
- constituent colours resolved through PODIO collection IDs and relation arrays, checked against the jet four-vectors;
- saved charged-track states when unambiguously linked; otherwise an explicitly reported momentum-based approximation;
- dashed neutral-particle directions;
- cluster positions recovered from their linked calorimeter reference positions when the stored cluster centre is zero and there is exactly one linked position.

The cylinders are schematic. The display performs no new track fit and adds no energy loss, scattering or detailed calorimeter showers. By default it caps a charged path after one turn or 12 m and marks capped endpoints with open circles. Fixed-size markers show positions, not energy deposits. The particle display threshold defaults to zero and does not change the event mass calculation.

## Individual events

The included extractor and renderer remain usable separately:

```bash
python extract_higgs_event.py "/actual/path/to/signal.root" --event 42 --output event_42.json
python plot_higgs_event_3d.py event_42.json --output-prefix event_42_refined
```

The single-event renderer includes its JavaScript inside the HTML, so that output is independently portable. The batch renderer shares the JavaScript across the gallery to reduce file size.

## Presentation use

Use a PNG on a Beamer slide. Keep the gallery open in a browser and switch to it for live rotation. An ordinary Beamer PDF does not run the Plotly HTML model inside the slide. Take the entire gallery folder, or its ZIP archive, to the presentation computer.

## Validation

The renderer was checked on the supplied signal event 42: 78 reconstructed particles, 44 saved track states and three jets with 27, 35 and 16 constituents. Its 34 cluster reference positions were recovered through the stored links.

The batch pipeline is tested separately with synthetic ROOT fixtures. Access to your CERN EOS data is needed to generate the requested 20 physical event displays; synthetic fixtures are not included as substitutes.

Underlying libraries: [Uproot](https://uproot.readthedocs.io/en/stable/basic.html), [Plotly HTML export](https://plotly.com/python/interactive-html-export/).

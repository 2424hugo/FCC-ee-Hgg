# Event PNGs and portable HTML

This tool reuses the JSON files exported by `create_event_gallery.py`. It keeps
the selected signal/background events and their original JSON bytes. It does
not read ROOT files or select replacements.

## Run on your CERN machine

Download `event_display_repair.zip` and upload it to the directory containing
your existing `event_gallery` folder. In that directory, run:

```bash
unzip event_display_repair.zip
python event_display_repair/repair_event_gallery.py event_gallery
```

The bundle includes its own copy of the plotting module. Keep its two Python
files together. Your original gallery scripts do not need to be overwritten.

If dependencies are missing:

```bash
python -m pip install numpy "plotly>=6,<7" "matplotlib>=3.6"
```

Python 3.9 or later is required. Matplotlib writes PNGs directly without a
browser, Chrome, Kaleido or a graphical desktop.

For the completed gallery of 10 signal and 10 background events, outputs are:

- `event_gallery_fixed/signal/`: 10 PNGs, 10 independent HTML pages, original
  JSON copies and plain-text scientific captions.
- `event_gallery_fixed/background/`: the corresponding 10 background events.
- `event_gallery_fixed/index.html`: a thumbnail gallery with links.
- `event_gallery_fixed/manifest.json`: event provenance and rendering notes.
- `event_gallery_fixed.zip`: the complete gallery.
- `event_gallery_fixed_pngs.zip`: PNGs, captions and the provenance manifest.

The script processes every event present in the input manifest. If the original
run stopped before producing 20 event JSONs, it reports the smaller actual
count; it does not manufacture or silently reselect missing events. If the
manifest is unavailable, it also recognises exported JSONs inside `signal/`
and `background/` subfolders. Missing JSONs named by a manifest produce an error.

For a single exported event:

```bash
python event_display_repair/repair_event_gallery.py event_42.json --sample signal
```

For PNGs only, add `--png-only`. For a centred view, add `--centred`. Default
resolution is 3840 × 2160 pixels (16:9); `--dpi 300` gives 4800 × 2700. These are
fixed-size images with equal horizontal/vertical scale, so circles stay circular.

Output goes to a separate folder. If you run the command again, choose another
output name with `--output event_gallery_fixed_v2`.

## Opening HTML

The older batch pages load `../assets/plotly.min.js`. Downloading a page alone
or moving it out of that folder can break that dependency. A notebook's HTML
preview can also restrict scripts. The exact cause on your machine has not
been established.

Each new event page embeds Plotly, its event data and a PNG preview. It has no
external script, font, CSS or image dependency. Download and extract the ZIP
onto your laptop, then open an event HTML in a browser outside the notebook
preview. You can also move an individual event HTML on its own. Interactive
3D still requires JavaScript and working WebGL; if it cannot start, the page
retains the static image and a PNG download link.

This follows Plotly's [self-contained HTML export](https://plotly.com/python/interactive-html-export/)
approach. The HTML files are several MB each because the JavaScript is embedded.

## What the PNG shows

The PNG projects the existing 3D trajectories into the x-y plane, looking from
+z along the beam axis. Positive x points right; positive y points up. All
events use the same coordinate orientation and schematic geometry. The default
composition leaves space on the left for presentation text.

- Positive charge: yellow. Negative charge: green. Neutral directions: blue dashed.
- All valid jet axes: cyan for J1, orange for J2, purple for J3, then further
  colours, ordered by decreasing energy. A jet parallel to the beam has no
  transverse arrow. Arrow lengths describe their geometric extent, not energy.
- Charged curves use saved track states where the original relationships are
  usable, otherwise the existing momentum-based propagation. No fit, material
  energy loss, scattering or calorimeter showers are added.
- Circles describe schematic radii, not exact engineering geometry. The
  existing defaults are tracker radius/half-length 2.0/2.0 m and calorimeter
  radius/half-length 2.35/2.5 m. Endcap crossings can project inside the rings.
- Paths stop at the existing boundaries or at the display cap of one turn or
  12 m. Open endpoint circles identify capped paths. Length is not energy.
- Hit/cluster markers and text labels are omitted for the clean slide layout.
  Use the accompanying captions when presenting a PNG as a scientific figure.
- The stored display energy threshold is retained when available in the input
  gallery configuration. The default is 0 GeV.

Signal/background labels come from the input gallery record or `--sample`;
the rendering does not determine the event's class from its appearance.

## Verification

PNG export was exercised on the uploaded signal event 42 and the full batch
workflow was exercised with a separate synthetic 20-event fixture. No synthetic
event images are included as user data. PNG dimensions, preserved JSON hashes,
event counts and lack of external HTML asset references were checked. Browser
playback could not be tested in this workspace because a browser runtime was
unavailable; it should be checked on your presentation laptop.

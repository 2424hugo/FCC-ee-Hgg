#!/usr/bin/env python3
"""Draw a 3D FCC-ee event from extract_higgs_event.py's JSON output.

Install (Python >=3.9):
    python -m pip install numpy "plotly>=6,<7" "matplotlib>=3.6"
Run:
    python plot_higgs_event_3d.py higgs_event.json
    python plot_higgs_event_3d.py higgs_event.json --open
    python plot_higgs_event_3d.py higgs_event.json --hits --min-energy 0.2

Outputs: higgs_event_3d.html (offline, rotatable, hover information) and
higgs_event_3d.png (300 dpi). The HTML camera button exports the current
view as a high-resolution PNG. Use --output-prefix to change output names.

Scientific interpretation:
* Input is the complete extracted event; no particles are randomly generated.
* Charged paths use saved EFlowTrack_1 states where their PODIO links can be
  resolved. The saved D0, Z0, reference point, phi, omega and tanLambda define
  each helix. There is no new fit to the stored points. Energy loss and
  scattering are not added. Missing/invalid states fall back to an explicitly
  reported origin-based approximation; --track-mode momentum forces that mode.
* Neutral/unknown-charge particles are drawn as straight direction lines.
* Default cylindrical dimensions are SCHEMATIC, not a detailed IDEA geometry.
  Change --tracker-radius, --tracker-half-length, --calo-radius and
  --calo-half-length using a validated detector model if needed (all metres).
* magFieldBz is interpreted in tesla; --bz overrides it. If absent/invalid,
  use straight lines and label that choice. Input momenta/energies are GeV.
* Stored cluster/hit positions are interpreted in mm by default. --position-unit m
  changes this. No calorimeter deposits are fabricated at track endpoints.
* Colours encode jet membership when metadata and ObjectID relations are valid;
  all jet axes are shown. --colour-by charge restores charge colours. Neutral
  directions are dashed. Membership must pass a constituent four-vector check;
  missing/ambiguous relations cause a labelled fallback, never a guessed link.
* Cluster centres that are zero can be represented at their linked stored
  calorimeter positions. Those are reference positions from fast simulation,
  not detailed shower deposits. Actual hit energies are not inferred.
* Rays/arcs end at schematic boundaries; their length is NOT an energy scale.
  Charged paths stop after at most one turn or 12 m, to keep curls readable.

References:
https://plotly.com/python/3d-line-plots/
https://plotly.com/python/interactive-html-export/
https://github.com/HEP-FCC/FCCeePhysicsPerformance/blob/master/General/README.md
"""

import argparse
import html
import json
import math
import re
import sys
import webbrowser
from pathlib import Path

try:
    import numpy as np
    import plotly.graph_objects as go
except ImportError as exc:
    raise SystemExit('Install dependencies: python -m pip install numpy "plotly>=6,<7" "matplotlib>=3.6"') from exc

RECO = "ReconstructedParticles/ReconstructedParticles."
JET = "Jet/Jet."
COLORS = {"Positive charge": "#f5da49", "Negative charge": "#8ade74",
          "Neutral": "#779eef", "Unknown charge": "#b4bfcf"}
JET_COLORS = ["#36d7ed", "#ffa167", "#ca95f5", "#a4d978", "#ee8eb4", "#e5d886"]
BG, FG, MUTED = "#050b15", "#edf3fa", "#a8bbcf"


def vector(data, name, length=None, required=False):
    values = data.get(name)
    if values is None:
        if required:
            raise ValueError(f"Missing required branch: {name}")
        return np.full(length, np.nan) if length is not None else np.array([])
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or (length is not None and len(arr) != length):
        raise ValueError(f"Unexpected shape for {name}: {arr.shape}")
    return arr


def four_vectors(data, prefix, required=True):
    energy = vector(data, prefix + "energy", required=required)
    momentum = np.column_stack([vector(data, prefix + "momentum." + axis,
                                      len(energy), required=required)
                                for axis in "xyz"])
    return energy, momentum


def field_value(data, override):
    if override is not None:
        return override, f"uniform Bz = {override:g} T (user override)"
    try:
        value = np.asarray(data.get("magFieldBz"), dtype=float).ravel()
        if value.size == 1 and np.isfinite(value[0]):
            return float(value[0]), f"uniform Bz = {value[0]:g} T (stored value)"
    except (TypeError, ValueError):
        pass
    return 0.0, "Bz unavailable: straight direction lines"


def trajectory(momentum, charge, bz, radius, half_length, max_path=12.0, max_turns=1.0):
    """Analytic constant-field helix from origin; x,y,z in m, p in GeV.

    s is 3D path length; k=dphi/ds=-0.299792458*q*Bz/|p|.
    Positive q in positive Bz bends from +x toward -y (q v cross B).
    Stop at FIRST cylinder/endcap intersection, one turn, or max_path.
    """
    p = float(np.linalg.norm(momentum))
    if not math.isfinite(p) or p <= 0:
        raise ValueError("A trajectory needs finite nonzero momentum.")
    direction = momentum / p
    transverse = float(np.hypot(direction[0], direction[1]))
    phi = math.atan2(direction[1], direction[0])
    k = -0.299792458 * charge * bz / p if math.isfinite(charge) else 0.0
    z_limit = half_length / abs(direction[2]) if abs(direction[2]) > 1e-14 else math.inf
    if abs(k) < 1e-10 or transverse < 1e-14:
        radial_limit = radius / transverse if transverse > 1e-14 else math.inf
        stop = min(radial_limit, z_limit, max_path)
        s = np.linspace(0, stop, 64)
        return s[:, None] * direction, stop < min(radial_limit, z_limit) - 1e-9

    rho = transverse / abs(k)
    radial_limit = (2 * math.asin(min(1.0, radius / (2 * rho))) / abs(k)
                    if radius <= 2 * rho else math.inf)
    boundary = min(radial_limit, z_limit)
    stop = min(boundary, max_turns * 2 * math.pi / abs(k), max_path)
    count = int(np.clip(64 + 60 * abs(k) * stop, 64, 500))
    s = np.linspace(0, stop, count)
    # sinc form remains stable in the nearly-straight limit.
    scale = transverse * s * np.sinc(k * s / (2 * math.pi))
    points = np.column_stack((scale * np.cos(phi + k * s / 2),
                              scale * np.sin(phi + k * s / 2), s * direction[2]))
    return points, stop < boundary - 1e-9


def collection_ids(payload):
    """Resolve collection IDs from the exported PODIO table, not fixed numbers."""
    for metadata in payload.get("file_metadata", {}).values():
        branches = metadata.get("branches", {})
        for key, values in branches.items():
            if key.endswith("/m_collectionIDs") and values:
                names = branches.get(key.rsplit("/", 1)[0] + "/m_names")
                if names and len(values[0]) == len(names[0]):
                    return {str(name): int(i) for name, i in zip(names[0], values[0])}
    return {}


def relation_links(data, owner, field, target_ids):
    """Return ObjectID lists per owner, using target IDs and stored offsets.

    No assumption that begin/end directly index reconstructed-particle arrays.
    Accept only a unique relation array whose target IDs match the requested
    collection(s). Reject out-of-range offsets and negative object indices.
    """
    prefix = owner + "/" + owner + "." + field
    begin, end = data.get(prefix + "_begin"), data.get(prefix + "_end")
    if begin is None or end is None or len(begin) != len(end) or not target_ids:
        return None
    if not end or max(end) == 0:
        return [[] for _ in begin]
    candidates = []
    pattern = re.compile(r"^" + re.escape(owner) + r"#\d+/" + re.escape(owner) + r"#\d+\.index$")
    for key, indices in data.items():
        if not pattern.match(key):
            continue
        ids = data.get(key[:-len("index")] + "collectionID", [])
        if len(indices) != len(ids) or len(indices) != max(end):
            continue
        if not all(0 <= a <= z <= len(indices) for a, z in zip(begin, end)):
            continue
        if not all(int(i) >= 0 and int(cid) in target_ids for i, cid in zip(indices, ids)):
            continue
        candidates.append([[(int(ids[k]), int(indices[k])) for k in range(a, z)]
                           for a, z in zip(begin, end)])
    return candidates[0] if len(candidates) == 1 else None


def verified_jet_membership(data, ids, energies, momenta, jet_energy, jet_momentum):
    reco_id = ids.get("ReconstructedParticles")
    links = relation_links(data, "Jet", "particles", {reco_id} if reco_id is not None else set())
    if links is None or len(links) != len(jet_energy):
        return None, "Jet relations unavailable/ambiguous; colours fall back to charge."
    assignment = np.full(len(energies), -1, dtype=int)
    for jet, objects in enumerate(links):
        ii = np.array([i for _, i in objects], dtype=int)
        if np.any(ii >= len(energies)) or len(np.unique(ii)) != len(ii) or np.any(assignment[ii] >= 0):
            return None, "Invalid or overlapping jet relations; colours fall back to charge."
        summed = np.r_[energies[ii].sum(), momenta[ii].sum(axis=0)]
        stored = np.r_[jet_energy[jet], jet_momentum[jet]]
        if not np.allclose(summed, stored, rtol=1e-4, atol=1e-4):
            return None, "Constituent four-vector check failed; colours fall back to charge."
        assignment[ii] = jet
    return assignment, "Jet ObjectID links verified against each stored jet four-vector."


def saved_track_states(data, ids, n_particles, position_scale):
    track_id = ids.get("EFlowTrack")
    links = relation_links(data, "ReconstructedParticles", "tracks",
                           {track_id} if track_id is not None else set())
    if links is None or len(links) != n_particles:
        return {}
    prefix = "EFlowTrack_1/EFlowTrack_1."
    state_fields = ["D0", "Z0", "phi", "omega", "tanLambda", "referencePoint.x",
                    "referencePoint.y", "referencePoint.z"]
    if not all(prefix + field in data for field in state_fields):
        return {}
    arrays = [np.asarray(data[prefix + field], dtype=float) for field in state_fields]
    if len({len(a) for a in arrays}) != 1:
        return {}
    starts = data.get("EFlowTrack/EFlowTrack.trackStates_begin", [])
    ends = data.get("EFlowTrack/EFlowTrack.trackStates_end", [])
    locations = data.get(prefix + "location", [0] * len(arrays[0]))
    result = {}
    for particle, objects in enumerate(links):
        if len(objects) != 1:
            continue
        track = objects[0][1]
        if track >= len(starts) or track >= len(ends):
            continue
        a, z = starts[track], ends[track]
        if not 0 <= a < z <= len(arrays[0]):
            continue
        state = next((k for k in range(a, z) if locations[k] == 1), a)
        values = np.array([v[state] for v in arrays])
        if not np.all(np.isfinite(values)):
            continue
        d0, z0, phi, omega, tanl, x, y, zref = values
        # EDM4hep/LCIO convention: x_PCA=x_ref-D0*sin(phi), y_PCA=y_ref+D0*cos(phi).
        # omega is signed inverse radius; direction evolves as phi(l)=phi0-omega*l.
        origin = position_scale * np.array([x-d0*np.sin(phi), y+d0*np.cos(phi), zref+z0])
        result[particle] = {"origin": origin, "phi": phi, "omega": omega/position_scale,
                            "tanLambda": tanl, "state_index": int(state), "track_index": int(track)}
    return result


def state_trajectory(state, radius, half_length, max_path, max_turns):
    """Propagate saved perigee parameters in metres to the first display boundary."""
    origin, phi, omega, tanl = (state[k] for k in ("origin", "phi", "omega", "tanLambda"))
    if np.hypot(*origin[:2]) >= radius or abs(origin[2]) >= half_length:
        raise ValueError("Saved reference point is outside the schematic tracker.")
    limit = max_path / math.sqrt(1 + tanl*tanl)
    if abs(omega) > 1e-12:
        limit = min(limit, max_turns * 2 * math.pi / abs(omega))
    def evaluate(length):
        length = np.asarray(length)
        angle = -omega * length
        scale = length * np.sinc(angle / (2 * math.pi))
        return origin + np.stack((scale*np.cos(phi+angle/2), scale*np.sin(phi+angle/2),
                                  length*tanl), axis=-1)
    count = int(np.clip(max(256, abs(omega)*limit/0.015), 256, 12000))
    lengths = np.linspace(0, limit, count)
    points = evaluate(lengths)
    outside = (np.hypot(points[:, 0], points[:, 1]) >= radius) | (np.abs(points[:, 2]) >= half_length)
    crossed = np.flatnonzero(outside)
    if not len(crossed):
        return points, True
    first = int(crossed[0]); lo, hi = lengths[first-1], lengths[first]
    for _ in range(35):
        mid = (lo+hi)/2; point = evaluate(mid)
        if np.hypot(*point[:2]) >= radius or abs(point[2]) >= half_length:
            hi = mid
        else:
            lo = mid
    return np.vstack((points[:first], evaluate((lo+hi)/2))), False


def cluster_positions(data, collection, ids, scale):
    """Use stored centres, or a linked hit position for a one-hit fast-sim cluster."""
    prefix = collection + "/" + collection + "."
    energy = vector(data, prefix + "energy")
    if not len(energy):
        return np.empty((0, 3)), energy, 0
    points = np.column_stack([vector(data, prefix + "position."+a, len(energy)) for a in "xyz"]) * scale
    good = np.all(np.isfinite(points), axis=1) & (np.linalg.norm(points, axis=1) > 1e-9)
    hit_id = ids.get("CalorimeterHits")
    links = relation_links(data, collection, "hits", {hit_id} if hit_id is not None else set())
    hprefix = "CalorimeterHits/CalorimeterHits.position."
    recovered = 0
    if links is not None and len(links) == len(energy) and all(hprefix+a in data for a in "xyz"):
        hits = np.column_stack([vector(data, hprefix+a) for a in "xyz"]) * scale
        for i in np.flatnonzero(~good):
            if len(links[i]) == 1:
                index = links[i][0][1]
                if index < len(hits) and np.all(np.isfinite(hits[index])) and np.linalg.norm(hits[index]) > 1e-9:
                    points[i] = hits[index]; recovered += 1
    return points, energy, recovered


def cylinder_lines(radius, half_length):
    angle = np.linspace(0, 2 * math.pi, 100)
    segments = [np.column_stack((radius * np.cos(angle), radius * np.sin(angle),
                                np.full_like(angle, z)))
                for z in np.linspace(-half_length, half_length, 9)]
    for phi in np.linspace(0, 2 * math.pi, 16, endpoint=False):
        segments.append(np.array([[radius * math.cos(phi), radius * math.sin(phi), -half_length],
                                  [radius * math.cos(phi), radius * math.sin(phi), half_length]]))
    return segments


def combine_segments(segments):
    return np.concatenate([np.vstack((segment, np.full((1, 3), np.nan)))
                           for segment in segments], axis=0)


def make_scene(payload, args):
    if payload.get("format") != "fcc-ee-event-display-extract-v1":
        raise ValueError("Use the JSON produced by extract_higgs_event.py.")
    data = payload["event_branches"]
    energies, momenta = four_vectors(data, RECO)
    charges = vector(data, RECO + "charge", len(energies))
    bz, field_note = field_value(data, args.bz)
    ids = collection_ids(payload)
    scale = 0.001 if args.position_unit == "mm" else 1.0
    jet_energy, jet_momentum = four_vectors(data, JET, required=False)
    membership, membership_note = verified_jet_membership(data, ids, energies, momenta, jet_energy, jet_momentum)
    use_jet_colours = membership is not None and args.colour_by != "charge"
    states = saved_track_states(data, ids, len(energies), scale) if args.track_mode == "states" else {}
    energy_order = np.argsort(-np.where(np.isfinite(jet_energy), jet_energy, -np.inf))
    ranks = {int(i): rank for rank, i in enumerate(energy_order)}
    jets = []
    for rank, i in enumerate(energy_order):
        p = np.linalg.norm(jet_momentum[i])
        if not np.isfinite(jet_energy[i]) or not np.all(np.isfinite(jet_momentum[i])) or p <= 0:
            continue
        end = trajectory(jet_momentum[i], 0, 0, args.calo_radius * 1.09,
                         args.calo_half_length * 1.09)[0][-1]
        count = int(np.sum(membership == i)) if membership is not None else None
        jets.append({"rank": rank + 1, "index": int(i), "end": end,
                     "color": JET_COLORS[rank % len(JET_COLORS)], "energy": float(jet_energy[i]),
                     "n_constituents": count})
    norms = np.linalg.norm(momenta, axis=1)
    valid = np.isfinite(energies) & np.all(np.isfinite(momenta), axis=1) & (norms > 0) & (energies >= 0)
    visible = valid & (energies >= args.min_energy)
    if not np.any(visible):
        raise ValueError("No finite, nonzero-momentum particles pass --min-energy.")
    paths = []
    limited_count = 0
    states_used = 0
    charged_fallback = 0
    for i in np.flatnonzero(visible):
        q = charges[i]
        group = ("Unknown charge" if not np.isfinite(q) else "Positive charge" if q > 0
                 else "Negative charge" if q < 0 else "Neutral")
        charged = np.isfinite(q) and q != 0
        colour = COLORS[group]
        if use_jet_colours:
            original_jet = int(membership[i])
            if original_jet >= 0:
                rank = ranks[original_jet]
                count = int(np.sum(membership == original_jet))
                group = f"J{rank+1} · {jet_energy[original_jet]:.1f} GeV · {count} particles"
                colour = JET_COLORS[rank % len(JET_COLORS)]
            else:
                group, colour = "Unassigned particles", "#b4bfcf"
        radius = args.tracker_radius if charged else args.calo_radius
        half_length = args.tracker_half_length if charged else args.calo_half_length
        mode = "neutral direction" if not charged else "momentum approximation"
        state = states.get(int(i)) if charged else None
        if state is not None:
            try:
                points, limited = state_trajectory(state, radius, half_length, args.max_path, args.max_turns)
                mode = "saved track state"; states_used += 1
            except ValueError:
                state = None
        if state is None:
            points, limited = trajectory(momenta[i], q, bz, radius, half_length, args.max_path, args.max_turns)
            charged_fallback += int(charged)
        limited_count += int(limited)
        paths.append({"index": int(i), "points": points, "group": group,
                      "color": colour, "energy": float(energies[i]),
                      "momentum": momenta[i], "charge": q,
                      "dashed": not charged, "limited": limited, "mode": mode,
                      "state_index": state["state_index"] if state is not None else None})

    clouds = []
    recovered_clusters = 0
    cloud_specs = []
    if not args.no_clusters:
        cloud_specs += [("EFlowPhoton", "Photon cluster positions", "#dae4f5", "energy"),
                        ("EFlowNeutralHadron", "Neutral-hadron cluster positions", "#a3bdcb", "energy")]
    if args.hits:
        cloud_specs += [("TrackerHits", "Stored tracker reference points", "#a8d4ef", "eDep")]
        if args.no_clusters:
            cloud_specs += [("CalorimeterHits", "Stored calorimeter positions", "#d3a9f1", "energy")]
    for collection, label, color, energy_name in cloud_specs:
        prefix = collection + "/" + collection + "."
        if collection in {"EFlowPhoton", "EFlowNeutralHadron"}:
            positions, en, recovered = cluster_positions(data, collection, ids, scale)
            recovered_clusters += recovered
            if not len(en):
                continue
        else:
            xs = vector(data, prefix + "position.x")
            if not len(xs):
                continue
            positions = np.column_stack((xs, vector(data, prefix + "position.y", len(xs)),
                                         vector(data, prefix + "position.z", len(xs)))) * scale
            en = vector(data, prefix + energy_name, len(xs))
        good = np.all(np.isfinite(positions), axis=1) & (np.linalg.norm(positions, axis=1) > 1e-9)
        # Do not fabricate detector positions for empty or all-zero collections.
        index = np.flatnonzero(good)
        if len(index) > args.max_points:
            index = index[np.linspace(0, len(index) - 1, args.max_points, dtype=int)]
        if len(index):
            clouds.append({"name": label, "points": positions[index], "energy": en[index],
                           "indices": index, "color": color,
                           "total_valid": int(good.sum()), "total_stored": len(en)})

    mass = None
    if np.all(np.isfinite(energies)) and np.all(np.isfinite(momenta)):
        e_sum = math.fsum(energies)
        p_sum = np.array([math.fsum(momenta[:, j]) for j in range(3)])
        m2 = e_sum**2 - float(p_sum @ p_sum)
        if m2 >= -1e-8:
            mass = math.sqrt(max(m2, 0.0))
    entry = payload.get("source", {}).get("entry_index_zero_based", "?")
    mass_text = f"{mass:.2f} GeV" if mass is not None else "unavailable"
    summary = f"Entry {entry}   |   reconstructed mass {mass_text}   |   {len(jet_energy)} jets"
    detail = f"{visible.sum()}/{len(energies)} particles shown   |   E ≥ {args.min_energy:g} GeV   |   {field_note}"
    notes = [f"{states_used} charged trajectories from saved track states; {charged_fallback} from momentum approximations.",
             "Schematic detector; no additional fit, energy loss or scattering. Neutral lines are directions.",
             membership_note]
    if args.bz is not None and states_used:
        notes.append("--bz changes fallback propagation only; saved track curvature is retained.")
    if limited_count:
        notes.append(f"{limited_count} paths capped at {args.max_turns:g} turn(s) or {args.max_path:g} m; circles mark capped endpoints.")
    if recovered_clusters:
        notes.append(f"{recovered_clusters} zero-centre clusters placed at their linked stored calorimeter positions.")
    if not jets:
        notes.append("Jet axes unavailable from the exported jet momenta.")
    if clouds:
        counts = "; ".join(f"{c['name']}: {len(c['points'])}/{c['total_stored']} points" for c in clouds)
        notes.append(counts + f"; positions interpreted in {args.position_unit}.")
    else:
        notes.append("No stored cluster/hit positions plotted.")
    if args.min_energy:
        notes.append("Displayed energy cut applies only to particle paths; event mass uses all particles.")
    if not np.all(valid):
        notes.append(f"{(~valid).sum()} invalid/zero-momentum particle paths omitted.")

    geometry = [("Tracker outline (schematic)", args.tracker_radius, args.tracker_half_length, "#638eaf"),
                ("Calorimeter outline (schematic)", args.calo_radius, args.calo_half_length, "#447194")]
    # Include all plotted positions in the view; never silently clip stored hits.
    extent = np.array([args.calo_radius * 1.16, args.calo_radius * 1.16, args.calo_half_length * 1.16])
    for cloud in clouds:
        extent = np.maximum(extent, np.max(np.abs(cloud["points"]), axis=0) * 1.08)
    # Look approximately perpendicular to the leading jet's transverse axis,
    # so the two jets are not hidden by foreshortening in the initial view.
    azimuth = math.atan2(jets[0]["end"][1], jets[0]["end"][0]) + math.pi / 2 if jets else 0.65
    return {"paths": paths, "jets": jets, "clouds": clouds, "geometry": geometry,
            "extent": extent, "summary": summary, "detail": detail, "notes": notes,
            "camera_azimuth": azimuth, "states_used": states_used,
            "charged_fallback": charged_fallback, "use_jet_colours": use_jet_colours,
            "limited_count": limited_count, "membership_verified": membership is not None,
            "max_turns": args.max_turns, "max_path": args.max_path,
            "roll": args.roll}


def plotly_figure(scene, title):
    fig = go.Figure()
    for label, radius, half_length, color in scene["geometry"]:
        points = combine_segments(cylinder_lines(radius, half_length))
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                  mode="lines", line=dict(color=color, width=1), opacity=0.22,
                                  name=label, hoverinfo="skip", legendgroup=label))
        a, z = np.meshgrid(np.linspace(0, 2 * math.pi, 50), [-half_length, half_length])
        fig.add_trace(go.Surface(x=radius*np.cos(a), y=radius*np.sin(a), z=z,
                                 surfacecolor=np.zeros_like(z), colorscale=[[0, color], [1, color]],
                                 opacity=0.045, showscale=False, hoverinfo="skip", showlegend=False,
                                 legendgroup=label))

    seen = set()
    for path in scene["paths"]:
        points = path["points"]
        p = path["momentum"]
        q = f"{path['charge']:g}" if np.isfinite(path["charge"]) else "unknown"
        hover = (f"Reconstructed particle {path['index']}<br>E = {path['energy']:.3f} GeV; q = {q} e"
                 f"<br>(px, py, pz) = ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) GeV"
                 f"<br>Source: {path['mode']}"
                 + (f" #{path['state_index']}" if path["state_index"] is not None else "")
                 + ("<br>Path display capped" if path["limited"] else ""))
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="lines",
                                  line=dict(color=path["color"], width=2.4,
                                            dash="dash" if path["dashed"] else "solid"),
                                  opacity=0.88, name=path["group"], legendgroup=path["group"],
                                  showlegend=path["group"] not in seen,
                                  hovertemplate=hover + "<extra></extra>"))
        seen.add(path["group"])
        if path["limited"]:
            end = points[-1]
            fig.add_trace(go.Scatter3d(x=[end[0]], y=[end[1]], z=[end[2]], mode="markers",
                                      marker=dict(size=4, color=path["color"], symbol="circle-open"),
                                      legendgroup=path["group"], showlegend=False,
                                      hovertemplate="Display-capped endpoint<extra></extra>"))

    for jet in scene["jets"]:
        end = jet["end"]
        name = f"Jet {jet['rank']} axis · {jet['energy']:.1f} GeV"
        group = (f"J{jet['rank']} · {jet['energy']:.1f} GeV · {jet['n_constituents']} particles"
                 if scene["use_jet_colours"] else name)
        fig.add_trace(go.Scatter3d(x=[0, end[0]], y=[0, end[1]], z=[0, end[2]], mode="lines+text",
                                  line=dict(color=jet["color"], width=7), text=["", f"J{jet['rank']}"],
                                  textfont=dict(color=jet["color"], size=18), name=name, legendgroup=group,
                                  showlegend=not scene["use_jet_colours"],
                                  hovertemplate=f"Original jet index {jet['index']}<br>{name}<extra></extra>"))
        unit = end / np.linalg.norm(end)
        fig.add_trace(go.Cone(x=[end[0]], y=[end[1]], z=[end[2]],
                             u=[unit[0]], v=[unit[1]], w=[unit[2]],
                             sizemode="absolute", sizeref=0.18, anchor="tip",
                             colorscale=[[0, jet["color"]], [1, jet["color"]]],
                             showscale=False, showlegend=False, hoverinfo="skip", legendgroup=group))
    for cloud in scene["clouds"]:
        points = cloud["points"]
        sizes = 3.5
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
                                  marker=dict(color=cloud["color"], size=sizes, symbol="square", opacity=0.8),
                                  name=cloud["name"], customdata=np.column_stack((cloud["indices"], cloud["energy"])),
                                  hovertemplate=(cloud["name"] + " %{customdata[0]:.0f}<br>E = %{customdata[1]:.3f} GeV"
                                                 "<br>x=%{x:.3f}, y=%{y:.3f}, z=%{z:.3f} m<extra></extra>")))
    beam = scene["extent"][2] * 0.93
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-beam, beam], mode="lines+text",
                              text=["−z", "+z · beam"], textfont=dict(color=MUTED),
                              line=dict(color=MUTED, width=2, dash="dot"), name="Beam axis", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                              marker=dict(color=FG, size=4), name="Propagation origin", hoverinfo="name"))
    axes = {axis + "axis": dict(title=f"{axis} [m]", range=[-extent, extent],
                               showbackground=False, showgrid=False, zeroline=False,
                               color=MUTED, ticks="", nticks=4)
            for axis, extent in zip("xyz", scene["extent"])}
    eye = np.array([1.85*math.cos(scene["camera_azimuth"]), 1.85*math.sin(scene["camera_azimuth"]), 0.85])
    view = eye/np.linalg.norm(eye)
    up = np.array([0., 0., 1.]) - view*view[2]; up /= np.linalg.norm(up)
    angle = math.radians(scene["roll"])
    up = math.cos(angle)*up + math.sin(angle)*np.cross(view, up)
    caption = (f"Schematic detector · {scene['states_used']} saved-state helices · neutral directions dashed"
               f"<br>{scene['limited_count']} paths display-capped; circles mark endpoints. No added material effects.")
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(family="Arial, sans-serif", color=FG, size=13),
                      title=dict(text=html.escape(title) + "<br><sup>" + html.escape(scene["summary"]) + "</sup>",
                                 x=0.04, y=0.97, font=dict(size=24)),
                      scene=dict(**axes, bgcolor=BG, aspectmode="data",
                                 camera=dict(eye=dict(zip("xyz", eye)), up=dict(zip("xyz", up))),
                                 dragmode="orbit", domain=dict(x=[0, 0.82], y=[0.11, 0.98])),
                      legend=dict(x=0.82, y=0.90, font=dict(size=12), bgcolor="rgba(0,0,0,0)",
                                  groupclick="togglegroup"),
                      margin=dict(l=20, r=20, t=105, b=45),
                      annotations=[dict(x=0.02, y=0.08, xref="paper", yref="paper", xanchor="left",
                                        yanchor="top", showarrow=False, align="left", font=dict(color=MUTED, size=12),
                                        text=html.escape(scene["detail"]) + "<br>" + caption)])
    return fig


def save_html(fig, scene, title, output, plotly_js=True):
    """Write a standalone display, or reference a shared local Plotly JS asset."""
    chart = fig.to_html(full_html=False, include_plotlyjs=plotly_js, default_height="88vh",
                        config=dict(responsive=True, displaylogo=False, scrollZoom=True,
                                    toImageButtonOptions=dict(format="png", filename=output.stem,
                                                              width=2400, height=1600, scale=1)))
    notes = "".join("<li>" + html.escape(note) + "</li>" for note in scene["notes"])
    document = ("<!doctype html><html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                "<title>" + html.escape(title) + "</title><style>"
                "body{margin:0;background:" + BG + ";color:" + FG + ";font:14px Arial,sans-serif}"
                "footer{padding:0 4vw 24px;color:" + MUTED + ";line-height:1.5}"
                "</style></head><body>" + chart + "<footer>"
                "Drag to rotate · scroll to zoom · click legend entries to toggle objects · camera button saves PNG."
                "<details><summary>Display assumptions and data handling</summary><ul>" + notes +
                "</ul></details></footer></body></html>")
    output.write_text(document, encoding="utf-8")


def save_png(scene, title, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError("PNG export requires matplotlib; install it or use --html-only.") from exc
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    ax = fig.add_axes([-0.005, 0.125, 0.81, 0.765], projection="3d", facecolor=BG)
    for _, radius, half_length, color in scene["geometry"]:
        for segment in cylinder_lines(radius, half_length):
            ax.plot(*segment.T, color=color, alpha=0.22, lw=0.55)
    seen = set()
    handles = []
    for path in scene["paths"]:
        ax.plot(*path["points"].T, color=path["color"], lw=0.9, alpha=0.85,
                linestyle="--" if path["dashed"] else "-")
        if path["limited"]:
            ax.scatter(*path["points"][-1], s=16, facecolors="none", edgecolors=path["color"],
                       linewidths=0.6, depthshade=False)
        if path["group"] not in seen:
            handles.append(Line2D([0], [0], color=path["color"], label=path["group"], lw=2,
                                  linestyle="--" if path["dashed"] else "-"))
            seen.add(path["group"])
    for jet in scene["jets"]:
        end = jet["end"]
        ax.quiver(0, 0, 0, *end, color=jet["color"], linewidth=2.3, arrow_length_ratio=0.09)
        ax.text(*(end * 1.05), f"J{jet['rank']}", color=jet["color"], fontsize=13, weight="bold")
        if not scene["use_jet_colours"]:
            handles.append(Line2D([0], [0], color=jet["color"], lw=3,
                                  label=f"Jet {jet['rank']} axis · {jet['energy']:.1f} GeV"))
    for cloud in scene["clouds"]:
        ax.scatter(*cloud["points"].T, s=9, c=cloud["color"], marker="s", alpha=0.85, depthshade=False)
        handles.append(Line2D([0], [0], marker="s", color=cloud["color"], linestyle="none", label=cloud["name"]))
    beam = scene["extent"][2] * 0.93
    ax.plot([0, 0], [0, 0], [-beam, beam], color=MUTED, lw=0.8, ls=":")
    ax.text(0, 0, beam, "+z · beam", color=MUTED, fontsize=10)
    ax.scatter([0], [0], [0], s=16, c=FG, depthshade=False)
    for dim, extent in zip("xyz", scene["extent"]):
        getattr(ax, f"set_{dim}lim")(-extent, extent)
    ax.set_box_aspect(scene["extent"], zoom=1.16)
    ax.view_init(elev=22, azim=math.degrees(scene["camera_azimuth"]), roll=scene["roll"])
    ax.set_axis_off()
    fig.text(0.05, 0.94, title, color=FG, fontsize=23, weight="bold")
    fig.text(0.05, 0.902, scene["summary"], color=MUTED, fontsize=13)
    fig.text(0.05, 0.112, scene["detail"], color=FG, fontsize=11)
    footer = (f"Schematic detector · {scene['states_used']} saved-state helices · neutral directions dashed. "
              "No added material effects.\n"
              f"{scene['limited_count']} paths capped at {scene['max_turns']:g} turn(s) or {scene['max_path']:g} m; "
              "open circles mark capped endpoints. Positions are from fast simulation.")
    if scene["charged_fallback"]:
        footer += f"\n{scene['charged_fallback']} charged paths use momentum approximations."
    fig.text(0.05, 0.079, footer, color=MUTED, fontsize=9, va="top", linespacing=1.5)
    handles.append(Line2D([0], [0], color="#638eaf", lw=1, label="Schematic cylinders"))
    handles += [Line2D([0], [0], color=MUTED, lw=1, linestyle="-", label="Charged trajectories"),
                Line2D([0], [0], color=MUTED, lw=1, linestyle="--", label="Neutral directions")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.77, 0.77),
               facecolor=BG, edgecolor="none", labelcolor=FG, fontsize=10, labelspacing=1.1)
    fig.savefig(output, dpi=300, facecolor=BG)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("event_json", help="JSON from extract_higgs_event.py")
    parser.add_argument("--output-prefix", default="higgs_event_3d", help="Output path without extension")
    parser.add_argument("--title", default="FCC-ee simulation · H → gg · √s = 125 GeV")
    parser.add_argument("--min-energy", type=float, default=0.0, help="Displayed particle energy cut [GeV]")
    parser.add_argument("--bz", type=float, help="Override Bz for momentum-mode/fallback paths [tesla]; saved curvature is unchanged")
    parser.add_argument("--track-mode", choices=["states", "momentum"], default="states",
                        help="Use saved states where linked, or momentum-based approximation")
    parser.add_argument("--colour-by", choices=["auto", "charge"], default="auto",
                        help="Use verified jet membership if available, or charge")
    parser.add_argument("--max-turns", type=float, default=1.0, help="Maximum displayed helix turns")
    parser.add_argument("--max-path", type=float, default=12.0, help="Maximum displayed path length [m]")
    parser.add_argument("--roll", type=float, default=55.0, help="Initial camera roll [degrees]")
    parser.add_argument("--tracker-radius", type=float, default=2.0, help="Schematic tracker radius [m]")
    parser.add_argument("--tracker-half-length", type=float, default=2.0, help="Schematic tracker half-length [m]")
    parser.add_argument("--calo-radius", type=float, default=2.35, help="Schematic calorimeter radius [m]")
    parser.add_argument("--calo-half-length", type=float, default=2.5, help="Schematic calorimeter half-length [m]")
    parser.add_argument("--position-unit", choices=["mm", "m"], default="mm", help="Units of stored positions")
    parser.add_argument("--hits", action="store_true", help="Also plot actual stored tracker/calorimeter hit positions")
    parser.add_argument("--max-points", type=int, default=10000, help="Maximum displayed points per cluster/hit collection")
    parser.add_argument("--no-clusters", action="store_true", help="Hide stored cluster positions")
    parser.add_argument("--html-only", action="store_true", help="Skip Matplotlib PNG export")
    parser.add_argument("--open", action="store_true", help="Open HTML in your default browser")
    parser.add_argument("--force", action="store_true", help="Replace existing output files")
    args = parser.parse_args(argv)
    dims = [args.tracker_radius, args.tracker_half_length, args.calo_radius, args.calo_half_length]
    if not all(math.isfinite(x) and x > 0 for x in dims):
        parser.error("Cylinder dimensions must be finite and positive.")
    if args.calo_radius < args.tracker_radius or args.calo_half_length < args.tracker_half_length:
        parser.error("Calorimeter outline must enclose tracker outline.")
    if not math.isfinite(args.min_energy) or args.min_energy < 0 or args.max_points < 1:
        parser.error("Require finite non-negative --min-energy and positive --max-points.")
    if args.bz is not None and not math.isfinite(args.bz):
        parser.error("--bz must be finite.")
    if not all(math.isfinite(v) and v > 0 for v in [args.max_turns, args.max_path]) or not math.isfinite(args.roll):
        parser.error("Path/turn limits must be finite and positive; camera roll must be finite.")
    prefix = Path(args.output_prefix).expanduser()
    html_path, png_path = Path(str(prefix) + ".html"), Path(str(prefix) + ".png")
    outputs = [html_path] if args.html_only else [html_path, png_path]
    for output in outputs:
        if output.exists() and not args.force:
            parser.error(f"{output} already exists; change --output-prefix or use --force.")
    with Path(args.event_json).expanduser().open(encoding="utf-8") as handle:
        payload = json.load(handle)
    scene = make_scene(payload, args)
    fig = plotly_figure(scene, args.title)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    save_html(fig, scene, args.title, html_path)
    print(f"Saved interactive display: {html_path.resolve()}", flush=True)
    if not args.html_only:
        save_png(scene, args.title, png_path)
        print(f"Saved 300 dpi figure: {png_path.resolve()}")
    print(scene["summary"])
    print(scene["detail"])
    for note in scene["notes"]:
        print(note)
    if args.open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

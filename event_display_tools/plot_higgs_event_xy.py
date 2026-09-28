#!/usr/bin/env python3
"""
Create a clean transverse (x-y) FCC-ee event display from the JSON output of
extract_higgs_event.py.

This is intended especially for a presentation title slide: it can generate a
16:9 PNG with the event shifted to the right, leaving empty dark space on the
left for title text.

Example:
    python plot_higgs_event_xy.py event_42.json --title-card \
        --output ee_H_gg_reconstructed_diagram.png \
        --min-energy 0.5

Dependencies:
    python -m pip install numpy matplotlib
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

RECO = "ReconstructedParticles/ReconstructedParticles."
JET = "Jet/Jet."

# Colour palette
BG = "#050b15"
FG = "#edf3fa"
MUTED = "#8ea4b8"

PARTICLE_COLORS = {
    "Positive charge": "#f5da49",   # yellow
    "Negative charge": "#8ade74",   # green
    "Neutral": "#77a8ff",           # blue
    "Unknown charge": "#b4bfcf",    # grey
}

JET_COLORS = ["#00d5ef", "#ff9057"]


def vector(data, name, length=None, required=False):
    values = data.get(name)
    if values is None:
        if required:
            raise ValueError(f"Missing required branch: {name}")
        return np.full(length, np.nan) if length is not None else np.array([])
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Unexpected shape for {name}: {arr.shape}")
    if length is not None and len(arr) != length:
        raise ValueError(f"Unexpected length for {name}: {len(arr)} != {length}")
    return arr


def four_vectors(data, prefix, required=True):
    energy = vector(data, prefix + "energy", required=required)
    momentum = np.column_stack([
        vector(data, prefix + "momentum." + axis, len(energy), required=required)
        for axis in "xyz"
    ])
    return energy, momentum


def field_value(data, override):
    if override is not None:
        return override
    try:
        value = np.asarray(data.get("magFieldBz"), dtype=float).ravel()
        if value.size == 1 and np.isfinite(value[0]):
            return float(value[0])
    except (TypeError, ValueError):
        pass
    return 0.0


def classify_charge(q):
    if not np.isfinite(q):
        return "Unknown charge"
    if q > 0:
        return "Positive charge"
    if q < 0:
        return "Negative charge"
    return "Neutral"


def trajectory(momentum, charge, bz, radius, half_length, max_path=12.0):
    """
    Analytic constant-field helix from origin, returned as points [x,y,z] in metres.
    Same basic model as your 3D script, but we will later plot only x and y.
    """
    p = float(np.linalg.norm(momentum))
    if not math.isfinite(p) or p <= 0:
        raise ValueError("Finite non-zero momentum required.")

    direction = momentum / p
    transverse = float(np.hypot(direction[0], direction[1]))
    phi = math.atan2(direction[1], direction[0])

    # sign convention matches the earlier 3D script
    k = -0.299792458 * charge * bz / p if math.isfinite(charge) else 0.0

    z_limit = half_length / abs(direction[2]) if abs(direction[2]) > 1e-14 else math.inf

    if abs(k) < 1e-10 or transverse < 1e-14:
        radial_limit = radius / transverse if transverse > 1e-14 else math.inf
        stop = min(radial_limit, z_limit, max_path)
        s = np.linspace(0, stop, 80)
        return s[:, None] * direction

    rho = transverse / abs(k)
    radial_limit = (
        2 * math.asin(min(1.0, radius / (2 * rho))) / abs(k)
        if radius <= 2 * rho else math.inf
    )

    boundary = min(radial_limit, z_limit)
    stop = min(boundary, 2 * math.pi / abs(k), max_path)

    npts = int(np.clip(80 + 80 * abs(k) * stop, 80, 500))
    s = np.linspace(0, stop, npts)

    scale = transverse * s * np.sinc(k * s / (2 * math.pi))
    points = np.column_stack((
        scale * np.cos(phi + k * s / 2),
        scale * np.sin(phi + k * s / 2),
        s * direction[2],
    ))
    return points


def particle_linewidth(energy):
    # modest energy-dependent line width
    return float(np.clip(0.5 + 0.25 * np.log10(max(energy, 0.05) / 0.05), 0.5, 2.2))


def particle_alpha(energy):
    return float(np.clip(0.18 + 0.22 * np.log10(max(energy, 0.05) / 0.05), 0.18, 0.85))


def add_detector(ax, tracker_radius, calo_radius):
    # main detector boundaries
    ax.add_patch(Circle((0, 0), tracker_radius, fill=False, lw=1.2,
                        ec="#6d8fae", alpha=0.30))
    ax.add_patch(Circle((0, 0), calo_radius, fill=False, lw=1.6,
                        ec="#4e7396", alpha=0.35))

    # faint inner rings for style/readability
    for frac, alpha in [(0.25, 0.10), (0.45, 0.12), (0.65, 0.14), (0.85, 0.16)]:
        ax.add_patch(Circle((0, 0), frac * calo_radius, fill=False, lw=0.8,
                            ec="#6b89a4", alpha=alpha))

    # spokes
    for phi in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        x = calo_radius * np.cos(phi)
        y = calo_radius * np.sin(phi)
        ax.plot([0, x], [0, y], color="#54718b", lw=0.5, alpha=0.10, zorder=0)


def add_jets(ax, jet_energy, jet_momentum, calo_radius, show_labels=False):
    if len(jet_energy) == 0:
        return

    order = np.argsort(-np.where(np.isfinite(jet_energy), jet_energy, -np.inf))[:2]
    for rank, i in enumerate(order):
        pvec = jet_momentum[i]
        p = np.linalg.norm(pvec)
        if not np.isfinite(jet_energy[i]) or not np.all(np.isfinite(pvec)) or p <= 0:
            continue

        direction = pvec[:2] / np.linalg.norm(pvec[:2]) if np.linalg.norm(pvec[:2]) > 0 else np.array([1.0, 0.0])
        end = 1.05 * calo_radius * direction

        arrow = FancyArrowPatch(
            (0, 0), (end[0], end[1]),
            arrowstyle='-|>',
            mutation_scale=18,
            lw=2.4,
            color=JET_COLORS[rank],
            alpha=0.95,
            zorder=8
        )
        ax.add_patch(arrow)

        if show_labels:
            ax.text(
                1.10 * end[0], 1.10 * end[1],
                f"J{rank+1}",
                color=JET_COLORS[rank],
                fontsize=11,
                ha="center", va="center", zorder=9
            )


def make_plot(payload, args):
    if payload.get("format") != "fcc-ee-event-display-extract-v1":
        raise ValueError("Input must be the JSON from extract_higgs_event.py")

    data = payload["event_branches"]
    energies, momenta = four_vectors(data, RECO, required=True)
    charges = vector(data, RECO + "charge", len(energies), required=False)

    bz = field_value(data, args.bz)

    norms = np.linalg.norm(momenta, axis=1)
    valid = (
        np.isfinite(energies)
        & np.all(np.isfinite(momenta), axis=1)
        & (norms > 0)
        & (energies >= 0)
    )
    visible = valid & (energies >= args.min_energy)

    if not np.any(visible):
        raise ValueError("No particles pass the selection.")

    # Build figure
    if args.title_card:
        fig_w = args.width
        fig_h = args.height
    else:
        fig_w = args.square_size
        fig_h = args.square_size

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
    ax.set_facecolor(BG)

    # Detector geometry
    add_detector(ax, args.tracker_radius, args.calo_radius)

    # Draw particles, highest energy last for visibility
    indices = np.flatnonzero(visible)
    indices = indices[np.argsort(energies[indices])]

    for i in indices:
        q = charges[i] if i < len(charges) else np.nan
        group = classify_charge(q)
        color = PARTICLE_COLORS[group]
        charged = np.isfinite(q) and q != 0

        radius = args.tracker_radius if charged else args.calo_radius
        half_length = args.tracker_half_length if charged else args.calo_half_length

        pts = trajectory(momenta[i], q, bz, radius, half_length)

        x = pts[:, 0]
        y = pts[:, 1]

        lw = particle_linewidth(energies[i])
        alpha = particle_alpha(energies[i])

        ax.plot(
            x, y,
            color=color,
            lw=lw,
            alpha=alpha,
            ls="-" if charged else (0, (2, 2)),
            zorder=4
        )

    # Leading jets
    jet_energy, jet_momentum = four_vectors(data, JET, required=False)
    add_jets(ax, jet_energy, jet_momentum, args.calo_radius, show_labels=args.jet_labels)

    # Collision point
    ax.scatter([0], [0], s=90, color="white", edgecolors="#b7e6ff",
               linewidths=0.8, zorder=10)
    ax.scatter([0], [0], s=320, color="#78d8ff", alpha=0.08, zorder=9)

    # Layout
    R = args.calo_radius
    margin = 0.15 * R

    if args.title_card:
        # extra empty space on the left for title text
        ax.set_xlim(-2.3 * R, 1.2 * R)
        ax.set_ylim(-1.15 * R, 1.15 * R)
    else:
        ax.set_xlim(-(R + margin), R + margin)
        ax.set_ylim(-(R + margin), R + margin)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Optional small annotation
    if args.caption:
        entry = payload.get("source", {}).get("entry_index_zero_based", "?")
        ax.text(
            0.02, 0.03,
            f"FCC-ee event display  •  entry {entry}  •  transverse (x-y) view",
            transform=ax.transAxes,
            color=MUTED,
            fontsize=11,
            ha="left", va="bottom"
        )

    # Save
    out = Path(args.output)
    fig.savefig(out, dpi=args.dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description="Plot a transverse x-y FCC-ee event display.")
    p.add_argument("json_file", help="JSON file from extract_higgs_event.py")
    p.add_argument("--output", default="ee_H_gg_reconstructed_diagram.png",
                   help="Output PNG filename")
    p.add_argument("--dpi", type=int, default=300)

    p.add_argument("--min-energy", type=float, default=0.5,
                   help="Minimum reconstructed particle energy [GeV] to display")

    p.add_argument("--bz", type=float, default=None,
                   help="Override Bz field [T]")

    p.add_argument("--tracker-radius", type=float, default=2.0,
                   help="Tracker radius [m]")
    p.add_argument("--tracker-half-length", type=float, default=2.4,
                   help="Tracker half-length [m]")
    p.add_argument("--calo-radius", type=float, default=3.3,
                   help="Calorimeter radius [m]")
    p.add_argument("--calo-half-length", type=float, default=4.0,
                   help="Calorimeter half-length [m]")

    p.add_argument("--title-card", action="store_true",
                   help="Use wide 16:9 layout with empty left space for slide title")
    p.add_argument("--width", type=float, default=13.333,
                   help="Figure width in inches for title-card mode")
    p.add_argument("--height", type=float, default=7.5,
                   help="Figure height in inches for title-card mode")

    p.add_argument("--square-size", type=float, default=8.0,
                   help="Square figure size in inches for non-title-card mode")

    p.add_argument("--caption", action="store_true",
                   help="Add a small caption in the lower-left corner")
    p.add_argument("--jet-labels", action="store_true",
                   help="Label leading jets J1 and J2")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.json_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    make_plot(payload, args)


if __name__ == "__main__":
    main()
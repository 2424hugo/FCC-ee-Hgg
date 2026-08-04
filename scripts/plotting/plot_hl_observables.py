"""Compare signal and background high-level observable distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scripts.data_processing.add_hl_observables import (
    add_all_hl_observables,
)


N_BINS = 60

SIGNAL_LABEL = r"$e^+e^- \rightarrow H \rightarrow gg$"
BACKGROUND_LABEL = r"$e^+e^- \rightarrow q\bar{q}$"


# One value for each selected jet.
JET_OBSERVABLES = {
    "jet_energy": {
        "xlabel": r"Jet energy [GeV]",
        "filename": "jet_energy.png",
    },
    "jet_mass": {
        "xlabel": r"Jet mass [GeV]",
        "filename": "jet_mass.png",
    },
    "jet_pt": {
        "xlabel": r"Jet $p_T$ [GeV]",
        "filename": "jet_pt.png",
    },
    "jet_p": {
        "xlabel": r"Jet $|\vec{p}|$ [GeV]",
        "filename": "jet_momentum.png",
    },
    "jet_phi": {
        "xlabel": r"Jet $\phi$ [rad]",
        "filename": "jet_phi.png",
        "range": (-np.pi, np.pi),
    },
    "jet_theta": {
        "xlabel": r"Jet $\theta$ [rad]",
        "filename": "jet_theta.png",
        "range": (0.0, np.pi),
    },
    "constituent_multiplicity": {
        "xlabel": "Constituent multiplicity",
        "filename": "constituent_multiplicity.png",
        "integer": True,
    },
    "e2_beta_0p2": {
        "xlabel": r"$e_2^{(0.2)}$",
        "filename": "e2_beta_0p2.png",
    },
}


# One value for each event.
EVENT_OBSERVABLES = {
    "event_invariant_mass": {
        "xlabel": r"Event invariant mass [GeV]",
        "filename": "event_invariant_mass.png",
    },
    "energy_asymmetry": {
        "xlabel": r"Energy asymmetry $A_E$",
        "filename": "energy_asymmetry.png",
        "range": (0.0, 1.0),
    },
    "dijet_pt": {
        "xlabel": r"Dijet $p_T$ [GeV]",
        "filename": "dijet_pt.png",
    },
    "dijet_opening_angle": {
        "xlabel": r"Dijet opening angle $\theta_{12}$ [rad]",
        "filename": "dijet_opening_angle.png",
        "range": (0.0, np.pi),
    },
    "dijet_acollinearity": {
        "xlabel": r"Dijet acollinearity $\pi-\theta_{12}$ [rad]",
        "filename": "dijet_acollinearity.png",
        "range": (0.0, np.pi),
    },
}


def finite_numpy(values: ak.Array) -> np.ndarray:
    """Convert an Awkward array to a one-dimensional finite NumPy array."""

    values = np.asarray(ak.to_numpy(values), dtype=np.float64)
    values = values.reshape(-1)

    return values[np.isfinite(values)]


def make_bin_edges(
    signal: np.ndarray,
    background: np.ndarray,
    *,
    fixed_range: tuple[float, float] | None = None,
    integer: bool = False,
) -> np.ndarray:
    """Construct common signal/background histogram bins."""

    combined = np.concatenate([signal, background])

    if len(combined) == 0:
        raise ValueError("No finite values available for histogram")

    if fixed_range is not None:
        lower, upper = fixed_range
    else:
        # Suppress only extreme numerical/outlier tails.
        lower, upper = np.quantile(combined, [0.001, 0.999])

        if lower == upper:
            lower = np.min(combined)
            upper = np.max(combined)

    if integer:
        lower = np.floor(lower)
        upper = np.ceil(upper)

        return np.arange(lower - 0.5, upper + 1.5, 1.0)

    return np.linspace(lower, upper, N_BINS + 1)


def draw_histogram(
    axis: plt.Axes,
    signal: np.ndarray,
    background: np.ndarray,
    bins: np.ndarray,
) -> None:
    """Draw normalized signal and background distributions."""

    axis.hist(
        background,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color="tab:blue",
        label=BACKGROUND_LABEL,
    )

    axis.hist(
        signal,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color="tab:red",
        label=SIGNAL_LABEL,
    )

    axis.set_ylabel("Probability density")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)


def plot_jet_observable(
    signal: ak.Array,
    background: ak.Array,
    field: str,
    settings: dict,
    output_dir: Path,
) -> None:
    """Plot leading and subleading distributions in separate panels."""

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        sharey=True,
    )

    jet_names = ["Leading jet", "Subleading jet"]

    for jet_index, (axis, jet_name) in enumerate(zip(axes, jet_names)):
        signal_values = finite_numpy(signal[field][:, jet_index])
        background_values = finite_numpy(background[field][:, jet_index])

        bins = make_bin_edges(
            signal_values,
            background_values,
            fixed_range=settings.get("range"),
            integer=settings.get("integer", False),
        )

        draw_histogram(
            axis,
            signal_values,
            background_values,
            bins,
        )

        axis.set_title(jet_name)
        axis.set_xlabel(settings["xlabel"])

    figure.suptitle(f"{settings['xlabel']}: signal vs background")
    figure.tight_layout()

    output_path = output_dir / settings["filename"]
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {output_path}")


def plot_event_observable(
    signal: ak.Array,
    background: ak.Array,
    field: str,
    settings: dict,
    output_dir: Path,
) -> None:
    """Plot a scalar event-level observable."""

    signal_values = finite_numpy(signal[field])
    background_values = finite_numpy(background[field])

    bins = make_bin_edges(
        signal_values,
        background_values,
        fixed_range=settings.get("range"),
    )

    figure, axis = plt.subplots(figsize=(6.5, 4.8))

    draw_histogram(
        axis,
        signal_values,
        background_values,
        bins,
    )

    axis.set_xlabel(settings["xlabel"])
    axis.set_title(f"{settings['xlabel']}: signal vs background")

    figure.tight_layout()

    output_path = output_dir / settings["filename"]
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--signal",
        type=Path,
        required=True,
        help="Input signal Parquet shard",
    )
    parser.add_argument(
        "--background",
        type=Path,
        required=True,
        help="Input background Parquet shard",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/plots/hl_observables"),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading signal:     {args.signal}")
    print(f"Loading background: {args.background}")

    signal = ak.from_parquet(args.signal)
    background = ak.from_parquet(args.background)

    print("Calculating signal observables...")
    signal = add_all_hl_observables(signal, beta=0.2)

    print("Calculating background observables...")
    background = add_all_hl_observables(background, beta=0.2)

    print(f"Signal events:     {len(signal):,}")
    print(f"Background events: {len(background):,}")

    for field, settings in JET_OBSERVABLES.items():
        plot_jet_observable(
            signal,
            background,
            field,
            settings,
            args.output_dir,
        )

    for field, settings in EVENT_OBSERVABLES.items():
        plot_event_observable(
            signal,
            background,
            field,
            settings,
            args.output_dir,
        )


if __name__ == "__main__":
    main()
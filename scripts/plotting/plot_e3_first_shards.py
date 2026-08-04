"""Plot e3^(0.2) for the first signal and background Parquet shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scripts.data_processing.add_hl_observables import (
    add_constituent_energy_fractions,
    add_e3,
)


BETA = 0.2
N_BINS = 60

SIGNAL_LABEL = r"$e^+e^- \rightarrow H \rightarrow gg$"
BACKGROUND_LABEL = r"$e^+e^- \rightarrow q\bar{q}$"


def calculate_e3_in_chunks(
    data: ak.Array,
    chunk_size: int,
    sample_name: str,
) -> np.ndarray:
    """Calculate e3 without constructing triples for the full shard at once."""

    e3_chunks = []
    number_of_events = len(data)

    for start in range(0, number_of_events, chunk_size):
        stop = min(start + chunk_size, number_of_events)

        chunk = data[start:stop]
        chunk = add_constituent_energy_fractions(chunk)
        chunk = add_e3(chunk, beta=BETA)

        chunk_e3 = np.asarray(
            ak.to_numpy(chunk.e3_beta_0p2),
            dtype=np.float64,
        )

        e3_chunks.append(chunk_e3)

        print(
            f"{sample_name}: processed "
            f"{stop:,}/{number_of_events:,} events"
        )

    return np.concatenate(e3_chunks, axis=0)


def finite_values(values: np.ndarray) -> np.ndarray:
    """Remove non-finite values from a one-dimensional array."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--signal",
        type=Path,
        required=True,
        help="First signal Parquet shard",
    )
    parser.add_argument(
        "--background",
        type=Path,
        required=True,
        help="First background Parquet shard",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/plots/hl_observables/"
            "e3_beta_0p2_first_shards.png"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Events processed simultaneously",
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading signal:     {args.signal}")
    print(f"Loading background: {args.background}")

    signal = ak.from_parquet(args.signal)
    background = ak.from_parquet(args.background)

    print(f"Signal events:     {len(signal):,}")
    print(f"Background events: {len(background):,}")

    signal_e3 = calculate_e3_in_chunks(
        signal,
        args.chunk_size,
        "Signal",
    )
    background_e3 = calculate_e3_in_chunks(
        background,
        args.chunk_size,
        "Background",
    )

    # Shape: events × 2 selected jets
    if signal_e3.ndim != 2 or signal_e3.shape[1] != 2:
        raise ValueError(f"Unexpected signal shape: {signal_e3.shape}")

    if background_e3.ndim != 2 or background_e3.shape[1] != 2:
        raise ValueError(
            f"Unexpected background shape: {background_e3.shape}"
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        sharey=True,
    )

    jet_names = ("Leading jet", "Subleading jet")

    for jet_index, (axis, jet_name) in enumerate(
        zip(axes, jet_names)
    ):
        signal_values = finite_values(signal_e3[:, jet_index])
        background_values = finite_values(
            background_e3[:, jet_index]
        )

        combined = np.concatenate(
            [signal_values, background_values]
        )

        # Common binning for signal and background.
        lower = float(np.min(combined))
        upper = float(np.max(combined))

        if lower == upper:
            upper = lower + 1.0

        bins = np.linspace(lower, upper, N_BINS + 1)

        axis.hist(
            background_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="tab:blue",
            label=BACKGROUND_LABEL,
        )

        axis.hist(
            signal_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="tab:red",
            label=SIGNAL_LABEL,
        )

        axis.set_title(jet_name)
        axis.set_xlabel(r"$e_3^{(0.2)}$")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)

        print(f"\n{jet_name}")
        print(
            "  Signal mean:     "
            f"{np.mean(signal_values):.6f}"
        )
        print(
            "  Background mean: "
            f"{np.mean(background_values):.6f}"
        )
        print(
            "  Signal median:   "
            f"{np.median(signal_values):.6f}"
        )
        print(
            "  Background median: "
            f"{np.median(background_values):.6f}"
        )

    axes[0].set_ylabel("Probability density")

    figure.suptitle(
        r"Three-point energy correlation: "
        r"$e_3^{(0.2)}$"
    )
    figure.tight_layout()
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
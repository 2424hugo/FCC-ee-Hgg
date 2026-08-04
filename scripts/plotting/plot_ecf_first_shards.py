"""Compare e2, e3, C2 and D2 for the first signal/background shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from scripts.data_processing.add_hl_observables import (
    add_c2_d2,
    add_constituent_energy_fractions,
    add_e2,
    add_e3,
)


BETA = 0.2
BETA_NAME = "0p2"
N_BINS = 60

SIGNAL_LABEL = r"$e^+e^- \rightarrow H \rightarrow gg$"
BACKGROUND_LABEL = r"$e^+e^- \rightarrow q\bar{q}$"

OBSERVABLES = {
    "e2_beta_0p2": {
        "symbol": r"$e_2^{(0.2)}$",
        "filename": "e2",
    },
    "e3_beta_0p2": {
        "symbol": r"$e_3^{(0.2)}$",
        "filename": "e3",
    },
    "c2_beta_0p2": {
        "symbol": r"$C_2^{(0.2)}$",
        "filename": "c2",
    },
    "d2_beta_0p2": {
        "symbol": r"$D_2^{(0.2)}$",
        "filename": "d2",
    },
}

JET_NAMES = {
    0: ("Leading jet", "leading"),
    1: ("Subleading jet", "subleading"),
}


def calculate_observables_in_chunks(
    data: ak.Array,
    chunk_size: int,
    sample_name: str,
) -> dict[str, np.ndarray]:
    """Calculate ECF observables without holding full-shard combinations."""

    output_chunks = {
        field: []
        for field in OBSERVABLES
    }

    number_of_events = len(data)

    for start in range(0, number_of_events, chunk_size):
        stop = min(start + chunk_size, number_of_events)

        chunk = data[start:stop]

        chunk = add_constituent_energy_fractions(chunk)
        chunk = add_e2(chunk, beta=BETA)
        chunk = add_e3(chunk, beta=BETA)
        chunk = add_c2_d2(chunk, beta=BETA)

        for field in OBSERVABLES:
            values = np.asarray(
                ak.to_numpy(chunk[field]),
                dtype=np.float64,
            )

            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError(
                    f"{field} has unexpected shape {values.shape}"
                )

            output_chunks[field].append(values)

        print(
            f"{sample_name}: processed "
            f"{stop:,}/{number_of_events:,} events"
        )

    return {
        field: np.concatenate(chunks, axis=0)
        for field, chunks in output_chunks.items()
    }


def finite_values(values: np.ndarray) -> np.ndarray:
    """Flatten an array and retain only finite values."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def common_bins(
    signal: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """Construct robust common bins for signal and background."""

    combined = np.concatenate([signal, background])

    if len(combined) == 0:
        raise ValueError("No finite values available")

    # Avoid allowing a few extreme C2/D2 ratios to dominate the axis.
    lower, upper = np.quantile(combined, [0.001, 0.999])

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Non-finite histogram range")

    if lower == upper:
        lower = float(np.min(combined))
        upper = float(np.max(combined))

    if lower == upper:
        upper = lower + 1.0

    return np.linspace(lower, upper, N_BINS + 1)


def plot_comparison(
    signal: np.ndarray,
    background: np.ndarray,
    xlabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Create one normalized signal/background comparison plot."""

    bins = common_bins(signal, background)

    figure, axis = plt.subplots(figsize=(7.0, 5.0))

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

    axis.set_xlabel(xlabel)
    axis.set_ylabel("Probability density")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {output_path}")


def print_summary(
    field: str,
    jet_name: str,
    signal: np.ndarray,
    background: np.ndarray,
) -> None:
    """Print basic checks for the plotted values."""

    print(f"\n{field}: {jet_name}")
    print(f"  Signal entries:       {len(signal):,}")
    print(f"  Background entries:   {len(background):,}")
    print(f"  Signal mean:          {np.mean(signal):.6g}")
    print(f"  Background mean:      {np.mean(background):.6g}")
    print(f"  Signal median:        {np.median(signal):.6g}")
    print(f"  Background median:    {np.median(background):.6g}")


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
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/plots/hl_observables/"
            "ecf_beta_0p2_first_shards"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of events processed simultaneously",
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading signal:     {args.signal}")
    print(f"Loading background: {args.background}")

    signal_data = ak.from_parquet(args.signal)
    background_data = ak.from_parquet(args.background)

    print(f"Signal events:      {len(signal_data):,}")
    print(f"Background events:  {len(background_data):,}")

    signal_results = calculate_observables_in_chunks(
        signal_data,
        args.chunk_size,
        "Signal",
    )

    background_results = calculate_observables_in_chunks(
        background_data,
        args.chunk_size,
        "Background",
    )

    for field, settings in OBSERVABLES.items():
        for jet_index, (jet_title, jet_filename) in JET_NAMES.items():
            signal_values = finite_values(
                signal_results[field][:, jet_index]
            )
            background_values = finite_values(
                background_results[field][:, jet_index]
            )

            print_summary(
                field,
                jet_title,
                signal_values,
                background_values,
            )

            output_path = args.output_dir / (
                f"{settings['filename']}_beta_{BETA_NAME}_"
                f"{jet_filename}.png"
            )

            plot_comparison(
                signal=signal_values,
                background=background_values,
                xlabel=settings["symbol"],
                title=f"{jet_title}: {settings['symbol']}",
                output_path=output_path,
            )


if __name__ == "__main__":
    main()
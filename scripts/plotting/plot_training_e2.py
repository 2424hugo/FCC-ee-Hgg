"""Plot e2(beta=0.2) for all signal and background training events."""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np


E2_FIELD = "e2_beta_0p2"

DEFAULT_SIGNAL_ROOT = Path("cache/analysis_dataset/signal/train")
DEFAULT_BACKGROUND_ROOT = Path("cache/analysis_dataset/background/train")
DEFAULT_OUTPUT = Path(
    "outputs/plots/training/e2_beta_0p2_signal_vs_background.png"
)


def find_shards(directory: Path) -> list[Path]:
    """Return all canonical Parquet shards below a training directory."""

    paths = sorted(
        path
        for path in directory.rglob("*part_*.parquet")
        if not path.name.startswith(".")
    )

    if not paths:
        raise FileNotFoundError(
            f"No Parquet shards matching '*part_*.parquet' found in "
            f"{directory}"
        )

    return paths


def load_e2(directory: Path, sample_name: str) -> np.ndarray:
    """Load the two e2 values per event from every shard."""

    paths = find_shards(directory)
    arrays = []
    number_of_events = 0

    print(f"\nLoading {sample_name}: {len(paths):,} shards")

    for index, path in enumerate(paths, start=1):
        data = ak.from_parquet(path, columns=[E2_FIELD])

        if E2_FIELD not in ak.fields(data):
            raise ValueError(f"{path}: missing field '{E2_FIELD}'")

        values = np.asarray(
            ak.to_numpy(data[E2_FIELD]),
            dtype=np.float64,
        )

        expected_shape = (len(data), 2)
        if values.shape != expected_shape:
            raise ValueError(
                f"{path}: expected {E2_FIELD} shape {expected_shape}, "
                f"found {values.shape}"
            )

        if not np.all(np.isfinite(values)):
            number_invalid = np.count_nonzero(~np.isfinite(values))
            raise ValueError(
                f"{path}: {number_invalid:,} non-finite {E2_FIELD} values"
            )

        arrays.append(values)
        number_of_events += len(data)

        print(
            f"  {index:>3}/{len(paths):<3} "
            f"{path.name}: {len(data):,} events",
            flush=True,
        )

    combined = np.concatenate(arrays, axis=0)

    print(
        f"{sample_name}: {number_of_events:,} events, "
        f"{combined.size:,} jet entries"
    )

    return combined


def common_bin_edges(
    signal: np.ndarray,
    background: np.ndarray,
    number_of_bins: int,
    x_min: float | None,
    x_max: float | None,
) -> np.ndarray:
    """Construct common linear bins for signal and background."""

    lower = (
        min(float(np.min(signal)), float(np.min(background)))
        if x_min is None
        else x_min
    )
    upper = (
        max(float(np.max(signal)), float(np.max(background)))
        if x_max is None
        else x_max
    )

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Histogram limits must be finite")

    if lower >= upper:
        raise ValueError(
            f"Histogram lower limit {lower} must be below upper limit {upper}"
        )

    return np.linspace(lower, upper, number_of_bins + 1)


def plot_distribution(
    axis: plt.Axes,
    signal: np.ndarray,
    background: np.ndarray,
    bins: np.ndarray,
    title: str,
) -> None:
    """Draw normalized signal and background distributions."""

    axis.hist(
        background,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color="tab:blue",
        label=r"$e^+e^-\rightarrow q\bar{q}$ background",
    )
    axis.hist(
        signal,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color="tab:orange",
        label=r"$H\rightarrow gg$ signal",
    )

    axis.set_title(title)
    axis.set_xlabel(r"$e_2^{(\beta=0.2)}$")
    axis.grid(alpha=0.25)


def make_plot(
    signal: np.ndarray,
    background: np.ndarray,
    output_path: Path,
    number_of_bins: int,
    x_min: float | None,
    x_max: float | None,
    log_y: bool,
) -> None:
    """Make leading, subleading and combined two-jet comparisons."""

    bins = common_bin_edges(
        signal,
        background,
        number_of_bins,
        x_min,
        x_max,
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(16, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    plot_distribution(
        axes[0],
        signal[:, 0],
        background[:, 0],
        bins,
        "Leading-energy jet",
    )
    plot_distribution(
        axes[1],
        signal[:, 1],
        background[:, 1],
        bins,
        "Subleading-energy jet",
    )
    plot_distribution(
        axes[2],
        signal.reshape(-1),
        background.reshape(-1),
        bins,
        "Both selected jets",
    )

    axes[0].set_ylabel("Probability density")
    axes[0].legend(frameon=False)

    if log_y:
        for axis in axes:
            axis.set_yscale("log")

    figure.suptitle(
        r"Training data: energy correlation function $e_2$"
        "\n"
        f"Signal: {len(signal):,} events; "
        f"background: {len(background):,} events",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"\nSaved plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot e2(beta=0.2) for all enriched signal and background "
            "training shards."
        )
    )
    parser.add_argument(
        "--signal-root",
        type=Path,
        default=DEFAULT_SIGNAL_ROOT,
    )
    parser.add_argument(
        "--background-root",
        type=Path,
        default=DEFAULT_BACKGROUND_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional lower x-axis limit",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional upper x-axis limit",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic y-axis",
    )

    args = parser.parse_args()

    if args.bins <= 0:
        raise ValueError("--bins must be positive")

    signal = load_e2(args.signal_root, "signal")
    background = load_e2(args.background_root, "background")

    make_plot(
        signal=signal,
        background=background,
        output_path=args.output,
        number_of_bins=args.bins,
        x_min=args.x_min,
        x_max=args.x_max,
        log_y=args.log_y,
    )


if __name__ == "__main__":
    main()
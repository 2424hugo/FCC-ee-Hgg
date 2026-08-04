"""Plot every stored numerical physics variable in the training data.

The script reads only the signal and background training directories.  It
uses a two-pass, shard-by-shard procedure so that all entries contribute to
the histograms without loading the complete dataset into memory.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SIGNAL_ROOT = Path("cache/analysis_dataset/signal/train")
DEFAULT_BACKGROUND_ROOT = Path("cache/analysis_dataset/background/train")
DEFAULT_OUTPUT_DIRECTORY = Path(
    "outputs/plots/training/all_variables"
)

# These fields describe bookkeeping rather than physical observables.
DEFAULT_EXCLUDED_FIELDS = {
    "source_file",
    "source_event",
    "label",
}

BACKGROUND_LABEL = r"$e^+e^-\rightarrow q\bar{q}$ background"
SIGNAL_LABEL = r"$H\rightarrow gg$ signal"

FIELD_LABELS = {
    "event_invariant_mass": r"$m_{\mathrm{event}}\ [\mathrm{GeV}]$",
    "n_jets_original": r"Original jet multiplicity",
    "jet_energy": r"$E_{\mathrm{jet}}\ [\mathrm{GeV}]$",
    "jet_px": r"$p_{x,\mathrm{jet}}\ [\mathrm{GeV}]$",
    "jet_py": r"$p_{y,\mathrm{jet}}\ [\mathrm{GeV}]$",
    "jet_pz": r"$p_{z,\mathrm{jet}}\ [\mathrm{GeV}]$",
    "jet_mass": r"$m_{\mathrm{jet}}\ [\mathrm{GeV}]$",
    "constituent_energy": r"$E_{\mathrm{constituent}}\ [\mathrm{GeV}]$",
    "constituent_px": r"$p_{x,\mathrm{constituent}}\ [\mathrm{GeV}]$",
    "constituent_py": r"$p_{y,\mathrm{constituent}}\ [\mathrm{GeV}]$",
    "constituent_pz": r"$p_{z,\mathrm{constituent}}\ [\mathrm{GeV}]$",
    "constituent_mass": r"$m_{\mathrm{constituent}}\ [\mathrm{GeV}]$",
    "constituent_charge": r"Constituent charge",
    "constituent_type": r"Constituent type",
    "constituent_multiplicity": r"Constituent multiplicity",
    "e2_beta_0p2": r"$e_2^{(\beta=0.2)}$",
    "e3_beta_0p2": r"$e_3^{(\beta=0.2)}$",
    "jet_pt": r"$p_{T,\mathrm{jet}}\ [\mathrm{GeV}]$",
    "jet_p": r"$|\vec{p}_{\mathrm{jet}}|\ [\mathrm{GeV}]$",
    "c2_beta_0p2": r"$C_2^{(\beta=0.2)}$",
    "d2_beta_0p2": r"$D_2^{(\beta=0.2)}$",
    "jet_theta": r"$\theta_{\mathrm{jet}}\ [\mathrm{rad}]$",
}


@dataclass(frozen=True)
class VariableSpec:
    """Description of one numerical field and its plot layout."""

    field: str
    kind: str
    panels: tuple[str, ...]


def find_shards(directory: Path) -> list[Path]:
    """Return all canonical Parquet shards below one training directory."""

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


def flattened_numpy(values: ak.Array) -> np.ndarray:
    """Flatten an Awkward array and return an ordinary one-dimensional array."""

    flattened = ak.flatten(values, axis=None)
    array = ak.to_numpy(flattened)

    if np.ma.isMaskedArray(array):
        array = array.compressed()

    return np.asarray(array).reshape(-1)


def is_numeric(values: ak.Array) -> bool:
    """Return whether an Awkward array has a numerical leaf type."""

    try:
        array = flattened_numpy(values)
    except (TypeError, ValueError):
        return False

    return np.issubdtype(array.dtype, np.number)


def infer_spec(field: str, values: ak.Array) -> VariableSpec | None:
    """Infer whether a field is event-level or contains two selected jets."""

    if not is_numeric(values):
        return None

    if values.ndim == 1:
        return VariableSpec(
            field=field,
            kind="event",
            panels=("Event distribution",),
        )

    number_per_event = np.asarray(
        ak.to_numpy(ak.num(values, axis=1))
    )

    if not np.all(number_per_event == 2):
        raise ValueError(
            f"{field}: expected two selected jets per event, found "
            f"axis-1 counts {np.unique(number_per_event)}"
        )

    return VariableSpec(
        field=field,
        kind="two_jet",
        panels=(
            "Leading-energy jet",
            "Subleading-energy jet",
            "Both selected jets",
        ),
    )


def extract_panels(
    values: ak.Array,
    spec: VariableSpec,
) -> dict[str, np.ndarray]:
    """Extract the arrays represented by the panels of one plot."""

    if spec.kind == "event":
        return {
            "Event distribution": flattened_numpy(values),
        }

    leading = flattened_numpy(values[:, 0])
    subleading = flattened_numpy(values[:, 1])

    return {
        "Leading-energy jet": leading,
        "Subleading-energy jet": subleading,
        "Both selected jets": np.concatenate(
            (leading, subleading)
        ),
    }


def discover_specs(
    signal_path: Path,
    background_path: Path,
    requested_fields: list[str] | None,
) -> list[VariableSpec]:
    """Discover numerical physics fields shared by both samples."""

    signal = ak.from_parquet(signal_path)
    background = ak.from_parquet(background_path)

    signal_fields = set(ak.fields(signal))
    background_fields = set(ak.fields(background))

    if signal_fields != background_fields:
        signal_only = sorted(signal_fields - background_fields)
        background_only = sorted(background_fields - signal_fields)
        raise ValueError(
            "Signal and background schemas differ. "
            f"Signal only: {signal_only}; "
            f"background only: {background_only}"
        )

    if requested_fields is None:
        fields = [
            field
            for field in ak.fields(signal)
            if field not in DEFAULT_EXCLUDED_FIELDS
        ]
    else:
        missing = set(requested_fields) - signal_fields
        if missing:
            raise ValueError(
                f"Requested fields are missing: {sorted(missing)}"
            )
        fields = requested_fields

    specs = []

    for field in fields:
        signal_spec = infer_spec(field, signal[field])
        background_spec = infer_spec(field, background[field])

        if signal_spec is None or background_spec is None:
            print(f"Skipping non-numerical field: {field}")
            continue

        if signal_spec != background_spec:
            raise ValueError(
                f"{field}: signal and background structures differ"
            )

        specs.append(signal_spec)

    if not specs:
        raise ValueError("No numerical fields were selected for plotting")

    del signal
    del background
    gc.collect()

    return specs


def finite_values(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Return finite values and the number of omitted non-finite values."""

    mask = np.isfinite(values)
    return values[mask], int(np.count_nonzero(~mask))


def scan_ranges(
    samples: dict[str, list[Path]],
    specs: list[VariableSpec],
) -> tuple[
    dict[str, tuple[float, float]],
    dict[str, int],
    dict[str, dict[str, int]],
]:
    """First pass: find a common finite range for each field."""

    ranges = {
        spec.field: (np.inf, -np.inf)
        for spec in specs
    }
    event_counts = {
        sample: 0
        for sample in samples
    }
    invalid_counts = {
        spec.field: {
            sample: 0
            for sample in samples
        }
        for spec in specs
    }
    selected_fields = [spec.field for spec in specs]

    print("\nPass 1/2: finding common histogram ranges")

    for sample, paths in samples.items():
        print(f"\nScanning {sample}: {len(paths):,} shards")

        for index, path in enumerate(paths, start=1):
            data = ak.from_parquet(path, columns=selected_fields)
            event_counts[sample] += len(data)

            missing = set(selected_fields) - set(ak.fields(data))
            if missing:
                raise ValueError(
                    f"{path}: missing fields {sorted(missing)}"
                )

            for spec in specs:
                panels = extract_panels(data[spec.field], spec)
                combined = panels[spec.panels[-1]]
                finite, number_invalid = finite_values(combined)
                invalid_counts[spec.field][sample] += number_invalid

                if finite.size == 0:
                    continue

                old_lower, old_upper = ranges[spec.field]
                ranges[spec.field] = (
                    min(old_lower, float(np.min(finite))),
                    max(old_upper, float(np.max(finite))),
                )

            print(
                f"  {index:>3}/{len(paths):<3} "
                f"{path.name}: {len(data):,} events",
                flush=True,
            )

            del data
            gc.collect()

    for field, (lower, upper) in ranges.items():
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"{field}: no finite values found")

        if lower == upper:
            padding = max(0.5, abs(lower) * 0.05)
            ranges[field] = (lower - padding, upper + padding)

    return ranges, event_counts, invalid_counts


def make_bin_edges(
    ranges: dict[str, tuple[float, float]],
    number_of_bins: int,
) -> dict[str, np.ndarray]:
    """Construct common linear bins for signal and background."""

    return {
        field: np.linspace(lower, upper, number_of_bins + 1)
        for field, (lower, upper) in ranges.items()
    }


def fill_histograms(
    samples: dict[str, list[Path]],
    specs: list[VariableSpec],
    bin_edges: dict[str, np.ndarray],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Second pass: fill every histogram from all training entries."""

    histograms = {
        spec.field: {
            sample: {
                panel: np.zeros(
                    len(bin_edges[spec.field]) - 1,
                    dtype=np.int64,
                )
                for panel in spec.panels
            }
            for sample in samples
        }
        for spec in specs
    }
    selected_fields = [spec.field for spec in specs]

    print("\nPass 2/2: filling histograms")

    for sample, paths in samples.items():
        print(f"\nFilling {sample}: {len(paths):,} shards")

        for index, path in enumerate(paths, start=1):
            data = ak.from_parquet(path, columns=selected_fields)

            for spec in specs:
                panels = extract_panels(data[spec.field], spec)

                for panel, values in panels.items():
                    finite, _ = finite_values(values)
                    counts, _ = np.histogram(
                        finite,
                        bins=bin_edges[spec.field],
                    )
                    histograms[spec.field][sample][panel] += counts

            print(
                f"  {index:>3}/{len(paths):<3} {path.name}",
                flush=True,
            )

            del data
            gc.collect()

    return histograms


def probability_density(
    counts: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Convert histogram counts to a unit-normalized probability density."""

    total = int(np.sum(counts))
    if total == 0:
        raise ValueError("Cannot normalize an empty histogram")

    return counts / (total * np.diff(edges))


def display_name(field: str) -> str:
    """Return a readable field name for the plot title."""

    return field.replace("_", " ")


def plot_field(
    spec: VariableSpec,
    histograms: dict[str, dict[str, np.ndarray]],
    edges: np.ndarray,
    event_counts: dict[str, int],
    output_directory: Path,
    log_y: bool,
) -> Path:
    """Create one signal-versus-background figure."""

    number_of_panels = len(spec.panels)
    figure, axes = plt.subplots(
        nrows=1,
        ncols=number_of_panels,
        figsize=(
            (6.2, 4.8)
            if number_of_panels == 1
            else (16, 4.8)
        ),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    if number_of_panels == 1:
        axes = np.asarray([axes])

    for axis, panel in zip(axes, spec.panels):
        background_density = probability_density(
            histograms["background"][panel],
            edges,
        )
        signal_density = probability_density(
            histograms["signal"][panel],
            edges,
        )

        axis.stairs(
            background_density,
            edges,
            linewidth=1.8,
            color="tab:blue",
            label=BACKGROUND_LABEL,
        )
        axis.stairs(
            signal_density,
            edges,
            linewidth=1.8,
            color="tab:orange",
            label=SIGNAL_LABEL,
        )

        axis.set_title(panel)
        axis.set_xlabel(
            FIELD_LABELS.get(spec.field, display_name(spec.field))
        )
        axis.grid(alpha=0.25)

        if log_y:
            axis.set_yscale("log")

    axes[0].set_ylabel("Probability density")
    axes[0].legend(frameon=False)

    figure.suptitle(
        f"Training data: {display_name(spec.field)}"
        "\n"
        f"Signal: {event_counts['signal']:,} events; "
        f"background: {event_counts['background']:,} events",
        fontsize=13,
    )

    output_path = output_directory / (
        f"{spec.field}_signal_vs_background.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot every stored numerical physics variable using only the "
            "signal and background training shards."
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
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic y-axis on every plot",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help=(
            "Optional list of fields to plot. By default, every numerical "
            "physics field is plotted."
        ),
    )

    args = parser.parse_args()

    if args.bins <= 0:
        raise ValueError("--bins must be positive")

    samples = {
        "signal": find_shards(args.signal_root),
        "background": find_shards(args.background_root),
    }

    specs = discover_specs(
        signal_path=samples["signal"][0],
        background_path=samples["background"][0],
        requested_fields=args.fields,
    )

    print("\nVariables to plot:")
    for spec in specs:
        print(f"  {spec.field} ({spec.kind})")

    ranges, event_counts, invalid_counts = scan_ranges(
        samples=samples,
        specs=specs,
    )
    bin_edges = make_bin_edges(
        ranges=ranges,
        number_of_bins=args.bins,
    )
    histograms = fill_histograms(
        samples=samples,
        specs=specs,
        bin_edges=bin_edges,
    )

    print("\nCreating plots")
    for spec in specs:
        plot_field(
            spec=spec,
            histograms=histograms[spec.field],
            edges=bin_edges[spec.field],
            event_counts=event_counts,
            output_directory=args.output_directory,
            log_y=args.log_y,
        )

    print("\nNon-finite entries omitted:")
    any_invalid = False
    for spec in specs:
        counts = invalid_counts[spec.field]
        if counts["signal"] or counts["background"]:
            any_invalid = True
            print(
                f"  {spec.field}: "
                f"signal={counts['signal']:,}, "
                f"background={counts['background']:,}"
            )

    if not any_invalid:
        print("  none")

    print(
        f"\nComplete: {len(specs):,} plots saved in "
        f"{args.output_directory}"
    )


if __name__ == "__main__":
    main()
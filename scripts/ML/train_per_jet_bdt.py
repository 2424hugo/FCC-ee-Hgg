#!/usr/bin/env python3
"""Train and validate a per-jet H->gg signal-versus-background BDT.

Each selected reconstructed jet is one classifier example.  The script reads
all Parquet shards recursively from the signal and background train/validation
directories, extracts the leading two jets while keeping every observable
aligned, and trains a histogram gradient-boosted decision tree.

Expected dataset layout
-----------------------
cache/analysis_dataset/
├── signal/
│   ├── train/**/*.parquet
│   └── validation/**/*.parquet  (or val/**/*.parquet)
└── background/
    ├── train/**/*.parquet
    └── validation/**/*.parquet  (or val/**/*.parquet)

The test split is deliberately never read.  The two selected jets from an
event remain in the same pre-existing split, preventing event leakage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_sample_weight

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURES = (
    "jet_mass",
    "constituent_multiplicity",
    "e2_beta_0p2",
    "e3_beta_0p2",
    "c2_beta_0p2",
    "d2_beta_0p2",
)

NUMBER_OF_SELECTED_JETS = 2
ORDERING_FEATURE = "jet_energy"

# Columns read from Parquet.  The ordering feature is loaded even when it is
# deliberately excluded from FEATURES, so changing the model feature set does
# not break the leading/subleading-jet ordering.
INPUT_COLUMNS = tuple(dict.fromkeys((*FEATURES, ORDERING_FEATURE)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an 11-variable per-jet histogram gradient-boosted decision "
            "tree using all signal/background Parquet shards."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("cache/analysis_dataset"),
        help="Directory containing signal/ and background/ (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ml/per_jet_bdt_11_variables"),
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--max-training-events-per-class",
        type=int,
        default=0,
        help="Training-event cap per class; 0 loads all events (default: %(default)s).",
    )
    parser.add_argument(
        "--max-validation-events-per-class",
        type=int,
        default=0,
        help="Validation-event cap per class; 0 loads all events (default: %(default)s).",
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument(
        "--target-signal-efficiencies",
        type=float,
        nargs="+",
        default=[0.50, 0.60, 0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--primary-signal-efficiency",
        type=float,
        default=0.70,
        help="Operating point saved in the model bundle (default: %(default)s).",
    )
    parser.add_argument(
        "--importance-jets",
        type=int,
        default=50_000,
        help="Maximum validation jets for permutation importance; 0 disables it.",
    )
    parser.add_argument("--importance-repeats", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for name in (
        "max_training_events_per_class",
        "max_validation_events_per_class",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    targets = [*args.target_signal_efficiencies, args.primary_signal_efficiency]
    if any(not 0.0 < target <= 1.0 for target in targets):
        raise ValueError("Signal-efficiency targets must lie in (0, 1]")
    if args.importance_jets < 0 or args.importance_repeats <= 0:
        raise ValueError("Invalid permutation-importance settings")


def parquet_files(directory: Path) -> list[Path]:
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under: {directory}")
    return files


def validation_directory(class_directory: Path) -> Path:
    for name in ("validation", "val"):
        candidate = class_directory / name
        if candidate.is_dir() and any(candidate.rglob("*.parquet")):
            return candidate
    raise FileNotFoundError(
        f"No validation split found at {class_directory / 'validation'} "
        f"or {class_directory / 'val'}"
    )


def is_numeric_list(data_type: pa.DataType) -> bool:
    is_list = (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    )
    if not is_list:
        return False
    value_type = data_type.value_type
    return (
        pa.types.is_integer(value_type)
        or pa.types.is_floating(value_type)
        or pa.types.is_decimal(value_type)
    ) and not pa.types.is_boolean(value_type)


def validate_schema(files: list[Path]) -> None:
    """Fail before the long load if a requested input is absent or malformed."""
    for file_path in files:
        schema = pq.ParquetFile(file_path).schema_arrow
        fields = {field.name: field.type for field in schema}
        missing = set(INPUT_COLUMNS) - fields.keys()
        if missing:
            raise ValueError(f"{file_path} is missing features: {sorted(missing)}")
        for feature in INPUT_COLUMNS:
            if not is_numeric_list(fields[feature]):
                raise TypeError(
                    f"{file_path}: {feature!r} has type {fields[feature]}; "
                    "a numeric jet-level list is required"
                )


def numeric_array(array: pa.Array) -> np.ndarray:
    """Convert nullable Arrow numerics to float32, retaining nulls as NaN."""
    converted = pc.cast(array, pa.float32(), safe=False)
    return np.asarray(converted.to_numpy(zero_copy_only=False), dtype=np.float32)


def first_two(array: pa.Array, feature: str) -> np.ndarray:
    lengths = pc.list_value_length(array)
    too_short = pc.any(pc.fill_null(pc.less(lengths, 2), True)).as_py()
    if too_short:
        raise ValueError(
            f"Feature {feature!r} contains an event with fewer than two jets"
        )
    try:
        return np.column_stack(
            (
                numeric_array(pc.list_element(array, 0)),
                numeric_array(pc.list_element(array, 1)),
            )
        )
    except pa.ArrowInvalid as error:
        raise ValueError(f"Could not extract two jets from {feature!r}") from error


def event_block(batch: pa.RecordBatch) -> np.ndarray:
    """Return [event, jet, feature], ordered by decreasing jet energy."""
    pairs = {
        feature: first_two(
            batch.column(batch.schema.get_field_index(feature)), feature
        )
        for feature in INPUT_COLUMNS
    }
    energy = pairs[ORDERING_FEATURE]
    swap = (
        np.isfinite(energy[:, 0])
        & np.isfinite(energy[:, 1])
        & (energy[:, 1] > energy[:, 0])
    )
    if np.any(swap):
        for values in pairs.values():
            values[swap] = values[swap, ::-1]
    return np.stack([pairs[feature] for feature in FEATURES], axis=2).astype(
        np.float32, copy=False
    )


def load_events(
    files: list[Path],
    batch_size: int,
    maximum_events: int,
    description: str,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """Load selected jets as [event, jet, feature] without reading test data."""
    blocks: list[np.ndarray] = []
    missing_counts = np.zeros(len(FEATURES), dtype=np.int64)
    infinity_counts = np.zeros(len(FEATURES), dtype=np.int64)
    loaded = 0

    print(f"\nLoading {description} from {len(files)} Parquet shard(s)")
    for file_number, file_path in enumerate(files, start=1):
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=list(INPUT_COLUMNS), use_threads=True
        ):
            block = event_block(batch)
            if maximum_events > 0:
                remaining = maximum_events - loaded
                if remaining <= 0:
                    break
                block = block[:remaining]

            missing_counts += np.isnan(block).sum(axis=(0, 1))
            infinite = np.isinf(block)
            infinity_counts += infinite.sum(axis=(0, 1))
            block[infinite] = np.nan
            blocks.append(block)
            loaded += len(block)
            if maximum_events > 0 and loaded >= maximum_events:
                break

        print(
            f"\r  shard {file_number:>4}/{len(files)} | events loaded: {loaded:,}",
            end="",
            flush=True,
        )
        if maximum_events > 0 and loaded >= maximum_events:
            break

    print()
    if not blocks:
        raise ValueError(f"No events loaded for {description}")
    events = np.concatenate(blocks, axis=0)
    missing = {
        feature: int(count)
        for feature, count in zip(FEATURES, missing_counts)
        if count > 0
    }
    infinities = {
        feature: int(count)
        for feature, count in zip(FEATURES, infinity_counts)
        if count > 0
    }
    return events, missing, infinities


def jet_matrix(events: np.ndarray) -> np.ndarray:
    return events.reshape(-1, events.shape[-1])


def combine_classes(
    signal_events: np.ndarray,
    background_events: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    signal = jet_matrix(signal_events)
    background = jet_matrix(background_events)
    features = np.concatenate((signal, background), axis=0)
    labels = np.concatenate(
        (
            np.ones(len(signal), dtype=np.int8),
            np.zeros(len(background), dtype=np.int8),
        )
    )
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(labels))
    return features[order], labels[order]


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [math.nan, math.nan]
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials**2)
        )
        / denominator
    )
    return [centre - half_width, centre + half_width]


def operating_point(
    target: float,
    labels: np.ndarray,
    scores: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, float | int | list[float]]:
    """Choose the tightest score cut with measured signal efficiency >= target."""
    candidates = np.flatnonzero(tpr >= target)
    if len(candidates) == 0:
        raise RuntimeError(f"ROC curve never reaches signal efficiency {target}")
    index = int(candidates[0])
    threshold = float(thresholds[index])
    signal_scores = scores[labels == 1]
    background_scores = scores[labels == 0]
    signal_accepted = int(np.count_nonzero(signal_scores >= threshold))
    background_accepted = int(np.count_nonzero(background_scores >= threshold))
    signal_efficiency = signal_accepted / len(signal_scores)
    background_efficiency = background_accepted / len(background_scores)
    return {
        "target_signal_efficiency": float(target),
        "threshold": threshold,
        "signal_efficiency": signal_efficiency,
        "background_efficiency": background_efficiency,
        "background_rejection_factor": (
            math.inf if background_efficiency == 0.0 else 1.0 / background_efficiency
        ),
        "signal_jets_accepted": signal_accepted,
        "signal_jets": int(len(signal_scores)),
        "background_jets_accepted": background_accepted,
        "background_jets": int(len(background_scores)),
        "background_efficiency_wilson_95": wilson_interval(
            background_accepted, len(background_scores)
        ),
        "roc_fpr": float(fpr[index]),
        "roc_tpr": float(tpr[index]),
    }


def event_efficiencies(
    signal_events: np.ndarray,
    background_events: np.ndarray,
    model: HistGradientBoostingClassifier,
    threshold: float,
) -> dict[str, float | int | list[float]]:
    """Measure two-tag efficiencies directly, without assuming independence."""
    signal_scores = model.predict_proba(jet_matrix(signal_events))[:, 1].reshape(
        len(signal_events), NUMBER_OF_SELECTED_JETS
    )
    background_scores = model.predict_proba(jet_matrix(background_events))[:, 1].reshape(
        len(background_events), NUMBER_OF_SELECTED_JETS
    )
    signal_pass = np.all(signal_scores >= threshold, axis=1)
    background_pass = np.all(background_scores >= threshold, axis=1)
    signal_accepted = int(np.count_nonzero(signal_pass))
    background_accepted = int(np.count_nonzero(background_pass))
    background_efficiency = background_accepted / len(background_events)
    return {
        "threshold": float(threshold),
        "signal_events_accepted": signal_accepted,
        "signal_events": int(len(signal_events)),
        "signal_event_efficiency": signal_accepted / len(signal_events),
        "background_events_accepted": background_accepted,
        "background_events": int(len(background_events)),
        "background_event_efficiency": background_efficiency,
        "background_event_rejection_factor": (
            math.inf if background_efficiency == 0.0 else 1.0 / background_efficiency
        ),
        "background_event_efficiency_wilson_95": wilson_interval(
            background_accepted, len(background_events)
        ),
        "signal_jet_score_correlation": float(
            np.corrcoef(signal_scores[:, 0], signal_scores[:, 1])[0, 1]
        ),
        "background_jet_score_correlation": float(
            np.corrcoef(background_scores[:, 0], background_scores[:, 1])[0, 1]
        ),
    }


def stratified_indices(
    labels: np.ndarray, maximum: int, random_state: int
) -> np.ndarray:
    if maximum <= 0 or maximum >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(random_state)
    signal_indices = np.flatnonzero(labels == 1)
    background_indices = np.flatnonzero(labels == 0)
    signal_count = min(len(signal_indices), maximum // 2)
    background_count = min(len(background_indices), maximum - signal_count)
    selected = np.concatenate(
        (
            rng.choice(signal_indices, signal_count, replace=False),
            rng.choice(background_indices, background_count, replace=False),
        )
    )
    rng.shuffle(selected)
    return selected


def save_roc_plot(
    fpr: np.ndarray, tpr: np.ndarray, auc: float, output_path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot(fpr, tpr, linewidth=2, label=f"Validation ROC (AUC = {auc:.4f})")
    axis.plot([0, 1], [0, 1], "--", color="black", alpha=0.6)
    axis.set(
        xlabel="Background-jet efficiency",
        ylabel="Signal-jet efficiency",
        title="6-variable substructure per-jet BDT",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_importance_plot(
    means: np.ndarray, standard_deviations: np.ndarray, output_path: Path
) -> None:
    order = np.argsort(means)
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(
        np.asarray(FEATURES)[order],
        means[order],
        xerr=standard_deviations[order],
        color="tab:blue",
        alpha=0.8,
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Decrease in validation ROC AUC after permutation",
        title="Per-jet BDT permutation importance",
    )
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)

    roots = {
        "signal_train": args.dataset_root / "signal" / "train",
        "background_train": args.dataset_root / "background" / "train",
        "signal_validation": validation_directory(args.dataset_root / "signal"),
        "background_validation": validation_directory(args.dataset_root / "background"),
    }
    files = {name: parquet_files(path) for name, path in roots.items()}
    validate_schema([path for group in files.values() for path in group])

    print("Per-jet input features:")
    for number, feature in enumerate(FEATURES, start=1):
        print(f"  {number:>2}. {feature}")

    loaded: dict[str, np.ndarray] = {}
    missing: dict[str, dict[str, int]] = {}
    infinities: dict[str, dict[str, int]] = {}
    for sample in (
        "signal_train",
        "background_train",
        "signal_validation",
        "background_validation",
    ):
        cap = (
            args.max_training_events_per_class
            if sample.endswith("train")
            else args.max_validation_events_per_class
        )
        loaded[sample], missing[sample], infinities[sample] = load_events(
            files[sample], args.batch_size, cap, sample.replace("_", " ")
        )

    train_features, train_labels = combine_classes(
        loaded["signal_train"], loaded["background_train"], args.random_state
    )
    validation_features, validation_labels = combine_classes(
        loaded["signal_validation"],
        loaded["background_validation"],
        args.random_state + 1,
    )
    training_weights = compute_sample_weight("balanced", train_labels)

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=args.learning_rate,
        max_iter=args.max_iterations,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        scoring="roc_auc",
        random_state=args.random_state,
        verbose=1,
    )
    print(
        f"\nTraining on {len(train_labels):,} jets "
        f"({np.count_nonzero(train_labels == 1):,} signal, "
        f"{np.count_nonzero(train_labels == 0):,} background)"
    )
    model.fit(train_features, train_labels, sample_weight=training_weights)

    scores = model.predict_proba(validation_features)[:, 1]
    fpr, tpr, thresholds = roc_curve(validation_labels, scores)
    auc = float(roc_auc_score(validation_labels, scores))
    average_precision = float(
        average_precision_score(validation_labels, scores)
    )
    targets = sorted(
        set([*args.target_signal_efficiencies, args.primary_signal_efficiency])
    )
    points = [
        operating_point(target, validation_labels, scores, fpr, tpr, thresholds)
        for target in targets
    ]
    primary = min(
        points,
        key=lambda point: abs(
            float(point["target_signal_efficiency"])
            - args.primary_signal_efficiency
        ),
    )
    double_tag = event_efficiencies(
        loaded["signal_validation"],
        loaded["background_validation"],
        model,
        float(primary["threshold"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "per_jet_bdt.joblib"
    metrics_path = args.output_dir / "metrics.json"
    roc_path = args.output_dir / "validation_roc.png"
    importance_path = args.output_dir / "permutation_importance.png"

    importance_summary: dict[str, dict[str, float]] | None = None
    if args.importance_jets > 0:
        indices = stratified_indices(
            validation_labels, args.importance_jets, args.random_state
        )
        print(
            f"\nCalculating permutation importance with {len(indices):,} "
            "validation jets"
        )
        importance = permutation_importance(
            model,
            validation_features[indices],
            validation_labels[indices],
            scoring="roc_auc",
            n_repeats=args.importance_repeats,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        save_importance_plot(
            importance.importances_mean,
            importance.importances_std,
            importance_path,
        )
        importance_summary = {
            feature: {
                "mean_auc_decrease": float(mean),
                "standard_deviation": float(std),
            }
            for feature, mean, std in sorted(
                zip(
                    FEATURES,
                    importance.importances_mean,
                    importance.importances_std,
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    model_bundle = {
        "model": model,
        "features": list(FEATURES),
        "signal_label": 1,
        "background_label": 0,
        "jet_definition": "two highest-energy candidate jets per event",
        "primary_signal_efficiency": args.primary_signal_efficiency,
        "decision_threshold": float(primary["threshold"]),
    }
    joblib.dump(model_bundle, model_path)
    save_roc_plot(fpr, tpr, auc, roc_path)

    metrics = {
        "model_definition": (
            "per-jet H->gg signal-candidate versus inclusive background "
            "classifier"
        ),
        "features": list(FEATURES),
        "number_of_features": len(FEATURES),
        "selected_jets_per_event": NUMBER_OF_SELECTED_JETS,
        "counts": {
            "training_signal_events": int(len(loaded["signal_train"])),
            "training_background_events": int(len(loaded["background_train"])),
            "training_signal_jets": int(np.count_nonzero(train_labels == 1)),
            "training_background_jets": int(np.count_nonzero(train_labels == 0)),
            "validation_signal_events": int(len(loaded["signal_validation"])),
            "validation_background_events": int(len(loaded["background_validation"])),
            "validation_signal_jets": int(np.count_nonzero(validation_labels == 1)),
            "validation_background_jets": int(np.count_nonzero(validation_labels == 0)),
        },
        "validation": {
            "roc_auc": auc,
            "average_precision": average_precision,
            "per_jet_operating_points": points,
            "primary_operating_point": primary,
            "direct_two_tag_event_performance": double_tag,
        },
        "training": {
            "classifier": "HistGradientBoostingClassifier",
            "learning_rate": args.learning_rate,
            "maximum_iterations": args.max_iterations,
            "iterations_used": int(model.n_iter_),
            "maximum_leaf_nodes": args.max_leaf_nodes,
            "minimum_samples_per_leaf": args.min_samples_leaf,
            "l2_regularization": args.l2_regularization,
            "early_stopping": True,
            "class_weighting": "balanced signal/background total weight",
            "random_state": args.random_state,
        },
        "data_quality": {
            "missing_values": missing,
            "infinite_values_converted_to_nan": infinities,
        },
        "permutation_importance": importance_summary,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")

    print("\nValidation results")
    print(f"  Signal jets:            {np.count_nonzero(validation_labels == 1):,}")
    print(f"  Background jets:        {np.count_nonzero(validation_labels == 0):,}")
    print(f"  ROC AUC:                {auc:.6f}")
    print(f"  Average precision:      {average_precision:.6f}")
    print("\nFixed signal-jet-efficiency operating points")
    print(" target    threshold    eps_signal      eps_background    B jets    rejection")
    for point in points:
        rejection = float(point["background_rejection_factor"])
        rejection_text = "infinite" if math.isinf(rejection) else f"{rejection:,.1f}"
        print(
            f" {float(point['target_signal_efficiency']):>6.2f}"
            f"  {float(point['threshold']):>11.6f}"
            f"  {float(point['signal_efficiency']):>12.6f}"
            f"  {float(point['background_efficiency']):>18.6e}"
            f"  {int(point['background_jets_accepted']):>8,}"
            f"  {rejection_text:>11}"
        )
    print(f"\nDirect double-tag event result at eps_signal ~= {args.primary_signal_efficiency:.2f}")
    print(
        "  Signal event efficiency:     "
        f"{float(double_tag['signal_event_efficiency']):.6e}"
    )
    print(
        "  Background event efficiency: "
        f"{float(double_tag['background_event_efficiency']):.6e}"
    )
    print(
        "  Background events accepted:  "
        f"{int(double_tag['background_events_accepted']):,}"
    )
    print("\nSaved:")
    print(f"  {model_path}")
    print(f"  {metrics_path}")
    print(f"  {roc_path}")
    if importance_summary is not None:
        print(f"  {importance_path}")


if __name__ == "__main__":
    main()
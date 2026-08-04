#!/usr/bin/env python3
"""
Train a boosted decision tree on 22 scalar event-level features.

The enriched shards contain scalar event columns, two-value jet columns and
variable-length constituent columns. Scikit-learn decision trees require a
rectangular scalar matrix, so this program constructs the following inputs:

* event_invariant_mass and n_jets_original;
* leading- and subleading-jet values for ten jet observables.

The two jets are ordered by jet_energy while loading. Raw constituent arrays
are deliberately not used; a constituent-based model such as ParticleNet is
more appropriate for those arrays.

Expected dataset layout
-----------------------
cache/analysis_dataset/
├── signal/
│   ├── train/*.parquet
│   └── validation/*.parquet  (or val/*.parquet)
└── background/
    ├── train/*.parquet
    └── validation/*.parquet  (or val/*.parquet)

The test split is deliberately not read. It should remain untouched until the
model and decision threshold have been fixed using the validation split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_sample_weight

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_NUMBER_OF_FEATURES = 22


@dataclass(frozen=True)
class FeatureDefinition:
    """Map one scalar model input to a Parquet source column."""

    name: str
    source_column: str
    jet_index: int | None = None


PAIR_FEATURE_COLUMNS = (
    "jet_energy",
    "jet_mass",
    "constituent_multiplicity",
    "e2_beta_0p2",
    "e3_beta_0p2",
    "jet_pt",
    "jet_p",
    "c2_beta_0p2",
    "d2_beta_0p2",
    "jet_theta",
)

DEFAULT_FEATURE_DEFINITIONS = (
    FeatureDefinition("event_invariant_mass", "event_invariant_mass"),
    FeatureDefinition("n_jets_original", "n_jets_original"),
    *(
        definition
        for source_column in PAIR_FEATURE_COLUMNS
        for definition in (
            FeatureDefinition(
                f"leading_{source_column}",
                source_column,
                0,
            ),
            FeatureDefinition(
                f"subleading_{source_column}",
                source_column,
                1,
            ),
        )
    ),
)

FEATURE_DEFINITIONS_BY_NAME = {
    definition.name: definition for definition in DEFAULT_FEATURE_DEFINITIONS
}

if len(DEFAULT_FEATURE_DEFINITIONS) != EXPECTED_NUMBER_OF_FEATURES:
    raise RuntimeError("The default BDT feature definition must contain 22 inputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and validate a histogram-based boosted decision tree using "
            "22 variables from signal and background Parquet shards."
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
        default=Path("outputs/ml/bdt_22_variables"),
        help="Directory for the model, metrics and plots (default: %(default)s).",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help=(
            "Optional ordered subset of derived scalar feature names. This "
            "22-variable program requires all 22; omit this option to use the "
            "validated default feature set."
        ),
    )
    parser.add_argument(
        "--max-events-per-class",
        type=int,
        default=0,
        help=(
            "Maximum events loaded from each class in each split. Use 0 for all "
            "events (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Parquet read batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Boosting learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=300,
        help="Maximum number of boosting iterations (default: %(default)s).",
    )
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=31,
        help="Maximum leaves in each decision tree (default: %(default)s).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=100,
        help="Minimum training events in a leaf (default: %(default)s).",
    )
    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=1.0,
        help="L2 regularisation strength (default: %(default)s).",
    )
    parser.add_argument(
        "--importance-events",
        type=int,
        default=50_000,
        help=(
            "Maximum validation events used for permutation importance; "
            "0 disables it (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--importance-repeats",
        type=int,
        default=5,
        help="Permutation repeats per feature (default: %(default)s).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for permutation importance (default: %(default)s).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: %(default)s).",
    )
    return parser.parse_args()


def parquet_files(directory: Path) -> list[Path]:
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under: {directory}")
    return files


def resolve_validation_directory(class_directory: Path) -> Path:
    """Accept either the name 'validation' or 'val'."""
    for name in ("validation", "val"):
        candidate = class_directory / name
        if candidate.is_dir() and any(candidate.rglob("*.parquet")):
            return candidate
    raise FileNotFoundError(
        f"Could not find a validation split at "
        f"{class_directory / 'validation'} or {class_directory / 'val'}"
    )


def is_scalar_numeric(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_decimal(data_type)
    ) and not pa.types.is_boolean(data_type)


def is_numeric_list(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    ) and is_scalar_numeric(data_type.value_type)


def resolve_feature_definitions(
    requested_names: list[str] | None,
) -> list[FeatureDefinition]:
    if requested_names is None:
        return list(DEFAULT_FEATURE_DEFINITIONS)

    unknown = [
        name for name in requested_names if name not in FEATURE_DEFINITIONS_BY_NAME
    ]
    if unknown:
        available = "\n  ".join(FEATURE_DEFINITIONS_BY_NAME)
        raise ValueError(
            f"Unknown derived feature name(s): {unknown}\n"
            f"Available names are:\n  {available}"
        )
    if len(requested_names) != EXPECTED_NUMBER_OF_FEATURES:
        raise ValueError(
            f"Expected exactly {EXPECTED_NUMBER_OF_FEATURES} features, received "
            f"{len(requested_names)}"
        )
    if len(set(requested_names)) != len(requested_names):
        raise ValueError("The feature list contains duplicate names")
    return [FEATURE_DEFINITIONS_BY_NAME[name] for name in requested_names]


def validate_feature_schema(
    files: list[Path],
    definitions: list[FeatureDefinition],
) -> None:
    """Check every shard before starting the potentially long data load."""
    required_definitions = {
        definition.source_column: definition for definition in definitions
    }
    if any(definition.jet_index is not None for definition in definitions):
        required_definitions["jet_energy"] = FeatureDefinition(
            "jet_energy_ordering",
            "jet_energy",
            0,
        )

    for file_path in files:
        schema = pq.ParquetFile(file_path).schema_arrow
        fields = {field.name: field.type for field in schema}
        missing = required_definitions.keys() - fields.keys()
        if missing:
            raise ValueError(f"{file_path} is missing features: {sorted(missing)}")

        for source_column, definition in required_definitions.items():
            data_type = fields[source_column]
            if definition.jet_index is None:
                valid_type = is_scalar_numeric(data_type)
                expected = "a scalar numeric column"
            else:
                valid_type = is_numeric_list(data_type)
                expected = "a numeric list containing the selected jets"
            if not valid_type:
                raise TypeError(
                    f"{file_path}: {source_column!r} has type {data_type}, "
                    f"but the BDT loader requires {expected}"
                )


def numeric_array(array: pa.Array) -> np.ndarray:
    """Convert a nullable Arrow numeric array to float32 with null -> NaN."""
    converted = pc.cast(array, pa.float32(), safe=False)
    return np.asarray(
        converted.to_numpy(zero_copy_only=False),
        dtype=np.float32,
    )


def two_jet_array(array: pa.Array, source_column: str) -> np.ndarray:
    """Extract the first two entries from one jet-level list column."""
    lengths = pc.list_value_length(array)
    too_short = pc.any(pc.less(lengths, 2)).as_py()
    if too_short:
        raise ValueError(
            f"Column {source_column!r} contains an event with fewer than two "
            "jet values. Recheck the enriched-shard selection."
        )

    try:
        first = numeric_array(pc.list_element(array, 0))
        second = numeric_array(pc.list_element(array, 1))
    except pa.ArrowInvalid as error:
        raise ValueError(
            f"Could not extract two jet values from {source_column!r}"
        ) from error
    return np.column_stack((first, second))


def load_parquet_matrix(
    files: list[Path],
    definitions: list[FeatureDefinition],
    batch_size: int,
    max_events: int,
    description: str,
) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    """
    Load only the selected columns as float32.

    HistGradientBoostingClassifier handles NaN values natively. Positive and
    negative infinities are converted to NaN and counted in the run summary.
    """
    blocks: list[np.ndarray] = []
    feature_names = [definition.name for definition in definitions]
    source_columns = list(
        dict.fromkeys(definition.source_column for definition in definitions)
    )
    pair_columns = {
        definition.source_column
        for definition in definitions
        if definition.jet_index is not None
    }
    if pair_columns and "jet_energy" not in source_columns:
        source_columns.append("jet_energy")

    events_loaded = 0
    missing_counts = np.zeros(len(definitions), dtype=np.int64)
    infinite_counts = np.zeros(len(definitions), dtype=np.int64)

    print(f"\nLoading {description} from {len(files)} Parquet shard(s)")
    for file_number, file_path in enumerate(files, start=1):
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=source_columns,
            use_threads=True,
        ):
            arrays = {
                name: batch.column(batch.schema.get_field_index(name))
                for name in source_columns
            }
            scalar_values = {
                definition.source_column: numeric_array(
                    arrays[definition.source_column]
                )
                for definition in definitions
                if definition.jet_index is None
            }
            pair_values = {
                source_column: two_jet_array(
                    arrays[source_column],
                    source_column,
                )
                for source_column in pair_columns | {"jet_energy"}
            }

            # Keep every jet-level observable aligned while defining the
            # leading jet as the higher-energy member of the stored pair.
            energy_pair = pair_values["jet_energy"]
            swap_jets = (
                np.isfinite(energy_pair[:, 0])
                & np.isfinite(energy_pair[:, 1])
                & (energy_pair[:, 1] > energy_pair[:, 0])
            )
            if np.any(swap_jets):
                for values in pair_values.values():
                    values[swap_jets] = values[swap_jets, ::-1]

            block = np.column_stack(
                [
                    (
                        scalar_values[definition.source_column]
                        if definition.jet_index is None
                        else pair_values[definition.source_column][
                            :, definition.jet_index
                        ]
                    )
                    for definition in definitions
                ]
            ).astype(np.float32, copy=False)

            if max_events > 0:
                remaining = max_events - events_loaded
                if remaining <= 0:
                    break
                block = block[:remaining]

            missing_counts += np.isnan(block).sum(axis=0)
            infinite = np.isinf(block)
            infinite_counts += infinite.sum(axis=0)
            block[infinite] = np.nan

            blocks.append(block)
            events_loaded += len(block)

            if max_events > 0 and events_loaded >= max_events:
                break

        print(
            f"\r  shard {file_number:>4}/{len(files)} | "
            f"events loaded: {events_loaded:,}",
            end="",
            flush=True,
        )
        if max_events > 0 and events_loaded >= max_events:
            break

    print()
    if not blocks:
        raise ValueError(f"No events were loaded for {description}")

    matrix = np.concatenate(blocks, axis=0)
    missing = {
        feature: int(count)
        for feature, count in zip(feature_names, missing_counts)
        if count > 0
    }
    infinite = {
        feature: int(count)
        for feature, count in zip(feature_names, infinite_counts)
        if count > 0
    }
    return matrix, missing, infinite


def combine_classes(
    signal: np.ndarray,
    background: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def best_youden_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    index_within_finite = np.argmax(
        true_positive_rate[finite] - false_positive_rate[finite]
    )
    best_index = np.flatnonzero(finite)[index_within_finite]
    return (
        float(thresholds[best_index]),
        false_positive_rate,
        true_positive_rate,
        thresholds,
    )


def stratified_importance_indices(
    labels: np.ndarray,
    maximum_events: int,
    random_state: int,
) -> np.ndarray:
    if maximum_events <= 0 or len(labels) <= maximum_events:
        return np.arange(len(labels))

    rng = np.random.default_rng(random_state)
    signal_indices = np.flatnonzero(labels == 1)
    background_indices = np.flatnonzero(labels == 0)

    signal_target = min(len(signal_indices), maximum_events // 2)
    background_target = min(
        len(background_indices), maximum_events - signal_target
    )
    selected = np.concatenate(
        (
            rng.choice(signal_indices, signal_target, replace=False),
            rng.choice(background_indices, background_target, replace=False),
        )
    )
    rng.shuffle(selected)
    return selected


def save_roc_plot(
    false_positive_rate: np.ndarray,
    true_positive_rate: np.ndarray,
    auc: float,
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"Validation ROC (AUC = {auc:.4f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.6)
    axis.set(
        xlabel="Background efficiency (false-positive rate)",
        ylabel="Signal efficiency (true-positive rate)",
        title="22-variable boosted decision tree",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_importance_plot(
    feature_names: list[str],
    means: np.ndarray,
    standard_deviations: np.ndarray,
    output_path: Path,
) -> None:
    order = np.argsort(means)
    fig_height = max(7.0, 0.35 * len(feature_names))
    fig, axis = plt.subplots(figsize=(9, fig_height))
    axis.barh(
        np.asarray(feature_names)[order],
        means[order],
        xerr=standard_deviations[order],
        color="tab:blue",
        alpha=0.8,
    )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Decrease in validation ROC AUC after permutation",
        title="Permutation feature importance",
    )
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def merge_counts(
    first: dict[str, int],
    second: dict[str, int],
) -> dict[str, int]:
    keys = set(first) | set(second)
    return {key: first.get(key, 0) + second.get(key, 0) for key in sorted(keys)}


def main() -> None:
    args = parse_args()
    if args.max_events_per_class < 0:
        raise ValueError("--max-events-per-class must be 0 or a positive integer")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    signal_root = args.dataset_root / "signal"
    background_root = args.dataset_root / "background"

    split_directories = {
        "signal_train": signal_root / "train",
        "background_train": background_root / "train",
        "signal_validation": resolve_validation_directory(signal_root),
        "background_validation": resolve_validation_directory(background_root),
    }
    files = {
        name: parquet_files(directory)
        for name, directory in split_directories.items()
    }

    selected_definitions = resolve_feature_definitions(args.features)
    selected_features = [
        definition.name for definition in selected_definitions
    ]

    print("Derived scalar input features:")
    for number, feature in enumerate(selected_features, start=1):
        print(f"  {number:>2}. {feature}")

    all_files = [path for group in files.values() for path in group]
    validate_feature_schema(all_files, selected_definitions)

    loaded: dict[str, np.ndarray] = {}
    missing_by_sample: dict[str, dict[str, int]] = {}
    infinite_by_sample: dict[str, dict[str, int]] = {}
    for sample_name in (
        "signal_train",
        "background_train",
        "signal_validation",
        "background_validation",
    ):
        matrix, missing, infinite = load_parquet_matrix(
            files=files[sample_name],
            definitions=selected_definitions,
            batch_size=args.batch_size,
            max_events=args.max_events_per_class,
            description=sample_name.replace("_", " "),
        )
        loaded[sample_name] = matrix
        missing_by_sample[sample_name] = missing
        infinite_by_sample[sample_name] = infinite

    train_features, train_labels = combine_classes(
        loaded["signal_train"],
        loaded["background_train"],
        args.random_state,
    )
    validation_features, validation_labels = combine_classes(
        loaded["signal_validation"],
        loaded["background_validation"],
        args.random_state + 1,
    )

    # Equal total signal and background weight prevents the larger class from
    # dominating training. These are classification weights, not physical
    # cross-section/luminosity weights.
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
        f"\nTraining on {len(train_labels):,} events "
        f"({np.count_nonzero(train_labels == 1):,} signal, "
        f"{np.count_nonzero(train_labels == 0):,} background)"
    )
    model.fit(train_features, train_labels, sample_weight=training_weights)

    validation_scores = model.predict_proba(validation_features)[:, 1]
    validation_auc = float(roc_auc_score(validation_labels, validation_scores))
    validation_average_precision = float(
        average_precision_score(validation_labels, validation_scores)
    )
    (
        threshold,
        false_positive_rate,
        true_positive_rate,
        _,
    ) = best_youden_threshold(validation_labels, validation_scores)

    predictions = (validation_scores >= threshold).astype(np.int8)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        validation_labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    signal_efficiency = true_positive / (true_positive + false_negative)
    background_efficiency = false_positive / (false_positive + true_negative)
    background_rejection = 1.0 - background_efficiency
    balanced_accuracy = balanced_accuracy_score(validation_labels, predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "bdt_model.joblib"
    metrics_path = args.output_dir / "metrics.json"
    roc_path = args.output_dir / "validation_roc.png"
    importance_path = args.output_dir / "permutation_importance.png"

    model_bundle = {
        "model": model,
        "features": selected_features,
        "feature_definitions": [
            {
                "name": definition.name,
                "source_column": definition.source_column,
                "jet_index_after_energy_ordering": definition.jet_index,
            }
            for definition in selected_definitions
        ],
        "decision_threshold": threshold,
        "signal_label": 1,
        "background_label": 0,
    }
    joblib.dump(model_bundle, model_path)
    save_roc_plot(
        false_positive_rate,
        true_positive_rate,
        validation_auc,
        roc_path,
    )

    importance_summary: dict[str, dict[str, float]] | None = None
    if args.importance_events > 0:
        importance_indices = stratified_importance_indices(
            validation_labels,
            args.importance_events,
            args.random_state,
        )
        print(
            f"\nCalculating permutation importance with "
            f"{len(importance_indices):,} validation events"
        )
        importance = permutation_importance(
            model,
            validation_features[importance_indices],
            validation_labels[importance_indices],
            scoring="roc_auc",
            n_repeats=args.importance_repeats,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        save_importance_plot(
            selected_features,
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
                    selected_features,
                    importance.importances_mean,
                    importance.importances_std,
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    metrics = {
        "features": selected_features,
        "number_of_features": len(selected_features),
        "event_counts": {
            "train_signal": int(np.count_nonzero(train_labels == 1)),
            "train_background": int(np.count_nonzero(train_labels == 0)),
            "validation_signal": int(np.count_nonzero(validation_labels == 1)),
            "validation_background": int(np.count_nonzero(validation_labels == 0)),
        },
        "validation": {
            "roc_auc": validation_auc,
            "average_precision": validation_average_precision,
            "youden_threshold": threshold,
            "signal_efficiency": float(signal_efficiency),
            "background_efficiency": float(background_efficiency),
            "background_rejection": float(background_rejection),
            "balanced_accuracy": float(balanced_accuracy),
            "confusion_matrix": {
                "true_negative": int(true_negative),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "true_positive": int(true_positive),
            },
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
            "class_weighting": "balanced",
            "random_state": args.random_state,
        },
        "data_quality": {
            "missing_values": missing_by_sample,
            "infinite_values_converted_to_nan": infinite_by_sample,
        },
        "permutation_importance": importance_summary,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    total_missing = merge_counts(
        merge_counts(
            missing_by_sample["signal_train"],
            missing_by_sample["background_train"],
        ),
        merge_counts(
            missing_by_sample["signal_validation"],
            missing_by_sample["background_validation"],
        ),
    )
    total_infinite = merge_counts(
        merge_counts(
            infinite_by_sample["signal_train"],
            infinite_by_sample["background_train"],
        ),
        merge_counts(
            infinite_by_sample["signal_validation"],
            infinite_by_sample["background_validation"],
        ),
    )

    print("\nValidation results")
    print(f"  ROC AUC:                {validation_auc:.6f}")
    print(f"  Average precision:      {validation_average_precision:.6f}")
    print(f"  Youden threshold:       {threshold:.6f}")
    print(f"  Signal efficiency:      {signal_efficiency:.6f}")
    print(f"  Background efficiency:  {background_efficiency:.6f}")
    print(f"  Background rejection:   {background_rejection:.6f}")
    print(f"  Balanced accuracy:      {balanced_accuracy:.6f}")
    print(f"  Missing values:         {total_missing or 'none'}")
    print(f"  Infinities converted:   {total_infinite or 'none'}")
    print("\nSaved:")
    print(f"  {model_path}")
    print(f"  {metrics_path}")
    print(f"  {roc_path}")
    if importance_summary is not None:
        print(f"  {importance_path}")


if __name__ == "__main__":
    main()
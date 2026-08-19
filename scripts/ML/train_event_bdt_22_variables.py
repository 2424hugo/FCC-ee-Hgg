#!/usr/bin/env python3
"""Train one event-level BDT from 2 event and 10-per-jet variables.

This is an ablation of the stacked ``jet BDT -> two scores -> event BDT``
architecture.  No jet-score prediction is used: the event classifier receives
the original observables directly (22 columns in total).

The script deliberately keeps training weights and evaluation weights separate.
Training weights are rescaled to give signal and background equal total weight;
validation yields and metrics use the physical weights from the normalization
manifest.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class FeatureSource:
    output_name: str
    column: str
    element: int | None = None


PREFIXES = {
    0: ("leading", "lead", "jet1", "jet_1", "j1"),
    1: ("subleading", "sublead", "jet2", "jet_2", "j2"),
}

DEFAULT_EVENT_FEATURES = ["event_invariant_mass", "n_jets_original"]
DEFAULT_JET_FEATURES = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--jet-model",
        type=Path,
        default=None,
        help="Deprecated compatibility option; the direct 22-variable model does not use it.",
    )
    parser.add_argument("--normalization-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--jet-features",
        nargs="+",
        default=DEFAULT_JET_FEATURES,
        help="The 10 base features used for both leading and subleading jets.",
    )
    parser.add_argument(
        "--event-features",
        nargs="+",
        default=DEFAULT_EVENT_FEATURES,
        help="The 2 scalar event features.",
    )
    parser.add_argument("--max-training-events-per-class", type=int, default=0)
    parser.add_argument("--max-validation-events-per-class", type=int, default=0)
    parser.add_argument("--background-systematics", type=float, nargs="+", default=[0.0, 0.0001, 0.01, 0.05, 0.10])
    parser.add_argument("--optimize-background-systematic", type=float, default=0.0001)
    parser.add_argument("--min-background-neff", type=float, default=100.0)
    parser.add_argument("--thresholds", type=int, default=2001)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=100)
    return parser.parse_args()


def feature_names_from_object(obj: Any) -> list[str] | None:
    """Find feature names in common joblib bundle/model layouts."""
    if isinstance(obj, dict):
        for key in (
            "feature_names",
            "features",
            "feature_columns",
            "input_features",
            "jet_features",
            "jet_feature_columns",
        ):
            value = obj.get(key)
            if isinstance(value, (list, tuple, np.ndarray)) and len(value):
                return [str(x) for x in value]
        for key in ("model", "classifier", "estimator", "pipeline"):
            if key in obj:
                found = feature_names_from_object(obj[key])
                if found:
                    return found
    names = getattr(obj, "feature_names_in_", None)
    if names is not None:
        return [str(x) for x in names]
    named_steps = getattr(obj, "named_steps", None)
    if named_steps:
        for step in reversed(list(named_steps.values())):
            found = feature_names_from_object(step)
            if found:
                return found
    return None


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the Parquet dataset") from exc
    return list(pq.ParquetFile(path).schema_arrow.names)


def candidate_columns(base: str, jet_index: int) -> list[str]:
    prefixes = PREFIXES[jet_index]
    suffixes = prefixes
    candidates = []
    for prefix in prefixes:
        candidates.extend((f"{prefix}_{base}", f"{prefix}.{base}", f"{prefix}/{base}"))
    for suffix in suffixes:
        candidates.extend((f"{base}_{suffix}", f"{base}.{suffix}", f"{base}/{suffix}"))
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def first_vector_length(path: Path, column: str) -> int | None:
    frame = pd.read_parquet(path, columns=[column]).head(20)
    for value in frame[column]:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            return len(value)
    return None


def resolve_feature_sources(
    event_feature_names: list[str],
    jet_feature_names: list[str],
    available: list[str],
    example_file: Path,
) -> list[FeatureSource]:
    available_set = set(available)
    sources: list[FeatureSource] = []
    unresolved: list[str] = []
    for name in event_feature_names:
        if name in available_set:
            sources.append(FeatureSource(name, name))
        else:
            unresolved.append(name)

    for base in jet_feature_names:
        pair: list[FeatureSource] = []
        for jet_index, label in ((0, "leading"), (1, "subleading")):
            matches = [name for name in candidate_columns(base, jet_index) if name in available_set]
            if len(matches) == 1:
                pair.append(FeatureSource(f"{label}_{base}", matches[0]))
            elif len(matches) > 1:
                raise RuntimeError(
                    f"Ambiguous columns for {label} jet feature {base!r}: {matches}"
                )
        if len(pair) == 2:
            sources.extend(pair)
            continue
        if base in available_set and first_vector_length(example_file, base) is not None:
            sources.extend(
                (
                    FeatureSource(f"leading_{base}", base, 0),
                    FeatureSource(f"subleading_{base}", base, 1),
                )
            )
            continue
        unresolved.append(base)

    if unresolved:
        preview = "\n  ".join(available[:120])
        raise RuntimeError(
            "Could not map these event/jet features onto Parquet columns: "
            f"{unresolved}.\nAvailable Parquet columns include:\n  {preview}\n"
            "Use --event-features/--jet-features if your schema uses different "
            "names, or rename jet columns to leading_<feature>/subleading_<feature>."
        )
    if len(sources) != 22:
        raise RuntimeError(f"Expected exactly 22 resolved inputs, found {len(sources)}")
    if len({source.output_name for source in sources}) != 22:
        raise RuntimeError("Resolved feature names are not unique")
    return sources


def belongs_to_split(path: Path, root: Path, split: str) -> bool:
    relative = path.relative_to(root)
    tokens: list[str] = []
    for part in relative.parts:
        tokens.extend(x for x in re.split(r"[^a-z0-9]+", part.lower()) if x)
    aliases = {"validation", "val"} if split == "validation" else {split}
    return bool(set(tokens) & aliases)


def sample_files(root: Path, pattern: str, split: str) -> list[Path]:
    raw_pattern = pattern if Path(pattern).is_absolute() else str(root / pattern)
    files = sorted(Path(p) for p in glob.glob(raw_pattern, recursive=True) if Path(p).is_file())
    selected = [p for p in files if belongs_to_split(p, root, split)]
    if not selected:
        raise FileNotFoundError(
            f"No {split} Parquet files matched {pattern!r} below {root}. "
            "The split name must occur in a directory or filename."
        )
    return selected


def numeric_scalar(value: Any, element: int | None) -> float:
    if element is None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan
    if value is None:
        return np.nan
    try:
        return float(value[element]) if len(value) > element else np.nan
    except (TypeError, ValueError, IndexError):
        return np.nan


def read_feature_frame(files: Iterable[Path], sources: list[FeatureSource]) -> pd.DataFrame:
    columns = list(dict.fromkeys(source.column for source in sources))
    chunks = []
    for path in files:
        raw = pd.read_parquet(path, columns=columns)
        data = {}
        for source in sources:
            series = raw[source.column]
            if source.element is None:
                data[source.output_name] = pd.to_numeric(series, errors="coerce")
            else:
                data[source.output_name] = series.map(
                    lambda value, i=source.element: numeric_scalar(value, i)
                )
        chunks.append(pd.DataFrame(data))
    frame = pd.concat(chunks, ignore_index=True)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def event_weight(sample: dict[str, Any], luminosity: float, split: str) -> float:
    generated = sample.get("generated_events", {})
    if split not in generated or float(generated[split]) <= 0:
        raise ValueError(f"Missing positive generated_events[{split!r}] for {sample.get('name')}")
    return (
        float(sample["cross_section_pb"])
        * luminosity
        * 1.0e6
        * float(sample.get("filter_efficiency", 1.0))
        * float(sample.get("k_factor", 1.0))
        * float(sample.get("branching_fraction", 1.0))
        / float(generated[split])
    )


def load_split(
    root: Path,
    manifest: dict[str, Any],
    split: str,
    sources: list[FeatureSource],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frames, labels, weights, processes = [], [], [], []
    luminosity = float(manifest["luminosity_ab_inv"])
    for sample in manifest["samples"]:
        sample_class = str(sample["class"]).lower()
        if sample_class not in {"signal", "background"}:
            raise ValueError(f"Unknown class {sample_class!r} for {sample.get('name')}")
        files = sample_files(root, str(sample["path_glob"]), split)
        frame = read_feature_frame(files, sources)
        n = len(frame)
        frames.append(frame)
        labels.append(np.full(n, sample_class == "signal", dtype=np.int8))
        weights.append(np.full(n, event_weight(sample, luminosity, split), dtype=float))
        processes.append(np.full(n, str(sample["name"]), dtype=object))
        print(
            f"{split:10s} {sample['name']:24s} rows={n:10,d} "
            f"weight={weights[-1][0]:.8g} yield={weights[-1].sum():.8g}"
        )
    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(labels),
        np.concatenate(weights),
        np.concatenate(processes),
    )


def cap_by_class(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    process: np.ndarray,
    cap: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if cap <= 0:
        return X, y, w, process
    rng = np.random.default_rng(seed)
    keep = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        if len(indices) > cap:
            indices = rng.choice(indices, size=cap, replace=False)
        keep.append(indices)
    selected = np.sort(np.concatenate(keep))
    return X.iloc[selected].reset_index(drop=True), y[selected], w[selected], process[selected]


def balanced_training_weights(y: np.ndarray, physical: np.ndarray) -> np.ndarray:
    result = np.zeros_like(physical, dtype=float)
    for label in (0, 1):
        mask = y == label
        total = physical[mask].sum()
        if total <= 0:
            raise ValueError(f"Class {label} has non-positive total training weight")
        result[mask] = physical[mask] / total
    # Mean one is convenient for sklearn's regularisation conventions.
    result *= len(result) / result.sum()
    return result


def asimov_significance(signal: float, background: float, frac_unc: float) -> float:
    if signal <= 0 or background <= 0:
        return 0.0
    if frac_unc <= 0:
        value = 2.0 * ((signal + background) * math.log1p(signal / background) - signal)
        return math.sqrt(max(value, 0.0))
    sigma2 = (frac_unc * background) ** 2
    first = (signal + background) * math.log(
        ((signal + background) * (background + sigma2))
        / (background * background + (signal + background) * sigma2)
    )
    second = (background * background / sigma2) * math.log(
        1.0 + sigma2 * signal / (background * (background + sigma2))
    )
    radicand = 2.0 * (first - second)
    if radicand > 0 and math.isfinite(radicand):
        return math.sqrt(radicand)

    # The two logarithmic terms can nearly cancel for the extreme S/B ratios
    # encountered here. Re-evaluate only those cases at higher precision.
    with localcontext() as context:
        context.prec = 50
        s, b, f = Decimal(str(signal)), Decimal(str(background)), Decimal(str(frac_unc))
        sigma2_decimal = (f * b) ** 2
        first_decimal = (s + b) * (
            ((s + b) * (b + sigma2_decimal))
            / (b * b + (s + b) * sigma2_decimal)
        ).ln()
        second_decimal = (b * b / sigma2_decimal) * (
            Decimal(1) + sigma2_decimal * s / (b * (b + sigma2_decimal))
        ).ln()
        precise = Decimal(2) * (first_decimal - second_decimal)
        return float(precise.sqrt()) if precise > 0 else 0.0


def threshold_scan(
    y: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    systematics: list[float],
    n_thresholds: int,
) -> pd.DataFrame:
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    order = np.argsort(scores)
    score_sorted, y_sorted, w_sorted = scores[order], y[order], weights[order]
    sig_w = w_sorted * (y_sorted == 1)
    bkg_w = w_sorted * (y_sorted == 0)
    bkg_w2 = bkg_w * bkg_w
    sig_tail = np.r_[np.cumsum(sig_w[::-1])[::-1], 0.0]
    bkg_tail = np.r_[np.cumsum(bkg_w[::-1])[::-1], 0.0]
    bkg2_tail = np.r_[np.cumsum(bkg_w2[::-1])[::-1], 0.0]
    index = np.searchsorted(score_sorted, thresholds, side="left")
    signal, background, sumw2 = sig_tail[index], bkg_tail[index], bkg2_tail[index]
    total_signal, total_background = sig_w.sum(), bkg_w.sum()
    data: dict[str, Any] = {
        "threshold": thresholds,
        "signal_yield": signal,
        "background_yield": background,
        "signal_efficiency": signal / total_signal,
        "background_efficiency": background / total_background,
        "background_neff": np.divide(
            background * background,
            sumw2,
            out=np.zeros_like(background),
            where=sumw2 > 0,
        ),
        "s_over_b": np.divide(signal, background, out=np.full_like(signal, np.nan), where=background > 0),
    }
    for frac in systematics:
        data[f"asimov_z_bkg_syst_{frac:g}"] = np.array(
            [asimov_significance(s, b, frac) for s, b in zip(signal, background)]
        )
    return pd.DataFrame(data)


def operating_points(
    y: np.ndarray, scores: np.ndarray, weights: np.ndarray, targets=(0.50, 0.25, 0.10, 0.01)
) -> pd.DataFrame:
    rows = []
    signal_scores = scores[y == 1]
    signal_weights = weights[y == 1]
    order = np.argsort(signal_scores)[::-1]
    cumulative = np.cumsum(signal_weights[order]) / signal_weights.sum()
    for target in targets:
        position = min(np.searchsorted(cumulative, target, side="left"), len(order) - 1)
        threshold = float(signal_scores[order[position]])
        selected = scores >= threshold
        s = weights[selected & (y == 1)].sum()
        b = weights[selected & (y == 0)].sum()
        b2 = np.square(weights[selected & (y == 0)]).sum()
        rows.append(
            {
                "target_signal_efficiency": target,
                "threshold": threshold,
                "signal_efficiency": s / weights[y == 1].sum(),
                "background_efficiency": b / weights[y == 0].sum(),
                "signal_yield": s,
                "background_yield": b,
                "background_neff": b * b / b2 if b2 > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def main() -> None:
    args = parse_args()
    if args.thresholds < 2:
        raise ValueError("--thresholds must be at least 2")
    if args.optimize_background_systematic not in args.background_systematics:
        args.background_systematics.append(args.optimize_background_systematic)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.normalization_manifest.read_text())
    event_feature_names = list(args.event_features)
    jet_feature_names = list(args.jet_features)
    if len(event_feature_names) != 2 or len(jet_feature_names) != 10:
        raise RuntimeError(
            "Expected 2 event features and 10 per-jet features "
            f"(22 inputs total), found {len(event_feature_names)} and "
            f"{len(jet_feature_names)} respectively."
        )

    first_pattern = str(manifest["samples"][0]["path_glob"])
    first_file = sample_files(args.dataset_root, first_pattern, "train")[0]
    sources = resolve_feature_sources(
        event_feature_names,
        jet_feature_names,
        parquet_columns(first_file),
        first_file,
    )
    output_names = [source.output_name for source in sources]
    print("Resolved 22 event inputs:")
    for source in sources:
        suffix = f"[{source.element}]" if source.element is not None else ""
        print(f"  {source.output_name:40s} <- {source.column}{suffix}")

    X_train, y_train, w_train_phys, p_train = load_split(
        args.dataset_root, manifest, "train", sources
    )
    X_val, y_val, w_val, p_val = load_split(
        args.dataset_root, manifest, "validation", sources
    )
    X_train, y_train, w_train_phys, p_train = cap_by_class(
        X_train, y_train, w_train_phys, p_train,
        args.max_training_events_per_class, args.random_state,
    )
    X_val, y_val, w_val, p_val = cap_by_class(
        X_val, y_val, w_val, p_val,
        args.max_validation_events_per_class, args.random_state + 1,
    )
    train_weights = balanced_training_weights(y_train, w_train_phys)

    model = HistGradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=args.random_state,
        verbose=1,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train, sample_weight=train_weights)
    fit_seconds = time.perf_counter() - start
    scores = model.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, scores, sample_weight=w_val)
    ap = average_precision_score(y_val, scores, sample_weight=w_val)
    scan = threshold_scan(
        y_val, scores, w_val, sorted(set(args.background_systematics)), args.thresholds
    )
    z_column = f"asimov_z_bkg_syst_{args.optimize_background_systematic:g}"
    eligible = scan[(scan["background_neff"] >= args.min_background_neff) & np.isfinite(scan[z_column])]
    if eligible.empty:
        raise RuntimeError(
            f"No threshold has background Neff >= {args.min_background_neff}. "
            "Increase validation statistics or lower --min-background-neff for diagnostics only."
        )
    best = eligible.loc[eligible[z_column].idxmax()]
    threshold = float(best["threshold"])

    selected = scores >= threshold
    composition_rows = []
    for process in np.unique(p_val):
        mask = selected & (p_val == process)
        all_mask = p_val == process
        sumw, sumw2 = w_val[mask].sum(), np.square(w_val[mask]).sum()
        composition_rows.append(
            {
                "process": process,
                "class": "signal" if y_val[all_mask][0] else "background",
                "yield": sumw,
                "efficiency": sumw / w_val[all_mask].sum(),
                "neff": sumw * sumw / sumw2 if sumw2 > 0 else 0.0,
            }
        )

    # Permutation importance on a bounded, class-balanced validation subset.
    X_imp, y_imp, w_imp, _ = cap_by_class(
        X_val, y_val, w_val, p_val, 25_000, args.random_state + 2
    )
    imp_weights = balanced_training_weights(y_imp, w_imp)
    importance_kwargs = dict(
        scoring="roc_auc", n_repeats=5, random_state=args.random_state, n_jobs=-1
    )
    try:
        importance = permutation_importance(
            model, X_imp, y_imp, sample_weight=imp_weights, **importance_kwargs
        )
    except TypeError:
        # ``sample_weight`` was added to permutation_importance in newer
        # scikit-learn releases. The fallback subset is class-balanced.
        importance = permutation_importance(
            model, X_imp, y_imp, **importance_kwargs
        )
    importance_frame = pd.DataFrame(
        {
            "feature": output_names,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    metrics = {
        "model": "direct_event_bdt_22_variables",
        "feature_count": len(output_names),
        "features": output_names,
        "fit_seconds": fit_seconds,
        "n_train": len(X_train),
        "n_validation": len(X_val),
        "weighted_roc_auc": auc,
        "weighted_average_precision": ap,
        "optimization_background_systematic": args.optimize_background_systematic,
        "min_background_neff": args.min_background_neff,
        "best_cut": best.to_dict(),
        "background_systematics": sorted(set(args.background_systematics)),
    }
    model_bundle = {
        "model": model,
        "feature_names": output_names,
        "event_features": event_feature_names,
        "base_jet_features": jet_feature_names,
        "feature_sources": [source.__dict__ for source in sources],
        "validation_threshold": threshold,
        "metrics": json_safe(metrics),
    }
    joblib.dump(model_bundle, args.output_dir / "event_bdt_22_variables.joblib")
    scan.to_csv(args.output_dir / "significance_scan.csv", index=False)
    operating_points(y_val, scores, w_val).to_csv(
        args.output_dir / "matched_signal_efficiency_points.csv", index=False
    )
    pd.DataFrame(composition_rows).to_csv(
        args.output_dir / "best_cut_process_composition.csv", index=False
    )
    importance_frame.to_csv(args.output_dir / "permutation_importance.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2) + "\n"
    )

    print("\nValidation result")
    print(f"  Weighted ROC AUC:             {auc:.6f}")
    print(f"  Weighted average precision:   {ap:.6g}")
    print(f"  Best threshold:               {threshold:.6f}")
    print(f"  Signal yield:                 {best['signal_yield']:.6g}")
    print(f"  Background yield:             {best['background_yield']:.6g}")
    print(f"  Signal efficiency:            {best['signal_efficiency']:.6g}")
    print(f"  Background efficiency:        {best['background_efficiency']:.6e}")
    print(f"  S/B:                          {best['s_over_b']:.6g}")
    for frac in sorted(set(args.background_systematics)):
        print(f"  Asimov Z ({100*frac:g}% bkg syst):     {best[f'asimov_z_bkg_syst_{frac:g}']:.6g}")
    print(f"  Background effective N:       {best['background_neff']:.3f}")
    print(f"\nSaved results to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
#!/usr/bin/env python3
"""Evaluate the extreme score tail of a trained H->gg event network.

This is an inference-only companion to ``train_event_nn_all_variables.py``.
It scans thresholds defined by the actual validation scores, optionally inside
reconstructed event-mass windows, and reports only operating points with enough
effective background Monte Carlo statistics to be interpretable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_event_bdt_22_variables as common
import train_event_nn_all_variables as training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--normalization-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--read-batch-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-validation-events-per-class", type=int, default=0)
    parser.add_argument("--mass-centre", type=float, default=125.0)
    parser.add_argument(
        "--mass-half-widths",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        help="Symmetric event-invariant-mass half-widths in GeV; no-window is also scanned.",
    )
    parser.add_argument("--background-systematics", type=float, nargs="+", default=[0.0, 0.0001, 0.01])
    parser.add_argument("--min-background-neff", type=float, default=100.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def normalizer_from_state(state: dict[str, list[float]]) -> training.StructuredNormalizer:
    return training.StructuredNormalizer(
        *(np.asarray(state[name], dtype=np.float32) for name in (
            "event_mean", "event_std", "jet_mean", "jet_std",
            "constituent_mean", "constituent_std",
        ))
    )


@torch.no_grad()
def predict_with_mass(model, samples, normalizer, device, args):
    scores, labels, weights, processes, masses = [], [], [], [], []
    for packed, batch_labels, batch_weights, _, batch_processes in training.iter_batches(
        samples, args, "validation", cap_per_class=args.max_validation_events_per_class
    ):
        # event_invariant_mass is the first untransformed event feature.
        masses.append(packed["event"][:, 0].astype(np.float32, copy=True))
        normalized = normalizer.apply(packed)
        batch_scores = []
        for start in range(0, len(batch_labels), args.batch_size):
            end = min(start + args.batch_size, len(batch_labels))
            logits = model(*training.tensors(training.slice_batch(normalized, start, end), device))
            batch_scores.append(torch.sigmoid(logits).cpu().numpy())
        scores.append(np.concatenate(batch_scores))
        labels.append(batch_labels.astype(np.int8))
        weights.append(batch_weights)
        processes.append(batch_processes)
    return tuple(np.concatenate(x) for x in (scores, labels, weights, processes, masses))


def exact_tail_scan(labels, scores, weights, systematics):
    # One row after every score tie: these are every distinct realizable cut.
    order = np.argsort(scores, kind="stable")[::-1]
    score = scores[order]
    y = labels[order]
    w = weights[order]
    signal = np.cumsum(w * (y == 1))
    background = np.cumsum(w * (y == 0))
    background2 = np.cumsum(np.square(w) * (y == 0))
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score) - 1]
    signal, background, background2 = signal[ends], background[ends], background2[ends]
    total_signal = weights[labels == 1].sum()
    total_background = weights[labels == 0].sum()
    frame = pd.DataFrame({
        "threshold": score[ends],
        "signal_yield": signal,
        "background_yield": background,
        "signal_efficiency": signal / total_signal,
        "background_efficiency": background / total_background,
        "background_neff": np.divide(background**2, background2, out=np.zeros_like(background), where=background2 > 0),
        "s_over_b": np.divide(signal, background, out=np.full_like(signal, np.nan), where=background > 0),
    })
    for fraction in systematics:
        frame[f"asimov_z_bkg_syst_{fraction:g}"] = [
            common.asimov_significance(float(s), float(b), fraction)
            for s, b in zip(signal, background)
        ]
    return frame


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["configuration"]
    # iter_batches reads these attributes.
    args.max_constituents = int(config["max_constituents"])
    args.type_buckets = int(config["type_buckets"])
    args.random_state = 42
    manifest = json.loads(args.normalization_manifest.read_text())
    samples = training.build_samples(args.dataset_root, manifest, "validation")
    device = training.choose_device(args.device)
    model = training.AllVariableNet(
        args.type_buckets, int(config["type_embedding_dim"]), float(config["dropout"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = normalizer_from_state(checkpoint["normalization"])
    scores, labels, weights, processes, masses = predict_with_mass(
        model, samples, normalizer, device, args
    )
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz",
        score=scores, label=labels, weight=weights, process=processes, event_invariant_mass=masses,
    )

    windows = [("none", np.ones(len(labels), dtype=bool))]
    windows += [
        (f"{args.mass_centre:g}+/-{width:g}", np.abs(masses - args.mass_centre) <= width)
        for width in args.mass_half_widths
    ]
    summaries = []
    for name, window in windows:
        if np.unique(labels[window]).size < 2:
            continue
        scan = exact_tail_scan(labels[window], scores[window], weights[window], args.background_systematics)
        scan.insert(0, "mass_window", name)
        safe_name = name.replace("+/-", "_pm_").replace(".", "p")
        scan.to_csv(args.output_dir / f"tail_scan_{safe_name}.csv", index=False)
        eligible = scan[scan.background_neff >= args.min_background_neff]
        for fraction in args.background_systematics:
            column = f"asimov_z_bkg_syst_{fraction:g}"
            if not eligible.empty:
                best = eligible.loc[eligible[column].idxmax()].to_dict()
                best.update({"optimized_systematic": fraction, "mass_window": name})
                summaries.append(best)

    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output_dir / "best_supported_operating_points.csv", index=False)
    print("\nBest statistically supported operating points")
    if summary.empty:
        print(f"  None with background Neff >= {args.min_background_neff:g}")
    else:
        print(summary[["mass_window", "optimized_systematic", "threshold", "signal_efficiency", "background_efficiency", "background_neff", "signal_yield", "background_yield", "s_over_b"]].to_string(index=False))
    print(f"\nSaved tail study to {args.output_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Evaluate a saved single-jet Particle Transformer checkpoint without training.

This script is designed for checkpoints produced by
``train_single_jet_particle_transformer_v2.py``.  It reconstructs the test
sample and model directly from the checkpoint's saved configuration, loads the
best model state, evaluates the frozen test subset, and writes:

  * metrics.json
  * predictions.csv
  * roc_curve.png
  * roc_curve_log_mistag.png
  * evaluation_configuration.json

The class convention is inherited from training: quark=0, gluon=1, so the
sigmoid score is P(gluon).

Typical use from the FCC-ee-Hgg repository root::

    python scripts/ML/evaluate_single_jet_particle_transformer.py \
        --checkpoint results/single_jet_particle_transformer/best_model.pt

By default outputs are written to an ``evaluation`` subdirectory next to the
checkpoint.  Use ``--output-dir`` to choose another location.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

# Reuse the exact dataset/model/evaluation implementation used for training.
from train_single_jet_particle_transformer_v2 import (
    SingleJetParticleTransformer,
    balanced_limit,
    choose_device,
    expand_paths,
    predict,
    save_results,
    valid_indices,
)
from single_jet_particlenet import (
    SingleJetParquetDataset,
    make_loader,
    read_jet_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved single-jet Particle Transformer checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("results/single_jet_particle_transformer/best_model.pt"),
        help="Checkpoint produced by train_single_jet_particle_transformer_v2.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Evaluation output directory. Defaults to <checkpoint-parent>/evaluation.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Evaluation device. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional evaluation batch-size override. Defaults to the training batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker override. Defaults to the training setting.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override mixed precision. By default uses the checkpoint training setting on CUDA.",
    )
    return parser.parse_args()


def require(mapping: dict, key: str):
    if key not in mapping:
        raise KeyError(
            f"Checkpoint configuration is missing required training argument {key!r}."
        )
    return mapping[key]


def main() -> None:
    cli = parse_args()
    start = time.perf_counter()

    if not cli.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {cli.checkpoint}")

    # Load on CPU first so the saved configuration can be inspected before the
    # model/device is created.
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint or "configuration" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain both 'model_state_dict' and 'configuration'."
        )

    configuration = checkpoint["configuration"]
    train_args = configuration.get("arguments")
    if not isinstance(train_args, dict):
        raise ValueError("Checkpoint configuration does not contain an arguments dictionary.")

    # ------------------------------------------------------------------
    # Reconstruct the exact frozen test subset used by the training run.
    # ------------------------------------------------------------------
    test_patterns = require(train_args, "test_parquet")
    test_paths = expand_paths(test_patterns)

    label_field = train_args.get("label_field", "jet_label")
    label_source = train_args.get("label_source", "per-jet")
    label_format = train_args.get("label_format", "binary")
    quark_pdgs = tuple(train_args.get("quark_pdgs", (1, 2, 3, 4, 5)))
    unknown_policy = train_args.get("unknown_label_policy", "error")
    seed = int(train_args.get("seed", 12345))
    limit_jets = train_args.get("limit_jets")

    label_args = dict(
        label_field=label_field,
        label_source=label_source,
        label_format=label_format,
        quark_pdgs=quark_pdgs,
    )
    test_labels = read_jet_labels(test_paths, **label_args)
    test_indices = sorted(
        balanced_limit(
            test_labels,
            valid_indices(test_labels, unknown_policy, "test"),
            limit_jets,
            seed + 2,
        )
    )

    vocabulary = tuple(configuration.get("particle_type_vocabulary", ()))
    dataset_args = dict(
        label_field=label_field,
        label_source=label_source,
        label_format=label_format,
        quark_pdgs=quark_pdgs,
        type_vocabulary=vocabulary,
        max_constituents=int(train_args.get("max_constituents", 100)),
        shard_cache_size=int(train_args.get("shard_cache_size", 2)),
    )
    test_dataset = SingleJetParquetDataset(test_paths, **dataset_args)

    selected_labels = test_labels[np.asarray(test_indices, dtype=np.int64)]
    n_quark = int(np.sum(selected_labels == 0))
    n_gluon = int(np.sum(selected_labels == 1))

    # ------------------------------------------------------------------
    # Device / loader.  Evaluation batch size may safely be overridden.
    # ------------------------------------------------------------------
    device = choose_device(cli.device)
    training_amp = bool(configuration.get("amp_enabled", False))
    amp = training_amp if cli.amp is None else bool(cli.amp)
    if device.type != "cuda":
        amp = False

    pin_memory = bool(device.type == "cuda")
    non_blocking = bool(pin_memory and device.type == "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    batch_size = int(
        cli.batch_size if cli.batch_size is not None else train_args.get("batch_size", 128)
    )
    num_workers = int(
        cli.num_workers if cli.num_workers is not None else train_args.get("num_workers", 2)
    )
    prefetch_factor = int(train_args.get("prefetch_factor", 2))

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
    )
    test_loader = make_loader(Subset(test_dataset, test_indices), **loader_kwargs)

    # ------------------------------------------------------------------
    # Reconstruct the architecture from the checkpoint, not command-line
    # guesses, then load the frozen best-model parameters.
    # ------------------------------------------------------------------
    use_pairwise_bias = not bool(train_args.get("disable_pairwise_bias", False))
    model = SingleJetParticleTransformer(
        num_particle_types=test_dataset.num_particle_types,
        type_embedding_dim=int(train_args.get("type_embedding_dim", 8)),
        d_model=int(require(train_args, "d_model")),
        num_heads=int(require(train_args, "num_heads")),
        num_layers=int(require(train_args, "num_layers")),
        ffn_dim=int(require(train_args, "ffn_dim")),
        dropout=float(train_args.get("dropout", 0.10)),
        pair_hidden_dim=int(train_args.get("pair_hidden_dim", 32)),
        use_pairwise_bias=use_pairwise_bias,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    output_dir = cli.output_dir or (cli.checkpoint.parent / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== CHECKPOINT / TEST AUDIT ===")
    print(f"Checkpoint: {cli.checkpoint}")
    print(f"Best epoch: {checkpoint.get('epoch', 'unknown')}")
    if "val_loss" in checkpoint:
        print(f"Best validation loss: {float(checkpoint['val_loss']):.6f}")
    print(
        "Architecture: "
        f"d_model={train_args['d_model']}, heads={train_args['num_heads']}, "
        f"layers={train_args['num_layers']}, ffn={train_args['ffn_dim']}, "
        f"pairwise_bias={use_pairwise_bias}"
    )
    print(f"Max constituents: {dataset_args['max_constituents']}")
    print(f"Particle type input: {'enabled' if len(vocabulary) else 'disabled'}")
    print(
        f"Test subset: quark(0)={n_quark:,}, gluon(1)={n_gluon:,}, "
        f"total={len(test_indices):,}"
    )
    print(f"Device: {device} | AMP: {amp} | batch size: {batch_size} | workers: {num_workers}")
    print("================================\n")

    labels, scores, events, jets = predict(
        model,
        test_loader,
        device,
        amp=amp,
        non_blocking=non_blocking,
    )
    metrics = save_results(labels, scores, events, jets, output_dir)

    eval_config = {
        "checkpoint": str(cli.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "device": str(device),
        "amp": amp,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "test_parquet": [str(path) for path in test_paths],
        "test_indices_count": len(test_indices),
        "test_quark": n_quark,
        "test_gluon": n_gluon,
        "training_arguments": train_args,
        "particle_type_vocabulary": list(vocabulary),
    }
    # Path objects may exist in the original argparse configuration.
    with (output_dir / "evaluation_configuration.json").open("w") as handle:
        json.dump(eval_config, handle, indent=2, default=str)

    print("Test results")
    print(f"  Accuracy: {metrics['test_accuracy']:.6f}")
    print(f"  ROC AUC:  {metrics['test_roc_auc_gluon']:.6f}")
    for target in (0.5, 0.7, 0.8, 0.9):
        key = f"quark_mistag_at_gluon_eff_{target:.1f}"
        print(f"  quark mistag @ gluon eff {target:.1f}: {metrics[key]:.6e}")
    print(f"Outputs written to: {output_dir}")
    print(f"Evaluation runtime: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
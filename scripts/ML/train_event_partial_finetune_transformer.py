#!/usr/bin/env python3
"""Partially fine-tune the jet Particle Transformer for H->gg event classification.

This script starts from the two models that have already been trained:

1. the single-jet quark/gluon Particle Transformer, and
2. the 22-high-level + two 128-d frozen-transformer-embedding event NN.

It then performs genuine end-to-end *partial* fine-tuning on the event task

    H -> gg  (signal, y=1)   versus   e+e- -> q qbar  (background, y=0),

while keeping most of the Particle Transformer frozen.

Default trainable transformer components
----------------------------------------
- final transformer attention block only
- final LayerNorm
- first Linear(128,128) projection in the transformer classifier head

The earlier transformer blocks, constituent input projection, particle-type
embedding, CLS token and pairwise-bias network remain frozen.  The final
Linear(128,1) quark/gluon classifier is bypassed: the event NN consumes the
128-dimensional pre-logit representation directly, exactly as in
``train_event_nn_with_two_transformer_embeddings.py``.

The event MLP is initialised from the already-trained 278-input hybrid model:

    [22 high-level features, h(J1)[128], h(J2)[128]]
        -> 256 -> 128 -> 64 -> 1

The preprocessing constants from that event checkpoint are kept fixed.  This
is important: at epoch zero the network exactly matches the already-validated
frozen-embedding pipeline (up to floating-point precision), and subsequent
changes can be attributed to event-level fine-tuning rather than refitting the
preprocessing transformation.

Why only the final block?
-------------------------
A full backward pass through all four transformer blocks for millions of jets
would be expensive on a Tesla T4.  Freezing the prefix means it is evaluated
under ``torch.no_grad()``, so autograd only stores activations for the final
block and event head.  This is substantially cheaper while still allowing the
constituent representation to adapt from the generic q/g objective to the
actual H->gg versus qqbar objective.

The default fine-tuning sample is intentionally smaller than the full event
training set (100k signal + 100k background events).  The test set is untouched
and, by default, evaluated in full.  If the method improves the test ROC, the
training subset can then be increased.

Expected repository files
-------------------------
Place this script in ``scripts/ML/``.  It reuses:

    scripts/ML/train_event_nn_architecture_sweep.py
    scripts/ML/train_event_nn_with_two_transformer_scores.py
    scripts/ML/train_single_jet_particle_transformer_v3_scheduler.py
    scripts/ML/single_jet_particlenet.py

Example
-------
python scripts/ML/train_event_partial_finetune_transformer.py \
  --dataset-root cache/analysis_dataset \
  --transformer-checkpoint \
    results/particle_transformer_250k_d128_h8_l4_pid_cosine/best_model.pt \
  --event-checkpoint \
    outputs/ml/nn_wide_plus_transformer_embeddings_250k/best_model.pt \
  --output-dir \
    outputs/ml/nn_transformer_partial_finetune_250k
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import Tensor, nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torch_geometric.utils import to_dense_batch

# -----------------------------------------------------------------------------
# Reuse the exact dissertation event MLP and already-validated data/model code.
# -----------------------------------------------------------------------------
try:
    from train_event_nn_architecture_sweep import (
        EventMLP,
        FEATURE_NAMES,
        fixed_signal_efficiencies,
        load_parquet_matrix,
        resolve_device,
        set_seed,
    )
except ImportError:
    from scripts.ML.train_event_nn_architecture_sweep import (
        EventMLP,
        FEATURE_NAMES,
        fixed_signal_efficiencies,
        load_parquet_matrix,
        resolve_device,
        set_seed,
    )

try:
    from train_event_nn_with_two_transformer_scores import (
        load_frozen_transformer,
        build_transformer_for_dataset,
        real_parquet_files,
        validation_directory,
    )
except ImportError:
    from scripts.ML.train_event_nn_with_two_transformer_scores import (
        load_frozen_transformer,
        build_transformer_for_dataset,
        real_parquet_files,
        validation_directory,
    )

try:
    from single_jet_particlenet import (
        SingleJetParquetDataset,
        SingleJetSample,
        SingleJetBatch,
        collate_single_jets,
    )
except ImportError:
    from scripts.ML.single_jet_particlenet import (
        SingleJetParquetDataset,
        SingleJetSample,
        SingleJetBatch,
        collate_single_jets,
    )


if len(FEATURE_NAMES) != 22:
    raise RuntimeError(f"Expected 22 dissertation high-level inputs, found {len(FEATURE_NAMES)}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    parser.add_argument("--dataset-root", type=Path, default=Path("cache/analysis_dataset"))
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--event-checkpoint",
        type=Path,
        required=True,
        help=(
            "best_model.pt from train_event_nn_with_two_transformer_embeddings.py. "
            "This supplies the pretrained 278-input event MLP and preprocessing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ml/nn_transformer_partial_finetune"),
    )

    # Event counts.  Training is balanced by construction by default.
    parser.add_argument("--train-events-per-class", type=int, default=100_000)
    parser.add_argument("--val-events-per-class", type=int, default=50_000)
    parser.add_argument(
        "--test-events-per-class",
        type=int,
        default=0,
        help="0 evaluates the full untouched test split for both classes.",
    )

    # Fine-tuning.
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=2.0e-5)
    parser.add_argument("--batch-size", type=int, default=128, help="Event batch size; transformer sees 2x jets.")
    parser.add_argument("--event-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--transformer-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=1,
        help="Number of final Particle Transformer attention blocks to fine-tune.",
    )
    parser.add_argument(
        "--unfreeze-prelogit-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also fine-tune final LayerNorm and Linear(128,128) pre-logit projection.",
    )
    parser.add_argument(
        "--lr-final-factor",
        type=float,
        default=0.15,
        help="Cosine schedule finishes at this fraction of each parameter group's initial LR.",
    )
    parser.add_argument(
        "--freeze-event-batchnorm-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep the pretrained event-NN BatchNorm running mean/variance fixed during "
            "balanced fine-tuning. Affine gamma/beta parameters remain trainable."
        ),
    )

    # Data / runtime.
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--parquet-batch-size", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision for fine-tuning/evaluation.",
    )

    return parser.parse_args()


# =============================================================================
# Event dataset: pair J1/J2 from the existing single-jet dataset
# =============================================================================


@dataclass(frozen=True)
class EventSample:
    jet1: SingleJetSample
    jet2: SingleJetSample
    high_level: Tensor
    y: Tensor


@dataclass(frozen=True)
class EventBatch:
    jets: SingleJetBatch              # contains 2*B jets ordered J1,J2,J1,J2,...
    high_level: Tensor                # [B, 22], unstandardised
    y: Tensor                         # [B]

    def to(self, device: torch.device, *, non_blocking: bool = False) -> "EventBatch":
        return EventBatch(
            jets=self.jets.to(device, non_blocking=non_blocking),
            high_level=self.high_level.to(device, non_blocking=non_blocking),
            y=self.y.to(device, non_blocking=non_blocking),
        )


class PairedEventDataset(Dataset[EventSample]):
    """Pair two selected jets from each event and attach the 22 high-level inputs."""

    def __init__(
        self,
        *,
        files: Sequence[Path],
        selected_event_indices: np.ndarray,
        selected_high_level: np.ndarray,
        event_label: int,
        transformer_train_args: dict,
        vocabulary: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.selected_event_indices = np.asarray(selected_event_indices, dtype=np.int64)
        self.high_level = np.asarray(selected_high_level, dtype=np.float32)
        self.event_label = int(event_label)

        if self.high_level.shape != (len(self.selected_event_indices), 22):
            raise ValueError(
                f"Expected selected high-level shape ({len(self.selected_event_indices)},22), "
                f"got {self.high_level.shape}"
            )

        self.jet_dataset = SingleJetParquetDataset(
            files,
            label_field=str(transformer_train_args.get("label_field", "label")),
            label_source=str(transformer_train_args.get("label_source", "event")),
            label_format=str(transformer_train_args.get("label_format", "binary")),
            quark_pdgs=tuple(transformer_train_args.get("quark_pdgs", (1, 2, 3, 4, 5))),
            type_vocabulary=vocabulary,
            max_constituents=int(transformer_train_args.get("max_constituents", 100)),
            shard_cache_size=int(transformer_train_args.get("shard_cache_size", 2)),
        )

        total_events = len(self.jet_dataset) // 2
        if 2 * total_events != len(self.jet_dataset):
            raise RuntimeError("Single-jet dataset length is not an even number of jets")
        if len(self.selected_event_indices):
            if self.selected_event_indices.min() < 0 or self.selected_event_indices.max() >= total_events:
                raise IndexError("Selected event index outside the source split")

    def __len__(self) -> int:
        return len(self.selected_event_indices)

    def source_event_index(self, local_index: int) -> int:
        return int(self.selected_event_indices[local_index])

    def shard_index(self, local_index: int) -> int:
        event = self.source_event_index(local_index)
        return int(self.jet_dataset.shard_index(2 * event))

    def __getitem__(self, local_index: int) -> EventSample:
        event = self.source_event_index(local_index)
        jet1 = self.jet_dataset[2 * event]
        jet2 = self.jet_dataset[2 * event + 1]
        if jet1.jet_index != 0 or jet2.jet_index != 1:
            raise RuntimeError("Unexpected jet ordering in SingleJetParquetDataset")
        return EventSample(
            jet1=jet1,
            jet2=jet2,
            high_level=torch.from_numpy(self.high_level[local_index]),
            y=torch.tensor(float(self.event_label), dtype=torch.float32),
        )


def collate_events(samples: Sequence[EventSample]) -> EventBatch:
    if not samples:
        raise ValueError("Cannot collate an empty event batch")
    ordered_jets: list[SingleJetSample] = []
    for sample in samples:
        ordered_jets.extend((sample.jet1, sample.jet2))
    return EventBatch(
        jets=collate_single_jets(ordered_jets),
        high_level=torch.stack([sample.high_level for sample in samples]),
        y=torch.stack([sample.y for sample in samples]),
    )


class BalancedShardBatchSampler(Sampler[list[int]]):
    """Balanced signal/background event batches while keeping I/O shard-local.

    Each batch contains half signal and half background events.  The signal half
    comes from one signal Parquet shard and the background half from one
    background shard, avoiding the severe cache thrashing caused by globally
    random event indices across class-pure shards.
    """

    def __init__(
        self,
        signal_dataset: PairedEventDataset,
        background_dataset: PairedEventDataset,
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError("Fine-tuning batch size must be an even integer >=2")
        self.signal_dataset = signal_dataset
        self.background_dataset = background_dataset
        self.batch_size = int(batch_size)
        self.half = self.batch_size // 2
        self.seed = int(seed)
        self.epoch = 0

        self.signal_groups = self._group_by_shard(signal_dataset)
        self.background_groups = self._group_by_shard(background_dataset)
        if not self.signal_groups or not self.background_groups:
            raise ValueError("Both signal and background datasets must be non-empty")

        # Equal per-class selected sizes are recommended.  Drop only incomplete
        # half-batches so BatchNorm always sees exactly balanced full batches.
        self._length = min(
            sum(len(v) // self.half for v in self.signal_groups.values()),
            sum(len(v) // self.half for v in self.background_groups.values()),
        )
        if self._length < 1:
            raise ValueError("Not enough selected events to form one balanced batch")

    @staticmethod
    def _group_by_shard(dataset: PairedEventDataset) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for local_index in range(len(dataset)):
            groups.setdefault(dataset.shard_index(local_index), []).append(local_index)
        return groups

    def __len__(self) -> int:
        return self._length

    def _chunks(self, groups: dict[int, list[int]], rng: random.Random) -> list[list[int]]:
        chunks: list[list[int]] = []
        for indices in groups.values():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            for start in range(0, len(shuffled) - self.half + 1, self.half):
                chunks.append(shuffled[start:start + self.half])
        rng.shuffle(chunks)
        return chunks

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        sig_chunks = self._chunks(self.signal_groups, rng)
        bkg_chunks = self._chunks(self.background_groups, rng)
        n_batches = min(len(sig_chunks), len(bkg_chunks))
        bkg_offset = len(self.signal_dataset)

        paired = list(zip(sig_chunks[:n_batches], bkg_chunks[:n_batches]))
        rng.shuffle(paired)
        for sig, bkg in paired:
            batch_indices = sig + [bkg_offset + index for index in bkg]
            rng.shuffle(batch_indices)
            yield batch_indices


# =============================================================================
# Hybrid model with frozen transformer prefix and trainable final block(s)
# =============================================================================


class PartialFineTuneHybrid(nn.Module):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        event_mlp: EventMLP,
        medians: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
        unfreeze_last_blocks: int,
        unfreeze_prelogit_head: bool,
        freeze_event_batchnorm_stats: bool,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.event_mlp = event_mlp
        self.embedding_dim = int(transformer.classifier[0].out_features)

        if unfreeze_last_blocks < 1 or unfreeze_last_blocks > len(transformer.blocks):
            raise ValueError(
                f"unfreeze_last_blocks must be 1..{len(transformer.blocks)}, "
                f"got {unfreeze_last_blocks}"
            )
        self.unfreeze_last_blocks = int(unfreeze_last_blocks)
        self.first_trainable_block = len(transformer.blocks) - self.unfreeze_last_blocks
        self.unfreeze_prelogit_head = bool(unfreeze_prelogit_head)
        self.freeze_event_batchnorm_stats = bool(freeze_event_batchnorm_stats)

        # Start from a completely frozen transformer, then selectively unfreeze.
        for parameter in self.transformer.parameters():
            parameter.requires_grad_(False)
        for block in self.transformer.blocks[self.first_trainable_block:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)

        if self.unfreeze_prelogit_head:
            for parameter in self.transformer.final_norm.parameters():
                parameter.requires_grad_(True)
            # classifier[0] = Linear(d,d). classifier[1] = GELU.
            for parameter in self.transformer.classifier[0].parameters():
                parameter.requires_grad_(True)

        # The original final q/g Linear(d,1) remains frozen and is not used.
        for parameter in self.transformer.classifier[-1].parameters():
            parameter.requires_grad_(False)

        # Fixed preprocessing learned from the frozen-embedding training sample.
        medians_t = torch.as_tensor(np.asarray(medians), dtype=torch.float32)
        means_t = torch.as_tensor(np.asarray(means), dtype=torch.float32)
        stds_t = torch.as_tensor(np.asarray(stds), dtype=torch.float32)
        expected = 22 + 2 * self.embedding_dim
        if medians_t.numel() != expected or means_t.numel() != expected or stds_t.numel() != expected:
            raise ValueError(
                f"Event checkpoint preprocessing must have {expected} values; "
                f"got medians={medians_t.numel()}, means={means_t.numel()}, stds={stds_t.numel()}"
            )
        self.register_buffer("medians", medians_t)
        self.register_buffer("means", means_t)
        self.register_buffer("stds", stds_t.clamp_min(1.0e-8))

    def set_mode(self, training: bool) -> None:
        # Keep the frozen transformer prefix deterministic (dropout disabled).
        self.transformer.eval()
        for block in self.transformer.blocks[self.first_trainable_block:]:
            block.train(training)
        self.transformer.final_norm.train(training and self.unfreeze_prelogit_head)
        self.transformer.classifier[0].train(training and self.unfreeze_prelogit_head)
        self.event_mlp.train(training)
        if training and self.freeze_event_batchnorm_stats:
            # Balanced fine-tuning changes the class mixture relative to the original
            # full event-NN training. Preserve the validated running statistics while
            # still allowing BatchNorm affine parameters to receive gradients.
            for module in self.event_mlp.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

    def _encode_prelogit(self, batch: SingleJetBatch) -> Tensor:
        """Return one d_model vector per jet with gradients only through final block(s)."""
        graph = batch.jets

        # Frozen input/pairwise stage.  Raw constituent tensors have no trainable
        # gradients, and the corresponding model parameters remain frozen.
        with torch.no_grad():
            dense_x, valid = to_dense_batch(graph.x, graph.batch)
            dense_pos, _ = to_dense_batch(graph.pos, graph.batch)

            features = dense_x
            if self.transformer.type_embedding is not None:
                dense_type, _ = to_dense_batch(
                    graph.particle_type,
                    graph.batch,
                    fill_value=0,
                )
                type_features = self.transformer.type_embedding(dense_type.long())
                features = torch.cat((features, type_features), dim=-1)

            tokens = self.transformer.input_projection(
                self.transformer.input_norm(features)
            )
            batch_size, particle_count, _ = tokens.shape

            cls = self.transformer.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat((cls, tokens), dim=1)
            cls_valid = torch.ones(
                batch_size, 1, dtype=torch.bool, device=valid.device
            )
            full_valid = torch.cat((cls_valid, valid), dim=1)

            full_pair_bias = None
            if self.transformer.pair_bias is not None:
                particle_bias = self.transformer.pair_bias(dense_x, dense_pos, valid)
                full_pair_bias = torch.zeros(
                    batch_size,
                    self.transformer.num_heads,
                    particle_count + 1,
                    particle_count + 1,
                    dtype=particle_bias.dtype,
                    device=particle_bias.device,
                )
                full_pair_bias[:, :, 1:, 1:] = particle_bias

            # Prefix is entirely frozen and therefore needs no autograd graph.
            for block in self.transformer.blocks[:self.first_trainable_block]:
                tokens = block(tokens, full_valid, full_pair_bias)

        # Make the boundary explicit: only the selected final blocks and later
        # modules participate in backward propagation.
        tokens = tokens.detach()
        if full_pair_bias is not None:
            full_pair_bias = full_pair_bias.detach()

        for block in self.transformer.blocks[self.first_trainable_block:]:
            tokens = block(tokens, full_valid, full_pair_bias)

        cls_embedding = self.transformer.final_norm(tokens[:, 0])
        prelogit = self.transformer.classifier[0](cls_embedding)
        prelogit = self.transformer.classifier[1](prelogit)  # GELU
        # Intentionally omit classifier[2] Dropout.  The frozen embedding model
        # was extracted in eval mode, where this Dropout was the identity.
        return prelogit

    def forward(self, batch: EventBatch) -> Tensor:
        jet_embeddings = self._encode_prelogit(batch.jets)
        n_events = batch.high_level.shape[0]
        if jet_embeddings.shape != (2 * n_events, self.embedding_dim):
            raise RuntimeError(
                f"Unexpected jet embedding shape {tuple(jet_embeddings.shape)} for {n_events} events"
            )

        paired = jet_embeddings.reshape(n_events, 2, self.embedding_dim)
        x = torch.cat(
            (batch.high_level.float(), paired[:, 0], paired[:, 1]),
            dim=1,
        )

        # Same finite-value treatment and standardisation as the frozen hybrid.
        x = torch.where(torch.isfinite(x), x, self.medians.unsqueeze(0))
        x = (x - self.means.unsqueeze(0)) / self.stds.unsqueeze(0)
        return self.event_mlp(x).reshape(-1)

    def transformer_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def event_trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.event_mlp.parameters() if p.requires_grad]


# =============================================================================
# Data preparation helpers
# =============================================================================


def choose_indices(total: int, requested: int, seed: int) -> np.ndarray:
    if requested <= 0 or requested >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = rng.choice(total, size=requested, replace=False)
    return np.sort(selected.astype(np.int64, copy=False))


def load_selected_class_dataset(
    *,
    files: list[Path],
    sample_name: str,
    split_name: str,
    requested_events: int,
    seed: int,
    label: int,
    transformer_train_args: dict,
    vocabulary: tuple[int, ...],
    parquet_batch_size: int,
) -> PairedEventDataset:
    matrix = load_parquet_matrix(
        files,
        0,
        parquet_batch_size,
        f"{sample_name} {split_name} high-level features",
    )
    total = len(matrix)
    indices = choose_indices(total, requested_events, seed)
    selected = np.asarray(matrix[indices], dtype=np.float32).copy()
    del matrix

    print(
        f"{sample_name:10s} {split_name:10s}: selected {len(indices):,}/{total:,} events"
    )
    return PairedEventDataset(
        files=files,
        selected_event_indices=indices,
        selected_high_level=selected,
        event_label=label,
        transformer_train_args=transformer_train_args,
        vocabulary=vocabulary,
    )


def make_train_loader(
    signal_dataset: PairedEventDataset,
    background_dataset: PairedEventDataset,
    *,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    combined = ConcatDataset((signal_dataset, background_dataset))
    sampler = BalancedShardBatchSampler(
        signal_dataset,
        background_dataset,
        batch_size=batch_size,
        seed=seed,
    )
    kwargs: dict[str, object] = {}
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(
        combined,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=collate_events,
        **kwargs,
    )


def make_eval_loader(
    signal_dataset: PairedEventDataset,
    background_dataset: PairedEventDataset,
    *,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
    combined = ConcatDataset((signal_dataset, background_dataset))
    kwargs: dict[str, object] = {}
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=collate_events,
        **kwargs,
    )


# =============================================================================
# Training / evaluation
# =============================================================================


def cosine_factor(epoch_index: int, total_epochs: int, final_factor: float) -> float:
    if total_epochs <= 1:
        return float(final_factor)
    progress = epoch_index / max(total_epochs - 1, 1)
    return final_factor + (1.0 - final_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_epoch(
    *,
    model: PartialFineTuneHybrid,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp: bool,
    gradient_clip: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.set_mode(training)
    non_blocking = device.type == "cuda"

    total_loss = 0.0
    total_events = 0
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            batch = batch.to(device, non_blocking=non_blocking)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                logits = model(batch)
                loss = criterion(logits, batch.y)

            if training:
                assert optimizer is not None
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            [p for group in optimizer.param_groups for p in group["params"]],
                            gradient_clip,
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            [p for group in optimizer.param_groups for p in group["params"]],
                            gradient_clip,
                        )
                    optimizer.step()

            n = len(batch.y)
            total_loss += float(loss.detach().item()) * n
            total_events += n
            all_scores.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
            all_labels.append(batch.y.detach().float().cpu().numpy())

    return (
        total_loss / max(total_events, 1),
        np.concatenate(all_scores),
        np.concatenate(all_labels),
    )


def evaluate_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return (
        float(roc_auc_score(labels, scores)),
        float(average_precision_score(labels, scores)),
    )


def operating_points_frame(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    points = fixed_signal_efficiencies(labels, scores, targets=[0.5, 0.7, 0.8, 0.9])
    rows = []
    for target, point in points.items():
        rows.append(
            {
                "target_signal_efficiency": target,
                "signal_efficiency": point["signal_efficiency"],
                "background_efficiency": point["background_efficiency"],
                "background_rejection": point["background_rejection"],
                "threshold": point["threshold"],
            }
        )
    return pd.DataFrame(rows)


def save_roc(labels: np.ndarray, scores: np.ndarray, auc: float, output: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.6f}")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set_xlabel("Background efficiency")
    ax.set_ylabel("Signal efficiency")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.random_seed)

    device = resolve_device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    pin_memory = device.type == "cuda"

    print("=" * 92)
    print("PARTIAL END-TO-END FINE-TUNING: PARTICLE TRANSFORMER + EVENT NN")
    print("=" * 92)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"Training subset: {args.train_events_per_class:,} signal + "
        f"{args.train_events_per_class:,} background events"
    )
    print(
        f"Validation subset: {args.val_events_per_class:,} signal + "
        f"{args.val_events_per_class:,} background events"
    )
    print(
        "Test: " + (
            "full untouched split" if args.test_events_per_class <= 0
            else f"{args.test_events_per_class:,} events per class"
        )
    )

    # ------------------------------------------------------------------
    # Load pretrained transformer checkpoint/configuration.
    # ------------------------------------------------------------------
    transformer_checkpoint, transformer_configuration, transformer_train_args, vocabulary, amp_stored = (
        load_frozen_transformer(args.transformer_checkpoint, device)
    )

    # ------------------------------------------------------------------
    # Load pretrained event embedding checkpoint and fixed preprocessing.
    # ------------------------------------------------------------------
    if not args.event_checkpoint.is_file():
        raise FileNotFoundError(f"Event checkpoint not found: {args.event_checkpoint}")
    event_checkpoint = torch.load(args.event_checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in event_checkpoint or "preprocessing" not in event_checkpoint:
        raise ValueError("Event checkpoint must contain model_state_dict and preprocessing")

    embedding_dim = int(transformer_train_args.get("d_model", 0))
    input_dim = 22 + 2 * embedding_dim
    if int(event_checkpoint.get("input_dim", -1)) != input_dim:
        raise ValueError(
            f"Event checkpoint input_dim={event_checkpoint.get('input_dim')} but expected {input_dim}"
        )
    transformer_meta = event_checkpoint.get("transformer", {})
    if transformer_meta.get("embedding_source", "prelogit") != "prelogit":
        raise ValueError("Fine-tuning script expects an event checkpoint trained with prelogit embeddings")

    hidden_dims = list(event_checkpoint.get("hidden_dims", [256, 128, 64]))
    dropout = float(event_checkpoint.get("dropout", 0.15))
    batch_norm = bool(event_checkpoint.get("batch_norm", True))
    activation = str(event_checkpoint.get("activation", "relu"))

    event_mlp = EventMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        batch_norm=batch_norm,
        activation=activation,
    ).to(device)
    event_mlp.load_state_dict(event_checkpoint["model_state_dict"], strict=True)

    preprocessing = event_checkpoint["preprocessing"]

    # ------------------------------------------------------------------
    # Resolve source Parquet splits.
    # ------------------------------------------------------------------
    signal_root = args.dataset_root / "signal"
    background_root = args.dataset_root / "background"
    split_files = {
        "signal_train": real_parquet_files(signal_root / "train"),
        "background_train": real_parquet_files(background_root / "train"),
        "signal_validation": real_parquet_files(validation_directory(signal_root)),
        "background_validation": real_parquet_files(validation_directory(background_root)),
        "signal_test": real_parquet_files(signal_root / "test"),
        "background_test": real_parquet_files(background_root / "test"),
    }

    # Build one tiny jet dataset first so the transformer obtains exactly the
    # same particle-type vocabulary size as during training.
    tiny_dataset = SingleJetParquetDataset(
        split_files["signal_train"][:1],
        label_field=str(transformer_train_args.get("label_field", "label")),
        label_source=str(transformer_train_args.get("label_source", "event")),
        label_format=str(transformer_train_args.get("label_format", "binary")),
        quark_pdgs=tuple(transformer_train_args.get("quark_pdgs", (1, 2, 3, 4, 5))),
        type_vocabulary=vocabulary,
        max_constituents=int(transformer_train_args.get("max_constituents", 100)),
        shard_cache_size=int(transformer_train_args.get("shard_cache_size", 2)),
    )
    transformer = build_transformer_for_dataset(
        transformer_checkpoint,
        transformer_train_args,
        tiny_dataset,
        device,
    )
    del tiny_dataset

    model = PartialFineTuneHybrid(
        transformer=transformer,
        event_mlp=event_mlp,
        medians=np.asarray(preprocessing["medians"]),
        means=np.asarray(preprocessing["means"]),
        stds=np.asarray(preprocessing["stds"]),
        unfreeze_last_blocks=args.unfreeze_last_blocks,
        unfreeze_prelogit_head=args.unfreeze_prelogit_head,
        freeze_event_batchnorm_stats=args.freeze_event_batchnorm_stats,
    ).to(device)

    transformer_trainable = model.transformer_trainable_parameters()
    event_trainable = model.event_trainable_parameters()
    print("\nInitialisation")
    print(f"  transformer checkpoint: {args.transformer_checkpoint}")
    print(f"  event checkpoint:       {args.event_checkpoint}")
    print(f"  starting event test AUC: {event_checkpoint.get('test_auc', float('nan'))}")
    print(f"  unfreezing final transformer block(s): {args.unfreeze_last_blocks}")
    print(f"  final LayerNorm/prelogit projection trainable: {args.unfreeze_prelogit_head}")
    print(f"  freeze event BatchNorm running stats:          {args.freeze_event_batchnorm_stats}")
    print(f"  transformer trainable parameters: {sum(p.numel() for p in transformer_trainable):,}")
    print(f"  event-NN trainable parameters:     {sum(p.numel() for p in event_trainable):,}")
    print(
        f"  LRs: transformer={args.transformer_learning_rate:.2e}, "
        f"event={args.event_learning_rate:.2e}"
    )

    # ------------------------------------------------------------------
    # Build selected train/validation datasets.
    # ------------------------------------------------------------------
    print("\nPreparing fine-tuning datasets")
    signal_train = load_selected_class_dataset(
        files=split_files["signal_train"],
        sample_name="signal",
        split_name="train",
        requested_events=args.train_events_per_class,
        seed=args.random_seed,
        label=1,
        transformer_train_args=transformer_train_args,
        vocabulary=vocabulary,
        parquet_batch_size=args.parquet_batch_size,
    )
    background_train = load_selected_class_dataset(
        files=split_files["background_train"],
        sample_name="background",
        split_name="train",
        requested_events=args.train_events_per_class,
        seed=args.random_seed + 11,
        label=0,
        transformer_train_args=transformer_train_args,
        vocabulary=vocabulary,
        parquet_batch_size=args.parquet_batch_size,
    )
    signal_val = load_selected_class_dataset(
        files=split_files["signal_validation"],
        sample_name="signal",
        split_name="validation",
        requested_events=args.val_events_per_class,
        seed=args.random_seed + 101,
        label=1,
        transformer_train_args=transformer_train_args,
        vocabulary=vocabulary,
        parquet_batch_size=args.parquet_batch_size,
    )
    background_val = load_selected_class_dataset(
        files=split_files["background_validation"],
        sample_name="background",
        split_name="validation",
        requested_events=args.val_events_per_class,
        seed=args.random_seed + 111,
        label=0,
        transformer_train_args=transformer_train_args,
        vocabulary=vocabulary,
        parquet_batch_size=args.parquet_batch_size,
    )

    train_loader = make_train_loader(
        signal_train,
        background_train,
        batch_size=args.batch_size,
        workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.random_seed,
        pin_memory=pin_memory,
    )
    val_loader = make_eval_loader(
        signal_val,
        background_val,
        batch_size=args.batch_size,
        workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=pin_memory,
    )

    # Balanced fine-tuning subset -> pos_weight = 1.  The untouched test AUC
    # remains prior-independent.  If unequal class sizes are selected later,
    # automatically restore the corresponding positive-class weight.
    pos_weight_value = len(background_train) / max(len(signal_train), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": event_trainable,
                "lr": args.event_learning_rate,
                "name": "event_head",
            },
            {
                "params": transformer_trainable,
                "lr": args.transformer_learning_rate,
                "name": "transformer_tail",
            },
        ],
        weight_decay=args.weight_decay,
    )

    def lr_lambda(epoch_index: int) -> float:
        return cosine_factor(epoch_index, args.epochs, args.lr_final_factor)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lr_lambda, lr_lambda],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    print("\nFine-tuning")
    print(
        f"  balanced batches: ~{args.batch_size//2} signal + "
        f"{args.batch_size//2} background events ({args.batch_size*2} jets/batch)"
    )
    print(f"  batches/epoch: {len(train_loader):,}")
    print(f"  validation events: {len(signal_val)+len(background_val):,}")
    print(f"  AMP: {amp}")

    # Epoch-zero sanity check: before any parameter update this should closely
    # reproduce the frozen 278-input embedding model on the selected validation
    # subset. A large discrepancy indicates ordering/preprocessing mismatch.
    initial_val_loss, initial_val_scores, initial_val_labels = run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        amp=amp,
        gradient_clip=0.0,
    )
    initial_val_auc, initial_val_ap = evaluate_metrics(initial_val_labels, initial_val_scores)
    print(
        f"  epoch-000 sanity check | val loss={initial_val_loss:.6f} | "
        f"val AUC={initial_val_auc:.6f} | val AP={initial_val_ap:.6f}"
    )

    history_rows: list[dict[str, float | int]] = []
    best_auc = -float("inf")
    best_epoch = 0
    best_state: dict | None = None
    stale = 0
    training_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            gradient_clip=args.gradient_clip,
        )
        val_loss, val_scores, val_labels = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            amp=amp,
            gradient_clip=0.0,
        )
        val_auc, val_ap = evaluate_metrics(val_labels, val_scores)
        elapsed = time.perf_counter() - epoch_start

        event_lr = optimizer.param_groups[0]["lr"]
        transformer_lr = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch {epoch:03d} | "
            f"lr_event={event_lr:.2e}, lr_T={transformer_lr:.2e} | "
            f"train loss={train_loss:.6f} | val loss={val_loss:.6f} | "
            f"val AUC={val_auc:.6f} | val AP={val_ap:.6f} | {elapsed:.1f}s"
        )

        history_rows.append(
            {
                "epoch": epoch,
                "event_lr": event_lr,
                "transformer_lr": transformer_lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_ap": val_ap,
                "seconds": elapsed,
            }
        )

        if val_auc > best_auc + args.min_delta:
            best_auc = val_auc
            best_epoch = epoch
            stale = 0
            best_state = {
                "transformer_state_dict": copy.deepcopy(model.transformer.state_dict()),
                "event_model_state_dict": copy.deepcopy(model.event_mlp.state_dict()),
            }
        else:
            stale += 1

        scheduler.step()
        if stale >= args.patience:
            print(
                f"Early stopping after epoch {epoch}: validation AUC did not improve "
                f"for {args.patience} epochs."
            )
            break

    training_seconds = time.perf_counter() - training_start
    if best_state is None:
        raise RuntimeError("Fine-tuning did not produce a valid checkpoint")

    model.transformer.load_state_dict(best_state["transformer_state_dict"], strict=True)
    model.event_mlp.load_state_dict(best_state["event_model_state_dict"], strict=True)
    print(f"Loaded best fine-tuned checkpoint from epoch {best_epoch} (val AUC={best_auc:.6f}).")

    # ------------------------------------------------------------------
    # Final validation and untouched test evaluation.
    # ------------------------------------------------------------------
    print("\nPreparing final test dataset")
    signal_test = load_selected_class_dataset(
        files=split_files["signal_test"],
        sample_name="signal",
        split_name="test",
        requested_events=args.test_events_per_class,
        seed=args.random_seed + 201,
        label=1,
        transformer_train_args=transformer_train_args,
        vocabulary=vocabulary,
        parquet_batch_size=args.parquet_batch_size,
    )
    background_test = load_selected_class_dataset(
        files=split_files["background_test"],
        sample_name="background",
        split_name="test",
        requested_events=args.test_events_per_class,
        seed=args.random_seed + 211,
        label=0,
        transformer_train_args=transformer_train_args,
        vocabulary=vocabulary,
        parquet_batch_size=args.parquet_batch_size,
    )
    test_loader = make_eval_loader(
        signal_test,
        background_test,
        batch_size=args.batch_size,
        workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=pin_memory,
    )

    val_loss, val_scores, val_labels = run_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        amp=amp,
        gradient_clip=0.0,
    )
    test_loss, test_scores, test_labels = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        amp=amp,
        gradient_clip=0.0,
    )

    val_auc, val_ap = evaluate_metrics(val_labels, val_scores)
    test_auc, test_ap = evaluate_metrics(test_labels, test_scores)
    val_ops = operating_points_frame(val_labels, val_scores)
    test_ops = operating_points_frame(test_labels, test_scores)

    # ------------------------------------------------------------------
    # Outputs.
    # ------------------------------------------------------------------
    history = pd.DataFrame(history_rows)
    history.to_csv(args.output_dir / "training_history.csv", index=False)
    val_ops.to_csv(args.output_dir / "validation_operating_points.csv", index=False)
    test_ops.to_csv(args.output_dir / "test_operating_points.csv", index=False)
    pd.DataFrame(
        {
            "label": test_labels.astype(np.int8),
            "event_score": test_scores.astype(np.float32),
        }
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)

    save_roc(
        val_labels,
        val_scores,
        val_auc,
        args.output_dir / "validation_roc.png",
        "Partially fine-tuned transformer + event NN: validation ROC",
    )
    save_roc(
        test_labels,
        test_scores,
        test_auc,
        args.output_dir / "test_roc.png",
        "Partially fine-tuned transformer + event NN: test ROC",
    )

    if len(history):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(history["epoch"], history["train_loss"], label="train loss")
        ax.plot(history["epoch"], history["val_loss"], label="validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE loss")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "training_loss.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(history["epoch"], history["val_auc"], label="validation AUC")
        ax.axhline(
            float(event_checkpoint.get("validation_auc", np.nan)),
            linestyle="--",
            label="frozen-embedding baseline",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ROC AUC")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "validation_auc.png", dpi=200)
        plt.close(fig)

    checkpoint_out = {
        "architecture": "partial_finetune_particle_transformer_plus_event_nn",
        "transformer_state_dict": best_state["transformer_state_dict"],
        "event_model_state_dict": best_state["event_model_state_dict"],
        "transformer_source_checkpoint": str(args.transformer_checkpoint),
        "event_source_checkpoint": str(args.event_checkpoint),
        "event_input_dim": input_dim,
        "event_hidden_dims": hidden_dims,
        "preprocessing": preprocessing,
        "unfreeze_last_blocks": args.unfreeze_last_blocks,
        "unfreeze_prelogit_head": args.unfreeze_prelogit_head,
        "freeze_event_batchnorm_stats": args.freeze_event_batchnorm_stats,
        "best_epoch": best_epoch,
        "validation_auc": val_auc,
        "test_auc": test_auc,
        "test_average_precision": test_ap,
        "arguments": vars(args),
    }
    torch.save(checkpoint_out, args.output_dir / "best_model.pt")

    metrics = {
        "model": "partial_finetune_particle_transformer_plus_event_nn",
        "baseline_frozen_embedding_test_auc": float(event_checkpoint.get("test_auc", float("nan"))),
        "best_epoch": best_epoch,
        "validation_loss": val_loss,
        "validation_auc": val_auc,
        "validation_average_precision": val_ap,
        "test_loss": test_loss,
        "test_auc": test_auc,
        "test_average_precision": test_ap,
        "test_operating_points": test_ops.to_dict(orient="records"),
        "train_events_signal": len(signal_train),
        "train_events_background": len(background_train),
        "validation_events_signal": len(signal_val),
        "validation_events_background": len(background_val),
        "test_events_signal": len(signal_test),
        "test_events_background": len(background_test),
        "transformer_trainable_parameters": sum(p.numel() for p in transformer_trainable),
        "event_trainable_parameters": sum(p.numel() for p in event_trainable),
        "training_seconds": training_seconds,
        "initial_validation_auc": initial_val_auc,
        "event_learning_rate": args.event_learning_rate,
        "transformer_learning_rate": args.transformer_learning_rate,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    print("\n" + "=" * 92)
    print("RESULT")
    print("=" * 92)
    print(f"Best fine-tuning epoch: {best_epoch}")
    print(f"Validation ROC AUC:     {val_auc:.6f}")
    print(f"Test ROC AUC:           {test_auc:.6f}")
    print(f"Test average precision: {test_ap:.6f}")
    baseline = float(event_checkpoint.get("test_auc", float("nan")))
    if np.isfinite(baseline):
        print(f"Frozen-embedding baseline test AUC: {baseline:.6f}")
        print(f"Delta test AUC:                    {test_auc - baseline:+.6f}")

    print("\nTest operating points")
    for row in test_ops.to_dict(orient="records"):
        print(
            f"  eps_s={row['signal_efficiency']:.3f} | "
            f"eps_b={row['background_efficiency']:.6e} | "
            f"R_b={row['background_rejection']:.1f}"
        )
    print(f"\nOutputs: {args.output_dir}")
    print(f"Fine-tuning time: {training_seconds:.1f}s")


if __name__ == "__main__":
    main()

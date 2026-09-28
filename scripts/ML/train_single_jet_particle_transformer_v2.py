#!/usr/bin/env python3
"""Train a single-jet Particle Transformer quark/gluon classifier.

This program is designed as a drop-in comparison with the existing
``single_jet_particlenet.py`` FCC-ee pipeline.  It reuses the same Parquet
loader and constituent representation, but replaces ParticleNet EdgeConv
blocks with physics-aware multi-head self-attention.

Class convention
----------------
quark = 0
gluon = 1
The returned sigmoid score is therefore P(gluon).

Architecture
------------
Each jet constituent is represented by the same seven continuous features used
by the existing ParticleNet dataset:

    [delta_eta, delta_phi, log(pt), log(E),
     log(pt / jet_pt), log(E / jet_E), charge]

Optionally, reconstructed particle type is embedded and concatenated before the
input projection.  A learned CLS token summarizes the jet after several
pre-normalized transformer blocks.

The attention logits are augmented by a learned pairwise physics bias derived
from four constituent-pair quantities inspired by Particle Transformer models:

    log(DeltaR), log(kT), log(z), log(m_pair^2)

where constituents are treated as approximately massless for the pair-mass
feature.

Example
-------
python train_single_jet_particle_transformer.py \
    --train-parquet 'cache/analysis_dataset/signal/train/*.parquet' \
                    'cache/analysis_dataset/background/train/*.parquet' \
    --val-parquet   'cache/analysis_dataset/signal/val/*.parquet' \
                    'cache/analysis_dataset/background/val/*.parquet' \
    --test-parquet  'cache/analysis_dataset/signal/test/*.parquet' \
                    'cache/analysis_dataset/background/test/*.parquet' \
    --label-field label --label-source event \
    --epochs 30 --batch-size 64 --max-constituents 100 \
    --d-model 64 --num-heads 4 --num-layers 3 --ffn-dim 256 \
    --amp

For a quick class-balanced debugging run, add for example:

    --limit-jets 50000

Requirements
------------
torch, torch-geometric, numpy, scikit-learn, matplotlib, awkward
and the existing ``single_jet_particlenet.py`` module in the Python path.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from torch import Tensor, nn
from torch.utils.data import Sampler, Subset
from torch_geometric.utils import to_dense_batch

from single_jet_particlenet import (
    SingleJetBatch,
    SingleJetParquetDataset,
    infer_particle_type_vocabulary,
    make_loader,
    read_jet_labels,
)


# -----------------------------------------------------------------------------
# Command-line configuration
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a physics-aware single-jet Particle Transformer tagger."
    )

    # Dataset / labels ---------------------------------------------------------
    parser.add_argument("--train-parquet", nargs="+", required=True)
    parser.add_argument("--val-parquet", nargs="+", required=True)
    parser.add_argument("--test-parquet", nargs="+", required=True)
    parser.add_argument(
        "--label-field",
        default="jet_label",
        help="Two-entry per-jet truth field, or scalar event truth with --label-source event.",
    )
    parser.add_argument(
        "--label-source",
        choices=("per-jet", "event"),
        default="per-jet",
    )
    parser.add_argument("--label-format", choices=("binary", "pdg"), default="binary")
    parser.add_argument("--quark-pdgs", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument(
        "--unknown-label-policy",
        choices=("error", "drop"),
        default="error",
    )

    # Training ----------------------------------------------------------------
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/single_jet_particle_transformer"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-constituents", type=int, default=100)
    parser.add_argument("--shard-cache-size", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatic mixed precision; defaults to enabled on CUDA.",
    )
    parser.add_argument("--use-particle-type", action="store_true")
    parser.add_argument("--particle-types", nargs="+", type=int, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument(
        "--limit-jets",
        type=int,
        default=None,
        help="Optional approximately class-balanced limit applied separately to each split.",
    )

    # Particle Transformer ----------------------------------------------------
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--type-embedding-dim", type=int, default=8)
    parser.add_argument("--pair-hidden-dim", type=int, default=32)
    parser.add_argument(
        "--disable-pairwise-bias",
        action="store_true",
        help="Use ordinary self-attention without the learned pairwise physics bias.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Dataset helpers -- intentionally mirror the existing ParticleNet training
# script so that the comparison uses the same data-selection logic.
# -----------------------------------------------------------------------------


def expand_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        paths.extend(matches or ([Path(pattern)] if Path(pattern).is_file() else []))
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError(f"No Parquet files matched: {list(patterns)}")
    return unique


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def valid_indices(labels: np.ndarray, policy: str, split: str) -> list[int]:
    bad = np.flatnonzero(labels < 0)
    if bad.size and policy == "error":
        raise ValueError(
            f"{split} contains {len(bad)} unsupported jet labels; examples: "
            f"{bad[:10].tolist()}. Use --unknown-label-policy drop only if intended."
        )
    return np.flatnonzero(labels >= 0).tolist()


def balanced_limit(
    labels: np.ndarray,
    indices: Sequence[int],
    limit: int | None,
    seed: int,
) -> list[int]:
    if limit is None or limit >= len(indices):
        return list(indices)
    if limit < 2:
        raise ValueError("--limit-jets must be at least 2")

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in (0, 1):
        candidates = np.asarray(
            [index for index in indices if labels[index] == label], dtype=np.int64
        )
        rng.shuffle(candidates)
        selected.extend(candidates[: limit // 2].tolist())

    remaining = limit - len(selected)
    if remaining:
        rest = np.setdiff1d(np.asarray(indices), np.asarray(selected))
        rng.shuffle(rest)
        selected.extend(rest[:remaining].tolist())

    rng.shuffle(selected)
    if len(selected) < 2 or len(np.unique(labels[selected])) != 2:
        raise ValueError("Selected sample must contain both quark and gluon jets")
    return selected


class MixedClassShardBatchSampler(Sampler[list[int]]):
    """Create mixed quark/gluon batches while retaining shard locality.

    The FCC-ee cache stores signal and background in separate Parquet shards.  A
    purely shard-local sampler therefore produces long runs of one-class batches.
    This sampler constructs a shuffled stream for each class, keeping each stream
    locally grouped by shard, then draws the appropriate number of examples from
    both streams for every mini-batch.  For a balanced ``--limit-jets`` sample a
    batch of 128 therefore contains approximately 64 quark and 64 gluon jets.

    The per-batch class fraction follows the selected training sample rather than
    forcing 50/50 for an imbalanced full-data run; BCE ``pos_weight`` therefore
    retains its intended meaning.
    """

    def __init__(
        self,
        dataset: SingleJetParquetDataset,
        indices: Sequence[int],
        labels: np.ndarray,
        batch_size: int,
        seed: int,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2 for mixed-class batches")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels[self.indices], dtype=np.int8)

        unique = np.unique(self.labels)
        if not np.array_equal(unique, np.array([0, 1], dtype=np.int8)):
            raise ValueError(f"Training selection must contain labels 0 and 1, got {unique.tolist()}")

        groups: dict[int, dict[int, list[int]]] = {0: {}, 1: {}}
        for subset_position, (dataset_index, label) in enumerate(zip(self.indices, self.labels)):
            shard = dataset.shard_index(int(dataset_index))
            groups[int(label)].setdefault(shard, []).append(subset_position)
        self.groups = {
            label: {shard: np.asarray(pos, dtype=np.int64) for shard, pos in shard_groups.items()}
            for label, shard_groups in groups.items()
        }

        self.class_counts = {label: int(np.sum(self.labels == label)) for label in (0, 1)}
        positive_fraction = self.class_counts[1] / len(self.indices)
        self.n_positive = int(round(self.batch_size * positive_fraction))
        self.n_positive = min(max(self.n_positive, 1), self.batch_size - 1)
        self.n_negative = self.batch_size - self.n_positive

    def _class_stream(self, label: int, rng: np.random.Generator) -> np.ndarray:
        shard_groups = self.groups[label]
        pieces: list[np.ndarray] = []
        shard_ids = list(shard_groups)
        rng.shuffle(shard_ids)
        for shard in shard_ids:
            values = shard_groups[shard].copy()
            rng.shuffle(values)
            pieces.append(values)
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        neg = self._class_stream(0, rng)
        pos = self._class_stream(1, rng)
        i0 = i1 = 0

        while i0 < len(neg) or i1 < len(pos):
            remaining0 = len(neg) - i0
            remaining1 = len(pos) - i1
            if remaining0 + remaining1 <= self.batch_size:
                batch = np.concatenate((neg[i0:], pos[i1:]))
                if len(batch) >= 2 and remaining0 and remaining1:
                    rng.shuffle(batch)
                    yield batch.tolist()
                elif len(batch):
                    # Only a tiny rounding remainder should reach this branch.
                    rng.shuffle(batch)
                    yield batch.tolist()
                break

            take0 = min(self.n_negative, remaining0)
            take1 = min(self.n_positive, remaining1)

            # If one stream is close to exhaustion, fill the unused slots from
            # the other while retaining at least one example from each class when possible.
            missing = self.batch_size - take0 - take1
            if missing > 0:
                extra0 = min(missing, remaining0 - take0)
                take0 += extra0
                missing -= extra0
            if missing > 0:
                extra1 = min(missing, remaining1 - take1)
                take1 += extra1

            batch = np.concatenate((neg[i0:i0 + take0], pos[i1:i1 + take1]))
            i0 += take0
            i1 += take1
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


# -----------------------------------------------------------------------------
# Particle Transformer
# -----------------------------------------------------------------------------


class PairwisePhysicsBias(nn.Module):
    """Map pairwise constituent kinematics to one additive bias per attention head.

    Input features for constituent pair (i, j):

        log(DeltaR_ij)
        log(kT_ij) = log(min(pt_i, pt_j) * DeltaR_ij)
        log(z_ij)  = log(min(pt_i, pt_j) / (pt_i + pt_j))
        log(m_ij^2) for a massless two-particle approximation

    The output has shape [B, H, N, N] and is added directly to attention logits.
    """

    def __init__(self, num_heads: int, hidden_dim: int = 32, eps: float = 1.0e-8):
        super().__init__()
        self.num_heads = num_heads
        self.eps = eps
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, x: Tensor, pos: Tensor, valid: Tensor) -> Tensor:
        # x:   [B, N, F], with x[..., 2] = log(pt)
        # pos: [B, N, 2] = (delta_eta, delta_phi)
        # valid: [B, N]
        deta = pos[:, :, None, 0] - pos[:, None, :, 0]
        raw_dphi = pos[:, :, None, 1] - pos[:, None, :, 1]
        dphi = torch.atan2(torch.sin(raw_dphi), torch.cos(raw_dphi))

        dr2 = deta.square() + dphi.square()
        log_dr = 0.5 * torch.log(dr2.clamp_min(self.eps))

        log_pt = x[..., 2]
        log_pt_i = log_pt[:, :, None]
        log_pt_j = log_pt[:, None, :]
        min_log_pt = torch.minimum(log_pt_i, log_pt_j)
        log_sum_pt = torch.logaddexp(log_pt_i, log_pt_j)

        log_kt = min_log_pt + log_dr
        log_z = min_log_pt - log_sum_pt

        # m_ij^2 = 2 pt_i pt_j [cosh(Delta eta) - cos(Delta phi)]
        angular = (
            torch.cosh(deta.clamp(min=-10.0, max=10.0)) - torch.cos(dphi)
        ).clamp_min(self.eps)
        log_m2 = math.log(2.0) + log_pt_i + log_pt_j + torch.log(angular)

        pair_features = torch.stack((log_dr, log_kt, log_z, log_m2), dim=-1)

        # Keep the MLP numerically well behaved for coincident / very soft pairs.
        pair_features = torch.nan_to_num(
            pair_features, nan=0.0, posinf=30.0, neginf=-30.0
        ).clamp(min=-30.0, max=30.0)

        bias = self.net(pair_features)  # [B, N, N, H]
        bias = bias.permute(0, 3, 1, 2).contiguous()  # [B, H, N, N]

        pair_valid = valid[:, None, :, None] & valid[:, None, None, :]
        return bias.masked_fill(~pair_valid, 0.0)


class ParticleAttentionBlock(nn.Module):
    """Pre-normalized self-attention block with optional pairwise physics bias."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        tokens: Tensor,
        valid: Tensor,
        pair_bias: Tensor | None,
    ) -> Tensor:
        # valid: [B, L], True for CLS/real particles, False for padding.
        batch_size, length, _ = tokens.shape
        normed = self.norm1(tokens)

        # MultiheadAttention accepts an additive attention mask with shape
        # [B * H, L, L].  Large negative values prevent attention to padded keys.
        if pair_bias is None:
            attention_bias = torch.zeros(
                batch_size,
                self.num_heads,
                length,
                length,
                dtype=normed.dtype,
                device=normed.device,
            )
        else:
            attention_bias = pair_bias.to(dtype=normed.dtype)

        invalid_keys = ~valid[:, None, None, :]
        attention_bias = attention_bias.masked_fill(invalid_keys, -1.0e4)
        attention_bias = attention_bias.reshape(
            batch_size * self.num_heads, length, length
        )

        attended, _ = self.attention(
            normed,
            normed,
            normed,
            attn_mask=attention_bias,
            need_weights=False,
        )
        tokens = tokens + self.dropout1(attended)
        tokens = tokens + self.dropout2(self.ffn(self.norm2(tokens)))
        return tokens


class SingleJetParticleTransformer(nn.Module):
    """Physics-aware Particle Transformer returning one gluon logit per jet."""

    def __init__(
        self,
        *,
        continuous_features: int = 7,
        num_particle_types: int = 1,
        type_embedding_dim: int = 8,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.10,
        pair_hidden_dim: int = 32,
        use_pairwise_bias: bool = True,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if min(d_model, num_heads, num_layers, ffn_dim) < 1:
            raise ValueError("Transformer dimensions and layer counts must be positive")

        self.num_heads = num_heads
        self.use_pairwise_bias = use_pairwise_bias
        self.type_embedding = (
            nn.Embedding(num_particle_types, type_embedding_dim, padding_idx=0)
            if num_particle_types > 1
            else None
        )

        input_dim = continuous_features + (
            type_embedding_dim if self.type_embedding is not None else 0
        )
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.pair_bias = (
            PairwisePhysicsBias(num_heads, hidden_dim=pair_hidden_dim)
            if use_pairwise_bias
            else None
        )
        self.blocks = nn.ModuleList(
            [
                ParticleAttentionBlock(d_model, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: SingleJetBatch) -> Tensor:
        graph = batch.jets

        # PyG -> dense padded constituent sequence.
        dense_x, valid = to_dense_batch(graph.x, graph.batch)
        dense_pos, _ = to_dense_batch(graph.pos, graph.batch)

        features = dense_x
        if self.type_embedding is not None:
            dense_type, _ = to_dense_batch(
                graph.particle_type,
                graph.batch,
                fill_value=0,
            )
            type_features = self.type_embedding(dense_type.long())
            features = torch.cat((features, type_features), dim=-1)

        tokens = self.input_projection(self.input_norm(features))
        batch_size, particle_count, _ = tokens.shape

        # Learned class token; no sequence positional encoding is used because a
        # jet is an unordered set.  Geometry enters through constituent features
        # and the pairwise attention bias.
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        cls_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=valid.device)
        full_valid = torch.cat((cls_valid, valid), dim=1)

        full_pair_bias: Tensor | None = None
        if self.pair_bias is not None:
            particle_bias = self.pair_bias(dense_x, dense_pos, valid)
            full_pair_bias = torch.zeros(
                batch_size,
                self.num_heads,
                particle_count + 1,
                particle_count + 1,
                dtype=particle_bias.dtype,
                device=particle_bias.device,
            )
            full_pair_bias[:, :, 1:, 1:] = particle_bias

        for block in self.blocks:
            tokens = block(tokens, full_valid, full_pair_bias)

        cls_embedding = self.final_norm(tokens[:, 0])
        return self.classifier(cls_embedding).squeeze(-1)


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader: Iterable[SingleJetBatch],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    amp: bool,
    scaler: torch.amp.GradScaler | None,
    non_blocking: bool,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct = torch.zeros((), device=device, dtype=torch.int64)
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            batch = batch.to(device, non_blocking=non_blocking)
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                logits = model(batch)
                loss = criterion(logits, batch.y)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            size = batch.y.numel()
            loss_sum += loss.detach().double() * size
            correct += ((logits >= 0) == (batch.y >= 0.5)).sum()
            count += size

    if not count:
        raise ValueError("DataLoader produced no jets")
    return float((loss_sum / count).item()), float((correct / count).item())


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: Iterable[SingleJetBatch],
    device: torch.device,
    *,
    amp: bool,
    non_blocking: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    events: list[np.ndarray] = []
    jets: list[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device, non_blocking=non_blocking)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            probability = torch.sigmoid(model(batch))
        labels.append(batch.y.cpu().numpy())
        scores.append(probability.cpu().numpy())
        events.append(batch.event_index.cpu().numpy())
        jets.append(batch.jet_index.cpu().numpy())

    return tuple(np.concatenate(items) for items in (labels, scores, events, jets))


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    axes[0].set(xlabel="Epoch", ylabel="BCE loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)


def mistag_at_gluon_efficiency(
    labels: np.ndarray,
    scores: np.ndarray,
    targets: Sequence[float] = (0.5, 0.7, 0.8, 0.9),
) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(labels, scores)
    return {
        f"quark_mistag_at_gluon_eff_{target:.1f}": float(np.interp(target, tpr, fpr))
        for target in targets
    }


def save_results(
    labels: np.ndarray,
    scores: np.ndarray,
    events: np.ndarray,
    jets: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    predictions = (scores >= 0.5).astype(np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    metrics: dict[str, object] = {
        "class_convention": {"0": "quark", "1": "gluon"},
        "test_jets": int(len(labels)),
        "test_accuracy": float(accuracy_score(labels, predictions)),
        "test_roc_auc_gluon": float(roc_auc_score(labels, scores)),
        "confusion_matrix_rows_truth_columns_prediction": matrix.tolist(),
    }
    metrics.update(mistag_at_gluon_efficiency(labels, scores))

    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("event_index", "jet_index", "truth", "gluon_probability", "prediction")
        )
        writer.writerows(zip(events, jets, labels.astype(int), scores, predictions))

    fpr, tpr, _ = roc_curve(labels, scores)
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot(fpr, tpr, label=f"AUC = {metrics['test_roc_auc_gluon']:.4f}")
    axis.plot((0, 1), (0, 1), "--", color="0.5")
    axis.set(
        xlabel="Quark mistag efficiency",
        ylabel="Gluon efficiency",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    # Log-scale mistag plot is particularly useful for FCC-ee operating points.
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.plot(fpr, tpr, label=f"AUC = {metrics['test_roc_auc_gluon']:.4f}")
    axis.set_xscale("log")
    axis.set(
        xlabel="Quark mistag efficiency",
        ylabel="Gluon efficiency",
        xlim=(max(1.0e-6, np.min(fpr[fpr > 0]) if np.any(fpr > 0) else 1.0e-6), 1),
        ylim=(0, 1),
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve_log_mistag.png", dpi=160)
    plt.close(fig)

    return metrics


def print_data_audit(
    *,
    train_dataset: SingleJetParquetDataset,
    train_indices: Sequence[int],
    train_labels: np.ndarray,
    val_indices: Sequence[int],
    val_labels: np.ndarray,
    test_indices: Sequence[int],
    test_labels: np.ndarray,
    batch_sampler: MixedClassShardBatchSampler,
) -> None:
    """Print a cheap pre-flight audit without loading constituent payloads."""

    feature_names = (
        "delta_eta", "delta_phi", "log_pt", "log_energy",
        "log_pt_fraction", "log_energy_fraction", "charge",
    )
    print("\n=== DATA / LABEL AUDIT ===")
    print("Transformer continuous inputs:", ", ".join(feature_names))
    print("Particle type input:", "enabled" if train_dataset.num_particle_types > 1 else "disabled")

    for name, labels, indices in (
        ("train", train_labels, train_indices),
        ("validation", val_labels, val_indices),
        ("test", test_labels, test_indices),
    ):
        selected = labels[np.asarray(indices, dtype=np.int64)]
        q = int(np.sum(selected == 0))
        g = int(np.sum(selected == 1))
        print(f"{name:10s}: quark(0)={q:,}, gluon(1)={g:,}, total={len(selected):,}")

    shard_counts: dict[int, list[int]] = {}
    for dataset_index in train_indices:
        shard = train_dataset.shard_index(int(dataset_index))
        label = int(train_labels[int(dataset_index)])
        shard_counts.setdefault(shard, [0, 0])[label] += 1
    pure = sum(1 for q, g in shard_counts.values() if (q == 0) ^ (g == 0))
    mixed = len(shard_counts) - pure
    print(f"Selected training shards: {len(shard_counts)} ({pure} class-pure, {mixed} mixed)")
    print(
        f"Mixed training batch target: ~{batch_sampler.n_negative} quark + "
        f"{batch_sampler.n_positive} gluon per full batch of {batch_sampler.batch_size}"
    )

    # Load one example from each class and verify that the dataset label agrees
    # with the lightweight label scan, while reporting the actual tensor content.
    selected_array = np.asarray(train_indices, dtype=np.int64)
    for class_value, class_name in ((0, "quark"), (1, "gluon")):
        matches = selected_array[train_labels[selected_array] == class_value]
        if not len(matches):
            continue
        dataset_index = int(matches[0])
        sample = train_dataset[dataset_index]
        if int(sample.y.item()) != class_value:
            raise RuntimeError(
                f"Label mismatch at dataset index {dataset_index}: "
                f"scan={class_value}, dataset={sample.y.item()}"
            )
        shard = train_dataset.shard_index(dataset_index)
        x = sample.graph.x.detach().cpu().numpy()
        if not np.isfinite(x).all():
            raise RuntimeError(f"Non-finite constituent inputs in audited {class_name} jet")
        print(
            f"{class_name:10s} sample: y={int(sample.y.item())}, "
            f"jet_index={sample.jet_index}, constituents={len(x)}, "
            f"shard={Path(train_dataset.paths[shard]).name}"
        )
        mins = np.min(x, axis=0)
        maxs = np.max(x, axis=0)
        print("  feature ranges: " + ", ".join(
            f"{name}=[{lo:.3g},{hi:.3g}]"
            for name, lo, hi in zip(feature_names, mins, maxs)
        ))

    # Inspect sampler labels only: no additional Parquet constituent payload is read here.
    # Restore the epoch counter afterwards so this audit does not alter training.
    original_epoch = batch_sampler.epoch
    probe = iter(batch_sampler)
    for number in range(1, min(4, len(batch_sampler)) + 1):
        subset_positions = next(probe)
        dataset_indices = np.asarray(train_indices, dtype=np.int64)[subset_positions]
        labels = train_labels[dataset_indices]
        print(
            f"batch {number:02d} labels: quark={int(np.sum(labels == 0))}, "
            f"gluon={int(np.sum(labels == 1))}"
        )
    batch_sampler.epoch = original_epoch
    print("==========================\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    if args.label_field == "label" and args.label_source != "event":
        raise ValueError("--label-field label requires --label-source event")
    if min(
        args.epochs,
        args.batch_size,
        args.max_constituents,
        args.shard_cache_size,
        args.prefetch_factor,
        args.d_model,
        args.num_heads,
        args.num_layers,
        args.ffn_dim,
    ) < 1:
        raise ValueError("Epochs, batch/cache sizes, constituent limit and model sizes must be positive")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must satisfy 0 <= dropout < 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_paths = expand_paths(args.train_parquet)
    val_paths = expand_paths(args.val_parquet)
    test_paths = expand_paths(args.test_parquet)

    path_sets = [set(train_paths), set(val_paths), set(test_paths)]
    if any(path_sets[i] & path_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Train, validation and test Parquet files must not overlap")

    label_args = dict(
        label_field=args.label_field,
        label_source=args.label_source,
        label_format=args.label_format,
        quark_pdgs=args.quark_pdgs,
    )
    train_labels = read_jet_labels(train_paths, **label_args)
    val_labels = read_jet_labels(val_paths, **label_args)
    test_labels = read_jet_labels(test_paths, **label_args)

    train_indices = balanced_limit(
        train_labels,
        valid_indices(train_labels, args.unknown_label_policy, "train"),
        args.limit_jets,
        args.seed,
    )
    val_indices = sorted(
        balanced_limit(
            val_labels,
            valid_indices(val_labels, args.unknown_label_policy, "validation"),
            args.limit_jets,
            args.seed + 1,
        )
    )
    test_indices = sorted(
        balanced_limit(
            test_labels,
            valid_indices(test_labels, args.unknown_label_policy, "test"),
            args.limit_jets,
            args.seed + 2,
        )
    )

    if args.particle_types is not None:
        vocabulary = tuple(sorted(set(args.particle_types)))
    elif args.use_particle_type:
        # Training split only: avoids information leakage from validation/test.
        vocabulary = infer_particle_type_vocabulary(train_paths)
    else:
        vocabulary = ()

    dataset_args = dict(
        label_field=args.label_field,
        label_source=args.label_source,
        label_format=args.label_format,
        quark_pdgs=args.quark_pdgs,
        type_vocabulary=vocabulary,
        max_constituents=args.max_constituents,
        shard_cache_size=args.shard_cache_size,
    )
    train_dataset = SingleJetParquetDataset(train_paths, **dataset_args)
    val_dataset = SingleJetParquetDataset(val_paths, **dataset_args)
    test_dataset = SingleJetParquetDataset(test_paths, **dataset_args)

    device = choose_device(args.device)
    if args.amp is None:
        amp = device.type == "cuda"
    else:
        amp = bool(args.amp)
    if amp and device.type != "cuda":
        raise ValueError("AMP is supported only on CUDA in this script")

    pin_memory = device.type == "cuda" if args.pin_memory is None else args.pin_memory
    non_blocking = bool(pin_memory and device.type == "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    common_loader_args = dict(
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
    )

    train_subset = Subset(train_dataset, train_indices)
    train_batch_sampler = MixedClassShardBatchSampler(
        train_dataset, train_indices, train_labels, args.batch_size, args.seed
    )
    print_data_audit(
        train_dataset=train_dataset,
        train_indices=train_indices,
        train_labels=train_labels,
        val_indices=val_indices,
        val_labels=val_labels,
        test_indices=test_indices,
        test_labels=test_labels,
        batch_sampler=train_batch_sampler,
    )
    train_loader = make_loader(
        train_subset,
        batch_sampler=train_batch_sampler,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        **common_loader_args,
    )
    val_loader = make_loader(
        Subset(val_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        persistent_workers=False,
        **common_loader_args,
    )
    test_loader = make_loader(
        Subset(test_dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        persistent_workers=False,
        **common_loader_args,
    )

    model = SingleJetParticleTransformer(
        num_particle_types=train_dataset.num_particle_types,
        type_embedding_dim=args.type_embedding_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        pair_hidden_dim=args.pair_hidden_dim,
        use_pairwise_bias=not args.disable_pairwise_bias,
    ).to(device)

    selected_train_labels = train_labels[np.asarray(train_indices)]
    positives = int(selected_train_labels.sum())
    negatives = len(selected_train_labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Training sample must contain both quark and gluon jets")

    positive_weight = negatives / positives
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    configuration = {
        "arguments": vars(args),
        "device": str(device),
        "amp_enabled": amp,
        "pin_memory": pin_memory,
        "class_convention": {"0": "quark", "1": "gluon"},
        "particle_type_vocabulary": vocabulary,
        "trainable_parameters": trainable_parameters,
        "positive_class_weight": positive_weight,
        "architecture": {
            "name": "physics_aware_particle_transformer",
            "continuous_features": 7,
            "pair_features": ["log_deltaR", "log_kT", "log_z", "log_m2"],
            "pairwise_bias": not args.disable_pairwise_bias,
            "pooling": "CLS token",
            "positional_encoding": "none; jet geometry enters through features and pairwise bias",
        },
    }
    with (args.output_dir / "configuration.json").open("w") as handle:
        json.dump(configuration, handle, indent=2, default=str)

    print(f"Device: {device}")
    print(
        f"Jets: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
    )
    print(
        f"Truth: field={args.label_field!r}, source={args.label_source}, "
        f"format={args.label_format}; 0=quark, 1=gluon"
    )
    print(f"Particle-type vocabulary size: {train_dataset.num_particle_types}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(
        f"Transformer: d_model={args.d_model}, heads={args.num_heads}, "
        f"layers={args.num_layers}, ffn={args.ffn_dim}, pairwise_bias={not args.disable_pairwise_bias}"
    )
    print(
        f"AMP: {amp} | pinned memory: {pin_memory} | workers: {args.num_workers} | "
        f"setup: {time.perf_counter() - start:.1f}s"
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    stale_epochs = 0
    checkpoint_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            amp=amp,
            scaler=scaler,
            non_blocking=non_blocking,
        )
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            amp=amp,
            scaler=None,
            non_blocking=non_blocking,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train loss={train_loss:.6f}, acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.6f}, acc={val_accuracy:.4f} | {row['seconds']:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "configuration": configuration,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(
                    f"Early stopping after epoch {epoch}: validation loss did not improve "
                    f"for {args.patience} epochs."
                )
                break

    save_history(history, args.output_dir)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"Loaded best checkpoint from epoch {checkpoint['epoch']} "
        f"(validation loss={checkpoint['val_loss']:.6f})."
    )

    labels, scores, events, jets = predict(
        model,
        test_loader,
        device,
        amp=amp,
        non_blocking=non_blocking,
    )
    metrics = save_results(labels, scores, events, jets, args.output_dir)

    print("\nTest results")
    print(f"  Accuracy: {metrics['test_accuracy']:.6f}")
    print(f"  ROC AUC:  {metrics['test_roc_auc_gluon']:.6f}")
    for target in (0.5, 0.7, 0.8, 0.9):
        key = f"quark_mistag_at_gluon_eff_{target:.1f}"
        print(f"  quark mistag @ gluon eff {target:.1f}: {metrics[key]:.6e}")
    print(f"Outputs written to: {args.output_dir}")
    print(f"Total runtime: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()

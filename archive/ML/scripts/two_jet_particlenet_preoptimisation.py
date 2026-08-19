"""ParticleNet adapter for two-jet FCC-ee event Parquet files.

The expected Awkward/Parquet layout is one record per event with two selected
jets and variable-length constituent arrays::

    constituent_{px,py,pz,energy,charge,type}: events x 2 x variable
    jet_{px,py,pz,...}:                       events x 2
    label:                                    events

The Parquet files are not rewritten.  :class:`TwoJetParquetDataset` loads one
shard at a time and returns two PyTorch-Geometric graphs plus a symmetric
event-level feature vector.  :class:`TwoJetParticleNet` applies one shared
ParticleNet encoder to both graphs and combines the embeddings as
``[h1 + h2, abs(h1 - h2)]``.

Required packages
-----------------
``awkward``, ``numpy``, ``torch`` and ``torch-geometric``.  Nearest-neighbour
graphs are built with native PyTorch operations, so a separately compiled
``torch-cluster`` CUDA extension is not required.

Minimal use
-----------
>>> dataset = TwoJetParquetDataset(["signal_part_0000.parquet",
...                                 "background_part_0000.parquet"])
>>> loader = make_loader(dataset, batch_size=64, shuffle=True)
>>> model = TwoJetParticleNet(high_level_dim=dataset.high_level_dim)
>>> batch = next(iter(loader))
>>> logits = model(batch)
>>> loss = torch.nn.functional.binary_cross_entropy_with_logits(
...     logits, batch.y
... )

Fit any feature normalisation using the training split only, and pass the
resulting mean and standard deviation to ``TwoJetParticleNet``.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import awkward as ak
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import BatchNorm, MessagePassing, global_mean_pool
from torch_geometric.utils import to_dense_batch


REQUIRED_FIELDS = (
    "label",
    "jet_px",
    "jet_py",
    "jet_pz",
    "constituent_energy",
    "constituent_px",
    "constituent_py",
    "constituent_pz",
    "constituent_charge",
    "constituent_type",
)

DEFAULT_SCALAR_FEATURES = (
    "event_invariant_mass",
    "n_jets_original",
)

# Each two-jet field becomes [jet_1 + jet_2, |jet_1 - jet_2|].
# This deliberately omits algebraically redundant jet_p, C2 and D2.
DEFAULT_JET_FEATURES = (
    "jet_energy",
    "jet_mass",
    "jet_theta",
    "e2_beta_0p2",
    "e3_beta_0p2",
)


@dataclass(frozen=True)
class TwoJetSample:
    """One event before batching."""

    jet1: Data
    jet2: Data
    high_level: Tensor
    y: Tensor
    source_file: str | None = None
    source_event: int | None = None


@dataclass(frozen=True)
class TwoJetBatch:
    """A mini-batch containing two independent batches of jet graphs."""

    jet1: Batch
    jet2: Batch
    high_level: Tensor
    y: Tensor

    def to(self, device: torch.device | str) -> "TwoJetBatch":
        return TwoJetBatch(
            jet1=self.jet1.to(device),
            jet2=self.jet2.to(device),
            high_level=self.high_level.to(device),
            y=self.y.to(device),
        )


def collate_two_jet_samples(samples: Sequence[TwoJetSample]) -> TwoJetBatch:
    """Collate paired variable-size graphs without constituent padding."""

    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    return TwoJetBatch(
        jet1=Batch.from_data_list([sample.jet1 for sample in samples]),
        jet2=Batch.from_data_list([sample.jet2 for sample in samples]),
        high_level=torch.stack([sample.high_level for sample in samples]),
        y=torch.stack([sample.y for sample in samples]),
    )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs: object,
) -> DataLoader:
    """Construct a loader using the paired-graph collator."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_two_jet_samples,
        **kwargs,
    )


class TwoJetParquetDataset(Dataset[TwoJetSample]):
    """Map Awkward Parquet event records to two constituent graphs.

    Parameters
    ----------
    paths:
        One or more Parquet shards.  Labels are read from each file; file names
        are never used to infer the class.
    scalar_features:
        Event-level scalar fields included directly.
    jet_features:
        Two-entry fields converted to sum and absolute difference, making the
        high-level branch invariant to interchange of the two jets.
    type_vocabulary:
        Raw reconstructed-particle type codes assigned embedding indices
        ``1..N``.  Index 0 is reserved for unknown values.  If omitted, all
        type codes map to 0.  Derive this vocabulary from the training split
        only if particle type is to be used.
    max_constituents:
        Optional limit per jet.  The highest-energy constituents are retained.
    shard_cache_size:
        Number of complete Parquet shards cached per worker.
    """

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        *,
        scalar_features: Sequence[str] = DEFAULT_SCALAR_FEATURES,
        jet_features: Sequence[str] = DEFAULT_JET_FEATURES,
        type_vocabulary: Sequence[int] | None = None,
        max_constituents: int | None = None,
        shard_cache_size: int = 1,
        epsilon: float = 1.0e-8,
        validate: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("At least one Parquet path is required")
        missing_paths = [str(path) for path in self.paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Parquet shard(s) not found: {missing_paths}")
        if max_constituents is not None and max_constituents < 1:
            raise ValueError("max_constituents must be positive or None")
        if shard_cache_size < 1:
            raise ValueError("shard_cache_size must be at least one")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        self.scalar_features = tuple(scalar_features)
        self.jet_features = tuple(jet_features)
        self.max_constituents = max_constituents
        self.shard_cache_size = shard_cache_size
        self.epsilon = float(epsilon)
        self.validate = validate
        self._type_to_index = {
            int(raw_type): index + 1
            for index, raw_type in enumerate(type_vocabulary or ())
        }
        self._columns = tuple(
            dict.fromkeys(
                REQUIRED_FIELDS
                + self.scalar_features
                + self.jet_features
                + ("source_file", "source_event")
            )
        )

        # Parquet footer metadata establishes the global index without reading
        # any event payloads.
        lengths: list[int] = []
        available_columns: list[set[str]] = []
        for path in self.paths:
            metadata = ak.metadata_from_parquet(path)
            lengths.append(int(metadata["num_rows"]))
            # Physical Parquet column names include list encodings; the
            # Awkward Form exposes the top-level record fields we request.
            available_columns.append(set(metadata["form"].fields))

        self._ends = np.cumsum(lengths, dtype=np.int64)
        self._available_columns = tuple(available_columns)
        self._cache: OrderedDict[int, ak.Array] = OrderedDict()

        if validate:
            requested = set(REQUIRED_FIELDS + self.scalar_features + self.jet_features)
            for path, columns in zip(self.paths, self._available_columns):
                missing = sorted(requested - columns)
                if missing:
                    raise ValueError(f"{path} is missing required fields: {missing}")

    @property
    def high_level_dim(self) -> int:
        return len(self.scalar_features) + 2 * len(self.jet_features)

    @property
    def num_particle_types(self) -> int:
        """Embedding size, including the unknown/padding index zero."""

        return len(self._type_to_index) + 1

    @property
    def high_level_names(self) -> tuple[str, ...]:
        names = list(self.scalar_features)
        for name in self.jet_features:
            names.extend((f"{name}_sum", f"{name}_absdiff"))
        return tuple(names)

    def __len__(self) -> int:
        return int(self._ends[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        start = 0 if shard_index == 0 else int(self._ends[shard_index - 1])
        return shard_index, index - start

    def _load_shard(self, shard_index: int) -> ak.Array:
        if shard_index in self._cache:
            self._cache.move_to_end(shard_index)
            return self._cache[shard_index]

        available = self._available_columns[shard_index]
        columns = [name for name in self._columns if name in available]
        shard = ak.from_parquet(self.paths[shard_index], columns=columns)
        self._cache[shard_index] = shard
        self._cache.move_to_end(shard_index)
        while len(self._cache) > self.shard_cache_size:
            self._cache.popitem(last=False)
        return shard

    def _type_indices(self, raw_types: np.ndarray) -> np.ndarray:
        if not self._type_to_index:
            return np.zeros(raw_types.shape, dtype=np.int64)
        return np.fromiter(
            (self._type_to_index.get(int(value), 0) for value in raw_types),
            dtype=np.int64,
            count=len(raw_types),
        )

    def _jet_graph(self, event: ak.Record, jet_index: int) -> Data:
        def values(field: str, dtype: np.dtype) -> np.ndarray:
            return np.asarray(ak.to_numpy(event[field][jet_index]), dtype=dtype)

        energy = values("constituent_energy", np.float64)
        px = values("constituent_px", np.float64)
        py = values("constituent_py", np.float64)
        pz = values("constituent_pz", np.float64)
        charge = values("constituent_charge", np.float64)
        raw_type = values("constituent_type", np.int64)

        lengths = {len(array) for array in (energy, px, py, pz, charge, raw_type)}
        if len(lengths) != 1:
            raise ValueError("Constituent fields have inconsistent lengths")
        if not energy.size:
            raise ValueError("ParticleNet cannot construct a graph for an empty jet")

        finite = (
            np.isfinite(energy)
            & np.isfinite(px)
            & np.isfinite(py)
            & np.isfinite(pz)
            & np.isfinite(charge)
        )
        positive_energy = energy > 0.0
        keep = finite & positive_energy
        if not np.any(keep):
            raise ValueError("Jet has no finite positive-energy constituents")
        energy, px, py, pz, charge, raw_type = (
            array[keep] for array in (energy, px, py, pz, charge, raw_type)
        )

        if self.max_constituents is not None and len(energy) > self.max_constituents:
            selected = np.argsort(energy)[-self.max_constituents :]
            # Descending energy gives deterministic constituent ordering.
            selected = selected[np.argsort(energy[selected])[::-1]]
            energy, px, py, pz, charge, raw_type = (
                array[selected] for array in (energy, px, py, pz, charge, raw_type)
            )

        pt = np.hypot(px, py)
        momentum = np.sqrt(px * px + py * py + pz * pz)
        eta = np.arctanh(np.clip(pz / np.maximum(momentum, self.epsilon),
                                 -1.0 + self.epsilon, 1.0 - self.epsilon))
        phi = np.arctan2(py, px)

        jet_px = float(event["jet_px"][jet_index])
        jet_py = float(event["jet_py"][jet_index])
        jet_pz = float(event["jet_pz"][jet_index])
        jet_pt = np.hypot(jet_px, jet_py)
        jet_p = np.sqrt(jet_pt * jet_pt + jet_pz * jet_pz)
        jet_eta = np.arctanh(
            np.clip(jet_pz / max(jet_p, self.epsilon),
                    -1.0 + self.epsilon, 1.0 - self.epsilon)
        )
        jet_phi = np.arctan2(jet_py, jet_px)

        delta_eta = eta - jet_eta
        delta_phi = np.arctan2(np.sin(phi - jet_phi), np.cos(phi - jet_phi))
        continuous = np.stack(
            (
                delta_eta,
                delta_phi,
                np.log(np.maximum(pt, self.epsilon)),
                np.log(np.maximum(energy, self.epsilon)),
                charge,
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        position = np.stack((delta_eta, delta_phi), axis=1).astype(
            np.float32, copy=False
        )

        return Data(
            x=torch.from_numpy(continuous),
            pos=torch.from_numpy(position),
            particle_type=torch.from_numpy(self._type_indices(raw_type)),
        )

    def _high_level(self, event: ak.Record) -> Tensor:
        features = [float(event[name]) for name in self.scalar_features]
        for name in self.jet_features:
            pair = np.asarray(ak.to_numpy(event[name]), dtype=np.float64)
            if pair.shape != (2,):
                raise ValueError(f"Field {name!r} must have exactly two values")
            features.extend((float(pair.sum()), float(abs(pair[0] - pair[1]))))
        array = np.asarray(features, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            bad = np.asarray(self.high_level_names)[~np.isfinite(array)].tolist()
            raise ValueError(f"Non-finite high-level feature(s): {bad}")
        return torch.from_numpy(array)

    def __getitem__(self, index: int) -> TwoJetSample:
        shard_index, local_index = self._locate(index)
        event = self._load_shard(shard_index)[local_index]
        label = float(event["label"])
        if self.validate and label not in (0.0, 1.0):
            raise ValueError(f"Expected binary label, got {label}")

        fields = set(ak.fields(event))
        source_file = str(event["source_file"]) if "source_file" in fields else None
        source_event = int(event["source_event"]) if "source_event" in fields else None
        return TwoJetSample(
            jet1=self._jet_graph(event, 0),
            jet2=self._jet_graph(event, 1),
            high_level=self._high_level(event),
            y=torch.tensor(label, dtype=torch.float32),
            source_file=source_file,
            source_event=source_event,
        )


def infer_particle_type_vocabulary(
    training_paths: str | Path | Sequence[str | Path],
) -> tuple[int, ...]:
    """Return sorted particle-type codes found in training Parquets only."""

    if isinstance(training_paths, (str, Path)):
        training_paths = [training_paths]
    values: set[int] = set()
    for path in training_paths:
        data = ak.from_parquet(path, columns=["constituent_type"])
        flat = ak.to_numpy(ak.flatten(data["constituent_type"], axis=None))
        values.update(int(value) for value in flat)
    return tuple(sorted(values))


class EdgeConvBlock(MessagePassing):
    """ParticleNet EdgeConv block with a residual connection."""

    def __init__(self, in_channels: int, channels: Sequence[int], k: int) -> None:
        super().__init__(aggr="max", flow="source_to_target")
        if len(channels) != 3:
            raise ValueError("Each EdgeConv block requires three channel widths")
        self.k = int(k)
        layers: list[nn.Module] = []
        width = 2 * in_channels
        for out_channels in channels:
            layers.extend(
                (nn.Linear(width, out_channels, bias=False),
                 BatchNorm(out_channels), nn.ReLU())
            )
            width = out_channels
        self.message_mlp = nn.Sequential(*layers)
        self.skip = nn.Sequential(
            nn.Linear(in_channels, channels[-1], bias=False),
            BatchNorm(channels[-1]),
        )
        self.activation = nn.ReLU()

    @staticmethod
    def _safe_knn_graph(points: Tensor, batch: Tensor, k: int) -> Tensor:
        """Build batched k-NN edges using native PyTorch CPU/CUDA kernels.

        ``to_dense_batch`` pads each graph only to the largest graph in the
        current mini-batch.  Pairwise distances therefore have shape
        ``[graphs, max_nodes, max_nodes]`` rather than mixing nodes from
        different jets.  Invalid padding and self-distances are masked before
        selecting neighbours.  Graphs with fewer than ``k + 1`` nodes simply
        use every available non-self neighbour; singleton graphs retain the
        previous self-loop behaviour.
        """

        if points.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=points.device)

        dense_points, mask = to_dense_batch(points, batch)
        node_ids, _ = to_dense_batch(
            torch.arange(points.size(0), device=points.device),
            batch,
            fill_value=-1,
        )
        n_graphs, max_nodes, _ = dense_points.shape

        if max_nodes == 1:
            nodes = node_ids[:, 0]
            return torch.stack((nodes, nodes))

        distances = torch.cdist(dense_points, dense_points)
        diagonal = torch.eye(
            max_nodes, dtype=torch.bool, device=points.device
        ).unsqueeze(0)
        invalid = diagonal | ~mask.unsqueeze(1) | ~mask.unsqueeze(2)
        distances.masked_fill_(invalid, torch.inf)

        neighbours_per_node = min(int(k), max_nodes - 1)
        neighbour_slots = distances.topk(
            neighbours_per_node, dim=-1, largest=False
        ).indices

        graph_slots = torch.arange(
            n_graphs, device=points.device
        )[:, None, None].expand_as(neighbour_slots)
        target_slots = torch.arange(
            max_nodes, device=points.device
        )[None, :, None].expand_as(neighbour_slots)
        source_nodes = node_ids[graph_slots, neighbour_slots]
        target_nodes = node_ids[graph_slots, target_slots]
        valid_edges = (
            mask.unsqueeze(-1)
            & (source_nodes >= 0)
            & torch.isfinite(distances.gather(-1, neighbour_slots))
        )

        edge_index = torch.stack(
            (source_nodes[valid_edges], target_nodes[valid_edges])
        )

        singleton_graphs = torch.nonzero(
            mask.sum(dim=1) == 1, as_tuple=False
        ).flatten()
        if singleton_graphs.numel():
            singleton_nodes = node_ids[singleton_graphs, 0]
            singleton_edges = torch.stack((singleton_nodes, singleton_nodes))
            edge_index = torch.cat((edge_index, singleton_edges), dim=1)
        return edge_index

    def forward(self, points: Tensor, features: Tensor, batch: Tensor) -> Tensor:
        edge_index = self._safe_knn_graph(points, batch, self.k)
        messages = self.propagate(edge_index, x=features)
        return self.activation(messages + self.skip(features))

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.message_mlp(torch.cat((x_i, x_j - x_i), dim=-1))


class ParticleNetEncoder(nn.Module):
    """One-jet ParticleNet encoder returning a fixed-size embedding."""

    def __init__(
        self,
        *,
        continuous_features: int = 5,
        num_particle_types: int = 1,
        type_embedding_dim: int = 8,
        conv_params: Sequence[tuple[int, Sequence[int]]] = (
            (16, (64, 64, 64)),
            (16, (128, 128, 128)),
            (16, (256, 256, 256)),
        ),
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_particle_types < 1:
            raise ValueError("num_particle_types must include index zero")
        self.use_particle_type = num_particle_types > 1
        self.type_embedding = (
            nn.Embedding(num_particle_types, type_embedding_dim, padding_idx=0)
            if self.use_particle_type else None
        )
        input_dim = continuous_features + (
            type_embedding_dim if self.use_particle_type else 0
        )
        self.input_norm = BatchNorm(input_dim)
        blocks: list[EdgeConvBlock] = []
        width = input_dim
        for k, channels in conv_params:
            blocks.append(EdgeConvBlock(width, channels, k))
            width = channels[-1]
        self.blocks = nn.ModuleList(blocks)
        self.projection = nn.Sequential(
            nn.Linear(width, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_dim = embedding_dim

    def forward(self, graph: Batch) -> Tensor:
        features = graph.x
        if self.type_embedding is not None:
            type_features = self.type_embedding(graph.particle_type)
            features = torch.cat((features, type_features), dim=-1)
        features = self.input_norm(features)
        points = graph.pos
        for block in self.blocks:
            features = block(points, features, graph.batch)
            points = features
        pooled = global_mean_pool(features, graph.batch)
        return self.projection(pooled)


class FeatureStandardizer(nn.Module):
    """Fixed standardisation stored with the model state."""

    def __init__(
        self,
        size: int,
        mean: Sequence[float] | Tensor | None = None,
        std: Sequence[float] | Tensor | None = None,
        epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        mean_tensor = torch.zeros(size) if mean is None else torch.as_tensor(mean)
        std_tensor = torch.ones(size) if std is None else torch.as_tensor(std)
        mean_tensor = mean_tensor.to(dtype=torch.float32)
        std_tensor = std_tensor.to(dtype=torch.float32)
        if mean_tensor.shape != (size,) or std_tensor.shape != (size,):
            raise ValueError(f"mean and std must both have shape ({size},)")
        if torch.any(std_tensor < 0):
            raise ValueError("standard deviations cannot be negative")
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("std", std_tensor.clamp_min(epsilon))

    def forward(self, values: Tensor) -> Tensor:
        return (values - self.mean) / self.std


class TwoJetParticleNet(nn.Module):
    """Shared jet encoder plus a swap-invariant binary event head."""

    def __init__(
        self,
        *,
        high_level_dim: int,
        num_particle_types: int = 1,
        type_embedding_dim: int = 8,
        jet_embedding_dim: int = 256,
        high_level_hidden: int = 128,
        classifier_hidden: int = 256,
        dropout: float = 0.1,
        high_level_mean: Sequence[float] | Tensor | None = None,
        high_level_std: Sequence[float] | Tensor | None = None,
    ) -> None:
        super().__init__()
        self.jet_encoder = ParticleNetEncoder(
            num_particle_types=num_particle_types,
            type_embedding_dim=type_embedding_dim,
            embedding_dim=jet_embedding_dim,
            dropout=dropout,
        )
        self.high_level_norm = FeatureStandardizer(
            high_level_dim, high_level_mean, high_level_std
        )
        self.high_level_encoder = nn.Sequential(
            nn.Linear(high_level_dim, high_level_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        combined_dim = 2 * jet_embedding_dim + high_level_hidden
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, batch: TwoJetBatch) -> Tensor:
        first = self.jet_encoder(batch.jet1)
        second = self.jet_encoder(batch.jet2)
        jet_event = torch.cat((first + second, torch.abs(first - second)), dim=-1)
        high_level = self.high_level_encoder(
            self.high_level_norm(batch.high_level)
        )
        return self.classifier(torch.cat((jet_event, high_level), dim=-1)).squeeze(-1)


@torch.no_grad()
def estimate_high_level_statistics(
    loader: Iterable[TwoJetBatch],
) -> tuple[Tensor, Tensor]:
    """Estimate feature mean/std in one pass over a training-only loader."""

    count = 0
    total: Tensor | None = None
    total_squared: Tensor | None = None
    for batch in loader:
        values = batch.high_level.to(dtype=torch.float64)
        count += values.shape[0]
        batch_sum = values.sum(dim=0)
        batch_squared = (values * values).sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        total_squared = (
            batch_squared if total_squared is None else total_squared + batch_squared
        )
    if count < 2 or total is None or total_squared is None:
        raise ValueError("At least two training events are required")
    mean = total / count
    variance = (total_squared - count * mean * mean) / (count - 1)
    return mean.float(), variance.clamp_min(0.0).sqrt().float()


def binary_accuracy(logits: Tensor, labels: Tensor) -> Tensor:
    """Threshold raw binary logits at zero."""

    return ((logits >= 0) == (labels >= 0.5)).float().mean()
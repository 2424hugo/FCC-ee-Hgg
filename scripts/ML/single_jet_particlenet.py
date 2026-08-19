"""Optimised v3 single-jet ParticleNet quark/gluon classifier for FCC-ee Parquet shards.

Each Parquet row may contain two reconstructed jets, but each jet is exposed as
an independent training example. Labels can be supplied either as two per-jet
values or as one event value when the dataset construction guarantees that both
selected jets have the event flavour (for example H -> gg versus qqbar samples).

Expected fields
---------------
``jet_px``, ``jet_py``, ``jet_pz``: events x 2
``constituent_{energy,px,py,pz,charge,type}``: events x 2 x variable
``jet_label`` (binary mode): events x 2, 0=quark and 1=gluon
``label`` (event-source binary mode): events, 0=quark and 1=gluon; duplicated
for the two jets only when both jets are known to share the event flavour

Alternatively, ``--label-format pdg`` may be used with a field such as
``jet_flavour`` containing generator PDG identifiers: abs(PDG) in 1..5 maps to
quark and PDG 21 maps to gluon.  Other values are rejected by the dataset and
can be excluded by the training script.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import awkward as ak
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import BatchNorm, MessagePassing, global_mean_pool
from torch_geometric.utils import to_dense_batch


KINEMATIC_FIELDS = (
    "jet_px", "jet_py", "jet_pz",
    "constituent_energy", "constituent_px", "constituent_py",
    "constituent_pz", "constituent_charge", "constituent_type",
)


def encode_jet_flavour(
    value: int | float,
    label_format: str,
    quark_pdgs: Sequence[int] = (1, 2, 3, 4, 5),
) -> int:
    """Return 0 for quark, 1 for gluon, and -1 for an unsupported PDG label."""

    if not np.isfinite(value):
        return -1
    integer = int(value)
    if float(integer) != float(value):
        return -1
    if label_format == "binary":
        if integer not in (0, 1):
            raise ValueError(f"Binary jet label must be 0 or 1, got {value}")
        return integer
    if label_format != "pdg":
        raise ValueError("label_format must be 'binary' or 'pdg'")
    absolute = abs(integer)
    if absolute in {abs(int(code)) for code in quark_pdgs}:
        return 0
    if absolute == 21:
        return 1
    return -1


@dataclass(frozen=True)
class SingleJetSample:
    graph: Data
    y: Tensor
    event_index: int
    jet_index: int


@dataclass(frozen=True)
class SingleJetBatch:
    jets: Batch
    y: Tensor
    event_index: Tensor
    jet_index: Tensor

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "SingleJetBatch":
        return SingleJetBatch(
            jets=self.jets.to(device, non_blocking=non_blocking),
            y=self.y.to(device, non_blocking=non_blocking),
            event_index=self.event_index.to(device, non_blocking=non_blocking),
            jet_index=self.jet_index.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "SingleJetBatch":
        return SingleJetBatch(
            jets=self.jets.pin_memory(),
            y=self.y.pin_memory(),
            event_index=self.event_index.pin_memory(),
            jet_index=self.jet_index.pin_memory(),
        )


def collate_single_jets(samples: Sequence[SingleJetSample]) -> SingleJetBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    return SingleJetBatch(
        jets=Batch.from_data_list([sample.graph for sample in samples]),
        y=torch.stack([sample.y for sample in samples]),
        event_index=torch.tensor([sample.event_index for sample in samples], dtype=torch.long),
        jet_index=torch.tensor([sample.jet_index for sample in samples], dtype=torch.long),
    )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
    **kwargs: object,
) -> DataLoader:
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    worker_args: dict[str, object] = {}
    if num_workers > 0:
        worker_args.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    loader_args: dict[str, object] = dict(
        dataset=dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_single_jets,
        **worker_args,
    )
    if "batch_sampler" not in kwargs:
        if batch_size is None:
            raise ValueError("batch_size is required when batch_sampler is absent")
        loader_args.update(batch_size=batch_size, shuffle=shuffle)
    loader_args.update(kwargs)
    return DataLoader(**loader_args)


class SingleJetParquetDataset(Dataset[SingleJetSample]):
    """Expose the two jets in each event record as independent graph samples."""

    def __init__(
        self,
        paths: str | Path | Sequence[str | Path],
        *,
        label_field: str = "jet_label",
        label_source: str = "per-jet",
        label_format: str = "binary",
        quark_pdgs: Sequence[int] = (1, 2, 3, 4, 5),
        type_vocabulary: Sequence[int] | None = None,
        max_constituents: int | None = 100,
        shard_cache_size: int = 2,
        epsilon: float = 1.0e-8,
        validate: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = tuple(Path(path) for path in paths)
        if not self.paths:
            raise ValueError("At least one Parquet path is required")
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Parquet shard(s) not found: {missing}")
        if label_source not in ("per-jet", "event"):
            raise ValueError("label_source must be 'per-jet' or 'event'")
        if label_source == "per-jet" and label_field == "label":
            raise ValueError(
                "label_field='label' is scalar event truth; use "
                "label_source='event' only when both jets share that flavour"
            )
        if label_format not in ("binary", "pdg"):
            raise ValueError("label_format must be 'binary' or 'pdg'")
        if max_constituents is not None and max_constituents < 1:
            raise ValueError("max_constituents must be positive or None")
        if shard_cache_size < 1 or epsilon <= 0:
            raise ValueError("shard_cache_size and epsilon must be positive")

        self.label_field = label_field
        self.label_source = label_source
        self.label_format = label_format
        self.quark_pdgs = tuple(abs(int(code)) for code in quark_pdgs)
        self.max_constituents = max_constituents
        self.shard_cache_size = shard_cache_size
        self.epsilon = float(epsilon)
        self.validate = validate
        self._type_values = np.asarray(
            sorted({int(value) for value in (type_vocabulary or ())}), dtype=np.int64
        )
        self._columns = KINEMATIC_FIELDS + (label_field,)
        event_lengths: list[int] = []
        available: list[set[str]] = []
        for path in self.paths:
            metadata = ak.metadata_from_parquet(path)
            event_lengths.append(int(metadata["num_rows"]))
            available.append(set(metadata["form"].fields))
        self._jet_ends = 2 * np.cumsum(event_lengths, dtype=np.int64)
        self._available_columns = tuple(available)
        self._cache: OrderedDict[int, ak.Array] = OrderedDict()
        if validate:
            required = set(self._columns)
            for path, columns in zip(self.paths, self._available_columns):
                missing_fields = sorted(required - columns)
                if missing_fields:
                    raise ValueError(f"{path} is missing required fields: {missing_fields}")

    @property
    def num_particle_types(self) -> int:
        return len(self._type_values) + 1

    def __len__(self) -> int:
        return int(self._jet_ends[-1])

    def _locate(self, index: int) -> tuple[int, int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self._jet_ends, index)
        start = 0 if shard == 0 else int(self._jet_ends[shard - 1])
        local_jet = index - start
        return shard, local_jet // 2, local_jet % 2

    def shard_index(self, index: int) -> int:
        return self._locate(index)[0]

    def event_group(self, index: int) -> tuple[int, int]:
        shard, event, _ = self._locate(index)
        return shard, event

    def _load_shard(self, shard: int) -> ak.Array:
        if shard in self._cache:
            self._cache.move_to_end(shard)
            return self._cache[shard]
        record = ak.from_parquet(self.paths[shard], columns=list(self._columns))
        self._cache[shard] = record
        while len(self._cache) > self.shard_cache_size:
            self._cache.popitem(last=False)
        return record

    def _type_indices(self, raw: np.ndarray) -> np.ndarray:
        if not self._type_values.size:
            return np.zeros(raw.shape, dtype=np.int64)
        locations = np.searchsorted(self._type_values, raw)
        valid = locations < len(self._type_values)
        known = np.zeros(raw.shape, dtype=bool)
        known[valid] = self._type_values[locations[valid]] == raw[valid]
        return np.where(known, locations + 1, 0).astype(np.int64, copy=False)

    def _graph(self, event: ak.Record, jet: int) -> Data:
        def values(name: str, dtype: np.dtype) -> np.ndarray:
            return np.asarray(ak.to_numpy(event[name][jet]), dtype=dtype)

        energy = values("constituent_energy", np.float64)
        px = values("constituent_px", np.float64)
        py = values("constituent_py", np.float64)
        pz = values("constituent_pz", np.float64)
        charge = values("constituent_charge", np.float64)
        particle_type = values("constituent_type", np.int64)
        if len({len(x) for x in (energy, px, py, pz, charge, particle_type)}) != 1:
            raise ValueError("Constituent fields have inconsistent lengths")
        keep = (
            np.isfinite(energy) & np.isfinite(px) & np.isfinite(py)
            & np.isfinite(pz) & np.isfinite(charge) & (energy > 0)
        )
        if not np.any(keep):
            raise ValueError("Jet has no finite positive-energy constituents")
        energy, px, py, pz, charge, particle_type = (
            array[keep] for array in (energy, px, py, pz, charge, particle_type)
        )
        if self.max_constituents is not None and len(energy) > self.max_constituents:
            selected = np.argpartition(energy, -self.max_constituents)[-self.max_constituents:]
            selected = selected[np.argsort(energy[selected])[::-1]]
            energy, px, py, pz, charge, particle_type = (
                array[selected] for array in (energy, px, py, pz, charge, particle_type)
            )

        pt = np.hypot(px, py)
        momentum = np.sqrt(px * px + py * py + pz * pz)
        eta = np.arctanh(np.clip(
            pz / np.maximum(momentum, self.epsilon),
            -1 + self.epsilon, 1 - self.epsilon,
        ))
        phi = np.arctan2(py, px)
        jet_px = float(event["jet_px"][jet])
        jet_py = float(event["jet_py"][jet])
        jet_pz = float(event["jet_pz"][jet])
        jet_pt = np.hypot(jet_px, jet_py)
        jet_p = np.sqrt(jet_pt * jet_pt + jet_pz * jet_pz)
        jet_eta = np.arctanh(np.clip(
            jet_pz / max(jet_p, self.epsilon),
            -1 + self.epsilon, 1 - self.epsilon,
        ))
        jet_phi = np.arctan2(jet_py, jet_px)
        delta_eta = eta - jet_eta
        delta_phi = np.arctan2(np.sin(phi - jet_phi), np.cos(phi - jet_phi))
        continuous = np.stack((
            delta_eta,
            delta_phi,
            np.log(np.maximum(pt, self.epsilon)),
            np.log(np.maximum(energy, self.epsilon)),
            np.log(np.maximum(pt / max(jet_pt, self.epsilon), self.epsilon)),
            np.log(np.maximum(energy / max(float(np.sum(energy)), self.epsilon), self.epsilon)),
            charge,
        ), axis=1).astype(np.float32, copy=False)
        position = np.stack((delta_eta, delta_phi), axis=1).astype(np.float32, copy=False)
        return Data(
            x=torch.from_numpy(continuous),
            pos=torch.from_numpy(position),
            particle_type=torch.from_numpy(self._type_indices(particle_type)),
        )

    def __getitem__(self, index: int) -> SingleJetSample:
        shard, local_event, jet = self._locate(index)
        event = self._load_shard(shard)[local_event]
        raw_label = event[self.label_field]
        if self.label_source == "event":
            values = np.asarray(raw_label)
            if values.ndim != 0:
                raise ValueError(f"{self.label_field!r} must be scalar for event labels")
            value = values.item()
        else:
            values = np.asarray(ak.to_numpy(raw_label))
            if values.shape != (2,):
                raise ValueError(f"{self.label_field!r} must have exactly two values per event")
            value = values[jet]
        label = encode_jet_flavour(value, self.label_format, self.quark_pdgs)
        if label < 0:
            raise ValueError(f"Unsupported jet flavour {value} at dataset index {index}")
        global_event = index // 2
        return SingleJetSample(
            graph=self._graph(event, jet),
            y=torch.tensor(float(label), dtype=torch.float32),
            event_index=global_event,
            jet_index=jet,
        )


def read_jet_labels(
    paths: Sequence[Path],
    *,
    label_field: str,
    label_source: str,
    label_format: str,
    quark_pdgs: Sequence[int] = (1, 2, 3, 4, 5),
) -> np.ndarray:
    """Read and encode labels without constructing constituent graphs."""

    encoded: list[np.ndarray] = []
    for path in paths:
        records = ak.from_parquet(path, columns=[label_field])
        values = np.asarray(ak.to_numpy(records[label_field]))
        if label_source == "event":
            if values.ndim != 1:
                raise ValueError(f"{path}: {label_field!r} must have shape (events,)")
            flat = np.repeat(values, 2)
        elif label_source == "per-jet":
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError(f"{path}: {label_field!r} must have shape (events, 2)")
            flat = values.reshape(-1)
        else:
            raise ValueError("label_source must be 'per-jet' or 'event'")
        encoded.append(np.asarray([
            encode_jet_flavour(value, label_format, quark_pdgs) for value in flat
        ], dtype=np.int8))
    return np.concatenate(encoded)


def infer_particle_type_vocabulary(paths: Sequence[Path]) -> tuple[int, ...]:
    values: set[int] = set()
    for path in paths:
        data = ak.from_parquet(path, columns=["constituent_type"])
        flat = ak.to_numpy(ak.flatten(data["constituent_type"], axis=None))
        values.update(np.unique(flat).astype(np.int64, copy=False).tolist())
    return tuple(sorted(values))


class EdgeConvBlock(MessagePassing):
    def __init__(self, in_channels: int, channels: Sequence[int], k: int) -> None:
        super().__init__(aggr="max", flow="source_to_target")
        if len(channels) != 3:
            raise ValueError("Each EdgeConv block requires three channel widths")
        self.k = int(k)
        layers: list[nn.Module] = []
        width = 2 * in_channels
        for out_channels in channels:
            layers.extend((
                nn.Linear(width, out_channels, bias=False),
                BatchNorm(out_channels), nn.ReLU(),
            ))
            width = out_channels
        self.message_mlp = nn.Sequential(*layers)
        self.skip = nn.Sequential(
            nn.Linear(in_channels, channels[-1], bias=False),
            BatchNorm(channels[-1]),
        )
        self.activation = nn.ReLU()

    @staticmethod
    def _safe_knn_graph(points: Tensor, batch: Tensor, k: int) -> Tensor:
        if points.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=points.device)
        dense, mask = to_dense_batch(points, batch)
        n_graphs, max_nodes, _ = dense.shape
        counts = mask.sum(dim=1)
        starts = counts.cumsum(dim=0) - counts
        node_ids = starts[:, None] + torch.arange(max_nodes, device=points.device)[None, :]
        node_ids = node_ids.masked_fill(~mask, -1)
        if max_nodes == 1:
            nodes = node_ids[:, 0]
            return torch.stack((nodes, nodes))
        with torch.autocast(device_type=points.device.type, enabled=False):
            dense_float = dense.float()
            norms = dense_float.square().sum(dim=-1, keepdim=True)
            distances = norms + norms.transpose(1, 2)
            distances.add_(torch.bmm(dense_float, dense_float.transpose(1, 2)), alpha=-2)
            distances.clamp_min_(0)
        diagonal = torch.eye(max_nodes, dtype=torch.bool, device=points.device).unsqueeze(0)
        distances.masked_fill_(diagonal | ~mask.unsqueeze(1) | ~mask.unsqueeze(2), torch.inf)
        neighbour_count = min(int(k), max_nodes - 1)
        slots = distances.topk(neighbour_count, dim=-1, largest=False).indices
        graph_slots = torch.arange(n_graphs, device=points.device)[:, None, None].expand_as(slots)
        target_slots = torch.arange(max_nodes, device=points.device)[None, :, None].expand_as(slots)
        sources = node_ids[graph_slots, slots]
        targets = node_ids[graph_slots, target_slots]
        valid = mask.unsqueeze(-1) & (sources >= 0) & torch.isfinite(distances.gather(-1, slots))
        edges = torch.stack((sources[valid], targets[valid]))
        singleton = torch.nonzero(mask.sum(dim=1) == 1, as_tuple=False).flatten()
        if singleton.numel():
            nodes = node_ids[singleton, 0]
            edges = torch.cat((edges, torch.stack((nodes, nodes))), dim=1)
        return edges

    def forward(self, points: Tensor, features: Tensor, batch: Tensor) -> Tensor:
        edges = self._safe_knn_graph(points, batch, self.k)
        return self.activation(self.propagate(edges, x=features) + self.skip(features))

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.message_mlp(torch.cat((x_i, x_j - x_i), dim=-1))


class SingleJetParticleNet(nn.Module):
    """Dynamic-graph ParticleNet returning one gluon logit per jet."""

    def __init__(
        self,
        *,
        continuous_features: int = 7,
        num_particle_types: int = 1,
        type_embedding_dim: int = 8,
        conv_params: Sequence[tuple[int, Sequence[int]]] = (
            (16, (64, 64, 64)),
            (16, (128, 128, 128)),
            (16, (256, 256, 256)),
        ),
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.type_embedding = (
            nn.Embedding(num_particle_types, type_embedding_dim, padding_idx=0)
            if num_particle_types > 1 else None
        )
        width = continuous_features + (type_embedding_dim if self.type_embedding is not None else 0)
        self.input_norm = BatchNorm(width)
        blocks: list[EdgeConvBlock] = []
        for k, channels in conv_params:
            blocks.append(EdgeConvBlock(width, channels, k))
            width = channels[-1]
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Sequential(
            nn.Linear(width, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: SingleJetBatch | Batch) -> Tensor:
        graph = batch.jets if isinstance(batch, SingleJetBatch) else batch
        features = graph.x
        if self.type_embedding is not None:
            features = torch.cat((features, self.type_embedding(graph.particle_type)), dim=-1)
        features = self.input_norm(features)
        points = graph.pos
        for block in self.blocks:
            features = block(points, features, graph.batch)
            points = features
        pooled = global_mean_pool(features, graph.batch)
        return self.classifier(pooled).squeeze(-1)
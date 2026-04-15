from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import awkward as ak
import numpy as np
import uproot


DEFAULT_PARTICLE_FEATURES: tuple[str, ...] = (
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_deta",
    "part_dphi",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
)

DEFAULT_JET_FEATURES: tuple[str, ...] = (
    "jet_pt",
    "jet_energy",
    "jet_nparticles",
)


@dataclass(frozen=True)
class DenseBatch:
    x_particles: np.ndarray
    x_jets: np.ndarray
    mask: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class ParticleNormalization:
    mean: np.ndarray
    std: np.ndarray


def discover_root_files(data_dir: str | Path, split: str, sample: str) -> list[Path]:
    pattern = f"{sample}_*.root"
    return sorted((Path(data_dir) / split).glob(pattern))


def _tree_paths(paths: Sequence[str | Path]) -> list[str]:
    return [f"{Path(path)}:tree" for path in paths]


def _pad_feature(
    jagged_array: ak.Array,
    max_constituents: int,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    padded = ak.pad_none(jagged_array, max_constituents, axis=1, clip=True)
    filled = ak.fill_none(padded, 0)
    return ak.to_numpy(filled).astype(dtype, copy=False)


def _build_mask(jagged_array: ak.Array, max_constituents: int) -> np.ndarray:
    counts = ak.to_numpy(ak.num(jagged_array, axis=1))
    clipped = np.minimum(counts, max_constituents)
    positions = np.arange(max_constituents, dtype=np.int32)[None, :]
    return positions < clipped[:, None]


def dense_from_awkward(
    arrays: ak.Array,
    particle_features: Sequence[str],
    jet_features: Sequence[str],
    max_constituents: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    particle_tensors = [
        _pad_feature(arrays[name], max_constituents=max_constituents)
        for name in particle_features
    ]
    x_particles = np.stack(particle_tensors, axis=-1)
    x_jets = np.stack(
        [ak.to_numpy(arrays[name]).astype(np.float32, copy=False)
         for name in jet_features],
        axis=-1,
    )
    mask = _build_mask(arrays[particle_features[0]],
                       max_constituents=max_constituents)
    return x_particles, x_jets, mask


def build_binary_labels(size: int, positive_label: int) -> np.ndarray:
    return np.full(size, positive_label, dtype=np.int64)


def apply_particle_normalization(
    x_particles: np.ndarray,
    mask: np.ndarray,
    normalization: ParticleNormalization | None,
) -> np.ndarray:
    if normalization is None:
        return x_particles

    normalized = (x_particles - normalization.mean) / normalization.std
    normalized = normalized.astype(np.float32, copy=False)
    normalized[~mask] = 0.0
    return normalized


def iter_dense_batches(
    paths: Sequence[str | Path],
    labels: Sequence[int],
    *,
    particle_features: Sequence[str] = DEFAULT_PARTICLE_FEATURES,
    jet_features: Sequence[str] = DEFAULT_JET_FEATURES,
    max_constituents: int = 128,
    step_size: str = "10 MB",
    verbose: bool = False,
    normalization: ParticleNormalization | None = None,
) -> Iterator[DenseBatch]:
    if len(paths) != len(labels):
        raise ValueError("paths and labels must have the same length")

    branches = [*particle_features, *jet_features]

    for path, label in zip(paths, labels, strict=True):
        if verbose:
            print(f"[data] reading {Path(path).name} label={label}", flush=True)
        for arrays in uproot.iterate(
            _tree_paths([path]),
            expressions=branches,
            library="ak",
            step_size=step_size,
        ):
            x_particles, x_jets, mask = dense_from_awkward(
                arrays,
                particle_features=particle_features,
                jet_features=jet_features,
                max_constituents=max_constituents,
            )
            x_particles = apply_particle_normalization(
                x_particles,
                mask,
                normalization=normalization,
            )
            labels_batch = build_binary_labels(
                len(x_particles), positive_label=label)
            yield DenseBatch(
                x_particles=x_particles,
                x_jets=x_jets,
                mask=mask,
                labels=labels_batch,
            )


def load_split_arrays(
    bb_paths: Sequence[str | Path],
    gg_paths: Sequence[str | Path],
    *,
    particle_features: Sequence[str] = DEFAULT_PARTICLE_FEATURES,
    jet_features: Sequence[str] = DEFAULT_JET_FEATURES,
    max_constituents: int = 128,
    step_size: str = "50 MB",
) -> DenseBatch:
    particle_batches: list[np.ndarray] = []
    jet_batches: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    sources = ((bb_paths, 1), (gg_paths, 0))
    for paths, label in sources:
        for batch in iter_dense_batches(
            paths,
            [label] * len(paths),
            particle_features=particle_features,
            jet_features=jet_features,
            max_constituents=max_constituents,
            step_size=step_size,
        ):
            particle_batches.append(batch.x_particles)
            jet_batches.append(batch.x_jets)
            masks.append(batch.mask)
            labels.append(batch.labels)

    if not particle_batches:
        raise ValueError("No events were loaded from the provided ROOT files")

    return DenseBatch(
        x_particles=np.concatenate(particle_batches, axis=0),
        x_jets=np.concatenate(jet_batches, axis=0),
        mask=np.concatenate(masks, axis=0),
        labels=np.concatenate(labels, axis=0),
    )


def compute_particle_normalization(
    paths: Sequence[str | Path],
    *,
    particle_features: Sequence[str] = DEFAULT_PARTICLE_FEATURES,
    jet_features: Sequence[str] = DEFAULT_JET_FEATURES,
    max_constituents: int = 128,
    step_size: str = "50 MB",
) -> ParticleNormalization:
    feature_count = len(particle_features)
    total_sum = np.zeros(feature_count, dtype=np.float64)
    total_sum_sq = np.zeros(feature_count, dtype=np.float64)
    total_count = 0

    for batch in iter_dense_batches(
        paths,
        [0] * len(paths),
        particle_features=particle_features,
        jet_features=jet_features,
        max_constituents=max_constituents,
        step_size=step_size,
        verbose=True,
        normalization=None,
    ):
        valid = batch.mask.reshape(-1)
        values = batch.x_particles.reshape(-1, feature_count)[valid]
        total_sum += values.sum(axis=0, dtype=np.float64)
        total_sum_sq += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        total_count += values.shape[0]

    if total_count == 0:
        raise ValueError("No particle entries found while computing normalization")

    mean = total_sum / total_count
    variance = total_sum_sq / total_count - np.square(mean)
    variance = np.maximum(variance, 1e-12)
    std = np.sqrt(variance)
    std = np.where(std < 1e-6, 1.0, std)

    return ParticleNormalization(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )


def save_particle_normalization(path: str | Path, normalization: ParticleNormalization) -> None:
    np.savez(
        Path(path),
        mean=normalization.mean,
        std=normalization.std,
    )


def load_particle_normalization(path: str | Path) -> ParticleNormalization:
    with np.load(Path(path)) as arrays:
        return ParticleNormalization(
            mean=arrays["mean"].astype(np.float32),
            std=arrays["std"].astype(np.float32),
        )

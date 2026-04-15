from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from core.data import (
    DEFAULT_PARTICLE_FEATURES,
    ParticleNormalization,
    discover_root_files,
    iter_dense_batches,
)
from core.model import SimpleParT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick CPU benchmark for SimpleParT")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/simple_part_best.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-constituents", type=int, default=128)
    parser.add_argument("--step-size", type=str, default="10 MB")
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--measure-batches", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = SimpleParT(input_dim=len(DEFAULT_PARTICLE_FEATURES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalization = None
    if checkpoint.get("normalization_mean") is not None and checkpoint.get("normalization_std") is not None:
        normalization = ParticleNormalization(
            mean=np.asarray(checkpoint["normalization_mean"], dtype=np.float32),
            std=np.asarray(checkpoint["normalization_std"], dtype=np.float32),
        )

    bb_paths = discover_root_files(args.data_dir, "val", "HToBB")[:1]
    batch_iter = iter_dense_batches(
        bb_paths,
        [1] * len(bb_paths),
        max_constituents=args.max_constituents,
        step_size=args.step_size,
        normalization=normalization,
    )

    for _ in range(args.warmup_batches):
        batch = next(batch_iter)
        x_particles = torch.from_numpy(batch.x_particles)
        mask = torch.from_numpy(batch.mask)
        with torch.inference_mode():
            _ = model(x_particles, mask)

    durations = []
    tracemalloc.start()
    for _ in range(args.measure_batches):
        batch = next(batch_iter)
        x_particles = torch.from_numpy(batch.x_particles)
        mask = torch.from_numpy(batch.mask)

        start = time.perf_counter()
        with torch.inference_mode():
            _ = model(x_particles, mask)
        durations.append(time.perf_counter() - start)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"mean_latency_s={sum(durations) / len(durations):.6f}")
    print(f"peak_python_memory_mb={peak_bytes / 1024 / 1024:.2f}")
    print(f"parameter_count={sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()

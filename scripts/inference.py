from __future__ import annotations

import argparse
import sys
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
    parser = argparse.ArgumentParser(description="Run inference with a trained SimpleParT checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/simple_part_best.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample", type=str, default="HToBB")
    parser.add_argument("--max-constituents", type=int, default=128)
    parser.add_argument("--step-size", type=str, default="10 MB")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint,
        map_location=args.device,
        weights_only=False,
    )
    normalization = None
    if checkpoint.get("normalization_mean") is not None and checkpoint.get("normalization_std") is not None:
        normalization = ParticleNormalization(
            mean=np.asarray(checkpoint["normalization_mean"], dtype=np.float32),
            std=np.asarray(checkpoint["normalization_std"], dtype=np.float32),
        )

    model = SimpleParT(input_dim=len(DEFAULT_PARTICLE_FEATURES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    paths = discover_root_files(args.data_dir, args.split, args.sample)
    if not paths:
        raise FileNotFoundError(
            f"No ROOT files found for split={args.split!r}, sample={args.sample!r}"
        )

    batch = next(
        iter_dense_batches(
            paths[:1],
            [1 if args.sample == "HToBB" else 0],
            particle_features=DEFAULT_PARTICLE_FEATURES,
            max_constituents=args.max_constituents,
            step_size=args.step_size,
            normalization=normalization,
        )
    )

    x_particles = torch.from_numpy(batch.x_particles).to(args.device)
    mask = torch.from_numpy(batch.mask).to(args.device)

    with torch.inference_mode():
        logits = model(x_particles, mask)
        probs = torch.softmax(logits, dim=1)

    print(f"batch_size={x_particles.size(0)}")
    print(f"first_logits={logits[:5].cpu()}")
    print(f"first_probs={probs[:5].cpu()}")


if __name__ == "__main__":
    main()

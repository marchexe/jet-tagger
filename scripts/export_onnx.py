from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from core.data import DEFAULT_PARTICLE_FEATURES
from core.model import SimpleParT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SimpleParT checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/simple_part_best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/exports/simple_part.onnx"),
    )
    parser.add_argument("--max-constituents", type=int, default=16)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


class SoftmaxWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x_particles: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        logits = self.model(x_particles, padding_mask)
        return torch.softmax(logits, dim=1)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    base_model = SimpleParT(input_dim=len(DEFAULT_PARTICLE_FEATURES))
    base_model.load_state_dict(checkpoint["model_state_dict"])
    model = SoftmaxWrapper(base_model)
    model.eval()

    batch_size = 1
    x_particles = torch.randn(
        batch_size,
        args.max_constituents,
        len(DEFAULT_PARTICLE_FEATURES),
        dtype=torch.float32,
    )
    padding_mask = torch.ones(
        batch_size,
        args.max_constituents,
        dtype=torch.bool,
    )

    with torch.inference_mode():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            (x_particles, padding_mask),
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["x_particles", "padding_mask"],
            output_names=["probabilities"],
            dynamic_axes={
                "x_particles": {0: "batch_size", 1: "num_particles"},
                "padding_mask": {0: "batch_size", 1: "num_particles"},
                "probabilities": {0: "batch_size"},
            },
        )

    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()

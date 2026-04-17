from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from torch import nn
import onnx

from core.benchmark import load_normalization_from_checkpoint
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


class ExportWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        normalization_mean: torch.Tensor | None,
        normalization_std: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.model = model
        if normalization_mean is not None and normalization_std is not None:
            self.register_buffer("normalization_mean", normalization_mean)
            self.register_buffer("normalization_std", normalization_std)
        else:
            self.normalization_mean = None
            self.normalization_std = None

    def forward(self, x_particles: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        if self.normalization_mean is not None and self.normalization_std is not None:
            x_particles = (x_particles - self.normalization_mean) / self.normalization_std
            x_particles = torch.where(
                padding_mask.unsqueeze(-1),
                x_particles,
                torch.zeros_like(x_particles),
            )
        return self.model(x_particles, padding_mask)


def attach_metadata(
    output_path: Path,
    *,
    has_embedded_normalization: bool,
    max_constituents: int,
) -> None:
    model = onnx.load(str(output_path), load_external_data=False)
    metadata = {
        "jet_tagger_model": "SimpleParT",
        "jet_tagger_output_kind": "logits",
        "jet_tagger_embedded_normalization": "true" if has_embedded_normalization else "false",
        "jet_tagger_particle_features": ",".join(DEFAULT_PARTICLE_FEATURES),
        "jet_tagger_max_constituents": str(max_constituents),
    }
    del model.metadata_props[:]
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value
    onnx.save(model, str(output_path))


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    base_model = SimpleParT(input_dim=len(DEFAULT_PARTICLE_FEATURES))
    base_model.load_state_dict(checkpoint["model_state_dict"])
    normalization = load_normalization_from_checkpoint(checkpoint)
    normalization_mean = None
    normalization_std = None
    if normalization is not None:
        normalization_mean = torch.as_tensor(
            normalization.mean,
            dtype=torch.float32,
        ).view(1, 1, -1)
        normalization_std = torch.as_tensor(
            normalization.std,
            dtype=torch.float32,
        ).view(1, 1, -1)

    model = ExportWrapper(base_model, normalization_mean, normalization_std)
    model.eval()

    x_particles = torch.randn(
        1,
        args.max_constituents,
        len(DEFAULT_PARTICLE_FEATURES),
        dtype=torch.float32,
    )
    padding_mask = torch.ones(
        1,
        args.max_constituents,
        dtype=torch.bool,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (x_particles, padding_mask),
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["x_particles", "padding_mask"],
            output_names=["logits"],
        )

    attach_metadata(
        args.output,
        has_embedded_normalization=normalization is not None,
        max_constituents=args.max_constituents,
    )
    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()

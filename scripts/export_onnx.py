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

from core.export import export_checkpoint_to_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SimpleParT checkpoint to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/simple_part_best.pt"),
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=("benchmark", "visual"),
        default="benchmark",
        help="Export benchmark-ready or visualization-friendly ONNX",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional custom output path",
    )
    parser.add_argument("--max-constituents", type=int, default=16)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def default_output_path(variant: str) -> Path:
    if variant == "benchmark":
        return Path("artifacts/exports/simple_part_benchmark.onnx")
    return Path("artifacts/exports/simple_part_visual.onnx")


def resolve_export_options(args: argparse.Namespace) -> tuple[Path, bool, bool]:
    output_path = args.output or default_output_path(args.variant)
    if args.variant == "benchmark":
        return output_path, True, True
    return output_path, False, False


def main() -> None:
    args = parse_args()
    output_path, embed_normalization, dynamic_batch = resolve_export_options(args)
    export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=output_path,
        max_constituents=args.max_constituents,
        opset=args.opset,
        variant=args.variant,
        embed_normalization=embed_normalization,
        dynamic_batch=dynamic_batch,
    )
    print(f"Exported {args.variant} ONNX model to {output_path}")


if __name__ == "__main__":
    main()

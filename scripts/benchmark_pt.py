from __future__ import annotations

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

from core.benchmark import BenchmarkConfig, RuntimeBenchmark, SplitArrays, load_normalization_from_checkpoint
from core.data import DEFAULT_PARTICLE_FEATURES, ParticleNormalization
from core.model import SimpleParT


def parse_args() -> dict:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark PyTorch inference")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/simple_part_best.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-constituents", type=int, default=16)
    parser.add_argument("--step-size", type=str, default="4 MB")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--measure-runs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--latency-max-batches", type=int, default=0)
    parser.add_argument("--memory-max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/logs/benchmark_pytorch.json"),
    )
    return vars(parser.parse_args())


class PyTorchBenchmark(RuntimeBenchmark):
    runtime_name = "pytorch"

    def __init__(self, config: BenchmarkConfig, *, checkpoint_path: Path, device: str) -> None:
        super().__init__(config)
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.checkpoint: dict | None = None
        self.normalization: ParticleNormalization | None = None
        self.model: torch.nn.Module | None = None

    def setup(self) -> dict | None:
        self.log("PyTorch Benchmark")
        self.log(f"device={self.device}, checkpoint={self.checkpoint_path}")
        self.log(
            f"split={self.config.split} batch_size={self.config.batch_size} "
            f"warmup={self.config.warmup_runs} measure={self.config.measure_runs}"
        )

        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.normalization = load_normalization_from_checkpoint(self.checkpoint)

        model = SimpleParT(input_dim=len(DEFAULT_PARTICLE_FEATURES))
        model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model = model.to(self.device)
        return None

    def get_normalization(self) -> ParticleNormalization | None:
        return self.normalization

    def prepare_batch(self, x_batch, mask_batch):
        return (
            torch.from_numpy(x_batch).to(self.device),
            torch.from_numpy(mask_batch).to(self.device),
        )

    def evaluate(self, split_arrays: SplitArrays) -> dict[str, float | str]:
        assert self.model is not None
        criterion = torch.nn.CrossEntropyLoss()
        x_t = torch.from_numpy(split_arrays.x_particles).to(self.device)
        mask_t = torch.from_numpy(split_arrays.mask).to(self.device)
        labels_t = torch.from_numpy(split_arrays.labels).to(self.device)

        self.model.eval()
        with torch.inference_mode():
            logits = self.model(x_t, mask_t)
            loss = float(criterion(logits, labels_t).item())
            preds = logits.argmax(dim=1)
            accuracy = float((preds == labels_t).float().mean().item())
        return {
            "accuracy": accuracy,
            "loss": loss,
            "output_kind": "logits",
        }

    def run_batch(self, batch) -> None:
        assert self.model is not None
        x_t, mask_t = batch
        self.model.eval()
        with torch.inference_mode():
            _ = self.model(x_t, mask_t)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def prepare_peak_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def read_peak_memory(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return torch.cuda.max_memory_allocated(self.device) / 1024 / 1024

    def build_artifacts(self) -> dict[str, str]:
        return {
            "checkpoint": str(self.checkpoint_path),
        }

    def build_extra(self) -> dict[str, object]:
        assert self.model is not None
        return {
            "device": str(self.device),
            "parameter_count": int(sum(param.numel() for param in self.model.parameters())),
            "normalization_embedded_in_model": False,
        }


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        data_dir=Path(args["data_dir"]),
        split=args["split"],
        max_constituents=args["max_constituents"],
        step_size=args["step_size"],
        batch_size=args["batch_size"],
        warmup_runs=args["warmup_runs"],
        measure_runs=args["measure_runs"],
        output_json=Path(args["output_json"]),
        max_events=args["max_events"] or None,
        latency_max_batches=args["latency_max_batches"] or None,
        memory_max_batches=args["memory_max_batches"] or None,
    )
    benchmark = PyTorchBenchmark(
        config,
        checkpoint_path=Path(args["checkpoint"]),
        device=args["device"],
    )
    benchmark.run()


if __name__ == "__main__":
    main()

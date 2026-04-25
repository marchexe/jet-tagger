from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core.benchmark import BenchmarkConfig, RuntimeBenchmark, SplitArrays, compute_classification_metrics
from core.data import ParticleNormalization, load_particle_normalization


try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnx

    HAS_ONNX_PARSER = True
except ImportError:
    HAS_ONNX_PARSER = False


def parse_args() -> dict:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime inference")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/exports/simple_part_benchmark.onnx"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max-constituents", type=int, default=16)
    parser.add_argument("--step-size", type=str, default="4 MB")
    parser.add_argument(
        "--norm-file",
        type=Path,
        default=Path("artifacts/checkpoints/particle_norm.npz"),
    )
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--measure-runs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--latency-max-batches", type=int, default=0)
    parser.add_argument("--memory-max-batches", type=int, default=0)
    parser.add_argument("--providers", type=str, default="CPUExecutionProvider")
    parser.add_argument(
        "--input-normalization",
        type=str,
        choices=("auto", "always", "never"),
        default="auto",
    )
    parser.add_argument(
        "--output-kind",
        type=str,
        choices=("auto", "logits", "probabilities"),
        default="auto",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/logs/benchmark_onnx.json"),
    )
    return vars(parser.parse_args())


def parse_bool_metadata(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    return None


def load_onnx_metadata(onnx_path: Path) -> dict[str, str]:
    if not HAS_ONNX_PARSER or not onnx_path.exists():
        return {}
    model = onnx.load(str(onnx_path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


class OnnxBenchmark(RuntimeBenchmark):
    runtime_name = "onnxruntime"

    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        onnx_path: Path,
        norm_path: Path,
        provider: str,
        input_normalization: str,
        output_kind: str,
    ) -> None:
        super().__init__(config)
        self.onnx_path = onnx_path
        self.norm_path = norm_path
        self.provider = provider
        self.input_normalization = input_normalization
        self.requested_output_kind = output_kind
        self.metadata: dict[str, str] = {}
        self.session: ort.InferenceSession | None = None
        self.normalization: ParticleNormalization | None = None
        self.output_kind = "auto"
        self.effective_batch_size = config.batch_size

    def setup(self) -> dict | None:
        if not HAS_ONNX:
            return self.build_unavailable_result("onnxruntime_not_installed")

        self.log("ONNX Runtime Benchmark")
        self.log(f"onnx={self.onnx_path}, provider={self.provider}")

        self.metadata = load_onnx_metadata(self.onnx_path)
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=[self.provider],
        )

        if self.should_apply_external_normalization():
            if self.norm_path.exists():
                self.normalization = load_particle_normalization(self.norm_path)
                self.log(f"Applying external normalization from {self.norm_path}")
        else:
            self.log("Using raw input tensors for ONNX session")

        self.output_kind = self.resolve_output_kind()
        self.log(f"ONNX outputs: {[output.name for output in self.session.get_outputs()]}")
        self.effective_batch_size = self.resolve_effective_batch_size()
        if self.effective_batch_size == 1:
            target_events = (
                self.config.max_events
                if self.config.max_events is not None and self.config.max_events > 0
                else "all loaded"
            )
            self.log(
                "This ONNX graph is effectively single-event only. "
                f"Latency will run one inference per event over {target_events} events."
            )
        return None

    def should_apply_external_normalization(self) -> bool:
        if self.input_normalization == "always":
            return True
        if self.input_normalization == "never":
            return False

        embedded = parse_bool_metadata(self.metadata.get("jet_tagger_embedded_normalization"))
        if embedded is not None:
            return not embedded

        assert self.session is not None
        output_names = [output.name for output in self.session.get_outputs()]
        if any("probab" in name.lower() for name in output_names):
            return False
        return self.norm_path.exists()

    def resolve_output_kind(self) -> str:
        if self.requested_output_kind != "auto":
            return self.requested_output_kind
        output_kind = self.metadata.get("jet_tagger_output_kind")
        if output_kind in {"logits", "probabilities"}:
            return output_kind
        return "auto"

    def get_normalization(self) -> ParticleNormalization | None:
        return self.normalization

    def cast_mask(self, mask):
        assert self.session is not None
        input_type = self.session.get_inputs()[1].type.lower()
        if "bool" in input_type:
            return mask.astype(bool, copy=False)
        if "uint8" in input_type:
            return mask.astype("uint8", copy=False)
        if "int64" in input_type:
            return mask.astype("int64", copy=False)
        return mask.astype(bool, copy=False)

    def run_model(self, x_particles, mask):
        assert self.session is not None
        ort_inputs = {
            self.session.get_inputs()[0].name: x_particles.astype("float32", copy=False),
            self.session.get_inputs()[1].name: self.cast_mask(mask),
        }
        try:
            return self.session.run(None, ort_inputs)[0]
        except Exception as exc:
            input_shapes = {
                input_meta.name: input_meta.shape
                for input_meta in self.session.get_inputs()
            }
            raise RuntimeError(
                "ONNX model input shape mismatch. "
                f"Session expects {input_shapes}, "
                f"but benchmark passed x_particles={tuple(x_particles.shape)} "
                f"and padding_mask={tuple(mask.shape)}. "
                "This usually means the ONNX file was exported with a static batch size. "
                "Re-export it with `python scripts/export_onnx.py --variant benchmark` and rerun the benchmark."
            ) from exc

    def resolve_effective_batch_size(self) -> int:
        requested = self.config.batch_size
        if requested <= 1:
            return 1

        probe_x = np.zeros(
            (requested, self.config.max_constituents, 16),
            dtype=np.float32,
        )
        probe_mask = np.ones(
            (requested, self.config.max_constituents),
            dtype=bool,
        )
        try:
            self.run_model(probe_x, probe_mask)
            return requested
        except RuntimeError:
            self.log(
                f"ONNX graph does not support batch_size={requested}; "
                "falling back to batch_size=1 for evaluation and latency."
            )
            return 1

    def evaluate(self, split_arrays: SplitArrays) -> dict[str, float | str]:
        outputs = []
        for start_idx in range(0, split_arrays.event_count, self.effective_batch_size):
            end_idx = min(start_idx + self.effective_batch_size, split_arrays.event_count)
            outputs.append(
                self.run_model(
                    split_arrays.x_particles[start_idx:end_idx],
                    split_arrays.mask[start_idx:end_idx],
                )
            )
        outputs = np.concatenate(outputs, axis=0)
        return compute_classification_metrics(
            outputs,
            split_arrays.labels,
            output_kind=self.output_kind,
        )

    def run_batch(self, batch):
        return self.run_model(batch[0], batch[1])

    def prepare_batches(self, split_arrays: SplitArrays):
        from core.benchmark import build_numpy_batches, describe_batches, format_batching_message

        numpy_batches = build_numpy_batches(
            split_arrays.x_particles,
            split_arrays.mask,
            batch_size=self.effective_batch_size,
        )
        batching = describe_batches(
            numpy_batches,
            requested_batch_size=self.config.batch_size,
        )
        self.log(format_batching_message(event_count=split_arrays.event_count, batching=batching)[12:])
        return numpy_batches, batching

    def build_artifacts(self) -> dict[str, str]:
        return {
            "onnx": str(self.onnx_path),
        }

    def build_extra(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "requested_output_kind": self.requested_output_kind,
            "external_input_normalization": self.normalization is not None,
            "effective_batch_size": self.effective_batch_size,
            "onnx_metadata": self.metadata,
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
    benchmark = OnnxBenchmark(
        config,
        onnx_path=Path(args["onnx"]),
        norm_path=Path(args["norm_file"]),
        provider=args["providers"],
        input_normalization=args["input_normalization"],
        output_kind=args["output_kind"],
    )
    benchmark.run()


if __name__ == "__main__":
    main()

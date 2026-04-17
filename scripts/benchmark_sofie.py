from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from core.benchmark import BenchmarkConfig, RuntimeBenchmark, SplitArrays, compute_classification_metrics
from core.data import ParticleNormalization, load_particle_normalization


try:
    import onnx

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import cppyy

    HAS_CPPYY = True
except ImportError:
    HAS_CPPYY = False

try:
    import ROOT  # type: ignore[import-not-found]

    HAS_PYROOT = True
except ImportError:
    HAS_PYROOT = False


def parse_args() -> dict:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark SOFIE inference from Python")
    parser.add_argument(
        "--header",
        type=Path,
        default=Path("artifacts/sofie/simple_part.hxx"),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("artifacts/sofie/simple_part.dat"),
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/exports/simple_part.onnx"),
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
    parser.add_argument(
        "--backend",
        type=str,
        choices=("auto", "cppyy", "pyroot"),
        default="auto",
    )
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
        default=Path("artifacts/logs/benchmark_sofie.json"),
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
    if not HAS_ONNX or not onnx_path.exists():
        return {}
    model = onnx.load(str(onnx_path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


def extract_sofie_namespace(header_path: Path) -> str:
    match = re.search(
        r"namespace\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{",
        header_path.read_text(encoding="utf-8", errors="replace"),
    )
    if not match:
        raise ValueError(f"Could not detect SOFIE namespace in {header_path}")
    return match.group(1)


class SofieRunner:
    def infer(self, x_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CppyySofieRunner(SofieRunner):
    def __init__(self, header_path: Path, weights_path: Path, namespace_name: str) -> None:
        cppyy.add_include_path(str(header_path.parent.resolve()))
        cppyy.include(str(header_path.resolve()))
        cppyy.cppdef(
            f"""
            #include <cstdint>
            namespace jet_tagger_sofie_bridge {{
            std::vector<float> infer_from_ptr(
                {namespace_name}::Session& session,
                std::size_t batch_size,
                std::size_t n_particles,
                std::uintptr_t x_ptr,
                std::uintptr_t mask_ptr
            ) {{
                return session.infer(
                    batch_size,
                    n_particles,
                    reinterpret_cast<float const*>(x_ptr),
                    reinterpret_cast<std::uint8_t const*>(mask_ptr)
                );
            }}
            }}
            """
        )
        self.session = getattr(cppyy.gbl, namespace_name).Session(str(weights_path))
        self.bridge = cppyy.gbl.jet_tagger_sofie_bridge

    def infer(self, x_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray:
        x_flat = np.ascontiguousarray(x_batch.reshape(-1), dtype=np.float32)
        mask_flat = np.ascontiguousarray(mask_batch.reshape(-1), dtype=np.uint8)
        result = self.bridge.infer_from_ptr(
            self.session,
            int(x_batch.shape[0]),
            int(x_batch.shape[1]),
            int(x_flat.ctypes.data),
            int(mask_flat.ctypes.data),
        )
        return np.asarray(result, dtype=np.float32).reshape(x_batch.shape[0], -1)


class PyRootSofieRunner(SofieRunner):
    def __init__(self, header_path: Path, weights_path: Path, namespace_name: str) -> None:
        ROOT.gInterpreter.AddIncludePath(str(header_path.parent.resolve()))
        ROOT.gInterpreter.Declare(f'#include "{header_path.resolve().as_posix()}"')
        ROOT.gInterpreter.Declare(
            f"""
            #include <cstdint>
            namespace jet_tagger_sofie_bridge {{
            std::vector<float> infer_from_ptr(
                {namespace_name}::Session& session,
                std::size_t batch_size,
                std::size_t n_particles,
                std::uintptr_t x_ptr,
                std::uintptr_t mask_ptr
            ) {{
                return session.infer(
                    batch_size,
                    n_particles,
                    reinterpret_cast<float const*>(x_ptr),
                    reinterpret_cast<std::uint8_t const*>(mask_ptr)
                );
            }}
            }}
            """
        )
        self.session = getattr(ROOT, namespace_name).Session(str(weights_path))
        self.bridge = ROOT.jet_tagger_sofie_bridge

    def infer(self, x_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray:
        x_flat = np.ascontiguousarray(x_batch.reshape(-1), dtype=np.float32)
        mask_flat = np.ascontiguousarray(mask_batch.reshape(-1), dtype=np.uint8)
        result = self.bridge.infer_from_ptr(
            self.session,
            int(x_batch.shape[0]),
            int(x_batch.shape[1]),
            int(x_flat.ctypes.data),
            int(mask_flat.ctypes.data),
        )
        return np.asarray(result, dtype=np.float32).reshape(x_batch.shape[0], -1)


class SofieBenchmark(RuntimeBenchmark):
    runtime_name = "sofie"

    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        header_path: Path,
        weights_path: Path,
        onnx_path: Path,
        norm_path: Path,
        backend: str,
        input_normalization: str,
        output_kind: str,
    ) -> None:
        super().__init__(config)
        self.header_path = header_path
        self.weights_path = weights_path
        self.onnx_path = onnx_path
        self.norm_path = norm_path
        self.backend = backend
        self.input_normalization = input_normalization
        self.requested_output_kind = output_kind
        self.metadata: dict[str, str] = {}
        self.normalization: ParticleNormalization | None = None
        self.runner: SofieRunner | None = None
        self.resolved_backend = ""
        self.output_kind = "auto"

    def setup(self) -> dict | None:
        self.log("SOFIE Benchmark")
        self.log(
            f"backend={self.backend} header={self.header_path} weights={self.weights_path}"
        )

        if not self.header_path.exists() or not self.weights_path.exists():
            return self.build_unavailable_result("sofie_artifacts_missing")

        namespace_name = extract_sofie_namespace(self.header_path)
        self.runner, reason = self.create_runner(namespace_name)
        if self.runner is None:
            return self.build_unavailable_result(reason)
        self.resolved_backend = reason

        self.metadata = load_onnx_metadata(self.onnx_path)
        if self.should_apply_external_normalization() and self.norm_path.exists():
            self.normalization = load_particle_normalization(self.norm_path)
            self.log(f"Applying external normalization from {self.norm_path}")
        else:
            self.log("Using raw input tensors for SOFIE session")

        self.output_kind = self.resolve_output_kind()
        return None

    def create_runner(self, namespace_name: str) -> tuple[SofieRunner | None, str]:
        if self.backend in {"auto", "cppyy"} and HAS_CPPYY:
            return CppyySofieRunner(self.header_path, self.weights_path, namespace_name), "cppyy"
        if self.backend == "cppyy":
            return None, "cppyy_not_installed"
        if self.backend in {"auto", "pyroot"} and HAS_PYROOT:
            return PyRootSofieRunner(self.header_path, self.weights_path, namespace_name), "pyroot"
        if self.backend == "pyroot":
            return None, "pyroot_not_installed"
        return None, "no_python_sofie_backend_available"

    def should_apply_external_normalization(self) -> bool:
        if self.input_normalization == "always":
            return True
        if self.input_normalization == "never":
            return False
        embedded = parse_bool_metadata(self.metadata.get("jet_tagger_embedded_normalization"))
        if embedded is not None:
            return not embedded
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

    def evaluate(self, split_arrays: SplitArrays) -> dict[str, float | str]:
        assert self.runner is not None
        outputs = self.runner.infer(split_arrays.x_particles, split_arrays.mask)
        return compute_classification_metrics(
            outputs,
            split_arrays.labels,
            output_kind=self.output_kind,
        )

    def run_batch(self, batch):
        assert self.runner is not None
        return self.runner.infer(batch[0], batch[1])

    def build_artifacts(self) -> dict[str, str]:
        return {
            "header": str(self.header_path),
            "weights": str(self.weights_path),
        }

    def build_extra(self) -> dict[str, object]:
        return {
            "backend": self.resolved_backend,
            "external_input_normalization": self.normalization is not None,
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
    )
    benchmark = SofieBenchmark(
        config,
        header_path=Path(args["header"]),
        weights_path=Path(args["weights"]),
        onnx_path=Path(args["onnx"]),
        norm_path=Path(args["norm_file"]),
        backend=args["backend"],
        input_normalization=args["input_normalization"],
        output_kind=args["output_kind"],
    )
    benchmark.run()


if __name__ == "__main__":
    main()

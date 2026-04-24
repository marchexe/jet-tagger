from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
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
    import ROOT  # type: ignore[import-not-found]

    HAS_PYROOT = True
except ImportError:
    HAS_PYROOT = False


def parse_args() -> dict:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark SOFIE inference through PyROOT")
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
        default=None,
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


def detect_sofie_header_api(header_text: str) -> tuple[str, str]:
    namespace_match = re.search(
        r"namespace\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{(?:(?!namespace\s+[A-Za-z_][A-Za-z0-9_]*\s*\{).)*?(?:class|struct)\s+Session\b",
        header_text,
        re.DOTALL,
    )
    if namespace_match is None:
        raise ValueError("Could not detect SOFIE namespace with Session declaration")

    dynamic_infer = re.search(
        r"\binfer\s*\(\s*size_t\s+[A-Za-z_][A-Za-z0-9_]*\s*,\s*size_t\s+[A-Za-z_][A-Za-z0-9_]*\s*,\s*float\s+const\s*\*\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*(?:std::)?uint8_t\s+const\s*\*",
        header_text,
        re.DOTALL,
    )
    if dynamic_infer is not None:
        return namespace_match.group(1), "dynamic_uint8"

    static_uint8_infer = re.search(
        r"\binfer\s*\(\s*float\s+const\s*\*\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*(?:std::)?uint8_t\s+const\s*\*",
        header_text,
        re.DOTALL,
    )
    if static_uint8_infer is not None:
        return namespace_match.group(1), "static_uint8"

    static_bool_infer = re.search(
        r"\binfer\s*\(\s*float\s+const\s*\*\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*bool\s+const\s*\*",
        header_text,
        re.DOTALL,
    )
    if static_bool_infer is not None:
        return namespace_match.group(1), "static_bool"

    raise ValueError("Could not detect supported SOFIE Session::infer signature")


def discover_root_include_paths() -> list[Path]:
    candidates: list[Path] = []

    def add_candidate(path: Path) -> None:
        header = path / "TMVA" / "SOFIE_common.hxx"
        if header.exists() and path not in candidates:
            candidates.append(path)

    for variable_name in ("ROOTSYS", "CONDA_PREFIX"):
        raw_value = os.environ.get(variable_name)
        if raw_value:
            root_path = Path(raw_value)
            add_candidate(root_path / "include")
            add_candidate(root_path / "include" / "root")

    include_path_value = os.environ.get("ROOT_INCLUDE_PATH", "")
    for raw_value in include_path_value.split(os.pathsep):
        if raw_value.strip():
            add_candidate(Path(raw_value.strip()))

    for fallback_root in (Path("/usr"), Path("/usr/local"), Path("/opt/homebrew")):
        add_candidate(fallback_root / "include")
        add_candidate(fallback_root / "include" / "root")

    return candidates


class PyRootSofieRunner:
    def __init__(
        self,
        header_path: Path,
        weights_path: Path,
        namespace_name: str,
        *,
        max_batch_size: int,
        max_constituents: int,
    ) -> None:
        self.header_text = header_path.read_text(encoding="utf-8", errors="replace")
        detected_namespace_name, self.infer_mode = detect_sofie_header_api(self.header_text)
        self.dynamic_infer = self.infer_mode == "dynamic_uint8"
        if detected_namespace_name != namespace_name:
            raise RuntimeError(
                "Parsed SOFIE namespace does not match the namespace selected by the benchmark "
                f"({detected_namespace_name!r} != {namespace_name!r})"
            )
        include_paths = discover_root_include_paths()
        for include_path in include_paths:
            ROOT.gInterpreter.AddIncludePath(str(include_path.resolve()))
        ROOT.gInterpreter.AddIncludePath(str(header_path.parent.resolve()))

        declare_ok = ROOT.gInterpreter.Declare(f'#include "{header_path.resolve().as_posix()}"')
        if not declare_ok:
            root_hint = ", ".join(str(path) for path in include_paths) or "<none found>"
            raise RuntimeError(
                "Failed to compile SOFIE header against the current ROOT installation. "
                f"Detected ROOT include paths: {root_hint}. "
                "Regenerate the SOFIE artifacts in the same WSL/ROOT environment with "
                "`python scripts/export_sofie.py`."
            )

        if self.dynamic_infer:
            bridge_code = f"""
                #include <cstdint>
                namespace jet_tagger_sofie_bridge {{
                std::vector<float> infer_dynamic_from_ptr(
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
        elif self.infer_mode == "static_bool":
            bridge_code = f"""
                #include <cstdint>
                namespace jet_tagger_sofie_bridge {{
                std::vector<float> infer_static_from_ptr(
                    {namespace_name}::Session& session,
                    std::uintptr_t x_ptr,
                    std::uintptr_t mask_ptr
                ) {{
                    return session.infer(
                        reinterpret_cast<float const*>(x_ptr),
                        reinterpret_cast<bool const*>(mask_ptr)
                    );
                }}
                }}
            """
        else:
            bridge_code = f"""
                #include <cstdint>
                namespace jet_tagger_sofie_bridge {{
                std::vector<float> infer_static_from_ptr(
                    {namespace_name}::Session& session,
                    std::uintptr_t x_ptr,
                    std::uintptr_t mask_ptr
                ) {{
                    return session.infer(
                        reinterpret_cast<float const*>(x_ptr),
                        reinterpret_cast<std::uint8_t const*>(mask_ptr)
                    );
                }}
                }}
            """

        ROOT.gInterpreter.Declare(bridge_code)

        session_type = getattr(ROOT, namespace_name).Session
        if self.dynamic_infer:
            self.session = session_type(
                str(weights_path),
                int(max_batch_size),
                int(max_constituents),
            )
        else:
            self.session = session_type(str(weights_path))
        self.bridge = ROOT.jet_tagger_sofie_bridge

    def infer(self, x_batch: np.ndarray, mask_batch: np.ndarray) -> np.ndarray:
        x_flat = np.ascontiguousarray(x_batch.reshape(-1), dtype=np.float32)
        if self.dynamic_infer:
            mask_flat = np.ascontiguousarray(mask_batch.reshape(-1), dtype=np.uint8)
            result = self.bridge.infer_dynamic_from_ptr(
                self.session,
                int(x_batch.shape[0]),
                int(x_batch.shape[1]),
                int(x_flat.ctypes.data),
                int(mask_flat.ctypes.data),
            )
        else:
            mask_dtype = np.bool_ if self.infer_mode == "static_bool" else np.uint8
            mask_flat = np.ascontiguousarray(mask_batch.reshape(-1), dtype=mask_dtype)
            result = self.bridge.infer_static_from_ptr(
                self.session,
                int(x_flat.ctypes.data),
                int(mask_flat.ctypes.data),
            )
        return np.asarray(result, dtype=np.float32).reshape(x_batch.shape[0], -1)


@dataclass
class SofieBatch:
    x_batch: np.ndarray
    mask_batch: np.ndarray
    actual_size: int


class SofieBenchmark(RuntimeBenchmark):
    runtime_name = "sofie"

    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        header_path: Path,
        weights_path: Path,
        onnx_path: Path | None,
        norm_path: Path,
        input_normalization: str,
        output_kind: str,
    ) -> None:
        super().__init__(config)
        self.header_path = header_path
        self.weights_path = weights_path
        self.onnx_path = onnx_path
        self.norm_path = norm_path
        self.input_normalization = input_normalization
        self.requested_output_kind = output_kind
        self.metadata: dict[str, str] = {}
        self.normalization: ParticleNormalization | None = None
        self.runner: PyRootSofieRunner | None = None
        self.output_kind = "auto"

    def setup(self) -> dict | None:
        self.log("SOFIE Benchmark")
        self.log(f"backend=pyroot header={self.header_path} weights={self.weights_path}")

        if not self.header_path.exists() or not self.weights_path.exists():
            return self.build_unavailable_result("sofie_artifacts_missing")
        if not HAS_PYROOT:
            return self.build_unavailable_result("pyroot_not_installed")

        header_text = self.header_path.read_text(encoding="utf-8", errors="replace")
        try:
            namespace_name, _ = detect_sofie_header_api(header_text)
        except ValueError as exc:
            self.log(f"pyroot setup failed: {exc}")
            return self.build_unavailable_result("unsupported_sofie_header")
        try:
            self.runner = PyRootSofieRunner(
                self.header_path,
                self.weights_path,
                namespace_name,
                max_batch_size=self.config.batch_size,
                max_constituents=self.config.max_constituents,
            )
        except Exception as exc:
            self.log(f"pyroot setup failed: {exc}")
            return self.build_unavailable_result("pyroot_setup_failed")

        self.metadata = load_onnx_metadata(self.onnx_path) if self.onnx_path is not None else {}
        if self.should_apply_external_normalization() and self.norm_path.exists():
            self.normalization = load_particle_normalization(self.norm_path)
            self.log(f"Applying external normalization from {self.norm_path}")
        else:
            self.log("Using raw input tensors for SOFIE session")

        self.output_kind = self.resolve_output_kind()
        return None

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
        header_text = self.header_path.read_text(encoding="utf-8", errors="replace")
        if "tensor_probabilities" in header_text:
            return "probabilities"
        if "tensor_logits" in header_text:
            return "logits"
        output_kind = self.metadata.get("jet_tagger_output_kind")
        if output_kind in {"logits", "probabilities"}:
            return output_kind
        return "auto"

    def get_normalization(self) -> ParticleNormalization | None:
        return self.normalization

    def evaluate(self, split_arrays: SplitArrays) -> dict[str, float | str]:
        assert self.runner is not None
        batches, _ = self.prepare_batches(split_arrays)
        outputs = []
        for batch in batches:
            batch_outputs = self.run_batch(batch)
            outputs.append(batch_outputs[: batch.actual_size])
        outputs = np.concatenate(outputs, axis=0)
        return compute_classification_metrics(
            outputs,
            split_arrays.labels,
            output_kind=self.output_kind,
        )

    def prepare_batch(self, x_batch: np.ndarray, mask_batch: np.ndarray) -> SofieBatch:
        actual_size = int(x_batch.shape[0])
        target_size = self.config.batch_size
        if actual_size == target_size:
            return SofieBatch(x_batch=x_batch, mask_batch=mask_batch, actual_size=actual_size)

        padded_x = np.zeros(
            (target_size, x_batch.shape[1], x_batch.shape[2]),
            dtype=x_batch.dtype,
        )
        padded_mask = np.zeros(
            (target_size, mask_batch.shape[1]),
            dtype=mask_batch.dtype,
        )
        padded_x[:actual_size] = x_batch
        padded_mask[:actual_size] = mask_batch
        return SofieBatch(x_batch=padded_x, mask_batch=padded_mask, actual_size=actual_size)

    def run_batch(self, batch):
        assert self.runner is not None
        return self.runner.infer(batch.x_batch, batch.mask_batch)

    def batch_event_count(self, batch) -> int:
        if isinstance(batch, SofieBatch):
            return batch.actual_size
        return super().batch_event_count(batch)

    def build_artifacts(self) -> dict[str, str]:
        return {
            "header": str(self.header_path),
            "weights": str(self.weights_path),
        }

    def build_extra(self) -> dict[str, object]:
        return {
            "backend": "pyroot",
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
        max_events=args["max_events"] or None,
    )
    benchmark = SofieBenchmark(
        config,
        header_path=Path(args["header"]),
        weights_path=Path(args["weights"]),
        onnx_path=Path(args["onnx"]) if args["onnx"] is not None else None,
        norm_path=Path(args["norm_file"]),
        input_normalization=args["input_normalization"],
        output_kind=args["output_kind"],
    )
    benchmark.run()


if __name__ == "__main__":
    main()

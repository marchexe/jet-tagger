from __future__ import annotations

import gc
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np
import psutil

from .data import (
    ParticleNormalization,
    discover_root_files,
    iter_dense_batches,
)


@dataclass(frozen=True)
class SplitArrays:
    x_particles: np.ndarray
    mask: np.ndarray
    labels: np.ndarray

    @property
    def event_count(self) -> int:
        return int(self.labels.shape[0])


@dataclass(frozen=True)
class BenchmarkConfig:
    data_dir: Path
    split: str
    max_constituents: int
    step_size: str
    batch_size: int
    warmup_runs: int
    measure_runs: int
    output_json: Path
    max_events: int | None = None


def load_normalization_from_checkpoint(
    checkpoint: dict[str, Any],
) -> ParticleNormalization | None:
    legacy = checkpoint.get("normalization")
    if isinstance(legacy, dict):
        mean = legacy.get("mean")
        std = legacy.get("std")
        if mean is not None and std is not None:
            return ParticleNormalization(
                mean=np.asarray(mean, dtype=np.float32),
                std=np.asarray(std, dtype=np.float32),
            )

    mean = checkpoint.get("normalization_mean")
    std = checkpoint.get("normalization_std")
    if mean is None or std is None:
        return None

    return ParticleNormalization(
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
    )


def collect_split_arrays(
    *,
    data_dir: Path,
    split: str,
    max_constituents: int,
    step_size: str,
    normalization: ParticleNormalization | None,
    max_events: int | None = None,
) -> SplitArrays:
    print(f"[benchmark] Loading {split} split...", flush=True)

    bb_paths = discover_root_files(data_dir, split, "HToBB")
    gg_paths = discover_root_files(data_dir, split, "HToGG")

    x_particles_list: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    loaded_events = 0

    sources = ((bb_paths, 1), (gg_paths, 0))
    for paths, label in sources:
        for batch in iter_dense_batches(
            paths,
            [label] * len(paths),
            max_constituents=max_constituents,
            step_size=step_size,
            normalization=normalization,
            verbose=False,
        ):
            remaining_events = None
            if max_events is not None and max_events > 0:
                remaining_events = max_events - loaded_events
                if remaining_events <= 0:
                    break

            x_particles = batch.x_particles
            mask = batch.mask
            labels = batch.labels
            if remaining_events is not None and x_particles.shape[0] > remaining_events:
                x_particles = x_particles[:remaining_events]
                mask = mask[:remaining_events]
                labels = labels[:remaining_events]

            x_particles_list.append(x_particles)
            masks_list.append(mask)
            labels_list.append(labels)
            loaded_events += int(labels.shape[0])

        if max_events is not None and max_events > 0 and loaded_events >= max_events:
            break

    if not x_particles_list:
        raise ValueError(
            f"No events were loaded from split={split!r} under {data_dir}"
        )

    split_arrays = SplitArrays(
        x_particles=np.concatenate(x_particles_list, axis=0),
        mask=np.concatenate(masks_list, axis=0),
        labels=np.concatenate(labels_list, axis=0),
    )
    print(f"[benchmark] Loaded {split_arrays.event_count} events", flush=True)
    return split_arrays


def build_numpy_batches(
    x_particles: np.ndarray,
    mask: np.ndarray,
    *,
    batch_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    num_events = int(x_particles.shape[0])
    if num_events == 0:
        raise ValueError("Cannot benchmark an empty event set")

    batches: list[tuple[np.ndarray, np.ndarray]] = []
    for start_idx in range(0, num_events, batch_size):
        end_idx = min(start_idx + batch_size, num_events)
        batches.append(
            (x_particles[start_idx:end_idx], mask[start_idx:end_idx]))
    return batches


def describe_batches(
    batches: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    requested_batch_size: int | None = None,
) -> dict[str, int]:
    batch_sizes = [int(x_batch.shape[0]) for x_batch, _ in batches]
    if not batch_sizes:
        raise ValueError("No batches were created")
    return {
        "batch_count": len(batch_sizes),
        "requested_batch_size": (
            int(requested_batch_size)
            if requested_batch_size is not None
            else max(batch_sizes)
        ),
        "min_batch_size": min(batch_sizes),
        "max_batch_size": max(batch_sizes),
    }


def format_batching_message(
    *,
    event_count: int,
    batching: dict[str, int],
) -> str:
    return (
        "[benchmark] "
        f"Latency/memory input: {event_count} events -> "
        f"{batching['batch_count']} batches "
        f"(requested_batch_size={batching['requested_batch_size']}, "
        f"min={batching['min_batch_size']}, max={batching['max_batch_size']})"
    )


def classify_output_kind(outputs: np.ndarray) -> str:
    if outputs.ndim != 2 or outputs.shape[1] == 0:
        raise ValueError(
            f"Expected 2D classifier outputs, got shape={outputs.shape}")

    if not np.isfinite(outputs).all():
        return "logits"

    row_sums = outputs.sum(axis=1)
    looks_like_probabilities = (
        np.all(outputs >= -1e-6)
        and np.all(outputs <= 1.0 + 1e-6)
        and np.allclose(row_sums, 1.0, atol=1e-3, rtol=1e-3)
    )
    return "probabilities" if looks_like_probabilities else "logits"


def compute_classification_metrics(
    outputs: np.ndarray,
    labels: np.ndarray,
    *,
    output_kind: str = "auto",
) -> dict[str, float | str]:
    outputs = np.asarray(outputs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if output_kind == "auto":
        resolved_output_kind = classify_output_kind(outputs)
    elif output_kind in {"logits", "probabilities"}:
        resolved_output_kind = output_kind
    else:
        raise ValueError(f"Unsupported output_kind={output_kind!r}")

    preds = outputs.argmax(axis=1)
    accuracy = float((preds == labels).astype(np.float32).mean())

    if resolved_output_kind == "probabilities":
        safe_probs = np.clip(outputs, 1e-12, 1.0)
        loss = float(-np.log(safe_probs[np.arange(len(labels)), labels]).mean())
    else:
        logits_safe = np.clip(outputs, -100.0, 100.0)
        row_max = np.max(logits_safe, axis=1, keepdims=True)
        logsumexp = row_max + \
            np.log(np.exp(logits_safe - row_max).sum(axis=1, keepdims=True))
        log_softmax = logits_safe - logsumexp
        loss = float(-log_softmax[np.arange(len(labels)), labels].mean())

    return {
        "accuracy": accuracy,
        "loss": loss,
        "output_kind": resolved_output_kind,
    }


def summarize_latency(
    durations_ms: Sequence[float],
    *,
    total_events: int,
) -> dict[str, float | int]:
    durations = np.asarray(durations_ms, dtype=np.float64)
    total_duration_s = float(durations.sum() / 1000.0)
    throughput = float(
        total_events / total_duration_s) if total_duration_s > 0 else 0.0
    return {
        "samples": int(durations.size),
        "mean_ms": float(np.mean(durations)),
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
        "p99_ms": float(np.percentile(durations, 99)),
        "throughput_events_per_s": throughput,
    }


def benchmark_latency(
    batches: Sequence[Any],
    *,
    run_batch: Callable[[Any], Any],
    batch_size_of: Callable[[Any], int],
    warmup_runs: int,
    measure_runs: int,
    synchronize: Callable[[], None] | None = None,
    progress_name: str = "Latency",
) -> dict[str, float | int]:
    print(f"[benchmark] Warmup {warmup_runs} runs...", flush=True)
    for _ in range(warmup_runs):
        for batch in batches:
            run_batch(batch)

    if synchronize is not None:
        synchronize()
    gc.collect()
    time.sleep(0.1)

    print(f"[benchmark] Measuring {measure_runs} runs...", flush=True)
    durations_ms: list[float] = []
    total_events = 0

    for _ in range(measure_runs):
        run_index = _ + 1
        for batch in batches:
            start = time.perf_counter()
            run_batch(batch)
            if synchronize is not None:
                synchronize()
            elapsed_s = time.perf_counter() - start
            durations_ms.append(elapsed_s * 1000.0)
            total_events += int(batch_size_of(batch))
        if measure_runs > 1:
            print(
                f"[benchmark] {progress_name} progress: run {run_index}/{measure_runs}",
                flush=True,
            )

    return summarize_latency(durations_ms, total_events=total_events)


def benchmark_memory(
    batches: Sequence[Any],
    *,
    run_batch: Callable[[Any], Any],
    synchronize: Callable[[], None] | None = None,
    prepare_peak_memory: Callable[[], None] | None = None,
    read_peak_memory: Callable[[], float | None] | None = None,
) -> dict[str, float | None]:
    process = psutil.Process()

    if prepare_peak_memory is not None:
        prepare_peak_memory()
    if synchronize is not None:
        synchronize()
    gc.collect()
    time.sleep(0.1)

    rss_baseline_mb = process.memory_info().rss / 1024 / 1024
    peak_rss_mb = rss_baseline_mb
    print(
        f"[benchmark] Memory baseline RSS={rss_baseline_mb:.2f} MB", flush=True)

    for _ in range(3):
        for batch in batches:
            run_batch(batch)
            if synchronize is not None:
                synchronize()
            current_rss_mb = process.memory_info().rss / 1024 / 1024
            peak_rss_mb = max(peak_rss_mb, current_rss_mb)

    peak_device_mb = None if read_peak_memory is None else read_peak_memory()

    return {
        "baseline_rss_mb": float(rss_baseline_mb),
        "peak_rss_mb": float(peak_rss_mb),
        "delta_rss_mb": float(peak_rss_mb - rss_baseline_mb),
        "peak_cuda_allocated_mb": None if peak_device_mb is None else float(peak_device_mb),
    }


def build_runtime_result(
    *,
    model: str,
    runtime: str,
    split: str,
    event_count: int,
    batching: dict[str, int],
    metrics: dict[str, float | str],
    latency: dict[str, float | int],
    memory: dict[str, float | int | None],
    artifacts: dict[str, str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 2,
        "model": model,
        "runtime": runtime,
        "split": split,
        "event_count": event_count,
        "batching": batching,
        "metrics": metrics,
        "latency": latency,
        "memory": memory,
        "artifacts": artifacts,
    }
    if extra:
        result["extra"] = extra
    return result


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


class RuntimeBenchmark(ABC):
    model_name = "SimpleParT"
    runtime_name: str

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def log(self, message: str) -> None:
        print(f"[benchmark] {message}", flush=True)

    def build_unavailable_result(self, reason: str) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "model": self.model_name,
            "runtime": self.runtime_name,
            "available": False,
            "reason": reason,
        }

    def write_result(self, result: dict[str, Any]) -> None:
        write_result(self.config.output_json, result)

    def setup(self) -> dict[str, Any] | None:
        return None

    def get_normalization(self) -> ParticleNormalization | None:
        return None

    def load_split_arrays(self) -> SplitArrays:
        return collect_split_arrays(
            data_dir=self.config.data_dir,
            split=self.config.split,
            max_constituents=self.config.max_constituents,
            step_size=self.config.step_size,
            normalization=self.get_normalization(),
            max_events=self.config.max_events,
        )

    def prepare_batch(self, x_batch: np.ndarray, mask_batch: np.ndarray) -> Any:
        return (x_batch, mask_batch)

    def prepare_batches(self, split_arrays: SplitArrays) -> tuple[list[Any], dict[str, int]]:
        numpy_batches = build_numpy_batches(
            split_arrays.x_particles,
            split_arrays.mask,
            batch_size=self.config.batch_size,
        )
        batching = describe_batches(
            numpy_batches,
            requested_batch_size=self.config.batch_size,
        )
        self.log(format_batching_message(
            event_count=split_arrays.event_count, batching=batching)[12:])
        return [self.prepare_batch(x_batch, mask_batch) for x_batch, mask_batch in numpy_batches], batching

    def batch_event_count(self, batch: Any) -> int:
        if isinstance(batch, tuple) and batch:
            first = batch[0]
            if hasattr(first, "shape"):
                return int(first.shape[0])
        raise TypeError("Override batch_event_count() for custom batch types")

    def synchronize(self) -> None:
        return None

    def prepare_peak_memory(self) -> None:
        return None

    def read_peak_memory(self) -> float | None:
        return None

    @abstractmethod
    def evaluate(self, split_arrays: SplitArrays) -> dict[str, float | str]:
        raise NotImplementedError

    @abstractmethod
    def run_batch(self, batch: Any) -> Any:
        raise NotImplementedError

    def build_artifacts(self) -> dict[str, str]:
        return {}

    def build_extra(self) -> dict[str, Any] | None:
        return None

    def run(self) -> dict[str, Any]:
        unavailable_result = self.setup()
        if unavailable_result is not None:
            self.write_result(unavailable_result)
            self.log(f"Wrote unavailable result to {self.config.output_json}")
            return unavailable_result

        split_arrays = self.load_split_arrays()

        print("\n[benchmark] Evaluating...", flush=True)
        metrics = self.evaluate(split_arrays)
        if "accuracy" in metrics and "loss" in metrics:
            summary = f"accuracy={metrics['accuracy']:.4f}, loss={metrics['loss']:.4f}"
            if "output_kind" in metrics:
                summary += f", output_kind={metrics['output_kind']}"
            self.log(summary)

        batches, batching = self.prepare_batches(split_arrays)

        print("\n[benchmark] Benchmarking latency...", flush=True)
        latency = benchmark_latency(
            batches,
            run_batch=self.run_batch,
            batch_size_of=self.batch_event_count,
            warmup_runs=self.config.warmup_runs,
            measure_runs=self.config.measure_runs,
            synchronize=self.synchronize,
        )

        print("\n[benchmark] Benchmarking memory...", flush=True)
        memory = benchmark_memory(
            batches,
            run_batch=self.run_batch,
            synchronize=self.synchronize,
            prepare_peak_memory=self.prepare_peak_memory,
            read_peak_memory=self.read_peak_memory,
        )

        result = build_runtime_result(
            model=self.model_name,
            runtime=self.runtime_name,
            split=self.config.split,
            event_count=split_arrays.event_count,
            batching=batching,
            metrics=metrics,
            latency=latency,
            memory=memory,
            artifacts=self.build_artifacts(),
            extra=self.build_extra(),
        )
        self.write_result(result)
        self.log(f"Results saved to {self.config.output_json}")
        print(json.dumps(result, indent=2), flush=True)
        return result

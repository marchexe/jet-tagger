from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark markdown and metric plots"
    )
    parser.add_argument(
        "--pytorch-json",
        type=Path,
        default=Path("artifacts/logs/benchmark_pytorch.json"),
    )
    parser.add_argument(
        "--onnx-json",
        type=Path,
        default=Path("artifacts/logs/benchmark_onnx.json"),
    )
    parser.add_argument(
        "--sofie-json",
        type=Path,
        default=Path("artifacts/logs/benchmark_sofie.json"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("artifacts/logs/benchmark_metrics.png"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("artifacts/logs/benchmark_table.md"),
    )
    return parser.parse_args()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, *, digits: int = 4, suffix: str = "") -> str:
    if value in (None, ""):
        return ""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return f"{value}{suffix}"
    return f"{numeric_value:.{digits}f}{suffix}"


def join_notes(parts: list[str]) -> str:
    return "; ".join(part for part in parts if part)


def row_from_runtime(data: dict[str, Any] | None, *, runtime_label: str) -> list[str]:
    if not data:
        return ["SimpleParT", runtime_label, "", "", "", "", "", "", "", "", "missing"]

    if data.get("available") is False:
        return [
            data.get("model", "SimpleParT"),
            runtime_label,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            data.get("reason", "unavailable"),
        ]

    metrics = data.get("metrics", {})
    latency = data.get("latency", {})
    memory = data.get("memory", {})
    batching = data.get("batching", {})
    extra = data.get("extra", {})

    notes = join_notes(
        [
            f"split={data.get('split', '')}" if data.get("split") else "",
            (
                f"provider={extra.get('provider')}"
                if extra.get("provider")
                else ""
            ),
            (
                f"batch={batching.get('requested_batch_size')}"
                if batching.get("requested_batch_size")
                else ""
            ),
            (
                f"range={batching.get('min_batch_size')}-{batching.get('max_batch_size')}"
                if batching.get("min_batch_size") is not None
                and batching.get("max_batch_size") is not None
                else ""
            ),
            (
                "external_norm=yes"
                if extra.get("external_input_normalization") is True
                else "external_norm=no"
                if extra.get("external_input_normalization") is False
                else ""
            ),
        ]
    )

    return [
        data.get("model", "SimpleParT"),
        runtime_label,
        str(data.get("event_count", "")),
        fmt(metrics.get("accuracy")),
        fmt(metrics.get("loss")),
        str(metrics.get("output_kind", "")),
        fmt(latency.get("mean_ms"), digits=3, suffix=" ms"),
        fmt(latency.get("p95_ms"), digits=3, suffix=" ms"),
        fmt(latency.get("throughput_events_per_s"), digits=1, suffix=" ev/s"),
        fmt(memory.get("peak_rss_mb"), digits=1, suffix=" MB"),
        notes,
    ]


def build_rows(
    pytorch: dict[str, Any] | None,
    onnx: dict[str, Any] | None,
    sofie: dict[str, Any] | None,
) -> list[list[str]]:
    return [
        row_from_runtime(pytorch, runtime_label="PyTorch"),
        row_from_runtime(onnx, runtime_label="ONNX Runtime"),
        row_from_runtime(sofie, runtime_label="SOFIE"),
    ]


def metric_value(data: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if current in (None, ""):
        return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def build_metric_records(
    pytorch: dict[str, Any] | None,
    onnx: dict[str, Any] | None,
    sofie: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidates = [
        ("PyTorch", pytorch),
        ("ONNX Runtime", onnx),
        ("SOFIE", sofie),
    ]
    records: list[dict[str, Any]] = []
    for runtime, data in candidates:
        if not data or data.get("available") is False or "metrics" not in data:
            continue
        records.append(
            {
                "runtime": runtime,
                "accuracy": metric_value(data, ("metrics", "accuracy")),
                "loss": metric_value(data, ("metrics", "loss")),
                "latency_mean_ms": metric_value(data, ("latency", "mean_ms")),
                "latency_p95_ms": metric_value(data, ("latency", "p95_ms")),
                "throughput_events_per_s": metric_value(
                    data,
                    ("latency", "throughput_events_per_s"),
                ),
                "peak_rss_mb": metric_value(data, ("memory", "peak_rss_mb")),
            }
        )
    return records


def write_markdown(headers: list[str], rows: list[list[str]], output: Path) -> None:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_metric_plots(records: list[dict[str, Any]], output: Path) -> None:
    if not HAS_MATPLOTLIB:
        print(f"Skipping PNG plot because matplotlib is not installed: {output}")
        return
    if not records:
        print(f"Skipping PNG plot because no benchmark metrics were found: {output}")
        return

    plt.rcParams["font.family"] = "DejaVu Sans"
    metric_specs = [
        ("Accuracy", "accuracy", ""),
        ("Loss", "loss", ""),
        ("Mean latency", "latency_mean_ms", "ms"),
        ("P95 latency", "latency_p95_ms", "ms"),
        ("Throughput", "throughput_events_per_s", "events/s"),
        ("Peak RSS", "peak_rss_mb", "MB"),
    ]
    colors = {
        "PyTorch": "#335c67",
        "ONNX Runtime": "#e09f3e",
        "SOFIE": "#7a9e7e",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle("SimpleParT Runtime Benchmark Metrics", fontsize=18, fontweight="bold")
    runtimes = [record["runtime"] for record in records]

    for ax, (title, key, unit) in zip(axes.flatten(), metric_specs, strict=True):
        values = [record.get(key) for record in records]
        numeric_values = [0.0 if value is None else float(value) for value in values]
        bars = ax.bar(
            runtimes,
            numeric_values,
            color=[colors.get(runtime, "#6c757d") for runtime in runtimes],
        )
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=18)
        if unit:
            ax.set_ylabel(unit)

        for bar, value in zip(bars, values, strict=True):
            if value is None:
                label = "n/a"
                y_pos = bar.get_height()
            elif abs(float(value)) >= 1000:
                label = f"{float(value):,.0f}"
                y_pos = bar.get_height()
            elif abs(float(value)) >= 10:
                label = f"{float(value):.2f}"
                y_pos = bar.get_height()
            else:
                label = f"{float(value):.4f}"
                y_pos = bar.get_height()
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pytorch = load_json_if_exists(args.pytorch_json)
    onnx = load_json_if_exists(args.onnx_json)
    sofie = load_json_if_exists(args.sofie_json)

    headers = [
        "Model",
        "Runtime",
        "Events",
        "Accuracy",
        "Loss",
        "Output",
        "Latency Mean",
        "Latency P95",
        "Throughput",
        "Peak RSS",
        "Notes",
    ]

    rows = build_rows(pytorch, onnx, sofie)
    records = build_metric_records(pytorch, onnx, sofie)
    write_markdown(headers, rows, args.output_md)
    draw_metric_plots(records, args.output_png)

    print(f"Wrote markdown table to {args.output_md}")
    if HAS_MATPLOTLIB:
        print(f"Wrote metric plot to {args.output_png}")


if __name__ == "__main__":
    main()

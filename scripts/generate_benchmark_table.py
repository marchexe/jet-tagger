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
        description="Generate a benchmark table figure with matplotlib"
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
        "--sofie-text",
        type=Path,
        default=Path("artifacts/logs/benchmark_sofie.txt"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("artifacts/logs/benchmark_table.png"),
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


def load_sofie_text(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result if result else None


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


def row_from_sofie(sofie: dict[str, Any] | None) -> list[str]:
    if not sofie:
        return ["SimpleParT", "SOFIE C++", "", "", "", "", "", "", "", "", "missing"]

    if sofie.get("available") is False:
        return [
            sofie.get("model", "SimpleParT"),
            "SOFIE",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            sofie.get("reason", "unavailable"),
        ]

    if "metrics" in sofie:
        return row_from_runtime(sofie, runtime_label="SOFIE")

    notes = join_notes(
        [
            f"batch={sofie.get('requested_batch_size', '')}" if sofie.get("requested_batch_size") else "",
            (
                f"range={sofie.get('min_batch_size', '')}-{sofie.get('max_batch_size', '')}"
                if sofie.get("min_batch_size") and sofie.get("max_batch_size")
                else ""
            ),
            f"init={sofie.get('init_time_ms', '')} ms" if sofie.get("init_time_ms") else "",
        ]
    )

    return [
        "SimpleParT",
        "SOFIE C++",
        sofie.get("events", ""),
        "",
        "",
        "",
        fmt(sofie.get("mean_latency_ms"), digits=3, suffix=" ms"),
        fmt(sofie.get("p95_latency_ms"), digits=3, suffix=" ms"),
        fmt(sofie.get("throughput_events_per_s"), digits=1, suffix=" ev/s"),
        "",
        notes,
    ]


def build_rows(
    pytorch: dict[str, Any] | None,
    onnx: dict[str, Any] | None,
    sofie: dict[str, str] | None,
) -> list[list[str]]:
    return [
        row_from_runtime(pytorch, runtime_label="PyTorch"),
        row_from_runtime(onnx, runtime_label="ONNX Runtime"),
        row_from_sofie(sofie),
    ]


def write_markdown(headers: list[str], rows: list[list[str]], output: Path) -> None:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_table(headers: list[str], rows: list[list[str]], output: Path) -> None:
    if not HAS_MATPLOTLIB:
        print(f"Skipping PNG table because matplotlib is not installed: {output}")
        return

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(22, 4.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.9)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f4e79")
        elif row == 2:
            cell.set_facecolor("#eef5ea")
        else:
            cell.set_facecolor("#f8f9f9" if row % 2 == 0 else "#ffffff")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pytorch = load_json_if_exists(args.pytorch_json)
    onnx = load_json_if_exists(args.onnx_json)
    sofie = load_json_if_exists(args.sofie_json)
    if sofie is None:
        sofie = load_sofie_text(args.sofie_text)

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
    write_markdown(headers, rows, args.output_md)
    draw_table(headers, rows, args.output_png)

    print(f"Wrote markdown table to {args.output_md}")
    if HAS_MATPLOTLIB:
        print(f"Wrote figure table to {args.output_png}")


if __name__ == "__main__":
    main()

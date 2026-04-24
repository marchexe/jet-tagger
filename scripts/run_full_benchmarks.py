from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.system_info import collect_system_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full PyTorch + ONNX + SOFIE benchmark pipeline and log system info."
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run the benchmark scripts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/simple_part_best.pt"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--measure-runs", type=int, default=20)
    parser.add_argument("--max-constituents", type=int, default=16)
    parser.add_argument("--step-size", type=str, default="4 MB")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--providers", type=str, default="CPUExecutionProvider")
    parser.add_argument(
        "--system-json",
        type=Path,
        default=Path("artifacts/logs/benchmark_system.json"),
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


def run_command(command: list[str]) -> None:
    print(f"[pipeline] Running: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()

    run_config = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "batch_size": args.batch_size,
        "max_events": None if args.max_events <= 0 else args.max_events,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "max_constituents": args.max_constituents,
        "step_size": args.step_size,
        "device": args.device,
        "providers": args.providers,
        "python": args.python,
    }

    args.system_json.parent.mkdir(parents=True, exist_ok=True)
    args.system_json.write_text(
        json.dumps(collect_system_info(run_config=run_config), indent=2),
        encoding="utf-8",
    )
    print(f"[pipeline] Wrote system info to {args.system_json}", flush=True)

    common_args = [
        "--data-dir",
        str(args.data_dir),
        "--split",
        args.split,
        "--max-constituents",
        str(args.max_constituents),
        "--step-size",
        args.step_size,
        "--batch-size",
        str(args.batch_size),
        "--warmup-runs",
        str(args.warmup_runs),
        "--measure-runs",
        str(args.measure_runs),
    ]
    if args.max_events > 0:
        common_args.extend(["--max-events", str(args.max_events)])

    python_executable = args.python

    run_command(
        [
            python_executable,
            "scripts/export_onnx.py",
            "--checkpoint",
            str(args.checkpoint),
            "--variant",
            "benchmark",
            "--max-constituents",
            str(args.max_constituents),
        ]
    )
    run_command(
        [
            python_executable,
            "scripts/export_onnx.py",
            "--checkpoint",
            str(args.checkpoint),
            "--variant",
            "sofie",
            "--max-constituents",
            str(args.max_constituents),
        ]
    )
    run_command(
        [
            python_executable,
            "scripts/export_sofie.py",
            "--batch-size",
            str(args.batch_size),
        ]
    )
    run_command(
        [
            python_executable,
            "scripts/benchmark_pt.py",
            "--checkpoint",
            str(args.checkpoint),
            "--device",
            args.device,
            "--output-json",
            str(args.pytorch_json),
            *common_args,
        ]
    )
    run_command(
        [
            python_executable,
            "scripts/benchmark_onnx.py",
            "--onnx",
            "artifacts/exports/simple_part_benchmark.onnx",
            "--providers",
            args.providers,
            "--output-json",
            str(args.onnx_json),
            *common_args,
        ]
    )
    run_command(
        [
            python_executable,
            "scripts/benchmark_sofie.py",
            "--onnx",
            "artifacts/exports/simple_part_sofie.onnx",
            "--output-json",
            str(args.sofie_json),
            *common_args,
        ]
    )
    run_command(
        [
            python_executable,
            "scripts/generate_benchmark_table.py",
            "--pytorch-json",
            str(args.pytorch_json),
            "--onnx-json",
            str(args.onnx_json),
            "--sofie-json",
            str(args.sofie_json),
            "--output-png",
            str(args.output_png),
            "--output-md",
            str(args.output_md),
        ]
    )

    print("[pipeline] Full benchmark pipeline completed.", flush=True)


if __name__ == "__main__":
    main()

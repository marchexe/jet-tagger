# Jet Tagger

Simple particle-transformer style classifier for jet tagging, with PyTorch,
ONNX Runtime, and optional SOFIE benchmarking.

The project uses one canonical PyTorch model (`SimpleParT`) and can export two
different ONNX artifacts from the same checkpoint:

- `simple_part_benchmark.onnx` for runtime benchmarking.
- `simple_part_visual.onnx` for Netron/model visualization.

## Project Layout

```text
core/
  data.py        ROOT reading, dense tensors, masks, normalization
  model.py       canonical PyTorch SimpleParT model
  export.py      ONNX export pipeline
  benchmark.py   shared benchmark framework

scripts/
  export_onnx.py              export benchmark or visual ONNX
  export_sofie.py             generate SOFIE C++ artifacts from ONNX
  benchmark_pt.py             PyTorch benchmark
  benchmark_onnx.py           ONNX Runtime benchmark
  benchmark_sofie.py          optional SOFIE benchmark
  run_full_benchmarks.py      end-to-end benchmark pipeline + system info
  generate_benchmark_table.py benchmark markdown + metric plot
  train_simple_part.py        training script

notebooks/
  benchmark_metrics.ipynb     notebook for plotting benchmark metrics

artifacts/
  checkpoints/  trained checkpoints and normalization
  exports/      ONNX exports
  logs/         benchmark JSON files and plots
```

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Optional notebook/dev extras:

```powershell
pip install -r requirements-dev.txt
```

All commands below assume they are run from the project root.

## Expected Inputs

The default scripts expect:

```text
artifacts/checkpoints/simple_part_best.pt
data/
  train/
  val/
  test/
```

The data loader searches for ROOT files named like:

```text
HToBB_*.root
HToGG_*.root
```

inside the selected split directory. `HToBB` is treated as label `1`, and
`HToGG` as label `0`.

## Export ONNX

There are two ONNX export variants.

### Benchmark ONNX

Use this for ONNX Runtime benchmarking:

```powershell
.\.venv\Scripts\python.exe scripts\export_onnx.py --variant benchmark
```

Default output:

```text
artifacts/exports/simple_part_benchmark.onnx
```

This export is optimized for benchmarking:

- output is `logits`
- input normalization is embedded
- batch axis is dynamic
- intended for `scripts/benchmark_onnx.py`

### Visual ONNX

Use this for Netron/model visualization:

```powershell
.\.venv\Scripts\python.exe scripts\export_onnx.py --variant visual
```

Default output:

```text
artifacts/exports/simple_part_visual.onnx
```

This export is meant to be easier to inspect visually:

- output is `probabilities`
- no embedded normalization by default
- static batch by default
- not intended as the main benchmark artifact

Optional flags:

```powershell
.\.venv\Scripts\python.exe scripts\export_onnx.py --variant visual --embed-normalization
.\.venv\Scripts\python.exe scripts\export_onnx.py --variant visual --dynamic-batch
```

## Run Benchmarks

### PyTorch

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_pt.py
```

Default output:

```text
artifacts/logs/benchmark_pytorch.json
```

Useful quick test:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_pt.py --max-events 2048
```

### ONNX Runtime

Export the benchmark ONNX first:

```powershell
.\.venv\Scripts\python.exe scripts\export_onnx.py --variant benchmark
```

Then run:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_onnx.py
```

Default output:

```text
artifacts/logs/benchmark_onnx.json
```

Useful quick test:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_onnx.py --max-events 2048
```

### SOFIE

SOFIE benchmarking is optional and is intended to run in WSL/Linux with
PyROOT from a ROOT installation in the same environment.

Recommended WSL setup:

```bash
micromamba create -n sofie310 -c conda-forge python=3.10 root numpy psutil awkward uproot onnx matplotlib
micromamba activate sofie310
cd /mnt/c/Users/ReDi_NRW_6489/Documents/jet-tagger
```

Export fresh SOFIE artifacts with the same ROOT version that will run the
benchmark. SOFIE uses the same benchmark ONNX graph as ONNX Runtime; if ROOT
cannot convert it, that is treated as a backend limitation rather than a
different model variant.

```bash
python scripts/export_onnx.py --variant benchmark
python scripts/export_sofie.py
```

Default generated files:

```text
artifacts/sofie/simple_part.hxx
artifacts/sofie/simple_part.dat
```

Then run:

```bash
python scripts/benchmark_sofie.py
```

Useful quick test:

```bash
python scripts/benchmark_sofie.py --max-events 2048 --warmup-runs 1 --measure-runs 1 --batch-size 128
```

The benchmark writes:

```text
artifacts/logs/benchmark_sofie.json
```

## Generate Benchmark Plots

After running benchmarks:

```powershell
.\.venv\Scripts\python.exe scripts\generate_benchmark_table.py
```

Outputs:

```text
artifacts/logs/benchmark_table.md
artifacts/logs/benchmark_metrics.png
```

The PNG is a metric plot, not a table screenshot. It shows:

- accuracy
- loss
- mean latency
- P95 latency
- throughput
- peak RSS

You can also open:

```text
notebooks/benchmark_metrics.ipynb
```

to inspect and regenerate the plot from a notebook. For notebook usage, install
the optional extras from `requirements-dev.txt`.

## Typical Workflow

For a full PyTorch vs ONNX Runtime comparison:

```powershell
.\.venv\Scripts\python.exe scripts\export_onnx.py --variant benchmark
.\.venv\Scripts\python.exe scripts\benchmark_pt.py
.\.venv\Scripts\python.exe scripts\benchmark_onnx.py
.\.venv\Scripts\python.exe scripts\generate_benchmark_table.py
```

For a full PyTorch + ONNX Runtime + SOFIE pipeline with system information:

```powershell
.\.venv\Scripts\python.exe scripts\run_full_benchmarks.py
```

This writes:

```text
artifacts/logs/benchmark_pytorch.json
artifacts/logs/benchmark_onnx.json
artifacts/logs/benchmark_sofie.json
artifacts/logs/benchmark_system.json
artifacts/logs/benchmark_table.md
artifacts/logs/benchmark_metrics.png
```

For Netron visualization:

```powershell
.\.venv\Scripts\python.exe scripts\export_onnx.py --variant visual
```

Then open:

```text
artifacts/exports/simple_part_visual.onnx
```

in Netron.

## Current Benchmark Result Shape

Benchmark JSON files use schema version `2` and contain:

- `metrics`: accuracy, loss, output kind
- `latency`: mean, p50, p95, p99, throughput
- `memory`: RSS and optional CUDA peak memory
- `batching`: requested/min/max batch size
- `artifacts`: checkpoint or ONNX file used
- `extra`: runtime-specific metadata

## Notes

- The PyTorch model is defined in `core/model.py`.
- ONNX exports are artifacts generated from the checkpoint; they are not new
  trained models.
- Benchmark ONNX and visual ONNX intentionally differ because they serve
  different purposes.
- Use `simple_part_benchmark.onnx` for measurement.
- Use `simple_part_visual.onnx` for visualization.

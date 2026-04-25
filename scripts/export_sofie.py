from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SOFIE C++ artifacts from an ONNX model using ROOT."
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/exports/simple_part_benchmark.onnx"),
        help="Input ONNX model to convert with TMVA SOFIE.",
    )
    parser.add_argument(
        "--output-header",
        type=Path,
        default=Path("artifacts/sofie/simple_part.hxx"),
        help="Target .hxx path. The .dat file will be written next to it with the same stem.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose ONNX parsing output from ROOT.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Concrete batch size to specialize dynamic SOFIE code generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import ROOT  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "PyROOT is not installed in the active environment. "
            "Install ROOT first in your WSL/Linux/SWAN environment, for example "
            "`micromamba install -n sofie-env -c conda-forge root`."
        ) from exc

    onnx_path = args.onnx.resolve()
    output_header = args.output_header.resolve()

    if not onnx_path.exists():
        raise SystemExit(
            f"ONNX file not found: {onnx_path}. "
            "Generate it first with `python scripts/export_onnx.py --variant benchmark`."
        )

    if output_header.suffix.lower() != ".hxx":
        raise SystemExit(f"--output-header must point to a .hxx file, got: {output_header}")

    output_header.parent.mkdir(parents=True, exist_ok=True)

    if not hasattr(ROOT.TMVA.Experimental.SOFIE, "RModelParser_ONNX"):
        ROOT.gInterpreter.Declare('#include "TMVA/RModelParser_ONNX.hxx"')

    if not hasattr(ROOT.TMVA.Experimental.SOFIE, "RModelParser_ONNX"):
        raise SystemExit(
            "ROOT is installed, but TMVA SOFIE ONNX parser is not available in this build. "
            "Make sure the active environment provides a ROOT build with SOFIE ONNX parsing support."
        )

    parser = ROOT.TMVA.Experimental.SOFIE.RModelParser_ONNX()
    model = parser.Parse(str(onnx_path), bool(args.verbose))
    generate_options = ROOT.TMVA.Experimental.SOFIE.Options.kDefault
    model.Generate(generate_options, int(args.batch_size), 0, bool(args.verbose))
    model.OutputGenerated(str(output_header))

    output_dat = output_header.with_suffix(".dat")
    print(f"Generated SOFIE header: {output_header}")
    print(f"Generated SOFIE weights: {output_dat}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path


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
    try:
        import onnx
    except ImportError:
        return {}

    if not onnx_path.exists():
        return {}

    model = onnx.load(str(onnx_path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}

from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


def _read_linux_cpu_model() -> str | None:
    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        return None
    for line in cpuinfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lower().startswith("model name"):
            _, value = line.split(":", 1)
            return value.strip()
    return None


def _root_version() -> str | None:
    try:
        import ROOT  # type: ignore[import-not-found]

        return str(ROOT.gROOT.GetVersion())
    except Exception:
        pass

    try:
        completed = subprocess.run(
            ["root-config", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def _optional_package_version(module_name: str, attr_name: str = "__version__") -> str | None:
    try:
        module = __import__(module_name)
        value = getattr(module, attr_name, None)
        return None if value is None else str(value)
    except Exception:
        return None


def collect_system_info(*, run_config: dict[str, Any] | None = None) -> dict[str, Any]:
    virtual_memory = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()

    system_info: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor() or _read_linux_cpu_model(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(),
            "is_wsl": "microsoft" in platform.release().lower() or "WSL_DISTRO_NAME" in os.environ,
        },
        "cpu": {
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "freq_current_mhz": None if cpu_freq is None else cpu_freq.current,
            "freq_max_mhz": None if cpu_freq is None else cpu_freq.max,
        },
        "memory": {
            "total_bytes": int(virtual_memory.total),
            "total_gb": round(float(virtual_memory.total) / 1024**3, 3),
        },
        "software": {
            "torch": _optional_package_version("torch"),
            "onnx": _optional_package_version("onnx"),
            "onnxruntime": _optional_package_version("onnxruntime"),
            "numpy": _optional_package_version("numpy"),
            "uproot": _optional_package_version("uproot"),
            "awkward": _optional_package_version("awkward"),
            "root": _root_version(),
        },
    }

    if run_config:
        system_info["run_config"] = run_config

    return system_info

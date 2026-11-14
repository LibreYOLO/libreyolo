"""Report which export matrix columns can run in the current environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def export_environment() -> dict[str, dict[str, object]]:
    """Return format availability and the missing requirement, if any."""
    onnx = _installed("onnx")
    return {
        "onnx": {"runnable": onnx, "requires": "onnx"},
        "torchscript": {"runnable": _installed("torch"), "requires": "torch"},
        "tensorrt": {
            "runnable": onnx and _installed("tensorrt"),
            "requires": "onnx and a CUDA-matched tensorrt install",
        },
        "openvino": {
            "runnable": onnx and _installed("openvino"),
            "requires": "onnx and openvino",
        },
        "axelera": {
            "runnable": sys.platform.startswith("linux")
            and onnx
            and _installed("axelera")
            and _installed("axelera.compiler"),
            "requires": "Linux, onnx, and Axelera's axelera-devkit (Axelera's package index)",
        },
        "ncnn": {"runnable": _installed("pnnx"), "requires": "pnnx"},
        "tflite": {
            "runnable": sys.version_info >= (3, 12) and onnx and _installed("onnx2tf"),
            "requires": "Python 3.12+, onnx, and onnx2tf 2.4.x",
        },
        "coreml": {
            "runnable": _installed("coremltools"),
            "requires": "coremltools; prediction parity additionally requires macOS",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    report = export_environment()
    if args.json_output:
        print(json.dumps(report, indent=2))
        return
    for fmt, status in report.items():
        state = "ready" if status["runnable"] else "unavailable"
        print(f"{fmt}: {state} ({status['requires']})")


if __name__ == "__main__":
    main()

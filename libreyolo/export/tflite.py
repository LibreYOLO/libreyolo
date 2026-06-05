"""TensorFlow Lite export implementation via onnx2tf."""

from __future__ import annotations

import logging
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_SUPPORTED_EXPORTS = {("yolo9", "detect"): "YOLO9 detect"}
_UNSUPPORTED_FAMILY_REASONS = {
    "rfdetr": (
        "RF-DETR needs separate validation for transformer decoder conversion, "
        "LiteRT operator coverage, and quantization behavior."
    ),
}


def supported_tflite_exports() -> tuple[tuple[str, str], ...]:
    """Return ``(family, task)`` pairs with validated TFLite export support."""
    return tuple(_SUPPORTED_EXPORTS)


def ensure_tflite_family_supported(
    model_family: str | None,
    task: str | None,
) -> None:
    """Raise a targeted error when a family/task has not been validated."""
    family = (model_family or "").lower()
    task = (task or "detect").lower()
    if (family, task) in _SUPPORTED_EXPORTS:
        return

    supported = ", ".join(_SUPPORTED_EXPORTS.values())
    reason = _UNSUPPORTED_FAMILY_REASONS.get(
        family,
        "This family/task has not been validated through the ONNX-to-TFLite path yet.",
    )
    raise NotImplementedError(
        f"TFLite export currently supports: {supported}. "
        f"Got model family {model_family!r}, task {task!r}. {reason}"
    )


def check_tflite_export_available() -> None:
    """Check whether the optional TFLite export toolchain is available."""
    if sys.version_info < (3, 12):
        raise ImportError(
            "TFLite export requires Python 3.12 or newer because onnx2tf 2.4.x "
            "does not publish wheels for older Python versions.\n\n"
            "Install in a Python 3.12 environment with:\n"
            "  pip install libreyolo[tflite]"
        )

    try:
        import onnx2tf  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TFLite export requires the optional onnx2tf converter.\n\n"
            "Install with:\n"
            "  pip install libreyolo[tflite]"
        ) from e


def _onnx2tf_command() -> list[str]:
    exe = shutil.which("onnx2tf")
    if exe:
        return [exe]
    return [sys.executable, "-m", "onnx2tf"]


def _find_converted_tflite(output_dir: Path, onnx_path: Path) -> Path:
    exact = output_dir / f"{onnx_path.stem}_float32.tflite"
    if exact.exists():
        return exact

    fp32_matches = sorted(output_dir.rglob("*_float32.tflite"))
    if fp32_matches:
        return fp32_matches[0]

    matches = sorted(output_dir.rglob("*.tflite"))
    if matches:
        return matches[0]

    produced = sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*"))
    raise RuntimeError(
        "onnx2tf did not produce a TFLite file. "
        f"Files found: {produced[:20]}"
    )


def _write_metadata_sidecar(output_path: Path, metadata: dict) -> None:
    sidecar_path = Path(str(output_path) + ".json")
    with open(sidecar_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata sidecar: %s", sidecar_path)


def export_tflite(
    onnx_path: str,
    output_path: str,
    *,
    half: bool = False,
    verbose: bool = False,
    onnx2tf_args: Iterable[str] | None = None,
    metadata: dict | None = None,
) -> str:
    """Convert a static ONNX model to TensorFlow Lite using onnx2tf."""
    if half:
        raise ValueError(
            "TFLite FP16 export is not supported yet. Omit half=True for FP32."
        )

    check_tflite_export_available()

    onnx_file = Path(onnx_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting ONNX to TFLite: %s", onnx_file)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_output = Path(tmpdir)
        cmd = [
            *_onnx2tf_command(),
            "-i",
            str(onnx_file),
            "-o",
            str(tmp_output),
            "-tb",
            "flatbuffer_direct",
            "-v",
            "info" if verbose else "warn",
        ]
        if onnx2tf_args is not None:
            cmd.extend(str(arg) for arg in onnx2tf_args)

        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
        )
        if result.returncode != 0:
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            raise RuntimeError(
                f"onnx2tf failed with exit code {result.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout: {stdout}\n"
                f"stderr: {stderr}"
            )

        converted = _find_converted_tflite(tmp_output, onnx_file)
        shutil.copy2(converted, dst)

    if metadata is not None:
        _write_metadata_sidecar(dst, metadata)

    logger.info("TFLite export complete: %s", dst)
    return str(dst)


__all__ = [
    "check_tflite_export_available",
    "ensure_tflite_family_supported",
    "export_tflite",
    "supported_tflite_exports",
]

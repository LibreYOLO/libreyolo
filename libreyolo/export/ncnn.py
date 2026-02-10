"""
ncnn export implementation via PNNX.

Converts PyTorch models to ncnn format using PNNX (direct PyTorch-to-ncnn
conversion). Falls back to ONNX-to-PNNX if direct tracing fails.

Supports FP32 and FP16 precision modes.
"""

import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import yaml


def check_ncnn_export_available() -> None:
    """Check if PNNX is available and raise helpful error if not."""
    try:
        import pnnx

        _ = pnnx.__version__
    except ImportError:
        raise ImportError(
            "ncnn export requires the 'pnnx' package.\n\n"
            "Installation:\n"
            "  pip install pnnx\n\n"
            "Or install with the libreyolo extras:\n"
            "  pip install libreyolo[ncnn]"
        )


def export_ncnn(
    nn_model,
    dummy,
    *,
    output_path: str,
    half: bool = False,
    opset: int = 13,
    simplify: bool = True,
    metadata: Optional[dict] = None,
) -> str:
    """
    Export a PyTorch model to ncnn format via PNNX.

    Tries direct PNNX conversion first (PyTorch -> ncnn). If that fails,
    falls back to ONNX -> PNNX conversion.

    The output is a directory containing model.ncnn.param, model.ncnn.bin,
    and optionally metadata.yaml.

    Args:
        nn_model: PyTorch model (nn.Module) in eval mode.
        dummy: Dummy input tensor for tracing.
        output_path: Output directory for ncnn model files.
        half: Export in FP16 precision (default: False).
        opset: ONNX opset version for fallback path (default: 13).
        simplify: Simplify ONNX graph in fallback path (default: True).
        metadata: Optional dict of model metadata to save as metadata.yaml.

    Returns:
        Path to exported ncnn model directory.

    Raises:
        ImportError: If pnnx is not installed.
        RuntimeError: If both direct and fallback export paths fail.
    """
    check_ncnn_export_available()

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        param_path, bin_path = _export_pnnx_direct(nn_model, dummy, output_dir, half)
        print("ncnn export via direct PNNX succeeded")
    except Exception as e:
        print(f"Direct PNNX export failed ({e}), trying ONNX fallback...")
        try:
            param_path, bin_path = _export_onnx_fallback(
                nn_model, dummy, output_dir, half, opset, simplify
            )
            print("ncnn export via ONNX fallback succeeded")
        except Exception as e2:
            raise RuntimeError(
                f"ncnn export failed with both direct and ONNX paths.\n"
                f"Direct error: {e}\n"
                f"Fallback error: {e2}"
            ) from e2

    if metadata:
        _save_metadata(output_dir, metadata)

    print(f"ncnn export complete: {output_dir}")
    return str(output_dir)


def _export_pnnx_direct(nn_model, dummy, output_dir: Path, half: bool):
    """Export using PNNX's direct PyTorch-to-ncnn conversion.

    Args:
        nn_model: PyTorch model in eval mode.
        dummy: Dummy input tensor.
        output_dir: Target directory for ncnn files.
        half: Whether to use FP16 precision.

    Returns:
        Tuple of (param_path, bin_path) in the output directory.
    """
    import pnnx

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # pnnx.export writes files relative to the torchscript path
        temp_pt = tmpdir / "model.pt"
        traced = __import__("torch").jit.trace(nn_model, dummy)
        traced.save(str(temp_pt))

        fp16_flag = 1 if half else 0
        pnnx.export(nn_model, str(temp_pt), dummy, fp16=fp16_flag)

        # PNNX creates files alongside the .pt file
        src_param = tmpdir / "model.ncnn.param"
        src_bin = tmpdir / "model.ncnn.bin"

        if not src_param.exists() or not src_bin.exists():
            # PNNX may use different naming; search for .ncnn.param/.ncnn.bin
            params = list(tmpdir.glob("*.ncnn.param"))
            bins = list(tmpdir.glob("*.ncnn.bin"))
            if not params or not bins:
                raise RuntimeError(
                    f"PNNX did not produce expected .ncnn.param/.ncnn.bin files "
                    f"in {tmpdir}. Files found: {list(tmpdir.iterdir())}"
                )
            src_param = params[0]
            src_bin = bins[0]

        dst_param = output_dir / "model.ncnn.param"
        dst_bin = output_dir / "model.ncnn.bin"
        shutil.copy2(str(src_param), str(dst_param))
        shutil.copy2(str(src_bin), str(dst_bin))

    return dst_param, dst_bin


def _export_onnx_fallback(
    nn_model, dummy, output_dir: Path, half: bool, opset: int, simplify: bool
):
    """Fallback: export to ONNX first, then convert to ncnn with pnnx CLI.

    Args:
        nn_model: PyTorch model in eval mode.
        dummy: Dummy input tensor.
        output_dir: Target directory for ncnn files.
        half: Whether to use FP16 precision.
        opset: ONNX opset version.
        simplify: Whether to simplify the ONNX graph.

    Returns:
        Tuple of (param_path, bin_path) in the output directory.
    """
    from .onnx import export_onnx

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Export to ONNX (always FP32 — PNNX handles precision)
        onnx_path = str(tmpdir / "model.onnx")
        export_onnx(
            nn_model,
            dummy,
            output_path=onnx_path,
            opset=opset,
            simplify=simplify,
            dynamic=False,
            half=False,
            metadata={},
        )

        # Step 2: Call pnnx CLI on the ONNX file
        shape_str = str(list(dummy.shape)).replace(" ", "")
        cmd = ["pnnx", onnx_path, f"inputshape={shape_str}"]
        if half:
            cmd.append("fp16=1")

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(tmpdir)
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pnnx CLI failed (exit code {result.returncode}):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Find the generated ncnn files
        params = list(tmpdir.glob("*.ncnn.param"))
        bins = list(tmpdir.glob("*.ncnn.bin"))
        if not params or not bins:
            raise RuntimeError(
                f"pnnx CLI did not produce .ncnn.param/.ncnn.bin files. "
                f"Files found: {list(tmpdir.iterdir())}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        dst_param = output_dir / "model.ncnn.param"
        dst_bin = output_dir / "model.ncnn.bin"
        shutil.copy2(str(params[0]), str(dst_param))
        shutil.copy2(str(bins[0]), str(dst_bin))

    return dst_param, dst_bin


def _save_metadata(output_dir: Path, metadata: dict) -> None:
    """Save metadata.yaml for ncnn model."""
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    print(f"Saved metadata: {metadata_path}")

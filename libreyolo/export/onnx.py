"""
ONNX export implementation.

Exports PyTorch models to ONNX format with optional graph simplification,
dynamic axes, FP16 precision, and embedded metadata.
"""

import importlib.util
import json
import warnings

import torch


def _get_version() -> str:
    """Return the installed libreyolo version string."""
    try:
        from importlib.metadata import version

        return version("libreyolo")
    except Exception:
        return "0.0.0.dev0"


def export_onnx(
    nn_model,
    dummy,
    *,
    output_path: str,
    opset: int,
    simplify: bool,
    dynamic: bool,
    half: bool,
    metadata: dict,
) -> str:
    """Export a PyTorch model to ONNX format.

    Args:
        nn_model: The PyTorch nn.Module to export.
        dummy: Dummy input tensor for tracing.
        output_path: Destination file path for the .onnx file.
        opset: ONNX opset version.
        simplify: Run onnxsim graph simplification.
        dynamic: Enable dynamic batch axis.
        half: Whether the model/input are FP16.
        metadata: Dict of metadata to embed in the ONNX model
            (keys like model_family, model_size, nb_classes, names, imgsz, etc.).

    Returns:
        The output_path string.
    """
    if importlib.util.find_spec("onnx") is None:
        raise ImportError(
            "ONNX export requires the 'onnx' package. "
            "Install with: uv sync --extra onnx  or  pip install onnx"
        )

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "images": {0: "batch"},
            "output": {0: "batch"},
        }

    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": ["images"],
        "output_names": ["output"],
        "dynamic_axes": dynamic_axes,
    }

    # PyTorch 2.1+ defaults to dynamo-based export which can fail on
    # complex models. Use legacy exporter for better compatibility.
    try:
        torch.onnx.export(nn_model, dummy, output_path, dynamo=False, **export_kwargs)
    except TypeError:
        # Older PyTorch versions don't have dynamo parameter
        torch.onnx.export(nn_model, dummy, output_path, **export_kwargs)

    _postprocess_onnx(output_path, simplify=simplify, dynamic=dynamic, half=half, metadata=metadata)

    return output_path


def _postprocess_onnx(
    path: str,
    *,
    simplify: bool,
    dynamic: bool,
    half: bool,
    metadata: dict,
) -> None:
    """Load the ONNX file, optionally simplify, embed metadata, and save."""
    try:
        import onnx
    except ImportError:
        return

    model_proto = onnx.load(path)

    # Simplify graph
    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify

            simplified, ok = onnx_simplify(model_proto)
            if ok:
                model_proto = simplified
        except ImportError:
            warnings.warn(
                "onnxsim is not installed — skipping ONNX graph simplification. "
                "Install with: pip install onnxsim",
                stacklevel=3,
            )
        except Exception as exc:
            warnings.warn(
                f"ONNX simplification failed (non-fatal): {exc}",
                stacklevel=3,
            )

    # Embed metadata
    for key, value in metadata.items():
        entry = model_proto.metadata_props.add()
        entry.key = key
        entry.value = value

    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, path)

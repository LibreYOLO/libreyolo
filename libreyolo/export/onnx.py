"""ONNX export implementation."""

import importlib.util
import warnings

import torch


def _get_version() -> str:
    """Return the installed libreyolo version string."""
    try:
        from importlib.metadata import version

        return version("libreyolo")
    except Exception:
        return "0.0.0.dev0"


def _uses_dfine_style_export_wrapper(model_family) -> bool:
    """Whether the family uses the ``(pred_logits, pred_boxes)`` export wrapper.

    These DETR-style families wrap the eval-mode model with a tracing-friendly
    module that returns a 2-tuple. ONNX export can skip the dynamic output
    probe for them, and they all need opset 17 for ``aten::scaled_dot_product``.
    """
    return model_family in {"dfine", "deim", "deimv2", "ec", "rfdetr", "rtdetrv4"}


def _set_metadata(model_proto, metadata: dict) -> None:
    """Replace ONNX metadata with the provided key/value pairs."""
    del model_proto.metadata_props[:]
    for key, value in metadata.items():
        entry = model_proto.metadata_props.add()
        entry.key = key
        entry.value = value


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

    _set_metadata(model_proto, metadata)

    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, path)


def _detect_num_outputs(nn_model, dummy):
    """Run a forward pass to detect how many outputs the model produces."""
    with torch.no_grad():
        out = nn_model(dummy)
    if isinstance(out, tuple):
        return len(out)
    return 1


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

    # Detect segmentation: prefer metadata flag from exporter, fall back
    # to output count heuristic for direct export_onnx() calls. For known
    # DETR detection families we already know the output schema, so skip
    # the probe forward pass entirely and reuse the count below.
    is_seg = metadata.get("segmentation") == "true"
    is_yolo9_seg = (
        metadata.get("model_family") == "yolo9"
        and metadata.get("task") == "segment"
    )
    is_obb = metadata.get("task") == "obb"
    known_detr_detection = _uses_dfine_style_export_wrapper(
        metadata.get("model_family")
    )
    num_outputs = None
    if not is_seg and not known_detr_detection:
        num_outputs = _detect_num_outputs(nn_model, dummy)
        is_seg = num_outputs >= 3

    model_family = metadata.get("model_family")
    if is_yolo9_seg:
        output_names = ["predictions", "proto", "mask_coeffs"]
        dynamic_axes = (
            {
                "images": {0: "batch"},
                "predictions": {0: "batch", 2: "anchors"},
                "proto": {0: "batch", 2: "mask_height", 3: "mask_width"},
                "mask_coeffs": {0: "batch", 2: "anchors"},
            }
            if dynamic
            else None
        )
        metadata["segmentation"] = "true"
    elif is_seg and not is_obb:
        output_names = (
            ["dets", "labels", "masks"]
            if model_family == "rfdetr"
            else ["boxes", "scores", "masks"]
        )
        input_name = "input" if model_family == "rfdetr" else "images"
        dynamic_axes = (
            {
                input_name: {0: "batch"},
                output_names[0]: {0: "batch"},
                output_names[1]: {0: "batch"},
                output_names[2]: {0: "batch"},
            }
            if dynamic
            else None
        )
        metadata["segmentation"] = "true"
    elif model_family == "rfdetr" and is_obb:
        input_name = "input"
        output_names = ["dets", "labels", "angles"]
        dynamic_axes = (
            {
                input_name: {0: "batch"},
                "dets": {0: "batch"},
                "labels": {0: "batch"},
                "angles": {0: "batch"},
            }
            if dynamic
            else None
        )
    elif model_family == "rfdetr":
        # RF-DETR's RFDETRExportWrapper returns (boxes, logits), and upstream
        # names those ONNX outputs dets/labels.
        input_name = "input"
        output_names = ["dets", "labels"]
        dynamic_axes = (
            {
                input_name: {0: "batch"},
                "dets": {0: "batch"},
                "labels": {0: "batch"},
            }
            if dynamic
            else None
        )
    elif known_detr_detection or num_outputs == 2:
        # DETR-style detection: (pred_logits, pred_boxes) as a tuple
        output_names = ["pred_logits", "pred_boxes"]
        dynamic_axes = (
            {
                "images": {0: "batch"},
                "pred_logits": {0: "batch"},
                "pred_boxes": {0: "batch"},
            }
            if dynamic
            else None
        )
    else:
        output_names = ["output"]
        dynamic_axes = (
            {"images": {0: "batch"}, "output": {0: "batch"}} if dynamic else None
        )

    input_names = ["input"] if model_family == "rfdetr" else ["images"]
    export_kwargs = {
        "export_params": True,
        "opset_version": opset,
        "do_constant_folding": True,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
    }

    # PyTorch 2.1+ defaults to dynamo-based export which can fail on
    # complex models. Use legacy exporter for better compatibility.
    try:
        torch.onnx.export(nn_model, dummy, output_path, dynamo=False, **export_kwargs)
    except TypeError:
        # Older PyTorch versions don't have dynamo parameter
        torch.onnx.export(nn_model, dummy, output_path, **export_kwargs)

    _postprocess_onnx(
        output_path, simplify=simplify, dynamic=dynamic, half=half, metadata=metadata
    )

    return output_path


def check_onnx_int8_available() -> None:
    """Check ONNX Runtime static quantization dependencies."""
    if importlib.util.find_spec("onnx") is None:
        raise ImportError(
            "ONNX INT8 export requires the 'onnx' package. "
            "Install with: uv sync --extra onnx  or  pip install onnx"
        )
    if importlib.util.find_spec("onnxruntime") is None:
        raise ImportError(
            "ONNX INT8 export requires the 'onnxruntime' package. "
            "Install with: uv sync --extra onnx  or  pip install onnxruntime"
        )
    try:
        from onnxruntime.quantization import quantize_static  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ONNX INT8 export requires ONNX Runtime quantization support. "
            "Install with: uv sync --extra onnx  or  pip install onnxruntime"
        ) from exc


class _CalibrationDataReader:
    """ONNX Runtime CalibrationDataReader backed by LibreYOLO calibration data."""

    def __init__(self, calibration_data, input_name: str):
        self.calibration_data = calibration_data
        self.input_name = input_name
        self._iterator = iter(calibration_data)

    def get_next(self):
        try:
            import numpy as np

            batch = next(self._iterator)
            batch = np.ascontiguousarray(batch, dtype=np.float32)
            return {self.input_name: batch}
        except StopIteration:
            return None

    def rewind(self) -> None:
        self._iterator = iter(self.calibration_data)


def _resolve_calibration_method(name: str):
    from onnxruntime.quantization import CalibrationMethod

    normalized = str(name).lower()
    if normalized == "minmax":
        return CalibrationMethod.MinMax
    if normalized == "entropy":
        return CalibrationMethod.Entropy
    raise ValueError(
        f"Unsupported ONNX INT8 calibration method: {name!r}. "
        "Use 'MinMax' or 'Entropy'."
    )


def _first_input_name(path: str) -> str:
    import onnx

    model_proto = onnx.load(path)
    if not model_proto.graph.input:
        raise ValueError(f"ONNX model has no inputs: {path}")
    return model_proto.graph.input[0].name


def embed_onnx_metadata(path: str, metadata: dict) -> None:
    """Replace metadata_props on an existing ONNX file."""
    import onnx

    model_proto = onnx.load(path)
    _set_metadata(model_proto, metadata)
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, path)


def quantize_onnx_int8(
    fp32_path: str,
    output_path: str,
    *,
    calibration_data,
    metadata: dict,
    preprocessed_path: str,
    calibrate_method: str = "MinMax",
    nodes_to_exclude: list[str] | None = None,
) -> str:
    """Quantize an FP32 ONNX model to QDQ INT8 with float32 inputs/outputs."""
    check_onnx_int8_available()

    from onnxruntime.quantization import QuantFormat, QuantType, quant_pre_process
    from onnxruntime.quantization import quantize_static

    if calibration_data is None:
        raise ValueError(
            "ONNX INT8 quantization requires calibration data. "
            "Pass data='path/to/data.yaml' or omit data to use coco8.yaml."
        )

    quant_pre_process(fp32_path, preprocessed_path)
    reader = _CalibrationDataReader(
        calibration_data,
        input_name=_first_input_name(preprocessed_path),
    )
    quantize_static(
        preprocessed_path,
        output_path,
        reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        calibrate_method=_resolve_calibration_method(calibrate_method),
        op_types_to_quantize=None,
        nodes_to_exclude=nodes_to_exclude,
        extra_options={
            "WeightSymmetric": True,
            "ActivationSymmetric": False,
        },
    )
    embed_onnx_metadata(output_path, metadata)
    return output_path

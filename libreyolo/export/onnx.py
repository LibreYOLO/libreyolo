"""ONNX export implementation."""

import importlib.util
import platform
import re
import warnings
from importlib import metadata as importlib_metadata

import torch

_DETR_TUPLE_OUTPUT_FAMILIES = {
    "deformable_detr",
    "detr",
    "dinodetr",
    "dfine",
    "gtr",
    "deim",
    "deimv2",
    "tinyformer",
    "ec",
    "lwdetr",
    "rfdetr",
    "rtdetr",
    "rtdetrv2",
    "rtdetrv4",
}


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
    return model_family in _DETR_TUPLE_OUTPUT_FAMILIES


def _requires_onnx_opset17(model_family) -> bool:
    """Whether the family needs opset 17 for ONNX auto-opset selection."""
    return model_family in _DETR_TUPLE_OUTPUT_FAMILIES or model_family in {
        "deit",
        "midas",
        "moge2",
        # V-JEPA 2 attention lowers to aten::scaled_dot_product_attention,
        # which ONNX only supports from opset 14.
        "vjepa2",
    }


def _set_metadata(model_proto, metadata: dict) -> None:
    """Replace ONNX metadata with the provided key/value pairs."""
    del model_proto.metadata_props[:]
    for key, value in metadata.items():
        entry = model_proto.metadata_props.add()
        entry.key = key
        entry.value = value


def _package_version(name: str) -> tuple[int, int, int] | None:
    try:
        raw_version = importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None
    match = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", raw_version)
    return tuple(int(part or 0) for part in match.groups()) if match else None


def _should_skip_onnx_simplify() -> bool:
    """Avoid the known onnxsim macOS-arm crash before it can abort Python."""
    if platform.system() != "Darwin" or platform.machine().lower() not in {
        "arm64",
        "aarch64",
    }:
        return False
    onnx_version = _package_version("onnx")
    onnxsim_version = _package_version("onnxsim")
    return (
        onnx_version is not None
        and onnx_version >= (1, 22, 0)
        and onnxsim_version is not None
        and onnxsim_version <= (0, 6, 5)
    )


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

    if simplify and _should_skip_onnx_simplify():
        warnings.warn(
            "Skipping ONNX simplification: onnx>=1.22 with onnxsim<=0.6.5 "
            "can crash Python on macOS arm64.",
            RuntimeWarning,
            stacklevel=3,
        )
        simplify = False

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
    nms: bool = False,
    deepstream: bool = False,
) -> str:
    """Export a PyTorch model to ONNX format.

    Args:
        nn_model: The PyTorch nn.Module to export.
        dummy: Dummy input tensor for tracing.
        output_path: Destination file path for the .onnx file.
        opset: ONNX opset version.
        simplify: Run onnxsim graph simplification.
        dynamic: Enable the format's supported dynamic axes.
        half: Whether the model/input are FP16.
        metadata: Dict of metadata to embed in the ONNX model
            (keys like model_family, model_size, nb_classes, names, imgsz, etc.).
        nms: When True, ``nn_model`` embeds NMS and returns the post-NMS
            ``(batch, max_det, 6)`` detection tensor first, followed by the raw
            detector tensor used by LibreYOLO backends for native postprocess
            parity. Skip the segmentation-probe / family output-schema logic.
        deepstream: When True, use the DeepStream-adapted schema for tasks with
            an nvinfer post-processor. Raw-tensor tasks preserve their regular
            ONNX output names and dynamic axes.

    Returns:
        The output_path string.
    """
    if metadata.get("model_family") == "yolo9" and (
        metadata.get("task") == "segment" or metadata.get("segmentation") == "true"
    ):
        raise NotImplementedError(
            "YOLO9 segmentation ONNX export is not supported. YOLO9 is "
            "detection-only in LibreYOLO."
        )

    if importlib.util.find_spec("onnx") is None:
        raise ImportError(
            "ONNX export requires the 'onnx' package. "
            "Install with: uv sync --extra onnx  or  pip install onnx"
        )

    task = metadata.get("task")
    deepstream_raw_outputs = False
    if deepstream:
        from .deepstream import deepstream_uses_raw_outputs

        deepstream_raw_outputs = deepstream_uses_raw_outputs(task)

    if deepstream and not deepstream_raw_outputs:
        # Parser-backed and native nvinfer tasks use one adapted output.
        # Raw-tensor tasks retain their normal ONNX names and dynamic axes
        # below so applications can decode every tensor from metadata.
        input_name = "input" if metadata.get("model_family") == "rfdetr" else "images"
        return _export_onnx_graph(
            nn_model,
            dummy,
            output_path=output_path,
            opset=opset,
            simplify=simplify,
            half=half,
            metadata=metadata,
            input_names=[input_name],
            output_names=["output"],
            dynamic_axes=(
                {input_name: {0: "batch"}, "output": {0: "batch"}} if dynamic else None
            ),
        )

    if nms:
        # Model embeds NMS: first output is the standalone post-NMS tensor; the
        # raw output lets LibreYOLO preserve native backend postprocess parity.
        return _export_onnx_graph(
            nn_model,
            dummy,
            output_path=output_path,
            opset=opset,
            simplify=simplify,
            half=half,
            metadata=metadata,
            input_names=["images"],
            output_names=["output", "raw"],
            dynamic_axes=(
                {
                    "images": {0: "batch"},
                    "output": {0: "batch"},
                    "raw": {0: "batch", 2: "anchors"},
                }
                if dynamic
                else None
            ),
        )

    # Detect segmentation: prefer metadata flag from exporter, fall back
    # to output count heuristic for direct export_onnx() calls. For known
    # DETR detection families we already know the output schema, so skip
    # the probe forward pass entirely and reuse the count below.
    model_family = metadata.get("model_family")
    is_seg = metadata.get("segmentation") == "true" or task == "segment"
    is_yolo9_pose = model_family == "yolo9" and task == "pose"
    is_hrnet_pose = model_family == "hrnet" and task == "pose"
    is_dekr_pose = model_family == "dekr" and task == "pose"
    is_rfdetr_pose = model_family == "rfdetr" and task == "pose"
    # GTR pose reuses the ECPose decoder and its (logits, keypoints) outputs.
    is_ec_pose = model_family in ("ec", "gtr") and task == "pose"
    is_yolonas_pose = model_family == "yolonas" and task == "pose"
    is_obb = task == "obb"
    is_yolonas_obb = model_family == "yolonas" and is_obb
    is_classify = task == "classify"
    is_semantic = task == "semantic"
    is_restore = task == "restore"
    is_matte = task == "matte"
    is_depth = task == "depth"
    is_normal = task == "normal"
    is_edge = task == "edge"
    is_gaze = task == "gaze"
    is_mask_rcnn = model_family == "mask_rcnn"
    is_faster_rcnn = model_family == "faster_rcnn"
    is_retinanet = model_family == "retinanet"
    is_ssd = model_family == "ssd"
    is_fcos = model_family == "fcos"
    known_detr_detection = _uses_dfine_style_export_wrapper(model_family)
    num_outputs = None
    if (
        not is_seg
        and not known_detr_detection
        and not is_mask_rcnn
        and not is_faster_rcnn
        and not is_retinanet
        and not is_ssd
        and not is_fcos
        and not is_restore
        and not is_matte
        and not is_depth
        and not is_normal
        and not is_edge
        and not is_semantic
        and not is_gaze
        # Supersedes the narrower is_hrnet_pose guard: every pose family has
        # its own output-name branch below, and probing here would misread a
        # multi-tensor pose head (rfdetr-pose 3, yolonas-pose 4) as
        # segmentation via num_outputs >= 3.
        and task != "pose"
    ):
        num_outputs = _detect_num_outputs(nn_model, dummy)
        is_seg = num_outputs >= 3

    if model_family == "yolo9" and is_seg:
        raise NotImplementedError(
            "YOLO9 segmentation ONNX export is not supported. YOLO9 is "
            "detection-only in LibreYOLO."
        )

    if is_mask_rcnn or is_faster_rcnn:
        output_names = ["boxes", "scores", "labels"]
        if is_mask_rcnn and task == "segment":
            output_names.append("masks")
        # Batch stays fixed at one, but the source spatial axes must remain
        # dynamic. GeneralizedRCNNTransform performs the upstream min/max
        # aspect resize in-graph; forcing a square canvas here would require
        # an extra resize/letterbox and break non-square prediction parity.
        dynamic_axes = (
            {
                "images": {2: "height", 3: "width"},
                "boxes": {0: "detections"},
                "scores": {0: "detections"},
                "labels": {0: "detections"},
            }
            if dynamic
            else None
        )
        if dynamic_axes is not None and "masks" in output_names:
            dynamic_axes["masks"] = {
                0: "detections",
                2: "mask_height",
                3: "mask_width",
            }
    elif is_retinanet:
        output_names = ["output"]
        dynamic_axes = (
            {
                "images": {2: "height", 3: "width"},
                "output": {1: "anchors"},
            }
            if dynamic
            else None
        )
    elif is_fcos:
        output_names = ["output"]
        # FCOS preprocessing is outside the graph and preserves aspect ratio,
        # so padded spatial dimensions vary with the source image.
        dynamic_axes = (
            {
                "images": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 1: "anchors"},
            }
            if dynamic
            else None
        )
    elif is_ssd:
        # One fixed-anchor tensor: decoded xyxy boxes followed by contiguous
        # class probabilities, transposed to the standard detector layout.
        output_names = ["output"]
        dynamic_axes = (
            {"images": {0: "batch"}, "output": {0: "batch"}}
            if dynamic
            else None
        )
    elif is_semantic:
        output_names = ["semantic_logits"]
        dynamic_axes = (
            {
                "images": {0: "batch"},
                "semantic_logits": {
                    0: "batch",
                    2: "mask_height",
                    3: "mask_width",
                },
            }
            if dynamic
            else None
        )
    elif is_gaze:
        output_names = ["yaw_logits", "pitch_logits"]
        dynamic_axes = (
            {
                "images": {0: "faces"},
                "yaw_logits": {0: "faces"},
                "pitch_logits": {0: "faces"},
            }
            if dynamic
            else None
        )
    elif is_hrnet_pose:
        output_names = ["heatmaps"]
        dynamic_axes = (
            {"images": {0: "people"}, "heatmaps": {0: "people"}} if dynamic else None
        )
    elif is_dekr_pose:
        # DEKR emits two dense stride-4 maps and nothing else: raw heatmap
        # logits (K + 1 channels, last one the person centre) and per-keypoint
        # offsets (2K channels). Peak finding, pose NMS and the derived-box
        # adapter all stay outside the graph.
        output_names = ["heatmap_logits", "offsets"]
        dynamic_axes = (
            {
                "images": {0: "batch"},
                "heatmap_logits": {0: "batch"},
                "offsets": {0: "batch"},
            }
            if dynamic
            else None
        )
    elif is_classify:
        # Classification emits a single logits tensor (B, num_classes).
        input_name = "input" if model_family == "rfdetr" else "images"
        output_names = ["output"]
        dynamic_axes = (
            {input_name: {0: "batch"}, "output": {0: "batch"}} if dynamic else None
        )
    elif is_restore:
        output_names = ["restored"]
        # Fully convolutional super-resolution families support dynamic spatial
        # dims; other restoration families keep the fixed-resolution contract.
        if dynamic and model_family in {"realesrgan", "quicksrnet"}:
            dynamic_axes = {
                "images": {0: "batch", 2: "height", 3: "width"},
                "restored": {0: "batch", 2: "out_height", 3: "out_width"},
            }
        else:
            dynamic_axes = (
                {"images": {0: "batch"}, "restored": {0: "batch"}} if dynamic else None
            )
    elif is_matte:
        # Single-channel logit map (B, 1, S, S); apply sigmoid downstream.
        output_names = ["matte"]
        dynamic_axes = (
            {"images": {0: "batch"}, "matte": {0: "batch"}} if dynamic else None
        )
    elif is_depth:
        # Dense relative inverse-depth map (B, 1, H, W) at the export canvas;
        # backends resize it back to the original image canvas (ADR 0006).
        output_names = ["depth"]
        dynamic_axes = (
            {"images": {0: "batch"}, "depth": {0: "batch"}} if dynamic else None
        )
    elif is_normal:
        # Dense OpenCV-frame unit normals (B, 3, H, W) at the export canvas;
        # backends resize, renormalize, and return HWC on the original canvas.
        output_names = ["normal"]
        dynamic_axes = (
            {"images": {0: "batch"}, "normal": {0: "batch"}} if dynamic else None
        )
    elif is_edge:
        # Fused edge probability map (B, 1, H, W) at the export canvas.
        output_names = ["edges"]
        dynamic_axes = (
            {"images": {0: "batch"}, "edges": {0: "batch"}} if dynamic else None
        )
    elif is_yolo9_pose:
        output_names = ["predictions", "keypoints"]
        dynamic_axes = (
            {
                "images": {0: "batch"},
                "predictions": {0: "batch", 2: "anchors"},
                "keypoints": {0: "batch", 1: "anchors", 2: "keypoints"},
            }
            if dynamic
            else None
        )
    elif is_ec_pose:
        output_names = ["pred_logits", "pred_keypoints"]
        dynamic_axes = (
            {
                "images": {0: "batch"},
                "pred_logits": {0: "batch", 1: "queries"},
                "pred_keypoints": {0: "batch", 1: "queries", 2: "keypoint_values"},
            }
            if dynamic
            else None
        )
    elif is_yolonas_obb:
        # Raw OBB export contract: boxes [B, N, 5] as cx, cy, w, h, rotation
        # plus sigmoid scores [B, N, C]. Rotated NMS stays out of the graph.
        output_names = ["boxes", "scores"]
        dynamic_axes = (
            {
                "images": {0: "batch"},
                "boxes": {0: "batch", 1: "anchors"},
                "scores": {0: "batch", 1: "anchors"},
            }
            if dynamic
            else None
        )
    elif is_yolonas_pose:
        output_names = [
            "boxes",
            "scores",
            "keypoints_xy",
            "keypoints_conf",
        ]
        dynamic_axes = (
            {
                "images": {0: "batch"},
                "boxes": {0: "batch", 1: "anchors"},
                "scores": {0: "batch", 1: "anchors"},
                "keypoints_xy": {0: "batch", 1: "anchors", 2: "keypoints"},
                "keypoints_conf": {0: "batch", 1: "anchors", 2: "keypoints"},
            }
            if dynamic
            else None
        )
    elif is_seg and not is_obb:
        output_names = (
            ["dets", "labels", "masks"]
            if model_family == "rfdetr"
            else ["pred_logits", "pred_boxes", "pred_masks"]
            if model_family in {"dfine", "ec"}
            else ["boxes", "scores", "masks"]
        )
        input_name = "input" if model_family == "rfdetr" else "images"
        dynamic_axes = (
            {
                input_name: {0: "batch"},
                output_names[0]: {0: "batch", 1: "queries"},
                output_names[1]: {0: "batch", 1: "queries"},
                output_names[2]: {
                    0: "batch",
                    1: "queries",
                    2: "mask_height",
                    3: "mask_width",
                },
            }
            if dynamic
            else None
        )
        metadata["segmentation"] = "true"
    elif is_rfdetr_pose:
        input_name = "input"
        output_names = ["dets", "labels", "keypoints"]
        dynamic_axes = (
            {
                input_name: {0: "batch"},
                "dets": {0: "batch"},
                "labels": {0: "batch"},
                "keypoints": {0: "batch"},
            }
            if dynamic
            else None
        )
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
    elif model_family == "ppyoloe":
        # Raw export contract: boxes [B, A, 4] as xyxy in input-canvas pixels
        # plus sigmoid scores [B, A, C]. NMS stays outside the graph and there
        # is no objectness column for a backend to multiply in twice.
        output_names = ["boxes", "scores"]
        dynamic_axes = (
            {
                "images": {0: "batch"},
                "boxes": {0: "batch", 1: "anchors"},
                "scores": {0: "batch", 1: "anchors"},
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
    return _export_onnx_graph(
        nn_model,
        dummy,
        output_path=output_path,
        opset=opset,
        simplify=simplify,
        half=half,
        metadata=metadata,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def _export_onnx_graph(
    nn_model,
    dummy,
    *,
    output_path: str,
    opset: int,
    simplify: bool,
    half: bool,
    metadata: dict,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict | None,
) -> str:
    """Run ``torch.onnx.export`` with the given IO names, then post-process."""
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
        output_path,
        simplify=simplify,
        dynamic=dynamic_axes is not None,
        half=half,
        metadata=metadata,
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


# Quantize only the heavy linear ops. Leaving the detection-head decode
# (sigmoid, box-distance math, the box+score concat) in float32 is deliberate:
# that concat mixes pixel-scale box coordinates (~0..imgsz) with [0, 1] class
# scores, so a single per-tensor activation scale — dominated by the box
# magnitude — would quantize every score to zero. Restricting quantization to
# Conv/Gemm keeps the size/speed win on the backbone while preserving scores.
_INT8_OP_TYPES = ["Conv", "Gemm"]


def quantize_onnx_int8(
    fp32_path: str,
    output_path: str,
    *,
    calibration_data,
    metadata: dict,
    preprocessed_path: str,
    calibrate_method: str = "MinMax",
    nodes_to_exclude: list[str] | None = None,
    op_types_to_quantize: list[str] | None = None,
    skip_symbolic_shape: bool = False,
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

    quant_pre_process(
        fp32_path,
        preprocessed_path,
        skip_symbolic_shape=skip_symbolic_shape,
    )
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
        op_types_to_quantize=op_types_to_quantize or _INT8_OP_TYPES,
        nodes_to_exclude=nodes_to_exclude,
        extra_options={
            "WeightSymmetric": True,
            "ActivationSymmetric": False,
        },
    )
    embed_onnx_metadata(output_path, metadata)
    return output_path

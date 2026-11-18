"""Unified model export with multiple backend support.

BaseExporter ABC with one subclass per format. Each subclass only
implements ``_export()``, while the template method in ``__call__`` handles
validation, model setup/teardown, calibration, and intermediate ONNX export.
"""

import copy
import importlib.util
import json
import logging
import math
import tempfile
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Optional, Tuple, Union

import torch

from ..tasks import task_to_suffix
from ..utils.image_size import round_imgsz_to_stride
from ..utils.serialization import SCHEMA_VERSION
from .onnx import (
    _get_version,
    _requires_onnx_opset17,
    check_onnx_int8_available,
    export_onnx,
    quantize_onnx_int8,
)
from .support import (
    BLOCKED_SIZES,
    UNMEASURED_SIZES,
    INT8_EXPORTS,
    SIZE_ACCURACY_NOTES,
    get_support,
    validated_alternatives,
)
from .torchscript import export_torchscript

logger = logging.getLogger(__name__)

DEFAULT_INT8_CALIBRATION_DATA = "coco8.yaml"


# Precision helpers


def _resolve_precision(half: bool, int8: bool) -> str:
    if int8:
        return "int8"
    if half:
        return "fp16"
    return "fp32"


def _precision_label(precision: str) -> str:
    return precision.upper()


def _is_rectangular_imgsz(imgsz: tuple[int, int]) -> bool:
    return int(imgsz[0]) != int(imgsz[1])


def _snapshot_rfdetr_export_state(root):
    """Capture RF-DETR export mutations so the live model can be restored."""
    snapshots = []
    if root is None or not hasattr(root, "modules"):
        return snapshots

    for module in root.modules():
        encoder = getattr(module, "encoder", None)
        embeddings = getattr(encoder, "embeddings", None)
        has_export_state = hasattr(module, "_export")
        has_position_state = embeddings is not None and hasattr(
            embeddings, "position_embeddings"
        )
        if not has_export_state and not has_position_state:
            continue

        state = {
            "forward": getattr(module, "forward", None),
            "had_forward_origin": hasattr(module, "_forward_origin"),
            "forward_origin": getattr(module, "_forward_origin", None),
        }
        if has_export_state:
            state["export"] = getattr(module, "_export")
        if hasattr(module, "shape"):
            state["shape"] = getattr(module, "shape")
        if has_position_state:
            state["position_embeddings"] = embeddings.position_embeddings
            state["interpolate_pos_encoding"] = embeddings.interpolate_pos_encoding
        snapshots.append((module, state))
    return snapshots


def _restore_rfdetr_export_state(snapshots):
    for module, state in reversed(snapshots):
        encoder = getattr(module, "encoder", None)
        embeddings = getattr(encoder, "embeddings", None)
        if embeddings is not None and "position_embeddings" in state:
            embeddings.position_embeddings = state["position_embeddings"]
            embeddings.interpolate_pos_encoding = state["interpolate_pos_encoding"]
        if "shape" in state:
            module.shape = state["shape"]
        if state.get("forward") is not None:
            module.forward = state["forward"]
        if state.get("had_forward_origin"):
            module._forward_origin = state["forward_origin"]
        elif hasattr(module, "_forward_origin"):
            delattr(module, "_forward_origin")
        if "export" in state:
            module._export = state["export"]


def _snapshot_tensor_slots(root):
    """Record every parameter and buffer slot of ``root`` with its tensor.

    ``Module.half()``/``.to()`` rebind parameter data and replace buffer
    tensors rather than writing into them, so keeping the original tensors
    is enough to restore exact values, dtype and device afterwards.
    """
    slots = []
    for module in root.modules():
        for name, param in module._parameters.items():
            if param is not None:
                slots.append((module._parameters, name, param, param.data))
        for name, buf in module._buffers.items():
            if buf is not None:
                slots.append((module._buffers, name, buf, None))
    return slots


def _restore_tensor_slots(slots):
    for store, name, tensor, data in slots:
        store[name] = tensor
        if data is not None:
            tensor.data = data


def _classify_eval_metadata(model) -> dict:
    """The model's classification eval pipeline as flat export metadata (#886).

    ``crop_pct`` / ``interpolation`` as before, plus ``norm_mean`` /
    ``norm_std`` (JSON-encoded RGB lists) and ``resize_mode`` for families that
    declare them. Exported backends rebuild the native eval transform from
    these, for both ``predict()`` and ``val()``.
    """
    meta = {}
    crop_pct = getattr(model, "crop_pct", None)
    interpolation = getattr(model, "interpolation", None)
    if crop_pct is not None:
        meta["crop_pct"] = float(crop_pct)
    if interpolation is not None:
        meta["interpolation"] = str(interpolation)
    for key in ("norm_mean", "norm_std"):
        value = getattr(model, key, None)
        if value is not None:
            meta[key] = json.dumps([float(v) for v in value])
    resize_mode = getattr(model, "resize_mode", None)
    if resize_mode is not None:
        meta["resize_mode"] = str(resize_mode)
    return meta


def _letterbox_pad_metadata(model) -> dict:
    """YOLO9-family letterbox placement, so exported runtimes pad like the .pt."""
    pad = getattr(model, "letterbox_pad", None)
    if pad is None:
        return {}
    from ..preprocess.letterbox import normalize_letterbox_pad

    return {"letterbox_pad": normalize_letterbox_pad(pad)}


def _pose_keypoint_shape_metadata(model) -> dict:
    num_keypoints = getattr(
        model, "num_keypoints", getattr(model, "POSE_NUM_KEYPOINTS", "")
    )
    keypoint_dim = getattr(model, "keypoint_dim", getattr(model, "KEYPOINT_DIM", ""))

    schema = getattr(model, "num_keypoints_per_class", None)
    inner = getattr(model, "model", None)
    if not schema and inner is not None:
        schema = getattr(inner, "num_keypoints_per_class", None)
    inner_model = getattr(inner, "model", None) if inner is not None else None
    if (
        not schema
        and inner_model is not None
        and hasattr(inner_model, "get_num_keypoints_per_class")
    ):
        schema = inner_model.get_num_keypoints_per_class()

    model_family = model._get_model_name() if hasattr(model, "_get_model_name") else ""
    if model_family in ("ec", "gtr"):
        # EC/GTR pose export raw xy-only tensors; visibility is appended by runtime
        # postprocessing after decoding.
        keypoint_dim = 2
    elif model_family == "rfdetr" and schema:
        # GroupPose RF-DETR exports the raw padded per-class tensor, whose
        # keypoint payload is (x, y, findable, visible, log_l11, l21, log_l22,
        # class_logit_boost).
        keypoint_dim = 8

    meta = {"num_keypoints": num_keypoints, "keypoint_dim": keypoint_dim}
    if schema:
        meta["num_keypoints_per_class"] = [int(count) for count in schema]
    return meta


_DOMEDETR_EXPORT_MESSAGE = (
    "Dome-DETR export is not supported. PAQI decides the query count per image "
    "(density-filtered proposals plus a greedy density-adaptive NMS), so the "
    "decoder output length is data dependent. Tracing bakes in whichever count "
    "the tracing image happened to produce, giving a graph that silently "
    "returns wrong results for every other image, and a static formulation "
    "needs the greedy suppression unrolled over all 250-1500 candidates. "
    "Reducing to a fixed top-k would remove exactly the tiny-object recall "
    "this family exists for. Use LibreDFINE if you need an exportable DETR."
)


_FIXED_SQUARE_EXPORT_FAMILIES = {
    "clip",
    "deformable_detr",
    "dinodetr",
    "detr",
    "dfine",
    "gtr",
    "deim",
    "deimv2",
    "tinyformer",
    "ec",
    "lwdetr",
    "moge2",
    "moge3",
    "ppdoclayoutv3",
    "riodetrv2",
    "rtdetr",
    "rtdetrv2",
    "rtdetrv4",
    "rfdetr",
    "siglip2",
    "ssd",
}
_RECTANGULAR_EXPORT_FAMILIES = {
    "hrnet",
    "yolo9",
    "yolo9_e2e",
    "yolo9_p2",
    "nafnet",
    # Every PP-LiteSeg size is natively rectangular (512x1024 / 768x1536);
    # a square export would not be the model the checkpoint was trained as.
    "ppliteseg",
    "unet",
    "realesrgan",
    "quicksrnet",
}
# (family, task) pairs that export a rectangular canvas although the family's
# other tasks are fixed-square. GTR semantic slides square windows over it.
_RECTANGULAR_EXPORT_TASKS = {("gtr", "semantic")}
_RECTANGULAR_EXPORT_FORMATS = {
    "coreai",
    "coreml",
    "ncnn",
    "onnx",
    "openvino",
    "tensorrt",
    "tflite",
    "torchscript",
}


class _RTDETRExportWrapper(torch.nn.Module):
    """Trace RT-DETR dict outputs as the two tensors used by exported backends."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs["pred_logits"], outputs["pred_boxes"]


class _SemanticExportWrapper(torch.nn.Module):
    """Expose only dense semantic logits from task-specific native outputs."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            if "semantic_logits" in output:
                return output["semantic_logits"]
            if "logits" in output:
                return output["logits"]
            if "predictions" in output:
                return output["predictions"]
            if "out" in output:
                return output["out"]
        if isinstance(output, (list, tuple)):
            return output[-1]
        return output


class _ImageEmbeddingExportWrapper(torch.nn.Module):
    """Trace an image tower as a normalized whole-image embedding graph."""

    def __init__(self, image_tower: torch.nn.Module):
        super().__init__()
        self.image_tower = image_tower

    def forward(self, x):
        return torch.nn.functional.normalize(self.image_tower(x).float(), dim=-1)


class _VideoEmbeddingExportWrapper(torch.nn.Module):
    """Trace a V-JEPA 2 encoder as a fixed-frame clip embedding graph.

    Input is the public 5D video layout ``(B, F, C, H, W)``; output is the
    pooled, L2-normalized ``(B, D)`` row. The raw token grid is deliberately
    not an export target.
    """

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        tokens = self.encoder(x)
        return torch.nn.functional.normalize(tokens.mean(dim=1).float(), dim=-1)


class _CLSVideoEmbeddingExportWrapper(torch.nn.Module):
    """Trace a CLS-based video encoder as a normalized clip embedding graph."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        tokens = self.encoder(x)
        return torch.nn.functional.normalize(tokens[:, 0].float(), dim=-1)


class _YOLONASExportWrapper(torch.nn.Module):
    """Expose decoded YOLO-NAS tensors without training-only auxiliaries."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        # Eager and torch.export capture return ``(decoded, raw)``. ONNX
        # tracing already returns ``decoded`` directly, so accept both forms.
        if (
            isinstance(output, tuple)
            and len(output) == 2
            and isinstance(output[0], tuple)
        ):
            return output[0]
        return output


class _YOLONASRawHeadsWrapper(torch.nn.Module):
    """YOLO-NAS graph cut right after the head's last convolutions.

    Returns the raw outputs of each level's final convolutions, level by level
    (stride 8, 16, 32), in the order of ``output_layout``. The DFL decode,
    anchors, sigmoids and keypoint / rotated-box geometry are left to the host
    (``libreyolo.postprocess.yolonas_raw``), which rebuilds exactly the tensors
    the decoded export emits. Integer accelerators handle the convolutions
    well but quantize pixel-range boxes and post-sigmoid scores poorly.

    ``output_windows`` (``{part: (low, high)}``, default per task from
    ``YOLONAS_RAW_OUTPUT_WINDOWS_BY_TASK``) emits those parts through
    :func:`_relu6_window`, so each gets an int8 range of ``[low, high]``
    instead of its full calibrated range; the host maps them back.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        task: str,
        output_windows: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__()
        from ..postprocess.yolonas_raw import (
            YOLONAS_RAW_LAYOUT_BY_TASK,
            YOLONAS_RAW_OUTPUT_WINDOWS_BY_TASK,
            check_output_windows,
            raw_head_output_names,
        )

        if task not in YOLONAS_RAW_LAYOUT_BY_TASK:
            raise ValueError(
                f"YOLO-NAS raw-head export supports tasks "
                f"{sorted(YOLONAS_RAW_LAYOUT_BY_TASK)}, got {task!r}."
            )
        self.model = model
        self.task = task
        heads = model.heads
        self.output_layout = YOLONAS_RAW_LAYOUT_BY_TASK[task]
        self.strides = tuple(int(stride) for stride in heads.fpn_strides)
        self.reg_max = int(heads.reg_max)
        self.output_names = raw_head_output_names(self.output_layout, self.strides)
        if output_windows is None:
            output_windows = YOLONAS_RAW_OUTPUT_WINDOWS_BY_TASK.get(task, {})
        self.output_windows = check_output_windows(self.output_layout, output_windows)
        if self.output_windows and task != "pose":
            raise ValueError(
                f"YOLO-NAS output windows are implemented for pose heads only, "
                f"not {task!r}."
            )

    def _pose_head_raw(self, head, x):
        # YoloNASPoseDFLHead.forward up to its last convolutions, without the
        # class/visibility split and keypoint reshape done after them.
        x = head.stem(x)
        pose_features = head.pose_stem(x)
        bbox_features = head.bbox_stem(x)
        return (
            self._final_conv("reg", head.reg_pred, head.reg_dropout_rate(head.reg_convs(bbox_features))),
            self._final_conv("cls", head.cls_pred, head.cls_dropout_rate(head.cls_convs(bbox_features))),
            self._final_conv("pose", head.pose_pred, head.reg_dropout_rate(head.pose_convs(pose_features))),
        )

    def _final_conv(self, part: str, conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        window = self.output_windows.get(part)
        if window is None:
            return conv(x)
        return _relu6_window(conv, x, *window)

    def forward(self, x):
        feats = self.model.neck(self.model.backbone(x))
        heads = self.model.heads
        outputs = []
        for index, feat in enumerate(feats[: heads.num_heads]):
            head = getattr(heads, f"head{index + 1}")
            if self.task == "pose":
                outputs.extend(self._pose_head_raw(head, feat))
            else:
                # Detect heads return (reg, cls); rotated heads return
                # (reg, cls, offset, rot): both already stop at the convs.
                outputs.extend(head(feat))
        return tuple(outputs)


#: Pre-activation bound for YOLO9 class towers in raw-head exports.
_YOLO9_CLASS_ACTIVATION_BOUND = 32.0


def _bounded_conv_bn_act(block: torch.nn.Module, x: torch.Tensor, bound: float) -> torch.Tensor:
    """``block(x)`` for a Conv -> BatchNorm -> activation block, with the
    pre-activation clamped to ``[-bound, bound]``.

    BatchNorm and ``1 / bound`` are folded into the convolution and the clamp
    is ``Clip(-1, 1)`` (HardTanh), which integer converters fuse into the
    convolution, so no int8 tensor carries the unbounded range. The scale is
    restored before the activation; SiLU of a bounded input is bounded.
    """
    conv, bn = block.conv, block.bn
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * (scale / bound).reshape(-1, 1, 1, 1)
    bias = (bn.bias - bn.running_mean * scale) / bound
    if conv.bias is not None:
        bias = bias + conv.bias * scale / bound
    z = torch.nn.functional.conv2d(
        x, weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups
    )
    return block.act(z.clamp(-1.0, 1.0) * bound)


class _YOLO9RawHeadsWrapper(torch.nn.Module):
    """YOLO9 graph cut right after each level's last box and class convs.

    Returns ``reg`` (``4 * reg_max`` distance-bin logits) and ``cls`` (class
    logits) per level, stride 8, 16, 32, as layout ``yolo9_raw_heads_v1``.
    The host rebuilds the decoded export's ``(B, 4 + nc, N)`` tensor
    (:func:`libreyolo.postprocess.yolonas_raw.decode_yolo9_raw`).
    ``output_windows`` works as in :class:`_YOLONASRawHeadsWrapper`; the
    default is ``YOLO9_RAW_OUTPUT_WINDOWS``.

    ``towers`` names the head's box and class tower lists: YOLO9-E2E cuts at
    its one-to-one towers (``one_to_one_anchor_convs`` /
    ``one_to_one_class_convs``), the branch its NMS-free inference decodes,
    so the host decode rebuilds the same tensor as its ONNX export.

    ``class_activation_bound`` caps the pre-activation of each class tower's
    second convolution at +-bound (:func:`_bounded_conv_bn_act`). Its SiLU
    output reaches about +540 on YOLO9-t while 99.9% of values stay below
    about 50, so one per-tensor int8 scale leaves the class evidence a few
    steps; the large values only push background logits further down, and
    bounding them at 32 changes no float32 detection (COCO). ``None`` keeps
    the plain tower.
    """

    output_layout = "yolo9_raw_heads_v1"
    #: Anchors sit at cell centres (``k * stride + stride // 2``).
    grid_cell_offset = 0.5

    def __init__(
        self,
        model: torch.nn.Module,
        task: str = "detect",
        output_windows: dict[str, tuple[float, float]] | None = None,
        class_activation_bound: float | None = _YOLO9_CLASS_ACTIVATION_BOUND,
        towers: tuple[str, str] = ("anchor_convs", "class_convs"),
    ):
        super().__init__()
        from ..postprocess.yolonas_raw import (
            YOLO9_RAW_OUTPUT_WINDOWS,
            check_output_windows,
            raw_head_output_names,
        )

        if task != "detect":
            raise ValueError(f"YOLO9 raw-head export supports detect only, got {task!r}.")
        self.model = model
        self.task = task
        head = model.head
        self.strides = tuple(int(stride) for stride in head.strides)
        self.reg_max = int(head.reg_max)
        self.output_names = raw_head_output_names(self.output_layout, self.strides)
        if output_windows is None:
            output_windows = YOLO9_RAW_OUTPUT_WINDOWS
        self.output_windows = check_output_windows(self.output_layout, output_windows)
        if class_activation_bound is not None and not class_activation_bound > 0:
            raise ValueError(
                f"class_activation_bound must be positive, got {class_activation_bound!r}."
            )
        self.class_activation_bound = class_activation_bound
        self.towers = tuple(towers)

    def forward(self, x):
        feats = self.model.neck(*self.model.backbone(x))
        head = self.model.head
        box_towers, class_towers = (getattr(head, name) for name in self.towers)
        outputs = []
        for box_tower, class_tower, feat in zip(box_towers, class_towers, feats):
            outputs.append(self._final_conv("reg", box_tower[-1], box_tower[:-1](feat)))
            hidden = class_tower[0](feat)
            if self.class_activation_bound is None:
                hidden = class_tower[1](hidden)
            else:
                hidden = _bounded_conv_bn_act(class_tower[1], hidden, self.class_activation_bound)
            outputs.append(self._final_conv("cls", class_tower[2], hidden))
        return tuple(outputs)

    def _final_conv(self, part: str, conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        window = self.output_windows.get(part)
        if window is None:
            return conv(x)
        return _relu6_window(conv, x, *window)


def _fused_yolonas_copy(model: torch.nn.Module, *, full_fusion: bool = True):
    """Deep-copied YOLO-NAS network with its QARepVGG branches fused.

    ``full_fusion`` also folds each block's post-BatchNorm into the fused
    convolution. The caller's model is left untouched.
    """
    fused = copy.deepcopy(model).eval()
    fused.fuse_reparam(full_fusion=full_fusion)
    return fused


#: Exponent of the pose-head activation equalization (0: none, 1: full).
_POSE_EQUALIZATION_ALPHA = 0.4


#: (last ConvBNReLU block, following 1x1 conv) pairs equalized in pose heads.
#: The class tower is left alone: equalizing it too measured 0.6973 against
#: 0.7061 keypoint mAP50-95 on the device (YOLO-NAS-pose-s, COCO persons 300).
_POSE_EQUALIZED_PAIRS = (("pose_convs", "pose_pred"),)


def _pose_head_input_ranges(model: torch.nn.Module, samples) -> dict:
    """Per-channel maximum feeding each equalized conv, ``{(head, pair): tensor}``."""
    heads = model.heads
    ranges: dict = {}
    handles = []
    for index in range(int(heads.num_heads)):
        head = getattr(heads, f"head{index + 1}")
        for pair, (tower, _) in enumerate(_POSE_EQUALIZED_PAIRS):

            def record(_module, _inputs, output, key=(index, pair)):
                peak = output.detach().amax(dim=(0, 2, 3))
                ranges[key] = peak if key not in ranges else torch.maximum(ranges[key], peak)

            handles.append(getattr(head, tower)[-1].register_forward_hook(record))
    wrapper = _YOLONASRawHeadsWrapper(model, "pose", output_windows={}).eval()
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for sample in samples:
                wrapper(torch.as_tensor(sample, dtype=torch.float32, device=device))
    finally:
        for handle in handles:
            handle.remove()
    return ranges


def _equalize_yolonas_pose_heads(
    model: torch.nn.Module, samples=None, alpha: float = _POSE_EQUALIZATION_ALPHA
) -> torch.nn.Module:
    """Even out the channel ranges feeding each pose head's final convs.

    Per-tensor INT8 gives the input of ``pose_pred`` (keypoint offsets) one
    scale, set by the widest channels (about 14x the median on
    YOLO-NAS-pose-s), so the narrow channels keep only a few levels; on a
    Metis device that cost most of the keypoint-mAP loss. Channel ``c`` of
    the last ``pose_convs`` block is scaled by ``1 / s_c`` (its BatchNorm
    scale and shift) and ``pose_pred``'s matching input weights by ``s_c``;
    ReLU is positively homogeneous, so the FP32 outputs are unchanged. ``s_c = (r_c / max r) ** alpha``, with ``r_c`` the
    channel's largest value over ``samples`` (calibration images) or, without
    samples, the estimate ``relu(beta_c + 3 |gamma_c|)`` from that
    BatchNorm. Works in place on ``model`` (an export copy); a model without
    pose heads is returned unchanged.
    """
    heads = model.heads
    if not hasattr(heads, "head1") or not hasattr(heads.head1, "pose_pred"):
        return model
    measured = _pose_head_input_ranges(model, samples) if samples else {}
    with torch.no_grad():
        for index in range(int(heads.num_heads)):
            head = getattr(heads, f"head{index + 1}")
            for pair, (tower, conv_name) in enumerate(_POSE_EQUALIZED_PAIRS):
                bn = getattr(head, tower)[-1].seq.bn
                ranges = measured.get((index, pair))
                if ranges is None:
                    ranges = torch.relu(bn.bias + 3.0 * bn.weight.abs())
                ranges = ranges.to(bn.weight.dtype).clamp_min(1e-3 * float(ranges.max()))
                scale = (ranges / ranges.max()) ** float(alpha)
                bn.weight.div_(scale)
                bn.bias.div_(scale)
                getattr(head, conv_name).weight.mul_(scale[None, :, None, None])
    return model


#: Calibration percentile at which the pose head's final-conv inputs are bounded.
_POSE_BOUND_PERCENTILE = 99.5
#: ConvBNReLU towers whose output gets the bound, with the conv each one feeds.
#: Only the class tower: capping the keypoint tower too cost keypoint mAP50-95
#: on n, m and l (-4.61 vs -1.30, -1.68 vs -0.64, -3.81 vs -0.38 against FP32;
#: its large activations carry keypoint detail) and helped only s.
_POSE_BOUNDED_TOWERS = (("cls_convs", "cls_pred"),)


def _bound_yolonas_pose_inputs(
    model: torch.nn.Module,
    samples,
    percentile: float = _POSE_BOUND_PERCENTILE,
    towers=_POSE_BOUNDED_TOWERS,
) -> dict:
    """Cap the inputs of each pose head's final convs at a calibration percentile.

    The last block of each tower ends in a ReLU whose rare large values set
    the INT8 scale of the next conv's input. Its BatchNorm is scaled by
    ``6 / cap`` and the ReLU becomes a ReLU6, so the block emits
    ``min(x, cap) * 6 / cap``; the next conv's input weights are scaled by
    ``cap / 6``. Values below ``cap`` are unchanged; the integer converter
    fuses the ReLU6 into the conv. Unlike the equalization this changes FP32
    outputs where an activation exceeds ``cap``, so it runs after the
    fusion parity check and its effect is measured with val, not parity.
    Returns ``{"head<i>.<tower>": cap}``.
    """
    heads = model.heads
    values: dict = {}
    handles = []
    for index in range(int(heads.num_heads)):
        head = getattr(heads, f"head{index + 1}")
        for tower, _ in towers:

            def record(_module, _inputs, output, key=(index, tower)):
                values.setdefault(key, []).append(output.detach().flatten())

            handles.append(getattr(head, tower)[-1].register_forward_hook(record))
    wrapper = _YOLONASRawHeadsWrapper(model, "pose", output_windows={}).eval()
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for sample in samples:
                wrapper(torch.as_tensor(sample, dtype=torch.float32, device=device))
    finally:
        for handle in handles:
            handle.remove()
    caps = {}
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for (index, tower), chunks in values.items():
            flat = torch.cat(chunks)
            if flat.numel() > 2_000_000:  # torch.quantile's input limit; a fixed subsample
                flat = flat[torch.randperm(flat.numel(), generator=generator)[:2_000_000]]
            cap = float(torch.quantile(flat.float(), percentile / 100.0))
            if cap <= 0:
                continue
            head = getattr(heads, f"head{index + 1}")
            block = getattr(head, tower)[-1]
            scale = 6.0 / cap
            block.seq.bn.weight.mul_(scale)
            block.seq.bn.bias.mul_(scale)
            block.seq.act = torch.nn.ReLU6()
            getattr(head, dict(towers)[tower]).weight.div_(scale)
            caps[f"head{index + 1}.{tower}"] = cap
    return caps


def _check_yolonas_fusion_parity(
    reference: torch.nn.Module,
    fused: torch.nn.Module,
    task: str,
    imgsz: tuple[int, int],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    seed: int = 0,
) -> float:
    """Assert fused and unfused raw head outputs agree on one random input.

    The tolerance is relative to each output's magnitude:
    ``max|a - b| <= atol + rtol * max|a|``. Returns the worst ratio of error
    to tolerance (below 1 passes).
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = next(reference.parameters()).device
    sample = torch.rand((1, 3, *imgsz), generator=generator).to(device)
    with torch.no_grad():
        expected = _YOLONASRawHeadsWrapper(reference, task).eval()(sample)
        actual = _YOLONASRawHeadsWrapper(fused, task).eval()(sample)
    worst = 0.0
    for index, (a, b) in enumerate(zip(expected, actual)):
        error = float((a - b).abs().max())
        tolerance = atol + rtol * float(a.abs().max())
        worst = max(worst, error / tolerance)
        if error > tolerance:
            raise AssertionError(
                f"YOLO-NAS fusion changed raw output {index}: max abs error "
                f"{error:.3g} exceeds {tolerance:.3g}."
            )
    return worst


# Logits beyond this bound are scores below 0.0004 or above 0.9996.
_INT8_LOGIT_BOUND = 8.0


def _relu6_window(
    conv: torch.nn.Conv2d, x: torch.Tensor, low: float, high: float
) -> torch.Tensor:
    """``conv(x)`` mapped from ``[low, high]`` onto ``[0, 6]`` and clamped.

    The map is folded into the conv's weights and bias, so integer converters
    lower the clamp to a RELU6 fused into the conv and give the output an int8
    range of ``[low, high]`` only.
    """
    k = 6.0 / (high - low)
    z = torch.nn.functional.conv2d(
        x,
        conv.weight * k,
        (conv.bias - low) * k,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
    )
    return z.clamp(0.0, 6.0)


def _bounded_sigmoid(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """``sigmoid(conv(x))`` with the logits bounded inside the conv output.

    Full-integer quantization gives the conv output one int8 scale over its
    calibrated range. Background logits reach about -380 on YOLO9, which
    leaves about 1.5 logit per int8 step and collapses mid-range scores. The
    conv is rescaled so ``[-B, B]`` maps onto ``[0, 6]`` and clamped there; the
    converter lowers that clamp to a RELU6 fused into the conv, so the int8
    range covers only ``[-B, B]``.
    """
    bound = _INT8_LOGIT_BOUND
    k = 6.0 / (2.0 * bound)
    return (_relu6_window(conv, x, -bound, bound) / k - bound).sigmoid()


class _YOLOXSplitOutputWrapper(torch.nn.Module):
    """YOLOX detection graph with boxes and scores as separate outputs.

    Full-integer TFLite gives every tensor one int8 scale. The regular export
    concatenates pixel boxes (0 to imgsz) with sigmoid scores (0 to 1), so a
    single scale near 2.7 rounds every score to zero. Here boxes are divided
    by the input canvas and never share a tensor with scores:
    ``boxes`` is ``(B, 4, N)`` cxcywh over ``(W, H, W, H)`` and ``scores`` is
    ``(B, N, 1 + nc)`` objectness then class probabilities.
    """

    def __init__(self, model: torch.nn.Module, canvas_hw: tuple[int, int]):
        super().__init__()
        self.model = model
        self.canvas_h, self.canvas_w = (int(v) for v in canvas_hw)

    def forward(self, x):
        head = self.model.head
        boxes, scores = [], []
        for k, (cls_conv, reg_conv, stride, feat) in enumerate(
            zip(head.cls_convs, head.reg_convs, head.strides, self.model.backbone(x))
        ):
            feat = head.stems[k](feat)
            cls_prob = _bounded_sigmoid(head.cls_preds[k], cls_conv(feat))
            reg_feat = reg_conv(feat)
            reg = head.reg_preds[k](reg_feat).flatten(2)  # (B, 4, HW)
            obj_prob = _bounded_sigmoid(head.obj_preds[k], reg_feat)
            h, w = feat.shape[-2:]
            yv, xv = torch.meshgrid(
                torch.arange(h, device=x.device),
                torch.arange(w, device=x.device),
                indexing="ij",
            )
            gx = xv.reshape(1, -1).to(reg.dtype)
            gy = yv.reshape(1, -1).to(reg.dtype)
            sx = float(stride) / self.canvas_w
            sy = float(stride) / self.canvas_h
            boxes.append(
                torch.stack(
                    [
                        (reg[:, 0] + gx) * sx,
                        (reg[:, 1] + gy) * sy,
                        torch.exp(reg[:, 2]) * sx,
                        torch.exp(reg[:, 3]) * sy,
                    ],
                    dim=1,
                )
            )
            scores.append(
                torch.cat([obj_prob, cls_prob], 1).flatten(2).permute(0, 2, 1)
            )
        return torch.cat(boxes, 2), torch.cat(scores, 1)


def _has_depthwise_convs(module: torch.nn.Module) -> bool:
    """True when ``module`` contains a depthwise convolution (groups == in_channels > 1)."""
    return any(
        isinstance(m, torch.nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels
        for m in module.modules()
    )


#: How the CSPNeXt pointwise-input equalization is exported: ``"conv"`` (a 1x1
#: depthwise conv, whose input the depthwise rule quantizes per channel),
#: ``"mul"`` (a constant Mul), or ``None`` to skip it.
CSPNEXT_POINTWISE_EQUALIZATION: str | None = "conv"
#: Per-channel factor: ``"act"`` (activation peak / mean peak) or ``"cle"``
#: (sqrt(activation peak / weight peak), geometric-mean normalized, which
#: balances the activation and the pointwise weight columns). ``None`` picks
#: the measured best per task (``CSPNEXT_POINTWISE_EQUALIZATION_METHOD_BY_TASK``).
CSPNEXT_POINTWISE_EQUALIZATION_METHOD: str | None = None
#: RTMDet on a Metis, 1000 COCO images: detect-t ``cle`` -2.29 vs ``act``
#: -2.84 mAP50-95; RTMDet-Ins-s collapses with ``act`` (mask 0.08 vs 0.40 on
#: 300) and holds with ``cle``; RTMDet-Ins-t ties (-1.44 vs -1.40 mask).
CSPNEXT_POINTWISE_EQUALIZATION_METHOD_BY_TASK: dict[str, str] = {
    "detect": "cle",
    "segment": "cle",
}


class _ChannelScale(torch.nn.Module):
    """Multiply each channel by a constant: a 1x1 depthwise conv, or a Mul."""

    def __init__(self, scale: torch.Tensor, as_conv: bool = True):
        super().__init__()
        channels = int(scale.numel())
        if as_conv:
            self.op = torch.nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
            with torch.no_grad():
                self.op.weight.copy_(scale.reshape(channels, 1, 1, 1))
        else:
            self.op = None
            self.register_buffer("scale", scale.reshape(1, channels, 1, 1).clone())

    def forward(self, x):
        return self.op(x) if self.op is not None else x * self.scale


def _equalize_cspnext_pointwise_inputs(
    model: torch.nn.Module, samples: list, mode: str = "conv", method: str = "act"
) -> int:
    """Equalize the per-channel ranges of every CSPNeXt pointwise-conv input, in place.

    The input of a depthwise-separable block's pointwise conv (the SiLU output
    of its depthwise conv) mixes channels of very different size (RTMDet-t
    stage 2: peaks up to 14.2 against a median of 2.7); with one int8 scale for
    the whole tensor that single tensor cost RTMDet-t about 4.4 mAP50-95 on a
    Metis. Channel ``c`` is scaled by ``1 / s_c`` just before the pointwise conv
    and the conv's input weights by ``s_c`` (``s_c`` = the channel's peak over
    ``samples`` divided by the mean peak), which leaves FP32 outputs unchanged
    up to rounding. Returns the number of blocks equalized. ``model`` must be a
    copy the caller owns.
    """
    from ..models.rtmdet.nn import DepthwiseSeparableConv

    blocks = [m for m in model.modules() if isinstance(m, DepthwiseSeparableConv)]
    if not blocks or not samples or mode is None:
        return 0
    peaks: dict[int, torch.Tensor] = {}

    def record(key):
        def hook(_module, inputs):
            peak = inputs[0].detach().abs().amax(dim=(0, 2, 3))
            peaks[key] = peak if key not in peaks else torch.maximum(peaks[key], peak)

        return hook

    handles = [
        block.pointwise_conv.register_forward_pre_hook(record(id(block))) for block in blocks
    ]
    try:
        with torch.no_grad():
            for sample in samples:
                model(torch.as_tensor(sample, dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()
    if method not in ("act", "cle"):
        raise ValueError(f"Unknown equalization method {method!r}; use 'act' or 'cle'.")
    for block in blocks:
        peak = peaks[id(block)].clamp_min(1e-6)
        if method == "cle":
            weight_peak = (
                block.pointwise_conv.conv.weight.detach().abs().amax(dim=(0, 2, 3)).clamp_min(1e-9)
            )
            scale = torch.sqrt(peak / weight_peak)
            scale = scale / torch.exp(torch.log(scale).mean())
        else:
            scale = peak / peak.mean()
        with torch.no_grad():
            block.pointwise_conv.conv.weight.mul_(scale.reshape(1, -1, 1, 1))
        block.pointwise_conv = torch.nn.Sequential(
            _ChannelScale(1.0 / scale, as_conv=(mode == "conv")), block.pointwise_conv
        )
    return len(blocks)


class _YOLOXRawHeadsWrapper(torch.nn.Module):
    """YOLOX graph cut right after each level's last convolutions.

    Returns ``reg`` (x/y cell offsets and log width/height), ``obj`` and
    ``cls`` logits per level, stride 8, 16, 32, as layout
    ``yolox_raw_heads_v1``. The host rebuilds the decoded export's
    ``(B, N, 5 + nc)`` tensor
    (:func:`libreyolo.postprocess.yolonas_raw.decode_yolox_raw`).
    ``output_windows`` works as in :class:`_YOLONASRawHeadsWrapper`; the
    default is ``YOLOX_RAW_OUTPUT_WINDOWS``.
    """

    output_layout = "yolox_raw_heads_v1"
    #: Anchors sit at the top-left corner of each cell (``k * stride``).
    grid_cell_offset = 0.0
    #: YOLOX regresses boxes directly: no distribution bins.
    reg_max = None

    def __init__(
        self,
        model: torch.nn.Module,
        task: str = "detect",
        output_windows: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__()
        from ..postprocess.yolonas_raw import (
            YOLOX_RAW_OUTPUT_WINDOWS,
            check_output_windows,
            raw_head_output_names,
        )

        if task != "detect":
            raise ValueError(f"YOLOX raw-head export supports detect only, got {task!r}.")
        self.model = model
        self.task = task
        self.strides = tuple(int(stride) for stride in model.head.strides)
        self.output_names = raw_head_output_names(self.output_layout, self.strides)
        if output_windows is None:
            output_windows = YOLOX_RAW_OUTPUT_WINDOWS
        self.output_windows = check_output_windows(self.output_layout, output_windows)

    def forward(self, x):
        head = self.model.head
        outputs = []
        for k, feat in enumerate(self.model.backbone(x)):
            feat = head.stems[k](feat)
            reg_feat = head.reg_convs[k](feat)
            outputs.append(self._final_conv("reg", head.reg_preds[k], reg_feat))
            outputs.append(self._final_conv("obj", head.obj_preds[k], reg_feat))
            outputs.append(self._final_conv("cls", head.cls_preds[k], head.cls_convs[k](feat)))
        return tuple(outputs)

    def _final_conv(self, part: str, conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        window = self.output_windows.get(part)
        if window is None:
            return conv(x)
        return _relu6_window(conv, x, *window)


def _split_anchor_head(
    weight: torch.Tensor, bias: torch.Tensor, anchors: int, num_classes: int
) -> dict[str, torch.nn.Conv2d]:
    """An ``A * (5 + nc)`` channel head conv as three convs with the same weights:
    ``reg`` (4 per anchor), ``obj`` (1 per anchor), ``cls`` (``nc`` per anchor)."""
    per_anchor = 5 + num_classes
    index = torch.arange(anchors * per_anchor).view(anchors, per_anchor)
    rows = {
        "reg": index[:, :4].reshape(-1),
        "obj": index[:, 4].reshape(-1),
        "cls": index[:, 5:].reshape(-1),
    }
    convs = {}
    for part, selected in rows.items():
        conv = torch.nn.Conv2d(weight.shape[1], len(selected), tuple(weight.shape[2:]))
        with torch.no_grad():
            conv.weight.copy_(weight[selected])
            conv.bias.copy_(bias[selected])
        convs[part] = conv
    return convs


class _PassThrough(torch.nn.Module):
    def forward(self, x):
        return x


class _AnchorYOLORawHeadsWrapper(torch.nn.Module):
    """Anchor-based YOLO graph (Darknet YOLOv3/v4, YOLOv7) cut before the box decode.

    ``model`` returns the inputs of each level's final 1x1 head convolution.
    That convolution is split into ``reg`` (four box terms per anchor),
    ``obj`` (one logit per anchor) and ``cls`` (``nc`` logits per anchor), so
    box terms and logits get separate INT8 scales; the host rebuilds the
    decoded export's ``(B, 4 + nc, N)`` tensor
    (:func:`libreyolo.postprocess.yolonas_raw.decode_anchor_yolo_raw`).
    ``output_windows`` works as in :class:`_YOLONASRawHeadsWrapper`.
    """

    #: Cells are indexed from their top-left corner.
    grid_cell_offset = 0.0
    reg_max = None

    def __init__(
        self,
        model: torch.nn.Module,
        heads,
        *,
        layout: str,
        strides,
        anchors,
        num_classes: int,
        scale_x_y=None,
        output_windows: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__()
        from ..postprocess.yolonas_raw import check_output_windows, raw_head_output_names

        self.model = model
        self.task = "detect"
        self.output_layout = layout
        self.strides = tuple(int(stride) for stride in strides)
        self.anchors = [[[float(w), float(h)] for w, h in level] for level in anchors]
        self.scale_x_y = None if scale_x_y is None else [float(s) for s in scale_x_y]
        self.output_names = raw_head_output_names(layout, self.strides)
        self.output_windows = check_output_windows(layout, output_windows or {})
        self.heads = torch.nn.ModuleList(
            torch.nn.ModuleDict(_split_anchor_head(weight, bias, len(level), num_classes))
            for (weight, bias), level in zip(heads, self.anchors)
        )

    def forward(self, x):
        outputs = []
        for feat, head in zip(self.model(x), self.heads):
            for part in ("reg", "obj", "cls"):
                window = self.output_windows.get(part)
                conv = head[part]
                outputs.append(conv(feat) if window is None else _relu6_window(conv, feat, *window))
        return tuple(outputs)


def _conv_output_rows(conv: torch.nn.Conv2d, rows) -> torch.nn.Conv2d:
    """A copy of ``conv`` that computes only the output channels ``rows``."""
    rows = list(rows)
    part = torch.nn.Conv2d(
        conv.in_channels,
        len(rows),
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    ).to(conv.weight.device)
    with torch.no_grad():
        part.weight.copy_(conv.weight[rows])
        if conv.bias is not None:
            part.bias.copy_(conv.bias[rows])
    return part


class _PicoDetRawHeadsWrapper(torch.nn.Module):
    """PicoDet graph cut before its box decode (layout ``picodet_raw_heads_v1``).

    When the head shares one final convolution between the class logits and
    the box distribution (``share_cls_reg``), it is split into a ``reg`` and a
    ``cls`` convolution with the same weights, so the two get separate INT8
    scales. The host rebuilds the decoded export's ``(B, N, 4 + nc)`` tensor
    (:func:`libreyolo.postprocess.yolonas_raw.decode_picodet_raw`).
    """

    output_layout = "picodet_raw_heads_v1"
    grid_cell_offset = 0.5

    def __init__(self, model: torch.nn.Module, output_windows=None):
        super().__init__()
        from ..postprocess.yolonas_raw import (
            PICODET_RAW_OUTPUT_WINDOWS,
            check_output_windows,
            raw_head_output_names,
        )

        self.output_windows = check_output_windows(
            self.output_layout,
            PICODET_RAW_OUTPUT_WINDOWS if output_windows is None else output_windows,
        )

        model = copy.deepcopy(model).eval()
        head = model.head
        self.backbone = model.backbone
        self.neck = model.neck
        self.cls_convs = head.cls_convs
        self.reg_convs = None if head.share_cls_reg else head.reg_convs
        self.strides = tuple(int(stride) for stride in head.strides)
        self.reg_max = int(head.reg_max)
        self.output_names = raw_head_output_names(self.output_layout, self.strides)
        classes, bins = int(head.num_classes), 4 * (self.reg_max + 1)
        self.reg_heads = torch.nn.ModuleList()
        self.cls_heads = torch.nn.ModuleList()
        for level in range(len(self.strides)):
            if head.share_cls_reg:  # PicoHead splits [classes, distribution]
                shared = head.gfl_cls[level]
                self.cls_heads.append(_conv_output_rows(shared, range(classes)))
                self.reg_heads.append(
                    _conv_output_rows(shared, range(classes, classes + bins))
                )
            else:
                self.cls_heads.append(head.gfl_cls[level])
                self.reg_heads.append(head.gfl_reg[level])

    def forward(self, x):
        outputs = []
        for level, feat in enumerate(self.neck(self.backbone(x))):
            cls_feat = feat
            for conv in self.cls_convs[level]:
                cls_feat = conv(cls_feat)
            reg_feat = cls_feat
            if self.reg_convs is not None:
                reg_feat = feat
                for conv in self.reg_convs[level]:
                    reg_feat = conv(reg_feat)
            for part, conv, part_feat in (
                ("reg", self.reg_heads[level], reg_feat),
                ("cls", self.cls_heads[level], cls_feat),
            ):
                window = self.output_windows.get(part)
                outputs.append(
                    conv(part_feat)
                    if window is None
                    else _relu6_window(conv, part_feat, *window)
                )
        return tuple(outputs)


def _fuse_darknet_batchnorms(net: torch.nn.Module) -> int:
    """Fold each Darknet convolution's batch norm into its weights and bias, in place.

    Darknet's norm is ``(x - mean) / (sqrt(var) + eps) * weight + bias``
    (:class:`~libreyolo.models.darknet.blocks.DarknetBatchNorm2d`), so with
    ``scale = weight / (sqrt(var) + eps)`` the fused convolution has weights
    ``W * scale`` and bias ``(b - mean) * scale + bias``: the same outputs up to
    float rounding. Unfused, every convolution exports as Conv -> Sub -> Div ->
    Mul -> Add, and the Axelera compiler stops on YOLOv3 in its in-core
    replication pass ("more than one ODR op in outlined composite"). Returns
    the number of fused layers.
    """
    from libreyolo.models.darknet.blocks import DarknetConv

    fused_count = 0
    for module in net.modules():
        if not isinstance(module, DarknetConv) or module.bn is None:
            continue
        conv, norm = module.conv, module.bn
        scale = norm.weight / (torch.sqrt(norm.running_var) + norm.eps)
        fused = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        ).to(conv.weight.device)
        with torch.no_grad():
            fused.weight.copy_(conv.weight * scale.view(-1, 1, 1, 1))
            base = conv.bias if conv.bias is not None else torch.zeros_like(scale)
            fused.bias.copy_((base - norm.running_mean) * scale + norm.bias)
        module.conv = fused
        module.bn = None
        fused_count += 1
    return fused_count


class _ReorgConv(torch.nn.Module):
    """Darknet's ``[reorg]`` as one convolution with one-hot weights.

    The layer is local (each output pixel reads only its own ``stride x stride``
    input block) but Darknet interleaves the channels with three reshapes and
    transposes, which an accelerator would have to run as layout transforms.
    A ``stride x stride`` convolution with stride ``stride`` and 0/1 weights,
    read off the reference layer on a basis input, computes the same tensor
    exactly (and its weights are exact in INT8).
    """

    def __init__(self, reorg: torch.nn.Module, channels: int):
        super().__init__()
        s = int(reorg.stride)
        cells = channels * s * s
        basis = torch.eye(cells).reshape(cells, channels, s, s)
        with torch.no_grad():
            picks = reorg(basis).reshape(cells, cells)  # picks[j, o] = 1 if output o reads input cell j
        self.conv = torch.nn.Conv2d(channels, cells, s, stride=s, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(picks.t().reshape(cells, channels, s, s))
        self.stride = s

    def forward(self, x):
        return self.conv(x)


def _reorg_layers_to_conv(net: torch.nn.Module) -> int:
    """Replace every Darknet ``[reorg]`` layer of ``net`` by a :class:`_ReorgConv`, in place."""
    from libreyolo.models.darknet.blocks import Reorg

    targets = [i for i, layer in enumerate(net.layers) if isinstance(layer, Reorg)]
    if not targets:
        return 0
    channels: dict[int, int] = {}
    hooks = [
        net.layers[i].register_forward_pre_hook(
            lambda _module, inputs, i=i: channels.__setitem__(i, int(inputs[0].shape[1]))
        )
        for i in targets
    ]
    try:
        side = 32 * max(int(net.layers[i].stride) for i in targets) * 2
        with torch.no_grad():
            net(torch.zeros(1, 3, side, side))
    finally:
        for hook in hooks:
            hook.remove()
    for i in targets:
        net.layers[i] = _ReorgConv(net.layers[i], channels[i])
    return len(targets)


class _LocalAsConvs(torch.nn.Module):
    """Darknet's ``[local]`` layer (YOLOv1) as one convolution per output position.

    ``[local]`` gives every output position its own filter bank and exports as
    an Einsum, which the Axelera quantizer does not support. Position ``(i, j)``
    is a ``size x size`` convolution (no padding) on its own patch of the
    padded input, with weights ``weight[i * out_w + j]`` viewed as
    ``(out, C, k, k)`` (``F.unfold``'s channel-major, kernel-row, kernel-col
    order, as in Darknet's im2col) and bias ``bias[:, i * out_w + j]``. The
    1x1 results are concatenated along W, then H: the same tensor as the
    reference layer up to float summation order.
    """

    def __init__(self, local: torch.nn.Module):
        super().__init__()
        k = int(local.size)
        self.size, self.stride, self.pad = k, int(local.stride), int(local.pad)
        self.out_h, self.out_w = int(local.out_h), int(local.out_w)
        weight = local.weight.detach().reshape(
            self.out_h * self.out_w, local.out_channels, local.in_channels, k, k
        )
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        self.bias = torch.nn.Parameter(local.bias.detach().t().clone(), requires_grad=False)
        self.act = local.act

    def forward(self, x):
        k, s = self.size, self.stride
        x = torch.nn.functional.pad(x, (self.pad,) * 4)
        rows = []
        for i in range(self.out_h):
            cols = []
            for j in range(self.out_w):
                loc = i * self.out_w + j
                patch = x[:, :, i * s : i * s + k, j * s : j * s + k]
                cols.append(torch.nn.functional.conv2d(patch, self.weight[loc], self.bias[loc]))
            rows.append(torch.cat(cols, dim=3))
        return self.act(torch.cat(rows, dim=2))


def _local_layers_to_convs(net: torch.nn.Module) -> int:
    """Replace every Darknet ``[local]`` layer of ``net`` by :class:`_LocalAsConvs`, in place."""
    from libreyolo.models.darknet.blocks import DarknetLocal

    count = 0
    for i, layer in enumerate(net.layers):
        if isinstance(layer, DarknetLocal):
            net.layers[i] = _LocalAsConvs(layer)
            count += 1
    return count


def _darknet_raw_heads_wrapper(
    holder: torch.nn.Module, task: str = "detect", output_windows=None
):
    """Raw-head wrapper for a Darknet ``[yolo]`` (LibreYOLO3, LibreYOLO4) or ``[region]`` (LibreYOLO2) detector.

    ``[region]`` heads (YOLOv2) use layout ``darknet_region_raw_heads_v1``: the
    same head split, anchors converted from grid cells to pixels, and a softmax
    over the classes on the host. ``[reorg]`` passthrough layers become one-hot
    convolutions (:class:`_ReorgConv`).
    """
    if task != "detect":
        raise ValueError(f"Darknet raw-head export supports detect only, got {task!r}.")
    net = copy.deepcopy(holder.net).eval()
    _fuse_darknet_batchnorms(net)
    _reorg_layers_to_conv(net)
    specs = net.detection_specs()
    kinds = {spec.kind for spec in specs}
    if kinds not in ({"yolo"}, {"region"}):
        raise ValueError("Raw-head export supports Darknet [yolo] or [region] heads only.")
    region = kinds == {"region"}
    heads = []
    for spec in specs:
        head = net.layers[spec.layer_index - 1]
        if head.bn is not None or head.conv.bias is None:
            raise ValueError("Darknet detection heads end in a linear conv with bias.")
        heads.append((head.conv.weight.detach(), head.conv.bias.detach()))
        net.layers[spec.layer_index - 1] = _PassThrough()
    anchors = [
        [(w * spec.stride, h * spec.stride) for w, h in spec.anchors] if region else spec.anchors
        for spec in specs
    ]
    return _AnchorYOLORawHeadsWrapper(
        net,
        heads,
        layout="darknet_region_raw_heads_v1" if region else "darknet_raw_heads_v1",
        strides=[spec.stride for spec in specs],
        anchors=anchors,
        num_classes=specs[0].num_classes,
        scale_x_y=[spec.scale_x_y for spec in specs],
        output_windows=output_windows,
    )


class _InputsOf(torch.nn.Module):
    def forward(self, xs):
        return xs


class _FusedRepConv(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv2d, act: torch.nn.Module):
        super().__init__()
        self.conv = conv
        self.act = act

    def forward(self, x):
        return self.act(self.conv(x))


def _fuse_yolo7_repconvs(net: torch.nn.Module) -> int:
    """Replace each YOLOv7 ``RepConv`` (3x3 and 1x1 Conv-BN branches, summed)
    by one 3x3 convolution with bias, in place; exact in FP32. Returns the count."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    from ..models.yolo7.blocks import RepConv

    targets = [(name, m) for name, m in net.named_modules() if isinstance(m, RepConv)]
    for name, rep in targets:
        k3 = fuse_conv_bn_eval(rep.conv1.conv, rep.conv1.bn)
        k1 = fuse_conv_bn_eval(rep.conv2.conv, rep.conv2.bn)
        if k3.stride != k1.stride or k3.groups != 1 or k1.groups != 1:
            raise ValueError(f"Cannot fuse RepConv {name!r}: branch strides or groups differ.")
        pad = (k3.kernel_size[0] - 1) // 2
        fused = torch.nn.Conv2d(
            k3.in_channels, k3.out_channels, k3.kernel_size, k3.stride, k3.padding, bias=True
        )
        with torch.no_grad():
            fused.weight.copy_(k3.weight + torch.nn.functional.pad(k1.weight, [pad] * 4))
            fused.bias.copy_(k3.bias + k1.bias)
        parent_name, _, child = name.rpartition(".")
        parent = net.get_submodule(parent_name) if parent_name else net
        setattr(parent, child, _FusedRepConv(fused, rep.act))
    return len(targets)


def _yolo7_raw_heads_wrapper(net: torch.nn.Module, task: str = "detect"):
    """Raw-head wrapper for LibreYOLO7: RepConv branches fused, and the
    implicit add / multiply of each ``IDetection`` head folded exactly into its
    split 1x1 convolutions."""
    if task != "detect":
        raise ValueError(f"YOLOv7 raw-head export supports detect only, got {task!r}.")
    net = copy.deepcopy(net).eval()
    _fuse_yolo7_repconvs(net)
    detection = net.layers[net._head_index]
    heads = []
    for idet in detection.heads:
        weight = idet.head_conv.weight.detach()
        add = idet.implicit_a.implicit.detach().reshape(-1)
        mul = idet.implicit_m.implicit.detach().reshape(-1)
        bias = (idet.head_conv.bias.detach() + weight[:, :, 0, 0] @ add) * mul
        heads.append((weight * mul.view(-1, 1, 1, 1), bias))
    net.layers[net._head_index] = _InputsOf()
    return _AnchorYOLORawHeadsWrapper(
        net,
        heads,
        layout="yolo7_raw_heads_v1",
        strides=net.strides,
        anchors=[list(zip(flat[0::2], flat[1::2])) for flat in net.anchors],
        num_classes=net.num_classes,
    )


class _PPYOLOERawHeadsWrapper(torch.nn.Module):
    """PP-YOLOE graph cut right after each level's class and distribution convs.

    Returns ``reg`` (``4 * (reg_max + 1)`` distribution logits) and ``cls``
    (class logits) per level in the head's stride order (32, 16, 8), as layout
    ``ppyoloe_raw_heads_v1``; the host decodes them like YOLO-NAS heads
    (:func:`libreyolo.postprocess.yolonas_raw.decode_yolonas_raw`). RepVGG
    blocks are fused into single 3x3 convolutions (exact) on a copy.
    ``output_windows`` works as in :class:`_YOLONASRawHeadsWrapper`; the
    default is ``PPYOLOE_RAW_OUTPUT_WINDOWS``.
    """

    output_layout = "ppyoloe_raw_heads_v1"

    def __init__(
        self,
        model: torch.nn.Module,
        task: str = "detect",
        output_windows: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__()
        from ..models.ppyoloe.nn import RepVGGBlock
        from ..postprocess.yolonas_raw import (
            PPYOLOE_RAW_OUTPUT_WINDOWS,
            check_output_windows,
            raw_head_output_names,
        )

        if task != "detect":
            raise ValueError(f"PP-YOLOE raw-head export supports detect only, got {task!r}.")
        model = copy.deepcopy(model).eval()
        for module in model.modules():
            if isinstance(module, RepVGGBlock):
                module.fuse_block_residual_branches()
        self.model = model
        self.task = task
        head = model.head
        self.strides = tuple(int(stride) for stride in head.fpn_strides)
        self.reg_max = int(head.reg_max)
        self.grid_cell_offset = float(head.grid_cell_offset)
        self.output_names = raw_head_output_names(self.output_layout, self.strides)
        if output_windows is None:
            output_windows = PPYOLOE_RAW_OUTPUT_WINDOWS
        self.output_windows = check_output_windows(self.output_layout, output_windows)

    def _final_conv(self, part: str, conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        window = self.output_windows.get(part)
        if window is None:
            return conv(x)
        return _relu6_window(conv, x, *window)

    def forward(self, x):
        model = self.model
        head = model.head
        outputs = []
        for i, feat in enumerate(model.neck(model.backbone(x))):
            avg_feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            outputs.append(self._final_conv("reg", head.pred_reg[i], head.stem_reg[i](feat, avg_feat)))
            outputs.append(
                self._final_conv("cls", head.pred_cls[i], head.stem_cls[i](feat, avg_feat) + feat)
            )
        return tuple(outputs)


class _RTMDetRawHeadsWrapper(torch.nn.Module):
    """RTMDet graph cut right after each level's ``rtm_reg`` and ``rtm_cls`` convs.

    Returns ``reg`` (raw l/t/r/b distance terms, before the stride scale and
    the optional exp) and ``cls`` (class logits) per level, stride 8, 16, 32,
    as layout ``rtmdet_raw_heads_v1``. The host rebuilds the decoded export's
    ``(B, N, 4 + nc)`` tensor
    (:func:`libreyolo.postprocess.yolonas_raw.decode_rtmdet_raw`).
    ``output_windows`` works as in :class:`_YOLONASRawHeadsWrapper`; the
    default is ``RTMDET_RAW_OUTPUT_WINDOWS``.
    """

    output_layout = "rtmdet_raw_heads_v1"
    #: mmdet's MlvlPointGenerator(offset=0): points on cell corners.
    grid_cell_offset = 0.0
    #: RTMDet regresses distances directly: no distribution bins.
    reg_max = None

    def __init__(
        self,
        model: torch.nn.Module,
        task: str = "detect",
        output_windows: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__()
        from ..postprocess.yolonas_raw import (
            RTMDET_RAW_OUTPUT_WINDOWS,
            check_output_windows,
            raw_head_output_names,
        )

        if task != "detect":
            raise ValueError(f"RTMDet raw-head export supports detect only, got {task!r}.")
        self.model = model
        self.task = task
        head = model.head
        self.strides = tuple(int(stride) for stride in head.strides)
        self.exp_on_reg = bool(head.exp_on_reg)
        self.output_names = raw_head_output_names(self.output_layout, self.strides)
        if output_windows is None:
            output_windows = RTMDET_RAW_OUTPUT_WINDOWS
        self.output_windows = check_output_windows(self.output_layout, output_windows)

    def forward(self, x):
        model = self.model
        head = model.head
        outputs = []
        for idx, feat in enumerate(model.neck(model.backbone(x))):
            cls_feat = reg_feat = feat
            for layer in head.cls_convs[idx]:
                cls_feat = layer(cls_feat)
            for layer in head.reg_convs[idx]:
                reg_feat = layer(reg_feat)
            outputs.append(self._final_conv("reg", head.rtm_reg[idx], reg_feat))
            outputs.append(self._final_conv("cls", head.rtm_cls[idx], cls_feat))
        return tuple(outputs)

    def _final_conv(self, part: str, conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        window = self.output_windows.get(part)
        if window is None:
            return conv(x)
        return _relu6_window(conv, x, *window)


class _RTMDetInsRawHeadsWrapper(torch.nn.Module):
    """RTMDet-Ins graph cut after each level's ``rtm_reg``/``rtm_cls``/``rtm_kernel``.

    Per level, stride 8, 16, 32: ``reg`` (ReLU'd l/t/r/b distances, before
    the stride scale), ``cls`` (class logits) and ``kernel`` (169 dynamic
    mask-conv parameters); then ``mask_feat`` (the stride-8 mask feature).
    Layout ``rtmdet_ins_raw_heads_v1``; the host rebuilds the head's native
    outputs (:func:`libreyolo.postprocess.yolonas_raw.decode_rtmdet_ins_raw`)
    and runs LibreYOLO's RTMDet-Ins postprocess. ``output_windows`` works as in
    :class:`_YOLONASRawHeadsWrapper` for the per-level parts; a ``reg`` window
    must start at 0, where it also replaces the ReLU.
    """

    output_layout = "rtmdet_ins_raw_heads_v1"
    grid_cell_offset = 0.0
    reg_max = None

    def __init__(
        self,
        model: torch.nn.Module,
        task: str = "segment",
        output_windows: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__()
        from ..postprocess.yolonas_raw import check_output_windows, raw_head_output_names

        if task != "segment":
            raise ValueError(f"RTMDet-Ins raw-head export supports segment only, got {task!r}.")
        self.model = model
        self.task = task
        head = model.head
        self.strides = tuple(int(stride) for stride in head.strides)
        self.output_names = raw_head_output_names(self.output_layout, self.strides)
        self.output_windows = check_output_windows(self.output_layout, output_windows or {})
        reg_window = self.output_windows.get("reg")
        if reg_window is not None and reg_window[0] != 0.0:
            raise ValueError(f"RTMDet-Ins reg window must start at 0 (ReLU), got {reg_window}.")

    def forward(self, x):
        model = self.model
        head = model.head
        feats = model.neck(model.backbone(x))
        outputs = []
        for idx, feat in enumerate(feats):
            cls_feat = reg_feat = kernel_feat = feat
            for layer in head.cls_convs[idx]:
                cls_feat = layer(cls_feat)
            for layer in head.reg_convs[idx]:
                reg_feat = layer(reg_feat)
            for layer in head.kernel_convs[idx]:
                kernel_feat = layer(kernel_feat)
            reg_window = self.output_windows.get("reg")
            if reg_window is None:
                outputs.append(torch.nn.functional.relu(head.rtm_reg[idx](reg_feat)))
            else:
                outputs.append(_relu6_window(head.rtm_reg[idx], reg_feat, *reg_window))
            outputs.append(self._final_conv("cls", head.rtm_cls[idx], cls_feat))
            outputs.append(self._final_conv("kernel", head.rtm_kernel[idx], kernel_feat))
        outputs.append(head.mask_head(feats))
        return tuple(outputs)

    def _final_conv(self, part: str, conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        window = self.output_windows.get(part)
        if window is None:
            return conv(x)
        return _relu6_window(conv, x, *window)


class _RetinaNetRawHeadsWrapper(torch.nn.Module):
    """RetinaNet graph cut right after each level's box-delta and class convs.

    Returns ``reg`` (``4 * A`` box deltas) and ``cls`` (``A * nc`` logits)
    per level P3-P7, as layout ``retinanet_raw_heads_v1``. The host rebuilds
    the anchors and the decoded export's ``(B, N, 4 + nc)`` tensor
    (:func:`libreyolo.postprocess.yolonas_raw.decode_retinanet_raw`). Exports
    take one fixed square canvas (the image fitted inside, padded
    bottom/right), so the anchors are fixed too.
    """

    output_layout = "retinanet_raw_heads_v1"
    grid_cell_offset = 0.0
    reg_max = None
    strides = (8, 16, 32, 64, 128)

    def __init__(self, model: torch.nn.Module, task: str = "detect"):
        super().__init__()
        from ..models.retinanet.nn import _ASPECT_RATIOS, _ANCHOR_SIZES, _COCO91_SOURCE_INDICES
        from ..postprocess.yolonas_raw import raw_head_output_names

        if task != "detect":
            raise ValueError(f"RetinaNet raw-head export supports detect only, got {task!r}.")
        self.model = model
        self.task = task
        self.output_names = raw_head_output_names(self.output_layout, self.strides)
        self.output_windows: dict[str, tuple[float, float]] = {}
        self.anchor_sizes = [list(sizes) for sizes in _ANCHOR_SIZES]
        self.aspect_ratios = [list(ratios) for ratios in _ASPECT_RATIOS]
        self.class_index_map = (
            list(_COCO91_SOURCE_INDICES) if model.num_classes == 91 else None
        )

    def forward(self, x):
        model = self.model
        head = model.head
        outputs = []
        for feature in model.backbone(x).values():
            reg = head.regression_head
            cls = head.classification_head
            outputs.append(reg.bbox_reg(reg.conv(feature)))
            outputs.append(cls.cls_logits(cls.conv(feature)))
        return tuple(outputs)



class _YOLO9SplitOutputWrapper(torch.nn.Module):
    """YOLO9 detection graph with boxes and scores as separate outputs.

    Same reason as :class:`_YOLOXSplitOutputWrapper`. Boxes are decoded per
    pyramid level into xyxy over ``(W, H, W, H)`` so each level's grid range
    is quantized on its own: ``boxes`` is ``(B, 4, N)`` and ``scores`` is
    ``(B, N, nc)`` sigmoid class probabilities.
    """

    def __init__(self, model: torch.nn.Module, canvas_hw: tuple[int, int]):
        super().__init__()
        self.model = model
        self.canvas_h, self.canvas_w = (int(v) for v in canvas_hw)

    def forward(self, x):
        feats = self.model.neck(*self.model.backbone(x))
        head = self.model.head
        boxes, scores = [], []
        for i, (feat, stride) in enumerate(zip(feats, head.strides)):
            h, w = feat.shape[-2:]
            # (B, 4, HW) ltrb distances in grid units
            distances = head.anchor2vec(head.anchor_convs[i](feat).flatten(2))
            shift_y, shift_x = torch.meshgrid(
                torch.arange(h, device=x.device, dtype=distances.dtype) + 0.5,
                torch.arange(w, device=x.device, dtype=distances.dtype) + 0.5,
                indexing="ij",
            )
            ax = shift_x.reshape(1, -1)
            ay = shift_y.reshape(1, -1)
            sx = float(stride) / self.canvas_w
            sy = float(stride) / self.canvas_h
            boxes.append(
                torch.stack(
                    [
                        (ax - distances[:, 0]) * sx,
                        (ay - distances[:, 1]) * sy,
                        (ax + distances[:, 2]) * sx,
                        (ay + distances[:, 3]) * sy,
                    ],
                    dim=1,
                )
            )
            cls_tower = head.class_convs[i]
            scores.append(
                _bounded_sigmoid(cls_tower[-1], cls_tower[:-1](feat))
                .flatten(2)
                .permute(0, 2, 1)
            )
        return torch.cat(boxes, 2), torch.cat(scores, 1)


# =============================================================================
# BaseExporter ABC
# =============================================================================


class BaseExporter(ABC):
    """Abstract base for all export formats.

    Subclasses set class-level attributes and implement ``_export()``.
    The ``__call__`` template method handles everything else.

    Example::

        from libreyolo.export import BaseExporter

        exporter = BaseExporter.create("onnx", model)
        path = exporter(output_path="model.onnx")

        # Or instantiate directly:
        from libreyolo.export import OnnxExporter
        path = OnnxExporter(model)(simplify=True, dynamic=True)
    """

    _registry: dict[str, type["BaseExporter"]] = {}

    # Alternate names accepted by create(). "litert" is Google's current name
    # for TensorFlow Lite; the format and .tflite suffix are unchanged.
    # The CLI and the `libreyolo formats` listing derive from this mapping,
    # so aliases live here and nowhere else.
    _aliases: dict[str, str] = {"engine": "tensorrt", "litert": "tflite"}

    # Class attributes (overridden by each subclass)
    format_name: str  # e.g. "onnx"
    suffix: str  # e.g. ".onnx"
    requires_onnx: bool  # TensorRT/OpenVINO need intermediate ONNX
    supports_int8: bool  # whether the format supports INT8 calibration
    supports_fp16: bool  # whether the format supports FP16 export
    apply_model_half: bool  # whether to cast model to fp16 (only ONNX/TorchScript)
    supports_embedded_nms: bool = False
    default_int8_calibration_data: bool = False
    # Export options read with kwargs.get() rather than named parameters;
    # the rest are the named parameters of __call__, _preflight and _export.
    _extra_export_kwargs: frozenset[str] = frozenset({"nms", "deepstream"})

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "format_name", None)
        if name is not None:
            BaseExporter._registry[name] = cls

    @classmethod
    def _accepted_export_kwargs(cls) -> set[str]:
        """Export options this format uses, from its signatures."""
        import inspect

        accepted = set(cls._extra_export_kwargs)
        for klass in cls.__mro__:
            for method in ("__call__", "_preflight", "_export"):
                function = klass.__dict__.get(method)
                if function is None:
                    continue
                accepted.update(
                    parameter.name
                    for parameter in inspect.signature(function).parameters.values()
                    if parameter.kind
                    in (parameter.KEYWORD_ONLY, parameter.POSITIONAL_OR_KEYWORD)
                )
        return accepted

    def __init__(self, model):
        self.model = model

    # Factory

    @classmethod
    def create(cls, format: str, model) -> "BaseExporter":
        """Look up *format* in the registry and return an exporter instance."""
        key = format.lower()
        key = cls._aliases.get(key, key)
        if key not in cls._registry:
            from libreyolo.utils.ultralytics_hints import export_format_hint

            valid = ", ".join(sorted(cls._registry))
            hint = export_format_hint(key)
            raise ValueError(
                f"Unsupported export format: {format!r}. Must be one of: {valid}"
                + (f". {hint}" if hint else "")
            )
        if model is not None and getattr(model, "FAMILY", None) == "domedetr":
            raise NotImplementedError(_DOMEDETR_EXPORT_MESSAGE)
        return cls._registry[key](model)

    # Template method

    def __call__(
        self,
        *,
        output_path: Optional[str] = None,
        imgsz: Optional[Union[int, Tuple[int, int]]] = None,
        opset: Optional[int] = None,
        simplify: bool = True,
        dynamic: bool = True,
        half: bool = False,
        int8: bool = False,
        batch: int = 1,
        device: Optional[str] = None,
        data: Optional[str] = None,
        fraction: float = 1.0,
        allow_download_scripts: bool = False,
        verbose: bool = False,
        quantize: Optional[Union[int, str]] = None,
        **kwargs,
    ) -> str:
        """Export the model.

        Args:
            output_path: Output file path (auto-generated if None).
            imgsz: Input resolution as ``(height, width)`` tuple or a single
                int for square (default: model's native size).
            opset: ONNX opset version (default: 13).
            simplify: Run ONNX graph simplification (default: True).
            dynamic: Enable dynamic axes for ONNX (default: True).
            half: Export in FP16 precision (default: False).
            int8: Export in INT8 precision (default: False).
            batch: Batch size for the model (default: 1).
            device: Device to trace on (default: model's current device).
            data: Path to data.yaml for INT8 calibration dataset.
            fraction: Fraction of calibration dataset to use (default: 1.0).
            allow_download_scripts: Allow embedded Python in dataset YAML downloads.
            verbose: Enable verbose logging (default: False).
            quantize: Precision as the ecosystem spells it: 16 (FP16, as
                ``half=True``), 8 (INT8, as ``int8=True``) or 32 (FP32).
            **kwargs: Format-specific parameters forwarded to ``_export()``.
                Options the format does not use are warned about and ignored.

        Returns:
            Path to the exported model file.
        """
        # Private plumbing (quantized_export): a callable run at the
        # post-validation mutation point below, so caller-side mutations
        # (reconstructing fp32 masters, enabling export mode) also wait for
        # every request rejection.
        pre_trace_hook = kwargs.pop("_pre_trace_hook", None)
        unknown = sorted(set(kwargs) - self._accepted_export_kwargs())
        if unknown:
            warnings.warn(
                f"Unknown {self.format_name} export arguments (ignored): {unknown}",
                stacklevel=3,
            )
        if quantize is not None:
            precision = {"16": "fp16", "8": "int8", "32": "fp32"}.get(str(quantize))
            if precision is None:
                raise ValueError(f"quantize must be 16, 8 or 32, got {quantize!r}.")
            if (half and precision != "fp16") or (int8 and precision != "int8"):
                raise ValueError(
                    f"quantize={quantize!r} conflicts with half={half}, int8={int8}."
                )
            half, int8 = precision == "fp16", precision == "int8"
        if isinstance(getattr(self.model, "input_profile", None), dict):
            if self.format_name != "onnx" or half or int8 or kwargs.get("nms", False):
                raise ValueError(
                    "Event histogram export currently supports FP32 ONNX without embedded NMS"
                )

        task = getattr(self.model, "task", "detect")
        model_name = self.model._get_model_name()
        if model_name == "ben2" and batch != 1:
            raise ValueError(
                "BEN2 export uses a fixed-resolution, batch-1 runtime "
                f"contract; got batch={batch}."
            )
        if task == "mesh":
            # Gated off for the first version, as semantic and point were: the
            # runtime metadata contract for a mesh graph (which body model,
            # how many betas, whether the body-model decoder is inside the
            # graph or applied afterwards) has to be defined before artifacts
            # exist that backends would have to keep reading.
            raise NotImplementedError(
                "Body-mesh export is not implemented yet. The exported-graph "
                "metadata contract for the mesh task is still to be defined; "
                "run mesh models through the PyTorch path for now."
            )
        if task in {"depth", "normal"}:
            # Dense-map export uses a fixed-resolution contract: backends
            # stretch-resize to the exported canvas and resize the result back
            # to the original canvas. The batch axis is static (dynamic is
            # forced off) and backends schedule one image per run, so a batch
            # != 1 artifact could never be fed correctly.
            task_label = "Depth" if task == "depth" else "Surface-normal"
            if batch != 1:
                raise ValueError(
                    f"{task_label} export uses a fixed-resolution, batch-1 runtime "
                    f"contract in v1; got batch={batch}."
                )
            if dynamic:
                warnings.warn(
                    f"{task_label} export uses a fixed-resolution runtime "
                    "contract in v1; forcing dynamic=False.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                dynamic = False
        if task == "edge":
            # Edge specialists export a single fused probability map on a
            # fixed square canvas; runtimes resize it back to the source.
            if batch != 1:
                raise ValueError(
                    "Edge export uses a fixed-resolution, batch-1 runtime "
                    f"contract in v1; got batch={batch}."
                )
            if dynamic:
                warnings.warn(
                    "Edge export uses a fixed-resolution runtime contract in "
                    "v1; forcing dynamic=False.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                dynamic = False
        if (
            getattr(self.model, "task", "detect") == "restore"
            and dynamic
            and self.model._get_model_name() not in {"realesrgan", "quicksrnet"}
        ):
            # Real-ESRGAN and QuickSRNet are fully convolutional and export with
            # dynamic H/W; other restore families keep the fixed-resolution v1
            # contract.
            warnings.warn(
                "Restore export uses a fixed-resolution runtime contract in "
                "v1; forcing dynamic=False.",
                RuntimeWarning,
                stacklevel=2,
            )
            dynamic = False
        if getattr(self.model, "task", "detect") == "matte" and dynamic:
            # BiRefNet's Swin relative-position tables are resolution-tied, so
            # the matte contract is the fixed native square (1024). Forcing a
            # dynamic graph would silently mis-interpolate them.
            warnings.warn(
                "Matte export uses a fixed-resolution runtime contract "
                "(native 1024); forcing dynamic=False.",
                RuntimeWarning,
                stacklevel=2,
            )
            dynamic = False
        half, int8 = self._validate(half, int8, data)
        self._preflight(half=half, int8=int8, data=data, **kwargs)
        data = self._resolve_calibration_data(int8, data)

        if opset is None:
            # DETR-style families use deformable attention / layer norm ops
            # which require opset 16+ (or 17 for ``aten::scaled_dot_product``
            # in the tuple export wrapper). Other families default to 13.
            opset = 17 if _requires_onnx_opset17(self.model._get_model_name()) else 13

        # The BiRefNet-derived decoders use torchvision deform_conv2d, which
        # maps to the standard ONNX ``DeformConv`` op (opset 19+). BEN2 is also
        # a matte family but has no deformable convolution, so it keeps the
        # normal exporter opset and needs no custom symbolic.
        if self.model._get_model_name() in {"birefnet", "feynobg"}:
            from ..models.birefnet.export import (
                MIN_OPSET as _MATTE_MIN_OPSET,
            )
            from ..models.birefnet.export import (
                register_deform_conv2d_onnx_symbolic,
            )

            if opset < _MATTE_MIN_OPSET:
                opset = _MATTE_MIN_OPSET
            register_deform_conv2d_onnx_symbolic(_MATTE_MIN_OPSET)

        imgsz, device, output_path = self._resolve_params(
            output_path,
            imgsz,
            device,
            half,
            int8,
        )
        # The CLI reports the canvas actually exported, which can differ from
        # the model's input size (see _square_fallback_for_restored_rect).
        self.model._last_export_imgsz = imgsz

        # ---- Post-validation mutation point ------------------------------
        # Every request rejection above — format support, precision
        # validation, option preflight, imgsz/path resolution — has fired;
        # from here on the live model may be mutated.
        #
        # A model fine-tuned with lora=True carries live PEFT adapter layers.
        # Fold them into dense weights so the traced graph is a plain model
        # with no peft dependency. The merge is destructive (adapters are
        # folded and removed), so it must not run any earlier.
        from ..training.lora import merge_lora_adapters, module_has_lora

        if module_has_lora(self.model.model):
            merge_lora_adapters(self.model.model)

        if pre_trace_hook is not None:
            pre_trace_hook()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        precision = _resolve_precision(half, int8)
        onnx_path = None

        try:
            with self._model_context(device, half, int8, batch, imgsz) as (
                nn_model,
                dummy,
            ):
                calibration_data = (
                    self._load_calibration(
                        data,
                        imgsz,
                        batch,
                        fraction,
                        allow_download_scripts,
                    )
                    if int8 and data is not None
                    else None
                )

                onnx_path = (
                    self._export_intermediate_onnx(
                        nn_model,
                        dummy,
                        output_path,
                        opset,
                        simplify,
                        dynamic,
                    )
                    if self.requires_onnx
                    else None
                )

                metadata = self._build_metadata(
                    precision,
                    dynamic,
                    onnx_path,
                    imgsz=imgsz,
                )

                result = self._export(
                    nn_model,
                    dummy,
                    output_path=output_path,
                    precision=precision,
                    metadata=metadata,
                    calibration_data=calibration_data,
                    onnx_path=onnx_path,
                    half=half,
                    int8=int8,
                    dynamic=dynamic,
                    opset=opset,
                    simplify=simplify,
                    verbose=verbose,
                    **kwargs,
                )
        finally:
            if onnx_path and Path(onnx_path).exists():
                Path(onnx_path).unlink()
            # Intermediate graphs a subclass registers before writing them, so
            # they go even when their export fails before onnx_path is set.
            for extra in getattr(self, "_intermediate_paths", ()):
                if Path(extra).exists():
                    Path(extra).unlink()

        self._print_summary(result, precision, imgsz)
        return result

    # Abstract export method

    @abstractmethod
    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path: str,
        precision: str = "fp32",
        metadata: dict | None = None,
        calibration_data=None,
        onnx_path: str | None = None,
        half: bool = False,
        int8: bool = False,
        dynamic: bool = False,
        opset: int = 13,
        simplify: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> str:
        """Format-specific export logic. Subclasses implement this only."""

    # Shared helpers

    def _validate(self, half: bool, int8: bool, data: Optional[str]):
        """Validate precision flags and calibration requirements."""
        if half and int8:
            warnings.warn(
                "Both half=True and int8=True specified. Using INT8 precision.",
                stacklevel=2,
            )
            half = False
        if half and not self.supports_fp16:
            raise NotImplementedError(
                f"{self.format_name.upper()} FP16 export is not supported."
            )
        if int8 and not self.supports_int8:
            raise NotImplementedError(
                f"{self.format_name.upper()} INT8 export is not supported."
            )
        if int8 and data is None and not self.default_int8_calibration_data:
            raise ValueError("INT8 export requires calibration data. Pass data=...")
        return half, int8

    def _resolve_calibration_data(
        self, int8: bool, data: Optional[str]
    ) -> Optional[str]:
        """Apply the default INT8 calibration dataset when data is omitted."""
        if not int8 or data is not None or not self.default_int8_calibration_data:
            return data
        logger.warning(
            "INT8 export requested without calibration data; using %s. "
            "This 8-image fallback is not representative. For accuracy validation, "
            "use a calibration dataset with roughly 300 or more representative images.",
            DEFAULT_INT8_CALIBRATION_DATA,
        )
        return DEFAULT_INT8_CALIBRATION_DATA

    def _preflight(self, *, half: bool, int8: bool, data: Optional[str], **kwargs):
        """Run cheap format-specific checks before model or calibration setup."""
        if kwargs.get("deepstream") and self.format_name != "onnx":
            raise ValueError(
                "deepstream=True is supported only for ONNX export "
                f"(format='onnx'); got format={self.format_name!r}."
            )
        family = self.model._get_model_name()
        task = getattr(self.model, "task", "detect")
        if not isinstance(task, str):
            task = "detect"
        support = get_support(family, task, self.format_name)
        if support.tier == "blocked":
            alternatives = validated_alternatives(family, task)
            alternatives_text = (
                f" Validated alternatives: {', '.join(alternatives)}."
                if alternatives
                else ""
            )
            raise NotImplementedError(
                f"{family} {task} export to {self.format_name} is blocked: "
                f"{support.reason}{alternatives_text}"
            )
        if support.tier == "hybrid":
            logger.info(
                "%s %s export to %s is hybrid: %s",
                family,
                task,
                self.format_name,
                support.reason,
            )
        size = str(getattr(self.model, "size", "") or "")
        structural = BLOCKED_SIZES.get((family, task, self.format_name), {}).get(size)
        if structural:
            raise NotImplementedError(
                f"{family}-{size} {task} export to {self.format_name} is blocked: "
                f"it {structural}."
            )
        unmeasured = UNMEASURED_SIZES.get((family, task, self.format_name), {}).get(size)
        if unmeasured:
            raise NotImplementedError(
                f"{family}-{size} {task} export to {self.format_name} is not "
                f"available yet: it {unmeasured}."
            )
        note = SIZE_ACCURACY_NOTES.get((family, task, self.format_name), {}).get(size)
        if note:
            # Supported, not validated: export anyway and say what was measured.
            logger.warning(
                "%s-%s %s export to %s is supported but not validated: %s.",
                family, size, task, self.format_name, note,
            )
        if kwargs.get("nms") and not self.supports_embedded_nms:
            raise NotImplementedError(
                f"{self.format_name.upper()} embedded NMS export is not supported."
            )
        if self.requires_onnx and importlib.util.find_spec("onnx") is None:
            raise ImportError(
                "ONNX export requires the 'onnx' package. "
                'Install with: pip install "libreyolo[onnx]"'
            )

    def _square_fallback_for_restored_rect(
        self, imgsz: tuple[int, int], model_name: str
    ) -> tuple[int, int]:
        """Keep default exports working for rectangular fine-tunes (#899).

        A square-native family (e.g. YOLOX) trained at ``imgsz=(h, w)`` reloads
        with that rectangular input size, but not every family/format pair can
        export it. Those exports defaulted to the family's square size before
        the size was restored, so fall back to a ``max(h, w)`` square, which
        letterboxes frames at the same scale the model was trained at. An
        explicit ``imgsz`` never reaches this path.
        """
        if not _is_rectangular_imgsz(imgsz):
            return imgsz
        if (
            model_name in _RECTANGULAR_EXPORT_FAMILIES
            and self._rectangular_imgsz_supported()
        ):
            return imgsz
        family_default = self.model._get_task_input_sizes().get(self.model.size)
        if isinstance(family_default, (tuple, list)):
            # Natively rectangular families keep their existing contract.
            return imgsz
        side = max(imgsz)
        logger.warning(
            "%s %s export does not support the checkpoint's rectangular imgsz=%s; "
            "exporting at the square imgsz=%d instead. Pass imgsz= to choose.",
            model_name,
            self.format_name,
            imgsz,
            side,
        )
        return side, side

    def _resolve_params(self, output_path, imgsz, device, half, int8):
        native_imgsz = self.model._get_input_size()
        model_name = self.model._get_model_name()
        if imgsz is None:
            if isinstance(native_imgsz, (tuple, list)):
                if len(native_imgsz) != 2:
                    raise ValueError(
                        "Native input size must be an int or (height, width), "
                        f"got {native_imgsz!r}."
                    )
                imgsz = (int(native_imgsz[0]), int(native_imgsz[1]))
                imgsz = self._square_fallback_for_restored_rect(imgsz, model_name)
            else:
                imgsz = (int(native_imgsz), int(native_imgsz))
        elif isinstance(imgsz, (tuple, list)):
            if len(imgsz) != 2:
                raise ValueError(f"imgsz tuple must be (height, width), got {imgsz}")
            imgsz = (int(imgsz[0]), int(imgsz[1]))
        else:
            imgsz = (int(imgsz), int(imgsz))
        if imgsz[0] <= 0 or imgsz[1] <= 0:
            raise ValueError(f"imgsz values must be positive, got {imgsz}.")
        imgsz = round_imgsz_to_stride(self.model, imgsz, "export")
        if model_name == "ben2":
            native_shape = (
                (int(native_imgsz[0]), int(native_imgsz[1]))
                if isinstance(native_imgsz, (tuple, list))
                else (int(native_imgsz), int(native_imgsz))
            )
            if imgsz != native_shape:
                raise ValueError(
                    "BEN2 export imgsz must match its fixed native resolution "
                    f"{native_shape[0]}x{native_shape[1]}, got {imgsz}."
                )
        if model_name in ("deit", "vgg") and imgsz != (native_imgsz, native_imgsz):
            raise ValueError(
                f"{model_name} export imgsz must match its fixed native resolution "
                f"{native_imgsz}x{native_imgsz}, got {imgsz}."
            )
        if model_name == "hrnet":
            native_h, native_w = (
                (int(native_imgsz[0]), int(native_imgsz[1]))
                if isinstance(native_imgsz, (tuple, list))
                else (int(native_imgsz), int(native_imgsz))
            )
            if imgsz != (native_h, native_w):
                raise ValueError(
                    "HRNet pose exports use the checkpoint's fixed person-crop "
                    f"canvas {(native_h, native_w)}, got {imgsz}."
                )
        imgsz_divisor = int(getattr(self.model, "IMGSZ_DIVISOR", 1) or 1)
        if imgsz[0] % imgsz_divisor or imgsz[1] % imgsz_divisor:
            raise ValueError(
                f"{model_name} export imgsz must be divisible by "
                f"{imgsz_divisor}, got {imgsz}."
            )
        if model_name == "nafnet":
            padder_size = int(getattr(self.model.model, "padder_size", 16))
            if imgsz[0] % padder_size or imgsz[1] % padder_size:
                raise ValueError(
                    "NAFNet export imgsz must be divisible by the network "
                    f"downsample factor {padder_size}, got {imgsz}."
                )
        dense_task = getattr(self.model, "task", "detect")
        divisor_attrs = {
            "depth": "depth_imgsz_divisor",
            "normal": "normal_imgsz_divisor",
            "semantic": "semantic_imgsz_divisor",
        }
        if dense_task in divisor_attrs:
            divisor_attr = divisor_attrs[dense_task]
            divisor = int(getattr(self.model, divisor_attr, 1) or 1)
            if imgsz[0] % divisor or imgsz[1] % divisor:
                raise ValueError(
                    f"{dense_task.capitalize()} export imgsz must be divisible "
                    "by the network "
                    f"stride {divisor}, got {imgsz}."
                )
        if getattr(self.model, "task", "detect") == "edge":
            divisor = int(getattr(self.model, "edge_imgsz_divisor", 1) or 1)
            if imgsz[0] % divisor or imgsz[1] % divisor:
                raise ValueError(
                    "Edge export imgsz must be divisible by the network "
                    f"stride {divisor}, got {imgsz}."
                )
        if model_name == "rfdetr":
            from ..models.rfdetr.imgsz import resolve_patch_window, validate_imgsz

            patch_size, num_windows = resolve_patch_window(self.model.model)
            imgsz = validate_imgsz(
                imgsz,
                patch_size=patch_size,
                num_windows=num_windows,
                name="RF-DETR export imgsz",
            )
        if model_name == "domedetr":
            raise NotImplementedError(_DOMEDETR_EXPORT_MESSAGE)
        rectangular_task = (
            model_name,
            getattr(self.model, "task", None),
        ) in _RECTANGULAR_EXPORT_TASKS
        if (
            _is_rectangular_imgsz(imgsz)
            and model_name in _FIXED_SQUARE_EXPORT_FAMILIES
            and not rectangular_task
        ):
            raise NotImplementedError(
                f"Rectangular imgsz export is not supported for {model_name}: "
                "this family uses a fixed square export/preprocessing spatial contract. "
                "Use the native square imgsz for now."
            )
        if (
            _is_rectangular_imgsz(imgsz)
            and model_name not in _RECTANGULAR_EXPORT_FAMILIES
            and not rectangular_task
        ):
            raise NotImplementedError(
                "Rectangular imgsz export is currently supported for "
                "YOLO9-family, HRNet, NAFNet, QuickSRNet, and Real-ESRGAN "
                "exports only."
            )
        if (
            _is_rectangular_imgsz(imgsz)
            and not self._rectangular_imgsz_supported()
        ):
            raise NotImplementedError(
                f"Rectangular imgsz export is not validated for format "
                f"{self.format_name!r}."
            )
        if device is None or str(device).lower() == "auto":
            if self.model._get_model_name() == "rfdetr":
                device = torch.device("cpu")
            else:
                device = self.model.device
        else:
            if isinstance(device, int):
                device = f"cuda:{device}"
            elif isinstance(device, str) and device.isdigit():
                device = f"cuda:{device}"
            device = torch.device(device)
        if output_path is None:
            output_path = self._auto_output_path(half, int8)
        return imgsz, device, output_path

    def _auto_output_path(self, half: bool, int8: bool) -> str:
        stem = self._auto_output_stem()
        precision_suffix = "_int8" if int8 else ("_fp16" if half else "")
        return str(Path("weights") / f"{stem}{precision_suffix}{self.suffix}")

    def _auto_output_stem(self) -> str:
        model_path = getattr(self.model, "model_path", None)
        if isinstance(model_path, (str, Path)):
            source = Path(model_path)
            if source.suffix.lower() in {".pt", ".pth", ".safetensors"}:
                return source.stem

        prefix = getattr(self.model, "FILENAME_PREFIX", None)
        size = getattr(self.model, "size", None)
        if isinstance(prefix, str) and prefix and isinstance(size, str) and size:
            task_suffix = self._auto_output_task_suffix()
            return f"{prefix}{size}{task_suffix}"

        model_name = self.model._get_model_name().lower()
        return f"{model_name}_{self.model.size}"

    def _auto_output_task_suffix(self) -> str:
        task = getattr(self.model, "task", "detect")
        if not isinstance(task, str):
            task = "detect"
        if getattr(self.model, "_is_segmentation", False) is True:
            task = "segment"
        try:
            suffix = task_to_suffix(task)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported task for auto output naming: {task!r}"
            ) from exc
        return f"-{suffix}" if suffix else ""

    def _rectangular_imgsz_supported(self) -> bool:
        """Whether this format keeps a natively rectangular family's canvas."""
        return self.format_name in _RECTANGULAR_EXPORT_FORMATS

    @contextmanager
    def _model_context(self, device, half, int8, batch, imgsz):
        """Setup model for export and restore state afterwards."""
        nn_model = self.model.model
        root_model = nn_model
        original_training = root_model.training
        root_model.eval()

        # A half-precision export must leave the caller's model untouched;
        # casting back with float() would keep the fp16 rounding.
        apply_half = half and not int8 and self.apply_model_half
        tensor_slots = _snapshot_tensor_slots(root_model) if apply_half else None

        original_device = next(root_model.parameters()).device
        root_model.to(device)

        # DETR-family export mode: wrap model so it returns a tuple instead
        # of dict and apply ``model.deploy()`` (BN fusion + prune non-eval
        # decoder layers). The wrapper is what gets traced; the original
        # model is restored on exit.
        dfine_wrapped = False
        rfdetr_export_activated = False
        # Non-None only for families whose graph consumes a 5D video clip.
        video_export_frames = None
        rfdetr_export_snapshots = []
        rfdetr_inner = None
        family = self.model._get_model_name()
        task = getattr(self.model, "task", "detect")
        moge2_onnx_mode = family in ("moge2", "moge3") and hasattr(
            root_model, "set_onnx_compatible_mode"
        )
        original_moge2_onnx_mode = bool(
            getattr(root_model, "onnx_compatible_mode", False)
        )
        if task == "semantic":
            nn_model = _SemanticExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "detr":
            from ..models.detr.nn import DETRExportWrapper

            nn_model = DETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "gtr" and getattr(self.model, "task", "detect") == "pose":
            from ..models.gtr.pose import GTRPoseExportWrapper

            nn_model = GTRPoseExportWrapper(copy.deepcopy(nn_model)).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "dfine" or (
            family == "gtr" and task in ("detect", "segment", "obb")
        ):
            from ..models.dfine.nn import DFINEExportWrapper

            # deploy() (BN fusion + decoder-layer pruning + head swap) mutates
            # the wrapped model in place; deepcopy first so the user's live
            # model is never destructively modified (mirrors the deimv2 path).
            nn_model = copy.deepcopy(nn_model)
            nn_model = DFINEExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "deim":
            from ..models.deim.nn import DEIMExportWrapper

            nn_model = copy.deepcopy(nn_model)
            nn_model = DEIMExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "deimv2":
            from ..models.deimv2.nn import DEIMv2ExportWrapper

            nn_model = copy.deepcopy(nn_model)
            nn_model = DEIMv2ExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "tinyformer":
            from ..models.tinyformer.nn import TinyFormerExportWrapper

            nn_model = copy.deepcopy(nn_model)
            nn_model = TinyFormerExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "ec":
            from ..models.ec.nn import ECExportWrapper

            nn_model = copy.deepcopy(nn_model)
            nn_model = ECExportWrapper(
                nn_model, task=getattr(self.model, "task", "detect")
            ).to(device)
            nn_model.eval()
            dfine_wrapped = True  # share the YOLOX-head-export skip path below
        elif family in {"yolo1", "yolo2", "yolo3", "yolo4"}:
            from ..models.darknet.export import DarknetExportWrapper

            # Bake the anchor-box decode (or the YOLOv1 dense-head decode) into the
            # graph so every export format emits a self-contained (B, 4+nc, N)
            # tensor consumed by the shared backend decode. No learned anchors
            # need to travel in metadata.
            nn_model = DarknetExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "yolo7":
            from ..models.yolo7.export import YOLO7ExportWrapper

            nn_model = YOLO7ExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family in {"yolonas", "ppyoloe"}:
            # Both heads return ``(decoded, raw)`` in eval; the wrapper keeps
            # the decoded (boxes, scores) pair and drops the training-only
            # auxiliaries.
            nn_model = _YOLONASExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "ppdoclayoutv3":
            from ..models.ppdoclayoutv3.nn import PPDocLayoutV3ExportWrapper

            nn_model = PPDocLayoutV3ExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "riodetrv2":
            # The encoder position embedding and decoder anchors are fixed-canvas
            # constants. At the native canvas the trained anchor buffers stay;
            # any other square size retargets both on a copy.
            from ..utils.image_size import imgsz_to_hw

            canvas = imgsz_to_hw(imgsz)
            if tuple(canvas) != tuple(nn_model.eval_spatial_size):
                nn_model = copy.deepcopy(nn_model)
                nn_model.set_eval_spatial_size(canvas)
            nn_model = _RTDETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family in {"rtdetr", "rtdetrv2", "rtdetrv4"}:
            nn_model = _RTDETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "lwdetr":
            from ..models.lwdetr.nn import LWDETRExportWrapper

            nn_model = LWDETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "mask_rcnn":
            from ..models.mask_rcnn.nn import MaskRCNNExportWrapper

            nn_model = MaskRCNNExportWrapper(
                nn_model,
                include_masks=task == "segment",
            ).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "faster_rcnn":
            from ..models.faster_rcnn.nn import FasterRCNNExportWrapper

            nn_model = FasterRCNNExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "retinanet":
            from ..models.retinanet.nn import RetinaNetExportWrapper

            nn_model = RetinaNetExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "ssd":
            from ..models.ssd.nn import SSDExportWrapper

            nn_model = SSDExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "fcos":
            from ..models.fcos.nn import FCOSExportWrapper

            nn_model = FCOSExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "efficientdet":
            from ..models.efficientdet.nn import EfficientDetExportWrapper

            if imgsz[0] != imgsz[1] or imgsz[0] != self.model.input_size:
                raise ValueError(
                    f"EfficientDet {self.model.size} exports require imgsz="
                    f"{self.model.input_size}; got {imgsz}."
                )
            nn_model = EfficientDetExportWrapper(
                nn_model,
                input_size=imgsz[0],
                # TensorRT's ITopK layer rejects K > 3840. Keep the exact
                # upstream 5000-point budget on every other runtime and use
                # its maximum only for the TensorRT graph.
                max_candidates=3840 if self.format_name == "tensorrt" else 5000,
                sparse_coco=(
                    getattr(self.model, "nb_classes", None) == 80
                    and getattr(self.model, "_arch_num_classes", None) == 90
                ),
            ).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "deformable_detr":
            from ..models.deformable_detr.nn import DeformableDETRExportWrapper

            nn_model = DeformableDETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "dinodetr":
            from ..models.dinodetr.nn import DINODETRExportWrapper

            nn_model = DINODETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "centernet":
            from ..models.centernet.nn import CenterNetExportWrapper

            # The portable grid-sample DCN flag is export-only. Keep the live
            # eager model on torchvision's exact native operator.
            nn_model = copy.deepcopy(nn_model)
            nn_model = CenterNetExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "rtmdet":
            # RTMDet intentionally aliases the head convolution weights across
            # feature levels while keeping one batch norm per level. XNNPACK's
            # batch-norm fusion assigns the shared parameters duplicate names,
            # so give the export-only copy independent modules with identical
            # weights. The user's live model and its sharing contract stay
            # untouched.
            nn_model = copy.deepcopy(nn_model)
            for tower_name in ("cls_convs", "reg_convs"):
                tower = getattr(nn_model.head, tower_name)
                for level in range(1, len(tower)):
                    for layer_index, layer in enumerate(tower[level]):
                        layer.conv = copy.deepcopy(tower[0][layer_index].conv)
            nn_model.to(device)
            nn_model.eval()
        elif family == "depth_anything3":
            nn_model = copy.deepcopy(nn_model)
            nn_model.export = True
            nn_model.to(device)
            nn_model.eval()
        elif family == "dinov2" and getattr(self.model, "task", None) in {
            "classify",
            "embed",
        }:
            # Classification and embedding have no detection decoder. Trace
            # their task-specific backbone path directly and bake the fixed
            # DINOv2 positional encoding before capture.
            if getattr(self.model, "task", None) == "classify":
                nn_model = nn_model.classifier.to(device)
            nn_model.eval()
            # Precompute static DINOv2 positional encodings for the fixed export
            # resolution; otherwise the dynamic bicubic-antialias interpolation
            # in the backbone is not ONNX-traceable.
            encoder = getattr(getattr(nn_model, "backbone", None), "encoder", None)
            if (
                encoder is not None
                and hasattr(encoder, "export")
                and not getattr(encoder, "_export", False)
            ):
                rfdetr_export_snapshots = _snapshot_rfdetr_export_state(nn_model)
                encoder.shape = (imgsz[0], imgsz[1])
                encoder.export()
                rfdetr_export_activated = True
            dfine_wrapped = True
        elif family in {"clip", "siglip2"} and task == "embed":
            image_tower = (
                nn_model.visual if family == "clip" else nn_model.vision_model
            )
            nn_model = _ImageEmbeddingExportWrapper(image_tower).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "vjepa2":
            # Every V-JEPA 2 graph takes a 5D clip. The embed graph pools to a
            # normalized row; the classify graph already ends at logits, so it
            # only needs the 5D dummy below.
            if task == "embed":
                nn_model = _VideoEmbeddingExportWrapper(nn_model).to(device)
            nn_model.eval()
            video_export_frames = int(getattr(self.model, "clip_frames", 64))
            dfine_wrapped = True
        elif family == "levjepa":
            nn_model = _CLSVideoEmbeddingExportWrapper(nn_model).to(device)
            nn_model.eval()
            video_export_frames = int(getattr(self.model, "clip_frames", 16))
            dfine_wrapped = True
        elif family in {"clip", "siglip2"} and task == "classify":
            text_embeds = getattr(self.model, "_text_embeds", None)
            if text_embeds is None:
                raise RuntimeError(
                    "No classes set; call set_classes() before export()."
                )
            scale = float(nn_model.logit_scale.exp().detach().cpu())
            weight = (scale * text_embeds).detach().to(device, torch.float32)
            if family == "clip":
                from ..models.clip.export import _FrozenCLIPClassifier

                nn_model = _FrozenCLIPClassifier(nn_model.visual, weight).to(device)
            else:
                from ..models.siglip2.export import _FrozenSigLIP2Classifier

                bias = nn_model.logit_bias.detach().to(
                    device=device,
                    dtype=torch.float32,
                )
                nn_model = _FrozenSigLIP2Classifier(
                    nn_model.vision_model,
                    weight,
                    bias.reshape(()),
                ).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "rfdetr":
            from ..models.rfdetr.nn import RFDETRExportWrapper

            rfdetr_inner = getattr(nn_model, "model", None)
            was_exported = getattr(rfdetr_inner, "_export", False)
            if not was_exported:
                rfdetr_export_snapshots = _snapshot_rfdetr_export_state(rfdetr_inner)
                if isinstance(getattr(self.model, "input_profile", None), dict):
                    # Bake positions at the graph's actual resolution. Baking at
                    # the RGB default and resizing again changes learned positions.
                    for module, state in rfdetr_export_snapshots:
                        if "shape" in state and "position_embeddings" in state:
                            module.shape = tuple(imgsz)
            nn_model = RFDETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
            rfdetr_export_activated = not was_exported

        # Set export mode for YOLOX/YOLOv9 heads
        original_export = None
        export_attr = None
        if (
            not dfine_wrapped
            and hasattr(nn_model, "head")
            and hasattr(nn_model.head, "export")
        ):
            export_attr = "head"
            original_export = nn_model.head.export
            nn_model.head.export = True

        # RF-DETR export mode
        rfdetr_layernorm_patches = []
        inner = rfdetr_inner or getattr(nn_model, "model", None)
        if (
            inner is not None
            and hasattr(inner, "forward_export")
            and hasattr(inner, "_export")
        ):
            if not inner._export:
                inner.export()
                rfdetr_export_activated = True

            try:
                from ..models.rfdetr.backbone import LayerNorm as RFDETRLayerNorm

                for m in nn_model.modules():
                    if isinstance(m, RFDETRLayerNorm):
                        rfdetr_layernorm_patches.append((m, m.forward))
                        ns = m.normalized_shape

                        def _static_forward(
                            x, _ns=ns, _w=m.weight, _b=m.bias, _eps=m.eps
                        ):
                            x = x.permute(0, 2, 3, 1)
                            x = torch.nn.functional.layer_norm(x, _ns, _w, _b, _eps)
                            return x.permute(0, 3, 1, 2)

                        m.forward = _static_forward
            except ImportError:
                pass

        h, w = imgsz
        if video_export_frames is not None:
            # Public clip layout (B, F, C, H, W). Frame count, crop and tubelet
            # geometry are fixed per graph; only batch may be dynamic.
            dummy = torch.randn(batch, video_export_frames, 3, h, w, device=device)
        else:
            from ..utils.input_channels import model_channels

            channels = (
                2
                if isinstance(getattr(self.model, "input_profile", None), dict)
                else model_channels(self.model)
            )
            dummy = torch.randn(batch, channels, h, w, device=device)

        if apply_half:
            nn_model.half()
            dummy = dummy.half()

        if moge2_onnx_mode:
            root_model.set_onnx_compatible_mode(True)
        try:
            yield nn_model, dummy
        finally:
            if moge2_onnx_mode:
                root_model.set_onnx_compatible_mode(original_moge2_onnx_mode)
            if rfdetr_export_snapshots:
                _restore_rfdetr_export_state(rfdetr_export_snapshots)
            nn_model.to(original_device)
            root_model.to(original_device)
            if tensor_slots is not None:
                _restore_tensor_slots(tensor_slots)
            if original_training:
                root_model.train()
                nn_model.train()
            if original_export is not None:
                getattr(nn_model, export_attr).export = original_export
            if (
                rfdetr_export_activated
                and not rfdetr_export_snapshots
                and inner is not None
            ):
                for module in inner.modules():
                    if hasattr(module, "_forward_origin"):
                        module.forward = module._forward_origin
                    if hasattr(module, "_export"):
                        module._export = False
            for m, orig_fwd in rfdetr_layernorm_patches:
                m.forward = orig_fwd

    def _load_calibration(
        self,
        data,
        imgsz,
        batch,
        fraction,
        allow_download_scripts=False,
    ):
        from .calibration import get_calibration_dataloader

        preprocess_fn = self.model._get_preprocess_numpy()
        from ..utils.input_channels import gray_preprocess_fn, model_channels

        if model_channels(self.model) == 1:
            # Grayscale graphs calibrate on the one-channel input they consume.
            preprocess_fn = gray_preprocess_fn(preprocess_fn)
        calibration_data = get_calibration_dataloader(
            data=data,
            imgsz=imgsz,
            batch=batch,
            fraction=fraction,
            preprocess_fn=preprocess_fn,
            allow_download_scripts=allow_download_scripts,
        )
        logger.info(
            "Calibration dataset: %d batches, %d images",
            len(calibration_data),
            calibration_data.num_samples,
        )
        return calibration_data

    def _export_intermediate_onnx(
        self, nn_model, dummy, output_path, opset, simplify, dynamic
    ):
        # Use a distinct intermediate name so it never collides with (and the
        # finally-block cleanup never deletes) a real ONNX export the user may
        # have written to the default ``<stem>.onnx`` path.
        out = Path(output_path)
        onnx_output = str(out.with_name(f"{out.stem}.export_intermediate.onnx"))
        logger.info("Step 1/2: Exporting to ONNX (%s)", onnx_output)
        return export_onnx(
            nn_model,
            dummy,
            output_path=onnx_output,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            half=False,
            metadata=self._build_onnx_metadata(
                dynamic=dynamic,
                half=False,
                imgsz=(dummy.shape[-2], dummy.shape[-1]),
            ),
        )

    def _build_metadata(
        self,
        precision: str,
        dynamic: bool,
        onnx_path: Optional[str],
        imgsz: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> dict:
        """Build metadata dict for non-ONNX formats (native Python types)."""
        task, supported_tasks, default_task = self._task_metadata()
        if imgsz is not None:
            if isinstance(imgsz, tuple):
                h, w = imgsz
                metadata_imgsz = max(h, w)
                meta_h, meta_w = h, w
            else:
                metadata_imgsz = int(imgsz)
                meta_h = meta_w = int(imgsz)
        else:
            native = self.model._get_input_size()
            if isinstance(native, (tuple, list)):
                meta_h, meta_w = int(native[0]), int(native[1])
                metadata_imgsz = max(meta_h, meta_w)
            else:
                metadata_imgsz = int(native)
                meta_h = meta_w = int(native)
        # TODO(schema-v1.1): keep legacy model_size/nb_classes aliases for one
        # transition window, then prefer the canonical size/nc keys only.
        meta = {
            "schema_version": SCHEMA_VERSION,
            "libreyolo_version": _get_version(),
            "model_family": self.model._get_model_name(),
            "size": self.model.size,
            "model_size": self.model.size,
            "task": task,
            "supported_tasks": supported_tasks,
            "default_task": default_task,
            "nc": self.model.nb_classes,
            "nb_classes": self.model.nb_classes,
            "names": {str(k): v for k, v in self.model.names.items()},
            "imgsz": metadata_imgsz,
            "imgsz_h": meta_h,
            "imgsz_w": meta_w,
            "precision": precision,
            "dynamic": dynamic,
            "obb": task == "obb",
        }
        if onnx_path is not None:
            meta["exported_from"] = str(Path(onnx_path).name)
        meta.update(_letterbox_pad_metadata(self.model))
        from ..utils.input_channels import channels_metadata

        meta.update(channels_metadata(self.model))
        # Classification eval preprocessing must travel with every artifact,
        # not just ONNX. Exported-backend predict() otherwise falls back to
        # crop_pct=0.875 and bilinear resize, which changes classifier logits
        # for families such as ResNet (0.95/bicubic).
        if task == "classify":
            meta.update(_classify_eval_metadata(self.model))
        if task == "pose":
            meta.update(_pose_keypoint_shape_metadata(self.model))
            if self.model._get_model_name() == "hrnet":
                meta["pose_input"] = "person_crop"
        if task == "gaze":
            meta.update(
                {
                    "num_bins": int(self.model.num_bins),
                    "bin_width_deg": float(self.model.bin_width_deg),
                    "offset_deg": float(self.model.offset_deg),
                    "gaze_input": "face_crop",
                }
            )
        return meta

    def _build_onnx_metadata(
        self,
        *,
        dynamic: bool,
        half: bool,
        imgsz: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> dict:
        """Build metadata dict for ONNX (all-string values, JSON-encoded names)."""
        task, supported_tasks, default_task = self._task_metadata()
        if imgsz is not None:
            if isinstance(imgsz, tuple):
                h, w = imgsz
                metadata_imgsz = str(max(h, w))
                meta_h = str(h)
                meta_w = str(w)
            else:
                metadata_imgsz = str(int(imgsz))
                meta_h = meta_w = str(int(imgsz))
        else:
            native = self.model._get_input_size()
            if isinstance(native, (tuple, list)):
                native_h, native_w = int(native[0]), int(native[1])
                metadata_imgsz = str(max(native_h, native_w))
                meta_h, meta_w = str(native_h), str(native_w)
            else:
                metadata_imgsz = str(int(native))
                meta_h = meta_w = str(int(native))
        # TODO(schema-v1.1): keep legacy model_size/nb_classes aliases for one
        # transition window, then prefer the canonical size/nc keys only.
        meta = {
            "schema_version": SCHEMA_VERSION,
            "libreyolo_version": _get_version(),
            "model_family": self.model._get_model_name(),
            "size": self.model.size,
            "model_size": self.model.size,
            "task": task,
            "supported_tasks": json.dumps(supported_tasks),
            "default_task": default_task,
            "nc": str(self.model.nb_classes),
            "nb_classes": str(self.model.nb_classes),
            "names": json.dumps({str(k): v for k, v in self.model.names.items()}),
            "imgsz": metadata_imgsz,
            "imgsz_h": meta_h,
            "imgsz_w": meta_w,
            "dynamic": str(dynamic),
            "precision": "fp16" if half else "fp32",
            "half": str(half),
            "segmentation": str(
                task == "segment" or getattr(self.model, "_is_segmentation", False)
            ).lower(),
            "obb": str(task == "obb").lower(),
        }
        from ..utils.event_histogram import input_metadata
        for key, value in input_metadata(self.model).items():
            meta[key] = json.dumps(value) if isinstance(value, dict) else str(value)
        from ..utils.input_channels import channels_metadata

        for key, value in channels_metadata(self.model).items():
            meta[key] = str(value)
        meta.update(_letterbox_pad_metadata(self.model))
        # Classification eval preprocessing — lets exported-backend inference
        # and validation match native predict()/val() (#886).
        for key, value in _classify_eval_metadata(self.model).items():
            meta[key] = str(value)
        if task == "pose":
            pose_meta = _pose_keypoint_shape_metadata(self.model)
            meta.update(
                {
                    "num_keypoints": str(pose_meta["num_keypoints"]),
                    "keypoint_dim": str(pose_meta["keypoint_dim"]),
                }
            )
            if "num_keypoints_per_class" in pose_meta:
                meta["num_keypoints_per_class"] = json.dumps(
                    pose_meta["num_keypoints_per_class"]
                )
            if self.model._get_model_name() == "hrnet":
                meta["pose_input"] = "person_crop"
        if task == "gaze":
            meta.update(
                {
                    "num_bins": str(int(self.model.num_bins)),
                    "bin_width_deg": str(float(self.model.bin_width_deg)),
                    "offset_deg": str(float(self.model.offset_deg)),
                    "gaze_input": "face_crop",
                }
            )
        return meta

    def _task_metadata(self) -> tuple[str, list[str], str]:
        task = getattr(self.model, "task", "detect")
        if not isinstance(task, str):
            task = "detect"
        supported_tasks = getattr(self.model, "SUPPORTED_TASKS", ("detect",))
        if not isinstance(supported_tasks, (list, tuple)):
            supported_tasks = ("detect",)
        default_task = getattr(self.model, "DEFAULT_TASK", "detect")
        if not isinstance(default_task, str):
            default_task = "detect"
        if self.model._get_model_name() in {"mask_rcnn", "rfdetr"}:
            return task, [task], task
        return task, list(supported_tasks), default_task

    def _print_summary(
        self, result: str, precision: str, imgsz: Union[int, Tuple[int, int]]
    ):
        if isinstance(imgsz, tuple):
            h, w = imgsz
        else:
            h = w = imgsz
        logger.info(
            "Export complete: %s\n"
            "  Model: %s %s\n"
            "  Format: %s\n"
            "  Precision: %s\n"
            "  Input size: %dx%d",
            result,
            self.model._get_model_name(),
            self.model.size,
            self.format_name,
            _precision_label(precision),
            w,
            h,
        )


# =============================================================================
# Subclasses — one per format
# =============================================================================


class OnnxExporter(BaseExporter):
    format_name = "onnx"
    suffix = ".onnx"
    requires_onnx = False
    supports_int8 = True
    supports_fp16 = True
    apply_model_half = True
    supports_embedded_nms = True
    default_int8_calibration_data = True

    def _resolve_params(self, output_path, imgsz, device, half, int8):
        imgsz, device, output_path = super()._resolve_params(
            output_path, imgsz, device, half, int8
        )
        family = self.model._get_model_name()
        size = getattr(self.model, "size", None)
        two_stage = family == "deformable_detr" and size == "r50twostage"
        if two_stage and half:
            raise NotImplementedError(
                "Deformable DETR two-stage ONNX export is validated in FP32 only."
            )
        # DINO-DETR selects its queries with the same two-stage encoder top-k.
        if (two_stage or family == "dinodetr") and device.type != "cpu":
            label = "DINO-DETR" if family == "dinodetr" else "Deformable DETR two-stage"
            warnings.warn(
                f"{label} ONNX export is traced on CPU because "
                "the legacy PyTorch exporter can terminate while lowering its "
                "CUDA top-k graph. The model is restored to its original device "
                "after export.",
                RuntimeWarning,
                stacklevel=3,
            )
            device = torch.device("cpu")
        return imgsz, device, output_path

    def _preflight(self, *, half: bool, int8: bool, data: Optional[str], **kwargs):
        if int8:
            task = getattr(self.model, "task", "detect")
            if not isinstance(task, str):
                task = "detect"
            if (self.model._get_model_name(), task) not in INT8_EXPORTS["onnx"]:
                raise NotImplementedError(
                    "ONNX INT8 export currently supports YOLO9 detection models only."
                )
            check_onnx_int8_available()
        if kwargs.get("nms"):
            task = getattr(self.model, "task", "detect")
            if not isinstance(task, str):
                task = "detect"
            if self.model._get_model_name() != "yolo9" or task != "detect":
                raise NotImplementedError(
                    "Embedded NMS ONNX export currently supports YOLO9 "
                    "detection models only."
                )
        if kwargs.get("deepstream"):
            from .deepstream import (
                deepstream_supported_families,
                deepstream_supported_tasks,
            )

            if kwargs.get("nms"):
                raise ValueError(
                    "deepstream=True and nms=True are mutually exclusive: "
                    "DeepStream runs suppression in its clustering stage."
                )
            task = getattr(self.model, "task", "detect")
            if not isinstance(task, str):
                task = "detect"
            family = self.model._get_model_name()
            supported = deepstream_supported_families(task)
            if not supported:
                supported_tasks = ", ".join(sorted(deepstream_supported_tasks()))
                raise NotImplementedError(
                    f"DeepStream export does not support the {task!r} task. "
                    f"Supported tasks: {supported_tasks}."
                )
            if family not in supported:
                raise NotImplementedError(
                    f"DeepStream {task} export supports families "
                    f"{sorted(supported)}; got {family!r}."
                )
        super()._preflight(half=half, int8=int8, data=data, **kwargs)

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        calibration_data,
        half,
        int8,
        dynamic,
        opset,
        simplify,
        nms=False,
        deepstream=False,
        iou=0.45,
        conf=0.25,
        max_det=300,
        calibrate_method="MinMax",
        nodes_to_exclude=None,
        **kwargs,
    ):
        imgsz = (dummy.shape[-2], dummy.shape[-1])

        if deepstream:
            from .deepstream import wrap_for_deepstream

            ds_task = getattr(self.model, "task", "detect")
            if not isinstance(ds_task, str):
                ds_task = "detect"
            nn_model = wrap_for_deepstream(
                nn_model,
                model_family=self.model._get_model_name(),
                imgsz=imgsz,
                model_size=getattr(self.model, "size", None),
                task=ds_task,
            ).eval()

        if nms:
            from .nms import EmbeddedNMSDetector

            if dummy.shape[0] != 1:
                raise NotImplementedError(
                    "Embedded NMS ONNX export currently requires batch=1."
                )
            if dynamic:
                logger.warning(
                    "Embedded NMS uses a fixed batch-1 graph; forcing dynamic=False."
                )
                dynamic = False
            nn_model = EmbeddedNMSDetector(
                nn_model,
                conf=conf,
                iou=iou,
                max_det=max_det,
            ).eval()

        def _onnx_metadata(precision_half: bool) -> dict:
            meta = self._build_onnx_metadata(
                dynamic=dynamic,
                half=precision_half,
                imgsz=imgsz,
            )
            if nms:
                meta["nms"] = "true"
                meta["nms_conf"] = str(conf)
                meta["nms_iou"] = str(iou)
                meta["max_det"] = str(max_det)
                meta["nms_raw_output"] = "true"
            if deepstream:
                meta["deepstream"] = "true"
            return meta

        from ..utils.input_channels import model_channels

        def _write_deepstream_sidecars(onnx_result_path: str) -> None:
            if not deepstream:
                return
            from .deepstream import write_deepstream_sidecars

            names = getattr(self.model, "names", None) or {}
            class_names = [names[k] for k in sorted(names, key=int)]
            write_deepstream_sidecars(
                onnx_result_path,
                model_family=self.model._get_model_name(),
                class_names=class_names,
                imgsz=imgsz,
                batch=dummy.shape[0],
                precision="int8" if int8 else ("fp16" if half else "fp32"),
                conf=conf,
                iou=iou,
                task=ds_task,
                letterbox_pad=_letterbox_pad_metadata(self.model).get("letterbox_pad"),
                channels=model_channels(self.model),
            )

        if int8:
            import tempfile

            output = Path(output_path)
            int8_metadata = _onnx_metadata(precision_half=False)
            int8_metadata["precision"] = "int8"
            with tempfile.TemporaryDirectory(
                prefix=f"{output.stem}_", dir=str(output.parent)
            ) as tmpdir:
                fp32_path = str(Path(tmpdir) / "model_fp32.onnx")
                preprocessed_path = str(Path(tmpdir) / "model_fp32_infer.onnx")
                export_onnx(
                    nn_model,
                    dummy,
                    output_path=fp32_path,
                    opset=opset,
                    simplify=simplify,
                    dynamic=dynamic,
                    half=False,
                    metadata=_onnx_metadata(precision_half=False),
                    nms=nms,
                    deepstream=deepstream,
                )
                # Without explicit nodes_to_exclude, the family's float
                # layers (YOLO9: first conv and the head) stay float, as in
                # model.quantize().
                keep_high_precision = ()
                if nodes_to_exclude is None:
                    from ..quant.api import default_keep_high_precision

                    keep_high_precision = default_keep_high_precision(
                        self.model._get_model_name()
                    )
                result = quantize_onnx_int8(
                    fp32_path,
                    output_path,
                    calibration_data=calibration_data,
                    metadata=int8_metadata,
                    preprocessed_path=preprocessed_path,
                    calibrate_method=calibrate_method,
                    nodes_to_exclude=nodes_to_exclude,
                    # ORT symbolic shape inference fails on dynamic-batch
                    # and embedded-NMS graphs ("Incomplete symbolic shape
                    # inference"); plain ONNX shape inference still runs.
                    skip_symbolic_shape=nms or dynamic,
                    keep_high_precision=keep_high_precision,
                )
                _write_deepstream_sidecars(result)
                return result

        result = export_onnx(
            nn_model,
            dummy,
            output_path=output_path,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            half=half,
            metadata=_onnx_metadata(precision_half=half),
            nms=nms,
            deepstream=deepstream,
        )
        _write_deepstream_sidecars(result)
        return result


class TorchScriptExporter(BaseExporter):
    format_name = "torchscript"
    suffix = ".torchscript"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = True
    apply_model_half = True

    def _resolve_params(self, output_path, imgsz, device, half, int8):
        if device is None or str(device).lower() == "auto":
            device = torch.device("cpu")
        return super()._resolve_params(output_path, imgsz, device, half, int8)

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        # The graph is traced at a fixed input shape, so report dynamic=False
        # regardless of the requested flag (mirrors the NCNN override).
        meta = super()._build_metadata(precision, dynamic, onnx_path, imgsz=imgsz)
        meta["dynamic"] = False
        return meta

    def _export(self, nn_model, dummy, *, output_path, metadata, **kwargs):
        from .coreai import _freeze_anchor_grid

        # A traced graph keeps the device of every tensor it creates, so a head
        # that builds its anchor grid in forward (YOLO9) would make that grid on
        # the trace device (CPU by default) after loading on CUDA. The graph is
        # fixed-shape anyway: pin the grid as constants, which
        # torch.jit.load(map_location=...) moves with the weights.
        restore = _freeze_anchor_grid(nn_model, dummy)
        try:
            return export_torchscript(
                nn_model, dummy, output_path=output_path, metadata=metadata
            )
        finally:
            restore()


class ExecuTorchExporter(BaseExporter):
    """Fixed-shape, batch-1, FP32 ExecuTorch export with XNNPACK delegation."""

    format_name = "executorch"
    suffix = ".pte"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False

    def __call__(
        self, *, dynamic: bool = False, batch: int = 1, **kwargs
    ) -> str:
        """Reject unsupported shapes before the destructive LoRA merge."""
        if batch != 1:
            raise ValueError(
                f"ExecuTorch v1 requires batch=1, got batch={batch}."
            )
        if dynamic:
            raise ValueError("ExecuTorch v1 requires dynamic=False.")
        return super().__call__(dynamic=False, batch=1, **kwargs)

    def _resolve_params(self, output_path, imgsz, device, half, int8):
        if device is not None and str(device).lower() not in {"auto", "cpu"}:
            raise ValueError("ExecuTorch XNNPACK export requires device='cpu'.")
        return super()._resolve_params(
            output_path, imgsz, torch.device("cpu"), half, int8
        )

    def _preflight(self, *, half: bool, int8: bool, data: Optional[str], **kwargs):
        delegate = str(kwargs.get("delegate", "xnnpack")).lower()
        if delegate != "xnnpack":
            raise ValueError(
                "ExecuTorch v1 supports delegate='xnnpack' only, "
                f"got {delegate!r}."
            )
        super()._preflight(half=half, int8=int8, data=data, **kwargs)
        from .executorch import check_executorch_available

        check_executorch_available()

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        meta = super()._build_metadata(
            precision, False, onnx_path, imgsz=imgsz
        )
        meta.update(_classify_eval_metadata(self.model))
        return meta

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        dynamic,
        delegate="xnnpack",
        **kwargs,
    ):
        if dummy.shape[0] != 1:
            raise ValueError(
                f"ExecuTorch v1 requires batch=1, got batch={dummy.shape[0]}."
            )
        if dynamic:
            raise ValueError("ExecuTorch v1 requires dynamic=False.")
        from .executorch import export_executorch

        return export_executorch(
            nn_model,
            dummy,
            output_path=output_path,
            metadata=metadata,
        )


class TensorRTExporter(BaseExporter):
    format_name = "tensorrt"
    suffix = ".engine"
    requires_onnx = True
    supports_int8 = True
    supports_fp16 = True
    apply_model_half = False

    def _preflight(self, **kwargs):
        super()._preflight(**kwargs)
        from .tensorrt import check_tensorrt_available

        check_tensorrt_available()

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        precision,
        metadata,
        calibration_data,
        onnx_path,
        half,
        int8,
        dynamic,
        verbose,
        workspace=4.0,
        min_batch=1,
        opt_batch=1,
        max_batch=8,
        hardware_compatibility="none",
        gpu_device=0,
        trt_config=None,
        **kwargs,
    ):
        from .tensorrt import export_tensorrt

        trt_metadata = dict(metadata or {})
        if dynamic:
            trt_metadata.update(
                {
                    "trt_min_batch": int(min_batch),
                    "trt_opt_batch": int(opt_batch),
                    "trt_max_batch": int(max_batch),
                }
            )

        logger.info("Step 2/2: Building TensorRT engine")
        return export_tensorrt(
            onnx_path=onnx_path,
            output_path=output_path,
            half=half,
            int8=int8,
            workspace=workspace,
            calibration_data=calibration_data,
            dynamic=dynamic,
            verbose=verbose,
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
            hardware_compatibility=hardware_compatibility,
            device=gpu_device,
            config=trt_config,
            metadata=trt_metadata,
        )


class OpenVINOExporter(BaseExporter):
    format_name = "openvino"
    suffix = "_openvino"
    requires_onnx = True
    supports_int8 = True
    supports_fp16 = True
    apply_model_half = False

    def _preflight(self, **kwargs):
        super()._preflight(**kwargs)
        from .openvino import check_openvino_available

        check_openvino_available()

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        calibration_data,
        onnx_path,
        half,
        int8,
        verbose,
        **kwargs,
    ):
        from .openvino import export_openvino

        logger.info("Step 2/2: Converting to OpenVINO IR")
        return export_openvino(
            onnx_path=onnx_path,
            output_path=output_path,
            half=half,
            int8=int8,
            calibration_data=calibration_data,
            verbose=verbose,
            metadata=metadata,
        )

class PaddleExporter(BaseExporter):
    """Static batch-1 FP32 Paddle export through X2Paddle."""

    format_name = "paddle"
    suffix = "_paddle"
    requires_onnx = True
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False

    def __call__(
        self,
        *,
        dynamic: bool = False,
        batch: int = 1,
        simplify: bool = True,
        opset: int | None = None,
        **kwargs,
    ) -> str:
        if dynamic:
            raise ValueError("Paddle export requires dynamic=False.")
        if batch != 1:
            raise ValueError(f"Paddle export currently requires batch=1, got {batch}.")
        if not simplify:
            raise ValueError(
                "Paddle export requires simplify=True for a fully static "
                "X2Paddle conversion graph."
            )
        if opset is not None and int(opset) != 15:
            raise ValueError(
                "Paddle export requires opset=15 for the validated X2Paddle "
                f"conversion graph, got {opset}."
            )
        return super().__call__(
            dynamic=False,
            batch=1,
            simplify=True,
            opset=15,
            **kwargs,
        )

    def _preflight(self, **kwargs):
        # Check support policy before importing the optional converter stack.
        super()._preflight(**kwargs)
        from .paddle import check_paddle_export_available

        check_paddle_export_available()

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        metadata = super()._build_metadata(
            precision, False, onnx_path, imgsz=imgsz
        )
        metadata["dynamic"] = False
        metadata.pop("exported_from", None)
        return metadata

    def _export(
        self, nn_model, dummy, *, onnx_path, output_path, metadata, **kwargs
    ):
        from .paddle import export_paddle

        logger.info("Step 2/2: Converting ONNX to Paddle")
        return export_paddle(
            onnx_path=onnx_path,
            output_path=output_path,
            metadata=metadata,
        )


class MNNExporter(BaseExporter):
    """Export a fixed-shape FP32 MNN artifact through ONNX."""

    format_name = "mnn"
    suffix = ".mnn"
    requires_onnx = True
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False

    def __call__(self, *args, dynamic: bool = False, **kwargs) -> str:
        if dynamic:
            raise ValueError("MNN v1 export requires dynamic=False.")
        return super().__call__(*args, dynamic=False, **kwargs)

    def _preflight(self, **kwargs):
        super()._preflight(**kwargs)
        from .mnn import check_mnn_available

        check_mnn_available()

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        meta = super()._build_metadata(precision, dynamic, onnx_path, imgsz=imgsz)
        meta["dynamic"] = False
        return meta

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        onnx_path,
        dynamic,
        verbose,
        **kwargs,
    ):
        if dynamic:
            raise ValueError("MNN v1 export requires dynamic=False.")
        from .mnn import export_mnn

        logger.info("Step 2/2: Converting to MNN")
        return export_mnn(
            onnx_path,
            output_path,
            metadata=metadata,
            batch=int(dummy.shape[0]),
            verbose=verbose,
        )


class RknnExporter(BaseExporter):
    """Compile an exact simulator-tested detector variant for RK3588."""

    format_name = "rknn"
    suffix = ".rknn"
    requires_onnx = True
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False

    def __call__(
        self,
        *args,
        dynamic: bool = False,
        batch: int = 1,
        imgsz: int | tuple[int, int] | None = None,
        opset: int | None = 19,
        **kwargs,
    ) -> str:
        if dynamic:
            raise ValueError("RKNN export requires static input shapes.")
        if batch != 1:
            raise NotImplementedError("RKNN export currently supports batch=1 only.")
        if opset not in {None, 19}:
            raise NotImplementedError(
                f"RKNN export is validated only with ONNX opset 19, got {opset}."
            )
        from .rknn import resolve_rknn_imgsz

        imgsz = resolve_rknn_imgsz(
            model_family=self.model._get_model_name(),
            model_size=self.model.size,
            task=getattr(self.model, "task", "detect"),
            imgsz=imgsz,
        )
        return super().__call__(
            *args,
            dynamic=False,
            batch=1,
            imgsz=imgsz,
            opset=19,
            **kwargs,
        )

    def _validate(self, half: bool, int8: bool, data: str | None):
        if half:
            raise NotImplementedError(
                "RKNN does not expose LibreYOLO's half=True contract. Omit half "
                "and int8 for the tested vendor floating-point build."
            )
        return super()._validate(half, int8, data)

    def _preflight(self, **kwargs):
        from .rknn import (
            check_rknn_available,
            resolve_rknn_target,
            validate_rknn_export_request,
        )

        target_platform = resolve_rknn_target(
            name=kwargs.get("name"),
            target=kwargs.get("target"),
            target_platform=kwargs.get("target_platform"),
        )
        validate_rknn_export_request(
            model_family=self.model._get_model_name(),
            model_size=self.model.size,
            task=getattr(self.model, "task", "detect"),
            target_platform=target_platform,
        )
        super()._preflight(**kwargs)
        check_rknn_available()

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        calibration_data,
        onnx_path,
        int8,
        opset,
        verbose,
        name=None,
        target=None,
        target_platform=None,
        rknn_config=None,
        rknn_build=None,
        verify=False,
        verify_input=None,
        verify_rtol=1e-3,
        verify_atol=1e-4,
        verify_min_cosine=0.9999,
        verify_max_normalized_rmse=0.02,
        **kwargs,
    ):
        from .rknn import (
            _publish_rknn_artifacts,
            _run_onnx_reference,
            compare_rknn_outputs,
            evaluate_rknn_metrics,
            export_rknn,
            export_rknn_with_simulator,
            resolve_rknn_target,
        )

        resolved_target = resolve_rknn_target(
            name=name,
            target=target,
            target_platform=target_platform,
        )
        rknn_metadata = dict(metadata or {})
        rknn_metadata.update(
            {
                "dynamic": False,
                "onnx_opset": int(opset),
                "rknn_target": resolved_target,
            }
        )
        logger.info("Step 2/2: Compiling RKNN for %s", resolved_target)

        if not verify:
            return export_rknn(
                onnx_path=onnx_path,
                output_path=output_path,
                target_platform=resolved_target,
                int8=int8,
                calibration_data=calibration_data,
                metadata=rknn_metadata,
                verbose=verbose,
                config=rknn_config,
                build=rknn_build,
            )

        if verify_input is None:
            generator = torch.Generator(device="cpu").manual_seed(0)
            verify_input = torch.rand(
                tuple(dummy.shape), generator=generator, dtype=torch.float32
            ).numpy()
        destination = Path(output_path)
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}.verify.",
            suffix=destination.suffix,
            dir=destination.parent,
            delete=False,
        ) as handle:
            staging_path = Path(handle.name)
        staging_path.unlink()

        try:
            result, simulated = export_rknn_with_simulator(
                onnx_path=onnx_path,
                output_path=str(staging_path),
                simulator_inputs=verify_input,
                target_platform=resolved_target,
                int8=int8,
                calibration_data=calibration_data,
                metadata=rknn_metadata,
                verbose=verbose,
                config=rknn_config,
                build=rknn_build,
            )
            reference = _run_onnx_reference(onnx_path, verify_input)
            metrics = compare_rknn_outputs(
                reference,
                simulated,
                rtol=float(verify_rtol),
                atol=float(verify_atol),
                raise_on_failure=False,
            )
            parity_passed, metrics = evaluate_rknn_metrics(
                metrics,
                min_cosine=float(verify_min_cosine),
                max_normalized_rmse=float(verify_max_normalized_rmse),
            )
            report = {
                "backend": "rknn-simulator",
                "target": resolved_target,
                "reference": "onnxruntime",
                "rtol": float(verify_rtol),
                "atol": float(verify_atol),
                "min_cosine": float(verify_min_cosine),
                "max_normalized_rmse": float(verify_max_normalized_rmse),
                "passed": parity_passed,
                "outputs": metrics,
            }
            staged_report = Path(f"{staging_path}.parity.json")
            temporary_report = staged_report.with_suffix(
                f"{staged_report.suffix}.tmp"
            )
            temporary_report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_report.replace(staged_report)
            if not parity_passed:
                report_path = Path(f"{destination}.failed.parity.json")
                staged_report.replace(report_path)
                failures = "; ".join(
                    f"output {item['index']}: "
                    f"max_abs={item['max_abs_error']:.6g}, "
                    f"cosine={item['cosine_similarity']:.6g}, "
                    f"normalized_rmse={item['normalized_rmse']:.6g}"
                    for item in metrics
                    if not item["accepted"]
                )
                raise AssertionError(
                    "RKNN simulator acceptance failed "
                    f"(min_cosine={verify_min_cosine}, "
                    f"max_normalized_rmse={verify_max_normalized_rmse}): "
                    f"{failures}. Metrics: {report_path}"
                )

            staged_metadata = Path(f"{staging_path}.metadata.json")
            report_path = Path(f"{destination}.parity.json")
            _publish_rknn_artifacts(
                staged_model=Path(result),
                staged_metadata=staged_metadata,
                staged_report=staged_report,
                destination=destination,
                remove_artifacts=(
                    Path(f"{destination}.failed.parity.json"),
                ),
            )
        finally:
            # Verification publishes the staged files only after all parity
            # gates pass. Any unsuccessful exit leaves a prior export intact.
            staging_path.unlink(missing_ok=True)
            Path(f"{staging_path}.metadata.json").unlink(missing_ok=True)
            Path(f"{staging_path}.parity.json").unlink(missing_ok=True)
            Path(f"{staging_path}.parity.json.tmp").unlink(missing_ok=True)
        logger.info("RKNN simulator parity passed: %s", report_path)
        return str(destination)


class _ClassifyLogitsWrapper(torch.nn.Module):
    """A classifier's full graph ending at its logits (layout ``classify_logits_v1``).

    Integer NPU exports of classifiers need no graph cut or host decode: the
    single output is the ``[B, nc]`` logits, and the backend applies the
    softmax of the shared classification path.
    """

    output_layout = "classify_logits_v1"
    output_names = ("logits",)
    output_windows: ClassVar[dict] = {}

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        return (logits,)


#: Dense tasks whose whole graph runs on Metis (layout ``graph_outputs_v1``), with
#: the output names of their ONNX export so both artifacts decode the same way.
_AXELERA_GRAPH_OUTPUTS = {
    ("fcn", "semantic"): ("semantic_logits",),
    ("quicksrnet", "restore"): ("restored",),
    ("midas", "depth"): ("depth",),
    ("deeplabv3", "semantic"): ("semantic_logits",),
    ("ppliteseg", "semantic"): ("semantic_logits",),
    ("pidnet", "semantic"): ("semantic_logits",),
    ("unet", "semantic"): ("semantic_logits",),
    # DexiNed: the fused edge probability map (sigmoid in the graph).
    ("dexined", "edge"): ("edges",),
    # EfficientDet's decoded export; the top-candidate decode (TopK, Gather,
    # Exp) lands in the compiled model's host postamble.
    ("efficientdet", "detect"): ("output",),
    # YOLOv1's decoded export (dense 7x7 grid, two boxes per cell); the decode
    # lands in the compiled model's host postamble.
    ("yolo1", "detect"): ("output",),
}


class _ShiftedDilatedConv(torch.nn.Module):
    """A dilated 3x3 convolution as per-tap 1x1 convolutions plus zero-filled shifts (exact).

    Tap ``(a, b)`` of a 3x3 kernel with dilation ``d`` reads the input shifted
    by ``((a - 1) d, (b - 1) d)``; a 1x1 convolution commutes with that shift,
    so the convolution is the sum over taps of ``shift(conv1x1(x, w[:, :, a, b]))``
    with zeros shifted in. Taps that only read padding are dropped. DeepLabV3's
    ASPP convolutions (dilation up to 36 at output stride 8) find no L1 tiling
    on the Metis AIPU as dilated convolutions; as 1x1 convolutions and shifts
    they compile.
    """

    def __init__(self, conv: torch.nn.Conv2d):
        super().__init__()
        out_channels, in_channels, kh, kw = conv.weight.shape
        if (kh, kw) != (3, 3) or conv.groups != 1 or tuple(conv.stride) != (1, 1):
            raise ValueError("Only stride-1, ungrouped 3x3 convolutions are rewritten.")
        if tuple(conv.dilation) != (conv.dilation[0],) * 2 or tuple(conv.padding) != tuple(conv.dilation):
            raise ValueError("Only 'same'-padded convolutions with equal dilations are rewritten.")
        self.dilation = int(conv.dilation[0])
        self.out_channels = int(out_channels)
        weight = conv.weight.detach().permute(2, 3, 0, 1).reshape(9 * out_channels, in_channels, 1, 1)
        self.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        self.bias = (
            torch.nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
            if conv.bias is not None
            else None
        )

    def forward(self, x):
        h, w, d = x.shape[-2], x.shape[-1], self.dilation
        out = None
        for k in range(9):
            dy, dx = (k // 3 - 1) * d, (k % 3 - 1) * d
            if abs(dy) >= h or abs(dx) >= w:
                continue  # this tap reads only zero padding
            tap = torch.nn.functional.conv2d(x, self.weight[k * self.out_channels : (k + 1) * self.out_channels])
            shifted = torch.nn.functional.pad(tap, (d, d, d, d))[:, :, d + dy : d + dy + h, d + dx : d + dx + w]
            out = shifted if out is None else out + shifted
        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        return out


def _aspp_shift(model: torch.nn.Module) -> int:
    """Rewrite every DeepLabV3 ASPP dilated convolution in ``model`` in place; returns the count."""
    from ..models.deeplabv3.nn import ASPPConv

    count = 0
    for module in model.modules():
        if isinstance(module, ASPPConv):
            module[0] = _ShiftedDilatedConv(module[0])
            count += 1
    return count


class _GraphOutputsWrapper(torch.nn.Module):
    """A dense model's full graph ending at its ONNX outputs (``graph_outputs_v1``).

    The device returns the same tensors as the family's ONNX export (super-
    resolved image, inverse depth, class logits, keypoint heatmaps), and the
    shared backend parsers turn them into results on the host.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        self.model = model
        self.output_names = tuple(output_names)

    def forward(self, x):
        output = self.model(x)
        if not isinstance(output, (list, tuple)):
            output = (output,)
        return tuple(output)[: len(self.output_names)]


#: Exact ONNX rewrites applied before the Axelera compile, per family (see
#: ``libreyolo.export.axelera.GRAPH_PASSES``). PicoDet's ESNet channel
#: shuffles become fixed 1x1 permutation convolutions: left as
#: Reshape/Transpose they are layout transforms inside the AIPU core, more
#: than the compiler accepts.
_AXELERA_GRAPH_PASSES: dict[str, tuple[str, ...]] = {
    "picodet": ("channel_shuffle_to_conv",),
    # EfficientNetV2-b1/b2/b3's stage-0 residual reads its skip through an
    # identity 1x1 conv (else the skip buffer is a 2nd compiled output; b0 has none).
    "efficientnetv2": ("stem_residual_skip_conv",),
    "deeplabv3": ("pooled_branch_split", "conv_reduction_split"),
    # VGG's fc6 (Gemm over 7x7x512) as row convolutions.
    "vgg": ("fc_rowconv",),
    # Spatial attention (channel mean/max, broadcast gating) in AIPU form; the
    # pyramid pooling's 1x1 branch as a broadcast add and its 2x2 and 4x4
    # upsampling to the 16x32 map (unequal factors up to 16) as factor-2 steps.
    "ppliteseg": (
        "channel_reduce_to_conv",
        "channel_max_tree",
        "broadcast_mul_expand",
        "conv_reduction_split",
        "resize_1x1_broadcast_add",
        "resize_corner_chain",
        "avgpool_staged",
    ),
    # PIDNet: the same attention rewrites; its DAPPM adds a 1x1 pooled branch
    # upsampled by 16 (a 32x32 depthwise window: too wide, F75).
    "pidnet": (
        "channel_reduce_to_conv",
        "broadcast_mul_expand",
        "resize_1x1_broadcast_add",
    ),
    # YOLOv1's [connected] layer (Gemm over 7x7x256 after its [local] layer).
    "yolo1": ("fc_rowconv",),
    # BiFPN weighted fusion and squeeze-excite averages in AIPU-runnable form.
    "efficientdet": (
        "float_div_to_mul",
        "stack_sum_to_add",
        "mean_hw_to_gap",
        "gap_staged",
    ),
}


def _identity_entry(t: torch.Tensor) -> torch.Tensor:
    """A 1x1 identity depthwise convolution: lets one tensor be a part output that also feeds a node, or a part
    input that fans out (the SDK's graph cleaner rejects both as plain tensors)."""
    return torch.nn.functional.conv2d(
        t, torch.ones(t.shape[1], 1, 1, 1, dtype=t.dtype, device=t.device), groups=t.shape[1]
    )


class _RRDBPart(torch.nn.Module):
    """One compiled part of Real-ESRGAN's RRDB network (x4) on Metis (``graph_outputs_v1``).

    The whole x4 graph's constants overflow the Metis DDR pool, so it runs as three compiled parts and one host
    step: ``head`` = ``conv_first`` + RRDB blocks ``[:split]`` -> (``skip``, ``mid``); ``tail`` = blocks
    ``[split:]`` + ``conv_body`` -> ``body``; the host adds ``skip + body``; ``upsample`` = the two nearest 2x
    upsampling convolutions, ``conv_hr`` and ``conv_last`` -> ``restored``. Exact: the parts compose to the
    network's own forward.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    _NAMES = {"head": ("skip", "mid"), "tail": ("body",), "upsample": ("restored",)}

    def __init__(self, net: torch.nn.Module, role: str, split: int):
        super().__init__()
        # ``holder``: the attribute name becomes the ONNX initializer prefix (holder.conv_first.weight, ...),
        # the names of the graphs that were compiled and measured.
        self.holder, self.role, self.split = net, role, int(split)
        self.output_names = self._NAMES[role]
        self.upscale = int(getattr(net, "scale", 4))
        # x2 (and x1) checkpoints pixel-unshuffle the input first (factor 2 / 4): an exact one-hot
        # stride-s convolution (output channel c*s*s + i*s + j reads input channel c at offset (i, j)).
        self.unshuffle = {4: 1, 2: 2, 1: 4}[self.upscale]
        if role == "head" and self.unshuffle > 1:
            s, c = self.unshuffle, int(net.conv_first.in_channels) // (self.unshuffle**2)
            weight = torch.zeros(c * s * s, c, s, s)
            for ch in range(c):
                for i in range(s):
                    for j in range(s):
                        weight[ch * s * s + i * s + j, ch, i, j] = 1.0
            self.register_buffer("unshuffle_weight", weight, persistent=False)

    def forward(self, x):
        net, F = self.holder, torch.nn.functional
        if self.role == "head":
            if self.unshuffle > 1:
                x = F.conv2d(x, self.unshuffle_weight.to(x.dtype), stride=self.unshuffle)
            skip = net.conv_first(x)
            return _identity_entry(skip), net.body[: self.split](skip)
        if self.role == "tail":
            return (net.conv_body(net.body[self.split :](_identity_entry(x))),)
        feat = net.lrelu(net.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest")))
        feat = net.lrelu(net.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return (net.conv_last(net.lrelu(net.conv_hr(feat))),)


class _NamedPart(torch.nn.Module):
    """A compiled part of a multi-part export: ``module`` with named outputs (``graph_outputs_v1``)."""

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}

    def __init__(self, module: torch.nn.Module, names):
        super().__init__()
        self.module, self.output_names = module.eval(), tuple(names)

    def forward(self, x):
        out = self.module(x)
        return (out,) if isinstance(out, torch.Tensor) else tuple(out)


class _FCOSTrunk(torch.nn.Module):
    """FCOS part 1 on Metis: the ResNet-FPN backbone's five levels (``lvl0`` .. ``lvl4``).

    P5 also feeds P6, so its output ends in an identity 1x1 convolution: the SDK's graph cleaner rejects a
    part output that also feeds another node.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    output_names = tuple(f"lvl{i}" for i in range(5))

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        outs = list(self.net.backbone(x).values())
        c = outs[2].shape[1]
        outs[2] = torch.nn.functional.conv2d(outs[2], torch.eye(c, dtype=outs[2].dtype, device=outs[2].device)[:, :, None, None])
        return tuple(outs)


def _fcos_level_boxes(shapes):
    """``[y, x, h, w]`` boxes for FPN level shapes on one canvas: level 0 on top, the rest side by side below,
    1-pixel zero gutters (a padded 3x3 convolution on the canvas then equals the per-level convolution)."""
    (h0, w0), rest = shapes[0], shapes[1:]
    boxes, x = [[0, 0, int(h0), int(w0)]], 0
    y = int(h0) + 1
    for h, w in rest:
        boxes.append([y, x, int(h), int(w)])
        x += int(w) + 1
    return boxes, [max(b[0] + b[2] for b in boxes), max(int(w0), x - 1)]


def _unfreeze_batchnorms(module: torch.nn.Module) -> None:
    """Replace FrozenBatchNorm2d by an equal eval-mode BatchNorm2d (folded into the convolution by the export)."""
    for name, child in list(module.named_children()):
        if type(child).__name__ == "FrozenBatchNorm2d":
            bn = torch.nn.BatchNorm2d(child.weight.numel(), eps=float(getattr(child, "eps", 1e-5)))
            with torch.no_grad():
                bn.weight.copy_(child.weight)
                bn.bias.copy_(child.bias)
                bn.running_mean.copy_(child.running_mean)
                bn.running_var.copy_(child.running_var)
            setattr(module, name, bn.eval())
        else:
            _unfreeze_batchnorms(child)


def _fcos_parts(net: torch.nn.Module, canvas: int):
    """Split FCOS (GroupNorm head) into Metis parts and host steps.

    The four head convolutions of the classification and regression towers run on Metis as one conv each
    (classification and regression stacked; grouped after the first), on all five levels packed onto one
    canvas; GroupNorm + ReLU per level runs on the host; one block-diagonal predictor conv gives
    ``cls | reg | ctrness``; the host decodes FCOSExportWrapper's tensor. Returns
    ``(trunk, layers, final, arrays, canvas_hw, groups, eps, n_cls)``.
    """
    from ..models.fcos.nn import FCOSExportWrapper

    _unfreeze_batchnorms(net.backbone)
    for p in net.parameters():
        p.requires_grad_(False)
    cls_h, reg_h = net.head.classification_head, net.head.regression_head
    convs = lambda h: [m for m in h.conv if isinstance(m, torch.nn.Conv2d)]  # noqa: E731
    norms = lambda h: [m for m in h.conv if isinstance(m, torch.nn.GroupNorm)]  # noqa: E731
    cc, rc, cg, rg = convs(cls_h), convs(reg_h), norms(cls_h), norms(reg_h)
    if not (len(cc) == len(rc) == len(cg) == len(rg) == 4):
        raise ValueError("FCOS Axelera export expects four conv + GroupNorm layers per head tower.")
    n_cls = int(cls_h.cls_logits.out_channels)
    width = int(cc[0].out_channels)

    def conv(cin, cout, groups=1):
        return torch.nn.Conv2d(cin, cout, 3, padding=1, groups=groups, bias=True)

    with torch.no_grad():
        layers = []
        for k in range(4):
            layer = conv(width if k == 0 else 2 * width, 2 * width, groups=1 if k == 0 else 2)
            layer.weight.copy_(torch.cat([cc[k].weight, rc[k].weight], 0))
            layer.bias.copy_(torch.cat([cc[k].bias, rc[k].bias], 0))
            layers.append(layer.eval())
        final = conv(2 * width, n_cls + 5)
        final.weight.zero_()
        final.weight[:n_cls, :width] = cls_h.cls_logits.weight
        final.weight[n_cls : n_cls + 4, width:] = reg_h.bbox_reg.weight
        final.weight[n_cls + 4 :, width:] = reg_h.bbox_ctrness.weight
        final.bias.copy_(torch.cat([cls_h.cls_logits.bias, reg_h.bbox_reg.bias, reg_h.bbox_ctrness.bias]))
        final.eval()
    arrays = {}
    for k in range(4):
        arrays[f"g{k + 1}"] = torch.cat([cg[k].weight, rg[k].weight]).numpy().astype("float32")
        arrays[f"b{k + 1}"] = torch.cat([cg[k].bias, rg[k].bias]).numpy().astype("float32")
    x0 = torch.zeros(1, 3, int(canvas), int(canvas))
    with torch.no_grad():
        feats = list(net.backbone(x0).values())
        anchors = net.anchor_generator.grid_anchors(x0, feats)
    anchors = anchors[0] if isinstance(anchors, (list, tuple)) else anchors
    boxes, canvas_hw = _fcos_level_boxes([tuple(t.shape[-2:]) for t in feats])
    import numpy as np

    arrays["boxes"] = np.asarray(boxes, dtype=np.int64)
    arrays["anchors"] = np.asarray(anchors, dtype=np.float32).reshape(-1, 4)
    arrays["class_indices"] = FCOSExportWrapper(net).class_indices.numpy().astype(np.int64)
    return _FCOSTrunk(net).eval(), layers, final, arrays, canvas_hw, int(cg[0].num_groups), float(cg[0].eps), n_cls, width


class _EfficientDetTrunk(torch.nn.Module):
    """EfficientDet part 1 on Metis: backbone, BiFPN, the box net, and the class net up to its last pointwise conv.

    Returns the five box regressions (``box_p3`` .. ``box_p7``) and the five class-tower outputs after the
    predictor's depthwise conv (``cls_dw_p3`` .. ``cls_dw_p7``). The predictor's pointwise conv runs as its own
    compiled part with min-max ranges (``class_head_part`` recipe): under the family's histogram ranges the class
    logits of the single-part build cannot represent logits above about 0 on p3, p4 and p7 (scores capped near 0.5).
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net
        levels = int(net.class_net.num_levels)
        self.output_names = tuple(
            [f"box_p{3 + i}" for i in range(levels)] + [f"cls_dw_p{3 + i}" for i in range(levels)]
        )

    def forward(self, x):
        net = self.net
        feats = net.fpn(net.backbone(x))
        boxes = net.box_net(feats)
        head = net.class_net
        dws = []
        for level in range(head.num_levels):
            t = feats[level]
            for conv, bn_levels in zip(head.conv_rep, head.bn_rep):
                t = head.act(bn_levels[level](conv(t)))
            dws.append(head.predict.conv_dw(t))
        return (*boxes, *dws)


class _EfficientDetCanvasDecode(torch.nn.Module):
    """Host ONNX tail of the EfficientDet parts export: per-level class logits cut from the packed canvas, then
    EfficientDetExportWrapper's decode (``output``)."""

    output_names = ("output",)

    def __init__(self, wrapper: torch.nn.Module, boxes):
        super().__init__()
        self.wrapper = wrapper
        self.boxes = [tuple(int(v) for v in b) for b in boxes]

    def forward(self, cls_logits, *box_levels):
        class_outputs = [cls_logits[:, :, y : y + h, x : x + w] for y, x, h, w in self.boxes]
        return self.wrapper.decode(class_outputs, list(box_levels))


class _RestoreBodyWrapper(torch.nn.Module):
    """Real-ESRGAN compact trunk up to its last convolution (``restore_shuffle_skip_v1``).

    The pixel shuffle and the nearest-upsampled input skip run on the host: a
    graph whose output depends directly on its input cannot be split into a
    host preamble and an accelerated core.
    """

    output_layout = "restore_shuffle_skip_v1"
    output_names = ("body",)
    output_windows: ClassVar[dict] = {}

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.upscale = int(model.upscale)

    def forward(self, x):
        out = x
        for layer in self.model.body:
            out = layer(out)
        return (out,)


class _SSDPartAWrapper(torch.nn.Module):
    """SSD300, part 1 of a two-part Axelera export (layout ``ssd_raw_heads_v1``).

    conv4_3's L2Norm (a per-pixel division by the channel norm) has no form on
    the Metis AIPU, and only the level-0 heads read its output. Part 1 runs the
    VGG trunk and returns conv4_3 as it is (``conv4_3``), then the five extra
    blocks and their box and class convolutions (levels 1-5). The host
    normalizes conv4_3 (``HOST_OPS["l2norm"]`` in the Axelera backend) and part
    2 (:class:`_SSDPartBWrapper`) runs the level-0 convolutions.
    """

    output_layout = "ssd_raw_heads_v1"
    grid_cell_offset = 0.5
    reg_max = None

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        from ..postprocess.yolonas_raw import SSD_STEPS, raw_head_output_names
        from ..utils.coco import COCO91_CATEGORY_IDS

        self.model = model
        self.strides = tuple(SSD_STEPS)
        #: The decoded heads' names, level by level (part 2 makes the first two).
        self.layout_names = raw_head_output_names(self.output_layout, self.strides)
        self.output_names = ["conv4_3", *self.layout_names[2:]]
        self.output_windows: dict = {}
        num_classes = int(model.num_classes)
        self.class_index_map = (
            list(COCO91_CATEGORY_IDS) if num_classes == 91 else list(range(1, num_classes))
        )
        self.l2norm_scale = [float(v) for v in model.backbone.scale_weight.detach().flatten()]

    def forward(self, x):
        backbone, head = self.model.backbone, self.model.head
        features = backbone.features(x)
        outputs = [features]
        level = features
        for index, block in enumerate(backbone.extra, start=1):
            level = block(level)
            outputs.append(head.regression_head.module_list[index](level))
            outputs.append(head.classification_head.module_list[index](level))
        return tuple(outputs)


class _SSDPartBWrapper(torch.nn.Module):
    """SSD300, part 2: the level-0 box and class convolutions on the host-normalized conv4_3.

    The graph input first goes through an identity 1x1 depthwise convolution:
    a graph input that fans out straight into two convolutions is cut into two
    AIPU functions by the compiler's lowering ("Expected 1 AIPU function, got
    2"), while one entry op followed by the fan-out compiles.
    """

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        self.model = model
        self.output_names = list(output_names)
        channels = int(model.head.regression_head.module_list[0].in_channels)
        self.entry = _ChannelScale(torch.ones(channels), as_conv=True)

    def forward(self, normalized):
        head = self.model.head
        features = self.entry(normalized)
        return (
            head.regression_head.module_list[0](features),
            head.classification_head.module_list[0](features),
        )


#: Transformer detectors exported as hybrids (support tier ``hybrid``): the
#: CNN backbone and the encoder's input projections compile for the Metis
#: AIPU, and the transformer encoder and decoder run as an FP32 ONNX graph on
#: the host CPU. The artifact ends in the family's ONNX outputs
#: (``graph_outputs_v1``), which the shared backend parser decodes.
#: Quantization scheme of a hybrid export's compiled neck part (see _export_hybrid_onnx).
_HYBRID_NECK_QUANTIZATION_SCHEME = "per_tensor_min_max"

#: Per-family neck-part scheme where per-tensor min-max loses accuracy: RT-DETRv2-r50's
#: neck part reaches a 0.68 lowest held-out cosine with min-max and 0.91 with histogram
#: ranges (a per-size recipe ``neck_quantization_scheme`` wins over both).
_HYBRID_NECK_QUANTIZATION_SCHEMES = {
    "rtdetrv2": "per_tensor_histogram",
    # RiO-DETRv2-n on 300 DOTA val tiles: histogram 0.4912 vs min-max 0.4742 mAP50-95 (FP32 0.5581).
    "riodetrv2": "per_tensor_histogram",
}

_AXELERA_HYBRID_OUTPUTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("rtdetr", "detect"): ("pred_logits", "pred_boxes"),
    ("rtdetrv2", "detect"): ("pred_logits", "pred_boxes"),
    ("rtdetrv2", "obb"): ("pred_logits", "pred_boxes"),
    ("riodetrv2", "obb"): ("pred_logits", "pred_boxes"),
    ("dfine", "detect"): ("pred_logits", "pred_boxes"),
    ("deim", "detect"): ("pred_logits", "pred_boxes"),
    ("deimv2", "detect"): ("pred_logits", "pred_boxes"),
    ("dfine", "segment"): ("pred_logits", "pred_boxes", "pred_masks"),
    ("rtdetrv4", "detect"): ("pred_logits", "pred_boxes"),
    ("detr", "detect"): ("pred_logits", "pred_boxes"),
    ("faster_rcnn", "detect"): ("boxes", "scores", "labels"),
    ("mask_rcnn", "detect"): ("boxes", "scores", "labels"),
    ("mask_rcnn", "segment"): ("boxes", "scores", "labels", "masks"),
    ("ppdoclayoutv3", "detect"): ("pred_logits", "pred_boxes", "order_votes"),
    ("deformable_detr", "detect"): ("pred_logits", "pred_boxes"),
    ("dinodetr", "detect"): ("pred_logits", "pred_boxes"),
}

#: Families whose sizes run whole on Metis unless their per-size recipe sets
#: ``hybrid_split`` (DeepLabV3-R101: its rewritten ASPP diverges on the device).
_AXELERA_RECIPE_HYBRID_OUTPUTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("deeplabv3", "semantic"): ("semantic_logits",),
    # VGG-19-BN: device drift compounds through conv block 5 (see _VGGFeaturesWrapper).
    ("vgg", "classify"): ("logits",),
    # RetinaNet-R50v2: the GroupNorm heads have no INT8 form; the ResNet-50 body runs on Metis.
    ("retinanet", "detect"): ("output",),
    # CenterNet-ResDCN18: the deformable up-path has no INT8 form; the ResNet-18 body runs on Metis.
    ("centernet", "detect"): ("output",),
    # HRNet-w32: the device drifts through stage 4's fusion modules; stem to stage 3 run on Metis.
    ("hrnet", "pose"): ("heatmaps",),
    # DEKR-w32: the same HRNet trunk; stem to stage 3 on Metis, stage 4 and the heads on the host.
    ("dekr", "pose"): ("heatmap_logits", "offsets"),
    # ZipDepth-bnpu: the GCNet softmax over positions has no AIPU form; the encoder before it runs on Metis.
    ("zipdepth", "depth"): ("depth",),
    # PIDNet-s: PAPPM's sums and the branch fusion after it break in INT8; layer 5 and before run on Metis.
    ("pidnet", "semantic"): ("semantic_logits",),
}

#: Hybrid families that only take the "backbone" split (no convolutional neck to compile).
_HYBRID_BACKBONE_ONLY = frozenset(
    {"detr", "faster_rcnn", "mask_rcnn", "ppdoclayoutv3", "deformable_detr", "dinodetr", "deeplabv3", "vgg", "retinanet", "centernet", "hrnet", "dekr", "zipdepth", "pidnet"}
)


def _unfrozen_batchnorms(module: torch.nn.Module) -> torch.nn.Module:
    """Replace every ``FrozenBatchNorm2d`` of ``module`` (in place) by an equal ``BatchNorm2d``.

    The frozen form exports as separate Mul/Add nodes the Axelera compiler
    does not fold; a plain batch norm in eval mode folds into its convolution
    during the ONNX export.
    """
    for name, child in list(module.named_children()):
        if type(child).__name__ == "FrozenBatchNorm2d":
            # DETR's FrozenBatchNorm2d hard-codes eps 1e-5 and has no ``eps`` attribute.
            norm = torch.nn.BatchNorm2d(
                int(child.weight.numel()), eps=float(getattr(child, "eps", 1e-5))
            )
            with torch.no_grad():
                norm.weight.copy_(child.weight)
                norm.bias.copy_(child.bias)
                norm.running_mean.copy_(child.running_mean)
                norm.running_var.copy_(child.running_var)
            setattr(module, name, norm.to(child.weight.device).eval())
        else:
            _unfrozen_batchnorms(child)
    return module


class _DETRBackboneWrapper(torch.nn.Module):
    """Part 1 of a hybrid DETR export: the CNN backbone and the encoder's input projections.

    Returns the projected feature maps by stride (``feat_s8``, ``feat_s16``,
    ``feat_s32`` for RT-DETR); every op is a convolution, a pooling or an
    activation, so the whole part runs on the Metis AIPU.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        strides = [int(v) for v in model.encoder.feat_strides]
        # D-FINE-seg n: the backbone returns one more (finer) stage than the encoder
        # takes; only the mask decoder reads it, so it leaves part 1 raw as "low_level".
        returned = getattr(model.backbone, "return_idx", None)
        self.low_level = returned is not None and len(returned) > len(model.encoder.input_proj)
        self.output_names = tuple(f"feat_s{stride}" for stride in strides) + (
            ("low_level",) if self.low_level else ()
        )

    def forward(self, x):
        features = self.model.backbone(x)
        low = None
        if self.low_level:
            low, features = features[0], features[1:]
        projections = self.model.encoder.input_proj
        out = tuple(projections[i](feature) for i, feature in enumerate(features))
        return out + (low,) if self.low_level else out


class _DocLayoutBackboneWrapper(torch.nn.Module):
    """Part 1 of a hybrid PP-DocLayoutV3 export: the HGNetV2 backbone and the encoder input projections.

    Returns the stride-4 backbone stage (``feat_s4``, the mask branch's input)
    and the projected stride-8/16/32 stages (``proj_s8`` ...); every op is a
    convolution, a pooling or an activation.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    hybrid_aipu = "HGNetV2 backbone and encoder input projections"
    hybrid_host = "hybrid encoder (AIFI attention, FPN/PAN, mask features), decoder, reading-order head (FP32 ONNX)"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        strides = [int(v) for v in model.cfg["feat_strides"]]
        self.output_names = ("feat_s4", *(f"proj_s{stride}" for stride in strides))

    def forward(self, x):
        features = self.model.backbone(x)
        projections = self.model.encoder_input_proj
        return (features[0], *(proj(f) for proj, f in zip(projections, features[1:])))


class _DocLayoutHostTailWrapper(torch.nn.Module):
    """Host step of a hybrid PP-DocLayoutV3 export: everything after the projections.

    Ends at the ONNX export's ``pred_logits``, ``pred_boxes`` and
    ``order_votes`` (PPDocLayoutV3ExportWrapper's outputs).
    """

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        backbone, model.backbone = model.backbone, None
        try:
            tail = copy.deepcopy(model).eval()
        finally:
            model.backbone = backbone
        tail.encoder_input_proj = None
        self.model = tail
        self.output_names = tuple(output_names)

    def forward(self, feat_s4, *proj_feats):
        from ..postprocess.ppdoclayoutv3 import reading_order_votes

        out = self.model.forward_features(feat_s4, list(proj_feats))
        values = {
            "pred_logits": out["pred_logits"],
            "pred_boxes": out["pred_boxes"],
            "order_votes": reading_order_votes(out["order_logits"]),
        }
        return tuple(values[name] for name in self.output_names)


class _RetinaNetBodyWrapper(torch.nn.Module):
    """Part 1 of a hybrid RetinaNet export (recipe ``hybrid_split``): the ResNet-50 body (C3, C4, C5).

    RetinaNet-R50v2's heads normalize every convolution with GroupNorm, which
    the INT8 toolchain has no form for; the body runs on Metis and the FPN,
    heads and box decode run on the host.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    output_names = ("c3", "c4", "c5")
    hybrid_aipu = "ResNet-50 body (C3, C4, C5)"
    hybrid_host = "FPN with P6/P7, GroupNorm heads and box decode (FP32 ONNX)"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return tuple(self.model.backbone.body(x).values())


class _RetinaNetHostTailWrapper(torch.nn.Module):
    """Host step of a hybrid RetinaNet export: FPN, heads, anchors and decode.

    Ends at the whole-graph ONNX export's ``output`` (decoded boxes plus
    contiguous scores), so the backend parses both exports the same way. The
    canvas size for the anchors is C3's stride (8) times its size.
    """

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        self.model = model
        self.output_names = tuple(output_names)
        self.keys = None

    def forward(self, c3, c4, c5):
        from collections import OrderedDict

        net = self.model
        keys = list(net.backbone.body.return_layers.values())
        features = list(net.backbone.fpn(OrderedDict(zip(keys, (c3, c4, c5)))).values())
        head_outputs = net.head(features)
        image_hw = (int(c3.shape[-2]) * 8, int(c3.shape[-1]) * 8)
        anchors, _ = net.anchor_generator(image_hw, features)
        decoded = torch.stack(
            [net.box_coder.decode_single(regression, anchors) for regression in head_outputs["bbox_regression"]]
        )
        scores = torch.sigmoid(head_outputs["cls_logits"])
        if net.num_classes == 91:
            scores = scores.index_select(2, net._coco91_source_indices)
        return (torch.cat((decoded, scores), dim=2),)


class _CenterNetBodyWrapper(torch.nn.Module):
    """Part 1 of a hybrid CenterNet export (recipe ``hybrid_split``): the body without deformable convolutions.

    ResDCN-18: the ResNet-18 body (stem to layer4), one output. DLA-34: the
    DLA base, its levels 2-5 (strides 4-32). The up-path's modulated deformable
    convolutions sample at learned offsets (grid_sample), which the INT8
    toolchain has no form for; the body runs on Metis and the deformable
    up-path, heads and peak decode run on the host.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    hybrid_host = "deformable up-path, heads and peak decode (FP32 ONNX)"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.dla = hasattr(model, "base")
        if self.dla:
            levels = range(model.first_level, len(model.base.channels))
            self.output_names = tuple(f"level{i}" for i in levels)
            self.hybrid_aipu = "DLA-34 base (levels 2-5)"
        else:
            self.output_names = ("feat",)
            self.hybrid_aipu = "ResNet-18 body (stem to layer4)"

    def forward(self, x):
        m = self.model
        if self.dla:
            return tuple(m.base(x)[m.first_level:])
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        return (m.layer4(m.layer3(m.layer2(m.layer1(x)))),)


class _CenterNetHostTailWrapper(torch.nn.Module):
    """Host step of a hybrid CenterNet export: deformable up-path, heads and peak decode.

    Ends at the whole-graph ONNX export's ``output`` (CenterNetExportWrapper:
    top-k peaks scaled to input pixels), so the backend parses both the same way.
    """

    def __init__(self, model: torch.nn.Module, output_names, topk: int = 100):
        super().__init__()
        from ..models.centernet.nn import set_portable_dcn

        self.model = model
        self.output_names = tuple(output_names)
        self.topk = topk
        set_portable_dcn(self.model, True)

    def forward(self, *feats):
        from ..postprocess.centernet import decode_centernet

        m = self.model
        if hasattr(m, "base"):  # DLA-34: DLAUp reads levels >= first_level only (and edits the list)
            features = m.dla_up([None] * m.first_level + list(feats))
            fused = [features[i].clone() for i in range(m.last_level - m.first_level)]
            m.ida_up(fused, 0, len(fused))
            x = fused[-1]
        else:
            x = m.deconv_layers(feats[0])
        out = {name: getattr(m, name)(x) for name in m.heads}
        return (
            decode_centernet(out["hm"], out["wh"], out["reg"], topk=self.topk)
            * out["hm"].new_tensor([4.0, 4.0, 4.0, 4.0, 1.0, 1.0]),
        )


def _hrnet_branches(transition, cfg, y):
    """HRNet's stage input: each branch through its transition (the new branch from the last one)."""
    return [transition[i](y[-1]) if transition[i] is not None else y[i] for i in range(cfg["NUM_BRANCHES"])]


class _HRNetBodyWrapper(torch.nn.Module):
    """Part 1 of a hybrid HRNet pose export (recipe ``hybrid_split``): stem to stage 3, all three branches.

    The whole graph compiles, but the device output drifts away from the SDK's
    own quantized model through stage 4's multi-branch fusion modules (device
    vs quantized cosine 0.983 after stage 3, 0.21 at the heatmaps); stage 3 and
    everything before it stay close, so stage 4 and the final layer run on the host.
    Each branch leaves through a weight-identity 1x1 convolution (exact): ending
    in the fuse layers' Add and ReLU, the quantizer would move that tail into a
    host postamble of the compiled part.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    hybrid_aipu = "HRNet stem, layer 1 and stages 2-3 (three resolution branches)"
    hybrid_host = "stage 4 (four-branch fusion) and the heatmap layer (FP32 ONNX)"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.output_names = tuple(f"branch{i}" for i in range(model.stage3_cfg["NUM_BRANCHES"]))
        channels = model._expanded_channels(model.stage3_cfg)
        self.identity = torch.nn.ModuleList()
        for c in channels:
            conv = torch.nn.Conv2d(c, c, 1, bias=False)
            with torch.no_grad():
                conv.weight.copy_(torch.eye(c).view(c, c, 1, 1))
            self.identity.append(conv)

    def _branches(self, x):
        m = self.model
        x = m.layer1(m.relu(m.bn2(m.conv2(m.relu(m.bn1(m.conv1(x)))))))
        y = m.stage2([t(x) if t is not None else x for t in m.transition1])
        return m.stage3(_hrnet_branches(m.transition2, m.stage3_cfg, y))

    def forward(self, x):
        return tuple(conv(t) for conv, t in zip(self.identity, self._branches(x)))


class _HRNetHostTailWrapper(torch.nn.Module):
    """Host step of a hybrid HRNet pose export: stage 4 and the heatmap layer.

    Ends at the whole-graph export's ``heatmaps``, so the backend parses both the same way.
    """

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        self.model = model
        self.output_names = tuple(output_names)

    def forward(self, *feats):
        m = self.model
        y = m.stage4(_hrnet_branches(m.transition3, m.stage4_cfg, list(feats)))
        return (m.final_layer(y[0]),)


def _dekr_stage_inputs(m, i, y):
    """DEKR's input to stage ``i + 2``: each branch through ``transition{i + 1}``."""
    transition = getattr(m, f"transition{i + 1}")
    return [
        transition[j](y[-1]) if transition[j] is not None else y[j]
        for j in range(int(m.stages_spec["NUM_BRANCHES"][i]))
    ]


class _DEKRBodyWrapper(torch.nn.Module):
    """Part 1 of a hybrid DEKR-w32 export (recipe ``hybrid_split``): the HRNet trunk's stem to stage 3.

    The whole 640x640 graph has no working Metis build (output paging fails
    the compile; without paging the device is lost), and HRNet's stage 4
    drifts on the device; stem to stage 3 run on Metis, each branch through a
    weight-identity 1x1 convolution (exact) so no host postamble is split off.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    hybrid_aipu = "HRNet-w32 stem, layer 1 and stages 2-3 (three resolution branches)"
    hybrid_host = "stage 4, the heatmap head and the per-keypoint offset heads (FP32 ONNX)"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        last = model.num_stages - 1  # the host runs the last stage
        channels = [int(c) for c in model.stages_spec["NUM_CHANNELS"][last - 1]]
        self.output_names = tuple(f"branch{i}" for i in range(len(channels)))
        self.identity = torch.nn.ModuleList()
        for c in channels:
            conv = torch.nn.Conv2d(c, c, 1, bias=False)
            with torch.no_grad():
                conv.weight.copy_(torch.eye(c).view(c, c, 1, 1))
            self.identity.append(conv)

    def forward(self, x):
        m = self.model
        x = m.layer1(m.relu(m.bn2(m.conv2(m.relu(m.bn1(m.conv1(x)))))))
        y = [x]
        for i in range(m.num_stages - 1):
            y = getattr(m, f"stage{i + 2}")(_dekr_stage_inputs(m, i, y))
        return tuple(conv(t) for conv, t in zip(self.identity, y))


class _DEKRHostTailWrapper(torch.nn.Module):
    """Host step of a hybrid DEKR export: stage 4 and the heads, ending at ``heatmap_logits`` and ``offsets``."""

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        self.model = model
        self.output_names = tuple(output_names)

    def forward(self, *feats):
        import torch.nn.functional as F

        m = self.model
        last = m.num_stages - 1
        y = getattr(m, f"stage{last + 2}")(_dekr_stage_inputs(m, last, list(feats)))
        h, w = y[0].shape[2], y[0].shape[3]
        features = torch.cat(
            [y[0]] + [F.interpolate(b, size=(h, w), mode="bilinear", align_corners=False) for b in y[1:]],
            dim=1,
        )
        heatmap = m.head_heatmap[1](m.head_heatmap[0](m.transition_heatmap(features)))
        offset_feature = m.transition_offset(features)
        k = m.offset_prekpt
        offsets = torch.cat(
            [
                m.offset_final_layer[j](m.offset_feature_layers[j](offset_feature[:, j * k : (j + 1) * k]))
                for j in range(m.num_joints)
            ],
            dim=1,
        )
        return heatmap, offsets


def _zipdepth_channel_attention(self, x):
    """ZipDepth ChannelAttention, exact: the channel mean as a global average pool and the gate repeated to H x W."""
    import torch.nn.functional as F

    gate = self.fc(F.adaptive_avg_pool2d(x, 1))
    return x * F.interpolate(gate, size=x.shape[-2:], mode="nearest")


def _zipdepth_strip_pooling(self, x):
    """ZipDepth StripPoolingAttention, exact: strip means as average pools, each repeated to H x W."""
    import torch.nn.functional as F

    h, w = x.shape[-2:]
    h_strip = F.interpolate(F.avg_pool2d(x, (1, w)), size=(h, w), mode="nearest")
    w_strip = F.interpolate(F.avg_pool2d(x, (h, 1)), size=(h, w), mode="nearest")
    return x * self.gate_conv(h_strip + w_strip)


#: Stage-3 blocks that run on the Metis in a hybrid ZipDepth export (measured on bnpu: device disparity
#: AbsRel vs FP32 0.030 with 4, 0.082 with 5, 0.098 with all 6 QARep blocks).
_ZIPDEPTH_STAGE3_METIS_BLOCKS = 4


def _zipdepth_host_from(stage3) -> int:
    """Index of the first stage-3 block that runs on the host.

    Stage 3 is six QARep blocks, a channel attention and the GCNet context
    block. The context block's softmax has no AIPU form; on the device the
    stage-3 output drifts away from FP32 through the last QARep blocks (a
    prefix bisect: cut after 4 blocks 0.030 disparity AbsRel, after 5 0.082,
    after 6 0.098), so the Metis part ends after the fourth.
    """
    blocks = list(stage3)
    attention = next(
        i for i, b in enumerate(blocks) if type(b).__name__ in ("ChannelAttention", "GlobalContextBlock")
    )
    return min(_ZIPDEPTH_STAGE3_METIS_BLOCKS, attention)


class _ZipDepthBodyWrapper(torch.nn.Module):
    """Part 1 of a hybrid ZipDepth export (recipe ``hybrid_split``): the encoder through stage 3's fourth block.

    The GCNet block weights the 576 stage-3 positions with a softmax that has
    no AIPU form within SDK 1.9; the encoder before it runs on Metis (its
    channel and strip attention as exact pooling forms), the context block,
    stage 4 and the decoder on the host. Each output (the decoder's skips and
    the stage-3 map) leaves through a weight-identity 1x1 conv (exact).
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    output_names = ("s_half", "s1", "s2", "s3_pre")
    hybrid_aipu = "ZipDepth encoder through stage 3's fourth QARep block"
    hybrid_host = (
        "the rest of stage 3 (two QARep blocks, channel attention, global context block), stage 4, "
        "SPPF, cross-scale exchange and the decoder (FP32 ONNX)"
    )

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        import types

        from ..models.zipdepth.nn import ChannelAttention, StripPoolingAttention

        self.model = copy.deepcopy(model)  # the attention rewrites stay on this copy
        for mod in self.model.modules():
            if isinstance(mod, ChannelAttention):
                mod.forward = types.MethodType(_zipdepth_channel_attention, mod)
            elif isinstance(mod, StripPoolingAttention):
                mod.forward = types.MethodType(_zipdepth_strip_pooling, mod)
        enc = self.model.encoder
        self.gcb_index = _zipdepth_host_from(enc.stage3)
        training = self.model.training
        self.model.eval()  # a private copy: one no-grad pass for the output widths, BatchNorm stats untouched
        with torch.no_grad():
            channels = [int(t.shape[1]) for t in self._features(torch.zeros(1, 3, 64, 64))]
        self.model.train(training)
        self.identity = torch.nn.ModuleList()
        for c in channels:
            conv = torch.nn.Conv2d(c, c, 1, bias=False)
            with torch.no_grad():
                conv.weight.copy_(torch.eye(c).view(c, c, 1, 1))
            self.identity.append(conv)

    def _features(self, x):
        m = self.model
        enc = m.encoder
        x = (x - m.mean) / m.std
        s_half = enc.stem_half(x)
        s1 = enc.stage1(enc.stem_quarter(s_half))
        s2 = enc.stage2(enc.down2(s1))
        s3 = enc.down3(s2)
        for block in list(enc.stage3)[: self.gcb_index]:
            s3 = block(s3)
        return s_half, s1, s2, s3

    def forward(self, x):
        return tuple(conv(t) for conv, t in zip(self.identity, self._features(x)))


class _ZipDepthHostTailWrapper(torch.nn.Module):
    """Host step of a hybrid ZipDepth export: the rest of stage 3 onward, ending at the whole-graph ``depth``."""

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        self.model = model
        self.output_names = tuple(output_names)
        self.gcb_index = _zipdepth_host_from(model.encoder.stage3)

    def forward(self, s_half, s1, s2, s3):
        m = self.model
        enc = m.encoder
        for block in list(enc.stage3)[self.gcb_index :]:
            s3 = block(s3)
        s4 = enc.spp(enc.stage4(enc.down4(s3)))
        s3, s4 = enc.cross_scale(s3, s4)
        h, w = s_half.shape[-2] * 2, s_half.shape[-1] * 2
        return (m.decoder(s_half, [s1, s2, s3, s4], (h, w)),)


def _ones_conv(cin: int, cout: int, value: float = 1.0) -> torch.nn.Conv2d:
    conv = torch.nn.Conv2d(cin, cout, 1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(value)
    return conv


def _pidnet_pag_forward(self, x, y):
    """PIDNet PagFM, exact INT8-friendly form: ``sigmoid(-z) * x + sigmoid(z) * y``.

    ``1 - sigmoid(z)`` is ``sigmoid(-z)``, so the blend needs neither a
    constant minus a tensor (the quantized model flips its sign: cosine -0.84
    vs FP32) nor ``x + sim * (y - x)`` (its two near-cancelling addends lose
    the result: 0.80). The channel sum and its negation are 1x1 convolutions
    of ones and minus ones; one-channel gates are repeated to every channel by
    a 1x1 convolution of ones. With this form the fusion keeps cosine 0.996.
    """
    import torch.nn.functional as F

    size = x.size()
    if self.after_relu:
        y = self.relu(y)
        x = self.relu(x)
    y_q = F.interpolate(self.f_y(y), size=[size[2], size[3]], mode="bilinear", align_corners=False)
    prod = self.f_x(x) * y_q
    if self.with_channel:
        z = self.up(prod)
        gate, gate_neg = torch.sigmoid(z), torch.sigmoid(self._int8_negate(z))
    else:
        gate = self._int8_expand(torch.sigmoid(self._int8_sum(prod)))
        gate_neg = self._int8_expand(torch.sigmoid(self._int8_sum_neg(prod)))
    y = F.interpolate(y, size=[size[2], size[3]], mode="bilinear", align_corners=False)
    return gate_neg * x + gate * y


def _ppliteseg_uafm_forward(self, x, skip):
    """PP-LiteSeg UAFM, exact INT8-friendly form: ``x * sigmoid(z) + skip * sigmoid(-z)``.

    The native blend ``x * a + skip * (1 - a)`` with ``a = sigmoid(z)`` needs a
    constant minus a tensor, which the quantized model gets wrong (the three
    UAFM outputs of PP-LiteSeg-t50 keep cosine 0.59-0.78 vs FP32 and the
    logits' argmax agrees on 56% of pixels, as on the device). ``1 - sigmoid(z)``
    is ``sigmoid(-z)``, the negation is a 1x1 convolution of -1, and the UAFM
    outputs keep 0.99 (argmax agreement 95%).
    """
    x = self.up_x(x)
    skip = self.proj_skip(skip)
    atten = torch.cat([*self._avg_max_spatial_reduce(x), *self._avg_max_spatial_reduce(skip)], dim=1)
    z = self.conv_atten(atten)
    return self.conv_out(x * torch.sigmoid(z) + skip * torch.sigmoid(self._int8_negate(z)))


def _ppliteseg_int8_blend(model: torch.nn.Module) -> int:
    """Rewrite every PP-LiteSeg UAFM of ``model`` (a copy the caller owns) to the exact blend above; returns how many."""
    import types

    count = 0
    for module in model.modules():
        if type(module).__name__ != "UAFM":
            continue
        negate = torch.nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            negate.weight.fill_(-1.0)
        module._int8_negate = negate
        module.forward = types.MethodType(_ppliteseg_uafm_forward, module)
        count += 1
    return count


class _PIDNetBodyWrapper(torch.nn.Module):
    """Part 1 of a hybrid PIDNet export (recipe ``hybrid_split``): the three branches through layer 5.

    INT8 breaks PIDNet twice: in its PagFM fusions (exact rewrite above) and,
    at the end, in the PAPPM pooling module's upsample-and-add sums (cosine
    0.3-0.5 while their inputs keep 0.994 or better) and the fusion after it.
    So the stem, the P, I and D branches and both PagFM fusions (rewritten, on
    this wrapper's private copy) run on Metis; PAPPM, the branch fusion and the
    segmentation head run on the host (about 13% of the MACs at 1024x1024).
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    output_names = ("i_layer5", "p_layer5", "d_layer5")
    hybrid_aipu = "PIDNet stem and the P, I and D branches through layer 5 (both PagFM fusions)"
    hybrid_host = "PAPPM pooling, the branch fusion and the segmentation head (FP32 ONNX)"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        import types

        self.model = copy.deepcopy(model)
        for mod in self.model.modules():
            if type(mod).__name__ != "PagFM":
                continue
            if mod.with_channel:
                c = [m for m in mod.up.modules() if isinstance(m, torch.nn.Conv2d)][-1].out_channels
                neg = torch.nn.Conv2d(c, c, 1, groups=c, bias=False)
                with torch.no_grad():
                    neg.weight.fill_(-1.0)
                mod._int8_negate = neg
            else:
                c = [m for m in mod.f_x.modules() if isinstance(m, torch.nn.Conv2d)][-1].out_channels
                cx = [m for m in mod.f_x.modules() if isinstance(m, torch.nn.Conv2d)][0].in_channels
                mod._int8_sum = _ones_conv(c, 1)
                mod._int8_sum_neg = _ones_conv(c, 1, -1.0)
                mod._int8_expand = _ones_conv(1, cx)
            mod.forward = types.MethodType(_pidnet_pag_forward, mod)
        m = self.model
        widths = [_conv_out(m.layer5), _conv_out(m.layer5_), _conv_out(m.layer5_d)]
        self.identity = torch.nn.ModuleList()
        for c in widths:
            conv = torch.nn.Conv2d(c, c, 1, bias=False)
            with torch.no_grad():
                conv.weight.copy_(torch.eye(c).view(c, c, 1, 1))
            self.identity.append(conv)

    def forward(self, x):
        import torch.nn.functional as F

        m = self.model
        x = (x - m.pixel_mean.to(dtype=x.dtype)) / m.pixel_std.to(dtype=x.dtype)
        ho, wo = x.shape[-2] // 8, x.shape[-1] // 8
        x = m.layer1(m.conv1(x))
        x = m.relu(m.layer2(m.relu(x)))
        x_ = m.layer3_(x)
        x_d = m.layer3_d(x)
        x = m.relu(m.layer3(x))
        x_ = m.pag3(x_, m.compression3(x))
        x_d = x_d + F.interpolate(m.diff3(x), size=[ho, wo], mode="bilinear", align_corners=False)
        x = m.relu(m.layer4(x))
        x_ = m.layer4_(m.relu(x_))
        x_d = m.layer4_d(m.relu(x_d))
        x_ = m.pag4(x_, m.compression4(x))
        x_d = x_d + F.interpolate(m.diff4(x), size=[ho, wo], mode="bilinear", align_corners=False)
        feats = (m.layer5(x), m.layer5_(m.relu(x_)), m.layer5_d(m.relu(x_d)))
        return tuple(conv(t) for conv, t in zip(self.identity, feats))


def _conv_out(module: torch.nn.Module) -> int:
    """Output channels of a block: its last convolution's (or last BatchNorm's) width."""
    width = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            width = m.out_channels
        elif isinstance(m, torch.nn.BatchNorm2d):
            width = m.num_features
    return int(width)


class _PIDNetHostTailWrapper(torch.nn.Module):
    """Host step of a hybrid PIDNet export: PAPPM, the branch fusion and the head, ending at ``semantic_logits``."""

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        self.model = model
        self.output_names = tuple(output_names)

    def forward(self, x5, p, d):
        import torch.nn.functional as F

        m = self.model
        x = F.interpolate(m.spp(x5), size=list(p.shape[-2:]), mode="bilinear", align_corners=False)
        return (m.final_layer(m.dfm(p, x, d)),)


class _DeepLabBackboneWrapper(torch.nn.Module):
    """Part 1 of a hybrid DeepLabV3 export (recipe ``hybrid_split``): the ResNet backbone through layer4.

    The whole backbone lowers correctly on Metis (device vs CPU-quantized
    cosine 0.995 on R101), while the rewritten ASPP diverges there (0.53), so
    the ASPP and the head run on the host instead.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    output_names = ("feat",)
    hybrid_aipu = "ResNet backbone (through layer4)"
    hybrid_host = "ASPP, classifier head and logit upsampling (FP32 ONNX)"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return (self.model.backbone(x)["out"],)


class _DeepLabHostTailWrapper(torch.nn.Module):
    """Host step of a hybrid DeepLabV3 export: the ASPP (plain dilated convolutions), head and upsampling.

    Ends at the whole-graph export's ``semantic_logits`` (class logits at the
    input canvas), so the backend parses both exports the same way.
    """

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        self.classifier = model.classifier
        self.output_names = tuple(output_names)
        self.upsample_to = None  # the input canvas (H, W), set by the exporter

    def forward(self, feat):
        logits = self.classifier(feat)
        return (
            torch.nn.functional.interpolate(
                logits, size=self.upsample_to, mode="bilinear", align_corners=False
            ),
        )


def _vgg_pool4_end(model: torch.nn.Module) -> int:
    """Index in ``model.features`` just past the fourth max pool (the hybrid VGG split)."""
    pools = [i for i, m in enumerate(model.features) if isinstance(m, torch.nn.MaxPool2d)]
    return pools[3] + 1


class _VGGFeaturesWrapper(torch.nn.Module):
    """Part 1 of a hybrid VGG export (recipe ``hybrid_split``): conv blocks 1-4, through pool4.

    On VGG-19-BN the device's deviation from the CPU-quantized model grows
    through conv block 5 (prefix cosine 0.984 at pool4, 0.785 after conv16)
    without one diverging operator, so block 5 runs on the host.
    """

    output_layout = "classify_logits_v1"
    output_windows: ClassVar[dict] = {}
    output_names = ("pool4",)
    hybrid_aipu = "convolution blocks 1-4 (through pool4)"
    hybrid_host = "convolution block 5, average pool and classifier (FP32 ONNX)"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.features = model.features[: _vgg_pool4_end(model)]

    def forward(self, x):
        return (self.features(x),)


class _VGGHostTailWrapper(torch.nn.Module):
    """Host step of a hybrid VGG export: conv block 5, average pool and classifier.

    Ends at the whole-graph export's ``logits``, so the backend parses both
    exports the same way.
    """

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        self.features = model.features[_vgg_pool4_end(model) :]
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        self.output_names = tuple(output_names)

    def forward(self, feat):
        pooled = self.avgpool(self.features(feat))
        return (self.classifier(torch.flatten(pooled, 1)),)


class _VanillaDETRBackboneWrapper(torch.nn.Module):
    """Part 1 of a hybrid export of vanilla DETR: the ResNet backbone and the input projection."""

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    output_names = ("feat",)
    hybrid_aipu = "ResNet backbone and input projection"
    hybrid_host = "positions, transformer encoder and decoder, heads (FP32 ONNX)"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        feature = self.model.backbone[0].body(x)["0"]
        return (self.model.input_proj(feature),)


class _VanillaDETRHostTailWrapper(torch.nn.Module):
    """Host step of a hybrid export of vanilla DETR: positions, transformer and heads.

    LibreYOLO feeds DETR one fixed square without padding, so the padding mask
    is all False and the sine positions depend only on the feature size.
    """

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        self.model = model
        self.output_names = tuple(output_names)

    def forward(self, feature):
        from ..models.detr.nn import NestedTensor

        mask = torch.zeros(
            (feature.shape[0], feature.shape[-2], feature.shape[-1]),
            dtype=torch.bool,
            device=feature.device,
        )
        positions = self.model.backbone[1](NestedTensor(feature, mask)).to(feature.dtype)
        hidden = self.model.transformer(
            feature, mask, self.model.query_embed.weight, positions
        )[0]
        logits = self.model.class_embed(hidden)[-1]
        boxes = self.model.bbox_embed(hidden).sigmoid()[-1]
        return logits, boxes


class _RCNNBackboneWrapper(torch.nn.Module):
    """Part 1 of a hybrid Faster R-CNN export: input normalization, backbone and FPN.

    Takes the backend's fixed canvas (RGB in [0, 1], already the model's
    ``min_size``, so the in-graph resize is the identity) and returns the FPN
    levels ``fpn0`` (finest) ... ``fpnN``. The extra ``pool`` level is a
    stride-2 subsample of the coarsest one and runs on the host.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    hybrid_aipu = "input normalization, backbone and FPN"
    hybrid_host = "RPN, RoIAlign, box head and NMS (FP32 ONNX)"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        mean = torch.tensor(model.transform.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(model.transform.image_std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("scale", 1.0 / std, persistent=False)
        self.register_buffer("shift", -mean / std, persistent=False)
        levels = len(model.backbone.fpn.inner_blocks)
        self.output_names = tuple(f"fpn{i}" for i in range(levels))

    def forward(self, x):
        levels = self.model.backbone(x * self.scale + self.shift)
        # The extra "pool" level is dropped here (the ONNX export prunes it) and
        # recomputed on the host: a strided 1x1 subsample has no safe AIPU form (F80).
        return tuple(value for key, value in levels.items() if key != "pool")


class _RCNNHostTailWrapper(torch.nn.Module):
    """Host step of a hybrid Faster R-CNN export: pool level, RPN, RoI heads, NMS, COCO labels.

    Ends at the ONNX export's ``boxes``, ``scores``, ``labels`` in canvas
    pixels, which the backend's fixed-shape parser maps back to the image.
    """

    def __init__(self, model: torch.nn.Module, output_names, canvas=(800, 800)):
        super().__init__()
        from ..models.faster_rcnn.nn import FasterRCNNExportWrapper

        self.model = model
        self.output_names = tuple(output_names)
        self.canvas = tuple(int(v) for v in canvas)
        self.register_buffer(
            "label_map", FasterRCNNExportWrapper(model).label_map.clone(), persistent=False
        )

    def forward(self, *levels):
        from collections import OrderedDict

        from torchvision.models.detection.image_list import ImageList

        features = OrderedDict((str(i), level) for i, level in enumerate(levels))
        features["pool"] = torch.nn.functional.max_pool2d(levels[-1], 1, 2, 0)
        h, w = self.canvas
        images = ImageList(levels[0].new_zeros((levels[0].shape[0], 3, h, w)), [(h, w)])
        proposals, _ = self.model.rpn(images, features)
        detections, _ = self.model.roi_heads(features, proposals, images.image_sizes)
        detection = self.model.transform.postprocess(detections, images.image_sizes, [(h, w)])[0]
        labels = self.label_map[detection["labels"]]
        keep = labels >= 0
        return detection["boxes"][keep], detection["scores"][keep], labels[keep]


class _MaskRCNNHostTailWrapper(_RCNNHostTailWrapper):
    """Host step of a hybrid Mask R-CNN export: the Faster R-CNN host step plus the mask head.

    The mask head (its RoIAlign, four convolutions and the mask predictor) runs
    on the host with the box head, and the soft masks are pasted onto the
    canvas, so ``masks`` is ``(N, 1, H, W)`` in canvas pixels; the backend's
    fixed-shape parser resizes them to the image like the boxes.
    """

    def forward(self, *levels):
        from collections import OrderedDict

        from torchvision.models.detection.image_list import ImageList

        features = OrderedDict((str(i), level) for i, level in enumerate(levels))
        features["pool"] = torch.nn.functional.max_pool2d(levels[-1], 1, 2, 0)
        h, w = self.canvas
        images = ImageList(levels[0].new_zeros((levels[0].shape[0], 3, h, w)), [(h, w)])
        proposals, _ = self.model.rpn(images, features)
        detections, _ = self.model.roi_heads(features, proposals, images.image_sizes)
        detection = self.model.transform.postprocess(detections, images.image_sizes, [(h, w)])[0]
        labels = self.label_map[detection["labels"]]
        keep = labels >= 0
        outputs = {
            "boxes": detection["boxes"][keep],
            "scores": detection["scores"][keep],
            "labels": labels[keep],
        }
        if "masks" in self.output_names:
            outputs["masks"] = detection["masks"][keep]
        return tuple(outputs[name] for name in self.output_names)


class _DeformableDETRBackboneWrapper(torch.nn.Module):
    """Part 1 of a hybrid Deformable DETR export: the ResNet body (C3, C4, C5).

    The input projections that follow normalize with GroupNorm, which the Metis
    has no form for, so they run on the host with the transformer.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    hybrid_aipu = "ResNet backbone"
    hybrid_host = "input projections (GroupNorm), positions, deformable transformer, heads (FP32 ONNX)"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.output_names = tuple(f"feat{i}" for i in range(len(model.backbone.strides)))

    def forward(self, x):
        features = self.model.backbone[0].body(x)
        return tuple(features[key] for key in sorted(features))


class _DeformableDETRHostTailWrapper(torch.nn.Module):
    """Host step of a hybrid Deformable DETR export: everything after the ResNet body.

    LibreYOLO feeds one fixed square without padding, so every padding mask is
    all False; the rest follows the model's own forward.
    """

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        self.model = model
        self.output_names = tuple(output_names)

    def forward(self, *levels):
        from ..models.deformable_detr.common import NestedTensor, inverse_sigmoid

        model = self.model
        masks = [torch.zeros((f.shape[0], f.shape[-2], f.shape[-1]), dtype=torch.bool, device=f.device) for f in levels]
        positions = [model.backbone[1](NestedTensor(f, m)).to(f.dtype) for f, m in zip(levels, masks)]
        srcs = [model.input_proj[i](f) for i, f in enumerate(levels)]
        for level in range(len(levels), model.num_feature_levels):
            src = model.input_proj[level](levels[-1] if level == len(levels) else srcs[-1])
            mask = torch.zeros((src.shape[0], src.shape[-2], src.shape[-1]), dtype=torch.bool, device=src.device)
            positions.append(model.backbone[1](NestedTensor(src, mask)).to(src.dtype))
            srcs.append(src)
            masks.append(mask)
        query_embeds = None if model.two_stage else model.query_embed.weight
        hidden, init_reference, inter_references, _, _ = model.transformer(srcs, masks, positions, query_embeds)
        level = hidden.shape[0] - 1
        reference = init_reference if level == 0 else inter_references[level - 1]
        reference = inverse_sigmoid(reference)
        logits = model.class_embed[level](hidden[level])
        delta = model.bbox_embed[level](hidden[level])
        if reference.shape[-1] == 4:
            delta = delta + reference
        else:
            delta = torch.cat([delta[..., :2] + reference, delta[..., 2:]], dim=-1)
        return logits, delta.sigmoid()


class _DINODETRBackboneWrapper(_DeformableDETRBackboneWrapper):
    """Part 1 of a hybrid DINO-DETR export: the ResNet body, as Deformable DETR.

    DINO-DETR projects its levels with GroupNorm too, so the cut is the same.
    """

    def __init__(self, model: torch.nn.Module):
        torch.nn.Module.__init__(self)
        self.model = model
        self.output_names = tuple(f"feat{i}" for i in range(len(model.backbone.num_channels)))


class _DINODETRHostTailWrapper(_DeformableDETRHostTailWrapper):
    """Host step of a hybrid DINO-DETR export: everything after the ResNet body.

    The padding masks are all False at LibreYOLO's fixed square; the
    transformer call and the last decoder layer's heads follow the model's own
    forward (no denoising queries at inference).
    """

    def forward(self, *levels):
        from ..models.dinodetr.common import NestedTensor, inverse_sigmoid

        model = self.model

        def no_padding(f):
            return torch.zeros((f.shape[0], f.shape[-2], f.shape[-1]), dtype=torch.bool, device=f.device)

        masks = [no_padding(f) for f in levels]
        positions = [model.backbone[1](NestedTensor(f, m)).to(f.dtype) for f, m in zip(levels, masks)]
        srcs = [model.input_proj[i](f) for i, f in enumerate(levels)]
        for level in range(len(levels), model.num_feature_levels):
            src = model.input_proj[level](levels[-1] if level == len(levels) else srcs[-1])
            mask = no_padding(src)
            positions.append(model.backbone[1](NestedTensor(src, mask)).to(src.dtype))
            srcs.append(src)
            masks.append(mask)
        hidden, references, _, _, _ = model.transformer(srcs, masks, None, positions, None, None)
        last = len(hidden) - 1
        boxes = (model.bbox_embed[last](hidden[last]) + inverse_sigmoid(references[last])).sigmoid()
        return model.class_embed[last](hidden[last]), boxes


#: Backbone-only hybrid families: (part-1 wrapper, host tail wrapper).
_HYBRID_BACKBONE_WRAPPERS = {
    "deformable_detr": ("_DeformableDETRBackboneWrapper", "_DeformableDETRHostTailWrapper"),
    "dinodetr": ("_DINODETRBackboneWrapper", "_DINODETRHostTailWrapper"),
    "detr": ("_VanillaDETRBackboneWrapper", "_VanillaDETRHostTailWrapper"),
    "faster_rcnn": ("_RCNNBackboneWrapper", "_RCNNHostTailWrapper"),
    "mask_rcnn": ("_RCNNBackboneWrapper", "_MaskRCNNHostTailWrapper"),
    "ppdoclayoutv3": ("_DocLayoutBackboneWrapper", "_DocLayoutHostTailWrapper"),
    "deeplabv3": ("_DeepLabBackboneWrapper", "_DeepLabHostTailWrapper"),
    "vgg": ("_VGGFeaturesWrapper", "_VGGHostTailWrapper"),
    "retinanet": ("_RetinaNetBodyWrapper", "_RetinaNetHostTailWrapper"),
    "centernet": ("_CenterNetBodyWrapper", "_CenterNetHostTailWrapper"),
    "hrnet": ("_HRNetBodyWrapper", "_HRNetHostTailWrapper"),
    "dekr": ("_DEKRBodyWrapper", "_DEKRHostTailWrapper"),
    "zipdepth": ("_ZipDepthBodyWrapper", "_ZipDepthHostTailWrapper"),
    "pidnet": ("_PIDNetBodyWrapper", "_PIDNetHostTailWrapper"),
}


class _DETRHostTailWrapper(torch.nn.Module):
    """Part 2 of a hybrid DETR export, on the host: encoder and decoder after the input projections.

    A deep copy of the model whose input projections are identities (and
    which has no backbone) runs its own encoder and decoder forward, so the
    host graph is the native model's code from the projected features to
    ``pred_logits`` and ``pred_boxes``. The caller's model is not touched.
    """

    def __init__(self, model: torch.nn.Module, output_names):
        super().__init__()
        backbone, model.backbone = model.backbone, None
        try:
            tail = copy.deepcopy(model).eval()
        finally:
            model.backbone = backbone
        tail.encoder.input_proj = torch.nn.ModuleList(
            torch.nn.Identity() for _ in tail.encoder.input_proj
        )
        self.model = tail
        self.output_names = tuple(output_names)

    def forward(self, *features):
        outputs = self.model.decoder(self.model.encoder(list(features)))
        return tuple(outputs[name] for name in self.output_names)


def _identity_depthwise(channels: int, stride: int = 1, scale: float = 1.0) -> torch.nn.Conv2d:
    """A 1x1 convolution with constant weights: a per-channel scale, optionally with a stride-``stride`` subsample.

    Without a stride it is depthwise. With one it is a dense convolution with
    the scale on its diagonal (zeros elsewhere, the same result): the Metis
    lowering rejects a strided 1x1 depthwise convolution on a stride-8 canvas
    (a DWPU command length over its limit), while the dense form compiles.
    """
    if stride == 1:
        conv = torch.nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        with torch.no_grad():
            conv.weight.fill_(float(scale))
        return conv
    conv = torch.nn.Conv2d(channels, channels, 1, stride=stride, bias=False)
    with torch.no_grad():
        conv.weight.zero_()
        index = torch.arange(channels)
        conv.weight[index, index, 0, 0] = float(scale)
    return conv


class _DETREncoderPackWrapper(torch.nn.Module):
    """Host step of a hybrid DETR export with the neck on Metis: the transformer encoder, then one canvas.

    Runs the encoder's attention layers (AIFI) on their feature level, as the
    native encoder does, and stacks the levels on channels at the finest level's
    size (coarser levels nearest-upsampled), so the neck part has the single
    input a compiled part takes. The neck subsamples them back exactly. Level
    ``i`` is multiplied by ``scales[i]`` so that every level spans the same
    range in the canvas's one INT8 scale; the neck divides it out again.
    """

    def __init__(self, encoder: torch.nn.Module, scales=None):
        super().__init__()
        self.encoder = encoder
        self.scales = [1.0] * len(encoder.input_proj) if scales is None else [float(v) for v in scales]

    def forward(self, *features):
        encoder = self.encoder
        features = list(features)
        for i, level in enumerate(encoder.use_encoder_idx):
            h, w = features[level].shape[2:]
            flat = features[level].flatten(2).permute(0, 2, 1)
            if encoder.eval_spatial_size is None:
                pos_embed = encoder.build_2d_sincos_position_embedding(
                    w, h, encoder.hidden_dim, encoder.pe_temperature
                ).to(flat.device)
            else:
                pos_embed = encoder._get_pos_embed(level, h, w, flat)
            memory = encoder.encoder[i](flat, pos_embed=pos_embed)
            features[level] = (
                memory.permute(0, 2, 1).reshape(-1, encoder.hidden_dim, h, w).contiguous()
            )
        size = features[0].shape[2:]
        levels = [features[0]] + [
            torch.nn.functional.interpolate(f, size=size, mode="nearest") for f in features[1:]
        ]
        levels = [f * k if k != 1.0 else f for f, k in zip(levels, self.scales)]
        return (torch.cat(levels, dim=1),)


class _DETRNeckWrapper(torch.nn.Module):
    """Compiled neck part of a hybrid DETR export: canvas -> fusion neck -> decoder input projections.

    The canvas goes through one identity depthwise conv first (a graph input
    that fans out straight into several ops is split into two AIPU functions),
    then each level is sliced off its channels and subsampled back to its own
    stride by a strided 1x1 conv (dense, diagonal) whose constant weight undoes
    the level's canvas scale. The encoder copy runs with its attention layers and input
    projections off, i.e. the convolutional fusion neck only; the decoder's
    1x1 input projections (conv + batch norm) follow.
    """

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}

    def __init__(
        self, encoder: torch.nn.Module, decoder_projections, channels: int, strides, scales=None
    ):
        super().__init__()
        self.encoder = encoder
        self.projections = decoder_projections
        self.channels = int(channels)
        strides = [int(v) for v in strides]
        levels = len(strides)
        scales = [1.0] * levels if scales is None else [float(v) for v in scales]
        self.entry = _identity_depthwise(self.channels * levels)
        self.picks = torch.nn.ModuleList(
            torch.nn.Identity()
            if i == 0 and scales[0] == 1.0
            else _identity_depthwise(self.channels, strides[i] // strides[0], 1.0 / scales[i])
            for i in range(levels)
        )
        self.output_names = tuple(f"memory_s{stride}" for stride in strides)

    def forward(self, canvas):
        x = self.entry(canvas)
        c = self.channels
        features = [pick(x[:, c * i : c * (i + 1)]) for i, pick in enumerate(self.picks)]
        fused = self.encoder(features)
        return tuple(self.projections[i](f) for i, f in enumerate(fused))


def _has_attention_encoder(model: torch.nn.Module) -> bool:
    """Whether a DETR-style model's encoder runs attention layers (RT-DETR's AIFI) or is all convolutions.

    DEIMv2's small sizes (atto, femto, pico) use a LiteEncoder: projections,
    pooling and CSP blocks only, so the whole encoder can run on the Metis.
    """
    encoder = model.encoder
    return hasattr(encoder, "use_encoder_idx") or hasattr(encoder, "num_encoder_layers")


class _GapFusionWithIdentity(torch.nn.Module):
    """DEIMv2 LiteEncoder's ``x + GAP(x) -> conv`` with a 1x1 identity conv on the pooled branch.

    A global pool feeding an Add that feeds a conv fails the Metis lowering
    ("Expecting all blocks to have the same iter_values", F77); an identity
    1x1 conv between the pool and the Add compiles. Exact.
    """

    def __init__(self, fusion: torch.nn.Module):
        super().__init__()
        channels = fusion.cv.conv.in_channels
        self.identity = torch.nn.Conv2d(channels, channels, 1, bias=False)
        with torch.no_grad():
            self.identity.weight.zero_()
            index = torch.arange(channels)
            self.identity.weight[index, index, 0, 0] = 1.0
        self.cv = fusion.cv

    def forward(self, x):
        return self.cv(x + self.identity(torch.nn.functional.adaptive_avg_pool2d(x, 1)))


class _DETRConvEncoderWrapper(torch.nn.Module):
    """Part 1 of an "encoder" hybrid split: backbone, the all-convolutional encoder and the decoder's input projections."""

    output_layout = "graph_outputs_v1"
    output_windows: ClassVar[dict] = {}
    hybrid_aipu = "backbone, convolutional encoder and decoder input projections"
    hybrid_host = "decoder (FP32 ONNX)"

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.backbone = model.backbone
        self.encoder = model.encoder
        for name, child in list(self.encoder.named_children()):
            if type(child).__name__ == "GAP_Fusion":
                setattr(self.encoder, name, _GapFusionWithIdentity(child))
        self.projections = model.decoder.input_proj
        # One memory map per decoder level (LiteEncoder lists one input level in
        # out_channels but returns two maps, FPN and PAN).
        self.output_names = tuple(f"memory{i}" for i in range(len(model.decoder.input_proj)))

    def forward(self, x):
        memory = self.encoder(self.backbone(x))
        return tuple(self.projections[i](m) for i, m in enumerate(memory))


def _detr_conv_encoder_parts(model: torch.nn.Module, output_names):
    """``(head, decoder)`` of an "encoder" hybrid split; each gets its own copy of what it runs."""
    head = _DETRConvEncoderWrapper(copy.deepcopy(model)).eval()
    decoder = copy.deepcopy(model.decoder)
    decoder.input_proj = torch.nn.ModuleList(torch.nn.Identity() for _ in decoder.input_proj)
    return head, _DETRDecoderWrapper(decoder, output_names).eval()


class _DETRDecoderWrapper(torch.nn.Module):
    """Host step of a hybrid DETR export with the neck on Metis: the decoder after its input projections.

    With ``low_level`` the last input is the backbone's raw finer stage, which
    D-FINE-seg's mask decoder reads (``low_level_feat``).
    """

    def __init__(self, decoder: torch.nn.Module, output_names, low_level: bool = False):
        super().__init__()
        self.decoder = decoder
        self.output_names = tuple(output_names)
        self.low_level = bool(low_level)

    def forward(self, *memory):
        if self.low_level:
            outputs = self.decoder(list(memory[:-1]), low_level_feat=memory[-1])
        else:
            outputs = self.decoder(list(memory))
        return tuple(outputs[name] for name in self.output_names)


class _MaskDecoderFront(torch.nn.Module):
    """D-FINE-seg's mask decoder up to its fusion convolution: laterals, GroupNorms and the upsampled sum.

    ``forward`` keeps its result in ``self.out`` for the wrapper that returns it.
    """

    def __init__(self, mask_decoder: torch.nn.Module):
        super().__init__()
        self.md = mask_decoder
        self.out = None

    def forward(self, feats):
        md = self.md
        x = md.bn[0](md.lateral[0](feats[0]))
        size = x.shape[-2:]
        for i in range(1, len(feats)):
            t = md.bn[i](md.lateral[i](feats[i]))
            x = x + torch.nn.functional.interpolate(t, size=size, mode="bilinear", align_corners=False)
        self.out = x
        return x


def _scaled_mask_embed(decoder, queries, mask_feat):
    """Stand-in for D-FINE-seg's per-query mask logits: keeps each scaled mask embedding (last layer first)."""
    embed = decoder.mask_head(queries)
    embed = embed * embed.shape[-1] ** -0.5
    decoder._mask_embeds.append(embed)
    return mask_feat


class _DETRDecoderMaskFrontWrapper(_DETRDecoderWrapper):
    """Host step of D-FINE-seg with the mask decoder's convolutions on Metis: the decoder and the mask front.

    Returns the detections, the last decoder layer's scaled mask embeddings
    (``mask_embed``) and the fusion convolution's input (``mask_fuse_in``);
    the two mask convolutions are compiled parts and the GroupNorms after them
    and the query-by-pixel product run in the host steps between and after.
    """

    def __init__(self, decoder: torch.nn.Module, output_names, low_level: bool = False):
        import types

        decoder = copy.deepcopy(decoder)
        super().__init__(decoder, output_names, low_level)
        self.front = _MaskDecoderFront(decoder.mask_decoder)
        decoder.mask_decoder = self.front
        decoder._mask_logits_from_queries = types.MethodType(_scaled_mask_embed, decoder)

    def forward(self, *memory):
        self.decoder._mask_embeds = []
        if self.low_level:
            outputs = self.decoder(list(memory[:-1]), low_level_feat=memory[-1])
        else:
            outputs = self.decoder(list(memory))
        embed = self.decoder._mask_embeds[0]
        self.decoder._mask_embeds = []
        return tuple(outputs[name] for name in self.output_names[:2]) + (embed, self.front.out)


class _MaskNormUp(torch.nn.Module):
    """Host step between the mask convolutions: GroupNorm, ReLU, then a 2x bilinear upsample."""

    def __init__(self, norm: torch.nn.Module):
        super().__init__()
        self.norm = norm
        self.output_names = ("mask_up_in",)

    def forward(self, x):
        x = torch.relu(self.norm(x))
        return torch.nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)


class _MaskLogits(torch.nn.Module):
    """Last host step: GroupNorm and ReLU of the mask map, each query's dot product with it, sigmoid."""

    def __init__(self, norm: torch.nn.Module):
        super().__init__()
        self.norm = norm
        self.output_names = ("pred_masks",)

    def forward(self, raw, embed):
        return torch.sigmoid(torch.einsum("bqc,bchw->bqhw", embed, torch.relu(self.norm(raw))))


def _detr_neck_parts(model: torch.nn.Module, output_names, samples=(), project_on_host: bool = False):
    """``(encoder_pack, neck, decoder)`` wrappers of a hybrid DETR export with the neck on Metis.

    Each gets its own deep copy of the modules it runs; ``model`` (an export
    copy whose backbone is part 1) is not modified. With calibration
    ``samples`` (``[1, 3, H, W]`` images), each canvas level is scaled by the
    power of two that brings its peak over them closest to the largest
    level's peak (exact in float: the neck divides it out).
    """
    encoder = model.encoder
    levels = len(encoder.input_proj)
    pack = _DETREncoderPackWrapper(copy.deepcopy(encoder)).eval()
    scales = None
    if samples:
        backbone = _DETRBackboneWrapper(model).eval()
        peaks = torch.zeros(levels)
        c = int(encoder.hidden_dim)
        with torch.no_grad():
            for sample in samples:
                canvas = pack(*backbone(torch.as_tensor(sample))[:levels])[0]
                peaks = torch.maximum(
                    peaks, torch.stack([canvas[:, c * i : c * (i + 1)].abs().max() for i in range(levels)])
                )
        if bool((peaks > 0).all()):
            # Powers of two, so scaling and unscaling are exact in float.
            scales = [float(2.0 ** round(math.log2(float(peaks.max() / peak)))) for peak in peaks]
            pack.scales = scales
    neck_encoder = copy.deepcopy(encoder)
    neck_encoder.input_proj = torch.nn.ModuleList(torch.nn.Identity() for _ in range(levels))
    neck_encoder.num_encoder_layers = 0
    neck_encoder.encoder = torch.nn.ModuleList()
    decoder = copy.deepcopy(model.decoder)
    if project_on_host:
        # The mask decoder (D-FINE-seg) reads the fusion neck's outputs before the
        # decoder's input projections, so those stay in the host decoder.
        projections = torch.nn.ModuleList(torch.nn.Identity() for _ in decoder.input_proj)
    else:
        projections = decoder.input_proj
        decoder.input_proj = torch.nn.ModuleList(torch.nn.Identity() for _ in projections)
    neck = _DETRNeckWrapper(
        neck_encoder, projections, encoder.hidden_dim, encoder.feat_strides, scales
    ).eval()
    low_level = _DETRBackboneWrapper(model).low_level
    return pack, neck, _DETRDecoderWrapper(decoder, output_names, low_level=low_level).eval()


def _efficientdet_at_canvas(model, canvas: int):
    """A copy of ``model`` (a LibreEfficientDet) that runs at a smaller square canvas.

    The BiFPN sizes its resampling from the configured image size when it is
    built; its weights do not depend on the spatial size, so the copy gets a
    BiFPN built for ``canvas`` with the original weights loaded. The caller's
    model is left untouched.
    """
    from ..models.efficientdet.config import SCALE_CONFIGS
    from ..models.efficientdet.nn import BiFpn

    net = copy.deepcopy(model.model).eval()
    cfg = SCALE_CONFIGS[model.size]
    fpn = BiFpn(
        int(canvas), net.backbone.feature_channels, cfg.fpn_channels, cfg.fpn_repeats
    )
    fpn.load_state_dict(net.fpn.state_dict())
    net.fpn = fpn.to(next(net.parameters()).device).eval()
    net.image_size = int(canvas)
    clone = copy.copy(model)
    clone.model = net
    clone.input_size = int(canvas)
    return clone


class AxeleraExporter(BaseExporter):
    """INT8 export for Axelera Metis AIPUs: a ``<stem>_axelera_model/`` directory.

    The directory holds ``<stem>.axm`` (the compiled model),
    ``compiler_config_final.toml`` and ``metadata.yaml``. For YOLO-NAS the
    graph is cut after the head convolutions and LibreYOLO decodes on the
    host (``libreyolo.postprocess.yolonas_raw``); classifiers export their
    whole graph up to the logits. Compiling needs Axelera's
    ``axelera-devkit`` on Linux, installed by the user.
    """

    format_name = "axelera"
    suffix = "_axelera_model"
    requires_onnx = True
    supports_int8 = True
    supports_fp16 = False
    apply_model_half = False
    supports_embedded_nms = False
    default_int8_calibration_data = True

    #: Calibration images used when ``data`` is omitted. OBB has no default:
    #: it needs aerial imagery, which LibreYOLO does not bundle.
    _CALIBRATION_DEFAULTS: ClassVar[dict[str, str]] = {
        "detect": "coco128.yaml",
        "pose": "coco128.yaml",
        "classify": "coco128.yaml",
    }
    _RECOMMENDED_CALIBRATION = (100, 400)
    _OPSET_RANGE = (14, 17)

    def __call__(
        self,
        *,
        dynamic: bool | None = None,
        half: bool = False,
        int8: bool = False,
        quantize: int | str | None = None,
        opset: int | None = None,
        batch: int = 1,
        **kwargs,
    ) -> str:
        if half:
            raise ValueError(
                "Axelera export is INT8 only; half=True is not supported. Omit "
                "half (INT8 calibration is applied automatically)."
            )
        if quantize is not None and str(quantize).strip() != "8":
            raise ValueError(
                f"Axelera export is INT8 only; quantize={quantize!r} is not "
                "supported. Omit quantize or pass quantize=8."
            )
        if dynamic:
            raise ValueError(
                "Axelera export compiles fixed input shapes; dynamic=True is not "
                "supported. Choose the input size with imgsz=."
            )
        if opset is None:
            opset = self._OPSET_RANGE[1]
        low, high = self._OPSET_RANGE
        if not low <= int(opset) <= high:
            raise ValueError(
                f"Axelera export supports ONNX opset {low}-{high}, got opset={opset}."
            )
        if int(batch) < 1:
            raise ValueError(f"batch must be >= 1, got {batch}.")
        if not int8:
            logger.info(
                "Axelera export is INT8: calibrating post-training quantization "
                "(int8=True)."
            )
        # The compiled graph takes one image; ``batch`` only sizes the
        # calibration loader.
        self._calibration_batch = int(batch)
        return super().__call__(
            dynamic=False,
            half=False,
            int8=True,
            opset=int(opset),
            batch=1,
            **kwargs,
        )

    def _task(self) -> str:
        task = getattr(self.model, "task", "detect")
        return task if isinstance(task, str) else "detect"


    def _size_recipe(self) -> dict:
        """The per-size recipe (``AXELERA_SIZE_RECIPES``) for this model, if any."""
        from .axelera import size_recipe

        recipe = size_recipe(
            self.model._get_model_name(),
            self._task(),
            str(getattr(self.model, "size", "") or ""),
        )
        if recipe and not getattr(self, "_recipe_logged", False):
            logger.info("Axelera: using the measured per-size recipe %s", recipe)
            self._recipe_logged = True
        return recipe
    def _preflight(self, **kwargs):
        # Support tier first, so a blocked family fails before any SDK check.
        super()._preflight(**kwargs)
        from .axelera import (
            check_compile_platform,
            normalize_compiler_overrides,
            require_axelera_devkit,
        )

        check_compile_platform()
        self._devkit_version = require_axelera_devkit()
        self._compiler_overrides = normalize_compiler_overrides(
            kwargs.get("compiler_config")
        )
        canvas = self._size_recipe().get("canvas")
        if (
            canvas
            and self.model._get_model_name() == "efficientdet"
            and int(self.model.input_size) != int(canvas)
        ):
            logger.warning(
                "EfficientDet-%s exports for Axelera at a fixed %dx%d canvas: at "
                "its native %d no tiling fits the AIPU's L1 memory. Images are "
                "resized to the smaller canvas, which costs accuracy on small "
                "objects (docs/axelera.md).",
                self.model.size,
                int(canvas),
                int(canvas),
                int(self.model.input_size),
            )
            self._source_model = self.model
            self.model = _efficientdet_at_canvas(self.model, int(canvas))

    def _resolve_calibration_data(self, int8: bool, data: str | None):
        if data is not None:
            return data
        task = self._task()
        if task not in self._CALIBRATION_DEFAULTS:
            raise ValueError(
                f"Axelera {task} export needs calibration images from your own "
                "domain: pass data= pointing at your dataset YAML (for YOLO-NAS-R, "
                "an aerial OBB dataset such as a DOTA-format one). LibreYOLO bundles "
                f"no default calibration set for {task}."
            )
        default = self._CALIBRATION_DEFAULTS[task]
        if task == "classify":
            logger.warning(
                "Axelera calibration: data= not given, using %s. Classifiers keep "
                "more top-1 accuracy when calibrated on their own domain: pass a "
                "dataset YAML whose train split holds %d-%d images spread over the "
                "classes (on ImageNet, ResNet-18 loses about 1.1 top-1 points with "
                "%s and 0.6 with ImageNet images).",
                default,
                *self._RECOMMENDED_CALIBRATION,
                default,
            )
            return default
        logger.info(
            "Axelera calibration: data= not given, using %s. For a fine-tuned "
            "model, calibrate with %d-%d images from its own domain.",
            default,
            *self._RECOMMENDED_CALIBRATION,
        )
        return default

    def _auto_output_path(self, half: bool, int8: bool) -> str:
        return str(Path("weights") / f"{self._auto_output_stem()}{self.suffix}")

    def _resolve_params(self, output_path, imgsz, device, half, int8):
        canvas = self._size_recipe().get("canvas")
        if imgsz is None and canvas:
            # The measured per-size canvas (e.g. RetinaNet-R50 at 1024x1024,
            # U-Net at its 512x1024 crop); an explicit imgsz= still wins.
            imgsz = tuple(int(v) for v in canvas) if isinstance(canvas, (tuple, list)) else int(canvas)
            height, width = imgsz if isinstance(imgsz, tuple) else (imgsz, imgsz)
            logger.info(
                "Axelera: %s-%s exports at its measured %dx%d canvas; pass imgsz= "
                "to choose another (docs/axelera.md).",
                self.model._get_model_name(),
                getattr(self.model, "size", ""),
                height,
                width,
            )
        imgsz, device, output_path = super()._resolve_params(
            output_path, imgsz, device, half, int8
        )
        # Graph preparation, calibration, output matching and verification all
        # feed host (CPU) tensors, and the Axelera toolchain runs on the host:
        # export on CPU. _model_context moves the model back afterwards.
        device = torch.device("cpu")
        if hasattr(self, "_source_model"):
            # The caller's model, not the canvas copy, reports what was exported.
            self._source_model._last_export_imgsz = imgsz
        stride = 32  # largest YOLO-NAS feature stride; the host decode needs whole cells
        raw_heads = (
            self._task() != "classify"
            and not self._graph_outputs()
            and not self._shuffle_skip_restore()
            # SSD300's default boxes are built for its fixed 300 x 300 canvas.
            and self.model._get_model_name() != "ssd"
        )
        if raw_heads and (imgsz[0] % stride or imgsz[1] % stride):
            raise ValueError(
                f"Axelera export needs imgsz to be a multiple of {stride}, got "
                f"{imgsz[0] if imgsz[0] == imgsz[1] else imgsz}."
            )
        return imgsz, device, output_path

    def _rectangular_imgsz_supported(self) -> bool:
        # Whole-graph dense exports and HRNet's hybrid keep their canvas (HRNet's
        # 256x192 person crop); the YOLO-NAS raw-head path stays square.
        return bool(self._graph_outputs()) or (
            self.model._get_model_name() == "hrnet" and bool(self._hybrid_outputs())
        )

    def _shuffle_skip_restore(self) -> bool:
        """Real-ESRGAN's compact network (x4t): trunk on Metis, tail on the host."""
        net = self.model.model
        return (
            self.model._get_model_name() == "realesrgan"
            and self._task() == "restore"
            and all(hasattr(net, name) for name in ("body", "upsampler", "upscale"))
        )

    def _rrdb_network(self):
        """Real-ESRGAN's RRDB network (x4/x2 checkpoints), else ``None``."""
        if self.model._get_model_name() != "realesrgan" or self._task() != "restore":
            return None
        for module in self.model.model.modules():
            if all(hasattr(module, n) for n in ("conv_first", "body", "conv_body", "conv_up1", "conv_hr")):
                return module
        return None

    def _rrdb_parts_restore(self) -> bool:
        """Real-ESRGAN-x4/x2's RRDB network as three compiled parts + a host add (recipe ``rrdb_split``)."""
        net = self._rrdb_network()
        return net is not None and bool(self._size_recipe().get("rrdb_split")) and int(getattr(net, "scale", 4)) in (2, 4)

    def _export_rrdb_parts_onnx(self, out: Path, imgsz, opset) -> str:
        """Export the three RRDB parts (``_RRDBPart``) and keep their run order in ``_hybrid_parts``.

        The parts are traced with plain ``torch.onnx.export`` (opset 17, no simplification): the graphs that
        were compiled and measured on a Metis (docs/axelera.md).
        """
        split = int(self._size_recipe()["rrdb_split"])
        net = copy.deepcopy(self._rrdb_network()).eval()
        head, tail, upsample = (_RRDBPart(net, role, split).eval() for role in ("head", "tail", "upsample"))
        x = torch.zeros(1, 3, *imgsz)
        paths = [str(out.with_name(f"{out.name}.export_intermediate{tag}.onnx")) for tag in ("", ".part2", ".part3")]
        self._intermediate_paths = list(paths)
        with torch.no_grad():
            skip, mid = head(x)
            feat = skip + tail(mid)[0]
        logger.info(
            "Step 1/2: Exporting Real-ESRGAN-%s as three parts (RRDB blocks split at %d; the host adds the trunk "
            "skip) (%s)",
            self.model.size,
            split,
            paths[0],
        )
        for module, sample, path, names in (
            (head, x, paths[0], ["images"]),
            (tail, mid, paths[1], ["mid"]),
            (upsample, feat, paths[2], ["feat"]),
        ):
            torch.onnx.export(
                module,
                (sample,),
                path,
                input_names=names,
                output_names=list(module.output_names),
                opset_version=17,
                dynamo=False,
            )
        self._raw_wrapper = head
        self._hybrid_parts = [
            {"compile": paths[1], "reference": tail, "input": "mid", "merge_pad": False},
            {"host": "add", "inputs": ["skip", "body"], "output": "feat"},
            {"compile": paths[2], "reference": upsample, "input": "feat", "merge_pad": False},
        ]
        return paths[0]

    def _export_fcos_parts_onnx(self, out: Path, imgsz) -> str:
        """FCOS as a multi-part export (GroupNorm on the host); the run order goes to ``_hybrid_parts``."""
        if int(imgsz[0]) != int(imgsz[1]):
            raise ValueError(f"FCOS exports for Axelera at a square canvas, got {imgsz[0]}x{imgsz[1]}.")
        net = copy.deepcopy(self.model.model).eval()
        trunk, layers, final, arrays, canvas, groups, eps, n_cls, width = _fcos_parts(net, int(imgsz[0]))
        work = out.with_name(f"{out.name}.export_intermediate")
        self._intermediate_paths = []

        def onnx_of(module, x, tag, in_name):
            path = f"{work}.{tag}.onnx"
            self._intermediate_paths.append(path)
            torch.onnx.export(
                module.eval(),
                (x,),
                path,
                input_names=[in_name],
                output_names=list(module.output_names),
                opset_version=17,
                dynamic_axes=None,
                dynamo=False,
            )
            return path

        logger.info(
            "Step 1/2: Exporting FCOS as %d compiled parts (trunk, 4 head convolutions on the packed %dx%d level "
            "canvas, predictor); GroupNorm and the decode run on the host",
            6,
            canvas[0],
            canvas[1],
        )
        first = onnx_of(trunk, torch.zeros(1, 3, *imgsz), "trunk", "images")
        steps = [
            {
                "host": "pack_levels",
                "inputs": list(trunk.output_names),
                "output": "packed",
                "boxes": "@npz:boxes",
                "canvas": [int(v) for v in canvas],
            }
        ]
        src = "packed"
        for k in range(4):
            named = _NamedPart(layers[k], [f"h{k + 1}raw"])
            sample = torch.zeros(1, width if k == 0 else 2 * width, *canvas)
            steps.append(
                {"compile": onnx_of(named, sample, f"l{k + 1}", src), "reference": named, "input": src, "merge_pad": False}
            )
            steps.append(
                {
                    "host": "gn_relu_levels",
                    "input": f"h{k + 1}raw",
                    "output": f"h{k + 1}",
                    "boxes": "@npz:boxes",
                    "groups": 2 * groups,
                    "gamma": f"@npz:g{k + 1}",
                    "beta": f"@npz:b{k + 1}",
                    "eps": eps,
                }
            )
            src = f"h{k + 1}"
        named = _NamedPart(final, ["headraw"])
        steps.append(
            {
                "compile": onnx_of(named, torch.zeros(1, 2 * width, *canvas), "final", src),
                "reference": named,
                "input": src,
                "merge_pad": False,
            }
        )
        steps.append(
            {
                "host": "fcos_decode",
                "input": "headraw",
                "output": "detections",
                "boxes": "@npz:boxes",
                "n_cls": n_cls,
                "anchors": "@npz:anchors",
                "class_indices": "@npz:class_indices",
            }
        )
        self._raw_wrapper = trunk
        self._hybrid_parts = steps
        self._hybrid_host_arrays = arrays
        return first

    def _export_efficientdet_parts_onnx(self, out: Path, imgsz, dummy, onnx_output, opset, simplify) -> str:
        """EfficientDet as two compiled parts: the network up to the class predictor's pointwise conv (family
        scheme), then that 1x1 conv on the five levels packed on one canvas with the recipe's scheme (min-max), so
        the class-logit outputs keep their top tail; the decode runs on the host as ONNX."""
        from ..models.efficientdet.nn import EfficientDetExportWrapper
        from .onnx import _export_onnx_graph

        if int(imgsz[0]) != int(imgsz[1]):
            raise ValueError(f"EfficientDet exports for Axelera at a square canvas, got {imgsz[0]}x{imgsz[1]}.")
        import numpy as np

        net = copy.deepcopy(self.model.model).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        trunk = _EfficientDetTrunk(net).eval()
        decode_wrapper = EfficientDetExportWrapper(
            net,
            input_size=int(imgsz[0]),
            max_candidates=5000,
            sparse_coco=(
                getattr(self.model, "nb_classes", None) == 80
                and getattr(self.model, "_arch_num_classes", None) == 90
            ),
        ).eval()
        x0 = torch.zeros(1, 3, *imgsz)
        with torch.no_grad():
            outs = trunk(x0)
        levels = len(outs) // 2
        boxes, canvas = _fcos_level_boxes([tuple(t.shape[-2:]) for t in outs[levels:]])
        predict = _NamedPart(net.class_net.predict.conv_pw, ["cls_logits"]).eval()
        tail = _EfficientDetCanvasDecode(decode_wrapper, boxes).eval()
        work = out.with_name(f"{out.name}.export_intermediate")
        self._intermediate_paths = []

        def onnx_of(module, args, tag, in_names):
            path = f"{work}.{tag}.onnx"
            self._intermediate_paths.append(path)
            torch.onnx.export(
                module.eval(),
                tuple(args),
                path,
                input_names=list(in_names),
                output_names=list(module.output_names),
                opset_version=17,
                dynamic_axes=None,
                dynamo=False,
            )
            return path

        scheme = self._size_recipe()["class_head_part"]
        logger.info(
            "Step 1/2: Exporting EfficientDet as 2 compiled parts (network up to the class predictor's pointwise "
            "conv; that conv on the packed %dx%d level canvas with %s ranges); the decode runs on the host",
            canvas[0],
            canvas[1],
            scheme,
        )
        # The trunk goes through the same ONNX export (simplify, shape inference) as the single-part graph, so the
        # family's graph passes (BiFPN fusion, staged global pools) find what they rewrite.
        first = _export_onnx_graph(
            trunk,
            dummy,
            output_path=onnx_output,
            opset=opset,
            simplify=simplify,
            half=False,
            metadata=self._build_onnx_metadata(dynamic=False, half=False, imgsz=imgsz),
            input_names=["images"],
            output_names=list(trunk.output_names),
            dynamic_axes=None,
        )
        width = int(outs[levels].shape[1])
        cls_shape = (1, int(net.class_net.predict.conv_pw.out_channels), *canvas)
        tail_args = [torch.zeros(cls_shape), *[torch.zeros_like(t) for t in outs[:levels]]]
        tail_inputs = ["cls_logits", *trunk.output_names[:levels]]
        steps = [
            {
                "host": "pack_levels",
                "inputs": list(trunk.output_names[levels:]),
                "output": "cls_packed",
                "boxes": "@npz:boxes",
                "canvas": [int(v) for v in canvas],
            },
            {
                "compile": onnx_of(predict, [torch.zeros(1, width, *canvas)], "cls_head", ["cls_packed"]),
                "reference": predict,
                "input": "cls_packed",
                "merge_pad": False,
                "quantization_scheme": scheme,
            },
            {
                "host_onnx": onnx_of(tail, tail_args, "decode", tail_inputs),
                "inputs": tail_inputs,
                "outputs": ["output"],
            },
        ]
        self._raw_wrapper = trunk
        self._hybrid_parts = steps
        self._hybrid_host_arrays = {"boxes": np.asarray(boxes, dtype=np.int64)}
        self._hybrid_first_passes = _AXELERA_GRAPH_PASSES.get("efficientdet", ())
        return first

    def _graph_outputs(self) -> tuple[str, ...]:
        """ONNX output names when the whole dense graph runs on Metis, else ``()``."""
        return _AXELERA_GRAPH_OUTPUTS.get((self.model._get_model_name(), self._task()), ())

    def _hybrid_outputs(self) -> tuple[str, ...]:
        """Final ONNX output names of a hybrid export (backbone on Metis, transformer on the host), else ``()``."""
        key = (self.model._get_model_name(), self._task())
        if key in _AXELERA_RECIPE_HYBRID_OUTPUTS:
            return _AXELERA_RECIPE_HYBRID_OUTPUTS[key] if self._size_recipe().get("hybrid_split") else ()
        return _AXELERA_HYBRID_OUTPUTS.get(key, ())

    def _calibration_preprocess(self):
        """``(rgb_hwc, imgsz) -> (chw, ratio)`` matching what the exported model sees.

        Calibration and ``verify`` images must go through the preprocessing the
        Axelera backend applies at predict time, which for fixed-canvas dense
        exports differs from the native model's: depth stretches to the canvas,
        super-resolution pads a crop to it.
        """
        import numpy as np

        from ..backends.base import _DEPTH_CUBIC_RESIZE_FAMILIES, BaseBackend

        task = self._task()
        if task == "depth":
            cubic = self.model._get_model_name() in _DEPTH_CUBIC_RESIZE_FAMILIES

            def depth(rgb, imgsz):
                tensor = BaseBackend._preprocess_depth(rgb, imgsz, "rgb", cubic=cubic)[0]
                return tensor[0].numpy(), 1.0

            return depth
        if task == "restore":

            def restore(rgb, imgsz):
                h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
                top = max(0, (rgb.shape[0] - h) // 2)
                left = max(0, (rgb.shape[1] - w) // 2)
                crop = np.ascontiguousarray(rgb[top : top + h, left : left + w])
                return BaseBackend._preprocess_restore(crop, (h, w), "rgb")[0][0].numpy(), 1.0

            return restore
        if self.model._get_model_name() in ("faster_rcnn", "mask_rcnn"):
            import cv2

            # The backend's fixed-shape Faster R-CNN preprocessing: stretch to the
            # canvas, RGB in [0, 1]; the compiled part normalizes.
            def canvas(rgb, imgsz):
                h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
                resized = cv2.resize(rgb, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
                return np.ascontiguousarray(resized.astype(np.float32).transpose(2, 0, 1) / 255.0), 1.0

            return canvas
        preprocess = self.model._get_preprocess_numpy()
        if self._hybrid_outputs() and self.model._get_model_name() not in ("retinanet", "fcos", "hrnet"):
            # DETR preprocessing takes one square size and returns NCHW.
            def square(rgb, imgsz):
                h, w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
                if int(h) != int(w):
                    raise ValueError(f"{self.model._get_model_name()} exports a square input, got {h}x{w}.")
                chw, ratio = preprocess(rgb, int(h))
                return (chw[0] if chw.ndim == 4 else chw), ratio

            return square
        if self.model._get_model_name() in ("retinanet", "fcos"):
            # RetinaNet's and FCOS's own preprocessing give every image its own
            # shape; the compiled graph takes one square canvas, so images are
            # fitted into it (the backend does the same for fixed-shape artifacts).
            import functools

            return functools.partial(preprocess, fixed_canvas=True)
        return preprocess

    def _prepare_quicksrnet(self, net) -> None:
        """Equalize QuickSRNet's hidden channels for per-channel INT8 weights."""
        from ..models.quicksrnet.quantize import prepare_for_int8

        samples = [torch.as_tensor(x) for x in self._equalization_samples()]
        if not samples:
            return
        learned = bool(self._size_recipe().get("learned_rounding"))
        rounding = [torch.as_tensor(x) for x in self._equalization_samples(limit=128)] if learned else None
        summary = prepare_for_int8(net, samples, learned_rounding=learned, rounding_samples=rounding)
        logger.info(
            "Axelera: QuickSRNet hidden channels equalized through their Hardtanh "
            "bounds (smallest calibration peak per layer %s); %s; %d biases corrected.",
            ", ".join(f"{v:.3f}" for v in summary["layer_min_peaks"]),
            f"INT8 rounding learned for {summary['learned_rounding']} convolutions on "
            f"{len(rounding)} calibration images" if learned else "nearest INT8 rounding",
            summary["bias_corrected"],
        )

    def _prepare_mobilenetv4(self, net, dummy) -> None:
        """Equalize and bound MobileNetV4's residual streams for per-tensor INT8."""
        from ..models.mobilenetv4.quantize import prepare_for_int8

        samples = [torch.as_tensor(x) for x in self._equalization_samples()]
        if not samples:
            return
        with torch.no_grad():
            before = [net(x) for x in samples[:4]]
        summary = prepare_for_int8(net, samples)
        with torch.no_grad():
            drift = max(float((net(x) - b).abs().max()) for x, b in zip(samples[:4], before))
        logger.info(
            "Axelera: MobileNetV4 residual streams equalized (stages %s) and bounded "
            "(%s); largest FP32 logit change on calibration images %.3g.",
            summary["equalized_stages"],
            ", ".join(f"stage {k} at {v:.1f}" for k, v in summary["stage_bounds"].items()) or "none",
            drift,
        )

    def _pointwise_method(self) -> str:
        """CSPNeXt pointwise equalization factor: the override, else the per-task best."""
        if CSPNEXT_POINTWISE_EQUALIZATION_METHOD is not None:
            return CSPNEXT_POINTWISE_EQUALIZATION_METHOD
        return CSPNEXT_POINTWISE_EQUALIZATION_METHOD_BY_TASK.get(self._task(), "act")

    def _equalization_samples(self, limit: int = 32) -> list:
        """Up to ``limit`` calibration images (``[1, C, H, W]``), or ``[]``."""
        loader = getattr(self, "_calibration_loader", None)
        samples = []
        for batch in loader or ():
            for image in batch:
                samples.append(image[None])
                if len(samples) >= limit:
                    return samples
        return samples

    def _load_calibration(self, data, imgsz, batch, fraction, allow_download_scripts=False):
        from .calibration import CalibrationDataLoader

        loader = CalibrationDataLoader(
            data,
            imgsz=imgsz,
            batch=getattr(self, "_calibration_batch", 1),
            fraction=fraction,
            preprocess_fn=self._calibration_preprocess(),
            allow_download_scripts=allow_download_scripts,
        )
        self._calibration_loader = loader
        count = int(loader.num_samples)
        low, high = self._RECOMMENDED_CALIBRATION
        if count < 2:
            raise ValueError(
                f"Axelera calibration needs at least 2 images; {data!r} with "
                f"fraction={fraction} gives {count}."
            )
        if count < low:
            logger.warning(
                "Axelera calibration uses %d images from %s; %d-%d representative "
                "images are recommended.",
                count,
                data,
                low,
                high,
            )
        self._calibration_files = {str(Path(f)) for f in loader.img_files[:count]}
        self._calibration_info = {
            "data": str(data),
            "num_images": count,
            "fraction": float(fraction),
            "batch": int(loader.batch),
            "preprocess": self._task(),
        }
        logger.info("Axelera calibration: %d images from %s", count, data)
        return _single_image_samples(loader, count)

    def _export_intermediate_onnx(
        self, nn_model, dummy, output_path, opset, simplify, dynamic
    ):
        from .onnx import _export_onnx_graph

        imgsz = (int(dummy.shape[-2]), int(dummy.shape[-1]))
        out = Path(output_path)
        onnx_output = str(out.with_name(f"{out.name}.export_intermediate.onnx"))
        part2_output = str(out.with_name(f"{out.name}.export_intermediate.part2.onnx"))
        self._intermediate_paths = [onnx_output, part2_output]
        if self._shuffle_skip_restore():
            if self._size_recipe().get("weight_equalization"):
                from ..models.realesrgan.quantize import equalize_conv_activation_ranges

                nn_model = copy.deepcopy(nn_model).eval()
                logger.info(
                    "Axelera: %d convolution pairs of the compact trunk equalized through "
                    "their activations (exact; docs/axelera.md).",
                    equalize_conv_activation_ranges(nn_model),
                )
            self._raw_wrapper = _RestoreBodyWrapper(nn_model).eval()
            logger.info("Step 1/2: Exporting restore trunk ONNX (%s)", onnx_output)
            return _export_onnx_graph(
                self._raw_wrapper,
                dummy,
                output_path=onnx_output,
                opset=opset,
                simplify=simplify,
                half=False,
                metadata=self._build_onnx_metadata(dynamic=False, half=False, imgsz=imgsz),
                input_names=["images"],
                output_names=list(self._raw_wrapper.output_names),
                dynamic_axes=None,
            )
        if self._rrdb_parts_restore():
            return self._export_rrdb_parts_onnx(out, imgsz, opset)
        if self.model._get_model_name() == "fcos" and self._task() == "detect":
            return self._export_fcos_parts_onnx(out, imgsz)
        if self.model._get_model_name() == "efficientdet" and self._size_recipe().get("class_head_part"):
            return self._export_efficientdet_parts_onnx(out, imgsz, dummy, onnx_output, opset, simplify)
        graph_outputs = () if self._hybrid_outputs() else self._graph_outputs()
        if graph_outputs:
            if self.model._get_model_name() == "quicksrnet":
                nn_model = copy.deepcopy(nn_model).eval()
                self._prepare_quicksrnet(nn_model)
            if self.model._get_model_name() == "yolo1":
                nn_model = copy.deepcopy(nn_model).eval()
                logger.info(
                    "Axelera: %d Darknet batch norms folded into their convolutions, "
                    "%d [local] layers as per-position convolutions (exact; docs/axelera.md).",
                    _fuse_darknet_batchnorms(nn_model.net),
                    _local_layers_to_convs(nn_model.net),
                )
            if self.model._get_model_name() == "ppliteseg":
                nn_model = copy.deepcopy(nn_model).eval()
                logger.info(
                    "Axelera: %d UAFM blends as x * sigmoid(z) + skip * sigmoid(-z) "
                    "(exact; docs/axelera.md).",
                    _ppliteseg_int8_blend(nn_model),
                )
            if self._size_recipe().get("aspp_shift"):
                nn_model = copy.deepcopy(nn_model).eval()
                logger.info(
                    "Axelera: %d ASPP dilated convolutions run as per-tap 1x1 "
                    "convolutions plus zero-filled shifts (exact; docs/axelera.md).",
                    _aspp_shift(nn_model),
                )
            self._raw_wrapper = _GraphOutputsWrapper(nn_model, graph_outputs).eval()
            logger.info("Step 1/2: Exporting %s ONNX (%s)", self._task(), onnx_output)
            return _export_onnx_graph(
                self._raw_wrapper,
                dummy,
                output_path=onnx_output,
                opset=opset,
                simplify=simplify,
                half=False,
                metadata=self._build_onnx_metadata(dynamic=False, half=False, imgsz=imgsz),
                input_names=["images"],
                output_names=list(graph_outputs),
                dynamic_axes=None,
            )
        if self._task() == "classify" and not self._hybrid_outputs():
            net = copy.deepcopy(self.model.model).eval()
            if self.model._get_model_name() == "mobilenetv4" and self._size_recipe().get(
                "stream_transform", True
            ):
                self._prepare_mobilenetv4(net, dummy)
            self._raw_wrapper = _ClassifyLogitsWrapper(net).eval()
            logger.info("Step 1/2: Exporting classifier ONNX (%s)", onnx_output)
            return _export_onnx_graph(
                self._raw_wrapper,
                dummy,
                output_path=onnx_output,
                opset=opset,
                simplify=simplify,
                half=False,
                metadata=self._build_onnx_metadata(dynamic=False, half=False, imgsz=imgsz),
                input_names=["images"],
                output_names=list(self._raw_wrapper.output_names),
                dynamic_axes=None,
            )
        hybrid = self._hybrid_outputs()
        if hybrid:
            return self._export_hybrid_onnx(dummy, out, opset, simplify, imgsz, hybrid)
        live = self.model.model
        self._chain_tail = None
        if self.model._get_model_name() == "ssd":
            # Two parts: conv4_3's L2Norm runs on the host between them.
            self._pose_bounds = {}
            net = copy.deepcopy(live).eval()
            self._raw_wrapper = _SSDPartAWrapper(net).eval()
            self._chain_tail = _SSDPartBWrapper(net, self._raw_wrapper.layout_names[:2]).eval()
        elif self.model._get_model_name() == "retinanet":
            self._pose_bounds = {}
            self._raw_wrapper = _RetinaNetRawHeadsWrapper(
                copy.deepcopy(live).eval(), self._task()
            ).eval()
        elif self.model._get_model_name() == "ppyoloe":
            self._pose_bounds = {}
            self._raw_wrapper = _PPYOLOERawHeadsWrapper(live, self._task()).eval()
        elif self.model._get_model_name() == "yolo7":
            self._pose_bounds = {}
            self._raw_wrapper = _yolo7_raw_heads_wrapper(live, self._task()).eval()
        elif self.model._get_model_name() in ("yolo2", "yolo3", "yolo4"):
            self._pose_bounds = {}
            self._raw_wrapper = _darknet_raw_heads_wrapper(
                live, self._task(), output_windows=self._size_recipe().get("output_windows")
            ).eval()
        elif self.model._get_model_name() == "picodet":
            if self._task() != "detect":
                raise ValueError(
                    f"PicoDet raw-head export supports detect only, got {self._task()!r}."
                )
            self._pose_bounds = {}
            self._raw_wrapper = _PicoDetRawHeadsWrapper(live).eval()
        elif self.model._get_model_name() in ("yolo9", "yolo9_e2e", "yolo9_p2"):
            # No reparameterization step; BatchNorm folds into the convs
            # during the ONNX export. YOLO9-E2E decodes its one-to-one branch;
            # YOLO9-P2 adds a stride-4 level (the strides travel in metadata).
            self._pose_bounds = {}
            towers = (
                ("one_to_one_anchor_convs", "one_to_one_class_convs")
                if self.model._get_model_name() == "yolo9_e2e"
                else ("anchor_convs", "class_convs")
            )
            self._raw_wrapper = _YOLO9RawHeadsWrapper(
                copy.deepcopy(live).eval(), self._task(), towers=towers
            ).eval()
        elif self.model._get_model_name() == "yolox":
            self._pose_bounds = {}
            self._raw_wrapper = _YOLOXRawHeadsWrapper(
                copy.deepcopy(live).eval(), self._task()
            ).eval()
        elif self.model._get_model_name() == "rtmdet":
            # Like YOLOX: BatchNorm folds into the convs during the ONNX
            # export; the copy keeps RTMDet's shared head convs shared.
            self._pose_bounds = {}
            copy_ = copy.deepcopy(live).eval()
            self._pointwise_equalized = _equalize_cspnext_pointwise_inputs(
                copy_,
                self._equalization_samples(),
                CSPNEXT_POINTWISE_EQUALIZATION,
                self._pointwise_method(),
            )
            self._pointwise_rounded = 0
            if self._size_recipe().get("pointwise_learned_rounding"):
                from ..models.rtmdet.quantize import learn_pointwise_int8_rounding

                samples = self._equalization_samples()
                self._pointwise_rounded = learn_pointwise_int8_rounding(copy_, samples)
                logger.info(
                    "Axelera: INT8 rounding learned for %d CSPNeXt pointwise convolutions "
                    "on %d calibration images.",
                    self._pointwise_rounded,
                    len(samples),
                )
            wrapper_cls = (
                _RTMDetInsRawHeadsWrapper
                if self._task() == "segment"
                else _RTMDetRawHeadsWrapper
            )
            self._raw_wrapper = wrapper_cls(copy_, self._task()).eval()
        else:
            fused = _fused_yolonas_copy(live, full_fusion=True)
            samples = self._equalization_samples() if self._task() == "pose" else []
            if self._task() == "pose":
                _equalize_yolonas_pose_heads(fused, samples)
            # Fusion and equalization must leave the FP32 raw outputs unchanged.
            self._fusion_parity = _check_yolonas_fusion_parity(
                live, fused, self._task(), imgsz
            )
            self._pose_bounds = (
                _bound_yolonas_pose_inputs(
                    fused,
                    samples,
                    towers=self._size_recipe().get("pose_bounded_towers", _POSE_BOUNDED_TOWERS),
                )
                if samples
                else {}
            )
            self._raw_wrapper = _YOLONASRawHeadsWrapper(fused, self._task()).eval()
        out = Path(output_path)
        onnx_output = str(out.with_name(f"{out.name}.export_intermediate.onnx"))
        logger.info("Step 1/2: Exporting raw-head ONNX (%s)", onnx_output)
        exported = _export_onnx_graph(
            self._raw_wrapper,
            dummy,
            output_path=onnx_output,
            opset=opset,
            simplify=simplify,
            half=False,
            metadata=self._build_onnx_metadata(dynamic=False, half=False, imgsz=imgsz),
            input_names=["images"],
            output_names=list(self._raw_wrapper.output_names),
            dynamic_axes=None,
        )
        if self._chain_tail is not None:
            with torch.no_grad():
                tail_input = torch.zeros_like(self._raw_wrapper(dummy)[0])
            self._chain_tail_onnx = _export_onnx_graph(
                self._chain_tail,
                tail_input,
                output_path=part2_output,
                opset=opset,
                simplify=simplify,
                half=False,
                metadata=self._build_onnx_metadata(dynamic=False, half=False, imgsz=imgsz),
                input_names=["conv4_3_norm"],
                output_names=list(self._chain_tail.output_names),
                dynamic_axes=None,
            )
        return exported

    def _hybrid_neck_scheme(self) -> str:
        """Quantization scheme of the compiled neck part: recipe, then family, then per tensor."""
        recipe = self._size_recipe().get("neck_quantization_scheme")
        if recipe:
            return str(recipe)
        family = self.model._get_model_name()
        return _HYBRID_NECK_QUANTIZATION_SCHEMES.get(family, _HYBRID_NECK_QUANTIZATION_SCHEME)

    def _hybrid_split(self) -> str:
        """Where a hybrid export stops running on Metis: ``"neck"`` (default) or ``"backbone"``."""
        family = self.model._get_model_name()
        if family not in _HYBRID_BACKBONE_ONLY and not _has_attention_encoder(self.model.model):
            return "encoder"
        default = "backbone" if family in _HYBRID_BACKBONE_ONLY else "neck"
        split = self._size_recipe().get("hybrid_split", default)
        if split not in ("neck", "backbone"):
            raise ValueError(f"Unknown hybrid split {split!r}; use 'neck' or 'backbone'.")
        if split == "neck" and family in _HYBRID_BACKBONE_ONLY:
            raise ValueError(f"{family} has no convolutional neck: its hybrid split is 'backbone'.")
        return split

    def _export_hybrid_onnx(self, dummy, out: Path, opset, simplify, imgsz, outputs):
        """Export the parts of a hybrid DETR export and keep their run order in ``_hybrid_parts``.

        ``"backbone"`` split: the backbone and the encoder's input projections on
        Metis, everything after them as one host ONNX graph. ``"neck"`` split
        (default): the attention encoder and the canvas packing on the host,
        then the fusion neck and the decoder's input projections on Metis, then
        the decoder on the host.
        """
        from .onnx import _export_onnx_graph

        net = _unfrozen_batchnorms(copy.deepcopy(self.model.model).eval())
        own = _HYBRID_BACKBONE_WRAPPERS.get(self.model._get_model_name())
        head_cls, tail_cls = (
            (globals()[own[0]], globals()[own[1]]) if own else (_DETRBackboneWrapper, _DETRHostTailWrapper)
        )
        split = self._hybrid_split()
        if split == "encoder":
            self._raw_wrapper, conv_encoder_tail = _detr_conv_encoder_parts(net, outputs)
        else:
            self._raw_wrapper = head_cls(net).eval()
        features = list(self._raw_wrapper.output_names)
        onnx_meta = self._build_onnx_metadata(dynamic=False, half=False, imgsz=imgsz)

        def export(module, sample, path, inputs, names):
            self._intermediate_paths.append(path)
            return _export_onnx_graph(
                module,
                sample,
                output_path=path,
                opset=opset,
                simplify=simplify,
                half=False,
                metadata=onnx_meta,
                input_names=list(inputs),
                output_names=list(names),
                dynamic_axes=None,
            )

        part1 = str(out.with_name(f"{out.name}.export_intermediate.onnx"))
        logger.info(
            "Step 1/2: Exporting the hybrid parts (%s split): backbone ONNX for Metis (%s)",
            split,
            part1,
        )
        exported = export(self._raw_wrapper, dummy, part1, ["images"], features)
        with torch.no_grad():
            sample = tuple(self._raw_wrapper(dummy))
        if split in ("backbone", "encoder"):
            tail = conv_encoder_tail if split == "encoder" else tail_cls(net, outputs).eval()
            if hasattr(tail, "upsample_to"):
                tail.upsample_to = tuple(int(v) for v in dummy.shape[-2:])
                self._hybrid_first_passes = _AXELERA_GRAPH_PASSES.get(self.model._get_model_name(), ())
            if hasattr(tail, "canvas"):
                canvas = tuple(int(v) for v in dummy.shape[-2:])
                min_size = int(net.transform.min_size[-1])
                if canvas != (min_size, min_size):
                    raise ValueError(
                        f"{self.model._get_model_name()} exports to Axelera at its {min_size}x{min_size} "
                        f"canvas (the model's min_size); got {canvas[0]}x{canvas[1]}."
                    )
                tail.canvas = canvas
            host = export(tail, sample, str(out.with_name(f"{out.name}.export_intermediate.host.onnx")), features, outputs)
            self._hybrid_parts = [
                {"host_onnx": host, "inputs": features, "outputs": list(outputs)},
            ]
            return exported
        pack, neck, decoder = _detr_neck_parts(
            net, outputs, self._equalization_samples(), project_on_host="pred_masks" in outputs
        )
        low = ["low_level"] if "low_level" in features else []
        levels = [name for name in features if name != "low_level"]
        self._hybrid_canvas_scales = list(pack.scales)
        logger.info(
            "Axelera hybrid: canvas level scales %s (powers of two that bring each level's "
            "calibration peak near the largest one; the neck divides them out)",
            ", ".join(f"{v:.3f}" for v in pack.scales),
        )
        level_sample = sample[: len(levels)]
        with torch.no_grad():
            canvas = pack(*level_sample)[0]
            memory = tuple(neck(canvas)) + tuple(sample[len(levels):])
        host1 = export(pack, level_sample, str(out.with_name(f"{out.name}.export_intermediate.host1.onnx")), levels, ["canvas"])
        part2 = export(neck, canvas, str(out.with_name(f"{out.name}.export_intermediate.part2.onnx")), ["canvas"], neck.output_names)
        host2 = export(
            decoder, memory, str(out.with_name(f"{out.name}.export_intermediate.host2.onnx")),
            list(neck.output_names) + low, outputs,
        )
        mask_parts = []
        # D-FINE-seg: the mask decoder's two convolutions (most of the host MACs) run on
        # Metis as their own parts; a recipe ``mask_decoder_on_metis: False`` keeps them on the host.
        if (
            "pred_masks" in outputs
            and hasattr(decoder.decoder, "mask_decoder")
            and self._size_recipe().get("mask_decoder_on_metis", True)
        ):
            mask_parts, host2 = self._mask_decoder_parts(decoder, memory, outputs, list(neck.output_names) + low, export, out)
        self._hybrid_parts = [
            {"host_onnx": host1, "inputs": levels, "outputs": ["canvas"]},
            # The neck part compiles per tensor whatever the backbone's scheme: its graph
            # input feeds a depthwise identity conv, and per-channel depthwise input ranges
            # (the depthwise scheme of HGNetv2 backbones) at a part's graph input fail the
            # compiler's input-quantization pass. The neck itself has no depthwise convs.
            {
                "compile": part2,
                "reference": neck,
                "input": "canvas",
                "merge_pad": False,
                "quantization_scheme": self._hybrid_neck_scheme(),
            },
            {
                "host_onnx": host2,
                "inputs": list(neck.output_names) + low,
                "outputs": list(outputs) if not mask_parts else [*outputs[:2], "mask_embed", "mask_fuse_in"],
            },
            *mask_parts,
        ]
        return exported

    def _mask_decoder_parts(self, decoder, memory, outputs, inputs, export, out: Path):
        """D-FINE-seg's mask convolutions as two compiled parts, with their GroupNorms on the host.

        Returns the steps after the host decoder and the host decoder graph
        that ends at the mask decoder's fusion input.
        """
        md = decoder.decoder.mask_decoder
        front = _DETRDecoderMaskFrontWrapper(decoder.decoder, outputs, low_level=decoder.low_level).eval()
        names = [*outputs[:2], "mask_embed", "mask_fuse_in"]
        host2 = export(front, memory, str(out.with_name(f"{out.name}.export_intermediate.host2.onnx")), inputs, names)
        with torch.no_grad():
            fuse_in = front(*memory)[3]
            fuse = _NamedPart(copy.deepcopy(md.fusion_conv), ["mask_fuse_raw"])
            norm_up = _MaskNormUp(copy.deepcopy(md.fusion_norm)).eval()
            up_in = norm_up(fuse(fuse_in)[0])
            up = _NamedPart(copy.deepcopy(md.up_conv), ["mask_up_raw"])
            logits = _MaskLogits(copy.deepcopy(md.bn1)).eval()
            up_raw = up(up_in)[0]
            embed = front(*memory)[2]
        scheme = self._hybrid_neck_scheme()
        stem = out.with_name(f"{out.name}.export_intermediate")
        logger.info("Axelera hybrid: D-FINE-seg mask convolutions as 2 compiled parts (GroupNorms on the host)")
        return [
            {
                "compile": export(fuse, fuse_in, f"{stem}.mask_fuse.onnx", ["mask_fuse_in"], ["mask_fuse_raw"]),
                "reference": fuse, "input": "mask_fuse_in", "merge_pad": False, "quantization_scheme": scheme,
            },
            {
                "host_onnx": export(norm_up, fuse(fuse_in)[0], f"{stem}.host3.onnx", ["mask_fuse_raw"], ["mask_up_in"]),
                "inputs": ["mask_fuse_raw"], "outputs": ["mask_up_in"],
            },
            {
                "compile": export(up, up_in, f"{stem}.mask_up.onnx", ["mask_up_in"], ["mask_up_raw"]),
                "reference": up, "input": "mask_up_in", "merge_pad": False, "quantization_scheme": scheme,
            },
            {
                "host_onnx": export(logits, (up_raw, embed), f"{stem}.host4.onnx", ["mask_up_raw", "mask_embed"], ["pred_masks"]),
                "inputs": ["mask_up_raw", "mask_embed"], "outputs": ["pred_masks"],
            },
        ], host2

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        meta = super()._build_metadata("int8", False, None, imgsz=imgsz)
        wrapper = self._raw_wrapper
        if isinstance(wrapper, _RestoreBodyWrapper):
            meta.update(
                {
                    "format": "axelera",
                    "precision": "int8",
                    "dynamic": False,
                    "batch": 1,
                    "output_layout": wrapper.output_layout,
                    "output_names": list(wrapper.output_names),
                    "upscale": wrapper.upscale,
                }
            )
            return meta
        if isinstance(wrapper, _RRDBPart):
            meta.update(
                {
                    "format": "axelera",
                    "precision": "int8",
                    "dynamic": False,
                    "batch": 1,
                    "output_layout": wrapper.output_layout,
                    "output_names": ["restored"],
                    "graph_outputs": ["restored"],
                    "upscale": wrapper.upscale,
                }
            )
            return meta
        if isinstance(wrapper, _EfficientDetTrunk):
            meta.update(
                {
                    "format": "axelera",
                    "precision": "int8",
                    "dynamic": False,
                    "batch": 1,
                    "output_layout": wrapper.output_layout,
                    "output_names": ["output"],
                    "graph_outputs": ["output"],
                }
            )
            return meta
        if isinstance(wrapper, _FCOSTrunk):
            meta.update(
                {
                    "format": "axelera",
                    "precision": "int8",
                    "dynamic": False,
                    "batch": 1,
                    "output_layout": wrapper.output_layout,
                    "output_names": ["detections"],
                    "graph_outputs": ["detections"],
                }
            )
            return meta
        if isinstance(
            wrapper,
            (
                _DETRBackboneWrapper,
                _VanillaDETRBackboneWrapper,
                _RCNNBackboneWrapper,
                _DETRConvEncoderWrapper,
                _DocLayoutBackboneWrapper,
                _DeformableDETRBackboneWrapper,
                _DeepLabBackboneWrapper,
                _VGGFeaturesWrapper,
                _RetinaNetBodyWrapper,
                _CenterNetBodyWrapper,
                _HRNetBodyWrapper,
                _DEKRBodyWrapper,
                _ZipDepthBodyWrapper,
                _PIDNetBodyWrapper,
            ),
        ):
            outputs = list(self._hybrid_outputs())
            own_split = hasattr(wrapper, "hybrid_aipu")
            meta.update(
                {
                    "format": "axelera",
                    "precision": "int8",
                    "dynamic": False,
                    "batch": 1,
                    "output_layout": wrapper.output_layout,
                    "output_names": outputs,
                    "graph_outputs": outputs,
                    "hybrid": (
                        {
                            "split": self._hybrid_split(),
                            "aipu": wrapper.hybrid_aipu
                            if own_split
                            else "backbone and encoder input projections",
                            "host": wrapper.hybrid_host
                            if own_split
                            else "attention encoder, fusion neck and decoder (FP32 ONNX)",
                        }
                        if self._hybrid_split() in ("backbone", "encoder")
                        else {
                            "split": "neck",
                            "aipu": "backbone, encoder input projections, fusion neck and "
                            "decoder input projections",
                            "host": "attention encoder and decoder (FP32 ONNX)",
                            "canvas_scales": [
                                round(float(v), 6)
                                for v in getattr(self, "_hybrid_canvas_scales", ())
                            ],
                            "neck_quantization_scheme": self._hybrid_neck_scheme(),
                        }
                    ),
                }
            )
            return meta
        if isinstance(wrapper, _GraphOutputsWrapper):
            meta.update(
                {
                    "format": "axelera",
                    "precision": "int8",
                    "dynamic": False,
                    "batch": 1,
                    "output_layout": wrapper.output_layout,
                    "output_names": list(wrapper.output_names),
                    # Graph order; ``output_names`` becomes the device order.
                    "graph_outputs": list(wrapper.output_names),
                }
            )
            return meta
        if isinstance(wrapper, _ClassifyLogitsWrapper):
            meta.update(
                {
                    "format": "axelera",
                    "precision": "int8",
                    "dynamic": False,
                    "batch": 1,
                    "output_layout": wrapper.output_layout,
                    "output_names": list(wrapper.output_names),
                }
            )
            return meta
        heads = (  # YOLO-NAS heads
            wrapper.model.heads if isinstance(wrapper, _YOLONASRawHeadsWrapper) else None
        )
        grid_cell_offset = (
            heads.grid_cell_offset if heads is not None else wrapper.grid_cell_offset
        )
        meta.update(
            {
                "format": "axelera",
                "precision": "int8",
                "dynamic": False,
                "batch": 1,
                "output_layout": wrapper.output_layout,
                "output_names": list(wrapper.output_names),
                "strides": list(wrapper.strides),
                "grid_cell_offset": float(grid_cell_offset),
            }
        )
        if isinstance(wrapper, _SSDPartAWrapper):
            # The backend assembles the decoded heads from both parts.
            meta["output_names"] = list(wrapper.layout_names)
            meta["class_index_map"] = list(wrapper.class_index_map)
        if wrapper.reg_max is not None:
            meta["reg_max"] = int(wrapper.reg_max)
        if getattr(wrapper, "anchors", None) is not None:
            meta["anchors"] = [[list(pair) for pair in level] for level in wrapper.anchors]
        if getattr(wrapper, "scale_x_y", None) is not None:
            meta["scale_x_y"] = list(wrapper.scale_x_y)
        if getattr(wrapper, "exp_on_reg", None) is not None:
            meta["exp_on_reg"] = bool(wrapper.exp_on_reg)
        if getattr(self, "_pointwise_equalized", 0):
            meta["pointwise_equalized"] = {
                "blocks": int(self._pointwise_equalized),
                "mode": str(CSPNEXT_POINTWISE_EQUALIZATION),
                "method": self._pointwise_method(),
            }
        if getattr(self, "_pointwise_rounded", 0):
            meta["pointwise_learned_rounding"] = {"convs": int(self._pointwise_rounded)}
        if getattr(wrapper, "anchor_sizes", None) is not None:  # RetinaNet
            meta["anchor_sizes"] = [list(sizes) for sizes in wrapper.anchor_sizes]
            meta["aspect_ratios"] = [list(r) for r in wrapper.aspect_ratios]
            if wrapper.class_index_map is not None:
                meta["class_index_map"] = list(wrapper.class_index_map)
        if getattr(self, "_pose_bounds", None):
            meta["pose_head_bounds"] = {
                name: round(float(cap), 4) for name, cap in self._pose_bounds.items()
            }
        if wrapper.output_windows:
            meta["output_windows"] = {
                part: [float(low), float(high)]
                for part, (low, high) in wrapper.output_windows.items()
            }
        if self._task() == "pose":
            meta["pose_offset_multiplier"] = float(heads.pose_offset_multiplier)
            meta["compensate_grid_cell_offset"] = bool(
                heads.compensate_grid_cell_offset
            )
        if self.model._get_model_name() == "yolonas":
            meta["weights_license"] = (
                "YOLO-NAS pretrained weights are Deci's, under "
                "LICENSE.YOLONAS.md (non-commercial, no redistribution). "
                "Artifacts compiled from them carry the same terms."
            )
        return meta

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        calibration_data,
        onnx_path,
        verbose,
        compiler_config=None,
        verify=False,
        **kwargs,
    ):
        from .axelera import (
            DEFAULT_QUANTIZATION_SCHEME_BY_MODEL,
            DEPTHWISE_QUANTIZATION_SCHEME,
            default_quantization_scheme,
            export_axelera,
        )

        family, task = self.model._get_model_name(), self._task()
        # A per-model exception wins; otherwise graphs with depthwise
        # convolutions get per-channel ranges for the depthwise inputs.
        recipe_scheme = self._size_recipe().get("quantization_scheme")
        if recipe_scheme:
            quantization_scheme = recipe_scheme
        elif (family, task) not in DEFAULT_QUANTIZATION_SCHEME_BY_MODEL and (
            _has_depthwise_convs(self._raw_wrapper)
        ):
            quantization_scheme = DEPTHWISE_QUANTIZATION_SCHEME
        else:
            quantization_scheme = default_quantization_scheme(family, task)

        if getattr(self, "_hybrid_parts", None):
            from .axelera import export_axelera_parts

            compiled = 1 + sum("compile" in part for part in self._hybrid_parts)
            logger.info(
                "Step 2/2: Quantizing and compiling for Axelera Metis (hybrid, %d compiled "
                "parts; the transformer runs on the host)",
                compiled,
            )
            return export_axelera_parts(
                parts=[
                    {
                        "compile": onnx_path,
                        "reference": self._raw_wrapper,
                        "input": "images",
                        **(
                            {"graph_passes": list(self._hybrid_first_passes)}
                            if getattr(self, "_hybrid_first_passes", None)
                            else {}
                        ),
                    },
                    *self._hybrid_parts,
                ],
                output_path=output_path,
                stem=Path(output_path).name.removesuffix(self.suffix) or "model",
                metadata=metadata,
                calibration=calibration_data,
                quantization_scheme=quantization_scheme,
                calibration_info=getattr(self, "_calibration_info", {}),
                compiler_overrides=getattr(self, "_compiler_overrides", {}),
                devkit_version=getattr(self, "_devkit_version", None),
                verify=bool(verify),
                verify_samples=self._verify_samples(dummy.shape[-2:]) if verify else None,
                host_arrays=getattr(self, "_hybrid_host_arrays", None),
            )
        if getattr(self, "_chain_tail", None) is not None:
            from .axelera import export_axelera_chain

            logger.info("Step 2/2: Quantizing and compiling for Axelera Metis (2 parts)")
            return export_axelera_chain(
                onnx_paths=[onnx_path, self._chain_tail_onnx],
                output_path=output_path,
                stem=Path(output_path).name.removesuffix(self.suffix) or "model",
                metadata=metadata,
                calibration=calibration_data,
                quantization_scheme=quantization_scheme,
                calibration_info=getattr(self, "_calibration_info", {}),
                compiler_overrides=getattr(self, "_compiler_overrides", {}),
                devkit_version=getattr(self, "_devkit_version", None),
                references=[self._raw_wrapper, self._chain_tail],
                host_step={
                    "host": "l2norm",
                    "input": "conv4_3",
                    "output": "conv4_3_norm",
                    "scale": list(self._raw_wrapper.l2norm_scale),
                },
                verify=bool(verify),
                verify_samples=self._verify_samples(dummy.shape[-2:]) if verify else None,
            )
        logger.info("Step 2/2: Quantizing and compiling for Axelera Metis")
        return export_axelera(
            onnx_path=onnx_path,
            output_path=output_path,
            stem=Path(output_path).name.removesuffix(self.suffix) or "model",
            metadata=metadata,
            calibration=calibration_data,
            quantization_scheme=quantization_scheme,
            calibration_info=getattr(self, "_calibration_info", {}),
            compiler_overrides=getattr(self, "_compiler_overrides", {}),
            devkit_version=getattr(self, "_devkit_version", None),
            verify=bool(verify),
            reference_model=self._raw_wrapper,
            verify_samples=self._verify_samples(dummy.shape[-2:]) if verify else None,
            verify_source=getattr(self, "_verify_source", "images"),
            graph_passes=_AXELERA_GRAPH_PASSES.get(self.model._get_model_name(), ()),
        )

    def _verify_samples(self, imgsz, limit: int = 32):
        """Held-out images for ``verify=True``: never the calibration images.

        Taken from the calibration dataset's val/test split; when that split
        has nothing left (coco128's val is its train split), the bundled
        sample images are used and the report says so.
        """
        import cv2
        import numpy as np

        from ..data.utils import get_img_files, load_data_config

        used = getattr(self, "_calibration_files", set())
        files: list[Path] = []
        try:
            config = load_data_config(self._calibration_info["data"], autodownload=False)
            root = str(config.get("path", "."))
            for split in ("val", "test"):
                listed = config.get(f"{split}_img_files")
                if listed is None and config.get(split):
                    listed = get_img_files(config[split], prefix=root)
                files.extend(Path(f) for f in (listed or []) if str(Path(f)) not in used)
        except Exception as exc:  # noqa: BLE001 - verification must not fail the export
            logger.warning("Axelera verify: could not list held-out images: %s", exc)
        files = sorted(dict.fromkeys(files))[:limit]
        source = "held-out dataset images"
        if not files:
            assets = Path(__file__).resolve().parents[1] / "assets"
            files = sorted(p for p in assets.glob("*.jpg") if not p.stem.endswith("_result"))
            source = "bundled sample images (no held-out dataset images)"
        preprocess = self._calibration_preprocess()
        size = tuple(int(v) for v in imgsz)
        samples = []
        for path in files:
            image = cv2.imread(str(path))
            if image is None:
                continue
            chw, _ = preprocess(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size)
            samples.append(np.ascontiguousarray(chw[None], dtype=np.float32))
        self._verify_source = source
        return samples


def _single_image_samples(loader, count: int):
    """Yield ``count`` calibration images one at a time as ``[1, 3, H, W]``.

    The loader pads its last batch by repeating images; those repeats are
    dropped so each image weighs once in the quantization statistics.
    """
    import numpy as np

    remaining = count
    for batch in loader:
        for image in batch:
            if remaining <= 0:
                return
            remaining -= 1
            yield np.ascontiguousarray(image[None], dtype=np.float32)


class NcnnExporter(BaseExporter):
    format_name = "ncnn"
    suffix = "_ncnn"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        meta = super()._build_metadata(precision, dynamic, onnx_path, imgsz=imgsz)
        meta["dynamic"] = False
        meta.pop("exported_from", None)
        return meta

    def _export(
        self, nn_model, dummy, *, output_path, metadata, half, opset, simplify, **kwargs
    ):
        from .ncnn import export_ncnn

        logger.info("Exporting to ncnn via PNNX")
        return export_ncnn(
            nn_model,
            dummy,
            output_path=output_path,
            half=half,
            opset=opset,
            simplify=simplify,
            metadata=metadata,
        )


class TFLiteExporter(BaseExporter):
    format_name = "tflite"
    suffix = ".tflite"
    requires_onnx = True
    supports_int8 = True
    supports_fp16 = False
    apply_model_half = False
    # Like ONNX INT8 and the YOLO export convention, int8=True without data=
    # calibrates on the default dataset and warns that it is not representative.
    default_int8_calibration_data = True

    def __call__(self, *args, dynamic: bool = False, **kwargs) -> str:
        if dynamic:
            raise ValueError("TFLite export requires static input shapes.")
        if kwargs.get("int8") and int(kwargs.get("batch") or 1) != 1:
            # onnx2tf feeds the representative dataset one image at a time.
            raise ValueError("TFLite INT8 export requires batch=1.")
        from .tflite import ensure_tflite_family_supported

        ensure_tflite_family_supported(
            self.model._get_model_name(),
            getattr(self.model, "task", "detect"),
        )
        return super().__call__(*args, dynamic=False, **kwargs)

    def _validate(self, half: bool, int8: bool, data: Optional[str]):
        if half:
            raise ValueError(
                "TFLite FP16 export is not supported yet. Omit half=True for FP32."
            )
        return super()._validate(half, int8, data)

    def _preflight(self, **kwargs):
        from .tflite import (
            TFLITE_INT8_EXPORTS,
            check_tflite_converter_version,
            check_tflite_export_available,
            check_tflite_int8_available,
        )

        # Validate the INT8 options before anything can touch the live model
        # (e.g. merging LoRA adapters), so a rejected request leaves it as is.
        int8_backend = kwargs.get("int8_backend", "converter")
        int8_output_dtype = kwargs.get("int8_output_dtype", "int8")
        if int8_backend not in {"converter", "onnx2tf"}:
            raise ValueError(
                f"int8_backend must be 'converter' or 'onnx2tf', got {int8_backend!r}"
            )
        if int8_output_dtype not in {"int8", "float32"}:
            raise ValueError(
                "int8_output_dtype must be 'int8' or 'float32', "
                f"got {int8_output_dtype!r}"
            )
        if (
            kwargs.get("int8")
            and int8_backend == "onnx2tf"
            and int8_output_dtype != "int8"
        ):
            raise ValueError(
                "int8_backend='onnx2tf' always writes int8 input and output tensors; "
                "int8_output_dtype='float32' requires the default in-process backend."
            )

        family = self.model._get_model_name()
        task = getattr(self.model, "task", "detect")
        self._split_outputs = False
        if kwargs.get("int8"):
            if (family, task) not in TFLITE_INT8_EXPORTS:
                supported = ", ".join(f"{f} {t}" for f, t in sorted(TFLITE_INT8_EXPORTS))
                raise NotImplementedError(
                    f"TFLite INT8 export currently supports: {supported}. "
                    f"Got model family {family!r}, task {task!r}."
                )
            self._split_outputs = True
        super()._preflight(**kwargs)
        check_tflite_export_available()
        check_tflite_converter_version(family, task)
        if kwargs.get("int8"):
            check_tflite_int8_available()

    def _export_intermediate_onnx(
        self, nn_model, dummy, output_path, opset, simplify, dynamic
    ):
        if getattr(self, "_split_outputs", False):
            wrapper = {
                "yolox": _YOLOXSplitOutputWrapper,
                "yolo9": _YOLO9SplitOutputWrapper,
            }[self.model._get_model_name()]
            nn_model = wrapper(nn_model, dummy.shape[-2:]).eval()
        return super()._export_intermediate_onnx(
            nn_model, dummy, output_path, opset, simplify, dynamic
        )

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        from .tflite import SPLIT_OUTPUT_LAYOUT

        meta = super()._build_metadata(precision, dynamic, onnx_path, imgsz=imgsz)
        if getattr(self, "_split_outputs", False):
            meta["output_layout"] = SPLIT_OUTPUT_LAYOUT
        return meta

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        onnx_path,
        half,
        int8,
        calibration_data,
        verbose,
        onnx2tf_args=None,
        int8_backend: str = "converter",
        int8_output_dtype: str = "int8",
        **kwargs,
    ):
        from .tflite import ensure_tflite_family_supported, export_tflite

        ensure_tflite_family_supported(
            metadata.get("model_family") if metadata else None,
            metadata.get("task") if metadata else None,
        )

        logger.info("Step 2/2: Converting to TensorFlow Lite")
        return export_tflite(
            onnx_path=onnx_path,
            output_path=output_path,
            half=half,
            int8=int8,
            calibration_data=calibration_data,
            verbose=verbose,
            onnx2tf_args=onnx2tf_args,
            metadata=metadata,
            int8_backend=int8_backend,
            int8_output_dtype=int8_output_dtype,
        )


class CoreAIExporter(BaseExporter):
    """Apple Core AI (``.aimodel``) export via ``torch.export``.

    Unlike the Core ML path this uses a real graph capture rather than a
    single recorded trace, so the static-eval monkey patches that path needs
    are not required here. Artifacts are static-shape in v1 and declare a
    minimum OS of v27, which is the only value the toolchain offers.
    """

    format_name = "coreai"
    suffix = ".aimodel"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False
    supports_embedded_nms = False

    def __call__(self, *, dynamic: bool = False, **kwargs) -> str:
        """Export a fixed-canvas Core AI artifact.

        The base exporter defaults ``dynamic=True`` for ONNX. Core AI has no
        dynamic-shape contract in v1, so its format-specific default is false
        and an explicit request is rejected rather than mislabeled.
        """
        if dynamic:
            raise NotImplementedError(
                "Core AI export uses a fixed input shape; dynamic=True is not "
                "supported."
            )
        return super().__call__(dynamic=False, **kwargs)

    def _preflight(self, **kwargs):
        # Support policy is checked before optional dependencies, as required
        # by ADR 0011. Dependency validation still happens before the
        # destructive LoRA merge in BaseExporter.__call__.
        super()._preflight(**kwargs)

        # Check the optional dependency HERE, not at conversion time. Preflight
        # runs before __call__ merges any live LoRA adapters, and that merge is
        # destructive. Discovering the missing package afterwards would leave
        # the caller's model permanently modified with no artifact to show for
        # it.
        from .coreai import _require_coreai

        _require_coreai()

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        # v1 artifacts are fixed-canvas, mirroring the CoreML/NCNN overrides.
        meta = super()._build_metadata(precision, dynamic, onnx_path, imgsz=imgsz)
        meta["dynamic"] = False
        return meta

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        precision,
        metadata,
        **kwargs,
    ):
        from .coreai import export_coreai

        return export_coreai(
            nn_model,
            dummy,
            output_path=output_path,
            precision=precision,
            metadata=metadata,
            model_family=self.model._get_model_name(),
        )


class CoreMLExporter(BaseExporter):
    format_name = "coreml"
    suffix = ".mlpackage"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = True
    apply_model_half = False  # ct.convert handles precision via compute_precision
    supports_embedded_nms = True
    _extra_export_kwargs = BaseExporter._extra_export_kwargs | {"max_det"}

    def _preflight(self, *, half: bool, int8: bool, data: Optional[str], **kwargs):
        if kwargs.get("nms"):
            family = self.model._get_model_name()
            task = getattr(self.model, "task", "detect")
            if not isinstance(task, str):
                task = "detect"
            if family == "yolo9" and task != "detect":
                raise NotImplementedError(
                    "CoreML embedded NMS currently supports YOLO9 detection "
                    "models only."
                )
            if family not in {"yolox", "yolo9"}:
                raise NotImplementedError(
                    "CoreML embedded NMS currently supports YOLOX and YOLO9 "
                    "detection models only."
                )
            if kwargs.get("max_det", 300) != 300:
                raise NotImplementedError(
                    "CoreML embedded NMS does not support max_det. "
                    "Use ONNX embedded NMS when max_det control is required."
                )
        super()._preflight(half=half, int8=int8, data=data, **kwargs)

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        # CoreML uses a hard-fixed ct.ImageType(shape=...); the exported graph
        # has a fixed input shape, so report dynamic=False regardless of the
        # requested flag (mirrors the NCNN override).
        meta = super()._build_metadata(precision, dynamic, onnx_path, imgsz=imgsz)
        meta["dynamic"] = False
        return meta

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        precision,
        metadata,
        compute_units="all",
        nms=False,
        iou=0.45,
        conf=0.25,
        **kwargs,
    ):
        from .coreml import export_coreml

        return export_coreml(
            nn_model,
            dummy,
            output_path=output_path,
            precision=precision,
            compute_units=compute_units,
            nms=nms,
            iou=iou,
            conf=conf,
            metadata=metadata,
            model_family=self.model._get_model_name(),
        )

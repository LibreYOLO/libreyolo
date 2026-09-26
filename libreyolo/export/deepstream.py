"""DeepStream-ready ONNX export: output wrapper + nvinfer config generation.

DeepStream's ``nvinfer`` element consumes ONNX directly (it builds the
TensorRT engine on device) but needs a custom bounding-box parser for
non-standard detector heads. The community-standard parser
(`marcoslucianops/DeepStream-Yolo <https://github.com/marcoslucianops/DeepStream-Yolo>`_,
MIT) expects a single output tensor of shape ``(batch, num_detections, 6)``
where each row is ``[x1, y1, x2, y2, score, class_id]`` in network-input
pixel coordinates. The parser applies the confidence threshold; suppression
runs in DeepStream's clustering stage (``cluster-mode=2`` + NMS IoU), so no
NMS is embedded in the graph.

This module provides:

- Graph adapters mapping each LibreYOLO detector family's export output to
  that contract (layout only; detection math is unchanged and verified by
  parity tests against the family's native postprocess). Families whose
  native preprocessing cannot be expressed by nvinfer's
  ``scale * (x - offsets)`` (per-channel std division) get the
  normalization baked into the DeepStream graph instead.
- ``write_deepstream_sidecars``, which generates a ready-to-use
  ``config_infer_primary_*.txt`` and ``labels.txt`` next to the exported
  ONNX, with preprocessing constants matching the family's native pipeline.

The tensor-contract description above is derived from the MIT-licensed
DeepStream-Yolo parser sources (nvdsinfer_custom_impl_Yolo); no code from
that project is included here.

Known preprocessing approximations (documented in the DeepStream guide):

- Letterbox families pad with gray (114/128) natively; nvinfer pads black.
- YOLO-NAS detection natively resizes the longest side to 636 inside a
  640 canvas; nvinfer's ``maintain-aspect-ratio`` scales to the full 640.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

# Families emitting the shared raw tensor ``(B, 4 + nc, N)``: xyxy boxes in
# input pixels then per-class scores (yolo9 head export; darknet/yolo7
# export wrappers bake obj*cls into the class scores).
_RAW_CHANNELS_FIRST_FAMILIES = {
    "yolo9",
    "yolo9_p2",
    "yolo9_e2e",
    "yolo1",
    "yolo2",
    "yolo3",
    "yolo4",
    "yolo7",
}

# Heads that emit at most one prediction per object (one-to-one assignment
# or Hungarian matching). DeepStream must not cluster their output.
_NMS_FREE_FAMILIES = {"yolo9_e2e"}

# Same semantics but already ``(B, N, 4 + nc)`` (rtmdet/picodet heads).
_RAW_CHANNELS_LAST_FAMILIES = {"rtmdet", "picodet"}

# YOLOX head export: ``(B, N, 5 + nc)`` — cxcywh input-pixel boxes,
# sigmoid objectness at channel 4, sigmoid class scores after (not
# multiplied in-graph).
_YOLOX_FAMILIES = {"yolox"}

# YOLO-NAS tracing export: two tensors ``(boxes (B, N, 4) xyxy pixels,
# scores (B, N, nc) sigmoid)``.
_TWO_TENSOR_FAMILIES = {"yolonas"}

# DETR-style families exporting ``(pred_logits, pred_boxes)`` with
# cxcywh boxes normalized to [0, 1] and raw (pre-sigmoid) logits.
_DETR_TUPLE_FAMILIES = {
    "dfine",
    "deim",
    "deimv2",
    "ec",
    "rtdetr",
    "rtdetrv2",
    "rtdetrv4",
}

# Families exporting ``(boxes, logits)`` in RF-DETR order (boxes first).
_BOXES_FIRST_DETR_FAMILIES = {"rfdetr"}

# Classification backbones exportable as an nvinfer classifier
# (``network-type=1``). They emit raw logits; the adapter softmaxes.
_CLASSIFY_FAMILIES = {
    "mobilenetv4",
    "convnext",
    "efficientnetv2",
    "resnet",
    "dinov2",
}

# Semantic segmentation families exportable as an nvinfer segmentation
# network (``network-type=2``). These normalize inside their own forward,
# so the DeepStream graph feeds them plain [0, 1] RGB and adds no
# normalization of its own. ``segformer`` is absent because it is not wired
# to the shared semantic export contract and cannot export to ONNX at all.
_SEMANTIC_FAMILIES = {
    "pidnet",
    "eomt",
    "dinov2",
    "lingbotvision",
}

# Instance segmentation families exportable as an nvinfer instance-mask
# network (``network-type=3``). All are DETR-style and export per-query
# masks as a third output. RTMDet-Ins and YOLO9 are absent: their seg
# export is blocked upstream in libreyolo.export.support.
_INSTANCE_SEG_FAMILIES = {"rfdetr", "dfine", "ec"}

# Depth families exported as a raw-tensor network (``network-type=100``
# with ``output-tensor-meta=1``): DeepStream has no depth post-processor,
# so the application reads the dense map from the tensor metadata. Both
# normalize inside their own forward, so the graph takes [0, 1] RGB.
# ``depth_anything3`` is absent: it has no export implementation at all
# (out of scope per ADR 0006).
_DEPTH_FAMILIES = {"depth_anything", "zipdepth"}

# Tasks DeepStream has no post-processor for. They export as raw-tensor
# networks (``network-type=100`` with ``output-tensor-meta=1``): the graph
# passes its native outputs through untouched and the application decodes
# them from the tensor metadata. Multi-output graphs are fine, every output
# layer reaches the metadata.
_POSE_FAMILIES = {"yolo9", "yolonas", "rfdetr", "ec"}
_RESTORE_FAMILIES = {"nafnet", "quicksrnet", "realesrgan", "swinir"}
_MATTE_FAMILIES = {"birefnet"}
_GAZE_FAMILIES = {"l2cs"}

_RAW_TENSOR_TASKS = {
    "depth": _DEPTH_FAMILIES,
    "pose": _POSE_FAMILIES,
    "restore": _RESTORE_FAMILIES,
    "matte": _MATTE_FAMILIES,
    "gaze": _GAZE_FAMILIES,
}


def _raw_tensor_families(task: str) -> set[str]:
    """Families exportable as a raw-tensor network for ``task``."""
    return _RAW_TENSOR_TASKS.get(task, set())


# Raw-tensor families that normalize with ImageNet stats *outside* the
# model, so the DeepStream graph must do it (nvinfer cannot divide per
# channel). Restoration families take plain [0, 1] and SwinIR subtracts its
# own mean inside forward, so neither appears here.
_RAW_TENSOR_IMAGENET_NORM = {"birefnet", "l2cs"}

# EoMT's "semantic_logits" are already per-pixel probabilities
# (``class.softmax() @ mask.sigmoid()``), so a second softmax would distort
# the values ``segmentation-threshold`` compares against.
_SEMANTIC_ALREADY_PROBABILITIES = {"eomt"}

# ImageNet constants, in [0, 1] units. nvinfer's preprocessing is
# ``scale * (x - offsets)`` with a scalar scale, which cannot express
# per-channel std division, so families normalizing with ImageNet stats
# get the mean/std baked into the DeepStream graph instead (input arrives
# from nvinfer as RGB in [0, 1] via net-scale-factor=1/255).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# In-graph normalization constants for families whose native pipeline
# normalizes in 0-255 space with unequal per-channel std. Keys are the
# family name; values are ((mean), (std)) in the family's channel order
# (rtmdet consumes BGR frames; picodet consumes RGB).
_GRAPH_NORM_0_255 = {
    "rtmdet": ((103.53, 116.28, 123.675), (57.375, 57.12, 58.395)),
    "picodet": ((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
}


def _uses_imagenet_norm(model_family: str, model_size: str | None) -> bool:
    """Whether the family's native preprocess applies ImageNet mean/std.

    These families receive [0, 1] input from nvinfer and normalize
    in-graph.
    """
    if model_family in {"rfdetr", "ec"}:
        return True
    if model_family == "deimv2":
        # DINO-backboned sizes normalize; the HGNetv2 tiny sizes do not.
        return model_size in {"s", "m", "l", "x"}
    return False


class _GraphNorm(nn.Module):
    """Bake ``(x - mean) / std`` into the graph ahead of the model."""

    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer(
            "_mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor):
        return self.model((x - self._mean.to(x.dtype)) / self._std.to(x.dtype))


def _maybe_normalize(
    nn_model: nn.Module, model_family: str, model_size: str | None
) -> nn.Module:
    """Bake the family's normalization into the graph when nvinfer can't do it."""
    if model_family in _GRAPH_NORM_0_255:
        mean, std = _GRAPH_NORM_0_255[model_family]
        return _GraphNorm(nn_model, mean, std)
    if _uses_imagenet_norm(model_family, model_size):
        return _GraphNorm(nn_model, _IMAGENET_MEAN, _IMAGENET_STD)
    return nn_model


def _rows_from(boxes, scores_all):
    """Concat max-class rows: ``[x1, y1, x2, y2, best_score, best_class]``."""
    scores, labels = scores_all.max(dim=-1)
    return torch.cat(
        (
            boxes,
            scores.unsqueeze(-1),
            labels.unsqueeze(-1).to(boxes.dtype),
        ),
        dim=-1,
    )


class DeepStreamRawOutput(nn.Module):
    """Adapt the shared raw detector tensor to the DeepStream parser layout.

    ``channels_first=True`` wraps ``(B, 4 + nc, N)`` output (transposed
    in-graph); ``False`` wraps ``(B, N, 4 + nc)``. Boxes are xyxy in input
    pixels, scores per-class (objectness already folded in where the family
    has one).
    """

    def __init__(self, model: nn.Module, channels_first: bool = True):
        super().__init__()
        self.model = model
        self.channels_first = channels_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.model(x)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        pred = raw.transpose(1, 2) if self.channels_first else raw
        return _rows_from(pred[..., :4].float(), pred[..., 4:].float())


class DeepStreamYOLOXOutput(nn.Module):
    """Adapt YOLOX export output ``(B, N, 5 + nc)`` to the parser layout.

    Boxes are cxcywh in input pixels; channel 4 is sigmoid objectness and
    the remaining channels are sigmoid class scores. Scores are
    ``obj * cls`` (matching the native postprocess) and boxes converted to
    xyxy.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.model(x)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        raw = raw.float()
        cxcy = raw[..., :2]
        half_wh = raw[..., 2:4] * 0.5
        boxes = torch.cat((cxcy - half_wh, cxcy + half_wh), dim=-1)
        scores_all = raw[..., 5:] * raw[..., 4:5]
        return _rows_from(boxes, scores_all)


class DeepStreamTwoTensorOutput(nn.Module):
    """Adapt two-tensor exports ``(boxes, scores)`` (YOLO-NAS).

    YOLO-NAS switches on ``torch.jit.is_tracing()``: the traced graph returns
    the decoded ``(boxes, scores)`` pair directly, while an eager call
    returns ``(decoded, raw_training_outputs)``. Unwrap the nested form so
    the adapter behaves identically in both.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out[0], (tuple, list)):
            out = out[0]
        boxes, scores_all = out[0], out[1]
        return _rows_from(boxes.float(), scores_all.float())


class DeepStreamDETROutput(nn.Module):
    """Adapt DETR-style ``(logits, boxes)`` outputs to the parser layout.

    ``boxes`` are cxcywh normalized to [0, 1]; ``logits`` are raw class
    logits scored with sigmoid (matching the family postprocess). Output is
    ``(B, Q, 6)`` rows of ``[x1, y1, x2, y2, best_score, best_class]`` in
    input-pixel coordinates.

    Args:
        model: Export-mode model (family export wrapper applied).
        imgsz: Export canvas ``(height, width)`` used to scale boxes.
        boxes_first: True for families returning ``(boxes, logits)``
            (RF-DETR order) instead of ``(logits, boxes)``.
    """

    def __init__(
        self,
        model: nn.Module,
        imgsz: tuple[int, int],
        boxes_first: bool,
    ):
        super().__init__()
        self.model = model
        self.boxes_first = boxes_first
        h, w = imgsz
        self.register_buffer(
            "_scale",
            torch.tensor([w, h, w, h], dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if self.boxes_first:
            boxes_norm, logits = out[0], out[1]
        else:
            logits, boxes_norm = out[0], out[1]
        scores_all = logits.float().sigmoid()
        cxcy = boxes_norm[..., :2].float()
        half_wh = boxes_norm[..., 2:4].float() * 0.5
        boxes = torch.cat((cxcy - half_wh, cxcy + half_wh), dim=-1) * self._scale
        return _rows_from(boxes, scores_all)


class DeepStreamClassifierOutput(nn.Module):
    """Softmax a classifier's logits for ``nvinfer``'s classifier mode.

    DeepStream's built-in classifier parser reads the output tensor as
    per-class probabilities and applies ``classifier-threshold``, so the
    softmax must live in the graph. Output stays ``(B, num_classes)``.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return torch.softmax(logits.float(), dim=-1)


class DeepStreamSemanticOutput(nn.Module):
    """Emit ``(B, C, H, W)`` per-class probabilities for segmentation mode.

    ``nvinfer``'s segmentation post-processing applies
    ``segmentation-threshold`` to these values and argmaxes into a class
    map. Argmax is unchanged by softmax; the transform exists so the
    threshold compares against real probabilities. Families whose head
    already outputs probabilities pass through unchanged.
    """

    def __init__(self, model: nn.Module, apply_softmax: bool = True):
        super().__init__()
        self.model = model
        self.apply_softmax = apply_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = logits.float()
        return torch.softmax(logits, dim=1) if self.apply_softmax else logits


class DeepStreamInstanceSegOutput(nn.Module):
    """Adapt DETR-style instance segmentation to the seg parser layout.

    The seg parser (``NvDsInferParseYoloSeg`` from DeepStream-Yolo-Seg, MIT)
    reads one tensor of ``(B, N, 6 + mask_size)`` rows: the usual
    ``[x1, y1, x2, y2, score, class]`` followed by that detection's mask
    flattened at exactly ``(netH / 4, netW / 4)`` — the parser hardcodes
    that resolution — as probabilities for ``segmentation-threshold``.

    LibreYOLO's seg families export per-query masks directly rather than
    prototype coefficients, so the adapter resizes them to the quarter
    canvas and sigmoids, with no RoI pooling or custom TensorRT plugin.
    """

    def __init__(
        self,
        model: nn.Module,
        imgsz: tuple[int, int],
        boxes_first: bool,
    ):
        super().__init__()
        self.model = model
        self.boxes_first = boxes_first
        h, w = imgsz
        self.mask_h = h // 4
        self.mask_w = w // 4
        self.register_buffer(
            "_scale",
            torch.tensor([w, h, w, h], dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if self.boxes_first:
            boxes_norm, logits, masks = out[0], out[1], out[2]
        else:
            logits, boxes_norm, masks = out[0], out[1], out[2]

        scores_all = logits.float().sigmoid()
        cxcy = boxes_norm[..., :2].float()
        half_wh = boxes_norm[..., 2:4].float() * 0.5
        boxes = torch.cat((cxcy - half_wh, cxcy + half_wh), dim=-1) * self._scale
        rows = _rows_from(boxes, scores_all)

        # (B, Q, mh, mw) -> quarter-canvas probabilities, flattened per query.
        masks = masks.float()
        b, q = masks.shape[0], masks.shape[1]
        masks = torch.nn.functional.interpolate(
            masks,
            size=(self.mask_h, self.mask_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        return torch.cat((rows, masks.reshape(b, q, self.mask_h * self.mask_w)), dim=-1)


def _build_deepstream_wrapper(
    nn_model: nn.Module,
    *,
    model_family: str,
    imgsz: tuple[int, int],
    model_size: str | None = None,
    task: str = "detect",
) -> nn.Module:
    """Pick the DeepStream output adapter for ``model_family``/``task``."""
    if task == "classify":
        if model_family not in _CLASSIFY_FAMILIES:
            raise NotImplementedError(
                f"DeepStream classifier export is not supported for family "
                f"{model_family!r}. Supported: {sorted(_CLASSIFY_FAMILIES)}"
            )
        # Classify transforms normalize with ImageNet stats outside the
        # model, and nvinfer cannot divide per channel: bake it in.
        return DeepStreamClassifierOutput(
            _GraphNorm(nn_model, _IMAGENET_MEAN, _IMAGENET_STD)
        )
    if task in _RAW_TENSOR_TASKS:
        allowed = _raw_tensor_families(task)
        if model_family not in allowed:
            raise NotImplementedError(
                f"DeepStream {task} export is not supported for family "
                f"{model_family!r}. Supported: {sorted(allowed)}"
            )
        # Outputs pass through untouched: DeepStream does not interpret
        # them, the application does. Only preprocessing the graph must own
        # is added.
        if model_family in _RAW_TENSOR_IMAGENET_NORM:
            return _GraphNorm(nn_model, _IMAGENET_MEAN, _IMAGENET_STD)
        if task == "pose" and _uses_imagenet_norm(model_family, model_size):
            return _GraphNorm(nn_model, _IMAGENET_MEAN, _IMAGENET_STD)
        return nn_model
    if task == "segment":
        if model_family not in _INSTANCE_SEG_FAMILIES:
            raise NotImplementedError(
                f"DeepStream instance segmentation export is not supported for "
                f"family {model_family!r}. Supported: "
                f"{sorted(_INSTANCE_SEG_FAMILIES)}"
            )
        return DeepStreamInstanceSegOutput(
            _maybe_normalize(nn_model, model_family, model_size),
            imgsz,
            boxes_first=model_family in _BOXES_FIRST_DETR_FAMILIES,
        )
    if task == "semantic":
        if model_family not in _SEMANTIC_FAMILIES:
            raise NotImplementedError(
                f"DeepStream segmentation export is not supported for family "
                f"{model_family!r}. Supported: {sorted(_SEMANTIC_FAMILIES)}"
            )
        # Semantic nets normalize inside their own forward; adding a graph
        # normalization here would apply ImageNet stats twice.
        return DeepStreamSemanticOutput(
            nn_model,
            apply_softmax=model_family not in _SEMANTIC_ALREADY_PROBABILITIES,
        )
    nn_model = _maybe_normalize(nn_model, model_family, model_size)

    if model_family in _RAW_CHANNELS_FIRST_FAMILIES:
        return DeepStreamRawOutput(nn_model, channels_first=True)
    if model_family in _RAW_CHANNELS_LAST_FAMILIES:
        return DeepStreamRawOutput(nn_model, channels_first=False)
    if model_family in _YOLOX_FAMILIES:
        return DeepStreamYOLOXOutput(nn_model)
    if model_family in _TWO_TENSOR_FAMILIES:
        return DeepStreamTwoTensorOutput(nn_model)
    if model_family in _DETR_TUPLE_FAMILIES:
        return DeepStreamDETROutput(nn_model, imgsz, boxes_first=False)
    if model_family in _BOXES_FIRST_DETR_FAMILIES:
        return DeepStreamDETROutput(nn_model, imgsz, boxes_first=True)
    raise NotImplementedError(
        f"DeepStream export is not supported for model family {model_family!r}. "
        f"Supported families: {sorted(deepstream_supported_families())}"
    )


def wrap_for_deepstream(
    nn_model: nn.Module,
    *,
    model_family: str,
    imgsz: tuple[int, int],
    model_size: str | None = None,
    task: str = "detect",
) -> nn.Module:
    """Wrap an export-mode model with the DeepStream output adapter."""
    wrapped = _build_deepstream_wrapper(
        nn_model,
        model_family=model_family,
        imgsz=imgsz,
        model_size=model_size,
        task=task,
    )
    if wrapped is nn_model:
        return wrapped
    # The adapters hold constant buffers (_scale, _mean, _std) built on CPU,
    # while the model being exported may already sit on CUDA. Line them up or
    # tracing dies on a device mismatch.
    params = next(nn_model.parameters(), None)
    if params is not None:
        wrapped = wrapped.to(params.device)
    return wrapped


def deepstream_supported_families(task: str = "detect") -> set[str]:
    """Families accepted by :func:`wrap_for_deepstream` for ``task``."""
    if task == "classify":
        return set(_CLASSIFY_FAMILIES)
    if task == "semantic":
        return set(_SEMANTIC_FAMILIES)
    if task == "segment":
        return set(_INSTANCE_SEG_FAMILIES)
    if task in _RAW_TENSOR_TASKS:
        return set(_raw_tensor_families(task))
    if task != "detect":
        return set()
    return (
        _RAW_CHANNELS_FIRST_FAMILIES
        | _RAW_CHANNELS_LAST_FAMILIES
        | _YOLOX_FAMILIES
        | _TWO_TENSOR_FAMILIES
        | _DETR_TUPLE_FAMILIES
        | _BOXES_FIRST_DETR_FAMILIES
    )


def deepstream_supported_tasks() -> set[str]:
    """Tasks with at least one DeepStream-supported family."""
    candidates = {"detect", "classify", "segment", "semantic"} | set(_RAW_TENSOR_TASKS)
    return {task for task in candidates if deepstream_supported_families(task)}


def deepstream_uses_raw_outputs(task: str | None) -> bool:
    """Whether DeepStream preserves the task's native ONNX output schema."""
    return task in _RAW_TENSOR_TASKS


# --- nvinfer sidecar generation -------------------------------------------

# Per-family nvinfer preprocessing profile. nvinfer computes
# ``y = net-scale-factor * (x - offsets)`` per channel on the input frame;
# combined with any in-graph normalization above, this must reproduce the
# family's native preprocessing. ``maintain_aspect_ratio`` /
# ``symmetric_padding`` mirror the family's letterbox geometry;
# ``model_color_format`` is 0 for RGB, 1 for BGR.
_PREPROCESS_PROFILES: dict[str, dict] = {
    # Letterbox + /255 RGB. Top-left pad unless the checkpoint stamps
    # letterbox_pad="center" (see write_deepstream_sidecars).
    "yolo9": {"maintain_aspect_ratio": 1},
    "yolo9_p2": {"maintain_aspect_ratio": 1},
    "yolo9_e2e": {"maintain_aspect_ratio": 1},
    "yolo7": {"maintain_aspect_ratio": 1},
    "yolo2": {"maintain_aspect_ratio": 1},
    "yolo3": {"maintain_aspect_ratio": 1},
    "yolo4": {"maintain_aspect_ratio": 1},
    # YOLOv1's dense head needs the plain stretch square.
    "yolo1": {},
    # YOLO-NAS detection letterboxes centered (native resize-to-636
    # approximated). Pose overrides the color and padding geometry below.
    "yolonas": {"maintain_aspect_ratio": 1, "symmetric_padding": 1},
    # YOLOX: BGR frames in raw 0-255 space, letterboxed.
    "yolox": {
        "net_scale_factor": 1.0,
        "model_color_format": 1,
        "maintain_aspect_ratio": 1,
    },
    # RTMDet: BGR 0-255 with in-graph mean/std, letterboxed.
    "rtmdet": {
        "net_scale_factor": 1.0,
        "model_color_format": 1,
        "maintain_aspect_ratio": 1,
    },
    # PicoDet: RGB 0-255 with in-graph mean/std, stretch resize.
    "picodet": {"net_scale_factor": 1.0},
    # DETR families: stretch resize, /255 RGB (ImageNet stats in-graph
    # where the family uses them).
    "rfdetr": {},
    "dfine": {},
    "deim": {},
    "deimv2": {},
    "ec": {},
    "rtdetr": {},
    "rtdetrv2": {},
    "rtdetrv4": {},
    # Classification: ImageNet mean/std baked into the graph, so nvinfer
    # just scales to [0, 1]. nvinfer stretches the frame/ROI to the network
    # input; the native transform resizes the shortest side then centre-
    # crops, so tight-ROI framing differs slightly (documented).
    "mobilenetv4": {},
    "convnext": {},
    "efficientnetv2": {},
    "resnet": {},
    "dinov2": {},
    # Semantic segmentation: nets normalize internally, so [0, 1] RGB only.
    # pidnet letterboxes natively; the rest stretch.
    "pidnet": {"maintain_aspect_ratio": 1},
    # Depth: nets normalize internally, so [0, 1] RGB; all stretch-resize.
    "depth_anything": {},
    "zipdepth": {},
    # Raw-tensor tasks. yolo9-pose letterboxes RGB as in detection; the rest
    # stretch-resize RGB unless overridden below.
    "nafnet": {},
    "realesrgan": {},
    "quicksrnet": {},
    "swinir": {},
    "birefnet": {},
    "l2cs": {},
    "eomt": {},
    "lingbotvision": {},
}

# A family can have task-specific preprocessing. YOLO-NAS pose consumes BGR
# and pads on the bottom/right at the full 640 resize, unlike YOLO-NAS
# detection's RGB, centered, resize-to-636 contract.
_TASK_PREPROCESS_PROFILES: dict[tuple[str, str], dict] = {
    ("yolonas", "pose"): {
        "model_color_format": 1,
        "maintain_aspect_ratio": 1,
        "symmetric_padding": 0,
    },
}


def write_deepstream_sidecars(
    onnx_path: str,
    *,
    model_family: str,
    class_names: list[str],
    imgsz: tuple[int, int],
    batch: int,
    precision: str,
    conf: float = 0.25,
    iou: float = 0.45,
    task: str = "detect",
    letterbox_pad: str | None = None,
) -> tuple[str, str]:
    """Write ``config_infer_primary_<stem>.txt`` and ``<stem>_labels.txt``.

    Returns the ``(config_path, labels_path)`` pair. The config targets the
    MIT DeepStream-Yolo parser library (``NvDsInferParseYolo``); the
    ``custom-lib-path`` is left pointing at the conventional build output of
    that project for the user to adjust. ``letterbox_pad`` is the YOLO9
    checkpoint pad placement; ``"center"`` maps to ``symmetric-padding=1``.
    """
    onnx_file = Path(onnx_path)
    stem = onnx_file.stem
    labels_path = onnx_file.with_name(f"{stem}_labels.txt")
    config_path = onnx_file.with_name(f"config_infer_primary_{stem}.txt")

    # Depth has no classes: no labels file, and no labelfile-path key.
    if class_names:
        labels_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    else:
        labels_path = None

    profile = {
        **_PREPROCESS_PROFILES.get(model_family, {}),
        **_TASK_PREPROCESS_PROFILES.get((model_family, task), {}),
    }
    net_scale = profile.get("net_scale_factor", 1.0 / 255.0)
    offsets = profile.get("offsets")
    maintain_ar = profile.get("maintain_aspect_ratio", 0)
    symmetric_pad = profile.get("symmetric_padding", 0)
    if letterbox_pad is not None and maintain_ar:
        from ..preprocess.letterbox import LETTERBOX_CENTER, normalize_letterbox_pad

        symmetric_pad = int(normalize_letterbox_pad(letterbox_pad) == LETTERBOX_CENTER)
    color_format = profile.get("model_color_format", 0)

    network_mode = {"fp32": 0, "int8": 1, "fp16": 2}.get(precision, 0)
    mode_name = {0: "fp32", 1: "int8", 2: "fp16"}[network_mode]
    # DeepStream-Yolo's custom engine builder serializes detection engines
    # with this fixed basename, regardless of the ONNX filename.
    engine_name = (
        f"model_b{batch}_gpu0_{mode_name}.engine"
        if task == "detect"
        else f"{onnx_file.name}_b{batch}_gpu0_{mode_name}.engine"
    )

    # Heads emitting one prediction per object (DETR Hungarian matching, or
    # YOLO9's one-to-one E2E head) must not be clustered: DeepStream's NMS
    # would merge genuinely distinct detections. Anchor/grid heads need it.
    nms_free = (
        model_family
        in _DETR_TUPLE_FAMILIES | _BOXES_FIRST_DETR_FAMILIES | _NMS_FREE_FAMILIES
    )
    cluster_mode = 4 if nms_free else 2

    common = [
        "[property]",
        "gpu-id=0",
        f"net-scale-factor={net_scale:.10g}",
        f"model-color-format={color_format}",
        f"onnx-file={onnx_file.name}",
        f"model-engine-file={engine_name}",
        *([f"labelfile-path={labels_path.name}"] if labels_path is not None else []),
        f"batch-size={batch}",
        f"network-mode={network_mode}",
        "interval=0",
        "gie-unique-id=1",
        f"infer-dims=3;{imgsz[0]};{imgsz[1]}",
        f"maintain-aspect-ratio={int(maintain_ar)}",
        f"symmetric-padding={int(symmetric_pad)}",
    ]

    if task == "classify":
        # Native nvinfer classifier: no custom parser library needed. Set
        # process-mode=2 and operate-on-gie-id to run it as a secondary
        # classifier behind a detector.
        lines = common + [
            "process-mode=1",
            "network-type=1",
            f"num-detected-classes={len(class_names)}",
            f"classifier-threshold={conf}",
            "classifier-async-mode=0",
        ]
    elif task in _RAW_TENSOR_TASKS:
        # No DeepStream post-processor for this task: hand the raw outputs
        # to the application through tensor metadata.
        lines = common + [
            "process-mode=1",
            "network-type=100",
            "output-tensor-meta=1",
        ]
    elif task == "segment":
        # Instance masks come from the DeepStream-Yolo-Seg parser library,
        # which is a separate build from the detection one.
        lines = common + [
            "process-mode=1",
            "network-type=3",
            f"num-detected-classes={len(class_names)}",
            # DETR-style seg heads emit one query per object: no clustering.
            "cluster-mode=4",
            "output-instance-mask=1",
            f"segmentation-threshold={conf}",
            "parse-bbox-instance-mask-func-name=NvDsInferParseYoloSeg",
            "custom-lib-path=nvdsinfer_custom_impl_Yolo_seg/"
            "libnvdsinfer_custom_impl_Yolo_seg.so",
            "",
            "[class-attrs-all]",
            f"pre-cluster-threshold={conf}",
        ]
    elif task == "semantic":
        # Native nvinfer segmentation: consumes (C, H, W) probabilities and
        # emits a class map in NvDsInferSegmentationMeta.
        lines = common + [
            "process-mode=1",
            "network-type=2",
            f"num-detected-classes={len(class_names)}",
            f"segmentation-threshold={conf}",
            "output-instance-mask=0",
        ]
    else:
        lines = common + [
            "process-mode=1",
            "network-type=0",
            f"num-detected-classes={len(class_names)}",
            f"cluster-mode={cluster_mode}",
            "parse-bbox-func-name=NvDsInferParseYolo",
            "custom-lib-path=nvdsinfer_custom_impl_Yolo/"
            "libnvdsinfer_custom_impl_Yolo.so",
            "engine-create-func-name=NvDsInferYoloCudaEngineGet",
            "",
            "[class-attrs-all]",
            f"pre-cluster-threshold={conf}",
            "topk=300",
        ]
        if not nms_free:
            lines.insert(len(lines) - 1, f"nms-iou-threshold={iou}")

    if offsets is not None:
        lines.insert(3, "offsets=" + ";".join(f"{o:.10g}" for o in offsets))

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(config_path), (str(labels_path) if labels_path is not None else "")

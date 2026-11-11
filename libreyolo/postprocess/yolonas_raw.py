"""Host-side decode for YOLO-NAS, YOLO9, YOLOX and anchor-based YOLO raw head outputs.

Some accelerators run a YOLO-NAS or YOLO9 graph cut right after the head's
last convolutions and leave the decode to the host: the DFL distance decode,
the anchor grid, the sigmoids and the keypoint / rotated-box geometry are
poorly suited to integer hardware. The functions here rebuild, in numpy,
exactly the tensors the decoded LibreYOLO export emits, so the shared
``_parse_yolonas*`` / ``_parse_yolo9`` parsers in ``backends/base.py`` consume
them unchanged.

Each layout is named and versioned (``output_layout`` in export metadata).
Outputs are ordered level by level (stride 8, 16, 32), and within a level in
the order the layout lists:

=============================  ===========================================
``yolonas_raw_heads_v1``       ``reg`` [B, 4*(reg_max+1), H, W],
                               ``cls`` [B, nc, H, W]
``yolonas_pose_raw_heads_v1``  ``reg`` [B, 4*(reg_max+1), H, W],
                               ``cls`` [B, nc + K, H, W] (class logits, then
                               one visibility logit per keypoint),
                               ``pose`` [B, 2K, H, W] (x, y per keypoint)
``yolonas_obb_raw_heads_v1``   ``reg`` [B, 2*(reg_max+1), H, W],
                               ``cls`` [B, nc, H, W],
                               ``offset`` [B, 2, H, W], ``rot`` [B, 1, H, W]
``yolo9_raw_heads_v1``         ``reg`` [B, 4*reg_max, H, W] (YOLO9's
                               ``reg_max`` counts the bins, 0..reg_max-1),
                               ``cls`` [B, nc, H, W]
``yolox_raw_heads_v1``         ``reg`` [B, 4, H, W] (x/y cell offsets, log
                               w/h), ``obj`` [B, 1, H, W], ``cls`` [B, nc, H, W]
``darknet_raw_heads_v1``,      per level and anchor (anchor-major):
``yolo7_raw_heads_v1``         ``reg`` [B, 4A, H, W] (tx, ty, tw, th),
                               ``obj`` [B, A, H, W], ``cls`` [B, A*nc, H, W]
=============================  ===========================================

An export may also store some parts *windowed* (``output_windows`` in export
metadata, ``{part: [low, high]}``): the graph maps that part from
``[low, high]`` onto ``[0, 6]`` and clamps it there, which an integer
accelerator lowers to a ReLU6 fused into the convolution, so the tensor's int8
scale covers only the useful range. :func:`decode_raw_heads` maps windowed
parts back before decoding.

This module is numpy-only so exported runtimes stay importable without torch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np

YOLONAS_RAW_LAYOUTS: dict[str, tuple[str, ...]] = {
    "yolonas_raw_heads_v1": ("reg", "cls"),
    "yolonas_pose_raw_heads_v1": ("reg", "cls", "pose"),
    "yolonas_obb_raw_heads_v1": ("reg", "cls", "offset", "rot"),
    "yolo9_raw_heads_v1": ("reg", "cls"),
    "yolox_raw_heads_v1": ("reg", "obj", "cls"),
    "darknet_raw_heads_v1": ("reg", "obj", "cls"),
    "yolo7_raw_heads_v1": ("reg", "obj", "cls"),
}

YOLONAS_RAW_LAYOUT_BY_TASK: dict[str, str] = {
    "detect": "yolonas_raw_heads_v1",
    "pose": "yolonas_pose_raw_heads_v1",
    "obb": "yolonas_obb_raw_heads_v1",
}

# Windows applied by default per task. YOLO-NAS-pose outputs span about
# [-1900, +175] at strides 16 and 32 (detect stays within about [-17, +15]):
# one int8 scale over that range rounds every confident score to 0.5 and
# moves keypoints by ~20 px. Where a person is detected, class and visibility
# logits stay within +-5, distribution logits within +-12 of zero and
# keypoint offsets within [-10.2, +18.3] grid cells (YOLO-NAS-pose n to l),
# so the windows below change no float32 result that matters (cocopose300
# keypoint mAP is unchanged). The keypoint window is asymmetric because
# offsets are: [-10, 18] gives a finer INT8 step than +-20.
YOLONAS_RAW_OUTPUT_WINDOWS_BY_TASK: dict[str, dict[str, tuple[float, float]]] = {
    "pose": {"cls": (-8.0, 8.0), "reg": (-12.0, 12.0), "pose": (-10.0, 18.0)},
}

# YOLO9 detect windows (yolo9_raw_heads_v1). Background class logits reach
# about -450 at stride 16 on YOLO9-t (one int8 step ~1.8 logits); where an
# object is detected the best class logit stays within [-1.1, +3.4] on every
# size. Distance-bin logits differ by size: t and s span about [-38, +10]
# and need bins down to about -12 near objects, m and c span about [-9, +19]
# with top bins up to +19. The windows below cover both and change no
# float32 result (COCO). The class bound matches _INT8_LOGIT_BOUND, which the
# full-integer TFLite export uses for the same reason.
YOLO9_RAW_OUTPUT_WINDOWS: dict[str, tuple[float, float]] = {
    "cls": (-8.0, 8.0),
    "reg": (-12.0, 20.0),
}

# YOLOX raw outputs stay within a few INT8 steps of their useful range on
# every size (objectness and class logits span about [-40, +7], step <= 0.17;
# where a box scores above 0.05 they stay within [-2.9, +7.0]), so the
# export keeps them unwindowed. Windowing them changed nothing on YOLOX-n.
YOLOX_RAW_OUTPUT_WINDOWS: dict[str, tuple[float, float]] = {}

# A windowed part is mapped from [low, high] onto [0, OUTPUT_WINDOW_SPAN].
OUTPUT_WINDOW_SPAN = 6.0

YOLONAS_STRIDES: tuple[int, ...] = (8, 16, 32)
YOLONAS_REG_MAX = 16
YOLONAS_GRID_CELL_OFFSET = 0.5


def raw_head_output_names(
    layout: str, strides: Sequence[int] = YOLONAS_STRIDES
) -> list[str]:
    """Deterministic output names for a layout, e.g. ``p3_reg``, ``p3_cls``.

    The level tag is ``p<log2(stride)>``, the usual pyramid naming.
    """
    parts = _layout_parts(layout)
    names = []
    for stride in strides:
        level = round(math.log2(int(stride)))
        names.extend(f"p{level}_{part}" for part in parts)
    return names


def _layout_parts(layout: str) -> tuple[str, ...]:
    try:
        return YOLONAS_RAW_LAYOUTS[layout]
    except KeyError:
        known = ", ".join(sorted(YOLONAS_RAW_LAYOUTS))
        raise ValueError(
            f"Unknown YOLO-NAS raw output layout {layout!r}. Known layouts: {known}."
        ) from None


def _split_levels(
    outputs: Sequence[np.ndarray], layout: str, strides: Sequence[int]
) -> list[dict[str, np.ndarray]]:
    parts = _layout_parts(layout)
    expected = len(parts) * len(strides)
    if len(outputs) != expected:
        raise ValueError(
            f"{layout} expects {expected} outputs ({len(strides)} levels x "
            f"{len(parts)} tensors), got {len(outputs)}."
        )
    levels = []
    for index in range(len(strides)):
        chunk = outputs[index * len(parts) : (index + 1) * len(parts)]
        level = {}
        for part, array in zip(parts, chunk):
            array = np.asarray(array, dtype=np.float32)
            if array.ndim != 4:
                raise ValueError(
                    f"{layout} output {part!r} at stride {strides[index]} must be "
                    f"4-D [B, C, H, W], got shape {array.shape}."
                )
            level[part] = array
        shapes = {array.shape[0:1] + array.shape[2:] for array in level.values()}
        if len(shapes) != 1:
            raise ValueError(
                f"{layout} outputs at stride {strides[index]} disagree on batch or "
                f"spatial size: {sorted(shapes)}."
            )
        levels.append(level)
    return levels


@lru_cache(maxsize=32)
def _anchor_grid(h: int, w: int, offset: float) -> np.ndarray:
    """Grid-cell centres in grid units, ``(h*w, 2)`` as ``(x, y)``, row-major."""
    xs = np.arange(w, dtype=np.float32) + np.float32(offset)
    ys = np.arange(h, dtype=np.float32) + np.float32(offset)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    anchors = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    anchors.setflags(write=False)
    return anchors


@lru_cache(maxsize=4)
def _projection(bins: int) -> np.ndarray:
    projection = np.arange(bins, dtype=np.float32).reshape(1, bins)
    projection.setflags(write=False)
    return projection


def _dfl_expectation(reg: np.ndarray, sides: int, reg_max: int) -> np.ndarray:
    """Expected bin index per side: ``[B, C, H, W]`` -> ``[B, sides, H*W]``.

    Computed as ``sum(exp * bin) / sum(exp)``, which equals the softmax
    expectation without normalizing every bin.
    """
    b, c, h, w = reg.shape
    bins = reg_max + 1
    if c != sides * bins:
        raise ValueError(
            f"Regression output has {c} channels; expected {sides} x (reg_max + 1) "
            f"= {sides * bins} for reg_max={reg_max}."
        )
    logits = reg.reshape(b, sides, bins, h * w)
    weights = logits - logits.max(axis=2, keepdims=True)
    np.exp(weights, out=weights)
    expectation = np.matmul(_projection(bins), weights)[:, :, 0, :]  # [B, sides, HW]
    expectation /= weights.sum(axis=2)
    return expectation


def _sigmoid_(x: np.ndarray) -> np.ndarray:
    """In-place logistic function on a float32 array."""
    np.negative(x, out=x)
    # exp overflows to inf for very negative logits; 1 / inf is the exact 0.
    with np.errstate(over="ignore"):
        np.exp(x, out=x)
    x += np.float32(1.0)
    np.reciprocal(x, out=x)
    return x


def _flatten_hw(array: np.ndarray) -> np.ndarray:
    """``[B, C, H, W]`` -> ``[B, H*W, C]`` (a view)."""
    b, c, h, w = array.shape
    return array.reshape(b, c, h * w).transpose(0, 2, 1)


def _check_strides(levels, strides) -> None:
    extents = {
        (level["cls"].shape[2] * int(stride), level["cls"].shape[3] * int(stride))
        for level, stride in zip(levels, strides)
    }
    if len(extents) != 1:
        raise ValueError(
            f"Feature-map sizes do not match strides {tuple(strides)}: the levels "
            f"cover input extents {sorted(extents)}."
        )


def _class_probabilities(levels) -> np.ndarray:
    """Concatenated ``[B, N, C]`` class-head outputs, sigmoid applied in place."""
    return _sigmoid_(np.concatenate([_flatten_hw(level["cls"]) for level in levels], axis=1))


def _write_xyxy(out: np.ndarray, distances: np.ndarray, anchors: np.ndarray, stride) -> None:
    """``out[B, HW, 4]`` = xyxy pixels from ltrb distances ``[B, 4, HW]``."""
    ax, ay = anchors[:, 0], anchors[:, 1]
    out[..., 0] = (ax - distances[:, 0]) * stride
    out[..., 1] = (ay - distances[:, 1]) * stride
    out[..., 2] = (ax + distances[:, 2]) * stride
    out[..., 3] = (ay + distances[:, 3]) * stride


def decode_yolonas_raw(
    outputs: Sequence[np.ndarray],
    *,
    strides: Sequence[int] = YOLONAS_STRIDES,
    reg_max: int = YOLONAS_REG_MAX,
    grid_cell_offset: float = YOLONAS_GRID_CELL_OFFSET,
) -> list[np.ndarray]:
    """Decode detection heads to ``[boxes (B, N, 4) xyxy px, scores (B, N, nc)]``."""
    layout = "yolonas_raw_heads_v1"
    levels = _split_levels(outputs, layout, strides)
    _check_strides(levels, strides)
    batch = levels[0]["cls"].shape[0]
    total = sum(level["cls"].shape[2] * level["cls"].shape[3] for level in levels)
    boxes = np.empty((batch, total, 4), dtype=np.float32)
    start = 0
    for level, stride in zip(levels, strides):
        _, _, h, w = level["cls"].shape
        anchors = _anchor_grid(h, w, float(grid_cell_offset))
        distances = _dfl_expectation(level["reg"], 4, reg_max)
        _write_xyxy(boxes[:, start : start + h * w], distances, anchors, np.float32(stride))
        start += h * w
    return [boxes, _class_probabilities(levels)]


def decode_yolonas_pose_raw(
    outputs: Sequence[np.ndarray],
    *,
    strides: Sequence[int] = YOLONAS_STRIDES,
    reg_max: int = YOLONAS_REG_MAX,
    grid_cell_offset: float = YOLONAS_GRID_CELL_OFFSET,
    pose_offset_multiplier: float = 1.0,
    compensate_grid_cell_offset: bool = True,
) -> list[np.ndarray]:
    """Decode pose heads to ``[boxes, scores, keypoints_xy, keypoint_scores]``.

    Shapes: ``(B, N, 4)`` xyxy px, ``(B, N, nc)``, ``(B, N, K, 2)`` px and
    ``(B, N, K)``. The keypoint count ``K`` comes from the ``pose`` channels
    (``2K``) and the class count from ``cls`` channels minus ``K``.
    """
    layout = "yolonas_pose_raw_heads_v1"
    levels = _split_levels(outputs, layout, strides)
    _check_strides(levels, strides)
    batch, cls_channels = levels[0]["cls"].shape[:2]
    pose_channels = levels[0]["pose"].shape[1]
    if pose_channels % 2:
        raise ValueError(
            f"Pose output has {pose_channels} channels; expected 2 per keypoint."
        )
    num_keypoints = pose_channels // 2
    num_classes = cls_channels - num_keypoints
    if num_classes < 1:
        raise ValueError(
            f"Pose class output has {cls_channels} channels, which cannot hold "
            f"{num_keypoints} keypoint visibilities plus at least one class."
        )
    total = sum(level["cls"].shape[2] * level["cls"].shape[3] for level in levels)
    boxes = np.empty((batch, total, 4), dtype=np.float32)
    keypoints = np.empty((batch, total, num_keypoints, 2), dtype=np.float32)
    start = 0
    for level, stride in zip(levels, strides):
        _, _, h, w = level["cls"].shape
        hw = h * w
        anchors = _anchor_grid(h, w, float(grid_cell_offset))
        stride_f = np.float32(stride)
        distances = _dfl_expectation(level["reg"], 4, reg_max)
        _write_xyxy(boxes[:, start : start + hw], distances, anchors, stride_f)
        # [B, 2K, H, W] -> [B, K, 2, HW] -> [B, HW, K, 2]
        pose = level["pose"].reshape(batch, num_keypoints, 2, hw).transpose(0, 3, 1, 2)
        if pose_offset_multiplier != 1.0:
            pose = pose * np.float32(pose_offset_multiplier)
        pose = pose + anchors[None, :, None, :]
        if compensate_grid_cell_offset:
            pose = pose - np.float32(grid_cell_offset)
        np.multiply(pose, stride_f, out=keypoints[:, start : start + hw])
        start += hw
    probabilities = _class_probabilities(levels)  # [B, N, nc + K]
    return [
        boxes,
        np.ascontiguousarray(probabilities[..., :num_classes]),
        keypoints,
        np.ascontiguousarray(probabilities[..., num_classes:]),
    ]


def decode_yolonas_obb_raw(
    outputs: Sequence[np.ndarray],
    *,
    strides: Sequence[int] = YOLONAS_STRIDES,
    reg_max: int = YOLONAS_REG_MAX,
    grid_cell_offset: float = YOLONAS_GRID_CELL_OFFSET,
) -> list[np.ndarray]:
    """Decode rotated heads to ``[boxes (B, N, 5) cx cy w h angle, scores (B, N, nc)]``.

    The angle is ``(sigmoid(rot) - 0.25) * pi`` radians, as in the native head;
    canonicalisation happens later in the shared OBB parser.
    """
    layout = "yolonas_obb_raw_heads_v1"
    levels = _split_levels(outputs, layout, strides)
    _check_strides(levels, strides)
    batch = levels[0]["cls"].shape[0]
    total = sum(level["cls"].shape[2] * level["cls"].shape[3] for level in levels)
    boxes = np.empty((batch, total, 5), dtype=np.float32)
    start = 0
    for level, stride in zip(levels, strides):
        _, _, h, w = level["cls"].shape
        hw = h * w
        out = boxes[:, start : start + hw]
        anchors = _anchor_grid(h, w, float(grid_cell_offset))
        stride_f = np.float32(stride)
        sizes = _dfl_expectation(level["reg"], 2, reg_max)  # [B, 2, HW]
        offsets = level["offset"].reshape(batch, 2, hw)
        out[..., 0] = (offsets[:, 0] + anchors[:, 0]) * stride_f
        out[..., 1] = (offsets[:, 1] + anchors[:, 1]) * stride_f
        out[..., 2] = sizes[:, 0] * stride_f
        out[..., 3] = sizes[:, 1] * stride_f
        rot = _sigmoid_(level["rot"].reshape(batch, hw).copy())
        out[..., 4] = (rot - np.float32(0.25)) * np.float32(math.pi)
        start += hw
    return [boxes, _class_probabilities(levels)]


def decode_yolo9_raw(
    outputs: Sequence[np.ndarray],
    *,
    strides: Sequence[int] = YOLONAS_STRIDES,
    reg_max: int = 16,
    grid_cell_offset: float = YOLONAS_GRID_CELL_OFFSET,
) -> list[np.ndarray]:
    """Decode YOLO9 heads to ``[(B, 4 + nc, N)]``: xyxy px, then class scores.

    The single tensor matches the decoded YOLO9 export: anchors at the cell
    centres, distances as the softmax expectation over ``reg_max`` bins
    scaled by the level stride.
    """
    layout = "yolo9_raw_heads_v1"
    levels = _split_levels(outputs, layout, strides)
    _check_strides(levels, strides)
    batch, num_classes = levels[0]["cls"].shape[:2]
    total = sum(level["cls"].shape[2] * level["cls"].shape[3] for level in levels)
    # Filled as [B, N, 4 + nc] and returned as a [B, 4 + nc, N] view, so the
    # parser's per-image transpose is contiguous again.
    out = np.empty((batch, total, 4 + num_classes), dtype=np.float32)
    start = 0
    for level, stride in zip(levels, strides):
        _, _, h, w = level["cls"].shape
        anchors = _anchor_grid(h, w, float(grid_cell_offset))
        # YOLO9 bins are 0..reg_max-1, i.e. YOLO-NAS's reg_max - 1.
        distances = _dfl_expectation(level["reg"], 4, int(reg_max) - 1)
        _write_xyxy(out[:, start : start + h * w, :4], distances, anchors, np.float32(stride))
        start += h * w
    out[..., 4:] = _class_probabilities(levels)
    return [out.transpose(0, 2, 1)]


def decode_yolox_raw(
    outputs: Sequence[np.ndarray],
    *,
    strides: Sequence[int] = YOLONAS_STRIDES,
    grid_cell_offset: float = 0.0,
    **_unused,
) -> list[np.ndarray]:
    """Decode YOLOX heads to ``[(B, N, 5 + nc)]``: cxcywh px, objectness, classes.

    The tensor matches the decoded YOLOX export: anchors at the top-left of
    each cell (``grid_cell_offset`` 0), centres ``(offset + cell) * stride``,
    sizes ``exp(log_wh) * stride``, sigmoid objectness and class scores.
    """
    layout = "yolox_raw_heads_v1"
    levels = _split_levels(outputs, layout, strides)
    _check_strides(levels, strides)
    batch, num_classes = levels[0]["cls"].shape[:2]
    total = sum(level["cls"].shape[2] * level["cls"].shape[3] for level in levels)
    out = np.empty((batch, total, 5 + num_classes), dtype=np.float32)
    start = 0
    for level, stride in zip(levels, strides):
        _, _, h, w = level["cls"].shape
        hw = h * w
        anchors = _anchor_grid(h, w, float(grid_cell_offset))  # (HW, 2) as (x, y)
        stride_f = np.float32(stride)
        reg = level["reg"].reshape(batch, 4, hw)
        block = out[:, start : start + hw]
        block[..., 0] = (reg[:, 0] + anchors[:, 0]) * stride_f
        block[..., 1] = (reg[:, 1] + anchors[:, 1]) * stride_f
        with np.errstate(over="ignore"):
            block[..., 2] = np.exp(reg[:, 2]) * stride_f
            block[..., 3] = np.exp(reg[:, 3]) * stride_f
        block[..., 4] = level["obj"].reshape(batch, hw)
        block[..., 5:] = _flatten_hw(level["cls"])
        start += hw
    _sigmoid_(out[..., 4:])
    return [out]


def decode_anchor_yolo_raw(
    outputs: Sequence[np.ndarray],
    *,
    layout: str,
    strides: Sequence[int],
    anchors: Sequence[Sequence[Sequence[float]]],
    scale_x_y: Sequence[float] | None = None,
    **_unused,
) -> list[np.ndarray]:
    """Decode anchor-based YOLO heads to ``[(B, 4 + nc, N)]``: xyxy px, obj * cls.

    Rows are ordered level, anchor, row, column, as in the decoded export.
    ``anchors`` holds each level's ``(w, h)`` pairs in input pixels.
    ``darknet_raw_heads_v1`` (Darknet YOLOv3/v4): centre ``(sigmoid(t) * s -
    (s - 1) / 2 + cell) * stride`` with ``s = scale_x_y`` of the level, size ``exp(t) *
    anchor``. ``yolo7_raw_heads_v1``: centre ``(2 * sigmoid(t) - 0.5 + cell) *
    stride``, size ``(2 * sigmoid(t)) ** 2 * anchor``.
    """
    levels = _split_levels(outputs, layout, strides)
    _check_strides(levels, strides)
    if len(anchors) != len(levels):
        raise ValueError(
            f"{layout} needs anchors for {len(levels)} levels, got {len(anchors)}."
        )
    scales = [1.0] * len(levels) if scale_x_y is None else [float(s) for s in scale_x_y]
    batch = levels[0]["obj"].shape[0]
    rows = []
    for level, stride, level_anchors, scale in zip(levels, strides, anchors, scales):
        _, count, h, w = level["obj"].shape
        pairs = np.asarray(level_anchors, dtype=np.float32).reshape(count, 2)
        nc = level["cls"].shape[1] // count
        reg = level["reg"].reshape(batch, count, 4, h, w)
        obj = _sigmoid_(level["obj"].reshape(batch, count, h, w).astype(np.float32, copy=True))
        cls = _sigmoid_(level["cls"].reshape(batch, count, nc, h, w).astype(np.float32, copy=True))
        col = np.arange(w, dtype=np.float32).reshape(1, 1, 1, w)
        row = np.arange(h, dtype=np.float32).reshape(1, 1, h, 1)
        anchor_w = pairs[:, 0].reshape(1, count, 1, 1)
        anchor_h = pairs[:, 1].reshape(1, count, 1, 1)
        if layout == "yolo7_raw_heads_v1":
            t = _sigmoid_(reg.astype(np.float32, copy=True)) * np.float32(2.0)
            cx = (t[:, :, 0] - np.float32(0.5) + col) * np.float32(stride)
            cy = (t[:, :, 1] - np.float32(0.5) + row) * np.float32(stride)
            half_w = t[:, :, 2] ** 2 * anchor_w / 2
            half_h = t[:, :, 3] ** 2 * anchor_h / 2
        else:
            txy = _sigmoid_(reg[:, :, 0:2].astype(np.float32, copy=True))
            if scale != 1.0:
                txy = txy * np.float32(scale) - np.float32((scale - 1.0) / 2.0)
            cx = (txy[:, :, 0] + col) * np.float32(stride)
            cy = (txy[:, :, 1] + row) * np.float32(stride)
            with np.errstate(over="ignore"):
                half_w = np.exp(reg[:, :, 2]) * anchor_w / 2
                half_h = np.exp(reg[:, :, 3]) * anchor_h / 2
        out = np.empty((batch, 4 + nc, count, h, w), dtype=np.float32)
        out[:, 0] = cx - half_w
        out[:, 1] = cy - half_h
        out[:, 2] = cx + half_w
        out[:, 3] = cy + half_h
        out[:, 4:] = cls.transpose(0, 2, 1, 3, 4) * obj[:, None]
        rows.append(out.reshape(batch, 4 + nc, count * h * w))
    return [np.concatenate(rows, axis=2)]


def _anchor_decoder(layout: str):
    def decode(outputs, **params):
        return decode_anchor_yolo_raw(outputs, layout=layout, **params)

    return decode


_DECODERS = {
    "yolonas_raw_heads_v1": decode_yolonas_raw,
    "yolonas_pose_raw_heads_v1": decode_yolonas_pose_raw,
    "yolonas_obb_raw_heads_v1": decode_yolonas_obb_raw,
    "yolo9_raw_heads_v1": decode_yolo9_raw,
    "yolox_raw_heads_v1": decode_yolox_raw,
    "darknet_raw_heads_v1": _anchor_decoder("darknet_raw_heads_v1"),
    "yolo7_raw_heads_v1": _anchor_decoder("yolo7_raw_heads_v1"),
}


def check_output_windows(
    layout: str, windows: Mapping[str, Sequence[float]] | None
) -> dict[str, tuple[float, float]]:
    """Validated ``{part: (low, high)}`` for ``layout``; ``{}`` when unset."""
    parts = _layout_parts(layout)
    checked: dict[str, tuple[float, float]] = {}
    for part, bounds in (windows or {}).items():
        if part not in parts:
            raise ValueError(
                f"Output window for {part!r}, which {layout} does not have "
                f"(parts: {', '.join(parts)})."
            )
        try:
            low, high = (float(value) for value in bounds)
        except (TypeError, ValueError):
            raise ValueError(
                f"Output window for {part!r} must be [low, high], got {bounds!r}."
            ) from None
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(
                f"Output window for {part!r} must satisfy low < high, got "
                f"[{low}, {high}]."
            )
        checked[part] = (low, high)
    return checked


def unwindow_raw_heads(
    layout: str,
    outputs: Sequence[np.ndarray],
    windows: Mapping[str, Sequence[float]] | None,
) -> list[np.ndarray]:
    """Map windowed parts from ``[0, 6]`` back to ``[low, high]``.

    Returns new arrays for windowed parts; the inputs are never modified.
    """
    checked = check_output_windows(layout, windows)
    if not checked:
        return list(outputs)
    parts = _layout_parts(layout)
    restored = []
    for index, array in enumerate(outputs):
        window = checked.get(parts[index % len(parts)])
        if window is None:
            restored.append(array)
            continue
        low, high = window
        scale = np.float32((high - low) / OUTPUT_WINDOW_SPAN)
        values = np.multiply(np.asarray(array, dtype=np.float32), scale)
        values += np.float32(low)
        restored.append(values)
    return restored


def decode_raw_heads(
    layout: str,
    outputs: Sequence[np.ndarray],
    *,
    output_windows: Mapping[str, Sequence[float]] | None = None,
    **params,
) -> list[np.ndarray]:
    """Decode raw head outputs for ``layout``; see the module docstring."""
    _layout_parts(layout)
    if output_windows:
        outputs = unwindow_raw_heads(layout, outputs, output_windows)
    return _DECODERS[layout](outputs, **params)


__all__ = [
    "OUTPUT_WINDOW_SPAN",
    "YOLO9_RAW_OUTPUT_WINDOWS",
    "YOLONAS_GRID_CELL_OFFSET",
    "YOLONAS_RAW_LAYOUTS",
    "YOLONAS_RAW_LAYOUT_BY_TASK",
    "YOLONAS_RAW_OUTPUT_WINDOWS_BY_TASK",
    "YOLONAS_REG_MAX",
    "YOLONAS_STRIDES",
    "YOLOX_RAW_OUTPUT_WINDOWS",
    "check_output_windows",
    "decode_anchor_yolo_raw",
    "decode_raw_heads",
    "decode_yolo9_raw",
    "decode_yolonas_obb_raw",
    "decode_yolonas_pose_raw",
    "decode_yolonas_raw",
    "decode_yolox_raw",
    "raw_head_output_names",
    "unwindow_raw_heads",
]

"""Shared letterbox geometry for families that pad a resized image.

Two pad conventions exist in the wild:

- ``topleft``: resized image in the top-left corner, pad on the right and
  bottom. This is what LibreYOLO YOLOv9 used through 1.5 (train, val, infer).
- ``center``: pad split on both sides. MultimediaTechLab/YOLO and WongKinYiu
  YOLOv9 train this way.

Unmarked YOLOv9 checkpoints must keep ``topleft`` so a 1.6 upgrade does not
silently move boxes on every user-trained weight. Newly converted official
MTL checkpoints stamp ``center``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

LETTERBOX_TOPLEFT = "topleft"
LETTERBOX_CENTER = "center"
LETTERBOX_PADS = (LETTERBOX_TOPLEFT, LETTERBOX_CENTER)

# Checkpoints written before letterbox_pad existed used top-left YOLOv9 pad.
DEFAULT_LETTERBOX_PAD = LETTERBOX_TOPLEFT


def normalize_letterbox_pad(value: str | None) -> str:
    """Return a canonical pad mode. Unknown/empty values become top-left."""
    if value is None:
        return DEFAULT_LETTERBOX_PAD
    pad = str(value).strip().lower().replace("-", "").replace("_", "")
    if pad in {"center", "centre"}:
        return LETTERBOX_CENTER
    if pad in {"topleft", "top-left", "tl"}:
        return LETTERBOX_TOPLEFT
    return DEFAULT_LETTERBOX_PAD


def letterbox_geometry(
    orig_h: int,
    orig_w: int,
    input_h: int,
    input_w: int,
    pad: str | None = None,
) -> Tuple[float, int, int, int, int]:
    """Scale and pad offsets for a letterboxed canvas.

    Returns ``(ratio, new_h, new_w, pad_left, pad_top)``. For ``topleft``,
    both pad offsets are 0 so existing ``coord / ratio`` undo stays exact.
    """
    pad = normalize_letterbox_pad(pad)
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"original size must be positive, got {(orig_h, orig_w)}")
    if input_h <= 0 or input_w <= 0:
        raise ValueError(f"input size must be positive, got {(input_h, input_w)}")
    ratio = min(input_h / orig_h, input_w / orig_w)
    new_h = max(int(orig_h * ratio), 1)
    new_w = max(int(orig_w * ratio), 1)
    if pad == LETTERBOX_CENTER:
        pad_left = (input_w - new_w) // 2
        pad_top = (input_h - new_h) // 2
    else:
        pad_left = 0
        pad_top = 0
    return float(ratio), new_h, new_w, pad_left, pad_top


def apply_letterbox_hwc(
    image: np.ndarray,
    input_h: int,
    input_w: int,
    *,
    pad: str | None = None,
    fill: int = 114,
) -> Tuple[np.ndarray, float, int, int]:
    """Letterbox an HWC uint8 image. Returns ``(canvas, ratio, pad_left, pad_top)``."""
    orig_h, orig_w = image.shape[:2]
    ratio, new_h, new_w, pad_left, pad_top = letterbox_geometry(
        orig_h, orig_w, input_h, input_w, pad
    )
    import cv2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if image.ndim == 3:
        canvas = np.full((input_h, input_w, image.shape[2]), fill, dtype=np.uint8)
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    else:
        canvas = np.full((input_h, input_w), fill, dtype=np.uint8)
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas, ratio, pad_left, pad_top


def unletterbox_xyxy(
    boxes,
    orig_w: int,
    orig_h: int,
    input_h: int,
    input_w: int,
    *,
    pad: str | None = None,
):
    """Map boxes from letterboxed canvas pixels back to the original image.

    ``topleft`` is a pure divide-by-ratio, matching historical YOLO9
    postprocess. ``center`` subtracts the pad first.
    """
    ratio, _new_h, _new_w, pad_left, pad_top = letterbox_geometry(
        orig_h, orig_w, input_h, input_w, pad
    )
    # Torch and numpy both support this indexing.
    boxes = boxes.clone() if hasattr(boxes, "clone") else boxes.copy()
    boxes[..., 0] = (boxes[..., 0] - pad_left) / ratio
    boxes[..., 2] = (boxes[..., 2] - pad_left) / ratio
    boxes[..., 1] = (boxes[..., 1] - pad_top) / ratio
    boxes[..., 3] = (boxes[..., 3] - pad_top) / ratio
    return boxes

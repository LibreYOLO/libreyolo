"""YOLO-format OBB label parsing helpers."""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np


def normalize_obb_angle(angle: float) -> float:
    """Normalize a rectangle angle to ``[-pi / 2, pi / 2)`` radians."""
    return (float(angle) + math.pi / 2) % math.pi - math.pi / 2


def canonicalize_xywhr(xywhr: Sequence[float]) -> np.ndarray:
    """Return ``xywhr`` with the long side as width and a normalized angle."""
    cx, cy, w, h, angle = map(float, xywhr)
    if w <= 0.0 or h <= 0.0:
        raise ValueError("OBB width and height must be positive")
    if h > w:
        w, h = h, w
        angle += math.pi / 2
    return np.array([cx, cy, w, h, normalize_obb_angle(angle)], dtype=np.float32)


def corners_to_xywhr(corners: np.ndarray) -> np.ndarray:
    """Convert four OBB corners to canonical ``xywhr``.

    The angle is in radians and follows the ``OBB`` result container contract:
    rotation of the width side around the center.
    """
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(corners)
    return canonicalize_xywhr((cx, cy, w, h, math.radians(angle_deg)))


def xywhr_to_corners(xywhr: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert ``xywhr`` rotated rectangles to four corners."""
    rects = np.asarray(xywhr, dtype=np.float32)
    single = rects.ndim == 1
    rects = rects.reshape(-1, 5)

    cx = rects[:, 0]
    cy = rects[:, 1]
    half_w = rects[:, 2] * 0.5
    half_h = rects[:, 3] * 0.5
    angle = rects[:, 4]
    cos = np.cos(angle)
    sin = np.sin(angle)

    width_vec = np.stack((cos * half_w, sin * half_w), axis=1)
    height_vec = np.stack((-sin * half_h, cos * half_h), axis=1)
    centers = np.stack((cx, cy), axis=1)
    corners = np.stack(
        (
            centers - width_vec - height_vec,
            centers + width_vec - height_vec,
            centers + width_vec + height_vec,
            centers - width_vec + height_vec,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    return corners[0] if single else corners


def scale_xywhr(
    xywhr: Sequence[float] | np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Scale rotated rectangles through corners and refit canonical ``xywhr``.

    Nonuniform x/y scaling turns a rotated rectangle into a parallelogram. The
    OBB contract still needs a rectangle, so this maps corners through the
    affine scale and returns OpenCV's minimum-area canonical rectangle.
    """
    rects = np.asarray(xywhr, dtype=np.float32)
    single = rects.ndim == 1
    rects = rects.reshape(-1, 5)
    if rects.shape[0] == 0:
        return np.zeros((0, 5), dtype=np.float32)
    corners = xywhr_to_corners(rects)
    corners[..., 0] *= float(scale_x)
    corners[..., 1] *= float(scale_y)
    scaled = np.stack([corners_to_xywhr(corner_set) for corner_set in corners])
    return scaled[0] if single else scaled.astype(np.float32, copy=False)


def xywhr_iou(xywhr_a: Sequence[float], xywhr_b: Sequence[float]) -> float:
    """Return exact IoU for two rotated rectangles in ``xywhr`` format."""
    rect_a = np.asarray(xywhr_a, dtype=np.float32)
    rect_b = np.asarray(xywhr_b, dtype=np.float32)
    area_a = float(rect_a[2] * rect_a[3])
    area_b = float(rect_b[2] * rect_b[3])
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    cv_rect_a = (
        (float(rect_a[0]), float(rect_a[1])),
        (float(rect_a[2]), float(rect_a[3])),
        float(math.degrees(rect_a[4])),
    )
    cv_rect_b = (
        (float(rect_b[0]), float(rect_b[1])),
        (float(rect_b[2]), float(rect_b[3])),
        float(math.degrees(rect_b[4])),
    )
    _status, points = cv2.rotatedRectangleIntersection(cv_rect_a, cv_rect_b)
    if points is None:
        return 0.0

    intersection = float(cv2.contourArea(points.astype(np.float32)))
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def xywhr_to_proxy_xyxy(xywhr: Sequence[float]) -> np.ndarray:
    """Convert ``xywhr`` to the horizontal proxy box used by YOLO9 OBB loss."""
    cx, cy, w, h, _ = map(float, xywhr)
    return np.array(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        dtype=np.float32,
    )


def parse_yolo_obb_label_line(
    line: str | Sequence[str],
    num_classes: int | None = None,
    *,
    clip: bool = False,
) -> tuple[int, np.ndarray]:
    """Parse one YOLO OBB label row into ``(class_id, corners)``.

    The accepted row shape is:

        class x1 y1 x2 y2 x3 y3 x4 y4

    Coordinates are normalized and returned as float32 corners with shape
    ``(4, 2)``. Set ``clip=True`` for dataset ingestion paths that should keep
    slightly out-of-frame crop-boundary boxes instead of dropping the row. The
    parser validates the file format but does not canonicalize point order or
    convert corners to an angle representation.
    """
    parts = line.split() if isinstance(line, str) else list(line)
    if len(parts) != 9:
        raise ValueError(f"Expected 9 fields for a YOLO OBB label, got {len(parts)}")

    try:
        class_value = float(parts[0])
    except ValueError as exc:
        raise ValueError(f"OBB class id must be numeric, got {parts[0]!r}") from exc

    if not np.isfinite(class_value) or not class_value.is_integer():
        raise ValueError(f"OBB class id must be an integer, got {parts[0]!r}")
    class_id = int(class_value)

    if class_id < 0:
        raise ValueError(f"OBB class id must be non-negative, got {class_id}")
    if num_classes is not None:
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if class_id >= num_classes:
            raise ValueError(
                f"OBB class id {class_id} out of range [0, {num_classes - 1}]"
            )

    try:
        corners = np.asarray(parts[1:], dtype=np.float32).reshape(4, 2)
    except ValueError as exc:
        raise ValueError("OBB coordinates must be numeric") from exc

    if not np.isfinite(corners).all():
        raise ValueError("OBB coordinates must be finite")
    out_of_bounds = (corners < 0.0) | (corners > 1.0)
    if out_of_bounds.any() and not clip:
        raise ValueError("OBB coordinates must be normalized to [0, 1]")
    if out_of_bounds.any():
        corners = np.clip(corners, 0.0, 1.0)

    x = corners[:, 0]
    y = corners[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area <= 0.0:
        raise ValueError("OBB corners must form a non-degenerate polygon")

    return class_id, corners

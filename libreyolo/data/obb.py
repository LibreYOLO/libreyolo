"""YOLO-format OBB label parsing helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def parse_yolo_obb_label_line(
    line: str | Sequence[str], num_classes: int | None = None
) -> tuple[int, np.ndarray]:
    """Parse one YOLO OBB label row into ``(class_id, corners)``.

    The accepted row shape is:

        class x1 y1 x2 y2 x3 y3 x4 y4

    Coordinates are normalized and returned as float32 corners with shape
    ``(4, 2)``. The parser validates the file format but does not canonicalize
    point order or convert corners to an angle representation.
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
    if ((corners < 0.0) | (corners > 1.0)).any():
        raise ValueError("OBB coordinates must be normalized to [0, 1]")

    x = corners[:, 0]
    y = corners[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area <= 0.0:
        raise ValueError("OBB corners must form a non-degenerate polygon")

    return class_id, corners

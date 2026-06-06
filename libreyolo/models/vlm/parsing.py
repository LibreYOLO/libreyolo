"""Tolerant parsing of VLM detection output into the LibreYOLO detection dict.

Vision-language detectors emit a JSON array of ``{"label", "bbox"}`` objects as
*generated text*, not a tensor. That text can arrive wrapped in markdown fences,
prefixed with prose, or truncated mid-array when generation hits the token
budget. These helpers turn that noisy text into the plain detection dict that
``InferenceRunner._wrap_results`` already knows how to turn into ``Results``.

Everything here is pure (no torch, no model) so it can be unit-tested offline.

The coordinate contract follows the documented LFM2-VL schema: ``bbox`` is
``[x1, y1, x2, y2]`` normalized to ``[0, 1]`` relative to the original image.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

__all__ = [
    "extract_detections",
    "normalize_bbox",
    "resolve_label",
    "build_detection_dict",
]

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```$")
# A flat ``{...}`` object with no nested braces; detection items are flat.
_OBJECT = re.compile(r"\{[^{}]*\}")


def _find_balanced_array(text: str) -> Optional[str]:
    """Return the first balanced ``[...]`` substring, or None if unterminated."""
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _loads_object(blob: str) -> Optional[dict]:
    for candidate in (blob, blob.replace("'", '"')):
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_detections(text: str) -> List[dict]:
    """Extract ``{"label", "bbox"}`` dicts from possibly-noisy model text.

    Defensive against markdown fences, surrounding prose, single quotes, and a
    truncated (unterminated) array. Returns ``[]`` rather than raising on any
    unparseable input, so an empty or chatty "no objects found" reply maps to
    zero detections.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", text.strip())).strip()

    array = _find_balanced_array(cleaned)
    if array is not None:
        try:
            data = json.loads(array)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: array was unterminated or invalid, recover individual objects.
    recovered = []
    for blob in _OBJECT.findall(cleaned):
        obj = _loads_object(blob)
        if obj is not None:
            recovered.append(obj)
    return recovered


def normalize_bbox(bbox) -> Optional[Tuple[float, float, float, float]]:
    """Validate/clean a normalized ``[x1, y1, x2, y2]`` box.

    Returns a 4-tuple clamped to ``[0, 1]`` with corners ordered, or None if the
    value is not four finite numbers. Coordinates are assumed already normalized
    to ``[0, 1]`` per the detection prompt contract.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
        return None
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in vals)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def resolve_label(label, name_to_id: Dict[str, int]) -> Optional[int]:
    """Map a free-text label to a class id (case-insensitive exact match).

    Returns None for labels outside the vocabulary, which the caller drops.
    This is what makes an open-vocabulary generator behave like a closed-set
    detector against a fixed ``names`` mapping.
    """
    if not isinstance(label, str):
        return None
    return name_to_id.get(label.strip().lower())


def build_detection_dict(
    items: List[dict],
    name_to_id: Dict[str, int],
    original_size: Tuple[int, int],
    conf_thres: float = 0.0,
    max_det: int = 300,
    default_score: float = 1.0,
    bbox_key: str = "bbox",
    coord_divisor: float = 1.0,
) -> dict:
    """Turn parsed items into the ``InferenceRunner`` detection dict.

    Boxes are read from ``item[bbox_key]``, divided by ``coord_divisor`` to
    reach the ``[0, 1]`` space (use 1.0 for already-normalized LFM2-VL output,
    1000.0 for Qwen-style ``bbox_2d`` on a 0-1000 scale), then scaled to pixel
    ``xyxy`` against ``original_size`` (W, H). Labels outside ``name_to_id`` and
    malformed boxes are skipped. ``default_score`` is the synthetic per-box
    confidence (the VLM emits none); rows below ``conf_thres`` are dropped so
    ``conf=`` still filters.
    """
    width, height = original_size
    boxes: List[List[float]] = []
    scores: List[float] = []
    classes: List[int] = []
    # Generative decoding can loop and emit the same object many times. A real
    # detector never reports an identical box twice, so drop duplicates (same
    # class + box rounded to ~0.1% of the image).
    seen = set()

    for item in items:
        class_id = resolve_label(item.get("label"), name_to_id)
        if class_id is None:
            continue
        raw = item.get(bbox_key)
        if coord_divisor != 1.0 and isinstance(raw, (list, tuple)) and len(raw) == 4:
            try:
                raw = [float(v) / coord_divisor for v in raw]
            except (TypeError, ValueError):
                raw = None
        box = normalize_bbox(raw)
        if box is None:
            continue
        if default_score < conf_thres:
            continue
        key = (class_id, *(round(v, 3) for v in box))
        if key in seen:
            continue
        seen.add(key)
        x1, y1, x2, y2 = box
        boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
        scores.append(default_score)
        classes.append(class_id)
        if len(boxes) >= max_det:
            break

    return {
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
        "num_detections": len(boxes),
    }

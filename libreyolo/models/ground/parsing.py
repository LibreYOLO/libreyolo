"""Tolerant parsing of grounding-model text into click points.

Grounders emit a click as generated text, not a tensor. The syntax varies by
family (``Click(x, y)``, ``<point>x y</point>``, JSON, a bare ``[x, y]``,
Florence ``<loc_N>`` tokens, or a box whose center is the click). Scaling is
*not* guessed here: the family declares ``COORD_SPACE`` and
``scale_point`` converts onto the original image canvas.

Pure (no torch, no model) so it can be unit-tested offline.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence

__all__ = [
    "COORD_SPACES",
    "build_point_dict",
    "coerce_queries",
    "extract_clicks",
    "scale_point",
]

COORD_SPACES = ("unit", "milli", "pixel", "pixel_view")

_CLICK_KWARGS = re.compile(
    r"\b(?:pyautogui\.)?click\s*\(\s*x\s*=\s*([\d.]+)\s*,\s*y\s*=\s*([\d.]+)\s*\)",
    re.IGNORECASE,
)
_CLICK_XY = re.compile(
    r"\b(?:pyautogui\.)?click\s*\(\s*(?:(?:point|start_box)\s*=\s*)?"
    r"(?:['\"]?<point>\s*)?"
    r"([\d.]+)\s*[, ]\s*([\d.]+)"
    r"(?:\s*</point>['\"]?)?"
    r"\s*\)",
    re.IGNORECASE,
)
_POINT_TAG = re.compile(
    r"<\s*point\s*>\s*([\d.]+)\s*[, ]\s*([\d.]+)\s*<\s*/\s*point\s*>",
    re.IGNORECASE,
)
_LOC_TOKEN = re.compile(r"<\s*loc_(\d+)\s*>", re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_OBJECT = re.compile(r"\{[^{}]*\}")
_PAIR = re.compile(r"\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]")


def coerce_queries(query: str | Sequence[str]) -> list[str]:
    """Normalize ``prompt=`` / ``query=`` / ``set_query`` input to a label list."""
    if isinstance(query, str):
        text = query.strip()
        if not text:
            raise ValueError("query must be a non-empty string.")
        return [text]
    if not isinstance(query, (list, tuple)):
        raise TypeError(
            f"query must be a string or a list of strings, not {type(query).__name__}."
        )
    labels = [str(item).strip() for item in query]
    if not labels or any(not item for item in labels):
        raise ValueError("query list must contain non-empty strings.")
    folded = [item.casefold() for item in labels]
    if len(folded) != len(set(folded)):
        raise ValueError("queries must be unique case-insensitively.")
    return labels


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def _pair(x: object, y: object) -> list[float] | None:
    try:
        point = [float(x), float(y)]
    except (TypeError, ValueError):
        return None
    return point if _finite(point) else None


def _from_mapping(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    for key in ("point", "position", "coordinate", "coordinates", "click", "xy"):
        raw = item.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            point = _pair(raw[0], raw[1])
            if point is not None:
                return {"label": item.get("label"), "point": point}
    if "x" in item and "y" in item:
        point = _pair(item.get("x"), item.get("y"))
        if point is not None:
            return {"label": item.get("label"), "point": point}
    box = item.get("bbox") or item.get("bbox_2d") or item.get("box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            return None
        if not _finite((x1, y1, x2, y2)):
            return None
        return {"label": item.get("label"), "point": [(x1 + x2) / 2.0, (y1 + y2) / 2.0]}
    return None


def _loads(blob: str):
    for variant in (blob, blob.replace("'", '"')):
        try:
            return json.loads(variant)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _iter_json_objects(text: str) -> list[dict]:
    stripped = _FENCE.sub("", text).strip()
    parsed = _loads(stripped)
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    objects: list[dict] = []
    seen = set()
    for blob in _OBJECT.findall(stripped):
        parsed = _loads(blob)
        if not isinstance(parsed, dict):
            continue
        key = json.dumps(parsed, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        objects.append(parsed)
    return objects


def extract_clicks(text: str | None) -> list[dict]:
    """Pull click candidates out of generated text, first match family wins.

    Preference order is the syntax families actually emit:

    1. ``click(x, y)`` / ``Click(x, y)`` / ``click(point='<point>x y</point>')``
    2. ``<point>x y</point>``
    3. Florence ``<loc_N>`` tokens (2 = point, 4 = box center)
    4. JSON objects with ``x``/``y``, ``point``, or a box
    5. A bare ``[x, y]`` pair (ShowUI)
    """
    if not isinstance(text, str) or not text.strip():
        return []

    clicks = [
        {"point": [float(match.group(1)), float(match.group(2))]}
        for match in _CLICK_KWARGS.finditer(text)
    ]
    if clicks:
        return clicks
    clicks = [
        {"point": [float(match.group(1)), float(match.group(2))]}
        for match in _CLICK_XY.finditer(text)
    ]
    if clicks:
        return clicks

    tagged = [
        {"point": [float(match.group(1)), float(match.group(2))]}
        for match in _POINT_TAG.finditer(text)
    ]
    if tagged:
        return tagged

    locs = [int(token) for token in _LOC_TOKEN.findall(text)]
    if len(locs) >= 4:
        x1, y1, x2, y2 = (float(v) for v in locs[:4])
        return [{"point": [(x1 + x2) / 2.0, (y1 + y2) / 2.0]}]
    if len(locs) >= 2:
        return [{"point": [float(locs[0]), float(locs[1])]}]

    from_json = []
    for obj in _iter_json_objects(text):
        parsed = _from_mapping(obj)
        if parsed is not None:
            from_json.append(parsed)
    if from_json:
        return from_json

    pairs = [_pair(match.group(1), match.group(2)) for match in _PAIR.finditer(text)]
    pairs = [pair for pair in pairs if pair is not None]
    if len(pairs) == 1:
        return [{"point": pairs[0]}]
    # Several pairs: keep those that do not look like a "[0, 1]" scale aside.
    meaningful = [pair for pair in pairs if pair != [0.0, 1.0] and pair != [0.0, 0.0]]
    if len(meaningful) == 1:
        return [{"point": meaningful[0]}]
    if meaningful:
        return [{"point": meaningful[-1]}]
    return []


def scale_point(
    x: float,
    y: float,
    original_size: tuple[int, int],
    coord_space: str,
    view_size: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Map a model-space click onto the original image canvas."""
    if coord_space not in COORD_SPACES:
        raise ValueError(
            f"Unknown coord space {coord_space!r}. Must be one of: {', '.join(COORD_SPACES)}."
        )
    width, height = original_size
    if coord_space == "unit":
        px, py = x * width, y * height
    elif coord_space == "milli":
        px, py = (x / 1000.0) * width, (y / 1000.0) * height
    elif coord_space == "pixel":
        px, py = x, y
    else:
        if view_size is None:
            raise ValueError(
                "pixel_view coordinates require view_size=(width, height)."
            )
        view_w, view_h = view_size
        if view_w <= 0 or view_h <= 0:
            raise ValueError(f"view_size must be positive, got {view_size!r}.")
        px, py = x * width / view_w, y * height / view_h
    return px, py


def _on_canvas(
    px: float,
    py: float,
    original_size: tuple[int, int],
    slack: float = 0.5,
    *,
    continuous_upper_edge: bool = False,
) -> tuple[float, float] | None:
    """Keep on-canvas clicks; drop points that are not on the image.

    A half-pixel of slack covers float overshoot. Unit and milli spaces use
    the continuous right/bottom canvas edge (``width``, ``height``), while
    pixel spaces use the last pixel center (``width - 1``, ``height - 1``).
    Anything farther is rejected instead of being silently clamped.
    """
    width, height = original_size
    if width <= 0 or height <= 0:
        return None
    upper_x = float(width if continuous_upper_edge else width - 1)
    upper_y = float(height if continuous_upper_edge else height - 1)
    if px < -slack or py < -slack or px > upper_x + slack or py > upper_y + slack:
        return None
    return (
        min(float(width - 1), max(0.0, px)),
        min(float(height - 1), max(0.0, py)),
    )


def build_point_dict(
    items: list[dict],
    name_to_id: dict[str, int],
    original_size: tuple[int, int],
    *,
    coord_space: str = "unit",
    view_size: tuple[int, int] | None = None,
    conf_thres: float = 0.0,
    max_det: int = 300,
    classes: list[int] | None = None,
    default_score: float = 1.0,
) -> dict:
    """Turn parsed click items into the point-task detection dict."""
    if max_det <= 0:
        return {"points": [], "num_detections": 0}

    default_id = next(iter(name_to_id.values())) if len(name_to_id) == 1 else None
    allowed = set(classes) if classes is not None else None
    points: list[list[float]] = []
    seen = set()

    for item in items:
        label = item.get("label")
        class_id = None
        if isinstance(label, str) and label.strip():
            class_id = name_to_id.get(label.strip().lower())
        if class_id is None:
            class_id = default_id
        if class_id is None:
            continue
        if allowed is not None and class_id not in allowed:
            continue
        raw = item.get("point")
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        try:
            x, y = float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            continue
        if not _finite((x, y)) or default_score < conf_thres:
            continue
        px, py = scale_point(x, y, original_size, coord_space, view_size)
        snapped = _on_canvas(
            px,
            py,
            original_size,
            continuous_upper_edge=coord_space in {"unit", "milli"},
        )
        if snapped is None:
            continue
        px, py = snapped
        key = (class_id, round(px, 1), round(py, 1))
        if key in seen:
            continue
        seen.add(key)
        points.append([px, py, float(class_id), default_score])
        if len(points) >= max_det:
            break

    return {"points": points, "num_detections": len(points)}

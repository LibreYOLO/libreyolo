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
import math
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

__all__ = [
    "extract_molmo_points",
    "extract_detections",
    "extract_bare_boxes",
    "normalize_bbox",
    "to_xyxy",
    "resolve_label",
    "build_detection_dict",
]

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```$")
# A flat ``{...}`` object with no nested braces; detection items are flat.
_OBJECT = re.compile(r"\{[^{}]*\}")


def _iter_balanced_arrays(text: str):
    """Yield every balanced top-level ``[...]`` substring, left to right.

    A model can emit a bracketed array in prose (e.g. "normalized to [0,1]")
    before the real detection array, so the caller must try each one rather than
    committing to the first.
    """
    i = 0
    n = len(text)
    while i < n:
        start = text.find("[", i)
        if start == -1:
            return
        depth = 0
        end = -1
        for j in range(start, n):
            ch = text[j]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            return  # unterminated; nothing more to yield
        yield text[start : end + 1]
        i = end + 1


def _is_detection_dict(d) -> bool:
    """A dict that looks like a detection item (carries a label or a box key).

    Keyed on the union of markers used by the families, since the parser has no
    access to a family's ``BBOX_KEY``. Used to prefer the real detection array
    over a dict-bearing preamble.
    """
    return isinstance(d, dict) and ("label" in d or "bbox" in d or "bbox_2d" in d)


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

    # Collect DETECTION-shaped dicts (carrying a label/box key) from EVERY
    # top-level array, so a dict-bearing preamble (a restated schema example, a
    # reasoning/metadata array) cannot shadow the real detections. A bracketed
    # array in prose (e.g. "[0,1]") or a bare ``bbox`` list contributes nothing;
    # an echoed example whose label is out of vocabulary is dropped downstream.
    # An any-dict array is remembered only as a last resort for an unforeseen key.
    collected = []
    fallback = None
    for array in _iter_balanced_arrays(cleaned):
        try:
            data = json.loads(array)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, list):
            continue
        collected.extend(d for d in data if _is_detection_dict(d))
        if fallback is None:
            dicts = [d for d in data if isinstance(d, dict)]
            if dicts:
                fallback = dicts
    # Recover flat objects the bracket scan missed and merge them in (deduped):
    # a real array truncated mid-content behind a complete (schema-echo) array is
    # never yielded by the scan, and a bare object has no enclosing array at all.
    # _OBJECT matches brace-flat objects, which detection items are. Any-shape
    # recoveries are kept only as a last resort.
    seen = {repr(d) for d in collected}
    recovered = []
    for blob in _OBJECT.findall(cleaned):
        obj = _loads_object(blob)
        if obj is None:
            continue
        recovered.append(obj)
        if _is_detection_dict(obj) and repr(obj) not in seen:
            collected.append(obj)
            seen.add(repr(obj))

    if collected:
        return collected
    if fallback is not None:
        return fallback
    return recovered


_BARE_QUAD = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def extract_bare_boxes(text: str) -> List[List[float]]:
    """Extract unlabeled ``[x1, y1, x2, y2]`` quads from model text.

    For grounding-style models (North Micro Vision) that answer a single-class
    query with ``[[x1, y1, x2, y2], ...]``, a flat ``[x1, y1, x2, y2]``, or
    newline-separated quads, never with labeled objects. A prose refusal ("No
    dogs are present ...") contains no numeric quad and maps to zero boxes.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return [[float(v) for v in match] for match in _BARE_QUAD.findall(text)]


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
    if x2 <= x1 or y2 <= y1:
        return None
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


def to_xyxy(box, box_format: str = "xyxy"):
    """Convert a 4-value box in the given layout to ``[x1, y1, x2, y2]``.

    Supported layouts: ``xyxy`` (corners, the default), ``xywh`` (top-left plus
    width/height), ``cxcywh`` (center plus width/height), and ``yxyx``
    (``[ymin, xmin, ymax, xmax]``, Gemma 4 / Gemini ``box_2d``). Returns None
    if the value is not four finite numbers or the layout is unknown.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        a, b, c, d = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if box_format == "xyxy":
        return [a, b, c, d]
    if box_format == "xywh":
        return [a, b, a + c, b + d]
    if box_format == "cxcywh":
        return [a - c / 2.0, b - d / 2.0, a + c / 2.0, b + d / 2.0]
    if box_format == "yxyx":
        ymin, xmin, ymax, xmax = a, b, c, d
        return [xmin, ymin, xmax, ymax]
    return None


def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def build_detection_dict(
    items: List[dict],
    name_to_id: Dict[str, int],
    original_size: Tuple[int, int],
    conf_thres: float = 0.0,
    max_det: int = 300,
    classes: Optional[List[int]] = None,
    default_score: float = 1.0,
    bbox_key: str = "bbox",
    coord_divisor: float = 1.0,
    box_format: str = "xyxy",
    iou_thres: Optional[float] = None,
) -> dict:
    """Turn parsed items into the ``InferenceRunner`` detection dict.

    Boxes are read from ``item[bbox_key]``, divided by ``coord_divisor`` to
    reach the ``[0, 1]`` space (1.0 for already-normalized LFM2-VL output, 1000.0
    for Qwen-style ``bbox_2d`` on a 0-1000 scale), converted from ``box_format``
    to corner layout (``xyxy`` / ``xywh`` / ``cxcywh`` / ``yxyx``), then scaled to pixel
    ``xyxy`` against ``original_size`` (W, H). Labels outside ``name_to_id`` and
    malformed boxes are skipped. If ``classes`` is provided, that class filter is
    applied before the ``max_det`` cap so requested classes are not dropped by an
    earlier out-of-filter prediction. ``default_score`` is the synthetic per-box
    confidence (the VLM emits none); rows below ``conf_thres`` are dropped so
    ``conf=`` still filters.
    """
    if max_det <= 0:
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0,
        }

    width, height = original_size
    boxes: List[List[float]] = []
    norm_boxes: List[Tuple[float, float, float, float]] = []
    scores: List[float] = []
    class_ids: List[int] = []
    allowed_classes = set(classes) if classes is not None else None
    # Generative decoding can loop and emit the same object many times. A real
    # detector never reports an identical box twice, so drop duplicates (same
    # class + box rounded to ~0.1% of the image).
    seen = set()

    for item in items:
        class_id = resolve_label(item.get("label"), name_to_id)
        if class_id is None:
            continue
        if allowed_classes is not None and class_id not in allowed_classes:
            continue
        raw = item.get(bbox_key)
        if raw is None:
            # Generative models drift between key conventions even within one
            # family (LFM2.5-VL-3B emits "bbox" on synthetic scenes but
            # Qwen-style "bbox_2d" on natural photos), so fall back to the
            # known aliases. The family's coord_divisor still applies.
            for alt in ("bbox", "bbox_2d"):
                if alt != bbox_key and alt in item:
                    raw = item[alt]
                    break
        box = None
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            try:
                scaled = [float(v) / coord_divisor for v in raw]
            except (TypeError, ValueError):
                scaled = None
            box = normalize_bbox(to_xyxy(scaled, box_format)) if scaled else None
        if box is None:
            continue
        if default_score < conf_thres:
            continue
        key = (class_id, *(round(v, 3) for v in box))
        if key in seen:
            continue
        if iou_thres is not None and any(
            class_id == kept_class and _iou_xyxy(box, kept_box) > iou_thres
            for kept_box, kept_class in zip(norm_boxes, class_ids)
        ):
            continue
        seen.add(key)
        x1, y1, x2, y2 = box
        norm_boxes.append(box)
        boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
        scores.append(default_score)
        class_ids.append(class_id)
        if len(boxes) >= max_det:
            break

    return {
        "boxes": boxes,
        "scores": scores,
        "classes": class_ids,
        "num_detections": len(boxes),
    }


_MOLMO_POINT_TAG = re.compile(
    r"<(?P<tag>points?)\b(?P<attrs>[^<>]*?)(?:/\s*>|>[^<>]*</(?P=tag)\s*>)",
    re.DOTALL,
)


def extract_molmo_points(text: str, label: str) -> list[dict]:
    """Parse Molmo single-image markup into normalized, query-labelled points.

    Molmo2 ``coords="1 id x y ..."`` uses a 0-1000 scale. Legacy ``x/y``
    and ``x1/y1/...`` attributes use percentages. The grammar determines the
    scale, never the numeric magnitude. Multi-image/video groups are rejected.
    ADR 0002 places text decoding in this pure parser module, separate from
    model loading and inference. Keeping the scalar coordinate validation here
    lets callers exercise it without a model or tensors. Only Molmo2 calls this
    helper; existing JSON box parsers and other VLM adapters are unchanged.
    See ``NOTICE`` for the upstream format reference.
    """
    if not isinstance(text, str):
        return []
    items = []
    for match in _MOLMO_POINT_TAG.finditer(text):
        try:
            attrs = ET.fromstring("<point" + match["attrs"] + "/>").attrib
        except ET.ParseError:
            continue
        pairs = []
        if "coords" in attrs:
            # XML normalizes literal tabs/newlines inside attributes to spaces.
            # Preserve the distinction between point spacing and frame separators.
            raw_coords = re.search(r"\bcoords\s*=\s*(['\"])(.*?)\1", match["attrs"], re.DOTALL)
            if raw_coords is None or any(c in raw_coords[2] for c in "\t\r\n:;,"):
                continue
            fields = attrs["coords"].split(" ")
            fields = [field for field in fields if field]
            # A single still image has index 1 and complete (id, x, y) triples.
            if not fields or fields[0] != "1" or (len(fields) - 1) % 3:
                continue
            if not all(re.fullmatch(r"[0-9]+", field) for field in fields):
                continue
            pairs = [(fields[i + 1], fields[i + 2]) for i in range(1, len(fields), 3)]
            divisor = 1000.0
        else:
            divisor = 100.0
            if match["tag"] == "point":
                pairs = [(attrs.get("x"), attrs.get("y"))]
            else:
                indices = sorted(
                    int(key[1:]) for key in attrs if re.fullmatch(r"x[1-9][0-9]*", key)
                )
                pairs = [(attrs[f"x{i}"], attrs.get(f"y{i}")) for i in indices]
        for raw_x, raw_y in pairs:
            try:
                x, y = float(raw_x) / divisor, float(raw_y) / divisor
            except (TypeError, ValueError, OverflowError):
                continue
            if all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
                items.append({"label": label, "point": [x, y]})
    return items

"""Serialize detection labels into the target text a VLM family is trained on.

This is the training-side mirror of ``libreyolo/models/vlm/parsing.py``: the
parser turns generated text into boxes using a family's declared convention
(``BBOX_KEY``, ``COORD_DIVISOR``, ``BOX_FORMAT``); this module turns dataset
boxes into exactly the text that parser expects back. Keeping both sides driven
by the same three class attributes is what guarantees a fine-tuned model emits
what ``predict()`` parses.

Pure functions, no torch, unit-tested offline including a parser round-trip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Sequence, Tuple

__all__ = ["FamilyFormat", "serialize_detections"]


@dataclass(frozen=True)
class FamilyFormat:
    """The output convention of one VLM family, captured for training.

    Built from a live model via :meth:`from_model` so the training pipeline can
    never drift from the inference adapter's declared convention.
    """

    family: str
    bbox_key: str
    coord_divisor: float
    box_format: str
    detection_prompt: str

    @classmethod
    def from_model(cls, model) -> "FamilyFormat":
        """Capture the convention (and current-vocabulary prompt) of a model."""
        return cls(
            family=model.FAMILY,
            bbox_key=model.BBOX_KEY,
            coord_divisor=float(model.COORD_DIVISOR),
            box_format=model.BOX_FORMAT,
            detection_prompt=model._detection_prompt(),
        )


def _convert_layout(
    box: Tuple[float, float, float, float], box_format: str
) -> Tuple[float, float, float, float]:
    """Convert a normalized xyxy box into the family's emitted layout."""
    x1, y1, x2, y2 = box
    if box_format == "xyxy":
        return (x1, y1, x2, y2)
    if box_format == "xywh":
        return (x1, y1, x2 - x1, y2 - y1)
    if box_format == "cxcywh":
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)
    raise ValueError(f"Unknown BOX_FORMAT {box_format!r}")


def _scale_coord(value: float, divisor: float) -> float | int:
    """Scale one normalized coordinate to the family's emitted range.

    ``COORD_DIVISOR == 1.0`` families write floats on [0, 1] (3 decimals, the
    precision the prompts show); larger divisors (e.g. 1000) write integers on
    [0, divisor], clamped, matching what the grounding-pretrained models emit.
    """
    if divisor == 1.0:
        return round(min(max(value, 0.0), 1.0), 3)
    scaled = int(round(min(max(value, 0.0), 1.0) * divisor))
    return min(max(scaled, 0), int(divisor))


def serialize_detections(
    boxes_xyxy: Sequence[Sequence[float]],
    labels: Sequence[str],
    fmt: FamilyFormat,
) -> str:
    """Render ground-truth boxes as the family's expected JSON answer text.

    Args:
        boxes_xyxy: Normalized ``[x1, y1, x2, y2]`` boxes on [0, 1], relative
            to the full image (the dataset-schema convention).
        labels: Class name per box (the phrase the model is prompted with).
        fmt: The family convention captured by :class:`FamilyFormat`.

    Returns:
        A JSON array string, one object per box, in deterministic reading
        order (top-to-bottom, then left-to-right), formatted the way the
        family's detection prompt asks for it. No boxes returns ``"[]"``,
        teaching the model the documented empty-image answer.
    """
    if len(boxes_xyxy) != len(labels):
        raise ValueError(
            f"boxes/labels length mismatch: {len(boxes_xyxy)} vs {len(labels)}"
        )
    order = sorted(
        range(len(boxes_xyxy)),
        key=lambda i: (boxes_xyxy[i][1], boxes_xyxy[i][0]),
    )
    items: List[dict] = []
    for i in order:
        x1, y1, x2, y2 = (float(v) for v in boxes_xyxy[i])
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        layout = _convert_layout((x1, y1, x2, y2), fmt.box_format)
        coords = [_scale_coord(v, fmt.coord_divisor) for v in layout]
        # Key order matches the prompt examples (box first for bbox_2d
        # families, label first for the [0,1] default) purely for target
        # naturalness; the parser accepts either order.
        if fmt.bbox_key == "bbox_2d":
            items.append({fmt.bbox_key: coords, "label": str(labels[i])})
        else:
            items.append({"label": str(labels[i]), fmt.bbox_key: coords})
    return json.dumps(items, ensure_ascii=False, separators=(", ", ": "))

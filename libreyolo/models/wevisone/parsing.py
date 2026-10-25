# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/Tencent/WeVisOne-VL
# (evaluation/utils/parser.py and parse_seg_boxes in wevisone_vl/engine.py)
# at commit 9a6bcd85404505df0d05cfb5d97512ccacf35dee.
# Copyright (C) 2026 Tencent. All rights reserved.
# Modified for LibreYOLO: one scanner serves boxes, points and polygons; a
# missing closing tag no longer swallows the next <ref>; every <OBJ> keeps its
# ordinal (None when its box is malformed) so masks stay aligned with the
# hidden states of the teacher-forced pass.
"""Parse WeVisOne-VL's tagged coordinate answers.

The checkpoint answers with tags such as::

    <ref>person</ref><box><12><40><380><990>,<500><42><880><991></box>
    <ref>cup</ref><point><512><640></point>
    <ref>dog</ref><box><100><200><300><400><OBJ></box>

Coordinates are bins on a 0-1000 scale of the original image. ``None`` means
the target is absent, and the closing tag may be missing when generation was
truncated. ``<OBJ>`` follows each box whose mask the pixel decoder predicts.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import List, Optional, Tuple

COORD_SCALE = 1000.0
OBJ_TOKEN = "<OBJ>"

_END_MARKERS = ("<|im_end|>", "<|endoftext|>")
_REF = re.compile(r"<ref>(.*?)</ref>", re.DOTALL)
_SHAPE = re.compile(
    r"<(box|point|polygon)>(.*?)(?:</\1>|(?=<ref>|<box>|<point>|<polygon>)|$)",
    re.DOTALL,
)
_BIN = re.compile(r"<(\d+)>")


def strip_response(text: str) -> str:
    """Cut the decoded text at the first end-of-turn marker."""
    text = text or ""
    for marker in _END_MARKERS:
        text = text.split(marker, 1)[0]
    return text.strip()


def _iter_shapes(text: str) -> Iterator[Tuple[Optional[str], str, int, int]]:
    """Yield ``(label, kind, body_start, body_end)`` for every shape tag.

    The label is the closest preceding ``<ref>`` text, or None for a bare tag.
    """
    refs = [(m.end(), m.group(1).strip()) for m in _REF.finditer(text)]
    for match in _SHAPE.finditer(text):
        label = None
        for end, name in refs:
            if end > match.start():
                break
            label = name
        yield label, match.group(1), match.start(2), match.end(2)


def _on_grid(values) -> bool:
    return all(0 <= v <= COORD_SCALE for v in values)


def _valid_box(values) -> bool:
    x1, y1, x2, y2 = values
    return _on_grid(values) and x2 > x1 and y2 > y1


def parse_shapes(text: str) -> List[dict]:
    """Return ``{"label", "kind", "coords"}`` items with coords in raw bins.

    ``kind`` is ``box`` (``[x1, y1, x2, y2]``), ``point`` (``[x, y]``) or
    ``polygon`` (flat ``[x0, y0, x1, y1, ...]``). Shapes with a coordinate
    off the 0-1000 grid and degenerate boxes are dropped.
    """
    text = strip_response(text)
    items: List[dict] = []
    for label, kind, start, end in _iter_shapes(text):
        body = text[start:end].replace(OBJ_TOKEN, "")
        for segment in body.strip().rstrip(",").split(","):
            values = [int(v) for v in _BIN.findall(segment)]
            if kind == "point":
                for k in range(0, len(values) - 1, 2):
                    if _on_grid(values[k : k + 2]):
                        items.append(
                            {"label": label, "kind": kind, "coords": values[k : k + 2]}
                        )
            elif kind == "polygon":
                if len(values) >= 6 and _on_grid(values):
                    items.append(
                        {
                            "label": label,
                            "kind": kind,
                            "coords": values[: len(values) - len(values) % 2],
                        }
                    )
            else:
                for k in range(0, len(values) - 3, 4):
                    box = values[k : k + 4]
                    if _valid_box(box):
                        items.append({"label": label, "kind": kind, "coords": box})
    return items


def parse_obj_boxes(
    text: str,
) -> List[Optional[Tuple[Optional[str], Tuple[float, float, float, float]]]]:
    """One entry per ``<OBJ>`` in order: ``(label, box01)`` or None.

    ``box01`` is the normalized ``xyxy`` box written right before that
    ``<OBJ>`` inside a ``<box>`` tag. None marks an ``<OBJ>`` without a valid
    box, so callers can still index the n-th ``<OBJ>`` hidden state.
    """
    text = strip_response(text)
    boxes = [
        (label, start, end)
        for label, kind, start, end in _iter_shapes(text)
        if kind == "box"
    ]
    entries: List[
        Optional[Tuple[Optional[str], Tuple[float, float, float, float]]]
    ] = []
    for match in re.finditer(re.escape(OBJ_TOKEN), text):
        position = match.start()
        entry = None
        for label, start, end in boxes:
            if start <= position <= end:
                prefix = text[start:position].rsplit(OBJ_TOKEN, 1)[-1]
                values = [int(v) for v in _BIN.findall(prefix)]
                if len(values) >= 4 and _valid_box(values[-4:]):
                    entry = (label, tuple(v / COORD_SCALE for v in values[-4:]))
                break
        entries.append(entry)
    return entries

"""Parse DeepSeek-OCR grounding output into OCR regions and page text.

Pure string handling, no model or tensor code. Grounded output interleaves
``<|ref|>LABEL<|/ref|><|det|>[[x1, y1, x2, y2], ...]<|/det|>`` blocks with
plain text. Coordinates are on a 0 to 999 grid (the release scales them back
with ``x / 999 * width``).

Two shapes occur:

* the OCR prompt puts the transcript inside ``<|ref|>`` and nothing after the
  box, so each box is one text line;
* the markdown prompt puts a layout label (``text``, ``title``, ``table``,
  ``image``...) inside ``<|ref|>`` and the block's markdown after the box.

One rule covers both: a box's transcript is the text that follows it up to
the next block when there is any, else the ``<|ref|>`` content. Boxes with an
empty transcript (figures) are not text regions and are dropped.
"""

from __future__ import annotations

import ast
import re
from typing import Dict, List, Tuple

COORD_GRID = 999.0
EOS_TEXT = "<｜end▁of▁sentence｜>"

_BLOCK_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>", re.DOTALL)
# A trailing, unterminated block left by a truncated generation.
_DANGLING_RE = re.compile(r"<\|ref\|>.*\Z", re.DOTALL)


def strip_eos(text: str) -> str:
    """Drop the end-of-sentence marker the decoder emits, as the release does."""
    if text.endswith(EOS_TEXT):
        text = text[: -len(EOS_TEXT)]
    return text


def _parse_boxes(raw: str) -> List[List[float]]:
    try:
        value = ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError):
        return []
    if (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(v, (int, float)) for v in value)
    ):
        value = [value]
    boxes = []
    if not isinstance(value, (list, tuple)):
        return boxes
    for box in value:
        if (
            isinstance(box, (list, tuple))
            and len(box) == 4
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in box
            )
        ):
            boxes.append([float(v) for v in box])
    return boxes


def parse_grounding(text: str) -> List[Dict]:
    """Grounded blocks in emission order.

    Each item is ``{"label": layout label, "text": transcript, "box": [x1, y1,
    x2, y2]}`` with the box still on the 0 to 999 grid. ``label`` is the
    ``<|ref|>`` content when it names a layout block (text follows the box),
    else ``"text"``. A ``<|det|>`` holding
    several boxes yields one item per box with the same transcript.
    """
    text = strip_eos(text)
    matches = list(_BLOCK_RE.finditer(text))
    items: List[Dict] = []
    for i, match in enumerate(matches):
        label = match.group(1).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        trailing = _DANGLING_RE.sub("", text[match.end() : end]).strip()
        transcript = trailing if trailing else label
        kind = label if trailing else "text"
        for box in _parse_boxes(match.group(2)):
            items.append({"label": kind, "text": transcript, "box": box})
    return items


def grid_box_to_pixels(
    box: List[float], width: int, height: int
) -> Tuple[float, float, float, float]:
    """Scale a 0 to 999 grid box to pixels, ordered and clamped to the image."""
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    def sx(v):
        return min(max(v / COORD_GRID * width, 0.0), float(width))

    def sy(v):
        return min(max(v / COORD_GRID * height, 0.0), float(height))

    return sx(x1), sy(y1), sx(x2), sy(y2)


def to_page_text(text: str) -> str:
    """Page-level text with the grounding markup removed.

    Mirrors how the release writes its ``result.mmd``: grounding blocks are
    deleted (figure blocks included, since no crops are saved here), two LaTeX
    colon macros are normalized, and the result is stripped.
    """
    text = strip_eos(text)
    text = _BLOCK_RE.sub("", text)
    text = _DANGLING_RE.sub("", text)
    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    return text.strip()


__all__ = [
    "COORD_GRID",
    "EOS_TEXT",
    "grid_box_to_pixels",
    "parse_grounding",
    "strip_eos",
    "to_page_text",
]

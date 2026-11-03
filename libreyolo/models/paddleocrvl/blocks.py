# SPDX-License-Identifier: Apache-2.0
# Adapted from PaddleX (https://github.com/PaddlePaddle/PaddleX, commit
# 4c968d30b9d0d4a693d9eaaef0881c20f8d7d0e0, Apache-2.0, Copyright (c) 2024
# PaddlePaddle Authors): paddlex/inference/pipelines/paddleocr_vl/pipeline.py
# (prompt routing, per-element pixel budgets, content clean-up) and
# paddlex/inference/pipelines/paddleocr_vl/uilts.py (crop_margin, OTSL to HTML,
# repetition truncation). See NOTICE.
"""Per-region preparation and content clean-up for PaddleOCR-VL.

Each layout region becomes one recognition request: a crop, a task prompt
("OCR:", "Table Recognition:", "Formula Recognition:", "Chart Recognition:")
and a pixel budget for the dynamic-resolution encoder. The model's raw text is
then cleaned the way the upstream pipeline does: runaway repetition is cut,
LaTeX delimiters are normalized to ``$``/``$$`` and table OTSL becomes HTML.
"""

from __future__ import annotations

import html
import itertools
import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

# Upstream per-element pixel budget for the vision encoder (min, max).
DEFAULT_MIN_PIXELS = 112896
DEFAULT_MAX_PIXELS = 1003520

# Layout classes that carry no text to recognize by default.
IMAGE_LABELS: Tuple[str, ...] = ("image", "header_image", "footer_image")


def skipped_labels(use_chart_recognition: bool = False) -> Tuple[str, ...]:
    """Labels never sent to the recognizer (pictures, plus charts and seals)."""
    labels = list(IMAGE_LABELS)
    if not use_chart_recognition:
        labels.append("chart")
    labels.append("seal")
    return tuple(labels)


def prompt_kind(label: str, use_chart_recognition: bool = False) -> str:
    """Map a layout label to the recognizer task (``ocr``/``table``/...)."""
    if label == "table":
        return "table"
    if label == "chart" and use_chart_recognition:
        return "chart"
    if "formula" in label and label != "formula_number":
        return "formula"
    return "ocr"


def crop_box(rgb: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """Integer-truncated crop of an ``(H, W, 3)`` page, as upstream does."""
    x1, y1, x2, y2 = (int(v) for v in box)
    return rgb[y1:y2, x1:x2].copy()


def crop_margin(img: np.ndarray) -> np.ndarray:
    """Trim the blank margin around a formula crop (RGB uint8 input)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    max_val, min_val = int(gray.max()), int(gray.min())
    if max_val == min_val:
        return img
    lut = np.zeros(256, dtype=np.uint8)
    for v in range(min_val, max_val + 1):
        lut[v] = int((v - min_val) / (max_val - min_val) * 255)
    data = cv2.LUT(gray, lut)
    _, binary = cv2.threshold(data, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y : y + h, x : x + w]


def prepare_region(
    rgb: np.ndarray, block: dict, use_chart_recognition: bool = False
) -> Optional[Tuple[np.ndarray, str]]:
    """Crop and prompt for one layout block, or None when it is skipped."""
    label = block["label"]
    if label in skipped_labels(use_chart_recognition):
        return None
    crop = crop_box(rgb, block["coordinate"])
    if crop.size == 0:
        return None
    kind = prompt_kind(label, use_chart_recognition)
    if kind == "formula":
        trimmed = crop_margin(crop)
        # Upstream compares (rows, cols) here despite naming them (w, h).
        if trimmed.shape[0] > 2 and trimmed.shape[1] > 2:
            crop = trimmed
    return crop, PROMPTS[kind]


# =============================================================================
# Repetition truncation
# =============================================================================


def _shortest_repeating_unit(s: str) -> Optional[str]:
    n = len(s)
    for i in range(1, n // 2 + 1):
        if n % i == 0 and s[:i] * (n // i) == s:
            return s[:i]
    return None


def _repeating_suffix(s: str, min_len: int = 8, min_repeats: int = 5):
    for i in range(len(s) // min_repeats, min_len - 1, -1):
        unit = s[-i:]
        if s.endswith(unit * min_repeats):
            count, rest = 0, s
            while rest.endswith(unit):
                rest = rest[:-i]
                count += 1
            return s[: len(s) - count * i], unit, count
    return None


def truncate_repetitive_content(
    content: str,
    line_threshold: int = 10,
    char_threshold: int = 10,
    min_len: int = 10,
    min_count: int = 3000,
) -> str:
    """Cut phrase, character or line repetition from a degenerate generation."""
    if len(content) < min_count:
        return content
    stripped = content.strip()
    if not stripped:
        return content
    if "\n" not in stripped and len(stripped) > 100:
        match = _repeating_suffix(stripped)
        if match:
            prefix, unit, count = match
            if len(unit) * count > len(stripped) * 0.5:
                return prefix
    if "\n" not in stripped and len(stripped) > min_len:
        unit = _shortest_repeating_unit(stripped)
        if unit and len(stripped) // len(unit) >= char_threshold:
            return unit
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    if len(lines) < line_threshold:
        return content
    line, count = Counter(lines).most_common(1)[0]
    if count >= line_threshold and count / len(lines) >= 0.8:
        return line
    return content


# =============================================================================
# OTSL (table structure tokens) to HTML
# =============================================================================

OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"
_OTSL_TAGS = (OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL)
_TAG_GROUP = "(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>)"
_OTSL_FIND = re.compile(f"{_TAG_GROUP}.*?(?={_TAG_GROUP}|$)", flags=re.DOTALL)
_OTSL_SPLIT = re.compile("(" + "|".join(_OTSL_TAGS) + ")")


def _otsl_pad_to_rectangle(otsl: str) -> str:
    """Pad or trim every row to the width that minimizes total edits."""
    otsl = otsl.strip()
    if OTSL_NL not in otsl:
        return otsl + OTSL_NL
    rows = []
    for line in otsl.split(OTSL_NL):
        cells = _OTSL_FIND.findall(line) if line else []
        if not cells:
            continue
        min_len = 0
        for i, cell in enumerate(cells):
            if cell.startswith(OTSL_FCEL):
                min_len = i + 1
        rows.append((cells, min_len))
    if not rows:
        return OTSL_NL
    lo = max(m for _, m in rows)
    hi = max(lo, max(len(c) for c, _ in rows))
    best_width, best_cost = hi, float("inf")
    for width in range(lo, hi + 1):
        cost = sum(abs(len(c) - width) for c, _ in rows)
        if cost < best_cost:
            best_cost, best_width = cost, width
    fixed = []
    for cells, _ in rows:
        cells = cells[:best_width] + [OTSL_ECEL] * max(0, best_width - len(cells))
        fixed.append("".join(cells))
    return OTSL_NL.join(fixed) + OTSL_NL


def _otsl_cells(texts: List[str], tokens: List[str]):
    """Parse OTSL tags and texts into cells ``(text, r0, r1, c0, c1)``."""
    grid = [list(g) for is_nl, g in itertools.groupby(tokens, lambda t: t == OTSL_NL) if not is_nl]
    if grid:
        width = max(len(r) for r in grid)
        for row in grid:
            row.extend([OTSL_ECEL] * (width - len(row)))
        rebuilt, k = [], 0
        for row in grid:
            for tag in row:
                rebuilt.append(tag)
                if k < len(texts) and texts[k] == tag:
                    k += 1
                    if k < len(texts) and texts[k] not in _OTSL_TAGS:
                        rebuilt.append(texts[k])
                        k += 1
            rebuilt.append(OTSL_NL)
            if k < len(texts) and texts[k] == OTSL_NL:
                k += 1
        texts = rebuilt

    def span(r: int, c: int, along_row: bool, tags) -> int:
        n = 0
        while True:
            if grid[r][c] not in tags:
                return n
            n += 1
            if along_row:
                c += 1
                if c >= len(grid[r]):
                    return n
            else:
                r += 1
                if r >= len(grid):
                    return n

    cells = []
    r = c = 0
    for i, text in enumerate(texts):
        if text in (OTSL_FCEL, OTSL_ECEL):
            row_span = col_span = 1
            offset = 1
            cell_text = ""
            if text != OTSL_ECEL:
                cell_text = texts[i + 1]
                offset = 2
            right = texts[i + offset] if i + offset < len(texts) else ""
            below = ""
            if r + 1 < len(grid) and c < len(grid[r + 1]):
                below = grid[r + 1][c]
            if right in (OTSL_LCEL, OTSL_XCEL):
                col_span += span(r, c + 1, True, (OTSL_LCEL, OTSL_XCEL))
            if below in (OTSL_UCEL, OTSL_XCEL):
                row_span += span(r + 1, c, False, (OTSL_UCEL, OTSL_XCEL))
            cells.append((cell_text.strip(), r, r + row_span, c, c + col_span))
        if text in (OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL):
            c += 1
        if text == OTSL_NL:
            r += 1
            c = 0
    return cells, grid


def otsl_to_html(otsl: str) -> str:
    """Convert an OTSL table string to an HTML ``<table>`` ("" when empty)."""
    otsl = _otsl_pad_to_rectangle(otsl)
    tokens = _OTSL_SPLIT.findall(otsl)
    texts = [t for t in _OTSL_SPLIT.split(otsl) if t.strip()]
    cells, grid = _otsl_cells(texts, tokens)
    if not cells:
        return ""
    nrows = len(grid)
    ncols = max(len(r) for r in grid) if grid else 0
    owner = [[None] * ncols for _ in range(nrows)]
    for cell in cells:
        for i in range(min(cell[1], nrows), min(cell[2], nrows)):
            for j in range(min(cell[3], ncols), min(cell[4], ncols)):
                owner[i][j] = cell
    body = ""
    for i in range(nrows):
        body += "<tr>"
        for j in range(ncols):
            cell = owner[i][j]
            if cell is None:
                cell = ("", i, i + 1, j, j + 1)
            text, r0, r1, c0, c1 = cell
            if r0 != i or c0 != j:
                continue
            tag = "td"
            if r1 - r0 > 1:
                tag += f' rowspan="{r1 - r0}"'
            if c1 - c0 > 1:
                tag += f' colspan="{c1 - c0}"'
            body += f"<{tag}>{html.escape(text.strip())}</td>"
        body += "</tr>"
    return f"<table>{body}</table>"


# =============================================================================
# Content clean-up
# =============================================================================


def clean_content(text: Optional[str], label: str) -> str:
    """Apply the upstream post-processing to one region's generated text."""
    text = text or ""
    text = truncate_repetitive_content(text, min_count=5000 if label == "table" else 50)
    if ("\\(" in text and "\\)" in text) or ("\\[" in text and "\\]" in text):
        text = text.replace("$", "")
        text = (
            text.replace("\\(", " $ ")
            .replace("\\)", " $")
            .replace("\\[\\[", "\\[")
            .replace("\\]\\]", "\\]")
            .replace("\\[", " $$ ")
            .replace("\\]", " $$ ")
        )
        if label == "formula_number":
            text = text.replace("$", "")
    if label == "table":
        try:
            table = otsl_to_html(text)
        except IndexError:  # malformed OTSL: keep the raw tokens
            table = ""
        if table:
            text = table
    return text


__all__ = [
    "PROMPTS",
    "DEFAULT_MIN_PIXELS",
    "DEFAULT_MAX_PIXELS",
    "IMAGE_LABELS",
    "skipped_labels",
    "prompt_kind",
    "crop_box",
    "crop_margin",
    "prepare_region",
    "truncate_repetitive_content",
    "otsl_to_html",
    "clean_content",
]

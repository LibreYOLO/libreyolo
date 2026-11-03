# SPDX-License-Identifier: Apache-2.0
# Adapted from PaddleX (https://github.com/PaddlePaddle/PaddleX, commit
# 4c968d30b9d0d4a693d9eaaef0881c20f8d7d0e0, Apache-2.0, Copyright (c) 2024
# PaddlePaddle Authors): paddlex/inference/common/result/converter/
# markdown_converter.py and markdown_format_funcs.py (plain, non-HTML mode),
# and the PaddleOCR-VL markdown_ignore_labels default. See NOTICE.
"""Assemble the page markdown from recognized regions in reading order."""

from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, Mapping, Tuple

# Labels left out of the page markdown by the PaddleOCR-VL defaults.
MARKDOWN_IGNORE_LABELS: Tuple[str, ...] = (
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
)

_NUMBERING = (
    r"(?:"
    r"[1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?|"
    r"[\(\（](?:[1-9][0-9]*|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[\)\）]|"
    r"[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+[、\.]?|"
    r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\s)"
    r")"
)
_TITLE_RE = re.compile(r"^\s*(" + _NUMBERING + r")(\s*)(.*)$")


def _collapse(s: str) -> str:
    return s.replace("-\n", "").replace("\n", " ")


def _paragraphs(s: str) -> str:
    return s.replace("\n\n", "\n").replace("\n", "\n\n")


def _title(s: str) -> str:
    match = _TITLE_RE.match(s)
    if match:
        s = match.group(1).strip() + " " + match.group(3).lstrip()
    s = s.rstrip(".")
    level = s.count(".") + 1 if "." in s else 1
    return _collapse(f"#{'#' * level} {s}")


def _first_line(
    s: str, templates: Iterable[str], fmt: Callable[[str], str], splitter: str
) -> str:
    lines = s.split(splitter)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if line.lower() in templates:
            lines[i] = fmt(line)
        break
    return splitter.join(lines)


def _plain(s: str) -> str:
    return s


HANDLERS: Dict[str, Callable[[str], str]] = {
    "paragraph_title": _title,
    "doc_title": lambda s: _collapse(f"# {s}"),
    "figure_title": _plain,
    "vision_footnote": _paragraphs,
    "text": _paragraphs,
    "vertical_text": _paragraphs,
    "reference_content": _paragraphs,
    "abstract": lambda s: _first_line(
        s, ("摘要", "abstract"), lambda x: f"## {x}\n", " "
    ),
    "content": lambda s: s.replace("-\n", "  \n").replace("\n", "  \n"),
    "display_formula": _plain,
    "inline_formula": _plain,
    "formula_number": _plain,
    "table": lambda s: (
        "\n"
        + ("\n" + s)
        .replace("<html>", "")
        .replace("</html>", "")
        .replace("<body>", "")
        .replace("</body>", "")
    ),
    "chart": _plain,
    "reference": lambda s: _first_line(
        s, ("参考文献", "references"), lambda x: f"## {x}", "\n"
    ),
    "algorithm": lambda s: s.strip("\n"),
    "number": _plain,
    "footnote": _plain,
    "header": _plain,
    "footer": _plain,
    "aside_text": _plain,
}


def build_markdown(
    regions: Iterable[Mapping[str, str]],
    ignore_labels: Iterable[str] = MARKDOWN_IGNORE_LABELS,
) -> str:
    """Join ``{"label", "text"}`` regions (reading order) into page markdown.

    Pictures are not exported, so image regions contribute nothing.
    """
    ignored = set(ignore_labels)
    parts = []
    for region in regions:
        label = region["label"]
        handler = HANDLERS.get(label)
        if handler is None or label in ignored:
            continue
        parts.append(handler(region["text"]))
    return "\n\n".join(parts)


__all__ = ["MARKDOWN_IGNORE_LABELS", "build_markdown"]

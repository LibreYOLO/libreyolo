# SPDX-License-Identifier: Apache-2.0
# Adapted from the GLM-OCR SDK (https://github.com/zai-org/GLM-OCR, commit
# 0288f96bb9e78cb439883eaa725eeae90bd177c5, Apache-2.0, Copyright (c) Z.ai /
# zai-org): glmocr/postprocess/result_formatter.py (ResultFormatter: content
# clean-up and formatting, formula-number merge, hyphenated text merge, bullet
# points, markdown) and glmocr/utils/result_postprocess_utils.py. See NOTICE.
"""Page assembly of the GLM-OCR pipeline: per-region clean-up and markdown.

Changes from the SDK: picture regions are not exported as image files, so
they carry no markdown line; everything else follows the SDK defaults
(``enable_merge_formula_numbers``, ``enable_merge_text_blocks`` and
``enable_format_bullet_points`` all on).
"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional

try:  # Optional, as in the SDK: better validation of hyphen-merged words.
    from wordfreq import zipf_frequency
except Exception:  # pragma: no cover - depends on the environment
    zipf_frequency = None

# ``label_visualization_mapping`` of the SDK config.
LABEL_GROUPS: Dict[str, tuple] = {
    "table": ("table",),
    "formula": ("display_formula", "inline_formula"),
    "image": ("chart", "image"),
    "text": (
        "abstract",
        "algorithm",
        "content",
        "doc_title",
        "figure_title",
        "paragraph_title",
        "reference_content",
        "text",
        "vertical_text",
        "vision_footnote",
        "seal",
        "formula_number",
    ),
}

_INLINE_FORMULA_RE = re.compile(r"(?<!\$)\$\s*((?:[^$\\]|\\.)+?)\s*\$(?!\$)")


# =============================================================================
# Content helpers (glmocr/utils/result_postprocess_utils.py)
# =============================================================================


def find_consecutive_repeat(
    s: str, min_unit_len: int = 10, min_repeats: int = 10
) -> Optional[str]:
    """Cut ``s`` after the first run of a unit repeated ``min_repeats`` times."""
    n = len(s)
    if n < min_unit_len * min_repeats:
        return None
    max_unit_len = n // min_repeats
    if max_unit_len < min_unit_len:
        return None
    pattern = re.compile(
        r"(.{"
        + str(min_unit_len)
        + ","
        + str(max_unit_len)
        + r"}?)\1{"
        + str(min_repeats - 1)
        + ",}",
        re.DOTALL,
    )
    match = pattern.search(s)
    if match:
        return s[: match.start()] + match.group(1)
    return None


def clean_repeated_content(
    content: str, min_len: int = 10, min_repeats: int = 10, line_threshold: int = 10
) -> str:
    """Remove consecutive and line-level repetition loops."""
    stripped = content.strip()
    if not stripped:
        return content
    if len(stripped) > min_len * min_repeats:
        result = find_consecutive_repeat(
            stripped, min_unit_len=min_len, min_repeats=min_repeats
        )
        if result is not None:
            return result
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    total = len(lines)
    if total >= line_threshold and lines:
        common, count = Counter(lines).most_common(1)[0]
        if count >= line_threshold and (count / total) >= 0.8:
            for i, line in enumerate(lines):
                if line == common:
                    consecutive = sum(
                        1
                        for j in range(i, min(i + 3, len(lines)))
                        if lines[j] == common
                    )
                    if consecutive >= 3:
                        original_lines = content.split("\n")
                        non_empty = 0
                        for idx, orig in enumerate(original_lines):
                            if orig.strip():
                                non_empty += 1
                                if non_empty == i + 1:
                                    return "\n".join(original_lines[: idx + 1])
                        break
    return content


def clean_formula_number(number_content: str) -> str:
    """``"(1)"`` -> ``"1"``, also with full-width parentheses."""
    number = number_content.strip()
    if number.startswith("(") and number.endswith(")"):
        number = number[1:-1]
    elif number.startswith("（") and number.endswith("）"):
        number = number[1:-1]
    return number


def normalize_inline_formula(content: str) -> str:
    """``$ x $`` -> ``$x$`` with one space between the formula and words."""
    if "$" not in content:
        return content
    parts: list = []
    last_end = 0
    for m in _INLINE_FORMULA_RE.finditer(content):
        formula = m.group(1).strip()
        if not formula:
            continue
        start, end = m.start(), m.end()
        before = content[last_end:start]
        if before and before[-1].isalnum():
            before += " "
        parts.append(before)
        parts.append(f"${formula}$")
        if end < len(content) and content[end].isalnum():
            parts.append(" ")
        last_end = end
    if not parts:
        return content
    parts.append(content[last_end:])
    return "".join(parts)


def clean_content(content: Optional[str]) -> str:
    """The SDK's ``_clean_content``."""
    if content is None:
        return ""
    content = re.sub(r"^(\\t)+", "", content).lstrip()
    content = re.sub(r"(\\t)+$", "", content).rstrip()
    content = re.sub(r"(\.)\1{2,}", r"\1\1\1", content)
    content = re.sub(r"(·)\1{2,}", r"\1\1\1", content)
    content = re.sub(r"(_)\1{2,}", r"\1\1\1", content)
    content = re.sub(r"(\\_)\1{2,}", r"\1\1\1", content)
    if len(content) >= 2048:
        content = clean_repeated_content(content)
    content = normalize_inline_formula(content)
    return content.strip()


def map_label(label: str) -> str:
    """Native layout class -> ``text`` / ``table`` / ``formula`` / ``image``."""
    for group in ("image", "text", "table", "formula"):
        if label in LABEL_GROUPS[group]:
            return group
    return label


def format_content(content: Optional[str], label: str, native_label: str):
    """The SDK's ``_format_content``: markdown for one region."""
    if content is None:
        return content
    if label == "table":
        if content.startswith("<table") and content.endswith("</table>"):
            content = content.strip()
        else:
            content = clean_content(str(content))
    elif label == "formula":
        if content.startswith("$$") and content.endswith("$$"):
            content = content.strip()
        else:
            content = clean_content(str(content))
    else:
        content = clean_content(str(content))

    if native_label == "doc_title":
        content = re.sub(r"^#+\s*", "", content)
        content = "# " + content
    elif native_label == "paragraph_title":
        if content.startswith("- ") or content.startswith("* "):
            content = content[2:].lstrip()
        content = re.sub(r"^#+\s*", "", content)
        content = "## " + content.lstrip()

    if label == "formula":
        if content.startswith(("$$", "\\[", "\\(")):
            content = content[2:].strip()
        if content.endswith(("$$", "\\]", "\\)")):
            content = content[:-2].strip()
        content = "$$\n" + content + "\n$$"

    if label == "text":
        if content.startswith("```") and not content.endswith("```"):
            content = content + "\n```"
        if (
            content.startswith("·")
            or content.startswith("•")
            or content.startswith("* ")
        ):
            content = "- " + content[1:].lstrip()
        match = re.match(r"^(\(|\（)(\d+|[A-Za-z])(\)|\）)(.*)$", content)
        if match:
            _, symbol, _, rest = match.groups()
            content = f"({symbol}) {rest.lstrip()}"
        match = re.match(r"^(\d+|[A-Za-z])(\.|\)|\）)(.*)$", content)
        if match:
            symbol, sep, rest = match.groups()
            sep = ")" if sep == "）" else sep
            content = f"{symbol}{sep} {rest.lstrip()}"
        content = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", content)
    return content


# =============================================================================
# Block merges (ResultFormatter)
# =============================================================================


def _is_likely_valid_merged_word(merged_word: str) -> bool:
    token = merged_word.strip().lower()
    if not token:
        return False
    if zipf_frequency is not None:
        try:
            return zipf_frequency(token, "en") >= 2.5
        except Exception:
            pass
    if not re.fullmatch(r"[a-z][a-z'\-]{2,30}", token):
        return False
    if "--" in token or "''" in token:
        return False
    return True


def merge_formula_numbers(blocks: List[dict]) -> List[dict]:
    """Fold a ``formula_number`` block into its neighbouring formula as
    ``\\tag{n}``; a formula number with no formula next to it is dropped."""
    if not blocks:
        return blocks
    merged, skip = [], set()
    for i, block in enumerate(blocks):
        if i in skip:
            continue
        native = block.get("native_label", "")
        if native == "formula_number":
            if i + 1 < len(blocks) and blocks[i + 1].get("label") == "formula":
                nxt = blocks[i + 1]
                number = clean_formula_number(block.get("content", "").strip())
                formula = nxt.get("content", "")
                out = deepcopy(nxt)
                if formula.endswith("\n$$"):
                    out["content"] = formula[:-3] + f" \\tag{{{number}}}\n$$"
                merged.append(out)
                skip.add(i + 1)
            continue
        if block.get("label") == "formula":
            if (
                i + 1 < len(blocks)
                and blocks[i + 1].get("native_label") == "formula_number"
            ):
                number = clean_formula_number(blocks[i + 1].get("content", "").strip())
                formula = block.get("content", "")
                out = deepcopy(block)
                if formula.endswith("\n$$"):
                    out["content"] = formula[:-3] + f" \\tag{{{number}}}\n$$"
                merged.append(out)
                skip.add(i + 1)
                continue
            merged.append(block)
            continue
        merged.append(block)
    for idx, block in enumerate(merged):
        block["index"] = idx
    return merged


def merge_text_blocks(blocks: List[dict]) -> List[dict]:
    """Join a text block ending in a hyphen with the next text block when the
    rejoined word looks valid (``exam-`` + ``ple`` -> ``example``)."""
    if not blocks:
        return blocks
    merged, skip = [], set()
    for i, block in enumerate(blocks):
        if i in skip:
            continue
        content = block.get("content", "")
        if block.get("label") != "text" or not isinstance(content, str):
            merged.append(block)
            continue
        stripped = content.rstrip()
        if not stripped or not stripped.endswith("-"):
            merged.append(block)
            continue
        done = False
        # The first later text block starting in lowercase is the candidate;
        # other blocks in between are skipped over (SDK behaviour).
        for j in range(i + 1, len(blocks)):
            if blocks[j].get("label") != "text":
                continue
            next_content = blocks[j].get("content", "")
            if not isinstance(next_content, str):
                continue
            next_stripped = next_content.lstrip()
            if next_stripped and next_stripped[0].islower():
                before = stripped[:-1].split()
                after = next_stripped.split()
                if before and after:
                    if _is_likely_valid_merged_word(before[-1] + after[0]):
                        out = deepcopy(block)
                        out["content"] = stripped[:-1] + next_content.lstrip()
                        merged.append(out)
                        skip.add(j)
                        done = True
                break
        if not done:
            merged.append(block)
    for idx, block in enumerate(merged):
        block["index"] = idx
    return merged


def format_bullet_points(
    blocks: List[dict], left_align_threshold: float = 10.0
) -> List[dict]:
    """Add a missing ``- `` to a text block sitting between two left-aligned
    bullet items."""
    if len(blocks) < 3:
        return blocks
    for i in range(1, len(blocks) - 1):
        cur, prev, nxt = blocks[i], blocks[i - 1], blocks[i + 1]
        if cur.get("native_label") != "text":
            continue
        if prev.get("native_label") != "text" or nxt.get("native_label") != "text":
            continue
        cur_content = cur.get("content", "")
        if cur_content.startswith("- "):
            continue
        if not (
            prev.get("content", "").startswith("- ")
            and nxt.get("content", "").startswith("- ")
        ):
            continue
        cur_box, prev_box, next_box = (
            cur.get("bbox_2d", []),
            prev.get("bbox_2d", []),
            nxt.get("bbox_2d", []),
        )
        if not (cur_box and prev_box and next_box):
            continue
        if (
            abs(cur_box[0] - prev_box[0]) <= left_align_threshold
            and abs(cur_box[0] - next_box[0]) <= left_align_threshold
        ):
            cur["content"] = "- " + cur_content
    return blocks


def format_page(regions: List[dict]) -> List[dict]:
    """Format one page's recognized regions (the SDK's layout-mode
    ``ResultFormatter.process``).

    Each input region carries ``index``, ``label`` (native layout class),
    ``bbox_2d``, ``task_type`` and ``content`` (``None`` for pictures or a
    failed read) plus any extra keys, which are kept. Returns the blocks in
    reading order with ``label`` mapped to ``text`` / ``table`` / ``formula``
    / ``image``, ``native_label`` set, and ``content`` as markdown.
    """
    blocks = []
    for item in sorted(regions, key=lambda x: x.get("index", 0)):
        block = deepcopy(item)
        block["native_label"] = block.get("label", "text")
        block["label"] = map_label(block["native_label"])
        block["content"] = format_content(
            block.get("content"), block["label"], block["native_label"]
        )
        is_image = block["label"] == "image" or block.get("task_type") == "skip"
        if not is_image:
            content = block.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                continue
        block["index"] = len(blocks)
        block["is_image"] = is_image
        block.pop("task_type", None)
        blocks.append(block)
    blocks = merge_formula_numbers(blocks)
    blocks = merge_text_blocks(blocks)
    return format_bullet_points(blocks)


def page_markdown(blocks: List[dict]) -> str:
    """Join the formatted blocks into the page markdown (pictures omitted)."""
    return "\n\n".join(
        b["content"] for b in blocks if not b.get("is_image") and b.get("content")
    )

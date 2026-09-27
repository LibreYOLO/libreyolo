"""Qwen3-VL used as a generalist point grounder.

Same weights as ``LibreVLM``'s Qwen3-VL family. The ask is a single click
on a 0–1000 grid, not a box list. Useful when no GUI-specialized model is
loaded yet; prefer ShowUI for screens.
"""

from __future__ import annotations

from typing import ClassVar

from .base import LibreGroundModel


class LibreGroundQwen3VL(LibreGroundModel):
    FAMILY = "ground_qwen3vl"
    FILENAME_PREFIX = "LibreGroundQwen3VL"

    HF_REPOS: ClassVar[dict[str, str]] = {
        "2b": "LibreYOLO/LibreGroundQwen3VL2b",
        "4b": "Qwen/Qwen3-VL-4B-Instruct",
        "8b": "Qwen/Qwen3-VL-8B-Instruct",
    }
    INPUT_SIZES: ClassVar[dict[str, int]] = {
        "2b": 1024,
        "4b": 1024,
        "8b": 1024,
    }
    COORD_SPACE = "milli"
    MAX_NEW_TOKENS = 128

    def _format_grounding_prompt(self, query: str) -> str:
        return (
            f"Point to: {query}. "
            "Reply with a JSON object "
            '{"point": [x, y], "label": "..."} '
            "where x and y are integers on a 0-1000 scale relative to the image. "
            "If the target is not visible, reply with {}."
        )

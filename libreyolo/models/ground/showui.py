"""ShowUI-2B: MIT-weight Qwen2-VL GUI grounder.

The pinned ``showlab/ShowUI-2B`` card declares MIT for the weights. The
ShowUI GitHub repository and the Qwen2-VL-2B base are Apache-2.0. Official
output is a bare ``[x, y]`` pair normalized to ``[0, 1]`` relative to the
screenshot. The system instruction below is copied verbatim from the pinned
MIT model card; see ``NOTICE``. ShowUI is the default ``LibreGround`` family
because it is small and loads through native transformers.
"""

from __future__ import annotations

from typing import ClassVar

from ..vlm.base import LibreVLMModel
from .base import LibreGroundModel


class LibreShowUI(LibreGroundModel):
    FAMILY = "showui"
    FILENAME_PREFIX = "LibreShowUI"

    HF_REPOS: ClassVar[dict[str, str]] = {
        "2b": "LibreYOLO/LibreShowUI2b",
    }
    INPUT_SIZES: ClassVar[dict[str, int]] = {
        "2b": 1344,
    }
    COORD_SPACE = "unit"
    MAX_NEW_TOKENS = 64
    # Upstream ships only pytorch_model.bin (no safetensors).
    SNAPSHOT_IGNORE_PATTERNS = tuple(
        pattern
        for pattern in LibreVLMModel.SNAPSHOT_IGNORE_PATTERNS
        if pattern not in ("*.bin", "*.bin.index.json")
    )

    def _format_grounding_prompt(self, query: str) -> str:
        return (
            "Based on the screenshot of the page, I give a text description "
            "and you give its corresponding location. The coordinate represents "
            "a clickable location [x, y] for an element, which is a relative "
            "coordinate on the screenshot, scaled from 0 to 1.\n"
            f"{query}"
        )

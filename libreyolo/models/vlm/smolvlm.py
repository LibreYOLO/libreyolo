"""LibreYOLO wrapper for HuggingFace's SmolVLM2 vision-language models.

SmolVLM2 (HuggingFaceTB, Apache-2.0) is a small general VLM. It follows the same
chat-template plus JSON-bbox output style as the base default (a ``bbox`` key on
a [0, 1] scale), so this family needs no parser override: it works through the
shared base with only the repo table declared. SmolVLM2 is a weak detector
compared with purpose-built grounding models, but it demonstrates that a new
model with no special handling drops straight into the tier.

Its processor depends on ``num2words`` (declared in the ``vlm`` extra).
"""

from __future__ import annotations

from typing import ClassVar, Dict

from .base import LibreVLMModel


class LibreSmolVLM2(LibreVLMModel):
    """SmolVLM2 used as an open-vocabulary detector (base default format)."""

    FAMILY = "smolvlm2"
    FILENAME_PREFIX = "LibreSmolVLM2"

    HF_REPOS: ClassVar[Dict[str, str]] = {
        "2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "500m": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "2.2b": 512,
        "500m": 512,
    }

    # Apache-2.0 weights: no restrictive-license notice needed.
    _LICENSE_NOTICE = ""

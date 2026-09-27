"""LibreYOLO wrapper for Liquid AI's LFM2-VL vision-language models.

LFM2.5-VL is a compact on-device VLM family (450M to 3B) with a native object-detection
prompt that returns ``[{"label", "bbox": [x1, y1, x2, y2]}]``. The coordinate
scale is size-dependent: [0, 1] on the 450M/1.6B, 0-1000 on the 3B (see
``_COORD_DIVISORS``). This family wraps it so it behaves like any LibreYOLO
detector.

Licensing note: LFM2-VL weights are published under the LFM Open License v1.0,
which is permissive for research and for organizations under a revenue
threshold but is NOT an OSI / MIT / Apache-2.0 license. LibreYOLO ships no LFM
source code (the model loads through the Apache-2.0 ``transformers`` API) and
does not redistribute the weights; the download is gated behind a one-time
license notice, mirroring the YOLO-NAS / L2CS precedents.
"""

from __future__ import annotations

from typing import ClassVar, Dict

from .base import LibreVLMModel

_LFM_LICENSE_URL = "https://www.liquid.ai/lfm-license"

_PROMPT_1000 = (
    "Detect all instances of: {labels}. "
    "Response must be a JSON array: "
    '[{{"label": ..., "bbox": [x1, y1, x2, y2]}}, ...]. '
    "Coordinates are on a 0-1000 scale relative to the image. "
    "Only include objects that are actually visible; if there are none, "
    "respond with an empty array []."
)


class LibreLFM2VL(LibreVLMModel):
    """Liquid AI LFM2-VL repurposed as a closed-set object detector."""

    FAMILY = "lfm2vl"
    FILENAME_PREFIX = "LibreLFM2VL"

    # LFM2.5-VL family (latest). 450m = smallest, 3b = largest.
    HF_REPOS: ClassVar[Dict[str, str]] = {
        "450m": "LiquidAI/LFM2.5-VL-450M",
        "1.6b": "LiquidAI/LFM2.5-VL-1.6B",
        "3b": "LiquidAI/LFM2.5-VL-3B",
    }
    # Nominal input size: the LFM2-VL processor owns the real (native-resolution)
    # resize, so this value is only used as the runner's default ``imgsz``.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "450m": 512,
        "1.6b": 512,
        "3b": 512,
    }

    # The 450M and 1.6B emit ``bbox`` normalized to [0, 1]; the 3B was trained
    # on 0-1000 grounding boxes and emits that scale no matter which convention
    # the prompt asks for (verified empirically on a known box: px
    # [200,150,400,300] of 800x600 comes back as ~[250,250,500,500]).
    # COORD_DIVISOR is a ClassVar, so the 3B shadows it per instance.
    _COORD_DIVISORS: ClassVar[Dict[str, float]] = {
        "450m": 1.0,
        "1.6b": 1.0,
        "3b": 1000.0,
    }

    _LICENSE_NOTICE = (
        "\n"
        "----------------------------------------------------------------\n"
        "LFM2-VL weights (Liquid AI) are distributed under the LFM Open\n"
        "License v1.0: permissive for research and for organizations below\n"
        "a revenue threshold, but NOT an OSI/MIT/Apache-2.0 license. By\n"
        "downloading them you accept those terms. Full license:\n"
        f"  {_LFM_LICENSE_URL}\n"
        "----------------------------------------------------------------\n"
    )

    def __init__(self, size: str, **kwargs):
        divisor = self._COORD_DIVISORS.get(size, 1.0)
        if divisor != 1.0:
            self.COORD_DIVISOR = divisor
        super().__init__(size, **kwargs)

    def _format_detection_prompt(self, labels: str) -> str:
        if self.COORD_DIVISOR == 1000.0:
            return _PROMPT_1000.format(labels=labels)
        return super()._format_detection_prompt(labels)

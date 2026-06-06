"""LibreYOLO wrapper for Liquid AI's LFM2-VL vision-language models.

LFM2.5-VL is a compact (450M) on-device VLM with a native object-detection
prompt that returns ``[{"label", "bbox": [x1, y1, x2, y2]}]`` with coordinates
normalized to ``[0, 1]``. This family wraps it so it behaves like any LibreYOLO
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


class LibreLFM2VL(LibreVLMModel):
    """Liquid AI LFM2-VL repurposed as a closed-set object detector."""

    FAMILY = "lfm2vl"
    FILENAME_PREFIX = "LibreLFM2VL"

    # LFM2.5-VL family (latest). 450m = smallest, 1.6b = larger variant.
    HF_REPOS: ClassVar[Dict[str, str]] = {
        "450m": "LiquidAI/LFM2.5-VL-450M",
        "1.6b": "LiquidAI/LFM2.5-VL-1.6B",
    }
    # Nominal input size: the LFM2-VL processor owns the real (native-resolution)
    # resize, so this value is only used as the runner's default ``imgsz``.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "450m": 512,
        "1.6b": 512,
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

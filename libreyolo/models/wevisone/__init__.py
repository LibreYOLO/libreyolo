"""WeVisOne-VL: one Qwen3-VL checkpoint for detection, referring, pointing,
OCR, segmentation and alpha matting.

The user-facing entry point is the ``LibreVLM`` factory
(``LibreVLM("wevisone-vl", task=...)``) or the ``LibreWeVisOneVL`` class
directly. See ``NOTICE`` in this directory for the ported-code provenance.
"""

from .model import LibreWeVisOneVL

__all__ = ["LibreWeVisOneVL"]

"""Falcon OCR: a 0.27B early-fusion document OCR VLM serving the ``ocr`` task.

Entry points: ``LibreVLM("falcon-ocr")`` or ``LibreFalconOCR`` directly.
See ``NOTICE`` in this directory for the ported-code provenance.
"""

from .model import LibreFalconOCR

__all__ = ["LibreFalconOCR"]

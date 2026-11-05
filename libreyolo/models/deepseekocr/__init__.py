"""DeepSeek-OCR: a 3B MoE document OCR VLM serving the ``ocr`` task.

Entry points: ``LibreVLM("deepseek-ocr")`` or ``LibreDeepSeekOCR`` directly.
See ``NOTICE`` in this directory for the ported-code provenance.
"""

from .model import LibreDeepSeekOCR

__all__ = ["LibreDeepSeekOCR"]

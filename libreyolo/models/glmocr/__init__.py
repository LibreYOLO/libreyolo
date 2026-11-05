"""GLM-OCR: PP-DocLayoutV3 layout regions read by a 0.9B vision-language
recognizer (text, tables, formulas) into ``Results.ocr`` and page markdown.

The user-facing entry point is the ``LibreVLM`` factory
(``LibreVLM("glm-ocr")``) or the ``LibreGLMOCR`` class directly.
See ``NOTICE`` in this directory for the ported-code provenance.
"""

from .model import LibreGLMOCR

__all__ = ["LibreGLMOCR"]

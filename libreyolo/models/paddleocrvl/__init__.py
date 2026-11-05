"""PaddleOCR-VL: PP-DocLayoutV2 layout regions read by a 0.9B vision-language
recognizer (text, tables, formulas, charts) into ``Results.ocr`` and page
markdown.

The user-facing entry point is the ``LibreVLM`` factory
(``LibreVLM("paddleocr-vl")``) or the ``LibrePaddleOCRVL`` class directly.
See ``NOTICE`` in this directory for the ported-code provenance.
"""

from .model import LibrePaddleOCRVL

__all__ = ["LibrePaddleOCRVL"]

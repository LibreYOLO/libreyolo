"""Image-classification validator for LibreViT.

The AugReg eval pipeline comes from the model (``LibreViT`` declares its
normalization, crop and interpolation), so this adds nothing to
:class:`ClassifyValidator`. It is kept because ``libreyolo.validation``
exports it and ``LibreViT.validator_class`` names it.
"""

from __future__ import annotations

from .classify_validator import ClassifyValidator


class ViTClassifyValidator(ClassifyValidator):
    """Top-1/top-5 validator for LibreViT."""

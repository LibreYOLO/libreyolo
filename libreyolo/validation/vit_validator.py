"""Image-classification validator for LibreViT.

The AugReg eval transform comes from the model itself (``eval_transform``,
#886), so this adds nothing to :class:`ClassifyValidator`; it is kept because
``libreyolo.validation`` exports it.
"""

from __future__ import annotations

from .classify_validator import ClassifyValidator


class ViTClassifyValidator(ClassifyValidator):
    """Top-1/top-5 validator for LibreViT."""

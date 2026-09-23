"""Normalization constants for LibreViT AugReg image classifiers.

The eval transform itself is the shared classification one
(:class:`~libreyolo.models.base.classify_preprocess.ClassifyPreprocessMixin`).
"""

VIT_MEAN = (0.5, 0.5, 0.5)
VIT_STD = (0.5, 0.5, 0.5)

__all__ = ["VIT_MEAN", "VIT_STD"]

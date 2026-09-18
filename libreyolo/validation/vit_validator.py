"""Image-classification validator with ViT AugReg preprocessing."""

from __future__ import annotations

from .classify_validator import ClassifyValidator


class ViTClassifyValidator(ClassifyValidator):
    """Top-1/top-5 validator using the published AugReg eval transform."""

    def _dataset_transform_kwargs(self) -> dict:
        return {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "interpolation": "bicubic",
            "crop_pct": self._resolve_crop_pct(0.9),
        }

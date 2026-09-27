"""Family-local photometric transform for the PP-LiteSeg training recipe.

The released recipes jitter brightness, contrast, and saturation at magnitude
0.5 (``SegColorJitter`` in the Apache-2.0 source, which is torchvision's
``ColorJitter`` applied to the image only). LibreYOLO's shared semantic dataset
defaults to HSV-gain jitter instead, so this keeps the source behavior
family-local rather than changing the default for every semantic family.
"""

from __future__ import annotations

import random
from typing import Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance


def _check_magnitude(value: float, name: str) -> Tuple[float, float]:
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return max(0.0, 1.0 - value), 1.0 + value


class SegColorJitter:
    """Random brightness / contrast / saturation jitter on an RGB image.

    Applied in a random order with independent uniform factors, matching
    torchvision ``ColorJitter``. Masks are never passed in: class IDs must not
    be recolored.
    """

    def __init__(
        self, brightness: float = 0.5, contrast: float = 0.5, saturation: float = 0.5
    ) -> None:
        self.brightness = _check_magnitude(brightness, "brightness")
        self.contrast = _check_magnitude(contrast, "contrast")
        self.saturation = _check_magnitude(saturation, "saturation")

    def _ops(self) -> Sequence[Tuple[type, Tuple[float, float]]]:
        return (
            (ImageEnhance.Brightness, self.brightness),
            (ImageEnhance.Contrast, self.contrast),
            (ImageEnhance.Color, self.saturation),
        )

    def __call__(self, img_rgb_hwc: np.ndarray) -> np.ndarray:
        ops = list(self._ops())
        random.shuffle(ops)
        pil = Image.fromarray(img_rgb_hwc)
        for enhancer, (low, high) in ops:
            if low == high == 1.0:
                continue
            pil = enhancer(pil).enhance(random.uniform(low, high))
        return np.asarray(pil, dtype=img_rgb_hwc.dtype)


__all__ = ["SegColorJitter"]

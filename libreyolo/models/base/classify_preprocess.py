"""One eval transform per classification family, shared by predict and val (#886).

A family declares its native eval pipeline once: ``crop_pct`` and
``interpolation`` on the instance, and :attr:`EVAL_MEAN` / :attr:`EVAL_STD` /
:attr:`EVAL_SQUARE_RESIZE` on the class. :meth:`eval_transform` builds it, and
every consumer calls that one method: ``predict()`` through :meth:`_preprocess`,
``val()`` and training validation through the classification dataset, and INT8
calibration through :meth:`_get_preprocess_numpy`. A model is therefore scored
on exactly the preprocessing it is deployed with.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ...data.augment.classify import (
    DEFAULT_CROP_PCT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_classify_transforms,
)
from ...utils.image_loader import ImageInput, ImageLoader


class ClassifyPreprocessMixin:
    """Single-source eval preprocessing for classification families."""

    #: Normalization of the native eval pipeline.
    EVAL_MEAN: Tuple[float, float, float] = IMAGENET_MEAN
    EVAL_STD: Tuple[float, float, float] = IMAGENET_STD
    #: Squash to ``(imgsz, imgsz)`` instead of shorter-side resize + center
    #: crop. An explicit ``crop_pct`` override switches back to the crop.
    EVAL_SQUARE_RESIZE: bool = False

    crop_pct: float = DEFAULT_CROP_PCT
    interpolation: str = "bilinear"

    def _check_eval_imgsz(self, imgsz: int) -> None:
        """Reject an input size the family cannot run; default accepts any."""

    def eval_transform(
        self, imgsz: Optional[int] = None, crop_pct: Optional[float] = None
    ) -> Callable[[Image.Image], torch.Tensor]:
        """The eval transform: an RGB PIL image to a normalized CHW tensor.

        ``imgsz`` defaults to the model's input size. ``crop_pct`` overrides
        the family's crop ratio (``val(crop_pct=...)``); predict never sets it.
        """
        if imgsz is None:
            imgsz = self.input_size
        if isinstance(imgsz, (list, tuple)):
            if len(imgsz) != 2 or int(imgsz[0]) != int(imgsz[1]):
                raise ValueError(
                    f"Classification preprocessing needs a square imgsz, got {imgsz}"
                )
            imgsz = imgsz[0]
        imgsz = int(imgsz)
        self._check_eval_imgsz(imgsz)
        return build_classify_transforms(
            imgsz,
            augment=False,
            mean=self.EVAL_MEAN,
            std=self.EVAL_STD,
            crop_pct=self.crop_pct if crop_pct is None else crop_pct,
            interpolation=self.interpolation,
            square_resize=self.EVAL_SQUARE_RESIZE and crop_pct is None,
        )

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        tensor = self.eval_transform(input_size)(img).unsqueeze(0)
        return tensor, img, img.size, 1.0

    def _get_preprocess_numpy(self):
        """RGB HWC array to ``(CHW float32 array, 1.0)`` for INT8 calibration."""

        def _preprocess_numpy(img_rgb_hwc, input_size=None):
            pil = (
                img_rgb_hwc
                if isinstance(img_rgb_hwc, Image.Image)
                else Image.fromarray(np.asarray(img_rgb_hwc).astype("uint8"))
            )
            return self.eval_transform(input_size)(pil).numpy(), 1.0

        return _preprocess_numpy

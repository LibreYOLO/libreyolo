"""EC pose recipes over the shared pose pipeline.

ECPose uses the same preprocessing contract at train, validation, and
inference time: resize directly to the square model input (non-square images
intentionally stretch), convert BGR to RGB, scale to [0, 1], then apply
ImageNet normalization.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ...data.augment.color import augment_hsv  # noqa: F401  (compat re-export)
from ...data.augment.pose import PoseTrainTransform, PoseValTransform


class ECPoseTrainTransform(PoseTrainTransform):
    """Train-time EC pose transform: augmentation plus direct square resize."""

    def __init__(
        self,
        num_keypoints: int,
        flip_idx: Optional[Sequence[int]] = None,
        max_labels: int = 100,
        flip_prob: float = 0.5,
        hsv_prob: float = 0.5,
        brightness_contrast_prob: float = 0.5,
        affine_prob: float = 0.75,
        degrees: float = 5.0,
        translate: float = 0.1,
        scale: tuple[float, float] = (0.75, 1.5),
        affine_interpolation: str = "linear",
        imagenet_norm: bool = True,
        to_rgb: bool = True,
    ):
        super().__init__(
            num_keypoints=num_keypoints,
            flip_idx=flip_idx,
            max_labels=max_labels,
            flip_prob=flip_prob,
            hsv_prob=hsv_prob,
            brightness_contrast_prob=brightness_contrast_prob,
            affine_prob=affine_prob,
            degrees=degrees,
            translate=translate,
            scale=scale,
            affine_interpolation=affine_interpolation,
            affine_border_value=114,
            final_layout="stretch",
            imagenet_norm=imagenet_norm,
            to_rgb=to_rgb,
        )


class ECPoseValTransform(PoseValTransform):
    """Validation EC pose transform: direct square resize only."""

    def __init__(
        self,
        num_keypoints: int,
        max_labels: int = 100,
        imagenet_norm: bool = True,
        to_rgb: bool = True,
    ):
        super().__init__(
            num_keypoints=num_keypoints,
            max_labels=max_labels,
            final_layout="stretch",
            imagenet_norm=imagenet_norm,
            to_rgb=to_rgb,
        )

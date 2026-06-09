"""YOLO9 pose recipes over the shared pose pipeline.

YOLO9 pose: top-left letterbox (matching the detection preprocessing), RGB
[0, 1] outputs, HSV + hflip + affine, no brightness/contrast jitter.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ...data.augment.pose import PoseTrainTransform, PoseValTransform


class YOLO9PoseTrainTransform(PoseTrainTransform):
    """Train-time YOLO9 pose transform with hflip, HSV, affine, and letterbox."""

    def __init__(
        self,
        num_keypoints: int,
        flip_idx: Optional[Sequence[int]] = None,
        max_labels: int = 100,
        flip_prob: float = 0.5,
        hsv_prob: float = 1.0,
        affine_prob: float = 0.5,
        degrees: float = 5.0,
        translate: float = 0.1,
        scale: tuple[float, float] = (0.75, 1.25),
    ):
        super().__init__(
            num_keypoints=num_keypoints,
            flip_idx=flip_idx,
            max_labels=max_labels,
            flip_prob=flip_prob,
            hsv_prob=hsv_prob,
            brightness_contrast_prob=0.0,
            affine_prob=affine_prob,
            degrees=degrees,
            translate=translate,
            scale=scale,
            affine_interpolation="linear",
            affine_border_value=114,
            final_layout="letterbox_top_left",
            pad_value=114,
            resize_size_cap=None,
            imagenet_norm=False,
            to_rgb=True,
        )


class YOLO9PoseValTransform(PoseValTransform):
    """Validation pose transform: YOLO9-compatible letterbox only."""

    def __init__(self, num_keypoints: int, max_labels: int = 100):
        super().__init__(
            num_keypoints=num_keypoints,
            max_labels=max_labels,
            final_layout="letterbox_top_left",
            pad_value=114,
            resize_size_cap=None,
            imagenet_norm=False,
            to_rgb=True,
        )

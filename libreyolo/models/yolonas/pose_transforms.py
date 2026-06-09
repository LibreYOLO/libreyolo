"""YOLO-NAS pose recipes over the shared pose pipeline.

Follows the public SuperGradients YOLO-NAS pose recipe where practical for
YOLO-format labels: BGR [0, 1] outputs, brightness/contrast + HSV jitter,
keypoint-aware hflip and affine (127 border), 640-capped resize. Training
pads in the center; validation pads bottom/right.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ...data.augment.color import augment_hsv  # noqa: F401  (compat re-export)
from ...data.augment.pose import PoseTrainTransform, PoseValTransform
from .utils import YOLO_NAS_POSE_PAD_VALUE, YOLO_NAS_POSE_RESIZE_SIZE


class YOLONASPoseTrainTransform(PoseTrainTransform):
    """Train-time pose transform: HSV jitter + keypoint-aware hflip + letterbox."""

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
        imagenet_norm: bool = False,
        to_rgb: bool = False,
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
            affine_border_value=YOLO_NAS_POSE_PAD_VALUE,
            final_layout="letterbox_center",
            pad_value=YOLO_NAS_POSE_PAD_VALUE,
            resize_size_cap=YOLO_NAS_POSE_RESIZE_SIZE,
            imagenet_norm=imagenet_norm,
            to_rgb=to_rgb,
        )


class YOLONASPoseValTransform(PoseValTransform):
    """Validation pose transform: letterbox only, no augmentation."""

    def __init__(
        self,
        num_keypoints: int,
        max_labels: int = 100,
        imagenet_norm: bool = False,
        to_rgb: bool = False,
    ):
        super().__init__(
            num_keypoints=num_keypoints,
            max_labels=max_labels,
            final_layout="letterbox_bottom_right",
            pad_value=YOLO_NAS_POSE_PAD_VALUE,
            resize_size_cap=YOLO_NAS_POSE_RESIZE_SIZE,
            imagenet_norm=imagenet_norm,
            to_rgb=to_rgb,
        )

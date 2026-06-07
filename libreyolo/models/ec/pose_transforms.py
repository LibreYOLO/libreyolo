"""EC pose training/validation transforms.

ECPose uses the same preprocessing contract at train, validation, and
inference time: resize directly to the square model input, convert BGR to RGB,
scale to [0, 1], then apply ImageNet normalization. No letterbox padding is
used, so non-square images intentionally stretch to the model input.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import cv2
import numpy as np

from ...training.augment import augment_hsv

_AFFINE_INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

_AFFINE_BORDER_VALUE = 114
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _as_hw(input_dim) -> tuple[int, int]:
    if isinstance(input_dim, int):
        return int(input_dim), int(input_dim)
    if len(input_dim) != 2:
        raise ValueError(f"input_dim must be int or (h, w), got {input_dim!r}")
    return int(input_dim[0]), int(input_dim[1])


def _finalize_image(img: np.ndarray, to_rgb: bool, imagenet_norm: bool) -> np.ndarray:
    """HWC uint8 BGR -> CHW float32, optionally RGB + ImageNet-normalized."""
    if to_rgb:
        img = img[:, :, ::-1]
    img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
    img /= 255.0
    if imagenet_norm:
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(img, dtype=np.float32)


def _brightness_contrast(img: np.ndarray) -> None:
    """In-place brightness/contrast jitter for uint8 BGR images."""
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-0.2, 0.2) * 255.0
    img[:] = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _random_affine(
    img: np.ndarray,
    bboxes: np.ndarray,
    kpts: np.ndarray,
    *,
    degrees: float,
    translate: float,
    scale_range: tuple[float, float],
    interpolation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a keypoint-aware affine transform in original image space."""
    h, w = img.shape[:2]
    angle = random.uniform(-degrees, degrees)
    scale = random.uniform(*scale_range)
    tx = random.uniform(-translate, translate) * w
    ty = random.uniform(-translate, translate) * h

    matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
    matrix[:, 2] += (tx, ty)
    warped = cv2.warpAffine(
        img,
        matrix,
        dsize=(w, h),
        flags=interpolation,
        borderValue=(_AFFINE_BORDER_VALUE,) * 3,
    )

    if len(bboxes) == 0:
        return warped, bboxes, kpts

    xyxy = np.concatenate(
        [
            bboxes[:, :2] - bboxes[:, 2:] * 0.5,
            bboxes[:, :2] + bboxes[:, 2:] * 0.5,
        ],
        axis=1,
    )
    corners = np.stack(
        [
            xyxy[:, [0, 1]],
            xyxy[:, [2, 1]],
            xyxy[:, [2, 3]],
            xyxy[:, [0, 3]],
        ],
        axis=1,
    )
    ones = np.ones((*corners.shape[:2], 1), dtype=np.float32)
    warped_corners = np.concatenate([corners, ones], axis=2) @ matrix.T
    new_xyxy = np.concatenate(
        [warped_corners.min(axis=1), warped_corners.max(axis=1)], axis=1
    )
    new_xyxy[:, [0, 2]] = new_xyxy[:, [0, 2]].clip(0, w)
    new_xyxy[:, [1, 3]] = new_xyxy[:, [1, 3]].clip(0, h)
    bboxes[:, :2] = (new_xyxy[:, :2] + new_xyxy[:, 2:]) * 0.5
    bboxes[:, 2:] = new_xyxy[:, 2:] - new_xyxy[:, :2]

    points = kpts[..., :2]
    warped_points = (
        np.concatenate([points, np.ones((*points.shape[:2], 1), dtype=np.float32)], axis=2)
        @ matrix.T
    )
    kpts[..., :2] = warped_points
    outside = (
        (kpts[..., 0] < 0)
        | (kpts[..., 0] >= w)
        | (kpts[..., 1] < 0)
        | (kpts[..., 1] >= h)
    )
    kpts[..., 0] = kpts[..., 0].clip(0, w)
    kpts[..., 1] = kpts[..., 1].clip(0, h)
    kpts[..., 2] = np.where(outside, 0.0, kpts[..., 2])
    return warped, bboxes, kpts


def _scale_targets_direct(
    bboxes: np.ndarray,
    kpts: np.ndarray,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> None:
    if len(bboxes) == 0:
        return
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scale_x = dst_w / float(src_w)
    scale_y = dst_h / float(src_h)
    bboxes[:, [0, 2]] *= scale_x
    bboxes[:, [1, 3]] *= scale_y
    kpts[..., 0] *= scale_x
    kpts[..., 1] *= scale_y


def _build_target(
    cls: np.ndarray,
    bboxes_px: np.ndarray,
    kpts_px: np.ndarray,
    num_keypoints: int,
    max_labels: int,
) -> np.ndarray:
    target = np.zeros((max_labels, 5 + 3 * num_keypoints), dtype=np.float32)
    if len(bboxes_px) == 0:
        return target

    keep = (
        (bboxes_px[:, 2] * bboxes_px[:, 3] > 1.0)
        & ((kpts_px[..., 2] > 0).sum(axis=1) >= 1)
    )
    bboxes_px, cls, kpts_px = bboxes_px[keep], cls[keep], kpts_px[keep]
    n = min(len(bboxes_px), max_labels)
    if n == 0:
        return target

    target[:n, 0] = cls[:n]
    target[:n, 1:5] = bboxes_px[:n]
    target[:n, 5:] = kpts_px[:n].reshape(len(kpts_px), -1)[:n]
    return target


class ECPoseTrainTransform:
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
        self.num_keypoints = num_keypoints
        self.max_labels = max_labels
        self.imagenet_norm = imagenet_norm
        self.to_rgb = to_rgb
        self.hsv_prob = hsv_prob
        self.brightness_contrast_prob = brightness_contrast_prob
        self.affine_prob = affine_prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.affine_interpolation = _AFFINE_INTERPOLATIONS.get(
            affine_interpolation, cv2.INTER_LINEAR
        )
        if flip_idx is not None and len(flip_idx) == num_keypoints:
            self.flip_idx = np.asarray(flip_idx, dtype=np.int64)
            self.flip_prob = flip_prob
        else:
            self.flip_idx = None
            self.flip_prob = 0.0

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        h, w = img.shape[:2]
        dst_hw = _as_hw(input_dim)

        bboxes = bboxes_norm.astype(np.float32).reshape(-1, 4)
        bboxes[:, [0, 2]] *= w
        bboxes[:, [1, 3]] *= h
        kpts = kpts_norm.astype(np.float32).reshape(-1, self.num_keypoints, 3)
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        cls = cls.astype(np.float32).reshape(-1)

        if self.hsv_prob > 0 and random.random() < self.hsv_prob:
            augment_hsv(img)
        if (
            self.brightness_contrast_prob > 0
            and random.random() < self.brightness_contrast_prob
        ):
            _brightness_contrast(img)

        if self.flip_idx is not None and random.random() < self.flip_prob:
            img = img[:, ::-1]
            if len(bboxes):
                bboxes[:, 0] = w - bboxes[:, 0]
                kpts[..., 0] = w - kpts[..., 0]
                kpts = kpts[:, self.flip_idx, :]

        if self.affine_prob > 0 and random.random() < self.affine_prob:
            img, bboxes, kpts = _random_affine(
                img,
                bboxes,
                kpts,
                degrees=self.degrees,
                translate=self.translate,
                scale_range=self.scale,
                interpolation=self.affine_interpolation,
            )

        img = cv2.resize(
            np.ascontiguousarray(img),
            (dst_hw[1], dst_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        _scale_targets_direct(bboxes, kpts, src_hw=(h, w), dst_hw=dst_hw)

        target = _build_target(cls, bboxes, kpts, self.num_keypoints, self.max_labels)
        img = _finalize_image(np.ascontiguousarray(img), self.to_rgb, self.imagenet_norm)
        return img, target


class ECPoseValTransform:
    """Validation EC pose transform: direct square resize only."""

    def __init__(
        self,
        num_keypoints: int,
        max_labels: int = 100,
        imagenet_norm: bool = True,
        to_rgb: bool = True,
    ):
        self.num_keypoints = num_keypoints
        self.max_labels = max_labels
        self.imagenet_norm = imagenet_norm
        self.to_rgb = to_rgb

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        h, w = img.shape[:2]
        dst_hw = _as_hw(input_dim)

        bboxes = bboxes_norm.astype(np.float32).reshape(-1, 4)
        bboxes[:, [0, 2]] *= w
        bboxes[:, [1, 3]] *= h
        kpts = kpts_norm.astype(np.float32).reshape(-1, self.num_keypoints, 3)
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        cls = cls.astype(np.float32).reshape(-1)

        img = cv2.resize(
            np.ascontiguousarray(img),
            (dst_hw[1], dst_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        _scale_targets_direct(bboxes, kpts, src_hw=(h, w), dst_hw=dst_hw)

        target = _build_target(cls, bboxes, kpts, self.num_keypoints, self.max_labels)
        img = _finalize_image(np.ascontiguousarray(img), self.to_rgb, self.imagenet_norm)
        return img, target

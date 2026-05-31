"""YOLO-NAS pose training/validation transforms.

Keypoint-aware preprocessing for the YOLO-format pose pipeline. Both transforms
take a raw BGR image plus normalized labels and return:

- ``image``: ``(3, H, W)`` float32 BGR in ``[0, 1]``
- ``target``: ``(max_labels, 5 + 3K)`` float32 — rows are
  ``[cls, cx, cy, w, h, kx1, ky1, v1, ...]`` in letterboxed pixel coordinates.

Augmentation follows the public SuperGradients YOLO-NAS pose recipe where it
is practical for YOLO-format labels: keypoint-aware hflip, brightness/contrast,
HSV jitter, random affine, resize, and padding. Training pads in the center;
validation pads bottom/right.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import cv2
import numpy as np

from ...training.augment import augment_hsv
from .utils import YOLO_NAS_POSE_PAD_VALUE, YOLO_NAS_POSE_RESIZE_SIZE

_AFFINE_INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def _letterbox(
    img: np.ndarray,
    input_dim,
    *,
    padding_mode: str = "center",
) -> tuple[np.ndarray, float, int, int]:
    """Resize-and-center-pad into ``input_dim``; return image, ratio, x/y pad."""
    ih, iw = input_dim
    h, w = img.shape[:2]
    resize_size = min(YOLO_NAS_POSE_RESIZE_SIZE, ih, iw)
    r = min(resize_size / h, resize_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((ih, iw, 3), YOLO_NAS_POSE_PAD_VALUE, dtype=np.uint8)
    if padding_mode == "bottom_right":
        pad_x = 0
        pad_y = 0
    elif padding_mode == "center":
        pad_x = (iw - nw) // 2
        pad_y = (ih - nh) // 2
    else:
        raise ValueError(f"Unsupported padding_mode={padding_mode!r}")
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, r, pad_x, pad_y


def _brightness_contrast(img: np.ndarray) -> None:
    """In-place SG-style brightness/contrast jitter for uint8 BGR images."""
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
    """Apply a lightweight keypoint-aware affine transform in image space."""
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
        borderValue=(YOLO_NAS_POSE_PAD_VALUE,) * 3,
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


def _apply_letterbox_to_targets(
    bboxes: np.ndarray, kpts: np.ndarray, ratio: float, pad_x: int, pad_y: int
):
    """Transform cxcywh boxes and xy keypoints into letterboxed pixel space."""
    if len(bboxes) == 0:
        return
    bboxes *= ratio
    bboxes[:, 0] += pad_x
    bboxes[:, 1] += pad_y
    kpts[..., :2] *= ratio
    kpts[..., 0] += pad_x
    kpts[..., 1] += pad_y


def _build_target(
    cls: np.ndarray,
    bboxes_px: np.ndarray,
    kpts_px: np.ndarray,
    num_keypoints: int,
    max_labels: int,
) -> np.ndarray:
    """Assemble the padded ``(max_labels, 5 + 3K)`` target slab.

    Valid rows are written contiguously from the front — the pose loss relies
    on this front-packing to slice each image's objects.
    """
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


class YOLONASPoseTrainTransform:
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
    ):
        self.num_keypoints = num_keypoints
        self.max_labels = max_labels
        self.hsv_prob = hsv_prob
        self.brightness_contrast_prob = brightness_contrast_prob
        self.affine_prob = affine_prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.affine_interpolation = _AFFINE_INTERPOLATIONS.get(
            affine_interpolation, cv2.INTER_LINEAR
        )
        # A horizontal flip needs the left/right keypoint permutation; without
        # a valid flip_idx, flipping would corrupt keypoint identities.
        if flip_idx is not None and len(flip_idx) == num_keypoints:
            self.flip_idx = np.asarray(flip_idx, dtype=np.int64)
            self.flip_prob = flip_prob
        else:
            self.flip_idx = None
            self.flip_prob = 0.0

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        h, w = img.shape[:2]

        # Normalized -> original-image pixels.
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

        img, r, pad_x, pad_y = _letterbox(
            np.ascontiguousarray(img), input_dim, padding_mode="center"
        )
        _apply_letterbox_to_targets(bboxes, kpts, r, pad_x, pad_y)

        target = _build_target(
            cls, bboxes, kpts, self.num_keypoints, self.max_labels
        )
        img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
        img /= 255.0
        return img, target


class YOLONASPoseValTransform:
    """Validation pose transform: letterbox only, no augmentation."""

    def __init__(self, num_keypoints: int, max_labels: int = 100):
        self.num_keypoints = num_keypoints
        self.max_labels = max_labels

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        h, w = img.shape[:2]
        bboxes = bboxes_norm.astype(np.float32).reshape(-1, 4)
        bboxes[:, [0, 2]] *= w
        bboxes[:, [1, 3]] *= h
        kpts = kpts_norm.astype(np.float32).reshape(-1, self.num_keypoints, 3)
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        cls = cls.astype(np.float32).reshape(-1)

        img, r, pad_x, pad_y = _letterbox(
            np.ascontiguousarray(img), input_dim, padding_mode="bottom_right"
        )
        _apply_letterbox_to_targets(bboxes, kpts, r, pad_x, pad_y)

        target = _build_target(
            cls, bboxes, kpts, self.num_keypoints, self.max_labels
        )
        img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
        img /= 255.0
        return img, target

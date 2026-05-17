"""YOLO-NAS pose training/validation transforms.

Keypoint-aware preprocessing for the YOLO-format pose pipeline. Both transforms
take a raw BGR image plus normalized labels and return:

- ``image``: ``(3, H, W)`` float32 RGB in ``[0, 1]``
- ``target``: ``(max_labels, 5 + 3K)`` float32 — rows are
  ``[cls, cx, cy, w, h, kx1, ky1, v1, ...]`` in letterboxed pixel coordinates.

Augmentation is intentionally minimal: HSV jitter and a keypoint-aware
horizontal flip (using the dataset ``flip_idx`` permutation). Letterboxing
matches the YOLO-NAS inference path — resize by a single ratio, center-pad
with value 114.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import cv2
import numpy as np

from ...training.augment import augment_hsv
from .utils import YOLO_NAS_RESIZE_SIZE


def _letterbox(img: np.ndarray, input_dim) -> tuple[np.ndarray, float, int, int]:
    """Resize-and-center-pad into ``input_dim``; return image, ratio, x/y pad."""
    ih, iw = input_dim
    h, w = img.shape[:2]
    resize_size = min(YOLO_NAS_RESIZE_SIZE, ih, iw)
    r = min(resize_size / h, resize_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
    pad_x = (iw - nw) // 2
    pad_y = (ih - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, r, pad_x, pad_y


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

    # Keep boxes with a sane size after transforms.
    keep = np.minimum(bboxes_px[:, 2], bboxes_px[:, 3]) > 1.0
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
    ):
        self.num_keypoints = num_keypoints
        self.max_labels = max_labels
        self.hsv_prob = hsv_prob
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

        if self.flip_idx is not None and random.random() < self.flip_prob:
            img = img[:, ::-1]
            if len(bboxes):
                bboxes[:, 0] = w - bboxes[:, 0]
                kpts[..., 0] = w - kpts[..., 0]
                kpts = kpts[:, self.flip_idx, :]

        img, r, pad_x, pad_y = _letterbox(np.ascontiguousarray(img), input_dim)
        _apply_letterbox_to_targets(bboxes, kpts, r, pad_x, pad_y)

        target = _build_target(
            cls, bboxes, kpts, self.num_keypoints, self.max_labels
        )
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
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

        img, r, pad_x, pad_y = _letterbox(np.ascontiguousarray(img), input_dim)
        _apply_letterbox_to_targets(bboxes, kpts, r, pad_x, pad_y)

        target = _build_target(
            cls, bboxes, kpts, self.num_keypoints, self.max_labels
        )
        img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
        img /= 255.0
        return img, target

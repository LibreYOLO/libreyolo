"""YOLO9 pose training transforms.

Clean-room implementation guided by the in-repo YOLO-NAS pose transform
contract and the existing YOLO9 detection preprocessing convention.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import cv2
import numpy as np

from ...training.augment import augment_hsv


def _letterbox_rgb_top_left(img: np.ndarray, input_dim) -> tuple[np.ndarray, float]:
    ih, iw = input_dim
    h, w = img.shape[:2]
    ratio = min(ih / h, iw / w)
    nh, nw = max(int(round(h * ratio)), 1), max(int(round(w * ratio)), 1)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    return canvas[:, :, ::-1], ratio


def _random_affine(
    img: np.ndarray,
    bboxes: np.ndarray,
    kpts: np.ndarray,
    *,
    degrees: float,
    translate: float,
    scale_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        flags=cv2.INTER_LINEAR,
        borderValue=(114, 114, 114),
    )
    if len(bboxes) == 0:
        return warped, bboxes, kpts

    xyxy = np.concatenate(
        [bboxes[:, :2] - bboxes[:, 2:] * 0.5, bboxes[:, :2] + bboxes[:, 2:] * 0.5],
        axis=1,
    )
    corners = np.stack(
        [xyxy[:, [0, 1]], xyxy[:, [2, 1]], xyxy[:, [2, 3]], xyxy[:, [0, 3]]],
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


def _build_target(cls, bboxes_px, kpts_px, num_keypoints, max_labels):
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
    target[:n, 5:] = kpts_px[:n].reshape(n, -1)
    return target


class YOLO9PoseTrainTransform:
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
        self.num_keypoints = int(num_keypoints)
        self.max_labels = int(max_labels)
        self.flip_prob = float(flip_prob)
        self.hsv_prob = float(hsv_prob)
        self.affine_prob = float(affine_prob)
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = scale
        self.flip_idx = (
            np.asarray(flip_idx, dtype=np.int64)
            if flip_idx is not None and len(flip_idx) == num_keypoints
            else None
        )
        if self.flip_idx is None:
            self.flip_prob = 0.0

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        h, w = img.shape[:2]
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

        if self.affine_prob > 0 and random.random() < self.affine_prob:
            img, bboxes, kpts = _random_affine(
                img,
                bboxes,
                kpts,
                degrees=self.degrees,
                translate=self.translate,
                scale_range=self.scale,
            )

        img, ratio = _letterbox_rgb_top_left(np.ascontiguousarray(img), input_dim)
        bboxes *= ratio
        kpts[..., :2] *= ratio
        target = _build_target(cls, bboxes, kpts, self.num_keypoints, self.max_labels)
        img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return img, target


class YOLO9PoseValTransform:
    """Validation pose transform: YOLO9-compatible letterbox only."""

    def __init__(self, num_keypoints: int, max_labels: int = 100):
        self.num_keypoints = int(num_keypoints)
        self.max_labels = int(max_labels)

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        h, w = img.shape[:2]
        bboxes = bboxes_norm.astype(np.float32).reshape(-1, 4)
        bboxes[:, [0, 2]] *= w
        bboxes[:, [1, 3]] *= h
        kpts = kpts_norm.astype(np.float32).reshape(-1, self.num_keypoints, 3)
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        cls = cls.astype(np.float32).reshape(-1)

        img, ratio = _letterbox_rgb_top_left(np.ascontiguousarray(img), input_dim)
        bboxes *= ratio
        kpts[..., :2] *= ratio
        target = _build_target(cls, bboxes, kpts, self.num_keypoints, self.max_labels)
        img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
        return img, target

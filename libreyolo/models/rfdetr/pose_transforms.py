"""RF-DETR pose training transforms.

Clean-room implementation based on LibreYOLO-local RF-DETR detection geometry
and the in-repo YOLO-NAS pose target contract.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import cv2
import numpy as np

from .seg_transforms import compute_multi_scale_scales

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _resolve_training_size(
    imgsz: int,
    *,
    multi_scale: bool,
    expanded_scales: bool,
    do_random_resize_via_padding: bool,
    patch_size: int,
    num_windows: int,
) -> int:
    if not multi_scale:
        return imgsz
    scales = compute_multi_scale_scales(imgsz, expanded_scales, patch_size, num_windows)
    if not scales or do_random_resize_via_padding:
        return imgsz
    return scales[-1]


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.zeros_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    out[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    out[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    out[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    return out


def _xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    out = np.zeros_like(boxes, dtype=np.float32)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def _resize_shortest_side(img: np.ndarray, size: int) -> tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    scale = size / max(1, min(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale, scale


def _resize_square(img: np.ndarray, input_dim) -> tuple[np.ndarray, float, float]:
    target_h, target_w = input_dim
    scale_x = target_w / img.shape[1]
    scale_y = target_h / img.shape[0]
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized[:, :, ::-1], scale_x, scale_y


def _crop_pose(
    img: np.ndarray,
    boxes_xyxy: np.ndarray,
    kpts: np.ndarray,
    *,
    left: int,
    top: int,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = img[top : top + size, left : left + size]
    if len(boxes_xyxy) == 0:
        return img, boxes_xyxy, kpts

    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]] - left, 0.0, float(size))
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]] - top, 0.0, float(size))

    kpts[..., 0] -= left
    kpts[..., 1] -= top
    outside = (
        (kpts[..., 0] < 0.0)
        | (kpts[..., 0] >= float(size))
        | (kpts[..., 1] < 0.0)
        | (kpts[..., 1] >= float(size))
    )
    kpts[..., 0] = np.clip(kpts[..., 0], 0.0, float(size))
    kpts[..., 1] = np.clip(kpts[..., 1], 0.0, float(size))
    kpts[..., 2] = np.where(outside, 0.0, kpts[..., 2])
    return img, boxes_xyxy, kpts


def _build_target(
    cls: np.ndarray,
    boxes_cxcywh: np.ndarray,
    kpts: np.ndarray,
    num_keypoints: int,
    max_labels: int,
) -> np.ndarray:
    target = np.zeros((max_labels, 5 + 3 * num_keypoints), dtype=np.float32)
    if len(boxes_cxcywh) == 0:
        return target

    keep = (
        (boxes_cxcywh[:, 2] > 1.0)
        & (boxes_cxcywh[:, 3] > 1.0)
        & ((kpts[..., 2] > 0).sum(axis=1) >= 1)
    )
    boxes_cxcywh = boxes_cxcywh[keep]
    cls = cls[keep]
    kpts = kpts[keep]
    n = min(len(boxes_cxcywh), max_labels)
    if n == 0:
        return target

    target[:n, 0] = cls[:n]
    target[:n, 1:5] = boxes_cxcywh[:n]
    target[:n, 5:] = kpts[:n].reshape(n, -1)
    return target


class RFDETRPoseTransform:
    """Pose transform for YOLO-format pose labels and RF-DETR square inputs."""

    def __init__(
        self,
        num_keypoints: int,
        *,
        flip_idx: Optional[Sequence[int]] = None,
        max_labels: int = 100,
        flip_prob: float = 0.5,
        imgsz: int = 384,
        multi_scale: bool = False,
        expanded_scales: bool = False,
        do_random_resize_via_padding: bool = False,
        patch_size: int = 16,
        num_windows: int = 4,
        crop_resize_prob: float = 0.0,
        crop_intermediate_sizes: tuple[int, ...] = (400, 500, 600),
        crop_min_size: int = 384,
        crop_max_size: int = 600,
    ):
        self.num_keypoints = int(num_keypoints)
        self.max_labels = int(max_labels)
        self.flip_prob = float(flip_prob)
        self.imgsz = int(imgsz)
        self.crop_resize_prob = float(crop_resize_prob)
        self.crop_intermediate_sizes = crop_intermediate_sizes
        self.crop_min_size = int(crop_min_size)
        self.crop_max_size = int(crop_max_size)
        self.target_size = _resolve_training_size(
            self.imgsz,
            multi_scale=multi_scale,
            expanded_scales=expanded_scales,
            do_random_resize_via_padding=do_random_resize_via_padding,
            patch_size=patch_size,
            num_windows=num_windows,
        )
        self.flip_idx = (
            np.asarray(flip_idx, dtype=np.int64)
            if flip_idx is not None and len(flip_idx) == self.num_keypoints
            else None
        )
        if self.flip_idx is None:
            self.flip_prob = 0.0

    def disable_strong_augs(self):
        self.crop_resize_prob = 0.0

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        del input_dim
        h, w = img.shape[:2]
        boxes_cxcywh = bboxes_norm.astype(np.float32).reshape(-1, 4)
        boxes_cxcywh[:, [0, 2]] *= w
        boxes_cxcywh[:, [1, 3]] *= h
        boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh)
        kpts = kpts_norm.astype(np.float32).reshape(-1, self.num_keypoints, 3)
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        cls = cls.astype(np.float32).reshape(-1)

        if self.flip_idx is not None and random.random() < self.flip_prob:
            img = img[:, ::-1].copy()
            if len(boxes_xyxy):
                boxes_xyxy[:, [0, 2]] = w - boxes_xyxy[:, [2, 0]]
                kpts[..., 0] = w - kpts[..., 0]
                kpts = kpts[:, self.flip_idx, :]

        if len(boxes_xyxy) and self.crop_resize_prob > 0 and random.random() < self.crop_resize_prob:
            img, scale_x, scale_y = _resize_shortest_side(
                img,
                random.choice(self.crop_intermediate_sizes),
            )
            boxes_xyxy[:, [0, 2]] *= scale_x
            boxes_xyxy[:, [1, 3]] *= scale_y
            kpts[..., 0] *= scale_x
            kpts[..., 1] *= scale_y

            h_mid, w_mid = img.shape[:2]
            max_crop = min(self.crop_max_size, h_mid, w_mid)
            min_crop = min(self.crop_min_size, max_crop)
            if max_crop >= 2:
                crop_size = random.randint(min_crop, max_crop)
                top = random.randint(0, max(0, h_mid - crop_size))
                left = random.randint(0, max(0, w_mid - crop_size))
                img, boxes_xyxy, kpts = _crop_pose(
                    img,
                    boxes_xyxy,
                    kpts,
                    left=left,
                    top=top,
                    size=crop_size,
                )

        target_h = target_w = self.target_size
        img_rgb, scale_x, scale_y = _resize_square(img, (target_h, target_w))
        if len(boxes_xyxy):
            boxes_xyxy[:, [0, 2]] *= scale_x
            boxes_xyxy[:, [1, 3]] *= scale_y
            boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0.0, float(target_w))
            boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0.0, float(target_h))
            kpts[..., 0] *= scale_x
            kpts[..., 1] *= scale_y
            outside = (
                (kpts[..., 0] < 0.0)
                | (kpts[..., 0] > float(target_w))
                | (kpts[..., 1] < 0.0)
                | (kpts[..., 1] > float(target_h))
            )
            kpts[..., 0] = np.clip(kpts[..., 0], 0.0, float(target_w))
            kpts[..., 1] = np.clip(kpts[..., 1], 0.0, float(target_h))
            kpts[..., 2] = np.where(outside, 0.0, kpts[..., 2])

        boxes_cxcywh = _xyxy_to_cxcywh(boxes_xyxy)
        target = _build_target(cls, boxes_cxcywh, kpts, self.num_keypoints, self.max_labels)

        img_out = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_out = (img_out - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.ascontiguousarray(img_out), target


__all__ = ["RFDETRPoseTransform"]

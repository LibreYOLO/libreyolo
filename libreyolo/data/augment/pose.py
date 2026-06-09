"""Keypoint-aware augmentation recipe shared by the pose pipelines.

One implementation of the pose train/val transforms previously copied across
YOLO9, YOLO-NAS, and EC. The model families differ only in parameters:

==============  ========  =========  ==========  ====================
                YOLO9     YOLO-NAS   EC          knob
==============  ========  =========  ==========  ====================
brightness      no        yes        yes         brightness_contrast_prob
affine border   114       127        114         affine_border_value
final geometry  top-left  center/BR  stretch     final_layout
pad value       114       127        114         pad_value
resize cap      none      640        n/a         resize_size_cap
color space     RGB       BGR        RGB         to_rgb
normalization   /255      /255       ImageNet    imagenet_norm
==============  ========  =========  ==========  ====================

Inputs follow the ``YOLOPoseDataset`` contract: normalized cxcywh boxes,
classes, and ``(N, K, 3)`` normalized keypoints with visibility. Output is a
CHW float32 image plus a front-packed ``(max_labels, 5 + 3K)`` target slab in
final-canvas pixel coordinates.

The RF-DETR pose recipe lives in :mod:`libreyolo.data.augment.rfdetr` — its
crop-and-square geometry follows that family's detection path instead.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import cv2
import numpy as np

from .color import augment_hsv, brightness_contrast

AFFINE_INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def as_hw(input_dim) -> tuple[int, int]:
    if isinstance(input_dim, int):
        return int(input_dim), int(input_dim)
    if len(input_dim) != 2:
        raise ValueError(f"input_dim must be int or (h, w), got {input_dim!r}")
    return int(input_dim[0]), int(input_dim[1])


def finalize_image(img: np.ndarray, to_rgb: bool, imagenet_norm: bool) -> np.ndarray:
    """HWC uint8 BGR -> CHW float32 [0, 1], optionally RGB + ImageNet-normalized."""
    if to_rgb:
        img = img[:, :, ::-1]
    img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
    img /= 255.0
    if imagenet_norm:
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(img, dtype=np.float32)


def letterbox_pose(
    img: np.ndarray,
    input_dim,
    *,
    padding_mode: str = "center",
    pad_value: int = 114,
    resize_size_cap: Optional[int] = None,
) -> tuple[np.ndarray, float, int, int]:
    """Resize-and-pad into ``input_dim``; return BGR canvas, ratio, x/y pad."""
    ih, iw = input_dim
    h, w = img.shape[:2]
    if resize_size_cap is None:
        r = min(ih / h, iw / w)
    else:
        resize_size = min(resize_size_cap, ih, iw)
        r = min(resize_size / h, resize_size / w)
    nh, nw = max(int(round(h * r)), 1), max(int(round(w * r)), 1)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((ih, iw, 3), pad_value, dtype=np.uint8)
    if padding_mode in ("top_left", "bottom_right"):
        pad_x = 0
        pad_y = 0
    elif padding_mode == "center":
        pad_x = (iw - nw) // 2
        pad_y = (ih - nh) // 2
    else:
        raise ValueError(f"Unsupported padding_mode={padding_mode!r}")
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, r, pad_x, pad_y


def apply_letterbox_to_pose_targets(
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


def scale_pose_targets_stretch(
    bboxes: np.ndarray,
    kpts: np.ndarray,
    *,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> None:
    """Scale cxcywh boxes and keypoints for a direct (non-letterbox) resize."""
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


def random_affine_pose(
    img: np.ndarray,
    bboxes: np.ndarray,
    kpts: np.ndarray,
    *,
    degrees: float,
    translate: float,
    scale_range: tuple[float, float],
    interpolation: int = cv2.INTER_LINEAR,
    border_value: int = 114,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keypoint-aware affine: rotate/scale/translate about the image center.

    Boxes (cxcywh pixels) are recomputed from warped corners; keypoints are
    warped and any that land outside the canvas get visibility zeroed.
    """
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
        borderValue=(border_value,) * 3,
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


def build_pose_target(
    cls: np.ndarray,
    bboxes_px: np.ndarray,
    kpts_px: np.ndarray,
    num_keypoints: int,
    max_labels: int,
) -> np.ndarray:
    """Assemble the padded ``(max_labels, 5 + 3K)`` target slab.

    Valid rows are written contiguously from the front — the pose loss relies
    on this front-packing to slice each image's objects. Instances are kept
    only if the box has area > 1 px^2 and at least one visible keypoint.
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
    target[:n, 5:] = kpts_px[:n].reshape(n, -1)
    return target


def _denormalize_pose(img, bboxes_norm, cls, kpts_norm, num_keypoints):
    h, w = img.shape[:2]
    bboxes = bboxes_norm.astype(np.float32).reshape(-1, 4)
    bboxes[:, [0, 2]] *= w
    bboxes[:, [1, 3]] *= h
    kpts = kpts_norm.astype(np.float32).reshape(-1, num_keypoints, 3)
    kpts[..., 0] *= w
    kpts[..., 1] *= h
    cls = cls.astype(np.float32).reshape(-1)
    return bboxes, cls, kpts


class PoseTrainTransform:
    """Train-time pose transform: HSV/brightness + hflip + affine + resize.

    ``final_layout`` selects the geometry that maps onto the model canvas:
    ``"letterbox_top_left"`` (YOLO9), ``"letterbox_center"`` (YOLO-NAS), or
    ``"stretch"`` (EC; non-square images intentionally stretch).
    """

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
        affine_border_value: int = 114,
        final_layout: str = "letterbox_center",
        pad_value: int = 114,
        resize_size_cap: Optional[int] = None,
        imagenet_norm: bool = False,
        to_rgb: bool = False,
    ):
        self.num_keypoints = int(num_keypoints)
        self.max_labels = int(max_labels)
        self.imagenet_norm = imagenet_norm
        self.to_rgb = to_rgb
        self.hsv_prob = float(hsv_prob)
        self.brightness_contrast_prob = float(brightness_contrast_prob)
        self.affine_prob = float(affine_prob)
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = scale
        self.affine_interpolation = AFFINE_INTERPOLATIONS.get(
            affine_interpolation, cv2.INTER_LINEAR
        )
        self.affine_border_value = int(affine_border_value)
        if final_layout not in ("letterbox_top_left", "letterbox_center", "stretch"):
            raise ValueError(f"Unsupported final_layout={final_layout!r}")
        self.final_layout = final_layout
        self.pad_value = int(pad_value)
        self.resize_size_cap = resize_size_cap
        # A horizontal flip needs the left/right keypoint permutation; without
        # a valid flip_idx, flipping would corrupt keypoint identities.
        if flip_idx is not None and len(flip_idx) == self.num_keypoints:
            self.flip_idx = np.asarray(flip_idx, dtype=np.int64)
            self.flip_prob = float(flip_prob)
        else:
            self.flip_idx = None
            self.flip_prob = 0.0

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        h, w = img.shape[:2]
        dst_hw = as_hw(input_dim)
        bboxes, cls, kpts = _denormalize_pose(
            img, bboxes_norm, cls, kpts_norm, self.num_keypoints
        )

        if self.hsv_prob > 0 and random.random() < self.hsv_prob:
            augment_hsv(img)
        if (
            self.brightness_contrast_prob > 0
            and random.random() < self.brightness_contrast_prob
        ):
            brightness_contrast(img)

        if self.flip_idx is not None and random.random() < self.flip_prob:
            img = img[:, ::-1]
            if len(bboxes):
                bboxes[:, 0] = w - bboxes[:, 0]
                kpts[..., 0] = w - kpts[..., 0]
                kpts = kpts[:, self.flip_idx, :]

        if self.affine_prob > 0 and random.random() < self.affine_prob:
            img, bboxes, kpts = random_affine_pose(
                img,
                bboxes,
                kpts,
                degrees=self.degrees,
                translate=self.translate,
                scale_range=self.scale,
                interpolation=self.affine_interpolation,
                border_value=self.affine_border_value,
            )

        if self.final_layout == "stretch":
            img = cv2.resize(
                np.ascontiguousarray(img),
                (dst_hw[1], dst_hw[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            scale_pose_targets_stretch(bboxes, kpts, src_hw=(h, w), dst_hw=dst_hw)
        else:
            padding_mode = (
                "top_left" if self.final_layout == "letterbox_top_left" else "center"
            )
            img, r, pad_x, pad_y = letterbox_pose(
                np.ascontiguousarray(img),
                dst_hw,
                padding_mode=padding_mode,
                pad_value=self.pad_value,
                resize_size_cap=self.resize_size_cap,
            )
            apply_letterbox_to_pose_targets(bboxes, kpts, r, pad_x, pad_y)

        target = build_pose_target(cls, bboxes, kpts, self.num_keypoints, self.max_labels)
        img = finalize_image(np.ascontiguousarray(img), self.to_rgb, self.imagenet_norm)
        return img, target


class PoseValTransform:
    """Validation pose transform: final-canvas geometry only, no augmentation."""

    def __init__(
        self,
        num_keypoints: int,
        max_labels: int = 100,
        final_layout: str = "letterbox_bottom_right",
        pad_value: int = 114,
        resize_size_cap: Optional[int] = None,
        imagenet_norm: bool = False,
        to_rgb: bool = False,
    ):
        self.num_keypoints = int(num_keypoints)
        self.max_labels = int(max_labels)
        if final_layout not in (
            "letterbox_top_left",
            "letterbox_bottom_right",
            "stretch",
        ):
            raise ValueError(f"Unsupported final_layout={final_layout!r}")
        self.final_layout = final_layout
        self.pad_value = int(pad_value)
        self.resize_size_cap = resize_size_cap
        self.imagenet_norm = imagenet_norm
        self.to_rgb = to_rgb

    def __call__(self, img, bboxes_norm, cls, kpts_norm, input_dim):
        h, w = img.shape[:2]
        dst_hw = as_hw(input_dim)
        bboxes, cls, kpts = _denormalize_pose(
            img, bboxes_norm, cls, kpts_norm, self.num_keypoints
        )

        if self.final_layout == "stretch":
            img = cv2.resize(
                np.ascontiguousarray(img),
                (dst_hw[1], dst_hw[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            scale_pose_targets_stretch(bboxes, kpts, src_hw=(h, w), dst_hw=dst_hw)
        else:
            padding_mode = (
                "top_left"
                if self.final_layout == "letterbox_top_left"
                else "bottom_right"
            )
            img, r, pad_x, pad_y = letterbox_pose(
                np.ascontiguousarray(img),
                dst_hw,
                padding_mode=padding_mode,
                pad_value=self.pad_value,
                resize_size_cap=self.resize_size_cap,
            )
            apply_letterbox_to_pose_targets(bboxes, kpts, r, pad_x, pad_y)

        target = build_pose_target(cls, bboxes, kpts, self.num_keypoints, self.max_labels)
        img = finalize_image(np.ascontiguousarray(img), self.to_rgb, self.imagenet_norm)
        return img, target

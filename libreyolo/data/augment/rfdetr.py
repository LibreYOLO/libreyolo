"""RF-DETR augmentation recipe: square resize + flip + optional crop-resize.

Mirrors the DETR recipe's output contract for detection (cxcywh pixel coords
on the resized canvas, ImageNet-normalized CHW float32 RGB) and additionally
rasterizes per-instance polygon rings to a dense (max_labels, H, W) float32
tensor at the transformed image resolution. Also used by EC segmentation.

Mosaic, mixup, photometric distort, IoU crop, and zoom-out are intentionally
omitted; RF-DETR training uses a direct square canvas with an optional
upstream-style random resize/crop branch.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import cv2
import numpy as np

from ..obb import normalize_obb_angle, scale_xywhr
from .segments import (
    copy_segments,
    crop_segments,
    filter_segments,
    flip_segments_lr,
    materialize_dense_masks_for_crop,
    rasterize_segments,
    scale_segments_xy,
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def compute_multi_scale_scales(
    resolution: int,
    expanded_scales: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
) -> list[int]:
    divisor = patch_size * num_windows
    base_num_patches_per_window = resolution // divisor
    offsets = (
        [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        if expanded_scales
        else [-3, -2, -1, 0, 1, 2, 3, 4]
    )
    return [
        (base_num_patches_per_window + offset) * divisor
        for offset in offsets
        if (base_num_patches_per_window + offset) * divisor >= divisor * 2
    ]


def resolve_training_size(
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
    if not scales:
        return imgsz
    # LibreYOLO stacks per-sample transform outputs directly. Upstream's
    # default also disables per-step random resize and uses the largest expanded
    # square scale, so keep one stable canvas per dataloader.
    if not do_random_resize_via_padding:
        return scales[-1]
    return imgsz


def resize_square(img: np.ndarray, input_dim) -> tuple[np.ndarray, float, float]:
    """BGR HWC -> resized RGB HWC, returning x/y scale factors."""
    target_h, target_w = input_dim
    scale_x = target_w / img.shape[1]
    scale_y = target_h / img.shape[0]
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized = resized[:, :, ::-1]  # BGR -> RGB
    return resized, scale_x, scale_y


def resize_shortest_side(img: np.ndarray, size: int) -> tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    scale = size / max(1, min(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale, scale


class RFDETRSegTransform:
    """Per-sample seg transform: square resize + flip + ImageNet norm + polygon rasterization.

    Output: ``(img_chw_float_rgb_imagenet, padded_labels [max_labels, 5] cxcywh-pixel,
    masks [max_labels, mask_h, mask_w] float32)``.

    The trainer's ``on_forward`` converts cxcywh-pixel -> cxcywh-normalized and slices
    masks to ``[num_valid, H, W]`` per image before passing to the criterion.
    """

    # Surfaced so YOLODataset/COCODataset hand us the original image, not the
    # pre-resized one — we need original pixel coords for polygon scaling.
    wants_unresized_image = True

    def __init__(
        self,
        max_labels: int = 300,
        flip_prob: float = 0.5,
        imgsz: int = 512,
        mask_downsample_ratio: int = 4,
        multi_scale: bool = False,
        expanded_scales: bool = False,
        do_random_resize_via_padding: bool = False,
        patch_size: int = 16,
        num_windows: int = 4,
        crop_resize_prob: float = 0.0,
        crop_intermediate_sizes: tuple[int, ...] = (400, 500, 600),
        crop_min_size: int = 384,
        crop_max_size: int = 600,
        target_dim: int = 5,
    ):
        if target_dim not in (5, 6):
            raise ValueError(f"RF-DETR target_dim must be 5 or 6, got {target_dim}")
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.imgsz = imgsz
        self.mask_downsample_ratio = mask_downsample_ratio
        self.multi_scale = multi_scale
        self.expanded_scales = expanded_scales
        self.do_random_resize_via_padding = do_random_resize_via_padding
        self.patch_size = patch_size
        self.num_windows = num_windows
        self.crop_resize_prob = crop_resize_prob
        self.crop_intermediate_sizes = crop_intermediate_sizes
        self.crop_min_size = crop_min_size
        self.crop_max_size = crop_max_size
        self.target_dim = target_dim
        self.target_size = resolve_training_size(
            imgsz,
            multi_scale=multi_scale,
            expanded_scales=expanded_scales,
            do_random_resize_via_padding=do_random_resize_via_padding,
            patch_size=patch_size,
            num_windows=num_windows,
        )

    def disable_strong_augs(self):
        # Compatibility shim: no strong augs to disable.
        return

    def __call__(self, image: np.ndarray, targets: np.ndarray, input_dim, segments=None):
        del input_dim
        target_h = target_w = self.target_size
        boxes = targets[:, :4].astype(np.float32, copy=True) if len(targets) else np.zeros((0, 4), np.float32)
        labels = targets[:, 4].astype(np.float32, copy=True) if len(targets) else np.zeros((0,), np.float32)
        angles = (
            targets[:, 5].astype(np.float32, copy=True)
            if self.target_dim == 6 and len(targets) and targets.shape[1] > 5
            else np.zeros((len(targets),), dtype=np.float32)
        )
        segments_t = copy_segments(segments)

        # Optional horizontal flip — applied before resize, on the original canvas.
        if random.random() < self.flip_prob:
            _, w_orig, _ = image.shape
            image = image[:, ::-1].copy()
            if len(boxes):
                boxes[:, [0, 2]] = w_orig - boxes[:, [2, 0]]
                if self.target_dim == 6:
                    angles = np.asarray(
                        [normalize_obb_angle(np.pi - float(a)) for a in angles],
                        dtype=np.float32,
                    )
            segments_t = flip_segments_lr(segments_t, w_orig)

        if len(boxes) and self.crop_resize_prob > 0 and random.random() < self.crop_resize_prob:
            image, scale_x, scale_y = resize_shortest_side(
                image,
                random.choice(self.crop_intermediate_sizes),
            )
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            segments_t = scale_segments_xy(segments_t, scale_x, scale_y)

            h_mid, w_mid = image.shape[:2]
            max_crop = min(self.crop_max_size, h_mid, w_mid)
            min_crop = min(self.crop_min_size, max_crop)
            if max_crop >= 2:
                crop_size = random.randint(min_crop, max_crop)
                top = random.randint(0, max(0, h_mid - crop_size))
                left = random.randint(0, max(0, w_mid - crop_size))
                segments_t = materialize_dense_masks_for_crop(segments_t, (h_mid, w_mid))
                image = image[top : top + crop_size, left : left + crop_size]
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] - left, 0.0, float(crop_size))
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] - top, 0.0, float(crop_size))
                segments_t = crop_segments(segments_t, left, top, crop_size, crop_size)

        # RF-DETR's square training/inference path resizes directly to the model
        # canvas, so boxes and masks use independent x/y scale factors.
        img_rgb, scale_x, scale_y = resize_square(image, (target_h, target_w))
        if len(boxes):
            if self.target_dim == 6:
                xywhr = np.stack(
                    (
                        (boxes[:, 0] + boxes[:, 2]) * 0.5,
                        (boxes[:, 1] + boxes[:, 3]) * 0.5,
                        boxes[:, 2] - boxes[:, 0],
                        boxes[:, 3] - boxes[:, 1],
                        angles,
                    ),
                    axis=1,
                )
                xywhr = scale_xywhr(xywhr, scale_x, scale_y)
                boxes[:, 0] = xywhr[:, 0] - xywhr[:, 2] * 0.5
                boxes[:, 1] = xywhr[:, 1] - xywhr[:, 3] * 0.5
                boxes[:, 2] = xywhr[:, 0] + xywhr[:, 2] * 0.5
                boxes[:, 3] = xywhr[:, 1] + xywhr[:, 3] * 0.5
                angles = xywhr[:, 4]
            else:
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
        segments_t = scale_segments_xy(segments_t, scale_x, scale_y)

        # Drop boxes that collapsed below 1px after resize. Apply the same keep mask
        # to segments so per-instance alignment is preserved.
        if len(boxes):
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            keep = (w > 1) & (h > 1)
            if not keep.all():
                boxes = boxes[keep]
                labels = labels[keep]
                angles = angles[keep]
                segments_t = filter_segments(segments_t, keep)

        # xyxy(pixel) -> cxcywh(pixel) on the resized canvas — matches the DETR contract.
        if len(boxes):
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            packed_columns = [labels, cx, cy, w, h]
            if self.target_dim == 6:
                packed_columns.append(angles)
            packed = np.stack(packed_columns, axis=1).astype(np.float32, copy=False)
        else:
            packed = np.zeros((0, self.target_dim), dtype=np.float32)

        padded = np.zeros((self.max_labels, self.target_dim), dtype=np.float32)
        n = min(len(packed), self.max_labels)
        if n:
            padded[:n] = packed[:n]

        # CHW float32 in [0, 1], then ImageNet normalize.
        img_out = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_out = (img_out - _IMAGENET_MEAN) / _IMAGENET_STD
        img_out = np.ascontiguousarray(img_out)

        mask_shape = (target_h, target_w)
        # Dense masks are authoritative here: every geometry op above
        # (flip/scale/crop) maintains them, and for RLE-sourced instances the
        # polygon ring is only a lossy contour approximation.
        masks = rasterize_segments(
            segments_t,
            image_shape=(target_h, target_w),
            mask_shape=mask_shape,
            max_masks=self.max_labels,
            prefer_dense=True,
        )

        return img_out, padded, masks


class RFDETRDetTransform(RFDETRSegTransform):
    """RF-DETR detection transform using the same square geometry."""

    def __call__(self, image: np.ndarray, targets: np.ndarray, input_dim):
        img, labels, _ = super().__call__(image, targets, input_dim, segments=None)
        return img, labels


class RFDETRSegPassThroughDataset:
    """Identity wrapper that runs the seg transform per item — no mosaic.

    Mirrors DETRPassThroughDataset's constructor contract so BaseTrainer's
    ``_setup_data`` can drop us in without special-casing.
    """

    def __init__(
        self,
        dataset,
        img_size,
        mosaic=True,
        preproc=None,
        degrees=0.0,
        translate=0.0,
        mosaic_scale=(1.0, 1.0),
        mixup_scale=(1.0, 1.0),
        shear=0.0,
        enable_mixup=False,
        mosaic_prob=0.0,
        mixup_prob=0.0,
    ):
        del mosaic, degrees, translate, mosaic_scale, mixup_scale, shear
        del enable_mixup, mosaic_prob, mixup_prob
        self.dataset = dataset
        self.img_size = img_size
        self.preproc = preproc or RFDETRSegTransform(imgsz=img_size[0])

    def __len__(self):
        return len(self.dataset)

    @property
    def input_dim(self):
        return self.img_size

    def set_stop_epoch(self, stop_epoch: int):
        # Compatibility shim — no strong-aug toggle here.
        return

    def set_epoch(self, epoch: int):
        # Compatibility shim — no per-epoch state.
        return

    def close_mosaic(self):
        # Compatibility shim — mosaic is never enabled here.
        return

    def __getitem__(self, idx):
        item = self.dataset.pull_item(idx)
        if len(item) == 5:
            img, label, img_info, img_id, segments = item
        else:
            img, label, img_info, img_id = item
            segments = None
        img, label, masks = self.preproc(img, label, self.input_dim, segments)
        return img, label, img_info, img_id, masks


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


def _build_pose_target_cxcywh(
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
        self.target_size = resolve_training_size(
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
            img, scale_x, scale_y = resize_shortest_side(
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
        img_rgb, scale_x, scale_y = resize_square(img, (target_h, target_w))
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
        target = _build_pose_target_cxcywh(
            cls, boxes_cxcywh, kpts, self.num_keypoints, self.max_labels
        )

        img_out = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_out = (img_out - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.ascontiguousarray(img_out), target

"""RF-DETR segmentation transforms: letterbox + flip + ImageNet norm + polygon rasterization.

Mirrors DFINETrainTransform's output contract for detection (cxcywh pixel coords on the
resized canvas, ImageNet-normalized CHW float32 RGB) and additionally rasterizes per-instance
polygon rings to a dense (max_labels, mask_h, mask_w) float32 tensor. The mask resolution is
input_dim / mask_downsample_ratio (default 4) to match the SegmentationHead's downsample.

Strong augmentations (mosaic, mixup, photometric distort, IoU crop, zoom out) are intentionally
omitted: torchvision.v2 supports Mask tv_tensors but the polygon-rings → boxes consistency under
those ops needs explicit per-instance copies that are out of scope for the initial seg port.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _letterbox(img: np.ndarray, input_dim) -> tuple[np.ndarray, float]:
    """BGR HWC → padded RGB HWC, returning the resize ratio."""
    padded = np.full((input_dim[0], input_dim[1], 3), 114, dtype=np.uint8)
    r = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
    new_w, new_h = int(round(img.shape[1] * r)), int(round(img.shape[0] * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded[:new_h, :new_w] = resized
    padded = padded[:, :, ::-1]  # BGR → RGB
    return padded, r


def _copy_segments(segments):
    if segments is None:
        return None
    return [[ring.copy() for ring in instance] for instance in segments]


def _flip_segments_lr(segments, width):
    if segments is None:
        return None
    out = []
    for instance in segments:
        flipped = []
        for ring in instance:
            if ring is None or len(ring) == 0:
                flipped.append(ring)
                continue
            r = ring.copy()
            r[:, 0] = width - r[:, 0]
            flipped.append(r)
        out.append(flipped)
    return out


def _scale_segments(segments, scale: float):
    if segments is None:
        return None
    out = []
    for instance in segments:
        scaled = []
        for ring in instance:
            if ring is None or len(ring) == 0:
                scaled.append(ring)
                continue
            scaled.append(ring.astype(np.float32, copy=True) * scale)
        out.append(scaled)
    return out


def _filter_segments(segments, keep_mask):
    if segments is None:
        return None
    keep = np.asarray(keep_mask, dtype=bool)
    n = min(len(segments), len(keep))
    return [segments[i] for i in range(n) if keep[i]]


def _rasterize_segments(segments, image_shape, mask_shape, max_masks):
    """Render per-instance polygon rings to a (max_masks, mask_h, mask_w) float32 array.

    Polygons are given in pixel coords on ``image_shape``; they are scaled into
    ``mask_shape`` before being filled into individual mask slots.
    """
    masks = np.zeros((max_masks, mask_shape[0], mask_shape[1]), dtype=np.float32)
    if not segments:
        return masks

    img_h, img_w = image_shape
    mask_h, mask_w = mask_shape
    sx = mask_w / max(float(img_w), 1.0)
    sy = mask_h / max(float(img_h), 1.0)

    for idx, instance in enumerate(segments[:max_masks]):
        polygons = []
        for ring in instance:
            if ring is None or len(ring) < 3:
                continue
            poly = ring.astype(np.float32, copy=True)
            poly[:, 0] *= sx
            poly[:, 1] *= sy
            polygons.append(np.round(poly).astype(np.int32))
        if polygons:
            cv2.fillPoly(masks[idx], polygons, color=1)
    return masks


class RFDETRSegTransform:
    """Per-sample seg transform: letterbox + flip + ImageNet norm + polygon rasterization.

    Output: ``(img_chw_float_rgb_imagenet, padded_labels [max_labels, 5] cxcywh-pixel,
    masks [max_labels, mask_h, mask_w] float32)``.

    The trainer's ``on_forward`` converts cxcywh-pixel → cxcywh-normalized and slices
    masks to ``[num_valid, H, W]`` per image before passing to the criterion.
    """

    # Surfaced so YOLODataset/COCODataset hand us the original image, not the
    # pre-letterboxed one — we need original pixel coords for polygon scaling.
    wants_unresized_image = True

    def __init__(
        self,
        max_labels: int = 300,
        flip_prob: float = 0.5,
        imgsz: int = 512,
        mask_downsample_ratio: int = 4,
    ):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.imgsz = imgsz
        self.mask_downsample_ratio = mask_downsample_ratio

    def disable_strong_augs(self):
        # Compatibility shim: no strong augs to disable.
        return

    def __call__(self, image: np.ndarray, targets: np.ndarray, input_dim, segments=None):
        target_h, target_w = input_dim
        boxes = targets[:, :4].astype(np.float32, copy=True) if len(targets) else np.zeros((0, 4), np.float32)
        labels = targets[:, 4].astype(np.float32, copy=True) if len(targets) else np.zeros((0,), np.float32)
        segments_t = _copy_segments(segments)

        # Optional horizontal flip — applied before resize, on the original canvas.
        if random.random() < self.flip_prob:
            _, w_orig, _ = image.shape
            image = image[:, ::-1].copy()
            if len(boxes):
                boxes[:, [0, 2]] = w_orig - boxes[:, [2, 0]]
            segments_t = _flip_segments_lr(segments_t, w_orig)

        # Letterbox-resize. Same logic as YOLO9's preproc — returns ratio r.
        img_rgb, r = _letterbox(image, input_dim)
        if len(boxes):
            boxes *= r
        segments_t = _scale_segments(segments_t, r)

        # Drop boxes that collapsed below 1px after resize. Apply the same keep mask
        # to segments so per-instance alignment is preserved.
        if len(boxes):
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            keep = (w > 1) & (h > 1)
            if not keep.all():
                boxes = boxes[keep]
                labels = labels[keep]
                segments_t = _filter_segments(segments_t, keep)

        # xyxy(pixel) → cxcywh(pixel) on the resized canvas — matches DFINE's contract.
        if len(boxes):
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            packed = np.stack([labels, cx, cy, w, h], axis=1).astype(np.float32, copy=False)
        else:
            packed = np.zeros((0, 5), dtype=np.float32)

        padded = np.zeros((self.max_labels, 5), dtype=np.float32)
        n = min(len(packed), self.max_labels)
        if n:
            padded[:n] = packed[:n]

        # CHW float32 in [0, 1], then ImageNet normalize.
        img_out = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_out = (img_out - _IMAGENET_MEAN) / _IMAGENET_STD
        img_out = np.ascontiguousarray(img_out)

        mask_shape = (
            target_h // self.mask_downsample_ratio,
            target_w // self.mask_downsample_ratio,
        )
        masks = _rasterize_segments(
            segments_t,
            image_shape=(target_h, target_w),
            mask_shape=mask_shape,
            max_masks=self.max_labels,
        )

        return img_out, padded, masks


class RFDETRSegPassThroughDataset:
    """Identity wrapper that runs the seg transform per item — no mosaic.

    Mirrors DFINEPassThroughDataset's constructor contract so BaseTrainer's
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


__all__ = ["RFDETRSegTransform", "RFDETRSegPassThroughDataset"]

"""RF-DETR transforms: resize/crop policy + flip + ImageNet norm + mask rasterization.

Mirrors DFINETrainTransform's output contract for detection (cxcywh pixel coords on the
resized canvas, ImageNet-normalized CHW float32 RGB) and additionally rasterizes per-instance
polygon rings to a dense (max_labels, H, W) float32 tensor at the transformed image resolution.

Mosaic, mixup, photometric distort, IoU crop, and zoom-out are intentionally omitted; RF-DETR
training uses a direct square canvas with an optional upstream-style random resize/crop branch.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


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
    if not scales:
        return imgsz
    # LibreYOLO stacks per-sample transform outputs directly. Upstream's
    # default also disables per-step random resize and uses the largest expanded
    # square scale, so keep one stable canvas per dataloader.
    if not do_random_resize_via_padding:
        return scales[-1]
    return imgsz


def _resize_square(img: np.ndarray, input_dim) -> tuple[np.ndarray, float, float]:
    """BGR HWC -> resized RGB HWC, returning x/y scale factors."""
    target_h, target_w = input_dim
    scale_x = target_w / img.shape[1]
    scale_y = target_h / img.shape[0]
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized = resized[:, :, ::-1]  # BGR -> RGB
    return resized, scale_x, scale_y


def _resize_shortest_side(img: np.ndarray, size: int) -> tuple[np.ndarray, float, float]:
    h, w = img.shape[:2]
    scale = size / max(1, min(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale, scale


def _copy_segments(segments):
    if segments is None:
        return None
    return [[ring.copy() for ring in instance] for instance in segments]


def _dense_mask(ring):
    return getattr(ring, "dense_mask", None)


def _set_dense_mask(ring, mask):
    if hasattr(ring, "dense_mask"):
        ring.dense_mask = np.ascontiguousarray(mask.astype(np.uint8))


def _instance_dense_mask(instance):
    for ring in instance:
        mask = _dense_mask(ring)
        if mask is not None:
            return mask
    return None


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
            mask = _dense_mask(r)
            if mask is not None:
                _set_dense_mask(r, mask[:, ::-1])
            flipped.append(r)
        out.append(flipped)
    return out


def _scale_segments_xy(segments, scale_x: float, scale_y: float):
    if segments is None:
        return None
    out = []
    for instance in segments:
        scaled = []
        for ring in instance:
            if ring is None or len(ring) == 0:
                scaled.append(ring)
                continue
            mask = _dense_mask(ring)
            ring_scaled = ring.astype(np.float32, copy=True)
            if mask is not None:
                ring_scaled = ring_scaled.view(type(ring))
            ring_scaled[:, 0] *= scale_x
            ring_scaled[:, 1] *= scale_y
            if mask is not None:
                new_w = max(1, int(round(mask.shape[1] * scale_x)))
                new_h = max(1, int(round(mask.shape[0] * scale_y)))
                scaled_mask = cv2.resize(
                    mask,
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                _set_dense_mask(ring_scaled, scaled_mask)
            scaled.append(ring_scaled)
        out.append(scaled)
    return out


def _crop_segments(segments, left: int, top: int, width: int, height: int):
    if segments is None:
        return None
    out = []
    for instance in segments:
        cropped = []
        for ring in instance:
            if ring is None or len(ring) == 0:
                cropped.append(ring)
                continue
            mask = _dense_mask(ring)
            r = ring.copy()
            r[:, 0] = np.clip(r[:, 0] - left, 0.0, float(width))
            r[:, 1] = np.clip(r[:, 1] - top, 0.0, float(height))
            if mask is not None:
                _set_dense_mask(r, mask[top : top + height, left : left + width])
            cropped.append(r)
        out.append(cropped)
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
        dense_mask = _instance_dense_mask(instance)
        if dense_mask is not None:
            mask = dense_mask
            if mask.shape != mask_shape:
                mask = cv2.resize(mask, (mask_w, mask_h), interpolation=cv2.INTER_NEAREST)
            masks[idx] = (mask > 0).astype(np.float32)
            continue

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
    """Per-sample seg transform: square resize + flip + ImageNet norm + polygon rasterization.

    Output: ``(img_chw_float_rgb_imagenet, padded_labels [max_labels, 5] cxcywh-pixel,
    masks [max_labels, mask_h, mask_w] float32)``.

    The trainer's ``on_forward`` converts cxcywh-pixel → cxcywh-normalized and slices
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
    ):
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
        self.target_size = _resolve_training_size(
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
        segments_t = _copy_segments(segments)

        # Optional horizontal flip — applied before resize, on the original canvas.
        if random.random() < self.flip_prob:
            _, w_orig, _ = image.shape
            image = image[:, ::-1].copy()
            if len(boxes):
                boxes[:, [0, 2]] = w_orig - boxes[:, [2, 0]]
            segments_t = _flip_segments_lr(segments_t, w_orig)

        if len(boxes) and self.crop_resize_prob > 0 and random.random() < self.crop_resize_prob:
            image, scale_x, scale_y = _resize_shortest_side(
                image,
                random.choice(self.crop_intermediate_sizes),
            )
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            segments_t = _scale_segments_xy(segments_t, scale_x, scale_y)

            h_mid, w_mid = image.shape[:2]
            max_crop = min(self.crop_max_size, h_mid, w_mid)
            min_crop = min(self.crop_min_size, max_crop)
            if max_crop >= 2:
                crop_size = random.randint(min_crop, max_crop)
                top = random.randint(0, max(0, h_mid - crop_size))
                left = random.randint(0, max(0, w_mid - crop_size))
                image = image[top : top + crop_size, left : left + crop_size]
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] - left, 0.0, float(crop_size))
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] - top, 0.0, float(crop_size))
                segments_t = _crop_segments(segments_t, left, top, crop_size, crop_size)

        # RF-DETR's square training/inference path resizes directly to the model
        # canvas, so boxes and masks use independent x/y scale factors.
        img_rgb, scale_x, scale_y = _resize_square(image, (target_h, target_w))
        if len(boxes):
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        segments_t = _scale_segments_xy(segments_t, scale_x, scale_y)

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

        mask_shape = (target_h, target_w)
        masks = _rasterize_segments(
            segments_t,
            image_shape=(target_h, target_w),
            mask_shape=mask_shape,
            max_masks=self.max_labels,
        )

        return img_out, padded, masks


class RFDETRDetTransform(RFDETRSegTransform):
    """RF-DETR detection transform using the same square geometry."""

    def __call__(self, image: np.ndarray, targets: np.ndarray, input_dim):
        img, labels, _ = super().__call__(image, targets, input_dim, segments=None)
        return img, labels


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

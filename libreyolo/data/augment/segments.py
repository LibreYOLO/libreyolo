"""Shared polygon and dense-mask augmentation helpers."""

import cv2
import numpy as np


def copy_segments(segments):
    if segments is None:
        return None
    return [[ring.copy() for ring in instance] for instance in segments]


def dense_mask(ring):
    return getattr(ring, "dense_mask", None)


def set_dense_mask(ring, mask):
    if hasattr(ring, "dense_mask"):
        ring.dense_mask = np.ascontiguousarray(mask.astype(np.uint8))


def instance_dense_mask(instance):
    for ring in instance:
        mask = dense_mask(ring)
        if mask is not None:
            return mask
    return None


def materialize_dense_masks_for_crop(segments, image_shape):
    if segments is None:
        return None

    from ..dataset import DenseMaskRing

    height, width = image_shape
    out = []
    for instance in segments:
        if instance_dense_mask(instance) is not None:
            out.append(instance)
            continue

        valid_rings = [
            np.asarray(ring, dtype=np.float32)
            for ring in instance
            if ring is not None and len(ring) >= 3
        ]
        if not valid_rings:
            out.append(instance)
            continue

        mask = np.zeros((height, width), dtype=np.uint8)
        polygons = [np.round(ring).astype(np.int32) for ring in valid_rings]
        cv2.fillPoly(mask, polygons, color=1)

        wrapped = []
        attached = False
        for ring in instance:
            if not attached and ring is not None and len(ring) >= 3:
                wrapped.append(DenseMaskRing(ring, mask))
                attached = True
            else:
                wrapped.append(ring)
        out.append(wrapped)
    return out


def transform_segments(
    segments,
    scale=1.0,
    padw=0.0,
    padh=0.0,
    width=None,
    height=None,
):
    if segments is None:
        return None
    transformed = []
    for instance in segments:
        rings = []
        for ring in instance:
            r = ring.astype(np.float32, copy=True)
            r[:, 0] = r[:, 0] * scale + padw
            r[:, 1] = r[:, 1] * scale + padh
            if width is not None:
                r[:, 0] = np.clip(r[:, 0], 0, width)
            if height is not None:
                r[:, 1] = np.clip(r[:, 1], 0, height)
            rings.append(r)
        transformed.append(rings)
    return transformed


def flip_segments_lr(segments, width):
    if segments is None:
        return None
    out = []
    for instance in segments:
        flipped = []
        for ring in instance:
            if ring is None or len(ring) == 0:
                flipped.append(ring)
                continue
            mask = dense_mask(ring)
            r = ring.copy() if mask is not None else ring.astype(np.float32, copy=True)
            r[:, 0] = width - r[:, 0]
            if mask is not None:
                set_dense_mask(r, mask[:, ::-1])
            flipped.append(r)
        out.append(flipped)
    return out


def flip_segments_ud(segments, height):
    if segments is None:
        return None
    out = []
    for instance in segments:
        flipped = []
        for ring in instance:
            if ring is None or len(ring) == 0:
                flipped.append(ring)
                continue
            mask = dense_mask(ring)
            r = ring.copy() if mask is not None else ring.astype(np.float32, copy=True)
            r[:, 1] = height - r[:, 1]
            if mask is not None:
                set_dense_mask(r, mask[::-1, :])
            flipped.append(r)
        out.append(flipped)
    return out


def scale_segments_xy(segments, scale_x: float, scale_y: float):
    if segments is None:
        return None
    out = []
    for instance in segments:
        scaled = []
        for ring in instance:
            if ring is None or len(ring) == 0:
                scaled.append(ring)
                continue
            mask = dense_mask(ring)
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
                set_dense_mask(ring_scaled, scaled_mask)
            scaled.append(ring_scaled)
        out.append(scaled)
    return out


def crop_segments(segments, left: int, top: int, width: int, height: int):
    if segments is None:
        return None
    out = []
    for instance in segments:
        cropped = []
        for ring in instance:
            if ring is None or len(ring) == 0:
                cropped.append(ring)
                continue
            mask = dense_mask(ring)
            r = ring.copy()
            r[:, 0] = np.clip(r[:, 0] - left, 0.0, float(width))
            r[:, 1] = np.clip(r[:, 1] - top, 0.0, float(height))
            if mask is not None:
                set_dense_mask(r, mask[top : top + height, left : left + width])
            cropped.append(r)
        out.append(cropped)
    return out


def filter_segments(segments, keep_mask):
    if segments is None:
        return None
    keep = np.asarray(keep_mask, dtype=bool)
    n = min(len(segments), len(keep))
    return [segments[i] for i in range(n) if keep[i]]


def rasterize_segments(segments, image_shape, mask_shape, max_masks):
    """Render per-instance polygon rings to a (max_masks, mask_h, mask_w) float32 array."""
    masks = np.zeros((max_masks, mask_shape[0], mask_shape[1]), dtype=np.float32)
    if not segments:
        return masks

    img_h, img_w = image_shape
    mask_h, mask_w = mask_shape
    sx = mask_w / max(float(img_w), 1.0)
    sy = mask_h / max(float(img_h), 1.0)

    for idx, instance in enumerate(segments[:max_masks]):
        dense = instance_dense_mask(instance)
        if dense is not None:
            mask = dense
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

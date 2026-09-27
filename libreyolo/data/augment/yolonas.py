"""YOLO-NAS training transforms and dataset wrapper.

Moved verbatim from ``libreyolo/models/yolonas/transforms.py``. Emits
``[class, cx, cy, w, h]`` pixel targets over an RGB /255 letterbox.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .boxes import adjust_box_anns, xyxy2cxcywh
from .color import augment_hsv
from .geometry import letterbox_preproc, mirror, mirror_vertical, random_affine


def preproc(img, input_size, swap=(2, 0, 1)):
    """Letterbox to RGB float32/0-1, matching the native inference path."""
    return letterbox_preproc(img, input_size, swap, to_rgb=True, scale=True)


class YOLONASTrainTransform:
    """Train transform emitting `[class, cx, cy, w, h]` pixel targets."""

    def __init__(self, max_labels=100, flip_prob=0.5, hsv_prob=0.5, flipud=0.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        # Vertical-flip probability (off by default). Guarded so a disabled
        # knob draws no random numbers and leaves existing behavior untouched.
        self.flipud = flipud

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        if len(boxes) == 0:
            padded_labels = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, _ = preproc(image, input_dim)
            return image, padded_labels

        image_o = image.copy()
        boxes_o = boxes.copy()
        labels_o = labels.copy()
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)

        image_t, boxes = mirror(image, boxes, self.flip_prob)
        if self.flipud > 0:
            image_t, boxes = mirror_vertical(image_t, boxes, self.flipud)
        image_t, r = preproc(image_t, input_dim)
        boxes = xyxy2cxcywh(boxes)
        boxes *= r

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5), dtype=np.float32)
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class YOLONASAffineMixupDataset:
    """Small YOLO-NAS-specific wrapper with affine + optional mixup.

    The constructor matches BaseTrainer's existing dataset-wrapper contract so
    the family can plug into shared training infrastructure without widening
    that interface first.
    """

    def __init__(
        self,
        dataset,
        img_size,
        mosaic=True,
        preproc=None,
        degrees=0.0,
        translate=0.25,
        mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5),
        shear=0.0,
        enable_mixup=False,
        mosaic_prob=0.0,
        mixup_prob=0.0,
        perspective=0.0,
    ):
        del mosaic, mosaic_prob
        self.dataset = dataset
        self.img_size = img_size
        self.preproc = preproc or YOLONASTrainTransform()
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mixup_scale
        self.enable_affine = True
        self.enable_mixup = enable_mixup
        self.mixup_prob = mixup_prob

    def __len__(self):
        return len(self.dataset)

    @property
    def input_dim(self):
        return self.img_size

    def close_mosaic(self):
        self.enable_affine = False
        self.enable_mixup = False

    def __getitem__(self, idx):
        img, label, img_info, img_id = self.dataset.pull_item(idx)

        if self.enable_affine:
            input_h, input_w = self.input_dim
            img, label = random_affine(
                img,
                label,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
                perspective=self.perspective,
            )

        if self.enable_mixup and len(label) > 0 and random.random() < self.mixup_prob:
            img, label = self._mixup(img, label)

        img, label = self.preproc(img, label, self.input_dim)
        return img, label, img_info, img_id

    def _mixup(self, origin_img, origin_labels):
        jit_factor = random.uniform(*self.mixup_scale)
        flip = random.uniform(0, 1) > 0.5

        cp_labels = []
        cp_index = None
        for _ in range(20):
            cp_index = random.randint(0, len(self.dataset) - 1)
            cp_labels = self.dataset.load_anno(cp_index)
            if len(cp_labels) > 0:
                break
        else:
            # Every sampled partner was background; skip mixup rather than
            # loop forever when the dataset has no foreground labels.
            return origin_img, origin_labels

        img, cp_labels, _, _ = self.dataset.pull_item(cp_index)
        input_dim = self.input_dim
        cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if flip:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)

        padded_cropped_img = padded_img[
            y_offset : y_offset + target_h, x_offset : x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if flip:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        cls_labels = cp_labels[:, 4:5].copy()
        labels = np.hstack((cp_bboxes_transformed_np, cls_labels))
        merged_labels = np.vstack((origin_labels, labels))

        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)
        return origin_img.astype(np.uint8), merged_labels


# ---------------------------------------------------------------------------
# OBB (YOLO-NAS-R) training transform
# ---------------------------------------------------------------------------


def _canonicalize_obb_rows(rows: np.ndarray) -> np.ndarray:
    """Put the long side first and fold the angle into ``[-pi/2, pi/2)``.

    ``rows`` are ``[cx, cy, w, h, angle]``. Vectorised twin of
    ``libreyolo.data.obb.canonicalize_xywhr`` (which is per-row and raises on
    degenerate boxes; here degenerate rows are dropped by the caller instead).
    """
    if len(rows) == 0:
        return rows
    out = rows.copy()
    swap = out[:, 3] > out[:, 2]
    w = np.where(swap, out[:, 3], out[:, 2])
    h = np.where(swap, out[:, 2], out[:, 3])
    angle = np.where(swap, out[:, 4] + np.pi / 2, out[:, 4])
    out[:, 2] = w
    out[:, 3] = h
    out[:, 4] = np.mod(angle + np.pi / 2, np.pi) - np.pi / 2
    return out


class YOLONASOBBTrainTransform:
    """Train transform for YOLO-NAS-R, emitting ``[class, cx, cy, w, h, angle]``.

    Augmentation is deliberately limited to HSV jitter and axis flips. Every
    geometric op here has an exact, tested effect on the angle; affine,
    mosaic and mixup do not (a shear turns a rectangle into a parallelogram,
    and a mosaic paste crops rotated boxes), so they are not offered rather
    than being offered as knobs that quietly corrupt the label.

    The resize/pad/normalize step calls ``preprocess_obb_numpy`` -- the same
    function inference and validation use -- so the three paths cannot drift.
    """

    def __init__(
        self,
        max_labels: int = 300,
        flip_prob: float = 0.5,
        hsv_prob: float = 0.5,
        flipud: float = 0.0,
    ):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.flipud = flipud

    def _pad(self, targets: np.ndarray) -> np.ndarray:
        padded = np.zeros((self.max_labels, 6), dtype=np.float32)
        n = min(len(targets), self.max_labels)
        if n:
            padded[:n] = targets[:n]
        return np.ascontiguousarray(padded, dtype=np.float32)

    def __call__(self, image, targets, input_dim):
        from libreyolo.preprocess.yolonas import preprocess_obb_numpy

        input_size = input_dim[0] if isinstance(input_dim, (tuple, list)) else input_dim

        if targets is None or len(targets) == 0:
            rgb = np.ascontiguousarray(image[:, :, ::-1])
            image_t, _ = preprocess_obb_numpy(rgb, input_size=input_size)
            return image_t, self._pad(np.zeros((0, 6), dtype=np.float32))

        targets = np.asarray(targets, dtype=np.float32)
        if targets.shape[1] < 6:
            raise ValueError(
                "YOLO-NAS OBB training expects six-column dataset rows "
                f"[x1, y1, x2, y2, class, angle], got {targets.shape[1]}"
            )

        # The dataset's xyxy block is the un-rotated proxy, so it decodes back
        # to (cx, cy, w, h) exactly (see data/dataset.py: xywhr_to_proxy_xyxy).
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        angles = targets[:, 5].copy()

        if random.random() < self.hsv_prob:
            augment_hsv(image)

        height, width = image.shape[:2]
        if random.random() < self.flip_prob:
            image = image[:, ::-1]
            x1 = boxes[:, 0].copy()
            boxes[:, 0] = width - boxes[:, 2]
            boxes[:, 2] = width - x1
            angles = -angles
        if self.flipud > 0 and random.random() < self.flipud:
            image = image[::-1]
            y1 = boxes[:, 1].copy()
            boxes[:, 1] = height - boxes[:, 3]
            boxes[:, 3] = height - y1
            angles = -angles

        rgb = np.ascontiguousarray(image[:, :, ::-1])
        image_t, ratio = preprocess_obb_numpy(rgb, input_size=input_size)

        cxcywh = xyxy2cxcywh(boxes) * ratio
        rows = np.concatenate([cxcywh, angles[:, None]], axis=1)
        rows = _canonicalize_obb_rows(rows)

        keep = np.minimum(rows[:, 2], rows[:, 3]) > 1
        rows = rows[keep]
        labels = labels[keep]

        targets_t = np.hstack((labels[:, None], rows)).astype(np.float32)
        return image_t, self._pad(targets_t)


class YOLONASOBBDataset:
    """Pass-through dataset wrapper for OBB training.

    Matches ``BaseTrainer``'s dataset-wrapper constructor signature but
    performs no mosaic, mixup or affine: see
    :class:`YOLONASOBBTrainTransform` for why. It exists so the OBB trainer
    can plug into the shared training loop unchanged.
    """

    def __init__(self, dataset, img_size, mosaic=True, preproc=None, **kwargs):
        del mosaic, kwargs
        self.dataset = dataset
        self.img_size = img_size
        self.preproc = preproc or YOLONASOBBTrainTransform()

    def __len__(self):
        return len(self.dataset)

    @property
    def input_dim(self):
        return self.img_size

    def close_mosaic(self):
        # Nothing to close; kept for the shared trainer's late-epoch hook.
        return None

    def __getitem__(self, idx):
        img, label, img_info, img_id = self.dataset.pull_item(idx)
        img, label = self.preproc(img, label, self.input_dim)
        return img, label, img_info, img_id

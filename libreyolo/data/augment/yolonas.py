"""YOLO-NAS augmentation recipe: random affine + paste mixup, RGB [0, 1] outputs."""

from __future__ import annotations

import random

import numpy as np

from .boxes import xyxy2cxcywh
from .color import augment_hsv
from .geometry import letterbox_rgb01, mirror, random_affine
from .mosaic import mixup_paste


class YOLONASTrainTransform:
    """Train transform emitting `[class, cx, cy, w, h]` pixel targets."""

    def __init__(self, max_labels=100, flip_prob=0.5, hsv_prob=0.5):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        if len(boxes) == 0:
            padded_labels = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, _ = letterbox_rgb01(image, input_dim)
            return image, padded_labels

        image_o = image.copy()
        boxes_o = boxes.copy()
        labels_o = labels.copy()
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)

        image_t, boxes = mirror(image, boxes, self.flip_prob)
        image_t, r = letterbox_rgb01(image_t, input_dim)
        boxes = xyxy2cxcywh(boxes)
        boxes *= r

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = letterbox_rgb01(image_o, input_dim)
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
    ):
        del mosaic, mosaic_prob
        self.dataset = dataset
        self.img_size = img_size
        self.preproc = preproc or YOLONASTrainTransform()
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
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
            )

        if self.enable_mixup and len(label) > 0 and random.random() < self.mixup_prob:
            img, label = mixup_paste(
                self.dataset, img, label, self.input_dim, self.mixup_scale
            )

        img, label = self.preproc(img, label, self.input_dim)
        return img, label, img_info, img_id

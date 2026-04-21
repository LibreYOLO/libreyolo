"""YOLO-NAS training transforms and dataset wrapper."""

from __future__ import annotations

import random

import cv2
import numpy as np

from ...training.augment import (
    adjust_box_anns,
    augment_hsv,
    mirror,
    random_affine,
    xyxy2cxcywh,
)


def preproc(img, input_size, swap=(2, 0, 1)):
    """Letterbox to RGB float32/0-1, matching the native inference path."""
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]  # BGR -> RGB
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0
    return padded_img, r


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
            image, _ = preproc(image, input_dim)
            return image, padded_labels

        image_o = image.copy()
        boxes_o = boxes.copy()
        labels_o = labels.copy()
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)

        image_t, boxes = mirror(image, boxes, self.flip_prob)
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
            img, label = self._mixup(img, label)

        img, label = self.preproc(img, label, self.input_dim)
        return img, label, img_info, img_id

    def _mixup(self, origin_img, origin_labels):
        jit_factor = random.uniform(*self.mixup_scale)
        flip = random.uniform(0, 1) > 0.5

        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, len(self.dataset) - 1)
            cp_labels = self.dataset.load_anno(cp_index)

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

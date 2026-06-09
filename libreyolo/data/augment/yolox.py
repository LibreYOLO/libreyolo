"""YOLOX-style augmentation recipe (mosaic + paste mixup + HSV + flip).

Adapted from the official YOLOX repository. Used by the YOLOX, RTMDet,
DAMO-YOLO, and PicoDet trainers; outputs ``[class, cx, cy, w, h]`` targets in
pixel coordinates on a BGR 0-255 CHW image.
"""

import random

import cv2
import numpy as np

from .boxes import xyxy2cxcywh
from .color import augment_hsv
from .geometry import letterbox, mirror, random_affine
from .mosaic import get_mosaic_coordinate, mixup_paste


class TrainTransform:
    """Transform for training data."""

    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = letterbox(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        boxes_o = xyxy2cxcywh(boxes_o)  # [xyxy] -> [cx,cy,w,h]

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = letterbox(image_t, input_dim)
        boxes = xyxy2cxcywh(boxes)  # [xyxy] -> [cx,cy,w,h]
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = letterbox(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """Transform for validation data."""

    def __init__(self, swap=(2, 0, 1)):
        self.swap = swap

    def __call__(self, img, res, input_size):
        img, _ = letterbox(img, input_size, self.swap)
        return img, np.zeros((1, 5))


class MosaicMixupDataset:
    """Dataset wrapper that applies YOLOX-style mosaic and mixup augmentation."""

    def __init__(
        self,
        dataset,
        img_size,
        mosaic=True,
        preproc=None,
        degrees=10.0,
        translate=0.1,
        mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5),
        shear=2.0,
        enable_mixup=True,
        mosaic_prob=1.0,
        mixup_prob=1.0,
    ):
        self.dataset = dataset
        self.img_size = img_size
        self.preproc = preproc or TrainTransform()
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

    def __len__(self):
        return len(self.dataset)

    @property
    def input_dim(self):
        return self.img_size

    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            return self._get_mosaic_item(idx)
        else:
            return self._get_normal_item(idx)

    def _get_normal_item(self, idx):
        img, label, img_info, img_id = self.dataset.pull_item(idx)
        img, label = self.preproc(img, label, self.input_dim)
        return img, label, img_info, img_id

    def _get_mosaic_item(self, idx):
        mosaic_labels = []
        input_h, input_w = self.input_dim[0], self.input_dim[1]

        # Mosaic center
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

        # 3 additional image indices
        indices = [idx] + [random.randint(0, len(self.dataset) - 1) for _ in range(3)]

        for i_mosaic, index in enumerate(indices):
            img, _labels, _, img_id = self.dataset.pull_item(index)
            h0, w0 = img.shape[:2]
            scale = min(1.0 * input_h / h0, 1.0 * input_w / w0)
            img = cv2.resize(
                img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
            )
            (h, w, c) = img.shape[:3]
            if i_mosaic == 0:
                mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
            )

            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            padw, padh = l_x1 - s_x1, l_y1 - s_y1

            labels = _labels.copy()
            if _labels.size > 0:
                labels[:, 0] = scale * _labels[:, 0] + padw
                labels[:, 1] = scale * _labels[:, 1] + padh
                labels[:, 2] = scale * _labels[:, 2] + padw
                labels[:, 3] = scale * _labels[:, 3] + padh
            mosaic_labels.append(labels)

        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
            np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
            np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
            np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

        mosaic_img, mosaic_labels = random_affine(
            mosaic_img,
            mosaic_labels,
            target_size=(input_w, input_h),
            degrees=self.degrees,
            translate=self.translate,
            scales=self.scale,
            shear=self.shear,
        )

        if (
            self.enable_mixup
            and len(mosaic_labels) > 0
            and random.random() < self.mixup_prob
        ):
            mosaic_img, mosaic_labels = self._mixup(mosaic_img, mosaic_labels)

        mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
        img_info = (mix_img.shape[1], mix_img.shape[0])

        return mix_img, padded_labels, img_info, img_id

    def _mixup(self, origin_img, origin_labels):
        return mixup_paste(
            self.dataset, origin_img, origin_labels, self.input_dim, self.mixup_scale
        )

    def close_mosaic(self):
        """Disable mosaic and mixup (called for final no-aug epochs)."""
        self.enable_mosaic = False
        self.enable_mixup = False

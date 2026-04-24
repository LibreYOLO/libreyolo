"""D-FINE training transforms (v1: hflip + plain resize, no mosaic/mixup).

Per the v1 plan, augmentation is intentionally minimal — paper-faithful augs
(``RandomZoomOut``/``RandomIoUCrop``/``RandomPhotometricDistort``) are deferred.
The transform emits the standard LibreYOLO ``(max_labels, 5)`` padded tensor
``[class, cx, cy, w, h]`` in *pixel* coordinates, which the trainer's
``on_forward`` hook translates to D-FINE's ``list[dict]`` cxcywh-normalized
target format.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


class DFINETrainTransform:
    """Plain resize + optional horizontal flip.

    Output:
        image: (3, H, W) RGB float32 in [0, 1].
        targets: (max_labels, 5) ``[class, cx, cy, w, h]`` in PIXEL coords on
            the resized image.
    """

    def __init__(self, max_labels: int = 120, flip_prob: float = 0.5):
        self.max_labels = max_labels
        self.flip_prob = flip_prob

    def __call__(self, image: np.ndarray, targets: np.ndarray, input_dim):
        target_h, target_w = input_dim
        orig_h, orig_w = image.shape[:2]

        boxes = targets[:, :4].astype(np.float32, copy=True) if len(targets) else np.zeros((0, 4), dtype=np.float32)
        labels = targets[:, 4].astype(np.float32, copy=True) if len(targets) else np.zeros((0,), dtype=np.float32)

        image_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if len(boxes):
            scale_x = target_w / max(orig_w, 1)
            scale_y = target_h / max(orig_h, 1)
            boxes[:, 0] *= scale_x
            boxes[:, 2] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 3] *= scale_y

        if random.random() < self.flip_prob and len(boxes):
            image_resized = image_resized[:, ::-1].copy()
            boxes[:, [0, 2]] = target_w - boxes[:, [2, 0]]

        # BGR -> RGB, /255, CHW
        image_t = image_resized[:, :, ::-1].transpose(2, 0, 1)
        image_t = np.ascontiguousarray(image_t, dtype=np.float32) / 255.0

        # xyxy → cxcywh, drop tiny boxes
        if len(boxes):
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            valid = (w > 1) & (h > 1)
            cx, cy, w, h = cx[valid], cy[valid], w[valid], h[valid]
            labels = labels[valid]
            packed = np.stack([labels, cx, cy, w, h], axis=1)
        else:
            packed = np.zeros((0, 5), dtype=np.float32)

        padded = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(packed):
            n = min(len(packed), self.max_labels)
            padded[:n] = packed[:n]
        return image_t, padded


class DFINEPassThroughDataset:
    """Identity wrapper that runs the train transform per item — no mosaic.

    Constructor signature matches ``BaseTrainer._setup_data``'s
    ``MosaicDatasetClass(...)`` contract; all mosaic/mixup kwargs are ignored.
    ``close_mosaic`` is a no-op since there is no mosaic to disable.
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
        self.preproc = preproc or DFINETrainTransform()

    def __len__(self):
        return len(self.dataset)

    @property
    def input_dim(self):
        return self.img_size

    def close_mosaic(self):
        # No mosaic to close. Kept for BaseTrainer.on_mosaic_disable() default.
        return None

    def __getitem__(self, idx):
        img, label, img_info, img_id = self.dataset.pull_item(idx)
        img, label = self.preproc(img, label, self.input_dim)
        return img, label, img_info, img_id

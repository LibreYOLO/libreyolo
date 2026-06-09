"""YOLO-NAS augmentation recipe: random affine + paste mixup, RGB [0, 1] outputs."""

from __future__ import annotations

import random

from .geometry import letterbox_rgb01, random_affine
from .mosaic import mixup_paste
from .yolox import TrainTransform


class YOLONASTrainTransform(TrainTransform):
    """YOLOX-style train transform on the RGB [0, 1] letterbox.

    Same HSV/flip/letterbox sequence as :class:`TrainTransform`; only the
    letterbox (BGR 0-255 -> RGB [0, 1]) and the defaults differ.
    """

    letterbox_fn = staticmethod(letterbox_rgb01)

    def __init__(self, max_labels=100, flip_prob=0.5, hsv_prob=0.5):
        super().__init__(
            max_labels=max_labels, flip_prob=flip_prob, hsv_prob=hsv_prob
        )


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

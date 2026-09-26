"""GTR training augmentation: upstream Mosaic and batch MixUp.

Ported from the pinned GTR recipe (``engine/data/transforms/mosaic.py`` and
``BatchImageCollateFunction`` in ``engine/data/dataloader.py``), which inherits
both from DEIM. See ``NOTICE`` for provenance.

Recipe, per the upstream COCO fine-tune configs (identical for S/M/L/X):

- Mosaic with probability 0.5 during the first ``mosaic_epochs`` epochs. Each
  tile is resized so its shorter side is half the training size, four tiles are
  pasted on a zero canvas from a 50-image per-worker cache, and a random affine
  (10 degrees, 0.1 translation, 0.5-1.5 scale, zero fill) follows. ZoomOut and
  IoUCrop are skipped for mosaic samples; photometric distortion still applies.
- MixUp with probability 0.5 per batch in the same epoch window. Each image is
  blended with its batch neighbour at a ratio drawn from [0.45, 0.55] and both
  label sets are kept.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as tv2

from ...data.augment.detr import _labels_at_index_2
from ..deim.transforms import DEIMPassThroughDataset, DEIMTrainTransform

MOSAIC_CACHE_SIZE = 50
MIXUP_RATIO_RANGE = (0.45, 0.55)


class GTRTrainTransform(DEIMTrainTransform):
    """DEIM transform plus the upstream post-mosaic strong pipeline."""

    def __init__(
        self,
        *args,
        degrees: float = 10.0,
        translate: float = 0.1,
        mosaic_scale=(0.5, 1.5),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._plain_strong = self._strong
        self._mosaic_strong = tv2.Compose(
            [
                tv2.RandomAffine(
                    degrees=degrees,
                    translate=(translate, translate),
                    scale=tuple(mosaic_scale),
                    fill=0,
                ),
                tv2.RandomPhotometricDistort(p=0.5),
                tv2.SanitizeBoundingBoxes(
                    min_size=self.sanitize_min_size, labels_getter=_labels_at_index_2
                ),
            ]
        )

    def __call__(self, image, targets, input_dim, mosaic: bool = False):
        self._strong = self._mosaic_strong if mosaic else self._plain_strong
        return super().__call__(image, targets, input_dim)


class GTRMosaicDataset(DEIMPassThroughDataset):
    """DEIM pass-through wrapper that adds the upstream cached Mosaic."""

    _default_transform_cls = GTRTrainTransform

    def __init__(self, dataset, img_size, *args, mosaic_prob=0.0, **kwargs):
        super().__init__(dataset, img_size, *args, **kwargs)
        self.mosaic_prob = float(mosaic_prob)
        self.mosaic_epochs = 0
        self._cache: list[tuple[np.ndarray, np.ndarray]] = []

    def set_mosaic_epochs(self, epochs: int):
        self.mosaic_epochs = int(epochs)

    def _mosaic_active(self) -> bool:
        stop = self._stop_epoch if self._stop_epoch is not None else 10**9
        return (
            self.mosaic_prob > 0
            and self._epoch < min(self.mosaic_epochs, stop)
            and getattr(self.preproc, "strong_augs", False)
        )

    def _tile_size(self) -> int:
        size = self.input_dim
        size = size[0] if isinstance(size, (tuple, list)) else size
        return max(1, int(size) // 2)

    def _resize_tile(self, img: np.ndarray, label: np.ndarray):
        h, w = img.shape[:2]
        scale = self._tile_size() / min(h, w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label = label.astype(np.float32, copy=True)
        if len(label):
            label[:, [0, 2]] *= new_w / w
            label[:, [1, 3]] *= new_h / h
        return img, label

    def _build_mosaic(self, img: np.ndarray, label: np.ndarray):
        tile = self._resize_tile(img, label)
        self._cache.append(tile)
        if len(self._cache) > MOSAIC_CACHE_SIZE:
            # Never evict the image just added.
            self._cache.pop(random.randint(0, len(self._cache) - 2))
        tiles = [tile] + random.choices(self._cache, k=3)

        max_h = max(t[0].shape[0] for t in tiles)
        max_w = max(t[0].shape[1] for t in tiles)
        canvas = np.zeros((2 * max_h, 2 * max_w, 3), dtype=np.uint8)
        labels = []
        for (tile_img, tile_label), (x0, y0) in zip(
            tiles, ((0, 0), (max_w, 0), (0, max_h), (max_w, max_h))
        ):
            h, w = tile_img.shape[:2]
            canvas[y0 : y0 + h, x0 : x0 + w] = tile_img
            if len(tile_label):
                shifted = tile_label.copy()
                shifted[:, [0, 2]] += x0
                shifted[:, [1, 3]] += y0
                labels.append(shifted)
        label = np.concatenate(labels) if labels else np.zeros((0, 5), np.float32)
        return canvas, label

    def __getitem__(self, idx):
        img, label, img_info, img_id = self.dataset.pull_item(idx)
        mosaic = self._mosaic_active() and random.random() <= self.mosaic_prob
        if mosaic:
            img, label = self._build_mosaic(img, label)
        img, label = self.preproc(img, label, self.input_dim, mosaic=mosaic)
        return img, label, img_info, img_id


class GTRMixUpCollate:
    """Upstream batch MixUp applied on top of the family collate."""

    def __init__(self, collate_fn, mixup_prob: float, mixup_epochs: int):
        self.collate_fn = collate_fn
        self.mixup_prob = float(mixup_prob)
        self.mixup_epochs = int(mixup_epochs)
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch
        if hasattr(self.collate_fn, "set_epoch"):
            self.collate_fn.set_epoch(epoch)

    def __call__(self, batch):
        imgs, labels, *rest = self.collate_fn(batch)
        if self._epoch < self.mixup_epochs and random.random() < self.mixup_prob:
            imgs, labels = mixup_batch(imgs, labels, random.uniform(*MIXUP_RATIO_RANGE))
        return (imgs, labels, *rest)


def mixup_batch(imgs: torch.Tensor, labels: torch.Tensor, beta: float):
    """Blend each image with its predecessor and keep both label sets.

    The padded label tensor widens when a blended pair has more objects than
    the per-image padding, so no visible object loses its target.
    """
    mixed = imgs.roll(shifts=1, dims=0).mul(1.0 - beta).add_(imgs.mul(beta))
    shifted = labels.roll(shifts=1, dims=0)
    merged = []
    for i in range(labels.shape[0]):
        own = labels[i][(labels[i, :, 3] > 0) & (labels[i, :, 4] > 0)]
        other = shifted[i][(shifted[i, :, 3] > 0) & (shifted[i, :, 4] > 0)]
        merged.append(torch.cat([own, other]))
    width = max([labels.shape[1]] + [len(m) for m in merged])
    out = labels.new_zeros((labels.shape[0], width, labels.shape[2]))
    for i, m in enumerate(merged):
        out[i, : len(m)] = m
    return mixed, out

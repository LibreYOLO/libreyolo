"""GTR oriented-box (DOTA) fine-tuning.

Recipe from Intellindust-AI-Lab/GTR (MIT), revision
782e737efe2e6437ac537fbdcee089673d3376c1: configs/obb/gtrobb_base.yml and
configs/obb/dota_finetune/gtrobb_{s,x}.yml. MAL + L1 + KLD losses with Chamfer/KLD
Hungarian matching over three query groups, oriented denoising, AdamW with the
size-specific backbone learning rate, quadratic warmup then a flat learning rate
for the whole run, EMA and gradient clipping. Augmentation ports
engine/data/transforms/obb_transforms.py: a random horizontal, vertical or
diagonal flip (p=0.75) and a random rotation (p=0.5, uniform in +-180 degrees,
or a multiple of 90 degrees when the image holds a square-like class), applied
on the fixed square canvas.

Differences from upstream: upstream trains on pre-split 1024px DOTA tiles, so
it never resizes. LibreYOLO datasets hold arbitrary images, which are resized
to fit the canvas and padded at the bottom and right with zeros, exactly as GTR
OBB inference does.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from ...data.obb import xywhr_to_corners
from ...training.trainer import BaseTrainer
from .config import GTRConfig
from .obb import IMAGENET_MEAN, IMAGENET_STD
from .obb_criterion import OBBGTRCriterion, OBBHungarianMatcher
from .obb_nn import OBB_INPUT_SIZE
from .trainer import GTRTrainer

# Upstream rect_obj_labels [9, 11] in DOTA v1.0 order.
SQUARE_LIKE_CLASS_NAMES = ("storage-tank", "roundabout")
ROTATE_PROB = 0.5
FLIP_DIRECTIONS = ("horizontal", "vertical", "diagonal")
MIN_BOX_SIZE = 2.0

# Per-size recipe values from configs/obb/dota_finetune/gtrobb_{s,x}.yml.
OBB_SIZE_RECIPES = {
    "s": {"epochs": 20, "backbone_lr_mult": 0.00018 / 0.0005, "weight_decay": 1e-4},
    "x": {"epochs": 30, "backbone_lr_mult": 0.000016 / 0.0005, "weight_decay": 1.25e-4},
}


@dataclass
class GTROBBConfig(GTRConfig):
    epochs: int | None = None
    imgsz: int = OBB_INPUT_SIZE
    # Upstream sets flat_epoch equal to epochs: after warmup the learning rate
    # stays flat for the whole fine-tune.
    flat_epochs: int | None = None
    no_aug_epochs: int = 0
    aug_stop_epoch_ratio: float = 1.0
    # Probability of one random flip (horizontal, vertical or diagonal).
    flip_prob: float = 0.75
    # Rotation range in degrees for the random rotation (0 disables it).
    degrees: float = 180.0
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    max_labels: int = 1000
    name: str = "gtr_obb_exp"

    def __post_init__(self):
        recipe = OBB_SIZE_RECIPES.get(self.size)
        if recipe is None:
            raise ValueError("GTR OBB training supports sizes 's' and 'x'")
        if self.epochs is None:
            self.epochs = recipe["epochs"]
        if self.flat_epochs is None:
            self.flat_epochs = self.epochs
        if self.backbone_lr_mult is None:
            self.backbone_lr_mult = recipe["backbone_lr_mult"]
        if self.weight_decay is None:
            self.weight_decay = recipe["weight_decay"]
        super().__post_init__()
        if self.mosaic_prob or self.mixup_prob:
            raise ValueError("GTR OBB training does not support Mosaic/MixUp")
        if not 0.0 <= self.flip_prob <= 1.0:
            raise ValueError("GTR OBB flip_prob must be between 0 and 1")
        if not 0.0 <= self.degrees <= 180.0:
            raise ValueError("GTR OBB degrees must be between 0 and 180")


class GTROBBTrainTransform:
    """Fit-and-pad to the square canvas, flip, rotate, normalize.

    Emits ``(chw float32, (max_labels, 6))`` rows ``[class, cx, cy, w, h, a]``
    with the box normalized by the canvas and ``a = theta / pi`` in the
    long-edge convention (``w >= h``, ``theta`` in ``[0, pi)``), the GTR OBB
    target format. Padding rows are all zero.
    """

    # The dataset hands over the original-resolution image and labels.
    wants_unresized_image = True

    def __init__(
        self,
        imgsz=OBB_INPUT_SIZE,
        max_labels=1000,
        flip_prob=0.75,
        degrees=180.0,
        square_like_labels=(),
    ):
        self.imgsz = int(imgsz)
        self.max_labels = int(max_labels)
        self.flip_prob = float(flip_prob)
        self.degrees = float(degrees)
        self.square_like_labels = {int(c) for c in square_like_labels}
        self.strong_augs = True

    def disable_strong_augs(self):
        self.strong_augs = False

    def _fit(self, image, corners):
        size = self.imgsz
        h, w = image.shape[:2]
        scale = min(size / w, size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        return canvas, corners * scale

    def _flip(self, image, corners):
        if random.random() >= self.flip_prob:
            return image, corners
        direction = random.choice(FLIP_DIRECTIONS)
        size = image.shape[1]
        corners = corners.copy()
        if direction in ("horizontal", "diagonal"):
            image = image[:, ::-1]
            corners[..., 0] = size - corners[..., 0]
        if direction in ("vertical", "diagonal"):
            image = image[::-1]
            corners[..., 1] = size - corners[..., 1]
        return np.ascontiguousarray(image), corners

    def _rotate(self, image, corners, labels):
        if self.degrees <= 0 or random.random() >= ROTATE_PROB:
            return image, corners, labels
        if any(int(c) in self.square_like_labels for c in labels):
            angle = float(random.choice((90, 180, -90, -180)))
        else:
            angle = self.degrees * (2 * random.random() - 1)
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
        image = cv2.warpAffine(
            image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
        )
        if len(corners):
            ones = np.ones((*corners.shape[:2], 1), dtype=np.float32)
            corners = np.concatenate([corners, ones], axis=-1) @ matrix.T
            centers = corners.mean(axis=1)
            keep = (
                (centers[:, 0] >= 0)
                & (centers[:, 0] < w)
                & (centers[:, 1] >= 0)
                & (centers[:, 1] < h)
            )
            corners, labels = corners[keep], labels[keep]
        return image, corners.astype(np.float32), labels

    def _to_rows(self, corners, labels):
        size = float(self.imgsz)
        rows = []
        for poly, cls in zip(corners, labels):
            (cx, cy), (bw, bh), angle = cv2.minAreaRect(poly.astype(np.float32))
            theta = math.radians(angle) % math.pi
            if bw < bh:
                bw, bh = bh, bw
                theta = (theta + math.pi / 2) % math.pi
            if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
                continue
            box = np.clip(
                [cx / size, cy / size, bw / size, bh / size, theta / math.pi],
                0.0,
                1.0,
            )
            box[4] = min(box[4], 1.0 - 1e-6)
            rows.append([cls, *box])
        padded = np.zeros((self.max_labels, 6), dtype=np.float32)
        if rows:
            rows = np.asarray(rows, dtype=np.float32)[: self.max_labels]
            padded[: len(rows)] = rows
        return padded

    def __call__(self, image, targets, input_dim):
        del input_dim
        targets = np.asarray(targets, dtype=np.float32).reshape(-1, 6)
        labels = targets[:, 4].astype(np.int64)
        if len(targets):
            x1, y1, x2, y2 = targets[:, :4].T
            xywhr = np.stack(
                [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, targets[:, 5]], 1
            )
            corners = xywhr_to_corners(xywhr).reshape(-1, 4, 2).astype(np.float32)
        else:
            corners = np.zeros((0, 4, 2), dtype=np.float32)

        rgb = np.ascontiguousarray(image[:, :, ::-1])
        rgb, corners = self._fit(rgb, corners)
        if self.strong_augs:
            rgb, corners = self._flip(rgb, corners)
            rgb, corners, labels = self._rotate(rgb, corners, labels)

        chw = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        chw = ((chw - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
        return chw, self._to_rows(corners, labels)


class GTROBBDataset:
    """Per-sample pass-through wrapper; flip/rotate stop at ``stop_epoch``."""

    def __init__(self, dataset, img_size, mosaic=True, preproc=None, **kwargs):
        del mosaic, kwargs
        self.dataset = dataset
        self.img_size = img_size
        self.preproc = preproc or GTROBBTrainTransform()
        self._stop_epoch = None

    def __len__(self):
        return len(self.dataset)

    @property
    def input_dim(self):
        return self.img_size

    def set_stop_epoch(self, stop_epoch):
        self._stop_epoch = stop_epoch

    def set_epoch(self, epoch):
        if self._stop_epoch is not None and epoch >= self._stop_epoch:
            self.preproc.disable_strong_augs()

    def close_mosaic(self):
        self.preproc.disable_strong_augs()

    def __getitem__(self, idx):
        img, label, img_info, img_id = self.dataset.pull_item(idx)
        img, label = self.preproc(img, label, self.input_dim)
        return img, label, img_info, img_id


class GTROBBTrainer(GTRTrainer):
    artifact_model_families = ("gtr",)
    # Select checkpoints on rotated mAP, not the axis-aligned proxy.
    best_metric_key = "metrics/mAP50-95(OBB)"

    @classmethod
    def _config_class(cls):
        return GTROBBConfig

    def get_model_tag(self):
        return f"GTR-OBB-{self.config.size}"

    def _square_like_labels(self):
        names = getattr(self.wrapper_model, "names", None) or {}
        return tuple(
            int(i) for i, n in names.items() if str(n) in SQUARE_LIKE_CLASS_NAMES
        )

    def create_transforms(self):
        return GTROBBTrainTransform(
            imgsz=self.config.imgsz,
            max_labels=self.config.max_labels,
            flip_prob=self.config.flip_prob,
            degrees=self.config.degrees,
            square_like_labels=self._square_like_labels(),
        ), GTROBBDataset

    def _setup_data(self):
        # The shared data path parses YOLO OBB labels (load_obb); GTR's
        # detection override adds Mosaic/MixUp, which OBB does not use.
        train_dataset = BaseTrainer._setup_data(self)
        from ...data.augment.detr import resolve_aug_stop_epoch

        train_dataset.set_stop_epoch(
            resolve_aug_stop_epoch(
                self.config.epochs,
                self.config.aug_stop_epoch_ratio,
                self.config.no_aug_epochs,
            )
        )
        return train_dataset

    def validate_validation_loss_config(self):
        if getattr(self.config, "val_loss", False):
            raise ValueError("val_loss=True is not supported for GTR OBB training")

    def build_criterion(self, *, distributed_normalize=True):
        matcher = OBBHungarianMatcher(
            weight_dict={"cost_class": 2.0, "cost_chamfer": 5.0, "cost_kld": 2.0},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        )
        return OBBGTRCriterion(
            matcher=matcher,
            weight_dict={
                "loss_mal": 1.0,
                "loss_bbox": 5.0,
                "loss_kld": 5.0,
                "loss_fgl": 0.15,
            },
            # Upstream gtrobb_base.yml enables only mal and boxes; loss_fgl keeps
            # its weight there but is never computed, and parity depends on it.
            losses=["mal", "boxes"],
            alpha=0.75,
            gamma=1.5,
            num_classes=self.config.num_classes,
            reg_max=32,
            group_detr=3,
            distributed_normalize=distributed_normalize,
        ).to(self.device)

    def on_forward(self, imgs, targets, polygons=None):
        del polygons
        target_list = rows_to_targets(targets, self.device)
        outputs = self.model(imgs, targets=target_list)
        losses = self.criterion(outputs, target_list)
        result = {"total_loss": sum(losses.values())}
        result.update(losses)
        return result

    def get_loss_components(self, outputs):
        return {
            name: sum(
                float(v.detach())
                for k, v in outputs.items()
                if k == f"loss_{name}" or k.startswith(f"loss_{name}_")
            )
            for name in ("mal", "bbox", "kld")
        }


def rows_to_targets(rows: torch.Tensor, device) -> list[dict]:
    """``(B, N, 6)`` padded ``[class, cx, cy, w, h, a]`` rows to GTR targets."""
    targets = []
    for sample in rows:
        valid = (sample[:, 3] > 0) & (sample[:, 4] > 0)
        sample = sample[valid]
        targets.append(
            {
                "labels": sample[:, 0].long().to(device),
                "boxes": sample[:, 1:6].float().to(device),
            }
        )
    return targets

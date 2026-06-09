"""ECSegTrainer — native EC instance-segmentation fine-tuning (EXPERIMENTAL).

EC's seg model shares the detect backbone/encoder/decoder and adds a mask head.
The training recipe therefore reuses the EC detect recipe (AdamW, FlatCosine,
MAL + L1 + GIoU + FGL + DDF, EMA 0.9999, ImageNet-normalized inputs) and layers
a point-sampled BCE + Dice mask loss on top.

Unlike :class:`ECTrainer` (which subclasses ``DFINETrainer``), this trainer
subclasses :class:`BaseTrainer` directly so the base data path keeps the
polygon/mask 5-tuple intact through ``yolox_collate_fn`` (D-FINE's multi-scale
collate drops it). The polygon→mask rasterization reuses LibreYOLO's RF-DETR seg
transform — the same lineage as EC's mask head — to avoid a second copy.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Type

import torch

from ...training.config import ECSegConfig, TrainConfig
from ...training.scheduler import FlatCosineScheduler
from ...training.trainer import BaseTrainer
from ...data.augment.rfdetr import RFDETRSegPassThroughDataset, RFDETRSegTransform
from .seg_loss import ECSegCriterion, ECSegHungarianMatcher


class ECSegTrainer(BaseTrainer):
    """Trainer for EC segmentation models (experimental)."""

    artifact_model_families = ("ec",)
    # max GT instances per image carried as rasterized target masks.
    _MAX_MASK_LABELS = 100

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return ECSegConfig

    def get_model_family(self) -> str:
        return "ec"

    def get_model_tag(self) -> str:
        return f"EC-Seg-{self.config.size}"

    @property
    def effective_lr(self) -> float:
        return self.config.lr0

    def _setup_device(self) -> torch.device:
        """Permit MPS with the per-op CPU fallback (see ECTrainer)."""
        device = super()._setup_device()
        if device.type == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            import logging

            logging.getLogger(__name__).info(
                "EC seg training on MPS: enabling PYTORCH_ENABLE_MPS_FALLBACK=1 "
                "(deformable attention's grid_sample backward runs on CPU)."
            )
        return device

    def on_num_classes_resolved(self):
        """EC training is out of scope for DETR class-count hardening."""

    def create_transforms(self):
        preproc = RFDETRSegTransform(
            max_labels=self._MAX_MASK_LABELS,
            flip_prob=self.config.flip_prob,
            imgsz=self.config.imgsz,
            mask_downsample_ratio=self.config.mask_downsample_ratio,
            multi_scale=False,
            crop_resize_prob=getattr(self.config, "crop_resize_prob", 0.0),
        )
        return preproc, RFDETRSegPassThroughDataset

    def create_scheduler(self, iters_per_epoch: int):
        return FlatCosineScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            no_aug_epochs=self.config.no_aug_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def on_setup(self):
        matcher = ECSegHungarianMatcher(
            weight_dict={
                "cost_class": 2.0,
                "cost_bbox": 1.0,
                "cost_giou": 1.0,
                "cost_mask_ce": self.config.mask_ce_loss_weight,
                "cost_mask_dice": self.config.mask_dice_loss_weight,
            },
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
            mask_point_sample_ratio=self.config.mask_point_sample_ratio,
        )
        self.criterion = ECSegCriterion(
            matcher=matcher,
            weight_dict={
                "loss_mal": 2.0,
                "loss_bbox": 1.0,
                "loss_giou": 1.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
                "loss_mask_ce": self.config.mask_ce_loss_weight,
                "loss_mask_dice": self.config.mask_dice_loss_weight,
            },
            losses=["mal", "boxes", "local", "masks"],
            num_classes=self.config.num_classes,
            alpha=0.75,
            gamma=1.5,
            reg_max=32,
            mask_point_sample_ratio=self.config.mask_point_sample_ratio,
        ).to(self.device)

    def on_mosaic_disable(self):
        super().on_mosaic_disable()
        if self.ema_model is not None:
            decay = getattr(self.config, "ema_restart_decay", self.config.ema_decay)
            self.ema_model.set_decay(decay)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """AdamW with {backbone, head} x {wd, no-wd} groups (EC recipe)."""
        backbone_wd, backbone_no_wd, head_wd, head_no_wd = [], [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_norm_or_bias = (
                "norm" in name
                or ".bn." in name
                or "bias" in name
                or "lab.scale" in name
            )
            is_backbone = name.startswith("backbone.")
            if is_backbone and is_norm_or_bias:
                backbone_no_wd.append(p)
            elif is_backbone:
                backbone_wd.append(p)
            elif is_norm_or_bias:
                head_no_wd.append(p)
            else:
                head_wd.append(p)

        lr = self.effective_lr
        wd = self.config.weight_decay
        bb_mult = float(getattr(self.config, "backbone_lr_mult", 1.0))

        groups = []
        if head_wd:
            groups.append({"params": head_wd, "lr": lr, "weight_decay": wd, "lr_mult": 1.0})
        if head_no_wd:
            groups.append({"params": head_no_wd, "lr": lr, "weight_decay": 0.0, "lr_mult": 1.0})
        if backbone_wd:
            groups.append({"params": backbone_wd, "lr": lr * bb_mult, "weight_decay": wd, "lr_mult": bb_mult})
        if backbone_no_wd:
            groups.append({"params": backbone_no_wd, "lr": lr * bb_mult, "weight_decay": 0.0, "lr_mult": bb_mult})
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))

    def _scale_lr(self, base_lr: float, param_group: dict) -> float:
        return base_lr * float(param_group.get("lr_mult", 1.0))

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        def _sum_with_prefix(prefix: str) -> float:
            total = 0.0
            for k, v in outputs.items():
                if k == prefix or k.startswith(prefix + "_"):
                    total += v.item() if isinstance(v, torch.Tensor) else float(v)
            return total

        return {
            "mal": _sum_with_prefix("loss_mal"),
            "bbox": _sum_with_prefix("loss_bbox"),
            "giou": _sum_with_prefix("loss_giou"),
            "fgl": _sum_with_prefix("loss_fgl"),
            "ddf": _sum_with_prefix("loss_ddf"),
            "mask_ce": _sum_with_prefix("loss_mask_ce"),
            "mask_dice": _sum_with_prefix("loss_mask_dice"),
        }

    def on_forward(
        self, imgs: torch.Tensor, targets: torch.Tensor, polygons: Optional[List] = None
    ) -> Dict:
        """Translate padded ``(B, max_labels, 5)`` targets + stacked masks into
        the criterion's per-image ``{labels, boxes, masks}`` dict list."""
        B = targets.shape[0]
        H, W = imgs.shape[-2], imgs.shape[-1]
        scale = torch.tensor([W, H, W, H], device=targets.device, dtype=targets.dtype)

        masks_batch = (
            polygons.to(self.device, non_blocking=True)
            if isinstance(polygons, torch.Tensor)
            else None
        )

        target_list = []
        for b in range(B):
            t = targets[b]
            valid = (t[:, 3] > 0) & (t[:, 4] > 0)
            t_valid = t[valid]
            if t_valid.numel() == 0:
                entry = {
                    "labels": torch.zeros(0, dtype=torch.int64, device=self.device),
                    "boxes": torch.zeros(0, 4, dtype=torch.float32, device=self.device),
                }
                if masks_batch is not None:
                    mh, mw = masks_batch.shape[-2], masks_batch.shape[-1]
                    entry["masks"] = torch.zeros(
                        0, mh, mw, dtype=torch.bool, device=self.device
                    )
            else:
                entry = {
                    "labels": t_valid[:, 0].long(),
                    "boxes": (t_valid[:, 1:5] / scale).clamp(0.0, 1.0),
                }
                if masks_batch is not None:
                    entry["masks"] = masks_batch[b][valid].to(
                        device=self.device, dtype=torch.bool
                    )
            target_list.append(entry)

        outputs = self.model(imgs, targets=target_list)
        losses = self.criterion(outputs, target_list)
        total = sum(losses.values())
        result = {"total_loss": total}
        result.update(losses)
        return result

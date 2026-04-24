"""DFINETrainer — BaseTrainer subclass for native D-FINE training.

The integration trick is in ``on_forward``: LibreYOLO's training pipeline
yields targets as a ``(B, max_labels, 5)`` padded tensor, but D-FINE's
criterion wants ``list[dict{labels, boxes_cxcywh_normalized}]`` per image.
The trainer translates between the two without touching ``BaseTrainer``.
"""

from __future__ import annotations

from typing import Dict, Type

import torch
import torch.nn as nn

from ...training.config import DFINEConfig, TrainConfig
from ...training.scheduler import FlatCosineScheduler
from ...training.trainer import BaseTrainer
from .loss import DFINECriterion
from .matcher import HungarianMatcher
from .transforms import DFINEPassThroughDataset, DFINETrainTransform


class DFINETrainer(BaseTrainer):
    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return DFINEConfig

    def get_model_family(self) -> str:
        return "dfine"

    def get_model_tag(self) -> str:
        return f"DFINE-{self.config.size}"

    def create_transforms(self):
        preproc = DFINETrainTransform(max_labels=120, flip_prob=self.config.flip_prob)
        return preproc, DFINEPassThroughDataset

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

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        def _scalar(v):
            if isinstance(v, torch.Tensor):
                return v.item()
            return float(v)

        return {
            "vfl": _scalar(outputs.get("loss_vfl", 0)),
            "bbox": _scalar(outputs.get("loss_bbox", 0)),
            "giou": _scalar(outputs.get("loss_giou", 0)),
            "fgl": _scalar(outputs.get("loss_fgl", 0)),
            "ddf": _scalar(outputs.get("loss_ddf", 0)),
        }

    def on_setup(self):
        matcher = HungarianMatcher(
            weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        )
        self.criterion = DFINECriterion(
            matcher=matcher,
            weight_dict={
                "loss_vfl": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            losses=["vfl", "boxes", "local"],
            alpha=0.75,
            gamma=2.0,
            num_classes=self.config.num_classes,
            reg_max=32,
        ).to(self.device)

    def on_mosaic_disable(self):
        super().on_mosaic_disable()
        # D-FINE's "EMA restart": switch to a constant decay for the final phase.
        if self.ema_model is not None:
            decay = getattr(self.config, "ema_restart_decay", self.config.ema_decay)
            self.ema_model.set_decay(decay)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """AdamW with no-WD on norms / biases / LearnableAffine params.

        Backbone-LR multiplier is intentionally **not** applied in v1 — we share
        a single LR across all groups. Adding it later requires either a custom
        scheduler (returning per-group LRs) or an override of ``_train_epoch``.
        """
        no_wd_params, wd_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Match all norm-layer params:
            #   - ``.bn.`` (BatchNorm in HGNetv2/encoder convs)
            #   - ``.norm.`` (encoder lateral conv layers)
            #   - ``.norm`` followed by digit (transformer ``norm1`` / ``norm2`` /
            #     ``norm3`` in encoder + decoder layers — the bug-prone case
            #     because ``.norm.`` does NOT match ``norm1``)
            is_norm_or_bias = (
                ".bn." in name
                or ".norm." in name
                or ".norm1." in name
                or ".norm2." in name
                or ".norm3." in name
                or name.endswith(".norm1.weight")
                or name.endswith(".norm2.weight")
                or name.endswith(".norm3.weight")
                or name.endswith(".bias")
                or "lab.scale" in name
                or "lab.bias" in name
            )
            (no_wd_params if is_norm_or_bias else wd_params).append(p)

        lr = self.effective_lr
        wd = self.config.weight_decay
        param_groups = [
            {"params": wd_params, "lr": lr, "weight_decay": wd},
        ]
        if no_wd_params:
            param_groups.append({"params": no_wd_params, "lr": lr, "weight_decay": 0.0})

        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forward + loss in one go.

        Translates the ``(B, max_labels, 5)`` ``[class, cx, cy, w, h]`` pixel
        target tensor into D-FINE's per-image dict list with cxcywh-normalized
        boxes, then runs model + criterion.
        """
        B = targets.shape[0]
        H, W = self.config.imgsz, self.config.imgsz
        scale = torch.tensor([W, H, W, H], device=targets.device, dtype=targets.dtype)

        target_list = []
        for b in range(B):
            t = targets[b]
            # Padding rows are zero in all 5 columns; valid boxes have w>0 and h>0.
            valid = (t[:, 3] > 0) & (t[:, 4] > 0)
            t_valid = t[valid]
            if t_valid.numel() == 0:
                target_list.append(
                    {
                        "labels": torch.zeros(0, dtype=torch.int64, device=self.device),
                        "boxes": torch.zeros(0, 4, dtype=torch.float32, device=self.device),
                    }
                )
            else:
                target_list.append(
                    {
                        "labels": t_valid[:, 0].long(),
                        "boxes": (t_valid[:, 1:] / scale).clamp(0.0, 1.0),
                    }
                )

        outputs = self.model(imgs, targets=target_list)
        losses = self.criterion(outputs, target_list)
        total = sum(losses.values())

        return {
            "total_loss": total,
            "loss_vfl": losses.get("loss_vfl", torch.tensor(0.0)),
            "loss_bbox": losses.get("loss_bbox", torch.tensor(0.0)),
            "loss_giou": losses.get("loss_giou", torch.tensor(0.0)),
            "loss_fgl": losses.get("loss_fgl", torch.tensor(0.0)),
            "loss_ddf": losses.get("loss_ddf", torch.tensor(0.0)),
        }

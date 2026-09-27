"""YOLO-NAS-R (OBB) trainer."""

from __future__ import annotations

from typing import Dict, Type

import torch

from ...training.config import TrainConfig, YOLONASOBBConfig
from ...training.scheduler import CosineAnnealingScheduler
from ...training.trainer import BaseTrainer
from .obb_loss import YOLONASOBBLoss


class YOLONASOBBTrainer(BaseTrainer):
    """Rotated-box training for the YOLO-NAS family.

    Reuses the shared ``BaseTrainer`` OBB data path (``load_obb=True``) and
    the shared :class:`~libreyolo.validation.obb_validator.OBBValidator`; the
    family-specific parts are the rotated loss, the angle-aware transform and
    the OBB metric key.
    """

    artifact_model_families = ("yolonas",)
    # Rotated mAP, not the axis-aligned bbox default: OBBValidator reports
    # both keys and selecting on the bbox one would pick checkpoints by the
    # proxy box (landmine #23).
    best_metric_key = "metrics/mAP50-95(OBB)"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return YOLONASOBBConfig

    def get_model_family(self) -> str:
        return "yolonas"

    def get_model_tag(self) -> str:
        return f"YOLO-NAS-R-{self.config.size}"

    def create_transforms(self):
        from ...data.augment.yolonas import (
            YOLONASOBBDataset,
            YOLONASOBBTrainTransform,
        )

        preproc = YOLONASOBBTrainTransform(
            max_labels=int(getattr(self.config, "max_labels", 300)),
            flip_prob=self.config.flip_prob,
            hsv_prob=self.config.hsv_prob,
            flipud=float(getattr(self.config, "flipud", 0.0)),
        )
        return preproc, YOLONASOBBDataset

    def create_scheduler(self, iters_per_epoch: int):
        return CosineAnnealingScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def on_setup(self):
        self.loss_fn = YOLONASOBBLoss(
            num_classes=self.config.num_classes,
            classification_loss_weight=self.config.classification_loss_weight,
            iou_loss_weight=self.config.iou_loss_weight,
            dfl_loss_weight=self.config.dfl_loss_weight,
            assigner_topk=self.config.bbox_assigner_topk,
            assigner_alpha=self.config.bbox_assigned_alpha,
            assigner_beta=self.config.bbox_assigned_beta,
            use_varifocal_loss=self.config.use_varifocal_loss,
        ).to(self.device)

    def validate_validation_loss_config(self) -> None:
        if getattr(self.config, "val_loss", False):
            raise ValueError(
                "val_loss=True is not supported for YOLO-NAS OBB training."
            )

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        def _scalar(v):
            return v.item() if isinstance(v, torch.Tensor) else v

        return {
            "cls": _scalar(outputs.get("cls", 0)),
            "iou": _scalar(outputs.get("iou", 0)),
            "dfl": _scalar(outputs.get("dfl", 0)),
        }

    def on_forward(
        self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None
    ) -> Dict:
        del polygons
        model_outputs = self.model(imgs)
        total_loss, log_losses = self.loss_fn(model_outputs, targets)
        return {
            "total_loss": total_loss,
            "cls": log_losses[0],
            "iou": log_losses[1],
            "dfl": log_losses[2],
        }

    def cuda_graph_train_spec(self):
        # The rotated loss runs a per-image Python loop over the assigner, so
        # the capture boundary the detect trainer relies on does not hold.
        return None

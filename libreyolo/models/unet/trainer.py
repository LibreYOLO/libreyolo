"""U-Net trainer: Cityscapes-style SGD + polynomial decay on the semantic path."""

from __future__ import annotations

from typing import Dict, Type

import torch

from ...training.config import TrainConfig, UNetConfig
from ...training.scheduler import PolyLRScheduler
from ...training.trainer import BaseTrainer
from ..base.semantic_cuda_graph import SemanticLogitsCudaGraphMixin
from ..base.semantic_validation_loss import SemanticValidationLossMixin
from .loss import IGNORE_INDEX, UNetLoss
from .nn import SIZE_CONFIGS


class UNetTrainer(SemanticLogitsCudaGraphMixin, SemanticValidationLossMixin, BaseTrainer):
    """Trainer for the LibreUNet semantic-segmentation family."""

    best_metric_key: str = "metrics/mIoU"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return UNetConfig

    def get_model_family(self) -> str:
        return "unet"

    def get_model_tag(self) -> str:
        return f"LibreUNet-{self.config.size}"

    def create_transforms(self):
        raise NotImplementedError(
            "LibreUNet is semantic-only; create_transforms() is never called for "
            "task='semantic' (BaseTrainer._setup_data routes to _setup_semantic_data)."
        )

    def create_scheduler(self, iters_per_epoch: int):
        return PolyLRScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            power=float(getattr(self.config, "poly_power", 0.9)),
            min_lr_ratio=self.config.min_lr_ratio,
        )

    @property
    def criterion(self) -> UNetLoss:
        cached = getattr(self, "_criterion", None)
        if cached is None:
            self._criterion = UNetLoss(
                ignore_index=IGNORE_INDEX,
                aux_weight=float(getattr(self.config, "aux_weight", 0.4)),
            ).to(self.device)
        return self._criterion

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        del polygons
        outputs = self.wrapper_model.model(imgs)
        components = self.criterion(outputs, targets)
        result = dict(components)
        result["total_loss"] = components["loss"]
        return result

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        return {
            key: float(value.item()) if torch.is_tensor(value) else float(value)
            for key, value in outputs.items()
            if key != "total_loss"
        }


__all__ = ["SIZE_CONFIGS", "UNetTrainer"]

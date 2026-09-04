"""U-Net trainer: Cityscapes-style SGD + polynomial decay on the semantic path."""

from __future__ import annotations

from typing import Any, Dict, Type

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
        # self.model is the trainer-owned module (SyncBN / DDP wrapped under
        # multi-GPU); the raw wrapper_model.model would skip gradient sync.
        outputs = self.model(imgs)
        components = self.criterion(outputs, targets)
        result = dict(components)
        result["total_loss"] = components["loss"]
        return result

    def _checkpoint_extra_metadata(self) -> Dict[str, Any]:
        # A fine-tune started from the Cityscapes checkpoint is a derivative
        # work and inherits its NON-COMMERCIAL term; carry the license fields
        # into best.pt / last.pt so reloading them keeps the restriction.
        extra = dict(super()._checkpoint_extra_metadata())
        wrapper = self.wrapper_model
        license_name = getattr(wrapper, "weight_license", None)
        if license_name:
            extra["weight_license"] = license_name
            for key in ("weight_license_url", "weight_dataset", "weight_commercial_use"):
                value = getattr(wrapper, key, None)
                if value is not None:
                    extra[key] = value
        return extra

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        return {
            key: float(value.item()) if torch.is_tensor(value) else float(value)
            for key, value in outputs.items()
            if key != "total_loss"
        }


__all__ = ["SIZE_CONFIGS", "UNetTrainer"]

"""LibreEfficientNetV2 classification trainer.

Thin :class:`BaseTrainer` subclass. The data, validation, and head-rebuild
machinery is the shared classification path in ``BaseTrainer``
(``_setup_classify_data`` / ``_run_classify_validation``); this class only
supplies the cross-entropy forward and a cosine LR schedule.
"""

from __future__ import annotations

from typing import Dict, Type

import torch

from ...training.classification import classification_loss
from ...training.config import TrainConfig
from ...training.scheduler import WarmupCosineScheduler
from ...training.trainer import BaseTrainer
from ..base.classify_cuda_graph import ClassifyCudaGraphMixin
from ..base.classify_validation_loss import ClassifyValidationLossMixin
from .config import EfficientNetV2Config


class EfficientNetV2Trainer(ClassifyCudaGraphMixin, ClassifyValidationLossMixin, BaseTrainer):
    """Cross-entropy fine-tuning trainer for EfficientNetV2 classification."""

    # Non-detect task: select the best checkpoint by top-1 accuracy.
    best_metric_key = "metrics/accuracy_top1"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return EfficientNetV2Config

    def get_model_family(self) -> str:
        return "efficientnetv2"

    def get_model_tag(self) -> str:
        return f"EfficientNetV2-{self.config.size}"

    def create_transforms(self):
        # Classification uses the ImageFolder pipeline in ``_setup_classify_data``,
        # so the detection transform factory is never invoked for this task.
        return None, None

    def create_scheduler(self, iters_per_epoch: int):
        return WarmupCosineScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            plateau_epochs=self.config.no_aug_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        logits = self.model(imgs)
        loss = classification_loss(
            logits, targets, getattr(self, "class_weights", None)
        )
        return {"total_loss": loss, "loss_ce": loss.detach()}

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        return {"ce": float(outputs.get("loss_ce", outputs["total_loss"]))}

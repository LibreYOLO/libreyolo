"""DINOv2 trainer — inherits RF-DETR's trainer with dinov2 family metadata."""

from __future__ import annotations

from typing import Type

from ...training.config import TrainConfig
from ..base.semantic_validation_loss import SemanticValidationLossMixin
from ..rfdetr.trainer import RFDETRTrainer
from .config import DINOv2Config


class DINOv2Trainer(SemanticValidationLossMixin, RFDETRTrainer):
    """Trainer for the LibreDINOv2 semantic-segmentation family.

    Inherits all training logic from RFDETRTrainer (which handles the
    semantic task path through its ``on_setup``, ``on_forward``, and
    ``get_loss_components`` semantic branches). Only the model-family
    metadata and config class are overridden so saved checkpoints carry
    ``model_family="dinov2"`` instead of ``"rfdetr"``.
    """

    artifact_model_families = ("dinov2",)

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return DINOv2Config

    def get_model_family(self) -> str:
        return "dinov2"

    def get_model_tag(self) -> str:
        return f"LibreDINOv2-{self.config.size}"

"""DDRNet trainer: the paper's Cityscapes recipe on the shared semantic path.

SGD + polynomial decay (power 0.9), OHEM cross-entropy on the main head plus
0.4x on the auxiliary head, SyncBN under DDP. ``BaseTrainer._setup_data``
routes ``task="semantic"`` to ``_setup_semantic_data``, so
``create_transforms`` is never called.
"""

from __future__ import annotations

from typing import Any

import torch

from ...training.config import DDRNetConfig, TrainConfig
from ...training.scheduler import PolyLRScheduler
from ...training.trainer import BaseTrainer
from ..base.semantic_cuda_graph import SemanticLogitsCudaGraphMixin
from ..base.semantic_validation_loss import SemanticValidationLossMixin
from .loss import IGNORE_INDEX, DDRNetLoss
from .nn import SIZE_CONFIGS


def _input_size_hw(input_size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    return int(input_size[0]), int(input_size[1])


class DDRNetTrainer(
    SemanticLogitsCudaGraphMixin, SemanticValidationLossMixin, BaseTrainer
):
    """Trainer for the LibreDDRNet semantic-segmentation family."""

    best_metric_key: str = "metrics/mIoU"

    @classmethod
    def _config_class(cls) -> type[TrainConfig]:
        return DDRNetConfig

    def get_model_family(self) -> str:
        return "ddrnet"

    def get_model_tag(self) -> str:
        return f"LibreDDRNet-{self.config.size}"

    def create_transforms(self):
        raise NotImplementedError(
            "LibreDDRNet is semantic-only; create_transforms() is never called for "
            "task='semantic' (BaseTrainer._setup_data routes to _setup_semantic_data)."
        )

    def create_scheduler(self, iters_per_epoch: int):
        scheduler_name = str(self.config.scheduler).lower()
        if scheduler_name != "poly":
            raise ValueError(
                "DDRNet trains with the source polynomial schedule; got "
                f"scheduler={self.config.scheduler!r}."
            )
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
    def criterion(self) -> DDRNetLoss:
        cached = getattr(self, "_criterion", None)
        if cached is None:
            self._criterion = DDRNetLoss(
                ignore_index=IGNORE_INDEX,
                aux_weight=float(getattr(self.config, "aux_weight", 0.4)),
                ohem=bool(getattr(self.config, "ohem", True)),
                ohem_thresh=float(getattr(self.config, "ohem_thresh", 0.9)),
                ohem_min_kept=int(getattr(self.config, "ohem_min_kept", 131072)),
            ).to(self.device)
        return self._criterion

    def validation_pixel_loss(self):
        # Validation runs in eval mode, so only the main head is scored: the
        # number is comparable to the training ``loss_ce`` component, not the
        # aux-weighted total. ``ohem_min_kept`` counts pixels per batch, so
        # the OHEM validation loss depends a little on val batch and imgsz.
        if not bool(getattr(self.config, "ohem", True)):
            return None
        return self.criterion.criterion

    def on_forward(
        self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None
    ) -> dict:
        del polygons
        # self.model is the trainer-owned module (SyncBN / DDP wrapped under
        # multi-GPU); the raw wrapper_model.model would skip gradient sync.
        outputs = self.model(imgs)
        components = self.criterion(outputs, targets)
        result = dict(components)
        result["total_loss"] = components["loss"]
        return result

    def _checkpoint_extra_metadata(self) -> dict[str, Any]:
        # Carry the source checkpoint's weight terms into best.pt / last.pt so
        # a fine-tune keeps the terms of the weights it started from.
        extra = dict(super()._checkpoint_extra_metadata())
        extra.update(self.wrapper_model._checkpoint_metadata())
        train_h, train_w = _input_size_hw(self.config.imgsz)
        extra.update(train_imgsz_h=train_h, train_imgsz_w=train_w)
        return extra

    def get_loss_components(self, outputs: dict) -> dict[str, float]:
        return {
            key: float(value.item()) if torch.is_tensor(value) else float(value)
            for key, value in outputs.items()
            if key != "total_loss"
        }


__all__ = ["SIZE_CONFIGS", "DDRNetTrainer"]

"""LingBot-Vision trainer — plugs into the shared semantic BaseTrainer path.

Follows the SegFormer pattern: a plain encoder + dense head, so the trainer
implements only what ``task="semantic"`` actually requires. The default recipe
is the upstream report's linear probe — backbone frozen, only the 1x1 head
trains — because that is what the published LingBot-Vision evaluation protocol
does and what the LibreYOLO-hosted weights were produced with. Set
``freeze_backbone=False`` (config) for a full fine-tune.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Type

import torch

from ...training.config import LingBotVisionConfig, TrainConfig
from ...training.distributed import is_main_process, unwrap_model
from ...training.optim import build_optimizer
from ...training.scheduler import FlatCosineScheduler, LinearLRScheduler
from ...training.trainer import BaseTrainer
from ..base.semantic_cuda_graph import SemanticLogitsCudaGraphMixin
from ..base.semantic_validation_loss import SemanticValidationLossMixin

logger = logging.getLogger(__name__)


class LingBotVisionTrainer(
    SemanticLogitsCudaGraphMixin, SemanticValidationLossMixin, BaseTrainer
):
    """Trainer for the LibreLingBotVision semantic-segmentation family."""

    best_metric_key: str = "metrics/mIoU"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return LingBotVisionConfig

    def get_model_family(self) -> str:
        return "lingbotvision"

    def get_model_tag(self) -> str:
        return f"LibreLingBotVision-{self.config.size}"

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """AdamW; with ``freeze_backbone`` (default) only the head trains.

        LayerNorm/bias params (``ndim <= 1``) get ``weight_decay=0``; every
        other weight gets ``config.weight_decay``. Params already frozen via
        the generic freeze config are skipped as usual.
        """
        base_lr = self.effective_lr
        wd = self.config.weight_decay
        raw = unwrap_model(self.model)

        if bool(getattr(self.config, "freeze_backbone", True)):
            for param in raw.backbone.parameters():
                param.requires_grad_(False)

        buckets: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}
        for name, param in raw.named_parameters():
            if not param.requires_grad:
                continue
            no_decay = param.ndim <= 1
            group_wd = 0.0 if no_decay else wd
            buckets.setdefault((1.0, group_wd), []).append(param)

        if not buckets:
            raise ValueError(
                "No trainable parameters remain for the LingBot-Vision optimizer; "
                "check the freeze configuration."
            )

        param_groups = [
            {"params": params, "lr": base_lr, "weight_decay": group_wd}
            for (_, group_wd), params in buckets.items()
        ]
        optimizer = build_optimizer(torch.optim.AdamW, param_groups, lr=base_lr)
        if is_main_process():
            n_trainable = sum(len(g["params"]) for g in param_groups)
            logger.info(
                "LingBot-Vision optimizer: AdamW, lr=%s, freeze_backbone=%s, trainable tensors=%d",
                base_lr,
                bool(getattr(self.config, "freeze_backbone", True)),
                n_trainable,
            )
        return optimizer

    def create_transforms(self):
        raise NotImplementedError(
            "LingBot-Vision is semantic-only; create_transforms() is never "
            "called for task='semantic' (BaseTrainer._setup_data routes "
            "straight to _setup_semantic_data)."
        )

    def create_scheduler(self, iters_per_epoch: int):
        scheduler_name = str(self.config.scheduler).lower()
        if scheduler_name == "linear":
            return LinearLRScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        if scheduler_name in ("cosine", "flat_cosine", "cos"):
            return FlatCosineScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                no_aug_epochs=getattr(self.config, "no_aug_epochs", 0),
                min_lr_ratio=self.config.min_lr_ratio,
            )
        raise ValueError(f"Unknown LingBot-Vision scheduler: {self.config.scheduler!r}")

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        return self.model(imgs, targets=targets)

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        value = outputs.get("sem", 0)
        return {"sem": value.item() if isinstance(value, torch.Tensor) else float(value)}


__all__ = ["LingBotVisionTrainer"]

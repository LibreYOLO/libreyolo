"""PP-LiteSeg trainer — the source Cityscapes recipe on the semantic path.

Recipe (SuperGradients ``cityscapes_default_train_params`` +
``cityscapes_pplite_seg{50,75}``): SGD momentum 0.9, weight decay 5e-4 with
BatchNorm and bias params exempt, two explicit parameter groups (STDC backbone
at ``lr0``, everything else at ``lr0 * head_lr_mult``), polynomial decay, 10
warmup epochs, EMA, mixed precision off, and Dice + cross-entropy + edge loss
over the main head and all three auxiliary heads.

``BaseTrainer._setup_data`` routes ``task="semantic"`` straight to
``_setup_semantic_data``, so ``create_transforms`` is never called.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Type

import torch

from ...training.config import PPLiteSegConfig, TrainConfig
from ...training.distributed import is_main_process, unwrap_model
from ...training.optim import build_optimizer
from ...training.scheduler import PolyLRScheduler
from ...training.trainer import BaseTrainer
from ..base.semantic_cuda_graph import SemanticLogitsCudaGraphMixin
from ..base.semantic_validation_loss import SemanticValidationLossMixin
from .loss import IGNORE_INDEX, PPLiteSegLoss

logger = logging.getLogger(__name__)

_BACKBONE_PREFIX = "encoder.backbone."


class PPLiteSegTrainer(SemanticLogitsCudaGraphMixin, SemanticValidationLossMixin, BaseTrainer):
    """Trainer for the LibrePPLiteSeg semantic-segmentation family."""

    best_metric_key: str = "metrics/mIoU"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return PPLiteSegConfig

    def get_model_family(self) -> str:
        return "ppliteseg"

    def get_model_tag(self) -> str:
        return f"LibrePPLiteSeg-{self.config.size}"

    # ------------------------------------------------------------------
    # Optimizer / schedule
    # ------------------------------------------------------------------

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """SGD with the source's two LR groups and BN/bias weight-decay exemption.

        The source expresses the split as ``initial_lr: {"encoder.backbone":
        0.01, default: 0.1}``. Here it is two explicit parameter groups carrying
        an ``lr_mult``, so ``_scale_lr`` reapplies the ratio on every scheduler
        step instead of re-matching parameter names each iteration.
        """
        base_lr = self.effective_lr
        wd = self.config.weight_decay
        head_lr_mult = float(getattr(self.config, "head_lr_mult", 10.0))
        zero_wd_on_bn_bias = bool(getattr(self.config, "zero_weight_decay_on_bias_and_bn", True))
        raw = unwrap_model(self.model)

        buckets: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}
        for name, param in raw.named_parameters():
            if not param.requires_grad:
                continue
            lr_mult = 1.0 if name.startswith(_BACKBONE_PREFIX) else head_lr_mult
            no_decay = zero_wd_on_bn_bias and param.ndim <= 1
            group_wd = 0.0 if no_decay else wd
            buckets.setdefault((lr_mult, group_wd), []).append(param)

        if not buckets:
            raise ValueError(
                "No trainable parameters remain for the PP-LiteSeg optimizer; "
                "check the freeze configuration."
            )

        param_groups = [
            {
                "params": params,
                "lr": base_lr * lr_mult,
                "weight_decay": group_wd,
                "lr_mult": lr_mult,
            }
            for (lr_mult, group_wd), params in buckets.items()
        ]
        optimizer = build_optimizer(
            torch.optim.SGD,
            param_groups,
            lr=base_lr,
            momentum=float(getattr(self.config, "momentum", 0.9)),
            nesterov=bool(getattr(self.config, "nesterov", False)),
        )
        if is_main_process():
            logger.info("PP-LiteSeg optimizer: SGD, backbone base lr=%s", base_lr)
            for (lr_mult, group_wd), params in buckets.items():
                logger.info(
                    "  - group: lr_mult=%s (lr=%s), weight_decay=%s, params=%d",
                    lr_mult,
                    base_lr * lr_mult,
                    group_wd,
                    len(params),
                )
        return optimizer

    def _scale_lr(self, base_lr: float, param_group: dict) -> float:
        """Keep the non-backbone 10x multiplier through warmup and poly decay."""
        return base_lr * float(param_group.get("lr_mult", 1.0))

    def create_transforms(self):
        raise NotImplementedError(
            "PP-LiteSeg is semantic-only; create_transforms() is never called for "
            "task='semantic' (BaseTrainer._setup_data routes straight to "
            "_setup_semantic_data)."
        )

    def create_scheduler(self, iters_per_epoch: int):
        scheduler_name = str(self.config.scheduler).lower()
        if scheduler_name != "poly":
            raise ValueError(
                f"PP-LiteSeg trains with the source polynomial schedule; got "
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

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @property
    def criterion(self) -> PPLiteSegLoss:
        num_classes = int(self.num_classes or self.config.num_classes)
        cached = getattr(self, "_criterion", None)
        if cached is None or cached.num_classes != num_classes:
            self._criterion = PPLiteSegLoss(
                num_classes=num_classes,
                edge_kernel=int(getattr(self.config, "edge_kernel", 5)),
                ignore_index=IGNORE_INDEX,
            ).to(self.device)
        return self._criterion

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        outputs = self.model(imgs)
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


__all__ = ["PPLiteSegTrainer"]

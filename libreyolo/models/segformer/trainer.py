"""SegFormer trainer — plugs into the shared semantic BaseTrainer path.

Unlike EoMT/PIDNet (inference-only) or DINOv2 (inherits RF-DETR's DETR-decoder
trainer machinery it doesn't need), SegFormer has no query decoder, matcher,
or NestedTensor plumbing — it is a plain encoder + dense head. This trainer
implements only what ``task="semantic"`` actually requires:
``BaseTrainer._setup_data`` dispatches straight to ``_setup_semantic_data``
and ``_run_semantic_validation`` without ever calling ``create_transforms``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Type

import torch

from ...training.config import SegformerConfig, TrainConfig
from ...training.distributed import is_main_process, unwrap_model
from ...training.optim import build_optimizer
from ...training.scheduler import FlatCosineScheduler, LinearLRScheduler
from ...training.trainer import BaseTrainer
from ..base.semantic_cuda_graph import SemanticLogitsCudaGraphMixin
from ..base.semantic_validation_loss import SemanticValidationLossMixin

logger = logging.getLogger(__name__)


class SegformerTrainer(
    SemanticLogitsCudaGraphMixin, SemanticValidationLossMixin, BaseTrainer
):
    """Trainer for the LibreSegformer semantic-segmentation family."""

    best_metric_key: str = "metrics/mIoU"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return SegformerConfig

    def get_model_family(self) -> str:
        return "segformer"

    def get_model_tag(self) -> str:
        return f"LibreSegformer-{self.config.size}"

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """AdamW with the SegFormer paramwise recipe (mmsegmentation ADE20K).

        - decode head parameters train at ``config.head_lr_mult`` (default 10)
          over the backbone base LR (``config.lr0``); ``_scale_lr`` re-applies
          the multiplier on every scheduler step so the ratio holds through
          warmup and decay. Set ``head_lr_mult=1.0`` for a uniform LR.
        - LayerNorm/bias params (``ndim <= 1``) and the Mix-FFN positional
          depthwise conv (``pos_block``, name contains ``dwconv``) get
          ``weight_decay=0``; every other weight gets ``config.weight_decay``.

        Frozen params (``requires_grad=False``) are skipped, so layer-freeze
        config still works.
        """
        base_lr = self.effective_lr
        wd = self.config.weight_decay
        head_lr_mult = float(getattr(self.config, "head_lr_mult", 10.0))
        raw = unwrap_model(self.model)

        # Bucket by (lr_mult, weight_decay) so at most four param groups form
        # (two when head_lr_mult == 1.0, i.e. a uniform LR).
        buckets: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}
        for name, param in raw.named_parameters():
            if not param.requires_grad:
                continue
            is_head = name.startswith("decode_head.")
            no_decay = param.ndim <= 1 or "dwconv" in name
            lr_mult = head_lr_mult if is_head else 1.0
            group_wd = 0.0 if no_decay else wd
            buckets.setdefault((lr_mult, group_wd), []).append(param)

        if not buckets:
            raise ValueError(
                "No trainable parameters remain for the SegFormer optimizer; "
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

        optimizer = build_optimizer(torch.optim.AdamW, param_groups, lr=base_lr)
        if is_main_process():
            logger.info("SegFormer optimizer: AdamW, backbone base lr=%s", base_lr)
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
        """Honor the per-group ``lr_mult`` set in ``_setup_optimizer`` (head=10x).

        The scheduler calls this for every param group on every step, so the
        decode-head 10x multiplier is preserved through warmup and decay.
        """
        return base_lr * float(param_group.get("lr_mult", 1.0))

    def create_transforms(self):
        raise NotImplementedError(
            "SegFormer is semantic-only; create_transforms() is never called "
            "for task='semantic' (BaseTrainer._setup_data routes straight to "
            "_setup_semantic_data)."
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
        raise ValueError(f"Unknown SegFormer scheduler: {self.config.scheduler!r}")

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        return self.model(imgs, targets=targets)

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        value = outputs.get("sem", 0)
        return {"sem": value.item() if isinstance(value, torch.Tensor) else float(value)}


__all__ = ["SegformerTrainer"]

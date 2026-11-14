"""Training configuration for native LibreRFDETR."""

import math
from dataclasses import dataclass
from typing import List

from libreyolo.training.config import TrainConfig


@dataclass(kw_only=True)
class RFDETRConfig(TrainConfig):
    """CLI-visible RF-DETR fine-tuning defaults."""

    epochs: int = 100
    batch: int = 4
    nbs: int | None = 16
    lr0: float = 1e-4
    device: str = "auto"

    workers: int = 0
    weight_decay: float = 1e-4
    eval_interval: int = 1
    warmup_epochs: int = 0
    warmup_lr_start: float = 1e-6
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.0
    lr_drop: int = 100

    ema: bool = True
    ema_decay: float = 0.993
    ema_tau: int = 100
    seed: int | None = None

    patience: int = 0
    optimizer: str = "adamw"
    matcher_backend: str = "auto"
    # None resolves from model arguments, then is recorded in the run config.
    lr_encoder: float | None = None
    lr_vit_layer_decay: float | None = None
    lr_component_decay: float | None = None
    focal_alpha: float | None = None
    cls_loss_coef: float | None = None
    scheduler: str = "step"
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    hsv_prob: float = 0.0
    degrees: float = 0.0
    translate: float = 0.0
    shear: float = 0.0
    multi_scale: bool = True
    expanded_scales: bool = True
    do_random_resize_via_padding: bool = False
    crop_resize_prob: float = 0.5
    # Copy-paste instance augmentation (segmentation task only). ``copy_paste``
    # is the per-sample probability (0 disables it). The pass-through pipeline
    # only supports the same-sample "flip" source, so ``copy_paste_mode`` other
    # than "flip" falls back to it with a warning.
    copy_paste: float = 0.0
    copy_paste_mode: str = "flip"
    amp: bool = True
    backbone_lr_mult: float = 0.1
    clip_max_norm: float = 0.1

    num_keypoints: int = 17
    keypoint_dim: int = 3
    oks_sigmas: List[float] | None = None
    # GroupPose keypoint loss coefficients (ported from RF-DETR v1.8.0
    # KeypointTrainConfig; all default 1.0).
    keypoint_l1_loss_coef: float = 1.0
    keypoint_findable_loss_coef: float = 1.0
    keypoint_visible_loss_coef: float = 1.0
    keypoint_nll_loss_coef: float = 1.0
    # None retains each existing task's effective loader defaults.
    pin_memory: bool | None = None
    prefetch_factor: int | None = None
    persistent_workers: bool | None = None
    decode_scale: int = 1

    name: str = "rfdetr_exp"

    def __post_init__(self):
        super().__post_init__()
        if self.matcher_backend not in ("auto", "scipy"):
            raise ValueError("matcher_backend must be 'auto' or 'scipy'")
        if self.focal_alpha is not None and not 0 <= self.focal_alpha <= 1:
            raise ValueError("focal_alpha must be between 0 and 1")
        for name in (
            "lr_encoder",
            "lr_vit_layer_decay",
            "lr_component_decay",
            "cls_loss_coef",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be nonnegative")
        if self.prefetch_factor is not None and self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive")

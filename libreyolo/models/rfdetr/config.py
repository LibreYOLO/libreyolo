"""Training configuration for native LibreRFDETR."""

from dataclasses import dataclass
from typing import List

from libreyolo.training.config import TrainConfig


def normalize_matcher_backend(value: str) -> str:
    """Validate the opt-in RF-DETR Hungarian-assignment implementation."""
    if isinstance(value, str):
        backend = value.strip().lower()
        if backend in ("scipy", "auto", "torch"):
            return backend
    raise ValueError("matcher_backend must be 'scipy', 'auto', or 'torch'")


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
    # Hungarian assignment: preserve SciPy by default. "auto" opts into the
    # optional torch-linear-assignment provider when eligible; "torch" requests it
    # explicitly. The criterion remains eager with compile enabled.
    matcher_backend: str = "scipy"

    num_keypoints: int = 17
    keypoint_dim: int = 3
    oks_sigmas: List[float] | None = None
    # GroupPose keypoint loss coefficients (ported from RF-DETR v1.8.0
    # KeypointTrainConfig; all default 1.0).
    keypoint_l1_loss_coef: float = 1.0
    keypoint_findable_loss_coef: float = 1.0
    keypoint_visible_loss_coef: float = 1.0
    keypoint_nll_loss_coef: float = 1.0
    pin_memory: bool = False
    prefetch_factor: int = 1
    persistent_workers: bool = True
    decode_scale: int = 1

    name: str = "rfdetr_exp"

    def __post_init__(self):
        super().__post_init__()
        self.matcher_backend = normalize_matcher_backend(self.matcher_backend)

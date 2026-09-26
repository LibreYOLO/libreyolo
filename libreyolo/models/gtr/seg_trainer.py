"""GTR instance-segmentation fine-tuning.

Reuses the EC seg data path (square resize, polygon rasterization, ImageNet
normalization; no Mosaic/MixUp, which have no mask support here) and swaps in
the GTR recipe: grouped matching (``group_detr=3``), MAL/box/FGL/DDF plus
point-sampled mask BCE/Dice with the upstream ``gtrseg_base.yml`` weights, the
GTR schedule, and the size-specific backbone LR multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ec.seg_trainer import ECSegTrainer
from .config import GTRConfig
from .criterion import GTRCriterion
from .matcher import HungarianMatcher
from .scheduler import GTRScheduler
from .seg import SEG_MASK_DOWNSAMPLE_RATIO


@dataclass
class GTRSegConfig(GTRConfig):
    # The seg data path is a per-sample passthrough without Mosaic/MixUp.
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    crop_resize_prob: float = 0.0
    mask_ce_loss_weight: float = 5.0
    mask_dice_loss_weight: float = 5.0
    mask_point_sample_ratio: int = 16
    mask_downsample_ratio: int = SEG_MASK_DOWNSAMPLE_RATIO
    name: str = "gtr_seg_exp"

    def __post_init__(self):
        super().__post_init__()
        if self.mosaic_prob or self.mixup_prob:
            raise ValueError("GTR segment training does not support Mosaic/MixUp")
        if self.mask_downsample_ratio != SEG_MASK_DOWNSAMPLE_RATIO:
            raise ValueError(
                f"GTR mask_downsample_ratio is fixed at {SEG_MASK_DOWNSAMPLE_RATIO}"
            )


class GTRSegTrainer(ECSegTrainer):
    artifact_model_families = ("gtr",)
    supports_lora = True

    @classmethod
    def _config_class(cls):
        return GTRSegConfig

    def get_model_family(self):
        return "gtr"

    def get_model_tag(self):
        return f"GTR-Seg-{self.config.size}"

    def create_scheduler(self, iters_per_epoch):
        return GTRScheduler(self.effective_lr, iters_per_epoch, self.config)

    def preserve_freeze_param(self, name, param):
        if not self.config.lora:
            return False
        from ...training.lora import is_lora_parameter_name

        return is_lora_parameter_name(name)

    def on_setup(self):
        if self.config.lora:
            from ...training.lora import apply_lora_to_gtr

            apply_lora_to_gtr(self.model)
        self.criterion = self.build_criterion()

    def build_criterion(self):
        cfg = self.config
        matcher = HungarianMatcher(
            weight_dict={
                "cost_class": 2.0,
                "cost_bbox": 1.0,
                "cost_giou": 1.0,
                "cost_mask_ce": cfg.mask_ce_loss_weight,
                "cost_mask_dice": cfg.mask_dice_loss_weight,
            },
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
            mask_point_sample_ratio=cfg.mask_point_sample_ratio,
        )
        return GTRCriterion(
            matcher=matcher,
            weight_dict={
                "loss_mal": 2.0,
                "loss_bbox": 1.0,
                "loss_giou": 1.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
                "loss_mask_ce": cfg.mask_ce_loss_weight,
                "loss_mask_dice": cfg.mask_dice_loss_weight,
            },
            losses=["mal", "boxes", "local", "masks"],
            num_classes=cfg.num_classes,
            alpha=0.75,
            gamma=1.5,
            reg_max=32,
            group_detr=3,
            mask_point_sample_ratio=cfg.mask_point_sample_ratio,
        ).to(self.device)

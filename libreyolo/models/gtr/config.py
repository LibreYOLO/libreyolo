"""GTR fine-tuning defaults, from the pinned upstream COCO recipes.

The portable augmentation pipeline remains deliberately smaller than upstream.
"""

from dataclasses import dataclass

from ...training.config import DFINEConfig


@dataclass
class GTRConfig(DFINEConfig):
    epochs: int = 30
    eval_interval: int = 1
    batch: int = 4
    lr0: float = 5e-4
    weight_decay: float | None = None
    backbone_lr_mult: float | None = None
    amp: bool = False
    multi_scale: bool = False
    min_lr_ratio: float = 0.5
    warmup_iters: int = 2000
    warmup_epochs: float | None = None
    warmup_lr_start: float = 0.0
    flat_epochs: int = 6
    no_aug_epochs: int = 2
    aug_stop_epoch_ratio: float = 28 / 30
    name: str = "gtr_exp"

    def __post_init__(self):
        super().__post_init__()
        if self.size not in ("s", "m", "l", "x"):
            raise ValueError(f"Unknown GTR size: {self.size!r}")
        if str(self.optimizer).lower() != "adamw":
            raise ValueError("GTR currently supports optimizer='adamw' only")
        if self.scheduler != "flat_cosine":
            raise ValueError("GTR currently supports scheduler='flat_cosine' only")
        if self.weight_decay is None:
            self.weight_decay = 1e-4 if self.size in ("s", "m") else 1.25e-4
        if self.backbone_lr_mult is None:
            self.backbone_lr_mult = {"s": 0.03, "m": 0.03, "l": 0.005, "x": 0.004}[
                self.size
            ]

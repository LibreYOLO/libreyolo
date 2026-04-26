"""Training configuration for RF-DETR.

RF-DETR delegates training to the upstream ``rfdetr`` package, but the CLI still
needs a family-local source of truth so dry-runs, cfg output, progress messages,
and adapter kwargs agree.
"""

from dataclasses import dataclass

from libreyolo.training.config import TrainConfig


@dataclass(kw_only=True)
class RFDETRConfig(TrainConfig):
    """CLI-visible RF-DETR training defaults matching the upstream adapter."""

    epochs: int = 100
    batch: int = 4
    lr0: float = 1e-4
    device: str | None = None

    workers: int = 2
    weight_decay: float = 1e-4
    eval_interval: int = 1
    warmup_epochs: int = 0

    ema: bool = True
    ema_decay: float = 0.993
    seed: int | None = None

    # RF-DETR early stopping is off by default. The CLI maps patience > 0 to
    # upstream early_stopping=True when the user opts in.
    patience: int = 0

    name: str = "rfdetr_exp"

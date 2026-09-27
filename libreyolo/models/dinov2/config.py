"""Training configuration for LibreDINOv2.

``DINOv2Trainer`` inherits all training logic from ``RFDETRTrainer`` (optimizer,
scheduler, freeze groups, augmentation), so this subclasses ``RFDETRConfig``
rather than the bare ``TrainConfig`` — it needs every field RFDETRTrainer reads
(``ema``, ``backbone_lr_mult``, ``amp``, ``scheduler``, ...) to stay in sync
with RF-DETR's config automatically. Only DINOv2-specific defaults are
overridden here.
"""

from dataclasses import dataclass

from ..rfdetr.config import RFDETRConfig


@dataclass(kw_only=True)
class DINOv2Config(RFDETRConfig):
    """CLI-visible LibreDINOv2 fine-tuning defaults."""

    name: str = "dinov2_exp"

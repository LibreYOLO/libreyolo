"""PP-YOLOE trainer: two-stage assigner schedule over the shared PP-YOLOE loss."""

from __future__ import annotations

from typing import Dict, Type

import torch

from ...training.config import PPYOLOEConfig, TrainConfig
from ...training.scheduler import CosineAnnealingScheduler
from ...training.trainer import BaseTrainer
from ..yolonas.loss import PPYoloELoss
from .transforms import PPYOLOETrainTransform

# Learning rates of the released 500-epoch COCO recipes
# (``coco2017_ppyoloe_train_params.yaml`` plus the per-size overrides). These
# are from-scratch values; ``PPYOLOEConfig.lr0`` deliberately defaults lower
# because LibreYOLO's ``train()`` entry point is a fine-tune path. Pass
# ``lr0=`` explicitly to reproduce the source recipe.
SOURCE_RECIPE_LR0 = {"s": 2e-3, "m": 1e-3, "l": 1e-3, "x": 2e-3}

# Epoch at which the released recipe switches ATSS -> TaskAlignedAssigner,
# and the total epoch budget that switch was tuned for.
SOURCE_STATIC_ASSIGNER_EPOCHS = 150
SOURCE_TOTAL_EPOCHS = 500


def resolve_static_assigner_epochs(
    static_assigner_epochs: int | None, total_epochs: int
) -> int:
    """Epochs of ATSS assignment before TaskAlignedAssigner takes over.

    ``None`` (the config default) scales the source switch point to the
    requested budget, so a 10-epoch fine-tune still gets both phases instead of
    spending all of it on the static assigner. An explicit value is used as
    given, clamped to the run length.
    """
    if static_assigner_epochs is None:
        fraction = SOURCE_STATIC_ASSIGNER_EPOCHS / SOURCE_TOTAL_EPOCHS
        resolved = int(round(total_epochs * fraction))
    else:
        resolved = int(static_assigner_epochs)
    return max(0, min(resolved, int(total_epochs)))


class PPYOLOETrainer(BaseTrainer):
    """Native PP-YOLOE detection trainer.

    Reuses ``libreyolo.models.yolonas.loss.PPYoloELoss`` (ATSS,
    TaskAlignedAssigner, GIoU, DFL) rather than duplicating it, and adds the
    piece YOLO-NAS never needed: the source's two-stage assignment schedule.
    """

    artifact_model_families = ("ppyoloe",)

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return PPYOLOEConfig

    def get_model_family(self) -> str:
        return "ppyoloe"

    def get_model_tag(self) -> str:
        return f"PP-YOLOE-{self.config.size}"

    def create_transforms(self):
        preproc = PPYOLOETrainTransform(
            max_labels=int(getattr(self.config, "max_labels", 100)),
            flip_prob=self.config.flip_prob,
            hsv_prob=self.config.hsv_prob,
            rot90_prob=getattr(self.config, "rot90_prob", 0.5),
            rgb2bgr_prob=getattr(self.config, "rgb2bgr_prob", 0.25),
        )
        from ..yolonas.transforms import YOLONASAffineMixupDataset

        return preproc, YOLONASAffineMixupDataset

    def create_scheduler(self, iters_per_epoch: int):
        return CosineAnnealingScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def on_setup(self):
        self.static_assigner_epochs = resolve_static_assigner_epochs(
            getattr(self.config, "static_assigner_epochs", None),
            self.config.epochs,
        )
        self.loss_fn = PPYoloELoss(
            num_classes=self.config.num_classes,
            use_static_assigner=self.static_assigner_epochs > 0,
            use_varifocal_loss=True,
        ).to(self.device)

    def uses_static_assigner(self, epoch: int) -> bool:
        """ATSS for ``epoch < static_assigner_epochs``, TaskAligned after.

        Derived from the epoch counter rather than latched, so resuming
        mid-run lands in the right phase without persisting extra state.
        """
        return int(epoch) < int(getattr(self, "static_assigner_epochs", 0))

    def validate_validation_loss_config(self) -> None:
        if not getattr(self.config, "val_loss", False):
            return

        from .nn import LibrePPYOLOEModel

        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        if task != "detect" or type(self.model) is not LibrePPYOLOEModel:
            raise ValueError(
                "val_loss=True currently supports PP-YOLOE detection only"
            )

    def build_validation_loss_adapter(self, model: torch.nn.Module):
        from .validation_loss import PPYOLOEValidationLoss

        return PPYOLOEValidationLoss(
            model,
            max_labels=int(getattr(self.config, "max_labels", 100)),
        )

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        def _scalar(v):
            return v.item() if isinstance(v, torch.Tensor) else v

        return {
            "cls": _scalar(outputs.get("cls", 0)),
            "iou": _scalar(outputs.get("iou", 0)),
            "dfl": _scalar(outputs.get("dfl", 0)),
        }

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        self.loss_fn.use_static_assigner = self.uses_static_assigner(self.current_epoch)
        model_outputs = self.model(imgs)
        total_loss, log_losses = self.loss_fn(model_outputs, targets)
        return {
            "total_loss": total_loss,
            "cls": log_losses[0],
            "iou": log_losses[1],
            "dfl": log_losses[2],
        }

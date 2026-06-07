"""
YOLOv9 Trainer for LibreYOLO.

Thin subclass of BaseTrainer with yolo9-specific transforms, scheduler,
and loss extraction.
"""

import torch
from typing import Dict, List, Type

from libreyolo.training.trainer import BaseTrainer
from libreyolo.training.config import TrainConfig, YOLO9Config
from libreyolo.training.freezing import FreezeGroup
from ...training.scheduler import LinearLRScheduler, CosineAnnealingScheduler
from .transforms import YOLO9TrainTransform, YOLO9MosaicMixupDataset


class YOLO9Trainer(BaseTrainer):
    """YOLOv9-specific trainer."""

    artifact_model_families = ("yolo9", "yolo9_e2e")

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return YOLO9Config

    def get_model_family(self) -> str:
        return "yolo9"

    def get_model_tag(self) -> str:
        return f"YOLOv9-{self.config.size}"

    def get_freeze_groups(self) -> List[FreezeGroup]:
        model = self.model
        backbone = getattr(model, "backbone", None)
        neck = getattr(model, "neck", None)
        head = getattr(model, "head", None)
        groups: List[FreezeGroup] = []
        if backbone is not None:
            for name in (
                "conv0",
                "conv1",
                "elan1",
                "down2",
                "elan2",
                "down3",
                "elan3",
                "down4",
                "elan4",
                "spp",
            ):
                module = getattr(backbone, name, None)
                if module is not None:
                    groups.append((f"backbone.{name}", module))
        if neck is not None:
            for name in (
                "elan_up1",
                "elan_up2",
                "down1",
                "elan_down1",
                "down2",
                "elan_down2",
            ):
                module = getattr(neck, name, None)
                if module is not None:
                    groups.append((f"neck.{name}", module))
        if head is not None:
            groups.append(("head", head))
        return groups or super().get_freeze_groups()

    def create_transforms(self):
        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        preproc = YOLO9TrainTransform(
            max_labels=100,
            flip_prob=self.config.flip_prob,
            vertical_flip_prob=self.config.flip_prob if task == "obb" else 0.0,
            hsv_prob=self.config.hsv_prob,
            mask_downsample_ratio=getattr(self.config, "mask_downsample_ratio", 4),
            output_label_dim=6 if task == "obb" else None,
        )
        if task == "segment":
            preproc.wants_unresized_image = True
        return preproc, YOLO9MosaicMixupDataset

    def create_scheduler(self, iters_per_epoch: int):
        scheduler_name = self.config.scheduler
        if scheduler_name == "linear":
            return LinearLRScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        elif scheduler_name in ("cos", "warmcos"):
            return CosineAnnealingScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        def _scalar(v):
            return v.item() if isinstance(v, torch.Tensor) else v

        if getattr(getattr(self, "wrapper_model", None), "task", "detect") == "classify":
            return {"cls": _scalar(outputs.get("cls", 0))}

        components = {
            "box": _scalar(outputs.get("box", 0)),
            "cls": _scalar(outputs.get("cls", 0)),
            "dfl": _scalar(outputs.get("dfl", 0)),
        }
        if "seg" in outputs:
            components["seg"] = _scalar(outputs.get("seg", 0))
        if "angle" in outputs:
            components["angle"] = _scalar(outputs.get("angle", 0))
        return components

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        if getattr(getattr(self, "wrapper_model", None), "task", "detect") == "segment":
            return self.model(imgs, targets=targets, masks=polygons)
        return self.model(imgs, targets=targets)

"""Native RF-DETR trainer for LibreYOLO."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type

import torch

from ...data import load_data_config
from ...training.config import TrainConfig
from ...training.scheduler import FlatCosineScheduler
from ...training.trainer import BaseTrainer
from .config import RFDETRConfig
from ..dfine.transforms import DFINEPassThroughDataset, DFINETrainTransform
from .seg_transforms import RFDETRSegPassThroughDataset, RFDETRSegTransform


class RFDETRTrainer(BaseTrainer):
    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return RFDETRConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_names = None
        if self.config.data:
            data_cfg = load_data_config(
                self.config.data,
                allow_scripts=self.config.allow_download_scripts,
            )
            self.config.num_classes = int(data_cfg.get("nc", self.config.num_classes))
            names = data_cfg.get("names")
            if isinstance(names, dict):
                self._class_names = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, (list, tuple)):
                self._class_names = {i: str(v) for i, v in enumerate(names)}

    @property
    def effective_lr(self) -> float:
        return self.config.lr0 * self.config.batch / 16

    def get_model_family(self) -> str:
        return "rfdetr"

    def get_model_tag(self) -> str:
        return f"LibreRFDETR-{self.config.size}"

    def create_transforms(self):
        if getattr(self.wrapper_model, "task", "detect") == "segment":
            preproc = RFDETRSegTransform(
                max_labels=300,
                flip_prob=self.config.flip_prob,
                imgsz=self.config.imgsz,
                mask_downsample_ratio=4,
            )
            return preproc, RFDETRSegPassThroughDataset
        preproc = DFINETrainTransform(
            max_labels=300,
            flip_prob=self.config.flip_prob,
            imgsz=self.config.imgsz,
            imagenet_norm=True,
            strong_augs=False,
        )
        return preproc, DFINEPassThroughDataset

    def create_scheduler(self, iters_per_epoch: int):
        return FlatCosineScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            no_aug_epochs=self.config.no_aug_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def on_setup(self):
        if self.model.nb_classes != self.config.num_classes:
            self.model.model.reinitialize_detection_head(self.config.num_classes + 1)
            self.model.nb_classes = self.config.num_classes
            self.model.args.num_classes = self.config.num_classes

        self.criterion, _ = self.model.build_criterion_and_postprocess()
        self.criterion.to(self.device)

        if self.wrapper_model is not None:
            self.wrapper_model.nb_classes = self.config.num_classes
            if self._class_names:
                self.wrapper_model.names = self.wrapper_model._sanitize_names(
                    self._class_names,
                    self.config.num_classes,
                )
            else:
                self.wrapper_model.names = {
                    i: f"class_{i}" for i in range(self.config.num_classes)
                }

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        backbone_wd, backbone_no_wd, head_wd, head_no_wd = [], [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = name.startswith("model.backbone.")
            no_wd = "norm" in name or "bias" in name or "pos_embed" in name or "position_embeddings" in name
            if is_backbone and no_wd:
                backbone_no_wd.append(param)
            elif is_backbone:
                backbone_wd.append(param)
            elif no_wd:
                head_no_wd.append(param)
            else:
                head_wd.append(param)

        lr = self.effective_lr
        wd = self.config.weight_decay
        bb_mult = float(self.config.backbone_lr_mult)
        groups = []
        if head_wd:
            groups.append({"params": head_wd, "lr": lr, "weight_decay": wd, "lr_mult": 1.0})
        if head_no_wd:
            groups.append({"params": head_no_wd, "lr": lr, "weight_decay": 0.0, "lr_mult": 1.0})
        if backbone_wd:
            groups.append({"params": backbone_wd, "lr": lr * bb_mult, "weight_decay": wd, "lr_mult": bb_mult})
        if backbone_no_wd:
            groups.append({"params": backbone_no_wd, "lr": lr * bb_mult, "weight_decay": 0.0, "lr_mult": bb_mult})
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))

    def _scale_lr(self, base_lr: float, param_group: dict) -> float:
        return base_lr * float(param_group.get("lr_mult", 1.0))

    def on_forward(
        self,
        imgs: torch.Tensor,
        targets: torch.Tensor,
        polygons: Optional[List] = None,
    ) -> Dict:
        batch_size = targets.shape[0]
        height, width = imgs.shape[-2], imgs.shape[-1]
        scale = torch.tensor([width, height, width, height], device=targets.device, dtype=targets.dtype)
        is_seg = getattr(self.wrapper_model, "task", "detect") == "segment"
        # ``polygons`` here is the collate-stacked output of RFDETRSegTransform:
        # a [B, max_labels, mask_h, mask_w] float32 tensor whose slot i aligns
        # with target slot i. Slice by the same ``valid`` box mask to hand the
        # criterion per-image ``[N_valid, mask_h, mask_w]`` tensors.
        masks_batch = polygons if is_seg and isinstance(polygons, torch.Tensor) else None

        target_list = []
        for batch_idx in range(batch_size):
            t = targets[batch_idx]
            valid = (t[:, 3] > 0) & (t[:, 4] > 0)
            t_valid = t[valid]
            if t_valid.numel() == 0:
                entry = {
                    "labels": torch.zeros(0, dtype=torch.int64, device=self.device),
                    "boxes": torch.zeros(0, 4, dtype=torch.float32, device=self.device),
                }
                if masks_batch is not None:
                    mh, mw = masks_batch.shape[-2], masks_batch.shape[-1]
                    entry["masks"] = torch.zeros(0, mh, mw, dtype=torch.bool, device=self.device)
            else:
                entry = {
                    "labels": t_valid[:, 0].long(),
                    "boxes": (t_valid[:, 1:] / scale).clamp(0.0, 1.0),
                }
                if masks_batch is not None:
                    m = masks_batch[batch_idx][valid]
                    entry["masks"] = m.to(device=self.device, dtype=torch.bool)
            target_list.append(entry)

        outputs = self.model(imgs, targets=target_list)
        loss_dict = self.criterion(outputs, target_list)
        weight_dict = self.criterion.weight_dict
        total = sum(loss_dict[key] * weight_dict[key] for key in loss_dict if key in weight_dict)
        result = {"total_loss": total}
        result.update(loss_dict)
        return result

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        def _sum_with_prefix(prefix: str) -> float:
            total = 0.0
            for key, value in outputs.items():
                if key == prefix or key.startswith(prefix + "_"):
                    total += value.item() if isinstance(value, torch.Tensor) else float(value)
            return total

        components = {
            "ce": _sum_with_prefix("loss_ce"),
            "bbox": _sum_with_prefix("loss_bbox"),
            "giou": _sum_with_prefix("loss_giou"),
        }
        if getattr(self.wrapper_model, "task", "detect") == "segment":
            components["mask_ce"] = _sum_with_prefix("loss_mask_ce")
            components["mask_dice"] = _sum_with_prefix("loss_mask_dice")
        return components


def train_rfdetr(
    data: str,
    size: str = "s",
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_dir: str = "runs/train",
    resume: str | None = None,
    pretrain_weights: str | None = None,
    segmentation: bool = False,
    **kwargs,
) -> Dict:
    """Compatibility helper around :class:`LibreRFDETR.train`."""
    from .model import LibreRFDETR

    model = LibreRFDETR(
        model_path=pretrain_weights,
        size=size,
        device=kwargs.pop("device", "auto"),
        segmentation=segmentation,
    )
    return model.train(
        data=data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        output_dir=str(Path(output_dir)),
        resume=resume,
        **kwargs,
    )


__all__ = ["RFDETRTrainer", "train_rfdetr"]

"""Native RF-DETR trainer for LibreYOLO."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Type

import torch
import torch.nn.functional as F

from ...data import load_data_config
from ...training.config import TrainConfig
from ...training.distributed import unwrap_model
from ...training.scheduler import BaseScheduler, CosineAnnealingScheduler, FlatCosineScheduler
from ...training.trainer import BaseTrainer
from .config import RFDETRConfig
from ..dfine.transforms import DFINEPassThroughDataset
from .seg_transforms import (
    RFDETRDetTransform,
    RFDETRSegPassThroughDataset,
    RFDETRSegTransform,
    compute_multi_scale_scales,
)


class RFDETRStepScheduler(BaseScheduler):
    """RF-DETR upstream-style warmup plus step decay schedule."""

    def __init__(
        self,
        lr: float,
        iters_per_epoch: int,
        total_epochs: int,
        warmup_epochs: float = 0.0,
        lr_drop: int = 100,
    ):
        super().__init__(lr, iters_per_epoch, total_epochs)
        self.warmup_iters = int(iters_per_epoch * warmup_epochs)
        self.drop_iter = int(iters_per_epoch * lr_drop)

    def update_lr(self, iters: int) -> float:
        if self.warmup_iters > 0 and iters < self.warmup_iters:
            return self.lr * float(iters) / float(max(1, self.warmup_iters))
        if iters < self.drop_iter:
            return self.lr
        return self.lr * 0.1


class RFDETRTrainer(BaseTrainer):
    artifact_model_families = ("rfdetr",)
    # RF-DETR has a DINOv2 ViT backbone whose attention/MLP projections are
    # nn.Linear layers, so LoRA fine-tuning is supported here.
    supports_lora = True

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return RFDETRConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_names = None
        # Classification resolves its class count from the ImageFolder dataset
        # in _setup_classify_data, not from a detection YAML.
        if getattr(self.wrapper_model, "task", "detect") == "classify":
            return
        if self.config.data:
            data_cfg = load_data_config(
                self.config.data,
                allow_scripts=self.config.allow_download_scripts,
            )
            names = data_cfg.get("names")
            data_nc = data_cfg.get("nc")
            if data_nc is None and names is not None:
                data_nc = len(names)
            self.config.num_classes = int(
                data_nc if data_nc is not None else self.config.num_classes
            )
            if isinstance(names, dict):
                self._class_names = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, (list, tuple)):
                self._class_names = {i: str(v) for i, v in enumerate(names)}

    @property
    def effective_lr(self) -> float:
        return self.config.lr0

    def get_model_family(self) -> str:
        return "rfdetr"

    def get_model_tag(self) -> str:
        return f"LibreRFDETR-{self.config.size}"

    def _ddp_find_unused_parameters(self) -> bool:
        # Detect/OBB: at least one transformer parameter receives no gradient
        # on some forward passes, so DDP must skip reduction for it.
        # Segmentation uses static_graph=True instead (see _ddp_static_graph).
        return getattr(self.wrapper_model, "task", "detect") in {"detect", "obb"}

    def _ddp_static_graph(self) -> bool:
        # Segmentation only: the seg head is invoked from both encoder and
        # decoder branches in one forward, so its parameters accumulate
        # gradients from two call sites. With find_unused_parameters=True the
        # DDP hook fires twice per step → "marked ready twice" crash.
        # static_graph=True locks the reducer after iteration 1 and handles
        # double accumulation correctly. Detect/OBB keep the default (False)
        # because find_unused_parameters=True requires dynamic graph traversal.
        return getattr(self.wrapper_model, "task", "detect") == "segment"

    def create_transforms(self):
        patch_size = int(getattr(self.model, "patch_size", 16))
        num_windows = int(getattr(self.model, "num_windows", 4))
        block_size = patch_size * num_windows
        # Validation always uses the literal imgsz, so divisibility is required
        # regardless of multi_scale mode.
        if self.config.imgsz % block_size != 0:
            lo = (self.config.imgsz // block_size) * block_size
            hi = lo + block_size
            raise ValueError(
                f"imgsz={self.config.imgsz} is not divisible by {block_size} "
                f"(patch_size={patch_size} x num_windows={num_windows}). "
                f"Use {lo} or {hi}."
            )
        task = getattr(self.wrapper_model, "task", "detect")
        if task == "segment":
            preproc = RFDETRSegTransform(
                max_labels=300,
                flip_prob=self.config.flip_prob,
                imgsz=self.config.imgsz,
                mask_downsample_ratio=4,
                multi_scale=self.config.multi_scale,
                expanded_scales=self.config.expanded_scales,
                do_random_resize_via_padding=self.config.do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
                crop_resize_prob=self.config.crop_resize_prob,
            )
            return preproc, RFDETRSegPassThroughDataset
        preproc = RFDETRDetTransform(
            max_labels=300,
            flip_prob=self.config.flip_prob,
            imgsz=self.config.imgsz,
            multi_scale=self.config.multi_scale,
            expanded_scales=self.config.expanded_scales,
            do_random_resize_via_padding=self.config.do_random_resize_via_padding,
            patch_size=patch_size,
            num_windows=num_windows,
            crop_resize_prob=self.config.crop_resize_prob,
            target_dim=6 if task == "obb" else 5,
        )
        return preproc, DFINEPassThroughDataset

    def create_scheduler(self, iters_per_epoch: int):
        scheduler = str(getattr(self.config, "scheduler", "step")).lower()
        if scheduler == "step":
            return RFDETRStepScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                lr_drop=getattr(self.config, "lr_drop", self.config.epochs),
            )
        if scheduler == "cosine":
            return CosineAnnealingScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=0.0,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        return FlatCosineScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            no_aug_epochs=self.config.no_aug_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def _multi_scale_scales(self) -> list[int]:
        if not self.config.multi_scale or self.config.do_random_resize_via_padding:
            return []
        raw = unwrap_model(self.model)
        patch_size = int(getattr(raw, "patch_size", 16))
        num_windows = int(getattr(raw, "num_windows", 4))
        return compute_multi_scale_scales(
            self.config.imgsz,
            self.config.expanded_scales,
            patch_size,
            num_windows,
        )

    def _apply_multi_scale_batch(
        self,
        imgs: torch.Tensor,
        targets: torch.Tensor,
        polygons,
        *,
        step: int,
    ):
        if getattr(self.wrapper_model, "task", "detect") == "classify":
            return imgs, targets, polygons
        scales = self._multi_scale_scales()
        if not scales:
            return imgs, targets, polygons

        rng = random.Random(step)
        scale = rng.choice(scales)
        current_h, current_w = imgs.shape[-2:]
        if current_h == scale and current_w == scale:
            return imgs, targets, polygons

        scale_x = scale / float(current_w)
        scale_y = scale / float(current_h)
        imgs = F.interpolate(
            imgs,
            size=(scale, scale),
            mode="bilinear",
            align_corners=False,
        )

        targets = targets.clone()
        targets[..., 1] *= scale_x
        targets[..., 2] *= scale_y
        targets[..., 3] *= scale_x
        targets[..., 4] *= scale_y

        if isinstance(polygons, torch.Tensor):
            polygons = F.interpolate(
                polygons.float(),
                size=(scale, scale),
                mode="nearest",
            )

        return imgs, targets, polygons

    def on_setup(self):
        # Classification computes cross-entropy inside the model head and sizes
        # its head from the dataset in _setup_classify_data — no detection
        # criterion or head reinitialization is needed.
        if getattr(self.wrapper_model, "task", "detect") == "classify":
            return
        if self.model.nb_classes != self.config.num_classes:
            self.model.model.reinitialize_detection_head(self.config.num_classes + 1)
            self.model.nb_classes = self.config.num_classes
            self.model.args.num_classes = self.config.num_classes

        if getattr(self.config, "lora", False):
            from ...training.lora import apply_lora_to_rfdetr

            core_model = getattr(self.model, "model", self.model)
            apply_lora_to_rfdetr(core_model)

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
        if getattr(self.wrapper_model, "task", "detect") == "classify":
            # The DINOv2 backbone + projector is LayerNorm-only (no BatchNorm),
            # so the shared BN/conv/bias grouping leaves an empty first group.
            # Use a standard AdamW with no weight decay on norms/biases — the
            # conventional recipe for fine-tuning a ViT classifier.
            decay, no_decay = [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
                    no_decay.append(param)
                else:
                    decay.append(param)
            groups = []
            if decay:
                groups.append({"params": decay, "weight_decay": self.config.weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "weight_decay": 0.0})
            return torch.optim.AdamW(groups, lr=self.effective_lr, betas=(0.9, 0.999))
        upstream_groups = self._setup_upstream_optimizer_groups()
        if upstream_groups:
            return torch.optim.AdamW(
                upstream_groups,
                lr=self.effective_lr,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
            )

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

    def _setup_upstream_optimizer_groups(self) -> list[dict]:
        core_model = getattr(self.model, "model", self.model)
        backbone = getattr(core_model, "backbone", None)
        if backbone is None:
            return []
        try:
            backbone_encoder = backbone[0]
        except (TypeError, IndexError):
            return []
        if not hasattr(backbone_encoder, "get_named_param_lr_pairs"):
            return []

        model_args = getattr(self.model, "args", getattr(core_model, "args", None))
        if model_args is None:
            return []
        args = SimpleNamespace(**vars(model_args))
        args.lr = self.effective_lr
        args.weight_decay = self.config.weight_decay

        backbone_param_by_name = backbone_encoder.get_named_param_lr_pairs(
            args,
            prefix="backbone.0",
        )
        if not backbone_param_by_name:
            return []

        base_lr = max(float(self.effective_lr), 1e-12)
        decoder_key = "transformer.decoder"
        groups = []
        decoder_params = []
        other_params = []
        for name, param in core_model.named_parameters():
            if not param.requires_grad:
                continue
            if name in backbone_param_by_name:
                continue
            if decoder_key in name:
                decoder_params.append(param)
            else:
                other_params.append(param)

        for param in other_params:
            groups.append({"params": param, "lr": self.effective_lr, "lr_mult": 1.0})

        for param_group in backbone_param_by_name.values():
            group = dict(param_group)
            group["lr_mult"] = float(group["lr"]) / base_lr
            groups.append(group)

        decoder_lr = self.effective_lr * float(getattr(args, "lr_component_decay", 1.0))
        decoder_lr_mult = decoder_lr / base_lr
        for param in decoder_params:
            groups.append({"params": param, "lr": decoder_lr, "lr_mult": decoder_lr_mult})

        return groups

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
        task = getattr(self.wrapper_model, "task", "detect")
        is_seg = task == "segment"
        is_obb = task == "obb"
        # ``polygons`` here is the collate-stacked output of RFDETRSegTransform:
        # a [B, max_labels, mask_h, mask_w] float32 tensor whose slot i aligns
        # with target slot i. Slice by the same ``valid`` box mask to hand the
        # criterion per-image ``[N_valid, mask_h, mask_w]`` tensors.
        masks_batch = (
            polygons.to(self.device, non_blocking=True)
            if is_seg and isinstance(polygons, torch.Tensor)
            else None
        )

        if getattr(self.wrapper_model, "task", "detect") == "classify":
            # ``targets`` are class indices [B]; the model head returns the
            # loss dict directly.
            return self.model(imgs, targets=targets)

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
                if is_obb:
                    entry["angles"] = torch.zeros(0, dtype=torch.float32, device=self.device)
                if masks_batch is not None:
                    mh, mw = masks_batch.shape[-2], masks_batch.shape[-1]
                    entry["masks"] = torch.zeros(0, mh, mw, dtype=torch.bool, device=self.device)
            else:
                entry = {
                    "labels": t_valid[:, 0].long(),
                    "boxes": (t_valid[:, 1:5] / scale).clamp(0.0, 1.0),
                }
                if is_obb:
                    entry["angles"] = t_valid[:, 5].float()
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
        if getattr(self.wrapper_model, "task", "detect") == "classify":
            value = outputs.get("cls", 0)
            return {"cls": value.item() if isinstance(value, torch.Tensor) else float(value)}

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
        if getattr(self.wrapper_model, "task", "detect") == "obb":
            components["angle"] = _sum_with_prefix("loss_angle")
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

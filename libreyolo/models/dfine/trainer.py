"""DFINETrainer — BaseTrainer subclass for native D-FINE training.

The integration tricks in this file:

1. ``on_forward`` translates LibreYOLO's padded ``(B, max_labels, 5)`` target
   tensor to D-FINE's ``list[dict{labels, boxes_cxcywh_normalized}]`` per-image
   format expected by the criterion.

2. ``_setup_optimizer`` builds 4 param groups (backbone wd / no-wd, head
   wd / no-wd) and stamps each with an ``lr_mult`` (backbone groups at
   ``config.backbone_lr_mult``, default 0.5).

3. ``_setup_data`` swaps the parent's standard collate for
   ``DFINEMultiScaleCollate`` (random per-batch resize until stop_epoch) when
   ``config.multi_scale=True``.

4. ``_train_epoch`` is a copy of ``BaseTrainer._train_epoch`` with three
   tweaks: per-epoch ``set_epoch`` propagation to the dataset and collate,
   gradient clipping at ``config.clip_max_norm``, and per-group LR (the
   scheduler's single output is multiplied by each group's ``lr_mult`` instead
   of being applied uniformly).

   Why a wholesale override: ``BaseTrainer._train_epoch`` doesn't expose
   pre-step / post-step hooks. The copy is intentionally kept structurally
   close to the parent so drift is easy to audit; if a third family ends up
   needing the same hooks, promote them into ``BaseTrainer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import torch
from torch.amp import autocast
from tqdm import tqdm

from ...data import get_img_files, img2label_paths, load_data_config
from ...data.dataset import COCODataset, YOLODataset
from ...training.config import DFINEConfig, TrainConfig
from ...training.scheduler import FlatCosineScheduler
from ...training.trainer import BaseTrainer
from .loss import DFINECriterion
from .matcher import HungarianMatcher
from .transforms import (
    DFINEMultiScaleCollate,
    DFINEPassThroughDataset,
    DFINETrainTransform,
)


class DFINETrainer(BaseTrainer):
    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return DFINEConfig

    def get_model_family(self) -> str:
        return "dfine"

    def get_model_tag(self) -> str:
        return f"DFINE-{self.config.size}"

    def create_transforms(self):
        preproc = DFINETrainTransform(max_labels=120, flip_prob=self.config.flip_prob)
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

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        # FGL/DDF are emitted only by the aux/dn paths (no main-loss key);
        # bare ``outputs.get("loss_ddf")`` was always 0. Aggregate over every
        # variant key so the tqdm display reflects the actual loss magnitude.
        def _sum_with_prefix(prefix: str) -> float:
            total = 0.0
            for k, v in outputs.items():
                if k == prefix or k.startswith(prefix + "_"):
                    total += v.item() if isinstance(v, torch.Tensor) else float(v)
            return total

        return {
            "vfl": _sum_with_prefix("loss_vfl"),
            "bbox": _sum_with_prefix("loss_bbox"),
            "giou": _sum_with_prefix("loss_giou"),
            "fgl": _sum_with_prefix("loss_fgl"),
            "ddf": _sum_with_prefix("loss_ddf"),
        }

    def _setup_device(self) -> torch.device:
        """Override the parent's device autodetect to avoid MPS.

        D-FINE's training backward pass crashes on Apple's MPS backend in the
        ``linear_backward`` op (the Integral's 33-bin softmax × W matmul hits
        a known MPS / MetalPerformanceShadersGraph compilation failure). Eval
        mode is fine — this only applies to training. Force CPU when the
        parent would have picked MPS.
        """
        device = super()._setup_device()
        if device.type == "mps":
            import logging

            logging.getLogger(__name__).warning(
                "D-FINE training on Apple MPS triggers a torch backward bug "
                "(mps_linear_backward in Metal). Falling back to CPU. "
                "Pass device='cuda' or device='cpu' explicitly to override."
            )
            return torch.device("cpu")
        return device

    def on_setup(self):
        matcher = HungarianMatcher(
            weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        )
        self.criterion = DFINECriterion(
            matcher=matcher,
            weight_dict={
                "loss_vfl": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            losses=["vfl", "boxes", "local"],
            alpha=0.75,
            gamma=2.0,
            num_classes=self.config.num_classes,
            reg_max=32,
        ).to(self.device)

    def on_mosaic_disable(self):
        super().on_mosaic_disable()
        # D-FINE's "EMA restart": switch to a constant decay for the final phase.
        if self.ema_model is not None:
            decay = getattr(self.config, "ema_restart_decay", self.config.ema_decay)
            self.ema_model.set_decay(decay)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """AdamW with 4 param groups: {backbone, head} × {wd, no-wd}.

        Each group gets an ``lr_mult`` that ``_train_epoch`` reads back when
        applying the scheduler-returned base LR. The default
        ``backbone_lr_mult=0.5`` matches D-FINE's published fine-tune recipe;
        head groups stay at 1.0×.
        """
        backbone_wd, backbone_no_wd, head_wd, head_no_wd = [], [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # Match upstream's regex semantics ``(?:norm|bn|bias)`` — substring,
            # not suffix. The previous ``endswith('.bias')`` missed
            # ``self_attn.in_proj_bias`` (PyTorch MHA's fused QKV bias) on five
            # parameters per model, which silently received weight decay.
            is_norm_or_bias = (
                "norm" in name
                or ".bn." in name
                or "bias" in name
                or "lab.scale" in name
            )
            is_backbone = name.startswith("backbone.")
            if is_backbone and is_norm_or_bias:
                backbone_no_wd.append(p)
            elif is_backbone:
                backbone_wd.append(p)
            elif is_norm_or_bias:
                head_no_wd.append(p)
            else:
                head_wd.append(p)

        lr = self.effective_lr
        wd = self.config.weight_decay
        bb_mult = float(getattr(self.config, "backbone_lr_mult", 1.0))

        param_groups = []
        if head_wd:
            param_groups.append(
                {"params": head_wd, "lr": lr, "weight_decay": wd, "lr_mult": 1.0}
            )
        if head_no_wd:
            param_groups.append(
                {"params": head_no_wd, "lr": lr, "weight_decay": 0.0, "lr_mult": 1.0}
            )
        if backbone_wd:
            param_groups.append(
                {
                    "params": backbone_wd,
                    "lr": lr * bb_mult,
                    "weight_decay": wd,
                    "lr_mult": bb_mult,
                }
            )
        if backbone_no_wd:
            param_groups.append(
                {
                    "params": backbone_no_wd,
                    "lr": lr * bb_mult,
                    "weight_decay": 0.0,
                    "lr_mult": bb_mult,
                }
            )

        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forward + loss in one go.

        Translates the ``(B, max_labels, 5)`` ``[class, cx, cy, w, h]`` pixel
        target tensor into D-FINE's per-image dict list with cxcywh-normalized
        boxes, then runs model + criterion.
        """
        B = targets.shape[0]
        # Read actual image size from the batch — multi-scale collate may have
        # resized to a non-default value (576..704), so we cannot trust
        # ``config.imgsz`` here.
        H, W = imgs.shape[-2], imgs.shape[-1]
        scale = torch.tensor([W, H, W, H], device=targets.device, dtype=targets.dtype)

        target_list = []
        for b in range(B):
            t = targets[b]
            # Padding rows are zero in all 5 columns; valid boxes have w>0 and h>0.
            valid = (t[:, 3] > 0) & (t[:, 4] > 0)
            t_valid = t[valid]
            if t_valid.numel() == 0:
                target_list.append(
                    {
                        "labels": torch.zeros(0, dtype=torch.int64, device=self.device),
                        "boxes": torch.zeros(
                            0, 4, dtype=torch.float32, device=self.device
                        ),
                    }
                )
            else:
                target_list.append(
                    {
                        "labels": t_valid[:, 0].long(),
                        "boxes": (t_valid[:, 1:] / scale).clamp(0.0, 1.0),
                    }
                )

        outputs = self.model(imgs, targets=target_list)
        losses = self.criterion(outputs, target_list)
        total = sum(losses.values())

        # Expose every named loss (including aux_/dn_/pre/enc variants) so
        # ``get_loss_components`` can aggregate by prefix. FGL/DDF appear only
        # in the aux/dn paths — bare ``loss_ddf`` would always be 0 otherwise.
        result = {"total_loss": total}
        result.update(losses)
        return result

    # =========================================================================
    # _setup_data override — wire DFINEMultiScaleCollate (when enabled)
    # =========================================================================

    def _setup_data(self):
        """Mirror of ``BaseTrainer._setup_data`` but uses ``DFINEMultiScaleCollate``.

        Built by hand instead of inheriting because ``create_dataloader`` doesn't
        expose ``collate_fn`` and we need our epoch-aware collate. Dataset-build
        logic is duplicated from the parent for clarity.
        """
        from torch.utils.data import DataLoader

        img_size = self.input_size
        preproc, MosaicDatasetClass = self.create_transforms()

        if self.config.data:
            data_cfg = load_data_config(self.config.data)
            data_dir = data_cfg["root"]
            self.num_classes = data_cfg.get("nc", self.config.num_classes)

            ann_file = Path(data_dir) / "annotations" / "instances_train2017.json"
            img_files = data_cfg.get("train_img_files")
            label_files = data_cfg.get("train_label_files")

            if img_files:
                train_dataset = YOLODataset(
                    img_files=img_files,
                    label_files=label_files,
                    img_size=img_size,
                    preproc=preproc,
                )
            elif ann_file.exists():
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name="train2017",
                    img_size=img_size,
                    preproc=preproc,
                )
            else:
                train_path = data_cfg.get("train", "images/train")
                try:
                    img_files = get_img_files(train_path, prefix=data_dir)
                except (FileNotFoundError, ValueError):
                    img_files = []
                if not img_files:
                    raise FileNotFoundError(f"No images found in {train_path}")
                label_files = img2label_paths(img_files)
                train_dataset = YOLODataset(
                    img_files=img_files,
                    label_files=label_files,
                    img_size=img_size,
                    preproc=preproc,
                )
        elif self.config.data_dir:
            data_dir = self.config.data_dir
            self.num_classes = self.config.num_classes
            if (Path(data_dir) / "annotations").exists():
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name="train2017",
                    img_size=img_size,
                    preproc=preproc,
                )
            else:
                train_dataset = YOLODataset(
                    data_dir=data_dir,
                    split="train",
                    img_size=img_size,
                    preproc=preproc,
                )
        else:
            raise ValueError("Either 'data' or 'data_dir' must be specified")

        train_dataset = MosaicDatasetClass(
            dataset=train_dataset,
            img_size=img_size,
            mosaic=True,
            preproc=preproc,
            degrees=self.config.degrees,
            translate=self.config.translate,
            mosaic_scale=self.config.mosaic_scale,
            mixup_scale=self.config.mixup_scale,
            shear=self.config.shear,
            enable_mixup=self.config.mixup_prob > 0,
            mosaic_prob=self.config.mosaic_prob,
            mixup_prob=self.config.mixup_prob,
        )

        # Wire stop_epoch on the dataset wrapper so set_epoch can disable
        # strong augs at the right moment.
        stop_epoch = int(
            self.config.epochs
            * float(getattr(self.config, "aug_stop_epoch_ratio", 1.0))
        )
        if hasattr(train_dataset, "set_stop_epoch"):
            train_dataset.set_stop_epoch(stop_epoch)

        # Multi-scale collate (or default yolox_collate_fn).
        if getattr(self.config, "multi_scale", False):
            collate_fn = DFINEMultiScaleCollate(
                base_size=self.config.imgsz,
                base_size_repeat=3,
                stop_epoch=stop_epoch,
            )
        else:
            from ...data.dataset import yolox_collate_fn

            collate_fn = yolox_collate_fn

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch,
            num_workers=self.config.workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

        return train_dataset

    # =========================================================================
    # _train_epoch override — set_epoch propagation, grad clip, per-group LR
    # =========================================================================

    def _train_epoch(self, epoch: int) -> Tuple[float, Optional[Dict[str, float]]]:
        """Copy of ``BaseTrainer._train_epoch`` with three D-FINE-specific tweaks:

        1. Propagate the current epoch to dataset + collate (drives stop_epoch
           augmentation/multi-scale gating).
        2. Apply gradient clipping at ``config.clip_max_norm`` before the
           optimizer step.
        3. Apply per-group LR multipliers (the scheduler returns one base LR;
           each param group's ``lr_mult`` scales it).
        """
        # 1. Epoch propagation.
        ds = self.train_loader.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        cf = getattr(self.train_loader, "collate_fn", None)
        if cf is not None and hasattr(cf, "set_epoch"):
            cf.set_epoch(epoch)

        clip_max_norm = float(getattr(self.config, "clip_max_norm", 0.0))

        self.model.train()
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            total=len(self.train_loader),
        )

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (imgs, targets, img_infos, img_ids) in enumerate(pbar):
            self.current_iter = epoch * len(self.train_loader) + batch_idx

            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.scaler is not None:
                with autocast("cuda"):
                    outputs = self.on_forward(imgs, targets)
                    loss = outputs["total_loss"]
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                if clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), clip_max_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.on_forward(imgs, targets)
                loss = outputs["total_loss"]
                self.optimizer.zero_grad()
                loss.backward()
                if clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), clip_max_norm
                    )
                self.optimizer.step()

            if self.ema_model is not None:
                self.ema_model.update(self.model)

            loss_val = loss.item()
            loss_components = self.get_loss_components(outputs)
            total_loss += loss_val
            del outputs, loss

            # 3. Per-group LR (scheduler returns one base LR; each group
            # multiplies by its ``lr_mult``).
            base_lr = self.lr_scheduler.update_lr(self.current_iter + 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = base_lr * pg.get("lr_mult", 1.0)
            num_batches += 1

            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{base_lr:.6f}"}
            postfix.update({k: f"{v:.4f}" for k, v in loss_components.items()})
            pbar.set_postfix(postfix)

            if self.tensorboard_writer and batch_idx % self.config.log_interval == 0:
                self.tensorboard_writer.add_scalar(
                    "train/loss", loss_val, self.current_iter
                )
                self.tensorboard_writer.add_scalar(
                    "train/lr", base_lr, self.current_iter
                )
                for name, val in loss_components.items():
                    self.tensorboard_writer.add_scalar(
                        f"train/{name}", val, self.current_iter
                    )

        avg_loss = total_loss / max(num_batches, 1)

        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("epoch/loss", avg_loss, epoch)

        val_metrics = None
        if (
            self.config.eval_interval > 0
            and (epoch + 1) % self.config.eval_interval == 0
        ):
            val_metrics = self._validate_epoch(epoch)
            if val_metrics and self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(
                    "val/mAP50", val_metrics["mAP50"], epoch
                )
                self.tensorboard_writer.add_scalar(
                    "val/mAP50_95", val_metrics["mAP50_95"], epoch
                )

        return avg_loss, val_metrics

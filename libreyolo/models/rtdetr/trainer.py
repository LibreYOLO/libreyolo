"""
RT-DETR Trainer for LibreYOLO.

Subclass of BaseTrainer with RT-DETR-specific transforms, scheduler,
loss extraction, and optimizer configuration.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

from libreyolo.training.trainer import BaseTrainer
from libreyolo.training.config import TrainConfig
from libreyolo.training.scheduler import LinearLRScheduler, CosineAnnealingScheduler
from libreyolo.training.augment import TrainTransform, MosaicMixupDataset

from .config import RTDETRConfig
from .loss import RTDETRLoss


logger = logging.getLogger(__name__)


def convert_targets_for_detr(targets: torch.Tensor, batch_size: int) -> List[Dict[str, torch.Tensor]]:
    """Convert YOLO-format batch targets to DETR format.
    
    Args:
        targets: [B, max_labels, 5] tensor where each row is [cls, x1, y1, x2, y2] in normalized coords
        batch_size: Number of images in batch
        
    Returns:
        List of dicts with 'labels' and 'boxes' (cxcywh format) for each image
    """
    detr_targets = []
    device = targets.device
    
    for i in range(batch_size):
        batch_targets = targets[i]
        
        # Valid boxes have x2 > x1 and y2 > y1 (columns 3 > 1 and 4 > 2)
        mask = (batch_targets[:, 3] > batch_targets[:, 1]) & (batch_targets[:, 4] > batch_targets[:, 2])
        valid_targets = batch_targets[mask]
        
        labels = valid_targets[:, 0].long()
        xyxy = valid_targets[:, 1:5]
        
        if len(labels) == 0:
            detr_targets.append({
                "labels": torch.zeros(0, dtype=torch.int64, device=device),
                "boxes": torch.zeros(0, 4, dtype=torch.float32, device=device)
            })
        else:
            # Convert xyxy to cxcywh
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]
            cx = xyxy[:, 0] + w / 2
            cy = xyxy[:, 1] + h / 2
            boxes = torch.stack([cx, cy, w, h], dim=-1)
            detr_targets.append({"labels": labels, "boxes": boxes})
            
    return detr_targets


class RTDETRTrainer(BaseTrainer):
    """RT-DETR-specific trainer."""

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return RTDETRConfig

    def get_model_family(self) -> str:
        return "rtdetr"

    def get_model_tag(self) -> str:
        return f"RT-DETR-{self.config.size}"

    def create_transforms(self):
        preproc = TrainTransform(
            max_labels=300,  # RTDETR uses more labels
            flip_prob=self.config.flip_prob,
            hsv_prob=self.config.hsv_prob,
        )
        return preproc, MosaicMixupDataset

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
        """Extract per-component losses for logging.
        
        The outputs dict comes from SetCriterion.forward() which includes
        total_loss and individual components.
        """
        def _scalar(v):
            return v.item() if isinstance(v, torch.Tensor) else v
        
        components = {"total": _scalar(outputs.get("total_loss", 0))}
        
        # Main loss components
        if "loss_vfl" in outputs:
            components["vfl"] = _scalar(outputs["loss_vfl"])
        elif "loss_focal" in outputs:
            components["focal"] = _scalar(outputs["loss_focal"])
        elif "loss_bce" in outputs:
            components["bce"] = _scalar(outputs["loss_bce"])
            
        if "loss_bbox" in outputs:
            components["bbox"] = _scalar(outputs["loss_bbox"])
        if "loss_giou" in outputs:
            components["giou"] = _scalar(outputs["loss_giou"])
            
        return components

    def on_setup(self):
        """Initialize the loss criterion."""
        self.criterion = RTDETRLoss(num_classes=self.config.num_classes)
        self.criterion.to(self.device)

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Run the model forward pass with DETR-specific target conversion.
        
        Args:
            imgs: [B, 3, H, W] image tensor
            targets: [B, max_labels, 5] YOLO-format targets [cls, x1, y1, x2, y2]
            
        Returns:
            Dict with total_loss and individual loss components
        """
        batch_size = imgs.shape[0]
        
        # Convert YOLO targets to DETR format
        detr_targets = convert_targets_for_detr(targets, batch_size)
        
        # Forward pass through model
        outputs = self.model(imgs, targets=detr_targets)
        
        # Compute losses
        loss_dict = self.criterion(outputs, detr_targets)
        
        return loss_dict

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with regex-based parameter group matching.
        
        Parameter groups (matched in order, first match wins):
          1. backbone + norm      -> lr=config.lr_backbone, weight_decay=0
          2. backbone + non-norm  -> lr=config.lr_backbone
          3. encoder/decoder + norm/bias -> weight_decay=0
          4. everything else      -> default lr and weight_decay
        """
        config = self.config
        base_lr = self.effective_lr
        base_wd = config.weight_decay
        betas = config.betas
        lr_bb = config.lr_backbone

        # Define param group rules: (regex_pattern, overrides)
        # Note: backbone (timm ResNet) uses 'bn' for BatchNorm layers,
        # while encoder/decoder use 'norm' (LayerNorm/BatchNorm via ConvNormLayer).
        group_rules = [
            (re.compile(r'^(?=.*backbone)(?=.*(?:norm|bn)).*$'),  {"lr": lr_bb, "weight_decay": 0.0}),
            (re.compile(r'^(?=.*backbone)(?!.*(?:norm|bn)).*$'),  {"lr": lr_bb}),
            (re.compile(r'^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$'), {"weight_decay": 0.0}),
        ]

        # Buckets: one per rule + a default bucket
        param_groups = [[] for _ in range(len(group_rules) + 1)]

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            matched = False
            for idx, (pattern, _) in enumerate(group_rules):
                if pattern.search(name):
                    param_groups[idx].append(param)
                    matched = True
                    break
            if not matched:
                param_groups[-1].append(param)

        # Build optimizer param group dicts
        opt_groups = []
        for idx, params in enumerate(param_groups):
            if not params:
                continue
            group = {"params": params, "lr": base_lr, "weight_decay": base_wd}
            if idx < len(group_rules):
                _, overrides = group_rules[idx]
                group.update(overrides)
            # Store lr_ratio so the scheduler can scale per-group LRs proportionally
            group["lr_ratio"] = group["lr"] / base_lr
            opt_groups.append(group)

        optimizer_name = config.optimizer.lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(opt_groups, lr=base_lr, betas=betas, weight_decay=base_wd)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(opt_groups, lr=base_lr, betas=betas, weight_decay=base_wd)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                opt_groups, 
                lr=base_lr, 
                momentum=config.momentum, 
                weight_decay=base_wd, 
                nesterov=getattr(config, 'nesterov', False)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        
        # Log parameter groups
        logger.info(f"Optimizer: {optimizer_name}")
        for i, g in enumerate(opt_groups):
            logger.info(f"  - Group {i}: lr={g['lr']:.6f}, wd={g.get('weight_decay', base_wd):.6f}, "
                       f"lr_ratio={g.get('lr_ratio', 1.0):.4f}, params={len(g['params'])}")
            
        return optimizer

    def _train_epoch(self, epoch: int) -> Tuple[float, Optional[Dict[str, float]]]:
        """Override to handle per-group LR scaling."""
        self.model.train()

        from tqdm import tqdm
        from torch.amp import autocast

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

            # Forward + backward
            if self.scaler is not None:
                with autocast("cuda"):
                    outputs = self.on_forward(imgs, targets)
                    loss = outputs["total_loss"]
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.on_forward(imgs, targets)
                loss = outputs["total_loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # EMA
            if self.ema_model is not None:
                self.ema_model.update(self.model)

            loss_val = loss.item()
            loss_components = self.get_loss_components(outputs)
            total_loss += loss_val

            del outputs, loss

            # LR update with per-group scaling
            base_lr = self.lr_scheduler.update_lr(self.current_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = base_lr * param_group.get("lr_ratio", 1.0)
            num_batches += 1

            # Progress bar
            postfix = {"loss": f"{loss_val:.4f}", "lr": f"base_lr:.6f"}
            postfix.update({k: f"{v:.4f}" for k, v in loss_components.items()})
            pbar.set_postfix(postfix)

            # TensorBoard
            if self.tensorboard_writer and batch_idx % self.config.log_interval == 0:
                self.tensorboard_writer.add_scalar(
                    "train/loss", loss_val, self.current_iter
                )
                self.tensorboard_writer.add_scalar("train/lr", base_lr, self.current_iter)
                for name, val in loss_components.items():
                    self.tensorboard_writer.add_scalar(
                        f"train/{name}", val, self.current_iter
                    )

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("epoch/loss", avg_loss, epoch)

        # Validation
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

"""
RT-DETR Trainer - Minimal v1 implementation.

Features:
- Hungarian matching + VFL/L1/GIoU losses
- AdamW with backbone LR multiplier
- AMP (Automatic Mixed Precision)
- EMA (Exponential Moving Average)
- Gradient clipping
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .config import RTDETRTrainConfig
from .criterion import RTDETRCriterion
from ..ema import ModelEMA
from ..dataset import YOLODataset


logger = logging.getLogger(__name__)


class RTDETRTrainer:
    """
    Simple RT-DETR Trainer.

    Follows YOLOXTrainer patterns while adapting for RT-DETR's
    transformer-based architecture and Hungarian matching.
    """

    def __init__(self, model: nn.Module, config: RTDETRTrainConfig):
        """
        Initialize trainer.

        Args:
            model: RT-DETR model (RTDETRModel instance)
            config: Training configuration
        """
        self.model = model
        self.config = config
        self.device = self._setup_device()

        # Training state
        self.epoch = 0
        self.best_loss = float("inf")

        # Components (initialized in setup)
        self.criterion: Optional[RTDETRCriterion] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.scaler: Optional[GradScaler] = None
        self.ema: Optional[ModelEMA] = None
        self.train_loader: Optional[DataLoader] = None

        # Paths
        self.save_dir: Optional[Path] = None

    def _setup_device(self) -> torch.device:
        """Setup and return training device."""
        device = self.config.device
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def setup(self):
        """Initialize all training components."""
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(self.config.project) / f"{self.config.name}_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "weights").mkdir(exist_ok=True)

        # Save config
        self.config.to_yaml(self.save_dir / "config.yaml")

        # Move model to device
        self.model.to(self.device)

        # Build criterion
        self.criterion = RTDETRCriterion(
            num_classes=self.config.num_classes,
            loss_vfl_weight=self.config.loss_vfl_weight,
            loss_bbox_weight=self.config.loss_bbox_weight,
            loss_giou_weight=self.config.loss_giou_weight,
            matcher_cost_class=self.config.matcher_cost_class,
            matcher_cost_bbox=self.config.matcher_cost_bbox,
            matcher_cost_giou=self.config.matcher_cost_giou,
        )

        # Build optimizer with parameter groups
        self.optimizer = self._build_optimizer()

        # Build scheduler (step LR at lr_drop_epochs)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.config.lr_drop_epochs, gamma=0.1
        )

        # AMP scaler
        if self.config.amp and self.device.type == "cuda":
            self.scaler = GradScaler()
            logger.info("Using AMP (Automatic Mixed Precision)")

        # EMA
        if self.config.ema:
            self.ema = ModelEMA(self.model, decay=self.config.ema_decay)
            logger.info(f"Using EMA with decay={self.config.ema_decay}")

        # Build data loader
        self.train_loader = self._build_dataloader()

        logger.info(f"Training on {self.device}")
        logger.info(f"Save directory: {self.save_dir}")
        logger.info(f"Dataset size: {len(self.train_loader.dataset)} images")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Epochs: {self.config.epochs}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer with separate backbone LR."""
        backbone_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name.lower():
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {
                "params": backbone_params,
                "lr": self.config.lr * self.config.backbone_lr_mult,
                "name": "backbone",
            },
            {"params": other_params, "lr": self.config.lr, "name": "other"},
        ]

        logger.info(
            f"Optimizer: AdamW with backbone_lr={self.config.lr * self.config.backbone_lr_mult:.6f}, "
            f"other_lr={self.config.lr:.6f}"
        )
        logger.info(
            f"  - Backbone params: {len(backbone_params)}, Other params: {len(other_params)}"
        )

        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )

    def _build_dataloader(self) -> DataLoader:
        """Build training DataLoader using YOLODataset."""
        # Load dataset config
        with open(self.config.data) as f:
            data_cfg = yaml.safe_load(f)

        data_root = Path(data_cfg.get("path", "."))
        if not data_root.is_absolute():
            # Make relative to data.yaml location
            data_root = Path(self.config.data).parent / data_root

        # Create dataset
        # YOLODataset expects img_size as tuple (height, width)
        img_size = (self.config.imgsz, self.config.imgsz)
        dataset = YOLODataset(
            data_dir=str(data_root),
            split="train",
            img_size=img_size,
            preproc=None,  # We handle preprocessing in collate_fn
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def _collate_fn(self, batch) -> tuple:
        """
        Custom collate function to convert YOLODataset format to RT-DETR format.

        YOLODataset returns: (img, target, info, img_id)
            - img: numpy array (H, W, 3) BGR, resized but aspect ratio preserved
            - target: numpy array (N, 5) with [x1, y1, x2, y2, class_id] in pixels

        RT-DETR format:
            - images: tensor (B, 3, H, W) normalized with ImageNet stats
            - targets: List[Dict] with 'labels' and 'boxes' (cxcywh normalized)
        """
        import numpy as np

        images = []
        targets = []

        target_h, target_w = self.config.imgsz, self.config.imgsz

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for img, target, info, img_id in batch:
            h, w = img.shape[:2]

            # Pad image to target size (640x640)
            padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            padded_img[:h, :w] = img

            # Convert BGR to RGB, then to tensor
            img_rgb = padded_img[:, :, ::-1].copy()  # BGR to RGB
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

            # Apply ImageNet normalization
            img_tensor = (img_tensor - mean) / std
            images.append(img_tensor)

            # Convert target from xyxy pixels to cxcywh normalized
            # target format: [x1, y1, x2, y2, class_id]
            if target is not None and len(target) > 0:
                labels = torch.from_numpy(target[:, 4].astype(np.int64)).long()

                # Convert xyxy to cxcywh and normalize by padded image size
                x1, y1, x2, y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]
                cx = (x1 + x2) / 2 / target_w
                cy = (y1 + y2) / 2 / target_h
                bw = (x2 - x1) / target_w
                bh = (y2 - y1) / target_h

                boxes = torch.from_numpy(
                    np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)
                )
                targets.append({"labels": labels, "boxes": boxes})
            else:
                targets.append(
                    {"labels": torch.zeros(0, dtype=torch.long), "boxes": torch.zeros(0, 4)}
                )

        images = torch.stack(images)
        return images, targets

    def train(self) -> Dict:
        """
        Main training loop.

        Returns:
            Dict with training results
        """
        self.setup()

        logger.info(f"Starting training for {self.config.epochs} epochs")
        start_time = time.time()

        for epoch in range(self.config.epochs):
            self.epoch = epoch

            # Train one epoch
            train_loss = self._train_epoch()

            # Step scheduler
            self.scheduler.step()

            # Save checkpoint
            is_best = train_loss < self.best_loss
            if is_best:
                self.best_loss = train_loss

            if (epoch + 1) % self.config.save_period == 0 or is_best or epoch == self.config.epochs - 1:
                self._save_checkpoint(is_best)

            # Log epoch summary
            lr = self.scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Loss: {train_loss:.4f} - "
                f"LR: {lr:.6f}"
            )

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 3600:.2f} hours")
        logger.info(f"Best loss: {self.best_loss:.4f}")
        logger.info(f"Saved to: {self.save_dir}")

        return {
            "best_loss": self.best_loss,
            "save_dir": str(self.save_dir),
            "total_time": total_time,
            "epochs": self.config.epochs,
        }

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {"vfl": 0.0, "bbox": 0.0, "giou": 0.0}

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.config.epochs}",
            total=len(self.train_loader),
        )

        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device, non_blocking=True)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict["loss"]

                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_grad
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict["loss"]

                self.optimizer.zero_grad()
                loss.backward()

                if self.config.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.clip_grad
                    )

                self.optimizer.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update(self.model)

            # Track losses
            total_loss += loss.item()
            loss_components["vfl"] += loss_dict["loss_vfl"].item()
            loss_components["bbox"] += loss_dict["loss_bbox"].item()
            loss_components["giou"] += loss_dict["loss_giou"].item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "vfl": f"{loss_dict['loss_vfl'].item():.4f}",
                    "bbox": f"{loss_dict['loss_bbox'].item():.4f}",
                    "giou": f"{loss_dict['loss_giou'].item():.4f}",
                }
            )

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def _save_checkpoint(self, is_best: bool):
        """Save training checkpoint."""
        model_to_save = self.ema.ema if self.ema else self.model

        checkpoint = {
            "epoch": self.epoch,
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config.to_dict(),
        }

        if self.scaler:
            checkpoint["scaler"] = self.scaler.state_dict()

        # Save last checkpoint
        last_path = self.save_dir / "weights" / "last.pt"
        torch.save(checkpoint, last_path)

        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / "weights" / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved (loss: {self.best_loss:.4f})")

        # Save periodic checkpoint
        if (self.epoch + 1) % self.config.save_period == 0:
            epoch_path = self.save_dir / "weights" / f"epoch_{self.epoch + 1}.pt"
            torch.save(checkpoint, epoch_path)

    def resume(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        logger.info(f"Resuming from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        if self.optimizer is not None and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        logger.info(f"Resumed from epoch {self.epoch}")

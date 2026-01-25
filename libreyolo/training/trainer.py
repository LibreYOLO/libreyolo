"""
YOLOX Trainer for LibreYOLO.

Provides a simplified training loop with essential features.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import YOLOXTrainConfig
from .scheduler import LRScheduler
from .ema import ModelEMA
from .augment import TrainTransform, MosaicMixupDataset
from .dataset import YOLODataset, COCODataset, create_dataloader, load_data_config


logger = logging.getLogger(__name__)


class YOLOXTrainer:
    """
    YOLOX Trainer.

    Handles the complete training loop with:
    - Mixed precision training (AMP)
    - Exponential Moving Average (EMA)
    - Learning rate scheduling
    - Mosaic/Mixup augmentation
    - Checkpoint saving
    - TensorBoard logging (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        config: YOLOXTrainConfig,
    ):
        """
        Initialize trainer.

        Args:
            model: YOLOX model to train
            config: Training configuration
        """
        self.model = model
        self.config = config

        # Setup device
        self.device = self._setup_device()

        # Training state
        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.best_loss = float('inf')

        # Will be initialized in setup()
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.ema_model = None
        self.train_loader = None
        self.val_loader = None
        self.tensorboard_writer = None

    def _setup_device(self) -> torch.device:
        """Setup and return the device for training."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)

        logger.info(f"Using device: {device}")
        return device

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        # Separate parameters for different weight decay
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases, no weight decay
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)  # BN weights, no weight decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # conv weights, with weight decay

        lr = self.config.effective_lr

        if self.config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                pg0,
                lr=lr,
                momentum=self.config.momentum,
                nesterov=self.config.nesterov,
            )
        elif self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(pg0, lr=lr)
        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(pg0, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Add parameter groups with appropriate weight decay
        optimizer.add_param_group({"params": pg1, "lr": lr, "weight_decay": self.config.weight_decay})
        optimizer.add_param_group({"params": pg2, "lr": lr})

        logger.info(f"Optimizer: {self.config.optimizer}")
        logger.info(f"  - pg0 (BN): {len(pg0)} params")
        logger.info(f"  - pg1 (Conv, wd={self.config.weight_decay}): {len(pg1)} params")
        logger.info(f"  - pg2 (Bias): {len(pg2)} params")

        return optimizer

    def _setup_scheduler(self, iters_per_epoch: int) -> LRScheduler:
        """Create learning rate scheduler."""
        scheduler = LRScheduler(
            name=self.config.scheduler,
            lr=self.config.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            no_aug_epochs=self.config.no_aug_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )
        return scheduler

    def _setup_data(self):
        """Setup training and validation data loaders."""
        img_size = self.config.input_size

        # Create preprocessing transform
        preproc = TrainTransform(
            max_labels=50,
            flip_prob=self.config.flip_prob,
            hsv_prob=self.config.hsv_prob,
        )

        # Determine dataset type and create datasets
        if self.config.data:
            # Load from data.yaml
            data_cfg = load_data_config(self.config.data)
            data_dir = data_cfg['root']
            self.num_classes = data_cfg.get('nc', self.config.num_classes)

            # Determine format (YOLO or COCO)
            if (Path(data_dir) / "annotations").exists():
                # COCO format
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name="train2017",
                    img_size=img_size,
                    preproc=preproc,
                )
            else:
                # YOLO format
                train_dataset = YOLODataset(
                    data_dir=data_dir,
                    split="train",
                    img_size=img_size,
                    preproc=preproc,
                )
        elif self.config.data_dir:
            # Direct path to dataset
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
            raise ValueError("Either 'data' or 'data_dir' must be specified in config")

        # Wrap with mosaic/mixup augmentation
        train_dataset = MosaicMixupDataset(
            dataset=train_dataset,
            img_size=img_size,
            mosaic=True,
            preproc=preproc,
            degrees=self.config.degrees,
            translate=self.config.translate,
            mosaic_scale=self.config.mosaic_scale,
            mixup_scale=self.config.mixup_scale,
            shear=self.config.shear,
            enable_mixup=True,
            mosaic_prob=self.config.mosaic_prob,
            mixup_prob=self.config.mixup_prob,
        )

        # Create data loader
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
        )

        logger.info(f"Training dataset: {len(train_dataset)} images")
        logger.info(f"Iterations per epoch: {len(self.train_loader)}")

        return train_dataset

    def setup(self):
        """Setup all training components."""
        logger.info("Setting up training...")

        # Move model to device
        self.model.to(self.device)

        # Initialize head biases for better convergence
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'initialize_biases'):
            self.model.head.initialize_biases(0.01)

        # Setup data
        train_dataset = self._setup_data()

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        self.lr_scheduler = self._setup_scheduler(len(self.train_loader))

        # Setup AMP scaler
        if self.config.amp and self.device.type == "cuda":
            self.scaler = GradScaler()
            logger.info("Using mixed precision training (AMP)")
        else:
            self.scaler = None

        # Setup EMA
        if self.config.ema:
            self.ema_model = ModelEMA(self.model, decay=self.config.ema_decay)
            logger.info(f"Using EMA with decay={self.config.ema_decay}")

        # Setup save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(self.config.save_dir) / f"exp_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.to_yaml(self.save_dir / "config.yaml")

        # Setup TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(self.save_dir / "tensorboard")
            logger.info(f"TensorBoard logging to {self.save_dir / 'tensorboard'}")
        except ImportError:
            self.tensorboard_writer = None
            logger.info("TensorBoard not available")

        logger.info(f"Saving to: {self.save_dir}")

    def train(self) -> Dict:
        """
        Run the training loop.

        Returns:
            Dictionary with training results
        """
        self.setup()

        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Model: YOLOX-{self.config.size}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.effective_lr}")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Disable mosaic in final epochs
            if epoch == self.config.epochs - self.config.no_aug_epochs:
                logger.info(f"Disabling mosaic/mixup for final {self.config.no_aug_epochs} epochs")
                self.train_loader.dataset.close_mosaic()
                self.model.head.use_l1 = True

            # Train one epoch
            epoch_loss = self._train_epoch(epoch)

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0 or epoch == self.config.epochs - 1:
                self._save_checkpoint(epoch, epoch_loss)

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 3600:.2f} hours")

        # Close tensorboard
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

        return {
            "best_loss": self.best_loss,
            "total_epochs": self.config.epochs,
            "total_time": total_time,
            "save_dir": str(self.save_dir),
        }

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
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

            # Move to device
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(imgs, targets)
                    loss = outputs["total_loss"]

                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(imgs, targets)
                loss = outputs["total_loss"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update EMA
            if self.ema_model is not None:
                self.ema_model.update(self.model)

            # Track metrics and save values for logging before deleting
            loss_val = loss.item()
            iou_loss_val = outputs.get('iou_loss', 0)
            obj_loss_val = outputs.get('obj_loss', 0)
            cls_loss_val = outputs.get('cls_loss', 0)
            total_loss += loss_val

            # Free memory to prevent GPU memory leak
            del outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Update learning rate
            lr = self.lr_scheduler.update_lr(self.current_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lr": f"{lr:.6f}",
                "iou": f"{iou_loss_val:.4f}",
                "obj": f"{obj_loss_val:.4f}",
                "cls": f"{cls_loss_val:.4f}",
            })

            # Log to TensorBoard
            if self.tensorboard_writer and batch_idx % self.config.log_interval == 0:
                self.tensorboard_writer.add_scalar("train/loss", loss.item(), self.current_iter)
                self.tensorboard_writer.add_scalar("train/lr", lr, self.current_iter)
                for key in ["iou_loss", "obj_loss", "cls_loss", "l1_loss"]:
                    if key in outputs:
                        self.tensorboard_writer.add_scalar(f"train/{key}", outputs[key], self.current_iter)

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

        # Log epoch metrics
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("epoch/loss", avg_loss, epoch)

        return avg_loss

    def _save_checkpoint(self, epoch: int, loss: float):
        """Save training checkpoint."""
        # Determine which model to save (EMA if available)
        if self.ema_model is not None:
            model_to_save = self.ema_model.ema
        else:
            model_to_save = self.model

        checkpoint = {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "loss": loss,
        }

        # Save latest checkpoint
        latest_path = self.save_dir / "last.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.save_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with loss: {loss:.4f}")

        # Save epoch checkpoint
        epoch_path = self.save_dir / f"epoch_{epoch + 1}.pt"
        torch.save(checkpoint, epoch_path)

        logger.info(f"Checkpoint saved: {latest_path}")

    def resume(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        logger.info(f"Resuming from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.start_epoch = checkpoint["epoch"] + 1

        if self.optimizer is not None and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "loss" in checkpoint:
            self.best_loss = checkpoint["loss"]

        logger.info(f"Resumed from epoch {self.start_epoch}")

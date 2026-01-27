"""
YOLOX Trainer for LibreYOLO.

Provides a simplified training loop with essential features.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
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
        wrapper_model: Optional[Any] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: YOLOX model to train
            config: Training configuration
            wrapper_model: LibreYOLO wrapper model (for validation), optional
        """
        self.model = model
        self.config = config
        self.wrapper_model = wrapper_model  # Used for validation

        # Setup device
        self.device = self._setup_device()

        # Training state
        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0

        # Metric tracking (following API spec)
        self.best_mAP50_95 = 0.0
        self.best_mAP50 = 0.0
        self.best_epoch = 0
        self.final_loss = 0.0
        self.patience_counter = 0

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

    def _get_save_dir(self) -> Path:
        """
        Get save directory with auto-incrementing exp names.

        Returns:
            Path to save directory (e.g., runs/train/exp, runs/train/exp2, ...)
        """
        project = Path(self.config.project)
        name = self.config.name

        if self.config.exist_ok:
            # Overwrite existing directory
            save_dir = project / name
        else:
            # Auto-increment: exp, exp2, exp3, ...
            save_dir = project / name
            if save_dir.exists():
                # Find next available number
                i = 2
                while (project / f"{name}{i}").exists():
                    i += 1
                save_dir = project / f"{name}{i}"

        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

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
                # YOLO format - construct paths from data.yaml
                train_path = data_cfg.get('train', 'images/train')

                # Full path to training images
                train_img_dir = Path(data_dir) / train_path

                # Collect image files
                img_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    img_files.extend(train_img_dir.glob(ext))
                    img_files.extend(train_img_dir.glob(ext.upper()))
                img_files = sorted(img_files)

                if len(img_files) == 0:
                    raise FileNotFoundError(f"No images found in {train_img_dir}")

                # Infer label paths (replace 'images' with 'labels', change extension to .txt)
                label_files = []
                for img_file in img_files:
                    # Replace /images/ with /labels/ and .jpg with .txt
                    label_file = Path(str(img_file).replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt')
                    label_files.append(label_file)

                # Create dataset using file list mode
                train_dataset = YOLODataset(
                    img_files=img_files,
                    label_files=label_files,
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
            batch_size=self.config.batch,
            num_workers=self.config.workers,
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

        # Setup save directory with auto-incrementing names
        self.save_dir = self._get_save_dir()

        # Save config
        self.config.to_yaml(self.save_dir / "train_config.yaml")  # Renamed to match spec

        # Setup TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(self.save_dir / "tensorboard")
            logger.info(f"TensorBoard logging to {self.save_dir / 'tensorboard'}")
        except Exception as e:
            # Handle ImportError, protobuf compatibility issues, or any other TensorBoard errors
            self.tensorboard_writer = None
            logger.warning(f"TensorBoard not available (skipping): {type(e).__name__}")
            logger.info("Training will continue without TensorBoard logging")

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
        logger.info(f"Batch size: {self.config.batch}")
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
            epoch_loss, val_metrics = self._train_epoch(epoch)

            # Track final loss (for return value)
            self.final_loss = epoch_loss

            # Save checkpoint
            if (epoch + 1) % self.config.save_period == 0 or epoch == self.config.epochs - 1:
                self._save_checkpoint(epoch, epoch_loss, val_metrics)

            # Early stopping check
            if self.patience_counter >= self.config.patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(patience={self.config.patience}, no improvement for {self.patience_counter} epochs)"
                )
                break

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 3600:.2f} hours")

        # Close tensorboard
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

        # Return results matching API spec format
        weights_dir = self.save_dir / "weights"
        return {
            'final_loss': self.final_loss,
            'best_mAP50': self.best_mAP50,
            'best_mAP50_95': self.best_mAP50_95,
            'best_epoch': self.best_epoch,
            'save_dir': str(self.save_dir),
            'best_checkpoint': str(weights_dir / 'best.pt'),
            'last_checkpoint': str(weights_dir / 'last.pt'),
        }

    def _validate_epoch(self, epoch: int) -> Optional[Dict[str, float]]:
        """
        Run validation and return metrics.

        Returns:
            dict with keys: mAP50, mAP50_95, or None if validation failed
        """
        try:
            from libreyolo.validation import DetectionValidator, ValidationConfig

            logger.info(f"Running validation for epoch {epoch + 1}")

            # Create validation config
            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=self.config.batch,
                imgsz=self.config.imgsz,  # Fixed: was img_size, should be imgsz
                conf_thres=0.001,
                iou_thres=0.65,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                verbose=False,  # Reduce validation output during training
            )

            # For validation, we need the wrapper model, not the raw PyTorch model
            # The wrapper model has methods like _get_val_preprocessor() that validator needs
            if self.wrapper_model is None:
                logger.error("Validation requires wrapper_model to be provided to trainer")
                return None

            # Create a temporary wrapper with EMA model if available
            # We'll use the wrapper's methods but swap out the underlying model
            eval_pytorch_model = self.ema_model.ema if self.ema_model else self.model

            # Temporarily swap the model for validation
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model

            try:
                # Create validator with the wrapper model
                validator = DetectionValidator(
                    model=self.wrapper_model,
                    config=val_config,
                )

                # Run validation (MUST be inside try block so model swap is active)
                results = validator.run()
            finally:
                # Restore original model
                self.wrapper_model.model = original_model

            # Debug: print what we got back
            print(f"[DEBUG] Validation results keys: {list(results.keys())}")

            # Extract metrics - results has keys like 'metrics/mAP50' and 'metrics/mAP50-95'
            metrics = {
                'mAP50': results.get('metrics/mAP50', 0.0),
                'mAP50_95': results.get('metrics/mAP50-95', 0.0),
            }

            print(f"[DEBUG] Extracted metrics: mAP50={metrics['mAP50']:.4f}, mAP50_95={metrics['mAP50_95']:.4f}")

            # Log to user (using print since logger may not show)
            print(f"Validation - mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50_95']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback
            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            return None

    def _train_epoch(self, epoch: int) -> Tuple[float, Optional[Dict[str, float]]]:
        """
        Train for one epoch and optionally validate.

        Returns:
            Tuple of (average_loss, validation_metrics)
        """
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
            l1_loss_val = outputs.get('l1_loss', 0)
            total_loss += loss_val

            # Free memory (don't call empty_cache every iteration - it's slow)
            del outputs, loss

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

            # Log to TensorBoard (use saved values, not deleted variables)
            if self.tensorboard_writer and batch_idx % self.config.log_interval == 0:
                self.tensorboard_writer.add_scalar("train/loss", loss_val, self.current_iter)
                self.tensorboard_writer.add_scalar("train/lr", lr, self.current_iter)
                self.tensorboard_writer.add_scalar("train/iou_loss", iou_loss_val, self.current_iter)
                self.tensorboard_writer.add_scalar("train/obj_loss", obj_loss_val, self.current_iter)
                self.tensorboard_writer.add_scalar("train/cls_loss", cls_loss_val, self.current_iter)
                if l1_loss_val:
                    self.tensorboard_writer.add_scalar("train/l1_loss", l1_loss_val, self.current_iter)

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")

        # Log epoch metrics
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("epoch/loss", avg_loss, epoch)

        # Run validation if configured
        val_metrics = None
        if self.config.eval_interval > 0 and (epoch + 1) % self.config.eval_interval == 0:
            val_metrics = self._validate_epoch(epoch)

            # Log validation metrics to TensorBoard
            if val_metrics and self.tensorboard_writer:
                self.tensorboard_writer.add_scalar("val/mAP50", val_metrics['mAP50'], epoch)
                self.tensorboard_writer.add_scalar("val/mAP50_95", val_metrics['mAP50_95'], epoch)

        return avg_loss, val_metrics

    def _save_checkpoint(self, epoch: int, loss: float, val_metrics: Optional[Dict[str, float]] = None):
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch number
            loss: Training loss for this epoch
            val_metrics: Validation metrics dict with mAP50, mAP50_95 (optional)
        """
        # Determine which model to save (EMA if available)
        if self.ema_model is not None:
            model_to_save = self.ema_model.ema
        else:
            model_to_save = self.model

        # Create checkpoint dict
        checkpoint = {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "loss": loss,
            "best_mAP50_95": self.best_mAP50_95,
            "best_mAP50": self.best_mAP50,
            "best_epoch": self.best_epoch,
        }

        # Save EMA state if available
        if self.ema_model is not None:
            checkpoint["ema_updates"] = self.ema_model.updates

        # Create weights directory if it doesn't exist
        weights_dir = self.save_dir / "weights"
        weights_dir.mkdir(exist_ok=True)

        # Save latest checkpoint
        latest_path = weights_dir / "last.pt"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint based on mAP (if validation metrics available)
        print(f"[DEBUG] _save_checkpoint: val_metrics={val_metrics}, best_mAP50_95={self.best_mAP50_95}")
        if val_metrics and val_metrics['mAP50_95'] > self.best_mAP50_95:
            self.best_mAP50_95 = val_metrics['mAP50_95']
            self.best_mAP50 = val_metrics['mAP50']
            self.best_epoch = epoch + 1
            self.patience_counter = 0  # Reset patience counter on improvement

            best_path = weights_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(
                f"New best model saved - Epoch {epoch + 1}: "
                f"mAP50={self.best_mAP50:.4f}, mAP50-95={self.best_mAP50_95:.4f}"
            )
        elif val_metrics:
            self.patience_counter += 1

        # Save epoch checkpoint (every save_period epochs)
        if (epoch + 1) % self.config.save_period == 0:
            epoch_path = weights_dir / f"epoch_{epoch + 1}.pt"
            torch.save(checkpoint, epoch_path)

        logger.info(f"Checkpoint saved: {latest_path}")

    def resume(self, checkpoint_path: str):
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.pt)

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint is incompatible
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        logger.info(f"Resuming from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        try:
            self.model.load_state_dict(checkpoint["model"])
        except Exception as e:
            raise RuntimeError(f"Cannot resume: model architecture mismatch - {e}")

        # Load training state
        self.start_epoch = checkpoint["epoch"] + 1

        # Load optimizer if available
        if self.optimizer is not None and "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info("Optimizer state restored")
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}")

        # Load best metrics (following API spec tracking)
        if "best_mAP50_95" in checkpoint:
            self.best_mAP50_95 = checkpoint["best_mAP50_95"]
            self.best_mAP50 = checkpoint.get("best_mAP50", 0.0)
            self.best_epoch = checkpoint.get("best_epoch", 0)
            logger.info(
                f"Restored best metrics: mAP50={self.best_mAP50:.4f}, "
                f"mAP50-95={self.best_mAP50_95:.4f} (epoch {self.best_epoch})"
            )
        elif "loss" in checkpoint:
            # Legacy checkpoint with loss tracking
            logger.warning("Old checkpoint format detected (loss-based). Converting to mAP tracking.")
            self.best_mAP50_95 = 0.0
            self.best_mAP50 = 0.0
            self.best_epoch = 0

        # Load EMA state if available
        if self.ema_model and "ema_updates" in checkpoint:
            self.ema_model.updates = checkpoint["ema_updates"]
            logger.info(f"EMA updates restored: {self.ema_model.updates}")

        # Reset patience counter on resume
        self.patience_counter = 0

        logger.info(
            f"Resumed from epoch {self.start_epoch} "
            f"(will train to epoch {self.config.epochs})"
        )

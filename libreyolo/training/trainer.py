"""Base trainer for LibreYOLO models.

Model-specific trainers subclass BaseTrainer and override hooks.
"""

import logging
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .config import TrainConfig
from .ema import ModelEMA
from ..data.dataset import YOLODataset, COCODataset, create_dataloader
from ..data import load_data_config, get_img_files, img2label_paths
from ..utils.serialization import load_trusted_torch_file


logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Base trainer for all LibreYOLO model families.

    Subclasses override hook methods to customise transforms, schedulers,
    loss extraction, and family-specific behaviour.
    """

    def __init__(
        self,
        model: nn.Module,
        wrapper_model: Optional[Any] = None,
        **kwargs,
    ):
        self.config = self._config_class().from_kwargs(**kwargs)
        self.model = model
        self.wrapper_model = wrapper_model

        # Device
        self.device = self._setup_device()

        # Training state
        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0

        # Metric tracking
        self.best_mAP50_95 = 0.0
        self.best_mAP50 = 0.0
        self.best_epoch = 0
        self.final_loss = 0.0
        self.epoch_losses: List[float] = []
        self.patience_counter = 0

        # Initialised in setup()
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.ema_model = None
        self.train_loader = None
        self.tensorboard_writer = None
        self._is_setup = False

    # =========================================================================
    # Config
    # =========================================================================

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        """Return the config dataclass for this trainer. Subclasses override."""
        return TrainConfig

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def effective_lr(self) -> float:
        """Learning rate scaled by batch size (linear scaling rule)."""
        return self.config.lr0 * self.config.batch / 64

    @property
    def input_size(self) -> Tuple[int, int]:
        return (self.config.imgsz, self.config.imgsz)

    # =========================================================================
    # Hook methods — subclasses override these
    # =========================================================================

    @abstractmethod
    def get_model_family(self) -> str:
        """Return canonical model family string for checkpoint metadata."""

    @abstractmethod
    def get_model_tag(self) -> str:
        """Return human-readable model tag for log messages (e.g. 'YOLOX-s')."""

    @abstractmethod
    def create_transforms(self):
        """Return (preproc_transform, mosaic_dataset_class)."""

    @abstractmethod
    def create_scheduler(self, iters_per_epoch: int):
        """Return a scheduler with an ``update_lr(iters)`` method."""

    @abstractmethod
    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        """Extract per-component losses for progress bar / TensorBoard.

        Returns:
            Dict mapping loss name → scalar value.
        """

    def on_setup(self):
        """Called after model is on device, before data setup (e.g. bias init)."""

    def on_mosaic_disable(self):
        """Called when mosaic is disabled for final no-aug epochs."""
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            self.train_loader.dataset.close_mosaic()

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Run the model forward pass. Override if call signature differs."""
        return self.model(imgs, targets)

    # =========================================================================
    # Shared infrastructure
    # =========================================================================

    def _setup_device(self) -> torch.device:
        device_str = str(self.config.device).strip().lower()
        if device_str in ("", "auto"):
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            if "," in device_str:
                raise NotImplementedError(
                    f"Multi-GPU training is not supported yet "
                    f"(got device={self.config.device!r}). Pass a single "
                    "index like '0' or 'cuda:0'."
                )
            # YOLO-style "0" -> "cuda:0"
            if device_str.isdigit():
                device_str = f"cuda:{device_str}"
            device = torch.device(device_str)
        logger.info(f"Using device: {device}")
        return device

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        pg0, pg1, pg2 = [], [], []
        for _k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)

        lr = self.effective_lr
        opt_name = self.config.optimizer

        if opt_name == "sgd":
            optimizer = torch.optim.SGD(
                pg0,
                lr=lr,
                momentum=self.config.momentum,
                nesterov=self.config.nesterov,
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(pg0, lr=lr)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(pg0, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        optimizer.add_param_group(
            {"params": pg1, "lr": lr, "weight_decay": self.config.weight_decay}
        )
        optimizer.add_param_group({"params": pg2, "lr": lr})

        logger.info(f"Optimizer: {opt_name}")
        logger.info(f"  - pg0 (BN): {len(pg0)} params")
        logger.info(f"  - pg1 (Conv, wd={self.config.weight_decay}): {len(pg1)} params")
        logger.info(f"  - pg2 (Bias): {len(pg2)} params")
        return optimizer

    def _get_save_dir(self) -> Path:
        project = Path(self.config.project)
        name = self.config.name

        save_dir = project / name
        if not self.config.exist_ok and save_dir.exists():
            i = 2
            while (project / f"{name}{i}").exists():
                i += 1
            save_dir = project / f"{name}{i}"

        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _setup_data(self):
        img_size = self.input_size
        preproc, MosaicDatasetClass = self.create_transforms()

        if self.config.data:
            data_cfg = load_data_config(
                self.config.data,
                allow_scripts=self.config.allow_download_scripts,
            )
            data_dir = data_cfg["root"]
            self.num_classes = data_cfg.get("nc", self.config.num_classes)

            ann_file = Path(data_dir) / "annotations" / "instances_train2017.json"

            # Prefer pre-resolved file lists from load_data_config (.txt format)
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
                train_img_dir = Path(train_path)
                if not train_img_dir.is_absolute():
                    train_img_dir = Path(data_dir) / train_img_dir

                try:
                    img_files = get_img_files(train_path, prefix=data_dir)
                except (FileNotFoundError, ValueError):
                    img_files = []

                if len(img_files) == 0:
                    raise FileNotFoundError(f"No images found in {train_img_dir}")

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

    # =========================================================================
    # Setup / train / epoch
    # =========================================================================

    def setup(self):
        if self._is_setup:
            return

        logger.info("Setting up training...")
        self.model.to(self.device)

        self.on_setup()

        self._setup_data()
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self.create_scheduler(len(self.train_loader))

        if self.config.amp and self.device.type == "cuda":
            self.scaler = GradScaler("cuda")
            logger.info("Using mixed precision training (AMP)")
        else:
            self.scaler = None

        if self.config.ema:
            self.ema_model = ModelEMA(self.model, decay=self.config.ema_decay)
            logger.info(f"Using EMA with decay={self.config.ema_decay}")

        self.save_dir = self._get_save_dir()

        self.config.to_yaml(self.save_dir / "train_config.yaml")

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard_writer = SummaryWriter(self.save_dir / "tensorboard")
            logger.info(f"TensorBoard logging to {self.save_dir / 'tensorboard'}")
        except Exception as e:
            self.tensorboard_writer = None
            logger.warning(f"TensorBoard not available (skipping): {type(e).__name__}")
            logger.info("Training will continue without TensorBoard logging")

        logger.info(f"Saving to: {self.save_dir}")
        self._is_setup = True

    def train(self) -> Dict:
        self.setup()

        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Model: {self.get_model_tag()}")
        logger.info(f"Batch size: {self.config.batch}")
        logger.info(f"Learning rate: {self.effective_lr}")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.config.epochs):
            self.current_epoch = epoch

            if epoch == self.config.epochs - self.config.no_aug_epochs:
                logger.info(
                    f"Disabling mosaic/mixup for final {self.config.no_aug_epochs} epochs"
                )
                self.on_mosaic_disable()

            epoch_loss, val_metrics = self._train_epoch(epoch)
            self.final_loss = epoch_loss
            self.epoch_losses.append(epoch_loss)

            if (
                epoch + 1
            ) % self.config.save_period == 0 or epoch == self.config.epochs - 1:
                self._save_checkpoint(epoch, epoch_loss, val_metrics)

            if self.patience_counter >= self.config.patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(patience={self.config.patience}, no improvement for {self.patience_counter} epochs)"
                )
                break

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 3600:.2f} hours")

        if self.tensorboard_writer:
            self.tensorboard_writer.close()

        weights_dir = self.save_dir / "weights"
        return {
            "final_loss": self.final_loss,
            "epoch_losses": list(self.epoch_losses),
            "best_mAP50": self.best_mAP50,
            "best_mAP50_95": self.best_mAP50_95,
            "best_epoch": self.best_epoch,
            "save_dir": str(self.save_dir),
            "best_checkpoint": str(weights_dir / "best.pt"),
            "last_checkpoint": str(weights_dir / "last.pt"),
        }

    def _scale_lr(self, base_lr: float, param_group: dict) -> float:
        """Hook for per-group LR scaling. Override in subclasses."""
        return base_lr

    def _train_epoch(self, epoch: int) -> Tuple[float, Optional[Dict[str, float]]]:
        self.model.train()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            total=len(self.train_loader),
            disable=not sys.stderr.isatty(),
            file=sys.stderr,
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

            # LR update
            lr = self.lr_scheduler.update_lr(self.current_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self._scale_lr(lr, param_group)
            num_batches += 1

            # Progress bar
            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{lr:.6f}"}
            postfix.update({k: f"{v:.4f}" for k, v in loss_components.items()})
            pbar.set_postfix(postfix)

            # TensorBoard
            if self.tensorboard_writer and batch_idx % self.config.log_interval == 0:
                self.tensorboard_writer.add_scalar(
                    "train/loss", loss_val, self.current_iter
                )
                self.tensorboard_writer.add_scalar("train/lr", lr, self.current_iter)
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

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_epoch(self, epoch: int) -> Optional[Dict[str, float]]:
        try:
            from libreyolo.validation import DetectionValidator, ValidationConfig

            logger.info(f"Running validation for epoch {epoch + 1}")

            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=self.config.batch,
                imgsz=self.config.imgsz,
                conf_thres=0.001,
                iou_thres=0.65,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                verbose=False,
                num_workers=self.config.workers,
            )

            if self.wrapper_model is None:
                logger.error(
                    "Validation requires wrapper_model to be provided to trainer"
                )
                return None

            eval_pytorch_model = self.ema_model.ema if self.ema_model else self.model
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model

            try:
                validator = DetectionValidator(
                    model=self.wrapper_model, config=val_config
                )
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            metrics = {
                "mAP50": results.get("metrics/mAP50", 0.0),
                "mAP50_95": results.get("metrics/mAP50-95", 0.0),
            }

            logger.debug(
                f"Extracted metrics: mAP50={metrics['mAP50']:.4f}, mAP50_95={metrics['mAP50_95']:.4f}"
            )
            logger.info(
                "Validation - mAP50: %.4f, mAP50-95: %.4f",
                metrics["mAP50"],
                metrics["mAP50_95"],
            )
            return metrics

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            import traceback

            logger.debug(f"Validation traceback:\n{traceback.format_exc()}")
            return None

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def _save_checkpoint(
        self, epoch: int, loss: float, val_metrics: Optional[Dict[str, float]] = None
    ):
        model_to_save = self.ema_model.ema if self.ema_model else self.model

        checkpoint = {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "loss": loss,
            "best_mAP50_95": self.best_mAP50_95,
            "best_mAP50": self.best_mAP50,
            "best_epoch": self.best_epoch,
            "nc": self.config.num_classes,
            "size": self.config.size,
            "model_family": self.get_model_family(),
        }
        if self.wrapper_model is not None:
            checkpoint["names"] = self.wrapper_model.names
        if self.ema_model is not None:
            checkpoint["ema_updates"] = self.ema_model.updates

        weights_dir = self.save_dir / "weights"
        weights_dir.mkdir(exist_ok=True)

        latest_path = weights_dir / "last.pt"
        torch.save(checkpoint, latest_path)

        if val_metrics and val_metrics["mAP50_95"] > self.best_mAP50_95:
            self.best_mAP50_95 = val_metrics["mAP50_95"]
            self.best_mAP50 = val_metrics["mAP50"]
            self.best_epoch = epoch + 1
            self.patience_counter = 0
            best_path = weights_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(
                f"New best model saved - Epoch {epoch + 1}: "
                f"mAP50={self.best_mAP50:.4f}, mAP50-95={self.best_mAP50_95:.4f}"
            )
        elif val_metrics:
            self.patience_counter += 1

        if (epoch + 1) % self.config.save_period == 0:
            epoch_path = weights_dir / f"epoch_{epoch + 1}.pt"
            torch.save(checkpoint, epoch_path)

        logger.info(f"Checkpoint saved: {latest_path}")

    def resume(self, checkpoint_path: str):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        logger.info(f"Resuming from {checkpoint_path}")
        checkpoint = load_trusted_torch_file(
            checkpoint_path,
            map_location=self.device,
            context="training resume checkpoint",
        )

        try:
            self.model.load_state_dict(checkpoint["model"])
        except Exception as e:
            raise RuntimeError(f"Cannot resume: model architecture mismatch - {e}")

        self.start_epoch = checkpoint["epoch"] + 1

        if self.optimizer is not None and "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info("Optimizer state restored")
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}")

        if "best_mAP50_95" in checkpoint:
            self.best_mAP50_95 = checkpoint["best_mAP50_95"]
            self.best_mAP50 = checkpoint.get("best_mAP50", 0.0)
            self.best_epoch = checkpoint.get("best_epoch", 0)
            logger.info(
                f"Restored best metrics: mAP50={self.best_mAP50:.4f}, "
                f"mAP50-95={self.best_mAP50_95:.4f} (epoch {self.best_epoch})"
            )
        elif "loss" in checkpoint:
            logger.warning(
                "Old checkpoint format detected (loss-based). Converting to mAP tracking."
            )
            self.best_mAP50_95 = 0.0
            self.best_mAP50 = 0.0
            self.best_epoch = 0

        if self.ema_model and "ema_updates" in checkpoint:
            self.ema_model.updates = checkpoint["ema_updates"]
            logger.info(f"EMA updates restored: {self.ema_model.updates}")

        self.patience_counter = 0
        logger.info(
            f"Resumed from epoch {self.start_epoch} "
            f"(will train to epoch {self.config.epochs})"
        )

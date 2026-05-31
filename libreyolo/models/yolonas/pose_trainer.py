"""YOLO-NAS pose-estimation trainer for native LibreYOLO training.

Unlike the detection trainer this trainer owns its data pipeline: it builds
:class:`~libreyolo.data.YOLOPoseDataset` loaders directly (keypoint-aware
transforms, padded ``(B, max_labels, 5 + 3K)`` targets) rather than going
through the shared mosaic/detection dataset path.

best.pt is selected by keypoint OKS-AP when metric validation is available,
with validation loss as the fallback signal.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, Type

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ...data import (
    YOLOPoseDataset,
    get_img_files,
    img2label_paths,
    load_data_config,
    pose_collate_fn,
)
from ...training.config import TrainConfig, YOLONASPoseConfig
from ...training.scheduler import CosineAnnealingScheduler
from ...training.trainer import BaseTrainer
from .loss import YoloNASPoseLoss
from .pose_transforms import YOLONASPoseTrainTransform, YOLONASPoseValTransform

logger = logging.getLogger(__name__)

# COCO 17-keypoint OKS sigmas — the upstream defaults.
_COCO17_OKS_SIGMAS = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
]


def _pose_worker_init_fn(worker_id: int) -> None:
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def default_oks_sigmas(num_keypoints: int) -> list[float]:
    """Per-keypoint OKS sigmas for COCO-style pose evaluation."""
    if num_keypoints == 17:
        return list(_COCO17_OKS_SIGMAS)
    return [1.0 / num_keypoints] * num_keypoints


class YOLONASPoseTrainer(BaseTrainer):
    """Trainer for YOLO-NAS pose models."""

    artifact_model_families = ("yolonas",)
    best_metric_key = "metrics/keypoints_mAP50-95"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return YOLONASPoseConfig

    def get_model_family(self) -> str:
        return "yolonas"

    def get_model_tag(self) -> str:
        return f"YOLO-NAS-Pose-{self.config.size}"

    @property
    def num_keypoints(self) -> int:
        return self.config.num_keypoints

    @property
    def effective_lr(self) -> float:
        return self.config.lr0

    # create_transforms is abstract on BaseTrainer; the pose trainer overrides
    # _setup_data entirely, so this hook is never exercised.
    def create_transforms(self):
        return None, None

    def create_scheduler(self, iters_per_epoch: int):
        return CosineAnnealingScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def _resolve_oks_sigmas(self) -> list[float]:
        sigmas = self.config.oks_sigmas
        if sigmas is not None:
            if len(sigmas) != self.num_keypoints:
                raise ValueError(
                    f"oks_sigmas has {len(sigmas)} entries but the dataset has "
                    f"{self.num_keypoints} keypoints"
                )
            return [float(s) for s in sigmas]
        return default_oks_sigmas(self.num_keypoints)

    def on_setup(self):
        self.loss_fn = YoloNASPoseLoss(
            oks_sigmas=self._resolve_oks_sigmas(),
            classification_loss_type=self.config.classification_loss_type,
            regression_iou_loss_type=self.config.regression_iou_loss_type,
            classification_loss_weight=self.config.classification_loss_weight,
            iou_loss_weight=self.config.iou_loss_weight,
            dfl_loss_weight=self.config.dfl_loss_weight,
            pose_cls_loss_weight=self.config.pose_cls_loss_weight,
            pose_reg_loss_weight=self.config.pose_reg_loss_weight,
            pose_classification_loss_type=self.config.pose_classification_loss_type,
            bbox_assigner_topk=self.config.bbox_assigner_topk,
            bbox_assigned_alpha=self.config.bbox_assigned_alpha,
            bbox_assigned_beta=self.config.bbox_assigned_beta,
            assigner_multiply_by_pose_oks=self.config.assigner_multiply_by_pose_oks,
            rescale_pose_loss_with_assigned_score=self.config.rescale_pose_loss_with_assigned_score,
        )
        self.loss_fn = self.loss_fn.to(self.device)
        self.val_loader = None

    def _build_dataset(self, img_files, label_files, preproc) -> YOLOPoseDataset:
        return YOLOPoseDataset(
            img_files=img_files,
            num_keypoints=self.num_keypoints,
            label_files=label_files,
            img_size=self.input_size,
            preproc=preproc,
            keypoint_dim=self.config.keypoint_dim,
            decode_scale=self.config.decode_scale,
        )

    def _setup_data(self):
        if not self.config.data:
            raise ValueError("Pose training requires 'data' (a dataset yaml path)")

        cfg = load_data_config(
            self.config.data, allow_scripts=self.config.allow_download_scripts
        )
        self.num_classes = 1
        flip_idx = cfg.get("flip_idx")

        train_imgs = cfg.get("train_img_files")
        train_lbls = cfg.get("train_label_files")
        if not train_imgs:
            if not cfg.get("train"):
                raise FileNotFoundError("Dataset yaml has no 'train' split")
            train_imgs = get_img_files(cfg["train"])
            train_lbls = img2label_paths(train_imgs)
        if not train_imgs:
            raise FileNotFoundError("No training images found for pose training")

        train_tf = YOLONASPoseTrainTransform(
            self.num_keypoints,
            flip_idx=flip_idx,
            flip_prob=self.config.flip_prob,
            hsv_prob=self.config.hsv_prob,
            brightness_contrast_prob=self.config.brightness_contrast_prob,
            affine_prob=self.config.affine_prob,
            degrees=self.config.degrees,
            translate=self.config.translate,
            scale=self.config.pose_scale,
            affine_interpolation=self.config.affine_interpolation,
        )
        train_ds = self._build_dataset(train_imgs, train_lbls, train_tf)
        drop_last = len(train_ds) >= self.config.batch
        loader_kwargs = {}
        if self.config.workers > 0:
            loader_kwargs.update(
                worker_init_fn=_pose_worker_init_fn,
                persistent_workers=self.config.persistent_workers,
                prefetch_factor=self.config.prefetch_factor,
            )
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch,
            shuffle=True,
            num_workers=self.config.workers,
            pin_memory=self.config.pin_memory,
            drop_last=drop_last,
            collate_fn=pose_collate_fn,
            **loader_kwargs,
        )

        val_imgs = cfg.get("val_img_files")
        val_lbls = cfg.get("val_label_files")
        if not val_imgs and cfg.get("val"):
            try:
                val_imgs = get_img_files(cfg["val"])
                val_lbls = img2label_paths(val_imgs)
            except (FileNotFoundError, ValueError):
                val_imgs = None
        if val_imgs:
            val_ds = self._build_dataset(
                val_imgs, val_lbls, YOLONASPoseValTransform(self.num_keypoints)
            )
            self.val_loader = DataLoader(
                val_ds,
                batch_size=self.config.batch,
                shuffle=False,
                num_workers=self.config.workers,
                pin_memory=self.config.pin_memory,
                drop_last=False,
                collate_fn=pose_collate_fn,
                **loader_kwargs,
            )
            logger.info("Validation dataset: %d images", len(val_ds))
        else:
            self.val_loader = None
            logger.warning(
                "No validation split found — best.pt cannot be selected by "
                "validation loss for this run"
            )

        logger.info("Training dataset: %d images", len(train_ds))
        logger.info("Iterations per epoch: %d", len(self.train_loader))
        return train_ds

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        keys = ("cls", "iou", "dfl", "pose_cls", "pose_reg")
        return {k: outputs.get(k, 0.0) for k in keys}

    def _checkpoint_extra_metadata(self) -> Dict:
        return {
            "num_keypoints": self.num_keypoints,
            "keypoint_dim": self.config.keypoint_dim,
            "oks_sigmas": self._resolve_oks_sigmas(),
        }

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        outputs = self.model(imgs)
        loss, log_losses = self.loss_fn(outputs, targets)
        # log_losses order: [cls, iou, dfl, pose_cls, pose_reg, total]
        return {
            "total_loss": loss,
            "cls": log_losses[0],
            "iou": log_losses[1],
            "dfl": log_losses[2],
            "pose_cls": log_losses[3],
            "pose_reg": log_losses[4],
        }

    def _validate_epoch(self, epoch: int):
        if getattr(self, "val_loader", None) is None:
            return None

        model = self.ema_model.ema if self.ema_model else self.model
        was_training = model.training
        model.eval()

        total_loss, num_batches = 0.0, 0
        pose_metrics = None
        try:
            with torch.no_grad():
                for batch in self.val_loader:
                    imgs = batch[0].to(self.device, non_blocking=True)
                    targets = batch[1].to(self.device, non_blocking=True)
                    loss, _ = self.loss_fn(model(imgs), targets)
                    total_loss += float(loss.item())
                    num_batches += 1

            pose_metrics = self._run_pose_metric_validation(model, epoch)
        finally:
            if was_training:
                model.train()

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {"loss/val": avg_loss}
        if pose_metrics:
            metrics.update(self._scalar_mapping(pose_metrics))
            mAP50 = metrics.get("metrics/keypoints_mAP50")
            mAP50_95 = metrics.get("metrics/keypoints_mAP50-95")
            logger.info(
                "Validation - loss/val: %.4f, keypoints_mAP50: %.4f, "
                "keypoints_mAP50-95: %.4f",
                avg_loss,
                mAP50 if mAP50 is not None else 0.0,
                mAP50_95 if mAP50_95 is not None else 0.0,
            )
            return {
                "best_metric": mAP50_95 if mAP50_95 is not None else 0.0,
                "best_metric_key": self.best_metric_key,
                "mAP50": mAP50,
                "mAP50_95": mAP50_95,
                "metrics": metrics,
            }

        logger.info("Validation - loss/val: %.4f", avg_loss)
        return {
            "best_metric": -avg_loss,  # higher-is-better convention; lower loss wins
            "best_metric_key": "loss/val",
            "mAP50": None,
            "mAP50_95": None,
            "metrics": metrics,
        }

    def _run_pose_metric_validation(
        self, eval_model: torch.nn.Module, epoch: int
    ) -> Dict[str, float] | None:
        if self.wrapper_model is None:
            logger.warning("Skipping pose mAP validation: wrapper_model is missing")
            return None

        try:
            from libreyolo.validation import PoseValidator, ValidationConfig

            val_config = ValidationConfig(
                data=self.config.data,
                split="val",
                batch_size=self.config.batch,
                imgsz=self.config.imgsz,
                conf_thres=0.001,
                iou_thres=0.65,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                verbose=False,
                num_workers=self.config.workers,
                allow_download_scripts=self.config.allow_download_scripts,
                oks_sigmas=self._resolve_oks_sigmas(),
                save_dir=str(self.save_dir / "val"),
            )

            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_model
            try:
                validator = PoseValidator(model=self.wrapper_model, config=val_config)
                return validator.run()
            finally:
                self.wrapper_model.model = original_model
        except Exception as exc:
            logger.error("Pose mAP validation failed at epoch %d: %s", epoch + 1, exc)
            return None

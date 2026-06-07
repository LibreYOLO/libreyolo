"""ECPoseTrainer — native EdgeCrafter ECPose (DETR-style) fine-tuning (EXPERIMENTAL).

ECPose is a keypoint transformer, so this trainer owns its data pipeline
(:class:`~libreyolo.data.YOLOPoseDataset` with keypoint-aware transforms, padded
``(B, max_labels, 5 + 3K)`` targets) like the YOLO-NAS pose trainer, but feeds a
DETR matcher/criterion (:class:`ECPoseCriterion`) instead of a grid loss.

EC-specific bits vs the YOLO-NAS pose trainer:
  * inputs are RGB + ImageNet-normalized (the ECViT backbone's contract);
  * AdamW with the EC backbone-LR multiplier + no-decay groups;
  * targets are translated to DETR per-image ``{labels, keypoints, vis, area}``.

best.pt is selected by keypoint OKS-AP via :class:`PoseValidator` when a val
split is available, falling back to validation loss otherwise.
"""

from __future__ import annotations

import logging
import os
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
from ...training.config import ECPoseConfig, TrainConfig
from ...training.scheduler import FlatCosineScheduler
from ...training.trainer import BaseTrainer
from .pose_loss import ECPoseCriterion, PoseHungarianMatcher, default_oks_sigmas
from .pose_transforms import ECPoseTrainTransform, ECPoseValTransform

logger = logging.getLogger(__name__)


def _pose_worker_init_fn(worker_id: int) -> None:
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _set_bn_eval(module: torch.nn.Module) -> None:
    """Put only BatchNorm layers into eval so a train-mode forward (needed to
    emit the decoder's aux/enc/dn outputs) does not pollute running stats."""
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


class ECPoseTrainer(BaseTrainer):
    """Trainer for EdgeCrafter ECPose models (experimental)."""

    artifact_model_families = ("ec",)
    best_metric_key = "metrics/keypoints_mAP50-95"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return ECPoseConfig

    def get_model_family(self) -> str:
        return "ec"

    def get_model_tag(self) -> str:
        return f"EC-Pose-{self.config.size}"

    @property
    def effective_lr(self) -> float:
        return self.config.lr0

    @property
    def num_keypoints(self) -> int:
        return self.config.num_keypoints

    def _setup_device(self) -> torch.device:
        device = super()._setup_device()
        if device.type == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            logger.info(
                "EC pose training on MPS: enabling PYTORCH_ENABLE_MPS_FALLBACK=1 "
                "(deformable attention's grid_sample backward runs on CPU)."
            )
        return device

    def on_num_classes_resolved(self):
        """ECPose is single-class (person); class count is fixed."""

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

    # create_transforms is abstract on BaseTrainer; pose owns _setup_data.
    def create_transforms(self):
        return None, None

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
        num_classes = int(getattr(self.model.decoder, "num_classes", 2))
        sigmas = self._resolve_oks_sigmas()
        if hasattr(self.model, "decoder"):
            self.model.decoder.oks_sigmas = sigmas
        matcher = PoseHungarianMatcher(
            num_keypoints=self.num_keypoints,
            cost_class=self.config.cls_loss_weight,
            cost_keypoints=self.config.keypoint_l1_loss_weight,
            cost_oks=self.config.oks_loss_weight,
            sigmas=sigmas,
        )
        self.criterion = ECPoseCriterion(
            matcher=matcher,
            num_keypoints=self.num_keypoints,
            num_classes=num_classes,
            weight_dict={
                "loss_vfl": self.config.cls_loss_weight,
                "loss_keypoints": self.config.keypoint_l1_loss_weight,
                "loss_oks": self.config.oks_loss_weight,
            },
            losses=("vfl", "keypoints"),
            sigmas=sigmas,
        ).to(self.device)
        # Configure the model's contrastive-denoising group from the config.
        if hasattr(self.model, "decoder"):
            self.model.decoder.dn_number = int(self.config.dn_number)
            self.model.decoder.label_noise_ratio = float(self.config.label_noise_ratio)
        self.val_loader = None

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        backbone_wd, backbone_no_wd, head_wd, head_no_wd = [], [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_norm_or_bias = (
                "norm" in name or ".bn." in name or "bias" in name or "lab.scale" in name
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

    def _ddp_find_unused_parameters(self) -> bool:
        # Contrastive denoising exercises label_enc/pose_enc, but some auxiliary
        # heads can still go unused on batches with no GT, so keep this on for
        # DDP safety (this is an experimental single-GPU-first path anyway).
        return True

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
        flip_idx = cfg.get("flip_idx") or self.config.flip_idx

        train_imgs = cfg.get("train_img_files")
        train_lbls = cfg.get("train_label_files")
        if not train_imgs:
            if not cfg.get("train"):
                raise FileNotFoundError("Dataset yaml has no 'train' split")
            train_imgs = get_img_files(cfg["train"])
            train_lbls = img2label_paths(train_imgs)
        if not train_imgs:
            raise FileNotFoundError("No training images found for pose training")

        train_tf = ECPoseTrainTransform(
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
            imagenet_norm=True,
            to_rgb=True,
        )
        train_ds = self._build_dataset(train_imgs, train_lbls, train_tf)
        per_rank_batch = max(1, self.config.batch // max(self.world_size, 1))
        train_sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_ds) >= self.world_size,
            )

        visible_samples = len(train_sampler) if train_sampler is not None else len(train_ds)
        drop_last = visible_samples >= per_rank_batch
        loader_kwargs = {}
        if self.config.workers > 0:
            loader_kwargs.update(
                worker_init_fn=_pose_worker_init_fn,
                persistent_workers=self.config.persistent_workers,
                prefetch_factor=self.config.prefetch_factor,
            )
        self.train_loader = DataLoader(
            train_ds,
            batch_size=per_rank_batch,
            shuffle=train_sampler is None,
            sampler=train_sampler,
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
            val_tf = ECPoseValTransform(
                self.num_keypoints, imagenet_norm=True, to_rgb=True
            )
            val_ds = self._build_dataset(val_imgs, val_lbls, val_tf)
            self.val_loader = DataLoader(
                val_ds,
                batch_size=per_rank_batch,
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
            logger.warning("No validation split found — best.pt falls back to val loss")

        logger.info("Training dataset: %d images", len(train_ds))
        logger.info(
            "Iterations per epoch: %d (batch_per_rank=%d, world_size=%d)",
            len(self.train_loader),
            per_rank_batch,
            self.world_size,
        )
        return train_ds

    def _build_pose_targets(self, targets: torch.Tensor, imgs: torch.Tensor):
        """``(B, max_labels, 5 + 3K)`` pixel slab -> per-image DETRPose target dicts.

        Each dict follows DETRPose's contract:
          - ``labels``   : (n,) all 0 (person)
          - ``boxes``    : (n,4) xyxy normalized (used by the denoising group)
          - ``keypoints``: (n, 2K + K) = [x1,y1,...,xK,yK | v1,...,vK], normalized
                           xy and binary visibility
          - ``area``     : (n,) normalized box area (w*h)
        """
        B = targets.shape[0]
        H, W = imgs.shape[-2], imgs.shape[-1]
        K = self.num_keypoints
        out = []
        for b in range(B):
            t = targets[b]
            valid = (t[:, 3] > 0) & (t[:, 4] > 0)
            tv = t[valid]
            if tv.numel() == 0:
                out.append(
                    {
                        "labels": torch.zeros(0, dtype=torch.int64, device=self.device),
                        "boxes": torch.zeros(0, 4, device=self.device),
                        "keypoints": torch.zeros(0, 3 * K, device=self.device),
                        "area": torch.zeros(0, device=self.device),
                    }
                )
                continue
            cx, cy, w, h = tv[:, 1], tv[:, 2], tv[:, 3], tv[:, 4]
            boxes = torch.stack(
                [(cx - w / 2) / W, (cy - h / 2) / H, (cx + w / 2) / W, (cy + h / 2) / H],
                dim=1,
            ).clamp(0.0, 1.0)
            area = ((w * h) / float(W * H)).clamp(min=1e-6)
            kpts = tv[:, 5:].reshape(tv.shape[0], K, 3)
            kxy = kpts[:, :, :2].clone()
            kxy[:, :, 0] = kxy[:, :, 0] / W
            kxy[:, :, 1] = kxy[:, :, 1] / H
            vis = (kpts[:, :, 2] > 0).to(kxy.dtype)  # binary visibility
            keypoints = torch.cat([kxy.reshape(tv.shape[0], 2 * K), vis], dim=1)
            out.append(
                {
                    "labels": torch.zeros(tv.shape[0], dtype=torch.int64, device=self.device),
                    "boxes": boxes,
                    "keypoints": keypoints,
                    "area": area,
                }
            )
        return out

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        def _sum_with_prefix(prefix: str) -> float:
            total = 0.0
            for k, v in outputs.items():
                if k == prefix or k.startswith(prefix + "_"):
                    total += v.item() if isinstance(v, torch.Tensor) else float(v)
            return total

        return {
            "vfl": _sum_with_prefix("loss_vfl"),
            "kpt": _sum_with_prefix("loss_keypoints"),
            "oks": _sum_with_prefix("loss_oks"),
        }

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        target_list = self._build_pose_targets(targets, imgs)
        # Pass targets so the decoder builds the contrastive-denoising group.
        outputs = self.model(imgs, targets=target_list)
        losses = self.criterion(outputs, target_list)
        total = sum(losses.values())
        result = {"total_loss": total}
        result.update(losses)
        return result

    def _checkpoint_extra_metadata(self) -> Dict:
        return {
            "num_keypoints": self.num_keypoints,
            "keypoint_dim": self.config.keypoint_dim,
            "oks_sigmas": self._resolve_oks_sigmas(),
        }

    def _validate_epoch(self, epoch: int, *, save_plots: bool | None = None):
        from ...training.distributed import barrier, is_main_process, unwrap_model

        if self.is_distributed and not is_main_process():
            barrier()
            return None

        if getattr(self, "val_loader", None) is None:
            if self.is_distributed:
                barrier()
            return None

        model = self.ema_model.ema if self.ema_model else unwrap_model(self.model)
        was_training = model.training
        model.eval()

        total_loss, num_batches = 0.0, 0
        pose_metrics = None
        try:
            with torch.no_grad():
                for batch in self.val_loader:
                    imgs = batch[0].to(self.device, non_blocking=True)
                    targets = batch[1].to(self.device, non_blocking=True)
                    target_list = self._build_pose_targets(targets, imgs)
                    # The pose decoder only emits aux/enc/dn outputs in train
                    # mode, but BatchNorm must NOT update running stats from val
                    # data (no_grad does not stop buffer writes). Run train mode
                    # for the decoder structure while freezing BN to eval.
                    model.train()
                    _set_bn_eval(model)
                    losses = self.criterion(
                        model(imgs, targets=target_list), target_list
                    )
                    model.eval()
                    total_loss += float(sum(losses.values()).item())
                    num_batches += 1
            pose_metrics = self._run_pose_metric_validation(model, epoch)
        finally:
            if was_training:
                model.train()
            if self.is_distributed:
                barrier()

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
            "best_metric": -avg_loss,
            "best_metric_key": "loss/val",
            "mAP50": None,
            "mAP50_95": None,
            "metrics": metrics,
        }

    def _run_pose_metric_validation(self, eval_model, epoch: int):
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

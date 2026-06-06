"""Native RF-DETR trainer for LibreYOLO."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Type

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ...data import (
    YOLOPoseDataset,
    default_oks_sigmas,
    get_img_files,
    img2label_paths,
    load_data_config,
    pose_collate_fn,
)
from ...training.config import TrainConfig
from ...training.distributed import is_main_process, unwrap_model
from ...training.scheduler import BaseScheduler, CosineAnnealingScheduler, FlatCosineScheduler
from ...training.trainer import BaseTrainer
from .config import RFDETRConfig
from ..dfine.transforms import DFINEPassThroughDataset
from .pose_transforms import RFDETRPoseTransform
from .seg_transforms import (
    RFDETRDetTransform,
    RFDETRSegPassThroughDataset,
    RFDETRSegTransform,
    compute_multi_scale_scales,
)

logger = logging.getLogger(__name__)


def _pose_worker_init_fn(worker_id: int) -> None:
    cv2.setNumThreads(0)
    torch.set_num_threads(1)
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


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
        if getattr(self.wrapper_model, "task", "detect") == "pose":
            self.best_metric_key = "metrics/keypoints_mAP50-95"
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
            if getattr(self.wrapper_model, "task", "detect") == "pose":
                self.config.num_classes = 1
                kpt_shape = data_cfg.get("kpt_shape")
                if kpt_shape is not None:
                    self.config.num_keypoints = int(kpt_shape[0])
                    self.config.keypoint_dim = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3

    @property
    def effective_lr(self) -> float:
        return self.config.lr0

    def get_model_family(self) -> str:
        return "rfdetr"

    def get_model_tag(self) -> str:
        return f"LibreRFDETR-{self.config.size}"

    def _ddp_find_unused_parameters(self) -> bool:
        # Detect/pose/OBB: at least one transformer parameter receives no gradient
        # on some forward passes, so DDP must skip reduction for it.
        # Segmentation uses static_graph=True instead (see _ddp_static_graph).
        return getattr(self.wrapper_model, "task", "detect") in {"detect", "pose", "obb"}

    def _ddp_static_graph(self) -> bool:
        # Segmentation only: the seg head is invoked from both encoder and
        # decoder branches in one forward, so its parameters accumulate
        # gradients from two call sites. With find_unused_parameters=True the
        # DDP hook fires twice per step → "marked ready twice" crash.
        # static_graph=True locks the reducer after iteration 1 and handles
        # double accumulation correctly. Detect/OBB keep the default (False)
        # because find_unused_parameters=True requires dynamic graph traversal.
        # Pose follows detection and keeps static_graph=False.
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
        if task == "pose":
            preproc = RFDETRPoseTransform(
                num_keypoints=self.config.num_keypoints,
                max_labels=100,
                flip_prob=self.config.flip_prob,
                imgsz=self.config.imgsz,
                multi_scale=self.config.multi_scale,
                expanded_scales=self.config.expanded_scales,
                do_random_resize_via_padding=self.config.do_random_resize_via_padding,
                patch_size=patch_size,
                num_windows=num_windows,
                crop_resize_prob=self.config.crop_resize_prob,
            )
            return preproc, None
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
        if getattr(self.wrapper_model, "task", "detect") == "pose" and targets.shape[-1] > 5:
            keypoints = targets[..., 5:].view(*targets.shape[:-1], -1, 3)
            keypoints[..., 0] *= scale_x
            keypoints[..., 1] *= scale_y

        if isinstance(polygons, torch.Tensor):
            polygons = F.interpolate(
                polygons.float(),
                size=(scale, scale),
                mode="nearest",
            )

        return imgs, targets, polygons

    def _resolve_oks_sigmas(self) -> list[float]:
        sigmas = self.config.oks_sigmas
        if sigmas is not None:
            if len(sigmas) != self.config.num_keypoints:
                raise ValueError(
                    f"oks_sigmas has {len(sigmas)} entries but the dataset has "
                    f"{self.config.num_keypoints} keypoints"
                )
            return [float(s) for s in sigmas]
        return default_oks_sigmas(self.config.num_keypoints)

    def _build_pose_dataset(self, img_files, label_files, preproc) -> YOLOPoseDataset:
        return YOLOPoseDataset(
            img_files=img_files,
            num_keypoints=self.config.num_keypoints,
            label_files=label_files,
            img_size=self.input_size,
            preproc=preproc,
            keypoint_dim=self.config.keypoint_dim,
            decode_scale=self.config.decode_scale,
        )

    def _setup_data(self):
        if getattr(self.wrapper_model, "task", "detect") != "pose":
            return super()._setup_data()
        if not self.config.data:
            raise ValueError("RF-DETR pose training requires 'data' (a dataset yaml path)")

        cfg = load_data_config(
            self.config.data,
            allow_scripts=self.config.allow_download_scripts,
        )
        kpt_shape = cfg.get("kpt_shape")
        if kpt_shape is not None:
            self.config.num_keypoints = int(kpt_shape[0])
            self.config.keypoint_dim = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3
        self.config.num_classes = 1
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
            raise FileNotFoundError("No training images found for RF-DETR pose training")

        patch_size = int(getattr(self.model, "patch_size", 16))
        num_windows = int(getattr(self.model, "num_windows", 4))
        block_size = patch_size * num_windows
        if self.config.imgsz % block_size != 0:
            lo = (self.config.imgsz // block_size) * block_size
            hi = lo + block_size
            raise ValueError(
                f"imgsz={self.config.imgsz} is not divisible by {block_size} "
                f"(patch_size={patch_size} x num_windows={num_windows}). "
                f"Use {lo} or {hi}."
            )
        train_tf = RFDETRPoseTransform(
            self.config.num_keypoints,
            flip_idx=flip_idx,
            max_labels=100,
            flip_prob=self.config.flip_prob,
            imgsz=self.config.imgsz,
            multi_scale=self.config.multi_scale,
            expanded_scales=self.config.expanded_scales,
            do_random_resize_via_padding=self.config.do_random_resize_via_padding,
            patch_size=patch_size,
            num_windows=num_windows,
            crop_resize_prob=self.config.crop_resize_prob,
        )
        train_ds = self._build_pose_dataset(train_imgs, train_lbls, train_tf)

        per_rank_batch = max(1, self.config.batch // max(self.world_size, 1))
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_ds) >= self.world_size,
            )

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
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.config.workers,
            pin_memory=self.config.pin_memory,
            drop_last=len(train_ds) >= per_rank_batch,
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
            val_tf = RFDETRPoseTransform(
                self.config.num_keypoints,
                max_labels=100,
                flip_prob=0.0,
                imgsz=self.config.imgsz,
                multi_scale=False,
                patch_size=patch_size,
                num_windows=num_windows,
                crop_resize_prob=0.0,
            )
            val_ds = self._build_pose_dataset(val_imgs, val_lbls, val_tf)
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
            if is_main_process():
                logger.info("Validation dataset: %d images", len(val_ds))
        else:
            self.val_loader = None
            logger.warning("No validation split found for RF-DETR pose training")

        if is_main_process():
            logger.info("Training dataset: %d images", len(train_ds))
            logger.info(
                "Iterations per epoch: %d (batch_per_rank=%d, world_size=%d)",
                len(self.train_loader),
                per_rank_batch,
                self.world_size,
            )
        return train_ds

    def on_setup(self):
        task = getattr(self.wrapper_model, "task", "detect")
        if self.model.nb_classes != self.config.num_classes:
            head_outputs = (
                self.config.num_classes
                if task == "pose"
                else self.config.num_classes + 1
            )
            self.model.model.reinitialize_detection_head(head_outputs)
            self.model.nb_classes = self.config.num_classes
            self.model.args.num_classes = (
                max(0, self.config.num_classes - 1)
                if task == "pose"
                else self.config.num_classes
            )
        if task == "pose":
            if getattr(self.model, "num_keypoints", None) != self.config.num_keypoints:
                self.model.model.reinitialize_keypoint_head(self.config.num_keypoints)
                self.model.num_keypoints = self.config.num_keypoints
            self.model.args.num_keypoints = self.config.num_keypoints
            self.model.args.oks_sigmas = self._resolve_oks_sigmas()
            self.model.args.keypoint_l1_loss_coef = self.config.keypoint_l1_loss_coef
            self.model.args.keypoint_oks_loss_coef = self.config.keypoint_oks_loss_coef
            self.model.args.keypoint_vis_loss_coef = self.config.keypoint_vis_loss_coef

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
            elif task == "pose" and self.config.num_classes == 1:
                self.wrapper_model.names = {0: "person"}
            else:
                self.wrapper_model.names = {
                    i: f"class_{i}" for i in range(self.config.num_classes)
                }
            if task == "pose":
                self.wrapper_model.num_keypoints = self.config.num_keypoints
                self.wrapper_model.keypoint_dim = self.config.keypoint_dim

    def _setup_optimizer(self) -> torch.optim.Optimizer:
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

    def _targets_to_rfdetr_list(
        self,
        targets: torch.Tensor,
        *,
        height: int,
        width: int,
        masks_batch: Optional[torch.Tensor] = None,
    ) -> list[dict]:
        batch_size = targets.shape[0]
        scale = torch.tensor([width, height, width, height], device=targets.device, dtype=targets.dtype)
        task = getattr(self.wrapper_model, "task", "detect")
        is_pose = task == "pose"
        is_obb = task == "obb"
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
                if is_pose:
                    entry["keypoints"] = torch.zeros(
                        0,
                        self.config.num_keypoints,
                        3,
                        dtype=torch.float32,
                        device=self.device,
                    )
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
                if is_pose:
                    keypoints = t_valid[:, 5:].view(-1, self.config.num_keypoints, 3).clone()
                    keypoint_scale = torch.tensor(
                        [width, height, 1.0],
                        device=targets.device,
                        dtype=targets.dtype,
                    )
                    keypoints = keypoints / keypoint_scale
                    outside = (
                        (keypoints[..., 0] < 0.0)
                        | (keypoints[..., 0] > 1.0)
                        | (keypoints[..., 1] < 0.0)
                        | (keypoints[..., 1] > 1.0)
                    )
                    keypoints[..., :2] = keypoints[..., :2].clamp(0.0, 1.0)
                    keypoints[..., 2] = torch.where(
                        outside,
                        torch.zeros_like(keypoints[..., 2]),
                        keypoints[..., 2],
                    )
                    entry["keypoints"] = keypoints
            target_list.append(entry)
        return target_list

    def on_forward(
        self,
        imgs: torch.Tensor,
        targets: torch.Tensor,
        polygons: Optional[List] = None,
    ) -> Dict:
        height, width = imgs.shape[-2], imgs.shape[-1]
        is_seg = getattr(self.wrapper_model, "task", "detect") == "segment"
        # ``polygons`` here is the collate-stacked output of RFDETRSegTransform:
        # a [B, max_labels, mask_h, mask_w] float32 tensor whose slot i aligns
        # with target slot i. Slice by the same ``valid`` box mask to hand the
        # criterion per-image ``[N_valid, mask_h, mask_w]`` tensors.
        masks_batch = (
            polygons.to(self.device, non_blocking=True)
            if is_seg and isinstance(polygons, torch.Tensor)
            else None
        )

        target_list = self._targets_to_rfdetr_list(
            targets,
            height=height,
            width=width,
            masks_batch=masks_batch,
        )

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
        if getattr(self.wrapper_model, "task", "detect") == "pose":
            components["keypoints_l1"] = _sum_with_prefix("loss_keypoints_l1")
            components["keypoints_oks"] = _sum_with_prefix("loss_keypoints_oks")
            components["keypoints_vis"] = _sum_with_prefix("loss_keypoints_vis")
        if getattr(self.wrapper_model, "task", "detect") == "obb":
            components["angle"] = _sum_with_prefix("loss_angle")
        return components

    def _checkpoint_extra_metadata(self) -> Dict:
        if getattr(self.wrapper_model, "task", "detect") != "pose":
            return {}
        return {
            "num_keypoints": self.config.num_keypoints,
            "keypoint_dim": self.config.keypoint_dim,
            "oks_sigmas": self._resolve_oks_sigmas(),
        }

    def _run_validation(
        self,
        epoch: int,
        *,
        save_plots: bool | None = None,
    ):
        if getattr(self.wrapper_model, "task", "detect") != "pose":
            return super()._run_validation(epoch, save_plots=save_plots)
        if getattr(self, "val_loader", None) is None:
            return None

        model = self.ema_model.ema if self.ema_model else unwrap_model(self.model)
        was_training = model.training
        model.eval()

        total_loss = 0.0
        num_batches = 0
        pose_metrics = None
        try:
            if not self.is_distributed:
                with torch.no_grad():
                    for batch in self.val_loader:
                        imgs = batch[0].to(self.device, non_blocking=True)
                        targets = batch[1].to(self.device, non_blocking=True)
                        target_list = self._targets_to_rfdetr_list(
                            targets,
                            height=imgs.shape[-2],
                            width=imgs.shape[-1],
                        )
                        outputs = model(imgs, targets=target_list)
                        loss_dict = self.criterion(outputs, target_list)
                        total = sum(
                            loss_dict[key] * self.criterion.weight_dict[key]
                            for key in loss_dict
                            if key in self.criterion.weight_dict
                        )
                        total_loss += float(total.item())
                        num_batches += 1
            pose_metrics = self._run_pose_metric_validation(model, epoch, save_plots=save_plots)
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
                "Validation - loss/val: %.4f, keypoints_mAP50: %.4f, keypoints_mAP50-95: %.4f",
                avg_loss,
                mAP50 if mAP50 is not None else 0.0,
                mAP50_95 if mAP50_95 is not None else 0.0,
            )
            return {
                "best_metric": mAP50_95 if mAP50_95 is not None else 0.0,
                "best_metric_key": "metrics/keypoints_mAP50-95",
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

    def _run_pose_metric_validation(
        self,
        eval_model: torch.nn.Module,
        epoch: int,
        *,
        save_plots: bool | None = None,
    ) -> Dict[str, float] | None:
        if self.wrapper_model is None:
            logger.warning("Skipping pose mAP validation: wrapper_model is missing")
            return None

        try:
            from libreyolo.validation import PoseValidator, ValidationConfig

            is_final_epoch = self._is_final_epoch(epoch)
            val_save_plots = (
                bool(save_plots)
                if save_plots is not None
                else bool(getattr(self.config, "save_plots", False)) and is_final_epoch
            )
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
                save_plots=val_save_plots,
                save_dir=str(self.save_dir / "val") if val_save_plots else None,
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
    pose: bool = False,
    **kwargs,
) -> Dict:
    """Compatibility helper around :class:`LibreRFDETR.train`."""
    from .model import LibreRFDETR

    model = LibreRFDETR(
        model_path=pretrain_weights,
        size=size,
        device=kwargs.pop("device", "auto"),
        segmentation=segmentation,
        task="pose" if pose else None,
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

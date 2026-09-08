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

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import logging

import torch
from tqdm import tqdm

from ...data import (
    get_coco_annotation_file,
    get_coco_image_dir,
    get_img_files,
    img2label_paths,
    load_data_config,
)
from ...data.dataset import COCODataset, YOLODataset
from ...training.config import (
    DFINEConfig,
    TrainConfig,
    resolve_dfine_base_size_repeat,
)
from ...training.scheduler import FlatCosineScheduler
from ...training.optim import build_optimizer
from ...training.trainer import BaseTrainer, ensure_mutation_reaches_workers
from .loss import DFINECriterion
from .matcher import HungarianMatcher
from .transforms import (
    DFINESegPassThroughDataset,
    DFINESegTransform,
    DFINEMultiScaleCollate,
    DFINEPassThroughDataset,
    DFINETrainTransform,
)
from ..base.detr_cuda_graph import DETREncoderCudaGraphMixin

logger = logging.getLogger(__name__)


class DFINETrainer(DETREncoderCudaGraphMixin, BaseTrainer):
    # D-FINE pairs a CNN (HGNetv2) backbone with a transformer encoder/decoder
    # whose projections are nn.Linear layers. lora=True freezes the backbone
    # and the transformer base weights and trains LoRA adapters on the
    # transformer Linears (see libreyolo/training/lora.py).
    supports_lora = True

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return DFINEConfig

    def get_model_family(self) -> str:
        return "dfine"

    def get_model_tag(self) -> str:
        return f"DFINE-{self.config.size}"

    def preserve_freeze_param(self, name: str, param: torch.nn.Parameter) -> bool:
        if not getattr(self.config, "lora", False):
            return False
        from ...training.lora import is_lora_parameter_name

        return is_lora_parameter_name(name)

    def _is_segment(self) -> bool:
        return (
            getattr(getattr(self, "wrapper_model", None), "task", "detect")
            == "segment"
        )

    def create_transforms(self):
        if self._is_segment():
            preproc = DFINESegTransform(
                max_labels=120,
                flip_prob=self.config.flip_prob,
                imgsz=self.config.imgsz,
                crop_resize_prob=getattr(self.config, "crop_resize_prob", 0.0),
            )
            return preproc, DFINESegPassThroughDataset

        preproc = DFINETrainTransform(
            max_labels=120,
            flip_prob=self.config.flip_prob,
            imgsz=self.config.imgsz,
        )
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
            "mask_bce": _sum_with_prefix("loss_mask_bce"),
            "mask_dice": _sum_with_prefix("loss_mask_dice"),
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

    def on_num_classes_resolved(self):
        num_classes = self._resolve_num_classes_from_data_config()
        self._sync_wrapped_model_num_classes(num_classes)

    def on_setup(self):
        if getattr(self.config, "lora", False):
            if self._is_segment():
                # The mask head path is untested with adapters; silently
                # training it fully while config says LoRA would misrepresent
                # the run.
                raise ValueError(
                    "D-FINE segment training does not support lora=True yet. "
                    "Use freeze='backbone' for parameter-efficient segment "
                    "fine-tuning."
                )
            from ...training.lora import apply_lora_to_detr

            apply_lora_to_detr(self.model)

        self.criterion = self.build_criterion()

    def build_criterion(self, *, distributed_normalize: bool = True):
        """Build the training criterion.

        Validation loss builds a second one with ``distributed_normalize``
        off, so both stay defined in exactly one place.
        """
        matcher_weights = {"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0}
        loss_weights = {
            "loss_vfl": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
        }
        losses = ["vfl", "boxes", "local"]
        if self._is_segment():
            matcher_weights.update(
                {
                    "cost_mask": self.config.mask_match_cost,
                    "cost_mask_dice": self.config.mask_dice_match_cost,
                }
            )
            loss_weights.update(
                {
                    "loss_mask_bce": self.config.mask_bce_loss_weight,
                    "loss_mask_dice": self.config.mask_dice_loss_weight,
                }
            )
            losses.append("masks")

        matcher = HungarianMatcher(
            weight_dict=matcher_weights,
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        )
        return DFINECriterion(
            matcher=matcher,
            weight_dict=loss_weights,
            losses=losses,
            alpha=0.75,
            gamma=2.0,
            num_classes=self.config.num_classes,
            reg_max=32,
            distributed_normalize=distributed_normalize,
        ).to(self.device)

    def validate_validation_loss_config(self) -> None:
        if not getattr(self.config, "val_loss", False):
            return

        task = getattr(getattr(self, "wrapper_model", None), "task", "detect")
        if task != "detect":
            raise ValueError(
                f"val_loss=True currently supports {self.get_model_family()} "
                "detection only; segment and other tasks are not supported"
            )

    def build_validation_loss_adapter(self, model: torch.nn.Module):
        from .validation_loss import DFINEValidationLoss

        return DFINEValidationLoss(
            model, self.build_criterion(distributed_normalize=False)
        )

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

        return build_optimizer(torch.optim.AdamW, param_groups, betas=(0.9, 0.999))

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
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
        masks_batch = (
            polygons.to(self.device, non_blocking=True)
            if isinstance(polygons, torch.Tensor)
            else None
        )

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
                if masks_batch is not None:
                    mh, mw = masks_batch.shape[-2], masks_batch.shape[-1]
                    target_list[-1]["masks"] = torch.zeros(
                        0, mh, mw, dtype=torch.bool, device=self.device
                    )
            else:
                entry = {
                    "labels": t_valid[:, 0].long(),
                    "boxes": (t_valid[:, 1:] / scale).clamp(0.0, 1.0),
                }
                if masks_batch is not None:
                    entry["masks"] = masks_batch[b][valid[: masks_batch.shape[1]]].to(
                        device=self.device,
                        dtype=torch.bool,
                    )
                target_list.append(entry)

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

    def _train_base_size_repeat(self) -> Optional[int]:
        """Multi-scale repeat for this run: config override or per-size default."""
        return resolve_dfine_base_size_repeat(
            getattr(self.config, "size", ""),
            getattr(self.config, "base_size_repeat", None),
        )

    def _setup_data(self):
        """Mirror of ``BaseTrainer._setup_data`` but uses ``DFINEMultiScaleCollate``.

        Built by hand instead of inheriting because ``create_dataloader`` doesn't
        expose ``collate_fn`` and we need our epoch-aware collate. Dataset-build
        logic is duplicated from the parent for clarity.
        """
        from torch.utils.data import DataLoader

        img_size = self.input_size
        preproc, MosaicDatasetClass = self.create_transforms()
        load_segments = self._is_segment()

        if self.config.data:
            data_cfg = load_data_config(
                self.config.data,
                single_cls=self.config.single_cls,
            )
            data_dir = data_cfg["root"]
            self.num_classes = data_cfg.get("nc", self.config.num_classes)

            ann_file = Path(data_dir) / "annotations" / "instances_train2017.json"
            coco_ann_file = get_coco_annotation_file(data_cfg, "train")
            img_files = data_cfg.get("train_img_files")
            label_files = data_cfg.get("train_label_files")

            if coco_ann_file:
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file=coco_ann_file,
                    name=get_coco_image_dir(data_cfg, "train", "train2017"),
                    img_size=img_size,
                    preproc=preproc,
                    num_classes=int(self.num_classes),
                    names=data_cfg.get("_original_names", data_cfg.get("names")),
                    load_segments=load_segments,
                    single_cls=self.config.single_cls,
                )
            elif img_files:
                train_dataset = YOLODataset(
                    img_files=img_files,
                    label_files=label_files,
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    num_classes=int(self.num_classes),
                    single_cls=self.config.single_cls,
                )
            elif ann_file.exists():
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name="train2017",
                    img_size=img_size,
                    preproc=preproc,
                    num_classes=int(self.num_classes),
                    names=data_cfg.get("_original_names", data_cfg.get("names")),
                    load_segments=load_segments,
                    single_cls=self.config.single_cls,
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
                    load_segments=load_segments,
                    num_classes=int(self.num_classes),
                    single_cls=self.config.single_cls,
                )
        elif self.config.data_dir:
            data_dir = self.config.data_dir
            self.num_classes = 1 if self.config.single_cls else self.config.num_classes
            if (Path(data_dir) / "annotations").exists():
                train_dataset = COCODataset(
                    data_dir=data_dir,
                    json_file="instances_train2017.json",
                    name="train2017",
                    img_size=img_size,
                    preproc=preproc,
                    num_classes=int(self.num_classes),
                    load_segments=load_segments,
                    single_cls=self.config.single_cls,
                )
            else:
                train_dataset = YOLODataset(
                    data_dir=data_dir,
                    split="train",
                    img_size=img_size,
                    preproc=preproc,
                    load_segments=load_segments,
                    num_classes=int(self.num_classes),
                    single_cls=self.config.single_cls,
                )
        else:
            raise ValueError("Either 'data' or 'data_dir' must be specified")

        train_dataset.enable_image_cache(getattr(self.config, "cache", False))

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
        # strong augs at the right moment (never later than the start of the
        # no-aug LR tail; see resolve_aug_stop_epoch).
        from ...data.augment.detr import resolve_aug_stop_epoch

        stop_epoch = resolve_aug_stop_epoch(
            self.config.epochs,
            getattr(self.config, "aug_stop_epoch_ratio", 1.0),
            getattr(self.config, "no_aug_epochs", 0),
        )
        if hasattr(train_dataset, "set_stop_epoch"):
            train_dataset.set_stop_epoch(stop_epoch)

        # Multi-scale collate (or default yolox_collate_fn).
        if getattr(self.config, "multi_scale", False) and not load_segments:
            # Upstream pins base_size_repeat per size (n disables multi-scale,
            # s 20, m 6, l 4, x 3); this used to be hardcoded to 3, which only
            # X actually wants. None makes the collate keep batches at
            # base_size, which is exactly upstream's `~` for the N size.
            collate_fn = DFINEMultiScaleCollate(
                base_size=self.config.imgsz,
                base_size_repeat=self._train_base_size_repeat(),
                stop_epoch=stop_epoch,
            )
        else:
            from ...data.dataset import yolox_collate_fn

            if getattr(self.config, "multi_scale", False) and load_segments:
                logger.info(
                    "D-FINE multi-scale training is not supported with "
                    "task='segment'; training at fixed %dpx.",
                    self.config.imgsz,
                )
            collate_fn = yolox_collate_fn

        # ``batch`` is the global batch under DDP. Each rank's loader is built
        # with ``batch // world_size`` over a DistributedSampler shard.
        per_rank_batch = max(1, self.config.batch // max(self.world_size, 1))
        train_sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=len(train_dataset) >= self.world_size,
            )

        # Under DDP each rank only sees ``len(sampler)`` samples, so base the
        # drop_last decision on the per-rank visible count. Otherwise a small
        # dataset split across ranks could drop every rank's only partial
        # batch and leave zero iterations.
        visible_samples = (
            len(train_sampler) if train_sampler is not None else len(train_dataset)
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_batch,
            num_workers=self.config.workers,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=visible_samples >= per_rank_batch,
        )

        return train_dataset

    # =========================================================================
    # _train_epoch override — set_epoch propagation, grad clip, per-group LR
    # =========================================================================

    def _train_epoch(
        self, epoch: int
    ) -> Tuple[float, Optional[Dict[str, float]], Dict[str, float], Dict[str, float]]:
        """Copy of ``BaseTrainer._train_epoch`` with three D-FINE-specific tweaks:

        1. Propagate the current epoch to dataset + collate (drives stop_epoch
           augmentation/multi-scale gating).
        2. Apply gradient clipping at ``config.clip_max_norm`` before the
           optimizer step.
        3. Apply per-group LR multipliers (the scheduler returns one base LR;
           each param group's ``lr_mult`` scales it).

        The profiler hooks (``wrap_loader`` / ``_prof_phase`` / ``prof.step``)
        mirror the parent loop exactly so ``profile=True`` works here too.
        """
        # Gradient accumulation is opt-in; delegate when enabled.
        if self._accum_steps > 1:
            return self._train_epoch_accum(epoch)

        # 1. Epoch propagation.
        # DistributedSampler needs its epoch set so shuffling differs per
        # epoch while staying deterministic for resume.
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        ds = self.train_loader.dataset
        if hasattr(ds, "set_epoch"):
            ensure_mutation_reaches_workers(self.train_loader, ds, "set_epoch")
            ds.set_epoch(epoch)
        cf = getattr(self.train_loader, "collate_fn", None)
        if cf is not None and hasattr(cf, "set_epoch"):
            ensure_mutation_reaches_workers(self.train_loader, cf, "set_epoch")
            cf.set_epoch(epoch)

        clip_max_norm = float(getattr(self.config, "clip_max_norm", 0.0))

        self.model.train()
        self._enforce_frozen_bn_eval()
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            total=len(self.train_loader),
        )

        total_loss = 0.0
        num_batches = 0
        loss_component_sums: Dict[str, float] = {}

        # getattr: test doubles may bypass BaseTrainer.__init__, same as _profiler.
        prof = getattr(self, "_profiler", None)
        loader = prof.wrap_loader(pbar) if prof is not None else pbar
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 5:
                imgs, targets, img_infos, img_ids, polygons = batch
            else:
                imgs, targets, img_infos, img_ids = batch
                polygons = None
            self.current_iter = epoch * len(self.train_loader) + batch_idx

            with self._prof_phase("to_device"):
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

            if self.scaler is not None:
                with self._prof_phase("forward"):
                    with self._autocast_context():
                        outputs = self._forward_train(imgs, targets, polygons)
                        loss = outputs["total_loss"]
                self.optimizer.zero_grad()
                with self._prof_phase("backward"):
                    self.scaler.scale(loss).backward()
                    if clip_max_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), clip_max_norm
                        )
                with self._prof_phase("optimizer"):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                with self._prof_phase("forward"):
                    outputs = self._forward_train(imgs, targets, polygons)
                    loss = outputs["total_loss"]
                self.optimizer.zero_grad()
                with self._prof_phase("backward"):
                    loss.backward()
                    if clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), clip_max_norm
                        )
                with self._prof_phase("optimizer"):
                    self.optimizer.step()

            if self.ema_model is not None:
                self.ema_model.update(self.model)

            loss_val = loss.item()
            loss_components = self._scalar_mapping(self.get_loss_components(outputs))
            total_loss += loss_val
            for name, value in loss_components.items():
                loss_component_sums[name] = loss_component_sums.get(name, 0.0) + value
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

            if prof is not None:
                prof.step()
                if prof.finished:
                    if getattr(self.config, "profile_then_stop", False):
                        self._stop_training = True
                        break
                    # Window closed: drop the hooks so the rest of the run
                    # pays nothing, and keep training.
                    logger.info(
                        "Profile window complete; training continues "
                        "(profile_then_stop=True stops here instead)"
                    )
                    self._profiler = None
                    prof = None

        num_batches = max(num_batches, 1)
        avg_loss = total_loss / num_batches
        avg_loss_components = {
            name: value / num_batches for name, value in loss_component_sums.items()
        }

        # Validation. A profile-only run (profile_then_stop) truncated the
        # epoch, so validating the barely-trained weights would waste time and
        # could poison the best-metric state.
        val_metrics = None
        if (
            not getattr(self, "_stop_training", False)
            and self.config.eval_interval > 0
            and (epoch + 1) % self.config.eval_interval == 0
        ):
            val_metrics = self._validate_epoch(epoch)

        return avg_loss, val_metrics, avg_loss_components, self._current_lrs()

    def _train_epoch_accum(
        self, epoch: int
    ) -> Tuple[float, Optional[Dict[str, float]], Dict[str, float], Dict[str, float]]:
        """``_train_epoch`` variant with gradient accumulation (``_accum_steps`` > 1).

        Same D-FINE tweaks as ``_train_epoch`` (epoch propagation, clipping,
        per-group LR), but gradients accumulate over ``accum`` micro-batches and
        the optimizer step, clipping, EMA and LR update fire once per window.
        """
        # 1. Epoch propagation.
        # DistributedSampler needs its epoch set so shuffling differs per
        # epoch while staying deterministic for resume.
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        ds = self.train_loader.dataset
        if hasattr(ds, "set_epoch"):
            ensure_mutation_reaches_workers(self.train_loader, ds, "set_epoch")
            ds.set_epoch(epoch)
        cf = getattr(self.train_loader, "collate_fn", None)
        if cf is not None and hasattr(cf, "set_epoch"):
            ensure_mutation_reaches_workers(self.train_loader, cf, "set_epoch")
            cf.set_epoch(epoch)

        clip_max_norm = float(getattr(self.config, "clip_max_norm", 0.0))

        self.model.train()
        self._enforce_frozen_bn_eval()
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config.epochs}",
            total=len(self.train_loader),
            disable=not sys.stderr.isatty(),
        )

        accum = self._accum_steps
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / accum))
        total_loss = 0.0
        num_batches = 0
        loss_component_sums: Dict[str, float] = {}
        actual_window = accum
        base_lr = self.optimizer.param_groups[0]["lr"]

        # getattr: test doubles may bypass BaseTrainer.__init__, same as _profiler.
        prof = getattr(self, "_profiler", None)
        loader = prof.wrap_loader(pbar) if prof is not None else pbar
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 5:
                imgs, targets, img_infos, img_ids, polygons = batch
            else:
                imgs, targets, img_infos, img_ids = batch
                polygons = None

            is_opt_step = (batch_idx + 1) % accum == 0 or batch_idx == len(self.train_loader) - 1
            opt_step = epoch * steps_per_epoch + batch_idx // accum
            self.current_iter = opt_step

            with self._prof_phase("to_device"):
                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

            if batch_idx % accum == 0:
                self.optimizer.zero_grad(set_to_none=True)
                actual_window = min(accum, len(self.train_loader) - batch_idx)

            if self.scaler is not None:
                with self._prof_phase("forward"):
                    with self._autocast_context():
                        outputs = self._forward_train(imgs, targets, polygons)
                        loss = outputs["total_loss"] / actual_window
                with self._prof_phase("backward"):
                    self.scaler.scale(loss).backward()
                if is_opt_step:
                    with self._prof_phase("optimizer"):
                        if clip_max_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), clip_max_norm
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
            else:
                with self._prof_phase("forward"):
                    outputs = self._forward_train(imgs, targets, polygons)
                    loss = outputs["total_loss"] / actual_window
                with self._prof_phase("backward"):
                    loss.backward()
                if is_opt_step:
                    with self._prof_phase("optimizer"):
                        if clip_max_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), clip_max_norm
                            )
                        self.optimizer.step()

            if is_opt_step:
                if self.ema_model is not None:
                    self.ema_model.update(self.model)
                # 3. Per-group LR (scheduler returns one base LR; each group
                # multiplies by its ``lr_mult``).
                base_lr = self.lr_scheduler.update_lr(opt_step + 1)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = base_lr * pg.get("lr_mult", 1.0)

            loss_val = loss.item() * actual_window
            loss_components = self._scalar_mapping(self.get_loss_components(outputs))
            total_loss += loss_val
            for name, value in loss_components.items():
                loss_component_sums[name] = loss_component_sums.get(name, 0.0) + value
            num_batches += 1
            del outputs, loss

            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{base_lr:.6f}"}
            postfix.update({k: f"{v:.4f}" for k, v in loss_components.items()})
            pbar.set_postfix(postfix)

            if prof is not None:
                prof.step()
                if prof.finished:
                    if getattr(self.config, "profile_then_stop", False):
                        self._stop_training = True
                        break
                    # Window closed: drop the hooks so the rest of the run
                    # pays nothing, and keep training.
                    logger.info(
                        "Profile window complete; training continues "
                        "(profile_then_stop=True stops here instead)"
                    )
                    self._profiler = None
                    prof = None

        num_batches = max(num_batches, 1)
        avg_loss = total_loss / num_batches
        avg_loss_components = {
            name: value / num_batches for name, value in loss_component_sums.items()
        }

        # Validation. A profile-only run (profile_then_stop) truncated the
        # epoch, so validating the barely-trained weights would waste time and
        # could poison the best-metric state.
        val_metrics = None
        if (
            not getattr(self, "_stop_training", False)
            and self.config.eval_interval > 0
            and (epoch + 1) % self.config.eval_interval == 0
        ):
            val_metrics = self._validate_epoch(epoch)

        return avg_loss, val_metrics, avg_loss_components, self._current_lrs()

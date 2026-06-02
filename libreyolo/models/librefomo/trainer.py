"""LibreFOMO trainer.

Thin subclass of BaseTrainer that:

* Completely overrides ``_setup_data`` to serve FOMO grid targets
  ``(B, H_grid, W_grid)`` instead of the standard box-format
  ``(B, max_labels, 5)`` used by the YOLO/DETR pipeline.
* Uses standard LR schedulers (defaulting to Cosine Annealing).
* Overrides ``_run_validation`` to use ``PointValidator`` and an F1 threshold
  sweep (mirroring the reference ``evaluate_sweep``).
* Sets ``best_metric_key = "metrics/F1"`` so ``_update_best_state`` and
  ``_save_checkpoint`` select and save the best model by F1 score.

Training is gated experimental — ``LibreFOMO.train()`` checks for
``allow_experimental=True`` before instantiating this class.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...training.config import FOMOConfig, TrainConfig
from ...training.scheduler import ConstantLRScheduler
from ...training.trainer import BaseTrainer
from ...training.distributed import (
    barrier,
    is_main_process,
    unwrap_model,
)
from .loss import LibreFOMOLoss
from .nn import CONFIGS

logger = logging.getLogger(__name__)

# Downsample factor: the backbone reduces spatial dims by 8×
_DOWNSAMPLE = 8


class LibreFOMOTrainer(BaseTrainer):
    """LibreFOMO point-localization trainer."""

    best_metric_key: str = "metrics/F1"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return FOMOConfig

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    def get_model_family(self) -> str:
        return "librefomo"

    def get_model_tag(self) -> str:
        return f"LibreFOMO-{self.config.size}"

    # -------------------------------------------------------------------------
    # Augmentation / transforms (not used — overridden by _setup_data)
    # -------------------------------------------------------------------------

    def create_transforms(self):
        """Not used; _setup_data is fully overridden."""
        return None, None

    # -------------------------------------------------------------------------
    # Data setup — bypass YOLO pipeline, serve FOMO grid targets
    # -------------------------------------------------------------------------

    def _setup_data(self):
        """Build FOMO dataloaders from a standard YOLO data.yaml.

        Produces ``(img, grid_target, img_info, img_id)`` 4-tuples so the
        inherited ``_train_epoch`` loop works without modification.
        """
        input_size = self.config.imgsz
        grid_size = input_size // _DOWNSAMPLE

        if not self.config.data:
            raise ValueError(
                "LibreFOMOTrainer requires a YOLO ``data`` (data.yaml path) in "
                "the training config. Pass ``data='path/to/data.yaml'`` to "
                "``model.train()``."
            )
        train_dataset, val_dataset = self._build_yolo_datasets(input_size, grid_size)

        self._val_dataset = val_dataset

        per_rank_batch = max(1, self.config.batch // max(self.world_size, 1))
        sampler = None
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=per_rank_batch,
            shuffle=(sampler is None),
            num_workers=self.config.workers,
            pin_memory=self.device.type == "cuda",
            sampler=sampler,
            drop_last=False,
        )

        if is_main_process():
            logger.info(f"FOMO training dataset: {len(train_dataset)} images")
            logger.info(
                f"Grid size: {grid_size}×{grid_size} "
                f"(imgsz={input_size}, downsample=8)"
            )
            logger.info(
                f"Iterations per epoch: {len(self.train_loader)} "
                f"(batch_per_rank={per_rank_batch}, world_size={self.world_size})"
            )
        return train_dataset

    def _build_yolo_datasets(self, input_size: int, grid_size: int):
        from .dataset import FOMOYOLODataset
        from ...data import load_data_config, get_img_files, img2label_paths

        data_cfg = load_data_config(
            self.config.data,
            allow_scripts=self.config.allow_download_scripts,
        )
        data_dir = data_cfg["root"]

        # Resolve training images
        train_img_files = data_cfg.get("train_img_files")
        train_label_files = data_cfg.get("train_label_files")
        if not train_img_files:
            train_path = data_cfg.get("train", "images/train")
            train_img_files = get_img_files(train_path, prefix=data_dir)
            train_label_files = img2label_paths(train_img_files)

        # Resolve validation images
        val_img_files = data_cfg.get("val_img_files")
        val_label_files = data_cfg.get("val_label_files")
        if not val_img_files:
            val_path = data_cfg.get("val", "images/val")
            try:
                val_img_files = get_img_files(val_path, prefix=data_dir)
                val_label_files = img2label_paths(val_img_files)
            except (FileNotFoundError, ValueError):
                val_img_files, val_label_files = [], []

        dataset_nc = data_cfg.get("nc", self.config.num_classes)
        if dataset_nc != getattr(self.model, "nc", self.config.num_classes):
            logger.info(
                "Dataset nc=%d differs from model nc=%d — rebuilding head.",
                dataset_nc,
                getattr(self.model, "nc", self.config.num_classes),
            )
            if self.wrapper_model is not None:
                self.wrapper_model._rebuild_for_new_classes(dataset_nc)
                # Keep self.model pointing at the freshly-built nn.Module.
                self.model = self.wrapper_model.model
            else:
                # Fallback: no wrapper available (unusual, but safe to skip rebuild).
                logger.warning(
                    "wrapper_model is None — cannot rebuild head for nc=%d. "
                    "Training will continue with the original head.",
                    dataset_nc,
                )
        self.config.num_classes = dataset_nc

        # Propagate dataset class names to the wrapper so checkpoints record
        # the real labels instead of the generic "class_0", "class_1", … defaults.
        raw_names = data_cfg.get("names")
        if raw_names is not None and self.wrapper_model is not None:
            from ..base import BaseModel
            self.wrapper_model.names = BaseModel._sanitize_names(
                raw_names if isinstance(raw_names, dict)
                else {i: n for i, n in enumerate(raw_names)},
                dataset_nc,
            )

        train_ds = FOMOYOLODataset(train_img_files, train_label_files, input_size, grid_size)
        val_ds = FOMOYOLODataset(val_img_files or [], val_label_files or [], input_size, grid_size)
        return train_ds, val_ds

    # -------------------------------------------------------------------------
    # Setup — build loss and plateau scheduler state
    # -------------------------------------------------------------------------

    def on_setup(self) -> None:
        nc = getattr(self.model, "nc", self.config.num_classes)
        fg_weight = getattr(self.config, "fg_weight", 100.0)
        self._loss_fn = LibreFOMOLoss(
            num_classes=nc,
            fg_weight=fg_weight,
            device=self.device,
        ).to(self.device)

        if is_main_process():
            logger.info(
                f"LibreFOMOLoss: nc={nc}, fg_weight={fg_weight}"
            )

    # -------------------------------------------------------------------------
    # Scheduler — constant/plateau, cosine, linear
    # -------------------------------------------------------------------------

    def create_scheduler(self, iters_per_epoch: int):
        sched_type = getattr(self.config, "scheduler", "cosine")

        if sched_type in ("cosine", "cos"):
            from ...training.scheduler import CosineAnnealingScheduler
            return CosineAnnealingScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=getattr(self.config, "warmup_epochs", 0),
                warmup_lr_start=getattr(self.config, "warmup_lr_start", 0.0),
                min_lr_ratio=getattr(self.config, "min_lr_ratio", 0.05),
            )
        elif sched_type == "flat_cosine":
            from ...training.scheduler import FlatCosineScheduler
            return FlatCosineScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=getattr(self.config, "warmup_epochs", 0),
                warmup_lr_start=getattr(self.config, "warmup_lr_start", 0.0),
                no_aug_epochs=getattr(self.config, "no_aug_epochs", 0),
                min_lr_ratio=getattr(self.config, "min_lr_ratio", 0.05),
            )
        elif sched_type == "linear":
            from ...training.scheduler import LinearLRScheduler
            return LinearLRScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=getattr(self.config, "warmup_epochs", 0),
                warmup_lr_start=getattr(self.config, "warmup_lr_start", 0.0001),
                min_lr_ratio=getattr(self.config, "min_lr_ratio", 0.01),
            )
        elif sched_type == "yoloxwarmcos":
            from ...training.scheduler import WarmupCosineScheduler
            return WarmupCosineScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=getattr(self.config, "warmup_epochs", 5),
                warmup_lr_start=getattr(self.config, "warmup_lr_start", 0.0),
                plateau_epochs=getattr(self.config, "no_aug_epochs", 15),
                min_lr_ratio=getattr(self.config, "min_lr_ratio", 0.05),
            )

        # Fallback to constant LR scheduler
        from ...training.scheduler import ConstantLRScheduler
        return ConstantLRScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=getattr(self.config, "warmup_epochs", 0),
            warmup_lr_start=getattr(self.config, "warmup_lr_start", 0.0),
        )

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------

    def on_forward(
        self,
        imgs: torch.Tensor,
        targets: torch.Tensor,
        polygons=None,
    ) -> Dict:
        """Run model and compute loss.

        Args:
            imgs: ``(B, 3, H, W)`` image tensor.
            targets: ``(B, H_grid, W_grid)`` int64 grid targets.
        """
        logits = self.model(imgs)

        if logits.shape[-2:] != targets.shape[-2:]:
            raise ValueError(
                f"Model output grid {tuple(logits.shape[-2:])} does not match "
                f"target grid {tuple(targets.shape[-2:])}. "
                f"Check that config.imgsz={self.config.imgsz} matches the "
                f"model variant (s:96, m:192, l:224)."
            )

        return self._loss_fn(logits, targets.long())

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        return {"ce": float(outputs.get("ce", 0.0))}

    # -------------------------------------------------------------------------
    # Validation — PointValidator with threshold sweep, plateau LR step
    # -------------------------------------------------------------------------

    def _run_validation(self, epoch: int) -> Optional[Dict[str, Any]]:
        """Run FOMO validation and step the plateau scheduler.

        Returns a metrics dict with ``best_metric`` = best F1 across the sweep,
        compatible with ``_update_best_state``.
        """
        try:
            val_dataset = getattr(self, "_val_dataset", None)
            if val_dataset is None or len(val_dataset) == 0:
                logger.warning("No validation dataset available; skipping validation.")
                return None

            logger.info(f"Running FOMO validation for epoch {epoch + 1}")

            # Use the EMA model if available, otherwise the raw model.
            eval_nn = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )

            conf_thresholds = tuple(getattr(self.config, "conf_thresholds", (0.25, 0.35, 0.50, 0.65, 0.80, 0.90)))
            nms_radii = tuple(int(r) for r in getattr(self.config, "nms_radii", (1, 2)))
            distance_tolerance = float(getattr(self.config, "distance_tolerance", 1.5))

            avg_val_loss, best_result = self._sweep_validation(
                eval_nn,
                val_dataset,
                conf_thresholds=conf_thresholds,
                nms_radii=nms_radii,
                distance_tolerance=distance_tolerance,
            )

            if best_result is None:
                logger.warning("Validation sweep returned no results.")
                return None

            f1 = best_result["f1"]
            precision = best_result["precision"]
            recall = best_result["recall"]
            mean_dist = best_result["mean_dist"]
            tp = best_result["tp"]
            fp = best_result["fp"]
            fn = best_result["fn"]

            metrics = {
                "best_metric": f1,
                "best_metric_key": "metrics/F1",
                "mAP50": f1,       # alias so base trainer logs mAP50 = F1
                "mAP50_95": f1,    # alias so best checkpoint tracks F1
                "metrics": {
                    "metrics/F1": f1,
                    "metrics/precision": precision,
                    "metrics/recall": recall,
                    "metrics/mean_distance": mean_dist,
                    "metrics/val_loss": avg_val_loss,
                    "metrics/TP": float(tp),
                    "metrics/FP": float(fp),
                    "metrics/FN": float(fn),
                    "decode/threshold": best_result["threshold"],
                    "decode/nms_radius": float(best_result["nms_radius"]),
                },
            }

            current_lr = self.optimizer.param_groups[0]["lr"]

            if is_main_process():
                logger.info(
                    f"Epoch {epoch + 1} val | "
                    f"loss={avg_val_loss:.4f} | "
                    f"F1={f1:.4f} | "
                    f"P={precision:.4f} | "
                    f"R={recall:.4f} | "
                    f"MeanDist={mean_dist:.3f} | "
                    f"thresh={best_result['threshold']:.2f} | "
                    f"nms_r={best_result['nms_radius']} | "
                    f"TP={tp} FP={fp} FN={fn} | "
                    f"LR={current_lr:.6f}"
                )

            return metrics

        except Exception as exc:
            import traceback
            logger.error(f"FOMO validation failed: {exc}")
            logger.debug(traceback.format_exc())
            return None

    def _sweep_validation(
        self,
        eval_nn: nn.Module,
        val_dataset,
        conf_thresholds,
        nms_radii,
        distance_tolerance: float,
    ) -> Tuple[float, Optional[Dict]]:
        """Run the model over the val set once, cache logits, sweep thresholds.

        Returns (avg_val_loss, best_result_dict).
        """
        from .loss import LibreFOMOLoss
        from .utils import decode_points_from_logits
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment

        per_rank_batch = max(1, self.config.batch // max(self.world_size, 1))
        val_loader = DataLoader(
            val_dataset,
            batch_size=per_rank_batch,
            shuffle=False,
            num_workers=min(self.config.workers, 4),
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

        nc = getattr(self.model, "nc", self.config.num_classes)
        fg_weight = getattr(self.config, "fg_weight", 100.0)
        val_loss_fn = LibreFOMOLoss(num_classes=nc, fg_weight=fg_weight, device=self.device)
        val_loss_fn.eval()

        eval_nn.eval()
        cached: list[tuple[torch.Tensor, torch.Tensor]] = []
        val_loss_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                imgs, grid_targets, _, _ = batch
                imgs = imgs.to(self.device, non_blocking=True)
                grid_targets = grid_targets.to(self.device, non_blocking=True)
                logits = eval_nn(imgs)
                loss_out = val_loss_fn(logits, grid_targets.long())
                val_loss_total += float(loss_out["total_loss"].item())
                cached.append((logits.cpu(), grid_targets.cpu()))

        avg_val_loss = val_loss_total / max(len(val_loader), 1)

        # Threshold / NMS sweep
        best: Optional[Dict] = None

        for threshold in conf_thresholds:
            for nms_radius in nms_radii:
                total_tp = total_fp = total_fn = 0
                total_dist = 0.0

                for logits_cpu, targets_cpu in cached:
                    decoded = decode_points_from_logits(
                        logits_cpu, conf_threshold=threshold, nms_radius=nms_radius
                    )
                    B = logits_cpu.shape[0]
                    for b in range(B):
                        rows = decoded[b]  # (N, 4): x, y, class_id (0-based), conf

                        # All foreground cells — grid value encodes class as (class+1).
                        fg_mask = targets_cpu[b] >= 1
                        ys, xs = torch.where(fg_mask)
                        if ys.numel():
                            true_cls = targets_cpu[b][ys, xs] - 1  # 0-based class
                            trues_xy = torch.stack((xs, ys), dim=1).float().numpy()
                            true_cls_np = true_cls.numpy()
                        else:
                            trues_xy = np.zeros((0, 2))
                            true_cls_np = np.zeros(0, dtype=np.int64)

                        if len(rows) > 0:
                            preds_xy = rows[:, :2].numpy()
                            preds_cls = rows[:, 2].long().numpy()  # 0-based
                        else:
                            preds_xy = np.zeros((0, 2))
                            preds_cls = np.zeros(0, dtype=np.int64)

                        if len(preds_xy) == 0 and len(trues_xy) == 0:
                            continue
                        if len(preds_xy) == 0:
                            total_fn += len(trues_xy)
                            continue
                        if len(trues_xy) == 0:
                            total_fp += len(preds_xy)
                            continue

                        dist_mat = cdist(preds_xy, trues_xy)
                        # Penalise class mismatches — set distance to inf so the
                        # Hungarian solver never matches cross-class pairs.
                        for pi in range(len(preds_cls)):
                            for ti in range(len(true_cls_np)):
                                if preds_cls[pi] != true_cls_np[ti]:
                                    dist_mat[pi, ti] = np.inf

                        row_ind, col_ind = linear_sum_assignment(
                            np.where(np.isfinite(dist_mat), dist_mat, 1e9)
                        )
                        matched_preds: set = set()
                        matched_trues: set = set()
                        for r, c in zip(row_ind, col_ind):
                            d = dist_mat[r, c]
                            if np.isfinite(d) and d <= distance_tolerance:
                                total_tp += 1
                                total_dist += d
                                matched_preds.add(r)
                                matched_trues.add(c)
                        total_fp += len(preds_xy) - len(matched_preds)
                        total_fn += len(trues_xy) - len(matched_trues)

                prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
                rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                mean_dist = total_dist / max(total_tp, 1)

                result = {
                    "threshold": float(threshold),
                    "nms_radius": int(nms_radius),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "mean_dist": mean_dist,
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                }
                if best is None or f1 > best["f1"]:
                    best = result

        return avg_val_loss, best



    # -------------------------------------------------------------------------
    # Checkpoint extra metadata — record best decode config
    # -------------------------------------------------------------------------

    def _checkpoint_extra_metadata(self) -> Dict[str, Any]:
        return {
            "task": "point",
            "best_metric_key": "metrics/F1",
        }

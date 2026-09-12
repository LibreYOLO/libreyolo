"""FOMO trainer."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

import torch
from torch.utils.data import DataLoader

from ...training.config import FOMOConfig, TrainConfig
from ...training.trainer import BaseTrainer
from ...training.distributed import (
    is_main_process,
    unwrap_model,
)
from .nn import CONFIGS
from ...validation.fomo_validator import FOMOValidator
from .dataset import build_fomo_datasets

logger = logging.getLogger(__name__)

_DOWNSAMPLE = 8


class FOMOTrainer(BaseTrainer):
    """FOMO point-localization trainer."""

    best_metric_key: str = "metrics/grid_F1"

    def __init__(self, *args, **kwargs):
        # FOMO overrides validation and only emits its grid metrics, so the
        # generic detect aliases would silently resolve to keys it never
        # produces. Reject up front rather than accept and ignore.
        if kwargs.get("best_metric") not in (None, ""):
            raise NotImplementedError(
                "best_metric is not supported for FOMO; it always tracks "
                "metrics/grid_F1."
            )
        super().__init__(*args, **kwargs)

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return FOMOConfig

    def get_model_family(self) -> str:
        return "fomo"

    def get_model_tag(self) -> str:
        return f"FOMO-{self.config.size}"

    def create_transforms(self):
        return None, None

    def _setup_data(self):
        input_size = self.config.imgsz
        grid_size = input_size // _DOWNSAMPLE

        if not self.config.data:
            raise ValueError(
                "FOMOTrainer requires a YOLO ``data`` (data.yaml path) in "
                "the training config. Pass ``data='path/to/data.yaml'`` to "
                "``model.train()``."
            )
        train_dataset, val_dataset, self.model, self._loss_fn = build_fomo_datasets(
            config=self.config,
            model=self.model,
            wrapper_model=self.wrapper_model,
            device=self.device,
            input_size=input_size,
            grid_size=grid_size,
        )

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
        
        from ...training.scheduler import ConstantLRScheduler
        return ConstantLRScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=getattr(self.config, "warmup_epochs", 0),
            warmup_lr_start=getattr(self.config, "warmup_lr_start", 0.0),
        )

    def on_forward(
        self,
        imgs: torch.Tensor,
        targets: torch.Tensor,
        polygons=None,
    ) -> Dict:
        logits = self.model(imgs)

        if logits.shape[-2:] != targets.shape[-2:]:
            raise ValueError(
                f"Model output grid {tuple(logits.shape[-2:])} does not match "
                f"target grid {tuple(targets.shape[-2:])}. "
                f"Check that config.imgsz={self.config.imgsz} matches the "
                f"model variant (s:96, m:192, l:224)."
            )

        return self._loss_fn(logits, targets.long())

    def cuda_graph_train_spec(self):
        """Capture spec: graph the whole network, keep the loss eager.

        FOMO's forward is a plain convolutional stack over images only, so
        the network is static-shaped for a fixed batch; the centroid loss
        (which reads the label grid) stays eager. The grid-shape check in
        ``on_forward`` is reproduced here so a mismatched ``imgsz`` still
        fails loudly instead of capturing a wrong graph.
        """
        from libreyolo.training.cuda_graph import (
            CudaGraphTrainSpec,
            GraphableNetwork,
        )

        task = getattr(getattr(self, "wrapper_model", None), "task", "point")
        if task != "point":
            return None
        if not isinstance(self.model, torch.nn.Module):
            return None
        if getattr(self, "_loss_fn", None) is None:
            return None

        network = GraphableNetwork(self.model)

        def assemble(flat, imgs, targets, polygons=None):
            logits = network.rebuild(flat)
            if logits.shape[-2:] != targets.shape[-2:]:
                raise ValueError(
                    f"Model output grid {tuple(logits.shape[-2:])} does not "
                    f"match target grid {tuple(targets.shape[-2:])}."
                )
            return self._loss_fn(logits, targets.long())

        return CudaGraphTrainSpec(network=network, assemble=assemble)

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        return {"ce": float(outputs.get("ce", 0.0))}

    def validate_validation_loss_config(self) -> None:
        # Accepted, but nothing to switch on: FOMOValidator always computes the
        # loss and publishes it under the shared metrics/loss keys. Raising
        # "not supported" here would be wrong, since it is.
        return

    def _run_validation(
        self, epoch: int, *, save_plots: bool | None = None
    ) -> Optional[Dict[str, Any]]:
        try:
            from ...validation import ValidationConfig

            logger.info(f"Running validation for epoch {epoch + 1}")

            is_final_epoch = self._is_final_epoch(epoch)
            val_save_plots = (
                bool(save_plots)
                if save_plots is not None
                else bool(getattr(self.config, "save_plots", False)) and is_final_epoch
            )
            val_save_dir = (
                str(self.save_dir / "val") if val_save_plots else None
            )

            val_config = ValidationConfig(
                data=self.config.data,
                batch_size=self.config.batch,
                imgsz=self.config.imgsz,
                conf_thres=0.001,
                iou_thres=0.65,
                max_det=self.config.max_det,
                eval_max_det=self.config.eval_max_det,
                device=str(self.device),
                half=self.config.amp and self.device.type == "cuda",
                amp_dtype=self.config.amp_dtype,
                verbose=False,
                num_workers=self.config.workers,
                save_plots=val_save_plots,
                save_dir=val_save_dir,
            )

            if self.wrapper_model is None:
                logger.error(
                    "Validation requires wrapper_model to be provided to trainer"
                )
                return None

            eval_pytorch_model = (
                self.ema_model.ema if self.ema_model else unwrap_model(self.model)
            )
            original_model = self.wrapper_model.model
            self.wrapper_model.model = eval_pytorch_model

            conf_thresholds = tuple(getattr(self.config, "conf_thresholds", (0.25, 0.35, 0.50, 0.65, 0.80, 0.90)))
            nms_radii = tuple(int(r) for r in getattr(self.config, "nms_radii", (1, 2)))
            distance_tolerance = float(getattr(self.config, "distance_tolerance", 1.5))
            grid_size = self.config.imgsz // _DOWNSAMPLE
            fg_weight = getattr(self.config, "fg_weight", 100.0)

            try:
                validator = FOMOValidator(
                    model=self.wrapper_model,
                    config=val_config,
                    grid_size=grid_size,
                    fg_weight=fg_weight,
                    conf_thresholds=conf_thresholds,
                    nms_radii=nms_radii,
                    distance_tolerance=distance_tolerance,
                )
                results = validator.run()
            finally:
                self.wrapper_model.model = original_model

            raw_metrics = self._scalar_mapping(results)
            best_key = getattr(self, "best_metric_key", "metrics/grid_F1")
            best_metric = raw_metrics.get(best_key, 0.0)

            metrics = {
                "mAP50": raw_metrics.get("metrics/grid_F1", 0.0),
                "mAP50_95": best_metric,
                "best_metric": best_metric,
                "best_metric_key": best_key,
                "metrics": raw_metrics,
            }

            if is_main_process():
                logger.info(
                    "Validation - Point Metrics | "
                    f"P={raw_metrics.get('metrics/precision', 0.0):.4f} | "
                    f"R={raw_metrics.get('metrics/recall', 0.0):.4f} | "
                    f"F1={raw_metrics.get('metrics/f1', 0.0):.4f} | "
                    f"mAP@0.01={raw_metrics.get('metrics/mAP@0.01', 0.0):.4f} | "
                    f"mAP_sweep={raw_metrics.get('metrics/mAP@[0.01:0.10]', 0.0):.4f} | "
                    f"MLE={raw_metrics.get('metrics/MLE', 0.0):.4f} | "
                    f"MAE={raw_metrics.get('metrics/MAE', 0.0):.4f} | "
                    f"RMSE={raw_metrics.get('metrics/RMSE', 0.0):.4f}"
                )

                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch + 1} grid val | "
                    f"loss={raw_metrics.get('metrics/val_loss', 0.0):.4f} | "
                    f"grid_F1={raw_metrics.get('metrics/grid_F1', 0.0):.4f} | "
                    f"grid_precision={raw_metrics.get('metrics/grid_precision', 0.0):.4f} | "
                    f"grid_recall={raw_metrics.get('metrics/grid_recall', 0.0):.4f} | "
                    f"MeanDist={raw_metrics.get('metrics/grid_mean_distance', 0.0):.3f} | "
                    f"thresh={raw_metrics.get('decode/threshold', 0.0):.2f} | "
                    f"nms_r={raw_metrics.get('decode/nms_radius', 0.0):.1f} | "
                    f"TP={raw_metrics.get('metrics/grid_TP', 0.0):.1f} FP={raw_metrics.get('metrics/grid_FP', 0.0):.1f} FN={raw_metrics.get('metrics/grid_FN', 0.0):.1f} | "
                    f"LR={current_lr:.6f}"
                )

            return metrics

        except Exception as exc:
            import traceback
            logger.error(f"FOMO training validation failed: {exc}")
            logger.debug(traceback.format_exc())
            raise

    def _checkpoint_extra_metadata(self) -> Dict[str, Any]:
        return {
            "task": "point",
            "best_metric_key": "metrics/grid_F1",
        }

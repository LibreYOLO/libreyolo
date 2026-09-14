"""Monocular-depth validator for LibreYOLO.

Computes the standard relative-depth metrics (AbsRel, RMSE, delta thresholds)
over a dense depth validation split, reusing the :class:`BaseValidator`
template (setup -> iterate -> finalize).

Predictions are relative inverse depth; ground truth is plain depth. Each
prediction is aligned to its ground truth with a per-image least-squares
scale and shift in inverse-depth space before metrics are computed, the
standard protocol for affine-invariant depth models. Pixels with invalid
ground truth (``<= 0``) are excluded.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.depth_dataset import (
    DepthDataset,
    depth_collate_fn,
    resolve_depth_data,
)
from .base import BaseValidator

logger = logging.getLogger(__name__)

_EPS = 1e-6

_METRIC_KEYS = (
    "metrics/abs_rel",
    "metrics/rmse",
    "metrics/delta1",
    "metrics/delta2",
    "metrics/delta3",
)


def align_inverse_depth(
    pred: torch.Tensor, gt_inverse: torch.Tensor
) -> torch.Tensor:
    """Least-squares scale-and-shift alignment of a prediction to a target.

    Solves ``min_{s,t} sum((s * pred + t - gt_inverse)^2)`` over the given
    flat tensors (valid pixels only) and returns the aligned prediction.
    Falls back to a median shift when the system is degenerate (e.g. a
    constant prediction) or when the fitted scale is non-positive. Depth
    predictions are relative inverse depth where larger means closer, so a
    negative scale would reward an inverted depth ordering.
    """
    pred = pred.double()
    gt_inverse = gt_inverse.double()
    n = pred.numel()
    sum_p = pred.sum()
    sum_pp = (pred * pred).sum()
    sum_g = gt_inverse.sum()
    sum_pg = (pred * gt_inverse).sum()
    det = sum_pp * n - sum_p * sum_p
    if float(det.abs()) < _EPS:
        return pred + (gt_inverse.median() - pred.median())
    scale = (sum_pg * n - sum_p * sum_g) / det
    shift = (sum_pp * sum_g - sum_p * sum_pg) / det
    if float(scale) <= _EPS:
        return pred + (gt_inverse.median() - pred.median())
    return scale * pred + shift


class DepthValidator(BaseValidator):
    """AbsRel / RMSE / delta-threshold validator for the depth task."""

    task = "depth"

    def _setup_dataloader(self) -> DataLoader:
        if not self.config.data:
            raise ValueError("Depth validation requires data= (a dataset YAML).")
        data_config = resolve_depth_data(
            self.config.data,
            allow_scripts=getattr(self.config, "allow_download_scripts", False),
        )
        split = self.config.split or "val"

        divisor = getattr(self.model, "depth_imgsz_divisor", None)
        if divisor and self.config.imgsz % int(divisor):
            raise ValueError(
                f"Depth validation imgsz={self.config.imgsz} must be "
                f"divisible by {int(divisor)} for this model family."
            )
        resize_mode = getattr(self.model, "depth_resize_mode", "letterbox")
        dataset = DepthDataset(
            data_config,
            split=split,
            imgsz=self.config.imgsz,
            augment=False,
            resize_mode=resize_mode,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=depth_collate_fn,
        )

    def _init_metrics(self) -> None:
        # Per-image accumulation (Eigen/MiDaS protocol): each metric is the
        # mean over an image's valid pixels, then averaged over images. RMSE
        # keeps its own running sum since it is not linear in the pixel terms.
        self._metric_sums = {
            "metrics/abs_rel": 0.0,
            "metrics/delta1": 0.0,
            "metrics/delta2": 0.0,
            "metrics/delta3": 0.0,
        }
        self._rmse_sum = 0.0
        self._image_count = 0

    def _preprocess_batch(self, batch: Any) -> tuple:
        images, targets, img_info, img_ids = batch
        return images, targets, img_info, img_ids

    def _postprocess_predictions(self, preds: Any, batch: Any) -> Any:
        """Decode raw model output into ``[B, H, W]`` inverse-depth maps."""
        output = preds
        if isinstance(output, dict):
            output = output.get("depth", output.get("predictions"))
        if isinstance(output, (list, tuple)):
            output = output[0]
        output = torch.as_tensor(output)

        targets = batch[1]
        target_hw = tuple(targets.shape[-2:])
        if output.ndim == 3:
            output = output.unsqueeze(1)
        if output.ndim != 4 or output.shape[1] != 1:
            raise ValueError(
                f"Depth validation expects [B, 1, H, W] output, got shape "
                f"{tuple(output.shape)}."
            )
        if tuple(output.shape[-2:]) != target_hw:
            output = F.interpolate(
                output.float(), size=target_hw, mode="bilinear", align_corners=False
            )
        return output[:, 0]

    def _update_metrics(
        self, preds: Any, targets: Any, img_info: Any, img_ids: Any = None
    ) -> None:
        preds = preds.detach().cpu().float()
        targets = targets.detach().cpu().float()
        for pred, gt_depth in zip(preds, targets):
            finite = torch.isfinite(pred) & torch.isfinite(gt_depth)
            valid = (gt_depth > 0) & finite
            valid_count = int(valid.sum())
            if valid_count < 2:
                continue
            pred_valid = pred[valid]
            gt = gt_depth[valid].double()
            encoding = getattr(self.model, "depth_encoding", "inverse_depth")
            depth = align_depth_prediction(pred_valid, gt, encoding=encoding)

            residual = depth - gt
            ratio = torch.maximum(depth / gt, gt / depth)
            # Per-image means, accumulated for a final average over images.
            self._metric_sums["metrics/abs_rel"] += float((residual.abs() / gt).mean())
            self._rmse_sum += float(torch.sqrt((residual * residual).mean()))
            self._metric_sums["metrics/delta1"] += float((ratio < 1.25).double().mean())
            self._metric_sums["metrics/delta2"] += float((ratio < 1.25**2).double().mean())
            self._metric_sums["metrics/delta3"] += float((ratio < 1.25**3).double().mean())
            self._image_count += 1

    def _compute_metrics(self) -> Dict[str, float]:
        if self._image_count == 0:
            raise ValueError(
                "Depth validation found no images with at least two valid pixels."
            )
        count = float(self._image_count)
        metrics = {
            "metrics/abs_rel": self._metric_sums["metrics/abs_rel"] / count,
            "metrics/rmse": self._rmse_sum / count,
            "metrics/delta1": self._metric_sums["metrics/delta1"] / count,
            "metrics/delta2": self._metric_sums["metrics/delta2"] / count,
            "metrics/delta3": self._metric_sums["metrics/delta3"] / count,
        }
        metrics["fitness"] = metrics["metrics/delta1"]
        return metrics

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Depth Validation Results")
        logger.info("=" * 50)
        logger.info("  AbsRel: %.4f", metrics.get("metrics/abs_rel", 0.0))
        logger.info("  RMSE:   %.4f", metrics.get("metrics/rmse", 0.0))
        logger.info("  delta1: %.4f", metrics.get("metrics/delta1", 0.0))
        logger.info("  delta2: %.4f", metrics.get("metrics/delta2", 0.0))
        logger.info("  delta3: %.4f", metrics.get("metrics/delta3", 0.0))
        logger.info("=" * 50)


def align_depth_prediction(pred, gt_depth, *, encoding="inverse_depth"):
    """Positive affine alignment in the declared space, then target depth units.

    Ground truth supplies the metric scale. A normalized log-depth prediction
    needs alignment before exponentiation; exp(raw) would invent a scale.
    """
    gt_depth = gt_depth.double()
    if encoding == "inverse_depth":
        aligned = align_inverse_depth(pred, 1.0 / gt_depth)
        return 1.0 / aligned.clamp_min(_EPS)
    if encoding == "depth":
        return align_inverse_depth(pred, gt_depth).clamp_min(_EPS)
    if encoding == "log_depth":
        aligned = align_inverse_depth(pred, gt_depth.log())
        return aligned.clamp(-50, 50).exp()
    raise ValueError(f"Unknown depth encoding: {encoding!r}")


__all__ = ["DepthValidator", "align_inverse_depth", "align_depth_prediction"]

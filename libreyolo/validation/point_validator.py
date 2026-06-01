"""Point-task validator."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

from .detection_validator import DetectionValidator
from .config import ValidationConfig

if TYPE_CHECKING:
    from libreyolo.models.base import BaseModel


def _linear_sum_assignment(cost: torch.Tensor) -> tuple[list[int], list[int]]:
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        return rows.tolist(), cols.tolist()
    except Exception:
        rows: list[int] = []
        cols: list[int] = []
        remaining_rows = set(range(cost.shape[0]))
        remaining_cols = set(range(cost.shape[1]))
        while remaining_rows and remaining_cols:
            best = None
            for r in remaining_rows:
                for c in remaining_cols:
                    value = float(cost[r, c])
                    if best is None or value < best[0]:
                        best = (value, r, c)
            if best is None:
                break
            _, r, c = best
            rows.append(r)
            cols.append(c)
            remaining_rows.remove(r)
            remaining_cols.remove(c)
        return rows, cols


class PointValidator(DetectionValidator):
    """Validator for point-localization models.

    Computes precision, recall, F1, and mean matched distance using one-to-one
    point matching. Model families own prediction decoding and target-space
    conversion; this class owns only matching and metric aggregation.
    """

    task = "point"

    def __init__(
        self,
        model: "BaseModel",
        config: Optional[ValidationConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, config, **kwargs)
        self.distance_tolerance = float(getattr(self.config, "point_distance_tolerance", 1.5))
        self.nms_radius = int(getattr(self.config, "point_nms_radius", 1))
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_distance = 0.0
        self._last_metric_shape = (self.config.imgsz, self.config.imgsz)

    def _init_metrics(self) -> None:
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_distance = 0.0

    def _postprocess_predictions(self, preds: Any, batch: Any) -> List[torch.Tensor]:
        decoder = getattr(self.model, "_decode_point_predictions", None)
        if not callable(decoder):
            raise NotImplementedError(
                f"{self.model.__class__.__name__} must implement "
                "_decode_point_predictions(...) for point validation."
            )
        metric_shape_fn = getattr(self.model, "_point_metric_shape", None)
        if callable(metric_shape_fn):
            self._last_metric_shape = metric_shape_fn(preds, self.config.imgsz)
        decoded = decoder(
            preds,
            conf_thres=self.config.conf_thres,
            max_det=self.config.max_det,
            nms_radius=self.nms_radius,
        )
        return [row.detach().cpu() if isinstance(row, torch.Tensor) else torch.as_tensor(row) for row in decoded]

    def _target_points_from_boxes(
        self, target: torch.Tensor, metric_h: int, metric_w: int
    ) -> torch.Tensor:
        target_encoder = getattr(self.model, "_point_targets_from_boxes", None)
        if callable(target_encoder):
            return target_encoder(
                target,
                metric_shape=(metric_h, metric_w),
                imgsz=self.config.imgsz,
            ).float()
        valid = target[:, 2] > target[:, 0]
        target = target[valid]
        if target.numel() == 0:
            return torch.zeros((0, 3), dtype=torch.float32)
        cx = (target[:, 0] + target[:, 2]) * 0.5 * (metric_w / self.config.imgsz)
        cy = (target[:, 1] + target[:, 3]) * 0.5 * (metric_h / self.config.imgsz)
        cls = target[:, 4]
        return torch.stack((cx, cy, cls), dim=1).float()

    def _update_metrics(
        self, preds: List[torch.Tensor], targets: torch.Tensor, img_info: Any, img_ids: Any = None
    ) -> None:
        if not preds:
            return
        metric_h, metric_w = self._last_metric_shape

        for b, pred_rows in enumerate(preds):
            pred_xy = pred_rows[:, :2].float() if len(pred_rows) else torch.zeros((0, 2))
            pred_cls = pred_rows[:, 2].float() if len(pred_rows) else torch.zeros((0,))
            true_rows = self._target_points_from_boxes(targets[b].cpu(), metric_h, metric_w)
            true_xy = true_rows[:, :2]
            true_cls = true_rows[:, 2]

            if len(pred_xy) == 0 and len(true_xy) == 0:
                continue
            if len(pred_xy) == 0:
                self.total_fn += len(true_xy)
                continue
            if len(true_xy) == 0:
                self.total_fp += len(pred_xy)
                continue

            dist = torch.cdist(pred_xy, true_xy)
            class_mismatch = pred_cls[:, None] != true_cls[None, :]
            dist = dist.masked_fill(class_mismatch, float("inf"))
            finite = torch.isfinite(dist)
            if not finite.any():
                self.total_fp += len(pred_xy)
                self.total_fn += len(true_xy)
                continue

            rows, cols = _linear_sum_assignment(dist)
            matched_preds = set()
            matched_trues = set()
            for r, c in zip(rows, cols):
                value = float(dist[r, c])
                if value <= self.distance_tolerance:
                    self.total_tp += 1
                    self.total_distance += value
                    matched_preds.add(r)
                    matched_trues.add(c)

            self.total_fp += len(pred_xy) - len(matched_preds)
            self.total_fn += len(true_xy) - len(matched_trues)

    def _compute_metrics(self) -> Dict[str, float]:
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) else 0.0
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        mean_distance = self.total_distance / self.total_tp if self.total_tp else 0.0
        return {
            "metrics/precision": precision,
            "metrics/recall": recall,
            "metrics/F1": f1,
            "metrics/mean_distance": mean_distance,
            "metrics/TP": float(self.total_tp),
            "metrics/FP": float(self.total_fp),
            "metrics/FN": float(self.total_fn),
        }

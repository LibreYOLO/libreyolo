"""Point validator for LibreYOLO."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from .base import BaseValidator
from .config import ValidationConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from libreyolo.models.base import BaseModel

# Default distance thresholds in normalised [0, 1] coordinate space.
_DEFAULT_DIST_THRESHOLDS: Tuple[float, ...] = tuple(
    round(v, 10) for v in np.linspace(0.01, 0.10, 10).tolist()
)

def _euclidean_distance_matrix(
    pred_xy: np.ndarray, gt_xy: np.ndarray
) -> np.ndarray:
    """Return an (N_pred, N_gt) Euclidean distance matrix in normalised space."""
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    gt_xy = np.asarray(gt_xy, dtype=np.float64)
    diff = pred_xy[:, np.newaxis, :] - gt_xy[np.newaxis, :, :]  # (N,M,2)
    return np.sqrt((diff ** 2).sum(axis=-1))


def _hungarian_match(
    dist_matrix: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Hungarian algorithm and split matches by distance threshold."""
    n_pred, n_gt = dist_matrix.shape

    if n_pred == 0 or n_gt == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.arange(n_pred, dtype=np.int64),
            np.arange(n_gt, dtype=np.int64),
        )

    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # Keep only pairs within the distance threshold
    dists = dist_matrix[row_ind, col_ind]
    valid = dists <= threshold
    tp_pairs = np.stack([row_ind[valid], col_ind[valid]], axis=1).astype(np.int64)

    matched_pred = set(row_ind[valid].tolist())
    matched_gt = set(col_ind[valid].tolist())

    fp_pred_indices = np.array(
        [i for i in range(n_pred) if i not in matched_pred], dtype=np.int64
    )
    fn_gt_indices = np.array(
        [j for j in range(n_gt) if j not in matched_gt], dtype=np.int64
    )
    return tp_pairs, fp_pred_indices, fn_gt_indices


def _precision_recall_f1(
    tp: int, fp: int, fn: int
) -> Tuple[float, float, float]:
    """Compute precision, recall and F1 from TP / FP / FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _average_precision_at_threshold(
    all_pred_scores: np.ndarray,
    all_pred_matched: np.ndarray,
    n_gt_total: int,
) -> float:
    """Compute AP via the area under the precision-recall curve. Implements the 101-point interpolated AP."""
    if n_gt_total == 0:
        return 0.0

    order = np.argsort(-all_pred_scores)
    matched_sorted = all_pred_matched[order]

    cum_tp = np.cumsum(matched_sorted)
    cum_fp = np.cumsum(~matched_sorted)

    recalls = cum_tp / n_gt_total
    precisions = cum_tp / (cum_tp + cum_fp)

    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[1.0], precisions, [0.0]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    recall_levels = np.linspace(0.0, 1.0, 101)
    ap = float(np.mean([
        precisions[np.searchsorted(recalls, r, side="left")]
        for r in recall_levels
    ]))
    return ap

class _ImageRecord:
    """Accumulates raw per-image matching results for later metric rollup."""

    __slots__ = (
        "pred_xy", "pred_scores", "pred_classes",
        "gt_xy", "gt_classes",
        "n_pred", "n_gt",
    )

    def __init__(
        self,
        pred_xy: np.ndarray,
        pred_scores: np.ndarray,
        pred_classes: np.ndarray,
        gt_xy: np.ndarray,
        gt_classes: np.ndarray,
    ) -> None:
        self.pred_xy = pred_xy
        self.pred_scores = pred_scores
        self.pred_classes = pred_classes
        self.gt_xy = gt_xy
        self.gt_classes = gt_classes
        self.n_pred = len(pred_xy)
        self.n_gt = len(gt_xy)

class PointValidator(BaseValidator):
    """Hungarian-matching point-localisation validator."""

    task = "point"

    def __init__(
        self,
        model: "BaseModel",
        config: Optional[ValidationConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, config, **kwargs)

        raw = getattr(self.config, "dist_thresholds", None)
        if raw is None or len(raw) == 0:
            self._dist_thresholds: Tuple[float, ...] = _DEFAULT_DIST_THRESHOLDS
        else:
            self._dist_thresholds = tuple(sorted(float(t) for t in raw))

        self._primary_threshold: float = self._dist_thresholds[0]
        self._records: List[_ImageRecord] = []
        self.nc: int = getattr(model, "nb_classes", 1)
        self.val_preproc = None  # set in _setup_dataloader
        self._actual_imgsz: int = 640

    def _setup_dataloader(self) -> DataLoader:
        """Build a point-label dataloader from a YOLO data.yaml."""
        from libreyolo.data import load_data_config, get_img_files, img2label_paths
        from libreyolo.data.dataset import YOLODataset

        if not self.config.data and not self.config.data_dir:
            raise ValueError(
                "PointValidator requires data= (yaml path) or data_dir=."
            )

        actual_imgsz = self._resolve_imgsz()
        self.config.imgsz = actual_imgsz
        self._actual_imgsz = actual_imgsz
        img_size = (actual_imgsz, actual_imgsz)

        img_files: Optional[List] = None
        label_files: Optional[List] = None

        if self.config.data:
            data_cfg = load_data_config(
                self.config.data,
                allow_scripts=self.config.allow_download_scripts,
            )
            self.nc = data_cfg.get("nc", self.nc)

            split = self.config.split
            img_files_key = f"{split}_img_files"
            label_files_key = f"{split}_label_files"

            if img_files_key in data_cfg:
                img_files = data_cfg[img_files_key]
                label_files = data_cfg.get(label_files_key)
            else:
                split_val = data_cfg.get(split, f"images/{split}")
                img_files = get_img_files(split_val)
                label_files = img2label_paths(img_files)

            data_dir = data_cfg["root"]
        else:
            data_dir = self.config.data_dir

        self.val_preproc = self.model._get_val_preprocessor(img_size=actual_imgsz)

        dataset = YOLODataset(
            data_dir=str(data_dir),
            split=self.config.split,
            img_size=img_size,
            preproc=self.val_preproc,
        ) if img_files is None else YOLODataset(
            img_files=img_files,
            label_files=label_files,
            img_size=img_size,
            preproc=self.val_preproc,
        )

        use_cuda = self.device.type == "cuda"
        nw = self.config.num_workers

        from libreyolo.validation.detection_validator import val_collate_fn

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=use_cuda,
            prefetch_factor=4 if nw > 0 else None,
            persistent_workers=nw > 0,
            collate_fn=val_collate_fn,
            drop_last=False,
        )

    def _resolve_imgsz(self) -> int:
        if self.config.imgsz is not None:
            return int(self.config.imgsz)
        fn = getattr(self.model, "_get_input_size", None)
        if callable(fn):
            return int(fn())
        return 640

    def _init_metrics(self) -> None:
        self._records = []

    def _preprocess_batch(
        self, batch: Tuple
    ) -> Tuple:
        import torch
        images, targets, img_info, img_ids = batch

        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images)

        images = images.float()

        if getattr(self.val_preproc, "custom_normalization", False):
            pass
        elif self.val_preproc.normalize:
            if images.max() > 1.0:
                images = images / 255.0
        else:
            if images.max() <= 1.0:
                images = images * 255.0

        if images.dim() == 3:
            images = images.unsqueeze(0)

        return images, targets, img_info, img_ids

    def _postprocess_predictions(
        self, preds: Any, batch: Tuple
    ) -> List[Dict[str, np.ndarray]]:
        """Decode raw model output into per-image point lists.

        Expects the model's ``_postprocess`` to return a dict 
        containing ``"points"``.
        """
        import torch

        _images, _targets, img_info, _img_ids = batch
        batch_size = len(img_info)
        uses_letterbox = (
            self.val_preproc is not None and self.val_preproc.uses_letterbox
        )

        results: List[Dict[str, np.ndarray]] = []
        for i in range(batch_size):
            orig_h, orig_w = img_info[i]

            preds_i = self._slice_batch_predictions(preds, i)

            result = self.model._postprocess(
                preds_i,
                conf_thres=self.config.conf_thres,
                iou_thres=self.config.iou_thres,
                original_size=(orig_w, orig_h),
                input_size=self._actual_imgsz,
                letterbox=uses_letterbox,
                max_det=self.config.max_det,
            )

            if result.get("num_detections", 0) > 0 and "points" in result:
                pts = result["points"]
                if isinstance(pts, torch.Tensor):
                    pts = pts.cpu().numpy()
                pts = np.asarray(pts, dtype=np.float64)

                raw_scores = result.get("scores", np.ones(len(pts), np.float32))
                raw_cls = result.get("classes", np.zeros(len(pts), int))

                if isinstance(raw_scores, torch.Tensor):
                    raw_scores = raw_scores.cpu().numpy()
                if isinstance(raw_cls, torch.Tensor):
                    raw_cls = raw_cls.cpu().numpy()

                scores = np.asarray(raw_scores, dtype=np.float32)
                classes = np.asarray(raw_cls, dtype=np.int64)

                xy_norm = pts.copy()
                xy_norm[:, 0] = xy_norm[:, 0] / float(orig_w)
                xy_norm[:, 1] = xy_norm[:, 1] / float(orig_h)
                xy_norm = np.clip(xy_norm, 0.0, 1.0)
            else:
                xy_norm = np.zeros((0, 2), dtype=np.float64)
                scores = np.zeros(0, dtype=np.float32)
                classes = np.zeros(0, dtype=np.int64)

            results.append({"xy_norm": xy_norm, "scores": scores, "classes": classes})

        return results

    def _slice_batch_predictions(self, preds: Any, batch_idx: int) -> Any:
        """Extract predictions for a single image from batched model output."""
        import torch

        if isinstance(preds, dict):
            sliced: Dict[str, Any] = {}
            for key, value in preds.items():
                if isinstance(value, torch.Tensor):
                    sliced[key] = value[batch_idx: batch_idx + 1]
                else:
                    sliced[key] = value
            return sliced
        elif isinstance(preds, torch.Tensor):
            return preds[batch_idx: batch_idx + 1]
        elif isinstance(preds, (list, tuple)):
            return type(preds)(
                self._slice_batch_predictions(p, batch_idx) for p in preds
            )
        return preds

    def _update_metrics(
        self,
        preds: List[Dict[str, np.ndarray]],
        targets: Any,
        img_info: List,
        img_ids: Any = None,
    ) -> None:
        """Parse GT box centres and store per-image records for rollup."""
        import torch

        if isinstance(targets, torch.Tensor):
            targets_np = targets.cpu().numpy()
        else:
            targets_np = np.asarray(targets)

        for i, pred in enumerate(preds):
            orig_h, orig_w = img_info[i]
            gt_xy_norm, gt_cls = self._parse_gt_points(
                targets_np[i], orig_h, orig_w
            )
            self._records.append(
                _ImageRecord(
                    pred_xy=pred["xy_norm"],
                    pred_scores=pred["scores"],
                    pred_classes=pred["classes"],
                    gt_xy=gt_xy_norm,
                    gt_classes=gt_cls,
                )
            )

    def _parse_gt_points(
        self, gt_row: np.ndarray, orig_h: int, orig_w: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a padded GT target row into normalised point coordinates."""
        arr = np.asarray(gt_row, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        if arr.shape[1] < 5:
            return np.zeros((0, 2), np.float64), np.zeros(0, np.int64)

        is_pixel_scaled = len(arr) > 0 and float(np.abs(arr[:, 1:5]).max()) > 1.5

        if is_pixel_scaled:
            valid = (arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1])
            vgt = arr[valid]
            if len(vgt) == 0:
                return np.zeros((0, 2), np.float64), np.zeros(0, np.int64)

            uses_lb = getattr(self.val_preproc, "uses_letterbox", False)
            if uses_lb:
                r, off_x, off_y = self.val_preproc.letterbox_scale(
                    orig_h, orig_w, self._actual_imgsz
                )
                x_orig_px = ((vgt[:, 0] + vgt[:, 2]) / 2.0 - off_x) / r
                y_orig_px = ((vgt[:, 1] + vgt[:, 3]) / 2.0 - off_y) / r
            else:
                sx = self._actual_imgsz / float(orig_w)
                sy = self._actual_imgsz / float(orig_h)
                x_orig_px = (vgt[:, 0] + vgt[:, 2]) / 2.0 / sx
                y_orig_px = (vgt[:, 1] + vgt[:, 3]) / 2.0 / sy

            x_norm = np.clip(x_orig_px / float(orig_w), 0.0, 1.0)
            y_norm = np.clip(y_orig_px / float(orig_h), 0.0, 1.0)
            xy_norm = np.stack([x_norm, y_norm], axis=1).astype(np.float64)
            classes = np.clip(vgt[:, 4].astype(int), 0, self.nc - 1)
        else:
            valid = (arr[:, 3] > 0) & (arr[:, 4] > 0)
            vgt = arr[valid]
            if len(vgt) == 0:
                return np.zeros((0, 2), np.float64), np.zeros(0, np.int64)
            xy_norm = np.stack(
                [np.clip(vgt[:, 1], 0.0, 1.0),
                 np.clip(vgt[:, 2], 0.0, 1.0)],
                axis=1,
            ).astype(np.float64)
            classes = np.clip(vgt[:, 0].astype(int), 0, self.nc - 1)

        return xy_norm, classes.astype(np.int64)

    # =========================================================================
    # Metrics computation
    # =========================================================================

    def _map_sweep_key(self) -> str:
        """Return the metric key for the mAP sweep over all distance thresholds."""
        lo = self._dist_thresholds[0]
        hi = self._dist_thresholds[-1]
        return f"metrics/mAP@[{lo:.2f}:{hi:.2f}]"

    def _compute_metrics(self) -> Dict[str, float]:
        """Roll up per-image records into final scalar metrics."""
        if not self._records:
            return self._zero_metrics()

        metrics: Dict[str, float] = {}

        map_at_thresholds: List[float] = []

        cls_n_gt: Dict[int, int] = {}
        for rec in self._records:
            for c in rec.gt_classes.tolist():
                cls_n_gt[c] = cls_n_gt.get(c, 0) + 1

        for threshold in self._dist_thresholds:
            dist_tp_list: List[float] = []
            cls_metrics: Dict[int, Dict[str, int]] = {
                c: {"tp": 0, "fp": 0, "fn": 0} for c in cls_n_gt
            }

            cls_scored_preds: Dict[int, List[Tuple[float, bool]]] = {}

            for rec in self._records:
                if rec.n_pred == 0 and rec.n_gt == 0:
                    continue

                all_classes = sorted(
                    set(rec.pred_classes.tolist()) | set(rec.gt_classes.tolist())
                )

                for cls in all_classes:
                    pred_mask = rec.pred_classes == cls
                    gt_mask = rec.gt_classes == cls

                    cls_pred_xy = rec.pred_xy[pred_mask]
                    cls_pred_scores = rec.pred_scores[pred_mask]
                    cls_gt_xy = rec.gt_xy[gt_mask]

                    n_pred_cls = int(pred_mask.sum())
                    n_gt_cls = int(gt_mask.sum())

                    if n_pred_cls == 0 and n_gt_cls == 0:
                        continue

                    if n_pred_cls == 0:
                        if cls in cls_metrics:
                            cls_metrics[cls]["fn"] += n_gt_cls
                        continue

                    if cls not in cls_scored_preds:
                        cls_scored_preds[cls] = []

                    if n_gt_cls == 0:
                        if cls in cls_metrics:
                            cls_metrics[cls]["fp"] += n_pred_cls
                        for score in cls_pred_scores:
                            cls_scored_preds[cls].append((float(score), False))
                        continue

                    dist_mat = _euclidean_distance_matrix(cls_pred_xy, cls_gt_xy)
                    tp_pairs, fp_idx, fn_idx = _hungarian_match(dist_mat, threshold)

                    if cls in cls_metrics:
                        cls_metrics[cls]["tp"] += len(tp_pairs)
                        cls_metrics[cls]["fp"] += len(fp_idx)
                        cls_metrics[cls]["fn"] += len(fn_idx)

                    if (
                        math.isclose(threshold, self._primary_threshold)
                        and len(tp_pairs) > 0
                    ):
                        dists = dist_mat[tp_pairs[:, 0], tp_pairs[:, 1]]
                        dist_tp_list.extend(dists.tolist())

                    matched_pred_set = (
                        set(tp_pairs[:, 0].tolist()) if len(tp_pairs) else set()
                    )
                    for pi, score in enumerate(cls_pred_scores):
                        cls_scored_preds[cls].append((float(score), pi in matched_pred_set))

            per_class_precisions: List[float] = []
            per_class_recalls: List[float] = []
            per_class_f1s: List[float] = []

            for cls in cls_n_gt:
                c_tp = cls_metrics[cls]["tp"]
                c_fp = cls_metrics[cls]["fp"]
                c_fn = cls_metrics[cls]["fn"]
                c_prec, c_rec, c_f1 = _precision_recall_f1(c_tp, c_fp, c_fn)
                per_class_precisions.append(c_prec)
                per_class_recalls.append(c_rec)
                per_class_f1s.append(c_f1)

            precision = float(np.mean(per_class_precisions)) if per_class_precisions else 0.0
            recall = float(np.mean(per_class_recalls)) if per_class_recalls else 0.0
            f1 = float(np.mean(per_class_f1s)) if per_class_f1s else 0.0

            # Macro-averaging
            per_class_aps: List[float] = []
            for cls, n_gt in cls_n_gt.items():
                pairs = cls_scored_preds.get(cls, [])
                if not pairs:
                    per_class_aps.append(0.0)
                else:
                    scores_arr = np.array([s for s, _ in pairs], dtype=np.float64)
                    matched_arr = np.array([m for _, m in pairs], dtype=bool)
                    per_class_aps.append(
                        _average_precision_at_threshold(scores_arr, matched_arr, n_gt)
                    )

            ap = float(np.mean(per_class_aps)) if per_class_aps else 0.0
            map_at_thresholds.append(ap)

            if math.isclose(threshold, self._primary_threshold):
                metrics["metrics/precision"] = precision
                metrics["metrics/recall"] = recall
                metrics["metrics/f1"] = f1
                metrics[f"metrics/mAP@{threshold:.2f}"] = ap
                if dist_tp_list:
                    metrics["metrics/MLE"] = float(np.mean(dist_tp_list))
                else:
                    metrics["metrics/MLE"] = 0.0

        metrics[self._map_sweep_key()] = float(np.mean(map_at_thresholds))

        count_errors = np.array(
            [r.n_pred - r.n_gt for r in self._records], dtype=np.float64
        )
        abs_errors = np.abs(count_errors)
        metrics["metrics/MAE"] = float(abs_errors.mean())
        metrics["metrics/RMSE"] = float(np.sqrt((count_errors ** 2).mean()))

        # mAP sweep average
        metrics["fitness"] = metrics[self._map_sweep_key()]

        return metrics

    def _zero_metrics(self) -> Dict[str, float]:
        """Return a zero-filled metrics dict when there are no records."""
        return {
            "metrics/precision": 0.0,
            "metrics/recall": 0.0,
            "metrics/f1": 0.0,
            f"metrics/mAP@{self._primary_threshold:.2f}": 0.0,
            self._map_sweep_key(): 0.0,
            "metrics/MLE": 0.0,
            "metrics/MAE": 0.0,
            "metrics/RMSE": 0.0,
            "fitness": 0.0,
        }

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 60)
        logger.info("Point Localisation Validation Results")
        logger.info("=" * 60)
        for key in (
            "metrics/precision",
            "metrics/recall",
            "metrics/f1",
            f"metrics/mAP@{self._primary_threshold:.2f}",
            self._map_sweep_key(),
            "metrics/MLE",
            "metrics/MAE",
            "metrics/RMSE",
        ):
            v = metrics.get(key, 0.0)
            logger.info("  %-35s %.4f", key, v)
        logger.info("-" * 60)
        logger.info("  Images processed: %d", self.seen)
        logger.info("  Distance thresholds: %s", list(self._dist_thresholds))
        logger.info("=" * 60)

    # TODO: implement plotting for point detection, 
    # such as showing predictions on images
    def _save_plots(self, *args, **kwargs) -> None:  # type: ignore[override]
        logger.info("save_plots not yet supported for PointValidator")


__all__ = ["PointValidator"]

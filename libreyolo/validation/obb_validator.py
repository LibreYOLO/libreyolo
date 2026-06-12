"""Oriented bounding box validator for YOLO OBB datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from libreyolo.data import get_img_files, img2label_paths, load_data_config
from libreyolo.data.dataset import YOLODataset
from libreyolo.data.obb import (
    corners_to_xywhr,
    parse_yolo_obb_label_line,
    xywhr_iou,
)

from ..postprocess.slicing import slice_batch_outputs
from .base import BaseValidator
from .config import ValidationConfig
from .detection_validator import val_collate_fn

if TYPE_CHECKING:
    from libreyolo.models.base import BaseModel

logger = logging.getLogger(__name__)


class _OBBValPreprocessor:
    """Delegate image preprocessing while tolerating six-column OBB targets."""

    def __init__(self, base_preprocessor):
        self.base_preprocessor = base_preprocessor

    def __getattr__(self, name):
        return getattr(self.base_preprocessor, name)

    def __call__(self, img: np.ndarray, targets: np.ndarray, input_size: tuple):
        target_view = targets[:, :5] if targets.shape[1] > 5 else targets
        return self.base_preprocessor(img, target_view, input_size)


class OBBValidator(BaseValidator):
    """Validate YOLO-format OBB datasets with rotated-IoU AP metrics."""

    task = "obb"

    def __init__(
        self,
        model: "BaseModel",
        config: Optional[ValidationConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, config, **kwargs)
        self.nc = model.nb_classes
        self.class_names: Optional[List[str]] = None
        self.iou_thresholds = tuple(float(v) for v in self.config.iou_thresholds)
        self.val_preproc = None
        self._actual_imgsz = self.config.imgsz
        self._gt_by_image: Dict[int, List[tuple[int, np.ndarray]]] = {}
        self._gt_by_class: Dict[int, Dict[int, List[np.ndarray]]] = {}
        self._num_gt_by_class: Dict[int, int] = {}
        self._predictions_by_class: Dict[int, List[dict]] = {}

    def _resolve_imgsz(self) -> int:
        if self.config.imgsz is not None:
            return int(self.config.imgsz)
        get_input_size = getattr(self.model, "_get_input_size", None)
        if callable(get_input_size):
            return int(get_input_size())
        return 640

    def _setup_dataloader(self) -> DataLoader:
        actual_imgsz = self._resolve_imgsz()
        self.config.imgsz = actual_imgsz
        self._actual_imgsz = actual_imgsz
        img_size = (actual_imgsz, actual_imgsz)

        img_files: List[Path] | None = None
        label_files: List[Path] | None = None

        if self.config.data:
            data_cfg = load_data_config(
                self.config.data,
                allow_scripts=self.config.allow_download_scripts,
            )
            names = data_cfg.get("names")
            data_nc = data_cfg.get("nc")
            if data_nc is None and names is not None:
                data_nc = len(names)
            if data_nc is not None:
                self.nc = int(data_nc)
            if isinstance(names, dict):
                parsed_names = {}
                for key, value in names.items():
                    try:
                        parsed_names[int(key)] = str(value)
                    except (TypeError, ValueError):
                        continue
                self.class_names = [
                    parsed_names.get(i, f"class_{i}") for i in range(self.nc)
                ]
            elif names is not None:
                parsed_names = [str(name) for name in list(names)[: self.nc]]
                self.class_names = parsed_names + [
                    f"class_{i}" for i in range(len(parsed_names), self.nc)
                ]

            split = self.config.split
            img_files = data_cfg.get(f"{split}_img_files")
            label_files = data_cfg.get(f"{split}_label_files")
            if img_files is None:
                split_path = Path(data_cfg.get(split, Path(data_cfg["path"]) / "images" / split))
                img_files = get_img_files(split_path)
                label_files = img2label_paths(img_files)
        elif self.config.data_dir:
            data_path = Path(self.config.data_dir)
            split_path = data_path / "images" / self.config.split
            img_files = get_img_files(split_path)
            label_files = img2label_paths(img_files)
        else:
            raise RuntimeError("OBB validation requires config.data or config.data_dir")

        if not img_files:
            raise RuntimeError(
                f"No {self.config.split} images found for OBB validation."
            )

        img_files = [Path(p) for p in img_files]
        label_files = [Path(p) for p in (label_files or img2label_paths(img_files))]
        self.val_preproc = _OBBValPreprocessor(
            self.model._get_val_preprocessor(img_size=actual_imgsz)
        )

        dataset = YOLODataset(
            img_files=img_files,
            label_files=label_files,
            img_size=img_size,
            preproc=self.val_preproc,
            load_obb=True,
            num_classes=self.nc,
        )
        self._gt_by_image = self._load_ground_truths(
            img_files,
            label_files,
            warn_invalid=False,
        )

        use_cuda = torch.cuda.is_available() and self.device.type == "cuda"
        nw = self.config.num_workers
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

    def _load_ground_truths(
        self,
        img_files: List[Path],
        label_files: List[Path],
        *,
        warn_invalid: bool = True,
    ) -> Dict[int, List[tuple[int, np.ndarray]]]:
        gt_by_image: Dict[int, List[tuple[int, np.ndarray]]] = {}
        skipped_rows = 0
        skipped_files = 0
        first_error = None
        for img_id, (img_file, label_file) in enumerate(zip(img_files, label_files)):
            with Image.open(img_file) as im:
                width, height = im.size
            rows: List[tuple[int, np.ndarray]] = []
            file_skipped_rows = 0
            if label_file.exists():
                for line in label_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        cls_id, corners = parse_yolo_obb_label_line(
                            line,
                            num_classes=self.nc,
                            clip=True,
                        )
                    except ValueError as exc:
                        skipped_rows += 1
                        file_skipped_rows += 1
                        first_error = first_error or f"{label_file.name}: {exc}"
                        continue
                    pixel_corners = corners.copy()
                    pixel_corners[:, 0] *= width
                    pixel_corners[:, 1] *= height
                    rows.append((cls_id, corners_to_xywhr(pixel_corners)))
            if file_skipped_rows:
                skipped_files += 1
            gt_by_image[img_id] = rows
        if skipped_rows and warn_invalid:
            logger.warning(
                "Skipped %d invalid YOLO OBB ground-truth rows across %d files. "
                "First invalid row: %s",
                skipped_rows,
                skipped_files,
                first_error,
            )
        return gt_by_image

    def _init_metrics(self) -> None:
        self._predictions_by_class = {i: [] for i in range(self.nc)}
        self._gt_by_class = {i: {} for i in range(self.nc)}
        self._num_gt_by_class = {i: 0 for i in range(self.nc)}
        for img_id, rows in self._gt_by_image.items():
            for cls_id, xywhr in rows:
                self._gt_by_class.setdefault(cls_id, {}).setdefault(img_id, []).append(xywhr)
                self._num_gt_by_class[cls_id] = self._num_gt_by_class.get(cls_id, 0) + 1

    def _preprocess_batch(self, batch: Any) -> tuple:
        images, targets, img_info, img_ids = batch
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        images = images.float()
        if getattr(self.val_preproc, "normalize", True) and images.max() > 1.0:
            images = images / 255.0
        return images, targets, img_info, img_ids

    def _postprocess_predictions(self, preds: Any, batch: Any) -> List[Dict[str, torch.Tensor]]:
        _images, _targets, img_info, _img_ids = batch
        batch_size = len(img_info)
        uses_letterbox = self.val_preproc is not None and self.val_preproc.uses_letterbox

        detections = []
        for i in range(batch_size):
            orig_h, orig_w = img_info[i]
            result = self.model._postprocess(
                self._slice_batch_predictions(preds, i),
                conf_thres=self.config.conf_thres,
                iou_thres=self.config.iou_thres,
                original_size=(orig_w, orig_h),
                input_size=self._actual_imgsz,
                letterbox=uses_letterbox,
                max_det=self.config.max_det,
            )
            raw_obb = result.get("obb")
            if result["num_detections"] > 0 and raw_obb is not None:
                obb = (
                    raw_obb.to(self.device).float()
                    if isinstance(raw_obb, torch.Tensor)
                    else torch.as_tensor(raw_obb, dtype=torch.float32, device=self.device)
                )
            else:
                obb = torch.zeros((0, 7), dtype=torch.float32, device=self.device)
            detections.append({"obb": obb})
        return detections

    def _slice_batch_predictions(self, preds: Any, batch_idx: int) -> Any:
        return slice_batch_outputs(preds, batch_idx)

    def _update_metrics(
        self,
        preds: List[Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        img_info: List,
        img_ids: List | None = None,
    ) -> None:
        del targets, img_info
        if img_ids is None:
            raise RuntimeError("img_ids are required for OBB validation")

        for pred, img_id in zip(preds, img_ids):
            image_key = int(img_id.item()) if hasattr(img_id, "item") else int(img_id)
            obb = pred["obb"].detach().cpu().numpy()
            for row in obb:
                cls_id = int(row[-1])
                if cls_id < 0 or cls_id >= self.nc:
                    continue
                self._predictions_by_class.setdefault(cls_id, []).append(
                    {
                        "image_id": image_key,
                        "score": float(row[-2]),
                        "xywhr": np.asarray(row[:5], dtype=np.float32),
                    }
                )

    @staticmethod
    def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
        if recall.size == 0:
            return 0.0
        points = np.linspace(0.0, 1.0, 101)
        values = [
            float(precision[recall >= point].max()) if np.any(recall >= point) else 0.0
            for point in points
        ]
        return float(np.mean(values))

    def _evaluate_class(self, cls_id: int, iou_threshold: float) -> tuple[float, float, float] | None:
        n_gt = self._num_gt_by_class.get(cls_id, 0)
        if n_gt == 0:
            return None

        preds = sorted(
            self._predictions_by_class.get(cls_id, []),
            key=lambda item: item["score"],
            reverse=True,
        )
        matched = {
            img_id: np.zeros(len(rows), dtype=bool)
            for img_id, rows in self._gt_by_class.get(cls_id, {}).items()
        }
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for pred_idx, pred in enumerate(preds):
            image_id = pred["image_id"]
            gt_rows = self._gt_by_class.get(cls_id, {}).get(image_id, [])
            if not gt_rows:
                fp[pred_idx] = 1.0
                continue

            ious = np.asarray(
                [xywhr_iou(pred["xywhr"], gt) for gt in gt_rows],
                dtype=np.float32,
            )
            best_idx = int(ious.argmax()) if ious.size else -1
            if (
                best_idx >= 0
                and float(ious[best_idx]) >= iou_threshold
                and not matched[image_id][best_idx]
            ):
                tp[pred_idx] = 1.0
                matched[image_id][best_idx] = True
            else:
                fp[pred_idx] = 1.0

        if len(preds) == 0:
            return 0.0, 0.0, 0.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / max(float(n_gt), 1.0)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        ap = self._average_precision(recall, precision)
        final_precision = float(cum_tp[-1] / max(cum_tp[-1] + cum_fp[-1], 1e-9))
        final_recall = float(cum_tp[-1] / max(float(n_gt), 1.0))
        return ap, final_precision, final_recall

    def _compute_metrics(self) -> Dict[str, float]:
        threshold_maps = []
        threshold_precisions = []
        threshold_recalls = []
        mAP50 = mAP75 = precision50 = recall50 = 0.0

        for threshold in self.iou_thresholds:
            aps = []
            precisions = []
            recalls = []
            for cls_id in range(self.nc):
                result = self._evaluate_class(cls_id, threshold)
                if result is None:
                    continue
                ap, precision, recall = result
                aps.append(ap)
                precisions.append(precision)
                recalls.append(recall)

            threshold_map = float(np.mean(aps)) if aps else 0.0
            threshold_precision = float(np.mean(precisions)) if precisions else 0.0
            threshold_recall = float(np.mean(recalls)) if recalls else 0.0
            threshold_maps.append(threshold_map)
            threshold_precisions.append(threshold_precision)
            threshold_recalls.append(threshold_recall)

            if abs(threshold - 0.50) < 1e-6:
                mAP50 = threshold_map
                precision50 = threshold_precision
                recall50 = threshold_recall
            if abs(threshold - 0.75) < 1e-6:
                mAP75 = threshold_map

        mAP = float(np.mean(threshold_maps)) if threshold_maps else 0.0
        metrics = {
            "metrics/precision": precision50,
            "metrics/recall": recall50,
            "metrics/mAP50": mAP50,
            "metrics/mAP75": mAP75,
            "metrics/mAP50-95": mAP,
            "metrics/precision(OBB)": precision50,
            "metrics/recall(OBB)": recall50,
            "metrics/mAP50(OBB)": mAP50,
            "metrics/mAP50-95(OBB)": mAP,
        }
        if self.config.save_json:
            logger.warning("OBB validation does not support save_json yet.")
        if self.config.save_plots:
            logger.warning("OBB validation plots are not implemented yet.")
        return metrics

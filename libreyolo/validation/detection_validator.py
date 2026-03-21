"""Detection validator for LibreYOLO."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import BaseValidator
from .config import ValidationConfig
from .metrics import DetMetrics
from .utils import process_batch

from libreyolo.utils.postprocess import resolve_nms_fn

if TYPE_CHECKING:
    from libreyolo.models.base import BaseModel


def val_collate_fn(batch):
    """Collate validation batch: stack preprocessed images and padded targets."""
    imgs, targets, img_infos, img_ids = zip(*batch)
    imgs = torch.from_numpy(np.stack(imgs))
    targets = torch.from_numpy(np.stack(targets))
    return imgs, targets, img_infos, img_ids


class DetectionValidator(BaseValidator):
    """
    Validator for object detection models.

    Computes mAP50, mAP50-95, precision, recall, and per-class AP.
    Supports both COCO evaluation API and legacy DetMetrics.
    """

    task = "detect"

    def __init__(
        self,
        model: "BaseModel",
        config: Optional[ValidationConfig] = None,
        postprocess=None,
        **kwargs,
    ) -> None:
        super().__init__(model, config, **kwargs)

        self.nms_fn = resolve_nms_fn(postprocess)
        self.metrics: Optional[DetMetrics] = None
        self.coco_evaluator = None
        self.class_names: Optional[List[str]] = None
        self.iou_thresholds = torch.tensor(self.config.iou_thresholds)
        self.nc = model.nb_classes
        self.val_preproc = None  # set in _setup_dataloader

    # =========================================================================
    # Setup
    # =========================================================================

    def _setup_dataloader(self) -> DataLoader:
        """
        Create validation dataloader from config.

        Supports directory-based datasets, .txt file format, and COCO JSON.
        """
        from libreyolo.data import load_data_config, get_img_files, img2label_paths
        from libreyolo.data.dataset import YOLODataset, COCODataset
        from torch.utils.data import DataLoader

        # Use model's native input size if available (e.g. YOLOX nano uses 416)
        model_input_size = (
            self.model._get_input_size()
            if hasattr(self.model, "_get_input_size")
            else None
        )
        if model_input_size is not None and model_input_size != self.config.imgsz:
            actual_imgsz = model_input_size
        else:
            actual_imgsz = self.config.imgsz

        self._actual_imgsz = actual_imgsz
        img_size = (actual_imgsz, actual_imgsz)

        img_files = None
        label_files = None
        split_name = self.config.split
        data_cfg = None

        if self.config.data:
            data_cfg = load_data_config(self.config.data)
            data_dir = data_cfg["root"]
            self.nc = data_cfg.get("nc", self.nc)

            names = data_cfg.get("names", None)
            if isinstance(names, dict):
                self.class_names = [names[i] for i in range(len(names))]
            else:
                self.class_names = names

            # Check for pre-resolved file lists (from .txt format)
            img_files_key = f"{self.config.split}_img_files"
            label_files_key = f"{self.config.split}_label_files"

            if img_files_key in data_cfg:
                img_files = data_cfg[img_files_key]
                label_files = data_cfg.get(label_files_key)
            else:
                split_path_str = data_cfg.get(
                    self.config.split, f"images/{self.config.split}"
                )

                if str(split_path_str).endswith(".txt"):
                    txt_path = Path(data_cfg["path"]) / split_path_str
                    if txt_path.exists():
                        try:
                            img_files = get_img_files(txt_path)
                            label_files = img2label_paths(img_files)
                        except (FileNotFoundError, ValueError):
                            pass
                else:
                    # Directory format
                    full_split_path = Path(data_cfg["path"]) / split_path_str

                    if full_split_path.exists():
                        img_files_list = []
                        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                            img_files_list.extend(full_split_path.glob(ext))
                            img_files_list.extend(full_split_path.glob(ext.upper()))

                        if img_files_list:
                            img_files = sorted(img_files_list)
                            label_files = img2label_paths(img_files)
                    else:
                        if "/" in str(split_path_str):
                            split_name = str(split_path_str).split("/")[-1]
                        else:
                            split_name = str(split_path_str)
        else:
            data_dir = self.config.data_dir
            self.class_names = None

        self.val_preproc = self.model._get_val_preprocessor(img_size=actual_imgsz)

        # Determine dataset format
        data_path = Path(data_dir)

        if img_files is not None:
            # File list mode (.txt format)
            dataset = YOLODataset(
                img_files=img_files,
                label_files=label_files,
                img_size=img_size,
                preproc=self.val_preproc,
            )
        elif (data_path / "annotations").exists():
            # COCO format (JSON annotations)
            json_file = f"instances_{self.config.split}2017.json"
            if not (data_path / "annotations" / json_file).exists():
                json_file = f"instances_{self.config.split}.json"

            split_name = (
                f"{self.config.split}2017" if "2017" in json_file else self.config.split
            )
            if (data_path / "images" / split_name).exists():
                split_name = f"images/{split_name}"

            dataset = COCODataset(
                data_dir=str(data_path),
                json_file=json_file,
                name=split_name,
                img_size=img_size,
                preproc=self.val_preproc,
            )
        else:
            # YOLO directory format
            dataset = YOLODataset(
                data_dir=str(data_path),
                split=split_name,
                img_size=img_size,
                preproc=self.val_preproc,
            )

        use_cuda = torch.cuda.is_available() and self.device.type == "cuda"
        nw = self.config.num_workers

        dataloader = DataLoader(
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

        return dataloader

    def _init_metrics(self) -> None:
        if self.config.use_coco_eval:
            try:
                from libreyolo.data import create_yolo_coco_api
                from libreyolo.validation import COCOEvaluator

                if self.config.verbose:
                    print("Initializing COCO evaluator...")

                coco_api = create_yolo_coco_api(self.config.data, self.config.split)
                self.coco_evaluator = COCOEvaluator(coco_api, iou_type="bbox")

                if self.config.verbose:
                    print(
                        f"COCO evaluator initialized with {len(coco_api.imgs)} images"
                    )
            except Exception as e:
                print(f"Warning: Failed to initialize COCO evaluator: {e}")
                print("Falling back to legacy DetMetrics")
                self.config.use_coco_eval = False
                self.coco_evaluator = None

        if not self.config.use_coco_eval or self.coco_evaluator is None:
            self.metrics = DetMetrics(
                nc=self.nc,
                conf=0.25,
                iou_thresholds=self.config.iou_thresholds,
            )

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    def _preprocess_batch(
        self, batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
        images, targets, img_info, img_ids = batch

        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        images = images.float()

        # Normalization depends on preprocessor:
        # - custom_normalization: already applied (e.g. RF-DETR ImageNet mean/std)
        # - normalize=True: model expects 0-1 (standard YOLO)
        # - normalize=False: model expects 0-255 (YOLOX)
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

    def _slice_batch_predictions(self, preds: Any, batch_idx: int) -> Any:
        """Extract predictions for a single image from batched model output."""
        if isinstance(preds, dict):
            sliced = {}
            for key, value in preds.items():
                if isinstance(value, dict):
                    sliced[key] = {
                        k: v[batch_idx : batch_idx + 1]
                        if isinstance(v, torch.Tensor)
                        else v
                        for k, v in value.items()
                    }
                elif isinstance(value, torch.Tensor):
                    sliced[key] = value[batch_idx : batch_idx + 1]
                else:
                    sliced[key] = value
            return sliced
        elif isinstance(preds, torch.Tensor):
            return preds[batch_idx : batch_idx + 1]
        elif isinstance(preds, (list, tuple)):
            return type(preds)(
                p[batch_idx : batch_idx + 1] if isinstance(p, torch.Tensor) else p
                for p in preds
            )
        else:
            return preds

    def _postprocess_predictions(
        self, preds: Any, batch: Tuple
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Postprocess raw model output into detection dicts.

        Returns:
            List of dicts per image with keys: boxes (xyxy), scores, classes.
        """
        images, targets, img_info, img_ids = batch
        batch_size = len(img_info)

        detections = []
        for i in range(batch_size):
            orig_h, orig_w = img_info[i]
            single_preds = self._slice_batch_predictions(preds, i)

            uses_letterbox = (
                self.val_preproc is not None and self.val_preproc.uses_letterbox
            )
            result = self.model._postprocess(
                single_preds,
                conf_thres=self.config.conf_thres,
                iou_thres=self.config.iou_thres,
                original_size=(orig_w, orig_h),  # (width, height)
                input_size=self._actual_imgsz,
                letterbox=uses_letterbox,
                nms_fn=self.nms_fn,
            )

            if result["num_detections"] > 0:
                boxes = torch.tensor(
                    result["boxes"], dtype=torch.float32, device=self.device
                )
                scores = torch.tensor(
                    result["scores"], dtype=torch.float32, device=self.device
                )
                classes = torch.tensor(
                    result["classes"], dtype=torch.int64, device=self.device
                )
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
                scores = torch.zeros(0, dtype=torch.float32, device=self.device)
                classes = torch.zeros(0, dtype=torch.int64, device=self.device)

            detections.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "classes": classes,
                }
            )

        return detections

    # =========================================================================
    # Metrics
    # =========================================================================

    def _update_metrics(
        self,
        preds: List[Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        img_info: List,
        img_ids: List | None = None,
    ) -> None:
        batch_size = len(preds)

        if self.coco_evaluator is not None and img_ids is not None:
            for i in range(batch_size):
                self.coco_evaluator.update(preds[i], img_ids[i])

        if self.coco_evaluator is not None:
            return

        uses_letterbox = (
            self.val_preproc is not None and self.val_preproc.uses_letterbox
        )

        for i in range(batch_size):
            pred = preds[i]
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_classes = pred["classes"]

            # targets: (B, max_labels, 5) with [x1, y1, x2, y2, class]
            if isinstance(targets, torch.Tensor):
                gt = targets[i]  # (max_labels, 5)
            else:
                gt = torch.from_numpy(targets[i])

            # Filter padding (all-zero boxes)
            valid_mask = gt[:, :4].sum(dim=1) > 0
            gt = gt[valid_mask]

            if len(gt) > 0:
                gt_boxes = gt[:, :4].clone().to(self.device)
                gt_classes = gt[:, 4].long().to(self.device)

                # Scale GT boxes from model input coords back to original image coords
                # (predictions are already in original coords from postprocess)
                orig_h, orig_w = img_info[i]
                img_h, img_w = self._actual_imgsz, self._actual_imgsz

                if uses_letterbox:
                    # Letterbox: GT scaled by r = min(img_h/orig_h, img_w/orig_w)
                    r = min(img_h / orig_h, img_w / orig_w)
                    gt_boxes[:, :4] = gt_boxes[:, :4] / r
                else:
                    # Simple resize: x_input = x_orig * (input_w/orig_w)
                    gt_boxes[:, 0] = gt_boxes[:, 0] * orig_w / img_w
                    gt_boxes[:, 1] = gt_boxes[:, 1] * orig_h / img_h
                    gt_boxes[:, 2] = gt_boxes[:, 2] * orig_w / img_w
                    gt_boxes[:, 3] = gt_boxes[:, 3] * orig_h / img_h
            else:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
                gt_classes = torch.zeros(0, dtype=torch.int64, device=self.device)

            correct, conf, pred_cls, target_cls = process_batch(
                pred_boxes,
                pred_scores,
                pred_classes,
                gt_boxes,
                gt_classes,
                self.iou_thresholds.to(self.device),
            )

            self.metrics.update(correct, conf, pred_cls, target_cls)

    def _compute_metrics(self) -> Dict[str, float]:
        if self.coco_evaluator is not None:
            if self.config.verbose:
                print("\nComputing COCO metrics...")

            save_json = None
            if self.config.save_json:
                save_json = str(self.save_dir / "predictions.json")

            coco_metrics = self.coco_evaluator.compute(save_json=save_json)

            return {
                "metrics/mAP50-95": coco_metrics["mAP"],
                "metrics/mAP50": coco_metrics["mAP50"],
                "metrics/mAP75": coco_metrics["mAP75"],
                "metrics/mAP_small": coco_metrics["mAP_small"],
                "metrics/mAP_medium": coco_metrics["mAP_medium"],
                "metrics/mAP_large": coco_metrics["mAP_large"],
                "metrics/AR1": coco_metrics["AR1"],
                "metrics/AR10": coco_metrics["AR10"],
                "metrics/AR100": coco_metrics["AR100"],
                "metrics/AR_small": coco_metrics["AR_small"],
                "metrics/AR_medium": coco_metrics["AR_medium"],
                "metrics/AR_large": coco_metrics["AR_large"],
            }
        else:
            return self.metrics.compute()

"""
Detection validator for LibreYOLO.

Implements validation for object detection models with mAP computation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import BaseValidator
from .config import ValidationConfig
from .metrics import ConfusionMatrix, DetMetrics
from .utils import process_batch

if TYPE_CHECKING:
    from libreyolo.common.base_model import LibreYOLOBase


def val_collate_fn(batch):
    """
    Collate function for validation dataloader.

    Args:
        batch: List of (image, targets, img_info, img_id) tuples.

    Returns:
        Tuple of (images, targets, img_infos, img_ids).
    """
    imgs, targets, img_infos, img_ids = zip(*batch)

    # Stack images (already preprocessed to same size)
    imgs = torch.from_numpy(np.stack(imgs))

    # Stack targets (already padded to same size)
    targets = torch.from_numpy(np.stack(targets))

    return imgs, targets, img_infos, img_ids


class DetectionValidator(BaseValidator):
    """
    Validator for object detection models.

    Computes detection metrics including:
    - mAP50: Mean Average Precision at IoU threshold 0.50
    - mAP50-95: Mean Average Precision at IoU thresholds 0.50 to 0.95
    - Precision and Recall at default IoU threshold
    - Per-class AP

    Attributes:
        metrics: DetMetrics instance for mAP computation.
        confusion_matrix: ConfusionMatrix for visualization.
        class_names: List of class names.
        iou_thresholds: IoU thresholds for evaluation.
    """

    task = "detect"

    def __init__(
        self,
        model: "LibreYOLOBase",
        config: Optional[ValidationConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize detection validator.

        Args:
            model: LibreYOLO model instance.
            config: ValidationConfig instance.
            **kwargs: Override config parameters.
        """
        super().__init__(model, config, **kwargs)

        # Detection-specific attributes
        self.metrics: Optional[DetMetrics] = None
        self.confusion_matrix: Optional[ConfusionMatrix] = None
        self.class_names: Optional[List[str]] = None
        self.iou_thresholds = torch.tensor(self.config.iou_thresholds)
        self.nc = model.nb_classes
        self.val_preproc = None  # Set in _setup_dataloader

    def _setup_dataloader(self) -> DataLoader:
        """
        Create validation dataloader from config.

        Supports both:
        - Directory-based datasets (images/val, labels/val structure)
        - .txt file format (val2017.txt listing image paths)

        Returns:
            DataLoader for validation data.
        """
        from libreyolo.data import load_data_config, get_img_files, img2label_paths
        from libreyolo.training.dataset import (
            YOLODataset,
            COCODataset,
        )
        from torch.utils.data import DataLoader

        img_size = (self.config.imgsz, self.config.imgsz)

        # Initialize variables for file list mode
        img_files = None
        label_files = None
        split_name = self.config.split  # Default split name
        data_cfg = None

        # Load data configuration (supports auto-download)
        if self.config.data:
            data_cfg = load_data_config(self.config.data)
            data_dir = data_cfg["root"]
            self.nc = data_cfg.get("nc", self.nc)

            # Handle names as dict or list
            names = data_cfg.get("names", None)
            if isinstance(names, dict):
                self.class_names = [names[i] for i in range(len(names))]
            else:
                self.class_names = names

            # Check if we have pre-resolved image files (from .txt format)
            img_files_key = f"{self.config.split}_img_files"
            label_files_key = f"{self.config.split}_label_files"

            if img_files_key in data_cfg:
                # Use pre-resolved file lists from load_data_config
                img_files = data_cfg[img_files_key]
                label_files = data_cfg.get(label_files_key)
            else:
                # Try to resolve files from the split path
                split_path_str = data_cfg.get(self.config.split, f"images/{self.config.split}")
                full_split_path = Path(data_cfg["path"]) / Path(split_path_str).name

                # Check if path ends with .txt (even if not yet downloaded)
                if str(split_path_str).endswith(".txt"):
                    # This is .txt format - try to resolve if file exists
                    txt_path = Path(data_cfg["path"]) / split_path_str
                    if txt_path.exists():
                        try:
                            img_files = get_img_files(txt_path)
                            label_files = img2label_paths(img_files)
                        except (FileNotFoundError, ValueError):
                            pass
                else:
                    # Directory format - extract split name for YOLODataset
                    if "/" in str(split_path_str):
                        split_name = str(split_path_str).split("/")[-1]
                    else:
                        split_name = str(split_path_str)
        else:
            data_dir = self.config.data_dir
            self.class_names = None

        # Get model-specific validation preprocessor
        self.val_preproc = self.model._get_val_preprocessor(img_size=self.config.imgsz)

        # Determine dataset format and create dataset
        data_path = Path(data_dir)

        if (data_path / "annotations").exists():
            # COCO format (JSON annotations)
            json_file = f"instances_{self.config.split}2017.json"
            if not (data_path / "annotations" / json_file).exists():
                # Try alternative naming
                json_file = f"instances_{self.config.split}.json"

            dataset = COCODataset(
                data_dir=str(data_path),
                json_file=json_file,
                name=f"{self.config.split}2017" if "2017" in json_file else self.config.split,
                img_size=img_size,
                preproc=self.val_preproc,
            )
        elif img_files is not None:
            # File list mode (.txt format)
            dataset = YOLODataset(
                img_files=img_files,
                label_files=label_files,
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

        # Create dataloader with validation collate function
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=False,  # MPS doesn't support pin_memory
            collate_fn=val_collate_fn,
            drop_last=False,
        )

        return dataloader

    def _init_metrics(self) -> None:
        """Initialize metrics containers."""
        self.metrics = DetMetrics(
            nc=self.nc,
            conf=0.25,  # Confidence threshold for precision/recall reporting
            iou_thresholds=self.config.iou_thresholds,
        )
        self.confusion_matrix = ConfusionMatrix(
            nc=self.nc,
            conf=0.25,
            iou_thres=0.5,
        )

    def _preprocess_batch(
        self, batch: Tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, List, List]:
        """
        Preprocess a batch of data for validation.

        Args:
            batch: (images, targets, img_info, img_ids) from dataset.

        Returns:
            Tuple of (preprocessed_images, targets, img_info, img_ids).
        """
        images, targets, img_info, img_ids = batch

        # Convert to tensor
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        images = images.float()

        # Normalize based on preprocessor config
        # Some models (e.g., YOLOX) expect 0-255 range
        if self.val_preproc.normalize and images.max() > 1.0:
            images = images / 255.0

        # Ensure NCHW format
        if images.dim() == 3:
            images = images.unsqueeze(0)

        return images, targets, img_info, img_ids

    def _slice_batch_predictions(self, preds: Any, batch_idx: int) -> Any:
        """
        Extract predictions for a single image from batched model output.

        Args:
            preds: Batched model output (dict with x8, x16, x32 keys).
            batch_idx: Index of the image in the batch.

        Returns:
            Sliced predictions for the single image.
        """
        if isinstance(preds, dict):
            # Handle dict output (YOLOv8/v11 style: {'x8': {'box': ..., 'cls': ...}, ...})
            sliced = {}
            for key, value in preds.items():
                if isinstance(value, dict):
                    # Nested dict like {'box': tensor, 'cls': tensor}
                    sliced[key] = {
                        k: v[batch_idx:batch_idx+1] if isinstance(v, torch.Tensor) else v
                        for k, v in value.items()
                    }
                elif isinstance(value, torch.Tensor):
                    sliced[key] = value[batch_idx:batch_idx+1]
                else:
                    sliced[key] = value
            return sliced
        elif isinstance(preds, torch.Tensor):
            # Simple tensor output
            return preds[batch_idx:batch_idx+1]
        elif isinstance(preds, (list, tuple)):
            # List/tuple of tensors
            return type(preds)(
                p[batch_idx:batch_idx+1] if isinstance(p, torch.Tensor) else p
                for p in preds
            )
        else:
            return preds

    def _postprocess_predictions(
        self, preds: Any, batch: Tuple
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Postprocess model predictions to detection format.

        Args:
            preds: Raw model output.
            batch: Original batch data (images, targets, img_info, img_ids).

        Returns:
            List of detection dicts per image with keys:
                - boxes: (N, 4) tensor in xyxy format
                - scores: (N,) tensor of confidence scores
                - classes: (N,) tensor of class indices
        """
        images, targets, img_info, img_ids = batch
        batch_size = len(img_info)

        detections = []

        # Process each image in batch
        for i in range(batch_size):
            # Get original image size
            orig_h, orig_w = img_info[i]

            # Slice predictions for this specific image
            single_preds = self._slice_batch_predictions(preds, i)

            # Run postprocessing using model's method
            result = self.model._postprocess(
                single_preds,
                conf_thres=self.config.conf_thres,
                iou_thres=self.config.iou_thres,
                original_size=(orig_w, orig_h),
            )

            # Convert to tensors
            if result["num_detections"] > 0:
                boxes = torch.tensor(result["boxes"], dtype=torch.float32, device=self.device)
                scores = torch.tensor(result["scores"], dtype=torch.float32, device=self.device)
                classes = torch.tensor(result["classes"], dtype=torch.int64, device=self.device)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
                scores = torch.zeros(0, dtype=torch.float32, device=self.device)
                classes = torch.zeros(0, dtype=torch.int64, device=self.device)

            detections.append({
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
            })

        return detections

    def _update_metrics(
        self,
        preds: List[Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        img_info: List,
    ) -> None:
        """
        Update metrics with batch predictions and targets.

        Args:
            preds: List of detection dicts per image.
            targets: Ground truth tensor (B, max_labels, 5) with [x1, y1, x2, y2, class].
            img_info: List of (height, width) tuples for each image.
        """
        batch_size = len(preds)

        for i in range(batch_size):
            pred = preds[i]
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_classes = pred["classes"]

            # Get ground truth for this image
            # targets shape: (B, max_labels, 5) where 5 is [x1, y1, x2, y2, class]
            if isinstance(targets, torch.Tensor):
                gt = targets[i]  # (max_labels, 5)
            else:
                gt = torch.from_numpy(targets[i])

            # Filter out padding (boxes with all zeros)
            valid_mask = gt[:, :4].sum(dim=1) > 0
            gt = gt[valid_mask]

            if len(gt) > 0:
                gt_boxes = gt[:, :4].clone().to(self.device)
                gt_classes = gt[:, 4].long().to(self.device)

                # Scale GT boxes from simple resize coords back to original image coords
                # GT boxes are in 640x640 coords (simple resize, no aspect preservation)
                # Predictions are in original coords, so scale GT boxes back
                orig_h, orig_w = img_info[i]
                img_h, img_w = self.config.imgsz, self.config.imgsz
                # Simple resize: x_640 = x_orig * (640/orig_w), so x_orig = x_640 * (orig_w/640)
                gt_boxes[:, 0] = gt_boxes[:, 0] * orig_w / img_w  # x1
                gt_boxes[:, 1] = gt_boxes[:, 1] * orig_h / img_h  # y1
                gt_boxes[:, 2] = gt_boxes[:, 2] * orig_w / img_w  # x2
                gt_boxes[:, 3] = gt_boxes[:, 3] * orig_h / img_h  # y2
            else:
                gt_boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
                gt_classes = torch.zeros(0, dtype=torch.int64, device=self.device)

            # Process batch for metrics
            correct, conf, pred_cls, target_cls = process_batch(
                pred_boxes,
                pred_scores,
                pred_classes,
                gt_boxes,
                gt_classes,
                self.iou_thresholds.to(self.device),
            )

            # Update DetMetrics
            self.metrics.update(correct, conf, pred_cls, target_cls)

            # Update confusion matrix
            self.confusion_matrix.update(
                pred_boxes,
                pred_classes,
                pred_scores,
                gt_boxes,
                gt_classes,
            )

    def _compute_metrics(self) -> Dict[str, float]:
        """
        Compute final metrics from accumulated stats.

        Returns:
            Dictionary with validation metrics.
        """
        return self.metrics.compute()

    def _generate_plots(self) -> None:
        """Generate and save visualization plots."""
        if self.confusion_matrix is not None:
            cm_path = self.save_dir / "confusion_matrix.png"
            self.confusion_matrix.plot(
                save_path=cm_path,
                names=self.class_names,
                normalize=True,
            )
            if self.config.verbose:
                print(f"Confusion matrix saved to: {cm_path}")


class ValDataset:
    """
    Simple validation dataset wrapper.

    Wraps images and annotations for validation without augmentation.
    """

    def __init__(
        self,
        img_paths: List[str],
        annotations: List[np.ndarray],
        img_size: Tuple[int, int] = (640, 640),
    ):
        """
        Initialize validation dataset.

        Args:
            img_paths: List of image file paths.
            annotations: List of annotation arrays (N, 5) with [x1, y1, x2, y2, class].
            img_size: Target image size (height, width).
        """
        self.img_paths = img_paths
        self.annotations = annotations
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple:
        import cv2

        # Load image
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img.shape[:2]

        # Resize
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))

        # Convert to tensor format (CHW)
        img = img.transpose(2, 0, 1).astype(np.float32)

        # Get annotations
        anno = self.annotations[idx].copy()

        # Scale annotations to resized image
        if len(anno) > 0:
            scale_x = self.img_size[1] / orig_w
            scale_y = self.img_size[0] / orig_h
            anno[:, [0, 2]] *= scale_x
            anno[:, [1, 3]] *= scale_y

        return img, anno, (orig_h, orig_w), idx

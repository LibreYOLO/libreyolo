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
from .metrics import DetMetrics
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
        self.coco_evaluator = None  # COCO evaluator (optional)
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

        # Use model's native input size if it differs from config
        # This is important for models like YOLOX nano/tiny that use 416x416
        # and RF-DETR variants that each have different resolutions
        model_input_size = self.model._get_input_size() if hasattr(self.model, '_get_input_size') else None
        if model_input_size is not None and model_input_size != self.config.imgsz:
            # Model has a specific input size requirement
            actual_imgsz = model_input_size
        else:
            actual_imgsz = self.config.imgsz

        # Store the actual imgsz for postprocessing
        self._actual_imgsz = actual_imgsz
        img_size = (actual_imgsz, actual_imgsz)

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
                    # Directory format - construct full path and collect image files
                    full_split_path = Path(data_cfg["path"]) / split_path_str

                    if full_split_path.exists():
                        # Use file list mode with explicit paths
                        img_files_list = []
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                            img_files_list.extend(full_split_path.glob(ext))
                            img_files_list.extend(full_split_path.glob(ext.upper()))

                        if img_files_list:
                            img_files = sorted(img_files_list)
                            label_files = img2label_paths(img_files)
                    else:
                        # Fallback: extract split name for legacy directory mode
                        if "/" in str(split_path_str):
                            split_name = str(split_path_str).split("/")[-1]
                        else:
                            split_name = str(split_path_str)
        else:
            data_dir = self.config.data_dir
            self.class_names = None

        # Get model-specific validation preprocessor
        self.val_preproc = self.model._get_val_preprocessor(img_size=actual_imgsz)

        # Determine dataset format and create dataset
        data_path = Path(data_dir)

        if img_files is not None:
            # File list mode (.txt format) - explicit list takes priority
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
                # Try alternative naming
                json_file = f"instances_{self.config.split}.json"

            # Determine image subdirectory: prefer images/val2017 (standard COCO layout)
            split_name = f"{self.config.split}2017" if "2017" in json_file else self.config.split
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

        # Create dataloader with validation collate function
        # Enable pin_memory for CUDA (faster CPU→GPU transfer)
        use_cuda = torch.cuda.is_available() and self.device.type == "cuda"
        nw = self.config.num_workers

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=use_cuda,  # Enable for CUDA, disable for MPS/CPU
            prefetch_factor=4 if nw > 0 else None,  # Prefetch more batches
            persistent_workers=nw > 0,  # Keep workers alive between batches
            collate_fn=val_collate_fn,
            drop_last=False,
        )

        return dataloader

    def _init_metrics(self) -> None:
        """Initialize metrics containers."""
        if self.config.use_coco_eval:
            # Use COCO evaluation API
            try:
                from libreyolo.data import create_yolo_coco_api
                from libreyolo.validation import COCOEvaluator

                if self.config.verbose:
                    print("Initializing COCO evaluator...")

                # Create COCO API from dataset
                coco_api = create_yolo_coco_api(self.config.data, self.config.split)
                self.coco_evaluator = COCOEvaluator(coco_api, iou_type='bbox')

                if self.config.verbose:
                    print(f"COCO evaluator initialized with {len(coco_api.imgs)} images")
            except Exception as e:
                print(f"Warning: Failed to initialize COCO evaluator: {e}")
                print("Falling back to legacy DetMetrics")
                self.config.use_coco_eval = False
                self.coco_evaluator = None

        # Initialize legacy metrics as fallback
        if not self.config.use_coco_eval or self.coco_evaluator is None:
            self.metrics = DetMetrics(
                nc=self.nc,
                conf=0.25,  # Confidence threshold for precision/recall reporting
                iou_thresholds=self.config.iou_thresholds,
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
        # - custom_normalization=True: preprocessor already applied its own
        #   normalization (e.g. RF-DETR ImageNet mean/std), pass through unchanged
        # - normalize=True: model expects 0-1 range (standard YOLO)
        # - normalize=False: model expects 0-255 range (YOLOX)
        if getattr(self.val_preproc, 'custom_normalization', False):
            pass  # Already normalized by preprocessor (e.g. RF-DETR ImageNet)
        elif self.val_preproc.normalize:
            if images.max() > 1.0:
                images = images / 255.0
        else:
            if images.max() <= 1.0:
                images = images * 255.0

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
            # Handle dict output (multi-scale style: {'x8': {'box': ..., 'cls': ...}, ...})
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

        Uses batched postprocessing when available for much better performance.
        Falls back to per-image processing for models that don't support batched mode.

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

        # Process each image in batch
        detections = []
        for i in range(batch_size):
            orig_h, orig_w = img_info[i]
            single_preds = self._slice_batch_predictions(preds, i)

            uses_letterbox = self.val_preproc is not None and self.val_preproc.uses_letterbox
            result = self.model._postprocess(
                single_preds,
                conf_thres=self.config.conf_thres,
                iou_thres=self.config.iou_thres,
                original_size=(orig_w, orig_h),  # (width, height) format as expected by postprocess
                input_size=self._actual_imgsz,  # Pass actual input size used
                letterbox=uses_letterbox,
            )

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
        img_ids: List = None,
    ) -> None:
        """
        Update metrics with batch predictions and targets.

        Args:
            preds: List of detection dicts per image.
            targets: Ground truth tensor (B, max_labels, 5) with [x1, y1, x2, y2, class].
            img_info: List of (height, width) tuples for each image.
            img_ids: List of image IDs for COCO evaluation (optional).
        """
        batch_size = len(preds)

        # Update COCO evaluator if enabled
        if self.coco_evaluator is not None and img_ids is not None:
            for i in range(batch_size):
                pred = preds[i]
                image_id = img_ids[i]
                self.coco_evaluator.update(pred, image_id)

        # Skip legacy metrics if using COCO eval
        if self.coco_evaluator is not None:
            return

        # Check if preprocessor uses letterbox (aspect-preserving) or simple resize
        uses_letterbox = self.val_preproc is not None and self.val_preproc.uses_letterbox

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

                # Scale GT boxes back to original image coords
                # GT boxes are in model input coords (640x640)
                # Predictions are already in original coords from postprocess
                orig_h, orig_w = img_info[i]
                img_h, img_w = self._actual_imgsz, self._actual_imgsz

                if uses_letterbox:
                    # Letterbox: GT boxes were scaled by r = min(img_h/orig_h, img_w/orig_w)
                    # To convert back: divide by r
                    r = min(img_h / orig_h, img_w / orig_w)
                    gt_boxes[:, :4] = gt_boxes[:, :4] / r
                else:
                    # Simple resize: x_640 = x_orig * (640/orig_w)
                    # To convert back: x_orig = x_640 * (orig_w/640)
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

    def _compute_metrics(self) -> Dict[str, float]:
        """
        Compute final metrics from accumulated stats.

        Returns:
            Dictionary with validation metrics.
        """
        if self.coco_evaluator is not None:
            # Compute COCO metrics
            if self.config.verbose:
                print("\nComputing COCO metrics...")

            save_json = None
            if self.config.save_json:
                save_json = str(self.save_dir / "predictions.json")

            coco_metrics = self.coco_evaluator.compute(save_json=save_json)

            # Map COCO metrics to LibreYOLO naming convention
            return {
                'metrics/mAP50-95': coco_metrics['mAP'],      # Primary metric
                'metrics/mAP50': coco_metrics['mAP50'],       # YOLO-style AP@0.5
                'metrics/mAP75': coco_metrics['mAP75'],       # Stricter AP@0.75
                'metrics/mAP_small': coco_metrics['mAP_small'],
                'metrics/mAP_medium': coco_metrics['mAP_medium'],
                'metrics/mAP_large': coco_metrics['mAP_large'],
                'metrics/AR1': coco_metrics['AR1'],
                'metrics/AR10': coco_metrics['AR10'],
                'metrics/AR100': coco_metrics['AR100'],
                'metrics/AR_small': coco_metrics['AR_small'],
                'metrics/AR_medium': coco_metrics['AR_medium'],
                'metrics/AR_large': coco_metrics['AR_large'],
            }
        else:
            # Use legacy DetMetrics
            return self.metrics.compute()

    def _generate_plots(self) -> None:
        """Generate and save visualization plots."""
        pass  # Plotting removed to reduce bloat


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

        # Load image (keep BGR format to match YOLOX training)
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        # Do NOT convert BGR to RGB - YOLOX is trained on BGR images

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

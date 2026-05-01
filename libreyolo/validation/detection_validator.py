"""Detection validator for LibreYOLO."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from .base import BaseValidator
from .config import ValidationConfig

logger = logging.getLogger(__name__)

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

    Computes the standard pycocotools COCO metrics:
    mAP50-95, mAP50, mAP75, mAP/AR by object size, AR at different maxDets.
    """

    task = "detect"

    def __init__(
        self,
        model: "BaseModel",
        config: Optional[ValidationConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, config, **kwargs)

        self.coco_evaluator = None
        self.class_names: Optional[List[str]] = None
        self.iou_thresholds = torch.tensor(self.config.iou_thresholds)
        self.nc = model.nb_classes
        self.val_preproc = None  # set in _setup_dataloader
        self._coco_annotation_file: Optional[Path] = None
        self._coco_label_to_category_id: Optional[Dict[int, int]] = None

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
            data_cfg = load_data_config(
                self.config.data,
                allow_scripts=self.config.allow_download_scripts,
            )
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
        self._coco_annotation_file = None
        self._coco_label_to_category_id = None
        coco_annotation_file = self._find_coco_annotation_file(data_path)

        if coco_annotation_file is not None:
            # Prefer official COCO JSON when it is present. This preserves
            # COCO image ids, category ids, crowd annotations, and area ranges.
            json_file = coco_annotation_file.name
            split_name = self._resolve_coco_image_dir(data_path, json_file)

            dataset = COCODataset(
                data_dir=str(data_path),
                json_file=json_file,
                name=split_name,
                img_size=img_size,
                preproc=self.val_preproc,
            )
            self._coco_annotation_file = coco_annotation_file
            self._coco_label_to_category_id = {
                label: category_id for label, category_id in enumerate(dataset.class_ids)
            }
        elif img_files is not None:
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

    def _find_coco_annotation_file(self, data_path: Path) -> Optional[Path]:
        annotations_dir = data_path / "annotations"
        if not annotations_dir.exists():
            return None

        candidates = [
            annotations_dir / f"instances_{self.config.split}2017.json",
            annotations_dir / f"instances_{self.config.split}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _resolve_coco_image_dir(self, data_path: Path, json_file: str) -> str:
        split_name = (
            f"{self.config.split}2017"
            if f"{self.config.split}2017" in json_file
            else self.config.split
        )
        if (data_path / "images" / split_name).exists():
            return f"images/{split_name}"
        return split_name

    def _init_metrics(self) -> None:
        from libreyolo.data import load_data_config
        from libreyolo.data.yolo_coco_api import YOLOCocoAPI
        from libreyolo.validation import COCOEvaluator

        if self.config.verbose:
            logger.info("Initializing COCO evaluator...")

        if self.config.data is None:
            raise RuntimeError(
                "config.data must be set to a yaml path or registry name "
                "to initialize the COCO evaluator."
            )

        if self._coco_annotation_file is not None:
            try:
                from pycocotools.coco import COCO
            except ImportError:
                raise ImportError(
                    "pycocotools is required for COCO format. "
                    "Install with: pip install pycocotools"
                )

            coco_api = COCO(str(self._coco_annotation_file))
            self.coco_evaluator = COCOEvaluator(
                coco_api,
                iou_type="bbox",
                label_to_category_id=self._coco_label_to_category_id,
            )
            if self.config.verbose:
                logger.info(
                    "COCO evaluator initialized from %s with %d images",
                    self._coco_annotation_file,
                    len(coco_api.imgs),
                )
            return

        # Resolve the (possibly registry-name) data argument through
        # load_data_config — that handles both relative `path:` fields and
        # registry shortcuts like "coco-val-only", returning absolute file
        # lists. Build YOLOCocoAPI directly from the resolved paths.
        data_cfg = load_data_config(
            self.config.data,
            allow_scripts=self.config.allow_download_scripts,
        )
        split = self.config.split
        img_files = data_cfg.get(f"{split}_img_files")
        label_files = data_cfg.get(f"{split}_label_files")
        if not img_files:
            raise RuntimeError(
                f"No {split} images resolved for data={self.config.data!r}. "
                "Check the dataset configuration."
            )

        names = data_cfg.get("names") or self.class_names or []
        if isinstance(names, dict):
            class_names = [names[i] for i in sorted(names.keys())]
        else:
            class_names = list(names)

        images_dir = Path(img_files[0]).parent
        labels_dir = (
            Path(label_files[0]).parent if label_files else images_dir.parent / "labels"
        )

        coco_api = YOLOCocoAPI(
            images_dir=images_dir,
            labels_dir=labels_dir,
            class_names=class_names,
        )
        self.coco_evaluator = COCOEvaluator(coco_api, iou_type="bbox")

        if self.config.verbose:
            logger.info("COCO evaluator initialized with %d images", len(coco_api.imgs))

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
        if img_ids is None:
            raise RuntimeError(
                "img_ids are required for COCO evaluation but were not provided "
                "by the dataloader."
            )
        for i in range(len(preds)):
            self.coco_evaluator.update(preds[i], img_ids[i])

    def _compute_metrics(self) -> Dict[str, float]:
        if self.config.verbose:
            logger.info("Computing COCO metrics...")

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

"""Detection validator for LibreYOLO."""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..postprocess.slicing import slice_batch_outputs
from .base import BaseValidator
from .config import ValidationConfig
from .loss import ValidationLossMixin

logger = logging.getLogger(__name__)

COCO_TOPK_FAMILIES = {"dfine", "deim", "deimv2", "tinyformer", "ec", "rfdetr", "rtdetr", "rtdetrv2", "rtdetrv4"}
_N_VAL_SAMPLES = 8  # maximum sample images stored for visualisation
BEST_CONF_KEY = "metrics/best_conf"
BEST_CONF_F1_KEY = "metrics/best_conf_f1"
BEST_CONF_PER_CLASS_KEY = "metrics/best_conf_per_class"


def _collapse_coco_ground_truth(coco_api, category_id: int):
    """Return an independent COCO API with every GT category collapsed."""
    from pycocotools.coco import COCO

    dataset = deepcopy(coco_api.dataset)
    dataset["categories"] = [
        {"id": int(category_id), "name": "object", "supercategory": "object"}
    ]
    for annotation in dataset.get("annotations", []):
        annotation["category_id"] = int(category_id)
    collapsed = COCO()
    collapsed.dataset = dataset
    collapsed.createIndex()
    return collapsed


if TYPE_CHECKING:
    from libreyolo.models.base import BaseModel
    from .loss import ValidationLossAdapter


def val_collate_fn(batch):
    """Collate validation batch: stack preprocessed images and padded targets."""
    if len(batch[0]) == 5:
        imgs, targets, img_infos, img_ids, _segments = zip(*batch)
    else:
        imgs, targets, img_infos, img_ids = zip(*batch)
    imgs = torch.from_numpy(np.stack(imgs))
    targets = torch.from_numpy(np.stack(targets))
    return imgs, targets, img_infos, img_ids


class DetectionValidator(ValidationLossMixin, BaseValidator):
    """
    Validator for object detection models.

    Computes the standard pycocotools COCO metrics:
    mAP50-95, mAP50, mAP75, mAP/AR by object size, AR at different maxDets.
    """

    task = "detect"

    # Class-level default so instances built without __init__ (a pattern the
    # test suite uses for narrow-scope validators) still resolve it.
    _gt_coco_api = None
    _single_cls_clip_warning_emitted = False

    def __init__(
        self,
        model: "BaseModel",
        config: Optional[ValidationConfig] = None,
        *,
        loss_adapter: Optional["ValidationLossAdapter"] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, config, **kwargs)

        self.coco_evaluator = None
        self.class_names: Optional[List[str]] = None
        self.iou_thresholds = torch.tensor(self.config.iou_thresholds)
        self.nc = model.nb_classes
        checkpoint_probe = getattr(model, "_checkpoint_train_config", None)
        checkpoint_config = checkpoint_probe() if callable(checkpoint_probe) else {}
        self._checkpoint_single_cls = bool(checkpoint_config.get("single_cls", False))
        if self._checkpoint_single_cls:
            self.config = self.config.update(single_cls=True)
        self.val_preproc = None  # set in _setup_dataloader
        self._coco_annotation_file: Optional[Path] = None
        self._coco_label_to_category_id: Optional[Dict[int, int]] = None
        self._yolo_coco_img_files: Optional[List[Path]] = None
        self._yolo_coco_label_files: Optional[List[Path]] = None
        self._init_validation_loss(loss_adapter)
        # Parsed ground truth, cached across runs of this instance. The GT is
        # immutable for the validator's lifetime; only the evaluator built on
        # top of it accumulates state and must stay per-run.
        self._gt_coco_api = None

    # =========================================================================
    # Setup
    # =========================================================================

    def _dataset_kwargs(self) -> Dict[str, Any]:
        return {}

    def _coco_api_kwargs(self) -> Dict[str, Any]:
        return {}

    def _single_cls_enabled(self) -> bool:
        return bool(getattr(self.config, "single_cls", False))

    def _resolve_imgsz(self) -> int | tuple[int, int]:
        """Return the validation image size, falling back to the model native size."""
        imgsz = self.config.imgsz
        if imgsz is not None:
            if isinstance(imgsz, (list, tuple)):
                return (int(imgsz[0]), int(imgsz[1]))
            return int(imgsz)

        get_input_size = getattr(self.model, "_get_input_size", None)
        if callable(get_input_size):
            raw = get_input_size()
            if isinstance(raw, (list, tuple)):
                return (int(raw[0]), int(raw[1]))
            return int(raw)

        return 640

    def _setup_dataloader(self) -> DataLoader:
        """
        Create validation dataloader from config.

        Supports directory-based datasets, .txt file format, and COCO JSON.
        """
        from libreyolo.data import (
            get_coco_annotation_file,
            get_coco_image_dir,
            get_img_files,
            img2label_paths,
            load_data_config,
        )
        from libreyolo.data.dataset import YOLODataset, COCODataset
        from torch.utils.data import DataLoader

        actual_imgsz = self._resolve_imgsz()
        self.config.imgsz = actual_imgsz
        self._actual_imgsz = actual_imgsz
        if isinstance(actual_imgsz, (list, tuple)):
            img_size = tuple(actual_imgsz)
        else:
            img_size = (actual_imgsz, actual_imgsz)

        img_files = None
        label_files = None
        split_name = self.config.split
        data_cfg = None
        explicit_coco_annotation = None

        if self.config.data:
            data_cfg = load_data_config(
                self.config.data,
                allow_scripts=self.config.allow_download_scripts,
                single_cls=self._single_cls_enabled(),
            )
            data_dir = data_cfg["root"]
            model_nc = int(self.nc)
            self.nc = int(data_cfg.get("nc", self.nc))

            original_nc = data_cfg.get("_original_nc")
            original_names = data_cfg.get("_original_names")
            if original_nc is None and original_names is not None:
                original_nc = len(original_names)
            if getattr(self, "_checkpoint_single_cls", False) and original_nc not in (
                None,
                1,
            ):
                logger.info(
                    "Checkpoint config has single_cls=True; collapsing %s dataset "
                    "classes to one validation class.",
                    original_nc,
                )
            elif not self._single_cls_enabled() and model_nc != self.nc:
                logger.warning(
                    "Checkpoint/model class count (%d) differs from dataset class "
                    "count (%d). Validation will continue, but metrics may be invalid.",
                    model_nc,
                    self.nc,
                )

            names = data_cfg.get("names", None)
            if isinstance(names, dict):
                self.class_names = [names[i] for i in range(len(names))]
            else:
                self.class_names = names

            # Check for pre-resolved file lists (from .txt format)
            img_files_key = f"{self.config.split}_img_files"
            label_files_key = f"{self.config.split}_label_files"
            explicit_coco_annotation = get_coco_annotation_file(
                data_cfg,
                self.config.split,
            )

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
                            img_files = sorted(set(img_files_list))
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
        self._ensure_validation_loss_target_capacity()
        dataset_kwargs = self._dataset_kwargs()

        # Determine dataset format
        data_path = Path(data_dir)
        self._coco_annotation_file = None
        self._coco_label_to_category_id = None
        self._yolo_coco_img_files = None
        self._yolo_coco_label_files = None
        coco_annotation_file = (
            Path(explicit_coco_annotation)
            if explicit_coco_annotation
            else self._find_coco_annotation_file(data_path)
        )

        if coco_annotation_file is not None:
            # Prefer official COCO JSON when it is present. This preserves
            # COCO image ids, category ids, crowd annotations, and area ranges.
            json_file = (
                str(coco_annotation_file)
                if explicit_coco_annotation
                else coco_annotation_file.name
            )
            if explicit_coco_annotation and data_cfg is not None:
                split_name = get_coco_image_dir(
                    data_cfg,
                    self.config.split,
                    self._resolve_coco_image_dir(data_path, coco_annotation_file.name),
                )
            else:
                split_name = self._resolve_coco_image_dir(
                    data_path,
                    coco_annotation_file.name,
                )

            dataset = COCODataset(
                data_dir=str(data_path),
                json_file=json_file,
                name=split_name,
                img_size=img_size,
                preproc=self.val_preproc,
                num_classes=int(self.nc),
                names=(
                    data_cfg.get("_original_names", data_cfg.get("names"))
                    if data_cfg is not None
                    else None
                ),
                single_cls=self._single_cls_enabled(),
                **dataset_kwargs,
            )
            self._coco_annotation_file = coco_annotation_file
            self._coco_label_to_category_id = dict(
                getattr(dataset, "label_to_category_id", {})
            )
        elif img_files is not None:
            # File list mode (.txt format)
            dataset = YOLODataset(
                img_files=img_files,
                label_files=label_files,
                img_size=img_size,
                preproc=self.val_preproc,
                num_classes=int(self.nc),
                single_cls=self._single_cls_enabled(),
                **dataset_kwargs,
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
                num_classes=int(self.nc),
                names=(
                    data_cfg.get("_original_names", data_cfg.get("names"))
                    if data_cfg is not None
                    else None
                ),
                single_cls=self._single_cls_enabled(),
                **dataset_kwargs,
            )
        else:
            # YOLO directory format
            dataset = YOLODataset(
                data_dir=str(data_path),
                split=split_name,
                img_size=img_size,
                preproc=self.val_preproc,
                num_classes=int(self.nc),
                single_cls=self._single_cls_enabled(),
                **dataset_kwargs,
            )

        if isinstance(dataset, YOLODataset):
            self._yolo_coco_img_files = list(dataset.img_files)
            self._yolo_coco_label_files = list(dataset.label_files)

        # Both dataset classes are ImageCacheMixin; every construction branch
        # above funnels through here, so this is the single cache switch.
        dataset.enable_image_cache(getattr(self.config, "cache", False))

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

    def _ensure_validation_loss_target_capacity(self) -> None:
        """Keep validation target padding at least as large as training's cap."""
        adapter = getattr(self, "loss_adapter", None)
        requested = (
            getattr(adapter, "max_labels", None) if adapter is not None else None
        )
        if requested is None:
            return
        requested = int(requested)
        if requested < 1:
            raise ValueError("Validation-loss max_labels must be at least 1")
        current = int(getattr(self.val_preproc, "max_labels", requested))
        if requested > current:
            self.val_preproc.max_labels = requested
            logger.info(
                "Raised validation target capacity from %d to %d for loss computation",
                current,
                requested,
            )

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
        from libreyolo.data import resolve_default_coco_image_dir

        return resolve_default_coco_image_dir(data_path, self.config.split, json_file)

    def _coco_max_det(self) -> int:
        """Resolve the opt-in evaluator cap without coupling it to NMS."""
        value = getattr(self.config, "eval_max_det", None)
        return 100 if value is None else int(value)

    def _init_metrics(self) -> None:
        from libreyolo.data import load_data_config
        from libreyolo.data.yolo_coco_api import YOLOCocoAPI
        from libreyolo.validation import COCOEvaluator

        self._reset_validation_loss()

        if self.config.verbose:
            logger.info("Initializing COCO evaluator...")

        # Always initialise plot-tracking state before any early returns
        self._confusion_matrix = None
        self._val_samples: List[Dict] = []
        if self.config.save_plots:
            from .val_plotter import ConfusionMatrix  # noqa: PLC0415
            self._confusion_matrix = ConfusionMatrix(nc=self.nc)

        # A COCO-annotation dataset (resolvable from either data= or data_dir=)
        # is handled here first; the config.data requirement below only applies
        # to the load_data_config / YOLOCocoAPI path.
        if self._coco_annotation_file is not None:
            try:
                from pycocotools.coco import COCO
            except ImportError:
                raise ImportError(
                    "pycocotools is required for COCO format. "
                    "Install with: pip install pycocotools"
                )

            coco_api = self._gt_coco_api
            if coco_api is None:
                coco_api = COCO(str(self._coco_annotation_file))
                if self._single_cls_enabled():
                    category_id = next(
                        iter((self._coco_label_to_category_id or {}).values()),
                        0,
                    )
                    coco_api = _collapse_coco_ground_truth(coco_api, category_id)
                    self._coco_label_to_category_id = {0: int(category_id)}
                self._gt_coco_api = coco_api
            self.coco_evaluator = COCOEvaluator(
                coco_api,
                iou_type="bbox",
                label_to_category_id=self._coco_label_to_category_id,
                max_det=self._coco_max_det(),
                faster_coco_eval=getattr(self.config, "faster_coco_eval", False),
            )
            if self.config.verbose:
                logger.info(
                    "COCO evaluator initialized from %s with %d images",
                    self._coco_annotation_file,
                    len(coco_api.imgs),
                )
            return

        if self.config.data is None:
            raise RuntimeError(
                "config.data must be set to a yaml path or registry name "
                "to initialize the COCO evaluator."
            )

        # Resolve the (possibly registry-name) data argument through
        # load_data_config — that handles both relative `path:` fields and
        # registry shortcuts like "coco-val-only", returning absolute file
        # lists. Build YOLOCocoAPI directly from the resolved paths.
        data_cfg = load_data_config(
            self.config.data,
            allow_scripts=self.config.allow_download_scripts,
            single_cls=self._single_cls_enabled(),
        )
        split = self.config.split
        img_files = data_cfg.get(f"{split}_img_files")
        label_files = data_cfg.get(f"{split}_label_files")

        # Reuse the file list already resolved in _setup_dataloader (e.g. the
        # images/<split> directory-glob fallback), which load_data_config does
        # not re-expose via the {split}_img_files key.
        if self._yolo_coco_img_files:
            img_files = self._yolo_coco_img_files
            if self._yolo_coco_label_files:
                label_files = self._yolo_coco_label_files

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
        image_files = self._yolo_coco_img_files or [Path(p) for p in img_files]
        yolo_label_files = self._yolo_coco_label_files or (
            [Path(p) for p in label_files] if label_files else None
        )

        coco_api = self._gt_coco_api
        if coco_api is None:
            coco_api = YOLOCocoAPI(
                images_dir=images_dir,
                labels_dir=labels_dir,
                class_names=class_names,
                image_files=image_files,
                label_files=yolo_label_files,
                single_cls=self._single_cls_enabled(),
                **self._coco_api_kwargs(),
            )
            self._gt_coco_api = coco_api
        self.coco_evaluator = COCOEvaluator(
            coco_api,
            iou_type="bbox",
            max_det=self._coco_max_det(),
            faster_coco_eval=getattr(self.config, "faster_coco_eval", False),
        )

        if self.config.verbose:
            logger.info("COCO evaluator initialized with %d images", len(coco_api.imgs))

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    def _update_batch_metrics(
        self,
        predictions: Any,
        images: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate an opt-in family loss from this batch's raw predictions."""
        self._accumulate_validation_loss(
            predictions,
            targets,
            image_size=(int(images.shape[-2]), int(images.shape[-1])),
        )

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
        return slice_batch_outputs(preds, batch_idx)

    def _det_from_result(self, result) -> Dict[str, torch.Tensor]:
        """Convert a Results object (from _predict_augment) to a detection dict."""
        if len(result) == 0:
            det: Dict[str, torch.Tensor] = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                "scores": torch.zeros(0, dtype=torch.float32, device=self.device),
                "classes": torch.zeros(0, dtype=torch.int64, device=self.device),
            }
            return det
        det = {
            "boxes": result.boxes.xyxy.to(self.device),
            "scores": result.boxes.conf.to(self.device),
            "classes": result.boxes.cls.long().to(self.device),
        }
        if result.masks is not None:
            det["masks"] = result.masks.data.to(self.device)
        return det

    def _resolve_img_path(self, dataset, global_idx: int, img_id) -> Optional[str]:
        """Resolve the file path for an image given its dataset index and COCO id."""
        from torch.utils.data import Subset
        if isinstance(dataset, Subset):
            actual_idx = dataset.indices[global_idx]
            actual_dataset = dataset.dataset
        else:
            actual_idx = global_idx
            actual_dataset = dataset

        img_files = getattr(actual_dataset, "img_files", None)
        if img_files is not None and actual_idx < len(img_files):
            return str(img_files[actual_idx])
        if hasattr(actual_dataset, "coco") and hasattr(actual_dataset, "data_dir"):
            coco_img = actual_dataset.coco.loadImgs(int(img_id))[0]
            img_dir = Path(actual_dataset.data_dir) / getattr(actual_dataset, "name", "images")
            return str(img_dir / coco_img["file_name"])
        return None

    def _uses_topk_coco_scoring(self) -> bool:
        family = getattr(self.model, "FAMILY", None) or getattr(
            self.model, "model_family", None
        )
        return family in COCO_TOPK_FAMILIES

    def _run_validation_augmented(self) -> None:
        """Per-image TTA validation using model._predict_augment (PIL-level flip+scale)."""
        import sys
        import time
        from tqdm import tqdm

        self.model.model.eval()
        dataset = self.dataloader.dataset
        n_images = len(dataset)
        n_passes = 2  # original + hflip

        if self.config.verbose:
            logger.info(
                "TTA enabled — %d augmentation passes per image (original + hflip). "
                "Running per-image inference on %d images.",
                n_passes,
                n_images,
            )

        pbar = tqdm(
            self.dataloader,
            desc=f"Validating (TTA ×{n_passes})",
            total=len(self.dataloader),
            disable=not self.config.verbose or not sys.stderr.isatty(),
            file=sys.stderr,
        )

        conf_thres = self.config.conf_thres
        if self._uses_topk_coco_scoring():
            conf_thres = 0.0

        total_start = time.time()
        global_idx = 0

        with torch.no_grad():
            for batch in pbar:
                _, targets, img_info, img_ids = batch
                batch_size = len(img_ids)

                t0 = time.time()
                detections = []
                for i in range(batch_size):
                    path = self._resolve_img_path(dataset, global_idx + i, img_ids[i])
                    if path is None:
                        detections.append({
                            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                            "scores": torch.zeros(0, dtype=torch.float32, device=self.device),
                            "classes": torch.zeros(0, dtype=torch.int64, device=self.device),
                        })
                        continue
                    result = self.model._predict_augment(
                        path,
                        conf=conf_thres,
                        iou=self.config.iou_thres,
                        imgsz=self._actual_imgsz,
                        max_det=self.config.max_det,
                    )
                    detections.append(self._det_from_result(result))

                elapsed = time.time() - t0
                self.speed["inference"] += elapsed
                ms_per_img = elapsed / batch_size * 1000
                pbar.set_postfix({"ms/img": f"{ms_per_img:.1f}"}, refresh=False)

                self._update_metrics(detections, targets, img_info, img_ids)
                self.seen += batch_size
                global_idx += batch_size

        self.speed["total"] = time.time() - total_start
        if self.config.verbose:
            total_s = self.speed["total"]
            logger.info(
                "TTA validation complete — %d images in %.1fs (%.1f ms/img)",
                self.seen,
                total_s,
                total_s / self.seen * 1000 if self.seen else 0,
            )

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
        uses_letterbox = self.val_preproc is not None and self.val_preproc.uses_letterbox

        conf_thres = self.config.conf_thres
        if self._uses_topk_coco_scoring():
            conf_thres = 0.0

        detections = []
        for i in range(batch_size):
            orig_h, orig_w = img_info[i]
            result = self.model._postprocess(
                self._slice_batch_predictions(preds, i),
                conf_thres=conf_thres,
                iou_thres=self.config.iou_thres,
                original_size=(orig_w, orig_h),
                input_size=self._actual_imgsz,
                letterbox=uses_letterbox,
                max_det=self.config.max_det,
            )
            if result["num_detections"] > 0:
                raw = result["boxes"]
                boxes = raw.to(self.device) if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=torch.float32, device=self.device)
                raw = result["scores"]
                scores = raw.to(self.device) if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=torch.float32, device=self.device)
                raw = result["classes"]
                classes = raw.to(self.device) if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=torch.int64, device=self.device)
                raw_masks = result.get("masks")
                masks = (
                    raw_masks.to(self.device) if isinstance(raw_masks, torch.Tensor)
                    else torch.tensor(raw_masks, device=self.device) if raw_masks is not None
                    else None
                )
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
                scores = torch.zeros(0, dtype=torch.float32, device=self.device)
                classes = torch.zeros(0, dtype=torch.int64, device=self.device)
                masks = None
            det: Dict[str, torch.Tensor] = {"boxes": boxes, "scores": scores, "classes": classes}
            if masks is not None:
                det["masks"] = masks
            detections.append(det)

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

        cfg = getattr(self, "config", None)
        if getattr(cfg, "save_plots", False):
            try:
                self._track_plots_data(preds, targets, img_info, img_ids)
            except Exception as exc:
                logger.warning("Failed to collect validation plot data: %s", exc)

    def _parse_gt_boxes(
        self, gt_row: torch.Tensor, orig_h: int, orig_w: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse a padded GT target row into xyxy pixel boxes and class indices.

        Two formats are auto-detected from the value range:
          YOLO  — [cls, cx_norm, cy_norm, w_norm, h_norm]  all coords in [0, 1]
          COCO  — [x1_scaled, y1_scaled, x2_scaled, y2_scaled, cls]  pixel coords
                  pre-scaled by the dataset letterbox ratio
        """
        arr = gt_row.cpu().numpy().astype(np.float32)

        # The validator dataloaders emit [x1, y1, x2, y2, cls] rows in scaled
        # pixel space. Prefer that schema whenever it forms a valid box so
        # tiny boxes near the origin are not mistaken for normalized YOLO rows.
        cls_col = arr[:, 4]
        class_like = (
            (cls_col >= 0)
            & (cls_col < self.nc)
            & np.isclose(cls_col, np.round(cls_col))
        )
        valid_xyxy = (
            class_like & (arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1])
        )
        vgt_xyxy = arr[valid_xyxy]
        if len(vgt_xyxy):
            uses_lb = getattr(self.val_preproc, "uses_letterbox", False)
            if uses_lb:
                r, off_x, off_y = self.val_preproc.letterbox_scale(
                    orig_h, orig_w, self._actual_imgsz
                )
                coords = vgt_xyxy[:, :4].copy()
                coords[:, [0, 2]] -= off_x
                coords[:, [1, 3]] -= off_y
                gt_boxes = (coords / r).astype(np.float32)
            else:
                if isinstance(self._actual_imgsz, (list, tuple)):
                    act_w, act_h = self._actual_imgsz[1], self._actual_imgsz[0]
                else:
                    act_w = act_h = self._actual_imgsz
                sx = act_w / orig_w
                sy = act_h / orig_h
                gt = vgt_xyxy[:, :4].copy()
                gt[:, [0, 2]] /= sx  # x1, x2
                gt[:, [1, 3]] /= sy  # y1, y2
                gt_boxes = gt.astype(np.float32)
            raw_classes = vgt_xyxy[:, 4].astype(int)
            self._warn_single_cls_nonzero_gt(raw_classes)
            gt_classes = np.clip(raw_classes, 0, self.nc - 1)
            return gt_boxes, gt_classes

        # If any value in columns 1-4 exceeds 1.5 the coords must be pixels
        is_coco_xyxy = (len(arr) > 0) and (float(np.abs(arr[:, 1:5]).max()) > 1.5)

        if is_coco_xyxy:
            # COCO [x1, y1, x2, y2, cls] — zero-padded rows have all zeros
            valid = (arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1])
            vgt = arr[valid]
            if len(vgt) == 0:
                return np.zeros((0, 4), np.float32), np.zeros(0, int)
            # Undo the coordinate transform applied by the val preprocessor.
            # Letterbox preprocessors (e.g. YOLO9) scale uniformly by
            #   r = min(imgsz/orig_h, imgsz/orig_w).
            # Non-letterbox preprocessors (e.g. RF-DETR, Standard) stretch
            # each axis independently: x *= imgsz/orig_w, y *= imgsz/orig_h.
            uses_lb = getattr(self.val_preproc, "uses_letterbox", False)
            if uses_lb:
                r, off_x, off_y = self.val_preproc.letterbox_scale(orig_h, orig_w, self._actual_imgsz)
                coords = vgt[:, :4].copy()
                coords[:, [0, 2]] -= off_x
                coords[:, [1, 3]] -= off_y
                gt_boxes = (coords / r).astype(np.float32)
            else:
                if isinstance(self._actual_imgsz, (list, tuple)):
                    act_w, act_h = self._actual_imgsz[1], self._actual_imgsz[0]
                else:
                    act_w = act_h = self._actual_imgsz
                sx = act_w / orig_w
                sy = act_h / orig_h
                gt = vgt[:, :4].copy()
                gt[:, [0, 2]] /= sx  # x1, x2
                gt[:, [1, 3]] /= sy  # y1, y2
                gt_boxes = gt.astype(np.float32)
            raw_classes = vgt[:, 4].astype(int)
            self._warn_single_cls_nonzero_gt(raw_classes)
            gt_classes = np.clip(raw_classes, 0, self.nc - 1)
        else:
            # YOLO [cls, cx_norm, cy_norm, w_norm, h_norm]
            valid = (arr[:, 3] > 0) & (arr[:, 4] > 0)
            vgt = arr[valid]
            if len(vgt) == 0:
                return np.zeros((0, 4), np.float32), np.zeros(0, int)
            cx = vgt[:, 1] * orig_w
            cy = vgt[:, 2] * orig_h
            bw = vgt[:, 3] * orig_w
            bh = vgt[:, 4] * orig_h
            gt_boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1).astype(np.float32)
            raw_classes = vgt[:, 0].astype(int)
            self._warn_single_cls_nonzero_gt(raw_classes)
            gt_classes = np.clip(raw_classes, 0, self.nc - 1)

        return gt_boxes, gt_classes

    def _warn_single_cls_nonzero_gt(self, class_ids: np.ndarray) -> None:
        """Warn once if clipping would hide a broken single-class GT remap."""
        if (
            getattr(getattr(self, "config", None), "single_cls", False)
            and not getattr(self, "_single_cls_clip_warning_emitted", False)
            and np.any(np.asarray(class_ids) != 0)
        ):
            logger.warning(
                "single_cls validation received non-zero ground-truth class ids "
                "before clipping. The dataset/evaluator remap may have regressed."
            )
            self._single_cls_clip_warning_emitted = True

    def _track_plots_data(
        self,
        preds: List[Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        img_info: List,
        img_ids: List,
    ) -> None:
        """Accumulate confusion-matrix entries and collect sample images."""
        for i, pred in enumerate(preds):
            orig_h, orig_w = img_info[i]

            gt_boxes, gt_classes = self._parse_gt_boxes(targets[i], orig_h, orig_w)

            # --- prediction arrays ---
            pb = pred["boxes"].cpu().numpy() if len(pred["boxes"]) else np.zeros((0, 4), np.float32)
            ps = pred["scores"].cpu().numpy() if len(pred["scores"]) else np.zeros(0, np.float32)
            pc = pred["classes"].cpu().numpy().astype(int) if len(pred["classes"]) else np.zeros(0, int)

            # Confusion matrix
            if self._confusion_matrix is not None:
                self._confusion_matrix.process_image(pb, pc, ps, gt_boxes, gt_classes)

            # Sample images (first _N_VAL_SAMPLES only)
            if len(self._val_samples) < _N_VAL_SAMPLES:
                global_idx = self.seen + i
                img_path = self._resolve_img_path(
                    self.dataloader.dataset, global_idx, img_ids[i]
                )
                pm = None
                masks_t = pred.get("masks")
                if masks_t is not None and len(masks_t) > 0:
                    pm = masks_t.cpu().numpy()
                self._val_samples.append({
                    "img_path": img_path,
                    "img_id": img_ids[i],
                    "gt_boxes": gt_boxes,
                    "gt_classes": gt_classes,
                    "pred_boxes": pb,
                    "pred_classes": pc,
                    "pred_scores": ps,
                    "pred_masks": pm,
                })

    def _save_plots(self, metrics: Dict[str, float]) -> None:
        from .val_plotter import ValPlotter  # noqa: PLC0415

        plots_dir = self.save_dir / "plots"
        _raw = self.class_names or []
        names = [_raw[i] if i < len(_raw) else str(i) for i in range(self.nc)]

        def _safe(fn, *args, **kwargs):
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                logger.warning("Plot failed (%s): %s", fn.__name__, exc)

        last_eval = getattr(self.coco_evaluator, "_last_coco_eval", None)

        # Box metrics bar chart — for segmentation only include (B) keys
        if self.task == "segment":
            bm = {
                k.replace("(B)", ""): v
                for k, v in metrics.items()
                if "(B)" in k and not k.startswith("speed/")
            }
        else:
            bm = {
                k: v
                for k, v in metrics.items()
                if not k.startswith(("speed/", "metrics/loss", "metrics/best_conf"))
            }

        # Inject per-IoU-threshold P/R; fallback to aggregate P/R when unavailable
        bm["p50-95"] = bm.get("metrics/precision", 0.0)
        bm["r50-95"] = bm.get("metrics/recall", 0.0)
        if last_eval is not None and getattr(last_eval, "eval", None):
            prec_arr = last_eval.eval.get("precision")   # (T, R, K, A, M)
            rec_arr  = last_eval.eval.get("recall")       # (T, K, A, M)
            if prec_arr is not None:
                def _mp(t, _pa=prec_arr):
                    p = _pa[t, :, :, 0, -1]
                    v = p[p > -1]
                    return float(v.mean()) if len(v) else 0.0
                bm["p50-95"] = float(np.mean([_mp(t) for t in range(prec_arr.shape[0])]))
                bm["p50"]    = _mp(0)
                bm["p75"]    = _mp(5)  # IoU thresholds: [.50,.55,.60,.65,.70,.75,...]; index 5 = 0.75
            if rec_arr is not None:
                def _mr(t, _ra=rec_arr):
                    r = _ra[t, :, 0, -1]
                    v = r[r > -1]
                    return float(v.mean()) if len(v) else 0.0
                bm["r50-95"] = float(np.mean([_mr(t) for t in range(rec_arr.shape[0])]))
                bm["r50"]    = _mr(0)
                bm["r75"]    = _mr(5)  # index 5 = IoU 0.75

        if bm:
            _safe(ValPlotter.plot_metrics_bar, bm, plots_dir / "box_metrics.png",
                  title="Box Metrics")

        # Per-class box AP and Recall (sorted desc)
        if last_eval is not None:
            _safe(ValPlotter.plot_per_class_ap, last_eval, names,
                  plots_dir / "per_class_ap_box.png", "Box")
            _safe(ValPlotter.plot_per_class_recall, last_eval, names,
                  plots_dir / "per_class_recall_box.png", "Box")

        # PR / P-conf / R-conf curves
        if last_eval is not None:
            _safe(ValPlotter.plot_pr_curves, last_eval, names, plots_dir, "box")

        # Confusion matrix
        if self._confusion_matrix is not None:
            _safe(ValPlotter.plot_confusion_matrix,
                  self._confusion_matrix.matrix, names,
                  plots_dir / "confusion_matrix.png")

        # Sample images → plots/samples/
        if self._val_samples:
            try:
                import cv2  # noqa: PLC0415
            except ImportError:
                logger.warning("opencv-python not found — skipping sample image plots")
                return
            samples_dir = plots_dir / "samples"
            for idx, sample in enumerate(self._val_samples):
                if sample["img_path"] is None:
                    continue
                img_bgr = cv2.imread(str(sample["img_path"]))
                if img_bgr is None:
                    continue
                _safe(
                    ValPlotter.plot_val_sample,
                    img_bgr,
                    sample["gt_boxes"],
                    sample["gt_classes"],
                    sample["pred_boxes"],
                    sample["pred_classes"],
                    sample["pred_scores"],
                    self.class_names,
                    samples_dir / f"val_sample_{idx:02d}.jpg",
                    sample.get("pred_masks"),
                    self._get_gt_masks_for_sample(sample, img_bgr),
                )

    def _get_gt_masks_for_sample(
        self, sample: Dict, img_bgr: np.ndarray
    ) -> Optional[np.ndarray]:
        return None

    def _compute_metrics(self) -> Dict[str, float]:
        if self.config.verbose:
            logger.info("Computing COCO metrics...")

        save_json = None
        if self.config.save_json:
            save_json = str(self.save_dir / "predictions.json")

        coco_metrics = self.coco_evaluator.compute(save_json=save_json)
        # Provenance for callers (surfaced as model.last_eval_backend).
        # getattr: tests substitute lightweight evaluator stubs.
        self.eval_backend = getattr(self.coco_evaluator, "last_backend", None)

        metrics = {
            "metrics/precision": coco_metrics["precision"],
            "metrics/recall": coco_metrics["recall"],
            "metrics/mAP50-95": coco_metrics["mAP"],
            "metrics/mAP50": coco_metrics["mAP50"],
            "metrics/mAP75": coco_metrics["mAP75"],
            "metrics/precision(B)": coco_metrics["precision"],
            "metrics/recall(B)": coco_metrics["recall"],
            "metrics/mAP50(B)": coco_metrics["mAP50"],
            "metrics/mAP50-95(B)": coco_metrics["mAP"],
            "metrics/mAP_small": coco_metrics["mAP_small"],
            "metrics/mAP_medium": coco_metrics["mAP_medium"],
            "metrics/mAP_large": coco_metrics["mAP_large"],
            "metrics/AR1": coco_metrics["AR1"],
            "metrics/AR10": coco_metrics["AR10"],
            "metrics/AR100": coco_metrics["AR100"],
            "metrics/AR_max_det": coco_metrics["AR_max_det"],
            "metrics/max_det": coco_metrics["max_det"],
            "metrics/AR_small": coco_metrics["AR_small"],
            "metrics/AR_medium": coco_metrics["AR_medium"],
            "metrics/AR_large": coco_metrics["AR_large"],
        }
        metrics.update(self._validation_loss_metrics())
        metrics.update(self._best_conf_metrics())
        return metrics

    def _class_display_name(self, label: int) -> str:
        names = getattr(self, "class_names", None)
        if names and 0 <= label < len(names):
            return str(names[label])
        return str(label)

    def _best_conf_metrics(self) -> Dict[str, Any]:
        """Free deploy-threshold metric from the finished COCO evaluation.

        Sweeps F1 over the already-scored, already-matched predictions
        (IoU 0.50 matching, see ``COCOEvaluator.best_conf_thresholds``) and
        reports the confidence threshold that maximizes it, globally
        (micro-averaged over all classes) and per class. Purely additive:
        existing metric keys are untouched.

        New keys:
            ``metrics/best_conf``: global F1-optimal confidence, NaN when no
                threshold achieves F1 > 0 (or the sweep source is missing).
            ``metrics/best_conf_f1``: F1 at that threshold, NaN likewise.
            ``metrics/best_conf_per_class``: dict of class name to threshold,
                NaN entries for all-FP or no-prediction classes.
        """
        nan = float("nan")
        self._best_conf_table = None
        sweep_fn = getattr(self.coco_evaluator, "best_conf_thresholds", None)
        sweep = sweep_fn() if callable(sweep_fn) else None
        if sweep is None:
            return {
                BEST_CONF_KEY: nan,
                BEST_CONF_F1_KEY: nan,
                BEST_CONF_PER_CLASS_KEY: {},
            }

        global_thr, global_f1 = sweep["global"]
        per_class: Dict[str, float] = {}
        table = [("all", float(global_thr), float(global_f1))]
        for label in sorted(sweep["per_class"]):
            thr, f1 = sweep["per_class"][label]
            name = self._class_display_name(label)
            per_class[name] = float(thr)
            table.append((name, float(thr), float(f1)))
        self._best_conf_table = table
        return {
            BEST_CONF_KEY: float(global_thr),
            BEST_CONF_F1_KEY: float(global_f1),
            BEST_CONF_PER_CLASS_KEY: per_class,
        }

    def _print_results(self, metrics: Dict[str, float]) -> None:
        printable = {
            k: v for k, v in metrics.items() if k != BEST_CONF_PER_CLASS_KEY
        }
        super()._print_results(printable)

        table = getattr(self, "_best_conf_table", None)
        if not table:
            return
        logger.info("Best confidence threshold (max F1 at IoU 0.50):")
        logger.info("  %-24s %10s %8s", "class", "best_conf", "F1")
        for name, thr, f1 in table:
            logger.info("  %-24s %10.4f %8.4f", name, thr, f1)
        if any(np.isnan(thr) for _, thr, _ in table):
            logger.info("  (nan: no threshold reaches F1 > 0 for that class)")


class SegmentationValidator(DetectionValidator):
    """Validator for instance segmentation models."""

    task = "segment"

    def _dataset_kwargs(self) -> Dict[str, Any]:
        return {"load_segments": True}

    def _coco_api_kwargs(self) -> Dict[str, Any]:
        return {"load_segments": True}

    def _get_gt_masks_for_sample(
        self, sample: Dict, img_bgr: np.ndarray
    ) -> Optional[np.ndarray]:
        """Fetch GT segmentation masks from the COCO API for a sample image."""
        img_id = sample.get("img_id")
        coco_gt = getattr(self.coco_evaluator, "coco_gt", None)
        if img_id is None or coco_gt is None:
            return None
        try:
            from pycocotools import mask as mask_utils  # noqa: PLC0415
            ann_ids = coco_gt.getAnnIds(imgIds=[int(img_id)], iscrowd=False)
            anns = coco_gt.loadAnns(ann_ids)
            if not anns:
                return None
            im_info = coco_gt.loadImgs(int(img_id))[0]
            orig_h, orig_w = im_info["height"], im_info["width"]
            masks = []
            for ann in anns:
                seg = ann.get("segmentation")
                if not seg:
                    continue
                if isinstance(seg, list):
                    rle = mask_utils.frPyObjects(seg, orig_h, orig_w)
                    rle = mask_utils.merge(rle)
                else:
                    rle = seg
                m = mask_utils.decode(rle).astype(bool)
                # Resize to the loaded image's dimensions if they differ
                h, w = img_bgr.shape[:2]
                if m.shape != (h, w):
                    import cv2  # noqa: PLC0415
                    m = cv2.resize(
                        m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                masks.append(m)
            return np.stack(masks) if masks else None
        except Exception as exc:
            logger.debug("GT mask fetch failed for img_id=%s: %s", img_id, exc)
            return None

    def _init_metrics(self) -> None:
        from libreyolo.validation import COCOEvaluator

        super()._init_metrics()
        self.bbox_evaluator = self.coco_evaluator
        self.mask_evaluator = COCOEvaluator(
            self.bbox_evaluator.coco_gt,
            iou_type="segm",
            label_to_category_id=self._coco_label_to_category_id,
            max_det=self._coco_max_det(),
            faster_coco_eval=getattr(self.config, "faster_coco_eval", False),
        )

    def _update_metrics(
        self,
        preds: List[Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        img_info: List,
        img_ids: List | None = None,
    ) -> None:
        # super() updates bbox_evaluator (== self.coco_evaluator) + tracks plots data
        super()._update_metrics(preds, targets, img_info, img_ids)
        if img_ids is None:
            return
        for i in range(len(preds)):
            self.mask_evaluator.update(preds[i], img_ids[i])

    def _save_plots(self, metrics: Dict[str, float]) -> None:
        from .val_plotter import ValPlotter  # noqa: PLC0415

        super()._save_plots(metrics)  # box metrics, CM, PR curves, sample images

        plots_dir = self.save_dir / "plots"
        _raw = self.class_names or []
        names = [_raw[i] if i < len(_raw) else str(i) for i in range(self.nc)]

        def _safe(fn, *args, **kwargs):
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                logger.warning("Plot failed (%s): %s", fn.__name__, exc)

        last_mask_eval = getattr(self.mask_evaluator, "_last_coco_eval", None)

        # Mask metrics bar chart — primary (no-suffix) keys are mask metrics;
        # precision/recall only exist with (M) suffix so merge both.
        mm: Dict[str, float] = {}
        for k, v in metrics.items():
            if k.startswith("speed/") or "(B)" in k:
                continue
            if "(M)" in k:
                mm[k.replace("(M)", "")] = v
            else:
                mm[k] = v

        # Inject per-IoU P/R for mask metrics
        mm["p50-95"] = mm.get("metrics/precision", 0.0)
        mm["r50-95"] = mm.get("metrics/recall", 0.0)
        if last_mask_eval is not None and getattr(last_mask_eval, "eval", None):
            prec_arr = last_mask_eval.eval.get("precision")
            rec_arr  = last_mask_eval.eval.get("recall")
            if prec_arr is not None:
                def _mmp(t, _pa=prec_arr):
                    p = _pa[t, :, :, 0, -1]
                    v = p[p > -1]
                    return float(v.mean()) if len(v) else 0.0
                mm["p50-95"] = float(np.mean([_mmp(t) for t in range(prec_arr.shape[0])]))
                mm["p50"]    = _mmp(0)
                mm["p75"]    = _mmp(5)  # IoU thresholds: [.50,.55,...,.75,...]; index 5 = 0.75
            if rec_arr is not None:
                def _mmr(t, _ra=rec_arr):
                    r = _ra[t, :, 0, -1]
                    v = r[r > -1]
                    return float(v.mean()) if len(v) else 0.0
                mm["r50-95"] = float(np.mean([_mmr(t) for t in range(rec_arr.shape[0])]))
                mm["r50"]    = _mmr(0)
                mm["r75"]    = _mmr(5)  # index 5 = IoU 0.75

        if mm:
            _safe(ValPlotter.plot_metrics_bar, mm, plots_dir / "mask_metrics.png",
                  title="Mask Metrics")

        # Per-class mask AP and Recall (sorted desc)
        if last_mask_eval is not None:
            _safe(ValPlotter.plot_per_class_ap, last_mask_eval, names,
                  plots_dir / "per_class_ap_mask.png", "Mask")
            _safe(ValPlotter.plot_per_class_recall, last_mask_eval, names,
                  plots_dir / "per_class_recall_mask.png", "Mask")

        # PR / P-conf / R-conf curves for masks
        if last_mask_eval is not None:
            _safe(ValPlotter.plot_pr_curves, last_mask_eval, names, plots_dir, "mask")

    def _compute_metrics(self) -> Dict[str, float]:
        if self.config.verbose:
            logger.info("Computing bbox and mask COCO metrics...")

        bbox_json = None
        mask_json = None
        if self.config.save_json:
            bbox_json = str(self.save_dir / "predictions_bbox.json")
            mask_json = str(self.save_dir / "predictions_masks.json")

        bbox = self.bbox_evaluator.compute(save_json=bbox_json)
        mask = self.mask_evaluator.compute(save_json=mask_json)
        # Provenance for callers (surfaced as model.last_eval_backend).
        # getattr: tests substitute lightweight evaluator stubs.
        self.eval_backend = getattr(self.bbox_evaluator, "last_backend", None)

        return {
            "metrics/mAP50-95": mask["mAP"],
            "metrics/mAP50": mask["mAP50"],
            "metrics/mAP75": mask["mAP75"],
            "metrics/mAP_small": mask["mAP_small"],
            "metrics/mAP_medium": mask["mAP_medium"],
            "metrics/mAP_large": mask["mAP_large"],
            "metrics/AR1": mask["AR1"],
            "metrics/AR10": mask["AR10"],
            "metrics/AR100": mask["AR100"],
            "metrics/AR_max_det": mask["AR_max_det"],
            "metrics/max_det": mask["max_det"],
            "metrics/AR_small": mask["AR_small"],
            "metrics/AR_medium": mask["AR_medium"],
            "metrics/AR_large": mask["AR_large"],
            "metrics/precision(B)": bbox["precision"],
            "metrics/recall(B)": bbox["recall"],
            "metrics/mAP50(B)": bbox["mAP50"],
            "metrics/mAP50-95(B)": bbox["mAP"],
            "metrics/precision(M)": mask["precision"],
            "metrics/recall(M)": mask["recall"],
            "metrics/mAP50(M)": mask["mAP50"],
            "metrics/mAP50-95(M)": mask["mAP"],
        }

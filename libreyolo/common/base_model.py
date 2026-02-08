"""
Base model class for LibreYOLO model wrappers.

Provides shared functionality for all YOLO model variants.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .image_loader import ImageInput, ImageLoader
from .results import Boxes, Results
from .utils import (
    COCO_CLASSES,
    draw_boxes,
    get_safe_stem,
    get_slice_bboxes,
    draw_tile_grid,
    nms,
    resolve_save_path,
)


class LibreYOLOBase(ABC):
    """
    Abstract base class for LibreYOLO model wrappers.

    Provides shared functionality for inference, saving, and tiling
    across all YOLO model variants.

    Subclasses must implement the abstract methods to provide model-specific
    behavior for initialization, forward pass, and postprocessing.
    """

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _init_model(self) -> nn.Module:
        """Initialize and return the neural network model."""
        pass

    @abstractmethod
    def _get_available_layers(self) -> Dict[str, nn.Module]:
        """Return mapping of layer names to module objects."""
        pass

    @abstractmethod
    def _get_valid_sizes(self) -> List[str]:
        """Return list of valid size codes for this model."""
        pass

    @abstractmethod
    def _get_model_name(self) -> str:
        """Return the model name for metadata."""
        pass

    @abstractmethod
    def _get_input_size(self) -> int:
        """Return the input size for this model."""
        pass

    @abstractmethod
    def _preprocess(
        self, image: ImageInput, color_format: str = "auto", input_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        """Preprocess image for inference.

        Args:
            image: Input image.
            color_format: Color format hint.
            input_size: Override input size (None = model default).
        """
        pass

    @abstractmethod
    def _forward(self, input_tensor: torch.Tensor) -> Any:
        """Run model forward pass."""
        pass

    @abstractmethod
    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
    ) -> Dict:
        """Postprocess model output to detections."""
        pass

    # =========================================================================
    # OVERRIDABLE METHODS - Subclasses may override for custom behavior
    # =========================================================================

    def _get_val_preprocessor(self, img_size: int = None):
        """
        Return the validation preprocessor for this model.

        Override in subclasses that need different preprocessing
        (e.g., YOLOX uses letterbox + no normalization).

        Args:
            img_size: Target image size. Defaults to model's native input size.

        Returns:
            A preprocessor instance with __call__(img, targets, input_size).
        """
        from libreyolo.validation.preprocessors import StandardValPreprocessor
        if img_size is None:
            img_size = self._get_input_size()
        return StandardValPreprocessor(img_size=(img_size, img_size))

    # =========================================================================
    # SHARED IMPLEMENTATION
    # =========================================================================

    def __init__(
        self,
        model_path: Union[str, dict, None],
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        """
        Initialize the model.

        Args:
            model_path: Path to weights file, pre-loaded state_dict, or None
                for random initialization (fresh model for training).
            size: Model size variant.
            nb_classes: Number of classes (default: 80 for COCO).
            device: Device for inference ("auto", "cuda", "mps", "cpu").
            **kwargs: Additional model-specific arguments.
        """
        # Validate size
        valid_sizes = self._get_valid_sizes()
        if size not in valid_sizes:
            raise ValueError(
                f"Invalid size: {size}. Must be one of: {', '.join(valid_sizes)}"
            )

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Store parameters
        self.size = size
        self.nb_classes = nb_classes

        # Build names dict (matches Ultralytics model.names)
        if nb_classes == 80:
            self.names: Dict[int, str] = {i: n for i, n in enumerate(COCO_CLASSES)}
        else:
            self.names: Dict[int, str] = {i: f"class_{i}" for i in range(nb_classes)}

        # Store extra kwargs for subclass use
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Initialize model (implemented by subclass)
        self.model = self._init_model()

        # Load weights (or skip for fresh model)
        if model_path is None:
            self.model_path = None
        elif isinstance(model_path, dict):
            self.model_path = None
            self.model.load_state_dict(model_path, strict=self._strict_loading())
        else:
            self.model_path = model_path
            self._load_weights(model_path)

        # Fresh models start in train mode; loaded models in eval mode
        if model_path is None:
            self.model.train()
        else:
            self.model.eval()
        self.model.to(self.device)

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        """Rebuild model with a new class count, preserving weights where shapes match.

        Used when training on a dataset with a different number of classes
        than the model was initialized with. Backbone/neck weights are preserved;
        head weights (which depend on nb_classes) are reinitialized.

        Args:
            new_nb_classes: The new number of classes.
        """
        old_state = self.model.state_dict()
        self.nb_classes = new_nb_classes
        self.model = self._init_model()

        # Transfer weights with matching shapes (backbone/neck preserved, head reinitialized)
        new_state = self.model.state_dict()
        for key in old_state:
            if key in new_state and old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key]

        self.model.load_state_dict(new_state)
        self.model.to(self.device)

    def _strict_loading(self) -> bool:
        """Return whether to use strict mode when loading weights.

        Override in subclasses that need non-strict loading.
        """
        return True

    def _load_weights(self, model_path: str):
        """Load model weights from file.

        Handles both raw state_dicts and training checkpoint dicts
        ({"model": state_dict, "optimizer": ..., "epoch": ...}).
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        try:
            loaded = torch.load(model_path, map_location="cpu", weights_only=False)

            if isinstance(loaded, dict):
                if "model" in loaded:
                    state_dict = loaded["model"]
                elif "state_dict" in loaded:
                    state_dict = loaded["state_dict"]
                else:
                    state_dict = loaded
            else:
                state_dict = loaded

            self.model.load_state_dict(state_dict, strict=self._strict_loading())
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {model_path}: {e}"
            ) from e

    def get_available_layer_names(self) -> List[str]:
        """Get list of available layer names."""
        return sorted(self._get_available_layers().keys())

    def __call__(
        self,
        source: ImageInput = None,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        batch: int = 1,
        # libreyolo-specific
        output_path: str = None,
        color_format: str = "auto",
        tiling: bool = False,
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> Union[Results, List[Results]]:
        """
        Run inference on an image or directory.

        Args:
            source: Input image or directory path.
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            imgsz: Input size override (None = model default).
            classes: Filter to specific class IDs.
            max_det: Maximum detections per image.
            save: If True, saves annotated image.
            batch: Batch size for directory processing.
            output_path: Optional output path.
            color_format: Color format hint.
            tiling: Enable tiled inference for large images.
            overlap_ratio: Tile overlap ratio.
            output_file_format: Output format ("jpg", "png", "webp").
            **kwargs: Additional arguments for postprocessing.

        Returns:
            Results instance or list of Results.
        """
        if output_file_format is not None:
            output_file_format = output_file_format.lower().lstrip(".")
            if output_file_format not in ("jpg", "jpeg", "png", "webp"):
                raise ValueError(
                    f"Invalid output_file_format: {output_file_format}. "
                    "Must be one of: 'jpg', 'png', 'webp'"
                )

        # Handle directory input
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            image_paths = ImageLoader.collect_images(source)
            if not image_paths:
                return []
            return self._process_in_batches(
                image_paths,
                batch=batch,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
                tiling=tiling,
                overlap_ratio=overlap_ratio,
                output_file_format=output_file_format,
                **kwargs,
            )

        # Use tiled inference if enabled
        if tiling:
            return self._predict_tiled(
                source,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
                overlap_ratio=overlap_ratio,
                output_file_format=output_file_format,
                **kwargs,
            )

        return self._predict_single(
            source,
            save=save,
            output_path=output_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            max_det=max_det,
            color_format=color_format,
            output_file_format=output_file_format,
            **kwargs,
        )

    def _process_in_batches(
        self,
        image_paths: List[Path],
        batch: int = 1,
        save: bool = False,
        output_path: str = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        tiling: bool = False,
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> List[Results]:
        """Process multiple images in batches."""
        results = []
        for i in range(0, len(image_paths), batch):
            chunk = image_paths[i : i + batch]
            for path in chunk:
                if tiling:
                    results.append(
                        self._predict_tiled(
                            path,
                            save=save,
                            output_path=output_path,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            classes=classes,
                            max_det=max_det,
                            color_format=color_format,
                            overlap_ratio=overlap_ratio,
                            output_file_format=output_file_format,
                            **kwargs,
                        )
                    )
                else:
                    results.append(
                        self._predict_single(
                            path,
                            save=save,
                            output_path=output_path,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            classes=classes,
                            max_det=max_det,
                            color_format=color_format,
                            output_file_format=output_file_format,
                            **kwargs,
                        )
                    )
        return results

    @staticmethod
    def _apply_classes_filter(
        boxes_t: torch.Tensor,
        conf_t: torch.Tensor,
        cls_t: torch.Tensor,
        classes: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter detections to keep only the requested class IDs."""
        mask = torch.zeros(len(cls_t), dtype=torch.bool, device=cls_t.device)
        for cid in classes:
            mask |= cls_t == cid
        return boxes_t[mask], conf_t[mask], cls_t[mask]

    def _wrap_results(
        self,
        detections: Dict,
        original_size: Tuple[int, int],
        image_path,
        classes: Optional[List[int]],
    ) -> Results:
        """Convert raw detection dict to a Results object.

        Args:
            detections: Dict with 'boxes', 'scores', 'classes', 'num_detections'.
            original_size: (width, height) from preprocessing.
            image_path: Source path or None.
            classes: Optional class filter list.
        """
        if detections["num_detections"] == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            conf_t = torch.zeros((0,), dtype=torch.float32)
            cls_t = torch.zeros((0,), dtype=torch.float32)
        else:
            raw_boxes = detections["boxes"]
            if isinstance(raw_boxes, torch.Tensor):
                boxes_t = raw_boxes.float()
            else:
                boxes_t = torch.tensor(raw_boxes, dtype=torch.float32)

            raw_conf = detections["scores"]
            if isinstance(raw_conf, torch.Tensor):
                conf_t = raw_conf.float()
            else:
                conf_t = torch.tensor(raw_conf, dtype=torch.float32)

            raw_cls = detections["classes"]
            if isinstance(raw_cls, torch.Tensor):
                cls_t = raw_cls.float()
            else:
                cls_t = torch.tensor(raw_cls, dtype=torch.float32)

        # Apply class filter
        if classes is not None and len(boxes_t) > 0:
            boxes_t, conf_t, cls_t = self._apply_classes_filter(
                boxes_t, conf_t, cls_t, classes
            )

        # original_size from preprocess is (W, H); orig_shape follows Ultralytics (H, W)
        orig_w, orig_h = original_size
        orig_shape = (orig_h, orig_w)

        return Results(
            boxes=Boxes(boxes_t, conf_t, cls_t),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    def _predict_single(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> Results:
        """Run inference on a single image."""
        image_path = image if isinstance(image, (str, Path)) else None

        # Resolve input size
        effective_imgsz = imgsz if imgsz is not None else self._get_input_size()

        # Preprocess
        input_tensor, original_img, original_size = self._preprocess(
            image, color_format, input_size=effective_imgsz
        )

        # Forward pass
        with torch.no_grad():
            output = self._forward(input_tensor.to(self.device))

        # Postprocess
        detections = self._postprocess(
            output, conf, iou, original_size, max_det=max_det, **kwargs
        )

        # Wrap into Results
        result = self._wrap_results(detections, original_size, image_path, classes)

        # Save annotated image
        if save:
            if len(result) > 0:
                annotated_img = draw_boxes(
                    original_img,
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                )
            else:
                annotated_img = original_img

            ext = output_file_format or "jpg"
            save_path = resolve_save_path(
                output_path,
                image_path,
                ext=ext,
                default_dir="runs/detections",
                model_name=f"{self._get_model_name()}_{self.size}",
            )
            annotated_img.save(save_path)
            result.saved_path = str(save_path)

        return result

    def _merge_tile_detections(
        self,
        boxes: List,
        scores: List,
        classes: List,
        iou_thres: float,
    ) -> Tuple[List, List, List]:
        """Merge detections from tiles using class-wise NMS."""
        if not boxes:
            return [], [], []

        boxes_t = torch.tensor(boxes, dtype=torch.float32, device=self.device)
        scores_t = torch.tensor(scores, dtype=torch.float32, device=self.device)
        classes_t = torch.tensor(classes, dtype=torch.int64, device=self.device)

        final_boxes, final_scores, final_classes = [], [], []

        for cls_id in torch.unique(classes_t):
            mask = classes_t == cls_id
            cls_boxes = boxes_t[mask]
            cls_scores = scores_t[mask]

            keep = nms(cls_boxes, cls_scores, iou_thres)

            final_boxes.extend(cls_boxes[keep].cpu().tolist())
            final_scores.extend(cls_scores[keep].cpu().tolist())
            final_classes.extend([cls_id.item()] * len(keep))

        return final_boxes, final_scores, final_classes

    def _predict_tiled(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> Results:
        """Run tiled inference on large images."""
        input_size = imgsz if imgsz is not None else self._get_input_size()
        img_pil = ImageLoader.load(image, color_format=color_format)
        orig_width, orig_height = img_pil.size
        image_path = image if isinstance(image, (str, Path)) else None

        # Skip tiling if image is small enough
        if orig_width <= input_size and orig_height <= input_size:
            return self._predict_single(
                image,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
                output_file_format=output_file_format,
                **kwargs,
            )

        # Get tile coordinates
        slices = get_slice_bboxes(
            orig_width, orig_height, slice_size=input_size, overlap_ratio=overlap_ratio
        )

        # Process tiles
        all_boxes, all_scores, all_classes = [], [], []
        tiles_data = []

        for idx, (x1, y1, x2, y2) in enumerate(slices):
            tile = img_pil.crop((x1, y1, x2, y2))

            if save:
                tiles_data.append(
                    {"index": idx, "coords": (x1, y1, x2, y2), "image": tile.copy()}
                )

            tile_result = self._predict_single(
                tile, save=False, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
                **kwargs,
            )

            # Shift boxes to original coordinates
            if len(tile_result) > 0:
                tile_boxes = tile_result.boxes.xyxy.tolist()
                for box in tile_boxes:
                    shifted_box = [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1]
                    all_boxes.append(shifted_box)
                all_scores.extend(tile_result.boxes.conf.tolist())
                all_classes.extend(tile_result.boxes.cls.tolist())

        # Merge detections
        final_boxes, final_scores, final_classes = self._merge_tile_detections(
            all_boxes, all_scores, all_classes, iou
        )

        # Build Results
        original_size = (orig_width, orig_height)
        detections = {
            "boxes": final_boxes,
            "scores": final_scores,
            "classes": final_classes,
            "num_detections": len(final_boxes),
        }
        result = self._wrap_results(detections, original_size, image_path, classes)

        # Attach tiling metadata as extra attributes
        result.tiled = True
        result.num_tiles = len(slices)

        # Save if requested
        if save:
            ext = output_file_format or "jpg"

            if isinstance(image_path, (str, Path)):
                stem = get_safe_stem(image_path)
            else:
                stem = "inference"
            model_tag = f"{self._get_model_name()}_{self.size}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            if output_path:
                base_path = Path(output_path)
                if base_path.suffix == "":
                    save_dir = base_path / f"{stem}_{model_tag}_{timestamp}"
                else:
                    save_dir = base_path.parent / f"{stem}_{model_tag}_{timestamp}"
            else:
                save_dir = Path("runs/tiled_detections") / f"{stem}_{model_tag}_{timestamp}"

            save_dir.mkdir(parents=True, exist_ok=True)

            # Save tiles
            tiles_dir = save_dir / "tiles"
            tiles_dir.mkdir(parents=True, exist_ok=True)
            for tile_data in tiles_data:
                tile_filename = f"tile_{tile_data['index']:03d}.{ext}"
                tile_data["image"].save(tiles_dir / tile_filename)

            # Save annotated image
            if len(result) > 0:
                annotated_img = draw_boxes(
                    img_pil,
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                )
            else:
                annotated_img = img_pil.copy()

            annotated_img.save(save_dir / f"final_image.{ext}")

            # Save grid visualization
            grid_img = draw_tile_grid(img_pil, slices)
            grid_path = save_dir / f"grid_visualization.{ext}"
            grid_img.save(grid_path)

            # Save metadata
            metadata = {
                "model": self._get_model_name(),
                "size": self.size,
                "image_source": str(image_path) if image_path else "PIL/numpy input",
                "original_size": [orig_width, orig_height],
                "num_tiles": len(slices),
                "tile_size": input_size,
                "overlap_ratio": overlap_ratio,
                "num_detections": len(result),
                "conf": conf,
                "iou": iou,
            }
            with open(save_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            result.saved_path = str(save_dir)
            result.tiles_path = str(tiles_dir)
            result.grid_path = str(grid_path)

        return result

    def export(
        self,
        format: str = "onnx",
        *,
        output_path: str = None,
        imgsz: int = None,
        opset: int = 13,
        simplify: bool = True,
        dynamic: bool = True,
        half: bool = False,
        batch: int = 1,
        device: str = None,
    ) -> str:
        """Export model to deployment format.

        Args:
            format: Target format ("onnx", "torchscript").
            output_path: Output file path (auto-generated if None).
            imgsz: Input resolution (default: model's native size).
            opset: ONNX opset version (default: 13).
            simplify: Run ONNX graph simplification (default: True).
            dynamic: Enable dynamic axes (default: True).
            half: Export in FP16 (default: False).
            batch: Batch size for static graph (default: 1).
            device: Device to trace on (default: model's current device).

        Returns:
            Path to the exported model file.
        """
        from libreyolo.export import Exporter

        return Exporter(self)(
            format,
            output_path=output_path,
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            half=half,
            batch=batch,
            device=device,
        )

    def predict(self, *args, **kwargs) -> Union[Results, List[Results]]:
        """Alias for __call__ method."""
        return self(*args, **kwargs)

    def val(
        self,
        data: str = None,
        batch: int = 16,
        imgsz: int = None,
        conf: float = 0.001,
        iou: float = 0.6,
        device: str = None,
        split: str = "val",
        save_json: bool = False,
        plots: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Run validation on a dataset.

        Computes standard object detection metrics including mAP50, mAP50-95,
        precision, and recall.

        Args:
            data: Path to data.yaml file containing dataset configuration.
            batch: Batch size for validation.
            imgsz: Image size for validation. Defaults to model's native input size.
            conf: Confidence threshold. Use low value (0.001) for mAP calculation.
            iou: IoU threshold for NMS.
            device: Device to use (default: same as model).
            split: Dataset split to validate on ("val", "test").
            save_json: Save predictions in COCO JSON format.
            plots: Generate confusion matrix and other plots.
            verbose: Print detailed metrics.
            **kwargs: Additional arguments passed to ValidationConfig.

        Returns:
            Dictionary with validation metrics:
                - metrics/precision: Mean precision at conf threshold
                - metrics/recall: Mean recall at conf threshold
                - metrics/mAP50: Mean AP at IoU=0.50
                - metrics/mAP50-95: Mean AP across IoU 0.50-0.95

        Example:
            >>> model = LIBREYOLO("weights/libreyoloXs.pt")
            >>> results = model.val(data="coco8.yaml", batch=16)
            >>> print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")
        """
        from libreyolo.validation import DetectionValidator, ValidationConfig

        # Use model's native input size if not specified
        if imgsz is None:
            imgsz = self._get_input_size()

        config = ValidationConfig(
            data=data,
            batch_size=batch,
            imgsz=imgsz,
            conf_thres=conf,
            iou_thres=iou,
            device=device or str(self.device),
            split=split,
            save_json=save_json,
            plots=plots,
            verbose=verbose,
            **kwargs,
        )

        validator = DetectionValidator(model=self, config=config)
        return validator()

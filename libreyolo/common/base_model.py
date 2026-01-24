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
from .utils import (
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
        self, image: ImageInput, color_format: str = "auto"
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        """Preprocess image for inference."""
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
        model_path: Union[str, dict],
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        """
        Initialize the model.

        Args:
            model_path: Path to weights file or pre-loaded state_dict.
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

        # Store extra kwargs for subclass use
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Initialize model (implemented by subclass)
        self.model = self._init_model()

        # Load weights
        if isinstance(model_path, dict):
            self.model_path = None
            self.model.load_state_dict(model_path, strict=self._strict_loading())
        else:
            self.model_path = model_path
            self._load_weights(model_path)

        # Set to evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)

    def _strict_loading(self) -> bool:
        """Return whether to use strict mode when loading weights.

        Override in subclasses that need non-strict loading.
        """
        return True

    def _load_weights(self, model_path: str):
        """Load model weights from file."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
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
        image: ImageInput,
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        color_format: str = "auto",
        batch_size: int = 1,
        tiling: bool = False,
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> Union[Dict, List[Dict]]:
        """
        Run inference on an image or directory.

        Args:
            image: Input image or directory path.
            save: If True, saves annotated image.
            output_path: Optional output path.
            conf_thres: Confidence threshold.
            iou_thres: IoU threshold for NMS.
            color_format: Color format hint.
            batch_size: Batch size for directory processing.
            tiling: Enable tiled inference for large images.
            overlap_ratio: Tile overlap ratio.
            output_file_format: Output format ("jpg", "png", "webp").
            **kwargs: Additional arguments for postprocessing.

        Returns:
            Detection dictionary or list of dictionaries.
        """
        if output_file_format is not None:
            output_file_format = output_file_format.lower().lstrip(".")
            if output_file_format not in ("jpg", "jpeg", "png", "webp"):
                raise ValueError(
                    f"Invalid output_file_format: {output_file_format}. "
                    "Must be one of: 'jpg', 'png', 'webp'"
                )

        # Handle directory input
        if isinstance(image, (str, Path)) and Path(image).is_dir():
            image_paths = ImageLoader.collect_images(image)
            if not image_paths:
                return []
            return self._process_in_batches(
                image_paths,
                batch_size=batch_size,
                save=save,
                output_path=output_path,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                color_format=color_format,
                tiling=tiling,
                overlap_ratio=overlap_ratio,
                output_file_format=output_file_format,
                **kwargs,
            )

        # Use tiled inference if enabled
        if tiling:
            return self._predict_tiled(
                image,
                save,
                output_path,
                conf_thres,
                iou_thres,
                color_format,
                overlap_ratio,
                output_file_format,
                **kwargs,
            )

        return self._predict_single(
            image,
            save,
            output_path,
            conf_thres,
            iou_thres,
            color_format,
            output_file_format,
            **kwargs,
        )

    def _process_in_batches(
        self,
        image_paths: List[Path],
        batch_size: int = 1,
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        color_format: str = "auto",
        tiling: bool = False,
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> List[Dict]:
        """Process multiple images in batches."""
        results = []
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i : i + batch_size]
            for path in chunk:
                if tiling:
                    results.append(
                        self._predict_tiled(
                            path,
                            save,
                            output_path,
                            conf_thres,
                            iou_thres,
                            color_format,
                            overlap_ratio,
                            output_file_format,
                            **kwargs,
                        )
                    )
                else:
                    results.append(
                        self._predict_single(
                            path,
                            save,
                            output_path,
                            conf_thres,
                            iou_thres,
                            color_format,
                            output_file_format,
                            **kwargs,
                        )
                    )
        return results

    def _predict_single(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        color_format: str = "auto",
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """Run inference on a single image."""
        image_path = image if isinstance(image, (str, Path)) else None

        # Preprocess
        input_tensor, original_img, original_size = self._preprocess(
            image, color_format
        )

        # Forward pass
        with torch.no_grad():
            output = self._forward(input_tensor.to(self.device))

        # Postprocess
        detections = self._postprocess(
            output, conf_thres, iou_thres, original_size, **kwargs
        )

        detections["source"] = str(image_path) if image_path else None

        # Save annotated image
        if save:
            if detections["num_detections"] > 0:
                annotated_img = draw_boxes(
                    original_img,
                    detections["boxes"],
                    detections["scores"],
                    detections["classes"],
                )
            else:
                annotated_img = original_img

            ext = output_file_format or "jpg"
            save_path = resolve_save_path(
                output_path,
                image_path,
                ext=ext,
                default_dir="runs/detections",
            )
            annotated_img.save(save_path)
            detections["saved_path"] = str(save_path)

        return detections

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
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        color_format: str = "auto",
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """Run tiled inference on large images."""
        input_size = self._get_input_size()
        img_pil = ImageLoader.load(image, color_format=color_format)
        orig_width, orig_height = img_pil.size
        image_path = image if isinstance(image, (str, Path)) else None

        # Skip tiling if image is small enough
        if orig_width <= input_size and orig_height <= input_size:
            return self._predict_single(
                image,
                save,
                output_path,
                conf_thres,
                iou_thres,
                color_format,
                output_file_format,
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

            result = self._predict_single(
                tile, save=False, conf_thres=conf_thres, iou_thres=iou_thres, **kwargs
            )

            # Shift boxes to original coordinates
            for box in result["boxes"]:
                shifted_box = [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1]
                all_boxes.append(shifted_box)
            all_scores.extend(result["scores"])
            all_classes.extend(result["classes"])

        # Merge detections
        final_boxes, final_scores, final_classes = self._merge_tile_detections(
            all_boxes, all_scores, all_classes, iou_thres
        )

        detections = {
            "boxes": final_boxes,
            "scores": final_scores,
            "classes": final_classes,
            "num_detections": len(final_boxes),
            "source": str(image_path) if image_path else None,
            "tiled": True,
            "num_tiles": len(slices),
        }

        # Save if requested
        if save:
            ext = output_file_format or "jpg"

            if isinstance(image_path, (str, Path)):
                stem = get_safe_stem(image_path)
            else:
                stem = "inference"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if output_path:
                base_path = Path(output_path)
                if base_path.suffix == "":
                    save_dir = base_path / f"{stem}_{timestamp}"
                else:
                    save_dir = base_path.parent / f"{stem}_{timestamp}"
            else:
                save_dir = Path("runs/tiled_detections") / f"{stem}_{timestamp}"

            save_dir.mkdir(parents=True, exist_ok=True)

            # Save tiles
            tiles_dir = save_dir / "tiles"
            tiles_dir.mkdir(parents=True, exist_ok=True)
            for tile_data in tiles_data:
                tile_filename = f"tile_{tile_data['index']:03d}.{ext}"
                tile_data["image"].save(tiles_dir / tile_filename)

            # Save annotated image
            if detections["num_detections"] > 0:
                annotated_img = draw_boxes(
                    img_pil,
                    detections["boxes"],
                    detections["scores"],
                    detections["classes"],
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
                "num_detections": detections["num_detections"],
                "conf_thres": conf_thres,
                "iou_thres": iou_thres,
            }
            with open(save_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            detections["saved_path"] = str(save_dir)
            detections["tiles_path"] = str(tiles_dir)
            detections["grid_path"] = str(grid_path)

        return detections

    def export(
        self, output_path: str = None, input_size: int = None, opset: int = 12
    ) -> str:
        """Export the model to ONNX format."""
        import importlib.util

        if importlib.util.find_spec("onnx") is None:
            raise ImportError(
                "ONNX export requires the optional ONNX dependencies. "
                "Install them with `uv sync --extra onnx` or `pip install -e '.[onnx]'`."
            )

        if input_size is None:
            input_size = self._get_input_size()

        if output_path is None:
            if self.model_path and isinstance(self.model_path, str):
                output_path = str(Path(self.model_path).with_suffix(".onnx"))
            else:
                output_path = f"{self._get_model_name().lower()}{self.size}.onnx"

        print(f"Exporting {self._get_model_name()} {self.size} to {output_path}...")

        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["images"],
                output_names=["output"],
                dynamic_axes={
                    "images": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch"},
                },
            )
            print(f"Export complete: {output_path}")
            return output_path
        except Exception as e:
            print(f"Export failed: {e}")
            raise

    def predict(self, *args, **kwargs) -> Union[Dict, List[Dict]]:
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
            >>> model = LIBREYOLO("weights/libreyolo8n.pt", size="n")
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

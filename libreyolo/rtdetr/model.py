"""
RT-DETR Model wrapper for LibreYOLO.

Provides the LIBREYOLORTDETR class with the same API as other LibreYOLO models.
"""

from datetime import datetime
from typing import Union, List, Optional, Dict
from pathlib import Path

import torch
from PIL import Image

from .nn import RTDETRModel
from .utils import preprocess_image, draw_boxes
from ..common.image_loader import ImageInput, ImageLoader


class LIBREYOLORTDETR:
    """
    RT-DETR model for object detection.

    Provides the same API as other LibreYOLO models (LIBREYOLO8, LIBREYOLO11, etc.)
    but uses the RT-DETR transformer-based architecture.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
        size: Model size variant. Must be one of: "s", "m", "l", "x"
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference. "auto" (default) uses CUDA if available.

    Example:
        >>> model = LIBREYOLORTDETR("rtdetrv2_r50vd.pth", size="l")
        >>> results = model("image.jpg", save=True)
        >>> print(f"Found {results['num_detections']} objects")
    """

    def __init__(
        self,
        model_path: Union[str, dict],
        size: str,
        nb_classes: int = 80,
        device: str = "auto"
    ):
        if size not in ['s', 'ms', 'm', 'l', 'x']:
            raise ValueError(f"Invalid size: {size}. Must be one of: 's', 'ms', 'm', 'l', 'x' (ms=M*, m=M)")

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

        self.size = size
        self.nb_classes = nb_classes
        self.input_size = 640

        # Initialize model
        self.model = RTDETRModel(config=size, nb_classes=nb_classes)

        # Load weights
        if isinstance(model_path, dict):
            self.model_path = None
            self._load_state_dict(model_path)
        else:
            self.model_path = model_path
            self._load_weights(model_path)

        # Set to evaluation/deploy mode
        self.model.deploy()
        self.model.to(self.device)

    def _load_weights(self, model_path: str):
        """Load model weights from file."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self._load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {model_path}: {e}") from e

    def _load_state_dict(self, checkpoint: dict):
        """Load state dict handling various checkpoint formats."""
        # Handle different checkpoint formats from RT-DETR
        if 'ema' in checkpoint and isinstance(checkpoint['ema'], dict):
            if 'module' in checkpoint['ema']:
                state_dict = checkpoint['ema']['module']
            else:
                state_dict = checkpoint['ema']
        elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Try to load, with fallback to non-strict mode
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # Some keys might be missing or extra, try non-strict
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Warning: Missing keys in state_dict: {len(missing)} keys")
            if unexpected:
                print(f"Warning: Unexpected keys in state_dict: {len(unexpected)} keys")

    def __call__(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.5,
        batch_size: int = 1,
        tiling: bool = False,
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
    ) -> Union[Dict, List[Dict]]:
        """
        Run inference on an image or directory of images.

        Args:
            image: Input image or directory. Supported types:
                - str: Local file path, directory path, or URL
                - pathlib.Path: Local file path or directory path
                - PIL.Image: PIL Image object
                - np.ndarray: NumPy array (HWC, RGB)
                - torch.Tensor: PyTorch tensor
                - bytes: Raw image bytes
            save: If True, saves the image with detections drawn.
            output_path: Optional path to save the annotated image.
            conf_thres: Confidence threshold (default: 0.5).
            batch_size: Number of images per batch for directory processing.
            tiling: Enable tiling for large images (default: False).
            overlap_ratio: Overlap ratio for tiling (default: 0.2).
            output_file_format: Output format ('jpg', 'png', 'webp').

        Returns:
            For single image: dict with keys 'boxes', 'scores', 'classes', 'num_detections'
            For directory: list of dicts
        """
        # Validate output format
        if output_file_format is not None:
            output_file_format = output_file_format.lower().lstrip('.')
            if output_file_format not in ('jpg', 'jpeg', 'png', 'webp'):
                raise ValueError(f"Invalid output_file_format: {output_file_format}")

        # Check if input is a directory
        if isinstance(image, (str, Path)) and Path(image).is_dir():
            image_paths = ImageLoader.collect_images(image)
            if not image_paths:
                return []
            return self._process_in_batches(
                image_paths, batch_size, save, output_path, conf_thres,
                tiling, overlap_ratio, output_file_format
            )

        # Handle tiling for large images
        if tiling:
            return self._predict_tiled(
                image, save, output_path, conf_thres, overlap_ratio, output_file_format
            )

        return self._predict_single(
            image, save, output_path, conf_thres, output_file_format
        )

    def _predict_single(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.5,
        output_file_format: Optional[str] = None
    ) -> Dict:
        """Run inference on a single image."""
        # Preprocess
        input_tensor, original_img, original_size = preprocess_image(
            image, input_size=self.input_size
        )
        input_tensor = input_tensor.to(self.device)

        # Prepare original size tensor [W, H]
        orig_size_tensor = torch.tensor(
            [[original_size[0], original_size[1]]],
            dtype=torch.float32,
            device=self.device
        )

        # Run inference
        with torch.no_grad():
            labels, boxes, scores = self.model(input_tensor, orig_size_tensor)

        # Filter by confidence threshold
        scores_np = scores[0].cpu()
        mask = scores_np > conf_thres

        filtered_boxes = boxes[0][mask].cpu().tolist()
        filtered_scores = scores_np[mask].tolist()
        filtered_classes = labels[0][mask].cpu().tolist()

        # Build result dict
        result = {
            "boxes": filtered_boxes,
            "scores": filtered_scores,
            "classes": [int(c) for c in filtered_classes],
            "num_detections": len(filtered_boxes),
        }

        # Add source info
        if isinstance(image, (str, Path)):
            result["source"] = str(image)

        # Save annotated image if requested
        if save:
            annotated = draw_boxes(
                original_img,
                filtered_boxes,
                filtered_scores,
                filtered_classes,
                threshold=0.0
            )

            if output_path is None:
                save_dir = Path("runs/rtdetr_detections")
                save_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                if isinstance(image, (str, Path)):
                    stem = Path(image).stem
                else:
                    stem = "image"
                ext = output_file_format or "jpg"
                save_path = save_dir / f"{stem}_{timestamp}.{ext}"
            else:
                save_path = Path(output_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)

            annotated.save(str(save_path))
            result["saved_path"] = str(save_path)

        return result

    def _predict_tiled(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.5,
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None
    ) -> Dict:
        """Run tiled inference for large images."""
        from ..common.utils import get_slice_bboxes

        # Load original image
        original_img = ImageLoader.load(image)
        orig_w, orig_h = original_img.size

        # Get tile coordinates
        slices = get_slice_bboxes(orig_w, orig_h, overlap_ratio=overlap_ratio)

        all_boxes = []
        all_scores = []
        all_classes = []

        for x1, y1, x2, y2 in slices:
            # Crop tile
            tile = original_img.crop((x1, y1, x2, y2))

            # Run inference on tile
            tile_result = self._predict_single(tile, save=False, conf_thres=conf_thres)

            # Shift boxes back to original coordinates
            for box, score, cls in zip(
                tile_result["boxes"],
                tile_result["scores"],
                tile_result["classes"]
            ):
                shifted_box = [
                    box[0] + x1,
                    box[1] + y1,
                    box[2] + x1,
                    box[3] + y1
                ]
                all_boxes.append(shifted_box)
                all_scores.append(score)
                all_classes.append(cls)

        # Merge overlapping detections with NMS
        if all_boxes:
            merged_boxes, merged_scores, merged_classes = self._merge_tile_detections(
                all_boxes, all_scores, all_classes, iou_thres=0.5
            )
        else:
            merged_boxes, merged_scores, merged_classes = [], [], []

        result = {
            "boxes": merged_boxes,
            "scores": merged_scores,
            "classes": merged_classes,
            "num_detections": len(merged_boxes),
            "tiled": True,
            "num_tiles": len(slices)
        }

        if isinstance(image, (str, Path)):
            result["source"] = str(image)

        # Save if requested
        if save:
            annotated = draw_boxes(
                original_img,
                merged_boxes,
                merged_scores,
                merged_classes,
                threshold=0.0
            )

            if output_path is None:
                save_dir = Path("runs/rtdetr_detections")
                save_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                if isinstance(image, (str, Path)):
                    stem = Path(image).stem
                else:
                    stem = "image"
                ext = output_file_format or "jpg"
                save_path = save_dir / f"{stem}_tiled_{timestamp}.{ext}"
            else:
                save_path = Path(output_path)

            annotated.save(str(save_path))
            result["saved_path"] = str(save_path)

        return result

    def _merge_tile_detections(
        self,
        boxes: List,
        scores: List,
        classes: List,
        iou_thres: float = 0.5
    ) -> tuple:
        """Merge detections from tiles using class-wise NMS."""
        import torchvision

        if not boxes:
            return [], [], []

        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        scores_t = torch.tensor(scores, dtype=torch.float32)
        classes_t = torch.tensor(classes, dtype=torch.int64)

        # Class-wise NMS
        keep_indices = torchvision.ops.batched_nms(
            boxes_t, scores_t, classes_t, iou_thres
        )

        return (
            boxes_t[keep_indices].tolist(),
            scores_t[keep_indices].tolist(),
            classes_t[keep_indices].tolist()
        )

    def _process_in_batches(
        self,
        image_paths: List[Path],
        batch_size: int,
        save: bool,
        output_path: str,
        conf_thres: float,
        tiling: bool,
        overlap_ratio: float,
        output_file_format: Optional[str]
    ) -> List[Dict]:
        """Process multiple images."""
        results = []
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i:i + batch_size]
            for path in chunk:
                if tiling:
                    results.append(self._predict_tiled(
                        path, save, output_path, conf_thres, overlap_ratio, output_file_format
                    ))
                else:
                    results.append(self._predict_single(
                        path, save, output_path, conf_thres, output_file_format
                    ))
        return results

    def predict(self, *args, **kwargs) -> Union[Dict, List[Dict]]:
        """Alias for __call__."""
        return self(*args, **kwargs)

    @staticmethod
    def get_available_sizes() -> List[str]:
        """
        Get list of available model sizes.

        Returns:
            List of size codes matching official RT-DETRv2 naming:
            - 's':  RT-DETRv2-S  (r18vd,  48.1 AP)
            - 'ms': RT-DETRv2-M* (r34vd,  49.9 AP)
            - 'm':  RT-DETRv2-M  (r50vd_m, 51.9 AP)
            - 'l':  RT-DETRv2-L  (r50vd,  53.4 AP)
            - 'x':  RT-DETRv2-X  (r101vd, 54.3 AP)
        """
        return ['s', 'ms', 'm', 'l', 'x']

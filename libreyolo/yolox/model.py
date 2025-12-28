"""
LIBREYOLOX implementation for LibreYOLO.
"""

from datetime import datetime
from typing import Union, List, Tuple
from pathlib import Path
import torch
from PIL import Image

from .nn import YOLOXModel
from .utils import preprocess_image, postprocess, nms
from ..common.utils import draw_boxes, get_safe_stem, get_slice_bboxes
from ..common.image_loader import ImageInput, ImageLoader


class LIBREYOLOX:
    """
    YOLOX model for object detection.

    Args:
        model_path: Path to model weights file or loaded state dict (required)
        size: Model size variant (required). Must be one of: "nano", "tiny", "s", "m", "l", "x"
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference. "auto" (default) uses CUDA if available, else MPS, else CPU.
                Can also specify directly: "cuda", "cuda:0", "mps", "cpu".
        tiling: Enable tiling for processing large/high-resolution images (default: False).
                When enabled, large images are automatically split into overlapping tiles,
                inference is run on each tile, and results are merged using NMS.

    Example:
        >>> model = LIBREYOLOX(model_path="yolox_s.pt", size="s")
        >>> detections = model(image="image.jpg", save=True)
    """

    # Default input sizes for different model variants
    DEFAULT_INPUT_SIZES = {
        'nano': 416,
        'tiny': 416,
        's': 640,
        'm': 640,
        'l': 640,
        'x': 640,
    }

    def __init__(
        self,
        model_path: Union[str, dict],
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        tiling: bool = False
    ):
        """
        Initialize the LIBREYOLOX model.

        Args:
            model_path: Path to model weights file or pre-loaded state dict
            size: Model size variant. Must be "nano", "tiny", "s", "m", "l", or "x"
            nb_classes: Number of classes (default: 80)
            device: Device for inference ("auto", "cuda", "mps", "cpu")
            tiling: Enable tiling for large images (default: False)
        """
        self.tiling = tiling

        if size not in ['nano', 'tiny', 's', 'm', 'l', 'x']:
            raise ValueError(f"Invalid size: {size}. Must be one of: 'nano', 'tiny', 's', 'm', 'l', 'x'")

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
        self.input_size = self.DEFAULT_INPUT_SIZES[size]

        # Initialize model
        self.model = YOLOXModel(config=size, nb_classes=nb_classes)

        # Load weights
        if isinstance(model_path, dict):
            self.model_path = None
            self.model.load_state_dict(model_path, strict=True)
        else:
            self.model_path = model_path
            self._load_weights(model_path)

        # Set to evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)

    def _load_weights(self, model_path: str):
        """Load model weights from file."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {model_path}: {e}") from e

    def __call__(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        color_format: str = "auto",
        batch_size: int = 1
    ) -> Union[dict, List[dict]]:
        """
        Run inference on an image or directory of images.

        Args:
            image: Input image or directory. Supported types:
                - str: Local file path, directory path, or URL (http/https/s3/gs)
                - pathlib.Path: Local file path or directory path
                - PIL.Image: PIL Image object
                - np.ndarray: NumPy array (HWC or CHW, RGB or BGR)
                - torch.Tensor: PyTorch tensor (CHW or NCHW)
                - bytes: Raw image bytes
                - io.BytesIO: BytesIO object containing image data
            save: If True, saves the image with detections drawn. Defaults to False.
            output_path: Optional path to save the annotated image.
            conf_thres: Confidence threshold (default: 0.25)
            iou_thres: IoU threshold for NMS (default: 0.45)
            color_format: Color format hint for NumPy/OpenCV arrays ("auto", "rgb", "bgr")
            batch_size: Number of images to process per batch (for directories)

        Returns:
            For single image: Dictionary containing detection results with keys:
                - boxes: List of bounding boxes in xyxy format
                - scores: List of confidence scores
                - classes: List of class IDs
                - num_detections: Number of detections
                - source: Source image path (if available)
                - saved_path: Path to saved image (if save=True)

            For directory: List of dictionaries, one per image processed.
        """
        # Check if input is a directory
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
                color_format=color_format
            )

        # Use tiled inference for large images when tiling is enabled
        if self.tiling:
            return self._predict_tiled(image, save, output_path, conf_thres, iou_thres, color_format)

        return self._predict_single(image, save, output_path, conf_thres, iou_thres, color_format)

    def _process_in_batches(
        self,
        image_paths: List[Path],
        batch_size: int = 1,
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        color_format: str = "auto"
    ) -> List[dict]:
        """Process multiple images in batches."""
        results = []
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i:i + batch_size]
            for path in chunk:
                results.append(
                    self._predict_single(path, save, output_path, conf_thres, iou_thres, color_format)
                )
        return results

    def _predict_single(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        color_format: str = "auto"
    ) -> dict:
        """Run inference on a single image."""
        # Store original image path for saving
        image_path = image if isinstance(image, (str, Path)) else None

        # Preprocess image (YOLOX-specific: letterbox, no normalization)
        input_tensor, original_img, original_size, ratio = preprocess_image(
            image,
            input_size=self.input_size,
            color_format=color_format
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor.to(self.device))

        # Postprocess
        detections = postprocess(
            outputs,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=self.input_size,
            original_size=original_size,
            ratio=ratio
        )

        # Add source path for traceability
        detections["source"] = str(image_path) if image_path else None

        # Draw and save if requested
        if save:
            if detections["num_detections"] > 0:
                annotated_img = draw_boxes(
                    original_img,
                    detections["boxes"],
                    detections["scores"],
                    detections["classes"]
                )
            else:
                annotated_img = original_img

            if output_path:
                final_output_path = Path(output_path)
                if final_output_path.suffix == "":
                    final_output_path.mkdir(parents=True, exist_ok=True)
                    if isinstance(image_path, (str, Path)):
                        stem = get_safe_stem(image_path)
                        ext = Path(image_path).suffix
                    else:
                        stem = "inference"
                        ext = ".jpg"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    final_output_path = final_output_path / f"{stem}_{timestamp}{ext}"
                else:
                    final_output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                if isinstance(image_path, (str, Path)):
                    stem = get_safe_stem(image_path)
                    ext = Path(image_path).suffix
                else:
                    stem = "inference"
                    ext = ".jpg"

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = Path("runs/detections")
                save_dir.mkdir(parents=True, exist_ok=True)
                final_output_path = save_dir / f"{stem}_{timestamp}{ext}"

            annotated_img.save(final_output_path)
            detections["saved_path"] = str(final_output_path)

        return detections

    def _merge_tile_detections(
        self,
        boxes: List,
        scores: List,
        classes: List,
        iou_thres: float
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
        color_format: str = "auto"
    ) -> dict:
        """Run tiled inference on large images."""
        # Load full image
        img_pil = ImageLoader.load(image, color_format=color_format)
        orig_width, orig_height = img_pil.size
        image_path = image if isinstance(image, (str, Path)) else None

        # Skip tiling if image is already small enough
        if orig_width <= self.input_size and orig_height <= self.input_size:
            return self._predict_single(image, save, output_path, conf_thres, iou_thres, color_format)

        # Get tile coordinates
        slices = get_slice_bboxes(orig_width, orig_height, slice_size=self.input_size)

        # Collect all detections from tiles
        all_boxes, all_scores, all_classes = [], [], []

        for x1, y1, x2, y2 in slices:
            # Crop tile from image
            tile = img_pil.crop((x1, y1, x2, y2))

            # Run inference on tile (without saving)
            result = self._predict_single(tile, save=False, conf_thres=conf_thres, iou_thres=iou_thres)

            # Shift boxes back to original image coordinates
            for box in result["boxes"]:
                shifted_box = [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1]
                all_boxes.append(shifted_box)
            all_scores.extend(result["scores"])
            all_classes.extend(result["classes"])

        # Merge detections from all tiles using class-wise NMS
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
            "num_tiles": len(slices)
        }

        # Draw and save if requested
        if save:
            if detections["num_detections"] > 0:
                annotated_img = draw_boxes(
                    img_pil,
                    detections["boxes"],
                    detections["scores"],
                    detections["classes"]
                )
            else:
                annotated_img = img_pil

            if output_path:
                final_output_path = Path(output_path)
                if final_output_path.suffix == "":
                    final_output_path.mkdir(parents=True, exist_ok=True)
                    if isinstance(image_path, (str, Path)):
                        stem = get_safe_stem(image_path)
                        ext = Path(image_path).suffix
                    else:
                        stem = "inference"
                        ext = ".jpg"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    final_output_path = final_output_path / f"{stem}_{timestamp}{ext}"
                else:
                    final_output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                if isinstance(image_path, (str, Path)):
                    stem = get_safe_stem(image_path)
                    ext = Path(image_path).suffix
                else:
                    stem = "inference"
                    ext = ".jpg"

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = Path("runs/detections")
                save_dir.mkdir(parents=True, exist_ok=True)
                final_output_path = save_dir / f"{stem}_{timestamp}{ext}"

            annotated_img.save(final_output_path)
            detections["saved_path"] = str(final_output_path)

        return detections

    def predict(
        self,
        image: ImageInput,
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        color_format: str = "auto",
        batch_size: int = 1
    ) -> Union[dict, List[dict]]:
        """
        Alias for __call__ method.

        Args:
            image: Input image or directory
            save: If True, saves the image with detections drawn
            output_path: Optional path to save the annotated image
            conf_thres: Confidence threshold (default: 0.25)
            iou_thres: IoU threshold for NMS (default: 0.45)
            color_format: Color format hint ("auto", "rgb", "bgr")
            batch_size: Number of images per batch

        Returns:
            For single image: Dictionary with detection results
            For directory: List of dictionaries
        """
        return self(
            image=image,
            save=save,
            output_path=output_path,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            color_format=color_format,
            batch_size=batch_size
        )

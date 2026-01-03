"""
LIBREYOLOX implementation for LibreYOLO.

Supports both inference and training.
"""

import json
from datetime import datetime
from typing import Union, List, Tuple, Optional, Dict
from pathlib import Path
import torch
from PIL import Image

from .nn import YOLOXModel
from .utils import preprocess_image, postprocess, nms, decode_outputs, cxcywh_to_xyxy
from ..common.utils import draw_boxes, get_safe_stem, get_slice_bboxes, draw_tile_grid
from ..common.image_loader import ImageInput, ImageLoader
from ..common.postprocessing import get_postprocessor, list_postprocessors, POSTPROCESSORS


class LIBREYOLOX:
    """
    YOLOX model for object detection.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
        size: Model size variant (required). Must be one of: "nano", "tiny", "s", "m", "l", "x"
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference. "auto" (default) uses CUDA if available, else MPS, else CPU.
                Can also specify directly: "cuda", "cuda:0", "mps", "cpu".

    Example:
        >>> model = LIBREYOLOX(model_path="libreyoloXs.pt", size="s")
        >>> detections = model(image="image.jpg", save=True)
        >>> # Use tiling for large images
        >>> detections = model(image=large_image_path, save=True, tiling=True)
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
        device: str = "auto"
    ):
        """
        Initialize the LIBREYOLOX model.

        Args:
            model_path: Model weights source. Can be:
                - str: Path to a .pt/.pth weights file
                - dict: Pre-loaded state_dict (e.g., from torch.load())
            size: Model size variant. Must be "nano", "tiny", "s", "m", "l", or "x"
            nb_classes: Number of classes (default: 80)
            device: Device for inference ("auto", "cuda", "mps", "cpu")
        """

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
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    # Official YOLOX checkpoint format
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # Assume it's a direct state dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

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
        batch_size: int = 1,
        tiling: bool = False,
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        postprocessor: Optional[str] = None,
        postprocessor_kwargs: Optional[dict] = None
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
            tiling: Enable tiling for processing large/high-resolution images (default: False).
                When enabled, large images are automatically split into overlapping tiles,
                inference is run on each tile, and results are merged using NMS.
            overlap_ratio: Fractional overlap between tiles when tiling is enabled (default: 0.2).
                Higher values reduce edge artifacts but increase processing time.
            output_file_format: Output image format when saving. Options: "jpg", "png", "webp".
                Defaults to "jpg" for maximum compatibility.
            postprocessor: Post-processing method to use. Options:
                - "nms": Standard NMS (default)
                - "soft_nms": Soft-NMS with score decay
                - "diou_nms": Distance-IoU based NMS
                - "ciou_nms": Complete-IoU based NMS
                Use get_available_postprocessors() to see all options.
            postprocessor_kwargs: Additional arguments for the post-processor.
                Examples:
                - For soft_nms: {"sigma": 0.5, "method": "gaussian"}
                - For any: {"agnostic": True} for class-agnostic NMS

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
        # Validate output_file_format
        if output_file_format is not None:
            output_file_format = output_file_format.lower().lstrip('.')
            if output_file_format not in ('jpg', 'jpeg', 'png', 'webp'):
                raise ValueError(f"Invalid output_file_format: {output_file_format}. Must be one of: 'jpg', 'png', 'webp'")

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
                color_format=color_format,
                tiling=tiling,
                overlap_ratio=overlap_ratio,
                output_file_format=output_file_format,
                postprocessor=postprocessor,
                postprocessor_kwargs=postprocessor_kwargs
            )

        # Use tiled inference for large images when tiling is enabled
        if tiling:
            return self._predict_tiled(image, save, output_path, conf_thres, iou_thres, color_format, overlap_ratio, output_file_format, postprocessor, postprocessor_kwargs)

        return self._predict_single(image, save, output_path, conf_thres, iou_thres, color_format, output_file_format, postprocessor, postprocessor_kwargs)

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
        postprocessor: Optional[str] = None,
        postprocessor_kwargs: Optional[dict] = None
    ) -> List[dict]:
        """Process multiple images in batches."""
        results = []
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i:i + batch_size]
            for path in chunk:
                if tiling:
                    results.append(
                        self._predict_tiled(path, save, output_path, conf_thres, iou_thres, color_format, overlap_ratio, output_file_format, postprocessor, postprocessor_kwargs)
                    )
                else:
                    results.append(
                        self._predict_single(path, save, output_path, conf_thres, iou_thres, color_format, output_file_format, postprocessor, postprocessor_kwargs)
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
        postprocessor: Optional[str] = None,
        postprocessor_kwargs: Optional[dict] = None
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

        # Postprocess - use new pluggable system if postprocessor is specified
        if postprocessor is not None:
            detections = self._postprocess_with_processor(
                outputs=outputs,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                original_size=original_size,
                ratio=ratio,
                postprocessor=postprocessor,
                postprocessor_kwargs=postprocessor_kwargs
            )
        else:
            # Use legacy postprocess for backwards compatibility
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

            # Determine output extension (default to jpg for compatibility)
            if output_file_format:
                ext = f".{output_file_format}" if not output_file_format.startswith('.') else output_file_format
            else:
                ext = ".jpg"  # Default to jpg for maximum compatibility

            if output_path:
                final_output_path = Path(output_path)
                if final_output_path.suffix == "":
                    final_output_path.mkdir(parents=True, exist_ok=True)
                    if isinstance(image_path, (str, Path)):
                        stem = get_safe_stem(image_path)
                    else:
                        stem = "inference"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    final_output_path = final_output_path / f"{stem}_{timestamp}{ext}"
                else:
                    final_output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                if isinstance(image_path, (str, Path)):
                    stem = get_safe_stem(image_path)
                else:
                    stem = "inference"

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = Path("runs/detections")
                save_dir.mkdir(parents=True, exist_ok=True)
                final_output_path = save_dir / f"{stem}_{timestamp}{ext}"

            annotated_img.save(final_output_path)
            detections["saved_path"] = str(final_output_path)

        return detections

    def _postprocess_with_processor(
        self,
        outputs: List[torch.Tensor],
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        ratio: float,
        postprocessor: str,
        postprocessor_kwargs: Optional[dict] = None
    ) -> dict:
        """
        Apply postprocessing using the pluggable postprocessor system.

        This decodes the raw YOLOX model output and applies the specified post-processor.
        YOLOX uses center/width/height format with objectness scores.

        Args:
            outputs: List of raw model output tensors from each scale
            conf_thres: Confidence threshold
            iou_thres: IoU threshold
            original_size: Original image size (width, height)
            ratio: Scale ratio from preprocessing
            postprocessor: Name of the post-processor to use
            postprocessor_kwargs: Additional kwargs for the post-processor

        Returns:
            Detection dictionary with boxes, scores, classes, num_detections
        """
        from ..common.postprocessing import get_postprocessor, filter_valid_boxes

        # Decode outputs to absolute coordinates (YOLOX format: cx, cy, w, h, obj, class_probs)
        decoded = decode_outputs(outputs)  # (B, N, 5+num_classes)

        # Take first batch
        decoded = decoded[0]  # (N, 5+num_classes)

        # Extract components
        boxes_cxcywh = decoded[:, :4]  # (N, 4) - center_x, center_y, width, height
        objectness = decoded[:, 4]     # (N,) - objectness score
        class_probs = decoded[:, 5:]   # (N, num_classes) - class probabilities

        # Final confidence = objectness * class_prob
        scores = objectness.unsqueeze(-1) * class_probs  # (N, num_classes)

        # Get max class score and class id for each prediction
        max_scores, class_ids = torch.max(scores, dim=1)  # (N,)

        # Filter by confidence threshold
        mask = max_scores > conf_thres
        if not mask.any():
            return {
                "boxes": [],
                "scores": [],
                "classes": [],
                "num_detections": 0
            }

        valid_boxes_cxcywh = boxes_cxcywh[mask]
        valid_scores = max_scores[mask]
        valid_classes = class_ids[mask]

        # Convert from cxcywh to xyxy format
        valid_boxes = cxcywh_to_xyxy(valid_boxes_cxcywh)

        # Scale boxes back to original image coordinates
        if ratio != 1.0:
            valid_boxes = valid_boxes / ratio

        # Clamp to image boundaries
        valid_boxes[:, [0, 2]] = torch.clamp(valid_boxes[:, [0, 2]], 0, original_size[0])
        valid_boxes[:, [1, 3]] = torch.clamp(valid_boxes[:, [1, 3]], 0, original_size[1])

        # Filter out invalid boxes
        valid_boxes, valid_scores, valid_classes = filter_valid_boxes(
            valid_boxes, valid_scores, valid_classes
        )

        if len(valid_boxes) == 0:
            return {
                "boxes": [],
                "scores": [],
                "classes": [],
                "num_detections": 0
            }

        # Get the post-processor
        pp_kwargs = postprocessor_kwargs or {}
        processor = get_postprocessor(
            postprocessor,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            **pp_kwargs
        )

        # Apply post-processing
        final_boxes, final_scores, final_classes = processor(
            valid_boxes, valid_scores, valid_classes
        )

        return {
            "boxes": final_boxes.cpu().tolist(),
            "scores": final_scores.cpu().tolist(),
            "classes": final_classes.cpu().tolist(),
            "num_detections": len(final_boxes)
        }

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
        color_format: str = "auto",
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        postprocessor: Optional[str] = None,
        postprocessor_kwargs: Optional[dict] = None
    ) -> dict:
        """
        Run tiled inference on large images.

        Splits the image into overlapping tiles, runs inference on each,
        shifts detections back to original coordinates, and merges with NMS.

        Args:
            image: Input image (path, PIL Image, numpy array, etc.)
            save: If True, saves the annotated image, individual tiles, and grid visualization.
            output_path: Optional path to save the outputs. When provided as a directory,
                outputs are saved there; otherwise a timestamped folder is created.
            conf_thres: Confidence threshold.
            iou_thres: IoU threshold for NMS.
            color_format: Color format hint for numpy arrays.
            overlap_ratio: Fractional overlap between tiles (default: 0.2).
            output_file_format: Output image format when saving.
            postprocessor: Post-processing method name.
            postprocessor_kwargs: Additional post-processor arguments.

        Returns:
            Dictionary with detection results including tiling metadata.
        """
        # Load full image
        img_pil = ImageLoader.load(image, color_format=color_format)
        orig_width, orig_height = img_pil.size
        image_path = image if isinstance(image, (str, Path)) else None

        # Skip tiling if image is already small enough
        if orig_width <= self.input_size and orig_height <= self.input_size:
            return self._predict_single(image, save, output_path, conf_thres, iou_thres, color_format, output_file_format, postprocessor, postprocessor_kwargs)

        # Get tile coordinates
        slices = get_slice_bboxes(orig_width, orig_height, slice_size=self.input_size, overlap_ratio=overlap_ratio)

        # Collect all detections from tiles
        all_boxes, all_scores, all_classes = [], [], []
        tiles_data = []  # Store tiles for saving

        for idx, (x1, y1, x2, y2) in enumerate(slices):
            # Crop tile from image
            tile = img_pil.crop((x1, y1, x2, y2))

            # Store tile data for saving if needed
            if save:
                tiles_data.append({
                    "index": idx,
                    "coords": (x1, y1, x2, y2),
                    "image": tile.copy()
                })

            # Run inference on tile (without saving)
            result = self._predict_single(tile, save=False, conf_thres=conf_thres, iou_thres=iou_thres, postprocessor=postprocessor, postprocessor_kwargs=postprocessor_kwargs)

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
                annotated_img = img_pil.copy()

            # Determine output extension (default to jpg for compatibility)
            if output_file_format:
                ext = f".{output_file_format}" if not output_file_format.startswith('.') else output_file_format
            else:
                ext = ".jpg"  # Default to jpg for maximum compatibility

            # Determine save directory
            if isinstance(image_path, (str, Path)):
                stem = get_safe_stem(image_path)
            else:
                stem = "inference"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if output_path:
                base_output_path = Path(output_path)
                if base_output_path.suffix == "":
                    # It's a directory
                    save_dir = base_output_path / f"{stem}_{timestamp}"
                else:
                    # It's a file path, use parent directory
                    save_dir = base_output_path.parent / f"{stem}_{timestamp}"
            else:
                save_dir = Path("runs/tiled_detections") / f"{stem}_{timestamp}"

            save_dir.mkdir(parents=True, exist_ok=True)

            # Create tiles subdirectory and save individual tiles
            tiles_dir = save_dir / "tiles"
            tiles_dir.mkdir(parents=True, exist_ok=True)
            tile_paths = []
            for tile_data in tiles_data:
                tile_filename = f"tile_{tile_data['index']:03d}{ext}"
                tile_path = tiles_dir / tile_filename
                tile_data["image"].save(tile_path)
                tile_paths.append(str(tile_path))

            # Save final annotated image
            final_image_path = save_dir / f"final_image{ext}"
            annotated_img.save(final_image_path)

            # Save grid visualization
            grid_img = draw_tile_grid(img_pil, slices)
            grid_path = save_dir / f"grid_visualization{ext}"
            grid_img.save(grid_path)

            # Save metadata
            metadata = {
                "model": "LIBREYOLOX",
                "size": self.size,
                "image_source": str(image_path) if image_path else "PIL/numpy input",
                "original_size": [orig_width, orig_height],
                "num_tiles": len(slices),
                "tile_size": self.input_size,
                "overlap_ratio": overlap_ratio,
                "tiles": [
                    {
                        "index": td["index"],
                        "coords": {"x1": td["coords"][0], "y1": td["coords"][1], "x2": td["coords"][2], "y2": td["coords"][3]},
                        "filename": f"tile_{td['index']:03d}{ext}"
                    }
                    for td in tiles_data
                ],
                "num_detections": detections["num_detections"],
                "conf_thres": conf_thres,
                "iou_thres": iou_thres
            }
            with open(save_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            detections["saved_path"] = str(save_dir)
            detections["tiles_path"] = str(tiles_dir)
            detections["grid_path"] = str(grid_path)

        return detections

    def predict(
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
        output_file_format: Optional[str] = None
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
            tiling: Enable tiling for processing large/high-resolution images (default: False)
            overlap_ratio: Fractional overlap between tiles (default: 0.2)
            output_file_format: Output image format when saving. Options: "jpg", "png", "webp".

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
            batch_size=batch_size,
            tiling=tiling,
            overlap_ratio=overlap_ratio,
            output_file_format=output_file_format
        )

    @classmethod
    def new(
        cls,
        size: str = "s",
        num_classes: int = 80,
        device: str = "auto"
    ) -> "LIBREYOLOX":
        """
        Create a new untrained YOLOX model.

        This is useful for training from scratch.

        Args:
            size: Model size variant ("nano", "tiny", "s", "m", "l", "x")
            num_classes: Number of classes (default: 80)
            device: Device for the model ("auto", "cuda", "mps", "cpu")

        Returns:
            LIBREYOLOX instance with randomly initialized weights

        Example:
            >>> model = LIBREYOLOX.new(size="s", num_classes=80)
            >>> model.train(data="coco.yaml", epochs=300)
        """
        if size not in ['nano', 'tiny', 's', 'm', 'l', 'x']:
            raise ValueError(f"Invalid size: {size}. Must be one of: 'nano', 'tiny', 's', 'm', 'l', 'x'")

        # Create instance without calling __init__
        instance = cls.__new__(cls)

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                instance.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                instance.device = torch.device("mps")
            else:
                instance.device = torch.device("cpu")
        else:
            instance.device = torch.device(device)

        instance.size = size
        instance.nb_classes = num_classes
        instance.input_size = cls.DEFAULT_INPUT_SIZES[size]
        instance.model_path = None

        # Create model with random initialization
        instance.model = YOLOXModel(config=size, nb_classes=num_classes)
        instance.model.to(instance.device)
        instance.model.train()  # Set to training mode for new models

        return instance

    def train(
        self,
        data: Optional[str] = None,
        epochs: int = 300,
        batch_size: int = 16,
        imgsz: int = None,
        config: Optional[Union[str, "YOLOXTrainConfig"]] = None,
        resume: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Train the model.

        Args:
            data: Path to data.yaml configuration file
            epochs: Number of training epochs (default: 300)
            batch_size: Batch size (default: 16)
            imgsz: Image size (default: model's default size)
            config: Path to config.yaml or YOLOXTrainConfig object
            resume: Path to checkpoint to resume training from
            **kwargs: Additional training parameters (see YOLOXTrainConfig)

        Returns:
            Dictionary with training results

        Example:
            >>> # Train from scratch
            >>> model = LIBREYOLOX.new(size="s", num_classes=80)
            >>> model.train(data="coco.yaml", epochs=300)

            >>> # Fine-tune pretrained model
            >>> model = LIBREYOLOX("libreyoloXs.pt", size="s")
            >>> model.train(data="custom.yaml", epochs=100)

            >>> # With config file
            >>> model.train(config="training_config.yaml")
        """
        from ..training import YOLOXTrainConfig, YOLOXTrainer

        # Build configuration
        if isinstance(config, str):
            cfg = YOLOXTrainConfig.from_yaml(config)
        elif config is not None:
            cfg = config
        else:
            cfg = YOLOXTrainConfig(
                size=self.size,
                num_classes=self.nb_classes,
            )

        # Override with explicit parameters
        if data is not None:
            cfg = cfg.update(data=data)
        if epochs != 300:
            cfg = cfg.update(epochs=epochs)
        if batch_size != 16:
            cfg = cfg.update(batch_size=batch_size)
        if imgsz is not None:
            cfg = cfg.update(imgsz=imgsz)
        elif imgsz is None:
            cfg = cfg.update(imgsz=self.input_size)

        # Apply any additional kwargs
        if kwargs:
            cfg = cfg.update(**kwargs)

        # Create trainer
        trainer = YOLOXTrainer(self.model, cfg)

        # Resume from checkpoint if specified
        if resume:
            trainer.resume(resume)

        # Run training
        results = trainer.train()

        # Update model to use best weights
        best_checkpoint = Path(results["save_dir"]) / "best.pt"
        if best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])

        return results

    def export(
        self,
        format: str = "onnx",
        output_path: Optional[str] = None,
        opset: int = 11,
        simplify: bool = True,
        dynamic: bool = False,
    ) -> str:
        """
        Export the model to a different format.

        Args:
            format: Export format ("onnx", "torchscript")
            output_path: Output file path (auto-generated if None)
            opset: ONNX opset version (default: 11)
            simplify: Simplify ONNX model (default: True)
            dynamic: Use dynamic input shapes (default: False)

        Returns:
            Path to the exported model file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"yolox_{self.size}_{timestamp}.{format}"

        self.model.eval()

        if format == "onnx":
            return self._export_onnx(output_path, opset, simplify, dynamic)
        elif format == "torchscript":
            return self._export_torchscript(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_onnx(
        self,
        output_path: str,
        opset: int = 11,
        simplify: bool = True,
        dynamic: bool = False,
    ) -> str:
        """Export to ONNX format."""
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)

        input_names = ["images"]
        output_names = ["outputs"]

        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                "images": {0: "batch", 2: "height", 3: "width"},
                "outputs": {0: "batch"},
            }

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamic_axes=dynamic_axes,
        )

        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                model = onnx.load(output_path)
                model_simplified, check = onnx_simplify(model)
                if check:
                    onnx.save(model_simplified, output_path)
            except ImportError:
                pass  # onnxsim not available

        return output_path

    def _export_torchscript(self, output_path: str) -> str:
        """Export to TorchScript format."""
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)

        traced_model = torch.jit.trace(self.model, dummy_input)
        traced_model.save(output_path)

        return output_path

    @staticmethod
    def get_available_postprocessors() -> dict:
        """
        Get available post-processing methods with their descriptions.

        Returns:
            Dictionary mapping post-processor names to metadata:
                - description: Human-readable description
                - supports_batched: Whether batched processing is supported

        Example:
            >>> LIBREYOLOX.get_available_postprocessors()
            {'nms': {'description': 'Standard NMS...', 'supports_batched': False}, ...}
        """
        return list_postprocessors()

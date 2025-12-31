"""
Libre YOLO8 implementation.
"""

import io
import json
from datetime import datetime
from typing import Union, List, Optional, Tuple
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from .nn import LibreYOLO8Model
from .utils import preprocess_image, postprocess, draw_boxes, make_anchors, decode_boxes, nms
from ..common.eigen_cam import compute_eigen_cam, overlay_heatmap
from ..common.cam import CAM_METHODS
from ..common.image_loader import ImageInput, ImageLoader
from ..common.utils import get_safe_stem, get_slice_bboxes, draw_tile_grid


class LIBREYOLO8:
    """
    Libre YOLO8 model for object detection.

    Args:
        model_path: Path to model weights file (required)
        size: Model size variant (required). Must be one of: "n", "s", "m", "l", "x"
        reg_max: Regression max value for DFL (default: 16)
        nb_classes: Number of classes (default: 80 for COCO)
        save_feature_maps: Feature map saving mode. Options:
            - False: Disabled (default)
            - True: Save all layers
            - List of layer names: Save only specified layers (e.g., ["backbone_p1", "neck_c2f21"])
        save_eigen_cam: If True, saves EigenCAM heatmap visualizations on each inference (default: False)
        cam_method: CAM method for explain(). Options: "eigencam", "gradcam", "gradcam++",
                   "xgradcam", "hirescam", "layercam", "eigengradcam" (default: "eigencam")
        cam_layer: Target layer for CAM computation (default: "neck_c2f22")
        device: Device for inference. "auto" (default) uses CUDA if available, else MPS, else CPU.
                Can also specify directly: "cuda", "cuda:0", "mps", "cpu".

    Example:
        >>> model = LIBREYOLO8(model_path="path/to/weights.pt", size="x", save_feature_maps=True)
        >>> detections = model(image=image_path, save=True)
        >>> # Use tiling for large images
        >>> detections = model(image=large_image_path, save=True, tiling=True)
        >>> # Use explain() for XAI heatmaps
        >>> heatmap = model.explain("image.jpg", method="gradcam")
    """
    
    def __init__(
        self,
        model_path: Union[str, dict],
        size: str,
        reg_max: int = 16,
        nb_classes: int = 80,
        save_feature_maps: Union[bool, List[str]] = False,
        save_eigen_cam: bool = False,
        cam_method: str = "eigencam",
        cam_layer: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize the Libre YOLO8 model.

        Args:
            model_path: Path to user-provided model weights file or loaded state dict
            size: Model size variant. Must be "n", "s", "m", "l", or "x"
            reg_max: Regression max value for DFL (default: 16)
            nb_classes: Number of classes (default: 80)
            save_feature_maps: Feature map saving mode. Options:
                - False: Disabled
                - True: Save all layers
                - List[str]: Save only specified layer names
            save_eigen_cam: If True, saves EigenCAM heatmap visualizations
            cam_method: Default CAM method for explain() (default: "eigencam")
            cam_layer: Target layer for CAM computation (default: "neck_c2f22")
            device: Device for inference ("auto", "cuda", "mps", "cpu")
        """
        
        if size not in ['n', 's', 'm', 'l', 'x']:
            raise ValueError(f"Invalid size: {size}. Must be one of: 'n', 's', 'm', 'l', 'x'")
        
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
        self.reg_max = reg_max
        self.nb_classes = nb_classes
        self.save_feature_maps = save_feature_maps
        self.save_eigen_cam = save_eigen_cam
        self.cam_method = cam_method.lower()
        self.feature_maps = {}
        self.hooks = []
        self._eigen_cam_layer = cam_layer or "neck_c2f22"  # Default layer for EigenCAM/CAM
        
        # Initialize model
        self.model = LibreYOLO8Model(config=size, reg_max=reg_max, nb_classes=nb_classes)
        
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
        
        # Register hooks for feature map extraction
        if self.save_feature_maps or self.save_eigen_cam:
            self._register_hooks()
    
    def _load_weights(self, model_path: str):
        """Load model weights from file."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")
        
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {model_path}: {e}") from e
    
    def get_available_layer_names(self) -> List[str]:
        """
        Get list of available layer names for feature map saving.
        
        Returns:
            List of layer names that can be used with save_feature_maps parameter.
        """
        return sorted(self._get_available_layers().keys())
    
    def _get_available_layers(self) -> dict:
        """Get mapping of layer names to module objects."""
        return {
            # Backbone layers
            "backbone_p1": self.model.backbone.p1,
            "backbone_p2": self.model.backbone.p2,
            "backbone_c2f1": self.model.backbone.c2f1,
            "backbone_p3": self.model.backbone.p3,
            "backbone_c2f2_P3": self.model.backbone.c2f2,
            "backbone_p4": self.model.backbone.p4,
            "backbone_c2f3_P4": self.model.backbone.c2f3,
            "backbone_p5": self.model.backbone.p5,
            "backbone_c2f4": self.model.backbone.c2f4,
            "backbone_sppf_P5": self.model.backbone.sppf,
            # Neck layers
            "neck_c2f21": self.model.neck.c2f21,
            "neck_c2f11": self.model.neck.c2f11,
            "neck_c2f12": self.model.neck.c2f12,
            "neck_c2f22": self.model.neck.c2f22,
            # Head layers
            "head8_conv11": self.model.head8.conv11,
            "head8_conv21": self.model.head8.conv21,
            "head16_conv11": self.model.head16.conv11,
            "head16_conv21": self.model.head16.conv21,
            "head32_conv11": self.model.head32.conv11,
            "head32_conv21": self.model.head32.conv21,
        }
    
    def _register_hooks(self):
        """Register forward hooks to capture feature maps from model layers."""
        def get_hook(name):
            def hook(module, input, output):
                # Detach and move to CPU to prevent memory leaks
                self.feature_maps[name] = output.detach().cpu()
            return hook
        
        available_layers = self._get_available_layers()
        layers_to_hook = set()
        
        if self.save_feature_maps is True:
            # Hook into all available layers
            layers_to_hook.update(available_layers.keys())
        elif isinstance(self.save_feature_maps, list):
            # Hook into specified layers only
            invalid_layers = [l for l in self.save_feature_maps if l not in available_layers]
            if invalid_layers:
                available = ", ".join(sorted(available_layers.keys()))
                raise ValueError(
                    f"Invalid layer names: {invalid_layers}. "
                    f"Available layers: {available}"
                )
            layers_to_hook.update(self.save_feature_maps)
        
        # Add EigenCAM layer if enabled
        if self.save_eigen_cam:
            layers_to_hook.add(self._eigen_cam_layer)
        
        # Register hooks for all required layers
        for layer_name in layers_to_hook:
            module = available_layers[layer_name]
            self.hooks.append(module.register_forward_hook(get_hook(layer_name)))
    
    def _save_feature_maps(self, image_path):
        """Save feature map visualizations to disk."""
        # Determine the base name for the output directory
        if isinstance(image_path, str):
            stem = get_safe_stem(image_path)
        else:
            stem = "inference"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("runs/feature_maps") / f"{stem}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "model": "LIBREYOLO8",
            "size": self.size,
            "input_size": [640, 640],
            "image_source": str(image_path) if isinstance(image_path, str) else "PIL/numpy input",
            "layers_captured": list(self.feature_maps.keys())
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature map visualizations
        for layer_name, fmap in self.feature_maps.items():
            # fmap shape: (batch, channels, H, W) - take first batch item
            fmap = fmap[0] if fmap.dim() == 4 else fmap
            
            # Create a 4x4 grid of the first 16 channels
            channels = min(fmap.shape[0], 16)
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            
            for i in range(16):
                ax = axes[i // 4, i % 4]
                if i < channels:
                    # Normalize the feature map for better visualization
                    channel_data = fmap[i].numpy()
                    ax.imshow(channel_data, cmap='viridis')
                ax.axis('off')
            
            plt.suptitle(f"Feature Maps: {layer_name}\nShape: {list(fmap.shape)}", fontsize=14)
            plt.tight_layout()
            plt.savefig(save_dir / f"{layer_name}.png", bbox_inches='tight', dpi=100)
            plt.close()
        
        # Clear feature maps after saving (only if not using eigen_cam)
        if not self.save_eigen_cam:
            self.feature_maps.clear()
        
        return str(save_dir)
    
    def _save_eigen_cam(self, image_path, original_img: Image.Image):
        """Save EigenCAM heatmap visualizations to disk."""
        # Get the activation from the target layer
        if self._eigen_cam_layer not in self.feature_maps:
            return None
        
        activation = self.feature_maps[self._eigen_cam_layer]
        # activation shape: (batch, channels, H, W) - take first batch item
        activation = activation[0].numpy() if activation.dim() == 4 else activation.numpy()
        
        # Compute EigenCAM heatmap
        heatmap = compute_eigen_cam(activation)
        
        # Determine the base name for the output directory
        if isinstance(image_path, str):
            stem = get_safe_stem(image_path)
        else:
            stem = "inference"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("runs/eigen_cam") / f"{stem}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert PIL Image to numpy array for overlay
        img_array = np.array(original_img)
        
        # Save heatmap overlay
        overlay = overlay_heatmap(img_array, heatmap, alpha=0.5)
        Image.fromarray(overlay).save(save_dir / "heatmap_overlay.jpg")
        
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        heatmap_gray = (heatmap_resized * 255).astype(np.uint8)
        Image.fromarray(heatmap_gray).save(save_dir / "heatmap_grayscale.png")
        
        # Save metadata
        metadata = {
            "model": "LIBREYOLO8",
            "size": self.size,
            "target_layer": self._eigen_cam_layer,
            "image_source": str(image_path) if isinstance(image_path, str) else "PIL/numpy input"
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Clear feature maps after saving
        self.feature_maps.clear()
        
        return str(save_dir)
    
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
        output_file_format: Optional[str] = None
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
            output_path: Optional path to save the annotated image. If not provided,
                         saves to 'runs/detections/' with a timestamped name.
            conf_thres: Confidence threshold (default: 0.25)
            iou_thres: IoU threshold for NMS (default: 0.45)
            color_format: Color format hint for NumPy/OpenCV arrays.
                - "auto": Auto-detect (default)
                - "rgb": Input is RGB format
                - "bgr": Input is BGR format (e.g., OpenCV)
            batch_size: Number of images to process per batch when handling multiple
                images (e.g., directories). Currently used for chunking at the Python
                level; true batched model inference is planned for future versions.
                Default: 1 (process one image at a time).
            tiling: Enable tiling for processing large/high-resolution images (default: False).
                When enabled, large images are automatically split into overlapping 640x640 tiles,
                inference is run on each tile, and results are merged using NMS.
            output_file_format: Output image format when saving. Options: "jpg", "png", "webp".
                Defaults to "jpg" for maximum compatibility.

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
                output_file_format=output_file_format
            )

        # Use tiled inference for large images when tiling is enabled
        if tiling:
            return self._predict_tiled(image, save, output_path, conf_thres, iou_thres, color_format, output_file_format)

        return self._predict_single(image, save, output_path, conf_thres, iou_thres, color_format, output_file_format)
    
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
        output_file_format: Optional[str] = None
    ) -> List[dict]:
        """
        Process multiple images, respecting batch_size for chunking.

        This method provides the scaffolding for batch processing. Currently, it
        processes images sequentially within each batch chunk. Future versions
        will implement true batched model inference for improved throughput.

        Args:
            image_paths: List of image paths to process.
            batch_size: Number of images per batch chunk.
            save: If True, saves annotated images.
            output_path: Optional output path for saved images.
            conf_thres: Confidence threshold.
            iou_thres: IoU threshold for NMS.
            color_format: Color format hint.
            tiling: Enable tiling for large images.
            output_file_format: Output image format when saving.

        Returns:
            List of detection dictionaries, one per image.
        """
        results = []
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i:i + batch_size]
            # TODO: Implement _predict_batch() for true batched model inference
            # For now, process images sequentially within each chunk
            for path in chunk:
                if tiling:
                    results.append(
                        self._predict_tiled(path, save, output_path, conf_thres, iou_thres, color_format, output_file_format)
                    )
                else:
                    results.append(
                        self._predict_single(path, save, output_path, conf_thres, iou_thres, color_format, output_file_format)
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
        output_file_format: Optional[str] = None
    ) -> dict:
        """
        Run inference on a single image.

        This is the internal implementation for single-image inference.
        Use __call__ for the public API which also supports directories.
        """
        # Store original image path for saving
        image_path = image if isinstance(image, (str, Path)) else None

        # Preprocess image
        input_tensor, original_img, original_size = preprocess_image(image, input_size=640, color_format=color_format)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor.to(self.device))

        # Postprocess
        detections = postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=640,
            original_size=original_size
        )

        # Add source path for traceability
        detections["source"] = str(image_path) if image_path else None

        # Save feature maps if enabled
        if self.save_feature_maps:
            feature_maps_path = self._save_feature_maps(image_path)
            detections["feature_maps_path"] = feature_maps_path

        # Save EigenCAM heatmap if enabled
        if self.save_eigen_cam:
            eigen_cam_path = self._save_eigen_cam(image_path, original_img)
            detections["eigen_cam_path"] = eigen_cam_path

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
                    # If directory, create it and use default naming
                    final_output_path.mkdir(parents=True, exist_ok=True)
                    if isinstance(image_path, (str, Path)):
                        stem = get_safe_stem(image_path)
                    else:
                        stem = "inference"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    final_output_path = final_output_path / f"{stem}_{timestamp}{ext}"
                else:
                    # If file path, ensure parent directory exists
                    final_output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Determine save directory (matching feature map style)
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
    
    def _merge_tile_detections(
        self,
        boxes: List,
        scores: List,
        classes: List,
        iou_thres: float
    ) -> Tuple[List, List, List]:
        """
        Merge detections from tiles using class-wise NMS.
        
        Args:
            boxes: List of boxes in xyxy format from all tiles.
            scores: List of confidence scores from all tiles.
            classes: List of class IDs from all tiles.
            iou_thres: IoU threshold for NMS.
        
        Returns:
            Tuple of (final_boxes, final_scores, final_classes) after merging.
        """
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
        output_file_format: Optional[str] = None
    ) -> dict:
        """
        Run tiled inference on large images.

        Splits the image into overlapping 640x640 tiles, runs inference on each,
        shifts detections back to original coordinates, and merges with NMS.

        Args:
            image: Input image (path, PIL Image, numpy array, etc.)
            save: If True, saves the annotated image, individual tiles, and grid visualization.
            output_path: Optional path to save the outputs. When provided as a directory,
                outputs are saved there; otherwise a timestamped folder is created.
            conf_thres: Confidence threshold.
            iou_thres: IoU threshold for NMS.
            color_format: Color format hint for numpy arrays.
            output_file_format: Output image format when saving.

        Returns:
            Dictionary with detection results including tiling metadata.
        """
        # Load full image
        img_pil = ImageLoader.load(image, color_format=color_format)
        orig_width, orig_height = img_pil.size
        image_path = image if isinstance(image, (str, Path)) else None

        # Skip tiling if image is already small enough
        if orig_width <= 640 and orig_height <= 640:
            return self._predict_single(image, save, output_path, conf_thres, iou_thres, color_format, output_file_format)

        # Get tile coordinates
        slices = get_slice_bboxes(orig_width, orig_height)

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
                "model": "LIBREYOLO8",
                "size": self.size,
                "image_source": str(image_path) if image_path else "PIL/numpy input",
                "original_size": [orig_width, orig_height],
                "num_tiles": len(slices),
                "tile_size": 640,
                "overlap_ratio": 0.2,
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
        
    def export(self, output_path: str = None, input_size: int = 640, opset: int = 12) -> str:
        """
        Export the model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX file. If None, uses the model's weights path with .onnx extension.
            input_size: The image size to export for (default: 640).
            opset: ONNX opset version (default: 12).
            
        Returns:
            Path to the exported ONNX file.
        """
        import inspect
        import torch.onnx

        # Torch's exporter requires the `onnx` package in the environment.
        # Use a spec check (instead of importing) so optional deps don't trigger
        # static-analysis import errors.
        import importlib.util

        if importlib.util.find_spec("onnx") is None:
            raise ImportError(
                "ONNX export requires the optional ONNX dependencies. "
                "Install them with `uv sync --extra onnx` (recommended) or "
                "`pip install -e '.[onnx]'`."
            )
        
        if output_path is None:
            if self.model_path and isinstance(self.model_path, str):
                output_path = str(Path(self.model_path).with_suffix('.onnx'))
            else:
                output_path = f"libreyolo8{self.size}.onnx"
        
        print(f"Exporting LibreYOLO8 {self.size} to {output_path}...")
        
        # 1. Create a dummy input (Batch, Channels, Height, Width)
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        
        # 2. Define a wrapper that decodes boxes for end-to-end inference
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                output = self.model(x)
                
                # Collect outputs from the 3 heads
                box_layers = [output['x8']['box'], output['x16']['box'], output['x32']['box']]
                cls_layers = [output['x8']['cls'], output['x16']['cls'], output['x32']['cls']]
                strides = [8, 16, 32]
                
                # Generate anchors (Traceable)
                anchors, stride_tensor = make_anchors(box_layers, strides)
                
                # Flatten and concatenate predictions
                # Box: (Batch, 4, H, W) -> (Batch, N, 4)
                box_preds = torch.cat([x.flatten(2).permute(0, 2, 1) for x in box_layers], dim=1)
                # Cls: (Batch, 80, H, W) -> (Batch, N, 80)
                cls_preds = torch.cat([x.flatten(2).permute(0, 2, 1) for x in cls_layers], dim=1)
                
                # Decode boxes to xyxy (Batch, N, 4)
                decoded_boxes = decode_boxes(box_preds, anchors, stride_tensor)
                
                # Apply sigmoid to class scores
                cls_scores = cls_preds.sigmoid()
                
                # Return concatenated [boxes, scores]: (Batch, N, 84)
                return torch.cat([decoded_boxes, cls_scores], dim=-1)

        wrapper = ONNXWrapper(self.model)
        wrapper.eval()
        
        # 3. Perform the export
        try:
            # Newer PyTorch versions may default to the "dynamo" ONNX exporter, which
            # pulls in extra deps like `onnxscript`. Prefer the legacy exporter by
            # explicitly setting `dynamo=False` when the argument exists.
            export_kwargs = {}
            try:
                if "dynamo" in inspect.signature(torch.onnx.export).parameters:
                    export_kwargs["dynamo"] = False
            except Exception:
                # If signature introspection fails for any reason, just proceed.
                pass

            torch.onnx.export(
                wrapper,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch'}
                },
                **export_kwargs,
            )
            print(f"Export complete: {output_path}")
            return output_path
        except Exception as e:
            print(f"Export failed: {e}")
            raise e

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
        output_file_format: Optional[str] = None
    ) -> Union[dict, List[dict]]:
        """
        Alias for __call__ method.

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
            batch_size: Number of images to process per batch when handling multiple
                images (e.g., directories). Default: 1.
            tiling: Enable tiling for processing large/high-resolution images (default: False).
            output_file_format: Output image format when saving. Options: "jpg", "png", "webp".

        Returns:
            For single image: Dictionary containing detection results.
            For directory: List of dictionaries, one per image processed.
        """
        return self(image=image, save=save, output_path=output_path, conf_thres=conf_thres, iou_thres=iou_thres, color_format=color_format, batch_size=batch_size, tiling=tiling, output_file_format=output_file_format)

    def explain(
        self,
        image: ImageInput,
        method: Optional[Union[str, List[str]]] = None,
        target_layer: Optional[str] = None,
        eigen_smooth: bool = False,
        save: bool = False,
        output_path: Optional[str] = None,
        alpha: float = 0.5,
        color_format: str = "auto"
    ) -> Union[dict, List[dict]]:
        """
        Generate explainability heatmap for the given image using CAM methods.

        **EXPERIMENTAL**: This feature is under active development. Results may vary
        between model versions (YOLO8 vs YOLO11) and CAM methods. Some methods may
        produce better visualizations than others depending on the model and image.

        This method provides visual explanations of what the model focuses on
        when making predictions. It supports multiple CAM (Class Activation Mapping)
        techniques including gradient-based and gradient-free methods.

        Args:
            image: Input image. Supported types:
                - str: Local file path or URL (http/https/s3/gs)
                - pathlib.Path: Local file path
                - PIL.Image: PIL Image object
                - np.ndarray: NumPy array (HWC or CHW, RGB or BGR)
                - torch.Tensor: PyTorch tensor (CHW or NCHW)
                - bytes: Raw image bytes
                - io.BytesIO: BytesIO object containing image data
            method: CAM method(s) to use. Can be a single string or a list of methods.
                Available methods:
                - "eigencam": Gradient-free, SVD-based (default)
                - "gradcam": Gradient-weighted class activation
                - "gradcam++": Improved GradCAM with second-order gradients
                - "xgradcam": Axiom-based GradCAM
                - "hirescam": High-resolution CAM
                - "layercam": Layer-wise CAM
                - "eigengradcam": Eigen-based gradient CAM
            target_layer: Layer name for CAM computation. Use get_available_layer_names()
                         to see options. Defaults to "neck_c2f22".
            eigen_smooth: Apply SVD smoothing to the heatmap (default: False).
            save: If True, saves the heatmap visualization to disk.
            output_path: Optional path to save the visualization.
            alpha: Blending factor for overlay (default: 0.5).
            color_format: Color format hint for NumPy/OpenCV arrays ("auto", "rgb", "bgr").

        Returns:
            If method is a single string: Dictionary containing:
                - heatmap: Grayscale heatmap array of shape (H, W) with values in [0, 1]
                - overlay: RGB overlay image as numpy array
                - original_image: Original image as PIL Image
                - method: CAM method used
                - target_layer: Target layer used
                - saved_path: Path to saved visualization (if save=True)

            If method is a list: List of dictionaries, one per method.

        Example:
            >>> model = LIBREYOLO8("yolo8n.pt", size="n")
            >>> # Single method
            >>> result = model.explain("image.jpg", method="gradcam", save=True)
            >>> # Multiple methods
            >>> results = model.explain("image.jpg", method=["eigencam", "gradcam"])
            >>> for r in results:
            ...     print(f"{r['method']}: heatmap shape {r['heatmap'].shape}")
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        # Handle list of methods
        if isinstance(method, list):
            results = []
            for m in method:
                results.append(self._explain_single(
                    image, m, target_layer, eigen_smooth, save, output_path, alpha, color_format
                ))
            return results

        return self._explain_single(
            image, method, target_layer, eigen_smooth, save, output_path, alpha, color_format
        )

    def _explain_single(
        self,
        image: ImageInput,
        method: Optional[str] = None,
        target_layer: Optional[str] = None,
        eigen_smooth: bool = False,
        save: bool = False,
        output_path: Optional[str] = None,
        alpha: float = 0.5,
        color_format: str = "auto"
    ) -> dict:
        """
        Internal method to generate a single explainability heatmap.

        **EXPERIMENTAL**: This feature is under active development.
        """
        method = (method or self.cam_method).lower()
        target_layer = target_layer or self._eigen_cam_layer

        if method not in CAM_METHODS:
            available = ", ".join(CAM_METHODS.keys())
            raise ValueError(f"Unknown CAM method '{method}'. Available: {available}")
        
        # Validate layer
        available_layers = self._get_available_layers()
        if target_layer not in available_layers:
            available = ", ".join(sorted(available_layers.keys()))
            raise ValueError(f"Unknown layer '{target_layer}'. Available: {available}")
        
        # Preprocess image
        input_tensor, original_img, original_size = preprocess_image(image, input_size=640, color_format=color_format)
        
        # Get target layer module
        target_module = available_layers[target_layer]
        
        # Create CAM instance
        cam_class = CAM_METHODS[method]
        cam = cam_class(
            model=self.model,
            target_layers=[target_module],
            reshape_transform=None
        )
        
        try:
            # Compute CAM
            grayscale_cam = cam(input_tensor.to(self.device), eigen_smooth=eigen_smooth)
            
            # Get the first batch item
            heatmap = grayscale_cam[0]
            
            # Resize heatmap to original image size
            heatmap_resized = cv2.resize(heatmap, (original_size[0], original_size[1]))
            
            # Normalize to [0, 1]
            heatmap_min, heatmap_max = heatmap_resized.min(), heatmap_resized.max()
            if heatmap_max - heatmap_min > 1e-8:
                heatmap_resized = (heatmap_resized - heatmap_min) / (heatmap_max - heatmap_min)
            
            # Create overlay
            img_array = np.array(original_img)
            overlay = overlay_heatmap(img_array, heatmap_resized, alpha=alpha)
            
            result = {
                "heatmap": heatmap_resized,
                "overlay": overlay,
                "original_image": original_img,
                "method": method,
                "target_layer": target_layer,
            }
            
            # Save if requested
            if save:
                image_path = image if isinstance(image, str) else None
                if isinstance(image_path, str):
                    stem = get_safe_stem(image_path)
                else:
                    stem = "inference"
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                if output_path:
                    save_dir = Path(output_path)
                    if save_dir.suffix:
                        save_dir = save_dir.parent
                else:
                    save_dir = Path(f"runs/{method}") / f"{stem}_{timestamp}"
                
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Save overlay
                Image.fromarray(overlay).save(save_dir / "heatmap_overlay.jpg")
                
                # Save grayscale heatmap
                heatmap_gray = (heatmap_resized * 255).astype(np.uint8)
                Image.fromarray(heatmap_gray).save(save_dir / "heatmap_grayscale.png")
                
                # Save metadata
                metadata = {
                    "model": "LIBREYOLO8",
                    "size": self.size,
                    "method": method,
                    "target_layer": target_layer,
                    "eigen_smooth": eigen_smooth,
                    "image_source": str(image) if isinstance(image, str) else "PIL/numpy input"
                }
                with open(save_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                
                result["saved_path"] = str(save_dir)
            
            return result
            
        finally:
            # Clean up CAM resources
            cam.release()

    @staticmethod
    def get_available_cam_methods() -> List[str]:
        """
        Get list of available CAM methods.
        
        Returns:
            List of CAM method names that can be used with explain().
        """
        return list(CAM_METHODS.keys())

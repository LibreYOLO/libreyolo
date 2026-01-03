"""
LibreYOLO-RD (Regional Diversity) inference wrapper.

Provides a high-level API for YOLO-RD object detection inference.
YOLO-RD extends YOLOv9-c with DConv for enhanced regional feature diversity.
"""

import json
from datetime import datetime
from typing import Union, List, Optional, Tuple
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from .nn import LibreYOLORDModel
from .utils import preprocess_image, postprocess, draw_boxes, nms
from ..common.eigen_cam import compute_eigen_cam, overlay_heatmap
from ..common.cam import CAM_METHODS
from ..common.image_loader import ImageInput, ImageLoader
from ..common.utils import get_safe_stem, get_slice_bboxes, draw_tile_grid
from ..common.postprocessing import get_postprocessor, list_postprocessors


class LIBREYOLORD:
    """
    LibreYOLO-RD (Regional Diversity) model for object detection.

    Based on YOLOv9-c architecture with DConv at B3 position for
    enhanced regional feature diversity.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
        size: Model size variant (required). Must be "c" (only variant supported)
        atoms: DConv atoms parameter (512 for rd-9c, 4096 for rd-9c-4096).
            If "auto", attempts to detect from weights.
        reg_max: Regression max value for DFL (default: 16)
        nb_classes: Number of classes (default: 80 for COCO)
        save_feature_maps: Feature map saving mode
        save_eigen_cam: If True, saves EigenCAM heatmap visualizations
        cam_method: CAM method for explain()
        cam_layer: Target layer for CAM computation
        device: Device for inference

    Example:
        >>> model = LIBREYOLORD(model_path="path/to/weights.pt", size="c")
        >>> detections = model(image=image_path, save=True)
        >>> # For 4096 variant:
        >>> model_4096 = LIBREYOLORD("path/to/rd_4096.pt", size="c", atoms=4096)
    """

    def __init__(
        self,
        model_path: Union[str, dict],
        size: str = "c",
        atoms: Union[int, str] = "auto",
        reg_max: int = 16,
        nb_classes: int = 80,
        save_feature_maps: Union[bool, List[str]] = False,
        save_eigen_cam: bool = False,
        cam_method: str = "eigencam",
        cam_layer: Optional[str] = None,
        device: str = "auto"
    ):
        if size != 'c':
            raise ValueError(f"YOLO-RD only supports size='c'. Got: {size}")

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
        self._eigen_cam_layer = cam_layer or "neck_elan_down2"

        # Load weights first if auto-detecting atoms
        if isinstance(model_path, dict):
            state_dict = model_path
            self.model_path = None
        else:
            self.model_path = model_path
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model weights file not found: {model_path}")
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

        # Detect atoms from weights if auto
        if atoms == "auto":
            atoms = self._detect_atoms(state_dict)

        self.atoms = atoms

        # Initialize model with detected atoms
        self.model = LibreYOLORDModel(config=size, reg_max=reg_max, nb_classes=nb_classes, atoms=atoms)

        # Load weights
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}") from e

        # Set to evaluation mode and move to device
        self.model.eval()
        self.model.to(self.device)

        # Register hooks for feature map extraction
        if self.save_feature_maps or self.save_eigen_cam:
            self._register_hooks()

    def _detect_atoms(self, state_dict: dict) -> int:
        """Detect atoms from weights by checking DConv layer shapes.

        The CG layer (backbone.elan2.dconv.CG.conv.weight) has shape [atoms, in_channels, 1, 1].
        Standard rd-9c uses atoms=512, rd-9c-4096 uses atoms=4096.
        """
        cg_key = "backbone.elan2.dconv.CG.conv.weight"
        if cg_key in state_dict:
            atoms = state_dict[cg_key].shape[0]
            return atoms
        # Fallback to default
        return 512

    def get_available_layer_names(self) -> List[str]:
        """Get list of available layer names for feature map saving."""
        return sorted(self._get_available_layers().keys())

    def _get_available_layers(self) -> dict:
        """Get mapping of layer names to module objects."""
        return {
            # Backbone layers
            "backbone_conv0": self.model.backbone.conv0,
            "backbone_conv1": self.model.backbone.conv1,
            "backbone_elan1": self.model.backbone.elan1,
            "backbone_down2": self.model.backbone.down2,
            "backbone_elan2": self.model.backbone.elan2,  # Contains DConv
            "backbone_down3": self.model.backbone.down3,
            "backbone_elan3": self.model.backbone.elan3,
            "backbone_down4": self.model.backbone.down4,
            "backbone_elan4": self.model.backbone.elan4,
            "backbone_spp": self.model.backbone.spp,
            # Neck layers
            "neck_elan_up1": self.model.neck.elan_up1,
            "neck_elan_up2": self.model.neck.elan_up2,
            "neck_elan_down1": self.model.neck.elan_down1,
            "neck_elan_down2": self.model.neck.elan_down2,
        }

    def _register_hooks(self):
        """Register forward hooks to capture feature maps."""
        def get_hook(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach().cpu()
            return hook

        available_layers = self._get_available_layers()
        layers_to_hook = set()

        if self.save_feature_maps is True:
            layers_to_hook.update(available_layers.keys())
        elif isinstance(self.save_feature_maps, list):
            invalid_layers = [l for l in self.save_feature_maps if l not in available_layers]
            if invalid_layers:
                available = ", ".join(sorted(available_layers.keys()))
                raise ValueError(f"Invalid layer names: {invalid_layers}. Available: {available}")
            layers_to_hook.update(self.save_feature_maps)

        if self.save_eigen_cam:
            layers_to_hook.add(self._eigen_cam_layer)

        for layer_name in layers_to_hook:
            module = available_layers[layer_name]
            self.hooks.append(module.register_forward_hook(get_hook(layer_name)))

    def _save_feature_maps(self, image_path):
        """Save feature map visualizations to disk."""
        if isinstance(image_path, str):
            stem = get_safe_stem(image_path)
        else:
            stem = "inference"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("runs/feature_maps") / f"{stem}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model": "LIBREYOLORD",
            "size": self.size,
            "input_size": [640, 640],
            "image_source": str(image_path) if isinstance(image_path, str) else "PIL/numpy input",
            "layers_captured": list(self.feature_maps.keys())
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        for layer_name, fmap in self.feature_maps.items():
            fmap = fmap[0] if fmap.dim() == 4 else fmap
            channels = min(fmap.shape[0], 16)
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))

            for i in range(16):
                ax = axes[i // 4, i % 4]
                if i < channels:
                    channel_data = fmap[i].numpy()
                    ax.imshow(channel_data, cmap='viridis')
                ax.axis('off')

            plt.suptitle(f"Feature Maps: {layer_name}\nShape: {list(fmap.shape)}", fontsize=14)
            plt.tight_layout()
            plt.savefig(save_dir / f"{layer_name}.png", bbox_inches='tight', dpi=100)
            plt.close()

        if not self.save_eigen_cam:
            self.feature_maps.clear()

        return str(save_dir)

    def _save_eigen_cam(self, image_path, original_img: Image.Image):
        """Save EigenCAM heatmap visualizations."""
        if self._eigen_cam_layer not in self.feature_maps:
            return None

        activation = self.feature_maps[self._eigen_cam_layer]
        activation = activation[0].numpy() if activation.dim() == 4 else activation.numpy()
        heatmap = compute_eigen_cam(activation)

        if isinstance(image_path, str):
            stem = get_safe_stem(image_path)
        else:
            stem = "inference"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path("runs/eigen_cam") / f"{stem}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        img_array = np.array(original_img)
        overlay = overlay_heatmap(img_array, heatmap, alpha=0.5)
        Image.fromarray(overlay).save(save_dir / "heatmap_overlay.jpg")

        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        heatmap_gray = (heatmap_resized * 255).astype(np.uint8)
        Image.fromarray(heatmap_gray).save(save_dir / "heatmap_grayscale.png")

        metadata = {
            "model": "LIBREYOLORD",
            "size": self.size,
            "target_layer": self._eigen_cam_layer,
            "image_source": str(image_path) if isinstance(image_path, str) else "PIL/numpy input"
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

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
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        postprocessor: Optional[str] = None,
        postprocessor_kwargs: Optional[dict] = None
    ) -> Union[dict, List[dict]]:
        """
        Run inference on an image or directory of images.

        Args:
            image: Input image or directory
            save: If True, saves the image with detections drawn
            output_path: Optional path to save the annotated image
            conf_thres: Confidence threshold (default: 0.25)
            iou_thres: IoU threshold for NMS (default: 0.45)
            color_format: Color format hint ("auto", "rgb", "bgr")
            batch_size: Number of images per batch for directories
            tiling: Enable tiling for large images
            overlap_ratio: Overlap between tiles
            output_file_format: Output format ("jpg", "png", "webp")
            postprocessor: Post-processing method
            postprocessor_kwargs: Additional post-processor arguments

        Returns:
            Detection results dictionary or list of dictionaries
        """
        if output_file_format is not None:
            output_file_format = output_file_format.lower().lstrip('.')
            if output_file_format not in ('jpg', 'jpeg', 'png', 'webp'):
                raise ValueError(f"Invalid output_file_format: {output_file_format}")

        if isinstance(image, (str, Path)) and Path(image).is_dir():
            image_paths = ImageLoader.collect_images(image)
            if not image_paths:
                return []
            return self._process_in_batches(
                image_paths, batch_size, save, output_path, conf_thres, iou_thres,
                color_format, tiling, overlap_ratio, output_file_format,
                postprocessor, postprocessor_kwargs
            )

        if tiling:
            return self._predict_tiled(
                image, save, output_path, conf_thres, iou_thres, color_format,
                overlap_ratio, output_file_format, postprocessor, postprocessor_kwargs
            )

        return self._predict_single(
            image, save, output_path, conf_thres, iou_thres, color_format,
            output_file_format, postprocessor, postprocessor_kwargs
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
        postprocessor: Optional[str] = None,
        postprocessor_kwargs: Optional[dict] = None
    ) -> List[dict]:
        """Process multiple images in batches."""
        results = []
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i:i + batch_size]
            for path in chunk:
                if tiling:
                    results.append(self._predict_tiled(
                        path, save, output_path, conf_thres, iou_thres,
                        color_format, overlap_ratio, output_file_format,
                        postprocessor, postprocessor_kwargs
                    ))
                else:
                    results.append(self._predict_single(
                        path, save, output_path, conf_thres, iou_thres,
                        color_format, output_file_format, postprocessor,
                        postprocessor_kwargs
                    ))
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
        image_path = image if isinstance(image, (str, Path)) else None

        # Preprocess
        input_tensor, original_img, original_size = preprocess_image(
            image, input_size=640, color_format=color_format
        )

        # Inference
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

        detections["source"] = str(image_path) if image_path else None

        # Save feature maps
        if self.save_feature_maps:
            feature_maps_path = self._save_feature_maps(image_path)
            detections["feature_maps_path"] = feature_maps_path

        # Save EigenCAM
        if self.save_eigen_cam:
            eigen_cam_path = self._save_eigen_cam(image_path, original_img)
            detections["eigen_cam_path"] = eigen_cam_path

        # Save annotated image
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

            ext = f".{output_file_format}" if output_file_format else ".jpg"

            if output_path:
                final_output_path = Path(output_path)
                if final_output_path.suffix == "":
                    final_output_path.mkdir(parents=True, exist_ok=True)
                    stem = get_safe_stem(image_path) if image_path else "inference"
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    final_output_path = final_output_path / f"{stem}_{timestamp}{ext}"
                else:
                    final_output_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                stem = get_safe_stem(image_path) if image_path else "inference"
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
        color_format: str = "auto",
        overlap_ratio: float = 0.2,
        output_file_format: Optional[str] = None,
        postprocessor: Optional[str] = None,
        postprocessor_kwargs: Optional[dict] = None
    ) -> dict:
        """Run tiled inference on large images."""
        img_pil = ImageLoader.load(image, color_format=color_format)
        orig_width, orig_height = img_pil.size
        image_path = image if isinstance(image, (str, Path)) else None

        if orig_width <= 640 and orig_height <= 640:
            return self._predict_single(
                image, save, output_path, conf_thres, iou_thres,
                color_format, output_file_format, postprocessor, postprocessor_kwargs
            )

        slices = get_slice_bboxes(orig_width, orig_height, overlap_ratio=overlap_ratio)

        all_boxes, all_scores, all_classes = [], [], []
        tiles_data = []

        for idx, (x1, y1, x2, y2) in enumerate(slices):
            tile = img_pil.crop((x1, y1, x2, y2))

            if save:
                tiles_data.append({
                    "index": idx,
                    "coords": (x1, y1, x2, y2),
                    "image": tile.copy()
                })

            result = self._predict_single(
                tile, save=False, conf_thres=conf_thres, iou_thres=iou_thres,
                postprocessor=postprocessor, postprocessor_kwargs=postprocessor_kwargs
            )

            for box in result["boxes"]:
                shifted_box = [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1]
                all_boxes.append(shifted_box)
            all_scores.extend(result["scores"])
            all_classes.extend(result["classes"])

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

            ext = f".{output_file_format}" if output_file_format else ".jpg"
            stem = get_safe_stem(image_path) if image_path else "inference"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if output_path:
                base_output_path = Path(output_path)
                if base_output_path.suffix == "":
                    save_dir = base_output_path / f"{stem}_{timestamp}"
                else:
                    save_dir = base_output_path.parent / f"{stem}_{timestamp}"
            else:
                save_dir = Path("runs/tiled_detections") / f"{stem}_{timestamp}"

            save_dir.mkdir(parents=True, exist_ok=True)

            tiles_dir = save_dir / "tiles"
            tiles_dir.mkdir(parents=True, exist_ok=True)
            for tile_data in tiles_data:
                tile_filename = f"tile_{tile_data['index']:03d}{ext}"
                tile_path = tiles_dir / tile_filename
                tile_data["image"].save(tile_path)

            final_image_path = save_dir / f"final_image{ext}"
            annotated_img.save(final_image_path)

            grid_img = draw_tile_grid(img_pil, slices)
            grid_path = save_dir / f"grid_visualization{ext}"
            grid_img.save(grid_path)

            metadata = {
                "model": "LIBREYOLORD",
                "size": self.size,
                "image_source": str(image_path) if image_path else "PIL/numpy input",
                "original_size": [orig_width, orig_height],
                "num_tiles": len(slices),
                "tile_size": 640,
                "overlap_ratio": overlap_ratio,
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
        """Export the model to ONNX format."""
        import importlib.util

        if importlib.util.find_spec("onnx") is None:
            raise ImportError(
                "ONNX export requires the optional ONNX dependencies. "
                "Install them with `uv sync --extra onnx` or `pip install -e '.[onnx]'`."
            )

        if output_path is None:
            if self.model_path and isinstance(self.model_path, str):
                output_path = str(Path(self.model_path).with_suffix('.onnx'))
            else:
                output_path = f"libreyolord{self.size}.onnx"

        print(f"Exporting LibreYOLO-RD {self.size} to {output_path}...")

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
                input_names=['images'],
                output_names=['output'],
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch'}
                },
            )
            print(f"Export complete: {output_path}")
            return output_path
        except Exception as e:
            print(f"Export failed: {e}")
            raise

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
        """Alias for __call__ method."""
        return self(
            image=image, save=save, output_path=output_path,
            conf_thres=conf_thres, iou_thres=iou_thres,
            color_format=color_format, batch_size=batch_size,
            tiling=tiling, output_file_format=output_file_format
        )

    @staticmethod
    def get_available_cam_methods() -> List[str]:
        """Get list of available CAM methods."""
        return list(CAM_METHODS.keys())

    @staticmethod
    def get_available_postprocessors() -> dict:
        """Get available post-processing methods."""
        return list_postprocessors()

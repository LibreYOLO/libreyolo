"""
Libre YOLO8 implementation.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from .nn import LibreYOLO8Model
from .utils import preprocess_image, postprocess, decode_boxes
from ..common.base_model import LibreYOLOBase
from ..common.utils import make_anchors, get_safe_stem, draw_boxes
from ..common.postprocessing import get_postprocessor, list_postprocessors
from ..common.eigen_cam import compute_eigen_cam, overlay_heatmap
from ..common.cam import CAM_METHODS
from ..common.image_loader import ImageInput


class LIBREYOLO8(LibreYOLOBase):
    """
    Libre YOLO8 model for object detection.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
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
        device: str = "auto",
    ):
        # Store reg_max before calling parent __init__
        self.reg_max = reg_max

        # Store XAI parameters before parent init
        self.save_feature_maps = save_feature_maps
        self.save_eigen_cam = save_eigen_cam
        self.cam_method = cam_method.lower()
        self._eigen_cam_layer = cam_layer or "neck_c2f22"
        self.feature_maps = {}
        self.hooks = []

        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
        )

        # Register hooks for feature map extraction after model is initialized
        if self.save_feature_maps or self.save_eigen_cam:
            self._register_hooks()

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================

    def _get_valid_sizes(self) -> List[str]:
        return ["n", "s", "m", "l", "x"]

    def _get_model_name(self) -> str:
        return "LIBREYOLO8"

    def _get_default_cam_layer(self) -> str:
        return "neck_c2f22"

    def _get_input_size(self) -> int:
        return 640

    def _init_model(self) -> nn.Module:
        return LibreYOLO8Model(
            config=self.size, reg_max=self.reg_max, nb_classes=self.nb_classes
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
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

    def _preprocess(
        self, image: Any, color_format: str = "auto"
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        return preprocess_image(image, input_size=640, color_format=color_format)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        postprocessor: Optional[str] = None,
        postprocessor_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> Dict:
        """
        Postprocess model output.

        Supports both legacy postprocessing and the pluggable postprocessor system.
        """
        if postprocessor is not None:
            return self._postprocess_with_processor(
                output=output,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                original_size=original_size,
                postprocessor=postprocessor,
                postprocessor_kwargs=postprocessor_kwargs,
            )

        # Legacy postprocess
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=640,
            original_size=original_size,
        )

    # =========================================================================
    # XAI METHODS - Feature Maps and CAM
    # =========================================================================

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
        """
        Run inference on a single image with XAI support.

        Overrides base class to add feature map and EigenCAM saving.
        """
        image_path = image if isinstance(image, (str, Path)) else None

        # Preprocess
        input_tensor, original_img, original_size = self._preprocess(image, color_format)

        # Forward pass
        with torch.no_grad():
            output = self._forward(input_tensor.to(self.device))

        # Postprocess
        detections = self._postprocess(
            output, conf_thres, iou_thres, original_size, **kwargs
        )

        detections["source"] = str(image_path) if image_path else None

        # Save feature maps if enabled (XAI)
        if self.save_feature_maps:
            feature_maps_path = self._save_feature_maps(image_path)
            detections["feature_maps_path"] = feature_maps_path

        # Save EigenCAM heatmap if enabled (XAI)
        if self.save_eigen_cam:
            eigen_cam_path = self._save_eigen_cam(image_path, original_img)
            detections["eigen_cam_path"] = eigen_cam_path

        # Save annotated image
        if save:
            from ..common.utils import resolve_save_path

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
        input_tensor, original_img, original_size = preprocess_image(
            image, input_size=640, color_format=color_format
        )

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

    # =========================================================================
    # YOLO8-SPECIFIC METHODS
    # =========================================================================

    def _postprocess_with_processor(
        self,
        output: dict,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        postprocessor: str,
        postprocessor_kwargs: Optional[dict] = None,
    ) -> dict:
        """Apply postprocessing using the pluggable postprocessor system."""
        from ..common.postprocessing import rescale_boxes, filter_valid_boxes

        # Decode boxes and get class predictions
        box_layers = [output["x8"]["box"], output["x16"]["box"], output["x32"]["box"]]
        cls_layers = [output["x8"]["cls"], output["x16"]["cls"], output["x32"]["cls"]]
        strides = [8, 16, 32]

        anchors, stride_tensor = make_anchors(box_layers, strides)

        box_preds = torch.cat(
            [x.flatten(2).permute(0, 2, 1) for x in box_layers], dim=1
        )
        cls_preds = torch.cat(
            [x.flatten(2).permute(0, 2, 1) for x in cls_layers], dim=1
        )

        decoded_boxes = decode_boxes(box_preds, anchors, stride_tensor)
        decoded_boxes = decoded_boxes[0]  # Remove batch dimension

        scores = cls_preds[0].sigmoid()
        max_scores, class_ids = torch.max(scores, dim=1)

        # Apply confidence threshold
        mask = max_scores > conf_thres
        if not mask.any():
            return {
                "boxes": [],
                "scores": [],
                "classes": [],
                "num_detections": 0,
            }

        valid_boxes = decoded_boxes[mask]
        valid_scores = max_scores[mask]
        valid_classes = class_ids[mask]

        # Rescale boxes to original image size
        valid_boxes = rescale_boxes(valid_boxes, 640, original_size, clip=True)

        # Filter out invalid boxes
        valid_boxes, valid_scores, valid_classes = filter_valid_boxes(
            valid_boxes, valid_scores, valid_classes
        )

        if len(valid_boxes) == 0:
            return {
                "boxes": [],
                "scores": [],
                "classes": [],
                "num_detections": 0,
            }

        # Get the post-processor
        pp_kwargs = postprocessor_kwargs or {}
        processor = get_postprocessor(
            postprocessor, conf_thres=conf_thres, iou_thres=iou_thres, **pp_kwargs
        )

        # Apply post-processing
        final_boxes, final_scores, final_classes = processor(
            valid_boxes, valid_scores, valid_classes
        )

        return {
            "boxes": final_boxes.cpu().tolist(),
            "scores": final_scores.cpu().tolist(),
            "classes": final_classes.cpu().tolist(),
            "num_detections": len(final_boxes),
        }

    def export(
        self, output_path: str = None, input_size: int = 640, opset: int = 12
    ) -> str:
        """
        Export the model to ONNX format.

        This override includes a wrapper that decodes boxes for end-to-end inference.

        Args:
            output_path: Path to save the ONNX file.
            input_size: The image size to export for (default: 640).
            opset: ONNX opset version (default: 12).

        Returns:
            Path to the exported ONNX file.
        """
        import inspect
        import importlib.util

        if importlib.util.find_spec("onnx") is None:
            raise ImportError(
                "ONNX export requires the optional ONNX dependencies. "
                "Install them with `uv sync --extra onnx` or `pip install -e '.[onnx]'`."
            )

        if output_path is None:
            if self.model_path and isinstance(self.model_path, str):
                output_path = str(Path(self.model_path).with_suffix(".onnx"))
            else:
                output_path = f"libreyolo8{self.size}.onnx"

        print(f"Exporting LibreYOLO8 {self.size} to {output_path}...")

        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

        # Wrapper that decodes boxes for end-to-end inference
        class ONNXWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                output = self.model(x)

                box_layers = [
                    output["x8"]["box"],
                    output["x16"]["box"],
                    output["x32"]["box"],
                ]
                cls_layers = [
                    output["x8"]["cls"],
                    output["x16"]["cls"],
                    output["x32"]["cls"],
                ]
                strides = [8, 16, 32]

                anchors, stride_tensor = make_anchors(box_layers, strides)

                box_preds = torch.cat(
                    [x.flatten(2).permute(0, 2, 1) for x in box_layers], dim=1
                )
                cls_preds = torch.cat(
                    [x.flatten(2).permute(0, 2, 1) for x in cls_layers], dim=1
                )

                decoded_boxes = decode_boxes(box_preds, anchors, stride_tensor)
                cls_scores = cls_preds.sigmoid()

                return torch.cat([decoded_boxes, cls_scores], dim=-1)

        wrapper = ONNXWrapper(self.model)
        wrapper.eval()

        try:
            export_kwargs = {}
            try:
                if "dynamo" in inspect.signature(torch.onnx.export).parameters:
                    export_kwargs["dynamo"] = False
            except Exception:
                pass

            torch.onnx.export(
                wrapper,
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
                **export_kwargs,
            )
            print(f"Export complete: {output_path}")
            return output_path
        except Exception as e:
            print(f"Export failed: {e}")
            raise

    @staticmethod
    def get_available_postprocessors() -> dict:
        """
        Get available post-processing methods with their descriptions.

        Returns:
            Dictionary mapping post-processor names to metadata.
        """
        return list_postprocessors()

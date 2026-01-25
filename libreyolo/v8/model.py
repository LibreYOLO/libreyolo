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
        device: str = "auto",
    ):
        # Store reg_max before calling parent __init__
        self.reg_max = reg_max

        # Store feature map parameters before parent init
        self.save_feature_maps = save_feature_maps
        self.feature_maps = {}
        self.hooks = []

        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
        )

        # Register hooks for feature map extraction after model is initialized
        if self.save_feature_maps:
            self._register_hooks()

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================

    def _get_valid_sizes(self) -> List[str]:
        return ["n", "s", "m", "l", "x"]

    def _get_model_name(self) -> str:
        return "LIBREYOLO8"

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

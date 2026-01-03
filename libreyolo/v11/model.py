"""
Libre YOLO11 implementation.
"""

from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from PIL import Image

from ..common.base_model import LibreYOLOBase
from ..common.image_loader import ImageInput
from ..common.utils import preprocess_image
from .nn import LibreYOLO11Model
from .utils import postprocess, make_anchors, decode_boxes


class LIBREYOLO11(LibreYOLOBase):
    """
    Libre YOLO11 model for object detection.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
        size: Model size variant (required). Must be one of: "n", "s", "m", "l", "x"
        reg_max: Regression max value for DFL (default: 16)
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference. "auto" (default) uses CUDA if available, else MPS, else CPU.

    Example:
        >>> model = LIBREYOLO11(model_path="path/to/weights.pt", size="x")
        >>> detections = model(image=image_path, save=True)
        >>> # Use tiling for large images
        >>> detections = model(image=large_image_path, save=True, tiling=True)
    """

    def __init__(
        self,
        model_path,
        size: str,
        reg_max: int = 16,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        self.reg_max = reg_max
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

    def _get_valid_sizes(self) -> List[str]:
        return ["n", "s", "m", "l", "x"]

    def _get_model_name(self) -> str:
        return "LIBREYOLO11"

    def _get_input_size(self) -> int:
        return 640

    def _init_model(self) -> nn.Module:
        return LibreYOLO11Model(
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
            "backbone_c2psa_P5": self.model.backbone.c2psa,
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
        self, image: ImageInput, color_format: str = "auto"
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
        **kwargs,
    ) -> Dict:
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=640,
            original_size=original_size,
        )

    def export(
        self, output_path: str = None, input_size: int = 640, opset: int = 12
    ) -> str:
        """
        Export the model to ONNX format with decoded boxes.

        Args:
            output_path: Path to save the ONNX file.
            input_size: The image size to export for (default: 640).
            opset: ONNX opset version (default: 12).

        Returns:
            Path to the exported ONNX file.
        """
        import importlib.util
        import inspect
        from pathlib import Path

        if importlib.util.find_spec("onnx") is None:
            raise ImportError(
                "ONNX export requires the optional ONNX dependencies. "
                "Install them with `uv sync --extra onnx` or `pip install -e '.[onnx]'`."
            )

        if output_path is None:
            if self.model_path and isinstance(self.model_path, str):
                output_path = str(Path(self.model_path).with_suffix(".onnx"))
            else:
                output_path = f"libreyolo11{self.size}.onnx"

        print(f"Exporting LibreYOLO11 {self.size} to {output_path}...")

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

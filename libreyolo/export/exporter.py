"""
Unified model export with multiple backend support.

Supports ONNX, TorchScript, TensorRT, and OpenVINO export formats with
various precision modes (FP32, FP16, INT8).
"""

import json
import warnings
from pathlib import Path
from typing import Optional

import torch

from .onnx import _get_version, export_onnx
from .torchscript import export_torchscript


class Exporter:
    """Export a LibreYOLO model to deployment formats.

    Supported formats:
        - onnx: ONNX format (universal interchange)
        - torchscript: TorchScript (PyTorch deployment)
        - tensorrt: TensorRT engine (NVIDIA GPU acceleration)
        - openvino: OpenVINO IR (Intel CPU/GPU/VPU acceleration)

    Args:
        model: A LibreYOLOBase subclass instance (LIBREYOLOX, LIBREYOLO9, etc.).

    Example::

        from libreyolo import LIBREYOLOX
        from libreyolo.export import Exporter

        model = LIBREYOLOX("weights.pt", size="s")
        exporter = Exporter(model)

        # Basic ONNX export
        path = exporter("onnx", simplify=True, dynamic=True)

        # TensorRT with FP16
        path = exporter("tensorrt", half=True)

        # TensorRT with INT8 (requires calibration dataset)
        path = exporter("tensorrt", int8=True, data="coco8.yaml")

        # Or use the model's export() facade directly:
        path = model.export(format="tensorrt", half=True)
    """

    FORMATS = {
        "onnx": {
            "suffix": ".onnx",
            "requires": None,
        },
        "torchscript": {
            "suffix": ".torchscript",
            "requires": None,
        },
        "tensorrt": {
            "suffix": ".engine",
            "requires": "onnx",  # TensorRT builds from ONNX
        },
        "openvino": {
            "suffix": "_openvino",
            "requires": "onnx",  # OpenVINO converts from ONNX
        },
    }

    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        format: str = "onnx",
        *,
        output_path: Optional[str] = None,
        imgsz: Optional[int] = None,
        opset: int = 13,
        simplify: bool = True,
        dynamic: bool = True,
        half: bool = False,
        int8: bool = False,
        batch: int = 1,
        device: Optional[str] = None,
        # Calibration parameters (for INT8)
        data: Optional[str] = None,
        fraction: float = 1.0,
        # TensorRT specific
        workspace: float = 4.0,
        hardware_compatibility: str = "none",
        gpu_device: int = 0,
        trt_config: Optional[str] = None,
        # Utility
        verbose: bool = False,
    ) -> str:
        """Export the model to a deployment format.

        Args:
            format: Target format ("onnx", "torchscript", "tensorrt", "openvino").
            output_path: Output file path (auto-generated if None).
            imgsz: Input resolution (default: model's native size).
            opset: ONNX opset version (default: 13).
            simplify: Run ONNX graph simplification (default: True).
            dynamic: Enable dynamic axes for ONNX (default: True).
                    For TensorRT, use with caution as it affects optimization.
            half: Export in FP16 precision (default: False).
                 Recommended for TensorRT on modern GPUs.
            int8: Export in INT8 precision (default: False).
                 Requires `data` parameter for calibration.
            batch: Batch size for the model (default: 1).
            device: Device to trace on (default: model's current device).
            data: Path to data.yaml for INT8 calibration dataset.
                 Required when int8=True.
            fraction: Fraction of calibration dataset to use (default: 1.0).
                     Lower values speed up calibration with slight accuracy loss.
            workspace: TensorRT workspace size in GiB (default: 4.0).
                      Larger values may find faster kernels during optimization.
            hardware_compatibility: TensorRT hardware compatibility level.
                - "none": Optimize for current GPU only (fastest, not portable)
                - "ampere_plus": Works on Ampere and newer GPUs
                - "same_compute_capability": Works on GPUs with same SM version
            gpu_device: GPU device ID for multi-GPU systems (default: 0).
            trt_config: Path to TensorRT export YAML config file.
                       When provided, overrides individual TensorRT parameters.
            verbose: Enable verbose logging (default: False).

        Returns:
            Path to the exported model file.

        Raises:
            ValueError: If format is unsupported or INT8 requested without data.
            ImportError: If required packages are not installed.
        """
        # --- validate format ---
        fmt = format.lower()
        if fmt not in self.FORMATS:
            valid = ", ".join(sorted(self.FORMATS))
            raise ValueError(
                f"Unsupported export format: {format!r}. Must be one of: {valid}"
            )
        fmt_info = self.FORMATS[fmt]

        # --- validate INT8 requirements ---
        if int8 and data is None and fmt in ("tensorrt", "openvino"):
            raise ValueError(
                "INT8 quantization requires calibration data.\n"
                "Provide data='path/to/data.yaml' or data='coco8' for built-in dataset."
            )

        # --- validate precision ---
        if half and int8:
            warnings.warn(
                "Both half=True and int8=True specified. Using INT8 precision."
            )
            half = False

        # --- resolve parameters ---
        if imgsz is None:
            imgsz = self.model._get_input_size()

        if device is None:
            device = self.model.device
        else:
            device = torch.device(device)

        if output_path is None:
            model_name = self.model._get_model_name().lower()
            precision_suffix = "_int8" if int8 else ("_fp16" if half else "")
            output_path = str(
                Path("weights")
                / f"{model_name}_{self.model.size}{precision_suffix}{fmt_info['suffix']}"
            )

        # --- ensure output directory exists ---
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # --- prepare model ---
        nn_model = self.model.model
        original_training = nn_model.training
        nn_model.eval()

        original_device = next(nn_model.parameters()).device
        nn_model.to(device)

        # --- set export mode for models that support it ---
        original_export = None
        export_attr = None  # which attribute holds the export flag
        if hasattr(nn_model, 'detect') and hasattr(nn_model.detect, 'export'):
            export_attr = 'detect'
            original_export = nn_model.detect.export
            nn_model.detect.export = True
        elif hasattr(nn_model, 'head') and hasattr(nn_model.head, 'export'):
            export_attr = 'head'
            original_export = nn_model.head.export
            nn_model.head.export = True

        # RF-DETR: activate LWDETR export mode (swaps forward to forward_export
        # which returns traceable tuple instead of mixed-type dict)
        rfdetr_export_activated = False
        rfdetr_layernorm_patches = []
        inner = getattr(nn_model, 'model', None)
        if inner is not None and hasattr(inner, 'forward_export') and hasattr(inner, '_export'):
            if not inner._export:
                inner.export()
                rfdetr_export_activated = True

            # Monkey-patch rfdetr LayerNorm to use static normalized_shape
            # instead of dynamic x.size(3) so the TorchScript ONNX exporter
            # can trace the graph.
            try:
                from rfdetr.models.backbone.projector import LayerNorm as RFDETRLayerNorm
                for m in nn_model.modules():
                    if isinstance(m, RFDETRLayerNorm):
                        rfdetr_layernorm_patches.append((m, m.forward))
                        ns = m.normalized_shape

                        def _static_forward(x, _ns=ns, _w=m.weight, _b=m.bias, _eps=m.eps):
                            x = x.permute(0, 2, 3, 1)
                            x = torch.nn.functional.layer_norm(x, _ns, _w, _b, _eps)
                            return x.permute(0, 3, 1, 2)

                        m.forward = _static_forward
            except ImportError:
                pass

        # --- build dummy input ---
        dummy = torch.randn(batch, 3, imgsz, imgsz, device=device)

        # For ONNX export, use FP16 if requested (TensorRT handles precision internally)
        if half and fmt == "onnx":
            nn_model.half()
            dummy = dummy.half()

        # --- prepare calibration data if needed ---
        calibration_data = None
        if int8 and data is not None:
            from .calibration import get_calibration_dataloader
            # Determine model type for correct preprocessing
            model_name = self.model._get_model_name().lower()
            if "yolox" in model_name:
                model_type = "yolox"
            elif "rfdetr" in model_name:
                model_type = "rfdetr"
            else:
                model_type = "yolov9"
            calibration_data = get_calibration_dataloader(
                data=data,
                imgsz=imgsz,
                batch=batch,
                fraction=fraction,
                model_type=model_type,
            )
            print(f"Calibration dataset: {len(calibration_data)} batches, "
                  f"{calibration_data.num_samples} images (preprocessing: {model_type})")

        # --- handle format dependencies ---
        onnx_path = None
        if fmt_info.get("requires") == "onnx":
            # Export to ONNX first as intermediate format
            onnx_output = str(Path(output_path).with_suffix(".onnx"))
            print(f"Step 1/2: Exporting to ONNX ({onnx_output})")
            onnx_path = export_onnx(
                nn_model,
                dummy,
                output_path=onnx_output,
                opset=opset,
                simplify=simplify,
                dynamic=False,  # TensorRT prefers static shapes in ONNX
                half=False,  # TensorRT handles precision internally
                metadata=self._build_onnx_metadata(dynamic=False, half=False),
            )

        # --- build metadata for the target format ---
        try:
            if fmt == "onnx":
                result = export_onnx(
                    nn_model,
                    dummy,
                    output_path=output_path,
                    opset=opset,
                    simplify=simplify,
                    dynamic=dynamic,
                    half=half,
                    metadata=self._build_onnx_metadata(dynamic=dynamic, half=half),
                )
            elif fmt == "torchscript":
                result = export_torchscript(
                    nn_model,
                    dummy,
                    output_path=output_path,
                )
            elif fmt == "tensorrt":
                from .tensorrt import export_tensorrt

                print("Step 2/2: Building TensorRT engine")

                if int8:
                    precision = "int8"
                elif half:
                    precision = "fp16"
                else:
                    precision = "fp32"

                metadata = {
                    "libreyolo_version": _get_version(),
                    "model_family": self.model._get_model_name(),
                    "model_size": self.model.size,
                    "nb_classes": self.model.nb_classes,
                    "names": {str(k): v for k, v in self.model.names.items()},
                    "imgsz": self.model._get_input_size(),
                    "precision": precision,
                    "dynamic": dynamic,
                    "exported_from": str(Path(onnx_path).name) if onnx_path else None,
                }

                result = export_tensorrt(
                    onnx_path=onnx_path,
                    output_path=output_path,
                    half=half,
                    int8=int8,
                    workspace=workspace,
                    calibration_data=calibration_data,
                    dynamic=dynamic,
                    verbose=verbose,
                    hardware_compatibility=hardware_compatibility,
                    device=gpu_device,
                    config=trt_config,
                    metadata=metadata,
                )
            elif fmt == "openvino":
                from .openvino import export_openvino

                print("Step 2/2: Converting to OpenVINO IR")

                if int8:
                    precision = "int8"
                elif half:
                    precision = "fp16"
                else:
                    precision = "fp32"

                metadata = {
                    "libreyolo_version": _get_version(),
                    "model_family": self.model._get_model_name(),
                    "model_size": self.model.size,
                    "nb_classes": self.model.nb_classes,
                    "names": {str(k): v for k, v in self.model.names.items()},
                    "imgsz": self.model._get_input_size(),
                    "precision": precision,
                    "dynamic": dynamic,
                    "exported_from": str(Path(onnx_path).name) if onnx_path else None,
                }

                result = export_openvino(
                    onnx_path=onnx_path,
                    output_path=output_path,
                    half=half,
                    int8=int8,
                    calibration_data=calibration_data,
                    verbose=verbose,
                    metadata=metadata,
                )
        finally:
            # restore model state
            nn_model.to(original_device)
            if half and fmt == "onnx":
                nn_model.float()
            if original_training:
                nn_model.train()
            if original_export is not None:
                getattr(nn_model, export_attr).export = original_export
            if rfdetr_export_activated:
                inner._export = False
                inner.forward = inner._forward_origin
            for m, orig_fwd in rfdetr_layernorm_patches:
                m.forward = orig_fwd

        # --- clean up intermediate ONNX file ---
        if onnx_path and Path(onnx_path).exists():
            Path(onnx_path).unlink()

        precision_str = "INT8" if int8 else ("FP16" if half else "FP32")
        print(
            f"\nExport complete: {result}\n"
            f"  Model: {self.model._get_model_name()} {self.model.size}\n"
            f"  Format: {fmt}\n"
            f"  Precision: {precision_str}\n"
            f"  Input size: {imgsz}x{imgsz}"
        )
        return result

    def _build_onnx_metadata(self, *, dynamic: bool, half: bool) -> dict:
        """Build the metadata dict for ONNX export."""
        return {
            "libreyolo_version": _get_version(),
            "model_family": self.model._get_model_name(),
            "model_size": self.model.size,
            "nb_classes": str(self.model.nb_classes),
            "names": json.dumps(
                {str(k): v for k, v in self.model.names.items()}
            ),
            "imgsz": str(self.model._get_input_size()),
            "dynamic": str(dynamic),
            "half": str(half),
        }

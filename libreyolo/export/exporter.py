"""
Unified model export with multiple backend support.

Supports ONNX, TorchScript, and TensorRT export formats with various
precision modes (FP32, FP16, INT8).
"""

import importlib.util
import json
import warnings
from pathlib import Path
from typing import Optional

import torch


class Exporter:
    """Export a LibreYOLO model to deployment formats.

    Supported formats:
        - onnx: ONNX format (universal interchange)
        - torchscript: TorchScript (PyTorch deployment)
        - tensorrt: TensorRT engine (NVIDIA GPU acceleration)

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
            "method": "_export_onnx",
            "requires": None,
        },
        "torchscript": {
            "suffix": ".torchscript",
            "method": "_export_torchscript",
            "requires": None,
        },
        "tensorrt": {
            "suffix": ".engine",
            "method": "_export_tensorrt",
            "requires": "onnx",  # TensorRT builds from ONNX
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
            format: Target format ("onnx", "torchscript", "tensorrt").
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
        if int8 and data is None and fmt == "tensorrt":
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
            output_path = f"{model_name}_{self.model.size}{precision_suffix}{fmt_info['suffix']}"

        # --- prepare model ---
        nn_model = self.model.model
        original_training = nn_model.training
        nn_model.eval()

        original_device = next(nn_model.parameters()).device
        nn_model.to(device)

        # --- set export mode for models that support it ---
        original_export = None
        if hasattr(nn_model, 'detect') and hasattr(nn_model.detect, 'export'):
            original_export = nn_model.detect.export
            nn_model.detect.export = True

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
            onnx_path = self._export_onnx(
                nn_model,
                dummy,
                output_path=onnx_output,
                opset=opset,
                simplify=simplify,
                dynamic=False,  # TensorRT prefers static shapes in ONNX
                half=False,  # TensorRT handles precision internally
            )

        # --- dispatch to format-specific method ---
        method = getattr(self, fmt_info["method"])
        try:
            if fmt == "tensorrt":
                print(f"Step 2/2: Building TensorRT engine")
            result = method(
                nn_model,
                dummy,
                output_path=output_path,
                opset=opset,
                simplify=simplify,
                dynamic=dynamic,
                half=half,
                int8=int8,
                onnx_path=onnx_path,
                calibration_data=calibration_data,
                workspace=workspace,
                hardware_compatibility=hardware_compatibility,
                gpu_device=gpu_device,
                trt_config=trt_config,
                verbose=verbose,
            )
        finally:
            # restore model state
            nn_model.to(original_device)
            if half and fmt == "onnx":
                nn_model.float()
            if original_training:
                nn_model.train()
            if original_export is not None:
                nn_model.detect.export = original_export

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

    # ------------------------------------------------------------------
    # ONNX
    # ------------------------------------------------------------------

    def _export_onnx(
        self,
        nn_model,
        dummy,
        *,
        output_path: str,
        opset: int,
        simplify: bool,
        dynamic: bool,
        half: bool,
        **_kwargs,
    ) -> str:
        if importlib.util.find_spec("onnx") is None:
            raise ImportError(
                "ONNX export requires the 'onnx' package. "
                "Install with: uv sync --extra onnx  or  pip install onnx"
            )

        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                "images": {0: "batch"},
                "output": {0: "batch"},
            }

        # Use legacy exporter (dynamo=False) for compatibility with models
        # that have complex shape operations not supported by torch.export
        export_kwargs = {
            "export_params": True,
            "opset_version": opset,
            "do_constant_folding": True,
            "input_names": ["images"],
            "output_names": ["output"],
            "dynamic_axes": dynamic_axes,
        }

        # PyTorch 2.1+ defaults to dynamo-based export which can fail on
        # complex models. Use legacy exporter for better compatibility.
        try:
            # Try with dynamo=False for PyTorch 2.1+
            torch.onnx.export(nn_model, dummy, output_path, dynamo=False, **export_kwargs)
        except TypeError:
            # Older PyTorch versions don't have dynamo parameter
            torch.onnx.export(nn_model, dummy, output_path, **export_kwargs)

        self._postprocess_onnx(output_path, simplify=simplify, dynamic=dynamic, half=half)

        return output_path

    def _postprocess_onnx(self, path: str, *, simplify: bool, dynamic: bool, half: bool) -> None:
        """Load the ONNX file once, optionally simplify, embed metadata, and save."""
        try:
            import onnx
        except ImportError:
            return

        model_proto = onnx.load(path)

        # Simplify graph
        if simplify:
            try:
                from onnxsim import simplify as onnx_simplify

                simplified, ok = onnx_simplify(model_proto)
                if ok:
                    model_proto = simplified
            except ImportError:
                warnings.warn(
                    "onnxsim is not installed — skipping ONNX graph simplification. "
                    "Install with: pip install onnxsim",
                    stacklevel=3,
                )
            except Exception as exc:
                warnings.warn(
                    f"ONNX simplification failed (non-fatal): {exc}",
                    stacklevel=3,
                )

        # Embed metadata
        metadata = {
            "libreyolo_version": self._get_version(),
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

        for key, value in metadata.items():
            entry = model_proto.metadata_props.add()
            entry.key = key
            entry.value = value

        onnx.checker.check_model(model_proto)
        onnx.save(model_proto, path)

    @staticmethod
    def _get_version() -> str:
        try:
            from importlib.metadata import version

            return version("libreyolo")
        except Exception:
            return "0.0.0.dev0"

    # ------------------------------------------------------------------
    # TorchScript
    # ------------------------------------------------------------------

    def _export_torchscript(
        self,
        nn_model,
        dummy,
        *,
        output_path: str,
        **_kwargs,
    ) -> str:
        traced = torch.jit.trace(nn_model, dummy)
        traced.save(output_path)
        return output_path

    # ------------------------------------------------------------------
    # TensorRT
    # ------------------------------------------------------------------

    def _export_tensorrt(
        self,
        nn_model,
        dummy,
        *,
        output_path: str,
        onnx_path: str,
        half: bool,
        int8: bool,
        calibration_data,
        workspace: float,
        hardware_compatibility: str,
        gpu_device: int,
        trt_config: Optional[str],
        verbose: bool,
        dynamic: bool,
        **_kwargs,
    ) -> str:
        """Export to TensorRT engine format."""
        from .tensorrt_export import export_tensorrt

        if int8:
            precision = "int8"
        elif half:
            precision = "fp16"
        else:
            precision = "fp32"

        metadata = {
            "libreyolo_version": self._get_version(),
            "model_family": self.model._get_model_name(),
            "model_size": self.model.size,
            "nb_classes": self.model.nb_classes,
            "names": {str(k): v for k, v in self.model.names.items()},
            "imgsz": self.model._get_input_size(),
            "precision": precision,
            "dynamic": dynamic,
            "exported_from": str(Path(onnx_path).name) if onnx_path else None,
        }

        return export_tensorrt(
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

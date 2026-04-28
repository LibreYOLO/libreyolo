"""Unified model export with multiple backend support.

BaseExporter ABC with one subclass per format. Each subclass only
implements ``_export()``, while the template method in ``__call__`` handles
validation, model setup/teardown, calibration, and intermediate ONNX export.
"""

import json
import logging
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch

from .onnx import _get_version, export_onnx
from .torchscript import export_torchscript

logger = logging.getLogger(__name__)


# Precision helpers


def _resolve_precision(half: bool, int8: bool) -> str:
    if int8:
        return "int8"
    if half:
        return "fp16"
    return "fp32"


def _precision_label(precision: str) -> str:
    return precision.upper()


# =============================================================================
# BaseExporter ABC
# =============================================================================


class BaseExporter(ABC):
    """Abstract base for all export formats.

    Subclasses set class-level attributes and implement ``_export()``.
    The ``__call__`` template method handles everything else.

    Example::

        from libreyolo.export import BaseExporter

        exporter = BaseExporter.create("onnx", model)
        path = exporter(output_path="model.onnx")

        # Or instantiate directly:
        from libreyolo.export import OnnxExporter
        path = OnnxExporter(model)(simplify=True, dynamic=True)
    """

    _registry: dict[str, type["BaseExporter"]] = {}

    # Class attributes (overridden by each subclass)
    format_name: str  # e.g. "onnx"
    suffix: str  # e.g. ".onnx"
    requires_onnx: bool  # TensorRT/OpenVINO need intermediate ONNX
    supports_int8: bool  # only TensorRT/OpenVINO support INT8 calibration
    supports_fp16: bool  # whether the format supports FP16 export
    apply_model_half: bool  # whether to cast model to fp16 (only ONNX/TorchScript)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "format_name", None)
        if name is not None:
            BaseExporter._registry[name] = cls

    def __init__(self, model):
        self.model = model

    # Factory

    @classmethod
    def create(cls, format: str, model) -> "BaseExporter":
        """Look up *format* in the registry and return an exporter instance."""
        key = format.lower()
        if key not in cls._registry:
            valid = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Unsupported export format: {format!r}. Must be one of: {valid}"
            )
        return cls._registry[key](model)

    # Template method

    def __call__(
        self,
        *,
        output_path: Optional[str] = None,
        imgsz: Optional[int] = None,
        opset: Optional[int] = None,
        simplify: bool = True,
        dynamic: bool = True,
        half: bool = False,
        int8: bool = False,
        batch: int = 1,
        device: Optional[str] = None,
        data: Optional[str] = None,
        fraction: float = 1.0,
        allow_download_scripts: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> str:
        """Export the model.

        Args:
            output_path: Output file path (auto-generated if None).
            imgsz: Input resolution (default: model's native size).
            opset: ONNX opset version (default: 13).
            simplify: Run ONNX graph simplification (default: True).
            dynamic: Enable dynamic axes for ONNX (default: True).
            half: Export in FP16 precision (default: False).
            int8: Export in INT8 precision (default: False).
            batch: Batch size for the model (default: 1).
            device: Device to trace on (default: model's current device).
            data: Path to data.yaml for INT8 calibration dataset.
            fraction: Fraction of calibration dataset to use (default: 1.0).
            allow_download_scripts: Allow embedded Python in dataset YAML downloads.
            verbose: Enable verbose logging (default: False).
            **kwargs: Format-specific parameters forwarded to ``_export()``.

        Returns:
            Path to the exported model file.
        """
        half, int8 = self._validate(half, int8, data)

        if opset is None:
            # D-FINE and ECDet use ``F.grid_sample`` (deformable attention)
            # which requires opset 16+. Default the rest of the families to
            # 13 to preserve compatibility with the broadest set of ONNX
            # runtimes.
            opset = 17 if self.model._get_model_name() in ("dfine", "ecdet") else 13

        imgsz, device, output_path = self._resolve_params(
            output_path,
            imgsz,
            device,
            half,
            int8,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        precision = _resolve_precision(half, int8)

        with self._model_context(device, half, batch, imgsz) as (nn_model, dummy):
            calibration_data = (
                self._load_calibration(
                    data,
                    imgsz,
                    batch,
                    fraction,
                    allow_download_scripts,
                )
                if int8 and data is not None
                else None
            )

            onnx_path = (
                self._export_intermediate_onnx(
                    nn_model,
                    dummy,
                    output_path,
                    opset,
                    simplify,
                )
                if self.requires_onnx
                else None
            )

            metadata = self._build_metadata(precision, dynamic, onnx_path)

            result = self._export(
                nn_model,
                dummy,
                output_path=output_path,
                precision=precision,
                metadata=metadata,
                calibration_data=calibration_data,
                onnx_path=onnx_path,
                half=half,
                int8=int8,
                dynamic=dynamic,
                opset=opset,
                simplify=simplify,
                verbose=verbose,
                **kwargs,
            )

        if onnx_path and Path(onnx_path).exists():
            Path(onnx_path).unlink()

        self._print_summary(result, precision, imgsz)
        return result

    # Abstract export method

    @abstractmethod
    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path: str,
        precision: str = "fp32",
        metadata: dict | None = None,
        calibration_data=None,
        onnx_path: str | None = None,
        half: bool = False,
        int8: bool = False,
        dynamic: bool = False,
        opset: int = 13,
        simplify: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> str:
        """Format-specific export logic. Subclasses implement this only."""

    # Shared helpers

    def _validate(self, half: bool, int8: bool, data: Optional[str]):
        """Validate precision flags and calibration requirements."""
        if int8 and data is None and self.supports_int8:
            raise ValueError(
                "INT8 quantization requires calibration data.\n"
                "Provide data='path/to/data.yaml' or data='coco8' for built-in dataset."
            )
        if half and int8:
            warnings.warn(
                "Both half=True and int8=True specified. Using INT8 precision."
            )
            half = False
        return half, int8

    def _resolve_params(self, output_path, imgsz, device, half, int8):
        if imgsz is None:
            imgsz = self.model._get_input_size()
        if device is None:
            device = self.model.device
        else:
            device = torch.device(device)
        if output_path is None:
            output_path = self._auto_output_path(half, int8)
        return imgsz, device, output_path

    def _auto_output_path(self, half: bool, int8: bool) -> str:
        model_name = self.model._get_model_name().lower()
        precision_suffix = "_int8" if int8 else ("_fp16" if half else "")
        return str(
            Path("weights")
            / f"{model_name}_{self.model.size}{precision_suffix}{self.suffix}"
        )

    @contextmanager
    def _model_context(self, device, half, batch, imgsz):
        """Setup model for export and restore state afterwards."""
        nn_model = self.model.model
        original_training = nn_model.training
        nn_model.eval()

        original_device = next(nn_model.parameters()).device
        nn_model.to(device)

        # DETR-family export mode: wrap model so it returns a tuple instead
        # of dict and apply ``model.deploy()`` (BN fusion + prune non-eval
        # decoder layers). The wrapper is what gets traced; the original
        # model is restored on exit.
        dfine_wrapped = False
        family = self.model._get_model_name()
        if family == "dfine":
            from ..models.dfine.nn import DFINEExportWrapper

            nn_model = DFINEExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "ecdet":
            from ..models.ecdet.nn import ECDetExportWrapper

            nn_model = ECDetExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True  # share the YOLOX-head-export skip path below

        # Set export mode for YOLOX/YOLOv9 heads
        original_export = None
        export_attr = None
        if (
            not dfine_wrapped
            and hasattr(nn_model, "head")
            and hasattr(nn_model.head, "export")
        ):
            export_attr = "head"
            original_export = nn_model.head.export
            nn_model.head.export = True

        # RF-DETR export mode
        rfdetr_export_activated = False
        rfdetr_layernorm_patches = []
        inner = getattr(nn_model, "model", None)
        if (
            inner is not None
            and hasattr(inner, "forward_export")
            and hasattr(inner, "_export")
        ):
            if not inner._export:
                inner.export()
                rfdetr_export_activated = True

            try:
                from rfdetr.models.backbone.projector import (
                    LayerNorm as RFDETRLayerNorm,
                )

                for m in nn_model.modules():
                    if isinstance(m, RFDETRLayerNorm):
                        rfdetr_layernorm_patches.append((m, m.forward))
                        ns = m.normalized_shape

                        def _static_forward(
                            x, _ns=ns, _w=m.weight, _b=m.bias, _eps=m.eps
                        ):
                            x = x.permute(0, 2, 3, 1)
                            x = torch.nn.functional.layer_norm(x, _ns, _w, _b, _eps)
                            return x.permute(0, 3, 1, 2)

                        m.forward = _static_forward
            except ImportError:
                pass

        dummy = torch.randn(batch, 3, imgsz, imgsz, device=device)

        if half and self.apply_model_half:
            nn_model.half()
            dummy = dummy.half()

        try:
            yield nn_model, dummy
        finally:
            nn_model.to(original_device)
            if half and self.apply_model_half:
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

    def _load_calibration(
        self,
        data,
        imgsz,
        batch,
        fraction,
        allow_download_scripts=False,
    ):
        from .calibration import get_calibration_dataloader

        preprocess_fn = self.model._get_preprocess_numpy()
        calibration_data = get_calibration_dataloader(
            data=data,
            imgsz=imgsz,
            batch=batch,
            fraction=fraction,
            preprocess_fn=preprocess_fn,
            allow_download_scripts=allow_download_scripts,
        )
        logger.info(
            "Calibration dataset: %d batches, %d images",
            len(calibration_data),
            calibration_data.num_samples,
        )
        return calibration_data

    def _export_intermediate_onnx(self, nn_model, dummy, output_path, opset, simplify):
        onnx_output = str(Path(output_path).with_suffix(".onnx"))
        logger.info("Step 1/2: Exporting to ONNX (%s)", onnx_output)
        return export_onnx(
            nn_model,
            dummy,
            output_path=onnx_output,
            opset=opset,
            simplify=simplify,
            dynamic=False,
            half=False,
            metadata=self._build_onnx_metadata(dynamic=False, half=False),
        )

    def _build_metadata(
        self, precision: str, dynamic: bool, onnx_path: Optional[str]
    ) -> dict:
        """Build metadata dict for non-ONNX formats (native Python types)."""
        meta = {
            "libreyolo_version": _get_version(),
            "model_family": self.model._get_model_name(),
            "model_size": self.model.size,
            "nb_classes": self.model.nb_classes,
            "names": {str(k): v for k, v in self.model.names.items()},
            "imgsz": self.model._get_input_size(),
            "precision": precision,
            "dynamic": dynamic,
        }
        if onnx_path is not None:
            meta["exported_from"] = str(Path(onnx_path).name)
        return meta

    def _build_onnx_metadata(self, *, dynamic: bool, half: bool) -> dict:
        """Build metadata dict for ONNX (all-string values, JSON-encoded names)."""
        return {
            "libreyolo_version": _get_version(),
            "model_family": self.model._get_model_name(),
            "model_size": self.model.size,
            "nb_classes": str(self.model.nb_classes),
            "names": json.dumps({str(k): v for k, v in self.model.names.items()}),
            "imgsz": str(self.model._get_input_size()),
            "dynamic": str(dynamic),
            "half": str(half),
            "segmentation": str(getattr(self.model, "_is_segmentation", False)).lower(),
        }

    def _print_summary(self, result: str, precision: str, imgsz: int):
        logger.info(
            "Export complete: %s\n"
            "  Model: %s %s\n"
            "  Format: %s\n"
            "  Precision: %s\n"
            "  Input size: %dx%d",
            result,
            self.model._get_model_name(),
            self.model.size,
            self.format_name,
            _precision_label(precision),
            imgsz,
            imgsz,
        )


# =============================================================================
# Subclasses — one per format
# =============================================================================


class OnnxExporter(BaseExporter):
    format_name = "onnx"
    suffix = ".onnx"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = True
    apply_model_half = True

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        half,
        dynamic,
        opset,
        simplify,
        **kwargs,
    ):
        return export_onnx(
            nn_model,
            dummy,
            output_path=output_path,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            half=half,
            metadata=self._build_onnx_metadata(dynamic=dynamic, half=half),
        )


class TorchScriptExporter(BaseExporter):
    format_name = "torchscript"
    suffix = ".torchscript"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = True
    apply_model_half = True

    def _export(self, nn_model, dummy, *, output_path, metadata, **kwargs):
        return export_torchscript(
            nn_model, dummy, output_path=output_path, metadata=metadata
        )


class TensorRTExporter(BaseExporter):
    format_name = "tensorrt"
    suffix = ".engine"
    requires_onnx = True
    supports_int8 = True
    supports_fp16 = True
    apply_model_half = False

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        precision,
        metadata,
        calibration_data,
        onnx_path,
        half,
        int8,
        dynamic,
        verbose,
        workspace=4.0,
        hardware_compatibility="none",
        gpu_device=0,
        trt_config=None,
        **kwargs,
    ):
        from .tensorrt import export_tensorrt

        logger.info("Step 2/2: Building TensorRT engine")
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


class OpenVINOExporter(BaseExporter):
    format_name = "openvino"
    suffix = "_openvino"
    requires_onnx = True
    supports_int8 = True
    supports_fp16 = True
    apply_model_half = False

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        calibration_data,
        onnx_path,
        half,
        int8,
        verbose,
        **kwargs,
    ):
        from .openvino import export_openvino

        logger.info("Step 2/2: Converting to OpenVINO IR")
        return export_openvino(
            onnx_path=onnx_path,
            output_path=output_path,
            half=half,
            int8=int8,
            calibration_data=calibration_data,
            verbose=verbose,
            metadata=metadata,
        )


class NcnnExporter(BaseExporter):
    format_name = "ncnn"
    suffix = "_ncnn"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False

    def _build_metadata(self, precision, dynamic, onnx_path):
        meta = super()._build_metadata(precision, dynamic, onnx_path)
        meta["dynamic"] = False
        meta.pop("exported_from", None)
        return meta

    def _export(
        self, nn_model, dummy, *, output_path, metadata, half, opset, simplify, **kwargs
    ):
        # NCNN can't handle DETR-style query selection: its op registry doesn't
        # include the topk/gather variants used by D-FINE and RT-DETR decoders.
        # Block early rather than producing a broken export directory.
        unsupported_family_names = {
            "dfine": "D-FINE",
            "rtdetr": "RT-DETR",
            "ecdet": "ECDet",
        }
        model_family = metadata.get("model_family") if metadata else None
        if model_family in unsupported_family_names:
            raise NotImplementedError(
                f"NCNN export is not supported for "
                f"{unsupported_family_names[model_family]}: NCNN's op registry "
                "lacks topk/gather variants that the DETR-style decoder "
                "requires. Use ONNX, OpenVINO, TorchScript, or TensorRT instead."
            )

        from .ncnn import export_ncnn

        logger.info("Exporting to ncnn via PNNX")
        return export_ncnn(
            nn_model,
            dummy,
            output_path=output_path,
            half=half,
            opset=opset,
            simplify=simplify,
            metadata=metadata,
        )

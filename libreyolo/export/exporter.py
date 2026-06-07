"""Unified model export with multiple backend support.

BaseExporter ABC with one subclass per format. Each subclass only
implements ``_export()``, while the template method in ``__call__`` handles
validation, model setup/teardown, calibration, and intermediate ONNX export.
"""

import copy
import importlib.util
import json
import logging
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from .onnx import (
    _get_version,
    _uses_dfine_style_export_wrapper,
    check_onnx_int8_available,
    export_onnx,
    quantize_onnx_int8,
)
from .torchscript import export_torchscript
from ..utils.serialization import SCHEMA_VERSION

logger = logging.getLogger(__name__)

DEFAULT_INT8_CALIBRATION_DATA = "coco8.yaml"


# Precision helpers


def _resolve_precision(half: bool, int8: bool) -> str:
    if int8:
        return "int8"
    if half:
        return "fp16"
    return "fp32"


def _precision_label(precision: str) -> str:
    return precision.upper()


def _is_rectangular_imgsz(imgsz: tuple[int, int]) -> bool:
    return int(imgsz[0]) != int(imgsz[1])


def _snapshot_rfdetr_export_state(root):
    """Capture RF-DETR export mutations so the live model can be restored."""
    snapshots = []
    if root is None or not hasattr(root, "modules"):
        return snapshots

    for module in root.modules():
        encoder = getattr(module, "encoder", None)
        embeddings = getattr(encoder, "embeddings", None)
        has_export_state = hasattr(module, "_export")
        has_position_state = embeddings is not None and hasattr(
            embeddings, "position_embeddings"
        )
        if not has_export_state and not has_position_state:
            continue

        state = {
            "forward": getattr(module, "forward", None),
            "had_forward_origin": hasattr(module, "_forward_origin"),
            "forward_origin": getattr(module, "_forward_origin", None),
        }
        if has_export_state:
            state["export"] = getattr(module, "_export")
        if hasattr(module, "shape"):
            state["shape"] = getattr(module, "shape")
        if has_position_state:
            state["position_embeddings"] = embeddings.position_embeddings
            state["interpolate_pos_encoding"] = embeddings.interpolate_pos_encoding
        snapshots.append((module, state))
    return snapshots


def _restore_rfdetr_export_state(snapshots):
    for module, state in reversed(snapshots):
        encoder = getattr(module, "encoder", None)
        embeddings = getattr(encoder, "embeddings", None)
        if embeddings is not None and "position_embeddings" in state:
            embeddings.position_embeddings = state["position_embeddings"]
            embeddings.interpolate_pos_encoding = state["interpolate_pos_encoding"]
        if "shape" in state:
            module.shape = state["shape"]
        if state.get("forward") is not None:
            module.forward = state["forward"]
        if state.get("had_forward_origin"):
            module._forward_origin = state["forward_origin"]
        elif hasattr(module, "_forward_origin"):
            delattr(module, "_forward_origin")
        if "export" in state:
            module._export = state["export"]


_FIXED_SQUARE_EXPORT_FAMILIES = {
    "dfine",
    "deim",
    "deimv2",
    "ec",
    "rtdetr",
    "rtdetrv2",
    "rtdetrv4",
    "rfdetr",
}
_RECTANGULAR_EXPORT_FAMILIES = {"yolo9", "yolo9_e2e"}
_RECTANGULAR_EXPORT_FORMATS = {
    "coreml",
    "ncnn",
    "onnx",
    "openvino",
    "tensorrt",
    "tflite",
    "torchscript",
}


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
    supports_int8: bool  # whether the format supports INT8 calibration
    supports_fp16: bool  # whether the format supports FP16 export
    apply_model_half: bool  # whether to cast model to fp16 (only ONNX/TorchScript)
    supports_embedded_nms: bool = False
    default_int8_calibration_data: bool = False

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
        imgsz: Optional[Union[int, Tuple[int, int]]] = None,
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
            imgsz: Input resolution as ``(height, width)`` tuple or a single
                int for square (default: model's native size).
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
        if getattr(self.model, "task", "detect") == "point":
            raise NotImplementedError(
                "Export for point-task models is not implemented yet. "
                "Add a point-aware export/runtime contract before exporting point models."
            )
        half, int8 = self._validate(half, int8, data)
        self._preflight(half=half, int8=int8, data=data, **kwargs)
        data = self._resolve_calibration_data(int8, data)

        if opset is None:
            # DETR-style families use deformable attention / layer norm ops
            # which require opset 16+ (or 17 for ``aten::scaled_dot_product``
            # in the tuple export wrapper). Other families default to 13.
            opset = (
                17
                if _uses_dfine_style_export_wrapper(self.model._get_model_name())
                else 13
            )

        imgsz, device, output_path = self._resolve_params(
            output_path,
            imgsz,
            device,
            half,
            int8,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        precision = _resolve_precision(half, int8)
        onnx_path = None

        try:
            with self._model_context(device, half, int8, batch, imgsz) as (
                nn_model,
                dummy,
            ):
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
                        dynamic,
                    )
                    if self.requires_onnx
                    else None
                )

                metadata = self._build_metadata(
                    precision,
                    dynamic,
                    onnx_path,
                    imgsz=imgsz,
                )

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
        finally:
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
        if half and int8:
            warnings.warn(
                "Both half=True and int8=True specified. Using INT8 precision.",
                stacklevel=2,
            )
            half = False
        if int8 and not self.supports_int8:
            raise NotImplementedError(
                f"{self.format_name.upper()} INT8 export is not supported."
            )
        if int8 and data is None and not self.default_int8_calibration_data:
            raise ValueError("INT8 export requires calibration data. Pass data=...")
        return half, int8

    def _resolve_calibration_data(self, int8: bool, data: Optional[str]) -> Optional[str]:
        """Apply the default INT8 calibration dataset when data is omitted."""
        if not int8 or data is not None or not self.default_int8_calibration_data:
            return data
        logger.warning(
            "INT8 export requested without calibration data; using %s. "
            "This 8-image fallback is not representative. For accuracy validation, "
            "use a calibration dataset with roughly 300 or more representative images.",
            DEFAULT_INT8_CALIBRATION_DATA,
        )
        return DEFAULT_INT8_CALIBRATION_DATA

    def _preflight(self, *, half: bool, int8: bool, data: Optional[str], **kwargs):
        """Run cheap format-specific checks before model or calibration setup."""
        if kwargs.get("nms") and not self.supports_embedded_nms:
            raise NotImplementedError(
                f"{self.format_name.upper()} embedded NMS export is not supported."
            )
        if self.requires_onnx and importlib.util.find_spec("onnx") is None:
            raise ImportError(
                "ONNX export requires the 'onnx' package. "
                "Install with: uv sync --extra onnx  or  pip install onnx"
            )

    def _resolve_params(self, output_path, imgsz, device, half, int8):
        native_imgsz = self.model._get_input_size()
        model_name = self.model._get_model_name()
        if imgsz is None:
            imgsz = (native_imgsz, native_imgsz)
        elif isinstance(imgsz, tuple):
            if len(imgsz) != 2:
                raise ValueError(f"imgsz tuple must be (height, width), got {imgsz}")
            imgsz = (int(imgsz[0]), int(imgsz[1]))
        else:
            imgsz = (int(imgsz), int(imgsz))
        if imgsz[0] <= 0 or imgsz[1] <= 0:
            raise ValueError(f"imgsz values must be positive, got {imgsz}.")
        if model_name == "deimv2" and imgsz != (
            int(native_imgsz),
            int(native_imgsz),
        ):
            raise ValueError(
                "DEIMv2 export uses fixed decoder anchors; imgsz must match "
                f"the native size {native_imgsz}, got {imgsz}."
            )
        if _is_rectangular_imgsz(imgsz) and model_name in _FIXED_SQUARE_EXPORT_FAMILIES:
            raise NotImplementedError(
                f"Rectangular imgsz export is not supported for {model_name}: "
                "this family uses a fixed square export/preprocessing spatial contract. "
                "Use the native square imgsz for now."
            )
        if (
            _is_rectangular_imgsz(imgsz)
            and model_name not in _RECTANGULAR_EXPORT_FAMILIES
        ):
            raise NotImplementedError(
                "Rectangular imgsz export is currently supported for "
                "YOLO9-family exports only."
            )
        if (
            _is_rectangular_imgsz(imgsz)
            and self.format_name not in _RECTANGULAR_EXPORT_FORMATS
        ):
            raise NotImplementedError(
                f"Rectangular imgsz export is not validated for format "
                f"{self.format_name!r}."
            )
        if device is None or str(device).lower() == "auto":
            if self.model._get_model_name() == "rfdetr":
                device = torch.device("cpu")
            else:
                device = self.model.device
        else:
            if isinstance(device, int):
                device = f"cuda:{device}"
            elif isinstance(device, str) and device.isdigit():
                device = f"cuda:{device}"
            device = torch.device(device)
        if output_path is None:
            output_path = self._auto_output_path(half, int8)
        return imgsz, device, output_path

    def _auto_output_path(self, half: bool, int8: bool) -> str:
        model_name = self.model._get_model_name().lower()
        task = getattr(self.model, "task", "detect")
        is_segment = (
            task == "segment" or getattr(self.model, "_is_segmentation", False) is True
        )
        if is_segment:
            task_suffix = "_seg"
        elif task == "obb":
            task_suffix = "_obb"
        elif task == "classify":
            task_suffix = "_cls"
        else:
            task_suffix = ""
        precision_suffix = "_int8" if int8 else ("_fp16" if half else "")
        return str(
            Path("weights")
            / f"{model_name}_{self.model.size}{task_suffix}{precision_suffix}{self.suffix}"
        )

    @contextmanager
    def _model_context(self, device, half, int8, batch, imgsz):
        """Setup model for export and restore state afterwards."""
        nn_model = self.model.model
        root_model = nn_model
        original_training = root_model.training
        root_model.eval()

        original_device = next(root_model.parameters()).device
        root_model.to(device)

        # DETR-family export mode: wrap model so it returns a tuple instead
        # of dict and apply ``model.deploy()`` (BN fusion + prune non-eval
        # decoder layers). The wrapper is what gets traced; the original
        # model is restored on exit.
        dfine_wrapped = False
        rfdetr_export_activated = False
        rfdetr_export_snapshots = []
        rfdetr_inner = None
        family = self.model._get_model_name()
        if family == "dfine":
            from ..models.dfine.nn import DFINEExportWrapper

            nn_model = DFINEExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "deim":
            from ..models.deim.nn import DEIMExportWrapper

            nn_model = DEIMExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "deimv2":
            from ..models.deimv2.nn import DEIMv2ExportWrapper

            nn_model = copy.deepcopy(nn_model)
            nn_model = DEIMv2ExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "ec":
            from ..models.ec.nn import ECExportWrapper

            nn_model = ECExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True  # share the YOLOX-head-export skip path below
        elif family == "rfdetr" and getattr(self.model, "task", None) == "classify":
            # Classification has no detection decoder; trace the backbone +
            # linear classifier directly (it returns logits). The detection
            # export wrapper forwards through ``model.model``, which is None
            # for classification.
            nn_model = nn_model.classifier.to(device)
            nn_model.eval()
            # Precompute static DINOv2 positional encodings for the fixed export
            # resolution; otherwise the dynamic bicubic-antialias interpolation
            # in the backbone is not ONNX-traceable.
            encoder = getattr(getattr(nn_model, "backbone", None), "encoder", None)
            if (
                encoder is not None
                and hasattr(encoder, "export")
                and not getattr(encoder, "_export", False)
            ):
                rfdetr_export_snapshots = _snapshot_rfdetr_export_state(nn_model)
                encoder.shape = (imgsz[0], imgsz[1])
                encoder.export()
                rfdetr_export_activated = True
            dfine_wrapped = True
        elif family == "rfdetr":
            from ..models.rfdetr.nn import RFDETRExportWrapper

            rfdetr_inner = getattr(nn_model, "model", None)
            was_exported = getattr(rfdetr_inner, "_export", False)
            if not was_exported:
                rfdetr_export_snapshots = _snapshot_rfdetr_export_state(rfdetr_inner)
            nn_model = RFDETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
            rfdetr_export_activated = not was_exported

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
        rfdetr_layernorm_patches = []
        inner = rfdetr_inner or getattr(nn_model, "model", None)
        if (
            inner is not None
            and hasattr(inner, "forward_export")
            and hasattr(inner, "_export")
        ):
            if not inner._export:
                inner.export()
                rfdetr_export_activated = True

            try:
                from ..models.rfdetr.backbone import LayerNorm as RFDETRLayerNorm

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

        h, w = imgsz
        dummy = torch.randn(batch, 3, h, w, device=device)

        if half and not int8 and self.apply_model_half:
            nn_model.half()
            dummy = dummy.half()

        try:
            yield nn_model, dummy
        finally:
            if rfdetr_export_snapshots:
                _restore_rfdetr_export_state(rfdetr_export_snapshots)
            nn_model.to(original_device)
            root_model.to(original_device)
            if half and not int8 and self.apply_model_half:
                nn_model.float()
                root_model.float()
            if original_training:
                root_model.train()
                nn_model.train()
            if original_export is not None:
                getattr(nn_model, export_attr).export = original_export
            if (
                rfdetr_export_activated
                and not rfdetr_export_snapshots
                and inner is not None
            ):
                for module in inner.modules():
                    if hasattr(module, "_forward_origin"):
                        module.forward = module._forward_origin
                    if hasattr(module, "_export"):
                        module._export = False
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

    def _export_intermediate_onnx(
        self, nn_model, dummy, output_path, opset, simplify, dynamic
    ):
        onnx_output = str(Path(output_path).with_suffix(".onnx"))
        logger.info("Step 1/2: Exporting to ONNX (%s)", onnx_output)
        return export_onnx(
            nn_model,
            dummy,
            output_path=onnx_output,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            half=False,
            metadata=self._build_onnx_metadata(
                dynamic=dynamic,
                half=False,
                imgsz=(dummy.shape[-2], dummy.shape[-1]),
            ),
        )

    def _build_metadata(
        self,
        precision: str,
        dynamic: bool,
        onnx_path: Optional[str],
        imgsz: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> dict:
        """Build metadata dict for non-ONNX formats (native Python types)."""
        task, supported_tasks, default_task = self._task_metadata()
        if imgsz is not None:
            if isinstance(imgsz, tuple):
                h, w = imgsz
                metadata_imgsz = max(h, w)
                meta_h, meta_w = h, w
            else:
                metadata_imgsz = int(imgsz)
                meta_h = meta_w = int(imgsz)
        else:
            native = self.model._get_input_size()
            metadata_imgsz = int(native)
            meta_h = meta_w = int(native)
        # TODO(schema-v1.1): keep legacy model_size/nb_classes aliases for one
        # transition window, then prefer the canonical size/nc keys only.
        meta = {
            "schema_version": SCHEMA_VERSION,
            "libreyolo_version": _get_version(),
            "model_family": self.model._get_model_name(),
            "size": self.model.size,
            "model_size": self.model.size,
            "task": task,
            "supported_tasks": supported_tasks,
            "default_task": default_task,
            "nc": self.model.nb_classes,
            "nb_classes": self.model.nb_classes,
            "names": {str(k): v for k, v in self.model.names.items()},
            "imgsz": metadata_imgsz,
            "imgsz_h": meta_h,
            "imgsz_w": meta_w,
            "precision": precision,
            "dynamic": dynamic,
            "obb": task == "obb",
        }
        if onnx_path is not None:
            meta["exported_from"] = str(Path(onnx_path).name)
        return meta

    def _build_onnx_metadata(
        self,
        *,
        dynamic: bool,
        half: bool,
        imgsz: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> dict:
        """Build metadata dict for ONNX (all-string values, JSON-encoded names)."""
        task, supported_tasks, default_task = self._task_metadata()
        if imgsz is not None:
            if isinstance(imgsz, tuple):
                h, w = imgsz
                metadata_imgsz = str(max(h, w))
                meta_h = str(h)
                meta_w = str(w)
            else:
                metadata_imgsz = str(int(imgsz))
                meta_h = meta_w = str(int(imgsz))
        else:
            native = self.model._get_input_size()
            metadata_imgsz = str(native)
            meta_h = meta_w = str(native)
        # TODO(schema-v1.1): keep legacy model_size/nb_classes aliases for one
        # transition window, then prefer the canonical size/nc keys only.
        return {
            "schema_version": SCHEMA_VERSION,
            "libreyolo_version": _get_version(),
            "model_family": self.model._get_model_name(),
            "size": self.model.size,
            "model_size": self.model.size,
            "task": task,
            "supported_tasks": json.dumps(supported_tasks),
            "default_task": default_task,
            "nc": str(self.model.nb_classes),
            "nb_classes": str(self.model.nb_classes),
            "names": json.dumps({str(k): v for k, v in self.model.names.items()}),
            "imgsz": metadata_imgsz,
            "imgsz_h": meta_h,
            "imgsz_w": meta_w,
            "dynamic": str(dynamic),
            "precision": "fp16" if half else "fp32",
            "half": str(half),
            "segmentation": str(getattr(self.model, "_is_segmentation", False)).lower(),
            "obb": str(task == "obb").lower(),
        }

    def _task_metadata(self) -> tuple[str, list[str], str]:
        task = getattr(self.model, "task", "detect")
        if not isinstance(task, str):
            task = "detect"
        supported_tasks = getattr(self.model, "SUPPORTED_TASKS", ("detect",))
        if not isinstance(supported_tasks, (list, tuple)):
            supported_tasks = ("detect",)
        default_task = getattr(self.model, "DEFAULT_TASK", "detect")
        if not isinstance(default_task, str):
            default_task = "detect"
        if self.model._get_model_name() == "rfdetr":
            return task, [task], task
        return task, list(supported_tasks), default_task

    def _print_summary(
        self, result: str, precision: str, imgsz: Union[int, Tuple[int, int]]
    ):
        if isinstance(imgsz, tuple):
            h, w = imgsz
        else:
            h = w = imgsz
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
            w,
            h,
        )


# =============================================================================
# Subclasses — one per format
# =============================================================================


class OnnxExporter(BaseExporter):
    format_name = "onnx"
    suffix = ".onnx"
    requires_onnx = False
    supports_int8 = True
    supports_fp16 = True
    apply_model_half = True
    supports_embedded_nms = True
    default_int8_calibration_data = True

    def _preflight(self, *, half: bool, int8: bool, data: Optional[str], **kwargs):
        if int8:
            task = getattr(self.model, "task", "detect")
            if not isinstance(task, str):
                task = "detect"
            if self.model._get_model_name() != "yolo9" or task != "detect":
                raise NotImplementedError(
                    "ONNX INT8 export currently supports YOLO9 detection models only."
                )
            check_onnx_int8_available()
        if kwargs.get("nms"):
            task = getattr(self.model, "task", "detect")
            if not isinstance(task, str):
                task = "detect"
            if self.model._get_model_name() != "yolo9" or task != "detect":
                raise NotImplementedError(
                    "Embedded NMS ONNX export currently supports YOLO9 "
                    "detection models only."
                )
        super()._preflight(half=half, int8=int8, data=data, **kwargs)

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        calibration_data,
        half,
        int8,
        dynamic,
        opset,
        simplify,
        nms=False,
        iou=0.45,
        conf=0.25,
        max_det=300,
        calibrate_method="MinMax",
        nodes_to_exclude=None,
        **kwargs,
    ):
        imgsz = (dummy.shape[-2], dummy.shape[-1])

        if nms:
            from .nms import EmbeddedNMSDetector

            if dummy.shape[0] != 1:
                raise NotImplementedError(
                    "Embedded NMS ONNX export currently requires batch=1."
                )
            if dynamic:
                logger.warning(
                    "Embedded NMS uses a fixed batch-1 graph; forcing dynamic=False."
                )
                dynamic = False
            nn_model = EmbeddedNMSDetector(
                nn_model,
                conf=conf,
                iou=iou,
                max_det=max_det,
            ).eval()

        def _onnx_metadata(precision_half: bool) -> dict:
            meta = self._build_onnx_metadata(
                dynamic=dynamic,
                half=precision_half,
                imgsz=imgsz,
            )
            if nms:
                meta["nms"] = "true"
                meta["nms_conf"] = str(conf)
                meta["nms_iou"] = str(iou)
                meta["max_det"] = str(max_det)
                meta["nms_raw_output"] = "true"
            return meta

        if int8:
            import tempfile

            output = Path(output_path)
            int8_metadata = _onnx_metadata(precision_half=False)
            int8_metadata["precision"] = "int8"
            with tempfile.TemporaryDirectory(
                prefix=f"{output.stem}_", dir=str(output.parent)
            ) as tmpdir:
                fp32_path = str(Path(tmpdir) / "model_fp32.onnx")
                preprocessed_path = str(Path(tmpdir) / "model_fp32_infer.onnx")
                export_onnx(
                    nn_model,
                    dummy,
                    output_path=fp32_path,
                    opset=opset,
                    simplify=simplify,
                    dynamic=dynamic,
                    half=False,
                    metadata=_onnx_metadata(precision_half=False),
                    nms=nms,
                )
                return quantize_onnx_int8(
                    fp32_path,
                    output_path,
                    calibration_data=calibration_data,
                    metadata=int8_metadata,
                    preprocessed_path=preprocessed_path,
                    calibrate_method=calibrate_method,
                    nodes_to_exclude=nodes_to_exclude,
                    skip_symbolic_shape=nms,
                )

        return export_onnx(
            nn_model,
            dummy,
            output_path=output_path,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            half=half,
            metadata=_onnx_metadata(precision_half=half),
            nms=nms,
        )


class TorchScriptExporter(BaseExporter):
    format_name = "torchscript"
    suffix = ".torchscript"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = True
    apply_model_half = True

    def _resolve_params(self, output_path, imgsz, device, half, int8):
        if device is None or str(device).lower() == "auto":
            device = torch.device("cpu")
        return super()._resolve_params(output_path, imgsz, device, half, int8)

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

    def _preflight(self, **kwargs):
        super()._preflight(**kwargs)
        from .tensorrt import check_tensorrt_available

        check_tensorrt_available()

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
        min_batch=1,
        opt_batch=1,
        max_batch=8,
        hardware_compatibility="none",
        gpu_device=0,
        trt_config=None,
        **kwargs,
    ):
        from .tensorrt import export_tensorrt

        trt_metadata = dict(metadata or {})
        if dynamic:
            trt_metadata.update(
                {
                    "trt_min_batch": int(min_batch),
                    "trt_opt_batch": int(opt_batch),
                    "trt_max_batch": int(max_batch),
                }
            )

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
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
            hardware_compatibility=hardware_compatibility,
            device=gpu_device,
            config=trt_config,
            metadata=trt_metadata,
        )


class OpenVINOExporter(BaseExporter):
    format_name = "openvino"
    suffix = "_openvino"
    requires_onnx = True
    supports_int8 = True
    supports_fp16 = True
    apply_model_half = False

    def _preflight(self, **kwargs):
        super()._preflight(**kwargs)
        from .openvino import check_openvino_available

        check_openvino_available()

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

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        meta = super()._build_metadata(precision, dynamic, onnx_path, imgsz=imgsz)
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
            "deim": "DEIM",
            "deimv2": "DEIMv2",
            "rtdetr": "RT-DETR",
            "ec": "EC",
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


class TFLiteExporter(BaseExporter):
    format_name = "tflite"
    suffix = ".tflite"
    requires_onnx = True
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False

    def __call__(self, *args, dynamic: bool = False, **kwargs) -> str:
        if dynamic:
            raise ValueError("TFLite export requires static input shapes.")
        from .tflite import ensure_tflite_family_supported

        ensure_tflite_family_supported(
            self.model._get_model_name(),
            getattr(self.model, "task", "detect"),
        )
        return super().__call__(*args, dynamic=False, **kwargs)

    def _validate(self, half: bool, int8: bool, data: Optional[str]):
        if half:
            raise ValueError(
                "TFLite FP16 export is not supported yet. Omit half=True for FP32."
            )
        if int8:
            raise ValueError(
                "TFLite INT8 quantization is not supported yet. "
                "Omit int8=True for FP32."
            )
        return super()._validate(half, int8, data)

    def _preflight(self, **kwargs):
        from .tflite import check_tflite_export_available

        check_tflite_export_available()
        super()._preflight(**kwargs)

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        onnx_path,
        half,
        verbose,
        onnx2tf_args=None,
        **kwargs,
    ):
        from .tflite import ensure_tflite_family_supported, export_tflite

        ensure_tflite_family_supported(
            metadata.get("model_family") if metadata else None,
            metadata.get("task") if metadata else None,
        )

        logger.info("Step 2/2: Converting to TensorFlow Lite")
        return export_tflite(
            onnx_path=onnx_path,
            output_path=output_path,
            half=half,
            verbose=verbose,
            onnx2tf_args=onnx2tf_args,
            metadata=metadata,
        )


class CoreMLExporter(BaseExporter):
    format_name = "coreml"
    suffix = ".mlpackage"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = True
    apply_model_half = False  # ct.convert handles precision via compute_precision
    supports_embedded_nms = True

    def _preflight(self, *, half: bool, int8: bool, data: Optional[str], **kwargs):
        if kwargs.get("nms"):
            family = self.model._get_model_name()
            task = getattr(self.model, "task", "detect")
            if not isinstance(task, str):
                task = "detect"
            if family == "yolo9" and task != "detect":
                raise NotImplementedError(
                    "CoreML embedded NMS currently supports YOLO9 detection "
                    "models only."
                )
            if family not in {"yolox", "yolo9"}:
                raise NotImplementedError(
                    "CoreML embedded NMS currently supports YOLOX and YOLO9 "
                    "detection models only."
                )
            if kwargs.get("max_det", 300) != 300:
                raise NotImplementedError(
                    "CoreML embedded NMS does not support max_det. "
                    "Use ONNX embedded NMS when max_det control is required."
                )
        super()._preflight(half=half, int8=int8, data=data, **kwargs)

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        precision,
        metadata,
        compute_units="all",
        nms=False,
        iou=0.45,
        conf=0.25,
        **kwargs,
    ):
        from .coreml import export_coreml

        return export_coreml(
            nn_model,
            dummy,
            output_path=output_path,
            precision=precision,
            compute_units=compute_units,
            nms=nms,
            iou=iou,
            conf=conf,
            metadata=metadata,
            model_family=self.model._get_model_name(),
        )

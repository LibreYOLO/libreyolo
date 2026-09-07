"""Unified model export with multiple backend support.

BaseExporter ABC with one subclass per format. Each subclass only
implements ``_export()``, while the template method in ``__call__`` handles
validation, model setup/teardown, calibration, and intermediate ONNX export.
"""

import copy
import importlib.util
import json
import logging
import tempfile
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from ..tasks import task_to_suffix
from ..utils.serialization import SCHEMA_VERSION
from .onnx import (
    _get_version,
    _requires_onnx_opset17,
    check_onnx_int8_available,
    export_onnx,
    quantize_onnx_int8,
)
from .support import get_support, validated_alternatives
from .torchscript import export_torchscript

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


def _pose_keypoint_shape_metadata(model) -> dict:
    num_keypoints = getattr(
        model, "num_keypoints", getattr(model, "POSE_NUM_KEYPOINTS", "")
    )
    keypoint_dim = getattr(model, "keypoint_dim", getattr(model, "KEYPOINT_DIM", ""))

    schema = getattr(model, "num_keypoints_per_class", None)
    inner = getattr(model, "model", None)
    if not schema and inner is not None:
        schema = getattr(inner, "num_keypoints_per_class", None)
    inner_model = getattr(inner, "model", None) if inner is not None else None
    if (
        not schema
        and inner_model is not None
        and hasattr(inner_model, "get_num_keypoints_per_class")
    ):
        schema = inner_model.get_num_keypoints_per_class()

    model_family = model._get_model_name() if hasattr(model, "_get_model_name") else ""
    if model_family == "ec":
        # EC pose exports raw xy-only tensors; visibility is appended by runtime
        # postprocessing after decoding.
        keypoint_dim = 2
    elif model_family == "rfdetr" and schema:
        # GroupPose RF-DETR exports the raw padded per-class tensor, whose
        # keypoint payload is (x, y, findable, visible, log_l11, l21, log_l22,
        # class_logit_boost).
        keypoint_dim = 8

    meta = {"num_keypoints": num_keypoints, "keypoint_dim": keypoint_dim}
    if schema:
        meta["num_keypoints_per_class"] = [int(count) for count in schema]
    return meta


_DOMEDETR_EXPORT_MESSAGE = (
    "Dome-DETR export is not supported. PAQI decides the query count per image "
    "(density-filtered proposals plus a greedy density-adaptive NMS), so the "
    "decoder output length is data dependent. Tracing bakes in whichever count "
    "the tracing image happened to produce, giving a graph that silently "
    "returns wrong results for every other image, and a static formulation "
    "needs the greedy suppression unrolled over all 250-1500 candidates. "
    "Reducing to a fixed top-k would remove exactly the tiny-object recall "
    "this family exists for. Use LibreDFINE if you need an exportable DETR."
)


_FIXED_SQUARE_EXPORT_FAMILIES = {
    "clip",
    "deformable_detr",
    "dinodetr",
    "detr",
    "dfine",
    "deim",
    "deimv2",
    "tinyformer",
    "ec",
    "lwdetr",
    "moge2",
    "rtdetr",
    "rtdetrv2",
    "rtdetrv4",
    "rfdetr",
    "siglip2",
    "ssd",
}
_RECTANGULAR_EXPORT_FAMILIES = {
    "hrnet",
    "yolo9",
    "yolo9_e2e",
    "yolo9_p2",
    "nafnet",
    # Every PP-LiteSeg size is natively rectangular (512x1024 / 768x1536);
    # a square export would not be the model the checkpoint was trained as.
    "ppliteseg",
    "unet",
    "realesrgan",
    "quicksrnet",
}
_RECTANGULAR_EXPORT_FORMATS = {
    "coreai",
    "coreml",
    "ncnn",
    "onnx",
    "openvino",
    "tensorrt",
    "tflite",
    "torchscript",
}


class _RTDETRExportWrapper(torch.nn.Module):
    """Trace RT-DETR dict outputs as the two tensors used by exported backends."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs["pred_logits"], outputs["pred_boxes"]


class _SemanticExportWrapper(torch.nn.Module):
    """Expose only dense semantic logits from task-specific native outputs."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            if "semantic_logits" in output:
                return output["semantic_logits"]
            if "logits" in output:
                return output["logits"]
            if "predictions" in output:
                return output["predictions"]
            if "out" in output:
                return output["out"]
        if isinstance(output, (list, tuple)):
            return output[-1]
        return output


class _ImageEmbeddingExportWrapper(torch.nn.Module):
    """Trace an image tower as a normalized whole-image embedding graph."""

    def __init__(self, image_tower: torch.nn.Module):
        super().__init__()
        self.image_tower = image_tower

    def forward(self, x):
        return torch.nn.functional.normalize(self.image_tower(x).float(), dim=-1)


class _VideoEmbeddingExportWrapper(torch.nn.Module):
    """Trace a V-JEPA 2 encoder as a fixed-frame clip embedding graph.

    Input is the public 5D video layout ``(B, F, C, H, W)``; output is the
    pooled, L2-normalized ``(B, D)`` row. The raw token grid is deliberately
    not an export target.
    """

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        tokens = self.encoder(x)
        return torch.nn.functional.normalize(tokens.mean(dim=1).float(), dim=-1)


class _YOLONASExportWrapper(torch.nn.Module):
    """Expose decoded YOLO-NAS tensors without training-only auxiliaries."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        # Eager and torch.export capture return ``(decoded, raw)``. ONNX
        # tracing already returns ``decoded`` directly, so accept both forms.
        if (
            isinstance(output, tuple)
            and len(output) == 2
            and isinstance(output[0], tuple)
        ):
            return output[0]
        return output


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

    # Alternate names accepted by create(). "litert" is Google's current name
    # for TensorFlow Lite; the format and .tflite suffix are unchanged.
    # The CLI and the `libreyolo formats` listing derive from this mapping,
    # so aliases live here and nowhere else.
    _aliases: dict[str, str] = {"engine": "tensorrt", "litert": "tflite"}

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
        key = cls._aliases.get(key, key)
        if key not in cls._registry:
            valid = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Unsupported export format: {format!r}. Must be one of: {valid}"
            )
        if model is not None and getattr(model, "FAMILY", None) == "domedetr":
            raise NotImplementedError(_DOMEDETR_EXPORT_MESSAGE)
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
        # Private plumbing (quantized_export): a callable run at the
        # post-validation mutation point below, so caller-side mutations
        # (reconstructing fp32 masters, enabling export mode) also wait for
        # every request rejection.
        pre_trace_hook = kwargs.pop("_pre_trace_hook", None)

        task = getattr(self.model, "task", "detect")
        model_name = self.model._get_model_name()
        if model_name == "ben2" and batch != 1:
            raise ValueError(
                "BEN2 export uses a fixed-resolution, batch-1 runtime "
                f"contract; got batch={batch}."
            )
        if task == "mesh":
            # Gated off for the first version, as semantic and point were: the
            # runtime metadata contract for a mesh graph (which body model,
            # how many betas, whether the body-model decoder is inside the
            # graph or applied afterwards) has to be defined before artifacts
            # exist that backends would have to keep reading.
            raise NotImplementedError(
                "Body-mesh export is not implemented yet. The exported-graph "
                "metadata contract for the mesh task is still to be defined; "
                "run mesh models through the PyTorch path for now."
            )
        if task in {"depth", "normal"}:
            # Dense-map export uses a fixed-resolution contract: backends
            # stretch-resize to the exported canvas and resize the result back
            # to the original canvas. The batch axis is static (dynamic is
            # forced off) and backends schedule one image per run, so a batch
            # != 1 artifact could never be fed correctly.
            task_label = "Depth" if task == "depth" else "Surface-normal"
            if batch != 1:
                raise ValueError(
                    f"{task_label} export uses a fixed-resolution, batch-1 runtime "
                    f"contract in v1; got batch={batch}."
                )
            if dynamic:
                warnings.warn(
                    f"{task_label} export uses a fixed-resolution runtime "
                    "contract in v1; forcing dynamic=False.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                dynamic = False
        if task == "edge":
            # Edge specialists export a single fused probability map on a
            # fixed square canvas; runtimes resize it back to the source.
            if batch != 1:
                raise ValueError(
                    "Edge export uses a fixed-resolution, batch-1 runtime "
                    f"contract in v1; got batch={batch}."
                )
            if dynamic:
                warnings.warn(
                    "Edge export uses a fixed-resolution runtime contract in "
                    "v1; forcing dynamic=False.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                dynamic = False
        if (
            getattr(self.model, "task", "detect") == "restore"
            and dynamic
            and self.model._get_model_name() not in {"realesrgan", "quicksrnet"}
        ):
            # Real-ESRGAN and QuickSRNet are fully convolutional and export with
            # dynamic H/W; other restore families keep the fixed-resolution v1
            # contract.
            warnings.warn(
                "Restore export uses a fixed-resolution runtime contract in "
                "v1; forcing dynamic=False.",
                RuntimeWarning,
                stacklevel=2,
            )
            dynamic = False
        if getattr(self.model, "task", "detect") == "matte" and dynamic:
            # BiRefNet's Swin relative-position tables are resolution-tied, so
            # the matte contract is the fixed native square (1024). Forcing a
            # dynamic graph would silently mis-interpolate them.
            warnings.warn(
                "Matte export uses a fixed-resolution runtime contract "
                "(native 1024); forcing dynamic=False.",
                RuntimeWarning,
                stacklevel=2,
            )
            dynamic = False
        half, int8 = self._validate(half, int8, data)
        self._preflight(half=half, int8=int8, data=data, **kwargs)
        data = self._resolve_calibration_data(int8, data)

        if opset is None:
            # DETR-style families use deformable attention / layer norm ops
            # which require opset 16+ (or 17 for ``aten::scaled_dot_product``
            # in the tuple export wrapper). Other families default to 13.
            opset = 17 if _requires_onnx_opset17(self.model._get_model_name()) else 13

        # The BiRefNet-derived decoders use torchvision deform_conv2d, which
        # maps to the standard ONNX ``DeformConv`` op (opset 19+). BEN2 is also
        # a matte family but has no deformable convolution, so it keeps the
        # normal exporter opset and needs no custom symbolic.
        if self.model._get_model_name() in {"birefnet", "feynobg"}:
            from ..models.birefnet.export import (
                MIN_OPSET as _MATTE_MIN_OPSET,
            )
            from ..models.birefnet.export import (
                register_deform_conv2d_onnx_symbolic,
            )

            if opset < _MATTE_MIN_OPSET:
                opset = _MATTE_MIN_OPSET
            register_deform_conv2d_onnx_symbolic(_MATTE_MIN_OPSET)

        imgsz, device, output_path = self._resolve_params(
            output_path,
            imgsz,
            device,
            half,
            int8,
        )

        # ---- Post-validation mutation point ------------------------------
        # Every request rejection above — format support, precision
        # validation, option preflight, imgsz/path resolution — has fired;
        # from here on the live model may be mutated.
        #
        # A model fine-tuned with lora=True carries live PEFT adapter layers.
        # Fold them into dense weights so the traced graph is a plain model
        # with no peft dependency. The merge is destructive (adapters are
        # folded and removed), so it must not run any earlier.
        from ..training.lora import merge_lora_adapters, module_has_lora

        if module_has_lora(self.model.model):
            merge_lora_adapters(self.model.model)

        if pre_trace_hook is not None:
            pre_trace_hook()

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
        if half and not self.supports_fp16:
            raise NotImplementedError(
                f"{self.format_name.upper()} FP16 export is not supported."
            )
        if int8 and not self.supports_int8:
            raise NotImplementedError(
                f"{self.format_name.upper()} INT8 export is not supported."
            )
        if int8 and data is None and not self.default_int8_calibration_data:
            raise ValueError("INT8 export requires calibration data. Pass data=...")
        return half, int8

    def _resolve_calibration_data(
        self, int8: bool, data: Optional[str]
    ) -> Optional[str]:
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
        if kwargs.get("deepstream") and self.format_name != "onnx":
            raise ValueError(
                "deepstream=True is supported only for ONNX export "
                f"(format='onnx'); got format={self.format_name!r}."
            )
        family = self.model._get_model_name()
        task = getattr(self.model, "task", "detect")
        if not isinstance(task, str):
            task = "detect"
        support = get_support(family, task, self.format_name)
        if support.tier == "blocked":
            alternatives = validated_alternatives(family, task)
            alternatives_text = (
                f" Validated alternatives: {', '.join(alternatives)}."
                if alternatives
                else ""
            )
            raise NotImplementedError(
                f"{family} {task} export to {self.format_name} is blocked: "
                f"{support.reason}{alternatives_text}"
            )
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
            if isinstance(native_imgsz, (tuple, list)):
                if len(native_imgsz) != 2:
                    raise ValueError(
                        "Native input size must be an int or (height, width), "
                        f"got {native_imgsz!r}."
                    )
                imgsz = (int(native_imgsz[0]), int(native_imgsz[1]))
            else:
                imgsz = (int(native_imgsz), int(native_imgsz))
        elif isinstance(imgsz, tuple):
            if len(imgsz) != 2:
                raise ValueError(f"imgsz tuple must be (height, width), got {imgsz}")
            imgsz = (int(imgsz[0]), int(imgsz[1]))
        else:
            imgsz = (int(imgsz), int(imgsz))
        if imgsz[0] <= 0 or imgsz[1] <= 0:
            raise ValueError(f"imgsz values must be positive, got {imgsz}.")
        if model_name == "ben2":
            native_shape = (
                (int(native_imgsz[0]), int(native_imgsz[1]))
                if isinstance(native_imgsz, (tuple, list))
                else (int(native_imgsz), int(native_imgsz))
            )
            if imgsz != native_shape:
                raise ValueError(
                    "BEN2 export imgsz must match its fixed native resolution "
                    f"{native_shape[0]}x{native_shape[1]}, got {imgsz}."
                )
        if model_name in ("deit", "vgg") and imgsz != (native_imgsz, native_imgsz):
            raise ValueError(
                f"{model_name} export imgsz must match its fixed native resolution "
                f"{native_imgsz}x{native_imgsz}, got {imgsz}."
            )
        if model_name == "hrnet":
            native_h, native_w = (
                (int(native_imgsz[0]), int(native_imgsz[1]))
                if isinstance(native_imgsz, (tuple, list))
                else (int(native_imgsz), int(native_imgsz))
            )
            if imgsz != (native_h, native_w):
                raise ValueError(
                    "HRNet pose exports use the checkpoint's fixed person-crop "
                    f"canvas {(native_h, native_w)}, got {imgsz}."
                )
        imgsz_divisor = int(getattr(self.model, "IMGSZ_DIVISOR", 1) or 1)
        if imgsz[0] % imgsz_divisor or imgsz[1] % imgsz_divisor:
            raise ValueError(
                f"{model_name} export imgsz must be divisible by "
                f"{imgsz_divisor}, got {imgsz}."
            )
        if model_name == "nafnet":
            padder_size = int(getattr(self.model.model, "padder_size", 16))
            if imgsz[0] % padder_size or imgsz[1] % padder_size:
                raise ValueError(
                    "NAFNet export imgsz must be divisible by the network "
                    f"downsample factor {padder_size}, got {imgsz}."
                )
        dense_task = getattr(self.model, "task", "detect")
        divisor_attrs = {
            "depth": "depth_imgsz_divisor",
            "normal": "normal_imgsz_divisor",
            "semantic": "semantic_imgsz_divisor",
        }
        if dense_task in divisor_attrs:
            divisor_attr = divisor_attrs[dense_task]
            divisor = int(getattr(self.model, divisor_attr, 1) or 1)
            if imgsz[0] % divisor or imgsz[1] % divisor:
                raise ValueError(
                    f"{dense_task.capitalize()} export imgsz must be divisible "
                    "by the network "
                    f"stride {divisor}, got {imgsz}."
                )
        if getattr(self.model, "task", "detect") == "edge":
            divisor = int(getattr(self.model, "edge_imgsz_divisor", 1) or 1)
            if imgsz[0] % divisor or imgsz[1] % divisor:
                raise ValueError(
                    "Edge export imgsz must be divisible by the network "
                    f"stride {divisor}, got {imgsz}."
                )
        if model_name == "rfdetr":
            from ..models.rfdetr.imgsz import resolve_patch_window, validate_imgsz

            patch_size, num_windows = resolve_patch_window(self.model.model)
            imgsz = validate_imgsz(
                imgsz,
                patch_size=patch_size,
                num_windows=num_windows,
                name="RF-DETR export imgsz",
            )
        if model_name == "domedetr":
            raise NotImplementedError(_DOMEDETR_EXPORT_MESSAGE)
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
                "YOLO9-family, HRNet, NAFNet, QuickSRNet, and Real-ESRGAN "
                "exports only."
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
        stem = self._auto_output_stem()
        precision_suffix = "_int8" if int8 else ("_fp16" if half else "")
        return str(Path("weights") / f"{stem}{precision_suffix}{self.suffix}")

    def _auto_output_stem(self) -> str:
        model_path = getattr(self.model, "model_path", None)
        if isinstance(model_path, (str, Path)):
            source = Path(model_path)
            if source.suffix.lower() in {".pt", ".pth", ".safetensors"}:
                return source.stem

        prefix = getattr(self.model, "FILENAME_PREFIX", None)
        size = getattr(self.model, "size", None)
        if isinstance(prefix, str) and prefix and isinstance(size, str) and size:
            task_suffix = self._auto_output_task_suffix()
            return f"{prefix}{size}{task_suffix}"

        model_name = self.model._get_model_name().lower()
        return f"{model_name}_{self.model.size}"

    def _auto_output_task_suffix(self) -> str:
        task = getattr(self.model, "task", "detect")
        if not isinstance(task, str):
            task = "detect"
        if getattr(self.model, "_is_segmentation", False) is True:
            task = "segment"
        try:
            suffix = task_to_suffix(task)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported task for auto output naming: {task!r}"
            ) from exc
        return f"-{suffix}" if suffix else ""

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
        # Non-None only for families whose graph consumes a 5D video clip.
        video_export_frames = None
        rfdetr_export_snapshots = []
        rfdetr_inner = None
        family = self.model._get_model_name()
        task = getattr(self.model, "task", "detect")
        moge2_onnx_mode = family == "moge2" and hasattr(
            root_model, "set_onnx_compatible_mode"
        )
        original_moge2_onnx_mode = bool(
            getattr(root_model, "onnx_compatible_mode", False)
        )
        if task == "semantic":
            nn_model = _SemanticExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "detr":
            from ..models.detr.nn import DETRExportWrapper

            nn_model = DETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "dfine":
            from ..models.dfine.nn import DFINEExportWrapper

            # deploy() (BN fusion + decoder-layer pruning + head swap) mutates
            # the wrapped model in place; deepcopy first so the user's live
            # model is never destructively modified (mirrors the deimv2 path).
            nn_model = copy.deepcopy(nn_model)
            nn_model = DFINEExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "deim":
            from ..models.deim.nn import DEIMExportWrapper

            nn_model = copy.deepcopy(nn_model)
            nn_model = DEIMExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "deimv2":
            from ..models.deimv2.nn import DEIMv2ExportWrapper

            nn_model = copy.deepcopy(nn_model)
            nn_model = DEIMv2ExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "tinyformer":
            from ..models.tinyformer.nn import TinyFormerExportWrapper

            nn_model = copy.deepcopy(nn_model)
            nn_model = TinyFormerExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "ec":
            from ..models.ec.nn import ECExportWrapper

            nn_model = copy.deepcopy(nn_model)
            nn_model = ECExportWrapper(
                nn_model, task=getattr(self.model, "task", "detect")
            ).to(device)
            nn_model.eval()
            dfine_wrapped = True  # share the YOLOX-head-export skip path below
        elif family in {"yolo1", "yolo2", "yolo3", "yolo4"}:
            from ..models.darknet.export import DarknetExportWrapper

            # Bake the anchor-box decode (or the YOLOv1 dense-head decode) into the
            # graph so every export format emits a self-contained (B, 4+nc, N)
            # tensor consumed by the shared backend decode. No learned anchors
            # need to travel in metadata.
            nn_model = DarknetExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "yolo7":
            from ..models.yolo7.export import YOLO7ExportWrapper

            nn_model = YOLO7ExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family in {"yolonas", "ppyoloe"}:
            # Both heads return ``(decoded, raw)`` in eval; the wrapper keeps
            # the decoded (boxes, scores) pair and drops the training-only
            # auxiliaries.
            nn_model = _YOLONASExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family in {"rtdetr", "rtdetrv2", "rtdetrv4"}:
            nn_model = _RTDETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "lwdetr":
            from ..models.lwdetr.nn import LWDETRExportWrapper

            nn_model = LWDETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "mask_rcnn":
            from ..models.mask_rcnn.nn import MaskRCNNExportWrapper

            nn_model = MaskRCNNExportWrapper(
                nn_model,
                include_masks=task == "segment",
            ).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "faster_rcnn":
            from ..models.faster_rcnn.nn import FasterRCNNExportWrapper

            nn_model = FasterRCNNExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "retinanet":
            from ..models.retinanet.nn import RetinaNetExportWrapper

            nn_model = RetinaNetExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "ssd":
            from ..models.ssd.nn import SSDExportWrapper

            nn_model = SSDExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "fcos":
            from ..models.fcos.nn import FCOSExportWrapper

            nn_model = FCOSExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "efficientdet":
            from ..models.efficientdet.nn import EfficientDetExportWrapper

            if imgsz[0] != imgsz[1] or imgsz[0] != self.model.input_size:
                raise ValueError(
                    f"EfficientDet {self.model.size} exports require imgsz="
                    f"{self.model.input_size}; got {imgsz}."
                )
            nn_model = EfficientDetExportWrapper(
                nn_model,
                input_size=imgsz[0],
                # TensorRT's ITopK layer rejects K > 3840. Keep the exact
                # upstream 5000-point budget on every other runtime and use
                # its maximum only for the TensorRT graph.
                max_candidates=3840 if self.format_name == "tensorrt" else 5000,
                sparse_coco=(
                    getattr(self.model, "nb_classes", None) == 80
                    and getattr(self.model, "_arch_num_classes", None) == 90
                ),
            ).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "deformable_detr":
            from ..models.deformable_detr.nn import DeformableDETRExportWrapper

            nn_model = DeformableDETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "dinodetr":
            from ..models.dinodetr.nn import DINODETRExportWrapper

            nn_model = DINODETRExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "centernet":
            from ..models.centernet.nn import CenterNetExportWrapper

            # The portable grid-sample DCN flag is export-only. Keep the live
            # eager model on torchvision's exact native operator.
            nn_model = copy.deepcopy(nn_model)
            nn_model = CenterNetExportWrapper(nn_model).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "rtmdet":
            # RTMDet intentionally aliases the head convolution weights across
            # feature levels while keeping one batch norm per level. XNNPACK's
            # batch-norm fusion assigns the shared parameters duplicate names,
            # so give the export-only copy independent modules with identical
            # weights. The user's live model and its sharing contract stay
            # untouched.
            nn_model = copy.deepcopy(nn_model)
            for tower_name in ("cls_convs", "reg_convs"):
                tower = getattr(nn_model.head, tower_name)
                for level in range(1, len(tower)):
                    for layer_index, layer in enumerate(tower[level]):
                        layer.conv = copy.deepcopy(tower[0][layer_index].conv)
            nn_model.to(device)
            nn_model.eval()
        elif family == "depth_anything3":
            nn_model = copy.deepcopy(nn_model)
            nn_model.export = True
            nn_model.to(device)
            nn_model.eval()
        elif family == "dinov2" and getattr(self.model, "task", None) in {
            "classify",
            "embed",
        }:
            # Classification and embedding have no detection decoder. Trace
            # their task-specific backbone path directly and bake the fixed
            # DINOv2 positional encoding before capture.
            if getattr(self.model, "task", None) == "classify":
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
        elif family in {"clip", "siglip2"} and task == "embed":
            image_tower = (
                nn_model.visual if family == "clip" else nn_model.vision_model
            )
            nn_model = _ImageEmbeddingExportWrapper(image_tower).to(device)
            nn_model.eval()
            dfine_wrapped = True
        elif family == "vjepa2":
            # Every V-JEPA 2 graph takes a 5D clip. The embed graph pools to a
            # normalized row; the classify graph already ends at logits, so it
            # only needs the 5D dummy below.
            if task == "embed":
                nn_model = _VideoEmbeddingExportWrapper(nn_model).to(device)
            nn_model.eval()
            video_export_frames = int(getattr(self.model, "clip_frames", 64))
            dfine_wrapped = True
        elif family in {"clip", "siglip2"} and task == "classify":
            text_embeds = getattr(self.model, "_text_embeds", None)
            if text_embeds is None:
                raise RuntimeError(
                    "No classes set; call set_classes() before export()."
                )
            scale = float(nn_model.logit_scale.exp().detach().cpu())
            weight = (scale * text_embeds).detach().to(device, torch.float32)
            if family == "clip":
                from ..models.clip.export import _FrozenCLIPClassifier

                nn_model = _FrozenCLIPClassifier(nn_model.visual, weight).to(device)
            else:
                from ..models.siglip2.export import _FrozenSigLIP2Classifier

                bias = nn_model.logit_bias.detach().to(
                    device=device,
                    dtype=torch.float32,
                )
                nn_model = _FrozenSigLIP2Classifier(
                    nn_model.vision_model,
                    weight,
                    bias.reshape(()),
                ).to(device)
            nn_model.eval()
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
        if video_export_frames is not None:
            # Public clip layout (B, F, C, H, W). Frame count, crop and tubelet
            # geometry are fixed per graph; only batch may be dynamic.
            dummy = torch.randn(batch, video_export_frames, 3, h, w, device=device)
        else:
            dummy = torch.randn(batch, 3, h, w, device=device)

        if half and not int8 and self.apply_model_half:
            nn_model.half()
            dummy = dummy.half()

        if moge2_onnx_mode:
            root_model.set_onnx_compatible_mode(True)
        try:
            yield nn_model, dummy
        finally:
            if moge2_onnx_mode:
                root_model.set_onnx_compatible_mode(original_moge2_onnx_mode)
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
        # Use a distinct intermediate name so it never collides with (and the
        # finally-block cleanup never deletes) a real ONNX export the user may
        # have written to the default ``<stem>.onnx`` path.
        out = Path(output_path)
        onnx_output = str(out.with_name(f"{out.stem}.export_intermediate.onnx"))
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
            if isinstance(native, (tuple, list)):
                meta_h, meta_w = int(native[0]), int(native[1])
                metadata_imgsz = max(meta_h, meta_w)
            else:
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
        # Classification eval preprocessing must travel with every artifact,
        # not just ONNX. Exported-backend predict() otherwise falls back to
        # crop_pct=0.875 and bilinear resize, which changes classifier logits
        # for families such as ResNet (0.95/bicubic).
        if task == "classify":
            crop_pct = getattr(self.model, "crop_pct", None)
            interpolation = getattr(self.model, "interpolation", None)
            if crop_pct is not None:
                meta["crop_pct"] = float(crop_pct)
            if interpolation is not None:
                meta["interpolation"] = str(interpolation)
        if task == "pose":
            meta.update(_pose_keypoint_shape_metadata(self.model))
            if self.model._get_model_name() == "hrnet":
                meta["pose_input"] = "person_crop"
        if task == "gaze":
            meta.update(
                {
                    "num_bins": int(self.model.num_bins),
                    "bin_width_deg": float(self.model.bin_width_deg),
                    "offset_deg": float(self.model.offset_deg),
                    "gaze_input": "face_crop",
                }
            )
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
            if isinstance(native, (tuple, list)):
                native_h, native_w = int(native[0]), int(native[1])
                metadata_imgsz = str(max(native_h, native_w))
                meta_h, meta_w = str(native_h), str(native_w)
            else:
                metadata_imgsz = str(int(native))
                meta_h = meta_w = str(int(native))
        # TODO(schema-v1.1): keep legacy model_size/nb_classes aliases for one
        # transition window, then prefer the canonical size/nc keys only.
        meta = {
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
            "segmentation": str(
                task == "segment" or getattr(self.model, "_is_segmentation", False)
            ).lower(),
            "obb": str(task == "obb").lower(),
        }
        # Classification eval preprocessing — lets exported-backend inference
        # match native predict()/val() (per-family crop_pct + interpolation).
        _crop_pct = getattr(self.model, "crop_pct", None)
        _interp = getattr(self.model, "interpolation", None)
        if _crop_pct is not None:
            meta["crop_pct"] = str(_crop_pct)
        if _interp is not None:
            meta["interpolation"] = str(_interp)
        if task == "pose":
            pose_meta = _pose_keypoint_shape_metadata(self.model)
            meta.update(
                {
                    "num_keypoints": str(pose_meta["num_keypoints"]),
                    "keypoint_dim": str(pose_meta["keypoint_dim"]),
                }
            )
            if "num_keypoints_per_class" in pose_meta:
                meta["num_keypoints_per_class"] = json.dumps(
                    pose_meta["num_keypoints_per_class"]
                )
            if self.model._get_model_name() == "hrnet":
                meta["pose_input"] = "person_crop"
        if task == "gaze":
            meta.update(
                {
                    "num_bins": str(int(self.model.num_bins)),
                    "bin_width_deg": str(float(self.model.bin_width_deg)),
                    "offset_deg": str(float(self.model.offset_deg)),
                    "gaze_input": "face_crop",
                }
            )
        return meta

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
        if self.model._get_model_name() in {"mask_rcnn", "rfdetr"}:
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

    def _resolve_params(self, output_path, imgsz, device, half, int8):
        imgsz, device, output_path = super()._resolve_params(
            output_path, imgsz, device, half, int8
        )
        family = self.model._get_model_name()
        size = getattr(self.model, "size", None)
        if family == "deformable_detr" and size == "r50twostage":
            if half:
                raise NotImplementedError(
                    "Deformable DETR two-stage ONNX export is validated in FP32 only."
                )
            if device.type != "cpu":
                warnings.warn(
                    "Deformable DETR two-stage ONNX export is traced on CPU because "
                    "the legacy PyTorch exporter can terminate while lowering its "
                    "CUDA top-k graph. The model is restored to its original device "
                    "after export.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                device = torch.device("cpu")
        return imgsz, device, output_path

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
        if kwargs.get("deepstream"):
            from .deepstream import (
                deepstream_supported_families,
                deepstream_supported_tasks,
            )

            if kwargs.get("nms"):
                raise ValueError(
                    "deepstream=True and nms=True are mutually exclusive: "
                    "DeepStream runs suppression in its clustering stage."
                )
            task = getattr(self.model, "task", "detect")
            if not isinstance(task, str):
                task = "detect"
            family = self.model._get_model_name()
            supported = deepstream_supported_families(task)
            if not supported:
                supported_tasks = ", ".join(sorted(deepstream_supported_tasks()))
                raise NotImplementedError(
                    f"DeepStream export does not support the {task!r} task. "
                    f"Supported tasks: {supported_tasks}."
                )
            if family not in supported:
                raise NotImplementedError(
                    f"DeepStream {task} export supports families "
                    f"{sorted(supported)}; got {family!r}."
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
        deepstream=False,
        iou=0.45,
        conf=0.25,
        max_det=300,
        calibrate_method="MinMax",
        nodes_to_exclude=None,
        **kwargs,
    ):
        imgsz = (dummy.shape[-2], dummy.shape[-1])

        if deepstream:
            from .deepstream import wrap_for_deepstream

            ds_task = getattr(self.model, "task", "detect")
            if not isinstance(ds_task, str):
                ds_task = "detect"
            nn_model = wrap_for_deepstream(
                nn_model,
                model_family=self.model._get_model_name(),
                imgsz=imgsz,
                model_size=getattr(self.model, "size", None),
                task=ds_task,
            ).eval()

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
            if deepstream:
                meta["deepstream"] = "true"
            return meta

        def _write_deepstream_sidecars(onnx_result_path: str) -> None:
            if not deepstream:
                return
            from .deepstream import write_deepstream_sidecars

            names = getattr(self.model, "names", None) or {}
            class_names = [names[k] for k in sorted(names, key=int)]
            write_deepstream_sidecars(
                onnx_result_path,
                model_family=self.model._get_model_name(),
                class_names=class_names,
                imgsz=imgsz,
                batch=dummy.shape[0],
                precision="int8" if int8 else ("fp16" if half else "fp32"),
                conf=conf,
                iou=iou,
                task=ds_task,
            )

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
                    deepstream=deepstream,
                )
                result = quantize_onnx_int8(
                    fp32_path,
                    output_path,
                    calibration_data=calibration_data,
                    metadata=int8_metadata,
                    preprocessed_path=preprocessed_path,
                    calibrate_method=calibrate_method,
                    nodes_to_exclude=nodes_to_exclude,
                    skip_symbolic_shape=nms,
                )
                _write_deepstream_sidecars(result)
                return result

        result = export_onnx(
            nn_model,
            dummy,
            output_path=output_path,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            half=half,
            metadata=_onnx_metadata(precision_half=half),
            nms=nms,
            deepstream=deepstream,
        )
        _write_deepstream_sidecars(result)
        return result


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

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        # The graph is traced at a fixed input shape, so report dynamic=False
        # regardless of the requested flag (mirrors the NCNN override).
        meta = super()._build_metadata(precision, dynamic, onnx_path, imgsz=imgsz)
        meta["dynamic"] = False
        return meta

    def _export(self, nn_model, dummy, *, output_path, metadata, **kwargs):
        return export_torchscript(
            nn_model, dummy, output_path=output_path, metadata=metadata
        )


class ExecuTorchExporter(BaseExporter):
    """Fixed-shape, batch-1, FP32 ExecuTorch export with XNNPACK delegation."""

    format_name = "executorch"
    suffix = ".pte"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False

    def __call__(
        self, *, dynamic: bool = False, batch: int = 1, **kwargs
    ) -> str:
        """Reject unsupported shapes before the destructive LoRA merge."""
        if batch != 1:
            raise ValueError(
                f"ExecuTorch v1 requires batch=1, got batch={batch}."
            )
        if dynamic:
            raise ValueError("ExecuTorch v1 requires dynamic=False.")
        return super().__call__(dynamic=False, batch=1, **kwargs)

    def _resolve_params(self, output_path, imgsz, device, half, int8):
        if device is not None and str(device).lower() not in {"auto", "cpu"}:
            raise ValueError("ExecuTorch XNNPACK export requires device='cpu'.")
        return super()._resolve_params(
            output_path, imgsz, torch.device("cpu"), half, int8
        )

    def _preflight(self, *, half: bool, int8: bool, data: Optional[str], **kwargs):
        delegate = str(kwargs.get("delegate", "xnnpack")).lower()
        if delegate != "xnnpack":
            raise ValueError(
                "ExecuTorch v1 supports delegate='xnnpack' only, "
                f"got {delegate!r}."
            )
        super()._preflight(half=half, int8=int8, data=data, **kwargs)
        from .executorch import check_executorch_available

        check_executorch_available()

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        meta = super()._build_metadata(
            precision, False, onnx_path, imgsz=imgsz
        )
        crop_pct = getattr(self.model, "crop_pct", None)
        interpolation = getattr(self.model, "interpolation", None)
        if crop_pct is not None:
            meta["crop_pct"] = float(crop_pct)
        if interpolation is not None:
            meta["interpolation"] = str(interpolation)
        return meta

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        dynamic,
        delegate="xnnpack",
        **kwargs,
    ):
        if dummy.shape[0] != 1:
            raise ValueError(
                f"ExecuTorch v1 requires batch=1, got batch={dummy.shape[0]}."
            )
        if dynamic:
            raise ValueError("ExecuTorch v1 requires dynamic=False.")
        from .executorch import export_executorch

        return export_executorch(
            nn_model,
            dummy,
            output_path=output_path,
            metadata=metadata,
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

class PaddleExporter(BaseExporter):
    """Static batch-1 FP32 Paddle export through X2Paddle."""

    format_name = "paddle"
    suffix = "_paddle"
    requires_onnx = True
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False

    def __call__(
        self,
        *,
        dynamic: bool = False,
        batch: int = 1,
        simplify: bool = True,
        opset: int | None = None,
        **kwargs,
    ) -> str:
        if dynamic:
            raise ValueError("Paddle export requires dynamic=False.")
        if batch != 1:
            raise ValueError(f"Paddle export currently requires batch=1, got {batch}.")
        if not simplify:
            raise ValueError(
                "Paddle export requires simplify=True for a fully static "
                "X2Paddle conversion graph."
            )
        if opset is not None and int(opset) != 15:
            raise ValueError(
                "Paddle export requires opset=15 for the validated X2Paddle "
                f"conversion graph, got {opset}."
            )
        return super().__call__(
            dynamic=False,
            batch=1,
            simplify=True,
            opset=15,
            **kwargs,
        )

    def _preflight(self, **kwargs):
        # Check support policy before importing the optional converter stack.
        super()._preflight(**kwargs)
        from .paddle import check_paddle_export_available

        check_paddle_export_available()

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        metadata = super()._build_metadata(
            precision, False, onnx_path, imgsz=imgsz
        )
        metadata["dynamic"] = False
        metadata.pop("exported_from", None)
        return metadata

    def _export(
        self, nn_model, dummy, *, onnx_path, output_path, metadata, **kwargs
    ):
        from .paddle import export_paddle

        logger.info("Step 2/2: Converting ONNX to Paddle")
        return export_paddle(
            onnx_path=onnx_path,
            output_path=output_path,
            metadata=metadata,
        )


class MNNExporter(BaseExporter):
    """Export a fixed-shape FP32 MNN artifact through ONNX."""

    format_name = "mnn"
    suffix = ".mnn"
    requires_onnx = True
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False

    def __call__(self, *args, dynamic: bool = False, **kwargs) -> str:
        if dynamic:
            raise ValueError("MNN v1 export requires dynamic=False.")
        return super().__call__(*args, dynamic=False, **kwargs)

    def _preflight(self, **kwargs):
        super()._preflight(**kwargs)
        from .mnn import check_mnn_available

        check_mnn_available()

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        meta = super()._build_metadata(precision, dynamic, onnx_path, imgsz=imgsz)
        meta["dynamic"] = False
        return meta

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        onnx_path,
        dynamic,
        verbose,
        **kwargs,
    ):
        if dynamic:
            raise ValueError("MNN v1 export requires dynamic=False.")
        from .mnn import export_mnn

        logger.info("Step 2/2: Converting to MNN")
        return export_mnn(
            onnx_path,
            output_path,
            metadata=metadata,
            batch=int(dummy.shape[0]),
            verbose=verbose,
        )


class RknnExporter(BaseExporter):
    """Compile an exact simulator-tested detector variant for RK3588."""

    format_name = "rknn"
    suffix = ".rknn"
    requires_onnx = True
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False

    def __call__(
        self,
        *args,
        dynamic: bool = False,
        batch: int = 1,
        imgsz: int | tuple[int, int] | None = None,
        opset: int | None = 19,
        **kwargs,
    ) -> str:
        if dynamic:
            raise ValueError("RKNN export requires static input shapes.")
        if batch != 1:
            raise NotImplementedError("RKNN export currently supports batch=1 only.")
        if opset not in {None, 19}:
            raise NotImplementedError(
                f"RKNN export is validated only with ONNX opset 19, got {opset}."
            )
        from .rknn import resolve_rknn_imgsz

        imgsz = resolve_rknn_imgsz(
            model_family=self.model._get_model_name(),
            model_size=self.model.size,
            task=getattr(self.model, "task", "detect"),
            imgsz=imgsz,
        )
        return super().__call__(
            *args,
            dynamic=False,
            batch=1,
            imgsz=imgsz,
            opset=19,
            **kwargs,
        )

    def _validate(self, half: bool, int8: bool, data: str | None):
        if half:
            raise NotImplementedError(
                "RKNN does not expose LibreYOLO's half=True contract. Omit half "
                "and int8 for the tested vendor floating-point build."
            )
        return super()._validate(half, int8, data)

    def _preflight(self, **kwargs):
        from .rknn import (
            check_rknn_available,
            resolve_rknn_target,
            validate_rknn_export_request,
        )

        target_platform = resolve_rknn_target(
            name=kwargs.get("name"),
            target=kwargs.get("target"),
            target_platform=kwargs.get("target_platform"),
        )
        validate_rknn_export_request(
            model_family=self.model._get_model_name(),
            model_size=self.model.size,
            task=getattr(self.model, "task", "detect"),
            target_platform=target_platform,
        )
        super()._preflight(**kwargs)
        check_rknn_available()

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        metadata,
        calibration_data,
        onnx_path,
        int8,
        opset,
        verbose,
        name=None,
        target=None,
        target_platform=None,
        rknn_config=None,
        rknn_build=None,
        verify=False,
        verify_input=None,
        verify_rtol=1e-3,
        verify_atol=1e-4,
        verify_min_cosine=0.9999,
        verify_max_normalized_rmse=0.02,
        **kwargs,
    ):
        from .rknn import (
            _publish_rknn_artifacts,
            _run_onnx_reference,
            compare_rknn_outputs,
            evaluate_rknn_metrics,
            export_rknn,
            export_rknn_with_simulator,
            resolve_rknn_target,
        )

        resolved_target = resolve_rknn_target(
            name=name,
            target=target,
            target_platform=target_platform,
        )
        rknn_metadata = dict(metadata or {})
        rknn_metadata.update(
            {
                "dynamic": False,
                "onnx_opset": int(opset),
                "rknn_target": resolved_target,
            }
        )
        logger.info("Step 2/2: Compiling RKNN for %s", resolved_target)

        if not verify:
            return export_rknn(
                onnx_path=onnx_path,
                output_path=output_path,
                target_platform=resolved_target,
                int8=int8,
                calibration_data=calibration_data,
                metadata=rknn_metadata,
                verbose=verbose,
                config=rknn_config,
                build=rknn_build,
            )

        if verify_input is None:
            generator = torch.Generator(device="cpu").manual_seed(0)
            verify_input = torch.rand(
                tuple(dummy.shape), generator=generator, dtype=torch.float32
            ).numpy()
        destination = Path(output_path)
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}.verify.",
            suffix=destination.suffix,
            dir=destination.parent,
            delete=False,
        ) as handle:
            staging_path = Path(handle.name)
        staging_path.unlink()

        try:
            result, simulated = export_rknn_with_simulator(
                onnx_path=onnx_path,
                output_path=str(staging_path),
                simulator_inputs=verify_input,
                target_platform=resolved_target,
                int8=int8,
                calibration_data=calibration_data,
                metadata=rknn_metadata,
                verbose=verbose,
                config=rknn_config,
                build=rknn_build,
            )
            reference = _run_onnx_reference(onnx_path, verify_input)
            metrics = compare_rknn_outputs(
                reference,
                simulated,
                rtol=float(verify_rtol),
                atol=float(verify_atol),
                raise_on_failure=False,
            )
            parity_passed, metrics = evaluate_rknn_metrics(
                metrics,
                min_cosine=float(verify_min_cosine),
                max_normalized_rmse=float(verify_max_normalized_rmse),
            )
            report = {
                "backend": "rknn-simulator",
                "target": resolved_target,
                "reference": "onnxruntime",
                "rtol": float(verify_rtol),
                "atol": float(verify_atol),
                "min_cosine": float(verify_min_cosine),
                "max_normalized_rmse": float(verify_max_normalized_rmse),
                "passed": parity_passed,
                "outputs": metrics,
            }
            staged_report = Path(f"{staging_path}.parity.json")
            temporary_report = staged_report.with_suffix(
                f"{staged_report.suffix}.tmp"
            )
            temporary_report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_report.replace(staged_report)
            if not parity_passed:
                report_path = Path(f"{destination}.failed.parity.json")
                staged_report.replace(report_path)
                failures = "; ".join(
                    f"output {item['index']}: "
                    f"max_abs={item['max_abs_error']:.6g}, "
                    f"cosine={item['cosine_similarity']:.6g}, "
                    f"normalized_rmse={item['normalized_rmse']:.6g}"
                    for item in metrics
                    if not item["accepted"]
                )
                raise AssertionError(
                    "RKNN simulator acceptance failed "
                    f"(min_cosine={verify_min_cosine}, "
                    f"max_normalized_rmse={verify_max_normalized_rmse}): "
                    f"{failures}. Metrics: {report_path}"
                )

            staged_metadata = Path(f"{staging_path}.metadata.json")
            report_path = Path(f"{destination}.parity.json")
            _publish_rknn_artifacts(
                staged_model=Path(result),
                staged_metadata=staged_metadata,
                staged_report=staged_report,
                destination=destination,
                remove_artifacts=(
                    Path(f"{destination}.failed.parity.json"),
                ),
            )
        finally:
            # Verification publishes the staged files only after all parity
            # gates pass. Any unsuccessful exit leaves a prior export intact.
            staging_path.unlink(missing_ok=True)
            Path(f"{staging_path}.metadata.json").unlink(missing_ok=True)
            Path(f"{staging_path}.parity.json").unlink(missing_ok=True)
            Path(f"{staging_path}.parity.json.tmp").unlink(missing_ok=True)
        logger.info("RKNN simulator parity passed: %s", report_path)
        return str(destination)


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

        super()._preflight(**kwargs)
        check_tflite_export_available()

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


class CoreAIExporter(BaseExporter):
    """Apple Core AI (``.aimodel``) export via ``torch.export``.

    Unlike the Core ML path this uses a real graph capture rather than a
    single recorded trace, so the static-eval monkey patches that path needs
    are not required here. Artifacts are static-shape in v1 and declare a
    minimum OS of v27, which is the only value the toolchain offers.
    """

    format_name = "coreai"
    suffix = ".aimodel"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = False
    apply_model_half = False
    supports_embedded_nms = False

    def __call__(self, *, dynamic: bool = False, **kwargs) -> str:
        """Export a fixed-canvas Core AI artifact.

        The base exporter defaults ``dynamic=True`` for ONNX. Core AI has no
        dynamic-shape contract in v1, so its format-specific default is false
        and an explicit request is rejected rather than mislabeled.
        """
        if dynamic:
            raise NotImplementedError(
                "Core AI export uses a fixed input shape; dynamic=True is not "
                "supported."
            )
        return super().__call__(dynamic=False, **kwargs)

    def _preflight(self, **kwargs):
        # Support policy is checked before optional dependencies, as required
        # by ADR 0011. Dependency validation still happens before the
        # destructive LoRA merge in BaseExporter.__call__.
        super()._preflight(**kwargs)

        # Check the optional dependency HERE, not at conversion time. Preflight
        # runs before __call__ merges any live LoRA adapters, and that merge is
        # destructive. Discovering the missing package afterwards would leave
        # the caller's model permanently modified with no artifact to show for
        # it.
        from .coreai import _require_coreai

        _require_coreai()

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        # v1 artifacts are fixed-canvas, mirroring the CoreML/NCNN overrides.
        meta = super()._build_metadata(precision, dynamic, onnx_path, imgsz=imgsz)
        meta["dynamic"] = False
        return meta

    def _export(
        self,
        nn_model,
        dummy,
        *,
        output_path,
        precision,
        metadata,
        **kwargs,
    ):
        from .coreai import export_coreai

        return export_coreai(
            nn_model,
            dummy,
            output_path=output_path,
            precision=precision,
            metadata=metadata,
            model_family=self.model._get_model_name(),
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

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        # CoreML uses a hard-fixed ct.ImageType(shape=...); the exported graph
        # has a fixed input shape, so report dynamic=False regardless of the
        # requested flag (mirrors the NCNN override).
        meta = super()._build_metadata(precision, dynamic, onnx_path, imgsz=imgsz)
        meta["dynamic"] = False
        return meta

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

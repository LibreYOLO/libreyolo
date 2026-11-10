"""GGUF export for vision.cpp.

Serializes the live, eval-mode network inside the exporter's model context
into a GGUF file: BatchNorm folded into the preceding convolution, RepConv
branches fused into one 3x3 convolution, training-only ``aux.*`` /
``aux_head.*`` modules left out, and the preprocessing / decode metadata
the C++ runtime needs written as GGUF key-values.

Tensor names are the LibreYOLO module paths of the ``nn.Conv2d`` that
carries the (folded) weights: ``<conv path>.weight`` (F32 or F16, OIHW in
PyTorch order, reversed ``ne`` order on disk as gguf-py writes it) and
``<conv path>.bias`` (always F32). A fused RepConv is stored under its 3x3
branch (``<repconv>.conv1.conv``); its 1x1 branch (``<repconv>.conv2``) is
absent from the file and ``yolov9.repconv_fused`` is true.

v1 covers YOLO9 detection only; every other family is blocked in
``support.py`` and raises here as well.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .exporter import BaseExporter

logger = logging.getLogger(__name__)

GGUF_SPEC_VERSION = 1
GGUF_FILE_TYPE_F32 = 0
GGUF_FILE_TYPE_F16 = 1

#: Families the GGUF writer knows how to describe, and their GGUF architecture.
_GGUF_ARCHITECTURES = {"yolo9": "yolov9"}

#: Top-level submodules that exist only for training and are never written.
_TRAINING_ONLY_PREFIXES = ("aux.", "aux_head.")


# =============================================================================
# Folding helpers
# =============================================================================


def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(weight, bias)`` of ``conv`` followed by eval-mode ``bn``.

    Computed in float64 with the module's own ``eps`` (1e-3 for YOLO9), then
    returned as float32. ``conv.bias`` is honoured when present.
    """
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError("BatchNorm without running statistics cannot be folded")
    w = conv.weight.detach().double()
    mean = bn.running_mean.detach().double()
    var = bn.running_var.detach().double()
    gamma = (
        bn.weight.detach().double() if bn.weight is not None else torch.ones_like(mean)
    )
    beta = bn.bias.detach().double() if bn.bias is not None else torch.zeros_like(mean)
    scale = gamma / torch.sqrt(var + float(bn.eps))
    conv_bias = (
        conv.bias.detach().double() if conv.bias is not None else torch.zeros_like(mean)
    )
    weight = w * scale.reshape(-1, 1, 1, 1)
    bias = beta + (conv_bias - mean) * scale
    return weight.float(), bias.float()


def _is_conv_bn(module: nn.Module) -> bool:
    return isinstance(getattr(module, "conv", None), nn.Conv2d) and isinstance(
        getattr(module, "bn", None), nn.BatchNorm2d
    )


def _is_repconv(module: nn.Module) -> bool:
    from ..models.yolo9.nn import RepConv

    return isinstance(module, RepConv)


def fuse_repconv(module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse a YOLO9 ``RepConv`` (kxk Conv-BN + 1x1 Conv-BN, summed) into one kxk conv."""
    conv_k, conv_1 = module.conv1.conv, module.conv2.conv
    k_h, k_w = conv_k.kernel_size
    if conv_1.kernel_size != (1, 1):
        raise ValueError(f"RepConv 1x1 branch has kernel {conv_1.kernel_size}")
    if (
        conv_k.stride != conv_1.stride
        or conv_k.groups != conv_1.groups
        or conv_k.dilation != (1, 1)
        or conv_1.padding != (0, 0)
        or conv_k.padding != ((k_h - 1) // 2, (k_w - 1) // 2)
    ):
        raise ValueError("RepConv branches are not centre-aligned; cannot fuse")
    w_k, b_k = fold_conv_bn(conv_k, module.conv1.bn)
    w_1, b_1 = fold_conv_bn(conv_1, module.conv2.bn)
    pad_h, pad_w = (k_h - 1) // 2, (k_w - 1) // 2
    w_1 = torch.nn.functional.pad(w_1, (pad_w, pad_w, pad_h, pad_h))
    return w_k + w_1, b_k + b_1


def iter_folded_tensors(
    nn_model: nn.Module, *, fuse_rep: bool = True
) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
    """Yield ``(conv path, weight, bias)`` for every inference convolution.

    Raises if a parameter of the inference graph is not covered by a
    convolution (a standalone BatchNorm, a Linear, ...), so new layer types
    can never be silently dropped from the file.
    """
    covered: set = set()
    modules = dict(nn_model.named_modules())

    def _skip(path: str) -> bool:
        return any((path + ".").startswith(p) for p in _TRAINING_ONLY_PREFIXES)

    for path, module in modules.items():
        if not path or _skip(path) or id(module) in covered:
            continue
        if fuse_rep and _is_repconv(module):
            weight, bias = fuse_repconv(module)
            for sub in module.modules():
                covered.add(id(sub))
            yield f"{path}.conv1.conv", weight, bias
        elif _is_conv_bn(module):
            weight, bias = fold_conv_bn(module.conv, module.bn)
            covered.update({id(module), id(module.conv), id(module.bn)})
            yield f"{path}.conv", weight, bias
        elif isinstance(module, nn.Conv2d):
            covered.add(id(module))
            weight = module.weight.detach().float()
            bias = (
                module.bias.detach().float()
                if module.bias is not None
                else torch.zeros(module.out_channels)
            )
            yield path, weight, bias

    for path, module in modules.items():
        if not path or _skip(path) or id(module) in covered:
            continue
        if any(True for _ in module.parameters(recurse=False)):
            raise NotImplementedError(
                f"GGUF export cannot serialize module {path!r} "
                f"({type(module).__name__}); only Conv2d / Conv-BN / RepConv are handled."
            )


# =============================================================================
# Metadata
# =============================================================================


def _yolo9_structure(model_wrapper, nn_model: nn.Module) -> Dict[str, object]:
    """Structural hyperparameters the C++ YOLO9 graph needs (``yolov9.*``)."""
    from ..models.yolo9.nn import YOLO9_CONFIGS

    size = str(model_wrapper.size)
    cfg = YOLO9_CONFIGS[size]
    head = nn_model.head
    stages = cfg["stages"]
    meta: Dict[str, object] = {
        "variant": size,
        "reg_max": int(head.reg_max),
        "strides": [int(s) for s in head.strides],
        "activation": "silu",
        "bn_eps": float(nn_model.backbone.conv0.bn.eps),
        "repconv_fused": True,
        "anchor_offset": 0.5,
        "first_block": str(cfg["first_block"]),
        "down_type": str(cfg["down_type"]),
        "csp_repeat": int(cfg["repeat_num"]),
        "csp_expand": 0.5,
        "stem_channels": [int(cfg["conv0_out"]), int(cfg["conv1_out"])],
        "first_block_channels": [
            int(cfg["first_block_out"]),
            int(cfg.get("first_block_part", cfg["first_block_out"])),
        ],
        "stage_down_channels": [int(s[0]) for s in stages],
        "stage_out_channels": [int(s[1]) for s in stages],
        "stage_part_channels": [int(s[2]) for s in stages],
        "spp_channels": [int(cfg["spp_out"]), int(cfg["spp_out"]) // 2],
        "spp_pool_kernel": 5,
        "spp_pool_repeat": 3,
        "neck_up1_channels": [int(v) for v in cfg["neck_elan_up1"]],
        "neck_up2_channels": [int(v) for v in cfg["neck_elan_up2"]],
        "neck_down1_channels": [int(cfg["neck_down1_out"])]
        + [int(v) for v in cfg["neck_elan_down1"]],
        "neck_down2_channels": [int(cfg["neck_down2_out"])]
        + [int(v) for v in cfg["neck_elan_down2"]],
        "head_in_channels": [int(c) for c in head.in_channels],
        "head_anchor_neck": int(head.anchor_neck),
        "head_class_neck": int(head.class_neck),
        "head_anchor_groups": int(head.groups),
    }
    return meta


def _pad_name(letterbox_pad: Optional[str]) -> str:
    from ..preprocess.letterbox import LETTERBOX_CENTER, normalize_letterbox_pad

    return "center" if normalize_letterbox_pad(letterbox_pad) == LETTERBOX_CENTER else "top_left"


def _labels(model_wrapper) -> List[str]:
    names = model_wrapper.names
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names]


def build_gguf_metadata(
    model_wrapper, nn_model: nn.Module, imgsz: Tuple[int, int], half: bool
) -> Dict[str, object]:
    """All GGUF key-values for a YOLO9 export, keyed by their full GGUF name."""
    from .onnx import _get_version

    family = model_wrapper._get_model_name()
    arch = _GGUF_ARCHITECTURES[family]
    labels = _labels(model_wrapper)
    name = f"{model_wrapper.FILENAME_PREFIX}{model_wrapper.size}"
    meta: Dict[str, object] = {
        "general.name": name,
        "general.license": "MIT",
        "general.file_type": GGUF_FILE_TYPE_F16 if half else GGUF_FILE_TYPE_F32,
        "libreyolo.version": _get_version(),
        "cv.spec_version": GGUF_SPEC_VERSION,
        "cv.task": "detect",
        "cv.labels": labels,
        "cv.num_classes": len(labels),
        "cv.input.height": int(imgsz[0]),
        "cv.input.width": int(imgsz[1]),
        "cv.input.multiple_of": 32,
        "cv.pre.resize": "cv2_linear",
        "cv.pre.fit": "letterbox",
        "cv.pre.pad": _pad_name(getattr(model_wrapper, "letterbox_pad", None)),
        "cv.pre.pad_value": 114,
        "cv.pre.channels": "rgb",
        "cv.pre.scale": 1.0 / 255.0,
        "cv.pre.mean": [0.0, 0.0, 0.0],
        "cv.pre.std": [1.0, 1.0, 1.0],
        "cv.post.kind": "dfl",
    }
    model_path = getattr(model_wrapper, "model_path", None)
    default_file = f"{name}.pt"
    if isinstance(model_path, (str, Path)) and Path(model_path).name == default_file:
        meta["general.source.url"] = f"https://huggingface.co/LibreYOLO/{name}"
    for key, value in _yolo9_structure(model_wrapper, nn_model).items():
        meta[f"{arch}.{key}"] = value
    return meta


# =============================================================================
# Writer
# =============================================================================


def _add_kv(writer, key: str, value) -> None:
    from gguf import GGUFValueType

    if isinstance(value, bool):
        writer.add_bool(key, value)
    elif isinstance(value, str):
        writer.add_string(key, value)
    elif isinstance(value, int):
        writer.add_uint32(key, value)
    elif isinstance(value, float):
        writer.add_float32(key, value)
    elif isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"GGUF key {key!r} has an empty array")
        first = value[0]
        if isinstance(first, str):
            sub = GGUFValueType.STRING
        elif isinstance(first, float):
            sub = GGUFValueType.FLOAT32
        elif isinstance(first, int):
            sub = GGUFValueType.UINT32
        else:
            raise TypeError(f"Unsupported array element for {key!r}: {type(first)}")
        writer.add_key_value(key, list(value), GGUFValueType.ARRAY, sub_type=sub)
    else:
        raise TypeError(f"Unsupported GGUF value for {key!r}: {type(value)}")


def write_gguf(
    model_wrapper,
    nn_model: nn.Module,
    output_path: str,
    *,
    imgsz: Tuple[int, int],
    half: bool = False,
) -> str:
    """Write ``nn_model`` (the live YOLO9 network) to ``output_path``."""
    try:
        import gguf
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "GGUF export requires the 'gguf' package. "
            "Install with: pip install 'libreyolo[gguf]'"
        ) from exc

    family = model_wrapper._get_model_name()
    if family not in _GGUF_ARCHITECTURES:
        raise NotImplementedError(f"GGUF export is not implemented for {family!r}.")
    arch = _GGUF_ARCHITECTURES[family]
    metadata = build_gguf_metadata(model_wrapper, nn_model, imgsz, half)

    writer = gguf.GGUFWriter(str(output_path), arch=arch)
    for key, value in metadata.items():
        if key == "general.file_type":
            writer.add_file_type(int(value))
        else:
            _add_kv(writer, key, value)

    count = 0
    with torch.no_grad():
        for path, weight, bias in iter_folded_tensors(nn_model):
            w = weight.cpu().numpy().astype(np.float16 if half else np.float32)
            writer.add_tensor(f"{path}.weight", np.ascontiguousarray(w))
            writer.add_tensor(
                f"{path}.bias", np.ascontiguousarray(bias.cpu().numpy().astype(np.float32))
            )
            count += 2

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    logger.info("Wrote %d tensors to %s", count, output_path)
    return str(output_path)


class GgufExporter(BaseExporter):
    """GGUF export for vision.cpp (YOLO9 detection in v1)."""

    format_name = "gguf"
    suffix = ".gguf"
    requires_onnx = False
    supports_int8 = False
    supports_fp16 = True
    # Weights are cast while writing; the live model stays FP32.
    apply_model_half = False

    def _resolve_params(self, output_path, imgsz, device, half, int8):
        # Serialization reads weights only; CPU keeps the FP64 fold portable.
        return super()._resolve_params(output_path, imgsz, torch.device("cpu"), half, int8)

    def _build_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        meta = super()._build_metadata(precision, dynamic, onnx_path, imgsz=imgsz)
        meta["dynamic"] = False
        return meta

    def _export(self, nn_model, dummy, *, output_path, half=False, **kwargs):
        imgsz = (int(dummy.shape[-2]), int(dummy.shape[-1]))
        return write_gguf(self.model, nn_model, output_path, imgsz=imgsz, half=half)


__all__ = [
    "GgufExporter",
    "build_gguf_metadata",
    "fold_conv_bn",
    "fuse_repconv",
    "iter_folded_tensors",
    "write_gguf",
]

"""Axelera Metis helpers for LibreYOLO export and inference.

LibreYOLO calls Axelera's Voyager SDK as a dependency the user installs:
``axelera-devkit`` (quantize and compile) and ``axelera-rt`` (run ``.axm``
models). Both are proprietary software under Axelera's EULA, published on
Axelera's own package index rather than PyPI; LibreYOLO never bundles or
vendors them.

Importing this module does not import torch or the SDK.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

# The SDK install and version helpers live with the backend, which must stay
# importable without torch; they are re-exported here for export callers.
from ..backends.axelera import (
    AUTOINSTALL_ENV,
    AXELERA_INDEX_URL,
    AXELERA_SDK_VERSION,
    DOCS_TROUBLESHOOTING,
    axelera_install_command,
    require_axelera_devkit,
    require_axelera_rt,
)

logger = logging.getLogger(__name__)

#: Compiler settings LibreYOLO owns: the artifact contract depends on them.
_RESERVED_CONFIG_FIELDS = {
    "output_axm_format": "LibreYOLO needs a .axm artifact (always True).",
    "aipu_cores_used": "The runtime spreads one single-core build across cores (always 1).",
    "output_dir": "LibreYOLO compiles in a temporary build directory.",
    "model_name": "The .axm is named after the checkpoint stem.",
    "model_metadata": "Framework-specific runtime hook; LibreYOLO decodes on the host.",
    "compiler_mode": "Export always quantizes and lowers.",
    "remove_output_dir": "LibreYOLO manages the build directory.",
}

#: Default compiler settings for the raw-head YOLO-NAS graph. Min-max
#: activation ranges: the SDK's histogram default clips the rare large
#: logits (positive class logits, distribution peaks), which measurably cost
#: accuracy on YOLO-NAS; see docs/axelera.md, section 7.
DEFAULT_QUANTIZATION_SCHEME = "per_tensor_min_max"
#: Per-task exceptions. Pose keeps more keypoint accuracy with histogram
#: activation ranges (0.7061 vs 0.6855 keypoint mAP50-95 on YOLO-NAS-pose-s).
DEFAULT_QUANTIZATION_SCHEME_BY_TASK = {"pose": "hybrid_per_tensor_per_channel"}
#: Per-model exceptions, keyed by (family, task); they win over the task
#: default. MobileNetV4 needs histogram ranges on top of its residual-stream
#: equalization: top-1 -1.7 -> +0.2 points on MobileNetV4-m (paired
#: 1000-image ImageNetV2 set, docs/axelera.md).
DEFAULT_QUANTIZATION_SCHEME_BY_MODEL = {
    ("mobilenetv4", "classify"): "hybrid_per_tensor_per_channel",
    # EfficientNetV2-b0: top-1 -1.7 (min-max) -> -1.0 / -1.1 points (same set).
    ("efficientnetv2", "classify"): "hybrid_per_tensor_per_channel",
    # MobileNetV3 depthwise backbone: per-channel ranges on depthwise inputs.
    ("deeplabv3", "semantic"): "hybrid_per_tensor_per_channel",
    # EfficientNet-B0 depthwise backbone: min-max lost 17.2 mAP50-95 points on
    # EfficientDet-d0, per-channel ranges 1.0 (300 COCO images, docs/axelera.md).
    ("efficientdet", "detect"): "hybrid_per_tensor_per_channel",
}


#: Per-size recipes, keyed by (family, task, size): the best measured way to
#: compile that size, applied by AxeleraExporter. Keys:
#: ``quantization_scheme`` (wins over the per-model, depthwise and task
#: defaults), ``output_windows`` (raw-head part -> (low, high), anchor heads),
#: ``stream_transform`` (MobileNetV4 residual-stream equalization and bound,
#: default on), ``pose_bounded_towers`` (YOLO-NAS-pose towers whose input is
#: bounded) and ``class_head_part`` (EfficientDet, opt-in: the class predictor's
#: pointwise conv as a second compiled part with this quantization scheme, e.g.
#: ``"per_tensor_min_max"``; under the family's histogram ranges the single-part
#: class logits cannot represent logits above about 0 on p3/p4/p7. Measured on
#: d0, cocoval300: -0.62 vs -1.00 single-part, not adopted) and
#: ``learned_rounding`` (QuickSRNet: learn each weight's INT8 rounding direction
#: after the Hardtanh equalization, deterministic, see
#: ``models/quicksrnet/quantize.learn_int8_rounding``) and
#: ``weight_equalization`` (Real-ESRGAN compact trunk: exact cross-layer
#: equalization through its PReLU activations,
#: ``models/realesrgan/quantize.equalize_conv_activation_ranges``) and
#: ``pointwise_learned_rounding`` (RTMDet: learned INT8 rounding of the CSPNeXt
#: pointwise convolutions, ``models/rtmdet/quantize.learn_pointwise_int8_rounding``). Results
#: with these recipes are in docs/axelera.md.
AXELERA_SIZE_RECIPES: dict[tuple[str, str, str], dict] = {
    # RT-DETRv2's neck part defaults to histogram ranges (exporter
    # _HYBRID_NECK_QUANTIZATION_SCHEMES); r18 and r34 were measured with the
    # per-tensor min-max neck (-1.25 / -2.03 on cocoval300) and keep it until
    # they are re-measured.
    ("rtdetrv2", "detect", "r18"): {"neck_quantization_scheme": "per_tensor_min_max"},
    ("rtdetrv2", "detect", "r34"): {"neck_quantization_scheme": "per_tensor_min_max"},
    # EfficientDet-d1 at its native 640 finds no L1 tiling (3.42 MB needed at
    # one-row tiles, 2.82 MB available); at d0's 512 canvas it compiles. The
    # exporter rebuilds the BiFPN for the canvas (same weights).
    ("efficientdet", "detect", "d1"): {"canvas": 512},
    # EfficientDet-d2 at its native 768: 3.15 MB needed, 2.54 MB available.
    ("efficientdet", "detect", "d2"): {"canvas": 512},
    # Real-ESRGAN-x4 (RRDB): the whole graph's constants overflow the DDR pool;
    # three compiled parts split at RRDB block 12 plus a host add fit.
    ("realesrgan", "restore", "x4"): {"rrdb_split": 12, "quantization_scheme": "per_tensor_min_max"},
    # FCOS-R50: the 800x800 canvas of PORTER E's measured multi-part gate.
    ("fcos", "detect", "r50"): {"canvas": 800},
    # U-Net-S at its 1024x2048 evaluation frame finds no L1 tiling (4.7 MB
    # needed, 4.1 MB available); at its 512x1024 training crop it compiles.
    ("unet", "semantic", "s"): {"canvas": (512, 1024)},
    # Real-ESRGAN-x2: the same RRDB body (its whole graph overflows the DDR pool too) after an exact
    # one-hot pixel-unshuffle convolution; 128x128 canvas (a 64x64 trunk, as x4 at 64).
    ("realesrgan", "restore", "x2"): {
        "rrdb_split": 12,
        "quantization_scheme": "per_tensor_min_max",
        "canvas": 128,
    },
    # RetinaNet-R50 resizes each image to a shorter side of 800 (longer up to
    # 1333); the fixed canvas fits the whole image inside, so at 800x800 most
    # images reach the network smaller (-2.25 end to end on 1000 COCO images).
    # A 1024x1024 canvas costs 1.64x the compute and recovers most of it
    # (-1.11); imgsz=800 stays available for speed.
    ("retinanet", "detect", "r50"): {"canvas": 1024},
    # RetinaNet-R50v2: GroupNorm heads; the ResNet-50 body on Metis, FPN + heads on the host.
    # Per-channel weights: with per-tensor min-max the body's quantized outputs drift to cosine 0.27-0.39
    # vs FP32 (0.81-0.97 with the hybrid scheme; -1.99 mAP50-95 end to end on CPU).
    # CenterNet-ResDCN18: deformable up-path on the host; the ResNet-18 body with per-channel weight ranges
    # (CPU-quantized end to end -0.83 mAP50-95; per-tensor ranges left the old multi-part route at 0.10).
    ("centernet", "detect", "resdcn18"): {
        "hybrid_split": "backbone",
        "quantization_scheme": "hybrid_per_tensor_per_channel",
    },
    # CenterNet-DLA34: DLA base (levels 2-5) on the Metis, DLAUp/IDAUp (deformable) on the host; per-channel
    # weight ranges (CPU-quantized end to end -0.61 mAP50-95 on cocoval100).
    ("centernet", "detect", "dla34"): {
        "hybrid_split": "backbone",
        "quantization_scheme": "hybrid_per_tensor_per_channel",
    },
    # HRNet-w32: stem to stage 3 on the Metis; stage 4 (where the device drifts from the quantized model) on the host.
    ("hrnet", "pose", "w32"): {
        "hybrid_split": "backbone",
        "quantization_scheme": "hybrid_per_tensor_per_channel",
    },
    # DEKR-w32 (640x640): the same trunk split; the whole graph has no working build (output paging / device lost).
    ("dekr", "pose", "w32"): {
        "hybrid_split": "backbone",
        "quantization_scheme": "hybrid_per_tensor_per_channel",
    },
    # ZipDepth-bnpu (384): encoder to stage 3 on the Metis; the GCNet context block (softmax) and the rest on the host.
    ("zipdepth", "depth", "bnpu"): {
        "hybrid_split": "backbone",
        "quantization_scheme": "hybrid_per_tensor_per_channel",
    },
    # PIDNet-s (1024): through layer 5 on the Metis (PagFM fusions in their exact sigmoid(-z) form); PAPPM, the
    # branch fusion and the head on the host.
    ("pidnet", "semantic", "s"): {
        "hybrid_split": "backbone",
        "quantization_scheme": "hybrid_per_tensor_per_channel",
    },
    ("retinanet", "detect", "r50v2"): {
        "canvas": 800,
        "hybrid_split": "backbone",
        "quantization_scheme": "hybrid_per_tensor_per_channel",
    },
    # DeepLabV3 r50/r101 at output stride 8: the ASPP convolutions (dilation up
    # to 36 on a 65x65 map) find no L1 tiling; per-tap 1x1 convolutions plus
    # zero-filled shifts are the same function and compile (the backbone alone
    # compiles; one 1x1 with all nine taps stacked does not).
    # DeepLabV3-R50: the same split recovers 4.7 mIoU (-0.40 vs -5.09 whole graph on 300 images).
    ("deeplabv3", "semantic", "r50"): {"aspp_shift": True, "hybrid_split": "backbone"},
    # DeepLabV3-R101's rewritten ASPP diverges on the device (cosine 0.53 vs the
    # CPU-quantized model, the backbone alone 0.995): the backbone runs on Metis
    # and the ASPP + head on the host. Without hybrid_split (aspp_shift alone)
    # the whole graph runs on Metis at -20.9 mIoU.
    ("deeplabv3", "semantic", "r101"): {"aspp_shift": True, "hybrid_split": "backbone"},
    # Histogram ranges recover most of the Mish backbone's loss (-6.8 -> -1.4);
    # the box-output window takes the rest of the way to -1.52 on 1000 images.
    ("yolo4", "detect", "b"): {
        "quantization_scheme": "hybrid_per_tensor_per_channel",
        "output_windows": {"reg": (-6.0, 6.0)},
    },
    # MobileNetV4: the stream transform + histogram recipe that validates m
    # costs s 2.6 and l 3.0 points; their best measured recipes differ.
    ("mobilenetv4", "classify", "s"): {
        "quantization_scheme": "hybrid_per_tensor_per_channel",
        "stream_transform": False,
    },
    ("mobilenetv4", "classify", "l"): {
        "quantization_scheme": "per_tensor_min_max",
        "stream_transform": True,
    },
    # YOLO-NAS-pose-s needs its keypoint tower bounded as well as the class tower.
    ("yolonas", "pose", "s"): {
        "pose_bounded_towers": (("cls_convs", "cls_pred"), ("pose_convs", "pose_pred")),
    },
    # VGG-16-BN: the folded batch norms widen min-max ranges; histogram ranges
    # take it from -2.9 to -0.2 top-1. VGG-19-BN does not follow (-6.0 vs -5.1
    # with min-max) and plain VGG-16/19 stay on min-max.
    ("vgg", "classify", "16bn"): {"quantization_scheme": "hybrid_per_tensor_per_channel"},
    # VGG-19-BN: on the device the error compounds through conv block 5, so
    # blocks 1-4 run on Metis and block 5 + classifier on the host (FP32):
    # top-1 -0.2 vs FP32 on 1000 ImageNetV2 images (+0.2 held-out) instead of -5.1.
    ("vgg", "classify", "19bn"): {"hybrid_split": "backbone"},
    # QuickSRNet-m2: learned INT8 rounding of the equalized weights (plus bias
    # correction) instead of nearest rounding + bias correction: PSNR -0.20 dB vs
    # FP32 on 296 COCO crops (-0.17 held-out) instead of -0.46 dB.
    ("quicksrnet", "restore", "m2"): {"learned_rounding": True},
    # Real-ESRGAN-x4t (SRVGGNetCompact): exact cross-layer equalization of the
    # conv+PReLU trunk before calibration: PSNR -0.19 dB vs FP32 on 296 COCO
    # crops (-0.20 held-out) instead of -0.49 dB.
    ("realesrgan", "restore", "x4t"): {"weight_equalization": True},
    # RTMDet-s: learned INT8 rounding of the CSPNeXt pointwise convolutions
    # after their input equalization (models/rtmdet/quantize).
    ("rtmdet", "detect", "s"): {"pointwise_learned_rounding": True},
}


def size_recipe(family: str, task: str, size: str) -> dict:
    """The per-size recipe for ``family``/``task``/``size`` (empty when the defaults apply)."""
    return dict(AXELERA_SIZE_RECIPES.get((family, task, str(size)), {}))


def default_quantization_scheme(family: str, task: str) -> str:
    """The activation-range scheme LibreYOLO compiles ``family``/``task`` with."""
    by_model = DEFAULT_QUANTIZATION_SCHEME_BY_MODEL.get((family, task))
    if by_model is not None:
        return by_model
    return DEFAULT_QUANTIZATION_SCHEME_BY_TASK.get(task, DEFAULT_QUANTIZATION_SCHEME)


#: Graphs with depthwise convolutions: per-channel ranges for the inputs of
#: depthwise convolutions (histogram ranges elsewhere). A depthwise conv never
#: mixes channels, so one per-tensor scale over an input whose channels span
#: very different ranges erases the small ones for good: YOLOX-n's first
#: depthwise input has channel maxima from 0.02 to 106 and scored 0.011 vs
#: FP32 0.261 mAP50-95 with per-tensor ranges (docs/axelera.md, section 7).
DEPTHWISE_QUANTIZATION_SCHEME = "hybrid_per_tensor_per_channel"
DEFAULT_RESOURCES_USED = 0.25


def check_compile_platform() -> None:
    """Axelera's compiler runs on Linux only (WSL2 counts as Linux)."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "Axelera export compiles with Axelera's toolchain, which runs on "
            f"Linux (or WSL2 on Windows); this platform is {sys.platform!r}. "
            "Export on Linux and copy the *_axelera_model directory to the "
            "machine that runs it."
        )


def _config_fields(config_cls) -> tuple[dict[str, str], set[str]]:
    """Map every accepted spelling (field name or alias) to its field name."""
    spellings: dict[str, str] = {}
    for name, field in config_cls.model_fields.items():
        spellings[name] = name
        alias = getattr(field, "alias", None)
        if isinstance(alias, str) and alias:
            spellings[alias] = name
    return spellings, set(config_cls.model_fields)


def normalize_compiler_overrides(
    overrides: Mapping[str, Any] | None, config_cls=None
) -> dict[str, Any]:
    """Validate user ``compiler_config`` keys against the real CompilerConfig."""
    if not overrides:
        return {}
    if not isinstance(overrides, Mapping):
        raise TypeError(
            "compiler_config must be a dict of CompilerConfig fields, got "
            f"{type(overrides).__name__}."
        )
    if config_cls is None:
        from axelera.compiler import CompilerConfig as config_cls
    spellings, fields = _config_fields(config_cls)
    normalized: dict[str, Any] = {}
    unknown = sorted(str(key) for key in overrides if key not in spellings)
    if unknown:
        raise ValueError(
            f"Unknown compiler_config field(s): {', '.join(unknown)}. "
            f"Valid fields: {', '.join(sorted(fields))}."
        )
    for key, value in overrides.items():
        name = spellings[key]
        if name in _RESERVED_CONFIG_FIELDS:
            raise ValueError(
                f"compiler_config cannot set {key!r}: {_RESERVED_CONFIG_FIELDS[name]}"
            )
        if name in normalized:
            raise ValueError(f"compiler_config sets {name!r} more than once.")
        normalized[name] = value
    return normalized


def build_compiler_config(
    model_name: str,
    *,
    build_dir: Path,
    quantization_scheme: str = DEFAULT_QUANTIZATION_SCHEME,
    resources_used: float = DEFAULT_RESOURCES_USED,
    overrides: Mapping[str, Any] | None = None,
):
    """Build the CompilerConfig for one export; returns ``(config, overrides)``."""
    from axelera.compiler import CompilerConfig

    normalized = normalize_compiler_overrides(overrides, CompilerConfig)
    settings: dict[str, Any] = {
        "quantization_scheme": quantization_scheme,
        "resources_used": resources_used,
        "save_error_artifact": True,
    }
    settings.update(normalized)
    settings.update(
        model_name=model_name,
        aipu_cores_used=1,
        output_axm_format=True,
        output_dir=Path(build_dir),
    )
    try:
        config = CompilerConfig(**settings)
    except Exception as exc:  # pydantic.ValidationError, without importing pydantic
        raise ValueError(f"Invalid compiler_config: {exc}") from exc
    return config, normalized


#: A constant Pad at or below this value in front of a MaxPool is -inf padding.
_MAXPOOL_PAD_FLOOR = -1e30


def merge_pad_into_maxpool(onnx_path: str, work_dir: Path) -> str:
    """Fold ``Pad(constant <= -1e30) -> MaxPool`` into the MaxPool's ``pads``.

    Exact: ONNX MaxPool pads with -inf. Darknet's stride-1 2x2 maxpool
    (yolov3-tiny) exports as a bottom/right Pad with -FLT_MAX followed by an
    unpadded MaxPool, and the AIPU accepts Pad only with a zero value. Returns
    ``onnx_path`` untouched when the graph has no such pair; otherwise writes
    the rewritten graph to ``work_dir`` and returns its path.
    """
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(onnx_path)
    graph = model.graph
    if not any(node.op_type == "Pad" for node in graph.node):
        return onnx_path
    constants = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    for node in graph.node:
        if (
            node.op_type == "Constant"
            and node.attribute
            and node.attribute[0].name == "value"
        ):
            constants[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    merged = []
    for pad in graph.node:
        if pad.op_type != "Pad" or len(pad.input) < 3 or not pad.input[2]:
            continue
        mode = next((a.s for a in pad.attribute if a.name == "mode"), b"constant")
        pads = constants.get(pad.input[1])
        value = constants.get(pad.input[2])
        if mode != b"constant" or pads is None or value is None:
            continue
        if float(np.asarray(value).reshape(-1)[0]) > _MAXPOOL_PAD_FLOOR:
            continue
        pads = [int(v) for v in np.asarray(pads).reshape(-1)]
        rank = len(pads) // 2
        users = consumers.get(pad.output[0], [])
        if (
            rank < 3
            or min(pads) < 0
            or pads[0]
            or pads[1]
            or pads[rank]
            or pads[rank + 1]
            or not users
            or any(user.op_type != "MaxPool" for user in users)
            or any(
                a.name == "auto_pad" and a.s not in (b"", b"NOTSET")
                for user in users
                for a in user.attribute
            )
        ):
            continue
        spatial = pads[2:rank] + pads[rank + 2 :]
        for user in users:
            old = next((a for a in user.attribute if a.name == "pads"), None)
            base = list(old.ints) if old is not None else [0] * len(spatial)
            if old is not None:
                user.attribute.remove(old)
            user.attribute.append(
                helper.make_attribute("pads", [b + s for b, s in zip(base, spatial)])
            )
            user.input[0] = pad.input[0]
        merged.append(pad)
    if not merged:
        return onnx_path
    for pad in merged:
        graph.node.remove(pad)
    out = Path(work_dir) / f"{Path(onnx_path).stem}.maxpool_pads.onnx"
    onnx.save(model, str(out))
    logger.info("Axelera: folded %d -inf Pad(s) into MaxPool padding", len(merged))
    return str(out)


def _channel_shuffle_to_conv(model) -> int:
    """Channel shuffles -> fixed 1x1 permutation convolutions; returns the count.

    Matches ``Reshape [B, g, C/g, H, W] -> Transpose [0, 2, 1, 3, 4] ->
    Reshape [B, C, H, W]`` (ShuffleNet-style blocks such as PicoDet's ESNet).
    Exact: output channel ``j`` copies input channel ``(j % g) * C/g + j // g``,
    and the 0/1 weights are exact in INT8. Inside the AIPU core the
    Reshape/Transpose pair is a layout transform, and the compiler rejects
    graphs with more than a handful of them.
    """
    import numpy as np
    from onnx import helper, numpy_helper, shape_inference

    graph = model.graph
    inferred = shape_inference.infer_shapes(model).graph
    shapes = {
        info.name: [dim.dim_value for dim in info.type.tensor_type.shape.dim]
        for info in (*inferred.value_info, *inferred.input, *inferred.output)
    }
    producers = {name: node for node in graph.node for name in node.output}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    count = 0
    for transpose in [node for node in graph.node if node.op_type == "Transpose"]:
        perm = next((list(a.ints) for a in transpose.attribute if a.name == "perm"), [])
        before = producers.get(transpose.input[0])
        after = consumers.get(transpose.output[0], [])
        if (
            perm != [0, 2, 1, 3, 4]
            or before is None
            or before.op_type != "Reshape"
            or len(after) != 1
            or after[0].op_type != "Reshape"
            or len(consumers.get(before.output[0], [])) != 1
        ):
            continue
        source = before.input[0]
        shape_in = shapes.get(source)
        shape_mid = shapes.get(before.output[0])
        if (
            not shape_in
            or len(shape_in) != 4
            or shape_in != shapes.get(after[0].output[0])
            or not shape_mid
            or len(shape_mid) != 5
            or shape_mid[1] * shape_mid[2] != shape_in[1]
        ):
            continue
        channels, groups, per_group = shape_in[1], shape_mid[1], shape_mid[2]
        weight = np.zeros((channels, channels, 1, 1), np.float32)
        for out in range(channels):
            weight[out, (out % groups) * per_group + out // groups] = 1.0
        name = f"{transpose.name or transpose.output[0]}_shuffle_conv".strip("/")
        graph.initializer.append(numpy_helper.from_array(weight, f"{name}_w"))
        conv = helper.make_node(
            "Conv",
            [source, f"{name}_w"],
            [after[0].output[0]],
            name=name,
            kernel_shape=[1, 1],
        )
        index = list(graph.node).index(before)
        for node in (before, transpose, after[0]):
            graph.node.remove(node)
        graph.node.insert(index, conv)
        count += 1
    return count


def _graph_tables(model):
    """Shapes, producers, consumers and constant initializers of an ONNX model."""
    from onnx import numpy_helper, shape_inference

    inferred = shape_inference.infer_shapes(model).graph
    shapes = {
        info.name: [dim.dim_value for dim in info.type.tensor_type.shape.dim]
        for info in (*inferred.value_info, *inferred.input, *inferred.output)
    }
    graph = model.graph
    producers = {name: node for node in graph.node for name in node.output}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    return shapes, producers, consumers, weights


def _attributes(node) -> dict:
    from onnx import helper

    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _drop_unused_initializers(graph) -> None:
    used = {name for node in graph.node for name in node.input}
    for init in [i for i in graph.initializer if i.name not in used]:
        graph.initializer.remove(init)


def _topological_sort(graph) -> None:
    """Reorder ``graph.node`` so every node follows its producers (stable)."""
    available = (
        {i.name for i in graph.input} | {i.name for i in graph.initializer} | {""}
    )
    nodes = list(graph.node)
    pending = list(range(len(nodes)))
    order = []
    while pending:
        ready = [
            i for i in pending if all(name in available for name in nodes[i].input)
        ]
        if not ready:
            raise ValueError("Axelera graph pass left a cycle or a dangling input.")
        for i in ready:
            order.append(i)
            available.update(nodes[i].output)
        done = set(ready)
        pending = [i for i in pending if i not in done]
    copies = [type(nodes[i])() for i in order]
    for copy_, i in zip(copies, order):
        copy_.CopyFrom(nodes[i])
    del graph.node[:]
    graph.node.extend(copies)


#: Depthwise convolutions whose inputs are split into range groups by
#: ``dw_channel_groups``: a channel joins group ``floor(log4(max_span /
#: span))``, capped here. Three groups recover most of what per-channel input
#: ranges give (PicoDet-s fake-quant box error: 1.62 px with one range per
#: tensor, 0.67 px with three groups, 0.62 px per channel).
DW_CHANNEL_GROUPS = 3
_ELEMENTWISE_ACTIVATIONS = (
    "Relu",
    "LeakyRelu",
    "Clip",
    "Sigmoid",
    "HardSwish",
    "HardSigmoid",
    "Tanh",
)


def _depthwise_input_site(node, producers, consumers, weights):
    """``(producer_conv, activation_nodes)`` feeding a depthwise conv, or None.

    Accepts ``Conv -> DW``, ``Conv -> act -> DW`` and the exported HardSwish
    ``Conv -> HardSigmoid -> Mul(conv, .) -> DW``, where the producer's output
    feeds nothing else.
    """
    attrs = _attributes(node)
    weight = weights.get(node.input[1])
    if weight is None or weight.ndim != 4 or weight.shape[1] != 1:
        return None
    if attrs.get("group", 1) != weight.shape[0] or weight.shape[0] < 2:
        return None
    source = producers.get(node.input[0])
    if source is None:
        return None
    if source.op_type == "Conv":
        conv, activation = source, []
    elif source.op_type in _ELEMENTWISE_ACTIVATIONS and len(source.input) == 1:
        conv, activation = producers.get(source.input[0]), [source]
    elif source.op_type == "Mul" and len(source.input) == 2:
        gate = producers.get(source.input[1]) or producers.get(source.input[0])
        if gate is None or gate.op_type != "HardSigmoid":
            return None
        if set(source.input) != {gate.input[0], gate.output[0]}:
            return None
        conv, activation = producers.get(gate.input[0]), [gate, source]
    else:
        return None
    if conv is None or conv.op_type != "Conv" or conv.input[1] not in weights:
        return None
    if _attributes(conv).get("group", 1) != 1:
        return None
    expected = {n.output[0] for n in activation} if activation else {node.output[0]}
    users = {n.output[0] for n in consumers.get(conv.output[0], [])}
    if users != expected:
        return None
    return conv, activation


def _dw_channel_groups(model, samples=None) -> int:
    """Give a depthwise conv's input channels a few per-tensor ranges.

    With one INT8 range per tensor, a depthwise input whose channels span very
    different ranges loses the small channels; a per-channel range is what
    helps (the SDK's hybrid scheme does that, but puts histogram ranges on
    every other activation). This pass splits the producer convolution by
    output channels into range groups (measured on ``samples``), repeats the
    activation per group, runs one depthwise conv per group, and restores
    the channel order with a fixed 1x1 permutation conv; consumers of the
    activation other than the depthwise conv get it back the same way. Each
    group's tensor then gets its own range, and the result is exact in FP32.
    Returns the number of rewritten depthwise convs (0 without samples).
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    if not samples:
        return 0
    _, producers, consumers, weights = _graph_tables(model)
    graph = model.graph
    sites = {}
    for node in graph.node:
        if node.op_type == "Conv":
            site = _depthwise_input_site(node, producers, consumers, weights)
            if site is not None:
                sites[node.input[0]] = (node, *site)
    if not sites:
        return 0
    import onnxruntime as ort

    probe = type(model)()
    probe.CopyFrom(model)
    known = {o.name for o in probe.graph.output}
    for name in sites:
        if name not in known:
            probe.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )
    session = ort.InferenceSession(
        probe.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]
    low = {name: None for name in sites}
    high = {name: None for name in sites}
    for sample in samples:
        values = dict(
            zip(
                out_names,
                session.run(None, {input_name: np.asarray(sample, np.float32)}),
            )
        )
        for name in sites:
            v = values[name]
            lo, hi = v.min(axis=(0, 2, 3)), v.max(axis=(0, 2, 3))
            low[name] = lo if low[name] is None else np.minimum(low[name], lo)
            high[name] = hi if high[name] is None else np.maximum(high[name], hi)
    count = 0
    for name, (dw, conv, activation) in sites.items():
        span = np.maximum(high[name] - low[name], 1e-9)
        group = np.minimum(
            np.floor(np.log(span.max() / span) / np.log(4.0)), DW_CHANNEL_GROUPS - 1
        ).astype(int)
        members = [np.flatnonzero(group == g) for g in range(DW_CHANNEL_GROUPS)]
        members = [m for m in members if m.size]
        if len(members) < 2:
            continue
        tag = (dw.name or dw.output[0]).strip("/").replace("/", "_") + "_dwg"
        conv_w, dw_w = weights[conv.input[1]], weights[dw.input[1]]
        conv_b = weights.get(conv.input[2]) if len(conv.input) > 2 else None
        dw_b = weights.get(dw.input[2]) if len(dw.input) > 2 else None
        conv_attrs, dw_attrs = _attributes(conv), _attributes(dw)
        nodes, act_outs, dw_outs = [], [], []
        for g, rows in enumerate(members):

            def init(array, suffix, prefix=f"{tag}{g}"):
                name = f"{prefix}_{suffix}"
                graph.initializer.append(
                    numpy_helper.from_array(
                        np.ascontiguousarray(array).astype(np.float32), name
                    )
                )
                return name

            ins = [conv.input[0], init(conv_w[rows], "pw")] + (
                [init(conv_b[rows], "pb")] if conv_b is not None else []
            )
            nodes.append(
                helper.make_node(
                    "Conv", ins, [f"{tag}{g}_c"], name=f"{tag}{g}_c", **conv_attrs
                )
            )
            current = f"{tag}{g}_c"
            if len(activation) == 2:  # HardSwish = x * HardSigmoid(x)
                gate_attrs = _attributes(activation[0])
                nodes.append(
                    helper.make_node(
                        "HardSigmoid",
                        [current],
                        [f"{tag}{g}_g"],
                        name=f"{tag}{g}_g",
                        **gate_attrs,
                    )
                )
                nodes.append(
                    helper.make_node(
                        "Mul",
                        [current, f"{tag}{g}_g"],
                        [f"{tag}{g}_a"],
                        name=f"{tag}{g}_a",
                    )
                )
                current = f"{tag}{g}_a"
            elif activation:
                act = activation[0]
                nodes.append(
                    helper.make_node(
                        act.op_type,
                        [current],
                        [f"{tag}{g}_a"],
                        name=f"{tag}{g}_a",
                        **_attributes(act),
                    )
                )
                current = f"{tag}{g}_a"
            act_outs.append(current)
            attrs = dict(dw_attrs, group=int(rows.size))
            ins = [current, init(dw_w[rows], "dw")] + (
                [init(dw_b[rows], "db")] if dw_b is not None else []
            )
            nodes.append(
                helper.make_node(
                    "Conv", ins, [f"{tag}{g}_y"], name=f"{tag}{g}_y", **attrs
                )
            )
            dw_outs.append(f"{tag}{g}_y")
        order = np.concatenate(members)
        channels = order.size
        perm = np.zeros((channels, channels, 1, 1), np.float32)
        perm[order, np.arange(channels), 0, 0] = (
            1.0  # concat position k holds channel order[k]
        )
        graph.initializer.append(numpy_helper.from_array(perm, f"{tag}_perm"))
        nodes.append(
            helper.make_node(
                "Concat", dw_outs, [f"{tag}_ycat"], name=f"{tag}_ycat", axis=1
            )
        )
        nodes.append(
            helper.make_node(
                "Conv",
                [f"{tag}_ycat", f"{tag}_perm"],
                [dw.output[0]],
                name=f"{tag}_yperm",
                kernel_shape=[1, 1],
            )
        )
        others = [n for n in consumers.get(name, []) if n.output[0] != dw.output[0]]
        if others:  # rebuild the activation in channel order for its other consumers
            nodes.append(
                helper.make_node(
                    "Concat", act_outs, [f"{tag}_acat"], name=f"{tag}_acat", axis=1
                )
            )
            nodes.append(
                helper.make_node(
                    "Conv",
                    [f"{tag}_acat", f"{tag}_perm"],
                    [f"{tag}_afull"],
                    name=f"{tag}_aperm",
                    kernel_shape=[1, 1],
                )
            )
            for other in others:
                for i, value in enumerate(other.input):
                    if value == name:
                        other.input[i] = f"{tag}_afull"
        for old in [conv, *activation, dw]:
            graph.node.remove(old)
        graph.node.extend(nodes)
        count += 1
    if count:
        _drop_unused_initializers(graph)
        _topological_sort(graph)
    return count


#: Passes that measure the graph on calibration samples (``samples=``).
_MAX_REDUCTION_BLOCKS = 128


def _pooled_branch_split(model) -> int:
    """Split ``Resize(1x1 -> HxW) -> Concat -> Conv 1x1`` (ASPP image pooling).

    DeepLabV3's image-pooling branch upsamples a 1x1 map to the feature size
    and concatenates it before a 1x1 projection. The compiler lowers that
    Resize to a depthwise op whose window (2H - 1 rows) does not fit the
    depthwise unit's scratchpad. The projection is linear, so it splits
    exactly into a convolution over the other branches plus a 1x1 convolution
    of the pooled vector, added as a per-channel ``[1, C, 1, 1]`` term.
    Returns the number of rewritten sites.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    shapes, _, consumers, weights = _graph_tables(model)
    graph = model.graph
    count = 0
    for resize in [node for node in graph.node if node.op_type == "Resize"]:
        source_shape = shapes.get(resize.input[0])
        users = consumers.get(resize.output[0], [])
        if not source_shape or len(source_shape) != 4 or source_shape[2:] != [1, 1]:
            continue
        if len(users) != 1 or users[0].op_type != "Concat":
            continue
        concat = users[0]
        if _attributes(concat).get("axis") not in (1, -3):
            continue
        projections = consumers.get(concat.output[0], [])
        if len(projections) != 1 or projections[0].op_type != "Conv":
            continue
        conv = projections[0]
        attrs = _attributes(conv)
        if (
            attrs.get("group", 1) != 1
            or list(attrs.get("kernel_shape", [1, 1])) != [1, 1]
            or conv.input[1] not in weights
        ):
            continue
        weight = weights[conv.input[1]]
        offsets, start = {}, 0
        for name in concat.input:
            offsets[name] = (start, start + shapes[name][1])
            start += shapes[name][1]
        others = [name for name in concat.input if name != resize.output[0]]
        keep = np.concatenate([np.arange(*offsets[name]) for name in others])
        low, high = offsets[resize.output[0]]
        tag = (conv.name or conv.output[0]).strip("/").replace("/", "_")
        graph.initializer.extend(
            [
                numpy_helper.from_array(
                    weight[:, keep].astype(np.float32), f"{tag}_main_w"
                ),
                numpy_helper.from_array(
                    weight[:, low:high].astype(np.float32), f"{tag}_pool_w"
                ),
            ]
        )
        bias = [conv.input[2]] if len(conv.input) > 2 and conv.input[2] else []
        nodes = [
            helper.make_node(
                "Concat", others, [f"{tag}_branches"], name=f"{tag}_branches", axis=1
            ),
            helper.make_node(
                "Conv",
                [resize.input[0], f"{tag}_pool_w"],
                [f"{tag}_pool"],
                name=f"{tag}_pool",
                kernel_shape=[1, 1],
            ),
            helper.make_node(
                "Conv",
                [f"{tag}_branches", f"{tag}_main_w", *bias],
                [f"{tag}_main"],
                name=f"{tag}_main",
                kernel_shape=[1, 1],
            ),
            helper.make_node(
                "Add",
                [f"{tag}_main", f"{tag}_pool"],
                [conv.output[0]],
                name=f"{tag}_add",
            ),
        ]
        for node in (resize, concat, conv):
            graph.node.remove(node)
        graph.node.extend(nodes)
        count += 1
    if count:
        _drop_unused_initializers(graph)
        _topological_sort(graph)
    return count


def _conv_reduction_split(model) -> int:
    """Split convolutions whose reduction is too large into channel slices.

    ``conv(x) = sum_i conv_i(x[:, slice_i])`` with the weight's input channels
    sliced the same way: exact up to summation order. Channel-axis Slice and
    same-shape Add run on the AIPU. Used for DeepLabV3-ResNet's ASPP 3x3
    convolutions on 2048 channels (288 blocks). Returns the number of
    rewritten convolutions.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    _, _, _, weights = _graph_tables(model)
    graph = model.graph
    count = 0
    for conv in [node for node in graph.node if node.op_type == "Conv"]:
        attrs = _attributes(conv)
        if attrs.get("group", 1) != 1 or conv.input[1] not in weights:
            continue
        weight = weights[conv.input[1]]
        _, channels, kh, kw = weight.shape
        if -(-channels * kh * kw // 64) <= _MAX_REDUCTION_BLOCKS:
            continue
        parts = next(
            (
                k
                for k in range(2, channels + 1)
                if channels % k == 0
                and (channels // k) % 64 == 0
                and -(-(channels // k) * kh * kw // 64) <= _MAX_REDUCTION_BLOCKS
            ),
            None,
        )
        if parts is None:
            continue
        step = channels // parts
        tag = (conv.name or conv.output[0]).strip("/").replace("/", "_") + "_split"
        graph.initializer.append(
            numpy_helper.from_array(np.array([1], np.int64), f"{tag}_axis")
        )
        bias = [conv.input[2]] if len(conv.input) > 2 and conv.input[2] else []
        nodes, total = [], None
        for part in range(parts):
            graph.initializer.extend(
                [
                    numpy_helper.from_array(
                        np.array([part * step], np.int64), f"{tag}_s{part}"
                    ),
                    numpy_helper.from_array(
                        np.array([(part + 1) * step], np.int64), f"{tag}_e{part}"
                    ),
                    numpy_helper.from_array(
                        np.ascontiguousarray(
                            weight[:, part * step : (part + 1) * step]
                        ).astype(np.float32),
                        f"{tag}_w{part}",
                    ),
                ]
            )
            nodes.append(
                helper.make_node(
                    "Slice",
                    [conv.input[0], f"{tag}_s{part}", f"{tag}_e{part}", f"{tag}_axis"],
                    [f"{tag}_x{part}"],
                    name=f"{tag}_x{part}",
                )
            )
            nodes.append(
                helper.make_node(
                    "Conv",
                    [f"{tag}_x{part}", f"{tag}_w{part}", *(bias if part == 0 else [])],
                    [f"{tag}_y{part}"],
                    name=f"{tag}_y{part}",
                    **attrs,
                )
            )
            if total is None:
                total = f"{tag}_y{part}"
                continue
            out = conv.output[0] if part == parts - 1 else f"{tag}_sum{part}"
            nodes.append(
                helper.make_node("Add", [total, f"{tag}_y{part}"], [out], name=out)
            )
            total = out
        index = list(graph.node).index(conv)
        graph.node.remove(conv)
        for offset, node in enumerate(nodes):
            graph.node.insert(index + offset, node)
        count += 1
    if count:
        _drop_unused_initializers(graph)
    return count


_FC_PASS_THROUGH = (
    "Relu",
    "LeakyRelu",
    "Sigmoid",
    "Tanh",
    "HardSwish",
    "Identity",
    "Dropout",
)


def _fc_rowconv(model) -> int:
    """Run a flattened-map classifier head as convolutions (VGG's fc6/fc7/fc8).

    ``Flatten`` of a ``[1, C, H, W]`` map with ``H * W > 1`` followed by a
    Gemm chain: the first Gemm (one reduction over ``C*H*W``, 25088 for
    VGG-16, which the compiler cannot map) becomes ``H`` row convolutions
    (a ``1 x W`` kernel on each H-slice of the map) summed by Adds, the later
    Gemms become 1x1 convolutions, and a Flatten restores ``[1, N]``. Exact
    up to summation order. A single ``H x W`` convolution is not an option:
    the quantizer's own large-kernel rewrite fails on it. Classifiers that
    pool to 1x1 first (ResNet, MobileNet) are left alone. Returns the
    number of rewritten heads.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    shapes, _, consumers, weights = _graph_tables(model)
    graph = model.graph
    count = 0
    for flatten in [node for node in graph.node if node.op_type == "Flatten"]:
        shape = shapes.get(flatten.input[0])
        if not shape or len(shape) != 4 or shape[0] != 1 or shape[2] * shape[3] <= 1:
            continue
        channels, height, width = shape[1:]
        chain, current = [], flatten.output[0]
        while True:
            users = consumers.get(current, [])
            if len(users) != 1:
                break
            user = users[0]
            if user.op_type == "Gemm":
                attrs = _attributes(user)
                if (
                    attrs.get("transA", 0)
                    or float(attrs.get("alpha", 1.0)) != 1.0
                    or float(attrs.get("beta", 1.0)) != 1.0
                    or user.input[1] not in weights
                ):
                    break
                chain.append(user)
            elif user.op_type in _FC_PASS_THROUGH:
                chain.append(user)
            else:
                break
            current = user.output[0]
        gemms = [n for n in chain if n.op_type == "Gemm"]
        if not gemms:
            continue
        tag = (flatten.name or flatten.output[0]).strip("/").replace("/", "_") + "_rows"
        graph.initializer.append(
            numpy_helper.from_array(np.array([2], np.int64), f"{tag}_axis")
        )
        nodes, previous, first = [], flatten.input[0], True
        for node in chain:
            out = f"{node.output[0]}_4d"
            if node.op_type != "Gemm":
                op = "Identity" if node.op_type == "Dropout" else node.op_type
                attrs = {} if op == "Identity" else _attributes(node)
                nodes.append(helper.make_node(op, [previous], [out], name=out, **attrs))
                previous = out
                continue
            weight = weights[node.input[1]]
            if not _attributes(node).get("transB", 0):
                weight = weight.T
            bias = [node.input[2]] if len(node.input) > 2 and node.input[2] else []
            if first:
                kernel = weight.reshape(weight.shape[0], channels, height, width)
                total = None
                for row in range(height):
                    name = f"{tag}{row}"
                    graph.initializer.extend(
                        [
                            numpy_helper.from_array(
                                np.array([row], np.int64), f"{name}_s"
                            ),
                            numpy_helper.from_array(
                                np.array([row + 1], np.int64), f"{name}_e"
                            ),
                            numpy_helper.from_array(
                                np.ascontiguousarray(
                                    kernel[:, :, row : row + 1, :]
                                ).astype(np.float32),
                                f"{name}_w",
                            ),
                        ]
                    )
                    nodes.append(
                        helper.make_node(
                            "Slice",
                            [previous, f"{name}_s", f"{name}_e", f"{tag}_axis"],
                            [f"{name}_x"],
                            name=f"{name}_x",
                        )
                    )
                    nodes.append(
                        helper.make_node(
                            "Conv",
                            [f"{name}_x", f"{name}_w", *(bias if row == 0 else [])],
                            [f"{name}_y"],
                            name=f"{name}_y",
                            kernel_shape=[1, width],
                        )
                    )
                    if total is None:
                        total = f"{name}_y"
                        continue
                    summed = out if row == height - 1 else f"{name}_sum"
                    nodes.append(
                        helper.make_node(
                            "Add", [total, f"{name}_y"], [summed], name=summed
                        )
                    )
                    total = summed
                if height == 1:
                    nodes.append(helper.make_node("Identity", [total], [out], name=out))
                first = False
            else:
                graph.initializer.append(
                    numpy_helper.from_array(
                        weight.reshape(*weight.shape, 1, 1).astype(np.float32),
                        f"{node.output[0]}_w1x1",
                    )
                )
                nodes.append(
                    helper.make_node(
                        "Conv",
                        [previous, f"{node.output[0]}_w1x1", *bias],
                        [out],
                        name=out,
                        kernel_shape=[1, 1],
                    )
                )
            previous = out
        last = chain[-1].output[0]
        nodes.append(
            helper.make_node(
                "Flatten", [previous], [last], name=f"{tag}_flatten", axis=1
            )
        )
        for node in [flatten, *chain]:
            graph.node.remove(node)
        graph.node.extend(nodes)
        count += 1
    if count:
        _drop_unused_initializers(graph)
        _topological_sort(graph)
    return count


def _constant(graph, weights, name):
    """Value of an initializer or Constant-node output, else None."""
    from onnx import numpy_helper

    if name in weights:
        return weights[name]
    for node in graph.node:
        if node.op_type == "Constant" and node.output[0] == name and node.attribute:
            return numpy_helper.to_array(node.attribute[0].t)
    return None


def _reduce_axes(graph, weights, node):
    attrs = _attributes(node)
    if "axes" in attrs:
        return [int(a) for a in attrs["axes"]]
    if len(node.input) > 1 and node.input[1]:
        value = _constant(graph, weights, node.input[1])
        return None if value is None else [int(a) for a in value.reshape(-1)]
    return None


def _float_div_to_mul(model) -> int:
    """``Div(x, c)`` by a floating-point constant becomes ``Mul(x, 1 / c)``.

    EfficientDet's BiFPN normalizes its learned fusion weights with a Div that
    the AIPU does not run; the weights are constant after export, so the
    division is a scaling. Integer divisions (index arithmetic in decodes) are
    left alone. Exact up to float rounding. Returns the number of rewrites.
    """
    import numpy as np
    from onnx import numpy_helper

    _, _, _, weights = _graph_tables(model)
    graph = model.graph
    count = 0
    for node in [n for n in graph.node if n.op_type == "Div"]:
        value = _constant(graph, weights, node.input[1])
        if value is None or not np.issubdtype(value.dtype, np.floating):
            continue
        if np.any(value == 0):
            continue
        name = f"{node.input[1]}_reciprocal_{count}"
        graph.initializer.append(
            numpy_helper.from_array((1.0 / value).astype(value.dtype), name)
        )
        node.op_type = "Mul"
        node.input[1] = name
        count += 1
    return count


def _stack_sum_to_add(model) -> int:
    """``ReduceSum(Concat(Unsqueeze(t_i)))`` over the stacking axis becomes ``t_0 + t_1 + ...``.

    The BiFPN fuses its inputs by stacking and summing them; the reduction
    over the new axis is not an AIPU op, an Add chain is. Exact up to
    summation order. Returns the number of rewrites.
    """
    from onnx import helper

    _, producers, consumers, weights = _graph_tables(model)
    graph = model.graph
    count = 0
    for reduce in [n for n in graph.node if n.op_type == "ReduceSum"]:
        concat = producers.get(reduce.input[0])
        if concat is None or concat.op_type != "Concat":
            continue
        if len(consumers.get(concat.output[0], [])) != 1:
            continue
        axis = _attributes(concat).get("axis")
        keep = _attributes(reduce).get("keepdims", 1)
        unsqueezes = [producers.get(name) for name in concat.input]
        if (
            keep != 0
            or _reduce_axes(graph, weights, reduce) != [axis]
            or any(
                u is None
                or u.op_type != "Unsqueeze"
                or _reduce_axes(graph, weights, u) != [axis]
                or len(consumers.get(u.output[0], [])) != 1
                for u in unsqueezes
            )
        ):
            continue
        terms = [u.input[0] for u in unsqueezes]
        nodes, total = [], terms[0]
        for index, term in enumerate(terms[1:]):
            out = (
                reduce.output[0]
                if index == len(terms) - 2
                else f"{reduce.output[0]}_sum{index}"
            )
            nodes.append(helper.make_node("Add", [total, term], [out], name=out))
            total = out
        for node in [reduce, concat, *unsqueezes]:
            graph.node.remove(node)
        graph.node.extend(nodes)
        count += 1
    if count:
        _drop_unused_initializers(graph)
        _topological_sort(graph)
    return count


def _mean_hw_to_gap(model) -> int:
    """``ReduceMean`` over H and W (keepdims) becomes ``GlobalAveragePool``. Exact."""
    from onnx import helper

    _, _, _, weights = _graph_tables(model)
    graph = model.graph
    count = 0
    for node in [n for n in graph.node if n.op_type == "ReduceMean"]:
        if _attributes(node).get("keepdims", 1) != 1:
            continue
        if sorted(_reduce_axes(graph, weights, node) or []) not in ([2, 3], [-2, -1]):
            continue
        index = list(graph.node).index(node)
        graph.node.remove(node)
        graph.node.insert(
            index,
            helper.make_node(
                "GlobalAveragePool", [node.input[0]], list(node.output), name=node.name
            ),
        )
        count += 1
    if count:
        _drop_unused_initializers(graph)
    return count


#: Largest map a GlobalAveragePool may average in one piece after gap_staged.
_GAP_MAX_SIDE = 4


def _gap_staged(model) -> int:
    """Average large maps in 2x2 steps before the GlobalAveragePool.

    A global average over a large map (EfficientDet-d0's first squeeze-excite
    sees 256x256) fails in the compiler's integer sum-reduce; so does a single
    16x16 averaging window. A chain of 2x2 / stride-2 AveragePools down to
    at most 4x4, then the global average, gives the same mean (equal windows).
    Returns the number of rewritten pools.
    """
    from onnx import helper

    shapes, _, _, _ = _graph_tables(model)
    graph = model.graph
    count = 0
    for node in [n for n in graph.node if n.op_type == "GlobalAveragePool"]:
        shape = shapes.get(node.input[0])
        if not shape or len(shape) != 4:
            continue
        height, width = shape[2], shape[3]
        nodes, current = [], node.input[0]
        while max(height, width) > _GAP_MAX_SIDE and height % 2 == 0 and width % 2 == 0:
            out = f"{node.input[0]}_avg2_{count}_{len(nodes)}"
            nodes.append(
                helper.make_node(
                    "AveragePool",
                    [current],
                    [out],
                    name=out,
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                )
            )
            current, height, width = out, height // 2, width // 2
        if not nodes:
            continue
        node.input[0] = current
        graph.node.extend(nodes)
        count += 1
    if count:
        _topological_sort(graph)
    return count

#: Largest averaging window (kh * kw) the compiler takes for a pooling window
#: that does not cover the whole map; PP-LiteSeg-t75's 12x24 window (288) fails.
_AVGPOOL_MAX_WINDOW = 255
#: Largest whole-map window measured to compile (PP-LiteSeg-t50's 16x32 pyramid bin).
_AVGPOOL_MAX_GLOBAL_WINDOW = 512


def _avgpool_staged(model) -> int:
    """Split wide non-overlapping average pools into a 2x2 step chain.

    PP-LiteSeg's pyramid pooling averages its stride-32 map over 2x2 bins; at
    768x1536 one bin is a 12x24 window, more than the compiler's averaging
    unit takes ("Input should be less than 256"). An ``AveragePool`` with
    ``kernel == stride``, no padding and no ceil mode averages equal windows,
    so first averaging 2x2 (or 2x1 / 1x2) blocks and then the remaining
    window gives the same mean. Steps are added until the window is below
    :data:`_AVGPOOL_MAX_WINDOW` (or, for a window that covers the whole map,
    at most :data:`_AVGPOOL_MAX_GLOBAL_WINDOW`). Exact up to summation order.
    Returns the number of rewritten pools.
    """
    from onnx import helper

    shapes, _, _, _ = _graph_tables(model)
    graph = model.graph
    count = 0
    for node in [n for n in graph.node if n.op_type == "AveragePool"]:
        attrs = _attributes(node)
        shape = shapes.get(node.input[0])
        kernel = list(attrs.get("kernel_shape", []))
        if (
            not shape
            or len(shape) != 4
            or len(kernel) != 2
            or list(attrs.get("strides", [1, 1])) != kernel
            or any(attrs.get("pads", [0, 0, 0, 0]))
            or attrs.get("ceil_mode", 0)
            or attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET")
            or list(attrs.get("dilations", [1, 1])) != [1, 1]
        ):
            continue
        height, width = shape[2], shape[3]
        kh, kw = kernel
        limit = (
            _AVGPOOL_MAX_GLOBAL_WINDOW
            if (kh, kw) == (height, width)
            else _AVGPOOL_MAX_WINDOW
        )
        nodes, current = [], node.input[0]
        while kh * kw > limit and (kh % 2 == 0 or kw % 2 == 0):
            step = [2 if kh % 2 == 0 else 1, 2 if kw % 2 == 0 else 1]
            out = f"{node.input[0]}_avgstep_{count}_{len(nodes)}"
            nodes.append(
                helper.make_node(
                    "AveragePool",
                    [current],
                    [out],
                    name=out,
                    kernel_shape=step,
                    strides=step,
                )
            )
            current, kh, kw = out, kh // step[0], kw // step[1]
        if not nodes:
            continue
        node.input[0] = current
        for attr in [a for a in node.attribute if a.name in ("kernel_shape", "strides")]:
            node.attribute.remove(attr)
        node.attribute.extend(
            [
                helper.make_attribute("kernel_shape", [kh, kw]),
                helper.make_attribute("strides", [kh, kw]),
            ]
        )
        graph.node.extend(nodes)
        count += 1
    if count:
        _topological_sort(graph)
    return count


def _channel_reduce_to_conv(model) -> int:
    """``ReduceSum`` / ``ReduceMean`` over channels (keepdims) becomes a 1x1 conv.

    PP-LiteSeg's and PIDNet's spatial attention reduce their features over
    the channel axis, which the AIPU only reduces over H and W. A 1x1
    convolution to one channel with weights 1 (or 1/C for the mean) computes
    the same sum. Exact up to summation order. Returns the number of rewrites.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    shapes, _, _, weights = _graph_tables(model)
    graph = model.graph
    count = 0
    for node in [n for n in graph.node if n.op_type in ("ReduceSum", "ReduceMean")]:
        shape = shapes.get(node.input[0])
        if _attributes(node).get("keepdims", 1) != 1 or not shape or len(shape) != 4:
            continue
        if _reduce_axes(graph, weights, node) not in ([1], [-3]):
            continue
        channels = shape[1]
        value = 1.0 if node.op_type == "ReduceSum" else 1.0 / channels
        tag = node.name or node.output[0]
        weight = f"{tag}_as_conv_w"
        graph.initializer.append(
            numpy_helper.from_array(
                np.full((1, channels, 1, 1), value, np.float32), weight
            )
        )
        index = list(graph.node).index(node)
        graph.node.remove(node)
        graph.node.insert(
            index,
            helper.make_node(
                "Conv",
                [node.input[0], weight],
                list(node.output),
                name=f"{tag}_as_conv",
                kernel_shape=[1, 1],
            ),
        )
        count += 1
    if count:
        _drop_unused_initializers(graph)
    return count


def _channel_slice(nodes, constants, base, source, start, end, tag) -> str:
    """Append a channel-axis Slice ``source[:, start:end]`` to ``nodes``."""
    import numpy as np
    from onnx import helper

    out = f"{base}_{tag}"
    constants[f"{out}_start"] = np.array([start], np.int64)
    constants[f"{out}_end"] = np.array([end], np.int64)
    nodes.append(
        helper.make_node(
            "Slice",
            [source, f"{out}_start", f"{out}_end", f"{base}_axis"],
            [out],
            name=out,
        )
    )
    return out


def _channel_max_tree(model) -> int:
    """``ReduceMax`` over channels (keepdims) becomes a halving tree of maxima.

    ``max(a, b) = b + relu(a - b)`` on two channel slices, repeated until one
    channel is left (an odd channel joins the next level). Slices, Mul by -1,
    Add and Relu all run on the AIPU. Exact. Returns the number of rewrites.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    shapes, _, _, weights = _graph_tables(model)
    graph = model.graph
    count = 0
    for node in [n for n in graph.node if n.op_type == "ReduceMax"]:
        shape = shapes.get(node.input[0])
        if _attributes(node).get("keepdims", 1) != 1 or not shape or len(shape) != 4:
            continue
        if _reduce_axes(graph, weights, node) not in ([1], [-3]):
            continue
        base = f"{node.name or node.output[0]}_max_tree"
        constants = {
            f"{base}_neg_one": np.array(-1.0, np.float32),
            f"{base}_axis": np.array([1], np.int64),
        }
        nodes = []
        current, channels, level = node.input[0], shape[1], 0
        while channels > 1:
            half = channels // 2
            a = _channel_slice(nodes, constants, base, current, 0, half, f"l{level}a")
            b = _channel_slice(
                nodes, constants, base, current, half, 2 * half, f"l{level}b"
            )
            names = [f"{base}_l{level}_{t}" for t in ("neg_b", "diff", "relu", "max")]
            nodes += [
                helper.make_node(
                    "Mul", [b, f"{base}_neg_one"], [names[0]], name=names[0]
                ),
                helper.make_node("Add", [a, names[0]], [names[1]], name=names[1]),
                helper.make_node("Relu", [names[1]], [names[2]], name=names[2]),
                helper.make_node("Add", [b, names[2]], [names[3]], name=names[3]),
            ]
            merged = names[3]
            if channels % 2:
                odd = _channel_slice(
                    nodes, constants, base, current, 2 * half, channels, f"l{level}odd"
                )
                merged = f"{base}_l{level}_cat"
                nodes.append(
                    helper.make_node(
                        "Concat", [names[3], odd], [merged], name=merged, axis=1
                    )
                )
                half += 1
            current, channels, level = merged, half, level + 1
        nodes.append(
            helper.make_node("Identity", [current], list(node.output), name=base)
        )
        graph.initializer.extend(
            numpy_helper.from_array(v, k) for k, v in constants.items()
        )
        index = list(graph.node).index(node)
        graph.node.remove(node)
        for offset, new in enumerate(nodes):
            graph.node.insert(index + offset, new)
        count += 1
    if count:
        _drop_unused_initializers(graph)
    return count


def _broadcast_mul_expand(model) -> int:
    """A one-channel map multiplied or added to a C-channel map is expanded first.

    The AIPU combines a 4-D map only with a same-shape tensor, a per-channel
    ``[1, C, 1, 1]`` constant or a scalar; an attention map ``[1, 1, H, W]``
    against ``[1, C, H, W]`` is broadcast with a 1x1 convolution of ones to C
    channels. Exact. Returns the number of expanded operands.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    shapes, _, _, _ = _graph_tables(model)
    graph = model.graph
    count = 0
    for node in [n for n in graph.node if n.op_type in ("Mul", "Add")]:
        if len(node.input) != 2:
            continue
        first, second = shapes.get(node.input[0]), shapes.get(node.input[1])
        if not first or not second or len(first) != 4 or len(second) != 4:
            continue
        for slot, (small, full) in enumerate(((first, second), (second, first))):
            if small[1] != 1 or full[1] <= 1 or small[2:] != full[2:]:
                continue
            channels = full[1]
            tag = node.name or node.output[0]
            weight = f"{tag}_expand_{slot}_w"
            graph.initializer.append(
                numpy_helper.from_array(
                    np.ones((channels, 1, 1, 1), np.float32), weight
                )
            )
            out = f"{node.input[slot]}_expand{channels}_{tag}"
            graph.node.insert(
                list(graph.node).index(node),
                helper.make_node(
                    "Conv",
                    [node.input[slot], weight],
                    [out],
                    name=f"{tag}_expand_{slot}",
                    kernel_shape=[1, 1],
                ),
            )
            node.input[slot] = out
            count += 1
            break
    return count


def _resize_1x1_broadcast_add(model) -> int:
    """Upsampling a 1x1 map into an addition becomes a broadcast addition.

    Any resize of a 1x1 map is the same value everywhere, so when each
    consumer adds it to a full-size map the per-channel ``[1, C, 1, 1]`` map
    can be added directly. The compiler would otherwise lower the resize
    (PP-LiteSeg's pyramid pooling, 1x1 to 16x32) to a depthwise window wider
    than the core's scratchpad. Exact. Returns the number of removed resizes.
    """
    shapes, _, consumers, _ = _graph_tables(model)
    graph = model.graph
    outputs = {o.name for o in graph.output}
    count = 0
    for node in [n for n in graph.node if n.op_type == "Resize"]:
        source, target = shapes.get(node.input[0]), shapes.get(node.output[0])
        if not source or not target or len(source) != 4 or source[2:] != [1, 1]:
            continue
        users = consumers.get(node.output[0], [])
        if node.output[0] in outputs or not users:
            continue
        if any(u.op_type != "Add" or len(u.input) != 2 for u in users):
            continue
        partners = [u.input[1 - list(u.input).index(node.output[0])] for u in users]
        if any(shapes.get(other) != target for other in partners):
            continue
        for user, other in zip(users, partners):
            user.input[0], user.input[1] = other, node.input[0]
        graph.node.remove(node)
        count += 1
    if count:
        _drop_unused_initializers(graph)
    return count


#: Largest bilinear factor a resize keeps in one piece: the compiler lowers a
#: factor-F upsampling to a depthwise window about 2F wide, and F = 16 (window
#: 32) already overflows the DWPU scratchpad.
_RESIZE_MAX_FACTOR = 4


def _factor_steps(factor: int) -> list[int] | None:
    """``factor`` as resize steps of 3 and 2 (largest first), or None."""
    steps, rest = [], int(factor)
    for prime in (3, 2):
        while rest % prime == 0 and rest > 1:
            steps.append(prime)
            rest //= prime
    return steps if rest == 1 and steps else None


def _resize_corner_chain(model) -> int:
    """Bilinear upsampling with a large or unequal factor becomes factor-2 steps.

    The compiler takes a bilinear resize only with the same whole factor on
    both axes, and only small factors fit its depthwise unit; PP-LiteSeg's
    pyramid pooling upsamples 2x2 and 4x4 maps to the 16x32 map of its
    512x1024 canvas (factors 8x16 and 4x8). For half-pixel sampling with
    factors ``(f, r * f)``, ``f`` even, ``F = r * f``:

    - the source is padded by one replicated pixel per side (a Concat of
      three distinct tensors per axis);
    - align-corners factor-2 and factor-3 resizes (n -> 2n - 1, n -> 3n - 2;
      ``F`` must be a product of 2s and 3s) build the corner-aligned factor-F
      samples: piecewise-linear interpolation through every source pixel,
      which such steps reproduce exactly, and constant across the replicated
      border;
    - a crop keeps the samples from half a source pixel before the first
      centre to half a pixel after the last;
    - a 2x2 average with stride 1 gives the half-pixel factor-F samples (each
      is the midpoint of two corner samples on one linear piece);
    - an ``r x 1`` average gives the smaller factor on its axis;
    - a 1x1 identity convolution ends the chain (see the comment there).

    Exact up to float rounding. A resize that is a graph output (the final
    upsampling the host runs) is left alone. Returns the number of rewrites.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    shapes, _, _, _ = _graph_tables(model)
    graph = model.graph
    outputs = {o.name for o in graph.output}
    count = 0
    for node in [n for n in graph.node if n.op_type == "Resize"]:
        attrs = _attributes(node)
        mode = attrs.get("mode", b"nearest")
        transform = attrs.get("coordinate_transformation_mode", b"half_pixel")
        mode = mode.decode() if isinstance(mode, bytes) else mode
        transform = transform.decode() if isinstance(transform, bytes) else transform
        if mode != "linear" or transform not in ("half_pixel", "pytorch_half_pixel"):
            continue
        source, target = shapes.get(node.input[0]), shapes.get(node.output[0])
        if node.output[0] in outputs or not source or not target:
            continue
        if len(source) != 4 or len(target) != 4:
            continue
        (height, width), (out_h, out_w) = source[2:], target[2:]
        if min(height, width) < 2 or out_h % height or out_w % width:
            continue
        factor_h, factor_w = out_h // height, out_w // width
        large, small = max(factor_h, factor_w), min(factor_h, factor_w)
        steps = _factor_steps(large)
        if large % small or small % 2 or steps is None:
            continue
        if factor_h == factor_w and large <= _RESIZE_MAX_FACTOR:
            continue
        tag = f"{node.name or node.output[0]}_corner_chain"
        constants, nodes, current = {}, [], node.input[0]
        for axis, size in ((2, height), (3, width)):
            constants[f"{tag}_axis{axis}"] = np.array([axis], np.int64)
            ends = (("first", 0, 1), ("last", size - 1, size))
            names = []
            for side, start, end in ends:
                name = f"{tag}_{side}{axis}"
                constants[f"{name}_start"] = np.array([start], np.int64)
                constants[f"{name}_end"] = np.array([end], np.int64)
                nodes.append(
                    helper.make_node(
                        "Slice",
                        [current, f"{name}_start", f"{name}_end", f"{tag}_axis{axis}"],
                        [name],
                        name=name,
                    )
                )
                names.append(name)
            padded = f"{tag}_pad{axis}"
            nodes.append(
                helper.make_node(
                    "Concat",
                    [names[0], current, names[1]],
                    [padded],
                    name=padded,
                    axis=axis,
                )
            )
            current = padded
        rows, cols = height + 2, width + 2
        for step, factor in enumerate(steps):
            rows, cols = factor * (rows - 1) + 1, factor * (cols - 1) + 1
            sizes = f"{tag}_sizes{step}"
            constants[sizes] = np.array([source[0], source[1], rows, cols], np.int64)
            out = f"{tag}_x{factor}_{step}"
            nodes.append(
                helper.make_node(
                    "Resize",
                    [current, "", "", sizes],
                    [out],
                    name=out,
                    mode="linear",
                    coordinate_transformation_mode="align_corners",
                )
            )
            current = out
        first = large - large // 2
        crop = f"{tag}_crop"
        constants[f"{crop}_start"] = np.array([first, first], np.int64)
        constants[f"{crop}_end"] = np.array(
            [first + height * large + 1, first + width * large + 1], np.int64
        )
        constants[f"{crop}_axes"] = np.array([2, 3], np.int64)
        nodes.append(
            helper.make_node(
                "Slice",
                [current, f"{crop}_start", f"{crop}_end", f"{crop}_axes"],
                [crop],
                name=crop,
            )
        )
        ratio = large // small
        averaged = f"{tag}_average"
        nodes.append(
            helper.make_node(
                "AveragePool",
                [crop],
                [averaged],
                name=f"{tag}_average",
                kernel_shape=[2, 2],
                strides=[1, 1],
            )
        )
        if ratio > 1:
            kernel = [ratio, 1] if factor_h < factor_w else [1, ratio]
            nodes.append(
                helper.make_node(
                    "AveragePool",
                    [averaged],
                    [f"{tag}_ratio"],
                    name=f"{tag}_ratio",
                    kernel_shape=kernel,
                    strides=kernel,
                )
            )
            averaged = f"{tag}_ratio"
        # A 1x1 identity convolution ends the chain: two pooled chains merging
        # into an Add that feeds a convolution (PP-LiteSeg's pyramid pooling)
        # otherwise stop lowering ("Expecting all blocks to have the same
        # iter_values"). Exact.
        channels = int(source[1])
        constants[f"{tag}_identity"] = np.eye(channels, dtype=np.float32).reshape(
            channels, channels, 1, 1
        )
        nodes.append(
            helper.make_node(
                "Conv",
                [averaged, f"{tag}_identity"],
                list(node.output),
                name=f"{tag}_identity",
                kernel_shape=[1, 1],
            )
        )
        graph.initializer.extend(
            numpy_helper.from_array(v, k) for k, v in constants.items()
        )
        index = list(graph.node).index(node)
        graph.node.remove(node)
        for offset, new in enumerate(nodes):
            graph.node.insert(index + offset, new)
        count += 1
    if count:
        _drop_unused_initializers(graph)
    return count


def _stem_residual_skip_conv(model) -> int:
    """A residual ``Add`` at the stem's resolution reads its skip through a 1x1 identity convolution.

    EfficientNetV2-b1/b2/b3 open with two stage-0 FusedMBConv blocks, the second
    adding its input back at the stem's stride-2 resolution. Compiled as is, the
    skip operand's buffer stays listed as a second graph output (pool
    ``pool_ddr_input``), so the compiled model has two output buffers for one
    output. With the skip read through an identity 1x1 convolution (identity
    weights, zero bias: exact) it compiles with one. Adds at other resolutions
    (b0 has no stage-0 residual) are left alone. Returns the number of rewrites.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    shapes, producers, _, weights = _graph_tables(model)
    graph = model.graph
    image = graph.input[0].name
    stem = next((n for n in graph.node if n.op_type == "Conv" and n.input[0] == image), None)
    if stem is None or len(shapes.get(stem.output[0], [])) != 4:
        return 0
    stem_hw = shapes[stem.output[0]][2:]
    count = 0
    for node in [n for n in graph.node if n.op_type == "Add"]:
        if len(node.input) != 2 or any(name in weights for name in node.input):
            continue
        a, b = (shapes.get(name, []) for name in node.input)
        if len(a) != 4 or a != b or a[2:] != stem_hw:
            continue
        channels = int(b[1])
        weight = np.eye(channels, dtype=np.float32).reshape(channels, channels, 1, 1)
        w_name, b_name = f"stem_skip_identity_w{count}", f"stem_skip_identity_b{count}"
        graph.initializer.append(numpy_helper.from_array(weight, w_name))
        graph.initializer.append(numpy_helper.from_array(np.zeros(channels, np.float32), b_name))
        out = f"{node.input[1]}_stem_skip_identity{count}"
        conv = helper.make_node(
            "Conv", [node.input[1], w_name, b_name], [out],
            name=f"stem_skip_identity_conv{count}", kernel_shape=[1, 1],
        )
        graph.node.insert(list(graph.node).index(node), conv)
        node.input[1] = out
        count += 1
    return count


_DATA_PASSES = {"dw_channel_groups"}

#: Named exact ONNX rewrites an export may request (``graph_passes``); each
#: rewrites a ModelProto in place and returns how many sites it changed.
GRAPH_PASSES = {
    "channel_shuffle_to_conv": _channel_shuffle_to_conv,
    "dw_channel_groups": _dw_channel_groups,
    "pooled_branch_split": _pooled_branch_split,
    "conv_reduction_split": _conv_reduction_split,
    "fc_rowconv": _fc_rowconv,
    "float_div_to_mul": _float_div_to_mul,
    "stack_sum_to_add": _stack_sum_to_add,
    "mean_hw_to_gap": _mean_hw_to_gap,
    "gap_staged": _gap_staged,
    "channel_reduce_to_conv": _channel_reduce_to_conv,
    "channel_max_tree": _channel_max_tree,
    "broadcast_mul_expand": _broadcast_mul_expand,
    "resize_1x1_broadcast_add": _resize_1x1_broadcast_add,
    "resize_corner_chain": _resize_corner_chain,
    "avgpool_staged": _avgpool_staged,
    "stem_residual_skip_conv": _stem_residual_skip_conv,
}


def apply_graph_passes(
    onnx_path: str, work_dir: Path, passes: Iterable[str], samples=None
) -> str:
    """Run the named :data:`GRAPH_PASSES` on ``onnx_path``.

    Returns ``onnx_path`` untouched when no pass changes anything; otherwise
    writes the rewritten graph to ``work_dir`` and returns its path.
    """
    passes = list(passes)
    if not passes:
        return onnx_path
    unknown = sorted(set(passes) - set(GRAPH_PASSES))
    if unknown:
        raise ValueError(f"Unknown Axelera graph pass(es): {', '.join(unknown)}.")
    import onnx

    model = onnx.load(onnx_path)
    changed = {}
    for name in passes:
        rewrite = GRAPH_PASSES[name]
        changed[name] = (
            rewrite(model, samples=samples) if name in _DATA_PASSES else rewrite(model)
        )
    if not any(changed.values()):
        return onnx_path
    out = Path(work_dir) / f"{Path(onnx_path).stem}.graph_passes.onnx"
    onnx.save(model, str(out))
    logger.info("Axelera graph passes: %s", changed)
    return str(out)


def compile_to_axm(
    model_input, calibration: Iterable, config, work_dir: Path, before_compile=None
):
    """Quantize and compile; return ``(axm_path, quantized_model)``.

    ``work_dir`` must be a fresh directory LibreYOLO owns: the compiler clears
    its ``output_dir`` (``work_dir / "build"``) and writes ``<model_name>.axm``
    next to it, in ``work_dir``. ``before_compile(quantized)`` runs between the
    two steps: compiling lowers the quantized model in place, so anything that
    evaluates it on CPU must happen there.
    """
    from axelera import compiler

    work_dir = Path(work_dir)
    build_dir = work_dir / "build"
    config.output_dir = build_dir
    quantized = compiler.quantize(
        model=model_input, config=config, calibration_dataset=iter(calibration)
    )
    if before_compile is not None:
        before_compile(quantized)
    try:
        compiler.compile(model=quantized, config=config, output_dir=build_dir)
    except Exception as exc:
        artifacts = sorted(str(path) for path in work_dir.rglob("*.zip"))
        detail = f" Error artifact: {', '.join(artifacts)}." if artifacts else ""
        raise RuntimeError(
            f"Axelera compilation failed: {exc}.{detail} "
            f"See {DOCS_TROUBLESHOOTING}."
        ) from exc
    expected = work_dir / f"{config.model_name}.axm"
    if expected.is_file():
        return expected, quantized
    found = sorted(work_dir.rglob("*.axm"))
    if len(found) == 1:
        return found[0], quantized
    raise RuntimeError(
        f"Axelera compilation finished but no single .axm was found under "
        f"{work_dir} (found {len(found)}). See {DOCS_TROUBLESHOOTING}."
    )


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, default_flow_style=False, sort_keys=False)


def _final_config_toml(config, build_dir: Path) -> str:
    """The resolved config: the compiler's own copy if it wrote one."""
    for candidate in sorted(Path(build_dir).rglob("compiler_config_final.toml")):
        return candidate.read_text(encoding="utf-8")
    return config.to_toml()


def _device_runner(axm: Path, output_names: list[str], reference_names: list[str]):
    """Callable running ``axm`` on a Metis device, outputs in graph order; or a reason string."""
    from ..backends.axelera import require_axelera_rt

    try:
        require_axelera_rt()
        from axelera.runtime import op

        model = op.load(str(axm))
    except Exception as exc:  # noqa: BLE001 - no runtime, no device or device busy: report, do not fail
        return f"not run: {exc}"
    order = [output_names.index(name) for name in reference_names]

    def run(sample):
        outputs = model(sample)
        outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        return [outputs[i] for i in order]

    return run


def collect_quantized_outputs(quantized, reference_model, samples) -> list[tuple]:
    """``(sample, fp32_outputs, quantized_outputs)`` per image, both in graph order.

    Call before ``compile()``: compiling lowers the quantized model in place.
    """
    import numpy as np
    import torch

    collected = []
    for sample in samples:
        sample = np.ascontiguousarray(sample, dtype=np.float32)
        tensor = torch.from_numpy(sample)
        with torch.no_grad():
            expected = [t.detach().cpu().numpy() for t in reference_model(tensor)]
            produced = quantized(tensor)
        if isinstance(produced, torch.Tensor):
            produced = (produced,)
        produced = [
            np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t)
            for t in produced
        ]
        # The quantized model returns compiled order; compare in graph order.
        collected.append((sample, expected, _match_outputs(expected, produced)))
    return collected


def parity_report(collected, names, device=None) -> dict[str, Any]:
    """Per-output cosine / NRMSE: quantized vs FP32, and the device vs both.

    ``device`` is a callable running the compiled model (outputs in graph
    order), or a string saying why it was not run.
    """
    import numpy as np

    from .rknn import compare_rknn_outputs

    pairs: dict[str, dict[int, list[dict[str, Any]]]] = {
        "quantized_vs_fp32": {},
        "device_vs_quantized": {},
        "device_vs_fp32": {},
    }

    constant: set[int] = set()

    def add(key, reference, candidate):
        # A constant reference (e.g. a windowed class map with every logit
        # below the window) has no direction: one quantization step of noise
        # would read as cosine 0. Such outputs are listed, not scored.
        keep = [i for i, ref in enumerate(reference) if np.ptp(np.asarray(ref)) > 0]
        constant.update(set(range(len(reference))) - set(keep))
        items = compare_rknn_outputs(
            [reference[i] for i in keep], [candidate[i] for i in keep], raise_on_failure=False
        )
        for position, item in enumerate(items):
            pairs[key].setdefault(keep[position], []).append(item)

    for sample, expected, quantized in collected:
        add("quantized_vs_fp32", expected, quantized)
        if callable(device):
            try:
                on_device = [np.asarray(t, dtype=np.float32) for t in device(sample)]
            except Exception as exc:  # noqa: BLE001 - the runtime connects on first call; a busy device must not discard the compiled model
                device = f"not run: {exc}"
                continue
            add("device_vs_quantized", quantized, on_device)
            add("device_vs_fp32", expected, on_device)

    def summarize(per_output):
        rows = []
        for index in sorted(per_output):
            items = per_output[index]
            rows.append(
                {
                    "name": names[index] if index < len(names) else str(index),
                    "min_cosine": min(i["cosine_similarity"] for i in items),
                    "mean_cosine": float(np.mean([i["cosine_similarity"] for i in items])),
                    "max_normalized_rmse": max(i["normalized_rmse"] for i in items),
                }
            )
        return rows

    report: dict[str, Any] = {
        "images": len(collected),
        "quantized_vs_fp32": summarize(pairs["quantized_vs_fp32"]),
    }
    never_scored = sorted(constant - set(pairs["quantized_vs_fp32"]))
    if never_scored:
        report["constant_reference"] = [
            names[i] if i < len(names) else str(i) for i in never_scored
        ]
    if callable(device):
        report["device_vs_quantized"] = summarize(pairs["device_vs_quantized"])
        report["device_vs_fp32"] = summarize(pairs["device_vs_fp32"])
    elif device is not None:
        report["device_vs_quantized"] = str(device)
    return report


def _match_outputs(expected, produced):
    """Reorder ``produced`` to ``expected``'s order by shape, cosine on ties."""
    free = set(range(len(produced)))
    ordered = []
    for reference in expected:
        candidates = [i for i in sorted(free) if produced[i].shape == reference.shape]
        if not candidates:
            raise RuntimeError(
                f"No quantized output matches graph output shape {tuple(reference.shape)}."
            )
        best = max(candidates, key=lambda i, ref=reference: _cosine(ref, produced[i]))
        free.discard(best)
        ordered.append(produced[best])
    return ordered


def device_output_order(quantized, reference_model, sample) -> list[str]:
    """Name the compiled model's outputs in the order it returns them.

    The compiler may reorder graph outputs (for YOLO-NAS it returns each
    level's class tensor before its box tensor). The quantized model returns
    outputs in the compiled order; each is matched to a graph output by shape,
    and by cosine similarity with the FP32 graph when shapes repeat.
    """
    import numpy as np
    import torch

    names = list(reference_model.output_names)
    tensor = torch.from_numpy(np.ascontiguousarray(sample, dtype=np.float32))
    with torch.no_grad():
        reference = [t.detach().cpu().numpy() for t in reference_model(tensor)]
        produced = quantized(tensor)
    if isinstance(produced, torch.Tensor):
        produced = (produced,)
    produced = [
        np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t)
        for t in produced
    ]
    if len(produced) != len(reference):
        raise RuntimeError(
            f"The compiled model has {len(produced)} outputs; the exported graph "
            f"has {len(reference)}. See {DOCS_TROUBLESHOOTING}."
        )
    order: list[str] = []
    free = set(range(len(reference)))
    for output in produced:
        candidates = [i for i in sorted(free) if reference[i].shape == output.shape]
        if not candidates:
            raise RuntimeError(
                f"A compiled output of shape {tuple(output.shape)} matches no "
                f"graph output ({', '.join(names)}). See {DOCS_TROUBLESHOOTING}."
            )
        if len(candidates) > 1:
            candidates.sort(key=lambda i, out=output: _cosine(reference[i], out), reverse=True)
        free.discard(candidates[0])
        order.append(names[candidates[0]])
    return order


def _cosine(a, b) -> float:
    import numpy as np

    a, b = np.ravel(a), np.ravel(b)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b)) / norm


def _remember_first(samples: Iterable, keep: list) -> Iterable:
    for sample in samples:
        if not keep:
            keep.append(sample)
        yield sample


def export_axelera(
    *,
    onnx_path: str,
    output_path: str,
    stem: str,
    metadata: Mapping[str, Any],
    calibration: Iterable,
    calibration_info: Mapping[str, Any],
    compiler_overrides: Mapping[str, Any] | None = None,
    devkit_version: str | None = None,
    quantization_scheme: str = DEFAULT_QUANTIZATION_SCHEME,
    resources_used: float = DEFAULT_RESOURCES_USED,
    verify: bool = False,
    reference_model=None,
    verify_samples: Iterable | None = None,
    verify_source: str = "images",
    graph_passes: Iterable[str] = (),
) -> str:
    """Compile ``onnx_path`` and publish ``<stem>.axm`` + sidecars at ``output_path``."""
    import json

    check_compile_platform()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    try:
        with tempfile.TemporaryDirectory(prefix="libreyolo-axelera-") as work:
            work_dir = Path(work)
            config, overrides = build_compiler_config(
                stem,
                build_dir=work_dir / "build",
                quantization_scheme=quantization_scheme,
                resources_used=resources_used,
                overrides=compiler_overrides,
            )
            first: list = []
            found: dict[str, Any] = {
                "output_names": list(metadata.get("output_names") or []),
                "collected": [],
            }

            def before_compile(quantized):
                if reference_model is None:
                    return
                if first:
                    found["output_names"] = device_output_order(
                        quantized, reference_model, first[0]
                    )
                if verify:
                    found["collected"] = collect_quantized_outputs(
                        quantized, reference_model, verify_samples or []
                    )

            model_input = merge_pad_into_maxpool(onnx_path, work_dir)
            graph_passes = list(graph_passes)
            samples = None
            if any(name in _DATA_PASSES for name in graph_passes):
                calibration = list(calibration)  # also feeds data-based passes
                samples = calibration[:16]
            model_input = apply_graph_passes(
                model_input, work_dir, graph_passes, samples=samples
            )
            axm, _ = compile_to_axm(
                model_input,
                _remember_first(calibration, first),
                config,
                work_dir,
                before_compile=before_compile,
            )
            output_names = found["output_names"]
            shutil.copy2(axm, staging / f"{stem}.axm")
            (staging / "compiler_config_final.toml").write_text(
                _final_config_toml(config, work_dir / "build"), encoding="utf-8"
            )
            if verify and reference_model is not None:
                device = _device_runner(
                    staging / f"{stem}.axm", output_names, list(reference_model.output_names)
                )
                report = parity_report(
                    found["collected"], list(reference_model.output_names), device=device
                )
                report["source"] = verify_source
                (staging / f"{stem}.parity.json").write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )
                worst = min(
                    (o["min_cosine"] for o in report["quantized_vs_fp32"]),
                    default=float("nan"),
                )
                logger.info(
                    "Axelera verify on %d %s: lowest per-output cosine %.4f "
                    "(quantized vs FP32); report %s.parity.json.",
                    report["images"],
                    verify_source,
                    worst,
                    stem,
                )
                if isinstance(report.get("device_vs_quantized"), str):
                    logger.warning(
                        "Axelera verify: device comparison %s. The compiled model "
                        "is kept; run val on the device to check it.",
                        report["device_vs_quantized"],
                    )
                if isinstance(report.get("device_vs_quantized"), list):
                    logger.info(
                        "Axelera verify: device vs quantized model lowest cosine %.4f; "
                        "device vs FP32 lowest cosine %.4f.",
                        min((o["min_cosine"] for o in report["device_vs_quantized"]), default=float("nan")),
                        min((o["min_cosine"] for o in report["device_vs_fp32"]), default=float("nan")),
                    )
                if worst < 0.99:
                    logger.warning(
                        "Axelera verify: quantized outputs drift from FP32 (cosine "
                        "%.4f). Check calibration data and run val on the device.",
                        worst,
                    )
        scheme = getattr(config, "quantization_scheme", quantization_scheme)
        meta = dict(metadata)
        meta["axm"] = f"{stem}.axm"
        meta["output_names"] = output_names
        meta["axelera"] = {
            "target": "metis",
            "devkit_version": devkit_version,
            "quantization_scheme": getattr(scheme, "value", scheme),
            "aipu_cores_used": int(config.aipu_cores_used),
            "resources_used": float(config.resources_used),
            "compiler_overrides": {k: _plain(v) for k, v in dict(overrides).items()},
            "calibration": dict(calibration_info),
        }
        _write_yaml(staging / "metadata.yaml", meta)
        publish_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return str(destination)


class OnnxReference:
    """An FP32 ONNX graph run by ONNX Runtime, called like a torch reference.

    Multi-part exports cut from one ONNX graph use it to calibrate and name
    each part: called with one torch tensor, it returns the graph's outputs as
    torch tensors, in the order of ``output_names``.
    """

    def __init__(self, onnx_path: str | Path):
        import onnxruntime

        self.path = str(onnx_path)
        self.session = onnxruntime.InferenceSession(self.path, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, feeds: Mapping[str, Any]) -> list:
        return self.session.run(self.output_names, dict(feeds))

    def __call__(self, tensor):
        import numpy as np
        import torch

        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
        return tuple(torch.from_numpy(out) for out in self.run({self.input_names[0]: array}))


def _part_names(spec: Mapping[str, Any], single: str, plural: str) -> list[str]:
    if plural in spec:
        return [str(name) for name in spec[plural]]
    return [str(spec[single])]


def _host_value(value):
    """A host-operation parameter as YAML: names as strings, integers as ints, other numbers as floats.

    Lists may nest (e.g. one ``[y, x, h, w]`` box per pyramid level).
    """
    import numpy as np

    if isinstance(value, (list, tuple)):
        return [_host_value(v) for v in value]
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return _plain(value)


def _run_fp32_chain(parts: list[Mapping[str, Any]], sample, host_ops, host_arrays, onnx_refs, keep, stop=None):
    """Run one image through the FP32 chain; ``keep(part_index, array)`` gets each compiled part's input.

    With ``stop`` the chain ends once part ``stop`` has its input (nothing after it runs).
    """
    import numpy as np
    import torch

    env: dict[str, Any] = {"images": np.ascontiguousarray(sample, dtype=np.float32)}
    for index, spec in enumerate(parts):
        if "compile" in spec:
            source = env[spec["input"]]
            keep(index, source)
            if index == stop:
                return env
            reference = spec["reference"]
            with torch.no_grad():
                outputs = reference(torch.from_numpy(np.ascontiguousarray(source, dtype=np.float32)))
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            env.update(
                zip(reference.output_names, (o.detach().cpu().numpy() for o in outputs))
            )
        elif "host" in spec:
            names = _part_names(spec, "output", "outputs")
            params = {
                key: host_arrays[value[5:]] if isinstance(value, str) and value.startswith("@npz:") else value
                for key, value in spec.items()
                if key not in ("host", "input", "output", "inputs", "outputs")
            }
            result = host_ops[spec["host"]](
                *(env[name] for name in _part_names(spec, "input", "inputs")), **params
            )
            env.update(zip(names, result if len(names) > 1 else (result,)))
        else:
            reference = onnx_refs[index]
            feeds = {name: np.ascontiguousarray(env[name]) for name in spec["inputs"]}
            env.update(zip(reference.output_names, reference.run(feeds)))
    return env


def export_axelera_parts(
    *,
    parts: list[Mapping[str, Any]],
    output_path: str,
    stem: str,
    metadata: Mapping[str, Any],
    calibration: Iterable,
    calibration_info: Mapping[str, Any],
    compiler_overrides: Mapping[str, Any] | None = None,
    devkit_version: str | None = None,
    quantization_scheme: str = DEFAULT_QUANTIZATION_SCHEME,
    resources_used: float = DEFAULT_RESOURCES_USED,
    verify: bool = False,
    verify_samples: Iterable | None = None,
    host_arrays: Mapping[str, Any] | None = None,
    item_limit: int = 8,
) -> str:
    """Compile a multi-part export and publish its ``.axm`` parts, host graphs and sidecars.

    ``parts`` lists the steps in run order, each one of:

    * ``{"compile": onnx_path, "reference": model, "input": name}``: a part
      compiled for the Metis AIPU. ``reference`` is the FP32 graph (a torch
      module or :class:`OnnxReference` with ``output_names``); the first part
      reads ``"images"``. ``"per_item": True`` makes the part take a stack of
      items (one at a time on the device); ``"graph_passes"`` lists graph
      passes for it, and ``"merge_pad"`` (default: first part only) merges
      Pad into MaxPool.
    * ``{"host": name, "input"/"inputs": ..., "output"/"outputs": ..., **params}``:
      a numpy operation from the backend's ``HOST_OPS``. A parameter
      ``"@npz:<key>"`` is ``host_arrays[<key>]``, saved next to the parts.
    * ``{"host_onnx": onnx_path, "inputs": [...], "outputs": [...]}``: an FP32
      ONNX graph the backend runs on the host CPU with ONNX Runtime.

    Every compiled part is calibrated on what it will see: the calibration
    images run through the FP32 chain (compiled parts as their references,
    host steps as the backend runs them), and each part takes its own input.
    Per-item parts take up to ``item_limit`` items per image. ``verify``
    compares each compiled part's quantized model with its FP32 reference on
    held-out images (``verify_samples``) carried through the FP32 chain the
    same way, plus the device when one is present.
    Metadata gains ``parts`` (and ``host_arrays`` when parameters use them);
    the first compiled part is ``<stem>.axm`` and the k-th ``<stem>_part<k>.axm``.
    """
    import json

    import numpy as np

    from ..backends.axelera import HOST_OPS

    check_compile_platform()
    if not parts or "compile" not in parts[0] or parts[0].get("input") != "images":
        raise ValueError("The first part of a multi-part Axelera export must be compiled and read the images.")
    host_arrays = dict(host_arrays or {})
    compiled = [i for i, spec in enumerate(parts) if "compile" in spec]
    stems = {index: stem if k == 0 else f"{stem}_part{k + 1}" for k, index in enumerate(compiled)}
    onnx_refs = {i: OnnxReference(spec["host_onnx"]) for i, spec in enumerate(parts) if "host_onnx" in spec}

    def part_inputs(images, part):
        """What compiled part ``part`` sees for each image: the FP32 chain run up to it.

        Computed for one part at a time, right before it compiles, so only that
        part's inputs are held in memory (a packed 512-channel canvas is tens of
        MB per image; holding every part's at once does not fit).
        """
        inputs: list = []

        def keep(index, array):
            if index != part:
                return
            if parts[index].get("per_item"):
                inputs.extend(array[i : i + 1] for i in range(min(len(array), item_limit)))
            else:
                inputs.append(array)

        for image in images:
            _run_fp32_chain(parts, image, HOST_OPS, host_arrays, onnx_refs, keep, stop=part)
        return inputs

    samples = list(calibration)
    if len(samples) < 2:
        raise ValueError("A multi-part Axelera export needs at least 2 calibration images.")
    held_out = list(verify_samples or []) if verify else []
    if verify and not held_out:
        logger.warning("Axelera verify: no held-out images for the multi-part export; run val on the device.")
    first_inputs: dict[int, Any] = {}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    try:
        with tempfile.TemporaryDirectory(prefix="libreyolo-axelera-") as work:
            work_dir = Path(work)
            output_names: dict[int, list[str]] = {}
            collected: dict[int, list] = {}
            configs = []
            overrides: Mapping[str, Any] = {}
            reports = []
            for k, index in enumerate(compiled):
                spec = parts[index]
                part_dir = work_dir / f"part{k + 1}"
                part_dir.mkdir()
                config, part_overrides = build_compiler_config(
                    stems[index],
                    build_dir=part_dir / "build",
                    quantization_scheme=spec.get("quantization_scheme", quantization_scheme),
                    resources_used=resources_used,
                    overrides=compiler_overrides,
                )
                if k == 0:
                    overrides = part_overrides
                configs.append(config)
                reference = spec["reference"]
                calibration_inputs = part_inputs(samples, index)
                if len(calibration_inputs) < 2:
                    raise ValueError(
                        f"Part {index} of the multi-part export got {len(calibration_inputs)} "
                        "calibration inputs from the calibration images; it needs at least 2."
                    )
                first_inputs[index] = calibration_inputs[0]
                verify_inputs = part_inputs(held_out, index) if held_out else []

                def before_compile(quantized, index=index, reference=reference, verify_inputs=verify_inputs):
                    output_names[index] = device_output_order(quantized, reference, first_inputs[index])
                    if verify_inputs:
                        collected[index] = collect_quantized_outputs(quantized, reference, verify_inputs)

                model_input = str(spec["compile"])
                if spec.get("merge_pad", k == 0):
                    model_input = merge_pad_into_maxpool(model_input, part_dir)
                passes = list(spec.get("graph_passes") or ())
                if passes:
                    model_input = apply_graph_passes(
                        model_input, part_dir, passes, samples=calibration_inputs[:16]
                    )
                axm, _ = compile_to_axm(
                    model_input, calibration_inputs, config, part_dir, before_compile=before_compile
                )
                del calibration_inputs, verify_inputs
                shutil.copy2(axm, staging / f"{stems[index]}.axm")
                suffix = "" if k == 0 else f".part{k + 1}"
                (staging / f"compiler_config_final{suffix}.toml").write_text(
                    _final_config_toml(config, part_dir / "build"), encoding="utf-8"
                )
                if index in collected:
                    device = _device_runner(
                        staging / f"{stems[index]}.axm", output_names[index], list(reference.output_names)
                    )
                    report = parity_report(collected[index], list(reference.output_names), device=device)
                    reports.append({"axm": f"{stems[index]}.axm", **report})
        meta_parts = []
        host_graphs = 0
        for index, spec in enumerate(parts):
            if "compile" in spec:
                entry = {"axm": f"{stems[index]}.axm", "input": spec["input"], "outputs": output_names[index]}
                if spec.get("per_item"):
                    entry["per_item"] = True
                    entry["item_shapes"] = {
                        name: [int(d) for d in np.asarray(out).shape[1:]]
                        for name, out in zip(
                            spec["reference"].output_names,
                            _reference_outputs(spec["reference"], first_inputs[index]),
                        )
                        if name in output_names[index]
                    }
                meta_parts.append(entry)
            elif "host" in spec:
                meta_parts.append({key: _host_value(value) for key, value in spec.items()})
            else:
                host_graphs += 1
                name = f"{stem}_host{host_graphs}.onnx"
                shutil.copy2(spec["host_onnx"], staging / name)
                meta_parts.append(
                    {"onnx": name, "inputs": list(spec["inputs"]), "outputs": list(spec["outputs"])}
                )
        if reports:
            (staging / f"{stem}.parity.json").write_text(
                json.dumps({"parts": reports}, indent=2) + "\n", encoding="utf-8"
            )
            worst = min(
                (o["min_cosine"] for r in reports for o in r["quantized_vs_fp32"]), default=float("nan")
            )
            logger.info(
                "Axelera verify (%d compiled parts, each on its FP32 input): lowest per-output "
                "cosine %.4f (quantized vs FP32); report %s.parity.json.",
                len(reports), worst, stem,
            )
            if worst < 0.99:
                logger.warning(
                    "Axelera verify: quantized outputs drift from FP32 (cosine %.4f). "
                    "Check calibration data and run val on the device.",
                    worst,
                )
        scheme = getattr(configs[0], "quantization_scheme", quantization_scheme)
        meta = dict(metadata)
        meta["axm"] = f"{stem}.axm"
        meta["parts"] = meta_parts
        if host_arrays:
            np.savez(staging / f"{stem}_host.npz", **{k: np.asarray(v) for k, v in host_arrays.items()})
            meta["host_arrays"] = f"{stem}_host.npz"
        meta["axelera"] = {
            "target": "metis",
            "devkit_version": devkit_version,
            "quantization_scheme": getattr(scheme, "value", scheme),
            "aipu_cores_used": int(configs[0].aipu_cores_used),
            "resources_used": float(configs[0].resources_used),
            "compiler_overrides": {k: _plain(v) for k, v in dict(overrides).items()},
            "calibration": dict(calibration_info),
        }
        _write_yaml(staging / "metadata.yaml", meta)
        publish_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return str(destination)


def _reference_outputs(reference, sample) -> list:
    import numpy as np
    import torch

    with torch.no_grad():
        outputs = reference(torch.from_numpy(np.ascontiguousarray(sample, dtype=np.float32)))
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    return [o.detach().cpu().numpy() for o in outputs]


def export_axelera_chain(
    *,
    onnx_paths: list[str],
    output_path: str,
    stem: str,
    metadata: Mapping[str, Any],
    calibration: Iterable,
    calibration_info: Mapping[str, Any],
    references: list,
    host_step: Mapping[str, Any],
    compiler_overrides: Mapping[str, Any] | None = None,
    devkit_version: str | None = None,
    quantization_scheme: str = DEFAULT_QUANTIZATION_SCHEME,
    resources_used: float = DEFAULT_RESOURCES_USED,
    verify: bool = False,
    verify_samples: Iterable | None = None,
) -> str:
    """Two compiled parts with one host operation between them (SSD300's form).

    A thin form of :func:`export_axelera_parts`: part 1 (``onnx_paths[0]``)
    takes the images, ``host_step`` turns one part-1 output into part 2's
    input. The metadata it writes is the two-part form SSD300 has always had.
    """
    head_ref, tail_ref = references
    return export_axelera_parts(
        parts=[
            {"compile": onnx_paths[0], "reference": head_ref, "input": "images"},
            dict(host_step),
            {"compile": onnx_paths[1], "reference": tail_ref, "input": host_step["output"], "merge_pad": False},
        ],
        output_path=output_path,
        stem=stem,
        metadata=metadata,
        calibration=calibration,
        calibration_info=calibration_info,
        compiler_overrides=compiler_overrides,
        devkit_version=devkit_version,
        quantization_scheme=quantization_scheme,
        resources_used=resources_used,
        verify=verify,
        verify_samples=verify_samples,
    )


def _plain(value):
    value = getattr(value, "value", value)
    if isinstance(value, Path):
        return str(value)
    return value


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def publish_directory(staging: Path, destination: Path) -> Path:
    """Move a finished artifact directory into place, replacing any old one.

    The previous directory is only removed after the new one is in place; on
    failure it is restored.
    """
    staging, destination = Path(staging), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if destination.exists():
        backup = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.old.", dir=destination.parent)
        )
        backup.rmdir()
        destination.replace(backup)
    try:
        staging.replace(destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    # The staging directory came from mkdtemp (mode 0700); give the artifact
    # the permissions any other new directory would get.
    destination.chmod(0o777 & ~_current_umask())
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
    return destination


__all__ = [
    "AUTOINSTALL_ENV",
    "AXELERA_INDEX_URL",
    "AXELERA_SDK_VERSION",
    "DEFAULT_QUANTIZATION_SCHEME",
    "DEFAULT_QUANTIZATION_SCHEME_BY_MODEL",
    "DEFAULT_QUANTIZATION_SCHEME_BY_TASK",
    "DEPTHWISE_QUANTIZATION_SCHEME",
    "DEFAULT_RESOURCES_USED",
    "axelera_install_command",
    "build_compiler_config",
    "check_compile_platform",
    "collect_quantized_outputs",
    "compile_to_axm",
    "default_quantization_scheme",
    "device_output_order",
    "OnnxReference",
    "export_axelera",
    "export_axelera_chain",
    "export_axelera_parts",
    "normalize_compiler_overrides",
    "parity_report",
    "publish_directory",
    "require_axelera_devkit",
    "require_axelera_rt",
]

"""Canonical export support tiers for model, task, and format combinations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from ..tasks import TASKS

#: ``validated``: parity-checked; ``available``: callable with the recorded
#: context; ``hybrid``: exports and runs, but only part of the network runs on
#: the target accelerator and the rest (named in the reason) on the host CPU;
#: ``blocked``: refused in preflight.
Tier = Literal["validated", "available", "hybrid", "blocked"]
EXPORT_FORMATS = (
    "onnx",
    "torchscript",
    "executorch",
    "tensorrt",
    "openvino",
    "paddle",
    "mnn",
    "rknn",
    "axelera",
    "ncnn",
    "tflite",
    "coreml",
    "coreai",
)


@dataclass(frozen=True)
class SupportEntry:
    """Support status and user-facing context for one export combination."""

    tier: Tier
    reason: str = ""
    #: The major.minor release LINE in which the capability first ships, not
    #: an exact tag: "1.5" covers 1.5.0 and its point releases, so rows that
    #: land between the .0 tag and the next point release stay truthful.
    since: str | None = None
    constraint: str | None = None


SUPPORT: dict[tuple[str, str, str], SupportEntry] = {}


def _add(
    tier: Tier,
    families: tuple[str, ...],
    tasks: tuple[str, ...],
    formats: tuple[str, ...],
    *,
    reason: str = "",
    since: str | None = None,
    constraint: str | None = None,
) -> None:
    entry = SupportEntry(tier, reason, since, constraint)
    keys = [
        (family, task, fmt) for family in families for task in tasks for fmt in formats
    ]
    seen: set[tuple[str, str, str]] = set()
    duplicates = []
    for key in keys:
        if key in SUPPORT or key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        rendered = ", ".join(repr(key) for key in duplicates)
        raise ValueError(f"Duplicate export support entries: {rendered}")
    for key in keys:
        SUPPORT[key] = entry


_add("available", ("gtr",), ("detect", "pose"), ("onnx", "torchscript"),
     reason="Portable gated recurrence with fixed input resolution.",
     constraint="FP32; static square input; no custom CUDA plugin")

_add("blocked", ("gtr",), ("detect", "pose"),
     tuple(fmt for fmt in EXPORT_FORMATS if fmt not in ("onnx", "torchscript")),
     reason="GTR currently implements only portable ONNX and TorchScript export.")

_add("available", ("gtr",), ("obb",), ("onnx", "torchscript"),
     reason="Portable gated recurrence with fixed 1024px input; NMS-free rotated boxes.",
     constraint="FP32; static square input; no custom CUDA plugin")

_add("blocked", ("gtr",), ("obb",),
     tuple(fmt for fmt in EXPORT_FORMATS if fmt not in ("onnx", "torchscript")),
     reason="GTR currently implements only portable ONNX and TorchScript export.")

_add("available", ("gtr",), ("depth",), ("onnx", "torchscript"),
     reason="Portable gated recurrence emitting a fixed-canvas inverse-depth map.",
     constraint="FP32; static square input; batch 1; no custom CUDA plugin")

_add("blocked", ("gtr",), ("depth",),
     tuple(fmt for fmt in EXPORT_FORMATS if fmt not in ("onnx", "torchscript")),
     reason="GTR currently implements only portable ONNX and TorchScript export.")

_add("available", ("gtr",), ("semantic",), ("onnx", "torchscript"),
     reason="Portable gated recurrence; the graph slides 1024px windows over a fixed canvas.",
     constraint="FP32; static 1024x2048 canvas; images are letterboxed into it")

_add("blocked", ("gtr",), ("semantic",),
     tuple(fmt for fmt in EXPORT_FORMATS if fmt not in ("onnx", "torchscript")),
     reason="GTR currently implements only portable ONNX and TorchScript export.")

_add("available", ("gtr",), ("segment",), ("onnx", "torchscript"),
     reason="Portable gated recurrence with fixed input resolution; raw mask logits.",
     constraint="FP32; static square input; no custom CUDA plugin")

_add("blocked", ("gtr",), ("segment",),
     tuple(fmt for fmt in EXPORT_FORMATS if fmt not in ("onnx", "torchscript")),
     reason="GTR currently implements only portable ONNX and TorchScript export.")

# Existing parity-backed paths. New validated rows must land with a parity test.
_add(
    "validated",
    ("convnextv2",),
    ("classify",),
    ("onnx", "torchscript"),
    reason=(
        "Native and exported logits, probabilities, labels, and preprocessing "
        "are covered by test_convnextv2_export.py."
    ),
    since="1.5",
    constraint="FP32; 224px ImageNet-1K classifiers; Atto CPU runtime parity",
)
_add(
    "validated",
    ("yolo9",),
    ("detect",),
    ("onnx", "torchscript", "tflite"),
    since="1.3",
)
_add(
    "validated",
    ("yolo9",),
    ("detect",),
    ("paddle",),
    reason=(
        "The trained LibreYOLO9t checkpoint is covered by raw-output, "
        "metadata, factory reload, and public detection parity in "
        "tests/e2e/test_paddle.py."
    ),
    since="1.5",
    constraint=(
        "X2Paddle 1.6.0, PaddlePaddle 2.6.2 CPU on Python 3.11 or earlier, ONNX 1.17/opset 15, FP32, "
        "batch 1, fixed square input; WSL2 Ubuntu 22.04"
    ),
)
_add(
    "blocked",
    ("rfdetr",),
    ("detect", "segment", "pose", "obb"),
    ("paddle",),
    reason=(
        "RF-DETR requires ONNX opset 17 and GridSample, while X2Paddle 1.6.0 "
        "accepts opset 15 or lower and has no GridSample mapper."
    ),
)
_add(
    "validated",
    ("yolo9_e2e", "ec", "rtdetrv4", "dfine", "deim", "deimv2"),
    ("detect",),
    ("paddle",),
    reason=(
        "Representative trained checkpoints have Paddle conversion, CPU "
        "runtime reload, raw-output parity, and matched public detection "
        "parity in tests/e2e/test_paddle.py."
    ),
    since="1.5",
    constraint=(
        "X2Paddle 1.6.0, PaddlePaddle 2.6.2 CPU on Python 3.11 or earlier, ONNX 1.17/opset 15, FP32, "
        "batch 1, fixed square input; WSL2 Ubuntu 22.04"
    ),
)
_add(
    "validated",
    ("yolo9_p2",),
    ("detect",),
    ("paddle",),
    reason=(
        "A YOLO9-P2-T conversion initialized from the SHA-256-pinned trained "
        "LibreYOLO9t checkpoint has raw-output and public detection parity; "
        "tests/e2e/test_paddle.py validates conversion, not P2 task accuracy."
    ),
    since="1.5",
    constraint=(
        "X2Paddle 1.6.0, PaddlePaddle 2.6.2 CPU on Python 3.11 or earlier, ONNX 1.17/opset 15, FP32, "
        "batch 1, fixed square input; WSL2 Ubuntu 22.04"
    ),
)
_add(
    "validated",
    ("ec",),
    ("pose", "segment"),
    ("paddle",),
    reason=(
        "Trained LibreECs pose and segmentation checkpoints have Paddle CPU "
        "raw-output parity plus task-aware public keypoint or mask parity in "
        "tests/e2e/test_paddle.py."
    ),
    since="1.5",
    constraint=(
        "X2Paddle 1.6.0, PaddlePaddle 2.6.2 CPU on Python 3.11 or earlier, ONNX 1.17/opset 15, FP32, "
        "batch 1, fixed square input; WSL2 Ubuntu 22.04"
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("detect", "pose"),
    ("paddle",),
    reason=(
        "Paddle CPU conversion, multi-output raw parity, and task-aware "
        "public detection/keypoint parity are covered in "
        "tests/e2e/test_paddle.py."
    ),
    since="1.5",
    constraint=(
        "X2Paddle 1.6.0, PaddlePaddle 2.6.2 CPU on Python 3.11 or earlier, ONNX 1.17/opset 15, FP32, "
        "batch 1, fixed square input; WSL2 Ubuntu 22.04"
    ),
)
_add(
    "blocked",
    ("rtdetr", "rtdetrv2"),
    ("detect",),
    ("paddle",),
    reason=(
        "The trained graphs require ONNX GridSample at opset 16 or newer, "
        "while X2Paddle 1.6.0 accepts opset 15 or lower."
    ),
)
_add(
    "blocked",
    ("dfine",),
    ("segment",),
    ("paddle",),
    reason=(
        "The trained LibreDFINEn segmentation graph converts and reloads, "
        "but mask-logit relative RMS error is 3.52% and minimum matched-mask "
        "IoU is only 0.582."
    ),
)
_add(
    "validated",
    ("efficientdet",),
    ("detect",),
    ("onnx", "torchscript"),
    reason=(
        "The official Apache-2.0 D0 checkpoint is covered by one-output "
        "artifact reload, metadata, runtime execution, and matched public "
        "post-NMS detection parity in tests/e2e/test_efficientdet_export.py."
    ),
    since="1.5",
    constraint="FP32, batch 1, fixed per-variant square input",
)
_add(
    "validated",
    ("efficientdet",),
    ("detect",),
    ("openvino",),
    reason=(
        "The official Apache-2.0 D0 checkpoint is covered by one-output "
        "artifact reload, metadata, OpenVINO CPU execution, and matched public "
        "post-NMS detection parity in tests/e2e/test_efficientdet_export.py."
    ),
    since="1.5",
    constraint=(
        "OpenVINO 2026.2, FP32, batch 1, fixed per-variant square input on CPU"
    ),
)
_add(
    "validated",
    ("efficientdet",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "The official Apache-2.0 D0 checkpoint is covered by engine build, "
        "artifact reload, metadata, runtime execution, and matched public "
        "post-NMS detection parity in tests/e2e/test_efficientdet_export.py."
    ),
    since="1.5",
    constraint=(
        "TensorRT 10.16, FP32, batch 1, fixed per-variant square input; "
        "TensorRT's ITopK limit uses 3840 candidates instead of the native "
        "5000-candidate budget"
    ),
)
_add(
    "validated",
    ("yolo9", "rfdetr"),
    ("detect",),
    ("executorch",),
    reason=(
        "Runtime and raw-output parity are covered in "
        "tests/e2e/test_executorch.py; trained-checkpoint detection parity is "
        "covered by its external-data flagship test."
    ),
    since="1.3",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "validated",
    ("yolo9", "rfdetr"),
    ("detect",),
    ("mnn",),
    reason=(
        "Trained-checkpoint conversion, fresh artifact reload, MNN CPU "
        "execution, metadata, and matched post-NMS detection parity are "
        "covered in tests/e2e/test_mnn.py."
    ),
    since="1.5",
    constraint="MNN 3.6.1, CPU, FP32, batch 1, fixed NCHW input shape",
)
_add(
    "validated",
    (
        "yolo9_e2e",
        "ec",
        "rtdetr",
        "rtdetrv2",
        "rtdetrv4",
        "dfine",
        "deim",
        "yolonas",
    ),
    ("detect",),
    ("mnn",),
    reason=(
        "Trained-checkpoint conversion, fresh artifact reload, MNN CPU "
        "execution, metadata, and matched post-NMS detection parity are "
        "covered in tests/e2e/test_mnn.py."
    ),
    since="1.5",
    constraint="MNN 3.6.1, CPU, FP32, batch 1, fixed NCHW input shape",
)
_add(
    "validated",
    ("yolo9_p2",),
    ("detect",),
    ("mnn",),
    reason=(
        "Conversion, fresh artifact reload, MNN CPU execution, metadata, and "
        "one-to-one public detection parity are covered with a deterministic "
        "strengthened-head fixture in tests/e2e/test_mnn.py."
    ),
    since="1.5",
    constraint="MNN 3.6.1, CPU, FP32, batch 1, fixed NCHW input shape",
)
_add(
    "available",
    ("deimv2",),
    ("detect",),
    ("mnn",),
    reason=(
        "The trained atto checkpoint converts, reloads, executes on MNN CPU, "
        "and preserves post-NMS detections, but the intermediate ONNX route "
        "has incomplete query-level score parity."
    ),
    since="1.5",
    constraint="MNN 3.6.1, CPU, FP32, batch 1, fixed NCHW input shape",
)
_add(
    "validated",
    ("ppyoloe",),
    ("detect",),
    ("onnx", "torchscript"),
    reason=(
        "The raw graph emits the decoded (boxes, scores) pair. TorchScript "
        "reproduces eager output exactly and ONNX to ~1e-3 px on boxes and "
        "~1e-7 on scores; public post-NMS detections match on batch 1 and 2 "
        "(tests/e2e/test_ppyoloe_export.py)."
    ),
    since="1.5",
    constraint="Fixed 640 canvas, FP32, batch 1 and 2",
)
_add(
    "validated",
    ("ppyoloe",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "Real engine build and inference, not an ONNX parse. A fixed-shape "
        "FP32 engine matches native post-NMS boxes to ~3e-02 px. Two "
        "limitations are deliberate: FP16 engines drift box localization "
        "materially on this head, and a dynamic-batch profile diverges at "
        "batch 2, so the validated claim is FP32 at fixed batch 1. OpenVINO "
        "carries the batch-1-and-2 compiled-backend coverage."
    ),
    since="1.5",
    constraint=(
        "TensorRT 10.16, FP32 engine (half=False), fixed 640 canvas, fixed batch 1"
    ),
)
_add(
    "validated",
    ("ppyoloe",),
    ("detect",),
    ("openvino",),
    reason=(
        "CPU FP32 conversion, raw two-tensor parity, factory reload and "
        "matching public post-NMS detections."
    ),
    since="1.5",
    constraint="OpenVINO CPU FP32, fixed 640 canvas, batch 1 and 2",
)
_add(
    "available",
    ("yolo9", "yolo9_e2e", "yolonas", "picodet"),
    ("detect",),
    ("rknn",),
    reason=(
        "Exact small variants passed RKNN Toolkit2 2.3.2 compilation, "
        "RK3588 PC-simulator raw-output gates, and matched post-NMS "
        "detections on a real image. Support is limited to YOLO9-t, "
        "YOLO9-E2E-t, YOLO-NAS-s, and PicoDet-s; on-device latency and "
        "parity have not been measured."
    ),
    constraint=(
        "RKNN Toolkit2 2.3.2, RK3588 PC simulator, vendor floating build, "
        "batch 1, fixed square input"
    ),
)
_add(
    "validated",
    (
        "ec",
        "picodet",
        "rtdetr",
        "rtdetrv2",
        "rtdetrv4",
        "yolo1",
        "yolo2",
        "yolo3",
        "yolo4",
        "yolo7",
        "yolo9_e2e",
        "yolox",
    ),
    ("detect",),
    ("executorch",),
    reason=(
        "Trained-checkpoint XNNPACK export, runtime execution, and matched "
        "post-NMS detection parity are covered by the external-data flagship "
        "test in tests/e2e/test_executorch.py."
    ),
    since="1.3",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "blocked",
    ("dfine",),
    ("detect",),
    ("executorch",),
    reason=(
        "Strict capture reaches an unsupported ContextVar read in deformable "
        "attention. Forcing the manual exported grid-sample path permits "
        "serialization, but ExecuTorch 1.2 runtime execution still fails with "
        "an invalid delegated tensor dimension order."
    ),
)
_add(
    "blocked",
    ("dfine",),
    ("segment",),
    ("executorch",),
    reason=(
        "Strict capture reaches the same untraceable deformable-attention "
        "ContextVar read as detection. Forcing the manual capture path permits "
        "serialization, but ExecuTorch 1.2 runtime execution fails with an "
        "invalid delegated tensor dimension order."
    ),
)
_add(
    "blocked",
    ("deim",),
    ("detect",),
    ("executorch",),
    reason=(
        "The trained nano model captures, lowers, and serializes, but "
        "ExecuTorch 1.2 runtime execution fails with an invalid delegated "
        "tensor dimension order."
    ),
)
_add(
    "available",
    ("rtmdet",),
    ("detect",),
    ("executorch",),
    reason=(
        "The export-only graph unshares RTMDet's cross-level head convolutions "
        "to avoid duplicate XNNPACK batch-norm fusion parameter names. Full "
        "conversion, runtime execution, input sensitivity, deterministic "
        "random-weight raw parity, and detection parsing are covered."
    ),
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "blocked",
    ("deimv2",),
    ("detect",),
    ("executorch",),
    reason=(
        "The trained atto model captures, lowers, and serializes, but the "
        "ExecuTorch 1.2 runtime process terminates while executing forward."
    ),
)
_add(
    "validated",
    ("yolo9_p2", "yolonas"),
    ("detect",),
    ("executorch",),
    reason=(
        "Deterministic input-sensitive fixtures cover XNNPACK conversion, "
        "runtime execution, per-output raw parity, metadata, and matched "
        "public post-NMS detection parity."
    ),
    since="1.5",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "validated",
    ("ec",),
    ("pose",),
    ("executorch",),
    reason=(
        "Deterministic input-sensitive fixtures cover XNNPACK conversion, "
        "runtime execution, per-output raw parity, metadata, and public "
        "postprocessing parity for boxes plus keypoints."
    ),
    since="1.5",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "validated",
    ("ec",),
    ("segment",),
    ("executorch",),
    reason=(
        "A deterministic input-sensitive fixture covers XNNPACK conversion, "
        "runtime execution, per-output raw parity, metadata, and public "
        "postprocessing parity for boxes plus masks."
    ),
    since="1.5",
    constraint=(
        "ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1; fixed input shape large "
        "enough for the top-300 query selection"
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("pose",),
    ("executorch",),
    reason=(
        "A deterministic input-sensitive fixture covers XNNPACK conversion, "
        "runtime execution, per-output raw parity, metadata, and matched "
        "public box and keypoint parity."
    ),
    since="1.5",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "validated",
    ("convnext",),
    ("classify",),
    ("executorch",),
    reason=(
        "A deterministic input-sensitive fixture covers XNNPACK conversion, "
        "runtime execution, per-output logits parity, metadata, and public "
        "probability cosine plus top-1 parity."
    ),
    since="1.5",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "validated",
    ("nafnet", "realesrgan"),
    ("restore",),
    ("executorch",),
    reason=(
        "Deterministic input-sensitive fixtures cover XNNPACK conversion, "
        "runtime execution, per-output image parity, metadata, and public "
        "restored-image parity above 40 dB PSNR."
    ),
    since="1.5",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "validated",
    ("fomo",),
    ("point",),
    ("executorch",),
    reason=(
        "A deterministic input-sensitive fixture covers XNNPACK conversion, "
        "runtime execution, per-output heatmap parity, metadata, and matched "
        "public point-coordinate and confidence parity."
    ),
    since="1.5",
    constraint=(
        "ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed square input shape"
    ),
)
_add(
    "validated",
    ("efficientnetv2", "mobilenetv4", "resnet"),
    ("classify",),
    ("executorch",),
    reason=(
        "Trained-checkpoint XNNPACK logits cosine and top-1 parity are covered "
        "by the external-data flagship test in tests/e2e/test_executorch.py."
    ),
    since="1.3",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "validated",
    ("pidnet",),
    ("semantic",),
    ("executorch",),
    reason=(
        "Trained-checkpoint XNNPACK semantic-mask parity is covered by the "
        "external-data flagship test in tests/e2e/test_executorch.py."
    ),
    since="1.3",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "validated",
    ("lingbotvision",),
    ("semantic",),
    ("executorch",),
    reason=(
        "Trained-checkpoint XNNPACK semantic-mask parity is covered by the "
        "external-data flagship test in tests/e2e/test_executorch.py."
    ),
    since="1.4",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "validated",
    ("dinov2",),
    ("semantic",),
    ("executorch",),
    reason=(
        "The real pretrained DINOv2 backbone has full XNNPACK conversion, "
        "runtime execution, input sensitivity, deterministic random-head raw "
        "parity, and public semantic-mask parity above 95% pixel agreement. "
        "This validates conversion compatibility, not trained task accuracy."
    ),
    since="1.5",
    constraint=(
        "ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed 518x518 input shape"
    ),
)
_add(
    "available",
    ("dinov2",),
    ("embed",),
    ("executorch",),
    reason=(
        "The real pretrained DINOv2 backbone has full XNNPACK conversion, "
        "runtime execution, input sensitivity, embedding-vector parity, "
        "normalization, and result parsing coverage."
    ),
    constraint=(
        "ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed 224x224 input shape"
    ),
)
_add(
    "validated",
    ("segformer",),
    ("semantic",),
    ("executorch",),
    reason=(
        "The b0 graph has full XNNPACK conversion, runtime execution, input "
        "sensitivity, deterministic random-weight logits parity, and public "
        "semantic-mask parity above 95% pixel agreement. This validates "
        "conversion compatibility, not trained task accuracy; published "
        "pretrained weights are non-commercial and are not used."
    ),
    since="1.5",
    constraint=(
        "ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape divisible by 32"
    ),
)
_add(
    "validated",
    ("depth_anything3",),
    ("depth",),
    ("executorch",),
    reason=(
        "The fixed-canvas graph exports raw depth and sky maps so LibreYOLO's "
        "runtime can apply the tensor-dependent sky correction and inverse-depth "
        "contract outside the portable graph. Conversion, runtime execution, "
        "raw-map parity, input sensitivity, and public depth-map parity above "
        "40 dB PSNR are covered."
    ),
    since="1.5",
    constraint=(
        "ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed square input shape "
        "divisible by 14"
    ),
)
_add(
    "validated",
    ("depth_anything3",),
    ("depth",),
    ("onnx", "torchscript", "openvino", "tensorrt"),
    reason=(
        "A deterministic input-sensitive fixture covers opset-17 conversion, "
        "artifact reload, two-image raw depth/sky parity with a 20x "
        "signal/error guard, metadata, and public depth-map parity above "
        "40 dB PSNR."
    ),
    since="1.5",
    constraint=(
        "FP32, batch 1, fixed square input divisible by 14; TensorRT evidence "
        "uses TensorRT 10.16 and OpenVINO evidence uses OpenVINO 2026.2"
    ),
)
_add(
    "blocked",
    ("eomt",),
    ("semantic",),
    ("executorch",),
    reason=(
        "Strict torch.export capture fails on a data-dependent symbolic "
        "expression in the mask path before XNNPACK lowering."
    ),
)
_add(
    "validated",
    ("moge2",),
    ("normal",),
    ("executorch",),
    reason=(
        "Trained-checkpoint XNNPACK angular normal-map parity is covered by "
        "the external-data flagship test in tests/e2e/test_executorch.py."
    ),
    since="1.4",
    constraint=(
        "ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed square input shape"
    ),
)
_add(
    "available",
    ("moge2",),
    ("normal",),
    ("ncnn",),
    reason=(
        "PNNX/NCNN 20260526 exports, reloads, and runs, but the measured "
        "two-image raw signal is only 4.5x conversion error; validation "
        "requires more than 20x."
    ),
)
_add(
    "blocked",
    ("moge2",),
    ("normal",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 flatbuffer-direct lowering cannot lower the encoder's "
        "cubic Resize because its input C/H/W signature remains dynamic."
    ),
)
_add(
    "validated",
    ("yolo9",),
    ("detect",),
    ("ncnn",),
    since="1.3",
    constraint=(
        "PNNX/NCNN 20260526 CPU FP32 with a fixed export canvas; trained MIT "
        "checkpoint covered by two-input raw parity, factory reload, metadata, "
        "and non-square public predict parity"
    ),
)
_add(
    "validated",
    ("yolo9",),
    ("detect",),
    ("tensorrt", "openvino"),
    reason=(
        "Runtime parity coverage lives in tests/e2e/test_tensorrt.py and "
        "tests/e2e/test_openvino.py."
    ),
    since="1.3",
)
_add(
    "blocked",
    ("yolo9",),
    ("segment",),
    EXPORT_FORMATS,
    reason="YOLO9 segmentation export is not supported; YOLO9 is detection-only in LibreYOLO.",
)
_add("validated", ("yolo9_p2",), ("detect",), ("onnx",), since="1.3")
_add(
    "validated",
    ("rfdetr",),
    ("detect",),
    ("onnx", "torchscript"),
    since="1.3",
)
_add(
    "validated",
    ("rfdetr",),
    ("detect",),
    ("tensorrt", "openvino"),
    reason=(
        "Runtime parity coverage lives in tests/e2e/test_tensorrt.py and "
        "tests/e2e/test_openvino.py."
    ),
    since="1.3",
)
_add(
    "blocked",
    ("rfdetr",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf emits a flatbuffer at the native 384x384 canvas, but LiteRT "
        "cannot allocate it because STRIDED_SLICE receives an input above its "
        "supported 5-D rank."
    ),
)
_add(
    "blocked",
    ("yolo1",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 emits an ONNX_EINSUM custom operation that LiteRT "
        "2.1.2 cannot prepare at the native 448x448 canvas."
    ),
)
_add(
    "validated",
    ("yolo9_e2e",),
    ("detect",),
    ("tflite",),
    reason=(
        "The trained LibreYOLO9E2Et checkpoint has raw-output and public "
        "predict parity in tests/e2e/test_tflite_round13.py (measured "
        "2026-11-01 on Linux)."
    ),
    since="1.7",
    constraint="onnx2tf 2.6.9 or newer, LiteRT 2.1.2, fixed 128x128 export canvas",
)
_add(
    "blocked",
    ("yolo9_p2",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.9 exports a runnable artifact, but public top-k class "
        "membership changes after LiteRT 2.1.2 conversion."
    ),
)
_add(
    "blocked",
    ("rtmdet",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 exports, reloads, and preserves raw output parity, but "
        "at the native 640x640 canvas public boxes fall to 0.911 IoU with "
        "29.9 px coordinate drift."
    ),
)
_add(
    "blocked",
    ("picodet",),
    ("detect",),
    ("tflite",),
    reason=(
        "LiteRT 2.1.2 cannot prepare the onnx2tf 2.6.7 artifact because a "
        "RESHAPE maps 19,200 input elements to 9,600 output elements."
    ),
)
_add(
    "blocked",
    ("dfine",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf flatbuffer-direct lowering crashes in GatherElements shape "
        "handling with an axis IndexError."
    ),
)
_add(
    "blocked",
    ("ec",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 emits an ONNX_LAYERNORMALIZATION custom operation that "
        "LiteRT 2.1.2 cannot prepare."
    ),
)
_add(
    "blocked",
    ("rtdetr",),
    ("detect",),
    ("tflite",),
    reason=(
        "LiteRT 2.1.2 rejects the onnx2tf 2.6.7 graph because a CONCATENATION "
        "receives incompatible 256 and 1 dimensions."
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("detect",),
    ("tflite",),
    since="1.5",
    constraint="fixed export canvas",
)
_add(
    "blocked",
    ("yolonas",),
    ("pose",),
    ("tflite",),
    reason=(
        "LiteRT rejects the converted pose graph because a CONCATENATION "
        "input has an unsupported/invalid tensor type."
    ),
)
_add(
    "validated",
    ("mobilenetv4", "convnext", "efficientnetv2", "resnet"),
    ("classify",),
    ("onnx", "torchscript"),
    since="1.3",
)
_add(
    "validated",
    ("vit",),
    ("classify",),
    ("onnx",),
    reason=(
        "A real AugReg ImageNet-1k checkpoint is covered by raw-logit and "
        "public probability parity in tests/unit/test_vit_export.py."
    ),
    since="1.5",
    constraint="FP32, fixed 224x224 input",
)
_add(
    "validated",
    ("deit",),
    ("classify",),
    ("onnx",),
    reason=(
        "The official DeiT-tiny checkpoint preserves upstream/native logits, "
        "exported raw logits, public probabilities, and top-1/top-5 results."
    ),
    since="1.5",
    constraint="CPU FP32, fixed 224x224 input; ONNX uses opset 17",
)
_add(
    "validated",
    ("deit",),
    ("classify",),
    ("torchscript",),
    reason=(
        "The official DeiT-tiny checkpoint preserves upstream/native logits, "
        "bit-exact exported raw logits and probabilities, and top-1/top-5 results."
    ),
    since="1.5",
    constraint="CPU FP32 with fixed 224x224 input",
)
_add(
    "validated",
    ("alexnet",),
    ("classify",),
    ("onnx",),
    reason=(
        "Tests cover raw logits plus ONNX Runtime artifact reload, public "
        "probabilities, metadata, and top-1 parity."
    ),
    since="1.5",
    constraint=(
        "FP32 at the native 224x224 input resolution; ONNX supports a dynamic "
        "batch axis"
    ),
)
_add(
    "validated",
    ("alexnet",),
    ("classify",),
    ("torchscript",),
    reason=(
        "Tests cover TorchScript artifact reload, public probabilities, "
        "metadata, and top-1 parity."
    ),
    since="1.5",
    constraint="FP32 at the native 224x224 input resolution",
)
_add(
    "validated",
    ("alexnet",),
    ("classify",),
    ("openvino",),
    reason=(
        "Official-checkpoint and deterministic-fixture runtime tests preserve "
        "probability cosine agreement and ordered top-k predictions."
    ),
    since="1.5",
    constraint="OpenVINO 2026.2 CPU FP32 at the fixed native 224x224 resolution",
)
_add(
    "validated",
    ("alexnet",),
    ("classify",),
    ("tensorrt",),
    reason=(
        "Official-checkpoint and deterministic-fixture runtime tests preserve "
        "probability cosine agreement and ordered top-k predictions."
    ),
    since="1.5",
    constraint="TensorRT 10.16 FP32 at the fixed native 224x224 resolution",
)
_add(
    "validated",
    ("swin",),
    ("classify",),
    ("onnx", "torchscript"),
    reason=(
        "The released Tiny ImageNet checkpoint is covered by artifact reload, "
        "trained-logit probability parity, metadata, and public top-1 parity "
        "in tests/e2e/test_swin_export.py."
    ),
    since="1.5",
    constraint="Swin V1 at its fixed 224x224 native input resolution",
)
_add(
    "validated",
    ("mobilenetv4", "convnext", "efficientnetv2", "resnet"),
    ("classify",),
    ("openvino",),
    since="1.5",
    constraint="fixed family-native input resolution",
)
_add(
    "validated",
    ("deit",),
    ("classify",),
    ("openvino",),
    reason=(
        "The official DeiT-tiny checkpoint preserves raw-logit and probability "
        "cosine above 0.999 with identical top-1/top-5 results."
    ),
    since="1.5",
    constraint="OpenVINO 2026.2 CPU FP32 with fixed 224x224 input",
)
_add(
    "validated",
    ("swin",),
    ("classify",),
    ("openvino",),
    reason=(
        "The released Tiny ImageNet checkpoint is covered by FP32 OpenVINO IR "
        "reload, trained probability cosine parity, metadata, and public "
        "top-1 parity in tests/e2e/test_swin_export.py."
    ),
    since="1.5",
    constraint="FP32 with a fixed 224x224 input resolution",
)
_add(
    "validated",
    ("mobilenetv4", "convnext", "efficientnetv2", "resnet"),
    ("classify",),
    ("tensorrt",),
    since="1.5",
    constraint="FP32 with fixed family-native input resolution",
)
_add(
    "validated",
    ("deit",),
    ("classify",),
    ("tensorrt",),
    reason=(
        "The official DeiT-tiny checkpoint preserves finite raw logits, "
        "probability cosine above 0.999, and identical top-1/top-5 results."
    ),
    since="1.5",
    constraint=(
        "TensorRT 10.16 FP16 on RTX 5070 Ti, fixed 224x224 batch-1 input, "
        "0.25 GiB tactic workspace"
    ),
)
_add(
    "validated",
    ("vgg",),
    ("classify",),
    ("onnx", "torchscript"),
    reason=(
        "A small-artifact PR-gate fixture covers conversion and raw-logit "
        "parity, while the official trained VGG-16 checkpoint covers native "
        "and backend probability/top-1 parity."
    ),
    since="1.5",
    constraint="FP32, batch 1, fixed 224x224 input",
)
_add(
    "validated",
    ("vgg",),
    ("classify",),
    ("openvino", "tensorrt"),
    reason=(
        "The official trained VGG-16 checkpoint is covered by fixed-224 FP32 "
        "backend probability parity and identical public top-1 output."
    ),
    since="1.5",
    constraint="FP32, batch 1, fixed 224x224 input",
)
_add(
    "validated",
    ("swin",),
    ("classify",),
    ("tensorrt",),
    reason=(
        "The released Tiny ImageNet checkpoint is covered by FP32 TensorRT "
        "engine reload, trained probability cosine parity, metadata, and "
        "public top-1 parity in tests/e2e/test_swin_export.py."
    ),
    since="1.5",
    constraint="FP32, batch 1, and a fixed 224x224 input resolution",
)
_add(
    "validated",
    ("clip", "siglip2"),
    ("classify",),
    ("onnx",),
    since="1.3",
    constraint="frozen-class labels and fixed input resolution",
)
_add(
    "validated",
    ("clip", "siglip2"),
    ("classify",),
    ("torchscript", "executorch", "tensorrt", "openvino"),
    reason=(
        "A deterministic input-sensitive frozen-class fixture covers artifact "
        "reload, two-input raw-logit parity with a 20x signal/error guard, "
        "metadata, class names, and public softmax/top-1 parity."
    ),
    since="1.5",
    constraint=(
        "batch 1, fixed square input, class set frozen at export time; "
        "SigLIP2 uses single-label softmax mode"
    ),
)
_add(
    "validated",
    ("siglip2",),
    ("classify",),
    ("tflite",),
    reason=(
        "A deterministic input-sensitive frozen-class fixture covers onnx2tf "
        "conversion, LiteRT reload, two-input raw-logit parity with a 20x "
        "signal/error guard, metadata, class names, and public softmax/top-1 parity."
    ),
    since="1.5",
    constraint=(
        "onnx2tf 2.6.7, LiteRT 2.1.2 CPU FP32, batch 1, fixed square input, "
        "class set frozen at export time, single-label softmax mode"
    ),
)
_add(
    "validated",
    ("pe",),
    ("classify", "embed"),
    ("onnx", "torchscript"),
    reason=(
        "Fixed-input graphs are compared against native PyTorch in "
        "tests/unit/test_pe_export.py: TorchScript reproduces native exactly "
        "(max_abs_diff 0.0) for the image-embed, frozen-class and fixed-frame "
        "video graphs; ONNX matches to 4.3e-07 (embed), 1.3e-07 (video) and "
        "1.5e-05 (classify, on logit_scale-amplified logits)."
    ),
    since="1.5",
    constraint=(
        "fixed square input; dynamic batch only. Video graphs have a static "
        "frame count -- a dynamic F is not advertised. The classify graph "
        "freezes the class set present at export time."
    ),
)
_add(
    "blocked",
    ("pe",),
    ("classify", "embed"),
    ("ncnn", "coreml", "coreai", "tflite", "paddle", "rknn"),
    reason=(
        "No parity-validated PE artifact exists for this runtime; the family "
        "ships only the ONNX and TorchScript graphs it was tested in."
    ),
)
_add(
    "blocked",
    ("clip",),
    ("classify",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 emits a LiteRT graph whose TRANSPOSE receives a rank-5 "
        "permutation for a rank-4 tensor."
    ),
)
_add(
    "blocked",
    ("clip", "siglip2"),
    ("classify",),
    ("ncnn", "coreml"),
    reason="No parity-valid frozen-class artifact is available for this runtime.",
)
_add(
    "blocked",
    ("dinov2",),
    ("classify",),
    ("ncnn", "tflite", "coreml"),
    reason="LibreDINOv2 classify export is not implemented for this format.",
)
_add(
    "validated",
    ("dinov2",),
    ("classify",),
    ("openvino",),
    reason=(
        "A deterministic input-sensitive fixture covers conversion, artifact "
        "reload, raw-logit parity, metadata, and public probability cosine "
        "plus top-1 parity."
    ),
    since="1.5",
    constraint="OpenVINO 2026.2 CPU FP32, batch 1, fixed 224x224 input",
)
_add(
    "available",
    ("dinov2",),
    ("classify",),
    ("tensorrt",),
    reason=(
        "A deterministic input-sensitive fixture exports, reloads, and runs, "
        "but changed-input logits carry only 2.2x more native signal than "
        "TensorRT 10.16 FP32 conversion error; validation requires more than 20x."
    ),
)
_add(
    "validated",
    ("dinov2",),
    ("classify",),
    ("executorch",),
    reason=(
        "A deterministic input-sensitive fixture covers XNNPACK conversion, "
        "runtime execution, raw-logit parity, metadata, and public probability "
        "cosine plus top-1 parity."
    ),
    since="1.5",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "validated",
    ("clip", "siglip2"),
    ("embed",),
    ("onnx", "torchscript", "openvino", "tensorrt", "executorch"),
    reason=(
        "Deterministic input-sensitive image-tower fixtures cover artifact "
        "reload, two-input raw embedding parity with a 20x signal/error guard, "
        "metadata, normalization, and public embedding parity."
    ),
    since="1.5",
    constraint=(
        "FP32, batch 1, fixed family-native square input; ExecuTorch uses "
        "1.2/XNNPACK, TensorRT uses 10.16, and OpenVINO uses 2026.2"
    ),
)
_add(
    "blocked",
    ("clip",),
    ("embed",),
    ("ncnn",),
    reason=(
        "PNNX 20260526 leaves unsupported pnnx.Expression nodes in the CLIP "
        "attention graph, so the generated NCNN network has no runnable input."
    ),
)
_add(
    "blocked",
    ("clip",),
    ("embed",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 emits a LiteRT graph whose TRANSPOSE receives a rank-5 "
        "permutation for a rank-4 tensor."
    ),
)
_add(
    "blocked",
    ("siglip2",),
    ("embed",),
    ("ncnn",),
    reason=(
        "PNNX 20260526 leaves unsupported pnnx.Expression nodes in the SigLIP2 "
        "attention graph, so the generated NCNN network has no runnable input."
    ),
)
_add(
    "validated",
    ("siglip2",),
    ("embed",),
    ("tflite",),
    reason=(
        "A deterministic input-sensitive image-tower fixture covers onnx2tf "
        "conversion, LiteRT reload, two-input raw embedding parity with a 20x "
        "signal/error guard, metadata, normalization, and public embedding parity."
    ),
    since="1.5",
    constraint="onnx2tf 2.6.7, LiteRT 2.1.2 CPU FP32, batch 1, fixed square input",
)
_add(
    "blocked",
    ("clip", "siglip2"),
    ("embed",),
    ("coreml", "coreai"),
    reason="No parity-valid embedding artifact is available for this runtime.",
)
_add(
    "validated",
    ("dinov2",),
    ("embed",),
    ("onnx", "torchscript"),
    reason=(
        "The pretrained Apache-2.0 backbone covers artifact reload, two-input "
        "raw embedding parity with a 20x signal/error guard, metadata, "
        "normalization, and public embedding parity."
    ),
    since="1.5",
    constraint="FP32, batch 1, fixed 224x224 input",
)
_add(
    "available",
    ("dinov2",),
    ("embed",),
    ("openvino",),
    reason=(
        "OpenVINO 2026.2 exports, reloads, and predicts, but 11.2% of embedding "
        "elements miss strict tolerance with maximum error 0.0124."
    ),
)
_add(
    "available",
    ("dinov2",),
    ("embed",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 exports, reloads, and predicts, but 0.52% of embedding "
        "elements miss strict tolerance with maximum error 0.00782."
    ),
)
_add(
    "blocked",
    ("dinov2",),
    ("embed",),
    ("ncnn",),
    reason=(
        "PNNX 20260526 cannot lower the DINOv2 attention graph's batch-axis "
        "broadcasts and leaves an unsupported pnnx.Expression node."
    ),
)
_add(
    "validated",
    ("dinov2",),
    ("embed",),
    ("tflite",),
    reason=(
        "The pretrained Apache-2.0 backbone covers onnx2tf conversion, LiteRT "
        "reload, two-input raw embedding parity with a 20x signal/error guard, "
        "metadata, normalization, and public embedding parity."
    ),
    since="1.5",
    constraint="onnx2tf 2.6.7, LiteRT 2.1.2 CPU FP32, batch 1, fixed square input",
)
_add(
    "blocked",
    ("dinov2",),
    ("embed",),
    ("coreml", "coreai"),
    reason="No parity-valid embedding artifact is available for this runtime.",
)
_add(
    "blocked",
    ("birefnet", "feynobg"),
    ("matte",),
    ("ncnn",),
    reason=(
        "BiRefNet's decoder requires torchvision deformable convolution, "
        "which PNNX/NCNN cannot lower to a runnable graph."
    ),
)
_add(
    "blocked",
    ("birefnet",),
    ("matte",),
    ("executorch",),
    reason=(
        "Strict capture succeeds at the fixed 1024x1024 canvas, but ExecuTorch "
        "1.2 lowering has no out variant for torchvision::deform_conv2d."
    ),
)
_add(
    "blocked",
    ("feynobg",),
    ("matte",),
    ("executorch",),
    reason=(
        "The fixed 1024x1024 large graph exceeded the local conversion "
        "timebox while its working set grew past 4.7 GB; no .pte artifact was "
        "produced, so runtime parity remains untested."
    ),
)

# Explicitly permitted but not yet parity-validated combinations.
_add(
    "blocked",
    ("rfdetr",),
    ("segment",),
    ("tflite",),
    reason=(
        "onnx2tf 2.4.x assigns an invalid NHWC layout to the segmentation-head "
        "Einsum (78 channels versus the required 256), so conversion fails."
    ),
)
_add(
    "available",
    ("birefnet", "feynobg"),
    ("matte",),
    ("onnx",),
    reason=(
        "The opset-19 DeformConv graph exports, but ONNX Runtime's CPU "
        "provider has no DeformConv implementation for runtime parity."
    ),
)
_add(
    "blocked",
    ("birefnet", "feynobg"),
    ("matte",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 reaches the shared ONNX DeformConv node but cannot "
        "parse it because ModulatedDeformConv2d is absent from the plugin "
        "registry."
    ),
)
_add(
    "validated",
    ("birefnet",),
    ("matte",),
    ("torchscript",),
    since="1.4",
    constraint="fixed 1024x1024 input",
)
_add(
    "validated",
    ("feynobg",),
    ("matte",),
    ("torchscript",),
    since="1.5",
    constraint="fixed 1024x1024 input",
)
_add(
    "validated",
    ("ben2",),
    ("matte",),
    ("onnx", "torchscript"),
    reason=(
        "The trained MIT checkpoint is covered by fixed-shape export, CPU "
        "runtime reload, raw-logit parity, and public matte preprocessing."
    ),
    since="1.5",
    constraint="FP32, batch 1, fixed 1024x1024 input",
)
_add(
    "blocked",
    ("rfdetr",),
    ("pose",),
    ("tflite",),
    reason=(
        "RF-DETR pose-x TFLite conversion exceeded the CPU timebox and 8 GB "
        "working memory without producing an artifact on this toolchain."
    ),
)
_add(
    "available",
    ("yolox", "yolo9", "rtdetr", "rfdetr"),
    ("detect",),
    ("coreml",),
    reason="Conversion is available, but runtime parity requires a macOS runner.",
)
_add(
    "validated",
    ("dinov2", "eomt", "lingbotvision"),
    ("semantic",),
    ("openvino",),
    since="1.5",
    constraint="fixed family-native export canvas",
)
_add(
    "validated",
    ("dinov2", "eomt"),
    ("semantic",),
    ("tensorrt",),
    since="1.5",
    constraint="FP32 with a fixed family-native export canvas",
)
_add(
    "available",
    ("lingbotvision",),
    ("semantic",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 FP32 exports, reloads, and predicts, but repeated "
        "builds produced raw-logit cosine as low as 0.9842, below the 0.999 "
        "promotion gate."
    ),
)
_add(
    "available",
    ("pidnet",),
    ("semantic",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 FP32 exports and runs, but repeated builds produced "
        "raw-logit cosine as low as 0.9970, below the 0.999 promotion gate."
    ),
)
_add(
    "validated",
    ("pidnet",),
    ("semantic",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "validated",
    ("pidnet",),
    ("semantic",),
    ("openvino",),
    since="1.5",
    constraint="fixed square input",
)
_add(
    "validated",
    ("ppliteseg",),
    ("semantic",),
    ("onnx", "torchscript", "openvino", "tensorrt"),
    since="1.5",
    constraint=(
        "fixed native rectangle per size (512x1024 for t50/b50, 768x1536 for "
        "t75/b75). All four sizes were exported and run in each format: "
        "raw-logit cosine vs eager >= 0.999994 at batch 1 and batch 2 (ONNX is "
        "exact to 8e-05 max abs diff), and original-canvas semantic-mask "
        "agreement >= 99.6% through the factory reload. TensorRT 10.x FP32 "
        "engines were built and executed, not just parsed"
    ),
)
_add(
    "validated",
    ("l2cs",),
    ("gaze",),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="head-only contract: each input image is one face crop",
)
_add(
    "validated",
    ("l2cs",),
    ("gaze",),
    ("executorch",),
    reason=(
        "A deterministic input-sensitive fixture covers XNNPACK conversion, "
        "runtime execution, two-head raw-logit parity, metadata, and public "
        "pitch/yaw parity for the fixed face-crop contract."
    ),
    since="1.5",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed 448x448 face crop"),
)
_add(
    "validated",
    ("depth_anything", "zipdepth"),
    ("depth",),
    ("executorch",),
    reason=(
        "Input-sensitive fixtures cover XNNPACK conversion, runtime execution, "
        "raw-depth parity with a 100x signal/error margin, metadata, and public "
        "depth-map parity above 40 dB PSNR."
    ),
    since="1.5",
    constraint=(
        "ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape; "
        "Depth Anything uses the Apache-2.0 Small checkpoint"
    ),
)
_add(
    "validated",
    ("rfdetr",),
    ("segment", "pose", "obb"),
    ("executorch",),
    reason=(
        "Input-sensitive fixtures cover XNNPACK conversion, runtime execution, "
        "query-aligned raw-output parity with a 100x signal/error margin, "
        "metadata, and task-aware public boxes plus masks, keypoints, or OBB "
        "geometry parity."
    ),
    since="1.5",
    constraint=(
        "ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed task-native input "
        "shape; segment and pose use Apache-2.0 trained checkpoints"
    ),
)
_add(
    "validated",
    ("nafnet",),
    ("restore",),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed-resolution export canvas",
)
_add(
    "validated",
    ("nafnet",),
    ("restore",),
    ("ncnn",),
    since="1.4",
    constraint=(
        "PNNX/NCNN 20260526 CPU FP32 with a fixed-resolution export canvas; "
        "two-input raw parity, factory reload, metadata, and public predict "
        "parity"
    ),
)
_add(
    "validated",
    ("nafnet",),
    ("restore",),
    ("openvino",),
    since="1.5",
    constraint="fixed-resolution export canvas",
)
_add(
    "validated",
    ("nafnet",),
    ("restore",),
    ("tensorrt",),
    since="1.5",
    constraint="FP32 with a fixed-resolution export canvas",
)
_add(
    "blocked",
    ("nafnet",),
    ("restore",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 converts the fixed-canvas graph, but LiteRT 2.1.2 fails "
        "at invoke time because input tensor 4539 lacks data."
    ),
)
_add(
    "validated",
    ("realesrgan",),
    ("restore",),
    ("onnx",),
    since="1.4",
    constraint="dynamic spatial input",
)
_add(
    "validated",
    ("realesrgan",),
    ("restore",),
    ("torchscript",),
    since="1.4",
    constraint="fixed-resolution export canvas",
)
_add(
    "validated",
    ("realesrgan",),
    ("restore",),
    ("ncnn",),
    since="1.4",
    constraint=(
        "PNNX/NCNN 20260526 CPU FP32 with a fixed-resolution export canvas; "
        "two-input raw parity, factory reload, metadata, and public predict "
        "parity"
    ),
)
_add(
    "validated",
    ("realesrgan",),
    ("restore",),
    ("openvino",),
    since="1.5",
    constraint="fixed-resolution export canvas",
)
_add(
    "validated",
    ("realesrgan",),
    ("restore",),
    ("tensorrt",),
    since="1.5",
    constraint="FP32 with a fixed-resolution export canvas",
)
_add(
    "validated",
    ("realesrgan",),
    ("restore",),
    ("tflite",),
    since="1.4",
    constraint="fixed-resolution export canvas",
)
_add(
    "validated",
    ("quicksrnet",),
    ("restore",),
    ("onnx",),
    reason=(
        "A deterministic input-sensitive fixture covers dynamic-shape export, "
        "runtime reload, scale metadata, and native-versus-runtime pixel parity."
    ),
    since="1.5",
    constraint="FP32, dynamic spatial input",
)
_add(
    "validated",
    ("quicksrnet",),
    ("restore",),
    ("torchscript",),
    reason=(
        "A deterministic input-sensitive fixture covers fixed-canvas export, "
        "runtime reload, scale metadata, and native-versus-runtime pixel parity."
    ),
    since="1.5",
    constraint="FP32, fixed-resolution export canvas",
)
_add(
    "blocked",
    ("depth_anything",),
    ("depth",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 converts the DINOv2 depth graph, but LiteRT 2.1.2 "
        "cannot broadcast [1,3,3,32] and [1,72,72,32] in a generated ADD."
    ),
)
_add(
    "validated",
    ("yolox",),
    ("detect",),
    ("tflite",),
    since="1.4",
)
_add(
    "validated",
    ("pidnet",),
    ("semantic",),
    ("ncnn",),
    since="1.4",
    constraint=(
        "PNNX/NCNN 20260526 CPU FP32 with a fixed export canvas; two-input raw "
        "parity, factory reload, metadata, and public predict parity"
    ),
)
_add(
    "validated",
    ("fomo",),
    ("point",),
    ("ncnn",),
    since="1.4",
    constraint=(
        "PNNX/NCNN 20260526 CPU FP32 with a fixed 96x96 input; two-input raw "
        "parity, factory reload, metadata, and public predict parity"
    ),
)
_add(
    "validated",
    ("picosam3",),
    ("segment",),
    ("onnx",),
    since="1.4",
    constraint="raw fixed-96 ROI contract: roi_image -> mask_logits",
)
_add(
    "blocked",
    ("picosam3",),
    ("segment",),
    tuple(fmt for fmt in EXPORT_FORMATS if fmt != "onnx"),
    reason="PicoSAM3 currently exports its raw ROI CNN through ONNX only.",
)
_add(
    "validated",
    ("instanthmr",),
    ("mesh",),
    ("onnx",),
    since="1.7",
    constraint=(
        "raw fixed-288 person-crop contract: image, cliff_cond -> mhr_params, "
        "shape_params, cam_trans, joints_2d, joints_3d, vertices; mesh faces "
        "in metadata"
    ),
)
_add(
    "validated",
    ("fomo",),
    ("point",),
    ("tensorrt",),
    since="1.5",
    constraint="FP32 with a fixed 96x96 input",
)
_add(
    "validated",
    ("fomo",),
    ("point",),
    ("openvino",),
    since="1.5",
    constraint="fixed square input",
)
_add(
    "validated",
    ("zipdepth",),
    ("depth",),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed-resolution export canvas",
)
_add(
    "validated",
    ("zipdepth",),
    ("depth",),
    ("ncnn",),
    since="1.4",
    constraint=(
        "PNNX/NCNN 20260526 CPU FP32 with a fixed-resolution export canvas; "
        "two-input raw parity, factory reload, metadata, and public predict "
        "parity"
    ),
)
_add(
    "validated",
    ("teed", "dexined"),
    ("edge",),
    ("onnx",),
    since="1.5",
    constraint="fixed-resolution batch-1 edge-probability canvas",
)
_add(
    "validated",
    ("teed", "dexined"),
    ("edge",),
    ("executorch",),
    reason=(
        "Deterministic input-sensitive fixtures cover XNNPACK conversion, "
        "runtime execution, two-image edge-probability parity, and metadata."
    ),
    since="1.5",
    constraint=("ExecuTorch 1.2, XNNPACK, CPU, FP32, batch 1, fixed input shape"),
)
_add(
    "blocked",
    ("teed", "dexined"),
    ("edge",),
    ("ncnn",),
    reason=(
        "PNNX 20260526 leaves an unsupported Tensor.index channel-reversal node, "
        "so the generated NCNN network has no runnable input."
    ),
)
_add(
    "blocked",
    ("teed", "dexined"),
    ("edge",),
    ("coreai", "coreml"),
    reason=("This edge runtime has no parity-valid artifact for the requested format."),
)
_add(
    "validated",
    ("teed", "dexined"),
    ("edge",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 and LiteRT 2.1.2 cover artifact reload, two-image raw "
        "edge-probability parity with a 20x signal/error guard, metadata, and "
        "public edge-map parity above 40 dB PSNR."
    ),
    since="1.5",
    constraint="LiteRT 2.1.2 CPU FP32, batch 1, fixed input shape",
)
_add(
    "validated",
    ("teed", "dexined"),
    ("edge",),
    ("torchscript",),
    reason=(
        "Deterministic input-sensitive fixtures cover conversion, artifact "
        "reload, two-image raw edge-probability parity, metadata, and public "
        "edge-map parity above 40 dB PSNR."
    ),
    since="1.5",
    constraint="TorchScript CPU FP32, batch 1, fixed input shape",
)
_add(
    "validated",
    ("teed", "dexined"),
    ("edge",),
    ("openvino",),
    reason=(
        "Deterministic input-sensitive fixtures cover conversion, artifact "
        "reload, two-image raw edge-probability parity, metadata, and public "
        "edge-map parity above 40 dB PSNR."
    ),
    since="1.5",
    constraint="OpenVINO 2026.2 CPU FP32, batch 1, fixed input shape",
)
_add(
    "validated",
    ("teed", "dexined"),
    ("edge",),
    ("tensorrt",),
    reason=(
        "Deterministic input-sensitive fixtures cover conversion, artifact "
        "reload, two-image raw edge-probability parity, metadata, and public "
        "edge-map parity above 40 dB PSNR."
    ),
    since="1.5",
    constraint="TensorRT 10.16 FP32, batch 1, fixed input shape",
)
_add(
    "validated",
    ("zipdepth",),
    ("depth",),
    ("openvino",),
    since="1.5",
    constraint="fixed-resolution export canvas",
)
_add(
    "available",
    ("zipdepth",),
    ("depth",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 FP32 exports, reloads, and predicts, but repeated "
        "builds produced raw depth PSNR as low as 30.27 dB, below the 40 dB "
        "promotion gate."
    ),
)
_add(
    "blocked",
    ("zipdepth",),
    ("depth",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 flatbuffer-direct conversion does not support the "
        "edge-mode Pad operation in ZipDepth's convex upsampler."
    ),
)
_add(
    "validated",
    ("picodet",),
    ("detect",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "validated",
    ("picodet",),
    ("detect",),
    ("ncnn",),
    since="1.4",
    constraint=(
        "PNNX/NCNN 20260526 CPU FP32 with a permissively licensed trained "
        "checkpoint; two-input raw parity, factory reload, metadata, and "
        "public predict parity"
    ),
)
_add(
    "validated",
    (
        "picodet",
        "rtmdet",
        "yolo1",
        "yolo2",
        "yolo3",
        "yolo4",
        "yolo7",
        "yolo9_e2e",
        "yolo9_p2",
        "yolox",
    ),
    ("detect",),
    ("openvino",),
    since="1.5",
    constraint="fixed export canvas; YOLO1 requires 448x448",
)
_add(
    "validated",
    ("yolo2", "yolo3", "yolo4"),
    ("detect",),
    ("tensorrt",),
    since="1.5",
    constraint="FP32 with a fixed export canvas",
)
_add(
    "validated",
    ("yolo1", "picodet", "rtmdet"),
    ("detect",),
    ("tensorrt",),
    since="1.5",
    constraint="TensorRT 10.16 FP32 with a fixed canvas; YOLO1 requires 448x448",
)
_add(
    "available",
    ("yolo7",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 FP32 exports and reloads, but the permissively licensed "
        "trained checkpoint changes the public top-k class membership."
    ),
)
_add(
    "available",
    ("yolo9_e2e",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "Repeated TensorRT 10.16 FP32 engine builds with the permissively "
        "licensed trained checkpoint alternate between public top-k class "
        "drift and parity."
    ),
)
_add(
    "available",
    ("yolo9_p2",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "TensorRT 10.16 FP32 exports and reloads, but the pinned permissive "
        "YOLO9 transfer fixture changes the public top-k class membership."
    ),
)
_add(
    "available",
    ("yolox",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "The permissively licensed trained checkpoint exports, reloads, and "
        "passes public predict parity, but normalized raw error is 1.6% and "
        "image signal is only 2.1 times the conversion error."
    ),
)
_add(
    "available",
    ("rtdetr",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "The permissively licensed trained checkpoint exports, reloads, and "
        "passes public predict parity, but normalized raw outputs drift by "
        "17% to 38% after TensorRT 10.16 conversion."
    ),
)
_add(
    "available",
    ("yolonas",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "A deterministic synthetic trained fixture exports, reloads, and "
        "passes public predict parity, but image signal is only 4 to 5 times "
        "the TensorRT conversion error."
    ),
)
_add(
    "available",
    ("yolonas",),
    ("pose",),
    ("tensorrt",),
    reason=(
        "A deterministic synthetic trained fixture exports, reloads, and "
        "passes public predict parity, but image signal is only 2 to 6 times "
        "the TensorRT conversion error."
    ),
)
_add(
    "available",
    ("dfine",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained checkpoint exports and reloads, but "
        "public top-k class membership changes after TensorRT 10.16 FP32 "
        "conversion."
    ),
)
_add(
    "available",
    ("dfine",),
    ("segment",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained segmentation checkpoint exports and "
        "reloads, but public top-k class membership changes after TensorRT "
        "10.16 FP32 conversion."
    ),
)
_add(
    "available",
    ("deim",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained checkpoint exports, reloads, and "
        "passes public predict parity, but normalized raw output error is "
        "0.41%, above the 0.1% promotion gate."
    ),
)
_add(
    "available",
    ("rtdetrv2",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "A deterministic synthetic fixture exports and reloads, but matched "
        "public boxes drift by at least 8 pixels and fall to 0.231 IoU."
    ),
)
_add(
    "available",
    ("rtdetrv4",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "A deterministic synthetic fixture exports, reloads, and predicts, "
        "but repeated TensorRT 10.16 FP32 builds change public top-k class "
        "membership or box geometry; a measured reconstruction reached "
        "0 IoU with 50.4-pixel coordinate drift."
    ),
)
_add(
    "available",
    ("ec",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained checkpoint exports, reloads, and "
        "passes public predict parity, but normalized raw output error is "
        "1.2%, above the 0.1% promotion gate."
    ),
)
_add(
    "available",
    ("ec",),
    ("pose",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained pose checkpoint exports and reloads, "
        "but matched public boxes fall to 0.920 IoU with 1.43-pixel "
        "coordinate drift."
    ),
)
_add(
    "available",
    ("ec",),
    ("segment",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained segmentation checkpoint exports and "
        "reloads, but public top-k class membership changes."
    ),
)
_add(
    "available",
    ("rfdetr",),
    ("segment",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained segmentation checkpoint exports and "
        "reloads, but public top-k class membership changes."
    ),
)
_add(
    "available",
    ("rfdetr",),
    ("pose",),
    ("tensorrt",),
    reason=(
        "A published Apache-2.0 trained pose checkpoint exports and reloads, "
        "but matched public boxes fall to 0.704 IoU with 41.4-pixel "
        "coordinate drift."
    ),
)
_add(
    "available",
    ("rfdetr",),
    ("obb",),
    ("tensorrt",),
    reason=(
        "A deterministic synthetic OBB fixture exports and reloads, but "
        "public top-k class membership changes."
    ),
)
_add(
    "validated",
    ("yolo3", "yolo4"),
    ("detect",),
    ("ncnn",),
    since="1.4",
    constraint=(
        "PNNX/NCNN 20260526 CPU FP32 with public-domain trained checkpoints; "
        "two-input raw parity, factory reload, metadata, and public predict "
        "parity"
    ),
)
_add(
    "blocked",
    ("yolo2",),
    ("detect",),
    ("ncnn",),
    reason=(
        "The public-domain trained checkpoint exports through PNNX 20260526, "
        "but NCNN 20260526 on Windows terminates the runtime with a native "
        "integer divide-by-zero during output extraction."
    ),
)
_add(
    "blocked",
    ("yolo2",),
    ("detect",),
    ("tflite",),
    reason=(
        "LiteRT 2.1.2 cannot prepare the onnx2tf 2.6.7 artifact because a "
        "RESHAPE maps 4,225 input elements to one output element."
    ),
)
_add(
    "blocked",
    ("yolo3",),
    ("detect",),
    ("tflite",),
    reason=(
        "A public-domain trained checkpoint exports, reloads, and preserves "
        "normalized raw parity, but public top-k class membership changes."
    ),
)
_add(
    "blocked",
    ("yolo4",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 exports and runs, but public boxes fall to 0 IoU with "
        "176 px coordinate drift on the deterministic full model."
    ),
)
_add(
    "validated",
    ("yolo7",),
    ("detect",),
    ("ncnn",),
    since="1.4",
    constraint=(
        "PNNX/NCNN 20260526 CPU FP32 with a permissively licensed trained "
        "checkpoint; two-input raw parity, factory reload, metadata, and "
        "public predict parity"
    ),
)
_add(
    "blocked",
    ("yolo7",),
    ("detect",),
    ("tflite",),
    reason=(
        "The converted LiteRT graph changes decoded box coordinates beyond "
        "the detector parity tolerance."
    ),
)
_add(
    "validated",
    ("yolo9_e2e", "yolox"),
    ("detect",),
    ("ncnn",),
    since="1.4",
    constraint=(
        "PNNX/NCNN 20260526 CPU FP32 with permissively licensed trained "
        "checkpoints; two-input raw parity, factory reload, metadata, and "
        "public predict parity"
    ),
)
_add(
    "available",
    ("yolo9_p2",),
    ("detect",),
    ("ncnn",),
    reason=(
        "The SHA-pinned MIT YOLO9 transfer fixture exports, reloads, and "
        "preserves raw NCNN parity, but changes near-noise public top-k "
        "classes and produces no detections above 0.05 on the bundled real "
        "image."
    ),
)
_add(
    "validated",
    ("yolo1",),
    ("detect",),
    ("ncnn",),
    since="1.4",
    constraint=(
        "fixed 448x448 input; PNNX/NCNN 20260526 CPU FP32 with a public-domain "
        "trained checkpoint; two-input raw parity, factory reload, metadata, "
        "and public predict parity"
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("detect", "pose"),
    ("ncnn",),
    since="1.4",
    constraint=(
        "PNNX/NCNN 20260526 CPU FP32 with deterministic synthetic trained "
        "fixtures; two-input raw parity, factory reload, metadata, and public "
        "predict parity; pose additionally validates matched keypoints; this "
        "validates conversion, not task accuracy"
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("detect", "pose"),
    ("openvino",),
    since="1.5",
    constraint="fixed export canvas",
)
_add(
    "validated",
    ("yolo2", "yolo3", "yolo4"),
    ("detect",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "validated",
    ("yolo1", "yolo7", "yolo9_e2e", "yolox"),
    ("detect",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "validated",
    ("yolo9_p2",),
    ("detect",),
    ("torchscript",),
    since="1.4",
)
_add(
    "validated",
    ("yolonas",),
    ("detect", "pose"),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "blocked",
    ("rtmdet",),
    ("detect",),
    ("ncnn",),
    reason=(
        "PNNX 20260526 reports an unregistered nn.Conv2d layer and leaves the "
        "RTMDet NCNN graph without usable input blobs."
    ),
)
_add(
    "validated",
    ("rtmdet",),
    ("detect",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "blocked",
    ("rtmdet",),
    ("segment",),
    tuple(fmt for fmt in EXPORT_FORMATS if fmt != "axelera"),
    reason=(
        "RTMDet-Ins export is not supported yet; the dynamic-kernel mask "
        "decode has no exported-runtime contract. Use native PyTorch "
        "inference for task='segment'."
    ),
)
_add(
    "validated",
    ("swinir",),
    ("restore",),
    ("onnx", "torchscript", "openvino", "tflite"),
    since="1.5",
    constraint=(
        "fixed export canvas; raw-output and predict parity are validated when "
        "the source dimensions exactly match that canvas. Smaller sources are "
        "padded to the canvas before the exported transformer and can diverge "
        "from native variable-size inference."
    ),
)
_add(
    "validated",
    ("swinir",),
    ("restore",),
    ("tensorrt",),
    since="1.5",
    constraint=(
        "FP32 with a fixed export canvas; raw-output and predict parity are "
        "validated when the source dimensions exactly match that canvas."
    ),
)
_add(
    "blocked",
    ("swinir",),
    ("restore",),
    ("ncnn",),
    reason=(
        "PNNX writes NCNN artifacts after reporting unsupported 5-rank "
        "Permute operations, but the NCNN runtime process exits while loading "
        "or executing the resulting graph."
    ),
)
_add(
    "blocked",
    ("swinir",),
    ("restore",),
    ("executorch",),
    reason=(
        "The fixed-canvas graph captures, lowers, serializes, and reloads, but "
        "ExecuTorch 1.2 runtime execution fails in aten::alias_copy.out because "
        "the source and destination tensors have different dimension orders."
    ),
)
_add(
    "blocked",
    ("birefnet", "feynobg"),
    ("matte",),
    ("openvino",),
    reason=(
        "OpenVINO 2026.2 cannot lower the shared matte decoder's standard "
        "ONNX DeformConv-19 operation."
    ),
)
_add(
    "blocked",
    ("dfine",),
    ("segment",),
    ("tflite",),
    reason=(
        "onnx2tf flatbuffer-direct lowering crashes in GatherElements shape "
        "handling with an axis IndexError."
    ),
)
_add(
    "blocked",
    ("rtdetrv4",),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf flatbuffer-direct lowering crashes in GatherElements shape "
        "handling with an axis IndexError at the native 640x640 canvas."
    ),
)
_add(
    "validated",
    ("dfine", "deim", "deimv2", "ec", "rtdetr", "rtdetrv2", "rtdetrv4"),
    ("detect",),
    ("torchscript",),
    since="1.4",
)
_add(
    "validated",
    ("dfine", "ec", "rtdetr"),
    ("detect",),
    ("onnx",),
    since="1.4",
)
_add(
    "validated",
    ("detr",),
    ("detect",),
    ("onnx", "torchscript"),
    reason=(
        "Official-checkpoint raw outputs and public predict results are covered "
        "by native, ONNX Runtime, and TorchScript parity tests."
    ),
    since="1.5",
    constraint="FP32, batch 1, fixed square input",
)
_add(
    "validated",
    ("deformable_detr",),
    ("detect",),
    ("onnx",),
    reason=(
        "All five official ResNet-50 variants preserve raw-logit, box, and "
        "public prediction parity through ONNX Runtime at a fixed square canvas."
    ),
    since="1.5",
    constraint="FP32, fixed square input, ONNX opset 17",
)
_add(
    "validated",
    ("dinodetr",),
    ("detect",),
    ("onnx",),
    reason=(
        "All three official ResNet-50 and Swin-L variants preserve raw-logit, "
        "box, and public prediction parity through ONNX Runtime."
    ),
    since="1.5",
    constraint="FP32, fixed square input, ONNX opset 17",
)
_add(
    "validated",
    ("lwdetr",),
    ("detect",),
    ("onnx", "torchscript"),
    reason=(
        "Runtime parity checked against native PyTorch on the same device: "
        "identical class ids and detection counts, scores within 3e-6 (ONNX) "
        "and 6e-8 (TorchScript)."
    ),
    since="1.5",
)
for _fmt, _constraint in (
    ("onnx", "FP32, fixed 800x800 input, ONNX opset 17"),
    ("torchscript", "FP32, fixed 800x800 input"),
):
    _add(
        "validated",
        ("ppdoclayoutv3",),
        ("detect",),
        (_fmt,),
        reason=(
            "Raw logits, boxes and reading-order votes match native PyTorch "
            "(TorchScript exactly, ONNX Runtime within fp32 noise); "
            "exported-backend predict returns the same regions, labels and "
            "reading order."
        ),
        since="1.7",
        constraint=_constraint,
    )
del _fmt, _constraint
_add(
    "validated",
    ("centernet",),
    ("detect",),
    ("onnx", "torchscript"),
    reason=(
        "Both official checkpoints preserve baked top-100 detections and public "
        "prediction parity through the portable grid-sample DCN graph."
    ),
    since="1.5",
    constraint="FP32, fixed square input; ONNX Runtime CPU or TorchScript",
)
_add(
    "blocked",
    ("centernet",),
    ("detect",),
    ("ncnn",),
    reason=(
        "NCNN cannot lower CenterNet's portable deformable sampling plus "
        "baked top-k decode contract. Use ONNX or TorchScript."
    ),
)
_add(
    "validated",
    ("faster_rcnn",),
    ("detect",),
    ("onnx",),
    reason=(
        "Official trained-checkpoint parity covers graph outputs and unified "
        "ONNX-backend detections against native PyTorch."
    ),
    since="1.5",
    constraint=(
        "ONNX Runtime, FP32, opset 18, batch 1, dynamic source H/W; upstream "
        "aspect resize and final class-wise NMS are embedded in the graph"
    ),
)
_add(
    "validated",
    ("ssd",),
    ("detect",),
    ("onnx",),
    reason=(
        "The official trained checkpoint preserves the decoded raw grid and "
        "public post-NMS predictions through ONNX Runtime."
    ),
    since="1.5",
    constraint="ONNX Runtime, FP32, opset 13, fixed 300 x 300 input",
)
_add(
    "blocked",
    ("ssd",),
    ("detect",),
    (
        "torchscript",
        "executorch",
        "tensorrt",
        "openvino",
        "ncnn",
        "tflite",
        "coreml",
        "coreai",
    ),
    reason=(
        "SSD's decoded fixed-default-box head has only been parity-validated "
        "through the ONNX Runtime contract."
    ),
)
_add(
    "blocked",
    ("faster_rcnn",),
    ("detect",),
    ("torchscript", "executorch"),
    reason=(
        "The variable-length two-stage detection graph has only been "
        "validated through the ONNX runtime contract."
    ),
)
_add(
    "blocked",
    ("faster_rcnn",),
    ("detect",),
    ("tensorrt", "openvino", "ncnn", "tflite", "coreml", "coreai"),
    reason=(
        "This runtime has no parity evidence for Faster R-CNN's proposal, "
        "RoIAlign, variable-length output, and embedded-NMS graph."
    ),
)
_add(
    "validated",
    ("retinanet",),
    ("detect",),
    ("onnx",),
    reason=(
        "Official-checkpoint parity covers decoded graph outputs and unified "
        "ONNX-backend detections against native PyTorch."
    ),
    since="1.5",
    constraint=(
        "ONNX Runtime, FP32, opset 13, batch 1, dynamic preprocessed H/W; "
        "class-aware NMS runs in the LibreYOLO backend"
    ),
)
_add(
    "validated",
    ("fcos",),
    ("detect",),
    ("onnx",),
    reason=(
        "The official trained checkpoint preserves the single-tensor raw "
        "contract and public post-NMS detections in ONNX Runtime."
    ),
    since="1.5",
    constraint=(
        "FP32, batch 1, out-of-graph aspect resize, opset 18, dynamic padded H/W"
    ),
)
_add(
    "validated",
    ("mask_rcnn",),
    ("detect", "segment"),
    ("onnx",),
    reason=(
        "Official trained-checkpoint parity covers final boxes, scores, labels, "
        "and full-image masks through ONNX Runtime and the unified backend."
    ),
    since="1.5",
    constraint=(
        "ONNX Runtime, FP32, opset 18, batch 1, dynamic source H/W; upstream "
        "aspect resize, class-wise NMS, RoIAlign, and mask paste are embedded"
    ),
)
_add(
    "validated",
    ("fcos",),
    ("detect",),
    ("torchscript",),
    reason=(
        "The official trained checkpoint preserves the single-tensor raw "
        "contract and public post-NMS detections in TorchScript."
    ),
    since="1.5",
    constraint="FP32, batch 1, out-of-graph aspect resize, variable padded H/W",
)
_add(
    "available",
    ("fcos",),
    ("detect",),
    ("openvino",),
    reason=(
        "FP32 dynamic-shape conversion and high-confidence public predictions "
        "pass, but small score/box drift can change low-confidence NMS ordering."
    ),
    since="1.5",
    constraint="OpenVINO CPU, FP32, batch 1, dynamic padded H/W",
)
_add(
    "blocked",
    ("fcos",),
    ("detect",),
    ("tensorrt",),
    reason=(
        "FCOS requires dynamic padded H/W to preserve its 800/1333 aspect "
        "transform, while the current TensorRT runtime profiles dynamic batch only."
    ),
)
_add(
    "blocked",
    ("retinanet",),
    ("detect",),
    (
        "torchscript",
        "executorch",
        "tensorrt",
        "openvino",
        "ncnn",
        "tflite",
        "coreml",
        "coreai",
    ),
    reason=(
        "RetinaNet's dynamic P3-P7 anchor graph and external class-aware "
        "postprocessing have parity evidence only through ONNX Runtime."
    ),
)
_add(
    "blocked",
    ("mask_rcnn",),
    ("detect", "segment"),
    tuple(fmt for fmt in EXPORT_FORMATS if fmt != "onnx"),
    reason=(
        "Only ONNX Runtime has parity evidence for Mask R-CNN's proposal, "
        "RoIAlign, variable-length detection, and full-image mask graph."
    ),
)
_add(
    "blocked",
    ("fcos",),
    ("detect",),
    ("executorch", "ncnn", "tflite", "coreml", "coreai"),
    reason=(
        "No runtime parity contract exists for FCOS dynamic anchor grids and "
        "variable padded spatial shapes in this format."
    ),
)
_add(
    "validated",
    ("deim",),
    ("detect",),
    ("onnx",),
    since="1.5",
    constraint="DETR query rows are aligned as an unordered set for parity",
)
_add(
    "validated",
    ("dfine", "ec", "rtdetr", "rtdetrv4"),
    ("detect",),
    ("openvino",),
    since="1.5",
    constraint="fixed export canvas",
)
_add(
    "available",
    ("deim",),
    ("detect",),
    ("openvino",),
    reason=(
        "The trained artifact reaches the elementwise tolerance, but its "
        "input signal is only 17.9x the conversion error; validation requires "
        "more than 20x."
    ),
)
_add(
    "available",
    ("deimv2",),
    ("detect",),
    ("openvino",),
    reason=(
        "After Hungarian query alignment, only 42.3% of scores meet the "
        "converted-runtime tolerance."
    ),
)
_add(
    "available",
    ("rtdetrv2",),
    ("detect",),
    ("openvino",),
    reason=(
        "After Hungarian query alignment, only 93.94% of trained raw elements "
        "meet the converted-runtime tolerance."
    ),
)
_add(
    "available",
    ("deimv2",),
    ("detect",),
    ("onnx",),
    reason=(
        "After Hungarian query alignment, only 43.7% of score values meet "
        "tolerance because ONNX top-k selects a different query set."
    ),
)
_add(
    "validated",
    ("rtdetrv2", "rtdetrv4"),
    ("detect",),
    ("onnx",),
    since="1.5",
    constraint=(
        "fixed export canvas; same-device CPU raw parity after one shared "
        "unordered-query permutation; published Apache-2.0 trained checkpoint "
        "covered by non-square public predict parity"
    ),
)
_add(
    "validated",
    ("rtdetrv2",),
    ("obb",),
    ("onnx", "torchscript"),
    reason=(
        "The official Apache-2.0 RT-DETRv2-N OBB checkpoint is covered by "
        "artifact reload, task metadata, raw five-coordinate output parity, "
        "and non-square public OBB prediction parity."
    ),
    since="1.5",
    constraint="FP32, batch 1, fixed 1024x1024 input canvas",
)
_add(
    "validated",
    ("riodetrv2",),
    ("obb",),
    ("onnx", "torchscript"),
    reason=(
        "The official Apache-2.0 RiO-DETRv2-N DOTA and DIOR-R checkpoints are "
        "covered by artifact reload, task metadata, raw five-coordinate output "
        "parity, and non-square public OBB prediction parity."
    ),
    since="1.7",
    constraint="FP32, batch 1, fixed square canvas (1024 DOTA, 800 DIOR-R)",
)
_add(
    "available",
    ("rtdetrv2",),
    ("obb",),
    ("openvino",),
    reason=(
        "The official N checkpoint exports, reloads, and preserves the top "
        "public OBB within 0.041 pixels, but the complete decoder query set "
        "does not meet raw-output parity after matching."
    ),
    since="1.5",
    constraint=(
        "OpenVINO 2026.2 CPU, FP32, batch 1, fixed 1024x1024 input canvas; "
        "export the ONNX intermediate on CPU"
    ),
)
_add(
    "available",
    ("rtdetrv2",),
    ("obb",),
    ("tensorrt",),
    reason=(
        "The official N checkpoint builds, reloads, and preserves the top "
        "public OBB within 0.057 pixels, but matched raw queries still drift "
        "by up to 0.078 in logits and 0.034 in normalized box coordinates."
    ),
    since="1.5",
    constraint=(
        "TensorRT 10.16 FP32 on RTX 5070 Ti, batch 1, fixed 1024x1024 input "
        "canvas; export the ONNX intermediate on CPU"
    ),
)
_add(
    "validated",
    ("dfine",),
    ("segment",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "validated",
    ("dfine",),
    ("segment",),
    ("openvino",),
    since="1.5",
    constraint="fixed export canvas",
)
_add(
    "validated",
    ("ec",),
    ("pose", "segment"),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed 640x640 input",
)
_add(
    "validated",
    ("ec",),
    ("segment",),
    ("openvino",),
    since="1.5",
    constraint="fixed 640x640 input",
)
_add(
    "available",
    ("ec",),
    ("pose",),
    ("openvino",),
    reason=(
        "Raw parity passes after Hungarian query alignment, but trained public "
        "boxes fall to 0.916 matched IoU."
    ),
)
_add(
    "validated",
    ("rfdetr",),
    ("segment", "pose", "obb"),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed task-native input resolution",
)
_add(
    "available",
    ("rfdetr",),
    ("segment", "pose", "obb"),
    ("openvino",),
    reason=(
        "After Hungarian query alignment, measured converted-runtime element "
        "match rates remain below validation: trained segment 69.0%, trained "
        "pose 72.75%, and input-sensitive OBB 91.25%."
    ),
)
_add(
    "validated",
    ("dinov2",),
    ("semantic",),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed 518x518 input",
)
_add(
    "validated",
    ("fcn",),
    ("semantic",),
    ("onnx", "torchscript"),
    reason=(
        "Both official trained checkpoints have two-input raw-logit parity, "
        "metadata reload, and public semantic-mask parity coverage."
    ),
    since="1.5",
    constraint="FP32, batch 1, fixed square input divisible by 8",
)
_add(
    "validated",
    ("fcn",),
    ("semantic",),
    ("openvino",),
    reason=(
        "Both official trained checkpoints have OpenVINO CPU FP32 raw-logit "
        "parity, input-sensitivity, metadata, and public-mask coverage."
    ),
    since="1.5",
    constraint="OpenVINO 2026.2 CPU FP32, batch 1, fixed square input divisible by 8",
)
_add(
    "validated",
    ("fcn",),
    ("semantic",),
    ("tensorrt",),
    reason=(
        "Both official trained checkpoints have TensorRT FP32 raw-logit "
        "parity, input-sensitivity, metadata, and public-mask coverage."
    ),
    since="1.5",
    constraint=("TensorRT 10.16 FP32, batch 1, fixed square input divisible by 8"),
)
_add(
    "blocked",
    ("fcn",),
    ("semantic",),
    ("executorch", "ncnn", "tflite", "coreai"),
    reason=(
        "This runtime has no parity-valid FCN artifact yet; only ONNX, "
        "TorchScript, TensorRT, and OpenVINO were assessed for this port."
    ),
)
_add(
    "blocked",
    ("fcn",),
    ("semantic",),
    ("coreml",),
    reason="The CoreML wrapper does not implement the dense semantic-logits contract.",
)
_add(
    "validated",
    ("segformer",),
    ("semantic",),
    ("onnx", "torchscript"),
    since="1.5",
    constraint="fixed square input divisible by 32",
)
_add(
    "validated",
    ("segformer",),
    ("semantic",),
    ("openvino",),
    since="1.5",
    constraint="fixed square input divisible by 32",
)
_add(
    "validated",
    ("segformer",),
    ("semantic",),
    ("tensorrt",),
    reason=(
        "A deterministic input-sensitive b0 fixture covers TensorRT 10.16 "
        "FP32 conversion, artifact reload, two-image raw-logit parity with a "
        "20x signal/error guard, metadata, and public semantic-mask parity "
        "above 95% pixel agreement."
    ),
    since="1.5",
    constraint=("TensorRT 10.16 FP32, batch 1, fixed square input divisible by 32"),
)
_add(
    "blocked",
    ("segformer",),
    ("semantic",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 emits a flatbuffer, but LiteRT 2.1.2 cannot prepare its "
        "attention reshape (1024 input elements versus 256 output elements)."
    ),
)
_add(
    "blocked",
    ("segformer",),
    ("semantic",),
    ("ncnn",),
    reason=(
        "PNNX leaves unsupported pnnx.Expression nodes in the SegFormer graph; "
        "the generated NCNN network reports 'network graph not ready' and has "
        "no runnable input blob."
    ),
)
_add(
    "validated",
    ("dinov2",),
    ("classify",),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed 224x224 input",
)
_add(
    "validated",
    ("eomt",),
    ("semantic",),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed 512x512 input",
)
_add(
    "validated",
    ("lingbotvision",),
    ("semantic",),
    ("onnx", "torchscript"),
    since="1.4",
    constraint="fixed 512x512 input",
)
_add(
    "blocked",
    ("fomo",),
    ("point",),
    ("coreml",),
    reason="The CoreML wrapper does not implement the raw point-heatmap contract.",
)
_add(
    "validated",
    ("fomo",),
    ("point",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "validated",
    ("depth_anything",),
    ("depth",),
    ("onnx", "torchscript"),
    since="1.4",
)
_add(
    "validated",
    ("midas",),
    ("depth",),
    ("onnx", "torchscript", "tensorrt", "openvino"),
    reason=(
        "The official Small and DPT-Large checkpoints cover opset-17 "
        "conversion, artifact reload, and two-image public depth-map parity "
        "above 46 dB PSNR with a signal/error margin above 10,000x across all "
        "four runtimes."
    ),
    constraint=(
        "FP32, batch 1, fixed square canvas: 256 for s and 384 for l; backend "
        "inference follows the ADR 0006 stretch-resize contract; TensorRT "
        "10.16 engines target the build GPU and OpenVINO evidence uses 2026.2"
    ),
)
_add(
    "validated",
    ("depth_anything",),
    ("depth",),
    ("openvino",),
    since="1.5",
    constraint="fixed input resolution divisible by 14",
)
_add(
    "validated",
    ("depth_anything",),
    ("depth",),
    ("tensorrt",),
    since="1.5",
    constraint="FP32 with a fixed input resolution divisible by 14",
)
_add(
    "blocked",
    ("depth_anything",),
    ("depth",),
    ("ncnn",),
    reason=(
        "PNNX 20260526 reports unsupported batch-index reshapes in the DINOv2 "
        "transformer graph; the produced NCNN artifact fails numeric parity."
    ),
)
_add(
    "validated",
    ("mobilenetv4", "convnext", "efficientnetv2", "resnet"),
    ("classify",),
    ("tflite",),
    since="1.4",
)
_add(
    "validated",
    ("mobilenetv4", "convnext", "efficientnetv2", "resnet"),
    ("classify",),
    ("ncnn",),
    since="1.4",
    constraint=(
        "PNNX/NCNN 20260526 CPU FP32 at the family-native input resolution; "
        "two-input raw parity, factory reload, metadata, and public predict "
        "parity"
    ),
)
_add(
    "blocked",
    ("fomo",),
    ("point",),
    ("tflite",),
    reason=(
        "LiteRT 2.1.2 cannot invoke the onnx2tf 2.6.7 graph because a "
        "DEPTHWISE_CONV_2D reports 16 filter channels versus zero input channels."
    ),
)
_add(
    "validated",
    ("pidnet",),
    ("semantic",),
    ("tflite",),
    since="1.4",
)
_add(
    "blocked",
    ("eomt", "lingbotvision"),
    ("semantic",),
    ("ncnn", "tflite"),
    reason=(
        "The dense-logits runtime contract is implemented, but this transformer "
        "graph has not produced a parity-valid edge-runtime artifact."
    ),
)
_add(
    "blocked",
    ("dinov2", "eomt", "pidnet", "lingbotvision"),
    ("semantic",),
    ("coreml",),
    reason="The CoreML wrapper does not implement the dense semantic-logits contract.",
)


_add(
    "validated",
    ("yolo1", "yolo2", "yolo3", "yolo4"),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed family-native canvases (YOLO1 448, YOLO2 608, YOLO3 416, "
        "YOLO4 608); representative published trained checkpoints are covered "
        "on Apple hardware by direct named-output parity with a 3e-04 "
        "tolerance and a 100x input-sensitivity margin; Core AI graph "
        "preparation exactly folds Darknet inference batch normalization into "
        "the preceding convolutions because Core AI 0.4.1 does not preserve "
        "Darknet's epsilon-after-square-root formula"
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("obb",),
    ("onnx", "torchscript", "openvino", "tensorrt"),
    since="1.5",
    constraint=(
        "fixed 1024x1024 export canvas, batch 1, FP32; the official "
        "YOLO-NAS-R-S DOTA2 checkpoint exports, reloads through the factory, "
        "and its public OBB results match native inference on the same image "
        "(max xywhr delta 6e-08 px TorchScript, 0.06 px ONNX, 0.04 px "
        "OpenVINO, 0.18 px TensorRT; max score delta 0.0, 1.0e-03, 1.5e-03 "
        "and 1.6e-03 respectively). Rotated NMS is not embedded in the graph"
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed 96x96 export canvas with pre-shaped canonical RGB tensors; a "
        "deterministic, license-clean synthetic "
        "YOLO-NAS-S state is covered on Apple hardware by direct named-output "
        "parity with a 3e-04 tolerance and a 100x input-sensitivity margin; "
        "the state receives 12 native training steps and a 20x regression-head "
        "scale to make both exported outputs non-degenerate; this validates "
        "conversion, not detection accuracy, raw-image preprocessing, or "
        "native-640 behavior, and does not convert restricted official weights"
    ),
)
_add(
    "available",
    ("dinov2",),
    ("classify",),
    ("coreai",),
    reason=(
        "Conversion has been measured, but the LibreDINOv2 classification "
        "checkpoint is not publicly downloadable for a reproducible trained-"
        "weight Core AI parity gate."
    ),
)
_add(
    "validated",
    (
        "deim",
        "deimv2",
        "ec",
        "picodet",
        "rtdetr",
        "rtdetrv2",
        "rtdetrv4",
        "rtmdet",
        "yolo9_e2e",
        "yolox",
    ),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; a representative published trained checkpoint "
        "for each family is covered on Apple hardware by direct named-output "
        "parity with a 3e-04 tolerance and a 100x input-sensitivity margin; "
        "RT-DETRv2 permits one shared whole-query permutation across its box "
        "and logit outputs because DETR query rows are an unordered set"
    ),
)
_add(
    "validated",
    ("convnext", "efficientnetv2", "mobilenetv4", "resnet"),
    ("classify",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; a representative published trained ImageNet "
        "checkpoint for each family is covered on Apple hardware by direct "
        "named-output parity with a 3e-04 tolerance and a 100x "
        "input-sensitivity margin"
    ),
)
_add(
    "validated",
    ("depth_anything", "zipdepth"),
    ("depth",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; permissively licensed trained checkpoints are "
        "covered on Apple hardware by direct named-output parity with a "
        "3e-04 tolerance and a 100x input-sensitivity margin"
    ),
)
_add(
    "validated",
    ("moge2",),
    ("normal",),
    ("onnx",),
    since="1.5",
    constraint=(
        "fixed square batch-1 export canvas divisible by 14; exported inference "
        "rejects non-square sources rather than stretching image-plane geometry; "
        "the official MIT ViT-S/B/L normal checkpoints are covered by FP32 "
        "same-canvas native-versus-ONNX angular parity below 0.1 degree"
    ),
)
_add(
    "blocked",
    ("dinov2",),
    ("semantic",),
    ("ncnn",),
    reason=(
        "PNNX 20260526 cannot lower the DINOv2 attention graph's batch-axis "
        "broadcasts and leaves an unsupported pnnx.Expression node."
    ),
)
_add(
    "blocked",
    ("dinov2",),
    ("semantic",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 flatbuffer-direct lowering cannot lower the backbone's "
        "cubic Resize because its input C/H/W signature remains dynamic."
    ),
)
_add(
    "validated",
    ("moge2",),
    ("normal",),
    ("torchscript", "openvino", "tensorrt"),
    reason=(
        "A deterministic input-sensitive ViT-S fixture covers conversion, "
        "artifact reload, two-image raw normal-map parity with a 20x "
        "signal/error guard, metadata, unit-vector normalization, and public "
        "angular parity below 0.1 degree."
    ),
    since="1.5",
    constraint=(
        "FP32, batch 1, fixed square input divisible by 14; TensorRT evidence "
        "uses TensorRT 10.16 and OpenVINO evidence uses OpenVINO 2026.2"
    ),
)
_add(
    "validated",
    ("moge3",),
    ("normal",),
    ("onnx", "torchscript"),
    reason=(
        "tests/unit/test_moge3.py covers same-canvas native-versus-exported "
        "angular normal parity, artifact reload, metadata, and unit-vector "
        "normalization on a random-init fixture; the normal graph is the "
        "MoGe-2 export contract and contains no sparse refiner."
    ),
    since="1.7",
    constraint=(
        "FP32, batch 1, fixed square input divisible by 14; exported "
        "inference rejects non-square sources"
    ),
)
_add(
    "blocked",
    ("moge3",),
    ("normal",),
    ("executorch", "ncnn", "tflite", "paddle", "mnn", "rknn", "coreml", "coreai"),
    reason=(
        "MoGe-3 normal export is validated for ONNX and TorchScript only; "
        "this runtime has not been run for the family."
    ),
)
_add(
    "blocked",
    ("moge3",),
    ("depth",),
    EXPORT_FORMATS,
    reason=(
        "The sparse refiner builds data-dependent voxel sets and camera "
        "recovery runs a CPU least-squares solve, so the depth graph is not "
        "capturable. Use the native .pt model for depth."
    ),
)
_add(
    "validated",
    ("nafnet", "realesrgan"),
    ("restore",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; permissively licensed trained restoration "
        "checkpoints are covered on Apple hardware by direct named-output "
        "parity with a 3e-04 tolerance and a 100x input-sensitivity margin"
    ),
)
_add(
    "validated",
    ("clip", "siglip2"),
    ("classify",),
    ("coreai",),
    since="1.5",
    constraint=(
        "frozen class set and fixed export canvas; permissively licensed "
        "trained checkpoints are covered on Apple hardware by direct named-"
        "output parity with a 3e-04 tolerance and a 100x input-sensitivity "
        "margin"
    ),
)

# HOW THE Core AI NUMBERS BELOW WERE MEASURED, and why an earlier set of them
# was withdrawn.
#
# Every figure is the worst relative error against a reference graph, with each
# artifact fed the input ITS OWN contract expects and reported alongside the
# reference's own input-sensitivity. Published trained weights are used where a
# permissive checkpoint exists. The FOMO, YOLO-NAS, and YOLO9-P2 entries state
# their license-clean synthetic or transfer fixture explicitly and make no
# accuracy claim.
#
# All three qualifiers were learned the hard way.
#
# Non-degenerate weights: a randomly initialised detection head emits nearly the same
# tensor whatever it is shown, because the constant anchor grid dominates its
# output. Measured on the ONNX reference between two very different probes,
# random-init yolox moved by 1.5e-09 and rtmdet by 8.9e-12. Agreement at 1e-08
# against a reference that moves 1.5e-09 certifies nothing. picodet was caught
# because it hit exactly zero and was recorded as blocked; its neighbours
# failed the same way by degrees and were recorded as validated.
#
# Input contract: _wrap_for_family wraps some families in a preprocessing
# module for the Apple formats, so a Core AI graph takes canonical RGB[0,1] and
# converts internally (YOLOX scales by 255 and swaps to BGR, RF-DETR applies
# ImageNet normalization). The ONNX exporter applies no such wrapper. Handing
# both the same tensor compares two different functions and reads ~0.5 however
# correct the conversion is.
#
# Sensitivity margin: a result counts only if parity is at least 100x below
# how far the reference itself moves between probes. Otherwise the honest
# answer is that the measurement cannot support a verdict.
_add(
    "validated",
    ("dfine",),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; trained LibreDFINEn weights are covered on "
        "macOS 27 by direct named-output parity with a 3e-04 tolerance and "
        "a 100x input-sensitivity margin"
    ),
)


_add(
    "validated",
    ("yolo9",),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; trained LibreYOLO9t weights are covered on "
        "macOS 27 by direct named-output parity with a 3e-04 tolerance and "
        "a 100x input-sensitivity margin"
    ),
)
_add(
    "validated",
    ("yolo9_p2",),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed 640x640 export canvas; a deterministic YOLO9-P2-T model "
        "initialized from the SHA-256-pinned, permissively licensed trained "
        "LibreYOLO9t checkpoint is covered on Apple hardware by direct "
        "named-output parity with a 3e-04 tolerance and a 100x "
        "input-sensitivity margin; this validates conversion, not P2 task "
        "accuracy, and does not depend on the restricted VisDrone "
        "research-preview checkpoint"
    ),
)
_add(
    "validated",
    ("yolo7",),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed 640x640 export canvas; trained LibreYOLO7b weights are covered on "
        "Apple hardware by direct named-output parity with a 3e-04 tolerance "
        "and a 100x input-sensitivity margin; the export decoder uses direct "
        "arange grids because Core AI 0.4.1 mislowers the equivalent "
        "cumulative-sum expression"
    ),
)
_add(
    "blocked",
    ("birefnet",),
    ("matte",),
    ("coreai",),
    reason=(
        "The decoder needs torchvision deform_conv2d, which the Core AI "
        "converter cannot lower ('unable to handle call function op: "
        "deform_conv2d.default'). The same operator already blocks the NCNN "
        "path. An encoder-only contract is the realistic route, matching the "
        "seam the CUDA graph work used."
    ),
)
_add(
    "validated",
    ("rfdetr",),
    ("detect",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed export canvas; trained LibreRFDETRn weights are covered on "
        "macOS 27 against the graph the exporter itself prepares, using "
        "direct named-output parity with a 3e-04 tolerance and a 100x "
        "input-sensitivity margin. "
        "Conversion needed _rebake_rfdetr_pos_embed in export/coreai.py: the "
        "backbone bakes its position embedding for its configured 384 canvas, "
        "so exporting at any other size left an antialiased bicubic in the "
        "graph and the converter has no lowering for "
        "aten._upsample_bicubic2d_aa. The rebake re-runs the model's OWN "
        "baking path for the actual canvas, so the interpolation happens "
        "eagerly, outside the graph, computing exactly what it computed "
        "before. "
        "NOTE the reference. This family is verified against the exporter's "
        "prepared graph, not against ONNX, and the difference is not a "
        "detail: at a 640 canvas the rfdetr ONNX artifact disagrees with that "
        "same prepared graph by 9.3e-01. Core AI's rebake preserves the "
        "antialiased resize the eager "
        "model performs, whereas the ONNX path disables antialiasing (the "
        "model checks torch.onnx.is_in_onnx_export). Which artifact is right "
        "is an ONNX question and is not settled here, but ONNX cannot be used "
        "as the reference for this family at a non-native canvas."
    ),
)
_add(
    "blocked",
    ("swinir",),
    ("restore",),
    ("coreai",),
    reason=(
        "The export process DIES rather than hangs, and the kill point moves "
        "between runs, which is the signature of memory exhaustion rather "
        "than a stuck loop. One run reached 'Step 3/3: Optimizing and writing "
        "the asset' before stopping; a later run of the same graph at the "
        "same 128 canvas died inside to_coreai() before returning, in both "
        "cases with a leaked-semaphore warning and no traceback. Window "
        "attention unrolls into a very large number of small ops, so the "
        "converter's peak memory is the prime suspect on a 16 GB machine. "
        "Next steps: watch RSS during conversion, try the smallest available "
        "size at a 64 canvas, and check the system log for a memory kill. Do "
        "NOT assume optimize() is at fault; an earlier note said so on the "
        "strength of a single run and the second run contradicted it."
    ),
)

_add(
    "validated",
    ("pidnet", "lingbotvision"),
    ("semantic",),
    ("coreai",),
    since="1.5",
    constraint=(
        "fixed family-native canvases (PIDNet 1024, LingBotVision 512); trained "
        "LibrePIDNets-sem and LibreLingBotVisions-sem checkpoints are covered "
        "on Apple hardware by direct named-output parity with a 3e-04 "
        "tolerance and a 100x input-sensitivity margin; exported backends "
        "already implement the shared dense-logit resize and argmax contract"
    ),
)
_add(
    "blocked",
    ("segformer",),
    ("semantic",),
    ("coreai",),
    reason=(
        "The SegFormer Core AI capture path has not been assessed. Its published "
        "weights are non-commercial regardless of export format."
    ),
)
_add(
    "blocked",
    ("eomt",),
    ("semantic",),
    ("coreai",),
    reason=(
        "torch.export refuses the graph: GuardOnDataDependentSymNode, "
        "'Could not guard on data-dependent expression Eq(u0, 1)'. Something "
        "in the mask path reads a value off a tensor and branches on it, "
        "which becomes an unbacked symbol with no hint the tracer can "
        "resolve. This is a real capture failure, not a missing operator and "
        "not the task gate: it was measured with the gate open. Fixing it "
        "means finding the host read and making the shape static for a fixed "
        "export canvas, the same shape of fix as the rfdetr torch._assert."
    ),
)
_add(
    "validated",
    ("fomo",),
    ("point",),
    ("coreai",),
    since="1.5",
    constraint=(
        "native 96 canvas; a deterministic model state trained from scratch "
        "for eight steps on synthetic tensors is covered on Apple hardware "
        "by direct named-output parity with a 3e-04 tolerance and a 100x "
        "input-sensitivity margin; this validates conversion and the existing "
        "heatmap contract, not point-localization accuracy"
    ),
)
_add(
    "blocked",
    ("l2cs",),
    ("gaze",),
    ("coreai",),
    reason=(
        "The model itself refuses: 'LibreL2CS export to coreai is not "
        "implemented. The gaze export contract supports ONNX, TorchScript, "
        "ExecuTorch, TensorRT, and OpenVINO only.' That is a model-side "
        "decision, unchanged by opening the support gate, so nothing about "
        "Core AI is being tested here."
    ),
)
_add(
    "validated",
    ("l2cs",),
    ("gaze",),
    ("openvino",),
    reason=(
        "A deterministic input-sensitive fixture covers conversion, artifact "
        "reload, two-head raw-logit parity, metadata, and public gaze-angle parity."
    ),
    since="1.5",
    constraint="OpenVINO 2026.2 CPU FP32, batch 1, fixed 448x448 face-crop input",
)
_add(
    "validated",
    ("l2cs",),
    ("gaze",),
    ("tensorrt",),
    reason=(
        "A deterministic input-sensitive fixture covers conversion, artifact "
        "reload, two-head raw-logit parity, metadata, and public gaze-angle parity."
    ),
    since="1.5",
    constraint="TensorRT 10.16 FP32, batch 1, fixed 448x448 face-crop input",
)
_add(
    "blocked",
    ("depth_anything3",),
    ("depth",),
    ("coreai",),
    reason=(
        "The model raises NotImplementedError for every format: depth export "
        "is out of scope per ADR 0006, the depth task contract. Depth Anything "
        "V2 exports and validates at 5.2e-06, so this is specific to the V3 "
        "family and not a Core AI limitation."
    ),
)

_add(
    "validated",
    ("deeplabv3",),
    ("semantic",),
    ("onnx", "torchscript"),
    reason=(
        "All three official checkpoints preserve 100% raw argmax and public-"
        "mask agreement. TorchScript logits are bit-exact; ONNX Runtime FP32 "
        "maximum absolute logit error is at most 3.07e-05."
    ),
    since="1.5",
    constraint="FP32, batch 1, fixed 520x520 input",
)
_add(
    "validated",
    ("deeplabv3",),
    ("semantic",),
    ("openvino",),
    reason=(
        "All three official checkpoints export, reload, and preserve at least "
        "99.987% public-mask agreement through the default CPU runtime."
    ),
    since="1.5",
    constraint=(
        "OpenVINO 2026.2 FP32 IR, CPU default inference precision, batch 1, "
        "fixed 520x520 input"
    ),
)
_add(
    "validated",
    ("hrnet",),
    ("pose",),
    ("onnx", "torchscript"),
    reason=(
        "Both official converted W32 and W48 checkpoints have raw-heatmap and "
        "public decoded-keypoint parity in tests/e2e/test_hrnet_exports.py; "
        "tests/unit/test_hrnet_parity.py separately proves the native graph, "
        "affine crop, normalization, flip-shift, and decoder against the pinned "
        "MIT upstream implementation."
    ),
    since="1.5",
    constraint=(
        "PyTorch 2.11, ONNX 1.20.1 / ONNX Runtime 1.26 or TorchScript, CPU "
        "FP32, batch 1, fixed checkpoint-native 256x192 (W32) or 384x288 "
        "(W48) person-crop input; the full-image person detector is not in-graph"
    ),
)
_add(
    "validated",
    ("hrnet",),
    ("pose",),
    ("openvino",),
    reason=(
        "Both official converted W32 and W48 checkpoints have conversion, "
        "artifact reload, raw-heatmap parity within 3e-3, metadata, and public "
        "decoded-keypoint parity in tests/e2e/test_hrnet_exports.py."
    ),
    since="1.5",
    constraint=(
        "OpenVINO 2026.2.1 CPU FP32, batch 1, fixed checkpoint-native 256x192 "
        "(W32) or 384x288 (W48) person-crop input; the full-image person "
        "detector is not in-graph"
    ),
)
_add(
    "validated",
    ("deeplabv3",),
    ("semantic",),
    ("tensorrt",),
    reason=(
        "All three official checkpoints build and reload with at least "
        "99.985% public-mask agreement on the validated GPU."
    ),
    since="1.5",
    constraint=("TensorRT 10.16 FP32, RTX 5070 Ti, batch 1, fixed 520x520 input"),
)
_add(
    "validated",
    ("hrnet",),
    ("pose",),
    ("tensorrt",),
    reason=(
        "Both official converted W32 and W48 checkpoints have conversion, "
        "artifact reload, raw-heatmap parity within 3e-3, metadata, and public "
        "decoded-keypoint parity in tests/e2e/test_hrnet_exports.py."
    ),
    since="1.5",
    constraint=(
        "TensorRT 10.16.1.11, CUDA 12.8, RTX 5070 Ti, FP32, batch 1, fixed "
        "checkpoint-native 256x192 (W32) or 384x288 (W48) person-crop input; "
        "the full-image person detector is not in-graph"
    ),
)


_add(
    "validated",
    ("dekr",),
    ("pose",),
    ("onnx", "torchscript"),
    reason=(
        "The released DEKR-W32-NO-DC checkpoint exports and reloads with raw "
        "heatmap-logit and offset parity at batch 1 and batch 2, and the "
        "reloaded raw outputs decode to the same poses as eager PyTorch, in "
        "tests/e2e/test_dekr_pose.py. TorchScript is exact; ONNX agrees within "
        "1e-2 absolute on the unbounded offset regressions. "
        "weights/parity_dekr.py separately proves the native graph against the "
        "pinned Apache-2.0 upstream implementation at max_abs_diff == 0."
    ),
    since="1.5",
    constraint=(
        "PyTorch 2.11, ONNX 1.20.1 / ONNX Runtime 1.26 or TorchScript, CPU "
        "FP32, batch 1 and 2 through a dynamic batch axis, fixed 640x640 "
        "spatial input; the graph emits only raw heatmap_logits and offsets, "
        "so peak finding, pose NMS and the derived-box adapter stay out-of-graph"
    ),
)
_add(
    "validated",
    ("dekr",),
    ("pose",),
    ("openvino",),
    reason=(
        "The released checkpoint converts to OpenVINO IR and reloads through "
        "the public backend route."
    ),
    since="1.5",
    constraint=(
        "OpenVINO 2026.2.1 CPU FP32, batch 1, fixed 640x640 input; decode and "
        "pose NMS stay out-of-graph"
    ),
)
_add(
    "validated",
    ("dekr",),
    ("pose",),
    ("tensorrt",),
    reason=(
        "The released checkpoint builds and executes a TensorRT engine on the "
        "validated GPU; parser success alone was not accepted as support."
    ),
    since="1.5",
    constraint=(
        "TensorRT 10.16.1.11, CUDA 12.8, RTX 5070 Ti, FP32, batch 1, fixed "
        "640x640 input; decode and pose NMS stay out-of-graph"
    ),
)


_TASK_BLOCKS = {
    "albedo": "Albedo export does not yet have a linear-RGB backend runtime contract.",
    "act": (
        "Action-chunk export is blocked until the policy sampling loop has an "
        "exportable graph and backend runtime contract (ADR 0028)."
    ),
    "generate": (
        "Image generation export is blocked until a sampling loop over several "
        "networks has an exportable graph and backend runtime contract (ADR 0030)."
    ),
    "generate3d": (
        "3D generation export is blocked until a sampling loop over several "
        "networks has an exportable graph and backend runtime contract (ADR 0031)."
    ),
    "detect3d": (
        "3D detection export is blocked until its graph outputs, camera metadata, "
        "and backend runtime contract are defined."
    ),
    "flow": (
        "Optical flow export is blocked until the two-input graph (image1, "
        "image2) and the backend vector-rescale contract of ADR 0032 are "
        "implemented for a family."
    ),
    "ocr": (
        "OCR uses two networks for detection and recognition with dynamic "
        "per-region cropping, so it does not fit the single-graph export contract."
    ),
    "point": (
        "This family is not wired to the shared point heatmap and backend "
        "peak-decoding export contract."
    ),
    "semantic": (
        "This family is not wired to the shared dense-logits and backend "
        "argmax semantic export contract."
    ),
    "mesh": (
        "Body-mesh export is blocked until its graph outputs, metadata, and "
        "backend runtime contract are defined."
    ),
    "normal": (
        "This family is not wired to the fixed-canvas dense unit-normal "
        "export and backend renormalization contract."
    ),
    "panoptic": "Panoptic export does not yet have a backend runtime contract.",
    "gaze": (
        "This family is not wired to the shared two-head logits and backend "
        "expectation-decoding gaze export contract."
    ),
}

_FAMILY_BLOCKS = {
    "video_depth_anything": (
        "Video Depth Anything rejects export for every format: the network "
        "reads a 32-frame window and windows are stitched on the host, which "
        "the single-image depth export contract does not describe."
    ),
    "marigold_v2": "Marigold V2 quantized diffusion export has not been integrated or validated.",
    "speridlabs_irisz": (
        "Speridlabs Iris-Z export is blocked for both tasks: image generation and "
        "depth run a multi-step sampling loop over a transformer and the FLUX.2 "
        "VAE, which has no exportable graph and backend runtime contract (ADR 0030)."
    ),
    "depth_anything3": (
        "Depth Anything 3 currently rejects export for every format; its "
        "depth graph has not been added to the exported-runtime contract."
    ),
    "domedetr": (
        "Dome-DETR rejects export for every format. PAQI sets the query count "
        "per image, so a traced graph is only valid for the image it was traced "
        "on; a static formulation would need the greedy density-adaptive NMS "
        "unrolled over all 250-1500 candidates. Use D-FINE for an exportable "
        "DETR."
    ),
    "eomt": "EoMT instance and panoptic export do not yet have runtime parsing.",
    "l2cs": (
        "The L2CS gaze export contract supports ONNX, TorchScript, ExecuTorch, "
        "TensorRT, and OpenVINO only."
    ),
    "hrnet": (
        "The HRNet person-crop pose-head export contract supports ONNX, "
        "TorchScript, OpenVINO, and TensorRT only."
    ),
    "dekr": (
        "The raw two-output DEKR pose graph export contract supports ONNX, "
        "TorchScript, OpenVINO, and TensorRT only."
    ),
    "sam": "Promptable model export is out of scope for the v1 runtime contract.",
    "sam2": "Promptable model export is out of scope for the v1 runtime contract.",
    "edgetam": "Promptable model export is out of scope for the v1 runtime contract.",
    "sam3": "Promptable model export is out of scope for the v1 runtime contract.",
    "sam31": "Promptable model export is out of scope for the v1 runtime contract.",
    "mobilesam": "Promptable model export is out of scope for the v1 runtime contract.",
    "grounding_dino": "Open-vocabulary runtime export is out of scope for v1.",
    "owlv2": "Open-vocabulary runtime export is out of scope for v1.",
    "omdet_turbo": "Open-vocabulary runtime export is out of scope for v1.",
    "ov_deim": "Open-vocabulary runtime export is out of scope for v1.",
    "florence2": "Generative VLM export is out of scope for v1.",
    "kosmos2": "Generative VLM export is out of scope for v1.",
    "lfm2vl": "Generative VLM export is out of scope for v1.",
    "internvl3": "Generative VLM export is out of scope for v1.",
    "qwen3vl": "Generative VLM export is out of scope for v1.",
    "qwen3vl_embedding": "VLM embedding export is out of scope for v1.",
    "smolvlm2": "Generative VLM export is out of scope for v1.",
    "locateanything": "Generative VLM export is out of scope for v1.",
    "northmicrovision": "Generative VLM export is out of scope for v1.",
    "gemma4": "Generative VLM export is out of scope for v1.",
    "moondream": "Generative VLM export is out of scope for v1.",
    "sensenovavision": "Generative VLM export is out of scope for v1.",
    "libremodus": "Generative VLM export is out of scope for v1.",
    "wevisonevl": "Generative VLM export is out of scope for v1.",
    "deepseekocr": "Generative VLM export is out of scope for v1.",
    "paddleocrvl": "Generative VLM export is out of scope for v1.",
    "glmocr": "Generative VLM export is out of scope for v1.",
    "falconocr": "Generative VLM export is out of scope for v1.",
    "falconperception": "Generative VLM export is out of scope for v1.",
}

_NCNN_BLOCKS = {
    "deformable_detr": "Deformable DETR",
    "detr": "DETR",
    "dinodetr": "DINO-DETR",
    "dfine": "D-FINE",
    "lwdetr": "LW-DETR",
    "deim": "DEIM",
    "deimv2": "DEIMv2",
    "tinyformer": "TinyFormer",
    "rtdetr": "RT-DETR",
    "riodetrv2": "RiO-DETRv2",
    "rtdetrv2": "RT-DETRv2",
    "rtdetrv4": "RT-DETRv4",
    "rfdetr": "RF-DETR",
    "ec": "EC",
    "ppdoclayoutv3": "PP-DocLayoutV3",
}


_add(
    "validated",
    ("yolonas",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the graph is cut after "
        "the head convolutions and LibreYOLO decodes on the host. On a Metis "
        "M.2 Max, YOLO-NAS-s/m/l are within 0.5 mAP50-95 of FP32 on a paired "
        "1000-image COCO val2017 subset (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: s, m, l. axelera-devkit / axelera-rt 1.9.0, Linux "
        "x86_64 compile, INT8, batch 1, fixed square input"
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("pose",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; LibreYOLO windows the "
        "raw outputs, equalizes the keypoint branch and caps the class "
        "branch, and decodes boxes and keypoints on the host. On a Metis M.2 "
        "Max, YOLO-NAS-pose-m/l are within 1.0 keypoint mAP50-95 of FP32 on "
        "300 COCO val2017 person images (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: m, l; n and s are supported with a measured drop "
        "(SIZE_ACCURACY_NOTES, docs/axelera.md). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, "
        "batch 1, fixed square input"
    ),
)
_add(
    "validated",
    ("yolonas",),
    ("obb",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; LibreYOLO decodes the "
        "rotated boxes on the host. On a Metis M.2 Max, YOLO-NAS-R-s/m/l are "
        "within 1.0 mAP50-95 of FP32 on a paired 300-tile DOTA-v2.0 val "
        "subset (docs/axelera.md); calibration data must be passed."
    ),
    since="1.7",
    constraint=(
        "validated sizes: s, m, l (300 DOTA-v2.0 tiles). axelera-devkit / "
        "axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, fixed square "
        "input"
    ),
)
_add(
    "validated",
    ("resnet",),
    ("classify",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the whole classifier "
        "graph runs on the Metis AIPU and the host applies the softmax. On a "
        "Metis M.2 Max, ResNet-18/34/50/101 are within 1.0 top-1 point of "
        "FP32 on a paired 1000-image ImageNetV2 set when calibrated on "
        "images from their own domain (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: 18, 34, 50, 101, calibrated with in-domain images "
        "(data=; the coco128 fallback costs about 0.5 more top-1). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, "
        "batch 1, fixed 224 input"
    ),
)
_add(
    "validated",
    ("mobilenetv4",),
    ("classify",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; LibreYOLO equalizes and "
        "bounds MobileNetV4's residual streams before export, calibrates with "
        "histogram activation ranges, and the whole classifier runs on the "
        "Metis AIPU. On a Metis M.2 Max, MobileNetV4-m is within 1.0 top-1 "
        "point of FP32 on a paired 1000-image ImageNetV2 set when calibrated on "
        "images from its own domain (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: m; s and l are supported with a measured drop "
        "(SIZE_ACCURACY_NOTES, docs/axelera.md), "
        "calibrated with in-domain images (data=). axelera-devkit / axelera-rt "
        "1.9.0, Linux x86_64 compile, INT8, batch 1, fixed 224 input"
    ),
)
_add(
    "available",
    ("efficientnetv2",),
    ("classify",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the whole classifier runs "
        "on the Metis AIPU with histogram activation ranges. Supported, not "
        "validated: on a Metis M.2 Max EfficientNetV2-b0 keeps top-1 1.0-1.1 "
        "points below FP32 on a paired 1000-image ImageNetV2 set "
        "(docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "supported sizes: b0 (measured drop, SIZE_ACCURACY_NOTES); b1, b2 and b3 "
        "do not compile with SDK 1.9.0 (BLOCKED_SIZES). axelera-devkit / "
        "axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, fixed 224 input"
    ),
)
_add(
    "validated",
    ("vgg",),
    ("classify",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the whole classifier "
        "runs on the Metis AIPU, with the fully connected head expressed as "
        "convolutions (fc6 as one row convolution per row of the 7x7 map, an "
        "exact graph pass), and the host applies the softmax. On a Metis M.2 "
        "Max, VGG-16, VGG-19 and VGG-16-BN (histogram activation ranges) are "
        "within 1.0 top-1 point of FP32 on a paired 1000-image ImageNetV2 set; "
        "VGG-19-BN is supported with a larger measured drop (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: 16, 19, 16bn; supported sizes: 19bn (measured drop, "
        "SIZE_ACCURACY_NOTES); calibrated with in-domain images (data=). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, "
        "fixed 224 input"
    ),
)
_add(
    "validated",
    ("alexnet",),
    ("classify",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the whole classifier "
        "runs on the Metis AIPU (its fully connected head included) and the "
        "host applies the softmax. On a Metis M.2 Max, AlexNet is within 1.0 "
        "top-1 point of FP32 on a paired 1000-image ImageNetV2 set "
        "(docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: b; calibrated with in-domain images (data=). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, "
        "fixed 224 input"
    ),
)
_add(
    "available",
    ("quicksrnet",),
    ("restore",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the whole super-resolution "
        "graph runs on the Metis AIPU after its hidden channels are equalized "
        "through their Hardtanh bounds and its biases corrected for INT8 weight "
        "rounding. Supported, not validated: QuickSRNet-m2 keeps PSNR 0.46 dB "
        "below FP32 on 296 COCO crops, 2x from 128 px (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "supported sizes: m2 (measured drop, SIZE_ACCURACY_NOTES). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, "
        "fixed input canvas (smaller images are padded)"
    ),
)
_add(
    "validated",
    ("realesrgan",),
    ("restore",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain. Real-ESRGAN-x4 (RRDB) runs "
        "as three compiled parts with the trunk skip added on the host (its whole "
        "graph's constants overflow the Metis DDR pool) and keeps PSNR within "
        "0.06 dB of FP32 on 296 COCO crops, 4x from 64 px (the library export "
        "gives the same 22.44 dB as the first measurement). For the compact "
        "network (x4t) the convolution "
        "trunk runs on the Metis AIPU and the host applies the pixel shuffle and "
        "the nearest-upsampled input skip (layout restore_shuffle_skip_v1); it "
        "keeps PSNR 0.49 dB below FP32 (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: x4 (three compiled parts); supported sizes: x4t "
        "(measured drop, SIZE_ACCURACY_NOTES); x2 is not measured on a Metis yet "
        "(UNMEASURED_SIZES). axelera-devkit / axelera-rt 1.9.0, Linux x86_64 "
        "compile, INT8, batch 1, fixed 64x64 input canvas (smaller images are padded)"
    ),
)
_add(
    "available",
    ("midas",),
    ("depth",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the whole MiDaS-s graph "
        "runs on the Metis AIPU and the host resizes the relative inverse depth. "
        "Supported, not validated: the device depth agrees with FP32 at AbsRel "
        "0.069 (delta1 0.94) on 300 COCO val2017 images at 256 px "
        "(docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "supported sizes: s (measured drop, SIZE_ACCURACY_NOTES); l (DPT ViT-L) "
        "does not quantize with SDK 1.9.0 (BLOCKED_SIZES). axelera-devkit / "
        "axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, fixed input canvas (images are stretched to it)"
    ),
)
_add(
    "available",
    ("deeplabv3",),
    ("semantic",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the whole DeepLabV3 graph "
        "runs on the Metis AIPU after exact graph rewrites (the ASPP "
        "image-pooling branch becomes a broadcast add, oversized ASPP "
        "convolutions are split by channels, and for R50/R101 each dilated ASPP "
        "convolution runs as per-tap 1x1 convolutions plus zero-filled shifts), "
        "and the host upsamples the class logits. Supported, not validated: on a "
        "paired 300-image COCO val2017 set with VOC-21 masks mIoU stays below "
        "FP32 by 3.77 points (MobileNetV3), 5.08 (R50) and 20.93 (R101; "
        "docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "supported sizes: mv3, r50, r101 (measured drops, SIZE_ACCURACY_NOTES). "
        "axelera-devkit / axelera-rt 1.9.0, Linux "
        "x86_64 compile, INT8, batch 1, fixed 520x520 input"
    ),
)
_add(
    "available",
    ("rtmdet",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the graph stops after "
        "each level's box and class convolutions, CSPNeXt pointwise-conv inputs "
        "are equalized on the export copy (FP32 outputs unchanged), and "
        "LibreYOLO decodes on the host. Supported, not validated: on a Metis M.2 "
        "Max the mAP50-95 drops below FP32 are RTMDet-t 2.29, RTMDet-s 6.75, "
        "RTMDet-m 2.66 and RTMDet-l 1.87 on a paired 1000-image COCO val2017 "
        "subset (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "supported sizes: t, s, m, l (measured drop, SIZE_ACCURACY_NOTES); x does "
        "not compile (BLOCKED_SIZES). axelera-devkit / axelera-rt 1.9.0, Linux x86_64 "
        "compile, INT8, batch 1, fixed input size"
    ),
)
_add(
    "available",
    ("rtmdet",),
    ("segment",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the device returns each "
        "level's box, class and dynamic-kernel outputs plus the stride-8 mask "
        "feature, and LibreYOLO's own RTMDet-Ins postprocess (box decode, NMS, "
        "dynamic-kernel mask decode) runs on the host. Supported, not "
        "validated: on a Metis M.2 Max the mask / box mAP50-95 drops below FP32 "
        "are RTMDet-Ins-t 1.44 / 2.51, RTMDet-Ins-s 5.80 / 7.26, RTMDet-Ins-m "
        "0.79 / 3.96 and RTMDet-Ins-l 0.50 / 5.34 on a paired 1000-image COCO "
        "val2017 subset (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "supported sizes: t, s, m, l (measured drop, SIZE_ACCURACY_NOTES); x does "
        "not compile (BLOCKED_SIZES). axelera-devkit / "
        "axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, fixed input size"
    ),
)
_add(
    "validated",
    ("fcn",),
    ("semantic",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the whole FCN graph runs "
        "on the Metis AIPU and the host upsamples the class logits and takes the "
        "argmax. On a Metis M.2 Max, FCN-ResNet50/101 are within 1.0 mIoU "
        "point of FP32 on a paired 300-image COCO val2017 set with VOC-21 masks "
        "(docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: r50, r101. axelera-devkit / axelera-rt 1.9.0, Linux "
        "x86_64 compile, INT8, batch 1, fixed 520x520 input"
    ),
)
_add(
    "validated",
    ("yolox",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the graph is cut after "
        "the box, objectness and class convolutions and LibreYOLO decodes on "
        "the host. On a Metis M.2 Max, YOLOX-t, -s, -m, -l and -x are within "
        "0.8 mAP50-95 of FP32 on a paired 1000-image COCO val2017 subset "
        "(docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: t, s, m, l, x; n is supported with a measured drop "
        "(on-device results in docs/axelera.md). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, "
        "batch 1, fixed input size"
    ),
)
_add(
    "validated",
    ("yolo2",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the [region] head "
        "convolution is split into box, objectness and class convolutions, "
        "the [reorg] passthrough runs as an exact one-hot convolution, and "
        "LibreYOLO decodes the anchors (softmax over the classes) on the host. "
        "On a Metis M.2 Max, YOLOv2-tiny and YOLOv2 are within 1.0 mAP50-95 "
        "of FP32 on a paired 1000-image COCO val2017 subset (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: t, b. axelera-devkit / axelera-rt 1.9.0, Linux x86_64 "
        "compile, INT8, batch 1, fixed input size"
    ),
)
_add(
    "validated",
    ("yolo4",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; each head convolution "
        "is split into box, objectness and class convolutions and LibreYOLO "
        "decodes the anchors on the host. On a Metis M.2 Max, YOLOv4-tiny is "
        "within 0.6 mAP50-95 of FP32 on a paired 1000-image COCO val2017 "
        "subset (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: t; b is supported with a measured drop (on-device "
        "results in docs/axelera.md). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, "
        "batch 1, fixed input size"
    ),
)
_add(
    "validated",
    ("yolo3",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; as for YOLOv4, each head "
        "convolution is split into box, objectness and class convolutions and "
        "LibreYOLO decodes the anchors on the host. On a Metis M.2 Max, "
        "YOLOv3-tiny, YOLOv3 and YOLOv3-SPP are within 0.8 mAP50-95 of FP32 "
        "on a paired "
        "1000-image COCO val2017 subset (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: t, b, spp (on-device results in docs/axelera.md). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, "
        "batch 1, fixed input size"
    ),
)
_add(
    "validated",
    ("yolo7",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; RepConv branches and "
        "the implicit-knowledge terms fold into the convolutions, each head "
        "convolution is split into box, objectness and class convolutions, and "
        "LibreYOLO decodes the anchors on the host. On a Metis M.2 Max, YOLOv7 "
        "is within 0.2 mAP50-95 of FP32 on a paired 1000-image COCO val2017 "
        "subset (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: b (on-device results in docs/axelera.md). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, "
        "batch 1, fixed input size"
    ),
)
_add(
    "validated",
    ("ppyoloe",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; RepVGG blocks are fused, "
        "the graph is cut after the distribution and class convolutions and "
        "LibreYOLO decodes on the host like YOLO-NAS. On a Metis M.2 Max, "
        "PP-YOLOE-l is within 0.4 mAP50-95 of FP32 on a paired 1000-image COCO "
        "val2017 subset (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: l; s, m and x are supported with a measured drop "
        "(SIZE_ACCURACY_NOTES, docs/axelera.md). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, "
        "batch 1, fixed input size"
    ),
)
_add(
    "available",
    ("picodet",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the graph stops after "
        "each level's box-distribution and class convolutions, ESNet's channel "
        "shuffles become fixed permutation convolutions, and LibreYOLO decodes "
        "on the host. Supported, not validated: on a Metis M.2 Max against FP32 "
        "on a paired 1000-image COCO val2017 subset PicoDet-s loses 2.3, "
        "PicoDet-m 2.9 and PicoDet-l 1.7 mAP50-95 (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "supported sizes: s, m, l (measured drop, SIZE_ACCURACY_NOTES). "
        "axelera-devkit / "
        "axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, fixed input size"
    ),
)
_add(
    "blocked",
    ("l2cs",),
    ("gaze",),
    ("axelera",),
    reason=(
        "L2CS's published checkpoint is trained on Gaze360, whose licence forbids "
        "redistributing models trained on it: LibreYOLO cannot mirror it, only "
        "fetch it from the authors, so no Axelera export is measured or shipped "
        "for it (non-redistributable weights; non-commercial ones would be "
        "fine)."
    ),
)
_add(
    "validated",
    ("ssd",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain in two parts with conv4_3's "
        "L2Norm on the host between them (no Metis operator for a per-pixel "
        "division by the channel norm): part 1 runs the VGG trunk and the level "
        "1-5 heads, part 2 the level-0 heads, and LibreYOLO decodes the 8,732 "
        "default boxes on the host. On a Metis M.2 Max, SSD300 is within 0.1 "
        "mAP50-95 of FP32 on a paired 1000-image COCO val2017 subset "
        "(docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: 300 (two .axm parts + metadata parts; keep them together). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, "
        "fixed 300x300 input"
    ),
)
_add(
    "available",
    ("retinanet",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain on one fixed square "
        "canvas (the image fitted inside, padded bottom-right; 1024x1024 by "
        "default for R50); the graph stops after the box-delta and class "
        "convolutions and LibreYOLO rebuilds the anchors on the host. Supported, "
        "not validated: on a Metis M.2 Max RetinaNet-R50 scores 1.11 mAP50-95 "
        "below the PyTorch model's own val() on 1000 COCO val2017 images (0.3410 "
        "vs 0.3521): 0.49 from the canvas (FP32 0.3472) and 0.62 from INT8. "
        "The 1024 canvas is about 1.64x the device compute of 800; imgsz=800 "
        "is faster and scores 2.25 below (0.3296; docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "supported sizes: r50 (measured drop, SIZE_ACCURACY_NOTES), fixed "
        "1024x1024 canvas by default (imgsz=800 for speed); r50v2 is blocked "
        "(GroupNorm head, BLOCKED_SIZES). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1"
    ),
)
_add(
    "available",
    ("efficientdet",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the whole EfficientDet "
        "graph runs on the Metis AIPU after exact rewrites of the BiFPN fusion "
        "(division by a constant as a multiplication, the stacked weighted sum "
        "as additions) and of the squeeze-excite averages (staged 2x2 pools), "
        "and the host decodes the boxes in float. Supported, not validated: on "
        "a Metis M.2 Max with per-channel activation ranges EfficientDet-d0 "
        "loses 1.78 mAP50-95 points against FP32 on 1000 COCO val2017 images; "
        "EfficientDet-d1 and -d2 run at d0's 512x512 canvas (their native 640 "
        "and 768 do not compile), which costs 3.10 and 5.31 points in FP32 and "
        "INT8 1.64 and 1.38 more (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "supported sizes: d0, d1 and d2 (d1 and d2 at a fixed 512x512 canvas; "
        "measured drops, SIZE_ACCURACY_NOTES); d3 and d4 do not compile "
        "(BLOCKED_SIZES). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, "
        "fixed input size"
    ),
)
_add(
    "validated",
    ("yolo1",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the whole YOLOv1 graph "
        "runs on the Metis AIPU after exact rewrites: the [local] locally "
        "connected layer (an Einsum) as one convolution per 7x7 output "
        "position, every Darknet batch norm folded into its convolution, and "
        "the [connected] layer as row convolutions; the dense-grid decode "
        "runs on the host. YOLOv1-b on a Metis M.2 Max is within 0.3 "
        "mAP50-95 of FP32 on a paired 1000-image COCO val2017 subset limited "
        "to the 20 VOC classes (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: b (t has no published weights). axelera-devkit / "
        "axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, fixed input size"
    ),
)
_add(
    "available",
    ("fcos",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 multi-part export compiled with Axelera's toolchain. FCOS's head "
        "normalizes every convolution with GroupNorm (per-image statistics), "
        "which the INT8 toolchain has no form for, so the ResNet-FPN trunk, the "
        "four head convolutions (classification and regression towers stacked, "
        "on all five levels packed onto one canvas) and the predictor run on "
        "the Metis AIPU as six compiled parts, and the host applies each "
        "GroupNorm + ReLU and decodes the boxes. Supported, not validated: on a "
        "Metis M.2 Max FCOS-R50 loses 4.13 mAP50-95 points against FP32 on 1000 "
        "COCO val2017 images at its 800x800 canvas (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "supported sizes: r50 (measured drop, SIZE_ACCURACY_NOTES); fixed "
        "800x800 canvas by default. axelera-devkit / axelera-rt 1.9.0, Linux "
        "x86_64 compile, INT8, batch 1"
    ),
)
_add(
    "validated",
    ("yolo9",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the graph is cut after "
        "the head convolutions, class-tower activations and the class and "
        "distance logits are bounded inside the convolutions, and LibreYOLO "
        "decodes on the host. YOLO9-t, -s, -m and -c on a Metis M.2 Max are "
        "within 0.6 mAP50-95 of FP32 on a paired 1000-image COCO val2017 subset "
        "(docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: t, s, m, c (on-device results in docs/axelera.md). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, "
        "batch 1, fixed input size"
    ),
)
_add(
    "validated",
    ("yolo9_p2",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; YOLO9-P2 keeps YOLO9's "
        "head towers and adds a stride-4 level, so the graph is cut after the "
        "four levels' head convolutions with the YOLO9 output windows and "
        "LibreYOLO decodes on the host. YOLO9-P2-s (the VisDrone research "
        "preview, non-commercial) on a Metis M.2 Max is within 0.3 mAP50-95 "
        "of FP32 on 448 VisDrone2019-DET val images, calibrated on 100 other "
        "val images (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: s (t has no published checkpoint). axelera-devkit / "
        "axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, fixed input size"
    ),
)
_add(
    "validated",
    ("yolo9_e2e",),
    ("detect",),
    ("axelera",),
    reason=(
        "INT8 export compiled with Axelera's toolchain; the graph is cut after "
        "the one-to-one head convolutions (the branch YOLO9-E2E's NMS-free "
        "inference decodes), with the YOLO9 output windows and class-tower "
        "bound, and LibreYOLO decodes on the host. YOLO9-E2E-t, -s, -m and -c "
        "on a Metis M.2 Max are within 0.8 mAP50-95 of FP32 on a paired "
        "1000-image COCO val2017 subset (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "validated sizes: t, s, m, c (on-device results in docs/axelera.md). "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, "
        "batch 1, fixed input size"
    ),
)
_add(
    "blocked",
    ("zipdepth",),
    ("depth",),
    ("axelera",),
    reason=(
        "ZipDepth's GCNet context block weights its 576 stage-3 positions with "
        "a softmax, which has no AIPU form within SDK 1.9's limits. Measured "
        "on ZipDepth-bnpu at 384 px: after exact rewrites of its channel and "
        "strip mean reductions the graph quantizes, but the softmax over "
        "positions either needs a reshape pair (4 layout transforms, at most "
        "3 allowed), a one-hot transposed convolution (5184 reduction blocks, "
        "too many for the MVM unit) or, written as an exponential-weighted "
        "pooling, Exp and Reciprocal, which the quantizer does not support. "
        "Running only the division on the host would need a compiled part "
        "with two inputs."
    ),
)
_add(
    "blocked",
    ("centernet",),
    ("detect",),
    ("axelera",),
    reason=(
        "Both CenterNet sizes (ResDCN-18 and DLA-34) upsample with modulated "
        "deformable convolutions, which sample the feature map at learned "
        "per-pixel offsets (exported as grid_sample); Axelera's 1.9 quantizer "
        "has no converter for GridSample. Measured multi-part route: ResDCN-18 "
        "as four compiled parts with its three deformable convolutions and the "
        "top-k decode on the host compiles and runs, but scores 0.098 mAP50-95 "
        "against FP32's 0.282 on 300 COCO val2017 images (0.081 with histogram "
        "ranges): the INT8 ResNet-18 trunk drifts (device vs quantized cosine "
        "0.91) and the deformable sampling amplifies it. DLA-34 has 16 "
        "deformable convolutions, about 17 parts, and was not attempted."
    ),
)
_add(
    "blocked",
    ("convnext",),
    ("classify",),
    ("axelera",),
    reason=(
        "ConvNeXt normalizes every block with a per-pixel LayerNorm (each pixel "
        "divided by the standard deviation of its own channels). The channel "
        "mean and variance have exact AIPU forms, but the reciprocal square "
        "root of that per-pixel variance has none, and it sits right before "
        "the block's GELU, so it cannot be folded into a convolution; with SDK "
        "1.9 quantization stops on the stem's channel ReduceMean. Running each "
        "LayerNorm on the host between accelerated parts would need about 23 "
        "parts for ConvNeXt-t (18 blocks, stem and three downsampling layers). "
        "A hybrid export (convolutions on the Metis, LayerNorms on the host) "
        "is the route if it is ever needed."
    ),
)
_add(
    "blocked",
    ("convnextv2",),
    ("classify",),
    ("axelera",),
    reason=(
        "ConvNeXt V2 normalizes every block with a per-pixel LayerNorm and "
        "Global Response Normalization (each channel scaled by its spatial L2 "
        "norm over the mean of those norms), data-dependent divisions and "
        "square roots that the AIPU does not run; with SDK 1.9 quantization "
        "stops on the first LayerNorm."
    ),
)
_add(
    "blocked",
    ("ec",),
    ("detect",),
    ("axelera",),
    reason=(
        "EC's decoder samples features with gather-based deformable attention "
        "(bounds masks, data-dependent Gather) and LayerNorm; with SDK 1.9 "
        "quantization has no converter for its GreaterOrEqual masks, and the "
        "host can only run operators before or after the accelerated graph."
    ),
)
_add(
    "blocked",
    ("tinyformer",),
    ("detect",),
    ("axelera",),
    reason=(
        "TinyFormer's decoder uses the same gather-based deformable attention "
        "as EC (bounds masks, data-dependent Gather) and LayerNorm; with SDK "
        "1.9 quantization has no converter for its GreaterOrEqual masks."
    ),
)
_add(
    "blocked",
    ("nafnet",),
    ("restore",),
    ("axelera",),
    reason=(
        "Not available yet: no trained LibreYOLO NAFNet weights are published, "
        "so there is no model to export and measure on a Metis. Known work for "
        "when weights exist: the input skip around the whole graph runs on the "
        "host (as for Real-ESRGAN), and every block's per-pixel LayerNorm has "
        "the same reciprocal square root as ConvNeXt's."
    ),
)
_add(
    "blocked",
    ("teed",),
    ("edge",),
    ("axelera",),
    reason=(
        "TEED's Smish activation takes a logarithm, which Axelera's 1.9 INT8 "
        "toolchain has no form for (quantization stops on the Log); no trained "
        "LibreYOLO weights are published to measure an exported model against."
    ),
)
_add(
    "blocked",
    ("pidnet",),
    ("semantic",),
    ("axelera",),
    reason=(
        "PIDNet-s compiles for Metis with SDK 1.9 (whole graph at 1024x1024, "
        "10.3 hours, after exact rewrites of its spatial attention and of the "
        "DAPPM pooling: a 17x17 average as two 1-D pools, the global average in "
        "2x2 steps), but INT8 collapses its output: cosine vs FP32 about 0 "
        "(-0.16 to 0.13) on the device and on the SDK's CPU-quantized model, "
        "with per-tensor min-max and with hybrid per-channel ranges alike. A "
        "per-stage quantization tap screen is the next step if PIDNet is "
        "revisited."
    ),
)
_add(
    "blocked",
    ("ppliteseg",),
    ("semantic",),
    ("axelera",),
    reason=(
        "Not available yet: PP-LiteSeg t50, b50, t75 and b75 compile and run "
        "on a Metis M.2 Max after exact graph rewrites (channel reductions as "
        "1x1 convolutions, pyramid-pool resizes as factor-2/3 steps, wide pool "
        "windows staged), but the device labels agree with FP32 on only "
        "48-56% of pixels (fidelity mIoU 0.17-0.19 on 1000 COCO val2017 "
        "images; 41% on street scenes) with either per-tensor min-max or "
        "hybrid ranges. Device prefix cuts show the STDC backbone drifting "
        "stage by stage (cosine vs FP32 0.996 at stride 4, 0.985 at stride 8, "
        "0.975 at stride 16, 0.951 at the stride-32 output, the largest "
        "step); no exact rewrite tried there helped."
    ),
)
_add(
    "blocked",
    ("unet",),
    ("semantic",),
    ("axelera",),
    reason=(
        "Not available yet: UNet has no ONNX semantic export contract in "
        "LibreYOLO yet, and the SDK 1.9 compile of UNet-s at its 512x1024 crop "
        "did not finish within 3 hours (quantization takes about 20 minutes); "
        "no structural error was seen there. At the full 1024x2048 frame "
        "lowering finds no L1 tiling (4.7 MB needed, 4.1 MB available)."
    ),
)
_add(
    "hybrid",
    ("rtdetr", "rtdetrv2", "dfine", "deim"),
    ("detect",),
    ("axelera",),
    reason=(
        "Hybrid INT8 export: the CNN backbone with the encoder's input "
        "projections, and the convolutional fusion neck with the decoder's input "
        "projections, are compiled with Axelera's toolchain and run on the Metis "
        "AIPU; the attention encoder layer and the decoder (query selection, "
        "deformable attention) run on the host CPU as FP32 ONNX graphs through "
        "ONNX Runtime (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "measured sizes in docs/axelera.md. Two .axm parts, two host ONNX graphs "
        "and metadata parts; keep them together. Needs onnxruntime at run time. "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, "
        "fixed square input"
    ),
)
_add(
    "hybrid",
    ("faster_rcnn",),
    ("detect",),
    ("axelera",),
    reason=(
        "Hybrid INT8 export at a fixed square canvas: the input normalization, "
        "the backbone and the FPN are compiled with Axelera's toolchain and run on "
        "the Metis AIPU; the region proposal network, RoIAlign, the box head and "
        "NMS run on the host CPU as an FP32 ONNX graph through ONNX Runtime "
        "(docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "measured sizes in docs/axelera.md. One .axm part, one host ONNX graph and "
        "metadata parts; keep them together. Needs onnxruntime at run time. "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, "
        "fixed square canvas (images are stretched to it)"
    ),
)
_add(
    "hybrid",
    ("detr",),
    ("detect",),
    ("axelera",),
    reason=(
        "Hybrid INT8 export: the ResNet backbone and the input projection are "
        "compiled with Axelera's toolchain and run on the Metis AIPU; the "
        "transformer encoder and decoder and the prediction heads run on the host "
        "CPU as an FP32 ONNX graph through ONNX Runtime (docs/axelera.md)."
    ),
    since="1.7",
    constraint=(
        "measured sizes in docs/axelera.md. One .axm part, one host ONNX graph and "
        "metadata parts; keep them together. Needs onnxruntime at run time. "
        "axelera-devkit / axelera-rt 1.9.0, Linux x86_64 compile, INT8, batch 1, "
        "fixed square input"
    ),
)
_add(
    "blocked",
    ("rfdetr",),
    ("detect",),
    ("axelera",),
    reason=(
        "RF-DETR needs attention, data-dependent TopK/Gather and grid_sample, "
        "which Axelera's compiler does not place on the AIPU, and the host can "
        "only run operators before or after the accelerated graph."
    ),
)


def get_support(family: str, task: str, fmt: str) -> SupportEntry:
    """Return the canonical support entry for an export combination."""
    family = str(family or "").lower()
    task = str(task or "detect").lower()
    fmt = str(fmt or "").lower()
    if task not in TASKS:
        return SupportEntry("blocked", f"{task!r} is not a canonical LibreYOLO task.")
    if fmt not in EXPORT_FORMATS:
        return SupportEntry("blocked", f"{fmt!r} is not a registered export format.")

    exact = SUPPORT.get((family, task, fmt))
    if exact is not None:
        return exact
    if family in _FAMILY_BLOCKS:
        return SupportEntry("blocked", _FAMILY_BLOCKS[family])
    if task in _TASK_BLOCKS:
        return SupportEntry("blocked", _TASK_BLOCKS[task])
    if fmt == "ncnn" and family in _NCNN_BLOCKS:
        label = _NCNN_BLOCKS[family]
        return SupportEntry(
            "blocked",
            f"NCNN export is not supported for {label}: the model requires decoder "
            "or sampling operations unavailable in NCNN. "
            "Use ONNX, OpenVINO, TorchScript, or TensorRT instead.",
        )
    if fmt == "mnn":
        return SupportEntry(
            "blocked",
            "MNN v1 has no implemented runtime contract for this family and task.",
        )
    if fmt == "rknn":
        return SupportEntry(
            "blocked",
            "RKNN v1 is limited to the exact simulator-tested detection variants: "
            "YOLO9-t, YOLO9-E2E-t, YOLO-NAS-s, and PicoDet-s on RK3588.",
        )
    if fmt == "axelera":
        return SupportEntry(
            "blocked",
            "This family and task have not been compiled or measured on "
            "Axelera Metis; docs/axelera.md lists the supported ones.",
        )
    if fmt in {"tensorrt", "openvino"}:
        runtime = "TensorRT" if fmt == "tensorrt" else "OpenVINO"
        return SupportEntry(
            "available",
            f"The converter path is available, but the project has not yet "
            f"recorded {runtime} runtime parity for this family and task.",
        )
    if fmt == "tflite":
        return SupportEntry(
            "blocked",
            "This family and task have not been validated through the ONNX-to-TFLite path.",
        )
    if fmt == "paddle":
        return SupportEntry(
            "blocked",
            "This family and task have not been validated through the "
            "ONNX-to-Paddle conversion path.",
        )
    if fmt == "coreai":
        return SupportEntry(
            "blocked",
            "This family and task have not been validated for Core AI export.",
        )
    if fmt == "coreml":
        return SupportEntry(
            "blocked",
            "This family and task are not covered by the family-aware CoreML wrapper.",
        )
    return SupportEntry(
        "available",
        "Conversion is implemented; numeric runtime parity has not been recorded "
        "for this combination.",
    )


def iter_entries(
    tier: Tier | None = None,
) -> Iterator[tuple[tuple[str, str, str], SupportEntry]]:
    """Iterate explicit matrix entries, optionally filtered by tier."""
    for key, entry in sorted(SUPPORT.items()):
        if tier is None or entry.tier == tier:
            yield key, entry


def iter_validated() -> Iterator[tuple[tuple[str, str, str], SupportEntry]]:
    """Iterate explicit parity-validated entries."""
    return iter_entries("validated")


def iter_blocked() -> Iterator[tuple[tuple[str, str, str], SupportEntry]]:
    """Iterate explicit blocked entries."""
    return iter_entries("blocked")


#: Sizes that export and run but sit more than 1.0 point below FP32 on their
#: paired measurement set: supported, not validated. Informational only: the
#: export logs the note as one warning and proceeds. Each note quotes the number
#: measured with the recipe the library uses for that size
#: (``libreyolo.export.axelera.AXELERA_SIZE_RECIPES``).
SIZE_ACCURACY_NOTES: dict[tuple[str, str, str], dict[str, str]] = {
    ("fcos", "detect", "axelera"): {
        "r50": (
            "measured mAP50-95 drop 4.13 points vs FP32 (0.3405 vs 0.3818 on 1000 "
            "COCO val2017 images at the 800x800 canvas; -3.43 on 300; six compiled "
            "parts with GroupNorm on the host)"
        ),
    },
    ("mobilenetv4", "classify", "axelera"): {
        "s": (
            "measured top-1 drop 1.7 points vs FP32 (0.607 vs 0.624 on a paired "
            "1000-image ImageNetV2 set; histogram ranges, no residual-stream "
            "transform)"
        ),
        "l": (
            "measured top-1 drop 1.7 points vs FP32 (0.710 vs 0.727 on a paired "
            "1000-image ImageNetV2 set; min-max ranges with the residual-stream "
            "transform)"
        ),
    },
    ("efficientnetv2", "classify", "axelera"): {
        "b0": (
            "measured top-1 drop 1.1 points vs FP32 (0.656 vs 0.667 on a paired "
            "1000-image ImageNetV2 set; 1.0 on a second compile)"
        ),
    },
    ("quicksrnet", "restore", "axelera"): {
        "m2": (
            "measured PSNR drop 0.46 dB vs FP32 (28.66 vs 29.12 dB, SSIM 0.866 vs "
            "0.877, on 296 COCO crops upscaled 2x from 128 px; the validated bar "
            "for restoration is 0.3 dB)"
        ),
    },
    ("realesrgan", "restore", "axelera"): {
        "x4t": (
            "measured PSNR drop 0.49 dB vs FP32 (22.00 vs 22.49 dB, SSIM 0.606 vs "
            "0.658, on 296 COCO crops upscaled 4x from 64 px; the validated bar "
            "for restoration is 0.3 dB)"
        ),
    },
    ("midas", "depth", "axelera"): {
        "s": (
            "measured depth error vs FP32 (AbsRel 0.069, delta1 0.94 between the "
            "device and the FP32 model on 300 COCO val2017 images at 256 px; "
            "the validated bar is AbsRel 0.02)"
        ),
    },
    ("picodet", "detect", "axelera"): {
        "s": (
            "measured mAP50-95 drop 2.34 points vs FP32 (0.2453 vs 0.2687 on "
            "1000 COCO val2017 images; 1.28 on 300)"
        ),
        "m": (
            "measured mAP50-95 drop 2.86 points vs FP32 (0.3163 vs 0.3449 on "
            "1000 COCO val2017 images)"
        ),
        "l": (
            "measured mAP50-95 drop 1.70 points vs FP32 (0.3801 vs 0.3971 on "
            "1000 COCO val2017 images)"
        ),
    },
    ("deeplabv3", "semantic", "axelera"): {
        "mv3": (
            "measured mIoU drop 3.77 points vs FP32 (0.4853 vs 0.5230 on 300 COCO "
            "val2017 images with VOC-21 masks at 520 px, the FP32 ONNX export "
            "through the same path)"
        ),
        "r50": (
            "measured mIoU drop 5.08 points vs FP32 (0.4735 vs 0.5243 on 300 COCO "
            "val2017 images with VOC-21 masks at 520 px, the FP32 ONNX export "
            "through the same path; ASPP as per-tap 1x1 convolutions plus shifts)"
        ),
        "r101": (
            "measured mIoU drop 20.93 points vs FP32 (0.3795 vs 0.5888 on 300 "
            "COCO val2017 images with VOC-21 masks at 520 px, the FP32 ONNX export "
            "through the same path; ASPP as per-tap 1x1 convolutions plus shifts). "
            "The loss happens on the device (device vs CPU-quantized output "
            "cosine 0.895): check accuracy on your data, or use r50 or mv3"
        ),
    },
    ("rtmdet", "detect", "axelera"): {
        "t": (
            "measured mAP50-95 drop 2.29 points vs FP32 (0.3677 vs 0.3906 on 1000 "
            "COCO val2017 images, 64 coco128 calibration images, per-channel "
            "depthwise ranges and balanced CSPNeXt pointwise equalization)"
        ),
        "s": (
            "measured mAP50-95 drop 6.75 points vs FP32 (0.3446 vs 0.4121 on 1000 "
            "COCO val2017 images, 64 coco128 calibration images, per-channel "
            "depthwise ranges and balanced CSPNeXt pointwise equalization)"
        ),
        "m": (
            "measured mAP50-95 drop 2.66 points vs FP32 (0.4461 vs 0.4727 on 1000 "
            "COCO val2017 images; -2.11 on 300; 64 coco128 calibration images, "
            "per-channel depthwise ranges and balanced CSPNeXt pointwise "
            "equalization)"
        ),
        "l": (
            "measured mAP50-95 drop 1.87 points vs FP32 (0.4618 vs 0.4805 on 1000 "
            "COCO val2017 images; -1.92 on 300; 64 coco128 calibration images, "
            "per-channel depthwise ranges and balanced CSPNeXt pointwise "
            "equalization)"
        ),
    },
    ("efficientdet", "detect", "axelera"): {
        "d0": (
            "measured mAP50-95 drop 1.78 points vs FP32 (0.3185 vs 0.3363 on 1000 "
            "COCO val2017 images; spread over small, medium and large objects)"
        ),
        "d1": (
            "measured mAP50-95 drop 4.74 points vs FP32 at its native 640 (0.3475 "
            "on the device at a fixed 512x512 canvas vs 0.3949 on 1000 COCO "
            "val2017 images): its native 640 does not compile (no L1 tiling fits), "
            "the smaller canvas costs 3.10 points (mostly small objects) and INT8 "
            "1.64"
        ),
        "d2": (
            "measured mAP50-95 drop 6.69 points vs FP32 at its native 768 (0.3606 "
            "on the device at a fixed 512x512 canvas vs 0.4275 on 1000 COCO "
            "val2017 images): its native 768 does not compile (no L1 tiling fits), "
            "the smaller canvas costs 5.31 points (mostly small objects) and INT8 "
            "1.38"
        ),
    },
    ("retinanet", "detect", "axelera"): {
        "r50": (
            "measured mAP50-95 drop 1.11 points vs FP32 PyTorch val() with "
            "RetinaNet's own resize at the default 1024x1024 canvas (0.3410 vs "
            "0.3521 on 1000 COCO val2017 images; FP32 on the same canvas 0.3472, "
            "so the canvas costs 0.49 and INT8 0.62); the 1024 canvas is about "
            "1.64x the device compute of 800, and imgsz=800 scores 0.3296 (-2.25: "
            "2.03 canvas, 0.22 INT8)"
        ),
    },
    ("rtmdet", "segment", "axelera"): {
        "t": (
            "measured mask mAP50-95 drop 1.44 points vs FP32 (0.3250 vs 0.3394 "
            "on 1000 COCO val2017 images; box 0.3574 vs 0.3825, -2.51; 64 "
            "coco128 calibration images, per-channel depthwise ranges and "
            "balanced CSPNeXt pointwise equalization)"
        ),
        "s": (
            "measured mask mAP50-95 drop 5.80 points vs FP32 (0.3139 vs 0.3719 "
            "on 1000 COCO val2017 images; box 0.3445 vs 0.4171, -7.26; 64 "
            "coco128 calibration images, per-channel depthwise ranges and "
            "balanced CSPNeXt pointwise equalization)"
        ),
        "m": (
            "measured box mAP50-95 drop 3.96 points vs FP32 (box 0.4149 vs 0.4545, "
            "mask 0.3932 vs 0.4011, -0.79, on 1000 COCO val2017 images; 64 "
            "coco128 calibration images, per-channel depthwise ranges and "
            "balanced CSPNeXt pointwise equalization)"
        ),
        "l": (
            "measured box mAP50-95 drop 5.34 points vs FP32 (box 0.4193 vs 0.4727, "
            "mask 0.4113 vs 0.4163, -0.50, on 1000 COCO val2017 images; 64 "
            "coco128 calibration images, per-channel depthwise ranges and "
            "balanced CSPNeXt pointwise equalization)"
        ),
    },
    ("vgg", "classify", "axelera"): {
        "19bn": (
            "measured top-1 drop 5.1 points vs FP32 (0.593 vs 0.644 on 1000 "
            "ImageNetV2 images, 250 ImageNetV2 calibration images, min-max "
            "ranges; histogram ranges gave -6.0)"
        ),
    },
    ("ppyoloe", "detect", "axelera"): {
        "s": (
            "measured mAP50-95 drop 1.23 points vs FP32 (0.396 vs 0.409 on 1000 "
            "COCO val2017 images)"
        ),
        "m": (
            "measured mAP50-95 drop 1.22 points vs FP32 (0.442 vs 0.454 on 1000 "
            "COCO val2017 images)"
        ),
        "x": (
            "measured mAP50-95 drop 1.71 points vs FP32 (0.475 vs 0.492 on 1000 "
            "COCO val2017 images)"
        ),
    },
    ("yolo4", "detect", "axelera"): {
        "b": (
            "measured mAP50-95 drop 1.52 points vs FP32 (0.465 vs 0.481 on 1000 "
            "COCO val2017 images; histogram ranges and a [-6, 6] box-output "
            "window, without which the drop is 6.8)"
        ),
    },
    ("yolox", "detect", "axelera"): {
        "n": (
            "measured mAP50-95 drop 1.81 points vs FP32 (0.243 vs 0.261 on 300 "
            "COCO val2017 images, 64 calibration images; 1.66 with 128)"
        ),
    },
    ("yolonas", "pose", "axelera"): {
        "n": (
            "measured keypoint mAP50-95 drop 1.30 points vs FP32 (0.696 vs 0.709 "
            "on 300 COCO val2017 person images; 1.11 when calibrated on person "
            "images)"
        ),
        "s": (
            "measured keypoint mAP50-95 drop 2.21 points vs FP32 (0.734 vs 0.757 "
            "on 300 COCO val2017 person images; class and keypoint towers both "
            "bounded)"
        ),
    },
}

#: Sizes that do not compile, do not run or give meaningless output on the
#: format, each with its structural reason. Accuracy never lands a size here
#: (that is SIZE_ACCURACY_NOTES); the export raises for these.
BLOCKED_SIZES: dict[tuple[str, str, str], dict[str, str]] = {
    ("efficientdet", "detect", "axelera"): {
        "d3": (
            "does not compile for Metis with the 1.9 SDK: lowering finds no L1 "
            "tiling even with one-row tiles, at its native 896 (2.52 MB needed, "
            "2.15 MB available) and at a 512 canvas (2.58 vs 2.14 MB); its "
            "160-channel BiFPN and heads set the need, not the canvas; use d2"
        ),
        "d4": (
            "does not compile for Metis with the 1.9 SDK: lowering finds no L1 "
            "tiling even with one-row tiles, at its native 1024 (2.69 MB needed, "
            "0.94 MB available) and at a 512 canvas (2.10 vs 1.22 MB). Prefix "
            "cuts at 512 put the overflow in the EfficientNet-B4 backbone: its "
            "first five stages compile, the 272-channel stage (blocks.5) alone "
            "needs 3.12 MB of L1 against 2.47 MB; use d2"
        ),
    },
    ("midas", "depth", "axelera"): {
        "l": (
            "does not quantize for Metis with the 1.9 SDK: its DPT ViT-L encoder "
            "(48 LayerNormalization, 24 Softmax, 28 Erf and the attention MatMuls "
            "at 384 px) stops at the quantizer's ONNX import ('unexpected input "
            "value type'), after 6 min and an 18 GB peak; use MiDaS-s"
        ),
    },
    ("efficientnetv2", "classify", "axelera"): {
        "b1": (
            "does not compile with Axelera's SDK 1.9.0: at 224 and 240 px the "
            "compiled model has two output buffers for its one output; the "
            "second is the channel-padded input (224x224x16 at 224 px), which "
            "the compiler lists as an output (b0 has no such buffer). Running "
            "the classifier as a 1x1 convolution, or compiling with first-layer "
            "replication, IO padding removal, memory paging or host/device "
            "double buffering switched off, fails the same way; at 256 px the "
            "squeeze-excitation pooling exceeds a reduction limit"
        ),
        "b2": (
            "does not compile with Axelera's SDK 1.9.0: like b1, the compiled "
            "model has two output buffers for its one output"
        ),
        "b3": (
            "does not compile with Axelera's SDK 1.9.0: at its native 300 px and "
            "at 224 px the compiled model has two output buffers for its one output"
        ),
    },
    ("retinanet", "detect", "axelera"): {
        "r50v2": (
            "does not quantize for Metis with the 1.9 SDK: like FCOS, its head "
            "normalizes every convolution with GroupNorm (per-image statistics) "
            "and quantization stops on the head's InstanceNormalization ops; use "
            "RetinaNet-r50"
        ),
    },
    ("rtmdet", "segment", "axelera"): {
        "x": (
            "does not compile for Metis at 640x640 with the 1.9 SDK: lowering "
            "fails in AssignTilingFactors (ScheduleMemory, LowerTIR) because no "
            "tiling factor fits L1 (3.99 MB needed, 3.32 MB available; after 2 h "
            "08 min, 8.1 GB peak RSS); use RTMDet-Ins-l"
        ),
    },
    ("rtmdet", "detect", "axelera"): {
        "x": (
            "does not compile for Metis at 640x640 with the 1.9 SDK: the "
            "compiler's LowerTIR pass still overflows its on-chip buffers after 4 "
            "iterations (after 2 h 21 min, 7 GB peak RSS); use RTMDet-l"
        ),
    },
}

#: Sizes not exportable yet because nobody has measured them on a Metis. This
#: is not a block (nothing is known to fail); the export raises until a
#: measurement moves the size to SIZE_ACCURACY_NOTES or to validated.
UNMEASURED_SIZES: dict[tuple[str, str, str], dict[str, str]] = {
    ("yolo9_p2", "detect", "axelera"): {
        "t": (
            "has not been measured on a Metis yet: no trained YOLO9-P2-t "
            "checkpoint is published"
        ),
    },
    ("dfine", "detect", "axelera"): {
        size: "has not been measured on a Metis yet" for size in ("l", "x")
    },
    ("deim", "detect", "axelera"): {
        size: "has not been measured on a Metis yet" for size in ("s", "m", "l", "x")
    },
    ("detr", "detect", "axelera"): {
        size: "has not been measured on a Metis yet" for size in ("r50dc5", "r101", "r101dc5")
    },
    ("faster_rcnn", "detect", "axelera"): {
        size: "has not been measured on a Metis yet" for size in ("s", "m", "l")
    },
    ("realesrgan", "restore", "axelera"): {
        "x2": (
            "has not been measured on a Metis yet: its whole-graph compile at 128 px "
            "with SDK 1.9.0 did not finish within 3.5 h (no error seen)"
        ),
    },
}

#: Formats whose INT8 export is limited to these (family, task) pairs.
INT8_EXPORTS: dict[str, frozenset[tuple[str, str]]] = {
    "onnx": frozenset({("yolo9", "detect")}),
    # Full-integer TFLite export has been checked on real weights for these.
    "tflite": frozenset({("yolox", "detect"), ("yolo9", "detect")}),
    # Axelera Metis is INT8-only; see the axelera rows above.
    "axelera": frozenset(
        {
            ("yolonas", "detect"),
            ("yolonas", "pose"),
            ("yolonas", "obb"),
            ("yolo9", "detect"),
            ("yolo9_p2", "detect"),
            ("yolo9_e2e", "detect"),
            ("resnet", "classify"),
            ("mobilenetv4", "classify"),
            ("efficientnetv2", "classify"),
            ("vgg", "classify"),
            ("alexnet", "classify"),
            ("quicksrnet", "restore"),
            ("realesrgan", "restore"),
            ("midas", "depth"),
            ("deeplabv3", "semantic"),
            ("fcn", "semantic"),
            ("yolox", "detect"),
            ("yolo1", "detect"),
            ("yolo2", "detect"),
            ("yolo4", "detect"),
            ("yolo3", "detect"),
            ("yolo7", "detect"),
            ("ppyoloe", "detect"),
            ("picodet", "detect"),
            ("rtmdet", "detect"),
            ("fcos", "detect"),
            ("retinanet", "detect"),
            ("ssd", "detect"),
            ("efficientdet", "detect"),
            ("rtmdet", "segment"),
            ("rtdetr", "detect"),
        }
    ),
}

#: Families whose export accepts FP32 only.
FP32_ONLY_FAMILIES = frozenset({"gtr"})


def precision_support(family: str, task: str, fmt: str) -> tuple[bool, bool]:
    """Return whether export() accepts ``(half=True, int8=True)`` for a combination.

    The format's own ``supports_fp16`` / ``supports_int8`` flags still apply on
    top of this.
    """
    family = str(family or "").lower()
    task = str(task or "detect").lower()
    fmt = str(fmt or "").lower()
    if get_support(family, task, fmt).tier == "blocked" or family in FP32_ONLY_FAMILIES:
        return False, False
    int8_pairs = INT8_EXPORTS.get(fmt)
    return True, int8_pairs is None or (family, task) in int8_pairs


def validated_alternatives(family: str, task: str) -> tuple[str, ...]:
    """Return validated formats for a concrete family and task."""
    return tuple(
        fmt
        for fmt in EXPORT_FORMATS
        if get_support(family, task, fmt).tier == "validated"
    )


__all__ = [
    "EXPORT_FORMATS",
    "FP32_ONLY_FAMILIES",
    "INT8_EXPORTS",
    "BLOCKED_SIZES",
    "SIZE_ACCURACY_NOTES",
    "UNMEASURED_SIZES",
    "SUPPORT",
    "SupportEntry",
    "Tier",
    "get_support",
    "iter_blocked",
    "iter_entries",
    "precision_support",
    "iter_validated",
    "validated_alternatives",
]


# --- V-JEPA 2 -------------------------------------------------------------
#
# Every V-JEPA 2 graph consumes a 5D video clip, so only the formats whose
# toolchains were actually exercised on a rank-5 input are claimed here.
_add(
    "validated",
    ("vjepa2",),
    ("embed",),
    ("onnx", "torchscript"),
    reason=(
        "Fixed-frame 5D clip graphs are covered by export, runtime reload, "
        "torch-vs-runtime embedding parity, unit-norm output, and a "
        "temporal-order sensitivity check that fails a graph which ignores "
        "frame order. Parity is measured by driving the runtime directly "
        "with a 5D clip."
    ),
    since="1.5",
    constraint=(
        "FP32, dynamic batch, fixed frame count / crop / tubelet geometry per "
        "graph; ONNX needs opset >= 14 for scaled_dot_product_attention. "
        "The graph must be driven with a 5D clip: LibreYOLO's exported-backend "
        "path still preprocesses to a 4D image batch and raises "
        "NotImplementedError for this family."
    ),
)
_add(
    "blocked",
    ("vjepa2",),
    ("embed", "classify"),
    ("ncnn", "tflite"),
    reason=(
        "Both toolchains are built around rank-4 image tensors; a rank-5 clip "
        "input has no supported conversion path."
    ),
)


# --- LeVJEPA ---------------------------------------------------------------
_add(
    "validated",
    ("levjepa",),
    ("embed",),
    ("torchscript",),
    reason=(
        "The 303M-parameter released checkpoint was exported and reloaded on "
        "CUDA. The graph matched native FP32 output exactly, retained unit-norm "
        "1024-D output, and changed when frame order was reversed."
    ),
    since="1.5",
    constraint=(
        "FP32, batch 1, fixed 16-frame 224x224 input. Drive the graph directly "
        "with a preprocessed 5D clip; exported-backend video preprocessing is "
        "not implemented."
    ),
)
_add(
    "blocked",
    ("levjepa",),
    ("embed",),
    ("ncnn", "tflite"),
    reason=(
        "Both toolchains are built around rank-4 image tensors; a rank-5 clip "
        "input has no supported conversion path."
    ),
)


# --- Guided and specialist restoration -----------------------------------
_add(
    "blocked",
    ("ddcolor",),
    ("restore",),
    EXPORT_FORMATS,
    reason=(
        "DDColor export needs a two-input contract that preserves the source "
        "image's original-resolution OpenCV Lab luminance plane through RGB "
        "reconstruction. Native prediction is supported; no exported runtime "
        "contract or parity gate is defined yet."
    ),
)
_add(
    "blocked",
    ("hvi_cidnet",),
    ("restore",),
    EXPORT_FORMATS,
    reason=(
        "HVI-CIDNet export has not defined metadata and runtime semantics for "
        "its gamma, saturation, and intensity controls, and no exported "
        "runtime parity gate has been recorded. Use native PyTorch inference."
    ),
)
_add(
    "blocked",
    ("lama",),
    ("restore",),
    EXPORT_FORMATS,
    reason=(
        "LibreLaMa already embeds and executes the exact upstream QDQ ONNX "
        "artifact. Re-exporting that opaque graph through PyTorch is neither "
        "meaningful nor supported."
    ),
)
_add(
    "blocked",
    ("vitmatte",),
    ("matte",),
    EXPORT_FORMATS,
    reason=(
        "ViTMatte needs a documented four-channel RGB-plus-trimap runtime "
        "input contract and guided-backend parity before export can be "
        "advertised. Use native PyTorch inference."
    ),
)

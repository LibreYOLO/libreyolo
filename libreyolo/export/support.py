"""Canonical export support tiers for model, task, and format combinations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from ..tasks import TASKS

Tier = Literal["validated", "available", "blocked"]
EXPORT_FORMATS = (
    "onnx",
    "torchscript",
    "executorch",
    "tensorrt",
    "openvino",
    "paddle",
    "mnn",
    "rknn",
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


# Existing parity-backed paths. New validated rows must land with a parity test.
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
        "X2Paddle 1.6.0, PaddlePaddle 2.6.2 CPU, ONNX 1.17/opset 15, FP32, "
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
        "X2Paddle 1.6.0, PaddlePaddle 2.6.2 CPU, ONNX 1.17/opset 15, FP32, "
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
        "X2Paddle 1.6.0, PaddlePaddle 2.6.2 CPU, ONNX 1.17/opset 15, FP32, "
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
        "X2Paddle 1.6.0, PaddlePaddle 2.6.2 CPU, ONNX 1.17/opset 15, FP32, "
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
        "X2Paddle 1.6.0, PaddlePaddle 2.6.2 CPU, ONNX 1.17/opset 15, FP32, "
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
    "blocked",
    ("yolo9_e2e", "yolo9_p2"),
    ("detect",),
    ("tflite",),
    reason=(
        "onnx2tf 2.6.7 exports a runnable artifact, but public top-k class "
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
    EXPORT_FORMATS,
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
    "smolvlm2": "Generative VLM export is out of scope for v1.",
    "locateanything": "Generative VLM export is out of scope for v1.",
    "northmicrovision": "Generative VLM export is out of scope for v1.",
    "gemma4": "Generative VLM export is out of scope for v1.",
    "moondream": "Generative VLM export is out of scope for v1.",
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
    "rtdetrv2": "RT-DETRv2",
    "rtdetrv4": "RT-DETRv4",
    "rfdetr": "RF-DETR",
    "ec": "EC",
}


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


def validated_alternatives(family: str, task: str) -> tuple[str, ...]:
    """Return validated formats for a concrete family and task."""
    return tuple(
        fmt
        for fmt in EXPORT_FORMATS
        if get_support(family, task, fmt).tier == "validated"
    )


__all__ = [
    "EXPORT_FORMATS",
    "SUPPORT",
    "SupportEntry",
    "Tier",
    "get_support",
    "iter_blocked",
    "iter_entries",
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

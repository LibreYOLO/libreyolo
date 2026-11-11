"""Smoke-test a fresh ``pip install "libreyolo[all]"`` environment (#989).

Imports the optional packages ``all`` provides on this platform, predicts with
YOLO9, then exports it to every backend in ``all`` that runs on a CPU-only
machine and checks each artifact predicts like PyTorch. Run with the
environment's python from an empty working directory; the YOLO9 weights are
downloaded from Hugging Face. Exit code 1 if any check fails.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
import time
import traceback

import numpy as np

import libreyolo
from libreyolo import SAMPLE_IMAGE, LibreYOLO

results: list[tuple[str, bool, str]] = []


def check(name, fn):
    start = time.time()
    try:
        detail = fn() or ""
        results.append((name, True, detail))
        print(f"ok    {name} ({time.time() - start:.1f}s) {detail}", flush=True)
    except Exception as exc:  # noqa: BLE001
        results.append((name, False, repr(exc)))
        print(f"FAIL  {name} ({time.time() - start:.1f}s) {exc!r}", flush=True)
        traceback.print_exc(limit=4)


print(
    f"libreyolo {libreyolo.__version__}, Python {sys.version.split()[0]}, "
    f"{sys.platform} {platform.machine()}",
    flush=True,
)

MODULES = [
    "onnx", "onnxruntime", "onnxsim", "openvino", "ncnn", "MNN",
    "tritonclient.http", "transformers", "peft", "timm", "yt_dlp", "mlflow",
    "wandb", "comet_ml", "clearml", "dvclive", "torch.utils.tensorboard",
    "huggingface_hub", "lmdb", "num2words",
]  # fmt: skip
if sys.version_info >= (3, 12):
    MODULES += ["onnx2tf", "ai_edge_litert"]
if sys.platform == "darwin" or (
    sys.platform == "linux" and platform.machine() == "x86_64"
):
    MODULES.append("coremltools")
for module in MODULES:
    check(f"import {module}", lambda m=module: importlib.import_module(m) and "")

# GPU-only packages cannot be imported on a CPU runner; check they installed.
if sys.platform != "darwin" and platform.machine().lower() in ("x86_64", "amd64"):
    for dist in ("tensorrt-cu12",):
        check(f"installed {dist}", lambda d=dist: importlib.metadata.version(d))

state: dict = {}


def _detections(result):
    def as_array(value):
        return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)

    boxes = result.boxes
    return (
        as_array(boxes.xyxy).reshape(-1, 4),
        as_array(boxes.cls).astype(int),
        as_array(boxes.conf).astype(float),
    )


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match(reference, export):
    """Pair reference and export detections one to one.

    A pair needs the same class and IoU >= 0.9; each export box is used at
    most once, so duplicated detections show up as extras. Returns the
    reference boxes left unpaired, the export boxes left unpaired and the
    largest score difference over the pairs.
    """
    used: set[int] = set()
    missing, worst_conf = 0, 0.0
    for i in np.argsort(-reference[2]):
        box, cls, conf = reference[0][i], reference[1][i], reference[2][i]
        best, best_iou = None, 0.9
        for j, (other, other_cls) in enumerate(zip(export[0], export[1])):
            if j not in used and other_cls == cls:
                iou = _iou(box, other)
                if iou >= best_iou:
                    best, best_iou = j, iou
        if best is None:
            missing += 1
        else:
            used.add(best)
            worst_conf = max(worst_conf, abs(float(export[2][best]) - float(conf)))
    return missing, len(export[1]) - len(used), worst_conf


def predict_torch():
    state["model"] = LibreYOLO("LibreYOLO9t.pt")
    state["ref"] = _detections(state["model"](SAMPLE_IMAGE))
    n = len(state["ref"][1])
    assert n > 0, "no detections on the sample image"
    return f"{n} detections"


check("predict yolo9 pytorch", predict_torch)

FORMATS = ["onnx", "openvino", "ncnn", "mnn"]
if sys.version_info >= (3, 12):
    FORMATS.append("tflite")
if sys.platform == "darwin":
    FORMATS.append("coreml")


def export_and_predict(fmt):
    path = state["model"].export(format=fmt)
    got = _detections(LibreYOLO(path)(SAMPLE_IMAGE))
    # Same classes, boxes and scores as PyTorch; one borderline detection
    # near the confidence threshold may appear on only one side.
    missing, extra, conf_diff = _match(state["ref"], got)
    assert missing <= 1 and extra <= 1, (
        f"{missing} PyTorch detections missing and {extra} unexpected "
        f"(same class, IoU >= 0.9) out of {len(state['ref'][1])}"
    )
    assert conf_diff <= 0.05, f"a matched score differs by {conf_diff:.3f}"
    return (
        f"{len(got[1])} detections, {missing} missing, {extra} extra, "
        f"max score diff {conf_diff:.4f}"
    )


if "ref" in state:
    for fmt in FORMATS:
        check(f"export+predict {fmt}", lambda f=fmt: export_and_predict(f))

failed = [name for name, ok, _ in results if not ok]
print(f"SUMMARY {len(results) - len(failed)}/{len(results)} ok", flush=True)
for name in failed:
    print(f"FAILED {name}")
sys.exit(1 if failed else 0)

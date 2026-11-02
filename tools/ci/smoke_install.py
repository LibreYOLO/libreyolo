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
    for dist in ("tensorrt-cu12", "cuda-python"):
        check(f"installed {dist}", lambda d=dist: importlib.metadata.version(d))

state: dict = {}


def predict_torch():
    state["model"] = LibreYOLO("LibreYOLO9t.pt")
    state["n"] = len(state["model"](SAMPLE_IMAGE))
    assert state["n"] > 0, "no detections on the sample image"
    return f"{state['n']} detections"


check("predict yolo9 pytorch", predict_torch)

FORMATS = ["onnx", "openvino", "ncnn", "mnn"]
if sys.version_info >= (3, 12):
    FORMATS.append("tflite")
if sys.platform == "darwin":
    FORMATS.append("coreml")


def export_and_predict(fmt):
    path = state["model"].export(format=fmt)
    n = len(LibreYOLO(path)(SAMPLE_IMAGE))
    assert abs(n - state["n"]) <= 1, f"{n} detections, PyTorch found {state['n']}"
    return f"{n} detections"


if "model" in state:
    for fmt in FORMATS:
        check(f"export+predict {fmt}", lambda f=fmt: export_and_predict(f))

failed = [name for name, ok, _ in results if not ok]
print(f"SUMMARY {len(results) - len(failed)}/{len(results)} ok", flush=True)
for name in failed:
    print(f"FAILED {name}")
sys.exit(1 if failed else 0)

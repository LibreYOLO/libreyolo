"""Torch-free inference contract.

LibreYOLO supports a lightweight ONNX-only install (``pip install --no-deps
libreyolo`` plus ``numpy``/``opencv-python-headless``/``onnxruntime``) for
deployments that cannot afford the torch wheel. That contract is easy to break
by accident: any new module-level ``import torch`` on the ONNX path silently
re-introduces the dependency, and CI would never notice because CI installs
torch.

These tests run the torch-free path in a subprocess with ``torch`` and
``torchvision`` blocked by a meta-path finder, which is the only way to observe
the failure from a process that already has torch imported.

See https://github.com/LibreYOLO/libreyolo/discussions/711.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


_BLOCK_TORCH_PREAMBLE = """
import sys, importlib.abc

BLOCKED = {"torch", "torchvision"}

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ModuleNotFoundError("No module named '%s'" % root, name=root)
        return None

sys.meta_path.insert(0, _Blocker())
for _m in list(sys.modules):
    if _m.split(".")[0] in BLOCKED:
        del sys.modules[_m]
"""


def _repo_root() -> Path:
    import libreyolo

    return Path(libreyolo.__file__).resolve().parent.parent


def run_without_torch(body: str) -> subprocess.CompletedProcess:
    """Execute ``body`` in a subprocess where importing torch raises."""
    script = _BLOCK_TORCH_PREAMBLE + textwrap.dedent(body)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_repo_root())
    # Keep the child from inheriting a torch-enabled sitecustomize.
    env.pop("PYTHONSTARTUP", None)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def _assert_ok(proc: subprocess.CompletedProcess) -> str:
    if proc.returncode != 0:
        pytest.fail(
            "torch-free subprocess failed\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout


def test_import_libreyolo_without_torch():
    """``import libreyolo`` must not pull torch."""
    proc = run_without_torch(
        """
        import libreyolo
        assert "torch" not in sys.modules, "torch was imported eagerly"
        print("OK", libreyolo.__version__)
        """
    )
    assert "OK" in _assert_ok(proc)


def test_ultralytics_name_hint_without_torch():
    """``from libreyolo import YOLO`` explains itself without importing torch."""
    proc = run_without_torch(
        """
        try:
            from libreyolo import YOLO
        except ImportError as e:
            assert "LibreYOLO(" in str(e), str(e)
        else:
            raise AssertionError("YOLO resolved")
        import libreyolo
        assert not hasattr(libreyolo, "YOLO")
        assert "torch" not in sys.modules, "the YOLO hint pulled torch"
        print("OK")
        """
    )
    assert "OK" in _assert_ok(proc)


def test_onnx_backend_imports_without_torch():
    """The ONNX backend module must be importable torch-free."""
    proc = run_without_torch(
        """
        from libreyolo.backends.onnx import OnnxBackend
        assert "torch" not in sys.modules, "importing OnnxBackend pulled torch"
        print("OK", OnnxBackend.__name__)
        """
    )
    assert "OK" in _assert_ok(proc)


def test_yolo9_detect_postprocess_without_torch():
    """A yolo9 detect result must be buildable from numpy arrays, torch-free."""
    proc = run_without_torch(
        """
        import numpy as np
        from libreyolo.backends.base import BaseBackend

        class _Dummy(BaseBackend):
            def __init__(self):
                super().__init__(
                    model_path="dummy", nb_classes=2, device="cpu", imgsz=640,
                    model_family="yolo9", names={0: "a", 1: "b"},
                    task="detect", supported_tasks=("detect",),
                )
            def _run_inference(self, blob):
                raise NotImplementedError

        backend = _Dummy()
        # Two heavily overlapping same-class boxes plus one distinct box:
        # NMS must collapse the pair and keep the loner.
        boxes = np.array(
            [[10., 10., 100., 100.],
             [12., 12., 102., 102.],
             [300., 300., 400., 400.]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        classes = np.array([0, 0, 1], dtype=np.int64)

        result = backend._build_result(
            boxes=boxes, max_scores=scores, class_ids=classes,
            orig_shape=(640, 640), image_path=None, iou=0.45,
            classes=None, max_det=300,
        )
        assert "torch" not in sys.modules, "building a Result pulled torch"
        n = len(result.boxes.xyxy)
        assert n == 2, f"expected 2 detections after NMS, got {n}"
        print("OK", n)
        """
    )
    assert "OK 2" in _assert_ok(proc)


def test_numpy_and_torch_nms_agree():
    """Parity guard: the numpy NMS must match torchvision's batched_nms.

    Runs in-process (torch available). If these ever diverge, the torch-free
    path silently returns different detections than the default path.
    """
    from torchvision.ops import batched_nms
    import torch

    from libreyolo.backends.base import _batched_nms_numpy

    rng = np.random.default_rng(0)
    for trial in range(25):
        n = int(rng.integers(1, 60))
        xy = rng.uniform(0, 500, size=(n, 2)).astype(np.float32)
        wh = rng.uniform(5, 120, size=(n, 2)).astype(np.float32)
        boxes = np.concatenate([xy, xy + wh], axis=1).astype(np.float32)
        scores = rng.uniform(0, 1, size=n).astype(np.float32)
        class_ids = rng.integers(0, 3, size=n).astype(np.int64)

        keep_np = _batched_nms_numpy(boxes, scores, class_ids, 0.45)
        keep_pt = batched_nms(
            torch.from_numpy(boxes),
            torch.from_numpy(scores),
            torch.from_numpy(class_ids),
            0.45,
        ).tolist()

        assert keep_np == keep_pt, (
            f"trial {trial}: numpy NMS diverged from torchvision\n"
            f"numpy: {keep_np}\ntorch: {keep_pt}"
        )


@pytest.mark.onnx
@pytest.mark.parametrize("classes", [None, [0]], ids=["all", "class-filter"])
def test_onnx_predict_matches_torch_path_end_to_end(tmp_path, classes):
    """The torch-free ONNX path must return exactly what the torch path returns.

    Exports a small YOLO9 to ONNX (needs torch), then runs the same model
    through ``OnnxBackend.predict`` twice: once normally, once in a subprocess
    with torch blocked. Boxes, scores and classes must match exactly. This is
    the guarantee the lightweight-install docs promise; if the numpy and torch
    branches ever diverge numerically, this fails.

    Parametrised over ``classes`` because the class-filter branch builds its own
    boolean mask and indexes masks/keypoints with it, which is a separate
    torch-conversion path from the unfiltered one.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from libreyolo.models.yolo9.model import LibreYOLO9

    model = LibreYOLO9({}, size="t", nb_classes=4, device="cpu")
    onnx_path = Path(model.export(format="onnx", imgsz=128)).resolve()

    body = f"""
        import json
        import numpy as np
        import libreyolo
        from libreyolo.backends.onnx import OnnxBackend

        backend = OnnxBackend(r"{onnx_path}")
        res = backend.predict(
            libreyolo.SAMPLE_IMAGE, imgsz=128, conf=0.0, max_det=25,
            classes={classes!r},
        )
        res = res[0] if isinstance(res, list) else res
        print("RESULT" + json.dumps({{
            "boxes": np.asarray(res.boxes.xyxy).round(4).tolist(),
            "conf": np.asarray(res.boxes.conf).round(6).tolist(),
            "cls": np.asarray(res.boxes.cls).tolist(),
            "torch": "torch" in sys.modules,
        }}))
        """

    free = json.loads(_assert_ok(run_without_torch(body)).split("RESULT")[1])
    assert free["torch"] is False, "the torch-free run imported torch"
    assert free["boxes"], "expected detections to compare against"

    import libreyolo
    from libreyolo.backends.onnx import OnnxBackend

    res = OnnxBackend(str(onnx_path)).predict(
        libreyolo.SAMPLE_IMAGE, imgsz=128, conf=0.0, max_det=25, classes=classes
    )
    res = res[0] if isinstance(res, list) else res
    ref = {
        "boxes": np.asarray(res.boxes.xyxy).round(4).tolist(),
        "conf": np.asarray(res.boxes.conf).round(6).tolist(),
        "cls": np.asarray(res.boxes.cls).tolist(),
    }

    assert free["boxes"] == ref["boxes"]
    assert free["conf"] == ref["conf"]
    assert free["cls"] == ref["cls"]


# One family per distinct torch-conversion shape on the ONNX path:
#   yolo9  - NMS path, preprocess_numpy returns CHW
#   rtdetr - preprocess_numpy already returns NCHW (as_input, not as_batched_input)
#   dfine  - DETR lineage, NMS-free postprocess
#   rfdetr - ImageNet normalisation constants moved alongside the recipe
_FAMILY_CASES = [
    ("yolo9", "t", 128),
    ("rtdetr", "r18", 128),
    ("dfine", "n", 640),
    ("rfdetr", "n", 128),
]


@pytest.mark.onnx
@pytest.mark.parametrize("family,size,imgsz", _FAMILY_CASES)
def test_g0_g1_families_predict_without_torch(family, size, imgsz):
    """Each G0/G1 preprocess shape must run ONNX detect torch-free, and match.

    All twelve G0/G1 families were verified by hand; these four cover the
    distinct torch-conversion shapes so a regression in any of them is caught
    without exporting twelve models on every run.
    """
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from libreyolo.cli.config import get_model_class

    # `{}` (empty state dict), not None: None makes RF-DETR fetch its DINOv2
    # backbone over HTTP, and the PR gate blocks external network access.
    model = get_model_class(family)({}, size=size, device="cpu")
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz)).resolve()

    body = f"""
        import json
        import numpy as np
        import libreyolo
        from libreyolo.backends.onnx import OnnxBackend

        res = OnnxBackend(r"{onnx_path}").predict(
            libreyolo.SAMPLE_IMAGE, imgsz={imgsz}, conf=0.0, max_det=10
        )
        res = res[0] if isinstance(res, list) else res
        print("RESULT" + json.dumps({{
            "boxes": np.asarray(res.boxes.xyxy).round(3).tolist(),
            "conf": np.asarray(res.boxes.conf).round(5).tolist(),
            "cls": np.asarray(res.boxes.cls).tolist(),
            "torch": "torch" in sys.modules,
        }}))
        """

    free = json.loads(_assert_ok(run_without_torch(body)).split("RESULT")[1])
    assert free["torch"] is False, f"{family}: the torch-free run imported torch"
    assert free["boxes"], f"{family}: expected detections to compare against"

    import libreyolo
    from libreyolo.backends.onnx import OnnxBackend

    res = OnnxBackend(str(onnx_path)).predict(
        libreyolo.SAMPLE_IMAGE, imgsz=imgsz, conf=0.0, max_det=10
    )
    res = res[0] if isinstance(res, list) else res

    assert free["boxes"] == np.asarray(res.boxes.xyxy).round(3).tolist()
    assert free["conf"] == np.asarray(res.boxes.conf).round(5).tolist()
    assert free["cls"] == np.asarray(res.boxes.cls).tolist()

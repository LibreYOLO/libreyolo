"""E2E test configuration and fixtures."""

import gc
import multiprocessing
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_python_env() -> dict[str, str]:
    """Return an env that makes one-shot test scripts import local sources."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    paths = [str(_REPO_ROOT)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env

# ---------------------------------------------------------------------------
# Force 'spawn' multiprocessing to prevent CUDA + fork segfaults.
#
# Export tests (ONNX, TensorRT) initialise the CUDA context.  Training tests
# then create DataLoader workers — with the default 'fork' start method on
# Linux the workers inherit the parent's CUDA context, which is not fork-safe.
# When those workers exit their stale CUDA cleanup can corrupt the parent
# process, producing a segfault (typically exit code 139 / signal 11).
#
# 'spawn' starts workers from scratch so they never inherit CUDA state.
# ---------------------------------------------------------------------------
multiprocessing.set_start_method("spawn", force=True)


def pytest_configure(config):
    """Register custom markers (e2e marker registered in root conftest)."""
    config.addinivalue_line(
        "markers",
        "export_backend: tests covering an export backend or serialized runtime format",
    )
    config.addinivalue_line(
        "markers",
        "supported_backend: tests covering a release-blocking supported backend",
    )
    config.addinivalue_line(
        "markers", "experimental_backend: tests covering an experimental backend"
    )
    config.addinivalue_line("markers", "onnx: tests covering ONNX export or inference")
    config.addinivalue_line(
        "markers", "torchscript: tests covering TorchScript export or inference"
    )
    config.addinivalue_line("markers", "tensorrt: tests requiring TensorRT")
    config.addinivalue_line(
        "markers", "trt: alias for TensorRT tests requiring TensorRT"
    )
    config.addinivalue_line("markers", "openvino: tests requiring OpenVINO")
    config.addinivalue_line("markers", "ncnn: tests requiring ncnn")
    config.addinivalue_line("markers", "yolox: tests covering the YOLOX model family")
    config.addinivalue_line("markers", "yolo9: tests covering the YOLO9 model family")
    config.addinivalue_line(
        "markers", "yolo9_e2e: tests covering the YOLOv9-E2E model family"
    )
    config.addinivalue_line(
        "markers", "yolonas: tests covering the YOLO-NAS model family"
    )
    config.addinivalue_line(
        "markers", "rfdetr: tests covering the RF-DETR model family"
    )
    config.addinivalue_line("markers", "dfine: tests covering the D-FINE model family")
    config.addinivalue_line("markers", "deim: tests covering the DEIM model family")
    config.addinivalue_line(
        "markers", "deimv2: tests covering the DEIMv2 model family"
    )
    config.addinivalue_line("markers", "ecdet: tests covering the ECDet model family")
    config.addinivalue_line(
        "markers", "rtdetr: tests covering the RT-DETR model family"
    )
    config.addinivalue_line("markers", "slow: slow tests that may take several minutes")
    config.addinivalue_line("markers", "rf1: RF1 training tests")
    config.addinivalue_line("markers", "rf5: RF5 training benchmark tests")


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def has_tensorrt():
    """Check if TensorRT is installed and usable."""
    if not has_cuda():
        return False
    try:
        import tensorrt as trt  # noqa: F401

        return True
    except ImportError:
        return False


def has_openvino():
    """Check if OpenVINO is installed and usable."""
    try:
        import openvino as ov

        _ = ov.__version__
        return True
    except ImportError:
        return False


def has_ncnn():
    """Check if ncnn is installed and usable."""
    try:
        import ncnn  # noqa: F401

        return True
    except ImportError:
        return False


def has_rfdetr_deps():
    """Check if RF-DETR dependencies are installed."""
    try:
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Skip decorators
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(not has_cuda(), reason="CUDA not available")

requires_tensorrt = pytest.mark.skipif(
    not has_tensorrt(), reason="TensorRT not installed or CUDA not available"
)

requires_openvino = pytest.mark.skipif(
    not has_openvino(), reason="OpenVINO not installed (pip install openvino)"
)

requires_ncnn = pytest.mark.skipif(
    not has_ncnn(), reason="ncnn not installed (pip install libreyolo[ncnn])"
)

requires_rfdetr = pytest.mark.skipif(
    not has_rfdetr_deps(),
    reason="RF-DETR dependencies not installed (pip install libreyolo[rfdetr])",
)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def cuda_cleanup():
    """Free GPU memory. Call after heavy tests."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Pre-spawned subprocess worker for CUDA isolation
#
# After hundreds of CUDA-heavy export + training tests, the CUDA driver state
# in the main pytest process becomes corrupted.  Any fork() / posix_spawn()
# from this process then triggers SIGSEGV (exit 139).
#
# Fix: fork a *worker* process at import time — before CUDA is ever used —
# so its CUDA context stays pristine.  Later, run_in_subprocess() sends
# commands to the worker via stdin/stdout pipes (no fork from the polluted
# main process).  The worker creates fresh subprocesses from its clean state.
# ---------------------------------------------------------------------------

_WORKER_SCRIPT = r"""
import json, os, subprocess, sys, tempfile

while True:
    line = sys.stdin.readline()
    if not line:
        break
    msg = json.loads(line)
    script_text, timeout = msg["s"], msg["t"]

    fd, path = tempfile.mkstemp(suffix=".py", prefix="ly_")
    os.write(fd, script_text.encode())
    os.close(fd)
    try:
        r = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
        resp = {"rc": r.returncode, "o": r.stdout[-4000:], "e": r.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        resp = {"rc": -1, "o": "", "e": f"Timed out after {timeout}s"}
    except Exception as exc:
        resp = {"rc": -1, "o": "", "e": str(exc)}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
"""


def _start_worker():
    """Start the subprocess worker. Called once at import time."""
    import atexit
    import subprocess as _sp
    import sys as _sys
    import tempfile as _tmp

    fd, path = _tmp.mkstemp(suffix=".py", prefix="ly_worker_")
    import os as _os

    _os.write(fd, _WORKER_SCRIPT.encode())
    _os.close(fd)

    proc = _sp.Popen(
        [_sys.executable, path],
        stdin=_sp.PIPE,
        stdout=_sp.PIPE,
        stderr=_sp.DEVNULL,
        cwd=str(_REPO_ROOT),
        env=_repo_python_env(),
        text=True,
    )

    def _cleanup():
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=5)
        try:
            _os.unlink(path)
        except OSError:
            pass

    atexit.register(_cleanup)
    return proc


_worker_proc = _start_worker()


def run_in_subprocess(script: str, *, timeout: int = 300) -> str:
    """Run Python code in a fresh subprocess via the pre-spawned worker.

    The worker was forked at import time (clean CUDA state) and creates
    child processes on demand — no fork from the polluted main process.
    """
    import json
    import textwrap

    msg = json.dumps({"s": textwrap.dedent(script), "t": timeout})
    _worker_proc.stdin.write(msg + "\n")
    _worker_proc.stdin.flush()

    # Read response (blocking).  The +60 allows the worker to report a
    # timeout rather than us killing it.
    resp_line = _worker_proc.stdout.readline()
    if not resp_line:
        raise RuntimeError(
            "Subprocess worker died unexpectedly.  Check stderr for details."
        )

    resp = json.loads(resp_line)
    if resp["rc"] != 0:
        raise RuntimeError(
            f"Subprocess exited with code {resp['rc']}\n"
            f"--- stdout (last 2000 chars) ---\n{resp['o'][-2000:]}\n"
            f"--- stderr (last 2000 chars) ---\n{resp['e'][-2000:]}"
        )
    return resp["o"]


def run_direct_subprocess(script: str, *, timeout: int = 300) -> str:
    """Run Python code in a one-shot subprocess.

    Use this for families that do not need the long-lived clean worker used by
    ``run_in_subprocess()``.
    """
    import os
    import subprocess
    import sys
    import tempfile
    import textwrap

    fd, path = tempfile.mkstemp(suffix=".py", prefix="ly_direct_")
    os.write(fd, textwrap.dedent(script).encode())
    os.close(fd)
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            cwd=str(_REPO_ROOT),
            env=_repo_python_env(),
            text=True,
            timeout=timeout,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess exited with code {result.returncode}\n"
            f"--- stdout (last 2000 chars) ---\n{result.stdout[-2000:]}\n"
            f"--- stderr (last 2000 chars) ---\n{result.stderr[-2000:]}"
        )
    return result.stdout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cuda_device():
    """Return CUDA device string if available."""
    if has_cuda():
        return "cuda"
    pytest.skip("CUDA not available")


@pytest.fixture(scope="session")
def gpu_info():
    """Return GPU information for logging."""
    if not has_cuda():
        return {"available": False}

    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }


@pytest.fixture(scope="session")
def tensorrt_version():
    """Return TensorRT version if available."""
    if not has_tensorrt():
        pytest.skip("TensorRT not available")

    import tensorrt as trt

    return trt.__version__


@pytest.fixture(scope="session")
def sample_image():
    """Get a sample image for inference tests."""
    from libreyolo import SAMPLE_IMAGE

    return SAMPLE_IMAGE


@pytest.fixture(scope="function")
def temp_export_dir(tmp_path):
    """Create a temporary directory for export artifacts."""
    return tmp_path / "exports"


@pytest.fixture(autouse=True, scope="function")
def cleanup_gpu_memory():
    """Clear GPU memory before and after each test to prevent state corruption."""
    yield
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="class")
def reset_gpu_state():
    """Force GPU state reset between test classes (useful for RF-DETR training)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# Model catalog — single source of truth
# ---------------------------------------------------------------------------

# (family, size, weights)
MODEL_CATALOG = [
    ("yolox", "n", "LibreYOLOXn.pt"),
    ("yolox", "t", "LibreYOLOXt.pt"),
    ("yolox", "s", "LibreYOLOXs.pt"),
    ("yolox", "m", "LibreYOLOXm.pt"),
    ("yolox", "l", "LibreYOLOXl.pt"),
    ("yolox", "x", "LibreYOLOXx.pt"),
    ("yolo9", "t", "LibreYOLO9t.pt"),
    ("yolo9", "s", "LibreYOLO9s.pt"),
    ("yolo9", "m", "LibreYOLO9m.pt"),
    ("yolo9", "c", "LibreYOLO9c.pt"),
    ("yolo9_e2e", "t", "LibreYOLO9E2Et.pt"),
    ("yolo9_e2e", "s", "LibreYOLO9E2Es.pt"),
    ("yolo9_e2e", "m", "LibreYOLO9E2Em.pt"),
    ("yolo9_e2e", "c", "LibreYOLO9E2Ec.pt"),
    ("yolonas", "s", "downloads/yolonas/yolo_nas_s_coco.pth"),
    ("yolonas", "m", "downloads/yolonas/yolo_nas_m_coco.pth"),
    ("yolonas", "l", "downloads/yolonas/yolo_nas_l_coco.pth"),
    ("rfdetr", "n", "LibreRFDETRn.pt"),
    ("rfdetr", "s", "LibreRFDETRs.pt"),
    ("rfdetr", "m", "LibreRFDETRm.pt"),
    ("rfdetr", "l", "LibreRFDETRl.pt"),
    ("dfine", "n", "LibreDFINEn.pt"),
    ("dfine", "s", "LibreDFINEs.pt"),
    ("dfine", "m", "LibreDFINEm.pt"),
    ("dfine", "l", "LibreDFINEl.pt"),
    ("dfine", "x", "LibreDFINEx.pt"),
    ("deim", "n", "weights/LibreDEIMn.pt"),
    ("deim", "s", "weights/LibreDEIMs.pt"),
    ("deim", "m", "weights/LibreDEIMm.pt"),
    ("deim", "l", "weights/LibreDEIMl.pt"),
    ("deim", "x", "weights/LibreDEIMx.pt"),
    ("deimv2", "atto", "LibreDEIMv2Atto.pt"),
    ("deimv2", "femto", "LibreDEIMv2Femto.pt"),
    ("deimv2", "pico", "LibreDEIMv2Pico.pt"),
    ("deimv2", "n", "LibreDEIMv2n.pt"),
    ("deimv2", "s", "LibreDEIMv2s.pt"),
    ("deimv2", "m", "LibreDEIMv2m.pt"),
    ("deimv2", "l", "LibreDEIMv2l.pt"),
    ("deimv2", "x", "LibreDEIMv2x.pt"),
    ("ecdet", "s", "LibreECDETs.pt"),
    ("ecdet", "m", "LibreECDETm.pt"),
    ("ecdet", "l", "LibreECDETl.pt"),
    ("ecdet", "x", "LibreECDETx.pt"),
    ("rtdetr", "r18", "LibreRTDETRr18.pt"),
    ("rtdetr", "r34", "LibreRTDETRr34.pt"),
    ("rtdetr", "r50", "LibreRTDETRr50.pt"),
    ("rtdetr", "r50m", "LibreRTDETRr50m.pt"),
    ("rtdetr", "r101", "LibreRTDETRr101.pt"),
    ("picodet", "s", "LibrePicoDets.pt"),
    ("picodet", "m", "LibrePicoDetm.pt"),
    ("picodet", "l", "LibrePicoDetl.pt"),
]

# Derived lists (no manual maintenance)
YOLOX_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "yolox"]
YOLO9_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "yolo9"]
YOLO9E2E_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "yolo9_e2e"]
YOLONAS_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "yolonas"]
RFDETR_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "rfdetr"]
DFINE_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "dfine"]
DEIM_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "deim"]
DEIMV2_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "deimv2"]
ECDET_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "ecdet"]
RTDETR_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "rtdetr"]
PICODET_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "picodet"]

ALL_MODELS = [(f, s) for f, s, _ in MODEL_CATALOG]
ALL_MODELS_WITH_WEIGHTS = MODEL_CATALOG
NON_RFDETR_MODELS = [(f, s) for f, s, _ in MODEL_CATALOG if f != "rfdetr"]

# Quick test set (for CI — smallest auto-available models only)
QUICK_TEST_MODELS = [("yolox", "n"), ("yolo9", "t"), ("rtdetr", "r18")]

# Full test set (all non-RF-DETR models)
FULL_TEST_MODELS = NON_RFDETR_MODELS

# RF-DETR test set (separate due to dependency)
RFDETR_TEST_MODELS = [(f, s) for f, s, _ in MODEL_CATALOG if f == "rfdetr"]

# RT-DETR test set (separate due to being transformer-based)
RTDETR_TEST_MODELS = [(f, s) for f, s, _ in MODEL_CATALOG if f == "rtdetr"]

FAMILY_MARKERS = {
    "yolox": pytest.mark.yolox,
    "yolo9": pytest.mark.yolo9,
    "yolo9_e2e": pytest.mark.yolo9_e2e,
    "yolonas": pytest.mark.yolonas,
    "rfdetr": pytest.mark.rfdetr,
    "dfine": pytest.mark.dfine,
    "deim": pytest.mark.deim,
    "deimv2": pytest.mark.deimv2,
    "ecdet": pytest.mark.ecdet,
    "rtdetr": pytest.mark.rtdetr,
    "picodet": pytest.mark.picodet,
}


def _normalize_marks(marks):
    """Normalize a mark or a collection of marks to a flat list."""
    if marks is None:
        return []
    if isinstance(marks, (list, tuple, set)):
        return [mark for mark in marks if mark is not None]
    return [marks]


def family_marks(family: str, marks=None):
    """Return pytest marks for a model family plus any extra marks."""
    return [FAMILY_MARKERS[family], *_normalize_marks(marks)]


def model_case(family: str, size: str, *, weights: str | None = None, marks=None):
    """Build a parametrized model case with family markers attached."""
    values = (family, size) if weights is None else (family, size, weights)
    return pytest.param(
        *values, marks=family_marks(family, marks), id=f"{family}-{size}"
    )


def model_cases(models, *, with_weights: bool = False, marks_resolver=None):
    """Attach family markers to a model matrix used in parametrized tests."""
    params = []
    for family, size, *rest in models:
        weights = rest[0] if with_weights else None
        marks = marks_resolver(family, size, *rest) if marks_resolver else None
        params.append(model_case(family, size, weights=weights, marks=marks))
    return params


QUICK_TEST_PARAMS = model_cases(QUICK_TEST_MODELS)
FULL_TEST_PARAMS = model_cases(FULL_TEST_MODELS)
RFDETR_TEST_PARAMS = model_cases(RFDETR_TEST_MODELS)
ALL_MODEL_WEIGHT_PARAMS = model_cases(ALL_MODELS_WITH_WEIGHTS, with_weights=True)


def get_model_weights(family: str, size: str) -> str:
    """Get the weight file name for a model family and size."""
    for f, s, w in MODEL_CATALOG:
        if f == family and s == size:
            return w
    raise ValueError(f"Unknown model: {family}-{size}")


@lru_cache(maxsize=None)
def _detect_local_weights_family(weights: str) -> str:
    """Detect a local checkpoint's family for skip-only environment validation."""
    from libreyolo import LibreYOLO

    model = LibreYOLO(weights)
    return model.FAMILY


def require_test_weights(weights: str, expected_family: str | None = None) -> str:
    """Skip cleanly if a test depends on missing or obviously wrong local weights."""
    path = Path(weights)
    if path.parent != Path("."):
        if not path.exists():
            pytest.skip(f"Required local weights not found: {weights}")
        if expected_family is not None:
            try:
                detected_family = _detect_local_weights_family(str(path))
            except Exception as exc:
                pytest.skip(
                    f"Local weights are unusable for testing: {weights} ({exc})"
                )
            if detected_family != expected_family:
                pytest.skip(
                    "Local weights do not match the expected family: "
                    f"{weights} detected as '{detected_family}', expected '{expected_family}'"
                )
    return weights


def make_ids(models):
    """Generate test IDs like 'yolox-n' from (family, size, ...) tuples."""
    return [f"{m[0]}-{m[1]}" for m in models]


# ---------------------------------------------------------------------------
# Common utility functions for E2E tests
# ---------------------------------------------------------------------------


def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format.

    Simple Python implementation for test utilities (no torch dependency).
    For GPU-accelerated IoU, see libreyolo.utils.box_ops.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def match_detections(results1, results2, iou_threshold=0.5):
    """
    Match detections between two result sets.

    Returns:
        Tuple of (match_rate, matched_count, total_count)
    """
    if len(results1) == 0 and len(results2) == 0:
        return 1.0, 0, 0  # Perfect match (both empty)

    if len(results1) == 0 or len(results2) == 0:
        return 0.0, 0, max(len(results1), len(results2))

    boxes1 = results1.boxes.xyxy.cpu().numpy()
    classes1 = results1.boxes.cls.cpu().numpy()

    boxes2 = results2.boxes.xyxy.cpu().numpy()
    classes2 = results2.boxes.cls.cpu().numpy()

    matched = 0
    for box1, cls1 in zip(boxes1, classes1):
        for box2, cls2 in zip(boxes2, classes2):
            if cls1 == cls2:
                iou = compute_iou(box1, box2)
                if iou >= iou_threshold:
                    matched += 1
                    break

    total = max(len(boxes1), len(boxes2))
    match_rate = matched / total if total > 0 else 1.0

    return match_rate, matched, total


def load_model(model_type: str, size: str, device: str = "cuda"):
    """Load a model by type and size."""
    from libreyolo import LibreYOLO

    weights = require_test_weights(
        get_model_weights(model_type, size),
        expected_family=model_type,
    )
    return LibreYOLO(weights, device=device)


def results_are_acceptable(
    match_rate: float, count1: int, count2: int, threshold: float = 0.7
) -> bool:
    """
    Check if results are acceptable.

    Args:
        match_rate: Detection match rate
        count1: Number of detections in first result
        count2: Number of detections in second result
        threshold: Minimum acceptable match rate (default 0.7)
    """
    if match_rate >= threshold:
        return True

    det_diff = abs(count1 - count2)
    if det_diff <= 2 and count1 > 0:
        return True

    if count1 == 0 and count2 == 0:
        return True

    return False


# ---------------------------------------------------------------------------
# Shared export test helpers
# ---------------------------------------------------------------------------


def run_export_compare_test(
    model_type,
    size,
    sample_image,
    tmp_path,
    format,
    export_kwargs=None,
    match_threshold=0.7,
    device="cpu",
):
    """
    Common export -> inference -> compare flow.

    Returns (exported_path, pt_results, export_results).
    """
    from libreyolo import LibreYOLO

    pt_model = load_model(model_type, size, device=device)
    pt_results = pt_model(sample_image, conf=0.25)

    export_path = str(tmp_path / f"{model_type}_{size}.{format}")
    exported_path = pt_model.export(
        format=format,
        output_path=export_path,
        **(export_kwargs or {}),
    )

    exported_model = LibreYOLO(exported_path, device=device)
    export_results = exported_model(sample_image, conf=0.25)

    match_rate, matched, total = match_detections(pt_results, export_results)
    assert results_are_acceptable(
        match_rate,
        len(pt_results),
        len(export_results),
        threshold=match_threshold,
    ), (
        f"Results mismatch: PT={len(pt_results)}, {format}={len(export_results)}, "
        f"matched={matched}/{total}, rate={match_rate:.2%}"
    )

    return exported_path, pt_results, export_results


def run_consistency_test(
    model_type,
    size,
    sample_image,
    tmp_path,
    format,
    export_kwargs=None,
    device="cpu",
    n_runs=5,
):
    """Export model and verify consistent inference results across N runs."""
    from libreyolo import LibreYOLO

    pt_model = load_model(model_type, size, device=device)
    export_path = str(tmp_path / f"{model_type}_{size}.{format}")
    exported_path = pt_model.export(
        format=format, output_path=export_path, **(export_kwargs or {})
    )

    model = LibreYOLO(exported_path, device=device)
    results = [len(model(sample_image, conf=0.25)) for _ in range(n_runs)]
    assert len(set(results)) == 1, f"Inconsistent results across runs: {results}"


def run_metadata_round_trip_test(
    model_type,
    size,
    tmp_path,
    format,
    export_kwargs=None,
    device="cpu",
):
    """Export model and verify metadata is preserved when loading."""
    from libreyolo import LibreYOLO

    pt_model = load_model(model_type, size, device=device)
    export_path = str(tmp_path / f"{model_type}_{size}.{format}")
    exported_path = pt_model.export(
        format=format, output_path=export_path, **(export_kwargs or {})
    )

    loaded_model = LibreYOLO(exported_path, device=device)
    assert loaded_model.nb_classes == pt_model.nb_classes
    assert loaded_model.names == pt_model.names

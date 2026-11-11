"""End-to-end Axelera Metis tests: real SDK, real compile, real device.

Skipped unless ``axelera-devkit`` and ``axelera-rt`` import and a Metis device
node exists. Each export compiles for tens of minutes, so the module compiles
YOLO-NAS-s detect once and reuses it; the CLI test compiles its own.

Run with::

    pytest tests/e2e/test_axelera.py -m axelera -v -p no:cacheprovider
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.extended_backend,
    pytest.mark.axelera,
    pytest.mark.yolonas,
    # Compiles take tens of minutes; the default e2e timeout is 900 s.
    pytest.mark.timeout(7200),
]

# mAP50-95 floor for on-device val of YOLO-NAS-s on coco128: measured value
# minus 2 points (see docs/axelera.md, section 7). None until recorded.
COCO128_MAP_FLOOR: float | None = None


def _sdk_and_device() -> str | None:
    for module in ("axelera.compiler", "axelera.runtime"):
        try:
            if importlib.util.find_spec(module) is None:
                return f"{module} is not installed"
        except (ImportError, ValueError):
            return f"{module} is not installed"
    if not sys.platform.startswith("linux") or not list(Path("/dev").glob("metis-*:*")):
        return "no Metis device found"
    return None


_missing = _sdk_and_device()
if _missing:
    pytestmark.append(pytest.mark.skip(reason=f"Axelera e2e needs the SDK and a device: {_missing}"))


@pytest.fixture(scope="module")
def detect_export(tmp_path_factory):
    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLONASs.pt")
    out = tmp_path_factory.mktemp("axelera") / "LibreYOLONASs_axelera_model"
    # Half of coco128 (64 images) keeps the e2e compile short.
    path = model.export(format="axelera", data="coco128.yaml", fraction=0.5, output_path=str(out), verify=True)
    return model, Path(path)


def test_export_layout_and_metadata(detect_export):
    import yaml

    _, path = detect_export
    names = sorted(p.name for p in path.iterdir())
    assert names == [
        "LibreYOLONASs.axm", "LibreYOLONASs.parity.json",
        "compiler_config_final.toml", "metadata.yaml",
    ]
    meta = yaml.safe_load((path / "metadata.yaml").read_text())
    assert meta["format"] == "axelera" and meta["precision"] == "int8"
    assert meta["output_layout"] == "yolonas_raw_heads_v1"
    report = json.loads((path / "LibreYOLONASs.parity.json").read_text())
    assert isinstance(report["device_vs_quantized"], list)
    # T4: the device and the CPU quantized model agree. Measured lowest
    # per-output cosines: 0.956 (YOLO-NAS-l detect), 0.971 (YOLO-NAS-s pose),
    # with mAP within 0.3 points of FP32; a wrong output mapping drops far
    # below 0.9.
    assert min(row["min_cosine"] for row in report["device_vs_quantized"]) >= 0.90


def test_predict_finds_people(detect_export, sample_image):
    from libreyolo import LibreYOLO

    model, path = detect_export
    npu = LibreYOLO(str(path))
    assert type(npu).__name__ == "AxeleraBackend"
    result = npu(sample_image, conf=0.25)
    names = [npu.names[int(c)] for c in result.boxes.cls]
    assert "person" in names
    h, w = result.orig_shape
    boxes = np.asarray(result.boxes.xyxy)
    assert (boxes[:, [0, 2]] >= 0).all() and (boxes[:, [0, 2]] <= w).all()
    assert (boxes[:, [1, 3]] >= 0).all() and (boxes[:, [1, 3]] <= h).all()
    # Same people as the PyTorch model at a confident threshold.
    reference = model(sample_image, conf=0.5)
    on_device = npu(sample_image, conf=0.5)
    assert abs(len(on_device) - len(reference)) <= max(1, len(reference) // 5)


def test_batched_predict_matches_single(detect_export, sample_image):
    from libreyolo import LibreYOLO

    _, path = detect_export
    npu = LibreYOLO(str(path))
    batched = npu.predict([sample_image, sample_image], conf=0.25, batch=2)
    single = npu(sample_image, conf=0.25)
    for result in batched:
        np.testing.assert_allclose(result.boxes.xyxy, single.boxes.xyxy, atol=1e-3)


def test_val_on_device(detect_export):
    from libreyolo import LibreYOLO

    _, path = detect_export
    metrics = LibreYOLO(str(path)).val(data="coco128.yaml", batch=8, plots=False)
    score = float(metrics["metrics/mAP50-95(B)"])
    assert score > 0.0
    if COCO128_MAP_FLOOR is not None:
        assert score >= COCO128_MAP_FLOOR


@pytest.mark.skip(
    reason="track() is implemented on native models only; exported backends "
    "(every format) have no track() yet. Maintainer decision pending."
)
def test_track_short_video(detect_export, sample_image, tmp_path):
    import cv2

    from libreyolo import LibreYOLO

    _, path = detect_export
    frame = cv2.imread(sample_image)
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (frame.shape[1], frame.shape[0]))
    for shift in range(6):
        writer.write(np.roll(frame, shift * 4, axis=1))
    writer.release()
    results = list(LibreYOLO(str(path)).track(str(video), conf=0.25, stream=True))
    assert len(results) == 6
    assert any(r.boxes.id is not None and len(r.boxes.id) for r in results)


def test_cli_round_trip(tmp_path, sample_image):
    """libreyolo export -> predict -> val, in a scratch working directory."""
    import shutil

    cli = shutil.which("libreyolo")
    assert cli, "the libreyolo console script is not on PATH"

    def run(*args):
        done = subprocess.run(
            [cli, *args, "--json"], cwd=tmp_path, capture_output=True, text=True, check=False
        )
        assert done.returncode == 0, done.stderr[-2000:]
        return json.loads(done.stdout)

    exported = run("export", "model=LibreYOLONASs.pt", "format=axelera", "fraction=0.5")
    artifact = Path(tmp_path, exported["output_path"])
    assert artifact.name.endswith("_axelera_model") and artifact.is_dir()
    assert exported["int8"] is True and exported["target"] == "metis"
    run("predict", f"model={artifact}", f"source={sample_image}")
    run("val", f"model={artifact}", "data=coco128.yaml")


def test_pose_s_export_is_refused(tmp_path):
    """YOLO-NAS-pose-s misses the 1.0-point bar on Metis (m and l pass)."""
    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLONASs-pose.pt")
    with pytest.raises(NotImplementedError, match="use m or l"):
        model.export(format="axelera", data="coco128.yaml", output_path=str(tmp_path / "pose_axelera_model"))

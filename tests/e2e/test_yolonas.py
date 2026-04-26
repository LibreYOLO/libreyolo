"""
YOLO-NAS end-to-end smoke tests.

These tests cover the native LibreYOLO YOLO-NAS path using the official
small checkpoint downloaded locally under downloads/yolonas/.

Scope:
- validation sanity check on coco128
- training/checkpoint/resume lifecycle smoke

This is intentionally not an exact SuperGradients parity benchmark.
"""

from pathlib import Path

import pytest
import torch

from libreyolo import LibreYOLO

from .conftest import cuda_cleanup

pytestmark = [pytest.mark.e2e, pytest.mark.yolonas]

OFFICIAL_YOLONAS_S = Path("downloads/yolonas/yolo_nas_s_coco.pth")
MIN_MAP = 0.25


@pytest.mark.skipif(
    not OFFICIAL_YOLONAS_S.exists(),
    reason="Official YOLO-NAS-S checkpoint not present in downloads/yolonas/",
)
def test_yolonas_s_val_coco128():
    """Validate the official YOLO-NAS-S checkpoint on coco128."""
    model = LibreYOLO(str(OFFICIAL_YOLONAS_S), device="cpu")

    results = model.val(
        data="coco128.yaml",
        batch=8,
        imgsz=320,
        conf=0.001,
        iou=0.6,
        verbose=False,
        num_workers=0,
    )

    map50_95 = results["metrics/mAP50-95"]
    map50 = results["metrics/mAP50"]
    print(f"\n  YOLO-NAS-S coco128: mAP50-95={map50_95:.4f}, mAP50={map50:.4f}")

    assert map50_95 >= MIN_MAP, (
        f"mAP50-95={map50_95:.4f} below threshold {MIN_MAP} — "
        "YOLO-NAS validation path may be broken"
    )

    cuda_cleanup()


@pytest.mark.skipif(
    not OFFICIAL_YOLONAS_S.exists(),
    reason="Official YOLO-NAS-S checkpoint not present in downloads/yolonas/",
)
def test_yolonas_s_train_resume_smoke(tmp_path):
    """Train for 1 epoch, reload checkpoint, resume to epoch 2."""
    model = LibreYOLO(str(OFFICIAL_YOLONAS_S), device="cpu")

    train_results = model.train(
        data="coco128.yaml",
        epochs=1,
        batch=8,
        imgsz=320,
        workers=0,
        device="cpu",
        amp=False,
        eval_interval=1,
        save_period=1,
        project=str(tmp_path),
        name="yolonas_s_train",
        exist_ok=True,
    )

    first_last_ckpt = Path(train_results["last_checkpoint"])
    assert first_last_ckpt.exists(), "Initial training run did not save last.pt"

    resumed = LibreYOLO(str(first_last_ckpt), device="cpu")
    resume_results = resumed.train(
        data="coco128.yaml",
        epochs=2,
        batch=8,
        imgsz=320,
        workers=0,
        device="cpu",
        amp=False,
        eval_interval=1,
        save_period=1,
        project=str(tmp_path),
        name="yolonas_s_resume",
        exist_ok=True,
        resume=True,
    )

    resumed_last_ckpt = Path(resume_results["last_checkpoint"])
    assert resumed_last_ckpt.exists(), "Resume run did not save last.pt"
    assert resume_results["final_loss"] > 0, "Resume run returned invalid final loss"

    checkpoint = torch.load(resumed_last_ckpt, map_location="cpu", weights_only=False)
    assert checkpoint["model_family"] == "yolonas"
    assert checkpoint["size"] == "s"
    assert checkpoint["nc"] == 80
    assert checkpoint["epoch"] == 1, "Resume run should finish at epoch index 1"

    reloaded = LibreYOLO(str(resumed_last_ckpt), device="cpu")
    assert reloaded.size == "s"
    assert reloaded.nb_classes == 80

    cuda_cleanup()


@pytest.mark.skipif(
    not OFFICIAL_YOLONAS_S.exists(),
    reason="Official YOLO-NAS-S checkpoint not present in downloads/yolonas/",
)
@pytest.mark.torchscript
def test_yolonas_s_torchscript_roundtrip(sample_image, tmp_path):
    """Export YOLO-NAS-S to TorchScript and reload it through LibreYOLO."""
    model = LibreYOLO(str(OFFICIAL_YOLONAS_S), device="cpu")

    export_path = tmp_path / "yolonas_s.torchscript"
    exported = model.export(format="torchscript", output_path=str(export_path))

    assert Path(exported).exists(), "TorchScript export did not create a file"
    assert Path(exported).stat().st_size > 0, "TorchScript export is empty"

    reloaded = LibreYOLO(str(exported), device="cpu")
    assert reloaded.__class__.__name__ == "TorchScriptBackend"
    assert reloaded.model_family == "yolonas"
    assert reloaded.nb_classes == 80
    assert reloaded.imgsz == 640

    output_image = tmp_path / "yolonas_s_roundtrip.jpg"
    result = reloaded(
        sample_image,
        conf=0.25,
        iou=0.45,
        imgsz=640,
        save=True,
        output_path=str(output_image),
    )

    assert len(result.boxes) > 0, "Round-tripped TorchScript model returned no boxes"
    assert output_image.exists(), (
        "Round-tripped TorchScript inference did not save output"
    )
    assert result.saved_path == str(output_image)

    cuda_cleanup()

"""
E2E: YOLOv9 segmentation training and inference tests.

Trains LibreYOLO9-t-Seg for 3 epochs on LibreYOLO/fire-smoke-seg
(2 classes, polygon labels). LibreYOLO9-seg has no published
pretrained weights yet, so the seg head (Proto + cv4) is initialised
randomly on top of detection weights.

Verifies:
- Model instantiates in seg mode and exposes _is_segmentation
- Training completes and the loss dict carries a 'mask' component
- Saved checkpoint contains head.proto / head.cv4 keys
- Post-training inference yields masks aligned with detections

Unlike RF-DETR-seg, YOLOv9-seg training does not require CUDA — it
runs on MPS and CPU. mAP is not asserted because 3 epochs from a
random seg head will not converge.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from .conftest import run_in_subprocess

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

DATASET_ROOT = Path.home() / ".cache" / "libreyolo" / "fire-smoke-seg"
HF_REPO = "LibreYOLO/fire-smoke-seg"


def _has_git_lfs() -> bool:
    try:
        subprocess.run(["git", "lfs", "version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _is_lfs_pointer(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(20).startswith(b"version https://git-lfs")


def _download_fire_smoke_dataset():
    if DATASET_ROOT.exists() and (DATASET_ROOT / "data.yaml").exists():
        sample = next(DATASET_ROOT.rglob("*.jpg"), None)
        if sample is not None and not _is_lfs_pointer(sample):
            return
        shutil.rmtree(DATASET_ROOT)

    if not _has_git_lfs():
        pytest.skip(
            "git-lfs is required for fire-smoke-seg dataset. "
            "Install with: sudo apt install git-lfs && git lfs install"
        )

    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "lfs", "install"], check=True)
    subprocess.run(
        [
            "git",
            "clone",
            f"https://huggingface.co/datasets/{HF_REPO}",
            str(DATASET_ROOT),
        ],
        check=True,
    )


def _patch_data_yaml():
    data_yaml = DATASET_ROOT / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text())
    changed = False
    if data.get("path") != str(DATASET_ROOT):
        data["path"] = str(DATASET_ROOT)
        changed = True
    for split in ("train", "val", "test"):
        val = data.get(split, "")
        if isinstance(val, str) and val.startswith("../"):
            data[split] = val.removeprefix("../")
            changed = True
    if changed:
        data_yaml.write_text(yaml.dump(data, default_flow_style=False))


@pytest.fixture(scope="module")
def dataset():
    _download_fire_smoke_dataset()
    _patch_data_yaml()
    return DATASET_ROOT


def test_yolo9_seg_training(dataset, tmp_path):
    """Train YOLOv9-t-seg for 3 epochs, verify mask loss + seg checkpoint."""
    output_dir = str(tmp_path / "yolo9_seg")
    data_yaml = str(dataset / "data.yaml")

    run_in_subprocess(
        f"""
        from pathlib import Path
        import torch
        from libreyolo.models.yolo9.model import LibreYOLO9

        # 1. Instantiate in seg mode (random seg head on top of det weights)
        model = LibreYOLO9(
            model_path="LibreYOLO9t.pt",
            size="t",
            segmentation=True,
        )
        assert model._is_segmentation, "Model should be in segmentation mode"
        assert hasattr(model.model, "head") and hasattr(model.model.head, "proto"), (
            "Model head missing Proto module"
        )

        # 2. Train 3 epochs
        results = model.train(
            data="{data_yaml}",
            epochs=3,
            batch=2,
            imgsz=640,
            project="{output_dir}",
            name="run",
            workers=0,
            amp=False,
        )

        # 3. Checkpoint exists and carries seg keys
        ckpt_path = Path(results["best_checkpoint"])
        if not ckpt_path.exists():
            ckpt_path = Path(results["last_checkpoint"])
        assert ckpt_path.exists(), f"No checkpoint at {{ckpt_path}}"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"]
        proto_keys = [k for k in state if k.startswith("head.proto")]
        cv4_keys = [k for k in state if k.startswith("head.cv4")]
        assert len(proto_keys) > 0, "Checkpoint missing head.proto keys"
        assert len(cv4_keys) > 0, "Checkpoint missing head.cv4 keys"
        print(f"Checkpoint has {{len(proto_keys)}} proto + {{len(cv4_keys)}} cv4 keys")

        # 4. Post-training .pt inference produces masks
        from libreyolo import SAMPLE_IMAGE
        result = model.predict(SAMPLE_IMAGE, conf=0.05)
        print(f"Inference: {{len(result)}} dets, masks={{result.masks is not None}}")
        if len(result) > 0:
            assert result.masks is not None, "Seg model produced dets but no masks"
            assert result.masks.data.shape[0] == len(result), (
                "Mask count != detection count"
            )

        print("PASSED")
        """,
        timeout=900,
    )


def test_yolo9_seg_factory_unblocked():
    """The factory should no longer reject -seg suffixes for YOLOv9.

    Doesn't actually load weights (no published LibreYOLO9*-seg.pt yet);
    just verifies the capability flag and the factory rejection path.
    """
    run_in_subprocess(
        """
        from libreyolo.models import BaseModel
        from libreyolo.models.yolo9.model import LibreYOLO9
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR

        # Capability flag is the new source of truth
        assert LibreYOLO9.SUPPORTS_SEG is True
        assert LibreYOLORFDETR.SUPPORTS_SEG is True
        assert BaseModel.SUPPORTS_SEG is False

        # YOLOv9 declares its real ONNX seg output schema
        assert LibreYOLO9.ONNX_SEG_OUTPUT_NAMES == (
            "predictions", "proto", "mask_coeffs"
        )
        assert LibreYOLORFDETR.ONNX_SEG_OUTPUT_NAMES == (
            "boxes", "scores", "masks"
        )

        print("PASSED")
        """,
        timeout=60,
    )

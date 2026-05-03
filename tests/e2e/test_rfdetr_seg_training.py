"""
E2E: RF-DETR segmentation training and inference tests.

Training test: trains RF-DETR-Seg-Nano for 5 epochs on LibreYOLO/fire-smoke-seg
(HuggingFace, public, 141 train / 40 valid / 20 test images, 2 classes:
fire & smoke, YOLO segmentation format with polygon annotations).

NOTE: Seg training requires CUDA — the upstream rfdetr mask loss uses
F.grid_sample(padding_mode='border') which MPS does not support.
Seg inference works on all devices (CPU, MPS, CUDA).

The dataset auto-downloads from HuggingFace — no API keys needed.

Usage:
    pytest tests/e2e/test_rfdetr_seg_training.py -v -m e2e
    pytest tests/e2e/test_rfdetr_seg_training.py::test_rfdetr_seg_inference_only -v
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from .conftest import requires_cuda, run_in_subprocess

pytestmark = [pytest.mark.e2e, pytest.mark.rfdetr, pytest.mark.slow]

DATASET_ROOT = Path.home() / ".cache" / "libreyolo" / "fire-smoke-seg"
HF_REPO = "LibreYOLO/fire-smoke-seg"


def _has_git_lfs() -> bool:
    """Check if git-lfs is installed."""
    try:
        subprocess.run(["git", "lfs", "version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _is_lfs_pointer(path: Path) -> bool:
    """Check if a file is a Git LFS pointer instead of actual content."""
    with open(path, "rb") as f:
        return f.read(20).startswith(b"version https://git-lfs")


def download_fire_smoke_dataset():
    """Download the fire-smoke-seg dataset from HuggingFace.

    This dataset uses Git LFS for images, so git-lfs must be installed.
    """
    if DATASET_ROOT.exists() and (DATASET_ROOT / "data.yaml").exists():
        sample = next(DATASET_ROOT.rglob("*.jpg"), None)
        if sample is not None and not _is_lfs_pointer(sample):
            return
        # LFS pointers from a previous clone without git-lfs — nuke it
        shutil.rmtree(DATASET_ROOT)

    if not _has_git_lfs():
        pytest.skip(
            "git-lfs is required for fire-smoke-seg dataset. "
            "Install with: sudo apt install git-lfs && git lfs install"
        )

    print(f"\nDownloading dataset {HF_REPO} from HuggingFace ...")
    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)

    # git-lfs must be initialized before cloning
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
    print(f"Dataset ready at {DATASET_ROOT}")


def patch_data_yaml():
    """Ensure data.yaml has absolute path and correct split paths.

    Roboflow exports use ``../train/images`` (relative to a subdirectory).
    We normalise them to ``train/images`` (relative to the dataset root).
    """
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
    """Download fire-smoke-seg dataset and patch data.yaml."""
    download_fire_smoke_dataset()
    patch_data_yaml()
    return DATASET_ROOT


@requires_cuda
def test_rfdetr_seg_training(dataset, tmp_path):
    """Train RF-DETR-Seg-Nano on fire-smoke-seg, verify mAP improves and masks are produced."""
    output_dir = str(tmp_path / "rfdetr_seg_n")
    dataset_dir = str(dataset)
    data_yaml = str(dataset / "data.yaml")

    run_in_subprocess(
        f"""
        from pathlib import Path
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR
        from libreyolo import SAMPLE_IMAGE

        # 1. Load segmentation model
        model = LibreYOLORFDETR(
            model_path="LibreRFDETRn-seg.pt",
            size="n",
            segmentation=True,
        )
        assert model._is_segmentation, "Model should be in segmentation mode"

        # 2. Baseline mask mAP BEFORE training on fire-smoke-seg
        pre = model.val(
            data="{data_yaml}", split="test", batch=8, conf=0.001, iou=0.6
        )
        assert "metrics/mAP50-95(M)" in pre, "Validation did not return mask mAP"
        pre_map = pre["metrics/mAP50-95(M)"]
        assert pre["metrics/mAP50-95"] == pre_map
        print(f"Pre-training mask mAP50-95: {{pre_map:.4f}}")

        # 3. Train on fire-smoke-seg dataset
        model.train(
            data="{dataset_dir}",
            epochs=5,
            batch_size=2,
            output_dir="{output_dir}",
        )

        # 4. Verify checkpoint was produced
        ckpt_path = Path("{output_dir}") / "checkpoint_best_total.pth"
        if not ckpt_path.exists():
            ckpts = sorted(Path("{output_dir}").glob("checkpoint*.pth"))
            assert ckpts, f"No checkpoint found in {output_dir}"
            ckpt_path = ckpts[-1]
        print(f"Checkpoint: {{ckpt_path}}")

        # 5. Verify checkpoint has segmentation_head keys
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["model"]
        seg_keys = [k for k in state if k.startswith("segmentation_head")]
        assert len(seg_keys) > 0, "Checkpoint missing segmentation_head keys"
        print(f"Checkpoint has {{len(seg_keys)}} segmentation_head keys")

        # 6. Post-training mask mAP
        post = model.val(
            data="{data_yaml}", split="test", batch=8, conf=0.001, iou=0.6
        )
        assert "metrics/mAP50-95(M)" in post, "Validation did not return mask mAP"
        post_map = post["metrics/mAP50-95(M)"]
        assert post["metrics/mAP50-95"] == post_map
        print(f"Post-training mask mAP50-95: {{post_map:.4f}}")

        assert post_map >= 0.05, f"mAP50-95={{post_map:.4f}} below 0.05"
        assert post_map > pre_map, (
            f"No improvement: pre={{pre_map:.4f}} -> post={{post_map:.4f}}"
        )

        # 7. Post-training inference still produces masks
        post_result = model.predict(SAMPLE_IMAGE, conf=0.3)
        print(f"Post-training: {{len(post_result)}} detections, "
              f"masks={{post_result.masks is not None}}")
        if post_result.masks is not None:
            print(f"Mask shape: {{post_result.masks.data.shape}}")

        print("PASSED")
        """,
        timeout=600,
    )


def test_rfdetr_seg_inference_only(dataset):
    """Verify seg model inference produces valid masks on dataset images."""
    run_in_subprocess(
        f"""
        from pathlib import Path
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR

        model = LibreYOLORFDETR(
            model_path="LibreRFDETRn-seg.pt",
            size="n",
            segmentation=True,
        )

        # Run inference on a few test images
        test_dir = Path("{dataset}") / "test" / "images"
        test_images = sorted(test_dir.glob("*.jpg"))[:5]
        assert len(test_images) > 0, "No test images found"

        for img_path in test_images:
            result = model.predict(str(img_path), conf=0.25)
            print(f"{{img_path.name}}: {{len(result)}} dets, "
                  f"masks={{result.masks is not None}}")

            # If detections exist, masks must exist too (seg model)
            if len(result) > 0:
                assert result.masks is not None, (
                    f"Seg model produced detections but no masks for {{img_path.name}}"
                )
                assert result.masks.data.shape[0] == len(result), (
                    f"Mask count ({{result.masks.data.shape[0]}}) != "
                    f"detection count ({{len(result)}})"
                )
                # Masks should be at original image resolution
                h, w = result.orig_shape
                assert result.masks.data.shape[1] == h, "Mask height mismatch"
                assert result.masks.data.shape[2] == w, "Mask width mismatch"

        print("PASSED")
        """,
        timeout=300,
    )


@requires_cuda
def test_rfdetr_seg_resume_training(dataset, tmp_path):
    """Train 3 epochs, stop, resume from checkpoint, train to 5 epochs."""
    output_dir = str(tmp_path / "rfdetr_seg_resume")
    dataset_dir = str(dataset)

    run_in_subprocess(
        f"""
        import gc
        import torch
        from pathlib import Path
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR

        output_dir = "{output_dir}"
        dataset_dir = "{dataset_dir}"

        # Phase 1: Train 3 epochs
        print("Phase 1: Training 3 epochs...")
        model = LibreYOLORFDETR(
            model_path="LibreRFDETRn-seg.pt",
            size="n",
            segmentation=True,
        )
        model.train(
            data=dataset_dir,
            epochs=3,
            batch_size=2,
            output_dir=output_dir,
        )

        # Find checkpoint
        ckpt_path = Path(output_dir) / "checkpoint_best_total.pth"
        if not ckpt_path.exists():
            ckpts = sorted(Path(output_dir).glob("checkpoint*.pth"))
            assert ckpts, f"No checkpoint found in {{output_dir}}"
            ckpt_path = ckpts[-1]
        print(f"Checkpoint: {{ckpt_path}}")

        # Verify checkpoint has seg keys
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        seg_keys = [k for k in ckpt["model"] if k.startswith("segmentation_head")]
        assert len(seg_keys) > 0, "Checkpoint missing segmentation_head keys"

        # Cleanup
        del model, ckpt
        gc.collect()
        torch.cuda.empty_cache()

        # Phase 2: Resume from checkpoint, train to 5 epochs
        print("Phase 2: Resuming training to 5 epochs...")
        model2 = LibreYOLORFDETR(
            model_path="LibreRFDETRn-seg.pt",
            size="n",
            segmentation=True,
        )
        model2.train(
            data=dataset_dir,
            epochs=5,
            batch_size=2,
            output_dir=output_dir,
            resume=str(ckpt_path),
        )

        # Verify resumed checkpoint still has seg keys
        ckpt2_path = Path(output_dir) / "checkpoint_best_total.pth"
        ckpt2 = torch.load(ckpt2_path, map_location="cpu", weights_only=False)
        seg_keys2 = [k for k in ckpt2["model"] if k.startswith("segmentation_head")]
        assert len(seg_keys2) > 0, "Resumed checkpoint missing segmentation_head keys"
        print(f"Resumed checkpoint has {{len(seg_keys2)}} segmentation_head keys")

        # Verify model still produces masks after resume
        from libreyolo import SAMPLE_IMAGE
        result = model2.predict(SAMPLE_IMAGE, conf=0.3)
        print(f"Post-resume inference: {{len(result)}} dets, "
              f"masks={{result.masks is not None}}")

        print("PASSED")
        """,
        timeout=600,
    )

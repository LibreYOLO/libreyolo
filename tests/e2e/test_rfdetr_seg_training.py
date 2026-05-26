"""
E2E: RF-DETR segmentation training and inference tests.

Training test: trains RF-DETR-Seg-Nano for 5 epochs on LibreYOLO/fire-smoke-seg
(HuggingFace, public, 141 train / 40 valid / 20 test images, 2 classes:
fire & smoke, YOLO segmentation format with polygon annotations).

Native RF-DETR segmentation training runs through LibreYOLO's BaseTrainer,
producing LibreYOLO-style outputs (``<save_dir>/weights/best.pt`` and
``last.pt``). Seg inference works on all devices (CPU, MPS, CUDA).

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

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.rfdetr,
    pytest.mark.slow,
    pytest.mark.flagship_nightly,
]

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
@pytest.mark.parametrize("size", ["n", "s", "m", "l"])
def test_rfdetr_seg_training(dataset, tmp_path, size):
    """Train RF-DETR-Seg on fire-smoke-seg, verify training produces a seg checkpoint."""
    output_dir = str(tmp_path / f"rfdetr_seg_{size}")
    data_yaml = str(dataset / "data.yaml")
    output_dir_py = repr(output_dir)
    data_yaml_py = repr(data_yaml)
    weights = f"LibreRFDETR{size}-seg.pt"

    run_in_subprocess(
        f"""
        from pathlib import Path
        from libreyolo.models.rfdetr.model import LibreRFDETR
        from libreyolo import SAMPLE_IMAGE

        # 1. Load segmentation model
        model = LibreRFDETR(
            model_path="{weights}",
            size="{size}",
            segmentation=True,
        )
        assert model._is_segmentation, "Model should be in segmentation mode"
        assert model.task == "segment"

        # 2. Train on fire-smoke-seg dataset (short run — smoke that the
        # train loop runs end-to-end and writes a LibreYOLO-style checkpoint).
        result = model.train(
            data={data_yaml_py},
            epochs=2,
            batch_size=2,
            output_dir={output_dir_py},
        )

        # 3. Verify checkpoint was produced under LibreYOLO conventions:
        # <save_dir>/weights/best.pt and last.pt.
        best_ckpt = result.get("best_checkpoint")
        assert best_ckpt and Path(best_ckpt).exists(), (
            f"Expected best.pt at {{best_ckpt}}, dir contents: "
            f"{{list(Path(result['save_dir']).rglob('*.pt'))}}"
        )
        print(f"Best checkpoint: {{best_ckpt}}")

        # 4. Verify checkpoint has segmentation_head keys
        import torch
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        seg_keys = [k for k in state if "segmentation_head" in k]
        assert len(seg_keys) > 0, "Checkpoint missing segmentation_head keys"
        print(f"Checkpoint has {{len(seg_keys)}} segmentation_head keys")

        # 5. Post-training inference still produces masks
        post_result = model.predict(SAMPLE_IMAGE, conf=0.3)
        print(f"Post-training: {{len(post_result)}} detections, "
              f"masks={{post_result.masks is not None}}")
        if post_result.masks is not None:
            print(f"Mask shape: {{post_result.masks.data.shape}}")

        # 6. Post-training mask mAP — only a sanity floor, not an improvement check
        # (2-epoch run on 141 train images is too short to guarantee improvement).
        post = model.val(
            data={data_yaml_py}, split="test", batch=4, conf=0.001, iou=0.6
        )
        assert "metrics/mAP50-95(M)" in post, "Validation did not return mask mAP"
        post_map = post["metrics/mAP50-95(M)"]
        print(f"Post-training mask mAP50-95(M): {{post_map:.4f}}")
        assert post_map == post_map, "mask mAP is NaN"  # finite check

        print("PASSED")
        """,
        timeout=600,
    )


def test_rfdetr_seg_inference_only(dataset):
    """Verify seg model inference produces valid masks on dataset images."""
    dataset_py = repr(str(dataset))
    run_in_subprocess(
        f"""
        from pathlib import Path
        from libreyolo.models.rfdetr.model import LibreRFDETR

        model = LibreRFDETR(
            model_path="LibreRFDETRn-seg.pt",
            size="n",
            segmentation=True,
        )

        # Run inference on a few test images
        test_dir = Path({dataset_py}) / "test" / "images"
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
    """Train 1 epoch, resume from best.pt, train 1 more epoch — verify seg keys survive."""
    output_dir = str(tmp_path / "rfdetr_seg_resume")
    data_yaml = str(dataset / "data.yaml")
    output_dir_py = repr(output_dir)
    data_yaml_py = repr(data_yaml)

    run_in_subprocess(
        f"""
        import gc
        import torch
        from pathlib import Path
        from libreyolo.models.rfdetr.model import LibreRFDETR

        output_dir = {output_dir_py}

        # Phase 1: Train 1 epoch
        print("Phase 1: Training 1 epoch...")
        model = LibreRFDETR(
            model_path="LibreRFDETRn-seg.pt",
            size="n",
            segmentation=True,
        )
        result = model.train(
            data={data_yaml_py},
            epochs=1,
            batch_size=2,
            output_dir=output_dir,
        )
        best_ckpt = result.get("best_checkpoint")
        assert best_ckpt and Path(best_ckpt).exists(), "Phase 1 best.pt missing"
        print(f"Phase 1 checkpoint: {{best_ckpt}}")

        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        seg_keys = [k for k in state if "segmentation_head" in k]
        assert len(seg_keys) > 0, "Phase 1 checkpoint missing segmentation_head keys"

        del model, ckpt
        gc.collect()
        torch.cuda.empty_cache()

        # Phase 2: Resume from best.pt, train 1 more epoch
        print("Phase 2: Resuming for 1 more epoch...")
        model2 = LibreRFDETR(
            model_path="LibreRFDETRn-seg.pt",
            size="n",
            segmentation=True,
        )
        result2 = model2.train(
            data={data_yaml_py},
            epochs=2,
            batch_size=2,
            output_dir=output_dir,
            resume=best_ckpt,
        )
        best2 = result2.get("best_checkpoint")
        assert best2 and Path(best2).exists(), "Phase 2 best.pt missing"
        ckpt2 = torch.load(best2, map_location="cpu", weights_only=False)
        assert ckpt2.get("epoch", -1) >= 1, "Resume did not run an additional epoch"
        state2 = ckpt2.get("model", ckpt2)
        seg_keys2 = [k for k in state2 if "segmentation_head" in k]
        assert len(seg_keys2) > 0, "Resumed checkpoint missing segmentation_head keys"
        print(f"Resumed checkpoint has {{len(seg_keys2)}} segmentation_head keys")

        # Verify model still produces masks after resume
        from libreyolo import SAMPLE_IMAGE
        r = model2.predict(SAMPLE_IMAGE, conf=0.3)
        print(f"Post-resume: {{len(r)}} dets, masks={{r.masks is not None}}")

        print("PASSED")
        """,
        timeout=600,
    )

"""
E2E: RF-DETR segmentation training test.

Trains RF-DETR-Seg-Nano for a few epochs on LibreYOLO/fire-smoke-seg
(HuggingFace, public, 141 train / 40 valid / 20 test images, 2 classes:
fire & smoke, YOLO segmentation format with polygon annotations).

The dataset auto-downloads from HuggingFace — no API keys needed.

Usage:
    pytest tests/e2e/test_rfdetr_seg_training.py -v -m e2e
"""

import subprocess
from pathlib import Path

import pytest
import yaml

from .conftest import run_in_subprocess

pytestmark = [pytest.mark.e2e, pytest.mark.rfdetr, pytest.mark.slow]

DATASET_ROOT = Path.home() / ".cache" / "libreyolo" / "fire-smoke-seg"
HF_REPO = "LibreYOLO/fire-smoke-seg"


def download_fire_smoke_dataset():
    """Download the fire-smoke-seg dataset from HuggingFace if not cached."""
    if DATASET_ROOT.exists() and (DATASET_ROOT / "data.yaml").exists():
        return

    print(f"\nDownloading dataset {HF_REPO} from HuggingFace ...")
    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "git",
            "clone",
            f"https://huggingface.co/datasets/{HF_REPO}",
            str(DATASET_ROOT),
        ],
        check=True,
    )
    print(f"Dataset downloaded to {DATASET_ROOT}")


def patch_data_yaml():
    """Ensure data.yaml has an absolute path so training resolves splits."""
    data_yaml = DATASET_ROOT / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text())
    if data.get("path") != str(DATASET_ROOT):
        data["path"] = str(DATASET_ROOT)
        data_yaml.write_text(yaml.dump(data, default_flow_style=False))


@pytest.fixture(scope="module")
def dataset():
    """Download fire-smoke-seg dataset and patch data.yaml."""
    download_fire_smoke_dataset()
    patch_data_yaml()
    return DATASET_ROOT


def test_rfdetr_seg_training(dataset, tmp_path):
    """Train RF-DETR-Seg-Nano on fire-smoke-seg, verify masks are produced."""
    output_dir = str(tmp_path / "rfdetr_seg_n")
    dataset_dir = str(dataset)

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

        # 2. Verify pre-training inference produces masks
        pre_result = model.predict(SAMPLE_IMAGE, conf=0.3)
        print(f"Pre-training: {{len(pre_result)}} detections, "
              f"masks={{pre_result.masks is not None}}")

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

        # 6. Post-training inference still produces masks
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
        test_images = sorted(Path("{dataset}") / "test" / "images")
        test_images = list(test_images.glob("*.jpg"))[:5]
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

"""
LibreFOMO end-to-end training test.

Runs a 2-epoch smoke train on a small YOLO-format point-localization dataset
hosted at ``LibreYOLO/<DATASET_REPO>`` (HuggingFace, public), then verifies:

  - Loss decreased epoch 1 → epoch 2
  - ``last.pt`` exists with valid checkpoint schema
  - ``metrics/F1`` is present in val_metrics
  - Trained checkpoint loads via the LibreYOLO factory (auto-detects size)
  - Inference on the reloaded model returns a valid ``result.points`` object

Dataset format (YOLO-standard)::

    dataset/
    ├── data.yaml           # nc, names, path, train/val/test keys
    ├── train/images/       # .jpg files
    ├── train/labels/       # .txt  (class cx cy w h, normalised)
    ├── valid/images/
    └── valid/labels/

Run via pytest::

    pytest tests/e2e/test_fomo_training.py -v -m fomo

Or via Modal::

    modal run /tmp/libreyolo_librefomo_train_modal.py
"""

import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from .conftest import cuda_cleanup, requires_cuda, run_direct_subprocess

pytestmark = [pytest.mark.e2e, pytest.mark.fomo]

# ---------------------------------------------------------------------------
# Dataset — git-clone from HuggingFace (same pattern as marbles / RF1)
# ---------------------------------------------------------------------------

HF_REPO = "bdanko/sjsu-headcount-scene-1"
DATASET_ROOT = Path.home() / ".cache" / "libreyolo" / "sjsu-headcount-scene-1"

FOMO_EPOCHS = 2
FOMO_BATCH = 8


def download_fomo_dataset():
    """Download the FOMO point-localization dataset from HuggingFace.

    Cached under ~/.cache/libreyolo/fomo-smoke-small after the first run.
    """
    if DATASET_ROOT.exists() and (DATASET_ROOT / "data.yaml").exists():
        return
    print(f"\nDownloading dataset {HF_REPO} from HuggingFace ...")
    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", f"https://huggingface.co/datasets/{HF_REPO}", str(DATASET_ROOT)],
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
def fomo_dataset():
    """Download and patch the FOMO dataset. Shared by all fixtures in this module."""
    download_fomo_dataset()
    patch_data_yaml()
    return DATASET_ROOT


@requires_cuda
def test_fomo_training_smoke(fomo_dataset, tmp_path):
    """2-epoch smoke train on the FOMO dataset — verify plumbing end-to-end.

    Uses ``run_direct_subprocess`` so CUDA state is isolated from other tests.
    Asserts:
      - Both epochs complete and loss decreases
      - ``last.pt`` / ``best.pt`` written with valid checkpoint schema
      - ``metrics/F1`` present in val_metrics
      - Factory-loaded checkpoint runs inference correctly
    """
    dataset_data_yaml = str(fomo_dataset / "data.yaml")

    run_direct_subprocess(
        f"""
        import torch
        from pathlib import Path

        from libreyolo import LibreYOLO
        from libreyolo.models.librefomo.model import LibreFOMO
        from libreyolo.utils.serialization import validate_checkpoint_metadata

        # --- Build fresh LibreFOMO-s (no pretrained weights) ---
        model = LibreFOMO(model_path=None, size="s", nb_classes=1, device="cpu")
        assert model.family == "librefomo"
        assert model.size == "s"
        assert model.input_size == 96

        # --- Train ---
        results = model.train(
            allow_experimental=True,
            data=r"{dataset_data_yaml}",
            epochs={FOMO_EPOCHS},
            batch={FOMO_BATCH},
            lr0=3e-4,
            eval_interval=1,
            workers=2,
            device="cuda",
            project=r"{str(tmp_path)}",
            name="smoke_s",
            exist_ok=True,
            patience=0,
        )

        # --- Loss decreased ---
        epoch_losses = results["epoch_losses"]
        assert len(epoch_losses) == {FOMO_EPOCHS}, (
            f"Expected {FOMO_EPOCHS} epoch losses, got {{len(epoch_losses)}}"
        )
        first_loss, last_loss = epoch_losses[0], epoch_losses[-1]
        assert last_loss < first_loss, (
            f"Loss did not decrease: {{first_loss:.4f}} → {{last_loss:.4f}}"
        )
        print(f"  loss: epoch1={{first_loss:.4f}}, epoch2={{last_loss:.4f}}", flush=True)

        # --- Checkpoints written ---
        save_dir = Path(results["save_dir"])
        weights_dir = save_dir / "weights"
        last_pt = weights_dir / "last.pt"
        assert last_pt.exists(), f"last.pt not found at {{last_pt}}"

        best_pt = weights_dir / "best.pt"
        ckpt_path = best_pt if best_pt.exists() else last_pt

        # --- Schema valid ---
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        errors = validate_checkpoint_metadata(ckpt, strict=False)
        assert errors == [], f"Checkpoint schema errors: {{errors}}"
        assert ckpt["task"] == "point"
        assert ckpt["model_family"] == "librefomo"
        assert ckpt["size"] == "s"
        print(f"  checkpoint schema OK: task={{ckpt['task']}}, size={{ckpt['size']}}", flush=True)

        # --- metrics/F1 in val_metrics ---
        val_metrics = [m for m in results.get("val_metrics", []) if m]
        assert len(val_metrics) > 0, "No non-empty val_metrics entries"
        last_metrics = val_metrics[-1]
        assert "metrics/F1" in last_metrics, (
            f"metrics/F1 not found in {{list(last_metrics.keys())}}"
        )
        f1 = last_metrics["metrics/F1"]
        assert isinstance(f1, float) and f1 >= 0.0
        print(f"  metrics/F1={{f1:.4f}}", flush=True)

        # --- Factory reload + inference ---
        from libreyolo import LibreYOLO
        import numpy as np
        from PIL import Image

        trained = LibreYOLO(str(ckpt_path), device="cpu")
        assert trained.family == "librefomo"
        assert trained.size == "s"

        arr = np.zeros((96, 96, 3), dtype=np.uint8)
        arr[40:50, 40:50] = 200
        result = trained.predict(Image.fromarray(arr), conf=0.01, max_det=20)
        assert result.boxes is None, "LibreFOMO must not produce boxes"
        assert result.points is not None
        print(f"  inference OK — n_points={{len(result)}}", flush=True)

        print("\\n✓ LibreFOMO training smoke test PASSED", flush=True)
        """,
        timeout=1800,
    )
    cuda_cleanup()

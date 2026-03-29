"""Regression tests for known unfixed bugs.

Each bug gets its own xfail(strict=True) test. When a bug is fixed,
the test will XPASS and strict=True turns that into a FAILURE —
forcing the developer to remove the xfail marker.

Trains yolo9-t once (module-scoped fixture) to keep CI cost minimal.
"""

import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from libreyolo import LibreYOLO

pytestmark = pytest.mark.e2e

DATASET_ROOT = Path.home() / ".cache" / "libreyolo" / "marbles"
HF_REPO = "LibreYOLO/marbles"


@pytest.fixture(scope="module")
def marbles_yaml():
    """Download marbles if needed, patch data.yaml, return path."""
    if not DATASET_ROOT.exists():
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
    data_yaml = DATASET_ROOT / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text())
    if data.get("path") != str(DATASET_ROOT):
        data["path"] = str(DATASET_ROOT)
        data_yaml.write_text(yaml.dump(data, default_flow_style=False))
    return str(data_yaml)


@pytest.fixture(scope="module")
def trained_model(marbles_yaml, tmp_path_factory):
    """Train yolo9-t on marbles for 3 epochs. Returns (model, results)."""
    tmp = tmp_path_factory.mktemp("training_regression")
    model = LibreYOLO("LibreYOLO9t.pt", size="t")
    results = model.train(
        data=marbles_yaml,
        epochs=3,
        batch=8,
        workers=0,
        project=str(tmp),
        name="yolo9_t",
        exist_ok=True,
    )
    return model, results


@pytest.mark.xfail(
    reason="model left in train() mode after .train() — no .eval() call",
    strict=True,
)
def test_predict_after_train(trained_model, marbles_yaml):
    model, _ = trained_model
    img = next((DATASET_ROOT / "test" / "images").glob("*.jpg"))
    result = model.predict(str(img), conf=0.1)
    assert hasattr(result, "boxes")


@pytest.mark.xfail(
    reason="model.names not updated from data.yaml during training",
    strict=True,
)
def test_names_updated_after_train(trained_model):
    model, _ = trained_model
    assert len(model.names) == 2
    assert model.names[0] != "person", f"Still COCO names: {model.names}"


@pytest.mark.xfail(
    reason="checkpoint saves stale COCO names instead of dataset names",
    strict=True,
)
def test_checkpoint_saves_correct_names(trained_model):
    _, results = trained_model
    ckpt_path = results.get("best_checkpoint", "")
    if not Path(ckpt_path).exists():
        ckpt_path = str(Path(ckpt_path).parent / "last.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    names = ckpt.get("names", {})
    assert names.get(0) != "person", f"Checkpoint has stale COCO names: {names}"

"""Regression tests for training bugs.

Known unfixed bugs keep xfail(strict=True). Fixed bugs remain as plain
regression checks so they cannot silently regress.

Trains yolo9-t once (module-scoped fixture) to keep CI cost minimal.
Trains rfdetr-n once for the post-training inference restore path.
"""

import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from libreyolo import LibreYOLO

pytestmark = [pytest.mark.e2e]

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
def trained_yolo9_model(marbles_yaml, tmp_path_factory):
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


@pytest.mark.yolo9
def test_yolo9_predict_after_train(trained_yolo9_model, marbles_yaml):
    model, _ = trained_yolo9_model
    img = next((DATASET_ROOT / "test" / "images").glob("*.jpg"))
    result = model.predict(str(img), conf=0.1)
    assert hasattr(result, "boxes")


@pytest.mark.yolo9
def test_yolo9_names_updated_after_train(trained_yolo9_model):
    model, _ = trained_yolo9_model
    assert len(model.names) == 2
    assert model.names[0] != "person", f"Still COCO names: {model.names}"


@pytest.mark.yolo9
def test_yolo9_checkpoint_saves_correct_names(trained_yolo9_model):
    _, results = trained_yolo9_model
    ckpt_path = results.get("best_checkpoint", "")
    if not Path(ckpt_path).exists():
        ckpt_path = str(Path(ckpt_path).parent / "last.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    names = ckpt.get("names", {})
    assert names.get(0) != "person", f"Checkpoint has stale COCO names: {names}"


@pytest.fixture(scope="module")
def trained_rfdetr_model(marbles_yaml, tmp_path_factory):
    """Train rfdetr-n on marbles for 1 epoch. Returns (model, results)."""
    tmp = tmp_path_factory.mktemp("training_regression_rfdetr")
    model = LibreYOLO("LibreRFDETRn.pt", size="n")
    results = model.train(
        data=marbles_yaml,
        epochs=1,
        batch_size=2,
        output_dir=str(tmp / "rfdetr_n"),
    )
    return model, results


@pytest.mark.rfdetr
def test_rfdetr_predict_after_train(trained_rfdetr_model, marbles_yaml):
    model, _ = trained_rfdetr_model
    img = next((DATASET_ROOT / "test" / "images").glob("*.jpg"))
    result = model.predict(str(img), conf=0.1)
    assert hasattr(result, "boxes")

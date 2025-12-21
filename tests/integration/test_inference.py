import shutil
from pathlib import Path

import pytest

from libreyolo import LIBREYOLO8, LIBREYOLO11

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "version, size",
    [
        ("8", "n"),
        ("8", "s"),
        ("8", "m"),
        ("8", "l"),
        ("8", "x"),
        ("11", "n"),
        ("11", "s"),
        ("11", "m"),
        ("11", "l"),
        ("11", "x"),
    ],
)
def test_inference_and_feature_maps(
    version, size, weights_dir, test_image, output_dir, tmp_path, monkeypatch
):
    weight_file = weights_dir / f"libreyolo{version}{size}.pt"
    if not weight_file.exists():
        pytest.skip(f"Weights {weight_file.name} not found")

    # keep test artifacts isolated per-test
    monkeypatch.chdir(tmp_path)

    # 1. Load Model
    ModelClass = LIBREYOLO8 if version == "8" else LIBREYOLO11
    model = ModelClass(model_path=str(weight_file), size=size)

    # 2. Standard Inference
    results = model.predict(test_image, save=False)
    assert results is not None
    assert "boxes" in results
    assert "scores" in results
    assert "classes" in results
    assert "num_detections" in results

    # 3. Feature Map Generation (isolated under tmp_path/runs/feature_maps)
    runs_dir = Path("runs/feature_maps")
    if runs_dir.exists():
        shutil.rmtree(runs_dir)

    model_fm = ModelClass(model_path=str(weight_file), size=size, save_feature_maps=True)
    model_fm.predict(test_image)

    assert runs_dir.exists(), "Feature maps directory was not created"

    subdirs = list(runs_dir.iterdir())
    assert subdirs, "No timestamped directory found in runs/feature_maps"

    fm_dir = sorted(subdirs)[-1]

    assert (fm_dir / "metadata.json").exists(), "metadata.json missing"

    pngs = list(fm_dir.glob("*.png"))
    assert pngs, "No feature map images generated"


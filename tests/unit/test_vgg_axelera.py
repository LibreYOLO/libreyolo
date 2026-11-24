"""VGG Axelera export: the whole graph runs on Metis."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tests.unit.test_axelera_backend import _dataset, _install_fake_sdk

pytestmark = [pytest.mark.unit, pytest.mark.vgg]
pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def sdk():
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield _install_fake_sdk(monkeypatch)


def _model(size="19bn"):
    from libreyolo import LibreVGG

    torch.manual_seed(0)
    model = LibreVGG(size=size, nb_classes=3, device="cpu")
    # The published 4096-wide classifier makes a random-weight artifact over
    # 500 MB; the feature graph stays whole and only the head is narrowed.
    model.model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 16), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
        nn.Linear(16, 16), nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(16, 3),
    )
    model.model.eval()
    return model


def test_the_whole_graph_runs_on_metis(sdk, tmp_path, monkeypatch):
    import yaml

    import libreyolo.export.axelera as axelera

    recipes = dict(axelera.AXELERA_SIZE_RECIPES)
    recipes.pop(("vgg", "classify", "19bn"), None)
    monkeypatch.setattr(axelera, "AXELERA_SIZE_RECIPES", recipes)
    artifact = Path(_model().export(
        format="axelera", data=str(_dataset(tmp_path)), imgsz=224, output_path=str(tmp_path / "vggw_axelera_model"),
    ))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert "parts" not in meta and "hybrid" not in meta and meta["output_names"] == ["logits"]
    assert meta["output_layout"] == "classify_logits_v1"

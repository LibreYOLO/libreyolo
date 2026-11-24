"""DeepLabV3 Axelera export: the whole graph runs on Metis (ASPP as per-tap 1x1 convolutions plus shifts)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.unit.test_axelera_backend import _dataset, _install_fake_sdk

pytestmark = pytest.mark.unit
pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def sdk():
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield _install_fake_sdk(monkeypatch)


def _model():
    from libreyolo import LibreDeepLabv3

    torch.manual_seed(0)
    return LibreDeepLabv3(None, size="r50", nb_classes=3, device="cpu")


def test_the_whole_graph_runs_on_metis(sdk, tmp_path, monkeypatch):
    import yaml

    import libreyolo.export.axelera as axelera

    recipes = dict(axelera.AXELERA_SIZE_RECIPES)
    recipes[("deeplabv3", "semantic", "r50")] = {"aspp_shift": True}
    monkeypatch.setattr(axelera, "AXELERA_SIZE_RECIPES", recipes)
    artifact = Path(_model().export(
        format="axelera", data=str(_dataset(tmp_path)), imgsz=64, output_path=str(tmp_path / "dlw_axelera_model"),
    ))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert "parts" not in meta and "hybrid" not in meta and meta["output_names"] == ["semantic_logits"]

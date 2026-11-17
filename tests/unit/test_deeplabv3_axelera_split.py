"""DeepLabV3 Axelera backbone split (per-size recipe ``hybrid_split``): backbone on Metis, ASPP + head on the host."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tests.unit.test_axelera_backend import _dataset, _image, _install_fake_sdk

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


def test_backbone_part_and_host_tail_equal_the_model():
    from libreyolo.export.exporter import _DeepLabBackboneWrapper, _DeepLabHostTailWrapper

    net = _model().model.eval()
    head = _DeepLabBackboneWrapper(net).eval()
    tail = _DeepLabHostTailWrapper(net, ("semantic_logits",)).eval()
    tail.upsample_to = (64, 64)
    x = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(tail(*head(x))[0], net(x), rtol=0, atol=0)


def test_r101_recipe_selects_the_split_and_r50_keeps_the_whole_graph():
    from libreyolo.export.axelera import size_recipe

    assert size_recipe("deeplabv3", "semantic", "r101").get("hybrid_split") == "backbone"
    assert "hybrid_split" not in size_recipe("deeplabv3", "semantic", "r50")


def test_split_export_runs_the_chain_like_the_onnx_backend(sdk, tmp_path, monkeypatch):
    import yaml

    import libreyolo.export.axelera as axelera
    from libreyolo import LibreYOLO

    model = _model()
    recipes = dict(axelera.AXELERA_SIZE_RECIPES)
    recipes[("deeplabv3", "semantic", "r50")] = {"aspp_shift": True, "hybrid_split": "backbone"}
    monkeypatch.setattr(axelera, "AXELERA_SIZE_RECIPES", recipes)
    artifact = Path(model.export(
        format="axelera", data=str(_dataset(tmp_path)), imgsz=64, output_path=str(tmp_path / "dl_axelera_model"),
    ))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["hybrid"]["split"] == "backbone"
    assert [("axm" if "axm" in part else "onnx") for part in meta["parts"]] == ["axm", "onnx"]
    assert meta["parts"][0]["outputs"] == ["feat"] and meta["output_names"] == ["semantic_logits"]
    backend = LibreYOLO(str(artifact))
    reference = LibreYOLO(model.export(format="onnx", imgsz=64, dynamic=False, output_path=str(tmp_path / "dl.onnx")))
    image = _image(4)
    actual, expected = backend(image)[0], reference(image)[0]
    assert np.mean(np.asarray(actual.semantic_mask.data) == np.asarray(expected.semantic_mask.data)) > 0.999


def test_without_the_recipe_the_whole_graph_runs_on_metis(sdk, tmp_path):
    import yaml

    artifact = Path(_model().export(
        format="axelera", data=str(_dataset(tmp_path)), imgsz=64, output_path=str(tmp_path / "dlw_axelera_model"),
    ))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert "parts" not in meta and "hybrid" not in meta and meta["output_names"] == ["semantic_logits"]

"""VGG-19-BN Axelera pool4 split (per-size recipe ``hybrid_split``): blocks 1-4 on Metis, block 5 + classifier on the host."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tests.unit.test_axelera_backend import _dataset, _image, _install_fake_sdk

_PARKED = pytest.mark.skip(
    reason="hybrid export parked by decision 2026-11-19 (part of the network runs on the host); branch axelera-hybrid-parked"
)

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


def test_features_part_and_host_tail_equal_the_model():
    from libreyolo.export.exporter import _VGGFeaturesWrapper, _VGGHostTailWrapper

    net = _model().model.eval()
    head = _VGGFeaturesWrapper(net).eval()
    tail = _VGGHostTailWrapper(net, ("logits",)).eval()
    # Blocks 1-4: 12 convolutions (each with its batch norm and ReLU) and 4 pools.
    assert sum(isinstance(m, nn.Conv2d) for m in head.features) == 12
    assert sum(isinstance(m, nn.MaxPool2d) for m in head.features) == 4
    assert sum(isinstance(m, nn.Conv2d) for m in tail.features) == 4
    x = torch.rand(1, 3, 224, 224, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        (feat,) = head(x)
        assert tuple(feat.shape) == (1, 512, 14, 14)
        torch.testing.assert_close(tail(feat)[0], net(x), rtol=0, atol=0)


@_PARKED
def test_only_19bn_selects_the_split():
    from libreyolo.export.axelera import size_recipe

    assert size_recipe("vgg", "classify", "19bn").get("hybrid_split") == "backbone"
    for size in ("16", "19", "16bn"):
        assert "hybrid_split" not in size_recipe("vgg", "classify", size)


@_PARKED
def test_split_export_runs_the_chain_like_the_onnx_backend(sdk, tmp_path):
    import yaml

    from libreyolo import LibreYOLO

    model = _model()
    artifact = Path(model.export(
        format="axelera", data=str(_dataset(tmp_path)), imgsz=224, output_path=str(tmp_path / "vgg_axelera_model"),
    ))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["output_layout"] == "classify_logits_v1" and meta["hybrid"]["split"] == "backbone"
    assert [("axm" if "axm" in part else "onnx") for part in meta["parts"]] == ["axm", "onnx"]
    assert meta["parts"][0]["outputs"] == ["pool4"] and meta["output_names"] == ["logits"]
    backend = LibreYOLO(str(artifact))
    reference = LibreYOLO(model.export(format="onnx", imgsz=224, dynamic=False, output_path=str(tmp_path / "vgg.onnx")))
    for seed in (2, 3):
        image = _image(seed, shape=(240, 300, 3))
        actual, expected = backend(image)[0], reference(image)[0]
        assert actual.probs.top1 == expected.probs.top1
        np.testing.assert_allclose(
            np.asarray(actual.probs.data, dtype=np.float32),
            np.asarray(expected.probs.data, dtype=np.float32),
            atol=1e-4,
        )


def test_without_the_recipe_the_whole_graph_runs_on_metis(sdk, tmp_path, monkeypatch):
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

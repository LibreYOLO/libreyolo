"""RTMDet-s Axelera recipe ``pointwise_learned_rounding``: learned INT8 rounding of the CSPNeXt pointwise convs."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from tests.unit.test_rtmdet_raw_heads import _build

pytestmark = pytest.mark.unit


def _pointwise_convs(net):
    from libreyolo.models.rtmdet.nn import DepthwiseSeparableConv
    from libreyolo.models.rtmdet.quantize import _pointwise

    return [_pointwise(m).conv for m in net.modules() if isinstance(m, DepthwiseSeparableConv)]


def test_rounded_weights_sit_on_the_int8_grid_and_keep_the_outputs_close():
    from libreyolo.export.exporter import _RTMDetRawHeadsWrapper, _equalize_cspnext_pointwise_inputs
    from libreyolo.models.rtmdet.quantize import learn_pointwise_int8_rounding

    _, net = _build()
    before = copy.deepcopy(net.state_dict())
    eq = copy.deepcopy(net).eval()
    samples = [torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(i)) * 255 for i in range(3)]
    _equalize_cspnext_pointwise_inputs(eq, samples, "conv", "cle")
    reference = copy.deepcopy(eq)
    peaks = [c.weight.detach().abs().amax(dim=(1, 2, 3)) for c in _pointwise_convs(eq)]
    count = learn_pointwise_int8_rounding(eq, samples, steps=40)
    convs = _pointwise_convs(eq)
    assert count == len(convs) > 0
    for conv, peak in zip(convs, peaks):
        step = (peak / 127.0).reshape(-1, 1, 1, 1)
        codes = conv.weight.detach() / step
        torch.testing.assert_close(codes, torch.round(codes), rtol=0, atol=1e-3)
        # The channel's largest weight keeps its value, so the compiler's per-channel scale is unchanged.
        torch.testing.assert_close(conv.weight.detach().abs().amax(dim=(1, 2, 3)), peak)
    x = torch.rand(1, 3, 64, 96, generator=torch.Generator().manual_seed(7)) * 255
    with torch.no_grad():
        for a, b in zip(_RTMDetRawHeadsWrapper(reference, output_windows={}).eval()(x),
                        _RTMDetRawHeadsWrapper(eq, output_windows={}).eval()(x)):
            assert float((b - a).abs().max()) < 0.05 * float(a.abs().max()) + 1e-3
    # Deterministic, and the caller's model is never touched.
    again = copy.deepcopy(reference)
    learn_pointwise_int8_rounding(again, samples, steps=40)
    assert all(torch.equal(p, q) for p, q in zip(again.state_dict().values(), eq.state_dict().values()))
    assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())


def test_without_samples_nothing_is_rounded():
    from libreyolo.models.rtmdet.quantize import learn_pointwise_int8_rounding

    _, net = _build()
    assert learn_pointwise_int8_rounding(copy.deepcopy(net), []) == 0


def test_only_rtmdet_s_detect_has_the_recipe():
    from libreyolo.export.axelera import size_recipe

    assert size_recipe("rtmdet", "detect", "s") == {"pointwise_learned_rounding": True}
    for size in ("t", "m", "l", "x"):
        assert "pointwise_learned_rounding" not in size_recipe("rtmdet", "detect", size)
    assert "pointwise_learned_rounding" not in size_recipe("rtmdet", "segment", "s")


def test_the_recipe_reaches_the_export(tmp_path, monkeypatch):
    import yaml

    import libreyolo.models.rtmdet.quantize as q
    from libreyolo import LibreRTMDet
    from tests.unit.test_axelera_backend import _dataset, _install_fake_sdk

    _install_fake_sdk(monkeypatch)
    monkeypatch.setattr(q.learn_pointwise_int8_rounding, "__defaults__", (5, q.INT8_ROUNDING_SEED))
    torch.manual_seed(0)
    model = LibreRTMDet(None, size="t", nb_classes=3, device="cpu")
    monkeypatch.setattr(model, "size", "s")  # the s recipe on a t-sized stand-in
    before = {k: v.clone() for k, v in model.model.state_dict().items()}
    out = Path(model.export(format="axelera", data=str(_dataset(tmp_path)), imgsz=64,
                            output_path=str(tmp_path / "rtms_axelera_model")))
    meta = yaml.safe_load((out / "metadata.yaml").read_text())
    assert meta["pointwise_learned_rounding"]["convs"] == meta["pointwise_equalized"]["blocks"] > 0
    assert all(torch.equal(before[k], v) for k, v in model.model.state_dict().items())
    model.size = "t"
    out = Path(model.export(format="axelera", data=str(_dataset(tmp_path / "t")), imgsz=64,
                            output_path=str(tmp_path / "rtmt_axelera_model")))
    assert "pointwise_learned_rounding" not in yaml.safe_load((out / "metadata.yaml").read_text())

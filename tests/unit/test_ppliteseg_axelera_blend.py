"""PP-LiteSeg Axelera export: UAFM blends in the exact INT8-friendly form x * sigmoid(z) + skip * sigmoid(-z)."""

from __future__ import annotations

import copy

import pytest
import torch

pytestmark = pytest.mark.unit


def _export_net(size="t50", canvas=(128, 256)):
    from libreyolo.models.ppliteseg.nn import LibrePPLiteSegNet

    torch.manual_seed(0)
    net = LibrePPLiteSegNet(size=size, num_classes=5, use_aux_heads=True).eval()
    net.remove_aux_heads()
    net.encoder.context_module.prep_model_for_conversion(canvas)
    with torch.no_grad():  # non-trivial batch-norm statistics so the attention gates are not all 0.5
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.5, 0.5)
                m.running_var.uniform_(0.5, 2.0)
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.3, 0.3)
    return net


@pytest.mark.parametrize("size", ["t50", "b50"])
def test_blend_rewrite_is_exact_in_fp32_and_float64(size):
    from libreyolo.export.exporter import _ppliteseg_int8_blend

    net = _export_net(size)
    rewritten = copy.deepcopy(net)
    assert _ppliteseg_int8_blend(rewritten) == 3
    x = torch.rand(2, 3, 128, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(rewritten(x), net(x), rtol=1e-4, atol=1e-4)  # FP32 rounding only
        torch.testing.assert_close(copy.deepcopy(rewritten).double()(x.double()),
                                   copy.deepcopy(net).double()(x.double()), rtol=0, atol=1e-10)
    # The rewrite lives on the copy only: the original UAFMs keep their class forward.
    assert all("forward" not in m.__dict__ for m in net.modules() if type(m).__name__ == "UAFM")
    negates = [m._int8_negate for m in rewritten.modules() if type(m).__name__ == "UAFM"]
    assert all(float(n.weight) == -1.0 and n.bias is None for n in negates)


def test_the_axelera_export_traces_the_rewritten_blend(tmp_path, monkeypatch):
    import dataclasses

    import libreyolo.export.exporter as exporter
    from libreyolo import LibrePPLiteSeg
    from tests.unit.test_axelera_backend import _dataset, _install_fake_sdk

    _install_fake_sdk(monkeypatch)
    original = exporter.get_support
    monkeypatch.setattr(  # the family's tier is decided by the device; the export path is under test here
        exporter, "get_support",
        lambda f, t, fmt: dataclasses.replace(original(f, t, fmt), tier="available") if f == "ppliteseg" else original(f, t, fmt),
    )
    torch.manual_seed(0)
    model = LibrePPLiteSeg(size="t50", nb_classes=5, device="cpu")
    calls = []
    real = exporter._ppliteseg_int8_blend
    monkeypatch.setattr(exporter, "_ppliteseg_int8_blend", lambda m: calls.append(1) or real(m))
    out = model.export(format="axelera", data=str(_dataset(tmp_path)), imgsz=(128, 256),
                       output_path=str(tmp_path / "ppl_axelera_model"))
    assert calls == [1] and out
    # The caller's model keeps the native UAFM forward.
    assert all("forward" not in m.__dict__ for m in model.model.modules() if type(m).__name__ == "UAFM")

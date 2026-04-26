"""Tests for RT-DETR backbone logging behavior."""

import pytest

from libreyolo.models.rtdetr.backbone import PResNet

pytestmark = pytest.mark.unit


def test_presnet_pretrained_uses_logger_not_print(monkeypatch):
    calls = {}

    monkeypatch.setattr("torch.hub.load_state_dict_from_url", lambda url: {})
    monkeypatch.setattr(PResNet, "load_state_dict", lambda self, state: None)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("print should not be called")))
    monkeypatch.setattr(
        "libreyolo.models.rtdetr.backbone.logger.info",
        lambda message, depth: calls.update({"message": message, "depth": depth}),
    )

    PResNet(depth=18, pretrained=True)

    assert calls == {"message": "Loaded PResNet%d state_dict", "depth": 18}

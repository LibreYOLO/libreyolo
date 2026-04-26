"""Tests for torch checkpoint loading helpers."""

import pytest

from libreyolo.utils import serialization

pytestmark = pytest.mark.unit


def test_untrusted_load_uses_weights_only(monkeypatch):
    calls = {}

    monkeypatch.setattr(serialization, "_supports_weights_only", lambda: True)

    def fake_load(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(serialization.torch, "load", fake_load)

    result = serialization.load_untrusted_torch_file("model.pt")

    assert result == {"ok": True}
    assert calls["kwargs"]["weights_only"] is True
    assert calls["kwargs"]["map_location"] == "cpu"


def test_trusted_load_uses_full_checkpoint_mode(monkeypatch):
    calls = {}

    monkeypatch.setattr(serialization, "_supports_weights_only", lambda: True)

    def fake_load(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(serialization.torch, "load", fake_load)

    result = serialization.load_trusted_torch_file("last.pt", map_location="cuda:0")

    assert result == {"ok": True}
    assert calls["kwargs"]["weights_only"] is False
    assert calls["kwargs"]["map_location"] == "cuda:0"


def test_untrusted_load_requires_modern_torch(monkeypatch):
    monkeypatch.setattr(serialization, "_supports_weights_only", lambda: False)

    with pytest.raises(RuntimeError, match="weights_only"):
        serialization.load_untrusted_torch_file("model.pt")

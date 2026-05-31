"""BaseValidator._setup_device normalisation tests."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from libreyolo.validation.base import BaseValidator
from libreyolo.validation.config import ValidationConfig

pytestmark = pytest.mark.unit


class _StubValidator(BaseValidator):
    def _setup_dataloader(self): pass
    def _init_metrics(self): pass
    def _preprocess_batch(self, b): pass
    def _postprocess_predictions(self, p, b): pass
    def _update_metrics(self, p, t, i, ids=None): pass
    def _compute_metrics(self): return {}


def _setup_device(device: str) -> "torch.device":
    config = ValidationConfig(data="x.yaml", device=device)
    v = object.__new__(_StubValidator)
    v.config = config
    return v._setup_device()


def _stub_validator(*, half: bool, device: str):
    validator = object.__new__(_StubValidator)
    validator.config = ValidationConfig(data="x.yaml", device=device, half=half)
    validator.device = torch.device(device)
    return validator


def test_bare_integer_device_string_normalised():
    with patch("torch.cuda.is_available", return_value=True):
        device = _setup_device("0")
    assert device.type == "cuda"
    assert str(device) == "cuda:0"


def test_bare_integer_string_two_digit():
    with patch("torch.cuda.is_available", return_value=True):
        device = _setup_device("10")
    assert device.type == "cuda"
    assert str(device) == "cuda:10"


def test_named_device_strings_pass_through():
    assert _setup_device("cpu").type == "cpu"
    assert str(_setup_device("cuda:0")) == "cuda:0"


@pytest.mark.parametrize(
    ("device", "expected_calls"),
    [("cuda", ["cuda"]), ("cpu", [])],
)
def test_half_validation_uses_cuda_autocast_only(monkeypatch, device, expected_calls):
    calls = []

    @contextmanager
    def fake_autocast(device_type):
        calls.append(device_type)
        yield

    monkeypatch.setattr("libreyolo.validation.base.torch.amp.autocast", fake_autocast)

    validator = _stub_validator(half=True, device=device)

    with validator._autocast_context():
        pass

    assert calls == expected_calls

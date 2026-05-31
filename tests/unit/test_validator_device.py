"""BaseValidator._setup_device normalisation tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from libreyolo.validation.config import ValidationConfig

pytestmark = pytest.mark.unit


def _setup_device(device: str) -> "torch.device":
    from libreyolo.validation.base import BaseValidator

    class _StubValidator(BaseValidator):
        def _setup_dataloader(self): pass
        def _init_metrics(self): pass
        def _preprocess_batch(self, b): pass
        def _postprocess_predictions(self, p, b): pass
        def _update_metrics(self, p, t, i, ids=None): pass
        def _compute_metrics(self): return {}

    config = ValidationConfig(data="x.yaml", device=device)
    v = object.__new__(_StubValidator)
    v.config = config
    return v._setup_device()


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

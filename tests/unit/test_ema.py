import math

import pytest
import torch.nn as nn

from libreyolo.training.ema import ModelEMA


pytestmark = pytest.mark.unit


def test_model_ema_uses_configurable_tau():
    model = nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.993, tau=100)

    assert ema.decay(1) == pytest.approx(0.993 * (1 - math.exp(-1 / 100)))


def test_model_ema_ramped_set_decay_keeps_configured_tau():
    model = nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.993, tau=100)

    ema.set_decay(0.9, ramp=True)

    assert ema.decay(1) == pytest.approx(0.9 * (1 - math.exp(-1 / 100)))


def test_model_ema_tau_zero_uses_constant_decay():
    model = nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.993, tau=0)

    assert ema.decay(1) == pytest.approx(0.993)

    ema.set_decay(0.9, ramp=True)

    assert ema.decay(1) == pytest.approx(0.9)

"""Unit tests for the TinyFormer trainer wiring (no real training)."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.tinyformer]


def test_tinyformer_size_defaults_cover_all_sizes():
    from libreyolo.models.tinyformer.nn import SIZE_CONFIGS
    from libreyolo.training.config import (
        DEIMV2_SIZE_DEFAULTS,
        TINYFORMER_SIZE_DEFAULTS,
    )

    assert set(TINYFORMER_SIZE_DEFAULTS) == set(SIZE_CONFIGS)
    # s/m/l/x reuse DEIMv2's DINO recipes verbatim; xl halves the base LR.
    for size in ("s", "m", "l", "x"):
        assert TINYFORMER_SIZE_DEFAULTS[size] == DEIMV2_SIZE_DEFAULTS[size]
    assert TINYFORMER_SIZE_DEFAULTS["xl"]["lr0"] == pytest.approx(2.5e-4)
    assert TINYFORMER_SIZE_DEFAULTS["xl"]["backbone_lr_mult"] == pytest.approx(0.02)


def test_tinyformer_trainer_config_class():
    from libreyolo.models.tinyformer.trainer import TinyFormerTrainer
    from libreyolo.training.config import TinyFormerConfig

    assert TinyFormerTrainer._config_class() is TinyFormerConfig


def test_tinyformer_optimizer_groups_split_dinov3_tower():
    """Only backbone.dinov3.* gets the reduced LR; the SSA stem and the
    proj_c* heads train at the base LR (upstream optimizer regex semantics)."""
    from libreyolo.models.tinyformer.nn import LibreTinyFormerModel
    from libreyolo.models.tinyformer.trainer import TinyFormerTrainer

    model = LibreTinyFormerModel(config="s", nb_classes=80)

    trainer = TinyFormerTrainer.__new__(TinyFormerTrainer)
    trainer.model = model

    class _Cfg:
        size = "s"
        lr0 = 5e-4
        weight_decay = 1e-4
        backbone_lr_mult = 0.05

    trainer.config = _Cfg()

    optimizer = trainer._setup_optimizer()
    assert isinstance(optimizer, torch.optim.AdamW)

    lows = [g for g in optimizer.param_groups if g["lr_mult"] == pytest.approx(0.05)]
    highs = [g for g in optimizer.param_groups if g["lr_mult"] == 1.0]
    assert lows and highs

    low_param_ids = {id(p) for g in lows for p in g["params"]}
    named = dict(model.named_parameters())
    dinov3_ids = {
        id(p) for n, p in named.items() if n.startswith("backbone.dinov3.")
    }
    sda_ids = {
        id(p)
        for n, p in named.items()
        if n.startswith(("backbone.sda.", "backbone.proj_c"))
    }
    assert dinov3_ids <= low_param_ids
    assert not (sda_ids & low_param_ids)

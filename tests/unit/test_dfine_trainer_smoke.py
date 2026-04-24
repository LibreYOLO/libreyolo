"""DFINETrainer smoke tests — wiring only, no data."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

from libreyolo import LibreDFINE


def test_trainer_target_translation_smoke():
    """Drive on_forward manually with synthetic padded targets.

    Goal: the (B, max_labels, 5) → list[dict] translation works, model+criterion
    run, and a backward pass produces gradients on trainable params.
    """
    from libreyolo.models.dfine.trainer import DFINETrainer

    wrapper = LibreDFINE(None, size="n", device="cpu")
    wrapper.model.train()

    trainer = DFINETrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=640,
        device="cpu",
        amp=False,
        ema=False,
        no_aug_epochs=0,
        warmup_epochs=0,
        eval_interval=-1,
    )
    trainer.on_setup()  # build criterion

    # Synthetic batch: 2 images, padded targets in pixel cxcywh on a 640×640 grid.
    imgs = torch.randn(2, 3, 640, 640)
    targets = torch.zeros(2, 120, 5)
    # Image 0: 2 boxes
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 100.0, 80.0])
    targets[0, 1] = torch.tensor([17.0, 200.0, 200.0, 60.0, 40.0])
    # Image 1: 1 box
    targets[1, 0] = torch.tensor([1.0, 400.0, 320.0, 120.0, 100.0])

    out = trainer.on_forward(imgs, targets)
    assert "total_loss" in out
    assert torch.isfinite(out["total_loss"]), "total_loss must be finite"
    assert out["total_loss"].item() > 0
    for k in ("loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf"):
        assert k in out

    out["total_loss"].backward()

    # Some non-frozen params must have nonzero grad.
    nonzero_grads = sum(
        1
        for p in wrapper.model.encoder.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert nonzero_grads > 0, "encoder must have at least one param with nonzero grad"


def test_trainer_handles_empty_targets():
    """A batch where one image has zero GT boxes still works (no NaN, finite loss)."""
    from libreyolo.models.dfine.trainer import DFINETrainer

    wrapper = LibreDFINE(None, size="n", device="cpu")
    wrapper.model.train()

    trainer = DFINETrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=640,
        device="cpu",
        amp=False,
        ema=False,
        no_aug_epochs=0,
        warmup_epochs=0,
        eval_interval=-1,
    )
    trainer.on_setup()

    imgs = torch.randn(2, 3, 640, 640)
    targets = torch.zeros(2, 120, 5)
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 100.0, 80.0])
    # Image 1: all padding (no boxes)

    out = trainer.on_forward(imgs, targets)
    assert torch.isfinite(out["total_loss"])


def test_optimizer_setup_groups_params_correctly():
    """Norm + bias params should land in the no-WD group; conv weights in WD group."""
    from libreyolo.models.dfine.trainer import DFINETrainer

    wrapper = LibreDFINE(None, size="n", device="cpu")
    trainer = DFINETrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=640,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
    )
    optimizer = trainer._setup_optimizer()
    groups = optimizer.param_groups
    assert len(groups) == 2
    wd_group, no_wd_group = groups
    assert wd_group["weight_decay"] > 0
    assert no_wd_group["weight_decay"] == 0
    assert sum(p.numel() for p in wd_group["params"]) > 0
    assert sum(p.numel() for p in no_wd_group["params"]) > 0


def test_optimizer_layernorm_weights_in_no_wd_group():
    """Regression: transformer norm1/norm2/norm3 weights MUST land in no-WD group.

    A bug in the original pattern (``.norm.`` only) silently put 8 LayerNorm
    weights into the weight-decay group, mis-regularizing decoder/encoder norms.
    """
    from libreyolo.models.dfine.trainer import DFINETrainer

    wrapper = LibreDFINE(None, size="n", device="cpu")
    trainer = DFINETrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=640,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
    )
    optimizer = trainer._setup_optimizer()
    wd_group, no_wd_group = optimizer.param_groups

    # Build name lookup so we can assert by name.
    name_by_id = {id(p): n for n, p in wrapper.model.named_parameters()}
    no_wd_names = {name_by_id[id(p)] for p in no_wd_group["params"]}

    expected_in_no_wd = [
        "encoder.encoder.0.layers.0.norm1.weight",
        "encoder.encoder.0.layers.0.norm2.weight",
        "decoder.decoder.layers.0.norm1.weight",
        "decoder.decoder.layers.0.norm3.weight",
        "decoder.decoder.layers.1.norm1.weight",
        "decoder.decoder.layers.1.norm3.weight",
        "decoder.decoder.layers.2.norm1.weight",
        "decoder.decoder.layers.2.norm3.weight",
    ]
    missing = [n for n in expected_in_no_wd if n not in no_wd_names]
    assert not missing, (
        f"These transformer norm weights should be no-WD but aren't: {missing}"
    )


def test_scheduler_warmup_then_flat():
    """FlatCosineScheduler: warmup → flat → cosine, no NaNs."""
    from libreyolo.training.scheduler import FlatCosineScheduler

    sched = FlatCosineScheduler(
        lr=0.001,
        iters_per_epoch=10,
        total_epochs=10,
        warmup_epochs=2,
        warmup_lr_start=1e-6,
        no_aug_epochs=2,
        min_lr_ratio=0.05,
    )
    # iter 0 → warmup_lr_start
    assert abs(sched.update_lr(0) - 1e-6) < 1e-7
    # iter 20 → end of warmup → ~lr
    assert abs(sched.update_lr(20) - 0.001) < 1e-6
    # iter 40 → flat
    assert abs(sched.update_lr(40) - 0.001) < 1e-6
    # iter 100 → end of cosine → min_lr (0.001 * 0.05)
    final = sched.update_lr(100)
    assert abs(final - 0.001 * 0.05) < 1e-5


def test_set_decay_changes_ema_behavior():
    """ModelEMA.set_decay swaps the schedule lambda."""
    from libreyolo.training.ema import ModelEMA

    m = LibreDFINE(None, size="n", device="cpu").model
    ema = ModelEMA(m, decay=0.9999)
    pre = ema.decay(1000)
    ema.set_decay(0.99, ramp=False)
    post = ema.decay(1000)
    assert abs(post - 0.99) < 1e-9
    assert post != pre

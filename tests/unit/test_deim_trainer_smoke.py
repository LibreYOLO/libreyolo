"""DEIMTrainer smoke tests — wiring only, no data."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from libreyolo import LibreDEIM

pytestmark = pytest.mark.unit


def _build_trainer(wrapper, **overrides):
    from libreyolo.models.deim.trainer import DEIMTrainer

    kwargs = dict(
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
    kwargs.update(overrides)
    return DEIMTrainer(**kwargs)


def test_trainer_metadata():
    """Family tag, model tag, and config class must reflect deim."""
    from libreyolo.training.config import DEIMConfig

    wrapper = LibreDEIM(None, size="n", device="cpu")
    trainer = _build_trainer(wrapper)
    assert trainer.get_model_family() == "deim"
    assert trainer.get_model_tag() == "DEIM-n"
    assert trainer._config_class() is DEIMConfig


def test_trainer_target_translation_smoke():
    """Drive on_forward manually with synthetic padded targets.

    Goal: the (B, max_labels, 5) → list[dict] translation works, the DEIM
    criterion runs, and a backward pass produces gradients on trainable
    params.
    """
    wrapper = LibreDEIM(None, size="n", device="cpu")
    wrapper.model.train()
    trainer = _build_trainer(wrapper)
    trainer.on_setup()

    imgs = torch.randn(2, 3, 640, 640)
    targets = torch.zeros(2, 120, 5)
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 100.0, 80.0])
    targets[0, 1] = torch.tensor([17.0, 200.0, 200.0, 60.0, 40.0])
    targets[1, 0] = torch.tensor([1.0, 400.0, 320.0, 120.0, 100.0])

    out = trainer.on_forward(imgs, targets)
    assert "total_loss" in out
    assert torch.isfinite(out["total_loss"]), "total_loss must be finite"
    assert out["total_loss"].item() > 0
    # MAL replaces VFL in the DEIM loss menu.
    assert any(k == "loss_mal" or k.startswith("loss_mal_") for k in out), sorted(out)
    # Each remaining loss family must appear.
    for prefix in ("loss_bbox", "loss_giou", "loss_fgl", "loss_ddf"):
        assert any(k == prefix or k.startswith(prefix + "_") for k in out), (
            f"no key matching {prefix}* in output: {sorted(out)}"
        )

    out["total_loss"].backward()
    nonzero_grads = sum(
        1
        for p in wrapper.model.encoder.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    assert nonzero_grads > 0, "encoder must have at least one param with nonzero grad"


def test_trainer_does_not_use_vfl_loss():
    """DEIM intentionally uses MAL instead of VFL. Loss output must not
    contain a bare ``loss_vfl`` key (and the menu must include ``mal``)."""
    wrapper = LibreDEIM(None, size="n", device="cpu")
    wrapper.model.train()
    trainer = _build_trainer(wrapper)
    trainer.on_setup()

    imgs = torch.randn(2, 3, 640, 640)
    targets = torch.zeros(2, 120, 5)
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 100.0, 80.0])

    out = trainer.on_forward(imgs, targets)
    assert "mal" in trainer.criterion.losses
    assert "vfl" not in trainer.criterion.losses
    assert not any(k == "loss_vfl" for k in out)


def test_amp_train_loop_uses_on_forward_for_polygon_passthrough():
    """The AMP branch must not bypass on_forward, or segment polygons get dropped."""
    from libreyolo.models.deim.trainer import DEIMTrainer

    class OneBatchLoader:
        def __init__(self, batch):
            self.batch = batch
            self.dataset = SimpleNamespace()
            self.collate_fn = None

        def __iter__(self):
            yield self.batch

        def __len__(self):
            return 1

    class FakeScaler:
        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            return None

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            return None

    param = torch.nn.Parameter(torch.tensor(1.0))
    polygons = [[[
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy(),
    ]]]
    imgs = torch.zeros(1, 3, 16, 16)
    targets = torch.zeros(1, 2, 5)
    batch = (imgs, targets, ((16, 16),), (0,), polygons)

    trainer = DEIMTrainer.__new__(DEIMTrainer)
    trainer.train_loader = OneBatchLoader(batch)
    trainer.config = SimpleNamespace(
        epochs=1,
        clip_max_norm=0.0,
        log_interval=1,
        eval_interval=-1,
    )
    trainer.model = torch.nn.Linear(1, 1)
    trainer.device = torch.device("cpu")
    trainer.scaler = FakeScaler()
    trainer.optimizer = torch.optim.SGD([param], lr=0.1)
    trainer.ema_model = None
    trainer.tensorboard_writer = None
    trainer.lr_scheduler = SimpleNamespace(update_lr=lambda _: 0.1)
    trainer.get_loss_components = lambda outputs: {}

    seen = {}

    def on_forward(batch_imgs, batch_targets, polygons=None):
        seen["imgs"] = batch_imgs
        seen["targets"] = batch_targets
        seen["polygons"] = polygons
        return {"total_loss": param.sum() * 0.0 + 1.0}

    trainer.on_forward = on_forward

    avg_loss, val_metrics = DEIMTrainer._train_epoch(trainer, 0)

    assert avg_loss == pytest.approx(1.0)
    assert val_metrics is None
    assert seen["polygons"] is polygons
    assert seen["imgs"].device.type == "cpu"


def test_trainer_handles_empty_targets():
    """A batch where one image has zero GT boxes still works (no NaN)."""
    wrapper = LibreDEIM(None, size="n", device="cpu")
    wrapper.model.train()
    trainer = _build_trainer(wrapper)
    trainer.on_setup()

    imgs = torch.randn(2, 3, 640, 640)
    targets = torch.zeros(2, 120, 5)
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 100.0, 80.0])
    # Image 1: all padding (no boxes)

    out = trainer.on_forward(imgs, targets)
    assert torch.isfinite(out["total_loss"])


def test_optimizer_setup_groups_params_correctly():
    """4 param groups: {backbone, head} × {wd, no-wd}."""
    wrapper = LibreDEIM(None, size="n", device="cpu")
    trainer = _build_trainer(wrapper)
    optimizer = trainer._setup_optimizer()
    groups = optimizer.param_groups

    assert len(groups) == 4
    wd_groups = [g for g in groups if g["weight_decay"] > 0]
    no_wd_groups = [g for g in groups if g["weight_decay"] == 0]
    assert len(wd_groups) == 2 and len(no_wd_groups) == 2
    for g in groups:
        assert sum(p.numel() for p in g["params"]) > 0


def test_optimizer_in_proj_bias_in_no_wd_group():
    """Self-attn ``in_proj_bias`` parameters must land in the no-WD group
    (matches upstream's ``(?:norm|bn|bias)`` substring regex). The previous
    ``endswith('.bias')`` check missed them."""
    wrapper = LibreDEIM(None, size="n", device="cpu")
    trainer = _build_trainer(wrapper)
    optimizer = trainer._setup_optimizer()
    name_by_id = {id(p): n for n, p in wrapper.model.named_parameters()}

    in_proj_bias_names = [
        n for n in name_by_id.values() if n.endswith("in_proj_bias")
    ]
    assert len(in_proj_bias_names) >= 1

    no_wd_names = set()
    for g in optimizer.param_groups:
        if g["weight_decay"] == 0:
            no_wd_names.update(name_by_id[id(p)] for p in g["params"])

    misclassified = [n for n in in_proj_bias_names if n not in no_wd_names]
    assert not misclassified, f"in_proj_bias under WD: {misclassified}"


def test_optimizer_backbone_lr_mult():
    """Backbone groups carry ``lr_mult=backbone_lr_mult``; head groups stay 1.0."""
    wrapper = LibreDEIM(None, size="n", device="cpu")
    trainer = _build_trainer(wrapper, backbone_lr_mult=0.5)
    optimizer = trainer._setup_optimizer()

    by_mult = {}
    for pg in optimizer.param_groups:
        by_mult.setdefault(pg["lr_mult"], []).append(pg)
    assert sorted(by_mult.keys()) == [0.5, 1.0]
    assert sum(len(pg["params"]) for pg in by_mult[1.0]) > 0
    assert sum(len(pg["params"]) for pg in by_mult[0.5]) > 0


def test_on_mosaic_disable_resets_ema_decay():
    """DEIM's published recipe restarts EMA with a constant decay for the
    final no-aug phase. on_mosaic_disable must call set_decay on the EMA."""
    from types import SimpleNamespace

    from libreyolo.training.ema import ModelEMA

    wrapper = LibreDEIM(None, size="n", device="cpu")
    trainer = _build_trainer(
        wrapper, ema=True, ema_decay=0.9999, ema_restart_decay=0.99
    )
    # Stub the parts of BaseTrainer.on_mosaic_disable that the unit isn't
    # exercising — train_loader is built in setup() which we don't run.
    trainer.train_loader = SimpleNamespace(dataset=SimpleNamespace())
    trainer.ema_model = ModelEMA(wrapper.model, decay=0.9999)

    pre = trainer.ema_model.decay(1000)
    trainer.on_mosaic_disable()
    post = trainer.ema_model.decay(1000)
    assert abs(post - 0.99) < 1e-9
    assert post != pre


def test_train_transform_smoke():
    """DEIMTrainTransform produces a (3, H, W) float32 image and preserves
    at least the GT box that survives augs."""
    import numpy as np

    from libreyolo.models.deim.transforms import DEIMTrainTransform

    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    targets = np.array([[100, 100, 200, 200, 0]], dtype=np.float32)
    t = DEIMTrainTransform(max_labels=120, flip_prob=0.0, imgsz=640)

    img_out, padded = t(img, targets, (640, 640))
    assert img_out.shape == (3, 640, 640)
    assert img_out.dtype == np.float32
    assert int((padded[:, 3] > 0).sum()) >= 1

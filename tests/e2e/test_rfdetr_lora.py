"""RF-DETR LoRA fine-tuning tests.

Exercise the real user path (``lora=True``) through the trainer: ``on_setup``
injects DoRA adapters into the DINOv2 backbone, the base ViT weights are frozen,
the optimizer only collects trainable params, and a few ``on_forward`` +
backward steps reduce the loss while updating adapters and leaving the base
weights untouched. The recipe is a faithful match of the RF-DETR reference
(DoRA, rank 16, alpha 16, attention query/key/value only). The nano model builds
offline (random init, no downloads).
"""

from __future__ import annotations

import warnings

import pytest
import torch

from libreyolo.models.rfdetr.model import LibreRFDETR
from libreyolo.models.rfdetr.trainer import RFDETRTrainer

pytestmark = [pytest.mark.e2e, pytest.mark.rfdetr, pytest.mark.slow]

peft = pytest.importorskip("peft")
from peft import PeftModel  # noqa: E402

NANO_IMGSZ = 384  # divisible by patch_size(16) * num_windows(2) = 32


def _build_trainer(
    lora: bool, num_classes: int = 2
) -> tuple[RFDETRTrainer, LibreRFDETR]:
    # Empty-dict model_path -> no weights loaded, random init, no network.
    wrapper = LibreRFDETR({}, size="n", device="cpu")
    wrapper.model.train()
    trainer = RFDETRTrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size="n",
        num_classes=num_classes,
        data=None,
        epochs=1,
        batch=2,
        imgsz=NANO_IMGSZ,
        device="cpu",
        amp=False,
        ema=False,
        warmup_epochs=0,
        eval_interval=-1,
        lora=lora,
    )
    return trainer, wrapper


def _synthetic_batch(num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.randn(2, 3, NANO_IMGSZ, NANO_IMGSZ)
    # padded targets (B, max_labels, 5) = [cls, cx, cy, w, h] in pixels
    targets = torch.zeros(2, 120, 5)
    targets[0, 0] = torch.tensor([0.0, 192.0, 192.0, 80.0, 60.0])
    targets[0, 1] = torch.tensor([1.0, 120.0, 100.0, 40.0, 40.0])
    targets[1, 0] = torch.tensor([1.0, 240.0, 200.0, 60.0, 50.0])
    return imgs, targets


def test_lora_injects_freezes_base_and_reduces_trainable_params():
    trainer, wrapper = _build_trainer(lora=True)
    core = wrapper.model.model

    trainable_before = sum(p.numel() for p in core.parameters() if p.requires_grad)
    trainer.on_setup()  # triggers apply_lora + head reinit + criterion build
    trainable_after = sum(p.numel() for p in core.parameters() if p.requires_grad)

    # encoder wrapped, base frozen, adapters present and trainable
    assert isinstance(core.backbone[0].encoder, PeftModel)
    assert trainable_after < trainable_before

    lora_params = [(n, p) for n, p in core.named_parameters() if "lora_" in n]
    base_enc = [
        (n, p)
        for n, p in core.named_parameters()
        if "backbone.0.encoder" in n and "lora_" not in n
    ]
    assert lora_params, "no LoRA adapters were created"
    assert all(p.requires_grad for _, p in lora_params)
    assert all(not p.requires_grad for _, p in base_enc), (
        "base ViT weights must be frozen"
    )


@pytest.mark.parametrize("size", ["n", "s", "m", "l"])
def test_lora_injects_across_all_detection_sizes(size):
    """Every RF-DETR detection size shares the DINOv2 backbone, so LoRA must
    inject and reduce trainable params for all of them, not just nano."""
    from libreyolo.models.rfdetr.nn import create_rfdetr_model
    from libreyolo.training.lora import apply_lora_to_rfdetr, count_trainable_parameters

    core = create_rfdetr_model(config=size, nb_classes=2, device="cpu").model
    before, total = count_trainable_parameters(core)
    apply_lora_to_rfdetr(core)
    after, _ = count_trainable_parameters(core)

    assert isinstance(core.backbone[0].encoder, PeftModel)
    assert after < before
    assert any("lora_" in n for n, _ in core.named_parameters())


def test_lora_recipe_is_dora_on_attention_only():
    """Faithful to the RF-DETR reference: DoRA (weight-decomposed LoRA) on the
    attention query/key/value projections, and NOT on the MLP fc1/fc2."""
    trainer, wrapper = _build_trainer(lora=True)
    trainer.on_setup()
    core = wrapper.model.model
    names = [n for n, _ in core.named_parameters()]

    # DoRA marker: per-layer magnitude vectors exist and are trainable.
    mags = [(n, p) for n, p in core.named_parameters() if "lora_magnitude_vector" in n]
    assert mags, "DoRA magnitude vectors missing (use_dora not active)"
    assert all(p.requires_grad for _, p in mags)

    # attention q/k/v are adapted
    assert any(".query.lora_A" in n for n in names)
    assert any(".key.lora_A" in n for n in names)
    assert any(".value.lora_A" in n for n in names)
    # MLP is NOT adapted (upstream targets attention only)
    assert not any("fc1.lora_A" in n for n in names)
    assert not any("fc2.lora_A" in n for n in names)


def test_lora_merges_on_export():
    """DoRA adapters must fold back into dense weights on export so the deployed
    model carries no peft dependency."""
    trainer, wrapper = _build_trainer(lora=True)
    trainer.on_setup()
    core = wrapper.model.model
    backbone = core.backbone[0]
    assert isinstance(backbone.encoder, PeftModel)

    # one optimizer step so the adapters are non-trivial before merging
    optimizer = trainer._setup_optimizer()
    imgs, targets = _synthetic_batch(num_classes=2)
    out = trainer.on_forward(imgs, targets)
    out["total_loss"].backward()
    optimizer.step()

    backbone.export()  # triggers merge_and_unload on the PeftModel encoder

    assert not isinstance(backbone.encoder, PeftModel), "adapters not merged on export"
    assert not any("lora_" in n for n, _ in backbone.encoder.named_parameters()), (
        "peft adapter tensors remain after merge"
    )


def test_lora_optimizer_only_collects_trainable_params():
    trainer, wrapper = _build_trainer(lora=True)
    trainer.on_setup()
    optimizer = trainer._setup_optimizer()

    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    # every optimizer param must be trainable
    for p in (p for g in optimizer.param_groups for p in g["params"]):
        assert p.requires_grad
    # at least one LoRA adapter param made it into the optimizer
    core = wrapper.model.model
    lora_ids = {id(p) for n, p in core.named_parameters() if "lora_" in n}
    assert opt_param_ids & lora_ids, "no LoRA params landed in the optimizer"
    # no frozen base-encoder param leaked in
    frozen_ids = {
        id(p)
        for n, p in core.named_parameters()
        if "backbone.0.encoder" in n and "lora_" not in n
    }
    assert not (opt_param_ids & frozen_ids), "frozen base params leaked into optimizer"


def test_lora_backward_updates_adapters_and_leaves_base_frozen():
    torch.manual_seed(0)
    trainer, wrapper = _build_trainer(lora=True)
    trainer.on_setup()
    core = wrapper.model.model

    # snapshot a frozen base query weight and a zero-init lora_B
    base_q_name = next(
        n
        for n, _ in core.named_parameters()
        if n.endswith("attention.attention.query.base_layer.weight")
    )
    base_q_ref = dict(core.named_parameters())[base_q_name].detach().clone()
    lora_B_name = next(n for n, _ in core.named_parameters() if "lora_B" in n)
    assert torch.count_nonzero(dict(core.named_parameters())[lora_B_name]) == 0

    optimizer = trainer._setup_optimizer()
    imgs, targets = _synthetic_batch(num_classes=2)

    out = trainer.on_forward(imgs, targets)
    loss = out["total_loss"]
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    assert dict(core.named_parameters())[lora_B_name].grad is not None
    optimizer.step()

    # adapters moved, base stayed put
    assert torch.count_nonzero(dict(core.named_parameters())[lora_B_name]) > 0
    assert torch.allclose(
        dict(core.named_parameters())[base_q_name].detach(), base_q_ref
    )


def test_lora_checkpoint_reloads_into_fresh_model():
    """A lora=True checkpoint must load into a fresh model: the loader replays the
    adapter injection so the saved PeftModel keys line up instead of being
    rejected as unexpected."""
    from libreyolo.training.lora import apply_lora_to_rfdetr

    # build an in-memory adapter checkpoint
    w1 = LibreRFDETR({}, size="n", device="cpu")
    apply_lora_to_rfdetr(w1.model.model)
    sd = dict(w1.model.state_dict())
    assert any("lora_" in k for k in sd), "checkpoint should carry adapter keys"
    ckpt = {"model_family": "rfdetr", "nc": 80, **sd}

    # a fresh model is not wrapped until the loader replays the injection
    w2 = LibreRFDETR({}, size="n", device="cpu")
    assert not isinstance(w2.model.model.backbone[0].encoder, PeftModel)
    w2._load_weights(ckpt)  # must not raise on the adapter keys
    assert isinstance(w2.model.model.backbone[0].encoder, PeftModel), (
        "loader did not replay LoRA injection for an adapter checkpoint"
    )


def test_lora_wrapped_checkpoint_reloads_into_fresh_model():
    from libreyolo.training.lora import apply_lora_to_rfdetr
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    w1 = LibreRFDETR({}, size="n", device="cpu")
    apply_lora_to_rfdetr(w1.model.model)
    ckpt = wrap_libreyolo_checkpoint(
        dict(w1.model.state_dict()),
        model_family="rfdetr",
        size="n",
        task="detect",
        nc=80,
        imgsz=NANO_IMGSZ,
        config={"lora": True},
    )

    w2 = LibreRFDETR({}, size="n", device="cpu")
    assert not isinstance(w2.model.model.backbone[0].encoder, PeftModel)
    w2._load_weights(ckpt)
    assert isinstance(w2.model.model.backbone[0].encoder, PeftModel)


def test_lora_apply_is_idempotent_on_wrapped_encoder():
    from libreyolo.training.lora import apply_lora_to_rfdetr

    wrapper = LibreRFDETR({}, size="n", device="cpu")
    apply_lora_to_rfdetr(wrapper.model.model)
    encoder = wrapper.model.model.backbone[0].encoder

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        wrapped_again = apply_lora_to_rfdetr(wrapper.model.model)

    assert wrapped_again is encoder
    assert wrapper.model.model.backbone[0].encoder is encoder
    assert not [
        w
        for w in seen
        if "PEFT" in str(w.message) or "multiple adapters" in str(w.message)
    ]


def test_lora_resume_checkpoint_enables_lora_before_setup(monkeypatch, tmp_path):
    from libreyolo.training.lora import apply_lora_to_rfdetr
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    source = LibreRFDETR({}, size="n", device="cpu")
    apply_lora_to_rfdetr(source.model.model)
    ckpt = wrap_libreyolo_checkpoint(
        dict(source.model.state_dict()),
        model_family="rfdetr",
        size="n",
        task="detect",
        nc=80,
        imgsz=NANO_IMGSZ,
        epoch=0,
        config={"lora": True},
    )
    resume_path = tmp_path / "last.pt"
    torch.save(ckpt, resume_path)

    captured = {}

    class _FakeTrainer:
        def __init__(self, model, wrapper_model=None, **kwargs):
            captured["kwargs"] = kwargs

        def setup(self):
            captured["setup"] = True

        def resume(self, path):
            captured["resume"] = path

        def train(self):
            return {"save_dir": str(tmp_path / "run")}

    monkeypatch.setattr("libreyolo.models.rfdetr.model.RFDETRTrainer", _FakeTrainer)

    wrapper = LibreRFDETR({}, size="n", device="cpu")
    wrapper.train(
        data="dummy.yaml",
        resume=resume_path,
        epochs=1,
        project=tmp_path,
        name="run",
        exist_ok=True,
    )

    assert captured["kwargs"]["lora"] is True
    assert captured["setup"] is True
    assert captured["resume"] == str(resume_path)


def test_lora_checkpoint_continue_train_does_not_reapply_adapters():
    from libreyolo.training.lora import apply_lora_to_rfdetr

    w1 = LibreRFDETR({}, size="n", device="cpu")
    apply_lora_to_rfdetr(w1.model.model)
    ckpt = {"model_family": "rfdetr", "nc": 80, **dict(w1.model.state_dict())}

    w2 = LibreRFDETR({}, size="n", device="cpu")
    w2._load_weights(ckpt)
    encoder = w2.model.model.backbone[0].encoder

    trainer = RFDETRTrainer(
        model=w2.model,
        wrapper_model=w2,
        size="n",
        num_classes=80,
        data=None,
        epochs=1,
        batch=2,
        imgsz=NANO_IMGSZ,
        device="cpu",
        amp=False,
        ema=False,
        eval_interval=-1,
        lora=True,
    )
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        trainer.on_setup()

    assert w2.model.model.backbone[0].encoder is encoder
    assert not [
        w
        for w in seen
        if "PEFT" in str(w.message) or "multiple adapters" in str(w.message)
    ]


def test_lora_rejected_on_unsupported_family():
    """A pure-CNN family must hard-error on lora=True rather than silently ignore it."""
    from libreyolo import LibreDFINE
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
        lora=True,
    )
    assert DFINETrainer.supports_lora is False
    with pytest.raises(ValueError, match="LoRA fine-tuning"):
        trainer.setup()

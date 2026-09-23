"""CPU contracts for the GTR detection port."""

import pytest
import torch

from libreyolo import LibreGTR, LibreYOLO
from libreyolo.models.gtr.attention import recurrent_gla
from libreyolo.models.gtr.nn import LibreGTRModel
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def cpu_threads():
    import random

    import numpy as np

    previous = torch.get_num_threads()
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)


def test_recurrence_matches_explicit_causal_sum_and_gradients():
    torch.manual_seed(8)
    q, k, v = [torch.randn(1, 5, 2, 3, requires_grad=True) for _ in range(3)]
    g = torch.nn.functional.logsigmoid(torch.randn(1, 5, 2, 3))
    actual = recurrent_gla(q, k, v, g)
    outputs = []
    for t in range(5):
        terms = []
        for j in range(t + 1):
            decay = g[:, j + 1 : t + 1].sum(1).exp()
            score = (q[:, t] * k[:, j] * decay).sum(-1) / 3**0.5
            terms.append(score.unsqueeze(-1) * v[:, j])
        outputs.append(sum(terms))
    expected = torch.stack(outputs, 1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    left = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    right = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("size", ["s", "m", "l", "x"])
def test_sizes_and_bidirectional_recognition(size):
    from libreyolo.models.deim.model import LibreDEIM
    from libreyolo.models.dfine.model import LibreDFINE
    from libreyolo.models.ec.model import LibreEC
    from libreyolo.models.yolox.model import LibreYOLOX

    model = LibreGTRModel(size, nb_classes=3, eval_spatial_size=(160, 160)).eval()
    sd = model.state_dict()
    assert LibreGTR.can_load(sd)
    assert LibreGTR.detect_size(sd) == size
    assert LibreGTR.detect_nb_classes(sd) == 3
    for cls in (LibreDFINE, LibreDEIM, LibreEC, LibreYOLOX):
        assert not cls.can_load(sd)
    assert not LibreGTR.can_load(
        {"decoder.pre_bbox_head.layers.0.weight": torch.zeros(1)}
    )
    with torch.no_grad():
        out = model(torch.randn(1, 3, 160, 160))
    assert out["pred_logits"].shape == (1, 300, 3)
    assert out["pred_boxes"].shape == (1, 300, 4)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_checkpoint_roundtrip_and_missing_weight_rejected(tmp_path):
    model = LibreGTRModel(nb_classes=2)
    ckpt = wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="gtr",
        size="s",
        nc=2,
        names={0: "a", 1: "b"},
        task="detect",
        imgsz=640,
    )
    path = tmp_path / "LibreGTRs.pt"
    torch.save(ckpt, path)
    loaded = LibreYOLO(str(path), device="cpu")
    assert isinstance(loaded, LibreGTR)
    assert not loaded.model.training
    assert loaded.names == {0: "a", 1: "b"}
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            value, loaded.model.state_dict()[key], rtol=0, atol=0
        )
    ckpt["model"].pop("backbone.backbone._model.blocks.0.attn.q_proj.weight")
    torch.save(ckpt, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        LibreYOLO(str(path), device="cpu")


def test_grouped_training_and_validation_loss():
    from libreyolo.models.gtr.config import GTRConfig
    from libreyolo.models.gtr.trainer import GTRTrainer

    trainer = object.__new__(GTRTrainer)
    trainer.config = GTRConfig(num_classes=2)
    trainer.device = torch.device("cpu")
    trainer.model = LibreGTRModel(nb_classes=2).train()
    trainer.criterion = trainer.build_criterion()
    images = torch.randn(1, 3, 160, 160)
    losses = trainer.on_forward(images, torch.tensor([[[1.0, 80.0, 80.0, 40.0, 32.0]]]))
    assert torch.isfinite(losses["total_loss"])
    losses["total_loss"].backward()
    grad = trainer.model.backbone.backbone._model.blocks[0].attn.q_proj.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    model = trainer.model.eval()
    adapter = trainer.build_validation_loss_adapter(model)
    with torch.no_grad(), adapter.forward_scope():
        prediction = model(images)
        result = adapter(
            prediction,
            torch.tensor([[[60.0, 64.0, 100.0, 96.0, 1.0]]]),
            image_size=(160, 160),
        )
    assert torch.isfinite(result["loss"])
    assert not model.decoder.emit_loss_outputs


def test_preprocess_and_topk_contract():
    from PIL import Image

    model = LibreGTR(None, size="s", device="cpu")
    tensor, _, shape, _ = model._preprocess(
        Image.new("RGB", (320, 200), "white"), input_size=160
    )
    torch.testing.assert_close(
        tensor[0, :, 0, 0],
        torch.tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]),
    )
    result = model._postprocess(
        {
            "pred_logits": torch.tensor([[[10.0, -10.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]]),
        },
        conf_thres=0.5,
        iou_thres=0.5,
        original_size=shape,
    )
    torch.testing.assert_close(
        torch.as_tensor(result["boxes"]), torch.tensor([[80.0, 50.0, 240.0, 150.0]])
    )
    with pytest.raises(ValueError, match="multiple of 32"):
        model._preprocess(Image.new("RGB", (50, 50)), input_size=144)


def test_export_gate_is_stable_for_extreme_logits(monkeypatch):
    from libreyolo.models.gtr.attention import GatedLinearAttention

    attention = GatedLinearAttention(hidden_size=64, num_heads=1).eval()
    with torch.no_grad():
        attention.gk_proj[1].weight.zero_()
        attention.gk_proj[1].bias.fill_(-1000)
    x = torch.randn(1, 5, 64)
    expected = attention(x)[0]
    monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
    actual = attention(x)[0]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "size,weight_decay,backbone_lr",
    [
        ("s", 1e-4, 0.03),
        ("m", 1e-4, 0.03),
        ("l", 1.25e-4, 0.005),
        ("x", 1.25e-4, 0.004),
    ],
)
def test_cli_and_python_share_size_recipe(size, weight_decay, backbone_lr):
    from libreyolo.cli.config import build_family_train_kwargs, get_family_defaults
    from libreyolo.models.gtr.config import GTRConfig

    defaults = get_family_defaults("gtr")
    cli = build_family_train_kwargs(defaults, "gtr", user_provided=set())
    assert cli == {}
    config = GTRConfig(size=size, **cli)
    assert config.weight_decay == weight_decay
    assert config.backbone_lr_mult == backbone_lr
    overrides = build_family_train_kwargs(
        {"weight_decay": 0.012, "lr0": 0.003},
        "gtr",
        user_provided={"weight_decay", "lr0"},
    )
    config = GTRConfig(size=size, **overrides)
    assert config.weight_decay == 0.012
    assert config.lr0 == 0.003


def test_gtr_scheduler_matches_upstream_recipe():
    from libreyolo.models.gtr.config import GTRConfig
    from libreyolo.models.gtr.scheduler import GTRScheduler

    schedule = GTRScheduler(0.0005, 1000, GTRConfig())
    # Independent values from the published 30-epoch, 6-flat, 2-tail recipe.
    assert schedule.update_lr(1000) == pytest.approx(0.000125)
    assert schedule.update_lr(6000) == pytest.approx(0.0005)
    assert schedule.update_lr(15000) == pytest.approx(0.0004102165696051787)
    assert schedule.update_lr(26000) == pytest.approx(0.00025506337829818784)
    assert schedule.update_lr(28000) == pytest.approx(0.00025)
    assert schedule.update_lr(30000) == pytest.approx(0.00025)
    short = GTRScheduler(0.001, 1, GTRConfig(epochs=1, warmup_epochs=0))
    assert short.update_lr(1) == pytest.approx(0.0005)


def test_resume_restores_saved_config_then_explicit_overrides(tmp_path, monkeypatch):
    import yaml

    from libreyolo.models.gtr.trainer import GTRTrainer

    data = tmp_path / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images",
                "val": "images",
                "names": ["one", "two"],
            }
        )
    )
    model = LibreGTRModel(nb_classes=2)
    checkpoint = wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="gtr",
        size="s",
        nc=2,
        task="detect",
        imgsz=160,
        names={0: "one", 1: "two"},
    )
    saved = {
        "data": str(data),
        "imgsz": 160,
        "batch": 2,
        "lr0": 0.001,
        "ema": False,
        "warmup_epochs": 0,
        "no_aug_epochs": 0,
        "weight_decay": 0.002,
        "seed": 42,
        "epochs": 7,
    }
    checkpoint.update(config=saved, epoch=5)
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    wrapper = LibreYOLO(str(path), device="cpu")
    captured = {}
    monkeypatch.setattr(GTRTrainer, "setup", lambda t: captured.update(config=t.config))
    monkeypatch.setattr(
        GTRTrainer, "resume", lambda t, source: captured.update(source=source)
    )
    monkeypatch.setattr(GTRTrainer, "train", lambda t: {})
    wrapper.train(resume=True, epochs=9, batch=3, device="cpu")
    config = captured["config"]
    for field in (
        "imgsz",
        "lr0",
        "ema",
        "warmup_epochs",
        "no_aug_epochs",
        "weight_decay",
        "seed",
    ):
        assert getattr(config, field) == saved[field]
    assert config.epochs == 9 and config.batch == 3
    assert captured["source"] == str(path)


def test_gtr_warns_about_ignored_augmentations():
    from libreyolo.cli.config import get_unsupported_train_params

    assert {"mosaic", "mixup", "degrees", "translate"} <= get_unsupported_train_params(
        "gtr"
    )


@pytest.mark.parametrize("kwargs", [{"optimizer": "sgd"}, {"scheduler": "cosine"}])
def test_unsupported_training_policy_is_rejected(kwargs):
    from libreyolo.models.gtr.config import GTRConfig

    with pytest.raises(ValueError, match="GTR currently supports"):
        GTRConfig(**kwargs)


def test_resume_keeps_resolved_optimizer_overrides(monkeypatch):
    from types import SimpleNamespace

    from libreyolo.models.gtr.config import GTRConfig
    from libreyolo.models.gtr.scheduler import GTRScheduler
    from libreyolo.models.gtr.trainer import GTRTrainer
    from libreyolo.training.trainer import BaseTrainer

    trainer = object.__new__(GTRTrainer)
    group = {"lr": 0.003, "lr_mult": 0.004, "weight_decay": 0.012}
    trainer.optimizer = SimpleNamespace(param_groups=[group])
    trainer.lr_scheduler = GTRScheduler(0.001, 1000, GTRConfig())
    trainer.start_epoch = 15
    monkeypatch.setattr(GTRTrainer, "_scheduler_steps_per_epoch", lambda t: 1000)

    def restore(t, path):
        t.optimizer.param_groups[0].update(lr=0.1, lr_mult=0.5, weight_decay=0.5)
        t.restored_path = path

    monkeypatch.setattr(BaseTrainer, "resume", restore)
    trainer.resume("last.pt")
    assert trainer.restored_path == "last.pt"
    assert group["lr_mult"] == 0.004
    assert group["weight_decay"] == 0.012
    assert group["lr"] == pytest.approx(0.0008204331392103574 * 0.004)


def test_training_transform_keeps_original_aspect_geometry(tmp_path):
    import numpy as np

    from libreyolo.models.gtr.config import GTRConfig
    from libreyolo.models.gtr.trainer import GTRTrainer

    trainer = object.__new__(GTRTrainer)
    trainer.config = GTRConfig(imgsz=160, flip_prob=0)
    transform, _ = trainer.create_transforms()
    assert transform.wants_unresized_image
    transform.disable_strong_augs()
    # A wide image must stretch to a square, not acquire letterbox padding.
    image = np.full((80, 320, 3), 255, dtype=np.uint8)
    targets = np.array([[80, 20, 240, 60, 0]], dtype=np.float32)
    tensor, labels = transform(image, targets, (160, 160))
    np.testing.assert_allclose(labels[0], [0, 80, 80, 80, 80])
    expected = np.array([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225])
    np.testing.assert_allclose(tensor[:, 0, 0], expected, atol=1e-6)


def test_scheduler_respects_overrides_that_fit_the_run():
    from libreyolo.models.gtr.config import GTRConfig
    from libreyolo.models.gtr.scheduler import GTRScheduler

    schedule = GTRScheduler(
        0.001, 1000, GTRConfig(warmup_epochs=5, flat_epochs=10, no_aug_epochs=8)
    )
    assert schedule.warmup_iters == 5000
    assert schedule.flat_iters == 10000
    assert schedule.tail_iters == 8000
    assert schedule.update_lr(5000) == pytest.approx(0.001)
    assert schedule.update_lr(10000) == pytest.approx(0.001)
    assert schedule.update_lr(22000) == pytest.approx(0.0005)

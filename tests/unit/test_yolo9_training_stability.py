"""YOLO9 training stability (issue #927): gradient clipping and fp32 matching."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def test_yolo9_family_configs_clip_gradient_norm_at_10():
    from libreyolo.models.yolo9_e2e.config import YOLO9E2EConfig
    from libreyolo.models.yolo9_p2.config import YOLO9P2Config
    from libreyolo.training.config import YOLO9Config

    for config in (YOLO9Config(), YOLO9P2Config(), YOLO9E2EConfig()):
        assert config.clip_max_norm == 10.0


@pytest.mark.parametrize(
    "kwargs,expected",
    [({}, 10.0), ({"clip_max_norm": 2.5}, 2.5), ({"clip_max_norm": 0}, 0.0)],
)
def test_yolo9_trainer_clip_max_norm_override(kwargs, expected):
    from libreyolo.models.yolo9.trainer import YOLO9Trainer
    from libreyolo.training.config import YOLO9Config

    trainer = object.__new__(YOLO9Trainer)
    trainer.config = YOLO9Config.from_kwargs(**kwargs)
    assert trainer._get_clip_max_norm() == expected
    assert trainer._should_clip_gradients() is (expected > 0)


def test_matcher_assigns_positives_with_fp16_background_logits():
    """Under fp16 the sigmoid of logits below about -17 is exactly 0; the
    matcher must still see fp32 scores and assign positives."""
    from libreyolo.models.yolo9.loss import YOLO9Loss

    assert torch.tensor(-20.0, dtype=torch.float16).sigmoid().item() == 0.0

    loss = YOLO9Loss(
        num_classes=2,
        reg_max=16,
        strides=[8, 16, 32],
        image_size=[64, 64],
        device=torch.device("cpu"),
        distributed_normalize=False,
    )
    raw = [torch.zeros(1, 4 * 16 + 2, s, s, dtype=torch.float16) for s in (8, 4, 2)]
    for level in raw:
        level[:, 4 * 16:] = -20.0
    targets = torch.tensor([[[1.0, 0.25, 0.25, 0.75, 0.75]]])

    seen = {}
    matcher = loss.matcher

    def _spy(target, predict):
        seen["dtypes"] = tuple(t.dtype for t in predict)
        return matcher(target, predict)

    loss.matcher = _spy
    out = loss(raw, targets)
    assert seen["dtypes"] == (torch.float32, torch.float32)
    assert float(out["num_fg"]) > 0
    assert torch.isfinite(out["total_loss"])


# -----------------------------------------------------------------------------
# PGI auxiliary branch defaults
# -----------------------------------------------------------------------------


def _save_yolo9t(tmp_path, *, with_aux, aux_class_neck=None):
    from libreyolo.models.yolo9.nn import YOLO9_CONFIGS, LibreYOLO9Model, YOLO9Head
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    raw = LibreYOLO9Model(config="t", nb_classes=2)
    if with_aux:
        raw.enable_aux(0.25)
        if aux_class_neck is not None:
            raw.aux_head = YOLO9Head(
                YOLO9_CONFIGS["t"]["head_channels"],
                2,
                reg_max=16,
                strides=(8, 16, 32),
                class_neck=aux_class_neck,
            )
    ckpt = wrap_libreyolo_checkpoint(
        raw.state_dict(),
        model_family="yolo9",
        size="t",
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
    )
    path = tmp_path / ("with_aux.pt" if with_aux else "no_aux.pt")
    torch.save(ckpt, path)
    return path


def _aux_attached_when_training(model, monkeypatch, **train_kwargs):
    """Run ``train()`` up to trainer construction; report whether PGI is attached."""
    seen = {}

    class _Trainer:
        def __init__(self, **kwargs):
            seen["aux"] = kwargs["model"].aux is not None

        def train(self):
            return {}

    monkeypatch.setattr(
        "libreyolo.data.load_data_config",
        lambda *args, **kwargs: {"nc": 2, "names": ["a", "b"]},
    )
    monkeypatch.setattr(type(model), "_trainer_class", lambda self: _Trainer)
    model.train(data="dummy.yaml", device="cpu", **train_kwargs)
    return seen["aux"]


def test_weights_without_pgi_tensors_train_the_main_head_only(tmp_path, monkeypatch):
    from libreyolo.models.yolo9.model import LibreYOLO9

    path = _save_yolo9t(tmp_path, with_aux=False)
    model = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert _aux_attached_when_training(model, monkeypatch) is False


def test_explicit_aux_weight_attaches_pgi_without_pretrained_tensors(tmp_path, monkeypatch):
    from libreyolo.models.yolo9.model import LibreYOLO9

    path = _save_yolo9t(tmp_path, with_aux=False)
    model = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert _aux_attached_when_training(model, monkeypatch, aux_weight=0.25) is True


def test_weights_with_pgi_tensors_attach_pgi_by_default(tmp_path, monkeypatch):
    from libreyolo.models.yolo9.model import LibreYOLO9

    path = _save_yolo9t(tmp_path, with_aux=True)
    model = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert _aux_attached_when_training(model, monkeypatch) is True


def test_from_scratch_training_attaches_pgi_by_default(monkeypatch):
    from libreyolo.models.yolo9.model import LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    assert _aux_attached_when_training(model, monkeypatch) is True


def test_pgi_head_takes_the_checkpoint_class_tower_width(tmp_path):
    """A PGI head whose class towers are wider than a fresh build loads fully."""
    from libreyolo.models.yolo9.model import LibreYOLO9

    path = _save_yolo9t(tmp_path, with_aux=True, aux_class_neck=128)
    saved = torch.load(path, map_location="cpu", weights_only=False)["model"]
    aux_keys = [k for k in saved if k.startswith(("aux.", "aux_head."))]

    model = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    model.model.enable_aux(0.25)
    assert model.model.aux_head.class_neck != 128
    loaded = model._reload_aux_from_path(str(path))
    assert model.model.aux_head.class_neck == 128
    assert loaded == len(aux_keys)


def test_scratch_ddp_worker_keeps_pgi(tmp_path, monkeypatch):
    """A DDP worker rebuilt from the bootstrap file trains the same graph as
    the from-scratch parent: the parent pins the PGI default before spawning."""
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.training.ddp_spawn import _bootstrap_checkpoint

    parent = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    train_kw = parent._ddp_prepare_train_kwargs({"data": "dummy.yaml"})
    assert train_kw["aux_weight"] == 0.25

    path = tmp_path / "bootstrap.pt"
    torch.save(_bootstrap_checkpoint(parent), path)
    worker = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert _aux_attached_when_training(
        worker, monkeypatch, aux_weight=train_kw["aux_weight"]
    ) is True


def test_ddp_prepare_leaves_fine_tunes_and_explicit_values_alone(tmp_path):
    from libreyolo.models.yolo9.model import LibreYOLO9

    scratch = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    assert scratch._ddp_prepare_train_kwargs({"aux_weight": 0})["aux_weight"] == 0
    assert "aux_weight" not in scratch._ddp_prepare_train_kwargs({"pretrained": True})
    assert "aux_weight" not in scratch._ddp_prepare_train_kwargs({"resume": True})

    path = _save_yolo9t(tmp_path, with_aux=False)
    fine_tune = LibreYOLO9(str(path), size="t", nb_classes=2, device="cpu")
    assert "aux_weight" not in fine_tune._ddp_prepare_train_kwargs({})


def test_ddp_aware_calls_the_family_hook_before_spawning(monkeypatch):
    from libreyolo.training import ddp_spawn

    seen = {}

    class _Model:
        def _ddp_prepare_train_kwargs(self, train_kw):
            return dict(train_kw, pinned=True)

        @ddp_spawn.ddp_aware()
        def train(self, data, device="", **kwargs):  # pragma: no cover
            raise AssertionError("the method body must not run on the parent")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        ddp_spawn,
        "spawn_for_model",
        lambda model, train_kw, nprocs, **kw: seen.update(train_kw) or {},
    )
    _Model().train("d.yaml", device="0,1")
    assert seen["pinned"] is True and seen["data"] == "d.yaml"


def test_pretrained_transfer_takes_the_pgi_class_tower_width(tmp_path):
    """``pretrained=`` loads a wider PGI head completely, like direct loading."""
    from libreyolo.models.yolo9.model import LibreYOLO9

    path = _save_yolo9t(tmp_path, with_aux=True, aux_class_neck=128)
    saved = torch.load(path, map_location="cpu", weights_only=False)["model"]
    aux_keys = [k for k in saved if k.startswith(("aux.", "aux_head."))]

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    model.model.enable_aux(0.25)
    stats = model._load_transfer_weights(path)
    assert model.model.aux_head.class_neck == 128
    assert stats["aux_loaded"] == len(aux_keys)
    assert stats["skipped"] == 0


# -----------------------------------------------------------------------------
# Name-bearing checkpoint metadata
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "legacy,current",
    [
        ("backbone.elan1.cv1.", "backbone.elan1.conv1."),
        ("backbone.elan1.cv1", "backbone.elan1.conv1"),
        ("neck.elan_up1.cv2.0.m.", "neck.elan_up1.conv2.0.bottleneck."),
        ("backbone.down2.cv", "backbone.down2.conv"),
        ("detect.cv3.", "head.class_convs."),
        ("head.cv2.1.", "head.anchor_convs.1."),
        ("aux_head.cv3", "aux_head.class_convs"),
        ("head.one2one_cv2.", "head.one_to_one_anchor_convs."),
        ("head.", "head."),
        ("backbone.conv0.", "backbone.conv0."),
        ("backbone.elan1.conv1.", "backbone.elan1.conv1."),
    ],
)
def test_upgrade_legacy_module_name(legacy, current):
    from libreyolo.models.yolo9.convert import upgrade_legacy_module_name

    assert upgrade_legacy_module_name(legacy) == current
    assert upgrade_legacy_module_name(current) == current


def test_quant_manifest_module_names_are_upgraded():
    from libreyolo.models.yolo9.model import LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    manifest = {
        "recipe": "int8",
        "keep_high_precision": ["head.", "backbone.conv0.", "backbone.elan1.cv1."],
        "fp8_tensorwise_weights": ("neck.elan_up1.cv1.conv",),
        "module_count": 207,
    }
    upgraded = model._upgrade_quant_manifest(manifest)
    assert upgraded["keep_high_precision"] == [
        "head.", "backbone.conv0.", "backbone.elan1.conv1.",
    ]
    assert upgraded["fp8_tensorwise_weights"] == ("neck.elan_up1.conv1.conv",)
    assert upgraded["module_count"] == 207
    assert manifest["keep_high_precision"][2] == "backbone.elan1.cv1."


def test_legacy_quantized_checkpoint_keeps_its_float_layers(tmp_path):
    """A quantized checkpoint whose keys and exclusions use the legacy names
    reloads with the same quantized-module count and the float layer intact."""
    from libreyolo import LibreYOLO
    from libreyolo.models.yolo9.model import LibreYOLO9

    model = LibreYOLO9(None, size="t", nb_classes=2, device="cpu")
    model.quantize(
        "int8",
        calib=None,
        keep_high_precision=("head.", "backbone.conv0.", "backbone.elan1.conv1."),
        verbose=False,
    )
    count = model.quant_info()["module_count"]
    float_weight = model.model.backbone.elan1.conv1.conv.weight.detach().clone()
    path = tmp_path / "quant.pt"
    model.save(path)

    # Respell the file the way pre-rename releases wrote it.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["model"] = {
        k.replace(".conv1.", ".cv1.", 1) if k.startswith("backbone.elan1.conv1.") else k: v
        for k, v in ckpt["model"].items()
    }
    ckpt["quant"]["keep_high_precision"] = ["head.", "backbone.conv0.", "backbone.elan1.cv1."]
    assert "backbone.elan1.cv1.conv.weight" in ckpt["model"]
    torch.save(ckpt, path)

    loaded = LibreYOLO(str(path), device="cpu")
    info = loaded.quant_info()
    assert sum(info["module_counts"].values()) == count
    layer = loaded.model.backbone.elan1.conv1.conv
    assert type(layer) is torch.nn.Conv2d
    assert torch.equal(layer.weight.detach().cpu(), float_weight)

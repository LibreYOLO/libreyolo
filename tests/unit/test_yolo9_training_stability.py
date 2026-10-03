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
# PGI auxiliary branch per size (AuxNeck for t/s, AuxBackbone for m/c)
# -----------------------------------------------------------------------------


def _is_aux_key(key):
    return key.startswith(("aux.", "aux_head."))


def _save_yolo9(tmp_path, size, *, branch=None, aux_class_neck=None, name="ckpt.pt"):
    """A training-style checkpoint with a PGI branch of the given kind."""
    from libreyolo.models.yolo9.nn import LibreYOLO9Model, YOLO9Head
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    raw = LibreYOLO9Model(config=size, nb_classes=2).enable_aux(0.25, branch=branch)
    if aux_class_neck is not None:
        raw.aux_head = YOLO9Head(
            raw.aux_head.in_channels,
            2,
            reg_max=16,
            strides=(8, 16, 32),
            class_neck=aux_class_neck,
        )
    with torch.no_grad():  # distinguishable from any fresh initialisation
        for key, param in raw.named_parameters():
            if _is_aux_key(key):
                param.add_(1.0)
    ckpt = wrap_libreyolo_checkpoint(
        raw.state_dict(),
        model_family="yolo9",
        size=size,
        task="detect",
        nc=2,
        names={0: "a", 1: "b"},
        imgsz=640,
    )
    path = tmp_path / name
    torch.save(ckpt, path)
    return path, ckpt["model"]


def _assert_aux_tensors_equal(model, saved):
    live = model.state_dict()
    aux_keys = [k for k in saved if _is_aux_key(k)]
    assert aux_keys and sorted(aux_keys) == sorted(k for k in live if _is_aux_key(k))
    for key in aux_keys:
        assert torch.equal(live[key], saved[key]), key


@pytest.mark.parametrize(
    "size,branch,head_channels",
    [
        ("t", "neck", (64, 96, 128)),
        ("s", "neck", (128, 192, 256)),
        ("m", "backbone", (240, 360, 480)),
        ("c", "backbone", (512, 512, 512)),
    ],
)
def test_enable_aux_builds_the_branch_of_the_size(size, branch, head_channels):
    from libreyolo.models.yolo9.nn import (
        AuxBackbone,
        AuxNeck,
        CBFuse,
        CBLinear,
        LibreYOLO9Model,
        supported_aux_branches,
    )

    model = LibreYOLO9Model(config=size, nb_classes=2)
    assert model.aux_branch is None
    model.enable_aux(0.25)
    assert supported_aux_branches(size)[0] == branch == model.aux_branch
    assert type(model.aux) is (AuxBackbone if branch == "backbone" else AuxNeck)
    assert model.aux_head.in_channels == head_channels
    has_cb = any(isinstance(m, (CBLinear, CBFuse)) for m in model.aux.modules())
    assert has_cb is (branch == "backbone")

    first, first_head = model.aux, model.aux_head
    model.enable_aux(0.5)  # idempotent: the attached branch is kept
    assert model.aux is first and model.aux_head is first_head
    assert model.aux_weight == 0.5
    model.disable_aux()
    assert model.aux is None and model.aux_head is None and model.aux_branch is None
    assert not any(_is_aux_key(k) for k in model.state_dict())


def test_aux_backbone_follows_the_upstream_layer_order_and_channels():
    """Module order, types and widths of the v9-m / v9-c ``auxiliary`` section."""
    from libreyolo.models.yolo9.nn import AuxBackbone

    expected_types = [
        ("cblinear3", "CBLinear"),
        ("cblinear4", "CBLinear"),
        ("cblinear5", "CBLinear"),
        ("conv0", "Conv"),
        ("conv1", "Conv"),
        ("elan1", "RepNCSPELAN"),
        ("down2", None),
        ("cbfuse3", "CBFuse"),
        ("elan2", "RepNCSPELAN"),
        ("down3", None),
        ("cbfuse4", "CBFuse"),
        ("elan3", "RepNCSPELAN"),
        ("down4", None),
        ("cbfuse5", "CBFuse"),
        ("elan4", "RepNCSPELAN"),
    ]
    expected = {
        # CBLinear splits, stem widths, first block, (down, stage) widths
        "m": ([[240], [240, 360], [240, 360, 480]], (32, 64, 128), "AConv",
              [(240, 240), (360, 360), (480, 480)]),
        "c": ([[256], [256, 512], [256, 512, 512]], (64, 128, 256), "ADown",
              [(256, 512), (512, 512), (512, 512)]),
    }
    for size, (splits, stem, down_type, stages) in expected.items():
        aux = AuxBackbone(size)
        children = list(aux.named_children())
        assert [name for name, _ in children] == [name for name, _ in expected_types]
        for (name, module), (_, type_name) in zip(children, expected_types):
            assert type(module).__name__ == (type_name or down_type), name
        assert [aux.cblinear3.out_channels, aux.cblinear4.out_channels,
                aux.cblinear5.out_channels] == splits
        assert [aux.cblinear3.conv.in_channels, aux.cblinear4.conv.in_channels,
                aux.cblinear5.conv.in_channels] == [out for _, out in stages]
        assert aux.cblinear3.conv.bias is not None
        assert [aux.cbfuse3.idx, aux.cbfuse4.idx, aux.cbfuse5.idx] == [[0, 0, 0], [1, 1], [2]]
        assert (aux.conv0.conv.out_channels, aux.conv1.conv.out_channels,
                aux.elan1.conv4.conv.out_channels) == stem
        for elan, (down_out, stage_out) in zip((aux.elan2, aux.elan3, aux.elan4), stages):
            assert elan.conv1.conv.in_channels == down_out
            assert elan.conv4.conv.out_channels == stage_out
        assert aux.out_channels == tuple(out for _, out in stages)


@pytest.mark.parametrize("size", ["t", "s"])
def test_aux_backbone_is_not_available_for_t_and_s(size):
    from libreyolo.models.yolo9.nn import AuxBackbone, LibreYOLO9Model

    with pytest.raises(ValueError):
        AuxBackbone(size)
    with pytest.raises(ValueError):
        LibreYOLO9Model(config=size, nb_classes=2).enable_aux(0.25, branch="backbone")


@pytest.mark.parametrize("size,aux_tensors", [("t", 432), ("s", 432)])
def test_ts_aux_state_dict_keys_are_stable(size, aux_tensors):
    """yolo9-t/s keep the ``AuxNeck`` key layout that 1.6.0 checkpoints use."""
    from libreyolo.models.yolo9.nn import AuxNeck, LibreYOLO9Model

    state = LibreYOLO9Model(config=size, nb_classes=2).enable_aux(0.25).state_dict()
    aux_keys = [k for k in state if _is_aux_key(k)]
    assert len(aux_keys) == aux_tensors
    assert {".".join(k.split(".")[:2]) for k in aux_keys} == {
        "aux.spp",
        "aux.elan_a4",
        "aux.elan_a3",
        "aux_head.anchor_convs",
        "aux_head.class_convs",
    }
    assert [k for k in aux_keys if k.startswith("aux.")] == [
        f"aux.{k}" for k in AuxNeck(size).state_dict()
    ]
    for key in (
        "aux.spp.conv1.conv.weight",
        "aux.spp.conv5.bn.running_var",
        "aux.elan_a4.conv1.conv.weight",
        "aux.elan_a4.conv2.0.bottleneck.2.conv1.conv2.bn.bias",
        "aux.elan_a3.conv3.1.conv.weight",
        "aux.elan_a3.conv4.bn.weight",
        "aux_head.anchor_convs.0.0.conv.weight",
        "aux_head.class_convs.2.2.bias",
    ):
        assert key in state, key


def test_aux_branch_kind_is_read_from_the_aux_keys():
    from libreyolo.models.yolo9.nn import aux_branch_from_state_dict

    assert aux_branch_from_state_dict({"aux.spp.conv1.conv.weight": 0}) == "neck"
    assert aux_branch_from_state_dict({"aux.elan_a3.conv1.conv.weight": 0}) == "neck"
    assert aux_branch_from_state_dict({"aux.cblinear3.conv.weight": 0}) == "backbone"
    assert aux_branch_from_state_dict({"aux_head.class_convs.0.2.weight": 0}) is None
    assert aux_branch_from_state_dict({"backbone.spp.conv1.conv.weight": 0}) is None


def test_legacy_mc_checkpoint_resumes_with_its_top_down_branch(tmp_path):
    """A 1.6.0 yolo9-m training checkpoint (``aux.spp`` / ``aux.elan_a*``)
    gets the branch it was trained with, and every aux tensor is loaded."""
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.models.yolo9.nn import AuxNeck

    path, saved = _save_yolo9(tmp_path, "m", branch="neck")
    aux_keys = [k for k in saved if _is_aux_key(k)]
    assert any(k.startswith("aux.spp.") for k in aux_keys)
    assert not any(k.startswith("aux.cblinear") for k in aux_keys)

    model = LibreYOLO9(str(path), size="m", nb_classes=2, device="cpu")
    assert model.model.aux is None
    loaded = model._maybe_enable_aux_from_path(str(path), 0.25)
    assert type(model.model.aux) is AuxNeck and model.model.aux_branch == "neck"
    assert loaded == len(aux_keys)
    _assert_aux_tensors_equal(model.model, saved)
    # The trainer's resume loads the saved model state strictly.
    model.model.load_state_dict(saved, strict=True)


@pytest.mark.parametrize("stored,expected", [("neck", "AuxNeck"), ("backbone", "AuxBackbone")])
def test_mc_fine_tune_keeps_the_checkpoint_branch(tmp_path, monkeypatch, stored, expected):
    """``train()`` from weights with PGI tensors builds the stored branch kind."""
    from libreyolo.models.yolo9.model import LibreYOLO9

    path, saved = _save_yolo9(tmp_path, "m", branch=stored)
    model = LibreYOLO9(str(path), size="m", nb_classes=2, device="cpu")
    assert _aux_attached_when_training(model, monkeypatch) is True
    assert type(model.model.aux).__name__ == expected
    _assert_aux_tensors_equal(model.model, saved)


def test_legacy_mc_branch_replaces_an_attached_new_branch(tmp_path):
    """Loading 1.6.0 PGI tensors into a model that already has the new branch
    rebuilds the branch instead of dropping the tensors."""
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.models.yolo9.nn import AuxBackbone, AuxNeck

    path, saved = _save_yolo9(tmp_path, "m", branch="neck")
    aux_keys = [k for k in saved if _is_aux_key(k)]

    model = LibreYOLO9(None, size="m", nb_classes=2, device="cpu")
    model.model.enable_aux(0.4)
    assert type(model.model.aux) is AuxBackbone
    assert model._reload_aux_from_path(str(path)) == len(aux_keys)
    assert type(model.model.aux) is AuxNeck and model.model.aux_weight == 0.4
    _assert_aux_tensors_equal(model.model, saved)

    transfer = LibreYOLO9(None, size="m", nb_classes=2, device="cpu")
    transfer.model.enable_aux(0.25)
    stats = transfer._load_transfer_weights(path)
    assert type(transfer.model.aux) is AuxNeck
    assert stats["aux_loaded"] == len(aux_keys) and stats["skipped"] == 0


def test_legacy_mc_training_checkpoint_loads_into_a_trained_wrapper(tmp_path):
    """The post-training reload path keeps the PGI tensors of either kind."""
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.models.yolo9.nn import AuxNeck

    path, saved = _save_yolo9(tmp_path, "m", branch="neck")
    model = LibreYOLO9(None, size="m", nb_classes=2, device="cpu")
    model.model.enable_aux(0.25)  # left attached by an earlier train()
    model._load_weights(str(path))
    assert type(model.model.aux) is AuxNeck
    _assert_aux_tensors_equal(model.model, saved)


def test_aux_backbone_head_takes_the_checkpoint_class_tower_width(tmp_path):
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.models.yolo9.nn import AuxBackbone

    path, saved = _save_yolo9(tmp_path, "m", aux_class_neck=256)
    aux_keys = [k for k in saved if _is_aux_key(k)]

    model = LibreYOLO9(str(path), size="m", nb_classes=2, device="cpu")
    model.model.enable_aux(0.25)
    assert model.model.aux_head.class_neck == 240
    assert model._reload_aux_from_path(str(path)) == len(aux_keys)
    assert type(model.model.aux) is AuxBackbone
    assert model.model.aux_head.class_neck == 256
    _assert_aux_tensors_equal(model.model, saved)


def test_rebuild_for_new_classes_covers_the_aux_backbone_head():
    from libreyolo.models.yolo9.model import LibreYOLO9

    wrapper = LibreYOLO9(None, size="c", nb_classes=80, device="cpu")
    wrapper.model.enable_aux(0.25)
    backbone_before = wrapper.model.aux.elan4.conv4.conv.weight.detach().clone()
    wrapper._rebuild_for_new_classes(5)
    aux_head = wrapper.model.aux_head
    assert wrapper.model.head.num_classes == aux_head.num_classes == 5
    assert all(t[2].out_channels == 5 for t in aux_head.class_convs)
    # The aux head reads the 512-wide A3, not the 256-wide main P3.
    assert [t[0].conv.in_channels for t in aux_head.class_convs] == [512, 512, 512]
    assert all(t[0].conv.out_channels == 512 for t in aux_head.class_convs)
    assert aux_head._loss_fn is None
    assert torch.equal(wrapper.model.aux.elan4.conv4.conv.weight, backbone_before)

    wrapper.model.train()
    targets = torch.zeros(1, 3, 5)
    targets[:, 0] = torch.tensor([4.0, 0.2, 0.2, 0.6, 0.6])
    out = wrapper.model(torch.rand(1, 3, 64, 96), targets=targets)
    assert torch.isfinite(out["total_loss"])


@pytest.mark.parametrize("size", ["t", "m"])
def test_pgi_training_forward_then_deepcopy(size):
    """The EMA deep-copies the model after training forwards; the PGI branch
    must leave no graph tensors on the modules, and must not change the main
    path."""
    import copy
    from types import SimpleNamespace

    from libreyolo.models.yolo9.nn import LibreYOLO9Model
    from libreyolo.models.yolo9.trainer import YOLO9Trainer

    torch.manual_seed(0)
    model = LibreYOLO9Model(config=size, nb_classes=3)
    x = torch.rand(2, 3, 64, 96)
    targets = torch.zeros(2, 3, 5)
    targets[:, 0] = torch.tensor([1.0, 0.2, 0.2, 0.6, 0.6])

    model.eval()
    with torch.no_grad():
        plain = model(x)["predictions"]
    model.enable_aux(0.25)
    with torch.no_grad():
        assert torch.equal(model(x)["predictions"], plain)  # eval ignores PGI

    model.train()
    reference = copy.deepcopy(model)
    main_only = copy.deepcopy(model)
    main_only.aux_weight = 0.0
    main_loss = main_only(x, targets=targets)

    loss = model(x, targets=targets)
    assert torch.isfinite(loss["total_loss"])
    assert loss["total_loss"] > main_loss["total_loss"]  # aux term added
    loss["total_loss"].backward()
    grads = [p.grad is not None for n, p in model.named_parameters() if _is_aux_key(n)]
    assert grads and all(grads)

    clone = copy.deepcopy(model)  # what ModelEMA does
    assert type(clone.aux) is type(model.aux)
    for key, value in model.state_dict().items():
        assert torch.equal(clone.state_dict()[key], value), key

    # The capture / compile boundary reproduces the model's own loss.
    host = SimpleNamespace(model=reference, wrapper_model=SimpleNamespace(task="detect"))
    spec = YOLO9Trainer.cuda_graph_train_spec(host)
    assert type(spec.network.module).__name__ == "_PGITrainForward"
    expected = copy.deepcopy(reference)(x, targets=targets)
    got = spec.assemble(spec.network(x), x, targets)
    for key in expected:
        assert torch.equal(expected[key], got[key]), key


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

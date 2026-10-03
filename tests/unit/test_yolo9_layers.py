"""YOLO9 building blocks, detection head contract and legacy checkpoint keys."""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

from libreyolo.models.yolo9.convert import upgrade_legacy_key, upgrade_legacy_state_dict
from libreyolo.models.yolo9.nn import (
    ADown,
    AConv,
    Anchor2Vec,
    Bottleneck,
    Conv,
    ELAN,
    LibreYOLO9Model,
    Pool,
    RepConv,
    RepNCSP,
    RepNCSPELAN,
    SPPELAN,
    YOLO9Head,
    auto_pad,
    create_activation_function,
    default_class_neck,
    round_up,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Helpers and blocks
# =============================================================================


def test_auto_pad_keeps_spatial_size():
    assert auto_pad(3) == (1, 1)
    assert auto_pad(1) == (0, 0)
    assert auto_pad((3, 5)) == (1, 2)
    assert auto_pad(3, dilation=2) == (2, 2)
    assert auto_pad(2) == (0, 0)


def test_create_activation_function():
    act = create_activation_function("SiLU")
    assert isinstance(act, nn.SiLU) and act.inplace
    assert isinstance(create_activation_function("silu"), nn.SiLU)
    for off in (None, False, "false", "None"):
        assert isinstance(create_activation_function(off), nn.Identity)
    assert isinstance(create_activation_function(True), nn.SiLU)
    assert isinstance(create_activation_function("GELU"), nn.GELU)
    with pytest.raises(ValueError, match="not found"):
        create_activation_function("NoSuchActivation")


def test_round_up():
    assert round_up(15, 4) == 16
    assert round_up(16, 4) == 16
    assert round_up(7) == 7


def test_conv_is_bias_free_with_mtl_batchnorm():
    conv = Conv(3, 8, 3, stride=2)
    assert conv.conv.bias is None
    assert conv.conv.padding == (1, 1)
    assert conv.bn.eps == pytest.approx(1e-3)
    assert conv.bn.momentum == pytest.approx(3e-2)
    assert isinstance(conv.act, nn.SiLU)
    assert conv(torch.zeros(1, 3, 32, 32)).shape == (1, 8, 16, 16)
    assert isinstance(Conv(3, 8, 1, activation=False).act, nn.Identity)


def test_pool_pads_automatically():
    assert Pool("max", 5, stride=1)(torch.zeros(1, 2, 9, 9)).shape == (1, 2, 9, 9)
    assert Pool("avg", kernel_size=2, stride=1)(torch.zeros(1, 2, 9, 9)).shape == (1, 2, 8, 8)


def test_repconv_sums_both_branches_before_activation():
    torch.manual_seed(0)
    block = RepConv(4, 6).eval()
    x = torch.randn(1, 4, 8, 8)
    expected = nn.functional.silu(block.conv1(x) + block.conv2(x))
    assert torch.equal(block(x), expected)
    assert block.conv1.conv.kernel_size == (3, 3)
    assert block.conv2.conv.kernel_size == (1, 1)
    assert isinstance(block.conv1.act, nn.Identity)


def test_bottleneck_residual_only_when_widths_match():
    torch.manual_seed(0)
    block = Bottleneck(4, 4).eval()
    x = torch.randn(1, 4, 8, 8)
    assert block.residual
    assert torch.equal(block(x), x + block.conv2(block.conv1(x)))
    assert not Bottleneck(4, 6).residual


def test_repncsp_shapes_and_repeats():
    block = RepNCSP(16, 32, repeat_num=3)
    assert len(block.bottleneck) == 3
    assert block.conv1.conv.out_channels == 16  # csp_expand 0.5
    assert block(torch.zeros(1, 16, 8, 8)).shape == (1, 32, 8, 8)


def test_elan_shapes():
    block = ELAN(32, 48, 32)
    assert block.conv2.conv.in_channels == 16
    assert block.conv2.conv.out_channels == 16  # process = part // 2
    assert block.conv4.conv.in_channels == 32 + 2 * 16
    assert block(torch.zeros(1, 32, 16, 16)).shape == (1, 48, 16, 16)


def test_repncspelan_shapes_and_repeats():
    block = RepNCSPELAN(64, 96, 64, csp_args={"repeat_num": 3})
    assert isinstance(block.conv2[0], RepNCSP) and isinstance(block.conv2[1], Conv)
    assert len(block.conv2[0].bottleneck) == 3
    assert len(block.conv3[0].bottleneck) == 3
    assert block.conv4.conv.in_channels == 64 + 2 * 32
    assert block(torch.zeros(1, 64, 8, 8)).shape == (1, 96, 8, 8)
    assert len(RepNCSPELAN(64, 96, 64).conv2[0].bottleneck) == 1


def test_downsampling_blocks_halve_resolution():
    assert AConv(16, 32)(torch.zeros(1, 16, 16, 16)).shape == (1, 32, 8, 8)
    down = ADown(16, 32)
    assert down.conv1.conv.in_channels == 8 and down.conv2.conv.out_channels == 16
    assert down(torch.zeros(1, 16, 16, 16)).shape == (1, 32, 8, 8)


def test_sppelan_shapes():
    block = SPPELAN(32, 64)
    assert block.conv1.conv.out_channels == 32  # neck = out // 2
    assert len(block.pools) == 3
    assert block(torch.zeros(1, 32, 8, 8)).shape == (1, 64, 8, 8)
    assert SPPELAN(32, 64, 16).conv5.conv.in_channels == 64


# =============================================================================
# Detection head
# =============================================================================

_CH = (16, 24, 32)
_HW = ((8, 8), (4, 4), (2, 2))  # a 64x64 input at strides 8/16/32


def _features(batch=2, sizes=_HW, channels=_CH):
    torch.manual_seed(0)
    return [torch.randn(batch, c, h, w) for c, (h, w) in zip(channels, sizes)]


def test_head_widths_follow_mtl_formulas():
    head = YOLO9Head(_CH, 3)
    assert head.anchor_neck == max(round_up(16 // 4, 4), 64, 16) == 64
    assert head.class_neck == max(16, min(3 * 2, 128)) == 16
    assert YOLO9Head((64, 96, 128), 80).class_neck == 128
    assert YOLO9Head((64, 96, 128), 80, class_neck=80).class_neck == 80
    assert YOLO9Head(_CH, 3, use_group=False).groups == 1


def test_default_class_neck_matches_libreyolo_widths():
    assert default_class_neck(64, 80) == 80
    assert default_class_neck(64, 40) == 64
    assert default_class_neck(64, 1) == 64
    assert default_class_neck(32, 40) == 40
    assert default_class_neck(256, 2) == 256
    assert default_class_neck(64, 1000) == 128


def test_head_towers_and_bias_init():
    head = YOLO9Head(_CH, 3)
    assert len(head.anchor_convs) == len(head.class_convs) == 3
    for tower, c in zip(head.anchor_convs, _CH):
        assert tower[0].conv.in_channels == c
        assert tower[1].conv.groups == 4
        assert tower[2].out_channels == 64 and tower[2].groups == 4
        assert torch.all(tower[2].bias == 1.0)
    for tower in head.class_convs:
        assert tower[2].out_channels == 3
        assert torch.all(tower[2].bias == -10.0)


def test_head_registers_all_box_towers_before_class_towers():
    names = [name for name, _ in YOLO9Head(_CH, 3).named_parameters()]
    last_anchor = max(i for i, n in enumerate(names) if n.startswith("anchor_convs."))
    first_class = min(i for i, n in enumerate(names) if n.startswith("class_convs."))
    assert last_anchor < first_class
    assert all(n.startswith(("anchor_convs.", "class_convs.")) for n in names)


def test_anchor2vec_is_stateless_softmax_expectation():
    a2v = Anchor2Vec(reg_max=16)
    assert a2v.state_dict() == {}
    logits = torch.randn(2, 64, 5, 7)
    expected = (
        logits.view(2, 4, 16, 5, 7).softmax(2)
        * torch.arange(16.0).view(1, 1, 16, 1, 1)
    ).sum(2)
    assert torch.allclose(a2v(logits), expected, atol=1e-6)
    assert a2v(logits.flatten(2)).shape == (2, 4, 35)


def test_head_train_mode_returns_raw_maps():
    head = YOLO9Head(_CH, 3).train()
    raw = head(_features())
    assert [tuple(t.shape) for t in raw] == [(2, 67, 8, 8), (2, 67, 4, 4), (2, 67, 2, 2)]


def test_head_eval_mode_returns_decoded_and_raw():
    head = YOLO9Head(_CH, 3).eval()
    with torch.no_grad():
        decoded, raw = head(_features())
    assert decoded.shape == (2, 4 + 3, 64 + 16 + 4)
    assert len(raw) == 3
    assert torch.equal(decoded, head.decode(raw))


def test_decode_places_boxes_around_cell_centres():
    head = YOLO9Head(_CH, 2).eval()
    raw = [torch.zeros(1, 64 + 2, h, w) for h, w in _HW]
    # Peaked bins: left/top = 1, right/bottom = 2 (grid units) everywhere.
    for level in raw:
        level[:, 0 * 16 + 1] = 60.0
        level[:, 1 * 16 + 1] = 60.0
        level[:, 2 * 16 + 2] = 60.0
        level[:, 3 * 16 + 2] = 60.0
        level[:, 64] = 0.0
        level[:, 65] = 3.0
    out = head.decode(raw)
    # First P3 cell: centre (4, 4), stride 8.
    assert torch.allclose(out[0, :4, 0], torch.tensor([4.0 - 8, 4.0 - 8, 4.0 + 16, 4.0 + 16]))
    # Last P5 cell of the 64x64 grid: centre (48, 48), stride 32.
    assert torch.allclose(out[0, :4, -1], torch.tensor([48.0 - 32, 48.0 - 32, 48.0 + 64, 48.0 + 64]))
    assert torch.allclose(out[0, 4], torch.full((84,), 0.5))
    assert torch.allclose(out[0, 5], torch.sigmoid(torch.tensor(3.0)).expand(84))


def test_decode_follows_rectangular_level_sizes():
    head = YOLO9Head(_CH, 1).eval()
    sizes = ((6, 10), (3, 5), (2, 3))  # 48x80 input (P5 rounded up)
    with torch.no_grad():
        decoded, _ = head(_features(batch=1, sizes=sizes))
    assert decoded.shape == (1, 5, 60 + 15 + 6)


def test_anchor_grid_cache_and_export_mode():
    head = YOLO9Head(_CH, 1).eval()
    feats = _features(batch=1)
    with torch.no_grad():
        first = head(feats)[0]
        cache = head._grid_cache
        head(feats)
        assert head._grid_cache is cache  # eager reuse
        head.export = True
        exported = head(feats)[0]
        assert head._grid_cache is cache  # export mode does not touch the cache
    assert torch.equal(first, exported)


def test_freeze_anchor_grid_pins_the_export_canvas():
    head = YOLO9Head(_CH, 1).eval()
    head.export = True
    feats = _features(batch=1)
    with torch.no_grad():
        live = head(feats)[0]
        head.freeze_anchor_grid((64, 64))
        frozen = head(feats)[0]
        assert torch.equal(live, frozen)
        with pytest.raises(ValueError, match="frozen"):
            head(_features(batch=1, sizes=((4, 4), (2, 2), (1, 1))))
        head.unfreeze_anchor_grid()
        head(_features(batch=1, sizes=((4, 4), (2, 2), (1, 1))))


def test_frozen_grid_traces_as_constants():
    head = YOLO9Head(_CH, 1).eval()
    head.export = True
    head.freeze_anchor_grid((64, 64))
    feats = _features(batch=1)

    class _Decode(nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, a, b, c):
            return self.head([a, b, c])[0]

    traced = torch.jit.trace(_Decode(head), tuple(feats))
    graph = str(traced.inlined_graph)
    assert "aten::arange" not in graph
    assert torch.equal(traced(*feats), head(feats)[0])


def test_set_num_classes_swaps_only_final_class_convs():
    head = YOLO9Head(_CH, 3, class_neck=40)
    hidden = [tower[0].conv.weight for tower in head.class_convs]
    anchors = [p.clone() for p in head.anchor_convs.parameters()]
    head._loss_fn = object()
    head.set_num_classes(7)
    assert head.num_classes == 7 and head._loss_fn is None
    for tower, weight in zip(head.class_convs, hidden):
        assert tower[0].conv.weight is weight
        assert tower[2].in_channels == 40 and tower[2].out_channels == 7
        assert torch.all(tower[2].bias == -10.0)
    assert all(torch.equal(a, b) for a, b in zip(anchors, head.anchor_convs.parameters()))


def test_build_helpers_and_init_bias():
    head = YOLO9Head(_CH, 3)
    towers = head.build_class_convs(_CH, 24, 5)
    assert [t[0].conv.out_channels for t in towers] == [24, 24, 24]
    assert [t[2].out_channels for t in towers] == [5, 5, 5]
    assert len(head.build_anchor_convs(_CH)) == 3
    with torch.no_grad():
        for tower in (*head.anchor_convs, *head.class_convs):
            tower[2].bias.zero_()
    head.init_bias()
    assert all(torch.all(t[2].bias == 1.0) for t in head.anchor_convs)
    assert all(torch.all(t[2].bias == -10.0) for t in head.class_convs)


def test_head_targets_path_returns_losses_and_caches_loss_fn():
    head = YOLO9Head(_CH, 3).train()
    targets = torch.tensor([[[1.0, 0.2, 0.2, 0.6, 0.7]], [[0.0, 0.1, 0.5, 0.4, 0.9]]])
    losses = head(_features(), targets=targets, img_size=(64, 64))
    assert {"total_loss", "box_loss", "dfl_loss", "cls_loss"} <= set(losses)
    assert torch.isfinite(losses["total_loss"])
    loss_fn = head._get_loss_fn("cpu")
    assert head._get_loss_fn(torch.device("cpu")) is loss_fn
    assert loss_fn.strides == [8, 16, 32] and loss_fn.num_classes == 3
    with pytest.raises(ValueError, match="img_size"):
        head(_features(), targets=targets)


def test_head_rejects_level_count_mismatch():
    with pytest.raises(ValueError):
        YOLO9Head(_CH, 3, strides=(8, 16))
    with pytest.raises(ValueError):
        YOLO9Head(_CH, 3).train()(_features()[:2])


def test_model_export_mode_returns_predictions_only():
    model = LibreYOLO9Model(config="t", nb_classes=3).eval()
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
        assert set(out) == {"predictions", "raw_outputs", "x8", "x16", "x32"}
        model.head.export = True
        exported = model(x)
    assert torch.equal(exported, out["predictions"])
    assert exported.shape == (1, 4 + 3, 84)


def test_model_uses_libreyolo_class_width_and_aux_head():
    model = LibreYOLO9Model(config="t", nb_classes=80)
    assert model.head.class_neck == 80
    model.enable_aux(0.25)
    assert isinstance(model.aux_head, YOLO9Head)
    assert model.aux_head.class_neck == 80
    assert model.aux_head.strides == (8, 16, 32)


# =============================================================================
# Legacy checkpoint keys
# =============================================================================

_LEGACY_TO_CURRENT = [
    ("detect.cv2.0.0.conv.weight", "head.anchor_convs.0.0.conv.weight"),
    ("head.cv2.2.2.bias", "head.anchor_convs.2.2.bias"),
    ("head.cv3.1.1.bn.running_var", "head.class_convs.1.1.bn.running_var"),
    ("aux_head.cv2.0.1.bn.weight", "aux_head.anchor_convs.0.1.bn.weight"),
    ("aux_head.cv3.2.2.weight", "aux_head.class_convs.2.2.weight"),
    ("head.one2one_cv2.1.2.weight", "head.one_to_one_anchor_convs.1.2.weight"),
    ("head.one2one_cv3.0.0.conv.weight", "head.one_to_one_class_convs.0.0.conv.weight"),
    ("backbone.elan1.cv1.conv.weight", "backbone.elan1.conv1.conv.weight"),
    ("backbone.elan1.cv4.bn.bias", "backbone.elan1.conv4.bn.bias"),
    (
        "backbone.elan2.cv2.0.m.1.cv1.conv1.bn.running_mean",
        "backbone.elan2.conv2.0.bottleneck.1.conv1.conv1.bn.running_mean",
    ),
    ("backbone.elan2.cv3.0.cv3.conv.weight", "backbone.elan2.conv3.0.conv3.conv.weight"),
    ("backbone.down2.cv.conv.weight", "backbone.down2.conv.conv.weight"),
    ("neck.down1.cv1.bn.num_batches_tracked", "neck.down1.conv1.bn.num_batches_tracked"),
    ("backbone.spp.cv5.conv.weight", "backbone.spp.conv5.conv.weight"),
    ("neck.elan_up1.cv2.0.m.0.cv2.conv.weight", "neck.elan_up1.conv2.0.bottleneck.0.conv2.conv.weight"),
    ("aux.spp.cv1.conv.weight", "aux.spp.conv1.conv.weight"),
    ("aux.elan_a3.cv2.1.bn.weight", "aux.elan_a3.conv2.1.bn.weight"),
]

_LEGACY_DROPPED = [
    "head.dfl.conv.weight",
    "aux_head.dfl.conv.weight",
    "detect.dfl.conv.weight",
    "head.stride",
    "head.anchors",
    "head.strides",
]


@pytest.mark.parametrize("legacy,current", _LEGACY_TO_CURRENT)
def test_upgrade_legacy_key_rules(legacy, current):
    assert upgrade_legacy_key(legacy) == current
    assert upgrade_legacy_key(current) == current  # idempotent


@pytest.mark.parametrize("legacy", _LEGACY_DROPPED)
def test_upgrade_legacy_key_drops_derived_head_state(legacy):
    assert upgrade_legacy_key(legacy) is None


def test_current_and_foreign_keys_pass_through():
    model = LibreYOLO9Model(config="c", nb_classes=2).enable_aux(0.25)
    for key in model.state_dict():
        assert upgrade_legacy_key(key) == key
    # Rules only apply to yolo9 module prefixes.
    assert upgrade_legacy_key("model.cv1.m.0.weight") == "model.cv1.m.0.weight"
    assert upgrade_legacy_key("head.cv4.0.2.weight") == "head.cv4.0.2.weight"


def test_upgrade_legacy_state_dict_is_order_preserving_and_idempotent():
    legacy = {"head.dfl.conv.weight": torch.zeros(1)}
    legacy.update({old: torch.full((1,), float(i)) for i, (old, _) in enumerate(_LEGACY_TO_CURRENT)})
    upgraded = upgrade_legacy_state_dict(legacy)
    assert list(upgraded) == [new for _, new in _LEGACY_TO_CURRENT]
    assert all(upgraded[new] is legacy[old] for old, new in _LEGACY_TO_CURRENT)
    assert upgrade_legacy_state_dict(upgraded) == upgraded


# Pre-rename attribute names per module type (the legacy layout).
_LEGACY_CHILD_NAMES = {
    Bottleneck: {"conv1": "cv1", "conv2": "cv2"},
    RepNCSP: {"conv1": "cv1", "conv2": "cv2", "conv3": "cv3", "bottleneck": "m"},
    ELAN: {"conv1": "cv1", "conv2": "cv2", "conv3": "cv3", "conv4": "cv4"},
    RepNCSPELAN: {"conv1": "cv1", "conv2": "cv2", "conv3": "cv3", "conv4": "cv4"},
    AConv: {"conv": "cv"},
    ADown: {"conv1": "cv1", "conv2": "cv2"},
    SPPELAN: {"conv1": "cv1", "conv5": "cv5"},
    YOLO9Head: {"anchor_convs": "cv2", "class_convs": "cv3"},
}


def _legacy_state_dict(model: nn.Module) -> dict:
    """The model's state dict spelled with the pre-rename key layout."""
    legacy = {}
    for key, value in model.state_dict().items():
        module, parts = model, []
        for part in key.split("."):
            parts.append(_LEGACY_CHILD_NAMES.get(type(module), {}).get(part, part))
            module = module._modules.get(part) if isinstance(module, nn.Module) else None
        legacy[".".join(parts)] = value
    return legacy


@pytest.mark.parametrize("size", ["t", "c"])
def test_legacy_state_dict_loads_strictly_after_upgrade(size):
    torch.manual_seed(0)
    source = LibreYOLO9Model(config=size, nb_classes=3).enable_aux(0.25)
    legacy = _legacy_state_dict(source)
    assert any(re.search(r"\.cv\d\.", k) for k in legacy)
    assert any(".m." in k for k in legacy) and any(k.startswith("head.cv3.") for k in legacy)
    legacy["head.dfl.conv.weight"] = torch.arange(16.0).view(1, 16, 1, 1)
    legacy = {("detect." + k[5:] if k.startswith("head.") else k): v for k, v in legacy.items()}

    upgraded = upgrade_legacy_state_dict(legacy)
    assert list(upgraded) == list(source.state_dict())

    torch.manual_seed(1)
    target = LibreYOLO9Model(config=size, nb_classes=3).enable_aux(0.25)
    target.load_state_dict(upgraded, strict=True)
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_yolo9_trainer_upgrades_every_saved_model_state():
    from libreyolo.models.yolo9.trainer import YOLO9Trainer
    from libreyolo.training.trainer import BaseTrainer

    legacy = {"backbone.elan1.cv1.conv.weight": torch.zeros(1), "head.dfl.conv.weight": torch.zeros(1)}
    checkpoint = {"model": dict(legacy), "train_model": dict(legacy), "ema": dict(legacy), "epoch": 3}
    trainer = object.__new__(YOLO9Trainer)
    upgraded = trainer.upgrade_resume_checkpoint(checkpoint)
    for key in ("model", "train_model", "ema"):
        assert list(upgraded[key]) == ["backbone.elan1.conv1.conv.weight"]
    assert upgraded["epoch"] == 3
    assert BaseTrainer.upgrade_resume_checkpoint(trainer, {"model": dict(legacy)}) == {
        "model": legacy
    }


# =============================================================================
# Loading through the wrappers
# =============================================================================


@pytest.mark.parametrize("size", ["t", "s", "m", "c"])
def test_registry_classmethods_accept_both_key_spellings(size):
    from libreyolo.models.yolo9.model import LibreYOLO9
    from libreyolo.models.yolo9_p2.model import LibreYOLO9P2

    model = LibreYOLO9Model(config=size, nb_classes=3)
    for state in (model.state_dict(), _legacy_state_dict(model)):
        assert LibreYOLO9.can_load(state) is True
        assert LibreYOLO9P2.can_load(state) is False
        assert LibreYOLO9.detect_size(state) == size
        assert LibreYOLO9.detect_nb_classes(state) == 3


@pytest.mark.parametrize(
    "extra_key",
    [
        "head.one2one_cv2.0.0.conv.weight",
        "head.one2one_cv3.0.2.weight",
        "head.one_to_one_anchor_convs.0.0.conv.weight",
        "head.one_to_one_class_convs.0.2.weight",
        "neck.elan_up3.cv1.conv.weight",
        "neck.elan_up3.conv1.conv.weight",
        "neck.elan_down0.conv1.conv.weight",
    ],
)
def test_base_can_load_leaves_e2e_and_p2_keys_to_their_families(extra_key):
    from libreyolo.models.yolo9.model import LibreYOLO9

    state = dict(LibreYOLO9Model(config="t", nb_classes=3).state_dict())
    state[extra_key] = torch.zeros(1)
    assert LibreYOLO9.can_load(state) is False


def _save_legacy_checkpoint(path, model, nc):
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    legacy = _legacy_state_dict(model)
    legacy["head.dfl.conv.weight"] = torch.arange(16.0).view(1, 16, 1, 1)
    torch.save(
        wrap_libreyolo_checkpoint(
            legacy,
            model_family="yolo9",
            size="t",
            task="detect",
            nc=nc,
            names={i: f"c{i}" for i in range(nc)},
            imgsz=640,
        ),
        path,
    )


def test_wrapper_loads_a_legacy_layout_checkpoint_file(tmp_path):
    from libreyolo.models.yolo9.model import LibreYOLO9

    torch.manual_seed(0)
    source = LibreYOLO9Model(config="t", nb_classes=3).eval()
    path = tmp_path / "legacy_t.pt"
    _save_legacy_checkpoint(path, source, nc=3)

    loaded = LibreYOLO9(str(path), size="t", device="cpu")
    assert loaded.nb_classes == 3
    state = loaded.model.state_dict()
    assert list(state) == list(source.state_dict())
    assert all(torch.equal(state[k], v) for k, v in source.state_dict().items())
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(loaded.model(x)["predictions"], source(x)["predictions"])


def test_wrapper_keeps_legacy_checkpoint_class_width(tmp_path):
    """A fine-tune keeps its source's COCO-width towers (t: 80 wide); the
    checkpoint width wins over a fresh build for the new class count."""
    from libreyolo.models.yolo9.model import LibreYOLO9

    source = LibreYOLO9Model(config="t", nb_classes=80)
    source.head.set_num_classes(2)
    path = tmp_path / "legacy_t_2cls.pt"
    _save_legacy_checkpoint(path, source, nc=2)

    loaded = LibreYOLO9(str(path), size="t", device="cpu")
    head = loaded.model.head
    assert loaded.nb_classes == 2 and head.num_classes == 2
    assert head.class_neck == 80
    assert head.class_convs[0][0].conv.out_channels == 80
    assert head.class_convs[0][2].out_channels == 2


def test_rebuild_for_new_classes_keeps_widths_and_rebuilds_aux():
    """A class-count change swaps the final class convs, keeps hidden widths,
    and (as LibreYOLO always has) re-applies the bias init to both towers."""
    from libreyolo.models.yolo9.model import LibreYOLO9

    wrapper = LibreYOLO9(None, size="t", nb_classes=80, device="cpu")
    wrapper.model.enable_aux(0.25)
    with torch.no_grad():
        wrapper.model.head.anchor_convs[0][2].bias.fill_(0.5)
    wrapper._rebuild_for_new_classes(4)
    for head in (wrapper.model.head, wrapper.model.aux_head):
        assert head.num_classes == 4 and head._loss_fn is None
        assert all(t[2].out_channels == 4 for t in head.class_convs)
        assert all(t[0].conv.out_channels == 80 for t in head.class_convs)
        assert all(torch.all(t[2].bias == -10.0) for t in head.class_convs)
        assert all(torch.all(t[2].bias == 1.0) for t in head.anchor_convs)

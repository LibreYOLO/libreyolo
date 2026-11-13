"""YOLO9 raw-head export contract: graph cut, output windows and host decode parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.export.exporter import _YOLO9RawHeadsWrapper
from libreyolo.postprocess.yolonas_raw import (
    decode_raw_heads,
    raw_head_output_names,
    unwindow_raw_heads,
)

pytestmark = pytest.mark.unit

_LAYOUT = "yolo9_raw_heads_v1"


def _build(size: str = "t", nb_classes: int = 5, seed: int = 0):
    from libreyolo import LibreYOLO9

    torch.manual_seed(seed)
    model = LibreYOLO9(None, size=size, nb_classes=nb_classes, device="cpu")
    net = model.model.eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        # Non-trivial BatchNorm statistics, spread-out bin logits and class
        # logits around zero so boxes and scores are both exercised.
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                n = module.num_features
                module.running_mean.copy_(torch.randn(n, generator=generator) * 0.1)
                module.running_var.copy_(torch.rand(n, generator=generator) + 0.5)
        for tower in (*net.head.anchor_convs, *net.head.class_convs):
            conv = tower[-1]
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.05)
            conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator))
    return model, net


def test_raw_wrapper_output_contract():
    _, net = _build(nb_classes=5)
    wrapper = _YOLO9RawHeadsWrapper(net, output_windows={}).eval()
    with torch.no_grad():
        outputs = wrapper(torch.rand(1, 3, 64, 96))

    assert wrapper.output_layout == _LAYOUT
    assert wrapper.strides == (8, 16, 32) and wrapper.reg_max == 16
    assert wrapper.output_names == raw_head_output_names(_LAYOUT)
    assert wrapper.output_names[:2] == ["p3_reg", "p3_cls"]
    for name, output in zip(wrapper.output_names, outputs):
        level, part = name.split("_")
        stride = 2 ** int(level[1:])
        channels = {"reg": 4 * 16, "cls": 5}[part]
        assert tuple(output.shape) == (1, channels, 64 // stride, 96 // stride)


def test_raw_wrapper_rejects_other_tasks():
    _, net = _build()
    with pytest.raises(ValueError, match="detect only"):
        _YOLO9RawHeadsWrapper(net, "segment")


@pytest.mark.parametrize(
    ("size", "hw"), [("t", (160, 160)), ("t", (192, 320)), ("s", (128, 128))]
)
@pytest.mark.parametrize("windowed", [False, True])
def test_host_decode_matches_in_graph_decode(size, hw, windowed):
    _, net = _build(size)
    windows = {"cls": (-8.0, 8.0), "reg": (-16.0, 16.0)} if windowed else {}
    wrapper = _YOLO9RawHeadsWrapper(net, output_windows=windows).eval()
    sample = torch.rand(2, 3, *hw, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        expected = net(sample)["predictions"].numpy()  # (B, 4 + nc, N)
        raw = [t.numpy() for t in wrapper(sample)]

    (decoded,) = decode_raw_heads(
        _LAYOUT, raw, strides=wrapper.strides, reg_max=wrapper.reg_max,
        output_windows=wrapper.output_windows,
    )
    assert decoded.shape == expected.shape and decoded.dtype == np.float32
    if windowed:
        # The windows clip only logits beyond +-8 (class) / +-16 (bins); the
        # random head keeps every logit inside them.
        restored = unwindow_raw_heads(_LAYOUT, raw, windows)
        assert max(float(np.abs(r).max()) for r in restored[1::2]) < 8.0
    # Boxes in pixels (float32 DFL sums, ~1e-5 px apart), scores exact-ish.
    np.testing.assert_allclose(decoded[:, :4], expected[:, :4], rtol=1e-6, atol=5e-4)
    np.testing.assert_allclose(decoded[:, 4:], expected[:, 4:], atol=1e-5)


def test_class_activation_bound_is_exact_inside_and_caps_outside():
    from libreyolo.export.exporter import _bounded_conv_bn_act

    _, net = _build("t")
    tower = net.head.class_convs[1]
    sample = torch.rand(1, 3, 128, 160, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        feats = net.neck(*net.backbone(sample))
        hidden = tower[0](feats[1])
        plain = tower[1](hidden)
        # BatchNorm and 1 / bound fold into the conv: exact while the bound is not reached.
        np.testing.assert_allclose(
            _bounded_conv_bn_act(tower[1], hidden, 1e4).numpy(), plain.numpy(), rtol=1e-5, atol=1e-5
        )
        pre = tower[1].bn(tower[1].conv(hidden))  # pre-activation
        bound = float(pre.max()) / 2
        capped = _bounded_conv_bn_act(tower[1], hidden, bound)
    silu_cap = bound / (1 + np.exp(-bound))
    assert float(capped.max()) <= silu_cap + 1e-4 < float(plain.max())
    inside = pre.abs() < bound * 0.99
    np.testing.assert_allclose(capped[inside].numpy(), plain[inside].numpy(), rtol=1e-5, atol=1e-5)


def test_class_activation_bound_default_and_validation():
    _, net = _build()
    assert _YOLO9RawHeadsWrapper(net).class_activation_bound == 32.0
    assert _YOLO9RawHeadsWrapper(net, class_activation_bound=None).class_activation_bound is None
    with pytest.raises(ValueError, match="positive"):
        _YOLO9RawHeadsWrapper(net, class_activation_bound=0.0)


@pytest.mark.parametrize("hw", [(160, 160), (192, 320)])
def test_e2e_cut_decodes_the_one_to_one_branch(hw):
    from libreyolo import LibreYOLO9E2E

    torch.manual_seed(0)
    net = LibreYOLO9E2E(None, size="t", nb_classes=5, device="cpu").model.eval()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                n = module.num_features
                module.running_mean.copy_(torch.randn(n, generator=generator) * 0.1)
                module.running_var.copy_(torch.rand(n, generator=generator) + 0.5)
        # Different last convs in the two branches, so a cut at the wrong
        # (dense) towers cannot match the one-to-one predictions.
        head = net.head
        for scale, towers in (
            (0.05, (*head.anchor_convs, *head.class_convs)),
            (0.08, (*head.one_to_one_anchor_convs, *head.one_to_one_class_convs)),
        ):
            for tower in towers:
                conv = tower[-1]
                conv.weight.copy_(
                    torch.randn(conv.weight.shape, generator=generator) * scale
                )
                conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator))
    wrapper = _YOLO9RawHeadsWrapper(
        net,
        output_windows={},
        towers=("one_to_one_anchor_convs", "one_to_one_class_convs"),
    ).eval()
    dense = _YOLO9RawHeadsWrapper(net, output_windows={}).eval()
    sample = torch.rand(1, 3, *hw, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        output = net(sample)
        expected = (
            output["predictions"] if isinstance(output, dict) else output[0]
        ).numpy()
        raw = [t.numpy() for t in wrapper(sample)]
        raw_dense = [t.numpy() for t in dense(sample)]

    (decoded,) = decode_raw_heads(
        _LAYOUT,
        raw,
        strides=wrapper.strides,
        reg_max=wrapper.reg_max,
        output_windows={},
    )
    (decoded_dense,) = decode_raw_heads(
        _LAYOUT,
        raw_dense,
        strides=dense.strides,
        reg_max=dense.reg_max,
        output_windows={},
    )
    np.testing.assert_allclose(decoded[:, :4], expected[:, :4], rtol=1e-6, atol=5e-4)
    np.testing.assert_allclose(decoded[:, 4:], expected[:, 4:], atol=1e-5)
    assert not np.allclose(decoded_dense[:, 4:], expected[:, 4:], atol=1e-3)

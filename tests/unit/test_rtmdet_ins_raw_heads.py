"""RTMDet-Ins raw-head export contract: graph cut, extra output, host path == native postprocess."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from libreyolo.backends.base import BaseBackend
from libreyolo.export.exporter import _RTMDetInsRawHeadsWrapper
from libreyolo.postprocess.rtmdet import postprocess as rtmdet_postprocess
from libreyolo.postprocess.yolonas_raw import (
    decode_raw_heads,
    raw_head_output_names,
    unwindow_raw_heads,
)

pytestmark = pytest.mark.unit

_LAYOUT = "rtmdet_ins_raw_heads_v1"


def _build(nb_classes: int = 5, seed: int = 0):
    from libreyolo import LibreRTMDet

    torch.manual_seed(seed)
    model = LibreRTMDet(None, size="t", nb_classes=nb_classes, task="segment", device="cpu")
    net = model.model.eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                n = module.num_features
                module.running_mean.copy_(torch.randn(n, generator=generator) * 0.1)
                module.running_var.copy_(torch.rand(n, generator=generator) + 0.5)
        head = net.head
        # Scores around 0.5, positive box distances of a few strides, small
        # kernels: enough survivors for NMS and non-trivial masks.
        for conv in head.rtm_cls:
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.02)
            conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator) * 0.5)
        for conv in head.rtm_reg:
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.02)
            conv.bias.fill_(2.0)
        for conv in head.rtm_kernel:
            conv.weight.copy_(torch.randn(conv.weight.shape, generator=generator) * 0.05)
            conv.bias.copy_(torch.randn(conv.bias.shape, generator=generator) * 0.2)
    return model, net


def test_raw_wrapper_output_contract():
    _, net = _build(nb_classes=5)
    wrapper = _RTMDetInsRawHeadsWrapper(net).eval()
    with torch.no_grad():
        outputs = wrapper(torch.rand(1, 3, 64, 96))

    assert wrapper.output_layout == _LAYOUT
    assert wrapper.strides == (8, 16, 32)
    names = raw_head_output_names(_LAYOUT)
    assert wrapper.output_names == names
    assert names[:3] == ["p3_reg", "p3_cls", "p3_kernel"] and names[-1] == "mask_feat"
    assert len(outputs) == 10
    for name, output in zip(names[:-1], outputs[:-1]):
        level, part = name.split("_")
        stride = 2 ** int(level[1:])
        channels = {"reg": 4, "cls": 5, "kernel": 169}[part]
        assert tuple(output.shape) == (1, channels, 64 // stride, 96 // stride)
        if part == "reg":
            assert float(output.min()) >= 0.0  # ReLU stays in the graph
    assert tuple(outputs[-1].shape) == (1, 8, 64 // 8, 96 // 8)


def test_raw_wrapper_rejects_other_tasks_and_bad_reg_window():
    _, net = _build()
    with pytest.raises(ValueError, match="segment only"):
        _RTMDetInsRawHeadsWrapper(net, "detect")
    with pytest.raises(ValueError, match="must start at 0"):
        _RTMDetInsRawHeadsWrapper(net, output_windows={"reg": (-1.0, 8.0)})


def test_existing_layouts_have_no_extra_outputs():
    assert raw_head_output_names("yolonas_raw_heads_v1") == [
        "p3_reg", "p3_cls", "p4_reg", "p4_cls", "p5_reg", "p5_cls"
    ]
    assert len(raw_head_output_names("rtmdet_raw_heads_v1")) == 6


@pytest.mark.parametrize("hw", [(160, 160), (192, 256)])
def test_host_path_matches_native_postprocess(hw):
    _, net = _build()
    wrapper = _RTMDetInsRawHeadsWrapper(copy.deepcopy(net)).eval()
    sample = torch.rand(1, 3, *hw, generator=torch.Generator().manual_seed(7)) * 255
    with torch.no_grad():
        native_outputs = net(sample)
        raw = [t.numpy() for t in wrapper(sample)]
    original = (hw[1] + 37, hw[0] + 11)  # (w, h), letterboxed into hw
    ratio = min(hw[1] / original[0], hw[0] / original[1])
    kwargs = dict(conf_thres=0.3, iou_thres=0.6, input_size=hw, original_size=original, ratio=ratio, max_det=100)
    expected = rtmdet_postprocess(native_outputs, **kwargs)
    assert expected["num_detections"] > 0

    decoded = decode_raw_heads(_LAYOUT, raw, strides=wrapper.strides)
    boxes, scores, classes, masks = BaseBackend._parse_rtmdet_segment(
        None, decoded, hw, original[0], original[1], 0.3, ratio, iou=0.6, max_det=100
    )
    np.testing.assert_allclose(boxes, expected["boxes"].numpy(), rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(scores, expected["scores"].numpy(), atol=1e-6)
    np.testing.assert_array_equal(classes, expected["classes"].numpy())
    assert masks.shape == tuple(expected["masks"].shape)
    # Mask decode is the same code on the same tensors up to the reg*stride float order.
    assert (masks != expected["masks"].numpy()).mean() < 1e-4

    # Through result building too: no second suppression on top of native NMS.
    class _Backend(BaseBackend):
        def _run_inference(self, blob):
            raise NotImplementedError

    backend = _Backend(
        model_path="dummy", nb_classes=5, device="cpu", imgsz=hw, model_family="rtmdet",
        names={i: str(i) for i in range(5)}, task="segment", supported_tasks=("detect", "segment"),
    )
    result = backend._build_result(
        boxes, scores, classes, masks=masks, orig_shape=(original[1], original[0]),
        image_path=None, iou=0.6, classes=None, max_det=100,
    )
    assert len(result.boxes) == expected["num_detections"]
    np.testing.assert_allclose(result.boxes.xyxy.numpy(), expected["boxes"].numpy(), rtol=1e-5, atol=1e-3)

    # The val path passes ratio=None: the parser derives the letterbox ratio itself.
    boxes_none, *_ = BaseBackend._parse_rtmdet_segment(
        None, decoded, hw, original[0], original[1], 0.3, None, iou=0.6, max_det=100
    )
    np.testing.assert_allclose(boxes_none, boxes, rtol=1e-6, atol=1e-4)


def test_unwindow_leaves_the_extra_output_alone():
    _, net = _build()
    windows = {"cls": (-8.0, 8.0)}
    wrapper = _RTMDetInsRawHeadsWrapper(net, output_windows=windows).eval()
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(torch.rand(1, 3, 64, 64) * 255)]
    restored = unwindow_raw_heads(_LAYOUT, raw, windows)
    assert restored[-1] is raw[-1]
    assert restored[0] is raw[0]  # p3_reg: not windowed
    assert restored[1] is not raw[1]  # p3_cls: windowed


def test_wrapper_on_deep_copy_leaves_the_model_unchanged():
    _, net = _build()
    before = copy.deepcopy(net.state_dict())
    wrapper = _RTMDetInsRawHeadsWrapper(copy.deepcopy(net).eval(), output_windows={"cls": (-8.0, 8.0)})
    with torch.no_grad():
        wrapper(torch.rand(1, 3, 64, 64))
    after = net.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_decode_rejects_wrong_output_count():
    _, net = _build()
    wrapper = _RTMDetInsRawHeadsWrapper(net).eval()
    with torch.no_grad():
        raw = [t.numpy() for t in wrapper(torch.rand(1, 3, 64, 64))]
    with pytest.raises(ValueError, match="expects 10 outputs"):
        decode_raw_heads(_LAYOUT, raw[:-1])

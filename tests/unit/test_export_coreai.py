"""Core AI exporter contracts that do not require Apple hardware."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from libreyolo.export import coreai, coreai_compat
from libreyolo.export.exporter import BaseExporter, CoreAIExporter

pytestmark = pytest.mark.unit


class _Wrapper:
    task = "detect"

    def __init__(self, family="yolo9"):
        self.family = family

    def _get_model_name(self):
        return self.family


def test_coreai_format_defaults_static_and_rejects_dynamic(monkeypatch):
    calls = []

    def fake_call(self, **kwargs):
        calls.append(kwargs)
        return "model.aimodel"

    monkeypatch.setattr(BaseExporter, "__call__", fake_call)
    exporter = CoreAIExporter(_Wrapper())

    assert exporter() == "model.aimodel"
    assert calls == [{"dynamic": False}]
    with pytest.raises(NotImplementedError, match="dynamic=True"):
        exporter(dynamic=True)


def test_coreai_rejects_unsupported_precision():
    exporter = CoreAIExporter(_Wrapper())
    with pytest.raises(NotImplementedError, match="FP16"):
        exporter._validate(True, False, None)
    with pytest.raises(NotImplementedError, match="INT8"):
        exporter._validate(False, True, None)


def test_low_level_coreai_rejects_dynamic_and_precision_before_import():
    model = nn.Identity()
    dummy = torch.zeros(1, 3, 8, 8)
    with pytest.raises(NotImplementedError, match="dynamic=True"):
        coreai.export_coreai(model, dummy, output_path="unused", dynamic=True)
    with pytest.raises(NotImplementedError, match="FP32 only"):
        coreai.export_coreai(model, dummy, output_path="unused", precision="fp16")


def test_blocked_support_is_checked_before_dependency(monkeypatch):
    def dependency_must_not_run():
        raise AssertionError("dependency check ran before support policy")

    monkeypatch.setattr(coreai, "_require_coreai", dependency_must_not_run)
    exporter = CoreAIExporter(_Wrapper("unwired_family"))
    with pytest.raises(NotImplementedError, match="coreai"):
        exporter._preflight(half=False, int8=False, data=None)


def test_structured_metadata_uses_json():
    assert coreai._stringify_metadata_value(["a", "b"]) == '["a","b"]'
    assert coreai._stringify_metadata_value({1: "one"}) == '{"1":"one"}'
    assert coreai._stringify_metadata_value(("x", 2)) == '["x",2]'


def test_frozen_classifier_helper_runs_shared_validation(monkeypatch):
    calls = []

    def fake_validate(self, half, int8, data):
        calls.append(("validate", half, int8, data))
        return half, int8

    def fake_preflight(self, **kwargs):
        calls.append(("preflight", kwargs))

    def fake_metadata(self, precision, dynamic, onnx_path, imgsz=None):
        calls.append(("metadata", precision, dynamic, onnx_path, imgsz))
        return {"precision": precision, "imgsz": imgsz}

    monkeypatch.setattr(CoreAIExporter, "_validate", fake_validate)
    monkeypatch.setattr(CoreAIExporter, "_preflight", fake_preflight)
    monkeypatch.setattr(CoreAIExporter, "_build_metadata", fake_metadata)
    owner = SimpleNamespace(input_size=224)

    size, output, metadata = coreai.prepare_frozen_classifier_export(
        owner,
        {"imgsz": (192, 192), "output_path": "frozen", "device": "cpu"},
        default_output="unused",
    )
    assert (size, output) == (192, "frozen")
    assert metadata == {"precision": "fp32", "imgsz": (192, 192)}
    assert calls == [
        ("validate", False, False, None),
        (
            "preflight",
            {"half": False, "int8": False, "data": None, "nms": False},
        ),
        ("metadata", "fp32", False, None, (192, 192)),
    ]


def test_compat_shim_declines_unverified_toolchain_version(monkeypatch):
    monkeypatch.setattr(coreai_compat, "version", lambda package: "0.4.2")
    assert coreai_compat._patch_avg_pool2d() is False


def test_manual_grid_sample_flag_is_nestable():
    from libreyolo.models.dfine.ms_deform import _FORCE_MANUAL_GRID_SAMPLE

    assert _FORCE_MANUAL_GRID_SAMPLE.get() is False
    restore_outer = coreai._force_manual_grid_sample()
    assert _FORCE_MANUAL_GRID_SAMPLE.get() is True
    restore_inner = coreai._force_manual_grid_sample()
    restore_inner()
    assert _FORCE_MANUAL_GRID_SAMPLE.get() is True
    restore_outer()
    assert _FORCE_MANUAL_GRID_SAMPLE.get() is False


def test_graph_preparation_restores_snapshot_when_prepare_fails(monkeypatch):
    restored = []
    monkeypatch.setattr(
        coreai,
        "_snapshot_rtdetr_static_eval",
        lambda model: lambda: restored.append(model),
    )

    def fail_prepare(*args):
        raise RuntimeError("preparation failed")

    monkeypatch.setattr(coreai, "_prepare_rtdetr_static_eval", fail_prepare)
    model = nn.Identity()
    dummy = torch.zeros(1, 3, 8, 8)
    with (
        pytest.raises(RuntimeError, match="preparation failed"),
        coreai._prepare_coreai_graph(model, dummy, "dfine"),
    ):
        pass
    assert restored == [model]


def test_graph_preparation_restores_replaced_pool_after_capture_error():
    model = nn.Sequential(nn.AdaptiveAvgPool2d(1))
    original = model[0]
    dummy = torch.zeros(1, 3, 8, 8)
    with (
        pytest.raises(RuntimeError, match="capture failed"),
        coreai._prepare_coreai_graph(model, dummy, "resnet"),
    ):
        assert model[0] is not original
        raise RuntimeError("capture failed")
    assert model[0] is original


def test_yolonas_contract_wrapper_keeps_only_decoded_outputs():
    class _SyntheticYoloNAS(nn.Module):
        def forward(self, x):
            return ((x + 1, x + 2), (x + 3, x + 4))

    x = torch.zeros(1, 1, 2, 2)
    wrapped = coreai._wrap_coreai_contract(_SyntheticYoloNAS(), "yolonas")
    boxes, scores = wrapped(x)
    torch.testing.assert_close(boxes, x + 1)
    torch.testing.assert_close(scores, x + 2)


def test_fomo_contract_wrapper_applies_native_normalization():
    x = torch.tensor([[[[0.0, 0.5, 1.0]]]]).expand(1, 3, 1, 3)
    wrapped = coreai._wrap_coreai_contract(nn.Identity(), "fomo")
    torch.testing.assert_close(wrapped(x), (x - 0.5) / 0.5)


def test_graph_preparation_fuses_exact_darknet_batchnorm_and_restores():
    from libreyolo.models.darknet.blocks import DarknetConv

    torch.manual_seed(7)
    model = DarknetConv(
        3,
        4,
        size=3,
        stride=1,
        batch_normalize=True,
        activation="leaky",
    ).eval()
    with torch.no_grad():
        model.bn.running_mean.copy_(torch.randn(4))
        model.bn.running_var.copy_(torch.rand(4) + 0.2)
        model.bn.weight.copy_(torch.randn(4))
        model.bn.bias.copy_(torch.randn(4))

    dummy = torch.randn(1, 3, 8, 8)
    original_conv = model.conv
    original_bn = model.bn
    expected = model(dummy)

    with coreai._prepare_coreai_graph(model, dummy, "yolo1"):
        assert model.conv is not original_conv
        assert model.bn is None
        torch.testing.assert_close(model(dummy), expected, rtol=2e-6, atol=2e-6)

    assert model.conv is original_conv
    assert model.bn is original_bn
    torch.testing.assert_close(model(dummy), expected, rtol=0, atol=0)


def test_darknet_batchnorm_fusion_failure_is_atomic(monkeypatch):
    from libreyolo.models.darknet.blocks import DarknetConv

    model = nn.Sequential(
        DarknetConv(
            3,
            4,
            size=3,
            stride=1,
            batch_normalize=True,
            activation="leaky",
        ),
        DarknetConv(
            4,
            5,
            size=3,
            stride=1,
            batch_normalize=True,
            activation="leaky",
        ),
    ).eval()
    original = [(block.conv, block.bn) for block in model]
    conv2d = nn.Conv2d
    calls = 0

    def fail_second_replacement(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic allocation failure")
        return conv2d(*args, **kwargs)

    monkeypatch.setattr(coreai.nn, "Conv2d", fail_second_replacement)
    with pytest.raises(RuntimeError, match="synthetic allocation failure"):
        coreai._fuse_darknet_batchnorm(model)

    assert [(block.conv, block.bn) for block in model] == original


def test_anchor_freeze_reaches_wrapped_model_and_restores_cache():
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.export = True
            self.frozen = None

        def freeze_anchor_grid(self, input_hw):
            self.frozen = tuple(input_hw)
            self.export = False  # restore must put the caller's flag back

        def unfreeze_anchor_grid(self):
            self.frozen = None

    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = Head()

        def forward(self, tensor):
            return tensor

    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, tensor):
            return self.model(tensor)

    detector = Detector()
    wrapped = Wrapper(detector)
    restore = coreai._freeze_anchor_grid(wrapped, torch.zeros(1, 3, 8, 16))
    assert detector.head.frozen == (8, 16)
    restore()

    assert detector.head.frozen is None
    assert detector.head.export is True


def test_anchor_freeze_is_a_noop_without_a_freezable_head():
    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Identity()

    restore = coreai._freeze_anchor_grid(Detector(), torch.zeros(1, 3, 8, 8))
    restore()


def test_anchor_freeze_pins_the_yolo9_grid_for_the_canvas():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    model = LibreYOLO9Model(config="t", nb_classes=2).eval()
    model.head.export = True
    restore = coreai._freeze_anchor_grid(model, torch.zeros(1, 3, 64, 96))
    try:
        sizes, anchors, _ = model.head._frozen_grid
        assert sizes == ((8, 12), (4, 6), (2, 3))
        assert anchors.shape[1] == 96 + 24 + 6
    finally:
        restore()
    assert model.head._frozen_grid is None and model.head.export is True


def test_rtdetr_snapshot_removes_new_attributes_and_buffers():
    class Encoder(nn.Module):
        use_encoder_idx = (0,)

        def build_2d_sincos_position_embedding(self):
            pass

    class Decoder(nn.Module):
        def _generate_anchors(self):
            pass

    model = SimpleNamespace(encoder=Encoder(), decoder=Decoder())
    restore = coreai._snapshot_rtdetr_static_eval(model)
    model.encoder.eval_spatial_size = (8, 8)
    model.encoder.pos_embed0 = torch.ones(1)
    model.decoder.eval_spatial_size = (8, 8)
    model.decoder.register_buffer("anchors", torch.ones(1))
    model.decoder.register_buffer("valid_mask", torch.ones(1))
    restore()

    assert not hasattr(model.encoder, "eval_spatial_size")
    assert not hasattr(model.encoder, "pos_embed0")
    assert not hasattr(model.decoder, "eval_spatial_size")
    assert "anchors" not in model.decoder._buffers
    assert "valid_mask" not in model.decoder._buffers


def test_rtdetr_snapshot_preserves_existing_none_buffer():
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("anchors", None)

        def _generate_anchors(self):
            pass

    model = SimpleNamespace(encoder=None, decoder=Decoder())
    restore = coreai._snapshot_rtdetr_static_eval(model)
    model.decoder.anchors = torch.ones(1)
    restore()
    assert "anchors" in model.decoder._buffers
    assert model.decoder.anchors is None

"""Hybrid Axelera exports (support tier "hybrid"): RT-DETR's CNN parts on Metis, its transformer on the host.

The fake SDK from test_axelera_backend plays the compiler and the device (it
runs each compiled part's FP32 ONNX), so the chain is checked end to end
without Axelera's packages.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.unit.test_axelera_backend import _dataset, _image, _install_fake_sdk

_PARKED = pytest.mark.skip(
    reason="hybrid export parked by decision 2026-11-19 (part of the network runs on the host); branch axelera-hybrid-parked"
)

pytestmark = pytest.mark.unit
pytest.importorskip("onnxruntime")

OUTPUTS = ("pred_logits", "pred_boxes")
IMGSZ = 160


@pytest.fixture(scope="module")
def sdk():
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield _install_fake_sdk(monkeypatch)


def _rtdetr():
    from libreyolo.models.rtdetr.model import LibreRTDETR

    return LibreRTDETR._from_scratch(size="r18", device="cpu", seed=0)


@_PARKED
def test_rtdetr_axelera_is_a_hybrid_int8_row():
    from libreyolo.export.support import get_support, precision_support

    entry = get_support("rtdetr", "detect", "axelera")
    assert entry.tier == "hybrid"
    assert "host CPU" in entry.reason and "onnxruntime" in entry.constraint
    assert precision_support("rtdetr", "detect", "axelera") == (True, True)


def test_unfrozen_batchnorms_keep_the_outputs():
    from libreyolo.export.exporter import _unfrozen_batchnorms
    from libreyolo.models.rtdetr.backbone import FrozenBatchNorm2d

    torch.manual_seed(0)
    norm = FrozenBatchNorm2d(4)
    norm.weight.uniform_(0.5, 2.0)
    norm.bias.normal_()
    norm.running_mean.normal_()
    norm.running_var.uniform_(0.5, 2.0)
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.Sequential(norm)).eval()
    x = torch.rand(2, 3, 8, 8)
    with torch.no_grad():
        expected = net(x)
        _unfrozen_batchnorms(net)
        actual = net(x)
    assert type(net[1][0]) is torch.nn.BatchNorm2d and not net[1][0].training
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _assert_same_queries(actual, expected, tol):
    """Decoder outputs match up to the order of the queries.

    Query selection is a TopK over the encoder scores: in a random-init model
    two near-tied scores can swap under float noise, and the decoder is
    permutation-equivariant over its queries, so the outputs swap rows too.
    """
    a = np.concatenate([np.asarray(v, dtype=np.float64)[0] for v in actual], axis=-1)
    e = np.concatenate([np.asarray(v, dtype=np.float64)[0] for v in expected], axis=-1)
    assert a.shape == e.shape
    distance = (np.abs(a[:, None, :] - e[None, :, :]) - tol * (1 + np.abs(e[None, :, :]))).max(-1)
    assert (distance.min(axis=1) <= 0).all() and (distance.min(axis=0) <= 0).all()
    assert (np.diagonal(distance) <= 0).mean() >= 0.95


@pytest.mark.parametrize("split", ["backbone", "neck"])
def test_rtdetr_hybrid_parts_reproduce_the_native_model(split):
    from libreyolo.export.exporter import (
        _DETRBackboneWrapper,
        _DETRHostTailWrapper,
        _detr_neck_parts,
        _unfrozen_batchnorms,
    )

    net = _rtdetr().model.eval()
    x = torch.rand(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        expected = net(x)
        copy_ = _unfrozen_batchnorms(copy.deepcopy(net).eval())
        features = _DETRBackboneWrapper(copy_).eval()(x)
        if split == "backbone":
            actual = _DETRHostTailWrapper(copy_, OUTPUTS).eval()(*features)
        else:
            samples = [torch.rand(1, 3, IMGSZ, IMGSZ) for _ in range(2)]
            pack, neck, decoder = _detr_neck_parts(copy_, OUTPUTS, samples)
            assert min(pack.scales) == 1.0
            assert all(float(k).hex().startswith("0x1.0000000000000p") for k in pack.scales)
            canvas = pack(*features)[0]
            assert canvas.shape == (1, 3 * 256, IMGSZ // 8, IMGSZ // 8)
            actual = decoder(*neck(canvas))
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )
    # The caller's model keeps its backbone and its own input projections.
    assert net.backbone is not None
    assert isinstance(net.encoder.input_proj[0], torch.nn.Sequential)
    assert isinstance(net.decoder.input_proj[0], torch.nn.Sequential)


def _export(model, tmp_path, name):
    return Path(
        model.export(
            format="axelera",
            data=str(_dataset(tmp_path)),
            imgsz=IMGSZ,
            output_path=str(tmp_path / f"{name}_axelera_model"),
        )
    )


@_PARKED
@pytest.mark.parametrize("split", ["neck", "backbone"])
def test_rtdetr_hybrid_export_runs_the_chain(sdk, tmp_path, monkeypatch, split):
    import yaml

    from libreyolo import LibreYOLO
    from libreyolo.export import axelera

    if split == "backbone":
        monkeypatch.setitem(
            axelera.AXELERA_SIZE_RECIPES, ("rtdetr", "detect", "r18"), {"hybrid_split": "backbone"}
        )
    model = _rtdetr()
    artifact = _export(model, tmp_path, f"rtdetr_{split}")
    assert not list(tmp_path.glob("*.export_intermediate*"))
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["output_layout"] == "graph_outputs_v1"
    assert meta["graph_outputs"] == meta["output_names"] == list(OUTPUTS)
    assert meta["hybrid"]["split"] == split
    if split == "neck":
        assert len(meta["hybrid"]["canvas_scales"]) == 3
    kinds = ["axm" if "axm" in part else "onnx" for part in meta["parts"]]
    if split == "neck":
        assert kinds == ["axm", "onnx", "axm", "onnx"]
        assert meta["parts"][2]["input"] == "canvas"
        assert sorted(p.name for p in artifact.glob("*.axm")) == [
            "rtdetr_neck.axm",
            "rtdetr_neck_part2.axm",
        ]
    else:
        assert kinds == ["axm", "onnx"]
    assert meta["parts"][0]["input"] == "images"
    assert sorted(meta["parts"][0]["outputs"]) == ["feat_s16", "feat_s32", "feat_s8"]
    assert meta["parts"][-1]["outputs"] == list(OUTPUTS)

    backend = LibreYOLO(str(artifact))
    assert type(backend).__name__ == "AxeleraBackend" and backend.model_family == "rtdetr"
    image = _image(shape=(IMGSZ, IMGSZ, 3))
    blob = model._preprocess(image, input_size=IMGSZ)[0].numpy()
    actual = backend._run_inference(blob)
    with torch.no_grad():
        expected = model.model.eval()(torch.from_numpy(blob))
    _assert_same_queries(actual, [expected[name].numpy() for name in OUTPUTS], 1e-3)
    assert backend.chain_timing["images"] == 1
    assert backend.chain_timing["aipu_ms"] > 0 and backend.chain_timing["host_ms"] > 0
    result = backend(image, conf=0.0)
    assert len(result.boxes) > 0


def test_rtdetr_hybrid_calibration_takes_square_sizes(sdk):
    from libreyolo.export.exporter import AxeleraExporter

    model = _rtdetr()
    preprocess = AxeleraExporter(model)._calibration_preprocess()
    chw, _ = preprocess(_image(shape=(48, 64, 3)), (IMGSZ, IMGSZ))
    assert chw.shape == (3, IMGSZ, IMGSZ) and chw.dtype == np.float32
    with pytest.raises(ValueError, match="square"):
        preprocess(_image(), (IMGSZ, 2 * IMGSZ))


@_PARKED
def test_unknown_hybrid_split_is_rejected(sdk, tmp_path, monkeypatch):
    from libreyolo.export import axelera

    monkeypatch.setitem(
        axelera.AXELERA_SIZE_RECIPES, ("rtdetr", "detect", "r18"), {"hybrid_split": "decoder"}
    )
    with pytest.raises(ValueError, match="hybrid split"):
        _export(_rtdetr(), tmp_path, "bad")


def test_unmeasured_rtdetr_sizes_are_not_available_yet():
    from libreyolo.export.support import BLOCKED_SIZES, UNMEASURED_SIZES

    unmeasured = UNMEASURED_SIZES.get(("rtdetr", "detect", "axelera"), {})
    assert "r18" not in unmeasured
    assert not set(unmeasured) & set(BLOCKED_SIZES.get(("rtdetr", "detect", "axelera"), {}))


@_PARKED
def test_hybrid_neck_part_compiles_per_tensor(sdk, tmp_path, monkeypatch):
    """Depthwise backbones compile per channel; the neck part (graph input into a depthwise conv) stays per tensor."""
    import yaml

    from libreyolo.export import axelera

    monkeypatch.setitem(
        axelera.AXELERA_SIZE_RECIPES,
        ("rtdetr", "detect", "r18"),
        {"quantization_scheme": "hybrid_per_tensor_per_channel"},
    )
    artifact = _export(_rtdetr(), tmp_path, "rtdetr_dw")
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["hybrid"]["neck_quantization_scheme"] == "per_tensor_min_max"
    first = (artifact / "compiler_config_final.toml").read_text()
    second = (artifact / "compiler_config_final.part2.toml").read_text()
    assert "hybrid_per_tensor_per_channel" in first
    assert "per_tensor_min_max" in second and "hybrid_per_tensor_per_channel" not in second


@_PARKED
def test_rtdetrv2_is_hybrid_and_its_parts_reproduce_the_native_model():
    from libreyolo.export.exporter import _AXELERA_HYBRID_OUTPUTS, _DETRBackboneWrapper, _detr_neck_parts
    from libreyolo.export.exporter import _unfrozen_batchnorms
    from libreyolo.export.support import get_support
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    assert get_support("rtdetrv2", "detect", "axelera").tier == "hybrid"
    assert _AXELERA_HYBRID_OUTPUTS[("rtdetrv2", "detect")] == OUTPUTS
    net = LibreRTDETRv2._from_scratch(size="r18", device="cpu", seed=0).model.eval()
    x = torch.rand(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        expected = net(x)
        copy_ = _unfrozen_batchnorms(copy.deepcopy(net).eval())
        features = _DETRBackboneWrapper(copy_).eval()(x)
        pack, neck, decoder = _detr_neck_parts(copy_, OUTPUTS, [torch.rand(1, 3, IMGSZ, IMGSZ)])
        actual = decoder(*neck(pack(*features)[0]))
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )


@_PARKED
def test_detr_hybrid_is_backbone_only_and_reproduces_the_native_model(sdk, tmp_path, monkeypatch):
    from libreyolo.export import axelera
    from libreyolo.export.exporter import (
        AxeleraExporter,
        _VanillaDETRBackboneWrapper,
        _VanillaDETRHostTailWrapper,
        _unfrozen_batchnorms,
    )
    from libreyolo.export.support import get_support
    from libreyolo.models.detr.model import LibreDETR

    assert get_support("detr", "detect", "axelera").tier == "hybrid"
    model = LibreDETR._from_scratch(size="r50", device="cpu", seed=0)
    net = model.model.eval()
    x = torch.rand(1, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        expected = net(x)
        copy_ = _unfrozen_batchnorms(copy.deepcopy(net).eval())
        assert not any(type(m).__name__ == "FrozenBatchNorm2d" for m in copy_.modules())
        features = _VanillaDETRBackboneWrapper(copy_).eval()(x)
        actual = _VanillaDETRHostTailWrapper(copy_, OUTPUTS).eval()(*features)
    for name, value in zip(OUTPUTS, actual):
        torch.testing.assert_close(value, expected[name], atol=1e-4, rtol=1e-4)
    assert AxeleraExporter(model)._hybrid_split() == "backbone"
    monkeypatch.setitem(axelera.AXELERA_SIZE_RECIPES, ("detr", "detect", "r50"), {"hybrid_split": "neck"})
    with pytest.raises(ValueError, match="no convolutional neck"):
        AxeleraExporter(model)._hybrid_split()


@_PARKED
@pytest.mark.parametrize("family", ["dfine", "deim", "rtdetrv4"])
def test_dfine_line_is_hybrid_and_its_parts_reproduce_the_native_model(family):
    from libreyolo.export.exporter import _AXELERA_HYBRID_OUTPUTS, _DETRBackboneWrapper, _detr_neck_parts
    from libreyolo.export.exporter import _unfrozen_batchnorms
    from libreyolo.export.support import get_support
    from libreyolo.models.deim.model import LibreDEIM
    from libreyolo.models.dfine.model import LibreDFINE

    assert get_support(family, "detect", "axelera").tier == "hybrid"
    assert _AXELERA_HYBRID_OUTPUTS[(family, "detect")] == OUTPUTS
    from libreyolo.models.rtdetrv4.model import LibreRTDETRv4

    cls = {"dfine": LibreDFINE, "deim": LibreDEIM, "rtdetrv4": LibreRTDETRv4}[family]
    net = cls._from_scratch(size="s" if family == "rtdetrv4" else "n", device="cpu", seed=0).model.eval()
    # D-FINE-n has two levels (strides 16 and 32): 320 px gives 500 positions for its 300 queries.
    size = 320
    x = torch.rand(1, 3, size, size)
    with torch.no_grad():
        expected = net(x)
        copy_ = _unfrozen_batchnorms(copy.deepcopy(net).eval())
        features = _DETRBackboneWrapper(copy_).eval()(x)
        pack, neck, decoder = _detr_neck_parts(copy_, OUTPUTS, [torch.rand(1, 3, size, size)])
        actual = decoder(*neck(pack(*features)[0]))
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )


@_PARKED
def test_faster_rcnn_hybrid_export_runs_the_chain(sdk, tmp_path, monkeypatch):
    import yaml

    from libreyolo import LibreYOLO
    from libreyolo.export import support
    from libreyolo.export.support import get_support
    from libreyolo.models.faster_rcnn.model import LibreFasterRCNN

    assert get_support("faster_rcnn", "detect", "axelera").tier == "hybrid"
    monkeypatch.setitem(support.UNMEASURED_SIZES, ("faster_rcnn", "detect", "axelera"), {})
    model = LibreFasterRCNN._from_scratch(size="n", device="cpu", seed=0)
    data = str(_dataset(tmp_path))
    artifact = Path(
        model.export(
            format="axelera",
            imgsz=320,
            data=data,
            output_path=str(tmp_path / "frcnn_axelera_model"),
        )
    )
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["hybrid"]["split"] == "backbone"
    assert [("axm" if "axm" in part else "onnx") for part in meta["parts"]] == ["axm", "onnx"]
    assert sorted(meta["parts"][0]["outputs"]) == ["fpn0", "fpn1"]
    assert meta["parts"][-1]["outputs"] == ["boxes", "scores", "labels"]
    backend = LibreYOLO(str(artifact))
    image = _image(shape=(240, 320, 3))
    result = backend(image, conf=0.0)
    assert result.boxes.xyxy.shape[-1] == 4
    with pytest.raises(ValueError, match="canvas"):
        model.export(
            format="axelera", imgsz=256, data=data, output_path=str(tmp_path / "bad_axelera_model")
        )


def test_hybrid_neck_scheme_is_per_family_with_a_recipe_override(sdk, monkeypatch):
    from libreyolo.export import axelera
    from libreyolo.export.exporter import AxeleraExporter
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    assert AxeleraExporter(_rtdetr())._hybrid_neck_scheme() == "per_tensor_min_max"
    v2 = LibreRTDETRv2._from_scratch(size="r18", device="cpu", seed=0)
    # r18 keeps the min-max neck it was measured with (a per-size recipe).
    assert AxeleraExporter(v2)._hybrid_neck_scheme() == "per_tensor_min_max"
    monkeypatch.setitem(axelera.AXELERA_SIZE_RECIPES, ("rtdetrv2", "detect", "r18"), {})
    assert AxeleraExporter(v2)._hybrid_neck_scheme() == "per_tensor_histogram"


def test_faster_rcnn_measured_sizes_leave_the_unmeasured_list():
    from libreyolo.export.support import UNMEASURED_SIZES

    assert ("faster_rcnn", "detect", "axelera") not in UNMEASURED_SIZES


@_PARKED
def test_deimv2_small_sizes_take_the_encoder_split_and_reproduce_the_native_model():
    from libreyolo.export.exporter import AxeleraExporter, _detr_conv_encoder_parts, _unfrozen_batchnorms
    from libreyolo.export.support import BLOCKED_SIZES, get_support
    from libreyolo.models.deimv2.model import LibreDEIMv2

    assert get_support("deimv2", "detect", "axelera").tier == "hybrid"
    assert "DINOv3" in BLOCKED_SIZES[("deimv2", "detect", "axelera")]["s"]
    model = LibreDEIMv2._from_scratch(size="atto", device="cpu", seed=0)
    assert AxeleraExporter(model)._hybrid_split() == "encoder"
    net = model.model.eval()
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        expected = net(x)
        head, tail = _detr_conv_encoder_parts(_unfrozen_batchnorms(copy.deepcopy(net).eval()), OUTPUTS)
        memory = head(x)
        assert len(memory) == len(head.output_names) == 2
        actual = tail(*memory)
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )


@pytest.mark.parametrize("task", ["segment", "detect"])
def test_mask_rcnn_hybrid_chain_matches_the_export_wrapper(task):
    """Mask R-CNN backbone-split hybrid in FP32: the Metis part (normalization, backbone, FPN)
    then the host step (pool level, RPN, RoI box and mask heads, mask paste) reproduce
    MaskRCNNExportWrapper at the canvas."""
    import copy

    from libreyolo.export.exporter import (
        _AXELERA_HYBRID_OUTPUTS,
        _HYBRID_BACKBONE_WRAPPERS,
        _MaskRCNNHostTailWrapper,
        _RCNNBackboneWrapper,
    )
    from libreyolo.models.mask_rcnn.nn import LibreMaskRCNNModel, MaskRCNNExportWrapper

    assert _HYBRID_BACKBONE_WRAPPERS["mask_rcnn"] == ("_RCNNBackboneWrapper", "_MaskRCNNHostTailWrapper")
    outputs = _AXELERA_HYBRID_OUTPUTS[("mask_rcnn", task)]
    torch.manual_seed(0)
    net = LibreMaskRCNNModel().eval()
    net.transform.min_size, net.transform.max_size = (256,), 256
    net.roi_heads.score_thresh = 0.0
    x = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = MaskRCNNExportWrapper(copy.deepcopy(net), include_masks=task == "segment").eval()(x)
        levels = _RCNNBackboneWrapper(net).eval()(x)
        got = _MaskRCNNHostTailWrapper(net, outputs, canvas=(256, 256)).eval()(*levels)
    assert len(got) == len(want) == len(outputs)
    assert got[0].shape[0] == want[0].shape[0] > 0
    # Random-init ResNet-50-FPN v2: float accumulation moves box corners by < 0.01 px (of 256).
    torch.testing.assert_close(got[0], want[0], atol=0.05, rtol=0)
    torch.testing.assert_close(got[1], want[1], atol=1e-3, rtol=0)
    assert torch.equal(got[2], want[2])
    if task == "segment":
        assert got[3].shape == want[3].shape and got[3].shape[-2:] == (256, 256)
        torch.testing.assert_close(got[3], want[3], atol=1e-2, rtol=0)


@_PARKED
def test_mask_rcnn_hybrid_segment_export_runs_the_chain(sdk, tmp_path, monkeypatch):
    import yaml

    from libreyolo import LibreYOLO
    from libreyolo.export import support
    from libreyolo.export.support import get_support
    from libreyolo.models.mask_rcnn.model import LibreMaskRCNN

    assert get_support("mask_rcnn", "segment", "axelera").tier == "hybrid"
    # r50 is measured for both tasks (docs/axelera.md).
    assert ("mask_rcnn", "segment", "axelera") not in support.UNMEASURED_SIZES
    assert ("mask_rcnn", "detect", "axelera") not in support.UNMEASURED_SIZES
    model = LibreMaskRCNN._from_scratch(size="r50", task="segment", device="cpu", seed=0)
    # A small canvas keeps the test light; the export takes the model's min_size as its canvas.
    model.model.transform.min_size, model.model.transform.max_size = (256,), 256
    artifact = Path(
        model.export(
            format="axelera",
            imgsz=256,
            data=str(_dataset(tmp_path)),
            output_path=str(tmp_path / "maskrcnn_axelera_model"),
        )
    )
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["hybrid"]["split"] == "backbone"
    assert [("axm" if "axm" in part else "onnx") for part in meta["parts"]] == ["axm", "onnx"]
    assert sorted(meta["parts"][0]["outputs"]) == ["fpn0", "fpn1", "fpn2", "fpn3"]
    assert meta["parts"][-1]["outputs"] == ["boxes", "scores", "labels", "masks"]
    backend = LibreYOLO(str(artifact))
    result = backend(_image(shape=(240, 320, 3)), conf=0.0)
    assert result.boxes.xyxy.shape[-1] == 4
    if len(result.boxes):
        assert tuple(result.masks.data.shape[-2:]) == (240, 320)


def test_ppdoclayout_hybrid_chain_matches_the_export_wrapper():
    """PP-DocLayoutV3 backbone-split hybrid in FP32: the Metis part (HGNetV2 and the encoder
    input projections) then the host step reproduce PPDocLayoutV3ExportWrapper."""
    import copy

    from libreyolo.export.exporter import (
        _AXELERA_HYBRID_OUTPUTS,
        _DocLayoutBackboneWrapper,
        _DocLayoutHostTailWrapper,
    )
    from libreyolo.models.ppdoclayoutv3.nn import PPDocLayoutV3ExportWrapper, PPDocLayoutV3Model

    outputs = _AXELERA_HYBRID_OUTPUTS[("ppdoclayoutv3", "detect")]
    torch.manual_seed(0)
    net = PPDocLayoutV3Model().eval()
    x = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = PPDocLayoutV3ExportWrapper(copy.deepcopy(net)).eval()(x)
        head = _DocLayoutBackboneWrapper(net).eval()
        levels = head(x)
        got = _DocLayoutHostTailWrapper(net, outputs).eval()(*levels)
    assert head.output_names == ("feat_s4", "proj_s8", "proj_s16", "proj_s32")
    assert net.backbone is not None and net.encoder_input_proj is not None
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4)


@_PARKED
def test_ppdoclayout_hybrid_export_runs_the_chain(sdk, tmp_path, monkeypatch):
    import yaml

    from libreyolo import LibreYOLO
    from libreyolo.export import support
    from libreyolo.export.support import get_support
    from libreyolo.models.ppdoclayoutv3.model import LibrePPDocLayoutV3

    assert get_support("ppdoclayoutv3", "detect", "axelera").tier == "hybrid"
    assert ("ppdoclayoutv3", "detect", "axelera") not in support.UNMEASURED_SIZES
    model = LibrePPDocLayoutV3._from_scratch(size="l", device="cpu", seed=0)
    artifact = Path(
        model.export(
            format="axelera",
            imgsz=256,
            data=str(_dataset(tmp_path)),
            output_path=str(tmp_path / "doclayout_axelera_model"),
        )
    )
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["hybrid"]["split"] == "backbone"
    assert [("axm" if "axm" in part else "onnx") for part in meta["parts"]] == ["axm", "onnx"]
    assert sorted(meta["parts"][0]["outputs"]) == ["feat_s4", "proj_s16", "proj_s32", "proj_s8"]
    assert meta["parts"][-1]["outputs"] == ["pred_logits", "pred_boxes", "order_votes"]
    result = LibreYOLO(str(artifact))(_image(shape=(240, 320, 3)), conf=0.0)
    assert result.boxes.xyxy.shape[-1] == 4


@_PARKED
def test_deformable_detr_hybrid_is_backbone_only_and_reproduces_the_native_model():
    from libreyolo.export.exporter import (
        AxeleraExporter,
        _DeformableDETRBackboneWrapper,
        _DeformableDETRHostTailWrapper,
        _unfrozen_batchnorms,
    )
    from libreyolo.export.support import get_support
    from libreyolo.models.deformable_detr.model import LibreDeformableDETR

    assert get_support("deformable_detr", "detect", "axelera").tier == "hybrid"
    model = LibreDeformableDETR._from_scratch(size="r50", device="cpu", seed=0)
    assert AxeleraExporter(model)._hybrid_split() == "backbone"
    net = model.model.eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        expected = net(x)
        copy_ = _unfrozen_batchnorms(copy.deepcopy(net).eval())
        actual = _DeformableDETRHostTailWrapper(copy_, OUTPUTS)(*_DeformableDETRBackboneWrapper(copy_)(x))
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )


@_PARKED
def test_dfine_segment_hybrid_chain_passes_the_low_level_tap(sdk, tmp_path, monkeypatch):
    import yaml

    from libreyolo import LibreYOLO
    from libreyolo.export import support
    from libreyolo.export.support import get_support
    from libreyolo.models.dfine.model import LibreDFINE

    assert get_support("dfine", "segment", "axelera").tier == "hybrid"
    monkeypatch.setitem(support.UNMEASURED_SIZES, ("dfine", "segment", "axelera"), {})
    model = LibreDFINE._from_scratch(size="n", device="cpu", seed=0, task="segment")
    artifact = Path(
        model.export(
            format="axelera", data=str(_dataset(tmp_path)), imgsz=320,
            output_path=str(tmp_path / "dfine_seg_axelera_model"),
        )
    )
    meta = yaml.safe_load((artifact / "metadata.yaml").read_text())
    assert meta["graph_outputs"] == ["pred_logits", "pred_boxes", "pred_masks"]
    assert "low_level" in meta["parts"][0]["outputs"]
    # The host decoder step takes the neck outputs plus low_level and hands the mask
    # decoder's fusion input to the compiled mask convolutions; the last host step
    # makes the masks.
    decoder_step = next(p for p in meta["parts"] if "mask_fuse_in" in p.get("outputs", []))
    assert decoder_step["inputs"][-1] == "low_level"
    assert "low_level" not in meta["parts"][1]["inputs"]
    assert meta["parts"][-1]["outputs"] == ["pred_masks"]
    assert meta["parts"][-1]["inputs"] == ["mask_up_raw", "mask_embed"]
    backend = LibreYOLO(str(artifact))
    blob = model._preprocess(_image(shape=(320, 320, 3)), input_size=320)[0].numpy()
    actual = backend._run_inference(blob)
    with torch.no_grad():
        expected = model.model.eval()(torch.from_numpy(blob))
    assert actual[2].shape == expected["pred_masks"].shape


def test_dfine_segment_mask_convolution_parts_reproduce_the_host_mask_decoder():
    from libreyolo.export.exporter import (
        _DETRBackboneWrapper,
        _DETRDecoderMaskFrontWrapper,
        _detr_neck_parts,
        _MaskLogits,
        _MaskNormUp,
        _unfrozen_batchnorms,
    )
    from libreyolo.models.dfine.model import LibreDFINE

    names = ("pred_logits", "pred_boxes", "pred_masks")
    model = LibreDFINE._from_scratch(size="n", device="cpu", seed=0, task="segment")
    net = _unfrozen_batchnorms(copy.deepcopy(model.model).eval())
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        backbone = _DETRBackboneWrapper(net).eval()
        feats = backbone(x)
        pack, neck, decoder = _detr_neck_parts(net, names, project_on_host=True)
        levels = len([n for n in backbone.output_names if n != "low_level"])
        memory = tuple(neck(pack(*feats[:levels])[0])) + tuple(feats[levels:])
        expected = decoder(*memory)
        front = _DETRDecoderMaskFrontWrapper(decoder.decoder, names, low_level=decoder.low_level).eval()
        logits, boxes, embed, fuse_in = front(*memory)
        md = decoder.decoder.mask_decoder
        up_in = _MaskNormUp(md.fusion_norm)(md.fusion_conv(fuse_in))
        masks = _MaskLogits(md.bn1)(md.up_conv(up_in), embed)
    for actual, want in zip((logits, boxes, masks), expected):
        assert torch.allclose(actual, want, atol=1e-5)


@_PARKED
def test_dinodetr_hybrid_is_backbone_only_and_reproduces_the_native_model():
    from libreyolo.export.exporter import (
        AxeleraExporter,
        _DINODETRBackboneWrapper,
        _DINODETRHostTailWrapper,
        _unfrozen_batchnorms,
    )
    from libreyolo.export.support import BLOCKED_SIZES, get_support
    from libreyolo.models.dinodetr.model import LibreDINODETR

    assert get_support("dinodetr", "detect", "axelera").tier == "hybrid"
    assert "swinl" in BLOCKED_SIZES[("dinodetr", "detect", "axelera")]
    model = LibreDINODETR._from_scratch(size="r50", device="cpu", seed=0)
    assert AxeleraExporter(model)._hybrid_split() == "backbone"
    net = model.model.eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        expected = net(x)
        copy_ = _unfrozen_batchnorms(copy.deepcopy(net).eval())
        actual = _DINODETRHostTailWrapper(copy_, OUTPUTS)(*_DINODETRBackboneWrapper(copy_)(x))
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )


@_PARKED
def test_every_deformable_detr_size_is_measured_on_axelera():
    from libreyolo.export.support import UNMEASURED_SIZES, get_support

    assert get_support("deformable_detr", "detect", "axelera").tier == "hybrid"
    assert ("deformable_detr", "detect", "axelera") not in UNMEASURED_SIZES


@_PARKED
def test_rtdetrv2_obb_hybrid_neck_split_reproduces_the_native_model():
    from libreyolo.export.exporter import (
        AxeleraExporter,
        _DETRBackboneWrapper,
        _detr_neck_parts,
        _unfrozen_batchnorms,
    )
    from libreyolo.export.support import get_support
    from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

    assert get_support("rtdetrv2", "obb", "axelera").tier == "hybrid"
    model = LibreRTDETRv2._from_scratch(size="n", device="cpu", seed=0, task="obb")
    assert AxeleraExporter(model)._hybrid_split() == "neck"
    net = _unfrozen_batchnorms(copy.deepcopy(model.model).eval())
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        expected = net(x)
        feats = _DETRBackboneWrapper(net).eval()(x)
        pack, neck, decoder = _detr_neck_parts(net, OUTPUTS)
        actual = decoder(*neck(pack(*feats)[0]))
    assert actual[1].shape[-1] == 5  # cx, cy, w, h, angle
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )


@_PARKED
@pytest.mark.parametrize("size", ["n", "s"])
def test_riodetrv2_obb_hybrid_neck_split_reproduces_the_native_model(size):
    from libreyolo.export.exporter import (
        AxeleraExporter,
        _DETRBackboneWrapper,
        _detr_neck_parts,
        _unfrozen_batchnorms,
    )
    from libreyolo.export.support import get_support
    from libreyolo.models.riodetrv2.model import LibreRiODETRv2

    assert get_support("riodetrv2", "obb", "axelera").tier == "hybrid"
    model = LibreRiODETRv2._from_scratch(size=size, device="cpu", seed=0, task="obb")
    assert AxeleraExporter(model)._hybrid_split() == "neck"
    net = _unfrozen_batchnorms(copy.deepcopy(model.model).eval())
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        expected = net(x)
        feats = _DETRBackboneWrapper(net).eval()(x)
        pack, neck, decoder = _detr_neck_parts(net, OUTPUTS)
        actual = decoder(*neck(pack(*feats)[0]))
    assert actual[1].shape[-1] == 5  # cx, cy, w, h, angle
    _assert_same_queries(
        [v.numpy() for v in actual], [expected[name].numpy() for name in OUTPUTS], 1e-4
    )


def test_riodetrv2_linear_angle_delta_exports_in_float32():
    """torch.onnx writes linspace as float64; the decoded angle must come back to float32 so the
    rotated sampling grid fed to GridSample matches its float32 input (ONNX Runtime has no mixed kernel)."""
    import io

    onnx = pytest.importorskip("onnx")
    from libreyolo.models.riodetrv2.decoder import decode_linear_angle_delta

    class Angle(torch.nn.Module):
        def forward(self, x):
            return decode_linear_angle_delta(x, 1 / 3)

    x = torch.rand(1, 4, 36)
    assert decode_linear_angle_delta(x, 1 / 3).dtype == torch.float32
    buf = io.BytesIO()
    torch.onnx.export(Angle(), x, buf, opset_version=17, dynamo=False)
    graph = onnx.load_from_string(buf.getvalue()).graph
    assert graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def test_riodetrv2_obb_is_measured_with_a_histogram_neck():
    from libreyolo.export.exporter import _HYBRID_NECK_QUANTIZATION_SCHEMES
    from libreyolo.export.support import UNMEASURED_SIZES

    assert _HYBRID_NECK_QUANTIZATION_SCHEMES["riodetrv2"] == "per_tensor_histogram"
    assert ("riodetrv2", "obb", "axelera") not in UNMEASURED_SIZES


def test_retinanet_r50v2_hybrid_body_split_reproduces_the_native_model():
    """RetinaNet-R50v2 recipe hybrid in FP32: the ResNet-50 body (Metis part) then FPN, GroupNorm heads,
    anchors and decode on the host reproduce RetinaNetExportWrapper's output."""
    import copy

    from libreyolo.export.axelera import size_recipe
    from libreyolo.export.exporter import AxeleraExporter, _RetinaNetBodyWrapper, _RetinaNetHostTailWrapper
    from libreyolo.models.retinanet.model import LibreRetinaNet
    from libreyolo.models.retinanet.nn import RetinaNetExportWrapper

    assert size_recipe("retinanet", "detect", "r50v2")["hybrid_split"] == "backbone"
    model = LibreRetinaNet._from_scratch(size="r50v2", device="cpu", seed=0)
    assert AxeleraExporter(model)._hybrid_outputs() == ("output",)
    assert AxeleraExporter(LibreRetinaNet._from_scratch(size="r50", device="cpu", seed=0))._hybrid_outputs() == ()
    net = model.model.eval()
    x = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = RetinaNetExportWrapper(copy.deepcopy(net)).eval()(x)
        got = _RetinaNetHostTailWrapper(net, ("output",)).eval()(*_RetinaNetBodyWrapper(net).eval()(x))[0]
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("size", ["resdcn18", "dla34"])
def test_centernet_hybrid_body_split_reproduces_the_native_model(size):
    """CenterNet recipe hybrid in FP32: the body (ResNet-18 or the DLA-34 base; Metis part) then the
    deformable up-path, heads and peak decode on the host reproduce CenterNetExportWrapper's output."""
    import copy

    from libreyolo.export.exporter import AxeleraExporter, _CenterNetBodyWrapper, _CenterNetHostTailWrapper
    from libreyolo.models.centernet.model import LibreCenterNet
    from libreyolo.models.centernet.nn import CenterNetExportWrapper

    model = LibreCenterNet._from_scratch(size=size, device="cpu", seed=0)
    net = model.model.eval()
    x = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = CenterNetExportWrapper(copy.deepcopy(net)).eval()(x)
        got = _CenterNetHostTailWrapper(net, ("output",)).eval()(*_CenterNetBodyWrapper(net).eval()(x))[0]
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)



def test_hrnet_w32_hybrid_stage3_split_reproduces_the_native_model():
    """HRNet-w32 recipe hybrid in FP32: stem to stage 3 (Metis part, three branches) then stage 4 and
    the heatmap layer on the host reproduce the native heatmaps, at the fixed 256x192 person crop."""
    from libreyolo import LibreHRNet
    from libreyolo.export.exporter import AxeleraExporter, _HRNetBodyWrapper, _HRNetHostTailWrapper

    model = LibreHRNet(size="w32", device="cpu")
    exporter = AxeleraExporter(model)
    assert exporter._hybrid_outputs() == ("heatmaps",) and exporter._hybrid_split() == "backbone"
    assert exporter._rectangular_imgsz_supported()
    net = model.model.eval()
    x = torch.rand(1, 3, 256, 192, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = net(x)
        body = _HRNetBodyWrapper(net).eval()
        feats = body(x)
        got = _HRNetHostTailWrapper(net, ("heatmaps",)).eval()(*feats)[0]
    assert body.output_names == ("branch0", "branch1", "branch2") and len(feats) == 3
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_dekr_w32_hybrid_stage3_split_reproduces_the_native_model():
    """DEKR-w32 recipe hybrid in FP32: the trunk's stem to stage 3 (Metis part, identity-conv outputs) then
    stage 4 and the heads on the host reproduce the native (heatmap_logits, offsets)."""
    from libreyolo.export.exporter import AxeleraExporter, _DEKRBodyWrapper, _DEKRHostTailWrapper
    from libreyolo.models.dekr.model import LibreDEKR

    model = LibreDEKR(size="w32", device="cpu")
    exporter = AxeleraExporter(model)
    assert exporter._hybrid_outputs() == ("heatmap_logits", "offsets") and exporter._hybrid_split() == "backbone"
    net = model.model.eval()
    x = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = net(x)
        body = _DEKRBodyWrapper(net).eval()
        feats = body(x)
        got = _DEKRHostTailWrapper(net, ("heatmap_logits", "offsets")).eval()(*feats)
    assert body.output_names == ("branch0", "branch1", "branch2")
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4)


def test_zipdepth_bnpu_hybrid_gcb_split_reproduces_the_native_model():
    """ZipDepth-bnpu recipe hybrid in FP32: the encoder to the GCNet block (Metis part; channel and strip
    attention in their exact pooling forms, identity-conv outputs) then the context block, stage 4 and the
    decoder on the host reproduce the native depth; the caller's model is left unpatched."""
    from libreyolo import LibreZipDepth
    from libreyolo.export.exporter import AxeleraExporter, _ZipDepthBodyWrapper, _ZipDepthHostTailWrapper

    model = LibreZipDepth(None, size="bnpu", device="cpu")
    exporter = AxeleraExporter(model)
    assert exporter._hybrid_outputs() == ("depth",) and exporter._hybrid_split() == "backbone"
    net = model.model.eval()
    x = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = net(x)
        body = _ZipDepthBodyWrapper(net).eval()
        got = _ZipDepthHostTailWrapper(net, ("depth",)).eval()(*body(x))[0]
        again = net(x)
    assert body.output_names == ("s_half", "s1", "s2", "s3_pre")
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(again, want)  # the rewrites stay on the wrapper's copy


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_pidnet_hybrid_layer5_split_reproduces_the_native_model(size):
    """PIDNet recipe hybrid (s, m, l) in FP32: the branches through layer 5 (Metis part; both PagFM fusions in the
    exact sigmoid(-z) * x + sigmoid(z) * y form, on the wrapper's copy) then PAPPM, the branch fusion and
    the head on the host reproduce the native logits; the caller's model is left unpatched."""
    from libreyolo import LibrePIDNet
    from libreyolo.export.exporter import AxeleraExporter, _PIDNetBodyWrapper, _PIDNetHostTailWrapper

    torch.manual_seed(0)
    model = LibrePIDNet(None, size=size, device="cpu")
    exporter = AxeleraExporter(model)
    assert exporter._hybrid_outputs() == ("semantic_logits",) and exporter._hybrid_split() == "backbone"
    # Float64: with random weights the FP32 blends round differently in the last bits (one element in
    # 19456 off by 4.7e-4 at FP32, where 1 - sigmoid(z) itself cancels); in float64 the rewrite is exact.
    net = model.model.eval().double()
    x = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    with torch.no_grad():
        want = net(x)
        body = _PIDNetBodyWrapper(net).eval().double()  # its added 1x1 convs are built in float32
        got = _PIDNetHostTailWrapper(net, ("semantic_logits",)).eval()(*body(x))[0]
        again = net(x)
    assert all(type(m).__name__ != "PagFM" or "forward" not in vars(m) for m in net.modules())
    torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(again, want)


def test_rtmdet_x_hybrid_backbone_split_reproduces_the_export_output():
    """RTMDet-x recipe hybrid in FP32: the CSPNeXt backbone (Metis part, identity-conv outputs) then the PAFPN
    neck and the head in export mode on the host reproduce the model's decoded export output."""
    from libreyolo import LibreRTMDet
    from libreyolo.export.exporter import AxeleraExporter, _RTMDetBodyWrapper, _RTMDetHostTailWrapper

    torch.manual_seed(0)
    model = LibreRTMDet(None, size="x", device="cpu")
    exporter = AxeleraExporter(model)
    assert exporter._hybrid_outputs() == ("output",) and exporter._hybrid_split() == "backbone"
    net = model.model.eval()
    x = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        net.head.export = True
        want = net(x)
        net.head.export = False
        got = _RTMDetHostTailWrapper(net, ("output",)).eval()(*_RTMDetBodyWrapper(net).eval()(x))[0]
    assert net.head.export is False
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@_PARKED
def test_tinyformer_stem_split_reproduces_the_native_model_within_the_sigmoid_gelu_bound():
    from libreyolo.export.exporter import _STEM_SIGMOID_GELU, AxeleraExporter, _vit_stem_parts
    from libreyolo.export.support import get_support
    from libreyolo.models.tinyformer.model import LibreTinyFormer

    assert get_support("tinyformer", "detect", "axelera").tier == "hybrid"
    model = LibreTinyFormer._from_scratch(size="s", device="cpu", seed=0)
    assert AxeleraExporter(model)._hybrid_split() == "stem"
    net = model.model.eval().deploy()
    stem, host, neck, decoder = _vit_stem_parts(net, OUTPUTS)
    x = torch.rand(1, 3, 640, 640)
    # The compiled stem runs its two inner GELUs as SiLU(a x) / a (a folded into the convolutions, exact), and
    # stops before the last GELU of each output, which the host step applies.
    assert sum(isinstance(m, torch.nn.SiLU) for m in stem.modules()) == 2
    with torch.no_grad():
        b = net.backbone
        sig = lambda t: t * torch.sigmoid(_STEM_SIGMOID_GELU * t)  # noqa: E731
        s4 = sig(b.sda[1][1](b.sda[1][0](sig(b.sda[0][1](b.sda[0][0](x))))))
        c1, s8 = stem(x)
        torch.testing.assert_close(c1, b.proj_c1[1](b.proj_c1[0](s4)), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(s8, b.sda[2][1](b.sda[2][0](s4)), atol=1e-4, rtol=1e-4)
        expected = net(x)
        assert tuple(host(x, c1, s8)[0].shape) == (1, 4 * 192, 160, 160)
        actual = decoder(*neck(host(x, c1, s8)[0]))
    # Documented bound (support text, measured on TinyFormer-s weights over cocoval300 images): the sorted top-10
    # detection scores move by less than 0.013 (sorted: the approximation may reorder near-tied queries).
    top = lambda logits: logits.sigmoid().flatten().topk(10).values  # noqa: E731
    assert float((top(actual[0]) - top(expected["pred_logits"])).abs().max()) < 0.013


@pytest.mark.parametrize("size", ["d3", "d4"])
def test_efficientdet_hybrid_backbone_split_reproduces_the_export_output(size):
    """EfficientDet-d3/d4 recipe hybrid in FP32, at a small canvas: the backbone (d3) or its stages 0-4 (d4) on
    the Metis side, then the remaining stages, BiFPN, heads and decode on the host reproduce
    EfficientDetExportWrapper's output."""
    from libreyolo import LibreEfficientDet
    from libreyolo.export.exporter import (
        AxeleraExporter,
        _efficientdet_at_canvas,
        _EfficientDetBodyWrapper,
        _EfficientDetHostTailWrapper,
    )
    from libreyolo.models.efficientdet.nn import EfficientDetExportWrapper

    torch.manual_seed(0)
    model = LibreEfficientDet(None, size=size, device="cpu")
    assert AxeleraExporter(model)._hybrid_outputs() == ("output",)
    net = _efficientdet_at_canvas(model, 256).model.eval()
    x = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want_cls, want_box = net(x)
        body = _EfficientDetBodyWrapper(net).eval()
        tail = _EfficientDetHostTailWrapper(net, ("output",)).eval()
        got_cls, got_box = tail.heads(*body(x))
        decoded = tail(*body(x))[0]
    assert body.output_names == (("stage2", "stage4", "stage6") if size == "d3" else ("stage2", "stage4"))
    # Compared before the top-k decode: with random weights near-tied logits make the candidate order fragile.
    for g, w in zip(list(got_cls) + list(got_box), list(want_cls) + list(want_box)):
        torch.testing.assert_close(g, w, atol=1e-4, rtol=1e-4)
    assert decoded.shape == EfficientDetExportWrapper(net, input_size=256, max_candidates=5000).eval()(x).shape

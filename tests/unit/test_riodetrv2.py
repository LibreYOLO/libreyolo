"""RiO-DETRv2: recognition, routing, upstream remap, canvas handling, shapes."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.riodetrv2]


def _scratch(size="n", nb_classes=15):
    from libreyolo.models.riodetrv2.model import LibreRiODETRv2

    return LibreRiODETRv2(None, size=size, nb_classes=nb_classes, device="cpu")


@pytest.fixture(scope="module")
def rio_n():
    return _scratch("n")


def _as_upstream(state_dict):
    """Rename native keys back to the upstream layout and add training tensors."""
    out = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        if key.startswith("encoder.input_proj.") and parts[3] in ("0", "1"):
            parts[3] = "conv" if parts[3] == "0" else "norm"
        out[".".join(parts)] = value
    out["encoder.feature_projector.0.weight"] = torch.zeros(768, 128)
    out["encoder.feature_projector.0.bias"] = torch.zeros(768)
    return out


def test_filenames_variants_and_download_urls():
    from libreyolo.models.riodetrv2.model import LibreRiODETRv2 as R

    assert R.detect_size_from_filename("LibreRiODETRv2x-obb.pt") == "x"
    assert R.detect_task_from_filename("LibreRiODETRv2x-obb.pt") == "obb"
    assert R.detect_variant_from_filename("LibreRiODETRv2s-obb-diorr.pt") == "diorr"
    assert R.get_download_url("LibreRiODETRv2n-obb.pt") == (
        "https://huggingface.co/LibreYOLO/LibreRiODETRv2n-obb/resolve/"
        f"{R.HF_REVISIONS['LibreRiODETRv2n-obb']}/LibreRiODETRv2n-obb.pt"
    )
    assert R.get_download_url("LibreRiODETRv2l-obb-diorr.pt").endswith(
        f"/LibreRiODETRv2l-obb-diorr/resolve/{R.HF_REVISIONS['LibreRiODETRv2l-obb-diorr']}"
        "/LibreRiODETRv2l-obb-diorr.pt"
    )
    assert len(R.HF_REVISIONS) == 10
    assert all(len(rev) == 40 for rev in R.HF_REVISIONS.values())
    # OBB is the only task, and its suffix is mandatory.
    assert R.detect_size_from_filename("LibreRiODETRv2n.pt") is None
    assert R.default_checkpoint_names(15)[0] == "plane"
    assert R.default_checkpoint_names(20)[19] == "windmill"


def test_recognition_size_and_classes(rio_n):
    from libreyolo.models.riodetrv2.model import LibreRiODETRv2 as R

    state = rio_n.model.state_dict()
    assert R.can_load(state)
    assert R.detect_size(state) == "n"
    assert R.detect_nb_classes(state) == 15
    assert R.detect_checkpoint_task(state) == "obb"


@pytest.mark.parametrize(
    ("hidden", "stem", "expected"),
    [(128, 16, "n"), (224, 16, "s"), (256, 24, "m"), (256, 32, "l"), (384, 32, "x")],
)
def test_detect_size_all_sizes(hidden, stem, expected):
    from libreyolo.models.riodetrv2.model import LibreRiODETRv2 as R

    state = {
        "encoder.input_proj.0.0.weight": torch.empty(hidden, 1, 1, 1),
        "backbone.stem.stem1.conv.weight": torch.empty(stem, 3, 3, 3),
    }
    assert R.detect_size(state) == expected


@pytest.mark.parametrize(
    ("module", "cls", "size", "kwargs"),
    [
        ("dfine", "LibreDFINE", "n", {}),
        ("deim", "LibreDEIM", "n", {}),
        ("rtdetrv4", "LibreRTDETRv4", "s", {}),
        ("rtdetr", "LibreRTDETR", "r18", {}),
        ("rtdetrv2", "LibreRTDETRv2", "n", {"task": "obb", "nb_classes": 15}),
    ],
)
def test_sibling_routing_is_bidirectional(rio_n, module, cls, size, kwargs):
    import importlib

    from libreyolo.models.riodetrv2.model import LibreRiODETRv2 as R

    sibling = getattr(importlib.import_module(f"libreyolo.models.{module}.model"), cls)
    sibling_state = sibling(None, size=size, device="cpu", **kwargs).model.state_dict()
    rio_state = rio_n.model.state_dict()
    assert not R.can_load(sibling_state)
    assert R.convert_upstream_state_dict(sibling_state) is None
    assert not sibling.can_load(rio_state)
    assert not sibling.can_load(_as_upstream(rio_state))


def test_upstream_layout_converts_and_loads_strictly(rio_n):
    from libreyolo.models.riodetrv2.convert import (
        convert_upstream,
        is_upstream_state_dict,
    )

    native = rio_n.model.state_dict()
    upstream = _as_upstream(native)
    assert is_upstream_state_dict(upstream)
    assert not is_upstream_state_dict(native)
    converted = convert_upstream(upstream)
    assert set(converted) == set(native)
    fresh = _scratch("n")
    fresh.model.load_state_dict(converted, strict=True)


def test_checkpoint_anchors_select_the_native_canvas():
    model = _scratch("n", nb_classes=20)
    assert model.input_size == 1024
    state = model.model.state_dict()
    # DIOR-R checkpoints save anchors for an 800 px canvas (50^2 + 25^2).
    state["decoder.anchors"] = torch.zeros(1, 3125, 4)
    state["decoder.valid_mask"] = torch.ones(1, 3125, 1, dtype=torch.bool)
    model._prepare_model_for_state_dict(state)
    assert model.input_size == 800
    assert tuple(model.model.decoder.anchors.shape) == (1, 3125, 4)
    assert model.model.encoder.pos_embed1.shape[1] == (800 // 32) ** 2
    model.model.load_state_dict(state, strict=True)


def test_forward_shapes_off_canvas(rio_n):
    with torch.no_grad():
        out = rio_n.model.eval()(torch.rand(1, 3, 256, 256))
    assert out["pred_logits"].shape == (1, 300, 15)
    assert out["pred_boxes"].shape == (1, 300, 5)
    assert torch.all((out["pred_boxes"][..., 4] >= 0) & (out["pred_boxes"][..., 4] < 1))


def test_angle_decoders():
    from libreyolo.models.riodetrv2.decoder import (
        decode_circular_angle_delta,
        decode_linear_angle_delta,
        wrap_angle_01,
    )

    logits = torch.full((1, 36), -1e4)
    logits[0, 9] = 0.0  # bin 9 of 36 is a quarter turn of the residual circle
    assert decode_circular_angle_delta(logits).item() == pytest.approx(0.25, abs=1e-6)
    logits = torch.full((1, 36), -1e4)
    logits[0, -1] = 0.0
    assert decode_linear_angle_delta(logits, 0.5 / 1.5).item() == pytest.approx(1 / 3)
    assert torch.allclose(wrap_angle_01(torch.tensor([-0.25, 1.25])), torch.tensor([0.75, 0.25]))


def test_local_frame_decode_rotates_the_center_offset():
    from libreyolo.models.riodetrv2.decoder import distance2obb_local

    ref = torch.tensor([[0.5, 0.5, 0.2, 0.2, 0.0]])
    reg_scale = torch.tensor([4.0])
    # right edge pushed out by one unit -> center moves along the box x-axis.
    distance = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    quarter_turn = torch.tensor([[0.5]])  # angle / pi
    xywh = distance2obb_local(ref, distance, quarter_turn, reg_scale)[0]
    shift = 0.5 * 1.0 * 0.2 / 4.0
    assert xywh[0].item() == pytest.approx(0.5, abs=1e-6)
    assert xywh[1].item() == pytest.approx(0.5 + shift, abs=1e-6)
    assert xywh[2].item() == pytest.approx(0.2 + 0.2 / 4.0)


def test_postprocess_float32_scale_is_opt_in():
    from libreyolo.postprocess.rtdetr import postprocess_obb

    torch.manual_seed(0)
    output = {"pred_logits": torch.randn(1, 300, 15), "pred_boxes": torch.rand(1, 300, 5)}
    a = postprocess_obb(output, 0.0, 0.5, (3840, 2560), input_size=800)
    b = postprocess_obb(output, 0.0, 0.5, (3840, 2560), input_size=800, float32_scale=True)
    assert torch.equal(a["scores"], b["scores"]) and torch.equal(a["classes"], b["classes"])
    assert torch.allclose(a["obb"], b["obb"], atol=1e-3)
    assert torch.equal(a["obb"][:, 4], b["obb"][:, 4])
    assert math.isclose(float((a["obb"][:, :4] - b["obb"][:, :4]).abs().max()), 0.0, abs_tol=1e-3)


def test_training_is_explicitly_out_of_scope(rio_n):
    with pytest.raises(NotImplementedError, match="inference-only"):
        rio_n.train(data="unused.yaml")


def _assert_raw_close(actual_logits, actual_boxes, expected):
    """Per-query agreement, tolerant of top-k tie-breaks in an untrained head."""
    exp_logits, exp_boxes = expected["pred_logits"], expected["pred_boxes"]
    assert actual_logits.shape == exp_logits.shape
    assert actual_boxes.shape == exp_boxes.shape
    per_query = torch.maximum(
        (actual_logits - exp_logits).abs().amax(dim=-1),
        (actual_boxes - exp_boxes).abs().amax(dim=-1),
    )
    assert int((per_query > 1e-4).sum()) <= 3, per_query.max()
    torch.testing.assert_close(
        actual_logits.flatten().sort().values, exp_logits.flatten().sort().values,
        rtol=0, atol=1e-4,
    )


@pytest.mark.parametrize("canvas", [1024, 800])
@pytest.mark.parametrize("fmt", ["onnx", "torchscript"])
def test_export_raw_roundtrip_and_metadata(tmp_path, fmt, canvas):
    if fmt == "onnx":
        ort = pytest.importorskip("onnxruntime")
    from libreyolo import LibreYOLO

    torch.manual_seed(1234)
    nc = 15 if canvas == 1024 else 20
    model = _scratch("n", nb_classes=nc)
    if canvas != 1024:
        model.model.set_eval_spatial_size((canvas, canvas))
        model.input_size = canvas
    # The zero-initialised angle head gives uniform angle logits, whose
    # circular mean is atan2(~0, ~0): numerically arbitrary across backends.
    # Trained heads never sit there; give the test model a non-degenerate one.
    with torch.no_grad():
        for head in model.model.decoder.dec_angle_head:
            head.layers[-1].weight.normal_(0.0, 0.05)
            head.layers[-1].bias.normal_(0.0, 1.0)
    model.model.eval()
    stem = "LibreRiODETRv2n-obb" + ("" if canvas == 1024 else "-diorr")
    output = tmp_path / f"{stem}.{fmt}"
    model.export(fmt, imgsz=canvas, output_path=str(output))

    image = torch.randn(1, 3, canvas, canvas, generator=torch.Generator().manual_seed(8))
    with torch.inference_mode():
        expected = model.model(image)
    if fmt == "onnx":
        # An untrained graph amplifies ONNX Runtime's reduction-order noise and
        # reorders near-tied queries, so random weights only pin the contract
        # here; numeric ONNX parity is recorded on the released checkpoints.
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        logits, boxes = session.run(None, {name: image.numpy()})
        assert logits.shape == tuple(expected["pred_logits"].shape)
        assert boxes.shape == tuple(expected["pred_boxes"].shape) and boxes.shape[-1] == 5
        assert np.isfinite(logits).all() and np.isfinite(boxes).all()
        zeros = session.run(None, {name: np.zeros_like(image.numpy())})
        assert np.abs(logits - zeros[0]).max() > 1e-5
    else:
        with torch.inference_mode():
            logits, boxes = torch.jit.load(str(output)).eval()(image)
        _assert_raw_close(logits, boxes, expected)

    backend = LibreYOLO(str(output), device="cpu")
    assert (backend.model_family, backend.model_size, backend.task) == ("riodetrv2", "n", "obb")
    assert len(backend.names) == nc


@pytest.mark.torchscript
def test_export_at_non_native_canvas_keeps_native_model_intact(tmp_path):
    from libreyolo import LibreYOLO

    model = _scratch("n")
    model.model.eval()
    output = tmp_path / "LibreRiODETRv2n-obb.torchscript"
    model.export("torchscript", imgsz=640, output_path=str(output))
    assert tuple(model.model.eval_spatial_size) == (1024, 1024)
    traced = torch.jit.load(str(output)).eval()
    with torch.inference_mode():
        logits, boxes = traced(torch.rand(1, 3, 640, 640))
    assert logits.shape == (1, 300, 15) and boxes.shape == (1, 300, 5)
    assert LibreYOLO(str(output), device="cpu").task == "obb"

"""Offline tests for the native PicoSAM3 port."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from libreyolo.models.picosam3.model import LibrePicoSAM3
from libreyolo.models.picosam3.nn import PicoSAM3Network
from libreyolo.models.picosam3.preprocess import padded_square_roi, preprocess_roi
from libreyolo.models.sam.model import _ALIASES, _PICO_SAM3

pytestmark = [pytest.mark.unit, pytest.mark.sam]


def _bare_picosam3() -> LibrePicoSAM3:
    model = object.__new__(LibrePicoSAM3)
    model.model = PicoSAM3Network().eval()
    for parameter in model.model.parameters():
        parameter.data.zero_()
    model.model.refine[3].bias.data.fill_(1.0)
    model.processor = None
    model.device = torch.device("cpu")
    model._model_dtype = torch.float32
    model._default_multimask = False
    model.names = {0: "object"}
    model.size = "pico"
    model.task = "segment"
    model._clear_image_state()
    return model


def test_top_level_export_resolves_lazily():
    from libreyolo import LibrePicoSAM3 as Exported

    assert Exported is LibrePicoSAM3


def test_aliases_resolve_without_download():
    assert _ALIASES["picosam3"] == (_PICO_SAM3, "pico")
    assert _ALIASES["picosam3-pico"] == (_PICO_SAM3, "pico")


def test_network_shape_and_parameter_count():
    model = PicoSAM3Network().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_371_418
    with torch.no_grad():
        output = model(torch.zeros((2, 3, 96, 96)))
    assert output.shape == (2, 1, 96, 96)


def test_upstream_roi_geometry_and_preprocess_are_deterministic():
    image = Image.new("RGB", (32, 24), color=(10, 20, 30))
    roi = padded_square_roi([8, 6, 24, 18], 32, 24)
    tensor = preprocess_roi(image, roi)

    assert roi == (6, 2, 25, 21)
    assert tensor.shape == (3, 96, 96)
    assert tensor[:, 0, 0].tolist() == pytest.approx(
        [-1.9466564655, -1.6855741739, -1.2815686464]
    )


def test_box_prompt_places_mask_back_into_original_image():
    model = _bare_picosam3()
    image = Image.new("RGB", (32, 24), color="white")

    result = model.predict(image, bboxes=[8, 6, 24, 18], conf=0.0)

    assert result.masks.data.shape == (1, 24, 32)
    assert result.masks.data[0, 2:21, 6:25].all()
    assert not result.masks.data[0, :2].any()
    assert result.boxes.xyxy.tolist() == [[6.0, 2.0, 24.0, 20.0]]


def test_multiple_boxes_are_batched_and_set_image_is_reused():
    model = _bare_picosam3().set_image(Image.new("RGB", (40, 30), color="white"))
    try:
        result = model.predict(bboxes=[[2, 2, 15, 20], [20, 5, 35, 25]])
        assert len(result.masks) == 2
    finally:
        model.reset_image()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"points": [4, 4]}, "supports only bboxes"),
        ({"text": "dog"}, "supports only bboxes"),
        ({"masks": torch.zeros(1)}, "supports only bboxes"),
        ({"multimask": True}, "does not support multimask"),
        ({}, "requires bboxes"),
    ],
)
def test_unsupported_prompt_contract_fails_clearly(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _bare_picosam3().predict(Image.new("RGB", (16, 16)), **kwargs)


def test_epoch1_picosam2_checkpoint_is_rejected(tmp_path):
    checkpoint = {"output_head.weight": torch.zeros((1, 40, 1, 1))}
    torch.save(checkpoint, tmp_path / LibrePicoSAM3.WEIGHT_FILE)
    model = object.__new__(LibrePicoSAM3)
    model.size = "pico"
    model._ensure_weights = lambda: str(tmp_path)

    with pytest.raises(ValueError, match="older PicoSAM2 architecture"):
        model._init_model()


def test_conversion_round_trip_writes_provenance_metadata(tmp_path):
    from weights.convert_picosam3_weights import convert

    source = tmp_path / "PicoSAM3_SAM3_student_best.pt"
    output = tmp_path / "LibrePicoSAM3pico.pt"
    torch.save(PicoSAM3Network().state_dict(), source)
    convert(source, output)
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)

    assert checkpoint["model_family"] == "picosam3"
    assert checkpoint["size"] == "pico"
    assert checkpoint["task"] == "segment"
    assert checkpoint["imgsz"] == 96
    assert checkpoint["license"] == "Apache-2.0"
    PicoSAM3Network().load_state_dict(checkpoint["model"], strict=True)


def test_raw_onnx_export_matches_pytorch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    model = _bare_picosam3()
    output = tmp_path / "picosam3.onnx"
    model.export(output=output)

    inputs = torch.randn((3, 3, 96, 96))
    with torch.no_grad():
        expected = model.model(inputs).numpy()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(["mask_logits"], {"roi_image": inputs.numpy()})[0]
    torch.testing.assert_close(
        torch.from_numpy(actual), torch.from_numpy(expected), atol=1e-5, rtol=1e-5
    )


def _roi_dataset(tmp_path):
    import numpy as np

    root = tmp_path / "rois"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(3):
        pixels = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / "train" / f"{index}.jpg")
        (root / "labels" / "train" / f"{index}.txt").write_text(
            "0 0.5 0.5 0.4 0.5\n1 0.3 0.4 0.2 0.3\n2 0.7 0.6 0.3 0.2\n"
        )
    data = tmp_path / "rois.yaml"
    data.write_text(
        f"path: {root}\ntrain: images/train\nval: images/train\nnames:\n  0: a\n  1: b\n  2: c\n"
    )
    return data


def test_axelera_and_onnx_exports_share_the_raw_roi_contract(tmp_path, monkeypatch):
    """format="axelera" compiles the same raw graph as format="onnx" (static batch 1):
    roi_image [1,3,96,96] -> mask_logits [1,1,96,96], calibrated on predict()'s own ROIs."""
    onnx = pytest.importorskip("onnx")
    import numpy as np

    import libreyolo.export.axelera as axelera

    model = _bare_picosam3()
    recorded = {}

    def recorder(**kwargs):
        recorded.update(kwargs)
        recorded["graph"] = onnx.load(kwargs["onnx_path"])
        return kwargs["output_path"]

    monkeypatch.setattr(axelera, "export_axelera", recorder)
    data = _roi_dataset(tmp_path)
    out = model.export("axelera", data=data, output_path=tmp_path / "pico_axelera_model")
    assert out == str(tmp_path / "pico_axelera_model")

    reference = onnx.load(model.export(output=tmp_path / "pico.onnx", dynamic=False))
    ort = pytest.importorskip("onnxruntime")

    def signature(graph):
        def dims(value):
            return [d.dim_value for d in value.type.tensor_type.shape.dim]

        return (
            [(v.name, dims(v)) for v in graph.graph.input],
            [(v.name, dims(v)) for v in graph.graph.output],
        )

    assert signature(recorded["graph"]) == signature(reference)
    assert signature(reference) == ([("roi_image", [1, 3, 96, 96])], [("mask_logits", [1, 1, 96, 96])])
    # Same function: the Axelera graph only differs by the exact dense form of the stem's depthwise conv.
    with torch.no_grad():
        for layer in model.model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.normal_(0, 0.2)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                layer.weight.fill_(1.0)
    recorded.clear()
    model.export("axelera", data=data, output_path=tmp_path / "pico_axelera_model")
    roi = np.random.default_rng(1).random((1, 3, 96, 96), dtype=np.float32)
    run = lambda graph: ort.InferenceSession(graph.SerializeToString(), providers=["CPUExecutionProvider"]).run(
        ["mask_logits"], {"roi_image": roi}
    )[0]
    np.testing.assert_allclose(
        run(recorded["graph"]), run(onnx.load(model.export(output=tmp_path / "pico2.onnx", dynamic=False))), atol=1e-5
    )
    assert isinstance(model.model.encoder_stage1[0], torch.nn.Conv2d) and model.model.encoder_stage1[0].groups == 3
    assert "Python" in recorded["metadata"]["contract"]
    assert recorded["metadata"]["input_names"] == ["roi_image"]
    calibration = recorded["calibration"]
    assert len(calibration) == 6  # two ROIs per image, three images
    assert all(c.shape == (1, 3, 96, 96) and c.dtype == np.float32 for c in calibration)
    # Normalized exactly as predict() normalizes an ROI (checked against preprocess_roi below).
    image = Image.open(sorted((tmp_path / "rois" / "images" / "train").glob("*.jpg"))[0]).convert("RGB")
    box = [0.3 * 160, 0.25 * 120, 0.7 * 160, 0.75 * 120]
    expected = preprocess_roi(image, padded_square_roi(box, 160, 120)).numpy()[None]
    np.testing.assert_array_equal(calibration[0], expected)
    assert recorded["quantization_scheme"] == LibrePicoSAM3.AXELERA_QUANTIZATION_SCHEME


@pytest.mark.parametrize(
    "kwargs",
    [{"data": "coco128.yaml"}, {"quantization_scheme": "per_tensor_min_max"}, {"verify": True}, {"fraction": 0.5}],
)
def test_onnx_export_rejects_axelera_only_arguments(tmp_path, kwargs):
    with pytest.raises(TypeError, match="Axelera-only"):
        _bare_picosam3().export(output=tmp_path / "pico.onnx", **kwargs)


def test_axelera_export_is_batch_one(tmp_path):
    with pytest.raises(ValueError, match="batch 1"):
        _bare_picosam3().export("axelera", dynamic=True, output_path=tmp_path / "x_axelera_model")

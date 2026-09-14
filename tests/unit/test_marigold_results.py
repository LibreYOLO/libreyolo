"""Numeric contracts needed for Marigold V2 without any model dependencies."""

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import AlbedoMap, DepthMap, NormalMap, Results
from libreyolo.data.albedo_dataset import AlbedoDataset, albedo_collate_fn
from libreyolo.tasks import detect_task_suffix, normalize_task
from libreyolo.validation.albedo_validator import AlbedoValidator
from libreyolo.validation.depth_validator import align_depth_prediction

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("encoding", ["inverse_depth", "depth", "log_depth"])
def test_depth_encoding_survives_result_transforms(encoding):
    result = Results(
        None,
        (2, 3),
        depth_map=DepthMap(torch.arange(6).reshape(2, 3), encoding=encoding),
    )
    for converted in (result.cpu(), result.numpy(), result.to("cpu"), result[0]):
        assert converted.depth_map.encoding == encoding
        np.testing.assert_array_equal(
            converted.depth_map.numpy().data, [[0, 1, 2], [3, 4, 5]]
        )
    assert result.depth_map.near_is_high == (encoding == "inverse_depth")
    if encoding != "inverse_depth":
        assert result.summary()[0]["encoding"] == encoding


@pytest.mark.parametrize("encoding", ["inverse_depth", "depth", "log_depth"])
def test_alignment_occurs_before_depth_decoding(encoding):
    gt = torch.tensor([0.25, 0.5, 1, 2, 4, 16], dtype=torch.float64)
    space = {"inverse_depth": 1 / gt, "depth": gt, "log_depth": gt.log()}[encoding]
    pred = (space - 3.5) / 2.75
    torch.testing.assert_close(align_depth_prediction(pred, gt, encoding=encoding), gt)
    wrong = align_depth_prediction(-pred, gt, encoding=encoding)
    assert not torch.allclose(wrong, gt)


def test_depth_rendering_keeps_near_far_colours_consistent():
    values = np.array([[1, 2], [3, 4]], np.float32)
    inverse = Results(None, (2, 2), depth_map=DepthMap(-values))
    log = Results(None, (2, 2), depth_map=DepthMap(values, encoding="log_depth"))
    np.testing.assert_array_equal(np.asarray(inverse.plot()), np.asarray(log.plot()))


def test_albedo_is_linear_float_data_and_display_is_srgb(tmp_path):
    data = np.array([[[0, 0.0031308, 0.5], [1, 0.25, 0.75]]], np.float32)
    result = Results(None, (1, 2), albedo=AlbedoMap(data))
    assert len(result) == 1
    assert result.summary() == [
        {"name": "albedo", "shape": [1, 2, 3], "color_space": "linear_rgb"}
    ]
    for converted in (result.cpu(), result.numpy(), result.to("cpu"), result[0]):
        np.testing.assert_array_equal(converted.albedo.numpy().data, data)
    np.testing.assert_array_equal(result.albedo.to_rgb()[0, 0], [0, 10, 188])
    path = tmp_path / "albedo.png"
    result.albedo.save(path)
    with Image.open(path) as rendered:
        np.testing.assert_array_equal(np.asarray(rendered), result.albedo.to_rgb())
    assert normalize_task("albedo-estimation") == "albedo"
    assert detect_task_suffix("LibreMarigoldV2b-albedo.pt") == "albedo"


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_albedo_rejects_invalid_reflectance(value):
    with pytest.raises(ValueError, match="finite"):
        AlbedoMap(np.full((2, 3, 3), value))


def test_ui_identifies_dense_payloads():
    from libreyolo.ui.server import _summarize_result

    normal = NormalMap(np.tile([0, 0, -1], (2, 3, 1)))
    assert _summarize_result(Results(None, (2, 3), normal_map=normal)) == ("normal", "normal map")
    albedo = AlbedoMap(np.full((2, 3, 3), 0.5))
    assert _summarize_result(Results(None, (2, 3), albedo=albedo)) == ("albedo", "albedo map")


def test_albedo_dataset_preserves_linear_values_and_metrics(tmp_path):
    images = tmp_path / "images/val"
    targets = tmp_path / "albedo/val"
    images.mkdir(parents=True)
    targets.mkdir(parents=True)
    Image.new("RGB", (16, 16), (120, 80, 30)).save(images / "sample.png")
    expected = np.full((16, 16, 3), 0.123456, np.float32)
    np.save(targets / "sample.npy", expected)
    dataset = AlbedoDataset({"val": str(images)}, "val", 16)
    batch = albedo_collate_fn([dataset[0]])
    torch.testing.assert_close(batch[1], torch.full((1, 3, 16, 16), 0.123456))
    validator = object.__new__(AlbedoValidator)
    validator._init_metrics()
    prediction = validator._postprocess_predictions({"albedo": batch[1]}, batch)
    validator._update_metrics(prediction, batch[1], batch[2])
    metrics = validator._compute_metrics()
    assert metrics["metrics/PSNR"] == 100.0
    assert metrics["metrics/SSIM"] == pytest.approx(1.0)
    np.save(targets / "sample.npy", np.zeros((8, 16, 3), np.float32))
    with pytest.raises(ValueError, match="shape"):
        dataset[0]

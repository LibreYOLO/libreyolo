"""Camera geometry and native FCOS3D inference contracts; no external bytes."""

import math

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.fcos3d import LibreFCOS3D
from libreyolo.models.fcos3d.nn import ModulatedConv
from libreyolo.models.fcos3d.utils import calibration, decode, payloads, preprocess

pytestmark = pytest.mark.unit
K = np.array([[100, 0, 4], [0, 100, 4], [0, 0, 1]], np.float32)


def raw_prediction():
    # One stride-eight point (4,4), offset (0,0), depth 10, xyz dimensions 4,2,6.
    return (
        [torch.tensor([10.0] + [-100.0] * 9).reshape(1, 10, 1, 1)],
        [
            torch.tensor([0.0, 0.0, 10.0, 4.0, 2.0, 6.0, 0.0, 0.0, 0.0]).reshape(
                1, 9, 1, 1
            )
        ],
        [torch.tensor([10.0, -10.0]).reshape(1, 2, 1, 1)],
        [torch.zeros(1, 9, 1, 1)],
        [torch.full((1, 1, 1, 1), 10.0)],
    )


def test_preprocess_caffe_channels_padding_and_no_resize():
    image = Image.new("RGB", (33, 17), (10, 20, 30))
    x = preprocess(image)
    assert x.shape == (1, 3, 32, 64)
    torch.testing.assert_close(
        x[0, :, 0, 0], torch.tensor([30 - 103.530, 20 - 116.280, 10 - 123.675])
    )
    assert torch.count_nonzero(x[:, :, 17:]) == 0
    assert torch.count_nonzero(x[:, :, :, 33:]) == 0


def test_deformable_zero_offset_half_mask():
    torch.manual_seed(1)
    conv = ModulatedConv(2, bias=True)
    x = torch.randn(1, 2, 5, 7)
    expected = (
        torch.nn.functional.conv2d(x, conv.weight, None, padding=1) * 0.5
        + conv.bias[None, :, None, None]
    )
    torch.testing.assert_close(conv(x), expected)


def test_decode_projected_center_and_direction():
    raw = raw_prediction()
    boxes, scores, labels = decode(raw, K)
    np.testing.assert_allclose(boxes[0, :6], [0, 0, 10, 4, 2, 6])
    # Local yaw 0 is placed in the [pi/4, pi+pi/4) bin: yaw becomes pi.
    assert boxes[0, 6] == pytest.approx(math.pi)
    assert labels.tolist() == [0]
    assert scores[0] == pytest.approx(torch.sigmoid(torch.tensor(10.0)).item() ** 2)
    raw[1][0][:, 0] = -10
    raw[2][0][:, 0] = -10
    raw[2][0][:, 1] = 10
    moved, _, _ = decode(raw, K)
    np.testing.assert_allclose(moved[0, :3], [1, 0, 10])
    assert moved[0, 6] == pytest.approx(2 * math.pi + math.atan(0.1))


def test_rotated_nms_suppresses_same_class_not_other_class():
    raw = raw_prediction()
    raw = tuple([torch.cat([group[0], group[0]], -1)] for group in raw)
    # Point 2 is x=12, so offset eight maps it onto the same 3D center.
    raw[1][0][0, 0, 0, 1] = 8
    raw[0][0][0, 1, 0, 1] = 9
    boxes, _, labels = decode(raw, K)
    assert len(boxes) == 2
    assert labels.tolist() == [0, 1]
    assert len(decode(raw, K, max_det=1)[0]) == 1
    assert len(decode(raw, K, conf=1)[0]) == 0


def test_cuboid_axes_projection_alignment_and_slicing():
    boxes = np.array([[0, 0, 10, 4, 2, 6, math.pi / 2]], np.float32)
    b, cuboids = payloads(boxes, np.array([0.8]), np.array([2]), K, (100, 100))
    assert not b.is_track
    assert b.orig_shape == (100, 100)
    np.testing.assert_allclose(cuboids.dimensions, [[6, 4, 2]])
    extent = np.ptp(cuboids.corners[0], axis=0)
    np.testing.assert_allclose(extent, [6, 2, 4], atol=1e-6)
    np.testing.assert_allclose(cuboids[0].intrinsics, K)
    np.testing.assert_allclose(cuboids.conf, b.conf)
    assert np.isfinite(b.xyxy).all()


@pytest.mark.parametrize(
    "k",
    [
        np.eye(4),
        np.zeros((3, 3)),
        [[1, 0.1, 0], [0, 1, 0], [0, 0, 1]],
        [[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]],
    ],
)
def test_bad_calibration(k):
    with pytest.raises(ValueError, match="intrinsics"):
        calibration(k)


def test_single_list_stream_and_validation_without_weights(tmp_path):
    class Network:
        def __call__(self, x):
            return raw_prediction()

    model = LibreFCOS3D.__new__(LibreFCOS3D)
    model.model = Network()
    model.device = torch.device("cpu")
    model.names = {0: "car"}
    image = Image.new("RGB", (32, 32))
    result = model(image, intrinsics=K, save=True, output_path=tmp_path / "out.png")
    assert result.boxes3d.data.shape == (1, 14)
    assert (tmp_path / "out.png").is_file()
    assert len(model([image, image], intrinsics=K)) == 2
    assert len(list(model([image], intrinsics=K, stream=True))) == 1
    for kw in (
        {"conf": float("nan")},
        {"iou": -1},
        {"max_det": 0},
        {"output_path": "x.png"},
    ):
        with pytest.raises(ValueError):
            model(image, intrinsics=K, **kw)
    for method in ("train", "val", "export", "track"):
        with pytest.raises(NotImplementedError):
            getattr(model, method)()

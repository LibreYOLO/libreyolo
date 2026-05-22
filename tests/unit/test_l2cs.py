"""Unit tests for LibreL2CS gaze inference.

Smoke tests only — they verify that the L2CS network builds at every size,
the bin-expectation decode produces sensible angles, the ``Gaze`` payload
round-trips through ``Results``, and end-to-end inference with a mocked
face detector returns a ``Results`` carrying both face boxes and gaze
angles. Numerical parity vs upstream is out of scope here — the published
weights aren't downloaded as part of the unit tier.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreL2CS
from libreyolo.models.l2cs.face import (
    CallableFaceDetector,
    FaceBox,
    resolve_face_detector,
)
from libreyolo.models.l2cs.nn import build_l2cs, detect_size_from_state_dict
from libreyolo.models.l2cs.utils import bin_logits_to_angles, preprocess_face_crops
from libreyolo.utils.results import Gaze, Results


pytestmark = pytest.mark.unit


SIZES = ["r18", "r34", "r50", "r101", "r152"]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_build_and_forward(size):
    """L2CS builds at every ResNet depth and produces the (B, 90) head shape."""
    model = build_l2cs(size, num_bins=90).eval()
    x = torch.zeros(2, 3, 448, 448)
    with torch.no_grad():
        yaw, pitch = model(x)
    assert yaw.shape == (2, 90)
    assert pitch.shape == (2, 90)


@pytest.mark.parametrize("size", SIZES)
def test_detect_size_from_state_dict_roundtrip(size):
    """State-dict fingerprinting recovers the size code we built with."""
    model = build_l2cs(size, num_bins=90)
    detected = detect_size_from_state_dict(model.state_dict())
    assert detected == size


# ---------------------------------------------------------------------------
# Bin decode
# ---------------------------------------------------------------------------


def test_bin_decode_zero_yields_offset():
    """Uniform logits put the expectation at the midpoint bin (89/2 ≈ 44.5).

    With offset=-180 and bin_width=4, the expected degree is
    44.5 * 4 - 180 = -2.0°, which is what the upstream pipeline produces
    for an unbiased softmax.
    """
    logits = torch.zeros(1, 90)
    angles = bin_logits_to_angles(logits, logits, num_bins=90)
    expected_deg = (89 / 2.0) * 4.0 - 180.0
    expected_rad = expected_deg * math.pi / 180.0
    assert torch.allclose(angles, torch.tensor([[expected_rad, expected_rad]]), atol=1e-6)


def test_bin_decode_one_hot_picks_bin():
    """A one-hot bin at index k decodes to ``k * 4 - 180`` degrees."""
    logits = torch.full((1, 90), -1e9)
    logits[0, 45] = 0  # pick bin 45 → 0°
    angles = bin_logits_to_angles(logits, logits, num_bins=90)
    assert torch.allclose(angles, torch.zeros(1, 2), atol=1e-5)


def test_bin_decode_batch():
    """Decode preserves the batch dimension."""
    logits = torch.zeros(4, 90)
    angles = bin_logits_to_angles(logits, logits, num_bins=90)
    assert angles.shape == (4, 2)


# ---------------------------------------------------------------------------
# Gaze result payload
# ---------------------------------------------------------------------------


def test_gaze_payload_basic():
    """Gaze exposes pitch / yaw and degree conversions."""
    data = torch.tensor([[math.pi / 6, -math.pi / 4]])
    g = Gaze(data)
    assert g.pitch.item() == pytest.approx(math.pi / 6)
    assert g.yaw.item() == pytest.approx(-math.pi / 4)
    assert g.pitch_deg.item() == pytest.approx(30.0)
    assert g.yaw_deg.item() == pytest.approx(-45.0)


def test_gaze_direction_3d_unit_norm():
    """3D direction vectors should be unit-length."""
    angles = torch.tensor([
        [0.0, 0.0],
        [math.pi / 4, math.pi / 6],
        [-math.pi / 6, math.pi / 3],
    ])
    vecs = Gaze(angles).direction_3d
    norms = torch.linalg.vector_norm(vecs, dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-6)


def test_gaze_device_roundtrip():
    """Gaze inherits _TensorPayload's to/cpu/numpy semantics."""
    g = Gaze(torch.tensor([[0.1, -0.2]]))
    g_np = g.numpy()
    assert isinstance(g_np.data, np.ndarray)
    g_back = Gaze(torch.as_tensor(g_np.data))
    assert torch.allclose(g_back.data, g.data, atol=1e-6)


def test_gaze_indexing():
    """Gaze supports row indexing like other result payloads."""
    g = Gaze(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    first = g[0]
    assert first.data.shape == (1, 2)
    assert first.pitch.item() == pytest.approx(0.1)


def test_results_carries_gaze_slot():
    """Results accepts a Gaze and exposes it via _keys."""
    from libreyolo.utils.results import Boxes

    boxes = Boxes(
        torch.zeros((1, 4)),
        torch.tensor([0.9]),
        torch.tensor([0.0]),
    )
    g = Gaze(torch.tensor([[0.1, -0.2]]))
    r = Results(boxes=boxes, orig_shape=(100, 100), gaze=g, names={0: "face"})
    assert r.gaze is g
    assert "gaze" in r._keys


# ---------------------------------------------------------------------------
# End-to-end with a mocked face detector
# ---------------------------------------------------------------------------


def _make_dummy_state_dict(size: str) -> dict:
    """Build a state dict for an L2CS model of given size, all-zero weights."""
    return build_l2cs(size, num_bins=90).state_dict()


def test_libre_l2cs_end_to_end_with_byo_bbox(tmp_path):
    """End-to-end: BYO bbox path, untrained weights, returns Gaze."""
    # Build and save dummy weights at r18 (smallest for speed)
    sd = _make_dummy_state_dict("r18")
    weights_path = tmp_path / "LibreL2CSr18.pt"
    torch.save(sd, weights_path)

    model = LibreL2CS(str(weights_path), size="r18", device="cpu")
    assert model.task == "gaze"
    assert model.names == {0: "face"}

    # 64x64 dummy image, full-frame face box
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    result = model(img, face_boxes=[(0, 0, 64, 64)])

    assert isinstance(result, Results)
    assert result.gaze is not None
    assert len(result.gaze) == 1
    assert result.gaze.data.shape == (1, 2)
    assert torch.isfinite(result.gaze.data).all()


def test_libre_l2cs_callable_face_detector(tmp_path):
    """A user-supplied callable becomes the face detector."""
    sd = _make_dummy_state_dict("r18")
    weights_path = tmp_path / "LibreL2CSr18.pt"
    torch.save(sd, weights_path)

    model = LibreL2CS(str(weights_path), size="r18", device="cpu")

    calls = {"n": 0}

    def fake_detector(image_rgb):
        calls["n"] += 1
        h, w = image_rgb.shape[:2]
        return [FaceBox(xyxy=(0, 0, w, h), score=0.99)]

    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    result = model(img, face_detector=fake_detector)
    assert calls["n"] == 1
    assert len(result.gaze) == 1
    assert isinstance(model.face_detector, type(None))  # not cached on instance


def test_libre_l2cs_no_face_raises(tmp_path):
    """No face_boxes and no face_detector → clear error, not a silent crash."""
    sd = _make_dummy_state_dict("r18")
    weights_path = tmp_path / "LibreL2CSr18.pt"
    torch.save(sd, weights_path)

    model = LibreL2CS(str(weights_path), size="r18", device="cpu")
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="no face source"):
        model(img)


def test_libre_l2cs_no_faces_returns_empty(tmp_path):
    """When the detector returns no faces, Results is empty but well-formed."""
    sd = _make_dummy_state_dict("r18")
    weights_path = tmp_path / "LibreL2CSr18.pt"
    torch.save(sd, weights_path)

    model = LibreL2CS(
        str(weights_path),
        size="r18",
        device="cpu",
        face_detector=CallableFaceDetector(fn=lambda img: []),
    )
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    result = model(img)
    assert len(result.boxes) == 0
    assert len(result.gaze) == 0


def test_libre_l2cs_rejects_augment_and_tiling(tmp_path):
    """TTA and tiling are nonsensical for gaze and should fail explicitly."""
    sd = _make_dummy_state_dict("r18")
    weights_path = tmp_path / "LibreL2CSr18.pt"
    torch.save(sd, weights_path)

    model = LibreL2CS(str(weights_path), size="r18", device="cpu")
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="augment"):
        model(img, face_boxes=[(0, 0, 64, 64)], augment=True)
    with pytest.raises(ValueError, match="[Tt]il"):
        model(img, face_boxes=[(0, 0, 64, 64)], tiling=True)


def test_libre_l2cs_blocks_train_val_export(tmp_path):
    """train/val/non-onnx export raise NotImplementedError with helpful messages."""
    sd = _make_dummy_state_dict("r18")
    weights_path = tmp_path / "LibreL2CSr18.pt"
    torch.save(sd, weights_path)

    model = LibreL2CS(str(weights_path), size="r18", device="cpu")
    with pytest.raises(NotImplementedError, match="upstream"):
        model.train()
    with pytest.raises(NotImplementedError, match="upstream"):
        model.val()
    with pytest.raises(NotImplementedError):
        model.export("torchscript")


# ---------------------------------------------------------------------------
# Regression tests for code-review findings
# ---------------------------------------------------------------------------


def test_preprocess_multiface_uniform_shapes():
    """Finding #1: crops of differing aspect ratio must batch without crashing."""
    crops = [
        Image.fromarray(np.zeros((160, 120, 3), dtype=np.uint8)),  # 4:3
        Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)),  # 1:2
        Image.fromarray(np.zeros((90, 90, 3), dtype=np.uint8)),    # 1:1
    ]
    batch = preprocess_face_crops(crops)
    assert batch.shape == (3, 3, 448, 448)


def test_preprocess_nonsquare_single():
    """A single non-square crop is resized to the fixed 448x448 square input."""
    batch = preprocess_face_crops(
        [Image.fromarray(np.zeros((300, 100, 3), dtype=np.uint8))]
    )
    assert batch.shape == (1, 3, 448, 448)


def test_libre_l2cs_multiface_end_to_end(tmp_path):
    """Finding #1 end-to-end: an image with two differently-shaped faces runs."""
    sd = _make_dummy_state_dict("r18")
    wp = tmp_path / "LibreL2CSr18.pt"
    torch.save(sd, wp)
    model = LibreL2CS(str(wp), size="r18", device="cpu")
    img = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))
    result = model(img, face_boxes=[(10, 10, 90, 150), (100, 20, 180, 90)])
    assert len(result.gaze) == 2
    assert result.gaze.data.shape == (2, 2)
    assert torch.isfinite(result.gaze.data).all()


def _craft_l2cs(tmp_path, yaw_bin, pitch_bin, num_bins=90):
    """Build an L2CS whose fc_yaw_gaze/fc_pitch_gaze heads fire fixed bins."""
    net = build_l2cs("r18", num_bins=num_bins)
    with torch.no_grad():
        net.fc_yaw_gaze.weight.zero_()
        net.fc_pitch_gaze.weight.zero_()
        net.fc_yaw_gaze.bias.zero_()
        net.fc_pitch_gaze.bias.zero_()
        net.fc_yaw_gaze.bias[yaw_bin] = 50.0
        net.fc_pitch_gaze.bias[pitch_bin] = 50.0
    wp = tmp_path / "LibreL2CSr18.pt"
    torch.save(net.state_dict(), wp)
    return str(wp)


def test_pitch_yaw_head_assignment(tmp_path):
    """Finding #2: result.gaze.pitch must come from fc_yaw_gaze (the pitch head).

    Per upstream L2CS-Net training, forward position 0 (fc_yaw_gaze) is
    supervised on pitch labels and position 1 (fc_pitch_gaze) on yaw labels.
    So result.gaze.pitch == decode(fc_yaw_gaze) and .yaw == decode(fc_pitch_gaze).
    """
    # fc_yaw_gaze -> bin 80 -> 80*4-180 = +140 deg; fc_pitch_gaze -> bin 10 -> -140 deg
    wp = _craft_l2cs(tmp_path, yaw_bin=80, pitch_bin=10)
    model = LibreL2CS(wp, size="r18", device="cpu")
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    res = model(img, face_boxes=[(0, 0, 64, 64)])
    assert float(res.gaze.pitch_deg[0]) == pytest.approx(140.0, abs=1.0)
    assert float(res.gaze.yaw_deg[0]) == pytest.approx(-140.0, abs=1.0)


def test_28bin_checkpoint_loads_and_decodes(tmp_path):
    """Finding #4: a 28-bin MPIIGaze-style checkpoint loads with correct geometry."""
    net = build_l2cs("r18", num_bins=28)
    wp = tmp_path / "LibreL2CSr18.pt"
    torch.save(net.state_dict(), wp)
    model = LibreL2CS(str(wp), size="r18", device="cpu")
    assert model.num_bins == 28
    assert model.bin_width_deg == 3.0
    assert model.offset_deg == -42.0
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    res = model(img, face_boxes=[(0, 0, 64, 64)])
    assert torch.isfinite(res.gaze.data).all()


def test_num_bins_inferred_from_state_dict():
    """Finding #4: num_bins is read off the checkpoint, not assumed to be 90."""
    assert LibreL2CS._detect_num_bins(build_l2cs("r18", num_bins=28).state_dict()) == 28
    assert LibreL2CS._detect_num_bins(build_l2cs("r18", num_bins=90).state_dict()) == 90
    assert LibreL2CS._detect_num_bins({"not": "a checkpoint"}) is None


def test_resolve_face_detector_rejects_unknown():
    """Finding #7: a non-model, non-callable object is rejected, not mis-wrapped."""

    class HasPredict:
        def predict(self):  # noqa: D102
            ...

    with pytest.raises(TypeError, match="Unsupported face_detector"):
        resolve_face_detector(HasPredict())
    # Plain callables and None are still accepted.
    assert isinstance(resolve_face_detector(lambda img: []), CallableFaceDetector)
    assert resolve_face_detector(None) is None


def test_invalid_output_file_format_raises(tmp_path):
    """Finding #8: an unsupported output_file_format fails fast with ValueError."""
    sd = _make_dummy_state_dict("r18")
    wp = tmp_path / "LibreL2CSr18.pt"
    torch.save(sd, wp)
    model = LibreL2CS(str(wp), size="r18", device="cpu")
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="output_file_format"):
        model(img, face_boxes=[(0, 0, 64, 64)], save=True, output_file_format="tif")


def test_get_download_url_is_none():
    """L2CS weights are never auto-downloaded — Gaze360 license forbids mirroring."""
    assert LibreL2CS.get_download_url("LibreL2CSr50.pt") is None


def test_missing_weights_gives_helpful_error():
    """A missing checkpoint raises an actionable error with the download link."""
    with pytest.raises(FileNotFoundError, match=r"drive\.google\.com") as exc:
        LibreL2CS("definitely_missing_l2cs.pkl", size="r18", device="cpu")
    msg = str(exc.value)
    assert "Gaze360" in msg
    assert "non-commercial" in msg
    assert "L2CSNet_gaze360.pkl" in msg

"""DEKR bottom-up pose: public inference route and export/reload parity.

Gated like the other weight-backed e2e suites: the checkpoint is fetched from
the source provider's CDN (LibreYOLO links rather than mirrors it), so these
tests skip when the artifact cannot be provisioned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from libreyolo import LibreYOLO  # noqa: E402
from libreyolo.postprocess.dekr import decode_poses  # noqa: E402
from libreyolo.preprocess.dekr import preprocess_numpy  # noqa: E402

from .conftest import DEKR_POSE_MODELS  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.dekr]

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE = REPO_ROOT / "libreyolo" / "assets" / "parkour.jpg"
FAMILY, SIZE, WEIGHTS = DEKR_POSE_MODELS[0]


@pytest.fixture(scope="module")
def model():
    """Load DEKR on CPU so export comparisons are not CUDA/CPU cross-device."""
    try:
        return LibreYOLO(WEIGHTS, device="cpu")
    except Exception as exc:  # pragma: no cover - provisioning failure
        pytest.skip(f"could not provision {WEIGHTS}: {exc}")


@pytest.fixture(scope="module")
def image_rgb():
    from PIL import Image

    if not IMAGE.exists():  # pragma: no cover
        pytest.skip(f"missing test image {IMAGE}")
    return np.asarray(Image.open(IMAGE).convert("RGB"))


@pytest.fixture(scope="module")
def batch(image_rgb):
    chw, scale = preprocess_numpy(image_rgb, 640)
    single = torch.from_numpy(chw)[None]
    return single, torch.cat([single, torch.zeros_like(single)], 0), scale


def test_family_metadata(model):
    assert model.family == "dekr"
    assert model.task == "pose"
    assert model.size == SIZE
    assert model.num_keypoints == 17
    assert model.variant == "no_dc"
    assert model.names == {0: "person"}


def test_predict_returns_flat_pose_results(model):
    result = model(str(IMAGE))
    result = result[0] if isinstance(result, list) else result
    assert result.keypoints is not None
    assert result.keypoints.data.ndim == 3
    assert result.keypoints.data.shape[1:] == (17, 3)
    # Bottom-up: multiple people with no detector in front of the model.
    assert len(result.boxes) >= 2
    assert result.keypoints.data.shape[0] == len(result.boxes)


def test_results_stay_on_the_original_canvas(model, image_rgb):
    result = model(str(IMAGE))
    result = result[0] if isinstance(result, list) else result
    height, width = image_rgb.shape[:2]
    keypoints = result.keypoints.data.numpy()
    assert keypoints[:, :, 0].max() <= width
    assert keypoints[:, :, 1].max() <= height
    boxes = result.boxes.xyxy.numpy()
    assert boxes[:, 0].min() >= 0 and boxes[:, 1].min() >= 0
    assert boxes[:, 2].max() <= width and boxes[:, 3].max() <= height


def test_derived_boxes_bound_their_own_keypoints(model):
    result = model(str(IMAGE))
    result = result[0] if isinstance(result, list) else result
    boxes = result.boxes.xyxy.numpy()
    keypoints = result.keypoints.data.numpy()
    for box, pose in zip(boxes, keypoints):
        confident = pose[pose[:, 2] > 0.05]
        if len(confident) < 2:
            continue
        assert box[0] <= confident[:, 0].min() + 1e-3
        assert box[2] >= confident[:, 0].max() - 1e-3


def test_raw_graph_contract(model, batch):
    single, _, _ = batch
    with torch.no_grad():
        heatmap, offsets = model.model(single)
    assert heatmap.shape == (1, 18, 160, 160)  # K + 1 channels at stride 4
    assert offsets.shape == (1, 34, 160, 160)  # 2K channels
    # No sigmoid inside the graph.
    assert heatmap.min() < 0.0


def test_batch_two_decodes_independently(model, batch):
    single, pair, _ = batch
    with torch.no_grad():
        one = model.model(single)
        two = model.model(pair)
    decoded_one = decode_poses(*one)
    decoded_two = decode_poses(*two)
    assert len(decoded_two) == 2
    np.testing.assert_allclose(
        decoded_two[0][0], decoded_one[0][0], rtol=0, atol=1e-4
    )
    # The zero-filled second item must not borrow poses from the first.
    assert len(decoded_two[1][0]) <= len(decoded_two[0][0])


def test_rectangular_source_restores_through_padding_and_scale(model, image_rgb):
    from PIL import Image

    portrait = Image.fromarray(image_rgb).resize((480, 800))
    landscape = Image.fromarray(image_rgb).resize((800, 480))
    for source in (portrait, landscape):
        result = model(source)
        result = result[0] if isinstance(result, list) else result
        keypoints = result.keypoints.data.numpy()
        if not len(keypoints):
            continue
        assert keypoints[:, :, 0].max() <= source.width
        assert keypoints[:, :, 1].max() <= source.height


def test_training_is_explicitly_out_of_scope(model):
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="coco8-pose.yaml")


def test_unsupported_export_format_is_rejected(model):
    with pytest.raises(NotImplementedError):
        model.export(format="ncnn")


@pytest.mark.parametrize("fmt", ["onnx", "torchscript"])
def test_export_reload_matches_eager_outputs(model, batch, fmt):
    single, pair, _ = batch
    with torch.no_grad():
        reference = [t.numpy() for t in model.model(single)]
        reference_pair = [t.numpy() for t in model.model(pair)]

    path = model.export(format=fmt)

    if fmt == "torchscript":
        module = torch.jit.load(path)
        module.eval()
        with torch.no_grad():
            got = [t.numpy() for t in module(single)]
            got_pair = [t.numpy() for t in module(pair)]
        tolerance = 0.0  # same runtime, so this must be exact
    else:
        ort = pytest.importorskip("onnxruntime")
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        assert [o.name for o in session.get_outputs()] == [
            "heatmap_logits",
            "offsets",
        ]
        name = session.get_inputs()[0].name
        got = session.run(None, {name: single.numpy()})
        got_pair = session.run(None, {name: pair.numpy()})
        tolerance = 1e-2

    for expected, actual in zip(reference, got):
        assert np.abs(expected - actual).max() <= tolerance
    for expected, actual in zip(reference_pair, got_pair):
        assert np.abs(expected - actual).max() <= tolerance


@pytest.mark.parametrize("fmt", ["onnx", "torchscript"])
def test_public_backend_route_matches_native_results(model, fmt):
    """An exported artifact reloaded through LibreYOLO must return the same poses.

    Raw-tensor agreement is not enough: the backend has to select the DEKR
    preprocessing and the DEKR decoder, or the dense maps reach the detection
    parser and produce garbage.
    """
    native = model(str(IMAGE))
    native = native[0] if isinstance(native, list) else native

    path = model.export(format=fmt)
    backend = LibreYOLO(path)
    exported = backend(str(IMAGE))
    exported = exported[0] if isinstance(exported, list) else exported

    assert len(exported.boxes) == len(native.boxes)
    assert exported.keypoints is not None
    assert exported.keypoints.data.shape == native.keypoints.data.shape

    # Sub-pixel rather than exact: the backend picks its own device, so this
    # comparison can straddle CPU and CUDA, which differ at the usual FP32
    # level. Raw-tensor exactness is asserted separately, same-device, in
    # test_export_reload_matches_eager_outputs.
    tolerance = 1.0
    np.testing.assert_allclose(
        np.sort(exported.boxes.conf.numpy()),
        np.sort(native.boxes.conf.numpy()),
        rtol=0,
        atol=0.05,
    )
    np.testing.assert_allclose(
        exported.keypoints.data.numpy()[:, :, :2],
        native.keypoints.data.numpy()[:, :, :2],
        rtol=0,
        atol=tolerance,
    )
    np.testing.assert_allclose(
        exported.boxes.xyxy.numpy(),
        native.boxes.xyxy.numpy(),
        rtol=0,
        atol=tolerance,
    )
    assert exported.boxes.cls.numpy().tolist() == native.boxes.cls.numpy().tolist()


@pytest.mark.parametrize("fmt", ["onnx", "torchscript"])
def test_exported_outputs_decode_to_the_same_poses(model, batch, fmt):
    single, _, _ = batch
    with torch.no_grad():
        reference = model.model(single)
    expected_poses, _ = decode_poses(*reference)[0]

    path = model.export(format=fmt)
    if fmt == "torchscript":
        module = torch.jit.load(path)
        module.eval()
        with torch.no_grad():
            raw = module(single)
    else:
        ort = pytest.importorskip("onnxruntime")
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        raw = [torch.from_numpy(t) for t in session.run(None, {name: single.numpy()})]

    poses, _ = decode_poses(*raw)[0]
    assert len(poses) == len(expected_poses)
    if len(poses):
        np.testing.assert_allclose(poses, expected_poses, rtol=0, atol=1.0)

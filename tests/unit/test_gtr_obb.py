"""CPU contracts for the GTR oriented-box (DOTA) port, with random weights."""

import math

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreGTR, LibreYOLO
from libreyolo.models.gtr.obb import (
    DOTA_NAMES,
    GTROBBValPreprocessor,
    is_gtr_obb_state_dict,
    preprocess_obb_image,
)
from libreyolo.models.gtr.obb_nn import LibreGTROBBModel
from libreyolo.models.gtr.obb_rbox import distance2rbox
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    state = torch.get_rng_state()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
        torch.set_rng_state(state)


@pytest.fixture(scope="module")
def obb_state():
    torch.manual_seed(0)
    return LibreGTROBBModel("s", 15).state_dict()


def _save(tmp_path, state, name="LibreGTRs-obb.pt", task="obb"):
    ckpt = wrap_libreyolo_checkpoint(
        state,
        model_family="gtr",
        size="s",
        nc=15,
        names=dict(DOTA_NAMES),
        task=task,
        imgsz=1024,
    )
    path = tmp_path / name
    torch.save(ckpt, path)
    return path


def test_obb_model_outputs_normalized_rboxes():
    model = LibreGTROBBModel("s", 15, (160, 160)).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 160, 160))
    assert out["pred_logits"].shape == (1, 300, 15)
    assert out["pred_boxes"].shape == (1, 300, 5)
    boxes = out["pred_boxes"]
    assert (boxes >= 0).all() and (boxes[..., 4] < 1).all()
    # Anchors regenerate for inputs other than the eval size.
    with torch.no_grad():
        assert model(torch.randn(1, 3, 192, 192))["pred_boxes"].shape == (1, 300, 5)


def test_only_published_obb_sizes_are_built():
    with pytest.raises(ValueError, match="no 'm' OBB weights"):
        LibreGTROBBModel("m")


def test_adr_decoding_returns_long_edge_angles():
    ref = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.25]])
    out = distance2rbox(ref, torch.zeros(1, 6), 4.0)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)
    # A box given short-edge-first comes back long-edge-first, angle + pi/2.
    swapped = distance2rbox(
        torch.tensor([[0.5, 0.5, 0.1, 0.2, 0.0]]), torch.zeros(1, 6), 4.0
    )
    torch.testing.assert_close(
        swapped, torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.5]]), atol=1e-5, rtol=0
    )


def test_checkpoint_task_detection_does_not_collide_with_detect(obb_state):
    from libreyolo.models.gtr.nn import LibreGTRModel

    detect_state = LibreGTRModel("s", 80).state_dict()
    assert LibreGTR.can_load(obb_state) and LibreGTR.can_load(detect_state)
    assert LibreGTR.detect_checkpoint_task(obb_state) == "obb"
    assert LibreGTR.detect_checkpoint_task(detect_state) is None
    assert is_gtr_obb_state_dict(obb_state) and not is_gtr_obb_state_dict(detect_state)
    assert LibreGTR.detect_size(obb_state) == "s"
    assert LibreGTR.default_checkpoint_names(15)[14] == "helicopter"


def test_obb_checkpoint_loads_strictly_and_predicts_rotated_boxes(tmp_path, obb_state):
    path = _save(tmp_path, obb_state)
    model = LibreYOLO(str(path), device="cpu")
    assert isinstance(model, LibreGTR)
    assert (model.task, model.size, model.input_size) == ("obb", "s", 1024)
    assert model.names[0] == "plane"
    for key, value in obb_state.items():
        torch.testing.assert_close(model.model.state_dict()[key], value, rtol=0, atol=0)

    image = Image.fromarray(np.full((120, 200, 3), 127, dtype=np.uint8))
    result = model.predict(image, imgsz=160, conf=0.0, max_det=5)
    assert result.obb.data.shape == (5, 7)
    assert result.boxes.data.shape == (5, 6)
    assert result.orig_shape == (120, 200)
    angles = result.obb.data[:, 4]
    assert ((angles >= 0) & (angles < math.pi)).all()

    missing = dict(obb_state)
    missing.pop("decoder.pre_bbox_head.layers.2.weight")
    with pytest.raises((RuntimeError, ValueError)):
        LibreYOLO(str(_save(tmp_path, missing, "broken.pt")), device="cpu")


def test_task_mismatch_is_rejected(tmp_path, obb_state):
    path = _save(tmp_path, obb_state, name="obb_weights.pt")
    with pytest.raises((ValueError, RuntimeError), match="task='obb'"):
        LibreGTR(str(path), size="s", nb_classes=15, device="cpu", task="detect")


def test_preprocessing_pads_bottom_right_and_normalizes():
    image = Image.fromarray(np.full((50, 100, 3), 255, dtype=np.uint8))
    tensor, _, size, scale = preprocess_obb_image(image, 64)
    assert tensor.shape == (1, 3, 64, 64) and size == (100, 50) and scale == 0.64
    white = (1 - 0.485) / 0.229
    black = (0 - 0.485) / 0.229
    assert tensor[0, 0, 0, 0].item() == pytest.approx(white, abs=1e-5)
    assert tensor[0, 0, -1, -1].item() == pytest.approx(black, abs=1e-5)

    val = GTROBBValPreprocessor(img_size=(64, 64))
    chw, _ = val(np.full((50, 100, 3), 255, np.uint8), np.zeros((0, 5)), (64, 64))
    np.testing.assert_allclose(chw, tensor[0].numpy(), atol=1e-5)
    assert val.custom_normalization and not val.normalize


def test_obb_val_preprocessor_survives_pickling():
    import pickle

    from libreyolo.validation.obb_validator import _OBBValPreprocessor

    wrapped = _OBBValPreprocessor(GTROBBValPreprocessor(img_size=(64, 64)))
    restored = pickle.loads(pickle.dumps(wrapped))
    assert restored.custom_normalization


def test_canonical_obb_filenames_resolve_to_task_repos():
    assert LibreGTR.detect_size_from_filename("LibreGTRx-obb.pt") == "x"
    assert LibreGTR.detect_task_from_filename("LibreGTRx-obb.pt") == "obb"
    url = LibreGTR.get_download_url("LibreGTRs-obb.pt")
    assert url.startswith("https://huggingface.co/LibreYOLO/LibreGTRs-obb/resolve/")
    assert url.endswith("/LibreGTRs-obb.pt")
    assert LibreGTR.get_download_url("LibreGTRm-obb.pt") is None
    assert "LibreGTRs/resolve/74193dc" in LibreGTR.get_download_url("LibreGTRs.pt")

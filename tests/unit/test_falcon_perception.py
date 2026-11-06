"""Falcon Perception adapter contract (no weights): vocabulary, tasks, postprocess."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.falcon_perception import LibreFalconPerception
from libreyolo.models.falcon_perception import model as fp_model
from libreyolo.models.falcon_perception.generate import Generation

pytestmark = pytest.mark.unit


def _bare(size="600m", task="detect", names=("cat",)):
    lib = LibreFalconPerception.__new__(LibreFalconPerception)
    lib.size = size
    lib.task = task
    lib.names = {i: n for i, n in enumerate(names)}
    lib._user_vocab = True
    lib._custom_prompt = None
    lib.model = SimpleNamespace(args=SimpleNamespace(do_segmentation=size == "600m"))
    return lib


def test_family_metadata():
    assert LibreFalconPerception.FAMILY == "falconperception"
    assert set(LibreFalconPerception.HF_REPOS) == {"300m", "600m"}
    assert set(LibreFalconPerception.HF_REVISIONS) == {"300m", "600m"}
    assert LibreFalconPerception.SUPPORTED_TASKS == ("detect", "segment")
    assert not LibreFalconPerception.TRUST_REMOTE_CODE


def test_query_needs_vocabulary():
    lib = _bare()
    lib._user_vocab = False
    with pytest.raises(ValueError, match="set_classes"):
        lib._queries()
    lib._user_vocab = True
    assert lib._queries() == [(0, "cat")]


def test_300m_rejects_segment():
    lib = _bare("300m")
    with pytest.raises(ValueError, match="no mask head"):
        lib.set_task("segment")
    lib.task = "segment"
    with pytest.raises(ValueError, match="no mask head"):
        lib._preprocess(Image.new("RGB", (32, 32)))


def test_chat_and_train_are_rejected():
    lib = _bare()
    with pytest.raises(NotImplementedError):
        lib.chat("x.jpg", "hi")
    with pytest.raises(NotImplementedError):
        lib.train(data="d.yaml")


def test_postprocess_boxes_masks_and_limits():
    lib = _bare(task="segment", names=("cat", "dog"))
    masks = torch.zeros(2, 4, 4, dtype=torch.bool)
    masks[0, :2, :2] = True
    raw = [
        {"x": 0.5, "y": 0.5}, {"h": 0.5, "w": 0.5},
        {"x": 0.25, "y": 0.25}, {"h": 0.25, "w": 0.25},
    ]
    output = [
        {"class_id": 0, "generation": Generation(tokens=torch.zeros(1), boxes_raw=raw, masks=masks)},
        {"class_id": 1, "generation": Generation(tokens=torch.zeros(1), boxes_raw=raw[:2], masks=masks[:1])},
    ]
    out = lib._postprocess(output, conf_thres=0.0, iou_thres=0.5, original_size=(200, 100))
    assert out["num_detections"] == 3
    assert out["classes"] == [0, 0, 1]
    np.testing.assert_allclose(out["boxes"][0], [50.0, 25.0, 150.0, 75.0])
    assert out["masks"].shape == (3, 100, 200) and out["masks"].dtype == np.uint8
    assert out["masks"][0, :50, :100].all() and not out["masks"][0, 50:].any()

    only_dog = lib._postprocess(
        output, 0.0, 0.5, (200, 100), classes=[1]
    )
    assert only_dog["classes"] == [1]
    assert lib._postprocess(output, 0.0, 0.5, (200, 100), max_det=1)["num_detections"] == 1

    lib.task = "detect"
    assert "masks" not in lib._postprocess(output, 0.0, 0.5, (200, 100))


def test_mask_resize_matches_nearest_and_pads_missing():
    masks = torch.zeros(1, 2, 2, dtype=torch.bool)
    masks[0, 0, 0] = True
    got = fp_model._mask_to_original(masks, 0, 4, 4)
    assert got.shape == (4, 4) and got[:2, :2].all() and got.sum() == 4
    assert fp_model._mask_to_original(masks, 3, 4, 4).sum() == 0
    assert fp_model._mask_to_original(None, 0, 4, 4).shape == (4, 4)

"""Tests for the shared batch-output slicer used by validation and predict."""

from __future__ import annotations

import pytest
import torch

from libreyolo.postprocess.slicing import slice_batch_outputs


pytestmark = pytest.mark.unit


def test_slices_bare_tensor_keeping_batch_dim():
    batched = torch.arange(24.0).reshape(3, 2, 4)

    sliced = slice_batch_outputs(batched, 1)

    assert sliced.shape == (1, 2, 4)
    assert torch.equal(sliced, batched[1:2])


def test_slices_dict_values_and_passes_non_tensors_through():
    output = {
        "predictions": torch.rand(2, 84, 100),
        "obb": True,
        "names": "unchanged",
        "maybe": None,
    }

    sliced = slice_batch_outputs(output, 0)

    assert torch.equal(sliced["predictions"], output["predictions"][0:1])
    assert sliced["obb"] is True
    assert sliced["names"] == "unchanged"
    assert sliced["maybe"] is None


def test_slices_one_level_of_nested_dicts():
    output = {
        "x8": {"features": torch.rand(2, 8, 4, 4), "tag": "keep"},
    }

    sliced = slice_batch_outputs(output, 1)

    assert torch.equal(sliced["x8"]["features"], output["x8"]["features"][1:2])
    assert sliced["x8"]["tag"] == "keep"


def test_recurses_into_picodet_style_tuple_of_lists():
    cls_scores = [torch.rand(2, 80, 8, 8), torch.rand(2, 80, 4, 4)]
    bbox_preds = [torch.rand(2, 32, 8, 8), torch.rand(2, 32, 4, 4)]
    output = (cls_scores, bbox_preds)

    sliced = slice_batch_outputs(output, 1)

    assert isinstance(sliced, tuple)
    assert isinstance(sliced[0], list)
    for original, single in zip(cls_scores, sliced[0]):
        assert torch.equal(single, original[1:2])
    for original, single in zip(bbox_preds, sliced[1]):
        assert torch.equal(single, original[1:2])


def test_list_of_per_scale_tensors_yolox_style():
    output = [torch.rand(3, 85, s, s) for s in (8, 4, 2)]

    sliced = slice_batch_outputs(output, 2)

    assert isinstance(sliced, list)
    for original, single in zip(output, sliced):
        assert torch.equal(single, original[2:3])


def test_scalar_passthrough():
    assert slice_batch_outputs(7, 0) == 7
    assert slice_batch_outputs(None, 1) is None
    assert slice_batch_outputs("flag", 2) == "flag"

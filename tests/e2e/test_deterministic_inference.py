"""Nightly native inference checks for every public model family.

This is the broad tier: one smallest pretrained case per family, native model
load, two inference passes, and stable task-appropriate outputs. Keep exports
and training out of this file; those belong to flagship or backend-specific
suites.
"""

import pytest
import torch
from PIL import Image

from libreyolo import LibreYOLO

from .conftest import (
    GENERAL_NIGHTLY_INFERENCE_PARAMS,
    cuda_cleanup,
    require_test_weights,
)

pytestmark = [pytest.mark.e2e, pytest.mark.general_nightly]


def _tensor(data):
    return (
        data.detach().cpu() if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    )


def _assert_detection_output_is_stable(family, first, second):
    assert first.boxes is not None, f"{family} did not return detection boxes"
    assert second.boxes is not None, f"{family} did not return detection boxes"
    assert len(first.boxes) > 0, f"{family} returned no detections"
    assert len(first.boxes) == len(second.boxes), (
        f"{family} detection count changed: {len(first.boxes)} -> {len(second.boxes)}"
    )
    assert first.orig_shape == second.orig_shape
    assert first.names, f"{family} result has no class names"

    n = min(5, len(first.boxes))
    first_boxes = _tensor(first.boxes.xyxy[:n])
    second_boxes = _tensor(second.boxes.xyxy[:n])
    first_conf = _tensor(first.boxes.conf[:n])
    second_conf = _tensor(second.boxes.conf[:n])
    first_cls = _tensor(first.boxes.cls[:n])
    second_cls = _tensor(second.boxes.cls[:n])

    assert torch.isfinite(first_boxes).all(), f"{family} produced non-finite boxes"
    assert torch.isfinite(first_conf).all(), f"{family} produced non-finite scores"

    # Same-process native inference should be stable. Keep a tiny tolerance so
    # GPU kernels and postprocess threshold edges do not create false nightly
    # failures while still catching meaningful drift.
    torch.testing.assert_close(first_boxes, second_boxes, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(first_conf, second_conf, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(first_cls, second_cls, rtol=0, atol=0)


def _assert_classification_output_is_stable(family, first, second):
    assert first.probs is not None, f"{family} did not return probabilities"
    assert second.probs is not None, f"{family} did not return probabilities"
    assert first.boxes is None and second.boxes is None
    assert first.orig_shape == second.orig_shape
    assert first.names, f"{family} result has no class names"

    first_probs = _tensor(first.probs.data)
    second_probs = _tensor(second.probs.data)
    assert first_probs.ndim == second_probs.ndim == 1
    assert first_probs.shape == second_probs.shape
    assert torch.isfinite(first_probs).all(), f"{family} produced non-finite probs"
    assert torch.isfinite(second_probs).all(), f"{family} produced non-finite probs"
    torch.testing.assert_close(first_probs.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(second_probs.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)
    torch.testing.assert_close(first_probs, second_probs, rtol=1e-5, atol=1e-6)
    assert first.probs.top1 == second.probs.top1


def _assert_pose_output_is_stable(family, first, second):
    _assert_detection_output_is_stable(family, first, second)
    assert first.keypoints is not None, f"{family} did not return keypoints"
    assert second.keypoints is not None, f"{family} did not return keypoints"
    assert len(first.keypoints) == len(first.boxes)
    assert len(second.keypoints) == len(second.boxes)

    first_keypoints = _tensor(first.keypoints.data)
    second_keypoints = _tensor(second.keypoints.data)
    assert first_keypoints.shape[1:] == (17, 3)
    assert torch.isfinite(first_keypoints).all(), (
        f"{family} produced non-finite keypoints"
    )
    assert torch.isfinite(second_keypoints).all(), (
        f"{family} produced non-finite keypoints"
    )
    torch.testing.assert_close(first_keypoints, second_keypoints, rtol=1e-4, atol=1e-4)


def _assert_batched_detection_output_matches_sequential(family, sequential, batched):
    assert sequential.boxes is not None, f"{family} did not return detection boxes"
    assert batched.boxes is not None, f"{family} did not return detection boxes"
    assert sequential.orig_shape == batched.orig_shape
    assert sequential.names, f"{family} result has no class names"
    assert len(sequential.boxes) == len(batched.boxes), (
        f"{family} batched detection count diverged: "
        f"{len(sequential.boxes)} -> {len(batched.boxes)}"
    )

    n = min(5, len(sequential.boxes))
    sequential_boxes = _tensor(sequential.boxes.xyxy[:n])
    batched_boxes = _tensor(batched.boxes.xyxy[:n])
    sequential_conf = _tensor(sequential.boxes.conf[:n])
    batched_conf = _tensor(batched.boxes.conf[:n])
    sequential_cls = _tensor(sequential.boxes.cls[:n])
    batched_cls = _tensor(batched.boxes.cls[:n])

    assert torch.isfinite(sequential_boxes).all(), (
        f"{family} produced non-finite sequential boxes"
    )
    assert torch.isfinite(batched_boxes).all(), (
        f"{family} produced non-finite batched boxes"
    )
    assert torch.isfinite(sequential_conf).all(), (
        f"{family} produced non-finite sequential scores"
    )
    assert torch.isfinite(batched_conf).all(), (
        f"{family} produced non-finite batched scores"
    )

    # Batch=1 and batch>1 can dispatch different CUDA kernels; with TF32 on
    # NVIDIA Ampere/L4 this has measured score drift up to ~3e-3. The caller
    # keeps the test away from threshold-edge detections while this assertion
    # allows that score drift and still requires retained boxes/classes to match.
    # Boxes are in pixels: the same kernel drift moved single coordinates by up
    # to ~0.1 px on the Guggenheim sample (L4), so require sub-pixel agreement.
    torch.testing.assert_close(sequential_boxes, batched_boxes, rtol=0, atol=0.5)
    torch.testing.assert_close(sequential_conf, batched_conf, rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(sequential_cls, batched_cls, rtol=0, atol=0)


def _assert_batched_classification_output_matches_sequential(
    family, sequential, batched
):
    assert sequential.probs is not None, f"{family} did not return probabilities"
    assert batched.probs is not None, f"{family} did not return probabilities"
    assert sequential.boxes is None and batched.boxes is None
    assert sequential.orig_shape == batched.orig_shape
    assert sequential.names, f"{family} result has no class names"
    assert sequential.names == batched.names

    sequential_probs = _tensor(sequential.probs.data)
    batched_probs = _tensor(batched.probs.data)
    assert sequential_probs.shape == batched_probs.shape
    assert torch.isfinite(sequential_probs).all()
    assert torch.isfinite(batched_probs).all()
    assert sequential.probs.top1 == batched.probs.top1
    # Batch-one and batch-two GEMMs can use different TF32 kernels. VGG-16 on
    # an RTX 5070 Ti measured 1.9e-3 relative and 1.7e-4 absolute drift while
    # retaining the same top-1 class.
    torch.testing.assert_close(sequential_probs, batched_probs, rtol=3e-3, atol=1e-5)


def _assert_batched_pose_output_matches_sequential(family, sequential, batched):
    _assert_batched_detection_output_matches_sequential(family, sequential, batched)
    assert sequential.keypoints is not None, f"{family} did not return keypoints"
    assert batched.keypoints is not None, f"{family} did not return keypoints"
    assert len(sequential.keypoints) == len(sequential.boxes)
    assert len(batched.keypoints) == len(batched.boxes)

    sequential_keypoints = _tensor(sequential.keypoints.data)
    batched_keypoints = _tensor(batched.keypoints.data)
    assert sequential_keypoints.shape[1:] == (17, 3)
    assert torch.isfinite(sequential_keypoints).all(), (
        f"{family} produced non-finite sequential keypoints"
    )
    assert torch.isfinite(batched_keypoints).all(), (
        f"{family} produced non-finite batched keypoints"
    )
    torch.testing.assert_close(
        sequential_keypoints, batched_keypoints, rtol=1e-3, atol=1e-3
    )


def _run_l2cs(weights, size):
    from libreyolo import LibreL2CS

    weights = require_test_weights(weights)
    model = LibreL2CS(
        weights, size=size, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    image = Image.new("RGB", (96, 96), color=(128, 128, 128))
    kwargs = {"face_boxes": [(8, 8, 88, 88)]}
    first = model(image, **kwargs)
    second = model(image, **kwargs)

    assert first.gaze is not None, "l2cs did not return gaze output"
    assert second.gaze is not None, "l2cs did not return gaze output"
    assert len(first.gaze) == 1
    assert len(second.gaze) == 1
    torch.testing.assert_close(
        _tensor(first.gaze.data),
        _tensor(second.gaze.data),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "family,size,weights",
    GENERAL_NIGHTLY_INFERENCE_PARAMS,
)
def test_native_inference_is_stable(family, size, weights, sample_image):
    """Every public family loads its smallest checkpoint and runs stable inference."""
    if family == "l2cs":
        _run_l2cs(weights, size)
        cuda_cleanup()
        return

    weights = require_test_weights(weights, expected_family=family)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO(weights, size=size, device=device)
    try:
        first = model(sample_image, conf=0.25)
        second = model(sample_image, conf=0.25)
        if model.task == "classify":
            _assert_classification_output_is_stable(family, first, second)
        elif family == "hrnet":
            _assert_pose_output_is_stable(family, first, second)
        else:
            _assert_detection_output_is_stable(family, first, second)
    finally:
        del model
        cuda_cleanup()


@pytest.mark.parametrize(
    "family,size,weights",
    GENERAL_NIGHTLY_INFERENCE_PARAMS,
)
def test_batched_list_predict_matches_sequential(family, size, weights, sample_image):
    """batch>1 over an in-memory list reproduces sequential results (issue #384).

    Real weights keep the check tied to production postprocess behavior. Score
    parity is TF32-aware because batch=1 and batch>1 may select different CUDA
    kernels even when batched routing is correct.
    """
    if family in ("l2cs", "vlm"):
        pytest.skip(f"{family} does not use the stacked batched predict path")

    from libreyolo.utils.image_loader import ImageLoader

    weights = require_test_weights(weights, expected_family=family)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO(weights, size=size, device=device)
    try:
        pil = ImageLoader.load(sample_image)
        images = [pil, pil.rotate(90, expand=True)]

        # This test verifies batched routing, not confidence-threshold boundary
        # behavior. A slightly higher threshold avoids TF32 nudging marginal
        # detections across the default 0.25 cutoff on L4-class GPUs.
        conf = 0.30
        sequential = model(images, conf=conf, batch=1)
        batched = model(images, conf=conf, batch=2)

        assert len(sequential) == len(batched) == 2
        if model.task == "classify":
            for r_seq, r_bat in zip(sequential, batched):
                _assert_batched_classification_output_matches_sequential(
                    family, r_seq, r_bat
                )
            return

        assert len(sequential[0]) > 0, f"{family} returned no detections"
        for r_seq, r_bat in zip(sequential, batched):
            # Full box/score/class parity for every image with detections
            # (the rotated view may legitimately drop to zero for some
            # families; the count check above still covers it).
            if len(r_seq) > 0:
                if family == "hrnet":
                    _assert_batched_pose_output_matches_sequential(
                        family, r_seq, r_bat
                    )
                else:
                    _assert_batched_detection_output_matches_sequential(
                        family, r_seq, r_bat
                    )
            else:
                assert len(r_bat) == 0, (
                    f"{family} batched detection count diverged: 0 -> {len(r_bat)}"
                )
                assert r_seq.orig_shape == r_bat.orig_shape
    finally:
        del model
        cuda_cleanup()

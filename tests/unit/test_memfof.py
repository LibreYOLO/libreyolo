"""LibreMEMFOF: checkpoint recognition, conversion and geometry (no weights)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from libreyolo import LibreMEMFOF
from libreyolo.models.memfof._vendor import MEMFOF
from libreyolo.models.memfof.utils import MIN_SIDE, flow_to_original, input_hw, preprocess_numpy
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = [pytest.mark.unit, pytest.mark.memfof]

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def upstream_state_dict():
    # Same keys and shapes as the published model.safetensors (vendored code).
    return MEMFOF().state_dict()


@pytest.fixture(scope="module")
def converter():
    sys.path.insert(0, str(REPO_ROOT / "weights"))
    try:
        import convert_memfof_weights
    finally:
        sys.path.remove(str(REPO_ROOT / "weights"))
    return convert_memfof_weights


def test_can_load_and_detect_size(upstream_state_dict):
    assert LibreMEMFOF.can_load(upstream_state_dict)
    assert LibreMEMFOF.detect_size(upstream_state_dict) == "m"
    assert LibreMEMFOF.detect_nb_classes(upstream_state_dict) == 1


def test_can_load_rejects_two_frame_raft_layouts(upstream_state_dict):
    # A RAFT-style context net reads two frames (6 channels) and has no att.*.
    two_frame = dict(upstream_state_dict)
    two_frame["cnet.resnet.conv1.weight"] = torch.zeros(64, 6, 7, 7)
    assert not LibreMEMFOF.can_load(two_frame)
    no_attention = {k: v for k, v in upstream_state_dict.items() if not k.startswith("att.")}
    assert not LibreMEMFOF.can_load(no_attention)
    assert not LibreMEMFOF.can_load({"model.0.weight": torch.zeros(1)})


@pytest.mark.parametrize(
    ("filename", "variant"),
    [
        ("LibreMEMFOFm-flow.pt", None),
        ("LibreMEMFOFm-flow-tartan.pt", "tartan"),
        ("LibreMEMFOFm-flow-tartan-t.pt", "tartan-t"),
        ("LibreMEMFOFm-flow-kitti.pt", "kitti"),
        ("LibreMEMFOFm-flow-sintel.pt", "sintel"),
        ("LibreMEMFOFm-flow-spring.pt", "spring"),
    ],
)
def test_filenames_resolve_to_one_repo_per_file(filename, variant):
    assert LibreMEMFOF.detect_size_from_filename(filename) == "m"
    assert LibreMEMFOF.detect_task_from_filename(filename) == "flow"
    assert LibreMEMFOF.detect_variant_from_filename(filename) == variant
    stem = filename[: -len(".pt")]
    assert LibreMEMFOF.get_download_url(filename) == (
        f"https://huggingface.co/LibreYOLO/{stem}/resolve/main/{filename}"
    )


def test_task_suffix_is_required():
    assert LibreMEMFOF.detect_size_from_filename("LibreMEMFOFm.pt") is None


def test_converter_wraps_metadata(converter, upstream_state_dict):
    ckpt = converter.convert_state_dict(
        upstream_state_dict,
        variant="tskh",
        source="egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH",
        source_revision="abc",
    )
    assert validate_checkpoint_metadata(ckpt, strict=True) == []
    assert ckpt["model_family"] == "memfof"
    assert ckpt["size"] == "m"
    assert ckpt["task"] == "flow"
    assert ckpt["nc"] == 1
    assert ckpt["names"] == {0: "flow"}
    assert ckpt["variant"] == "tskh"
    assert ckpt["libreyolo_version"]
    converter.check_strict_load(ckpt)


def test_converter_rejects_foreign_state_dict(converter):
    with pytest.raises(ValueError, match="MEMFOF"):
        converter.convert_state_dict({"backbone.weight": torch.zeros(1)}, variant="tskh")


def test_converter_filenames(converter):
    assert converter.output_filename("tskh") == "LibreMEMFOFm-flow.pt"
    for variant in ("tartan", "tartan-t", "kitti", "sintel", "spring"):
        name = converter.output_filename(variant)
        assert name == f"LibreMEMFOFm-flow-{variant}.pt"
        assert LibreMEMFOF.detect_variant_from_filename(name) == variant
    with pytest.raises(ValueError):
        converter.output_filename("things")


def test_converted_checkpoint_loads_through_factory(tmp_path, converter, upstream_state_dict):
    from libreyolo import LibreYOLO

    ckpt = converter.convert_state_dict(upstream_state_dict, variant="spring")
    path = tmp_path / "LibreMEMFOFm-flow-spring.pt"
    torch.save(ckpt, path)
    model = LibreYOLO(str(path), device="cpu")
    assert isinstance(model, LibreMEMFOF)
    assert model.task == "flow"
    assert model.names == {0: "flow"}
    assert model.FLOW_CONTEXT == (1, 0)


def test_preprocess_keeps_native_canvas_and_caps_long_side():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    chw, ratio = preprocess_numpy(img, 1920)
    assert chw.shape == (3, 300, 400) and chw.dtype == np.float32
    assert ratio == (1.0, 1.0)
    chw, ratio = preprocess_numpy(np.zeros((600, 1000, 3), dtype=np.uint8), 500)
    assert chw.shape == (3, 300, 500)
    assert ratio == pytest.approx((2.0, 2.0))
    assert input_hw(1080, 1920, 1920) == (1080, 1920)
    assert MIN_SIDE == 241


def test_flow_to_original_rescales_vectors():
    flow = torch.stack([torch.full((10, 20), 1.5), torch.full((10, 20), -2.0)])
    out = flow_to_original(flow, 30, 40)
    assert out.shape == (30, 40, 2)
    assert torch.allclose(out[..., 0], torch.full((30, 40), 3.0))
    assert torch.allclose(out[..., 1], torch.full((30, 40), -6.0))
    same = flow_to_original(flow, 10, 20)
    assert torch.equal(same, flow.permute(1, 2, 0))


def test_window_contract_returns_forward_flow():
    model = LibreMEMFOF(None, device="cpu", iters=1)
    frames = torch.rand(3, 3, 256, 256) * 255
    out = model._forward_flow(frames)
    assert out["flow"].shape == (1, 2, 256, 256)
    assert out["flow_backward"].shape == (1, 2, 256, 256)
    post = model._postprocess(out, 0.0, 0.0, (256, 256))
    assert post["flow"].shape == (256, 256, 2)
    with pytest.raises(ValueError, match="three"):
        model._forward_flow(torch.rand(2, 3, 256, 256))


def test_train_is_not_supported():
    with pytest.raises(NotImplementedError):
        LibreMEMFOF(None, device="cpu").train()

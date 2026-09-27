"""LibreDEKR family: architecture, recognition, preprocessing and Results."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.models.dekr.model import LibreDEKR
from libreyolo.models.dekr.nn import LibreDEKRModel
from libreyolo.preprocess.dekr import DEKR_PAD_VALUE, preprocess_numpy

pytestmark = [pytest.mark.unit, pytest.mark.dekr]


@pytest.fixture(scope="module")
def tiny_state() -> dict:
    """State dict of a 3-keypoint DEKR, cheap enough to build per module."""
    return LibreDEKRModel(num_keypoints=3).state_dict()


# --- architecture --------------------------------------------------------


def test_forward_shapes_and_stride():
    model = LibreDEKRModel(num_keypoints=3).eval()
    with torch.no_grad():
        heatmap, offsets = model(torch.zeros(2, 3, 64, 64))
    assert heatmap.shape == (2, 4, 16, 16)  # K + 1 channels, stride 4
    assert offsets.shape == (2, 6, 16, 16)  # 2K channels


def test_forward_accepts_non_square_input_divisible_by_four():
    model = LibreDEKRModel(num_keypoints=2).eval()
    with torch.no_grad():
        heatmap, offsets = model(torch.zeros(1, 3, 96, 64))
    assert heatmap.shape[2:] == (24, 16)
    assert offsets.shape[2:] == (24, 16)


def test_graph_emits_raw_logits_not_probabilities():
    torch.manual_seed(0)
    model = LibreDEKRModel(num_keypoints=3).eval()
    with torch.no_grad():
        heatmap, _ = model(torch.randn(1, 3, 64, 64))
    # A hidden sigmoid would bound the output to (0, 1); assert it does not.
    assert heatmap.min() < 0.0 or heatmap.max() > 1.0


def test_offset_head_uses_dilation_five_standard_convs():
    model = LibreDEKRModel(num_keypoints=2)
    block = model.offset_feature_layers[0][0]
    assert block.conv1.dilation == (5, 5)
    assert block.conv1.padding == (5, 5)
    assert not any("adapt_conv" in name for name, _ in model.named_modules())


def test_heatmap_head_uses_dilation_one():
    model = LibreDEKRModel(num_keypoints=2)
    assert model.head_heatmap[0][0].conv1.dilation == (1, 1)


def test_replace_head_rebuilds_both_heads():
    model = LibreDEKRModel(num_keypoints=17)
    model.replace_head(5)
    assert len(model.offset_final_layer) == 5
    assert model.head_heatmap[1].out_channels == 6
    assert model.transition_offset[0].out_channels == 5 * 15
    with torch.no_grad():
        heatmap, offsets = model.eval()(torch.zeros(1, 3, 64, 64))
    assert heatmap.shape[1] == 6
    assert offsets.shape[1] == 10


def test_rejects_zero_keypoints():
    with pytest.raises(ValueError):
        LibreDEKRModel(num_keypoints=0)


# --- family recognition --------------------------------------------------


def test_can_load_accepts_native_layout(tiny_state):
    assert LibreDEKR.can_load(tiny_state) is True
    assert LibreDEKR.detect_num_keypoints(tiny_state) == 3
    assert LibreDEKR.detect_nb_classes(tiny_state) == 1


def test_detect_size_reads_branch_widths_not_filename(tiny_state):
    assert LibreDEKR.detect_size(tiny_state) == "w32"
    narrowed = dict(tiny_state)
    narrowed["stage4.0.branches.3.0.conv1.weight"] = torch.zeros(384, 384, 3, 3)
    assert LibreDEKR.detect_size(narrowed) is None


def test_rejects_deformable_dekr_checkpoint(tiny_state):
    deformable = dict(tiny_state)
    deformable["offset_feature_layers.0.0.adapt_conv.weight"] = torch.zeros(15, 15, 3, 3)
    assert LibreDEKR.can_load(deformable) is False


def test_rejects_incomplete_offset_branch_run(tiny_state):
    gapped = dict(tiny_state)
    del gapped["offset_final_layer.1.weight"]
    assert LibreDEKR.can_load(gapped) is False


def test_rejects_offset_branch_with_wrong_channel_count(tiny_state):
    wrong = dict(tiny_state)
    wrong["offset_final_layer.0.weight"] = torch.zeros(2, 16, 1, 1)
    assert LibreDEKR.can_load(wrong) is False


def test_rejects_heatmap_head_disagreeing_with_offset_branches(tiny_state):
    mismatched = dict(tiny_state)
    mismatched["head_heatmap.1.weight"] = torch.zeros(9, 32, 1, 1)
    assert LibreDEKR.can_load(mismatched) is False


def test_rejects_unrelated_state_dicts():
    assert LibreDEKR.can_load({"conv1.weight": torch.zeros(64, 3, 3, 3)}) is False
    assert LibreDEKR.can_load({}) is False


def test_hrnet_and_dekr_reject_each_other(tiny_state):
    """Both are HRNet-lineage pose families and must not steal each other's files."""
    from libreyolo.models.hrnet.model import LibreHRNet
    from libreyolo.models.hrnet.nn import HRNetPoseModel

    assert LibreHRNet.can_load(tiny_state) is False
    hrnet_state = HRNetPoseModel(width=32, num_keypoints=17).state_dict()
    assert LibreDEKR.can_load(hrnet_state) is False


def test_filename_and_download_route():
    assert LibreDEKR.detect_size_from_filename("LibreDEKRw32-pose.pt") == "w32"
    assert LibreDEKR.detect_size_from_filename("dekr_w32_no_dc_coco_pose.pth") == "w32"
    assert LibreDEKR.detect_task_from_filename("dekr_w32_no_dc_coco_pose.pth") == "pose"
    url = LibreDEKR.get_download_url("LibreDEKRw32-pose.pt")
    # Weights are linked from the source CDN, never mirrored on the LibreYOLO org.
    assert url.endswith("/dekr_w32_no_dc_coco_pose.pth")
    assert "huggingface" not in url
    assert LibreDEKR.get_download_url("LibreDEKRw48-pose.pt") is None


def test_family_is_registered_and_enrolled():
    from libreyolo.models import BaseModel
    from libreyolo.models.registry import MODEL_GROUPS

    assert LibreDEKR in BaseModel._registry
    # Inference-only families belong in g3, not the trainable coverage sets.
    assert MODEL_GROUPS["dekr"] == "g3"


def test_verify_downloaded_file_rejects_unknown_artifact(tmp_path):
    blob = tmp_path / "whatever.pth"
    blob.write_bytes(b"not the real checkpoint")
    with pytest.raises(RuntimeError, match="no pinned checksum"):
        LibreDEKR.verify_downloaded_file(str(blob), "https://cdn/models/whatever.pth")


def test_verify_downloaded_file_rejects_tampered_artifact(tmp_path):
    blob = tmp_path / LibreDEKR._SOURCE_FILENAME
    blob.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        LibreDEKR.verify_downloaded_file(
            str(blob), f"https://cdn/models/{LibreDEKR._SOURCE_FILENAME}"
        )
    assert not blob.exists()  # fails closed: the bad file is removed


# --- checkpoint handling -------------------------------------------------


def test_unwrap_drops_optimizer_and_scaler_state(tiny_state):
    from libreyolo.models.dekr.utils import strip_module_prefix, unwrap_dekr_checkpoint

    released = {
        "net": {f"module.{k}": v for k, v in tiny_state.items()},
        "acc": 0.63,
        "epoch": 140,
        "optimizer_state_dict": {"state": {}},
        "scaler_state_dict": {"scale": 1.0},
    }
    prepared = strip_module_prefix(unwrap_dekr_checkpoint(released))
    assert set(prepared) == set(tiny_state)
    assert not any(
        key in prepared for key in ("optimizer_state_dict", "scaler_state_dict", "acc")
    )


def test_unwrap_rejects_non_mapping_checkpoint():
    from libreyolo.models.dekr.utils import unwrap_dekr_checkpoint

    with pytest.raises(TypeError):
        unwrap_dekr_checkpoint(object())


def test_strip_module_prefix_removes_exactly_one_level():
    from libreyolo.models.dekr.utils import strip_module_prefix

    stripped = strip_module_prefix({"module.module.conv1.weight": 1, "bn1.weight": 2})
    assert set(stripped) == {"module.conv1.weight", "bn1.weight"}


# --- preprocessing -------------------------------------------------------


@pytest.mark.parametrize(
    ("height", "width"),
    [(480, 640), (640, 480), (721, 1281), (100, 100), (1000, 300)],
)
def test_preprocess_letterboxes_to_square_canvas(height, width):
    image = np.full((height, width, 3), 200, dtype=np.uint8)
    chw, scale = preprocess_numpy(image, 640)
    assert chw.shape == (3, 640, 640)
    assert scale == pytest.approx(min(640 / height, 640 / width))


def test_preprocess_pads_bottom_right_with_127():
    # 320x640 upscales to 640x1280? No: longest side governs, so 320x640 -> 320x640.
    image = np.zeros((320, 640, 3), dtype=np.uint8)
    chw, scale = preprocess_numpy(image, 640)
    assert scale == pytest.approx(1.0)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    expected_pad = ((DEKR_PAD_VALUE / 255.0) - mean) / std
    # Top-left region is content, the bottom band is padding.
    np.testing.assert_allclose(chw[:, 500, 0], expected_pad[:, 0, 0], rtol=1e-6)
    assert not np.allclose(chw[:, 0, 0], expected_pad[:, 0, 0])


def test_preprocess_uses_round_half_up_resize():
    # 641 * (640/1281) = 320.2498..., and 1281 -> 640. Locks the odd-pixel case.
    image = np.zeros((641, 1281, 3), dtype=np.uint8)
    chw, scale = preprocess_numpy(image, 640)
    expected_rows = int(641 * scale + 0.5)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    pad = float(((DEKR_PAD_VALUE / 255.0) - mean[0, 0, 0]) / std[0, 0, 0])
    assert chw[0, expected_rows, 0] == pytest.approx(pad, rel=1e-6)
    assert chw[0, expected_rows - 1, 0] != pytest.approx(pad, rel=1e-6)


def test_preprocess_rejects_non_rgb():
    with pytest.raises(ValueError):
        preprocess_numpy(np.zeros((32, 32), dtype=np.uint8), 640)

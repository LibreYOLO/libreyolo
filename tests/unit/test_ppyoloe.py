"""PP-YOLOE architecture, checkpoint recognition, preprocessing and decoding."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.models.ppyoloe.convert import (
    convert_upstream,
    detect_nb_classes_from_state,
    detect_size_from_state,
    is_upstream_state_dict,
    unwrap_ppyoloe_checkpoint,
)
from libreyolo.models.ppyoloe.model import LibrePPYOLOE
from libreyolo.models.ppyoloe.nn import LibrePPYOLOEModel, PPYOLOE_CONFIGS
from libreyolo.models.ppyoloe.utils import (
    PPYOLOE_MEAN,
    PPYOLOE_STD,
    preprocess_numpy,
)
from libreyolo.postprocess.ppyoloe import postprocess

pytestmark = [pytest.mark.unit, pytest.mark.ppyoloe]

SIZES = tuple(PPYOLOE_CONFIGS)

# Neck output widths per size, in head order (stride 32, 16, 8).
EXPECTED_HEAD_WIDTHS = {
    "s": (384, 192, 96),
    "m": (576, 288, 144),
    "l": (768, 384, 192),
    "x": (960, 480, 240),
}


@pytest.fixture(scope="module")
def tiny_models():
    return {size: LibrePPYOLOEModel(size=size, nb_classes=4).eval() for size in SIZES}


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_head_widths_match_source_rounding(tiny_models, size):
    head = tiny_models[size].head
    assert head.in_channels == EXPECTED_HEAD_WIDTHS[size]


@pytest.mark.parametrize("size", SIZES)
def test_regression_head_emits_four_times_reg_max_plus_one(tiny_models, size):
    for conv in tiny_models[size].head.pred_reg:
        assert conv.out_channels == 4 * (16 + 1) == 68


def test_backbone_returns_strides_8_16_32():
    model = LibrePPYOLOEModel(size="s", nb_classes=4).eval()
    with torch.no_grad():
        feats = model.backbone(torch.zeros(1, 3, 256, 256))
    assert [f.shape[-1] for f in feats] == [32, 16, 8]


def test_neck_emits_head_order_deepest_first():
    model = LibrePPYOLOEModel(size="s", nb_classes=4).eval()
    with torch.no_grad():
        feats = model.neck(model.backbone(torch.zeros(1, 3, 256, 256)))
    # Head consumes stride 32 first, so the neck's first output is the
    # coarsest: 8x8 for a 256px input, then 16x16 (stride 16), then 32x32.
    assert [f.shape[-1] for f in feats] == [8, 16, 32]
    assert [f.shape[1] for f in feats] == list(EXPECTED_HEAD_WIDTHS["s"])


def test_eval_forward_shapes_and_no_objectness():
    model = LibrePPYOLOEModel(size="s", nb_classes=4).eval()
    with torch.no_grad():
        (boxes, scores), raw = model(torch.zeros(2, 3, 640, 640))
    anchors = 20 * 20 + 40 * 40 + 80 * 80
    assert boxes.shape == (2, anchors, 4)
    # 4 class columns, not 4 + 1: the head has no objectness output.
    assert scores.shape == (2, anchors, 4)
    assert raw[1].shape == (2, anchors, 68)
    assert list(raw[4]) == [400, 1600, 6400]


def test_scores_are_sigmoid_probabilities():
    model = LibrePPYOLOEModel(size="s", nb_classes=4).eval()
    with torch.no_grad():
        (_, scores), raw = model(torch.randn(1, 3, 320, 320))
    torch.testing.assert_close(scores, raw[0].sigmoid())
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0


def test_anchor_points_are_cell_centres():
    model = LibrePPYOLOEModel(size="s", nb_classes=4).eval()
    with torch.no_grad():
        feats = model.neck(model.backbone(torch.zeros(1, 3, 320, 320)))
        points, strides = model.head._generate_anchors(feats)
    assert torch.equal(points[0], torch.tensor([0.5, 0.5]))
    assert float(strides[0]) == 32.0
    assert float(strides[-1]) == 8.0


def test_batch_sizes_do_not_share_cached_anchors():
    model = LibrePPYOLOEModel(size="s", nb_classes=4).eval()
    with torch.no_grad():
        (boxes1, _), _ = model(torch.zeros(1, 3, 320, 320))
        (boxes2, _), _ = model(torch.zeros(2, 3, 320, 320))
        (boxes1_again, _), _ = model(torch.zeros(1, 3, 320, 320))
    assert boxes2.shape[0] == 2
    torch.testing.assert_close(boxes1, boxes1_again)
    torch.testing.assert_close(boxes2[0], boxes1[0])


def test_non_square_input_is_supported():
    model = LibrePPYOLOEModel(size="s", nb_classes=4).eval()
    with torch.no_grad():
        (boxes, _), _ = model(torch.zeros(1, 3, 640, 960))
    assert boxes.shape[1] == 20 * 30 + 40 * 60 + 80 * 120


def test_replace_num_classes_keeps_backbone_weights():
    model = LibrePPYOLOEModel(size="s", nb_classes=80)
    stem_before = model.backbone.stem.conv1.seq.conv.weight.clone()
    model.replace_num_classes(3)
    assert model.head.num_classes == 3
    assert [c.out_channels for c in model.head.pred_cls] == [3, 3, 3]
    assert [c.out_channels for c in model.head.pred_reg] == [68, 68, 68]
    torch.testing.assert_close(model.backbone.stem.conv1.seq.conv.weight, stem_before)


def test_unknown_size_rejected():
    with pytest.raises(ValueError, match="Unknown PP-YOLOE size"):
        LibrePPYOLOEModel(size="n")


# ---------------------------------------------------------------------------
# Checkpoint recognition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_detect_size_and_classes_from_state(tiny_models, size):
    state = tiny_models[size].state_dict()
    assert LibrePPYOLOE.can_load(state)
    assert detect_size_from_state(state) == size
    assert detect_nb_classes_from_state(state) == 4
    assert LibrePPYOLOE.detect_size(state) == size
    assert LibrePPYOLOE.detect_nb_classes(state) == 4


def test_module_prefix_is_stripped(tiny_models):
    state = {f"module.{k}": v for k, v in tiny_models["s"].state_dict().items()}
    assert is_upstream_state_dict(state)
    converted = convert_upstream(state)
    assert not any(k.startswith("module.") for k in converted)
    assert LibrePPYOLOE.can_load(converted)


def test_net_wrapper_is_unwrapped(tiny_models):
    checkpoint = {"net": {f"module.{k}": v for k, v in tiny_models["s"].state_dict().items()}}
    state = unwrap_ppyoloe_checkpoint(checkpoint)
    assert is_upstream_state_dict(state)


def test_convert_upstream_rejects_foreign_state():
    with pytest.raises(ValueError, match="does not look like a PP-YOLOE checkpoint"):
        convert_upstream({"backbone.stem.conv1.seq.conv.weight": torch.zeros(1)})


def test_detect_size_rejects_inconsistent_depth(tiny_models):
    """A width signature alone must not decide the size."""
    state = dict(tiny_models["l"].state_dict())
    # Delete stage-1 blocks past index 3 so the depth signature says "m"
    # while the head widths still say "l".
    state = {
        k: v
        for k, v in state.items()
        if not any(k.startswith(f"backbone.stages.1.blocks.{i}.") for i in (4, 5))
    }
    assert detect_size_from_state(state) is None


def test_rejects_sibling_family_states():
    """PP-YOLOE must not claim YOLO-NAS or PicoDet checkpoints, or vice versa."""
    from libreyolo.models.picodet.model import LibrePICODET
    from libreyolo.models.picodet.nn import LibrePICODETModel
    from libreyolo.models.yolonas.model import LibreYOLONAS
    from libreyolo.models.yolonas.nn import LibreYOLONASModel

    ours = LibrePPYOLOEModel(size="s", nb_classes=4).state_dict()
    yolonas = LibreYOLONASModel(config="s", nb_classes=4).state_dict()
    picodet = LibrePICODETModel(size="s", nb_classes=4).state_dict()

    assert LibrePPYOLOE.can_load(yolonas) is False
    assert LibrePPYOLOE.can_load(picodet) is False
    assert LibreYOLONAS.can_load(ours) is False
    assert LibrePICODET.can_load(ours) is False
    assert LibrePPYOLOE.convert_upstream_state_dict(yolonas) is None
    assert LibrePPYOLOE.convert_upstream_state_dict(picodet) is None


# ---------------------------------------------------------------------------
# Filenames and weight hosting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", SIZES)
def test_canonical_and_native_filenames(size):
    assert LibrePPYOLOE.detect_size_from_filename(f"LibrePPYOLOE{size}.pt") == size
    assert LibrePPYOLOE.detect_size_from_filename(f"ppyoloe_{size}_coco.pth") == size


def test_download_url_points_at_source_cdn_not_libreyolo():
    url = LibrePPYOLOE.get_download_url("LibrePPYOLOEs.pt")
    assert url.endswith("/ppyoloe_s_coco.pth")
    assert "huggingface.co/LibreYOLO" not in url


def test_every_size_has_a_pinned_digest():
    for size in SIZES:
        url = LibrePPYOLOE.get_download_url(f"LibrePPYOLOE{size}.pt")
        assert url.rsplit("/", 1)[-1] in LibrePPYOLOE._CHECKPOINT_SHA256


def test_unpinned_download_is_refused(tmp_path):
    blob = tmp_path / "ppyoloe_unknown.pth"
    blob.write_bytes(b"not a checkpoint")
    with pytest.raises(RuntimeError, match="no pinned checksum"):
        LibrePPYOLOE.verify_downloaded_file(
            str(blob), "https://example.invalid/models/ppyoloe_unknown.pth"
        )
    assert not blob.exists()


def test_checksum_mismatch_deletes_the_file(tmp_path):
    blob = tmp_path / "ppyoloe_s_coco.pth"
    blob.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        LibrePPYOLOE.verify_downloaded_file(
            str(blob),
            "https://d2gjn4b69gu75n.cloudfront.net/models/ppyoloe_s_coco.pth",
        )
    assert not blob.exists()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_preprocess_is_a_stretch_resize_with_source_normalization():
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    chw, ratio = preprocess_numpy(img, 640)
    assert chw.shape == (3, 640, 640)
    assert ratio == 1.0
    expected = (255.0 - np.array(PPYOLOE_MEAN)) / np.array(PPYOLOE_STD)
    np.testing.assert_allclose(chw[:, 0, 0], expected, rtol=1e-6)


def test_val_preprocessor_matches_the_inference_normalization():
    from libreyolo.validation.preprocessors import PPYOLOEValPreprocessor

    preprocessor = PPYOLOEValPreprocessor(img_size=(640, 640))
    assert preprocessor.custom_normalization is True
    assert preprocessor.uses_letterbox is False
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 256, size=(73, 121, 3), dtype=np.uint8)
    val_chw, _ = preprocessor(bgr, np.zeros((0, 5), np.float32), (640, 640))
    infer_chw, _ = preprocess_numpy(bgr[:, :, ::-1].copy(), 640)
    np.testing.assert_allclose(val_chw, infer_chw, rtol=1e-5, atol=1e-4)


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


def _fake_output(boxes, scores):
    return (torch.tensor(boxes)[None], torch.tensor(scores)[None])


def test_postprocess_reverses_stretch_independently_on_x_and_y():
    out = _fake_output([[0.0, 0.0, 320.0, 640.0]], [[0.9, 0.1]])
    result = postprocess(out, conf_thres=0.5, input_size=640, original_size=(1280, 320))
    assert result["num_detections"] == 1
    np.testing.assert_allclose(result["boxes"][0], [0.0, 0.0, 640.0, 320.0])
    assert result["classes"] == [0]


def test_postprocess_multi_label_emits_both_classes():
    out = _fake_output([[10.0, 10.0, 20.0, 20.0]], [[0.9, 0.8]])
    multi = postprocess(out, conf_thres=0.5, input_size=640, original_size=(640, 640))
    single = postprocess(
        out, conf_thres=0.5, input_size=640, original_size=(640, 640), multi_label=False
    )
    assert sorted(multi["classes"]) == [0, 1]
    assert single["classes"] == [0]


def test_postprocess_respects_max_det():
    rng = np.random.default_rng(3)
    boxes = rng.uniform(0, 100, size=(50, 4)).astype(np.float32)
    boxes[:, 2:] += 200  # keep x2/y2 > x1/y1 and boxes mostly disjoint
    boxes += np.arange(50, dtype=np.float32)[:, None] * 300
    scores = np.full((50, 1), 0.9, dtype=np.float32)
    out = _fake_output(boxes, scores)
    result = postprocess(
        out, conf_thres=0.5, input_size=640, original_size=(640, 640), max_det=5
    )
    assert result["num_detections"] == 5


def test_postprocess_empty_below_threshold():
    out = _fake_output([[0.0, 0.0, 10.0, 10.0]], [[0.1]])
    result = postprocess(out, conf_thres=0.5, input_size=640, original_size=(640, 640))
    assert result == {"boxes": [], "scores": [], "classes": [], "num_detections": 0}


def test_postprocess_accepts_the_eager_two_level_tuple():
    decoded = _fake_output([[0.0, 0.0, 10.0, 10.0]], [[0.9]])
    result = postprocess(
        (decoded, ("raw", "placeholder")),
        conf_thres=0.5,
        input_size=640,
        original_size=(640, 640),
    )
    assert result["num_detections"] == 1


# ---------------------------------------------------------------------------
# Factory / registry
# ---------------------------------------------------------------------------


def test_family_metadata_is_declared_explicitly():
    assert LibrePPYOLOE.FAMILY == "ppyoloe"
    assert LibrePPYOLOE.FILENAME_PREFIX == "LibrePPYOLOE"
    assert LibrePPYOLOE.SUPPORTED_TASKS == ("detect",)
    assert LibrePPYOLOE.DEFAULT_TASK == "detect"
    assert set(LibrePPYOLOE.INPUT_SIZES) == set(SIZES)
    assert set(LibrePPYOLOE.INPUT_SIZES.values()) == {640}


def test_exported_backends_use_class_aware_nms():
    """Native postprocess uses batched_nms; exported runtimes must match.

    ``_parse_ppyoloe`` emits one candidate per (anchor, class) pair above the
    threshold, so a shared anchor can carry two classes with the same box.
    Class-agnostic NMS downstream would delete the lower-scored class and make
    exported results disagree with native inference.
    """
    import inspect

    from libreyolo.backends import base as backend_base

    source = inspect.getsource(backend_base.BaseBackend)
    marker = source.index("_batched_nms_numpy(boxes, max_scores, class_ids, iou)")
    allowlist = source[source.rindex("if self.model_family in (", 0, marker) : marker]
    assert '"ppyoloe"' in allowlist


def test_postprocess_keeps_overlapping_classes_on_one_anchor():
    """Two classes on the same box must both survive class-aware NMS."""
    out = _fake_output([[10.0, 10.0, 110.0, 110.0]], [[0.9, 0.8]])
    result = postprocess(out, conf_thres=0.5, input_size=640, original_size=(640, 640))
    assert result["num_detections"] == 2
    assert sorted(result["classes"]) == [0, 1]


def test_registered_in_model_groups():
    from libreyolo.models.registry import MODEL_GROUPS

    assert MODEL_GROUPS["ppyoloe"] == "g2"


def test_exported_from_package_root():
    import libreyolo

    assert libreyolo.LibrePPYOLOE is LibrePPYOLOE

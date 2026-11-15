"""Fast FCOS family integration and discriminator tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.models.fcos.model import LibreFCOS
from libreyolo.models.fcos.utils import resize_dimensions
from libreyolo.postprocess.fcos import postprocess
from libreyolo.utils.coco import COCO91_TO_COCO80
from libreyolo.validation.preprocessors import FCOSValPreprocessor


pytestmark = pytest.mark.unit


def _valid_fcos_state() -> dict[str, torch.Tensor]:
    return {
        "head.regression_head.bbox_ctrness.weight": torch.empty(1, 256, 3, 3),
        "head.classification_head.cls_logits.weight": torch.empty(91, 256, 3, 3),
        "backbone.fpn.extra_blocks.p6.weight": torch.empty(256, 256, 3, 3),
        "backbone.fpn.extra_blocks.p7.weight": torch.empty(256, 256, 3, 3),
        "backbone.body.conv1.weight": torch.empty(64, 3, 7, 7),
    }


def test_fcos_checkpoint_detection() -> None:
    state = _valid_fcos_state()
    assert LibreFCOS.can_load(state)
    assert LibreFCOS.detect_size(state) == "r50"
    assert LibreFCOS.detect_nb_classes(state) == 80

    for required in (
        "head.regression_head.bbox_ctrness.weight",
        "head.classification_head.cls_logits.weight",
        "backbone.fpn.extra_blocks.p6.weight",
        "backbone.fpn.extra_blocks.p7.weight",
    ):
        incomplete = dict(state)
        incomplete.pop(required)
        assert not LibreFCOS.can_load(incomplete)


@pytest.mark.parametrize(
    ("family", "state"),
    [
        (
            "resnet",
            {
                "conv1.weight": torch.empty(64, 3, 7, 7),
                "fc.weight": torch.empty(1000, 2048),
                "layer1.0.conv1.weight": torch.empty(64, 64, 3, 3),
            },
        ),
        (
            "rtdetr",
            {
                "backbone.res_layers.0.conv1.weight": torch.empty(1),
                "encoder.input_proj.0.weight": torch.empty(1),
                "decoder.input_proj.0.weight": torch.empty(1),
                "decoder.dec_score_head.0.weight": torch.empty(1),
            },
        ),
        (
            "rtdetrv2",
            {
                "backbone.res_layers.0.conv1.weight": torch.empty(1),
                "encoder.input_proj.0.weight": torch.empty(1),
                "decoder.input_proj.0.weight": torch.empty(1),
                "decoder.dec_score_head.0.weight": torch.empty(1),
                "decoder.decoder.layers.0.cross_attn.num_points_scale": torch.empty(1),
            },
        ),
        ("yolox", {"head.stems.0.conv.weight": torch.empty(1)}),
        (
            "picodet",
            {
                "head.gfl_cls.0.weight": torch.empty(1),
                "backbone.blocks.0.conv.weight": torch.empty(1),
            },
        ),
        ("rtmdet", {"bbox_head.rtm_cls.0.weight": torch.empty(1)}),
        (
            "faster_rcnn",
            {
                "rpn.head.cls_logits.weight": torch.empty(1),
                "roi_heads.box_predictor.cls_score.weight": torch.empty(91, 1024),
                "roi_heads.box_predictor.bbox_pred.weight": torch.empty(364, 1024),
            },
        ),
    ],
)
def test_fcos_rejects_other_registry_families(
    family: str,
    state: dict[str, torch.Tensor],
) -> None:
    assert family
    assert not LibreFCOS.can_load(state)


def test_fcos_postprocess_decodes_and_maps_sparse_coco_label() -> None:
    logits = torch.full((1, 1, 91), -20.0)
    logits[0, 0, 1] = 10.0
    output = {
        "cls_logits": logits,
        "bbox_regression": torch.full((1, 1, 4), 0.5),
        "bbox_ctrness": torch.full((1, 1, 1), 10.0),
        "anchors": torch.tensor([[[0.0, 0.0, 8.0, 8.0]]]),
        "level_sizes": torch.tensor([[1]]),
    }

    result = postprocess(
        output,
        conf_thres=0.2,
        iou_thres=0.6,
        original_size=(8, 8),
        max_det=100,
        class_map=COCO91_TO_COCO80,
        input_size=8,
    )

    assert result["num_detections"] == 1
    np.testing.assert_array_equal(result["boxes"], [[0.0, 0.0, 8.0, 8.0]])
    np.testing.assert_array_equal(result["classes"], [0])
    np.testing.assert_allclose(result["scores"], [torch.sigmoid(torch.tensor(10.0))])


def test_fcos_resize_and_validation_preprocessor_geometry() -> None:
    assert resize_dimensions(576, 768, 800) == (800, 1066, 800 / 576)

    preprocessor = FCOSValPreprocessor(img_size=(32, 32), max_labels=2)
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    targets = np.array([[1.0, 1.0, 4.0, 3.0, 2.0]], dtype=np.float32)
    processed, padded_targets = preprocessor(image, targets, (32, 32))
    scale = 32 / 6
    assert processed.shape == (3, 32, 64)
    np.testing.assert_allclose(
        padded_targets[0],
        [scale, scale, 4 * scale, 3 * scale, 2.0],
    )
    assert preprocessor.letterbox_scale(6, 8, 32) == (scale, 0.0, 0.0)


def test_fcos_public_defaults_and_training_rejection() -> None:
    model = object.__new__(LibreFCOS)
    model._runner_instance = lambda source, **kwargs: (source, kwargs)

    source, defaults = model("image")
    assert source == "image"
    assert defaults["conf"] == 0.2
    assert defaults["iou"] == 0.6
    assert defaults["max_det"] == 100

    _, overrides = model("image", conf=0.4, iou=0.5, max_det=7)
    assert overrides == {"conf": 0.4, "iou": 0.5, "max_det": 7}

    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train()


def test_fixed_canvas_fits_the_image_into_one_square():
    from libreyolo.models.fcos.utils import preprocess_numpy

    image = np.random.default_rng(0).integers(0, 256, (240, 400, 3), dtype=np.uint8)
    chw, ratio = preprocess_numpy(image, 320, fixed_canvas=True)
    assert chw.shape == (3, 320, 320) and ratio == pytest.approx(0.8)
    assert resize_dimensions(240, 400, 320, fixed_canvas=True) == (192, 320, pytest.approx(0.8))
    pair, _ = preprocess_numpy(image, (320, 320), fixed_canvas=True)  # (h, w) from export calibration
    np.testing.assert_array_equal(pair, chw)
    dynamic, _ = preprocess_numpy(image, 320)
    assert dynamic.shape[1:] != (320, 320)  # torchvision's min/max resize stays the default


def test_val_preprocessor_takes_the_fixed_canvas():
    image = np.random.default_rng(1).integers(0, 256, (240, 400, 3), dtype=np.uint8)
    targets = np.array([[10.0, 20.0, 110.0, 120.0, 3.0]], dtype=np.float32)
    fixed = FCOSValPreprocessor(img_size=(320, 320), fixed_canvas=True)
    chw, scaled = fixed(image, targets, (320, 320))
    assert chw.shape == (3, 320, 320)
    np.testing.assert_allclose(scaled[0, :4], targets[0, :4] * 0.8, rtol=1e-6)
    assert fixed.letterbox_scale(240, 400, 320)[0] == pytest.approx(0.8)
    assert not FCOSValPreprocessor(img_size=(320, 320)).fixed_canvas

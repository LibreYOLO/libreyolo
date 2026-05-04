"""Unit tests for model factory heuristics."""

import torch
import pytest

from libreyolo import LibreYOLO
from libreyolo.models import _needs_rfdetr_registration
from libreyolo.models.yolo9.nn import LibreYOLO9Model

pytestmark = pytest.mark.unit


def test_rfdetr_lazy_registration_detects_enc_out_markers():
    weights_dict = {
        "transformer.enc_out_class_embed.0.weight": object(),
        "transformer.enc_out_bbox_embed.0.layers.0.weight": object(),
    }

    assert _needs_rfdetr_registration(weights_dict) is True


def test_rfdetr_lazy_registration_ignores_rtdetr_signature():
    weights_dict = {
        "backbone.stages.0.conv.weight": object(),
        "encoder.input_proj.0.0.weight": object(),
        "decoder.input_proj.0.conv.weight": object(),
        "decoder.dec_score_head.0.weight": object(),
    }

    assert _needs_rfdetr_registration(weights_dict) is False


def test_factory_loads_yolo9_t_metadata_checkpoint_with_coco_class_width(tmp_path):
    model = LibreYOLO9Model(config="t", nb_classes=80)

    # Mimic a fine-tuned checkpoint saved from a COCO-width YOLO9-t model:
    # only the final class conv is rebuilt to 2 classes, while the class
    # branch hidden width stays at 80.
    for seq in model.head.cv3:
        in_channels = seq[-1].weight.shape[1]
        seq[-1] = torch.nn.Conv2d(in_channels, 2, 1)

    ckpt_path = tmp_path / "yolo9_t_best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "nc": 2,
            "names": {0: "red", 1: "white"},
            "model_family": "yolo9",
            "size": "t",
        },
        ckpt_path,
    )

    loaded = LibreYOLO(str(ckpt_path), size="t", device="cpu")

    assert loaded.nb_classes == 2
    assert loaded.names == {0: "red", 1: "white"}
    assert loaded.model.head.cv3[0][0].conv.weight.shape[0] == 80


def test_factory_rejects_unsupported_explicit_task_from_filename():
    with pytest.raises(ValueError, match="not supported"):
        LibreYOLO("LibreYOLOXs.pt", task="segment")

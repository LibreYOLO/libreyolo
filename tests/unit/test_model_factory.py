"""Unit tests for model factory heuristics."""

import pytest

from libreyolo.models import _needs_rfdetr_registration

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

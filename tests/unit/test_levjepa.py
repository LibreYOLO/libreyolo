"""Structural and API tests for the inference-only LeVJEPA family."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.export.support import get_support
from libreyolo.models.levjepa.model import LibreLeVJEPA
from libreyolo.models.levjepa.nn import (
    LeVJEPAConfig,
    LeVJEPAModel,
    _block_causal_mask,
)
from libreyolo.models.levjepa.preprocess import clip_frame_indices
from libreyolo.models.vjepa2.model import LibreVJEPA2


pytestmark = [pytest.mark.unit, pytest.mark.levjepa]


def _tiny(**overrides) -> LeVJEPAConfig:
    values = {
        "img_size": 32,
        "patch_size": 16,
        "num_frames": 2,
        "tubelet_size": 1,
        "embed_dim": 96,
        "depth": 2,
        "num_heads": 3,
        "mlp_ratio": 2.0,
    }
    values.update(overrides)
    return LeVJEPAConfig(**values)


def test_released_config_contract():
    config = LeVJEPAConfig.for_size("l")
    assert config.patch_tokens == 3136
    assert config.embed_dim == 1024
    assert config.depth == 24
    assert config.num_heads == 16
    assert config.attn_mode == "block_causal"


def test_tiny_forward_returns_cls_and_time_major_patches():
    model = LeVJEPAModel(_tiny()).eval()
    output = model(torch.zeros(1, 2, 3, 32, 32))
    assert output.shape == (1, 9, 96)  # CLS + 2 * 2 * 2 patches


def test_block_causal_mask_makes_cls_read_only():
    mask = _block_causal_mask(_tiny())
    assert mask is not None
    assert mask.shape == (9, 9)
    assert mask[0].all()  # CLS reads every patch.
    assert not mask[1:, 0].any()  # No patch reads CLS.
    assert mask[1:5, 1:5].all()  # First frame is bidirectional within-frame.
    assert not mask[1:5, 5:9].any()  # First frame cannot read the future.
    assert mask[5:9, 1:9].all()  # Second frame reads current and past frames.


def test_family_recognition_is_specific_and_rejects_vjepa2():
    state = {
        "encoder.patch_embed.proj.weight": torch.zeros(1024, 3, 1, 16, 16),
        "encoder.cls_token": torch.zeros(1, 1, 1024),
        "encoder.blocks.0.attn.qkv.weight": torch.zeros(3072, 1024),
        "encoder.norm.weight": torch.zeros(1024),
    }
    assert LibreLeVJEPA.can_load(state)
    assert LibreLeVJEPA.detect_size(state) == "l"
    assert LibreLeVJEPA.detect_nb_classes(state) == 1
    assert not LibreVJEPA2.can_load(state)

    vjepa2 = {
        "embeddings.patch_embeddings.proj.weight": torch.zeros(1024, 3, 2, 16, 16),
        "layernorm.weight": torch.zeros(1024),
    }
    assert LibreVJEPA2.can_load(vjepa2)
    assert not LibreLeVJEPA.can_load(vjepa2)


def test_canonical_filename_requires_embed_suffix():
    assert LibreLeVJEPA.detect_size_from_filename("LibreLeVJEPAl-embed.pt") == "l"
    assert LibreLeVJEPA.detect_task_from_filename("LibreLeVJEPAl-embed.pt") == "embed"
    assert LibreLeVJEPA.detect_size_from_filename("LibreLeVJEPAl.pt") is None


def test_centered_sampling_tracks_source_fps():
    indices = clip_frame_indices(300, source_fps=30.0)
    assert len(indices) == 16
    assert indices[1] - indices[0] == 4
    assert indices[-1] - indices[0] == 60
    assert indices[0] > 0


def test_short_clip_holds_last_real_frame():
    indices = clip_frame_indices(5, source_fps=15.0)
    assert indices[:3] == [0, 2, 4]
    assert indices[3:] == [4] * 13


def test_video_sampling_seeks_to_centered_window(monkeypatch):
    import cv2

    class FakeCapture:
        def __init__(self):
            self.position = 0
            self.reads = 0
            self.seek_targets = []

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 300
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_POS_FRAMES:
                return self.position
            raise AssertionError(f"unexpected property: {prop}")

        def set(self, prop, value):
            assert prop == cv2.CAP_PROP_POS_FRAMES
            self.position = int(value)
            self.seek_targets.append(self.position)
            return True

        def read(self):
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            self.position += 1
            self.reads += 1
            return True, frame

        def release(self):
            pass

    capture = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: capture)

    frames = LibreLeVJEPA.sample_clip_frames(None, "long.mp4", 16)

    assert len(frames) == 16
    assert capture.seek_targets == [119]
    assert capture.reads == 61


def test_download_notice_is_explicitly_noncommercial():
    notice = LibreLeVJEPA.get_download_notice("LibreLeVJEPAl-embed.pt", "ignored")
    assert "CC BY-NC 4.0" in notice
    assert "NON-COMMERCIAL" in notice


def test_export_support_matches_validated_video_contract():
    entry = get_support("levjepa", "embed", "torchscript")
    assert entry.tier == "validated"
    assert "16-frame" in entry.constraint
    assert get_support("levjepa", "embed", "ncnn").tier == "blocked"
    assert get_support("levjepa", "embed", "tflite").tier == "blocked"

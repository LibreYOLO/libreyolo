"""Offline smoke test for the YOLO-World open-vocabulary scaffold.

Validates the real architecture + API end-to-end on CPU:
  - YOLOv8 backbone + RepVL-PAN neck + BNContrastive head assemble,
  - text encoder produces normalized embeddings,
  - forward pass produces the right per-level shapes,
  - prompt hot-swap updates the classification head width,
  - full inference call returns a populated Results-like dict.

Does NOT validate accuracy — weights are random-init in this MVP.
See docs/agentic-features/blog/yolo-world-integration.md for scope.

Run: pytest tests/smoke/test_yoloworld_smoke.py -v -m smoke
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.yoloworld import (
    LibreYOLOWorld,
    LibreYOLOWorldModel,
    CLIP_EMBED_DIM,
    MaxSigmoidAttnBlock,
    BNContrastiveHead,
)


pytestmark = pytest.mark.smoke

# Smoke tests use the smallest size so CPU runs in seconds.
SMOKE_SIZE = "s"


def _synthetic_image(size: int = 128) -> Image.Image:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Core module tests
# ---------------------------------------------------------------------------


def test_model_instantiates_and_requires_prompts():
    model = LibreYOLOWorldModel(size=SMOKE_SIZE, imgsz=128)
    assert hasattr(model, "backbone")
    assert hasattr(model, "neck")
    assert hasattr(model, "head")
    assert hasattr(model, "text_encoder")
    with pytest.raises(RuntimeError, match="No text prompts"):
        model(torch.randn(1, 3, 128, 128))


def test_text_encoder_returns_normalized_embeddings():
    model = LibreYOLOWorldModel(size=SMOKE_SIZE, imgsz=128)
    embeds = model.text_encoder.encode(["a photo of a cat", "a photo of a dog", "chair"])
    assert embeds.shape == (3, CLIP_EMBED_DIM)
    norms = embeds.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_max_sigmoid_attn_block_output_shape():
    blk = MaxSigmoidAttnBlock(in_channels=64, out_channels=64, guide_channels=CLIP_EMBED_DIM,
                              embed_channels=64, num_heads=8)
    x = torch.randn(2, 64, 16, 16)
    guide = torch.randn(2, 4, CLIP_EMBED_DIM)  # B=2, N_cls=4
    out = blk(x, guide)
    assert out.shape == (2, 64, 16, 16)


def test_bn_contrastive_head_output_shape():
    head = BNContrastiveHead(embed_dim=CLIP_EMBED_DIM)
    embed = torch.randn(2, CLIP_EMBED_DIM, 8, 8)
    text = torch.randn(2, 7, CLIP_EMBED_DIM)
    text = text / text.norm(dim=-1, keepdim=True)
    logits = head(embed, text)
    assert logits.shape == (2, 7, 8, 8)


def test_forward_produces_three_level_outputs():
    model = LibreYOLOWorldModel(size=SMOKE_SIZE, imgsz=128)
    model.set_prompts(["person", "car", "bicycle", "dog"])
    x = torch.randn(1, 3, 128, 128)
    out = model(x)

    assert len(out["bbox_dists"]) == 3
    assert len(out["cls_logits"]) == 3
    assert out["strides"] == [8.0, 16.0, 32.0]

    # Expected feature sizes at stride 8/16/32 for imgsz=128: 16x16, 8x8, 4x4
    expected = [(16, 16), (8, 8), (4, 4)]
    for i, ((b, c), (eh, ew)) in enumerate(zip(zip(out["bbox_dists"], out["cls_logits"]), expected)):
        assert b.shape[-2:] == (eh, ew)
        assert c.shape[-2:] == (eh, ew)
        assert c.shape[1] == 4  # N_prompts
        assert b.shape[1] == 64  # 4 * reg_max = 4 * 16
        assert torch.isfinite(b).all() and torch.isfinite(c).all()


def test_prompts_are_hot_swappable():
    model = LibreYOLOWorldModel(size=SMOKE_SIZE, imgsz=128)
    model.set_prompts(["cat", "dog"])
    out_a = model(torch.randn(1, 3, 128, 128))
    assert out_a["cls_logits"][0].shape[1] == 2

    model.set_prompts(["red apple", "green apple", "banana", "orange"])
    out_b = model(torch.randn(1, 3, 128, 128))
    assert out_b["cls_logits"][0].shape[1] == 4
    assert model.num_prompts == 4
    assert model.prompts == ["red apple", "green apple", "banana", "orange"]


def test_empty_prompts_rejected():
    model = LibreYOLOWorldModel(size=SMOKE_SIZE, imgsz=128)
    with pytest.raises(ValueError):
        model.set_prompts([])
    with pytest.raises(ValueError):
        model.set_prompts("not a list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Wrapper tests
# ---------------------------------------------------------------------------


def test_wrapper_instantiates_with_prompts():
    model = LibreYOLOWorld(size=SMOKE_SIZE, imgsz=128, prompts=["person", "dog"], device="cpu")
    assert model.prompts == ["person", "dog"]
    assert model.nb_classes == 2


def test_wrapper_inference_returns_detections(tmp_path):
    model = LibreYOLOWorld(size=SMOKE_SIZE, imgsz=128, prompts=["person", "dog"], device="cpu")
    img_path = tmp_path / "sample.jpg"
    _synthetic_image(128).save(img_path)

    result = model(str(img_path), conf=0.0, iou=0.5, max_det=10)
    res = result[0] if isinstance(result, list) else result
    assert hasattr(res, "boxes")
    # Random-init weights → detections may be 0; pipeline must not raise.
    if len(res.boxes) > 0:
        cls_ids = res.boxes.cls.long().tolist()
        assert all(0 <= c < 2 for c in cls_ids)


def test_wrapper_prompt_swap_changes_output_classes(tmp_path):
    model = LibreYOLOWorld(size=SMOKE_SIZE, imgsz=128, prompts=["cat"], device="cpu")
    img_path = tmp_path / "sample.jpg"
    _synthetic_image(128).save(img_path)

    _ = model(str(img_path), conf=0.0, iou=0.5, max_det=5)
    model.set_prompts(["dog", "cat", "bird", "fish", "snake"])
    assert model.nb_classes == 5

    result = model(str(img_path), conf=0.0, iou=0.5, max_det=5)
    res = result[0] if isinstance(result, list) else result
    if len(res.boxes) > 0:
        cls_ids = res.boxes.cls.long().tolist()
        assert all(0 <= c < 5 for c in cls_ids)

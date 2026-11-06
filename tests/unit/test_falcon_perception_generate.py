"""Falcon Perception generation helpers: prompt attention mask and box pairing."""

from types import SimpleNamespace

import pytest
import torch

from libreyolo.models.falcon_perception.generate import (
    build_prompt_mask,
    generate,
    pair_bbox_entries,
)
from libreyolo.models.falcon_perception.nn import FalconPerception, ModelArgs

pytestmark = pytest.mark.unit


def test_prompt_mask_image_block_is_bidirectional():
    # text, [cls, reg, img, img], end_of_image, text
    soi, eoi = 5, 6
    toks = torch.tensor([1, soi, 2, 3, 3, eoi, 4])
    m = build_prompt_mask(toks, soi, eoi)
    assert m[1, 4] and m[2, 3] and m[3, 4] and m[4, 1]  # inside the block, both directions
    assert not m[0, 1]  # text before the image cannot see it
    assert m[5, 1] and m[5, 4]  # eoi is causal and sees the whole block
    assert not m[5, 6] and m[6, 0]  # causal for text
    assert not m[3, 5]  # image tokens do not see the end-of-image token


def test_pair_bbox_entries():
    raw = [{"x": 0.5, "y": 0.5}, {"h": 0.2, "w": 0.1}, {"x": 0.1, "y": 0.2}, {"h": 0.3, "w": 0.4}]
    boxes = pair_bbox_entries(raw)
    assert boxes == [
        {"x": 0.5, "y": 0.5, "h": 0.2, "w": 0.1},
        {"x": 0.1, "y": 0.2, "h": 0.3, "w": 0.4},
    ]
    assert pair_bbox_entries([{"x": 0.5, "y": 0.5}]) == []


@pytest.mark.parametrize("budget", [1, 3])
def test_generation_stops_at_the_requested_budget(budget):
    # Ids: pad 0, eos 1, cls 2, regs 3-6, end-of-image 7, image 30, coord 20, size 21.
    args = ModelArgs(
        max_seq_len=512, dim=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        vocab_size=64, ffn_dim=48, coord_enc_dim=16, coord_dec_dim=32, coord_out_dim=16,
        size_enc_dim=16, size_dec_dim=32, size_out_dim=16, perception_heads=True,
        do_segmentation=False, img_id=30, coord_token_id=20, size_token_id=21,
    )
    torch.manual_seed(0)
    net = FalconPerception(args).eval()
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    tok = SimpleNamespace(
        image_cls_token_id=2, end_of_image_token_id=7, pad_token_id=0, eos_token_id=1,
        seg_token_id=22,
    )
    n_img = 4  # a 32x32 canvas at patch size 16
    tokens = torch.tensor([[9, 2, 3, 4, 5, 6] + [30] * n_img + [7, 10]])
    L = tokens.shape[1]
    pos_t = torch.arange(L)[None]
    pos_hw = torch.zeros(1, L, 2)
    pixel_values = torch.zeros(1, 1, 32, 32, 3)
    pixel_mask = torch.ones(1, 1, 32, 32, dtype=torch.bool)

    gen = generate(
        net, tok, tokens, pos_t, pos_hw, pixel_values, pixel_mask,
        segmentation=False, max_new_tokens=budget, stop_token_ids=[-1],
    )
    # -1 never occurs, so only the budget can end generation. The rounded cache
    # size (128) must not extend it.
    assert gen.tokens.shape[0] == L + budget

"""Falcon Perception network: config, masked attention, KV cache, heads, AnyUp window."""

import pytest
import torch

from libreyolo.models.falcon_perception.nn import (
    FalconPerception,
    KVCache,
    ModelArgs,
    get_model_args,
    masked_attention,
    window2d,
    window_keep_mask,
)

pytestmark = pytest.mark.unit

COORD, SIZE, SEG = 20, 21, 22


def tiny_args(**kw) -> ModelArgs:
    base = dict(
        max_seq_len=256, dim=32, n_layers=2, n_heads=4, head_dim=8, n_kv_heads=2,
        vocab_size=64, ffn_dim=48, coord_enc_dim=16, coord_dec_dim=32, coord_out_dim=16,
        size_enc_dim=16, size_dec_dim=32, size_out_dim=16, perception_heads=True,
        do_segmentation=False, coord_token_id=COORD, size_token_id=SIZE,
        img_id=30,
    )
    base.update(kw)
    return ModelArgs(**base)


def tiny_net(seed: int = 0) -> FalconPerception:
    torch.manual_seed(seed)
    net = FalconPerception(tiny_args()).eval()
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.05)
    with torch.no_grad():
        net.freqs_cis_golden.normal_(std=0.1)
    return net


def test_model_args_per_size():
    big, small = get_model_args("600m"), get_model_args("300m")
    assert (big.n_layers, big.dim, big.head_dim, big.do_segmentation) == (28, 1024, 128, True)
    assert (small.n_layers, small.dim, small.head_dim, small.do_segmentation) == (22, 768, 64, False)
    with pytest.raises(ValueError):
        get_model_args("1b")


def test_masked_attention_matches_sdpa_and_returns_natural_lse():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril()
    out, lse = masked_attention(q, k, v, mask, return_lse=True)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(out, ref, atol=1e-6)
    scores = (q @ k.transpose(-2, -1)) / 8**0.5
    scores = scores.masked_fill(~mask, float("-inf"))
    assert torch.allclose(lse, torch.logsumexp(scores, -1), atol=1e-6)


def test_window_mask_matches_bounds():
    H = W = 8
    h = w = 4
    windows = window2d((h, w), (H, W), 0.1)
    keep = window_keep_mask(0, H * W, W, h, w, windows)
    assert keep.shape == (H * W, h * w)
    assert keep.any(dim=1).all()  # every query sees at least one key
    # chunking is exact
    parts = [window_keep_mask(a, min(a + 10, H * W), W, h, w, windows) for a in range(0, H * W, 10)]
    assert torch.equal(torch.cat(parts), keep)


def test_kv_cache_decode_matches_full_prefill():
    net = tiny_net()
    S = 9
    toks = torch.randint(0, 19, (1, S))
    pos_t = torch.arange(S)[None]
    pos_hw = torch.zeros(1, S, 2)
    causal = torch.ones(S, S, dtype=torch.bool).tril()[None, None]

    full_cache = KVCache(1, 16, 4, 8, 2)
    with torch.inference_mode():
        full_logits, _ = net(toks, causal, full_cache, rope_pos_t=pos_t, rope_pos_hw=pos_hw)

        inc_cache = KVCache(1, 16, 4, 8, 2)
        net(
            toks[:, : S - 1],
            causal[:, :, : S - 1, : S - 1],
            inc_cache,
            rope_pos_t=pos_t,
            rope_pos_hw=pos_hw,
        )
        step_logits, _ = net(toks[:, S - 1 :], torch.ones(1, 1, 1, S, dtype=torch.bool), inc_cache)
    assert torch.allclose(full_logits[:, -1], step_logits[:, -1], atol=1e-5)


def test_coord_and_size_heads():
    net = tiny_net()
    h = torch.randn(2, 32)
    tokens = torch.tensor([COORD, SIZE])
    xy, hw, is_coord, is_size, logits = net.sample_bbox(h, tokens)
    assert xy.shape == hw.shape == (2, 2) and logits.shape == (2, 2, 8)
    assert is_coord.tolist() == [True, False] and is_size.tolist() == [False, True]
    assert ((xy >= 0) & (xy < 1)).all()
    assert ((hw > 0) & (hw <= 1)).all()


def test_dedup_moves_a_repeated_centre():
    logits = torch.zeros(2, 16)
    logits[0, 4], logits[0, 7] = 5.0, 4.0
    logits[1, 8], logits[1, 2] = 5.0, 4.0
    xy = torch.tensor([4 / 16, 8 / 16])
    history = xy.clone()[None]
    FalconPerception.dedup_single_coord(
        xy, torch.tensor(True), history, torch.tensor([True]), logits, threshold=0.01
    )
    # One x-bin and one y-bin are masked per retry, so both coordinates move.
    assert xy.tolist() == [7 / 16, 2 / 16]

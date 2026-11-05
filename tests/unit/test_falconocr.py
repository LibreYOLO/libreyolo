"""Offline Falcon OCR checks: input layout, mask and attention rules, decoding, wiring.

Weight-backed parity against the upstream release runs on CPU and is recorded
in the pull request; nothing here loads the real checkpoint.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreFalconOCR, LibreVLM
from libreyolo.models.falconocr import model as fo_model
from libreyolo.models.falconocr import processing as P
from libreyolo.models.falconocr.network import (
    FalconOCRConfig,
    FalconOCRNetwork,
    KVCache,
    SequenceMask,
    sink_attention,
)

pytestmark = pytest.mark.unit

TOKENS = P.FalconOCRTokens()


# -------------------------------------------------------------- processing


@pytest.mark.parametrize(
    "size,expected",
    [
        ((800, 600), (800, 600)),  # inside [64, 1536]: untouched
        ((3000, 1500), (1536, 768)),  # landscape, too large
        ((1000, 4000), (384, 1536)),  # portrait, too large
        ((538, 40), (860, 64)),  # too thin: short side up to 64
    ],
)
def test_resize_image_if_necessary(size, expected):
    out = P.resize_image_if_necessary(Image.new("RGB", size))
    assert out.size == expected


def test_smart_resize_rounds_half_to_even_and_rejects_extremes():
    assert P.smart_resize_shape(259, 538) == (256, 544)
    assert P.smart_resize_shape(24, 24) == (64, 64)  # min_pixels 56*56
    with pytest.raises(ValueError):
        P.smart_resize_shape(8, 100)
    with pytest.raises(ValueError):
        P.smart_resize_shape(16, 16 * 201)


def test_pixels_are_scaled_to_minus_one_one():
    img = Image.new("RGB", (64, 64), (255, 0, 128))
    px = P.image_to_pixels(img)
    assert px.shape == (64, 64, 3) and px.dtype == np.float32
    np.testing.assert_allclose(px[0, 0], [1.0, -1.0, 128 / 255 * 2 - 1], atol=1e-6)


def test_patchify_is_row_major_with_channel_last_patches():
    px = np.arange(32 * 48 * 3, dtype=np.float32).reshape(32, 48, 3)
    patches = P.patchify(px)
    assert patches.shape == (2 * 3, 16 * 16 * 3)
    # second patch in the first row starts at column 16
    np.testing.assert_array_equal(patches[1, :3].numpy(), px[0, 16])
    np.testing.assert_array_equal(patches[3, -3:].numpy(), px[31, 15])


def test_input_ids_positions_and_grid():
    ids = P.build_input_ids([[], [7, 8]], 6, TOKENS)
    assert ids == [244, 245, 246, 247, 248] + [227] * 6 + [230, 7, 8]
    t, hw = P.positions(torch.tensor(ids), 2, 3, TOKENS)
    # cls opens at 0; registers, patches and end of image share it; text counts on
    assert t.tolist() == [0] * 12 + [1, 2]
    on = ~torch.isnan(hw).any(-1)
    assert on.tolist() == [False] * 5 + [True] * 6 + [False] * 3
    lim = (3 / 2) ** 0.5
    np.testing.assert_allclose(hw[5:11, 1].numpy(), [-lim, 0, lim] * 2, atol=1e-6)
    np.testing.assert_allclose(
        hw[5:11, 0].numpy(), [-1 / lim] * 3 + [1 / lim] * 3, atol=1e-6
    )
    with pytest.raises(ValueError, match="exactly one"):
        P.build_input_ids([[1]], 4, TOKENS)


# ------------------------------------------------------- mask and attention


def _reference_mask(ids, cfg):
    """The release's FlexAttention mask_mod, evaluated pointwise."""
    n = len(ids)
    ids = torch.tensor(ids)
    soi = torch.cumsum(ids == cfg.image_cls_token_id, 0)
    eoi = torch.cumsum(ids == cfg.img_end_id, 0)
    img = (soi - eoi) > 0
    idx = soi * img
    eos = ids == cfg.eos_id
    eos[-1] = True
    cum = torch.cumsum(eos.long(), 0)
    doc = torch.zeros_like(cum)
    doc[1:] = cum[:-1]
    out = torch.zeros(n, n, dtype=torch.bool)
    for q in range(n):
        for k in range(n):
            image = bool(img[q] and img[k] and idx[q] == idx[k])
            causal = k <= q and doc[q] == doc[k]
            out[q, k] = image or causal
    return out


def test_sequence_mask_matches_release_rule():
    cfg = FalconOCRConfig()
    ids = [5, 244, 245, 227, 227, 230, 9, 11, 3, 4]
    dense = SequenceMask(torch.tensor(ids), cfg).dense(0, len(ids), len(ids))
    assert torch.equal(dense, _reference_mask(ids, cfg))
    # image tokens see later image tokens, text after the image is causal
    assert dense[3, 4] and not dense[5, 6] and dense[6, 3]
    # a generated token attends to its document: everything after the eos
    row = SequenceMask(torch.tensor(ids), cfg).dense(
        len(ids), len(ids) + 1, len(ids) + 1
    )
    assert row[0].tolist() == [False] * 8 + [True] * 3
    no_eos = torch.tensor([244, 227, 230, 7])
    assert SequenceMask(no_eos, cfg).dense(4, 5, 5).all()


def test_sink_attention_matches_sink_as_extra_key():
    torch.manual_seed(0)
    cfg = FalconOCRConfig()
    ids = torch.tensor([244, 245, 227, 227, 227, 230, 7, 8, 9])
    mask = SequenceMask(ids, cfg)
    q, k, v = (torch.randn(1, 2, 9, 8, dtype=torch.float64) for _ in range(3))
    sinks = torch.randn(2, dtype=torch.float64)
    out, lse = sink_attention(q, k, v, mask, 0)
    got = out * torch.sigmoid(lse - sinks.view(1, -1, 1)).unsqueeze(-1)
    # Same thing with the sink as one extra key whose logit is the sink and value zero.
    scores = q @ k.transpose(-1, -2) / 8**0.5
    scores = scores.masked_fill(~mask.dense(0, 9, 9), float("-inf"))
    scores = torch.cat([scores, sinks.view(1, 2, 1, 1).expand(1, 2, 9, 1)], -1)
    ref = torch.softmax(scores, -1)[..., :-1] @ v
    torch.testing.assert_close(got, ref)


def test_sink_attention_matches_flex_attention():
    fa = pytest.importorskip("torch.nn.attention.flex_attention")
    if not hasattr(fa, "AuxRequest"):
        pytest.skip("flex_attention without AuxRequest")
    torch.manual_seed(0)
    cfg = FalconOCRConfig()
    ids = torch.tensor(
        [244, 245, 246, 247, 248] + [227] * 6 + [230] + list(range(20, 24))
    )
    n = ids.numel()
    m = _reference_mask(ids.tolist(), cfg)
    block = fa.create_block_mask(
        lambda b, h, qi, ki: m[qi, ki], 1, None, n, n, device="cpu"
    )
    q, k, v = (torch.randn(1, 2, n, 16) for _ in range(3))
    ref, aux = fa.flex_attention(
        q, k, v, block_mask=block, return_aux=fa.AuxRequest(lse=True)
    )
    out, lse = sink_attention(q, k, v, SequenceMask(ids, cfg), 0)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lse, aux.lse, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------- network


def _tiny():
    cfg = FalconOCRConfig(
        dim=32,
        n_layers=2,
        n_heads=4,
        head_dim=8,
        n_kv_heads=2,
        vocab_size=300,
        ffn_dim=48,
        max_seq_len=512,
    )
    torch.manual_seed(0)
    net = FalconOCRNetwork(cfg)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.2)
    torch.nn.init.normal_(net.freqs_cis_golden)
    return net.eval(), cfg


def _tiny_inputs(cfg, grid=(2, 3)):
    tokens = P.FalconOCRTokens()
    ids = torch.tensor(P.build_input_ids([[], [7, 8, 9]], grid[0] * grid[1], tokens))
    pos_t, pos_hw = P.positions(ids, *grid, tokens)
    patches = torch.randn(grid[0] * grid[1], 16 * 16 * 3)
    return ids, pos_t, pos_hw, patches


def test_network_matches_checkpoint_layout():
    with torch.device("meta"):
        net = FalconOCRNetwork(FalconOCRConfig())
    sd = net.state_dict()
    assert len(sd) == 115
    assert tuple(sd["freqs_cis_golden"].shape) == (16, 16, 2)
    assert tuple(sd["layers.0.attention.wqkv.weight"].shape) == (2048, 768)
    assert tuple(sd["layers.21.feed_forward.w13.weight"].shape) == (4608, 768)
    assert tuple(sd["img_projector.weight"].shape) == (768, 768)
    assert "freqs_cis" not in sd


def test_incremental_decode_matches_full_prefill():
    net, cfg = _tiny()
    ids, pos_t, pos_hw, patches = _tiny_inputs(cfg)
    with torch.no_grad():
        new = net.generate(ids, pos_t, pos_hw, patches, max_new_tokens=6, stop_ids=[])
    # 6 rounded up to the 128 block: decoding runs until capacity, as upstream.
    assert len(new) == 128 - len(ids)
    # Re-run the prompt plus the first 3 generated tokens in one causal pass.
    full = torch.cat([ids, torch.tensor(new[:3])])
    t = torch.cat([pos_t, pos_t[-1] + torch.arange(1, 4)])
    hw = torch.cat([pos_hw, torch.full((3, 2), float("nan"))])
    mask = SequenceMask(ids, cfg)
    with torch.no_grad():
        logits = net.prefill(
            full.view(1, -1),
            t.view(1, -1),
            hw.view(1, -1, 2),
            patches,
            mask,
            KVCache(cfg.n_layers, 128),
        )
    assert logits[0, -4:].argmax(-1).tolist() == new[:4]


def test_generate_stops_on_stop_token_and_checks_context():
    net, cfg = _tiny()
    ids, pos_t, pos_hw, patches = _tiny_inputs(cfg)
    with torch.no_grad():
        first = net.generate(
            ids, pos_t, pos_hw, patches, max_new_tokens=4, stop_ids=[]
        )[0]
        out = net.generate(
            ids, pos_t, pos_hw, patches, max_new_tokens=4, stop_ids=[first]
        )
    assert out == [first]
    with pytest.raises(ValueError, match="context"):
        net.generate(ids, pos_t, pos_hw, patches, max_new_tokens=600, stop_ids=[])


def test_patch_count_must_match_image_tokens():
    net, cfg = _tiny()
    ids, pos_t, pos_hw, patches = _tiny_inputs(cfg)
    with pytest.raises(ValueError, match="patch count"):
        net.generate(ids, pos_t, pos_hw, patches[:-1], max_new_tokens=1, stop_ids=[])


# ----------------------------------------------------------- prompts, layout


def test_prompt_resolution():
    plain = fo_model.resolve_prompt(None)
    assert plain == "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>"
    assert "table content" in fo_model.resolve_prompt("Table")
    assert fo_model.resolve_prompt("Read it.") == "<|image|>Read it.\n<|OCR_PLAIN|>"
    assert fo_model.resolve_prompt("<|image|>x") == "<|image|>x"
    with pytest.raises(ValueError):
        fo_model.resolve_prompt("<|image|><|image|>")
    assert fo_model.clean_text(" hi<|end_of_query|><|end_of_text|> ") == "hi"


def test_filter_nested_and_crop_rules():
    dets = [
        {"category": "text", "bbox": [0, 0, 200, 200], "score": 0.9},
        {"category": "display_formula", "bbox": [10, 10, 50, 50], "score": 0.8},
        {"category": "table", "bbox": [0, 300, 400, 500], "score": 0.9},
        {"category": "image", "bbox": [0, 600, 100, 700], "score": 0.9},
        {"category": "title", "bbox": [0, 800, 10, 900], "score": 0.9},
    ]
    kept = fo_model.filter_nested(dets)
    assert [d["category"] for d in kept] == ["text", "table", "image", "title"]
    page = Image.new("RGB", (400, 1000))
    crops = fo_model.layout_crops(page, dets)
    # display_formula has no prompt category upstream, image is skipped,
    # the 10 px wide title is under the 16 px minimum.
    assert [d["category"] for d, _, _ in crops] == ["text", "table"]
    assert crops[1][1].size == (400, 200) and "table content" in crops[1][2]
    whole = fo_model.layout_crops(page, [dets[3]])
    assert whole[0][0] is None and whole[0][1] is page


def test_build_ocr_dict_page_and_regions():
    page = fo_model.build_ocr_dict([(None, "Hello")], (100, 50), 300, 1.0)["ocr"]
    assert page["polygons"] == [[[0, 0], [100, 0], [100, 50], [0, 50]]]
    assert page["labels"] == ["plain"] and page["markdown"] == "Hello"
    items = [
        ({"category": "doc_title", "bbox": [1, 2, 30, 40], "score": 0.7}, "# T"),
        ({"category": "text", "bbox": [1, 50, 30, 60], "score": 0.6}, ""),
        ({"category": "text", "bbox": [1, 70, 30, 90], "score": 0.5}, "body"),
    ]
    out = fo_model.build_ocr_dict(items, (100, 100), 300, 1.0)["ocr"]
    assert out["texts"] == ["# T", "body"] and out["labels"] == ["doc_title", "text"]
    assert out["det_confidences"] == [0.7, 0.5] and out["markdown"] == "# T\n\nbody"


# ------------------------------------------------------------------ wiring


@pytest.fixture
def harness(monkeypatch):
    """A LibreFalconOCR whose weights, layout and generation are stubbed."""
    state = {"calls": []}

    def fake_load(self, snapshot_dir):
        self._tokens = P.FalconOCRTokens()
        self._stop_ids = [11, 263]
        return torch.nn.Linear(1, 1), SimpleNamespace()

    def fake_read(self, image, prompt, max_new_tokens=None):
        state["calls"].append((image.size, prompt))
        return state.get("text", "page text")

    monkeypatch.setattr(LibreFalconOCR, "_ensure_weights", lambda self: "unused")
    monkeypatch.setattr(LibreFalconOCR, "_load_pretrained", fake_load)
    monkeypatch.setattr(LibreFalconOCR, "_read", fake_read)
    monkeypatch.setattr(
        LibreFalconOCR,
        "_detect_layout",
        lambda self, image: [
            {"category": "doc_title", "bbox": [0, 0, 300, 40], "score": 0.9},
            {"category": "table", "bbox": [0, 60, 300, 200], "score": 0.8},
        ],
    )
    return state


@pytest.mark.parametrize("alias", ["falcon-ocr", "FalconOCR", "falcon-ocr-270m"])
def test_factory_and_defaults(alias, harness):
    model = LibreVLM(alias, device="cpu")
    assert isinstance(model, LibreFalconOCR)
    assert model.task == "ocr" and model.names == {0: "text"}
    assert model.prompt == fo_model.resolve_prompt("plain") and not model.layout
    assert model.HF_REPOS["270m"] == "LibreYOLO/LibreFalconOCR270m"
    assert len(model.HF_REVISIONS["270m"]) == 40


def test_predict_whole_page_and_layout(harness):
    model = LibreVLM("falcon-ocr", device="cpu", prompt="table")
    result = model.predict(Image.new("RGB", (300, 400), "white"))
    assert harness["calls"][-1] == ((300, 400), fo_model.resolve_prompt("table"))
    assert result.boxes is None and len(result.ocr) == 1
    assert result.ocr.texts == ["page text"] and result.ocr.labels == ["plain"]
    assert result.ocr.markdown == "page text"
    np.testing.assert_allclose(result.ocr.xyxy[0].numpy(), [0, 0, 300, 400])

    layout = LibreVLM("falcon-ocr", device="cpu", layout=True)
    res = layout.predict(Image.new("RGB", (300, 400), "white"))
    assert [c[0] for c in harness["calls"][-2:]] == [(300, 40), (300, 140)]
    assert "title content" in harness["calls"][-2][1]
    assert res.ocr.labels == ["doc_title", "table"]
    assert res.ocr.markdown == "page text\n\npage text"


def test_chat_and_unsupported(harness):
    model = LibreVLM("falcon-ocr", device="cpu")
    assert model.chat(Image.new("RGB", (8, 8)), "formula") == "page text"
    assert "formula content" in harness["calls"][-1][1]
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="x.yaml")
    with pytest.raises(NotImplementedError):
        model.export()


def test_pins_inventory_and_registry():
    from libreyolo.models.inventory import collect_model_inventory
    from libreyolo.models.registry import MODEL_GROUPS

    assert not LibreFalconOCR.TRUST_REMOTE_CODE
    assert set(LibreFalconOCR.HF_REPOS) == set(LibreFalconOCR.HF_REVISIONS) == {"270m"}
    assert MODEL_GROUPS["falconocr"] == "s"
    entry = collect_model_inventory()["falconocr"]
    assert entry["optional_extra"] == "vlm" and entry["tasks"] == ["ocr"]

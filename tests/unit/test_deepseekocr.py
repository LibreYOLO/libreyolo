"""Offline DeepSeek-OCR checks: parsing, input layout, decoding rules, wiring.

Weight-backed token parity against the upstream release runs on a GPU and is
recorded in the pull request; nothing here loads the real checkpoint.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreDeepSeekOCR, LibreVLM
from libreyolo.models.deepseekocr import model as ds_model
from libreyolo.models.deepseekocr import parsing, processing
from libreyolo.models.deepseekocr.language import (
    DeepSeekOCRTextConfig,
    NoRepeatNGram,
    causal_mask,
)
from libreyolo.models.deepseekocr.network import DeepSeekOCRNetwork
from libreyolo.utils.results import OCRRegions, Results

pytestmark = pytest.mark.unit

OCR_OUTPUT = (
    "<|ref|>Quarterly Report 2026<|/ref|><|det|>[[81, 64, 633, 98]]<|/det|>\n"
    "<|ref|>Schedule project amount<|/ref|><|det|>[[78, 138, 712, 162]]<|/det|>"
    "<｜end▁of▁sentence｜>"
)
MARKDOWN_OUTPUT = (
    "<|ref|>sub_title<|/ref|><|det|>[[78, 61, 637, 102]]<|/det|>\n"
    "## Quarterly Report 2026  \n\n"
    "<|ref|>image<|/ref|><|det|>[[10, 500, 300, 700]]<|/det|>\n\n"
    "<|ref|>text<|/ref|><|det|>[[74, 135, 773, 572]]<|/det|>\n"
    "Line one \\coloneqq two<｜end▁of▁sentence｜>"
)


# ----------------------------------------------------------------- parsing


def test_ocr_prompt_output_gives_one_line_per_box():
    items = parsing.parse_grounding(OCR_OUTPUT)
    assert [i["text"] for i in items] == [
        "Quarterly Report 2026",
        "Schedule project amount",
    ]
    assert [i["label"] for i in items] == ["text", "text"]
    assert items[0]["box"] == [81.0, 64.0, 633.0, 98.0]
    assert parsing.to_page_text(OCR_OUTPUT) == ""


def test_markdown_output_uses_trailing_text_and_drops_figures():
    items = parsing.parse_grounding(MARKDOWN_OUTPUT)
    assert [(i["label"], i["text"]) for i in items] == [
        ("sub_title", "## Quarterly Report 2026"),
        ("text", "Line one \\coloneqq two"),
    ]
    page = parsing.to_page_text(MARKDOWN_OUTPUT)
    assert page == "## Quarterly Report 2026  \n\n\n\n\nLine one := two"


def test_figure_only_markdown_page_has_no_text_regions():
    figure = "<|ref|>image<|/ref|><|det|>[[10, 10, 500, 500]]<|/det|>"
    assert parsing.parse_grounding(figure, layout=True) == []
    assert [i["text"] for i in parsing.parse_grounding(figure, layout=False)] == ["image"]
    out = ds_model.build_ocr_dict(figure, (100, 100), 300, 1.0, layout=True)["ocr"]
    assert out["polygons"] == [] and out["markdown"] == ""


def test_multi_box_det_truncation_and_garbage():
    text = (
        "<|ref|>a<|/ref|><|det|>[[0, 0, 10, 10], [20, 20, 30, 30]]<|/det|>"
        "<|ref|>bad<|/ref|><|det|>[[1, 2]]<|/det|>"
        "<|ref|>x<|/ref|><|det|>not a list<|/det|>"
        "<|ref|>cut off<|/ref|><|det|>[[1, 2,"
    )
    items = parsing.parse_grounding(text)
    assert [i["box"] for i in items] == [
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 20.0, 30.0, 30.0],
    ]
    assert parsing.to_page_text(text) == ""


def test_grid_box_to_pixels_scales_by_999_orders_and_clamps():
    assert parsing.grid_box_to_pixels([999, 0, 0, 999], 1000, 500) == (
        0.0,
        0.0,
        1000.0,
        500.0,
    )
    x1, y1, x2, y2 = parsing.grid_box_to_pixels([81, 64, 633, 98], 1000, 1300)
    assert x1 == pytest.approx(81 / 999 * 1000) and y2 == pytest.approx(98 / 999 * 1300)
    assert parsing.grid_box_to_pixels([-5, 0, 2000, 10], 100, 100)[::2] == (0.0, 100.0)


def test_build_ocr_dict_reading_order_scores_and_markdown():
    text = (
        "<|ref|>right<|/ref|><|det|>[[600, 100, 900, 150]]<|/det|>"
        "<|ref|>left<|/ref|><|det|>[[0, 100, 300, 150]]<|/det|>"
        "<|ref|>top<|/ref|><|det|>[[0, 0, 300, 50]]<|/det|>"
    )
    out = ds_model.build_ocr_dict(text, (999, 999), max_det=300, score=1.0)["ocr"]
    assert out["texts"] == ["top", "left", "right"]
    assert out["confidences"] == [1.0] * 3 and out["det_confidences"] == [1.0] * 3
    assert out["labels"] == ["text"] * 3
    assert out["polygons"][0] == [[0.0, 0.0], [300.0, 0.0], [300.0, 50.0], [0.0, 50.0]]
    assert out["markdown"] == ""
    capped = ds_model.build_ocr_dict(text, (999, 999), max_det=1, score=1.0)["ocr"]
    assert capped["texts"] == ["right"]
    free = ds_model.build_ocr_dict("Plain page text", (10, 10), 300, 1.0)["ocr"]
    assert free["polygons"] == [] and free["markdown"] == "Plain page text"


# -------------------------------------------------------------- processing


@pytest.mark.parametrize(
    "size,resolution,grid,tokens",
    [
        # The release laid out 903 placeholders for this page in Gundam mode
        # (910 input ids: BOS, 903 image tokens, 6 prompt tokens).
        ((1000, 1300), "gundam", (2, 3), 903),
        ((1000, 1300), "base", (1, 1), 273),
        ((1000, 1300), "tiny", (1, 1), 73),
        ((1000, 1300), "small", (1, 1), 111),
        ((1000, 1300), "large", (1, 1), 421),
        ((640, 480), "gundam", (1, 1), 273),
    ],
)
def test_views_and_token_layout(size, resolution, grid, tokens):
    views = processing.prepare_views(Image.new("RGB", size, "white"), resolution)
    assert views.grid == grid
    if resolution == "gundam" and grid != (1, 1):
        assert views.crops.shape == (grid[0] * grid[1], 3, 640, 640)
        assert views.global_view.shape == (1, 3, 1024, 1024)
    else:
        assert views.crops is None
    assert views.num_tokens == tokens
    # Pad colour is the normalized mean, i.e. 127/255 mapped to about -0.004.
    assert (
        float(views.global_view.min()) >= -1.0 and float(views.global_view.max()) <= 1.0
    )


def test_unknown_resolution_and_prompt_placeholders():
    with pytest.raises(ValueError, match="resolution"):
        processing.resolve_resolution("huge")
    assert processing.normalize_prompt(" Free OCR. ") == "<image>\nFree OCR."
    with pytest.raises(ValueError, match="exactly one"):
        processing.normalize_prompt("<image><image>")


class _FakeTokenizer:
    add_bos = True
    bos_token_id = 0
    eos_token_id = 1
    image_token_id = 128815

    def encode(self, text):
        return [ord(c) for c in text]


def test_build_input_ids_places_bos_image_tokens_and_text():
    ids, mask = processing.build_input_ids(_FakeTokenizer(), "<image>\nOCR.", 5)
    assert ids == [0] + [128815] * 5 + [ord(c) for c in "\nOCR."]
    assert mask == [False] + [True] * 5 + [False] * 5
    tok = _FakeTokenizer()
    tok.add_bos = False
    ids, _ = processing.build_input_ids(tok, "<image>\nOCR.", 2)
    assert ids[0] == 128815


def test_tokenizer_reads_bos_from_config(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers

    vocab = {"<s>": 0, "</s>": 1, "<image>": 2, "a": 3, "b": 4, " ": 5}
    tok = tokenizers.Tokenizer(models.WordLevel(vocab, unk_token="a"))
    tok.pre_tokenizer = pre_tokenizers.Split(" ", "isolated")
    tok.add_special_tokens(["<s>", "</s>", "<image>"])
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"add_bos_token": True, "bos_token": "</s>", "eos_token": "<s>"})
    )
    wrapped = processing.DeepSeekOCRTokenizer(tmp_path)
    assert (wrapped.bos_token_id, wrapped.eos_token_id, wrapped.image_token_id) == (
        1,
        0,
        2,
    )
    assert wrapped.encode("a b") == [3, 5, 4]


# ---------------------------------------------------------------- decoding


def _reference_banned(tokens, n):
    """Recompute-from-scratch no_repeat_ngram rule (the standard definition)."""
    if len(tokens) + 1 < n:
        return []
    grams = {}
    for gram in zip(*[tokens[i:] for i in range(n)]):
        grams.setdefault(tuple(gram[:-1]), []).append(gram[-1])
    return sorted(set(grams.get(tuple(tokens[len(tokens) + 1 - n :]), [])))


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_incremental_no_repeat_ngram_matches_reference(n):
    rng = np.random.default_rng(n)
    seq = rng.integers(0, 4, size=60).tolist()
    ngram = NoRepeatNGram(n, seq[:10])
    for tok in seq[10:]:
        assert ngram.banned() == _reference_banned(ngram.tokens, n)
        ngram.append(tok)


def test_causal_mask_values():
    mask = causal_mask(3, torch.bfloat16, torch.device("cpu"))
    assert mask.shape == (1, 1, 3, 3) and mask.dtype == torch.bfloat16
    assert torch.all(mask[0, 0].tril() == 0)
    assert mask[0, 0, 0, 1] == torch.finfo(torch.bfloat16).min


def test_network_matches_checkpoint_layout():
    with torch.device("meta"):
        net = DeepSeekOCRNetwork(DeepSeekOCRTextConfig())
    keys = set(net.state_dict())
    assert len(keys) == 2710
    for key in (
        "lm_head.weight",
        "model.view_seperator",
        "model.projector.layers.weight",
        "model.sam_model.blocks.11.attn.rel_pos_h",
        "model.vision_model.transformer.layers.23.self_attn.qkv_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.11.mlp.experts.63.down_proj.weight",
        "model.layers.11.mlp.shared_experts.up_proj.weight",
        "model.layers.1.mlp.gate.weight",
    ):
        assert key in keys
    assert not any("inv_freq" in k or "position_ids" in k for k in keys)


def test_text_config_rejects_unsupported_routing(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"use_mla": True}))
    with pytest.raises(ValueError, match="latent attention"):
        DeepSeekOCRTextConfig.from_json(path)
    path.write_text(
        json.dumps({"language_config": {"hidden_size": 64}, "vocab_size": 7})
    )
    cfg = DeepSeekOCRTextConfig.from_json(path)
    assert (cfg.hidden_size, cfg.vocab_size) == (64, 7)


# ------------------------------------------------------------------ wiring


@pytest.fixture
def harness(monkeypatch):
    """A LibreDeepSeekOCR whose weights and generation are stubbed."""
    state = {}

    def fake_load(self, snapshot_dir):
        tok = SimpleNamespace(
            eos_token_id=1, decode=lambda ids: state["text"], image_token_id=2
        )
        return torch.nn.Linear(1, 1), tok

    monkeypatch.setattr(LibreDeepSeekOCR, "_ensure_weights", lambda self: "unused")
    monkeypatch.setattr(LibreDeepSeekOCR, "_load_pretrained", fake_load)
    monkeypatch.setattr(
        LibreDeepSeekOCR,
        "_generate",
        lambda self, image, prompt, max_new_tokens=None: state.setdefault(
            "calls", []
        ).append((image.size, prompt, self.resolution))
        or [5],
    )
    return state


@pytest.mark.parametrize("alias", ["deepseek-ocr", "DeepSeekOCR"])
def test_factory_and_defaults(alias, harness):
    model = LibreVLM(alias, device="cpu")
    assert isinstance(model, LibreDeepSeekOCR)
    assert model.task == "ocr" and model.names == {0: "text"}
    assert model.resolution == "gundam" and model.prompt == ds_model.PROMPTS["ocr"]
    assert model.HF_REPOS["1"] == "LibreYOLO/LibreDeepSeekOCR1"
    assert len(model.HF_REVISIONS["1"]) == 40


def test_predict_returns_ocr_regions_and_markdown(harness):
    model = LibreVLM("deepseek-ocr", device="cpu", resolution="base")
    harness["text"] = OCR_OUTPUT
    result = model.predict(Image.new("RGB", (999, 999), "white"))
    assert harness["calls"][-1] == ((999, 999), ds_model.PROMPTS["ocr"], "base")
    assert result.boxes is None and len(result.ocr) == 2
    assert result.ocr.texts == ["Quarterly Report 2026", "Schedule project amount"]
    np.testing.assert_allclose(result.ocr.xyxy[0].numpy(), [81, 64, 633, 98])
    assert result.ocr.markdown == ""
    harness["text"] = MARKDOWN_OUTPUT
    md = model.set_prompt("markdown").predict(Image.new("RGB", (999, 999), "white"))
    assert harness["calls"][-1][1] == ds_model.PROMPTS["markdown"]
    assert md.ocr.labels == ["sub_title", "text"]
    assert md.ocr.markdown.startswith("## Quarterly Report 2026")
    harness["text"] = "just text"
    free = model.set_prompt("free").predict(Image.new("RGB", (64, 64)))
    assert len(free.ocr) == 0 and free.ocr.markdown == "just text"


def test_chat_custom_prompt_and_unsupported(harness):
    model = LibreVLM("deepseek-ocr", device="cpu", prompt="Parse the figure.")
    assert model.prompt == "Parse the figure."
    harness["text"] = "figure<｜end▁of▁sentence｜>"
    assert model.chat(Image.new("RGB", (8, 8)), "Describe.") == "figure"
    with pytest.raises(ValueError, match="resolution"):
        model.resolution = "huge"
    with pytest.raises(NotImplementedError, match="inference-only"):
        model.train(data="x.yaml")
    with pytest.raises(NotImplementedError):
        model.export()


def test_ocr_regions_markdown_survives_moves_and_slices():
    regions = OCRRegions(
        np.zeros((2, 4, 2), dtype=np.float32),
        ["a", "b"],
        labels=["t", "u"],
        markdown="# p",
    )
    for moved in (regions.cpu(), regions.numpy(), regions[1:], regions[0]):
        assert moved.markdown == "# p"
    result = Results(boxes=None, orig_shape=(4, 4), ocr=regions)
    assert result.ocr.markdown == "# p" and result.ocr.labels == ["t", "u"]


def test_pins_inventory_and_registry():
    from libreyolo.models.inventory import collect_model_inventory
    from libreyolo.models.registry import MODEL_GROUPS

    assert not LibreDeepSeekOCR.TRUST_REMOTE_CODE
    assert (
        set(LibreDeepSeekOCR.HF_REPOS)
        == set(LibreDeepSeekOCR.HF_REVISIONS)
        == {"1", "2"}
    )
    assert MODEL_GROUPS["deepseekocr"] == "s"
    entry = collect_model_inventory()["deepseekocr"]
    assert entry["optional_extra"] == "vlm"
    assert entry["tasks"] == ["ocr"]
    assert set(processing.RESOLUTIONS) == {"tiny", "small", "base", "large", "gundam"}


# ------------------------------------------------------- DeepSeek-OCR-2 (size 2)


@pytest.mark.parametrize(
    "size,resolution,grid,tokens",
    [
        # 1024 global view (256) + separator + 144 per 768 crop, no newlines.
        ((1000, 1300), "gundam", (2, 3), 256 + 1 + 6 * 144),
        ((1000, 1300), "base", (1, 1), 257),
        ((700, 500), "gundam", (1, 1), 257),
    ],
)
def test_v2_views_and_token_layout(size, resolution, grid, tokens):
    views = processing.prepare_views(
        Image.new("RGB", size, "white"), resolution, processing.V2
    )
    assert views.grid == grid and views.num_tokens == tokens
    if grid != (1, 1):
        assert views.crops.shape == (grid[0] * grid[1], 3, 768, 768)
    with pytest.raises(ValueError, match="resolution"):
        processing.prepare_views(Image.new("RGB", size), "tiny", processing.V2)


def test_v2_flow_mask():
    from libreyolo.models.deepseekocr.encoder2 import flow_mask

    mask = flow_mask(5, 2, torch.float32, torch.device("cpu"))[0, 0]
    allowed = mask == 0
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(allowed, expected)


def test_v2_network_matches_checkpoint_layout():
    with torch.device("meta"):
        net = DeepSeekOCRNetwork(DeepSeekOCRTextConfig(), version=2)
    keys = set(net.state_dict())
    assert len(keys) == 2707
    for key in (
        "model.qwen2_model.query_768.weight",
        "model.qwen2_model.model.model.layers.23.self_attn.q_proj.bias",
        "model.qwen2_model.model.model.norm.weight",
        "model.projector.layers.weight",
    ):
        assert key in keys
    assert "model.image_newline" not in keys
    assert net.model.sam_model.net_3.out_channels == 896


def test_v2_factory(harness):
    model = LibreVLM("deepseek-ocr-2", device="cpu")
    assert model.size == "2" and model.resolution == "gundam"
    assert model.HF_REPOS["2"] == "LibreYOLO/LibreDeepSeekOCR2"
    with pytest.raises(ValueError, match="resolution"):
        model.resolution = "large"

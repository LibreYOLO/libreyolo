"""Offline WeVisOne-VL prompt, grammar, pixel-decoder and public-contract checks."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from libreyolo import LibreVLM, LibreWeVisOneVL
from libreyolo.models.wevisone import parsing, prompts
from libreyolo.models.wevisone.feature_bridge import premerge_tokens_to_2d
from libreyolo.models.wevisone.mask_decoder import (
    Qwen3VLSegConfig,
    build_mask_decoder,
)

pytestmark = pytest.mark.unit

SEG_ID = 99
VIS_DIM, TEXT_DIM, GRID = 8, 12, 4
PREFIX_TOKENS = 3
TINY_CFG = {
    "vision_feature_layers": [0, 1, 2, 3],
    "hidden_dim": 16,
    "mask_feature_dim": 8,
    "num_fourier_freqs": 4,
    "msda_encoder_layers": 1,
    "msda_num_levels": 4,
    "msda_num_points": 2,
    "transformer_layers": 1,
    "transformer_heads": 2,
    "transformer_ffn_dim": 32,
    "group_norm_groups": 4,
    "stem_mid_dim": 8,
    "alpha_cascade_dim": 8,
    "alpha_cascade_blocks": 1,
    "cnn_image_size": 64,
}


# =============================================================================
# Grammar and prompts
# =============================================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "<ref>person</ref><box><12><40><380><990>,<500><42><880><991></box>",
            [
                ("person", "box", [12, 40, 380, 990]),
                ("person", "box", [500, 42, 880, 991]),
            ],
        ),
        (
            "<ref>a</ref><box><1><2><3><4></box>, <ref>b</ref><box><5><6><7><8></box>",
            [("a", "box", [1, 2, 3, 4]), ("b", "box", [5, 6, 7, 8])],
        ),
        (
            "<ref>a</ref><box><1><2><3><4>, <ref>b</ref><box><5><6><7><8>",
            [("a", "box", [1, 2, 3, 4]), ("b", "box", [5, 6, 7, 8])],
        ),
        ("<ref>cup</ref><point><512><640></point>", [("cup", "point", [512, 640])]),
        ("<ref>dog</ref><box><1><2><3><4><OBJ></box>", [("dog", "box", [1, 2, 3, 4])]),
        (
            "<ref>EXIT</ref><polygon><0><0><100><0><100><100></polygon>",
            [("EXIT", "polygon", [0, 0, 100, 0, 100, 100])],
        ),
        (
            "<box><1><2><3><4></box><|im_end|><box><5><6><7><8></box>",
            [(None, "box", [1, 2, 3, 4])],
        ),
        ("<ref>x</ref><box><10><10><5><20></box>", []),
        ("<ref>x</ref><box><10><10><50><1001></box>", []),
        ("<ref>x</ref><box><10><10><50></box>", []),
        ("None", []),
        ("", []),
    ],
)
def test_parse_shapes(text, expected):
    items = parsing.parse_shapes(text)
    assert [(i["label"], i["kind"], i["coords"]) for i in items] == expected


def test_obj_boxes_keep_their_ordinal():
    text = (
        "<ref>dog</ref><box><100><100><500><500><OBJ>,<600><600><500><500><OBJ></box>"
        "<ref>cat</ref><box><0><0><1000><1000><OBJ></box><|im_end|>"
    )
    assert parsing.parse_obj_boxes(text) == [
        ("dog", (0.1, 0.1, 0.5, 0.5)),
        None,
        ("cat", (0.0, 0.0, 1.0, 1.0)),
    ]
    assert parsing.parse_obj_boxes("None") == []


def test_prompts_match_the_released_instructions():
    assert prompts.SYSTEM_PROMPT == "You are a helpful assistant."
    assert prompts.detection_prompt(["person", " dog "]) == (
        "Locate all the instances that match the following description: person. dog."
    )
    assert prompts.referring_prompt("the man in red") == (
        "Locate the instances that match the following description: the man in red."
    )
    assert prompts.referring_prompt("which cup is full?").endswith("full?")
    assert prompts.point_prompt(["cup"]) == "point to: cup"
    assert prompts.matting_prompt("person") == (
        "Locate the instances that match the following description and "
        "provide the alpha mattes: person."
    )
    assert prompts.OCR_PROMPT == "detect all the word in the box format"


# =============================================================================
# Offline model harness
# =============================================================================


class _Batch(dict):
    def to(self, device):
        return _Batch({k: v.to(device) for k, v in self.items()})


class _Visual(nn.Module):
    """Qwen3-VL vision tower layout: DeepStack mergers plus a final merger."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(VIS_DIM, VIS_DIM) for _ in range(4))
        self.deepstack_visual_indexes = [0, 1, 2]
        self.deepstack_merger_list = nn.ModuleList(
            nn.Linear(VIS_DIM, TEXT_DIM) for _ in range(3)
        )
        self.merger = nn.Linear(VIS_DIM, TEXT_DIM)

    def forward(self, pixel_values):
        x = pixel_values
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.deepstack_visual_indexes:
                self.deepstack_merger_list[self.deepstack_visual_indexes.index(idx)](x)
        return self.merger(x)


class _Language(nn.Module):
    """Hidden state at position ``p`` is the constant ``p``."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity()])

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32)
        hidden = positions[None, :, None].expand(input_ids.shape[0], -1, TEXT_DIM)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _FakeQwen(nn.Module):
    def __init__(self, harness):
        super().__init__()
        self.harness = harness
        self.model = nn.Module()
        self.model.visual = _Visual()
        self.model.language_model = _Language()
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(hidden_size=VIS_DIM, spatial_merge_size=2),
            text_config=SimpleNamespace(hidden_size=TEXT_DIM),
        )

    def generate(self, **kwargs):
        self.harness.generations.append(kwargs)
        return torch.cat([kwargs["input_ids"], torch.tensor([[42]])], dim=1)

    def forward(self, input_ids, pixel_values, image_grid_thw, **kwargs):
        self.harness.teacher_calls.append(kwargs)
        self.model.visual(pixel_values)
        self.model.language_model(input_ids)
        return SimpleNamespace(logits=torch.zeros(1, 1, 4))


class _Tokenizer:
    def get_vocab(self):
        return {parsing.OBJ_TOKEN: SEG_ID}

    def convert_tokens_to_ids(self, token):
        return SEG_ID if token == parsing.OBJ_TOKEN else 0


class _Processor:
    def __init__(self, harness):
        self.harness = harness
        self.tokenizer = _Tokenizer()
        self._prompt = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        self.harness.messages.append(messages)
        prompt = messages[1]["content"][1]["text"]
        if add_generation_prompt:
            self.harness.prompts.append(prompt)
            self._prompt = prompt
            return prompt
        return prompt + messages[2]["content"]

    def __call__(self, text, images, padding, return_tensors):
        objs = text[0].count(parsing.OBJ_TOKEN)
        ids = list(range(1, PREFIX_TOKENS + 1)) + [7, SEG_ID] * objs
        generator = torch.Generator().manual_seed(0)
        return _Batch(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
            pixel_values=torch.randn(GRID * GRID, VIS_DIM, generator=generator),
            image_grid_thw=torch.tensor([[1, GRID, GRID]]),
        )

    def batch_decode(self, tokens, skip_special_tokens, clean_up_tokenization_spaces):
        assert tokens.tolist() == [[42]]
        assert skip_special_tokens is False
        return [self.harness.responses.get(self._prompt, "None") + "<|im_end|>"]


def _write_sidecar(directory: Path, seg_token_id=SEG_ID) -> None:
    torch.manual_seed(0)
    decoder = build_mask_decoder(
        VIS_DIM, TEXT_DIM, Qwen3VLSegConfig.from_dict(TINY_CFG)
    )
    torch.save(
        {
            "state_dict": decoder.state_dict(),
            "cfg": TINY_CFG,
            "seg_token_id": seg_token_id,
            "spatial_merge_size": 2,
        },
        directory / "mask_decoder.pt",
    )


@pytest.fixture
def harness(monkeypatch, tmp_path):
    state = SimpleNamespace(
        responses={}, prompts=[], generations=[], messages=[], teacher_calls=[]
    )
    _write_sidecar(tmp_path)

    def load_pretrained(self, snapshot_dir):
        model, processor = _FakeQwen(state), _Processor(state)
        self._load_mask_decoder(Path(snapshot_dir), model, processor)
        return model, processor

    monkeypatch.setattr(LibreWeVisOneVL, "_ensure_weights", lambda self: str(tmp_path))
    monkeypatch.setattr(LibreWeVisOneVL, "_load_pretrained", load_pretrained)
    return state


IMAGE_SIZE = (400, 200)


def _image():
    return Image.new("RGB", IMAGE_SIZE)


# =============================================================================
# Public API
# =============================================================================


@pytest.mark.parametrize(
    "alias,size",
    [
        ("wevisone-vl", "2b"),
        ("wevisone", "2b"),
        ("wevisone-vl-2b", "2b"),
        ("wevisone-vl-4b", "4b"),
    ],
)
def test_factory(alias, size, harness):
    model = LibreVLM(alias, names=["boat"], device="cpu")
    assert isinstance(model, LibreWeVisOneVL)
    assert model.size == size
    assert model.task == "detect"
    assert model.seg_token_id == SEG_ID
    assert model.decoder_cfg.cnn_image_size == 64


def test_detect_parses_tagged_boxes(harness):
    detect = prompts.detection_prompt(["person", "dog"])
    harness.responses[detect] = (
        "<ref>person</ref><box><100><200><300><400></box>, "
        "<ref>dog</ref><box><500><500><900><1000></box>"
    )
    model = LibreWeVisOneVL(names=["person", "dog"], device="cpu")
    result = model.predict(_image())
    assert harness.prompts == [detect]
    np.testing.assert_allclose(
        result.boxes.xyxy.cpu(), [[40, 40, 120, 80], [200, 100, 360, 200]]
    )
    assert result.boxes.cls.tolist() == [0, 1]
    assert result.masks is None
    generation = harness.generations[-1]
    assert generation["max_new_tokens"] == 4096 and generation["do_sample"] is False
    assert generation["repetition_penalty"] == 1.05
    assert harness.messages[-1][0] == {
        "role": "system",
        "content": prompts.SYSTEM_PROMPT,
    }
    assert model.predict(_image(), classes=[1]).boxes.cls.tolist() == [1]


def test_referring_queries_each_phrase_and_absent_targets_are_empty(harness):
    phrases = ["the man in red", "a cat"]
    harness.responses[prompts.referring_prompt(phrases[0])] = (
        "<ref>the man in red</ref><box><0><0><500><500></box>"
    )
    harness.responses[prompts.referring_prompt(phrases[1])] = "None"
    model = LibreWeVisOneVL(names=phrases, referring=True, device="cpu")
    result = model.predict(_image())
    assert harness.prompts == [prompts.referring_prompt(p) for p in phrases]
    assert result.boxes.cls.tolist() == [0]
    np.testing.assert_allclose(result.boxes.xyxy.cpu(), [[0, 0, 200, 100]])


@pytest.mark.parametrize("task", ["segment", "matte"])
def test_per_phrase_tasks_require_a_vocabulary(task, harness):
    model = LibreWeVisOneVL(device="cpu")
    with pytest.raises(ValueError, match="set_classes"):
        model.set_task(task).predict(_image())
    referring = LibreWeVisOneVL(referring=True, device="cpu")
    with pytest.raises(ValueError, match="set_classes"):
        referring.predict(_image())


def test_point_queries_one_label_at_a_time(harness):
    harness.responses[prompts.point_prompt(["dog"])] = (
        "<ref>dog</ref><point><250><750></point>"
    )
    harness.responses[prompts.point_prompt(["cat"])] = (
        "<ref>cat</ref><point><500><500></point>,<point><100><100></point>"
    )
    model = LibreVLM("wevisone-vl", names=["dog", "cat"], task="point", device="cpu")
    result = model.predict(_image())
    assert harness.prompts == ["point to: dog", "point to: cat"]
    assert result.boxes is None
    np.testing.assert_allclose(
        result.points.xy.cpu(), [[100, 150], [200, 100], [40, 20]]
    )
    assert result.points.cls.tolist() == [0, 1, 1]


def test_segment_aligns_obj_hidden_states_with_valid_boxes(harness):
    matting = prompts.matting_prompt("dog")
    harness.responses[matting] = (
        "<ref>dog</ref><box><100><100><500><500><OBJ>,<600><600><500><500><OBJ>,"
        "<600><600><900><900><OBJ>,<100><100><500><500><OBJ></box>"
    )
    model = LibreWeVisOneVL(names=["dog"], device="cpu", decoder_batch_size=2)
    model.set_task("segment")
    encode_calls, refine_calls = [], []
    _spy(model.mask_decoder, "encode", encode_calls)
    _spy(model.mask_decoder, "refine", refine_calls)
    result = model.predict(_image())

    assert harness.prompts == [matting]
    teacher = harness.messages[-1]
    assert teacher[-1] == {"role": "assistant", "content": harness.responses[matting]}
    assert harness.teacher_calls[-1]["use_cache"] is False
    assert len(encode_calls) == 1
    assert [len(call[2]) for call in refine_calls] == [2, 1]
    canvas, visual_maps, t_seg, boxes = encode_calls[0]
    assert tuple(canvas.shape) == (1, 3, 64, 64)
    assert [tuple(m.shape) for m in visual_maps] == [(1, VIS_DIM, GRID, GRID)] * 4
    # The n-th <OBJ> sits at PREFIX_TOKENS + 1 + 2n; the malformed box n=1 is skipped.
    obj_positions = [PREFIX_TOKENS + 1 + 2 * n for n in (0, 2, 3)]
    assert t_seg[:, 0].tolist() == obj_positions
    np.testing.assert_allclose(
        boxes.cpu(), [[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9], [0.1, 0.1, 0.5, 0.5]]
    )
    # The duplicate of the first box is suppressed.
    np.testing.assert_allclose(
        result.boxes.xyxy.cpu(), [[40, 20, 200, 100], [240, 120, 360, 180]], atol=1e-4
    )
    assert result.boxes.cls.tolist() == [0, 0]
    assert tuple(result.masks.data.shape) == (2, IMAGE_SIZE[1], IMAGE_SIZE[0])

    model.mask_threshold = 0.0
    assert bool(model.predict(_image()).masks.data.all())


def _spy(module, name, calls):
    original = getattr(module, name)

    def wrapper(*args):
        calls.append(args)
        return original(*args)

    setattr(module, name, wrapper)


def test_decoder_batch_size_does_not_change_mattes(harness):
    response = (
        "<ref>dog</ref><box><100><100><500><500><OBJ>,<600><600><900><900><OBJ>,"
        "<200><500><700><950><OBJ></box>"
    )
    model = LibreWeVisOneVL(names=["dog"], device="cpu")
    alphas = {}
    for size in (1, 2, 3):
        model.decoder_batch_size = size
        instances = model._decode_instances(
            _image(), prompts.matting_prompt("dog"), response
        )
        alphas[size] = np.stack([instance["alpha"] for instance in instances])
    assert alphas[1].shape == (3, IMAGE_SIZE[1], IMAGE_SIZE[0])
    np.testing.assert_allclose(alphas[2], alphas[1], atol=1e-6)
    np.testing.assert_allclose(alphas[3], alphas[1], atol=1e-6)


def test_decoder_refinement_chunks_match_joint_decoding():
    torch.manual_seed(0)
    decoder = build_mask_decoder(
        VIS_DIM, TEXT_DIM, Qwen3VLSegConfig.from_dict(TINY_CFG)
    ).eval()
    image = torch.rand(1, 3, 64, 64)
    maps = [torch.randn(1, VIS_DIM, GRID, GRID) for _ in range(4)]
    t_seg = torch.randn(3, TEXT_DIM)
    boxes = torch.tensor(
        [[0.1, 0.1, 0.5, 0.5], [0.4, 0.2, 0.9, 0.8], [0.0, 0.5, 0.6, 1.0]]
    )
    with torch.no_grad():
        joint = decoder(image, maps, t_seg, boxes)
        queries, pixel_feature = decoder.encode(image, maps, t_seg, boxes)
        chunked = torch.cat(
            [decoder.refine(image, pixel_feature, queries[k : k + 1]) for k in range(3)]
        )
        alone = decoder(image, maps, t_seg[:1], boxes[:1])
    torch.testing.assert_close(chunked, joint, atol=1e-5, rtol=1e-5)
    # Objects attend to each other, so decoding one in isolation differs.
    assert not torch.allclose(alone, joint[:1], atol=1e-3)


class _BoxDecoder(nn.Module):
    """Decoder stand-in that paints each query's box."""

    def encode(self, image, visual_maps, t_seg, boxes):
        return boxes, None

    def refine(self, image, pixel_feature, boxes):
        size = image.shape[-1]
        logits = torch.full((len(boxes), size, size), -30.0)
        for k, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
            logits[
                k, int(y1 * size) : int(y2 * size), int(x1 * size) : int(x2 * size)
            ] = 30.0
        return logits


def test_matte_is_the_union_of_instance_mattes(harness):
    harness.responses[prompts.matting_prompt("person")] = (
        "<ref>person</ref><box><0><0><500><1000><OBJ>,<500><0><1000><500><OBJ></box>"
    )
    model = LibreWeVisOneVL(names=["person"], device="cpu").set_task("matte")
    model.mask_decoder = _BoxDecoder()
    result = model.predict(_image())
    assert result.boxes is None
    matte = result.matte.data.cpu().numpy()
    assert matte.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0])
    assert matte.min() >= 0.0 and matte.max() <= 1.0
    assert matte[150, 100] > 0.99 and matte[50, 300] > 0.99
    assert matte[150, 300] < 0.01


def test_ocr_returns_reading_order_quads_with_transcripts(harness):
    harness.responses[prompts.OCR_PROMPT] = (
        "<ref>EXIT</ref><polygon><500><500><700><500><700><600><500><600></polygon>"
        "<ref>Polar</ref><box><100><100><300><200></box>"
    )
    model = LibreWeVisOneVL(device="cpu").set_task("ocr")
    result = model.predict(_image())
    assert harness.prompts == [prompts.OCR_PROMPT]
    assert result.ocr.texts == ["Polar", "EXIT"]
    np.testing.assert_allclose(
        result.ocr.polygons.cpu(),
        [
            [[40, 20], [120, 20], [120, 40], [40, 40]],
            [[200, 100], [280, 100], [280, 120], [200, 120]],
        ],
    )


def test_custom_prompt_and_chat(harness):
    harness.responses["Find the boats."] = (
        "<ref>boat</ref><box><0><0><1000><1000></box>"
    )
    model = LibreWeVisOneVL(names=["boat"], prompt="Find the boats.", device="cpu")
    assert model.predict(_image()).boxes.cls.tolist() == [0]
    harness.responses["How many wheels?"] = "<ref>wheel</ref><point><1><2></point>"
    reply = model.chat(_image(), "How many wheels?", max_new_tokens=17)
    assert reply == "<ref>wheel</ref><point><1><2></point>"
    assert harness.generations[-1]["max_new_tokens"] == 17


def test_unsupported_tasks_and_operations(harness):
    with pytest.raises(ValueError):
        LibreVLM("wevisone-vl", task="pose", device="cpu")
    model = LibreWeVisOneVL(names=["boat"], device="cpu")
    for method in [model.train, model.val, model.export]:
        with pytest.raises(NotImplementedError):
            method()


def test_sidecar_errors(harness, tmp_path):
    model = LibreWeVisOneVL(names=["boat"], device="cpu")
    backbone, processor = _FakeQwen(harness), _Processor(harness)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="mask_decoder.pt"):
        model._load_mask_decoder(empty, backbone, processor)
    mismatched = tmp_path / "mismatched"
    mismatched.mkdir()
    _write_sidecar(mismatched, seg_token_id=SEG_ID + 1)
    with pytest.raises(RuntimeError, match="<OBJ> token mismatch"):
        model._load_mask_decoder(mismatched, backbone, processor)


# =============================================================================
# Static contracts
# =============================================================================


def test_pins_inventory_and_registry():
    from libreyolo.models.inventory import collect_model_inventory
    from libreyolo.models.registry import MODEL_GROUPS

    assert not LibreWeVisOneVL.TRUST_REMOTE_CODE
    assert (
        set(LibreWeVisOneVL.HF_REPOS)
        == set(LibreWeVisOneVL.HF_REVISIONS)
        == {"2b", "4b"}
    )
    assert all(
        len(sha) == 40 and int(sha, 16) for sha in LibreWeVisOneVL.HF_REVISIONS.values()
    )
    assert MODEL_GROUPS["wevisonevl"] == "s"
    entry = collect_model_inventory()["wevisonevl"]
    assert entry["optional_extra"] == "vlm"
    assert entry["tasks"] == ["detect", "point", "segment", "matte", "ocr"]


@pytest.mark.parametrize("llm_dim,params", [(2048, 9_511_460), (2560, 9_609_764)])
def test_released_decoder_shape(llm_dim, params):
    decoder = build_mask_decoder(visual_dim=1024, llm_dim=llm_dim)
    assert sum(p.numel() for p in decoder.parameters()) == params
    assert len(decoder.state_dict()) == 296


def test_premerge_tokens_round_trip():
    h, w, m, c = 4, 6, 2, 3
    grid = torch.arange(h * w * c, dtype=torch.float32).reshape(h, w, c)
    merge_local = (
        grid.reshape(h // m, m, w // m, m, c).permute(0, 2, 1, 3, 4).reshape(h * w, c)
    )
    restored = premerge_tokens_to_2d(merge_local, (h, w), m)
    assert torch.equal(restored[0].permute(1, 2, 0), grid)
    with pytest.raises(ValueError, match="divisible"):
        premerge_tokens_to_2d(merge_local, (3, 8), m)

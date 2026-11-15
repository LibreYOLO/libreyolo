"""Offline unit tests for the Qwen3-VL-Embedding adapter (no model download).

The model and processor are replaced with small stubs, so these tests pin the
adapter's contract (aliases, instruction handling, pooling, shapes, rejected
surfaces), not the network's numbers. Numerical parity against the upstream
reference embedder is recorded in the PR.
"""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from libreyolo.models.vlm.qwen3vl_embedding import (
    DEFAULT_INSTRUCTION,
    LibreQwen3VLEmbedding,
    normalize_instruction,
    pool_last_token,
)

pytestmark = pytest.mark.unit

DIM = 2048


class _Encoding(dict):
    def to(self, device):
        return self


class _StubProcessor:
    """Records rendered prompts; tokenizes to right-padded fake ids."""

    def __init__(self):
        self.rendered = []
        self.calls = []

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
        assert add_generation_prompt is True and tokenize is False
        self.rendered.append(conversation)
        system = conversation[0]["content"][0]["text"]
        user = conversation[1]["content"]
        return f"{system}|{len(user)}|{user[-1].get('text', '<img>')}"

    def __call__(self, text, images=None, padding=True, return_tensors="pt", **kw):
        self.calls.append({"text": text, "images": images, **kw})
        lengths = [len(t) for t in text]
        width = max(lengths)
        mask = torch.zeros(len(text), width, dtype=torch.long)
        for i, n in enumerate(lengths):
            mask[i, :n] = 1
        ids = torch.arange(width).repeat(len(text), 1) * mask
        return _Encoding(input_ids=ids, attention_mask=mask)


class _StubInner(torch.nn.Module):
    def forward(self, input_ids, attention_mask, **kwargs):
        # Hidden state encodes the position, so pooling picks are checkable.
        b, t = input_ids.shape
        pos = torch.arange(t, dtype=torch.float32).view(1, t, 1).expand(b, t, DIM)
        hidden = pos + 1.0 + torch.arange(DIM, dtype=torch.float32) / DIM
        return SimpleNamespace(last_hidden_state=hidden)


def _bare(prompt=None):
    m = object.__new__(LibreQwen3VLEmbedding)
    m.size = "2b"
    m.device = torch.device("cpu")
    m._custom_prompt = prompt
    m._model_dtype = torch.float32
    m.processor = _StubProcessor()
    m.model = SimpleNamespace(model=_StubInner())
    return m


class TestFactory:
    def test_aliases_resolve(self):
        from libreyolo.models.vlm import _ALIASES

        assert _ALIASES["qwen3-vl-embedding"] == (LibreQwen3VLEmbedding, "2b")
        assert _ALIASES["qwen3-vl-embedding-2b"] == (LibreQwen3VLEmbedding, "2b")

    def test_top_level_export(self):
        import libreyolo

        assert libreyolo.LibreQwen3VLEmbedding is LibreQwen3VLEmbedding

    def test_contract_class_vars(self):
        assert LibreQwen3VLEmbedding.SUPPORTED_TASKS == ("embed",)
        assert LibreQwen3VLEmbedding.DEFAULT_TASK == "embed"
        assert LibreQwen3VLEmbedding.HF_REPOS == {
            "2b": "LibreYOLO/LibreQwen3VLEmbedding2b"
        }
        assert LibreQwen3VLEmbedding.TRUST_REMOTE_CODE is False

    @pytest.mark.parametrize("key", ["names", "max_new_tokens", "checkpoint_dir"])
    def test_detector_kwargs_rejected_before_load(self, key):
        with pytest.raises(TypeError):
            LibreQwen3VLEmbedding("2b", **{key: ["x"]})

    def test_invalid_size_raises_before_load(self):
        with pytest.raises(ValueError):
            LibreQwen3VLEmbedding("8b")


class TestInstruction:
    def test_default(self):
        assert normalize_instruction(None) == DEFAULT_INSTRUCTION
        assert normalize_instruction("   ") == DEFAULT_INSTRUCTION

    def test_appends_period_only_without_punctuation(self):
        assert normalize_instruction(" Find the page ") == "Find the page."
        assert normalize_instruction("Find the page?") == "Find the page?"
        assert normalize_instruction("Find the page.") == "Find the page."

    def test_model_instruction_follows_constructor_prompt(self):
        assert _bare().instruction == DEFAULT_INSTRUCTION
        assert _bare("Retrieve pages").instruction == "Retrieve pages."


class TestPooling:
    def test_last_attended_token_with_right_padding(self):
        hidden = torch.arange(2 * 4, dtype=torch.float32).view(2, 4, 1)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        assert pool_last_token(hidden, mask).flatten().tolist() == [2.0, 7.0]


class TestEmbedText:
    def test_shape_dtype_and_unit_rows(self):
        m = _bare()
        out = m.embed_text(["a", "much longer text"])
        assert out.shape == (2, DIM) and out.dtype == torch.float32
        assert torch.allclose(out.norm(dim=-1), torch.ones(2))

    def test_single_string_and_empty(self):
        m = _bare()
        assert m.embed_text("a").shape == (1, DIM)
        assert m.embed_text([]).shape == (0, DIM)

    def test_rejects_non_strings(self):
        m = _bare()
        with pytest.raises(TypeError):
            m.embed_text(["a", 3])
        with pytest.raises(TypeError):
            m.embed_text("a", prompt=3)

    def test_prompt_override_and_default(self):
        m = _bare("Doc default")
        m.embed_text("q", prompt="Find relevant pages")
        m.embed_text("d")
        systems = [c[0]["content"][0]["text"] for c in m.processor.rendered]
        assert systems == ["Find relevant pages.", "Doc default."]

    def test_batches_and_truncation(self):
        m = _bare()
        m.embed_text([str(i) for i in range(LibreQwen3VLEmbedding.TEXT_BATCH + 1)])
        assert len(m.processor.calls) == 2
        call = m.processor.calls[0]
        assert call["truncation"] is True
        assert call["max_length"] == LibreQwen3VLEmbedding.MAX_LENGTH

    def test_padding_does_not_change_a_row(self):
        m = _bare()
        alone = m.embed_text("short")
        padded = m.embed_text(["short", "a much much longer input text"])[:1]
        assert torch.allclose(alone, padded)


class TestImagePath:
    def test_resize_to_grid_within_budget(self):
        img = Image.new("RGB", (850, 1100))
        out = LibreQwen3VLEmbedding._resize(img)
        w, h = out.size
        assert w % 32 == 0 and h % 32 == 0
        assert LibreQwen3VLEmbedding.MIN_PIXELS <= w * h
        assert w * h <= LibreQwen3VLEmbedding.MAX_PIXELS

    def test_large_image_is_capped(self):
        out = LibreQwen3VLEmbedding._resize(Image.new("RGB", (4000, 3000)))
        assert out.size[0] * out.size[1] <= LibreQwen3VLEmbedding.MAX_PIXELS

    def test_grid_aligned_image_is_untouched(self):
        img = Image.new("RGB", (768, 576))
        assert LibreQwen3VLEmbedding._resize(img) is img

    def test_preprocess_forward_postprocess(self):
        m = _bare()
        inputs, _img, size, ratio = m._preprocess(Image.new("RGB", (64, 48)))
        assert size == (64, 48) and ratio == 1.0
        call = m.processor.calls[-1]
        assert call["do_resize"] is False and len(call["images"]) == 1
        conv = m.processor.rendered[-1]
        assert conv[0]["content"][0]["text"] == DEFAULT_INSTRUCTION
        assert conv[1]["content"][0]["type"] == "image"
        payload = m._postprocess(m._forward(inputs), 0.25, 0.45, size)
        emb = payload["embeddings"]
        assert emb.shape == (1, DIM)
        assert torch.allclose(emb.norm(dim=-1), torch.ones(1))


class TestRejectedSurface:
    @pytest.mark.parametrize(
        "call",
        [
            lambda m: m.chat("img.jpg", "hi"),
            lambda m: m.set_classes(["dog"]),
            lambda m: m.val(data="x.yaml"),
            lambda m: m.export("onnx"),
            lambda m: m.train(data="x.yaml"),
        ],
    )
    def test_raises_not_implemented(self, call):
        with pytest.raises(NotImplementedError):
            call(_bare())

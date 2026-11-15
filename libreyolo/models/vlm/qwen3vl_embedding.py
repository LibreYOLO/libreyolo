"""LibreYOLO wrapper for Qwen3-VL-Embedding, a VLM-based multimodal embedder.

Qwen3-VL-Embedding-2B (Apache-2.0) is Qwen3-VL-2B-Instruct fine-tuned to emit
one vector per input: the input is rendered as a chat (a system instruction,
then the user's image and/or text), the decoder runs once, and the hidden
state of the last token is the embedding. Images and texts land in the same
2048-d space, which is what visual document retrieval needs: embed page images
once, embed text queries at search time, rank by dot product.

It lives in the LibreVLM tier (same snapshot download and transformers loader
as ``LibreQwen3VL``) but serves the ``embed`` task contract (ADR 0015), not
detection:

* ``predict(image)`` returns ``Results.embeddings`` of shape ``(1, 2048)``,
  float32 and L2-normalized; ``embed(source)`` stacks rows like any embed
  family.
* ``embed_text(texts, prompt=None)`` returns ``(M, 2048)`` normalized rows in
  the same space, mirroring CLIP / SigLIP2 / PE.
* ``prompt`` is the model's instruction (a system message). The upstream
  default is ``"Represent the user's input."``. Set it at construction to
  change the default for images and texts, or pass ``prompt=`` to
  ``embed_text`` per call: retrieval queries usually carry a task
  instruction while documents keep the default.

Preprocessing follows the upstream reference embedder
(``scripts/qwen3_vl_embedding.py`` in the model repo, QwenLM/Qwen3-VL-Embedding,
Apache-2.0): images are resized on a 32-pixel grid to between 4,096 and
1,843,200 pixels with PIL bicubic, instructions get a trailing period when they
lack punctuation, and the pooled token is the last non-padding token after the
assistant generation prompt.

Out of scope in v1: generation (``chat``), detection (``set_classes``),
training, dataset validation, export, combined image+text inputs, video clip
embeddings, and Matryoshka (MRL) truncation. For MRL, slice the first ``k``
columns of the returned rows and re-normalize.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from .base import LibreVLMModel

DEFAULT_INSTRUCTION = "Represent the user's input."

_UNSUPPORTED_KWARGS = ("names", "max_new_tokens", "checkpoint_dir")


def normalize_instruction(instruction: str | None) -> str:
    """Strip an instruction and end it with punctuation (upstream rule).

    ``None`` or blank returns the default instruction.
    """
    text = (instruction or "").strip()
    if not text:
        return DEFAULT_INSTRUCTION
    if not unicodedata.category(text[-1]).startswith("P"):
        text += "."
    return text


def pool_last_token(
    hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Hidden state of the last attended token per row (right padding)."""
    last = attention_mask.shape[1] - attention_mask.flip(dims=[1]).argmax(dim=1) - 1
    rows = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[rows, last]


class LibreQwen3VLEmbedding(LibreVLMModel):
    """Qwen3-VL-Embedding: images and texts to one shared vector space."""

    FAMILY = "qwen3vl_embedding"
    FILENAME_PREFIX = "LibreQwen3VLEmbedding"

    HF_REPOS: ClassVar[dict[str, str]] = {
        "2b": "LibreYOLO/LibreQwen3VLEmbedding2b",
    }
    # Nominal only; images are resized by pixel budget (see _resize).
    INPUT_SIZES: ClassVar[dict[str, int]] = {"2b": 1024}
    EMBEDDING_DIMS: ClassVar[dict[str, int]] = {"2b": 2048}

    SUPPORTED_TASKS: ClassVar[tuple] = ("embed",)
    DEFAULT_TASK: ClassVar[str] = "embed"

    # Upstream reference embedder constants.
    IMAGE_FACTOR: ClassVar[int] = 32
    MIN_PIXELS: ClassVar[int] = 4 * 32 * 32
    MAX_PIXELS: ClassVar[int] = 1800 * 32 * 32
    MAX_LENGTH: ClassVar[int] = 8192
    TEXT_BATCH: ClassVar[int] = 8

    TRAINABLE: ClassVar[bool] = False
    TRAIN_UNSUPPORTED_REASON: ClassVar[str] = (
        "Training is not supported for LibreQwen3VLEmbedding: it is an "
        "inference-only embedder in this LibreYOLO release."
    )
    _LICENSE_NOTICE = ""

    def __init__(self, size: str = "2b", *, prompt: str | None = None, **kwargs):
        for key in _UNSUPPORTED_KWARGS:
            if kwargs.get(key) is not None:
                raise TypeError(
                    f"{type(self).__name__} is an embedding model and does not "
                    f"accept {key}=."
                )
            kwargs.pop(key, None)
        super().__init__(size, prompt=prompt, **kwargs)

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIMS[self.size]

    @property
    def instruction(self) -> str:
        """The instruction applied to images and to texts without ``prompt=``."""
        return normalize_instruction(self._custom_prompt)

    def embed_text(
        self, texts: str | Sequence[str], prompt: str | None = None
    ) -> torch.Tensor:
        """Embed text rows into the same space as image embeddings.

        Args:
            texts: One string or a sequence of strings.
            prompt: Instruction for these texts. Defaults to the model's
                instruction (``"Represent the user's input."`` unless set at
                construction). For retrieval, pass the task instruction here
                for queries and keep the default for documents.

        Returns:
            ``(M, D)`` float32 CPU tensor, rows L2-normalized.
        """
        items = [texts] if isinstance(texts, str) else list(texts)
        if any(not isinstance(text, str) for text in items):
            raise TypeError("embed_text() expects a string or a sequence of strings.")
        if prompt is not None and not isinstance(prompt, str):
            raise TypeError("embed_text() prompt must be a string.")
        if not items:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)
        instruction = (
            self.instruction if prompt is None else normalize_instruction(prompt)
        )
        out: list[torch.Tensor] = []
        for start in range(0, len(items), self.TEXT_BATCH):
            batch = items[start : start + self.TEXT_BATCH]
            texts_in = [
                self._render(self._conversation(instruction, text=t)) for t in batch
            ]
            inputs = self.processor(
                text=texts_in,
                padding=True,
                truncation=True,
                max_length=self.MAX_LENGTH,
                return_tensors="pt",
            )
            feats = self._embed_inputs(inputs)
            out.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(out, dim=0)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    @staticmethod
    def _conversation(
        instruction: str, *, text: str | None = None, image: Any = None
    ) -> list:
        content: list = []
        if image is not None:
            content.append({"type": "image", "image": image})
        if text:
            content.append({"type": "text", "text": text})
        if not content:
            # Upstream encodes an empty input as the literal "NULL".
            content.append({"type": "text", "text": "NULL"})
        return [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

    def _render(self, conversation: list) -> str:
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

    @classmethod
    def _resize(cls, img: Image.Image) -> Image.Image:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            smart_resize,
        )

        height, width = smart_resize(
            img.height,
            img.width,
            factor=cls.IMAGE_FACTOR,
            min_pixels=cls.MIN_PIXELS,
            max_pixels=cls.MAX_PIXELS,
        )
        if (width, height) == img.size:
            return img
        return img.resize((width, height), Image.BICUBIC)

    def _embed_inputs(self, inputs: Any) -> torch.Tensor:
        """Run the decoder once and pool the last attended token."""
        inputs = self._prepare_generation_inputs(inputs.to(self.device))
        attention_mask = inputs["attention_mask"]
        with torch.inference_mode():
            hidden = self.model.model(**inputs).last_hidden_state
        return pool_last_token(hidden, attention_mask)

    # ------------------------------------------------------------------
    # InferenceRunner hooks
    # ------------------------------------------------------------------

    def _load_pretrained(self, snapshot_dir: str):
        model, processor = super()._load_pretrained(snapshot_dir)
        # Last-token pooling assumes right padding (upstream setting).
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "right"
        return model, processor

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: int | None = None,
    ) -> tuple[Any, Any, tuple[int, int], float]:
        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        resized = self._resize(img.convert("RGB"))
        text = self._render(self._conversation(self.instruction, image=resized))
        inputs = self.processor(
            text=[text],
            images=[resized],
            do_resize=False,
            padding=True,
            return_tensors="pt",
        )
        return inputs, img, original_size, 1.0

    def _forward(self, inputs: Any) -> torch.Tensor:
        return self._embed_inputs(inputs)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> dict:
        return self._postprocess_embeddings(
            output,
            gallery=kwargs.get("gallery"),
            threshold=kwargs.get("threshold"),
        )

    # ------------------------------------------------------------------
    # Generative / detection surface does not apply
    # ------------------------------------------------------------------

    def chat(self, *args, **kwargs) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} is an embedding model and does not generate "
            "text. Use predict()/embed() for images and embed_text() for text."
        )

    def set_classes(self, classes: list) -> LibreQwen3VLEmbedding:
        raise NotImplementedError(
            f"{type(self).__name__} is an embedding model, not a detector. "
            "Compare embed_text() rows with image embeddings instead."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            f"Dataset validation is not supported for {type(self).__name__}: "
            "LibreYOLO has no retrieval validator yet."
        )

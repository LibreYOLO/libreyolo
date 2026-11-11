"""Prompt embedders for Iris-Z: the FLUX.2 text encoders through ``transformers``.

Ported from the Iris-Z SDK (``iris_z/model/text_encoder.py``, from FLUX.2).
See ``NOTICE``. Each embedder stacks the hidden states of three decoder layers
of a 512-token, max-length-padded chat-formatted prompt into one
``[B, 512, 3 * hidden]`` tensor:

- Iris-Z-big (FLUX.2 [dev]): Mistral Small 3.2 24B, layers 10, 20, 30, with the
  FLUX.2 system message, through the Mistral Small 3.1 processor (as upstream).
- Iris-Z-small (FLUX.2 [klein] Base 9B): Qwen3 8B FP8, layers 9, 18, 27, with
  the chat template's thinking turned off.

Only the embedding path is ported; upstream's prompt upsampling and content
filters are not called by Iris-Z and are left out. Weights come from their
publishers (Apache-2.0) at pinned revisions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)

MAX_LENGTH = 512

MISTRAL_REPO = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
MISTRAL_REVISION = "95a6d26c4bfb886c58daf9d3f7332c857cb27b43"
MISTRAL_PROCESSOR_REPO = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
MISTRAL_PROCESSOR_REVISION = "68faf511d618ef198fef186659617cfd2eb8e33a"
MISTRAL_LAYERS = (10, 20, 30)
# FLUX.2's system message for prompt embedding.
SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured responses "
    "focusing on object relationships, object\nattribution and actions without speculation."
)

QWEN3_REPO = "Qwen/Qwen3-8B-FP8"
QWEN3_REVISION = "220b46e3b2180893580a4454f21f22d3ebb187d3"
QWEN3_LAYERS = (9, 18, 27)


def _stack_layers(hidden_states: Sequence[torch.Tensor], layers: Sequence[int]) -> torch.Tensor:
    """``[B, L, D]`` per layer -> ``[B, L, len(layers) * D]``, layer-major per token."""
    stacked = torch.stack([hidden_states[k] for k in layers], dim=1)
    batch, count, length, dim = stacked.shape
    return stacked.permute(0, 2, 1, 3).reshape(batch, length, count * dim)


class MistralSmallEmbedder(nn.Module):
    """Mistral Small 3.2 prompt embeddings for FLUX.2 [dev]."""

    def __init__(self, device="cpu"):
        super().__init__()
        from transformers import AutoProcessor, Mistral3ForConditionalGeneration

        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            MISTRAL_REPO, revision=MISTRAL_REVISION, dtype=torch.bfloat16
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(
            MISTRAL_PROCESSOR_REPO, revision=MISTRAL_PROCESSOR_REVISION, use_fast=False
        )

    @property
    def device(self) -> torch.device:
        return self.model.device

    @staticmethod
    def _messages(prompts: List[str]) -> List[List[Dict]]:
        return [
            [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
                {"role": "user", "content": [{"type": "text", "text": text.replace("[IMG]", "")}]},
            ]
            for text in prompts
        ]

    @torch.no_grad()
    def forward(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.processor.apply_chat_template(
            self._messages(prompts),
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        output = self.model(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
            output_hidden_states=True,
            use_cache=False,
        )
        return _stack_layers(output.hidden_states, MISTRAL_LAYERS)


class Qwen3Embedder(nn.Module):
    """Qwen3 8B (FP8) prompt embeddings for FLUX.2 [klein] Base 9B."""

    def __init__(self, device="cpu"):
        super().__init__()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(QWEN3_REPO, revision=QWEN3_REVISION)
        self.fp8 = torch.device(device).type == "cuda"
        self.model = None
        if self.fp8:
            self.model = self._load(device, dequantize=False)
            try:
                # transformers loads the FP8 matmul kernel on the first call.
                with torch.no_grad():
                    self.model(input_ids=torch.tensor([[0]], device=self.model.device))
            except ImportError as exc:
                logger.warning(
                    "Qwen3 8B FP8 runs dequantized in bfloat16 because the FP8 kernel is "
                    "unavailable (%s). Prompt embeddings are close to upstream's, not identical.",
                    exc,
                )
                self.model = None
                self.fp8 = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if self.model is None:
            self.model = self._load(device, dequantize=True)

    @staticmethod
    def _load(device, dequantize: bool):
        from transformers import AutoModelForCausalLM, FineGrainedFP8Config

        kwargs = {}
        if dequantize:
            # FP8 needs CUDA and transformers' FP8 kernel; elsewhere the
            # checkpoint's block-scaled weights are expanded to bfloat16.
            kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        return AutoModelForCausalLM.from_pretrained(
            QWEN3_REPO,
            revision=QWEN3_REVISION,
            # Upstream loads with torch_dtype=None under transformers 4.57, which
            # keeps the unquantized layers in float32; transformers 5 defaults to "auto".
            dtype=torch.bfloat16 if dequantize else torch.float32,
            device_map=str(device),
            **kwargs,
        )

    @property
    def device(self) -> torch.device:
        return self.model.device

    @torch.no_grad()
    def forward(self, prompts: List[str]) -> torch.Tensor:
        input_ids, attention_masks = [], []
        for text in prompts:
            chat = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            tokens = self.tokenizer(
                chat,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            input_ids.append(tokens["input_ids"])
            attention_masks.append(tokens["attention_mask"])
        output = self.model(
            input_ids=torch.cat(input_ids).to(self.device),
            attention_mask=torch.cat(attention_masks).to(self.device),
            output_hidden_states=True,
            use_cache=False,
        )
        return _stack_layers(output.hidden_states, QWEN3_LAYERS)


TEXT_ENCODERS = {"mistral_small": MistralSmallEmbedder, "qwen3_8b": Qwen3Embedder}

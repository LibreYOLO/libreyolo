"""Batch collation for VLM detection fine-tuning.

Turns dataset samples (PIL image + prompt + target text) into the tensors a
chat-template VLM trains on. The one delicate step is label masking: only the
assistant's answer tokens are supervised; the user turn (including every image
token) and padding are masked to -100.

Masking is done by the prompt-prefix method: the batch is tokenized twice, once
as the full conversation and once as the user turn plus generation prompt.
With right padding, the prompt tokens are a prefix of the full sequence, so
masking up to the common prefix is family-agnostic; no per-family image token
ids are needed.

One boundary subtlety: the last prompt token may retokenize when the answer is
appended (e.g. a template ending in ``<think>\n`` merges that newline into a
``\n\n`` token once text follows), so the mask runs to the LONGEST COMMON
PREFIX of the two tokenizations, with a small tolerance. A template that
genuinely rewrites the user turn when an answer is present would produce a
short common prefix, and that still fails loudly instead of silently training
on prompt tokens.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, List, Sequence

import torch

logger = logging.getLogger(__name__)

__all__ = ["VLMChatCollator"]


class VLMChatCollator:
    """Collate ``{"image", "prompt", "target"}`` samples into a training batch."""

    # Processor outputs that generative forward passes do not accept.
    DROP_KEYS: ClassVar[tuple] = ("token_type_ids",)
    # How many trailing prompt tokens may retokenize at the prompt/answer
    # boundary before it is treated as a broken template. Merged-newline
    # boundaries use 1-2; anything larger means the template rewrote the turn.
    BOUNDARY_TOLERANCE: ClassVar[int] = 4

    def __init__(self, processor, max_length_warn: int | None = None) -> None:
        self.processor = processor
        self.max_length_warn = max_length_warn
        self._warned_length = False

    def _tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full: List[list] = []
        prompt_only: List[list] = []
        for sample in samples:
            user = {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["prompt"]},
                ],
            }
            assistant = {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["target"]}],
            }
            full.append([user, assistant])
            prompt_only.append([user])

        tokenizer = self._tokenizer()
        previous_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "right"
        try:
            batch = self.processor.apply_chat_template(
                full,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            prompts = self.processor.apply_chat_template(
                prompt_only,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                add_generation_prompt=True,
            )
        finally:
            tokenizer.padding_side = previous_side

        labels = batch["input_ids"].clone()
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        prompt_lengths = prompts["attention_mask"].sum(dim=1).tolist()
        for i, prompt_len in enumerate(prompt_lengths):
            full_row = batch["input_ids"][i, :prompt_len]
            prompt_row = prompts["input_ids"][i, :prompt_len]
            diverging = (full_row != prompt_row).nonzero()
            common = int(diverging[0]) if len(diverging) else prompt_len
            if prompt_len - common > self.BOUNDARY_TOLERANCE:
                raise RuntimeError(
                    "Chat template broke the prompt-prefix property: the user "
                    "turn tokenizes differently with and without the assistant "
                    "answer. Label masking would supervise prompt tokens; "
                    "this family needs a dedicated collator."
                )
            labels[i, :common] = -100
        batch["labels"] = labels

        if (
            self.max_length_warn
            and not self._warned_length
            and batch["input_ids"].shape[1] > self.max_length_warn
        ):
            self._warned_length = True
            logger.warning(
                "Training sequence length %d exceeds %d tokens; expect high "
                "VRAM use. Lower max_pixels or simplify the vocabulary.",
                batch["input_ids"].shape[1],
                self.max_length_warn,
            )

        for key in self.DROP_KEYS:
            batch.pop(key, None)
        return batch

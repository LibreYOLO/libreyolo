"""Per-family VLM fine-tuning recipes.

Fixed per family, not user-facing knobs (the docs/lora.md philosophy): rank,
alpha, targets, what freezes, and the optimizer defaults live here. A family
becomes trainable by adding a recipe AND setting ``TRAINABLE = True`` on its
adapter class; the trainer refuses families without both.

Qwen3-VL: LoRA on the language model's attention and MLP projections, vision
tower and merger frozen. Matches the official qwen-vl-finetune defaults
(rank 16 scale, frozen ViT) and the ms-swift best-practice recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

__all__ = ["VLMTrainRecipe", "get_recipe"]


@dataclass(frozen=True)
class VLMTrainRecipe:
    """Family-fixed training recipe."""

    # LoRA adapter shape. ``target_modules`` is a regex matched against module
    # names; it must only match inside the language model so the vision tower
    # stays untouched (asserted at injection time).
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = ""
    # Module-name prefixes that must never receive adapters and are frozen in
    # full fine-tuning too.
    frozen_prefixes: tuple = ()
    # Optimizer defaults (LoRA path). Full fine-tuning uses ``full_ft_lr0``.
    lr0: float = 1e-4
    full_ft_lr0: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    clip_grad_norm: float = 1.0
    # Soft ceiling for a training sequence before the collator warns.
    max_length_warn: int = 8192


_RECIPES: Dict[str, VLMTrainRecipe] = {
    "qwen3vl": VLMTrainRecipe(
        target_modules=(
            r".*language_model.*\."
            r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        ),
        frozen_prefixes=("model.visual",),
    ),
}


def get_recipe(family: str) -> VLMTrainRecipe:
    """Return the fixed recipe for a family, or raise for unknown families."""
    recipe = _RECIPES.get(family)
    if recipe is None:
        raise NotImplementedError(
            f"No VLM training recipe is defined for family {family!r}."
        )
    return recipe

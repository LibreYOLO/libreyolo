"""Parameter-efficient fine-tuning (LoRA) helpers for LibreYOLO.

LibreYOLO's own integration of LoRA on top of the optional ``peft`` dependency.
It targets transformer (DINOv2 ViT) backbones such as RF-DETR, where the
attention projections are ``nn.Linear`` layers that LoRA adapts cheaply. The
backbone base weights are frozen and only the small low-rank adapters remain
trainable in the backbone, while the projector, decoder, and detection head keep
training normally. This lets users with limited GPU memory fine-tune on a custom
dataset.

The adapter recipe is a faithful match of the RF-DETR reference (Apache-2.0):
DoRA (weight-decomposed LoRA) with rank 16 and alpha 16 on the DINOv2 attention
projections. The public surface is a single boolean ``lora=True`` training
argument; the hyperparameters below are fixed, not a user-facing API.
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

# Fixed adapter hyperparameters, matching the RF-DETR reference. Not exposed:
# the public API is ``lora=True``.
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
USE_DORA = True  # weight-decomposed LoRA (DoRA), as in upstream RF-DETR

# Target-module suffixes for the DINOv2 ViT, matching the RF-DETR reference list
# verbatim. On LibreYOLO's transformers-based DINOv2 only the attention
# ``query``/``key``/``value`` ``nn.Linear`` layers actually match; the remaining
# entries are inert for this backbone but kept for parity:
#   - ``q_proj``/``k_proj``/``v_proj``/``qkv`` name other DINOv2 variants' fused
#     or renamed projections that this implementation does not use.
#   - ``cls_token``/``register_tokens`` are ``nn.Parameter`` attributes, not
#     ``nn.Linear`` modules, so peft's module-name matching does not adapt them
#     here (this matches upstream behavior on the same HF backbone).
DINOV2_TARGET_MODULES = (
    "q_proj",
    "v_proj",
    "k_proj",
    "qkv",
    "query",
    "key",
    "value",
    "cls_token",
    "register_tokens",
)

_PEFT_INSTALL_HINT = (
    'LoRA fine-tuning requires the optional "peft" package. '
    'Install it with: pip install "libreyolo[lora]"'
)


def _require_peft():
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(_PEFT_INSTALL_HINT) from exc
    return LoraConfig, get_peft_model


def is_peft_available() -> bool:
    """Return True when the optional ``peft`` dependency is importable."""
    try:
        import peft  # noqa: F401
    except ImportError:
        return False
    return True


def count_trainable_parameters(module: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts for *module*."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def state_dict_has_lora(state_dict: dict) -> bool:
    """Return True when *state_dict* carries LoRA adapter tensors."""
    return any(
        ("lora_A" in k or "lora_B" in k or "lora_magnitude" in k) for k in state_dict
    )


def module_has_lora(module: nn.Module) -> bool:
    """Return True when *module* already carries PEFT/LoRA adapters."""
    if hasattr(module, "peft_config"):
        return True
    return any("lora_" in name for name, _ in module.named_parameters())


def apply_lora_to_rfdetr(core_model: nn.Module) -> nn.Module:
    """Inject LoRA adapters into an RF-DETR DINOv2 encoder, in place.

    Wraps ``core_model.backbone[0].encoder`` (the DINOv2 ViT) with a PEFT model
    so the base weights are frozen and only the low-rank adapters are trainable.
    The wrapped encoder exposes ``merge_and_unload`` which the backbone's
    ``export`` path already uses to bake adapters back into dense weights.

    Args:
        core_model: the LWDETR core module (``LibreRFDETRModel.model``) that
            owns the ``backbone`` Joiner.

    Returns:
        The PEFT-wrapped encoder module that replaced the original encoder.

    Raises:
        ImportError: if ``peft`` is not installed.
        ValueError: if the model does not expose the expected backbone layout.
    """
    backbone = getattr(core_model, "backbone", None)
    if backbone is None:
        raise ValueError("RF-DETR model has no .backbone; cannot apply LoRA.")
    try:
        encoder_owner = backbone[0]
    except (TypeError, IndexError) as exc:
        raise ValueError("RF-DETR model.backbone[0] is not indexable.") from exc
    if not hasattr(encoder_owner, "encoder"):
        raise ValueError("RF-DETR model.backbone[0] has no .encoder to adapt.")

    if module_has_lora(encoder_owner.encoder):
        return encoder_owner.encoder

    LoraConfig, get_peft_model = _require_peft()
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        use_dora=USE_DORA,
        target_modules=list(DINOV2_TARGET_MODULES),
        bias="none",
    )
    wrapped = get_peft_model(encoder_owner.encoder, lora_config)
    encoder_owner.encoder = wrapped

    n_adapted = sum(
        1 for name, _ in wrapped.named_modules() if name.endswith(".lora_A.default")
    )
    if n_adapted == 0:
        raise ValueError(
            "LoRA injection matched zero modules in the RF-DETR backbone. "
            f"Expected target suffixes {DINOV2_TARGET_MODULES} in the DINOv2 encoder; "
            "the backbone module naming may have changed."
        )

    trainable, total = count_trainable_parameters(core_model)
    logger.info(
        "Applied LoRA to RF-DETR backbone: %d adapted modules, "
        "%d/%d trainable params (%.2f%%).",
        n_adapted,
        trainable,
        total,
        100.0 * trainable / max(1, total),
    )
    return wrapped


__all__ = [
    "apply_lora_to_rfdetr",
    "is_peft_available",
    "state_dict_has_lora",
    "module_has_lora",
    "count_trainable_parameters",
    "DINOV2_TARGET_MODULES",
    "LORA_RANK",
    "LORA_ALPHA",
    "USE_DORA",
]

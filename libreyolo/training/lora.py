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
    return any(is_lora_parameter_name(k) for k in state_dict)


def is_lora_parameter_name(name: str) -> bool:
    """Return True for PEFT LoRA adapter parameter names."""
    return "lora_A" in name or "lora_B" in name or "lora_magnitude" in name


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


# ---------------------------------------------------------------------------
# D-FINE / DEIM (DETR decoder) recipe
# ---------------------------------------------------------------------------
# These families pair a CNN (HGNetv2) backbone with a transformer hybrid
# encoder + deformable decoder. The CNN backbone has no nn.Linear layers to
# adapt, so the recipe differs from RF-DETR's ViT-backbone recipe:
#   - the backbone is frozen entirely (no gradients flow through it at all,
#     which also skips its backward pass),
#   - the transformer blocks (AIFI encoder layers + decoder layers) freeze
#     their base weights and train LoRA adapters on their nn.Linear layers,
#   - everything else (encoder conv fusion, input projections, prediction
#     heads, query embeddings) keeps training normally, since custom class
#     counts require freshly trained heads.
#
# Plain LoRA, not DoRA: several decoder Linears (Gate.gate, MSDeformableAttention
# sampling_offsets / attention_weights) are zero-init by design, and DoRA's
# magnitude normalization divides by the weight norm, which is 0 for a
# freshly built model. Rank/alpha match the RF-DETR recipe.
DETR_LORA_RANK = 16
DETR_LORA_ALPHA = 16
DETR_LORA_DROPOUT = 0.0

# Module classes that delimit the frozen+adapted transformer zone. Both
# D-FINE and DEIM define their own copies of these classes with identical
# names (as does RT-DETR, for future adoption).
DETR_BLOCK_CLASSES = ("TransformerEncoderLayer", "TransformerDecoderLayer")

# Leaf names of nn.Linear layers adapted inside those blocks. ``value_proj``
# and ``output_proj`` are inert for D-FINE/DEIM (their deformable attention
# has neither) but kept for parity with other deformable-DETR decoders.
# nn.MultiheadAttention is intentionally NOT adapted: its forward reads
# ``out_proj.weight`` directly, so a swapped-in LoRA layer would be silently
# bypassed. Self-attention stays frozen instead.
DETR_TARGET_LINEAR_NAMES = (
    "linear1",
    "linear2",
    "gate",
    "sampling_offsets",
    "attention_weights",
    "value_proj",
    "output_proj",
    # DEIMv2's decoder swaps the classic linear1/linear2 FFN for SwiGLU:
    "w12",
    "w3",
)

# DEIMv2 S/M/L/X carry a ViT backbone under ``backbone.dinov3``, in one of two
# vendorings: the DINOv3 ``SelfAttentionBlock`` or the ViT-Tiny ``Block``
# (vit_tiny.py). Both use one fused ``qkv`` Linear per block; adapting it
# mirrors the RF-DETR reference recipe, which lists ``qkv`` for exactly this
# fused layout. The attention output ``proj`` is not adapted (the reference
# does not adapt output projections either). ``Block`` is generic, so ViT
# block discovery is scoped to the ``backbone.dinov3`` subtree.
DINOV3_BLOCK_CLASSES = ("SelfAttentionBlock", "Block")
DINOV3_TARGET_LINEAR_NAMES = ("qkv",)


def apply_lora_to_detr(core_model: nn.Module) -> nn.Module:
    """Inject LoRA adapters into a D-FINE/DEIM style DETR core model, in place.

    Uses ``peft.inject_adapter_in_model`` (no PeftModel wrapper) so the module
    tree, attribute surface, and checkpoint key layout stay put; only the
    targeted ``nn.Linear`` layers become ``lora.Linear`` (their dense weight
    moves under ``.base_layer.``). After injection the trainability policy is
    applied explicitly:

    - ``lora_*`` adapter params: trainable
    - ``backbone.*`` and all params inside transformer blocks: frozen
    - everything else: left exactly as it was (heads/neck stay trainable,
      intentionally fixed params like D-FINE's ``decoder.up`` stay fixed)

    Args:
        core_model: a model exposing ``backbone``/``encoder``/``decoder``
            (``LibreDFINEModel`` / ``LibreDEIMModel``).

    Returns:
        *core_model*, modified in place.

    Raises:
        ImportError: if ``peft`` is not installed.
        ValueError: if the model does not look like a DETR core or no target
            module matched.
    """
    _require_detr_layout(core_model)
    if module_has_lora(core_model):
        return core_model

    block_roots = _discover_block_roots(core_model, DETR_BLOCK_CLASSES)
    if not block_roots:
        raise ValueError(
            "No transformer encoder/decoder blocks found; expected "
            f"{DETR_BLOCK_CLASSES} classes in the model."
        )
    target_modules = _collect_linear_targets(
        core_model, block_roots, DETR_TARGET_LINEAR_NAMES, skip_frozen=True
    )
    frozen_prefixes = ("backbone.",) + tuple(f"{root}." for root in block_roots)
    return _inject_lora(
        core_model, target_modules, frozen_prefixes, label="DETR transformer blocks"
    )


def apply_lora_to_deimv2(core_model: nn.Module) -> nn.Module:
    """Inject LoRA adapters into a DEIMv2 core model, in place.

    Same contract as :func:`apply_lora_to_detr`, with two DEIMv2 additions:

    - The decoder FFN is SwiGLU (``w12``/``w3``), covered by the shared
      target list.
    - S/M/L/X sizes carry a DINOv3 ViT backbone under ``backbone.dinov3``.
      There, the ViT base is frozen and its fused attention ``qkv`` Linears
      get adapters (as in the RF-DETR reference, which targets ``qkv`` for
      fused layouts), while the Spatial Tuning Adapter (STA) conv pyramid
      around it keeps training as the projector analog. The ``qkv`` layers
      are adapted even when the config shipped the ViT frozen
      (``finetune=False``): adapting a frozen backbone is the point of LoRA.
      HGNetv2 sizes fall back to the plain DETR recipe (backbone frozen
      entirely).
    """
    _require_detr_layout(core_model)
    if module_has_lora(core_model):
        return core_model

    detr_roots = _discover_block_roots(core_model, DETR_BLOCK_CLASSES)
    if not detr_roots:
        raise ValueError(
            "No transformer encoder/decoder blocks found; expected "
            f"{DETR_BLOCK_CLASSES} classes in the model."
        )
    target_modules = _collect_linear_targets(
        core_model, detr_roots, DETR_TARGET_LINEAR_NAMES, skip_frozen=True
    )

    frozen_roots = list(detr_roots)
    dinov3 = getattr(core_model.backbone, "dinov3", None)
    if dinov3 is not None:
        # Scoped to the ViT subtree: "Block" is too generic a class name to
        # match across the whole model.
        vit_roots = [
            f"backbone.dinov3.{name}"
            for name, module in dinov3.named_modules()
            if name and type(module).__name__ in DINOV3_BLOCK_CLASSES
        ]
        if not vit_roots:
            raise ValueError(
                "backbone.dinov3 present but no SelfAttentionBlock modules "
                "found; the DINOv3 vendoring may have changed."
            )
        target_modules += _collect_linear_targets(
            core_model, vit_roots, DINOV3_TARGET_LINEAR_NAMES, skip_frozen=False
        )
        frozen_roots += vit_roots
        backbone_prefix = "backbone.dinov3."
    else:
        backbone_prefix = "backbone."

    frozen_prefixes = (backbone_prefix,) + tuple(f"{root}." for root in frozen_roots)
    return _inject_lora(
        core_model, target_modules, frozen_prefixes, label="DEIMv2 transformer blocks"
    )


def apply_lora_to_ec(core_model: nn.Module) -> nn.Module:
    """Inject LoRA adapters into an EC (EdgeCrafter) core model, in place.

    EC is a DETR whose backbone is a ViTAdapter: a plain ViT (fused ``qkv``
    attention ``Block``s, same layout as DEIMv2's tiny ViT) under
    ``backbone.backbone``, surrounded by a trainable conv projector pyramid.
    Recipe: freeze the ViT base and adapt its ``qkv`` Linears, freeze the
    transformer encoder/decoder block bases and adapt their Linears, keep the
    ViTAdapter projector, input projections, and heads dense-trainable.
    """
    _require_detr_layout(core_model)
    if module_has_lora(core_model):
        return core_model

    detr_roots = _discover_block_roots(core_model, DETR_BLOCK_CLASSES)
    if not detr_roots:
        raise ValueError(
            "No transformer encoder/decoder blocks found; expected "
            f"{DETR_BLOCK_CLASSES} classes in the model."
        )
    target_modules = _collect_linear_targets(
        core_model, detr_roots, DETR_TARGET_LINEAR_NAMES, skip_frozen=True
    )

    vit = getattr(core_model.backbone, "backbone", None)
    if vit is None:
        raise ValueError(
            "EC model has no backbone.backbone ViT; the ViTAdapter layout "
            "may have changed."
        )
    # Scoped to the ViT subtree: "Block" is too generic a class name to match
    # across the whole model.
    vit_roots = [
        f"backbone.backbone.{name}"
        for name, module in vit.named_modules()
        if name and type(module).__name__ in DINOV3_BLOCK_CLASSES
    ]
    if not vit_roots:
        raise ValueError(
            "backbone.backbone present but no ViT Block modules found; "
            "the EC backbone vendoring may have changed."
        )
    target_modules += _collect_linear_targets(
        core_model, vit_roots, DINOV3_TARGET_LINEAR_NAMES, skip_frozen=False
    )

    frozen_prefixes = ("backbone.backbone.",) + tuple(
        f"{root}." for root in detr_roots + vit_roots
    )
    return _inject_lora(
        core_model, target_modules, frozen_prefixes, label="EC transformer blocks"
    )


# GTR's backbone is a ViTAdapter around a gated-linear-attention ViT
# (``Block`` under ``backbone.backbone``). Its separate q/k/v projections play
# the role of the fused ``qkv`` adapted elsewhere; the gate, output and MLP
# projections are not adapted, matching the attention-input-only reference.
GTR_TARGET_LINEAR_NAMES = ("q_proj", "k_proj", "v_proj")
# The oriented-box decoder layer carries the same Linear names under its own
# class name.
GTR_DECODER_BLOCK_CLASSES = DETR_BLOCK_CLASSES + ("OBBTransformerDecoderLayer",)


def apply_lora_to_gtr(core_model: nn.Module) -> nn.Module:
    """Inject LoRA adapters into a GTR detection core model, in place.

    Same shape as :func:`apply_lora_to_ec`: freeze the recurrent ViT base and
    adapt its q/k/v projections, freeze the decoder layer bases and adapt their
    Linears, keep the ViTAdapter projector, conv encoder and heads trainable.
    """
    _require_detr_layout(core_model)
    if module_has_lora(core_model):
        return core_model

    detr_roots = _discover_block_roots(core_model, GTR_DECODER_BLOCK_CLASSES)
    if not detr_roots:
        raise ValueError(
            "No transformer decoder blocks found; expected "
            f"{GTR_DECODER_BLOCK_CLASSES} classes in the model."
        )
    target_modules = _collect_linear_targets(
        core_model, detr_roots, DETR_TARGET_LINEAR_NAMES, skip_frozen=True
    )

    vit = getattr(core_model.backbone, "backbone", None)
    if vit is None:
        raise ValueError(
            "GTR model has no backbone.backbone ViT; the ViTAdapter layout "
            "may have changed."
        )
    vit_roots = [
        f"backbone.backbone.{name}"
        for name, module in vit.named_modules()
        if name and type(module).__name__ == "Block"
    ]
    if not vit_roots:
        raise ValueError(
            "backbone.backbone present but no ViT Block modules found; "
            "the GTR backbone vendoring may have changed."
        )
    target_modules += _collect_linear_targets(
        core_model, vit_roots, GTR_TARGET_LINEAR_NAMES, skip_frozen=False
    )

    frozen_prefixes = ("backbone.backbone.",) + tuple(
        f"{root}." for root in detr_roots + vit_roots
    )
    return _inject_lora(
        core_model, target_modules, frozen_prefixes, label="GTR transformer blocks"
    )


# ConvNeXt: a conv classifier whose blocks nonetheless carry channels-last
# nn.Linear MLPs (timm layout, ``fc1``/``fc2``). Those take the adapters; the
# depthwise convs, norms, and layer-scale gammas freeze; the classification
# head stays dense-trainable (custom class counts rebuild it).
CONVNEXT_BLOCK_CLASSES = ("ConvNeXtBlock",)
CONVNEXT_TARGET_LINEAR_NAMES = ("fc1", "fc2")


def apply_lora_to_convnext(core_model: nn.Module) -> nn.Module:
    """Inject LoRA adapters into a ConvNeXt classifier core, in place."""
    for attr in ("stages", "head"):
        if not hasattr(core_model, attr):
            raise ValueError(
                f"Model has no .{attr}; cannot apply the ConvNeXt LoRA recipe."
            )
    if module_has_lora(core_model):
        return core_model

    block_roots = _discover_block_roots(core_model, CONVNEXT_BLOCK_CLASSES)
    if not block_roots:
        raise ValueError(
            "No ConvNeXtBlock modules found; the ConvNeXt vendoring may have "
            "changed."
        )
    target_modules = _collect_linear_targets(
        core_model, block_roots, CONVNEXT_TARGET_LINEAR_NAMES, skip_frozen=True
    )

    # Freeze every top-level child except the classification head.
    frozen_prefixes = tuple(
        f"{name}." for name, _ in core_model.named_children() if name != "head"
    )
    return _inject_lora(
        core_model, target_modules, frozen_prefixes, label="ConvNeXt block MLPs"
    )


def _require_detr_layout(core_model: nn.Module) -> None:
    for attr in ("backbone", "encoder", "decoder"):
        if not hasattr(core_model, attr):
            raise ValueError(
                f"Model has no .{attr}; cannot apply the DETR LoRA recipe."
            )


def _discover_block_roots(
    core_model: nn.Module, block_classes: tuple[str, ...]
) -> list[str]:
    return [
        name
        for name, module in core_model.named_modules()
        if type(module).__name__ in block_classes
    ]


def _collect_linear_targets(
    core_model: nn.Module,
    block_roots: list[str],
    leaf_names: tuple[str, ...],
    *,
    skip_frozen: bool,
) -> list[str]:
    """Fully-qualified names of the nn.Linear layers to adapt inside blocks.

    ``skip_frozen=True`` respects layers frozen by design (e.g.
    ``cross_attn_method="discrete"`` freezes sampling_offsets): no adapter
    there. ViT backbone passes set it False, because a config-frozen backbone
    is exactly where adapters belong.
    """
    targets = []
    for root in block_roots:
        block = core_model.get_submodule(root)
        for sub_name, sub in block.named_modules():
            if not isinstance(sub, nn.Linear):
                continue
            if sub_name.rsplit(".", 1)[-1] not in leaf_names:
                continue
            params = list(sub.parameters())
            if skip_frozen and params and all(not p.requires_grad for p in params):
                continue
            targets.append(f"{root}.{sub_name}")
    return targets


def _inject_lora(
    core_model: nn.Module,
    target_modules: list[str],
    frozen_prefixes: tuple[str, ...],
    *,
    label: str,
) -> nn.Module:
    """Inject plain LoRA (r16/a16) on *target_modules* and apply the
    trainability policy: adapters train, *frozen_prefixes* freeze, everything
    else is restored to its pre-injection state."""
    if not target_modules:
        raise ValueError(
            f"LoRA injection matched zero modules in the {label}. "
            f"Expected Linear leaf names {DETR_TARGET_LINEAR_NAMES}; "
            "the module naming may have changed."
        )

    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as exc:
        raise ImportError(_PEFT_INSTALL_HINT) from exc

    # Snapshot trainability before injection so the policy below can restore
    # everything outside the frozen zones no matter what peft toggled.
    pre_requires_grad = {
        name: p.requires_grad for name, p in core_model.named_parameters()
    }

    lora_config = LoraConfig(
        r=DETR_LORA_RANK,
        lora_alpha=DETR_LORA_ALPHA,
        lora_dropout=DETR_LORA_DROPOUT,
        use_dora=False,
        target_modules=target_modules,
        bias="none",
    )
    inject_adapter_in_model(lora_config, core_model, adapter_name="default")

    n_adapted = sum(
        1
        for name, _ in core_model.named_modules()
        if name.endswith(".lora_A.default")
    )
    if n_adapted == 0:
        raise ValueError(
            "peft matched the target list but created no adapters; "
            f"targets were {target_modules[:5]}..."
        )

    for name, p in core_model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        elif name.startswith(frozen_prefixes):
            p.requires_grad = False
        elif name in pre_requires_grad:
            p.requires_grad = pre_requires_grad[name]

    trainable, total = count_trainable_parameters(core_model)
    logger.info(
        "Applied LoRA to %s: %d adapted modules, "
        "%d/%d trainable params (%.2f%%).",
        label,
        n_adapted,
        trainable,
        total,
        100.0 * trainable / max(1, total),
    )
    return core_model


def merge_lora_adapters(module: nn.Module) -> int:
    """Merge injected LoRA layers back into dense weights, in place.

    For models adapted with :func:`apply_lora_to_detr` (peft injection without
    a PeftModel wrapper), folds every adapter into its base ``nn.Linear`` and
    swaps the original layer back in, so the module carries no peft dependency
    afterwards.

    Returns:
        Number of merged adapter layers.
    """
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError as exc:
        raise ImportError(_PEFT_INSTALL_HINT) from exc

    merged = 0
    for name, sub in list(module.named_modules()):
        if not isinstance(sub, BaseTunerLayer):
            continue
        sub.merge(safe_merge=True)
        parent = (
            module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
        )
        setattr(parent, name.rsplit(".", 1)[-1], sub.get_base_layer())
        merged += 1

    # inject_adapter_in_model stamps ``peft_config`` on the model; drop it once
    # every tuner layer is folded back so ``module_has_lora`` reflects the now
    # fully-dense graph (and a second export is a no-op).
    if merged and hasattr(module, "peft_config"):
        try:
            del module.peft_config
        except AttributeError:
            pass
    return merged


__all__ = [
    "apply_lora_to_rfdetr",
    "apply_lora_to_detr",
    "apply_lora_to_deimv2",
    "apply_lora_to_ec",
    "apply_lora_to_gtr",
    "apply_lora_to_convnext",
    "merge_lora_adapters",
    "is_peft_available",
    "is_lora_parameter_name",
    "state_dict_has_lora",
    "module_has_lora",
    "count_trainable_parameters",
    "DINOV2_TARGET_MODULES",
    "DETR_BLOCK_CLASSES",
    "DETR_TARGET_LINEAR_NAMES",
    "DETR_LORA_RANK",
    "DETR_LORA_ALPHA",
    "LORA_RANK",
    "LORA_ALPHA",
    "USE_DORA",
]

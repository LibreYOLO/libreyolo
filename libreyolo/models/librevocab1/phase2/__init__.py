"""LibreVocab1 Phase 2 — own-decoder open-vocab detector on top of CRADIOv4.

Replaces the vendored SAM3 head from Phase 1 with a LibreYOLO-owned, permissively
licensed detection head:

    Image ─► CRADIOv4 (frozen, NVIDIA Open Model License)
              │
              ├─► forward_intermediates([L1, L2, L3])
              │       ↓
              │    DEIMv2 SpatialPriorModulev2 + 1x1 conv + SyncBN
              │       ↓
              │    3-scale pyramid (1/8, 1/16, 1/32)
              │       ↓
              │    RT-DETRv2 HybridEncoder
              │       ↓
              │    RT-DETRv2 TransformerDecoder + text cross-attention
              │       (300 queries, 6 layers, denoising)
              │
              └─► siglip2-g adaptor (frozen, comes with CRADIOv4 ckpt)
                       ↓
                  encode_text(prompts) ─► (K, 1152)
                       ↓ text_proj
                       └─► fused into decoder (cross-attn + cosine class head)

Classification: ``cls_logit = (qproj(query) @ text_emb.T) / temperature``.
Box regression: standard 4-d sigmoid cxcywh head.

License composition target:
    - CRADIOv4 weights:       NVIDIA Open Model License
    - SigLIP2 text branch:    Apache-2.0 (loaded via CRADIOv4's adaptor)
    - DEIMv2-derived decoder: Apache-2.0
    - LibreYOLO glue + trained head weights: MIT
    No SAM License surface in Phase 2.

Status: scaffolding. Heavy components (SPMv2, RT-DETRv2 hybrid encoder + decoder)
are deferred until branch ``151-add-deimv2`` merges to ``main``; once it does,
this module will import them from ``libreyolo.models.deimv2.engine.*`` rather
than re-implementing them.

This family is **not** registered in the factory. ``can_load`` returns False so
the factory cannot route real checkpoints here. Construction goes through
``LibreVocab1Phase2(...)`` directly.
"""

from .model import LibreVocab1Phase2

__all__ = ["LibreVocab1Phase2"]

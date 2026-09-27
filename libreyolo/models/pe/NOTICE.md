# LibrePE - third-party notices

The LibrePE family is a native LibreYOLO implementation of Meta's **Perception
Encoder (PE) Core** dual-tower vision-language encoder
(<https://arxiv.org/abs/2504.13181>). Neither `timm` nor `open_clip` is
imported at runtime; both are used only as pinned adaptation sources and as the
offline parity oracle.

## Adapted code

### huggingface/pytorch-image-models - Apache-2.0

- Upstream: <https://github.com/huggingface/pytorch-image-models>
- Pinned revision: tag `v1.0.28`, commit `8ef73809f622e0031bd7f4940265734aef8b9978`
- Files referenced: `timm/models/eva.py`, `timm/layers/pos_embed_sincos.py`,
  `timm/layers/attention_pool.py`
- Adapted into `libreyolo/models/pe/nn.py` as `PEVisionTransformer`, `PEBlock`,
  `PEAttentionRope`, `PEAttentionPoolLatent`, `PatchEmbed`, `Mlp`,
  `RotaryEmbeddingCat`, `build_rotary_pos_embed`, `apply_rot_embed_cat` and the
  frequency-band helpers.
- This is the selected **architecture and configuration** source. The per-size
  values in `PE_CONFIGS` are transcribed from the `vit_pe_core_*` model
  definitions.

Licensed under the Apache License, Version 2.0. A copy is available at
<http://www.apache.org/licenses/LICENSE-2.0>.

### mlfoundations/open_clip - MIT

- Upstream: <https://github.com/mlfoundations/open_clip>
- Pinned revision: tag `v3.2.0`, commit `6f939057c792a2f3d4d58df748de60ca47c4aed4`
- File referenced: `src/open_clip/transformer.py`
- Adapted into `libreyolo/models/pe/nn.py` as `PETextTransformer`,
  `Transformer` and `ResidualAttentionBlock`.
- This is the selected **text, projection, preprocessing and parity** source.

### Tokenizer

The OpenAI BPE tokenizer is **reused** from the existing in-tree
`libreyolo/models/clip/` implementation. No vocabulary or tokenizer code is
duplicated for this family; that file's own provenance applies.

## Semantic upstream (documentation only)

- `facebookresearch/perception_models`, commit
  `3e352cca660658d4b5c90f42a7808b11469e4c66`.
- The repository ships `LICENSE.PE` with Apache-2.0 terms and its root
  documentation describes PE code and PE checkpoints as Apache-2.0, but the
  same pinned revision's `setup.py` declares a noncommercial/proprietary
  package license. **This inconsistency is unresolved.** No code was copied or
  adapted from that repository; it was used only as behavioral and
  documentation evidence. The independently permissive timm/OpenCLIP
  conversion route was selected precisely to avoid the ambiguity.
- The Perception Language Model (`LICENSE.PLM`, noncommercial), PE Spatial and
  PE-AV are explicitly out of scope and were not inspected or adapted.

## Weights

Rehosted checkpoints are converted from the OpenCLIP-compatible `timm/PE-Core-*`
repositories, each of whose model card declares Apache-2.0. Exact repository ids
and revisions are recorded per checkpoint in `weights/LICENSE_NOTICE.txt` and in
each checkpoint's own metadata (`source_repo`, `source_revision`,
`source_license`). These are converted artifacts targeting the OpenCLIP-compatible
modeling implementation, not unmodified official `facebook/PE-Core-*` package
checkpoints.

## Datasets

No dataset is downloaded, bundled, mirrored, or required at runtime. In
particular the published `facebook/PE-Video` dataset is CC-BY-NC-4.0 and is
**not** used by any test, example, CI job, or download helper.

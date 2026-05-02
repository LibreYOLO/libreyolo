"""LibreVocab1 — experimental open-vocabulary detection family.

Architecture (Path A: SAM3 + CRADIOv4 vision encoder swap):

    Image ─► CRADIOv4 (with `sam3` adaptor head)
              │
              ▼ dense vision features in SAM3 vision-feature space
              │
    SAM3 stack (vendored from facebookresearch/sam3 via mranzinger/sam3-radio):
              │
              ├── FPN-style neck (Sam3DualViTDetNeck)
              ├── Text encoder (VETextEncoder, CLIP-style BPE)
              ├── Fusion encoder (TransformerEncoderFusion: pools text into image)
              ├── Transformer decoder (200 queries, 6 layers, text cross-attn,
              │                         box-refine, presence token, DAC)
              ├── DotProductScoring (query @ prompt_mlp(text_emb).T)
              └── UniversalSegmentationHead (PixelDecoder + masks)
                                                       │
                                                       ▼
                                       (boxes, masks, scores) per text prompt

This is the configuration NVIDIA validated empirically in figures 6–8 of the
C-RADIOv4 tech report (arXiv 2601.17237). We vendor the inference path of
``sam3-radio`` and call the standard ``torch.hub.load('NVlabs/RADIO', ...)``
entry point for the backbone.

LICENSES (read before redistributing):
    - LibreYOLO core glue (this directory's *.py except sam3/): MIT.
    - Vendored SAM3 subtree (sam3/): Meta SAM License (see sam3/LICENSE_SAM3).
      Permissive enough for inference + distribution; viral on derivatives;
      includes indemnification + no-reverse-engineer + Trade Controls clauses.
    - CRADIOv4 weights (downloaded at runtime): NVIDIA Open Model License.
      Commercially permissive per NVIDIA. Not redistributed by us.
    - SAM3 weights (downloaded at runtime from facebookresearch/sam3 on HF):
      SAM License. Not redistributed by us.

Status: experimental scaffold. NOT registered in the LibreYOLO factory. Build
manually via :func:`build_librevocab1` until inference parity vs sam3-radio is
verified on a test image.
"""

from .builder import build_librevocab1
from .model import LibreVocab1

__all__ = ["LibreVocab1", "build_librevocab1"]

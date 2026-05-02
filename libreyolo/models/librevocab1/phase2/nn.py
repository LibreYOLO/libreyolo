"""LibreVocab1 Phase 2 architecture (scaffold).

Defines class shapes and forward-pass plumbing. Heavy components — SpatialPrior
ModuleV2, RT-DETRv2 HybridEncoder, RT-DETRv2 TransformerDecoder — are imported
from ``libreyolo.models.deimv2.engine.*`` once branch 151 merges to main; for
now they are stubbed. The CRADIOv4 backbone wrapper is real (we already use it
in Phase 1).

Forward shapes assume:
    - input image (B, 3, H, W) with H, W multiples of 16
    - CRADIOv4-SO400M: embed_dim=1152, depth=27
    - CRADIOv4-H:      embed_dim=1280, depth=32
    - intermediate layer indices scale with depth (deep / mid / shallow)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Backbone wrapper
# ---------------------------------------------------------------------------


class CRADIOv4Backbone(nn.Module):
    """Frozen CRADIOv4 backbone exposing 3 intermediate ViT layers + the
    SigLIP2-aligned dense stream from the ``siglip2-g`` adaptor.

    Loaded via ``torch.hub.load('NVlabs/RADIO', 'radio_model', version, ...)``.
    Always frozen and in eval mode in Phase 2.
    """

    # Default intermediate layer indices per backbone size. Roughly evenly
    # spaced across depth, biased toward the second half (where features are
    # more semantic). Will be re-validated empirically in the H200 run.
    _DEFAULT_INDICES: Dict[str, List[int]] = {
        "c-radio_v4-so400m": [13, 19, 25],  # depth 27
        "c-radio_v4-h": [15, 23, 31],       # depth 32
    }

    def __init__(
        self,
        version: str = "c-radio_v4-so400m",
        intermediate_indices: Optional[List[int]] = None,
        load_siglip2_adaptor: bool = True,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if version not in ("c-radio_v4-so400m", "c-radio_v4-h"):
            raise ValueError(f"unsupported CRADIOv4 version: {version}")
        self.version = version
        self.indices = (
            list(intermediate_indices)
            if intermediate_indices is not None
            else list(self._DEFAULT_INDICES[version])
        )
        self.embed_dim = 1152 if version == "c-radio_v4-so400m" else 1280
        self.patch_size = 16
        self.load_siglip2_adaptor = load_siglip2_adaptor

        # Lazy: actual torch.hub.load happens on first build because it's slow
        # and we want CPU unit tests to be able to construct the wrapper without
        # downloading 1.6 GB.
        self._radio: Optional[nn.Module] = None
        self._device_target = device

    def _ensure_radio(self) -> nn.Module:
        if self._radio is None:
            adaptor_names: List[str] = []
            if self.load_siglip2_adaptor:
                adaptor_names.append("siglip2-g")
            self._radio = torch.hub.load(
                "NVlabs/RADIO",
                "radio_model",
                self.version,
                adaptor_names=adaptor_names if adaptor_names else None,
                skip_validation=True,
            )
            self._radio.eval()
            for p in self._radio.parameters():
                p.requires_grad = False
        return self._radio

    @property
    def text_encoder(self) -> nn.Module:
        """Frozen SigLIP2 text encoder, accessible via the siglip2-g adaptor.

        Phase 2 detection runs this at training time on the per-batch class
        prompts and at inference time on the user-supplied prompts.
        """
        radio = self._ensure_radio()
        if not hasattr(radio, "adaptors") or "siglip2-g" not in radio.adaptors:
            raise RuntimeError(
                "siglip2-g adaptor not loaded; pass load_siglip2_adaptor=True"
            )
        return radio.adaptors["siglip2-g"].text_model

    def encode_text(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        """L2-normalized text embeddings, shape ``(len(prompts), 1152)``."""
        radio = self._ensure_radio()
        adaptor = radio.adaptors["siglip2-g"]
        tokens = adaptor.tokenizer(prompts).to(device)
        with torch.no_grad():
            return adaptor.encode_text(tokens, normalize=True)

    def forward_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return three intermediate features as ``(B, embed_dim, H/16, W/16)``."""
        # Real implementation pending; will use radio.forward_intermediates.
        # The stub returns a single repeated map so downstream shape tests pass.
        raise NotImplementedError(
            "CRADIOv4Backbone.forward_pyramid: implementation pending Phase 2 wiring."
        )


# ---------------------------------------------------------------------------
# Neck / encoder / decoder — deferred to DEIMv2 branch merge
# ---------------------------------------------------------------------------


class SPMv2Neck(nn.Module):
    """DEIMv2 SpatialPriorModuleV2 + 1x1 conv + SyncBN.

    Scaffold. The real implementation lives in
    ``libreyolo/models/deimv2/engine/backbone/dinov3_adapter.py`` on branch
    ``151-add-deimv2``. Once that merges to main, this class will import it
    rather than re-implement it.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def forward(
        self, intermediates: List[torch.Tensor], image: torch.Tensor
    ) -> List[torch.Tensor]:
        raise NotImplementedError("SPMv2Neck pending DEIMv2 branch merge.")


class HybridEncoder(nn.Module):
    """RT-DETRv2 hybrid encoder. Multi-scale fusion + intra-scale attention.

    Scaffold. Implementation pending DEIMv2 branch merge.
    """

    def __init__(self, hidden_dim: int = 256, num_levels: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels

    def forward(self, pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError("HybridEncoder pending DEIMv2 branch merge.")


class OpenVocabDecoder(nn.Module):
    """RT-DETRv2 transformer decoder with text cross-attention and cosine
    classification.

    Differences from upstream RT-DETRv2:
        - Adds a per-layer cross-attention block between (queries, text tokens)
          before the cross-attention to image features. Same pattern as
          Grounding-DINO and SAM3.
        - Replaces the standard ``Linear(hidden, num_classes)`` classification
          head with a parameter-free cosine head:
              ``cls_logits = qproj(query) @ text_emb.T / temperature``.
          ``num_classes`` is dynamic — set by the prompt list at forward time.

    Scaffold. Implementation pending DEIMv2 branch merge.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_layers: int = 6,
        text_dim: int = 1152,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.text_dim = text_dim
        self.temperature = temperature

    def forward(
        self,
        encoded: List[torch.Tensor],
        text_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return ``{'pred_logits': (B, Q, K), 'pred_boxes': (B, Q, 4)}``.

        Text embeddings come from the frozen SigLIP2 text encoder; ``K`` is the
        number of prompts in the current batch.
        """
        raise NotImplementedError("OpenVocabDecoder pending DEIMv2 branch merge.")


# ---------------------------------------------------------------------------
# Top-level network
# ---------------------------------------------------------------------------


class LibreVocab1Phase2Network(nn.Module):
    """Composes backbone + neck + encoder + decoder for end-to-end forward.

    Inference call:
        ``out = network(image, prompts=['a photo of a person', ...])``
    """

    def __init__(
        self,
        size: str = "s",
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_layers: int = 6,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        version = "c-radio_v4-so400m" if size == "s" else "c-radio_v4-h"
        self.backbone = CRADIOv4Backbone(version=version, device=device)
        self.neck = SPMv2Neck(embed_dim=self.backbone.embed_dim, hidden_dim=hidden_dim)
        self.encoder = HybridEncoder(hidden_dim=hidden_dim, num_levels=3)
        self.decoder = OpenVocabDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            text_dim=1152,
        )

    def forward(self, image: torch.Tensor, prompts: List[str]) -> Dict[str, Any]:
        intermediates = self.backbone.forward_pyramid(image)
        pyramid = self.neck(intermediates, image)
        encoded = self.encoder(pyramid)
        text_emb = self.backbone.encode_text(prompts, image.device)
        return self.decoder(encoded, text_emb)

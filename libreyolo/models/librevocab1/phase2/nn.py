"""LibreVocab1 Phase 2 architecture.

CRADIOv4 backbone (frozen) + DEIMv2-style SPMv2 neck + lightweight RT-DETR-style
hybrid encoder + open-vocab transformer decoder with text cross-attention and
cosine classification.

Forward shapes assume:
    - input image (B, 3, H, W) with H, W multiples of 16.
    - CRADIOv4-SO400M: embed_dim=1152, depth=27.
    - CRADIOv4-H:      embed_dim=1280, depth=32.
    - 3 intermediate ViT layers reshaped to stride-16 maps, then upsampled
      / downsampled to a (1/8, 1/16, 1/32) feature pyramid.

We deliberately do *not* import from ``libreyolo.models.deimv2.engine.*`` here.
Phase 2 is the open-vocab variant; while it borrows DEIMv2's structural ideas,
keeping the implementation local insulates us from churn on branch 151 and
gives us a clean place to evolve our own decoder + text-cross-attention head.

Top-level entry:
    out = LibreVocab1Phase2Network(size='s')(images, text_emb)
    out['pred_logits']  # (B, Q, K) cosine logits; K = text_emb.shape[0]
    out['pred_boxes']   # (B, Q, 4) cxcywh in [0, 1]
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Backbone wrapper
# ---------------------------------------------------------------------------


class CRADIOv4Backbone(nn.Module):
    """Frozen CRADIOv4 backbone exposing 3 intermediate ViT layers and the
    SigLIP2 text encoder via the ``siglip2-g`` adaptor.

    Loaded lazily via ``torch.hub.load('NVlabs/RADIO', 'radio_model', version)``.
    Always frozen and eval-mode in Phase 2 — we do not train the backbone.
    """

    _DEFAULT_INDICES: Dict[str, List[int]] = {
        "c-radio_v4-so400m": [13, 19, 25],   # depth 27
        "c-radio_v4-h": [15, 23, 31],         # depth 32
    }

    def __init__(
        self,
        version: str = "c-radio_v4-so400m",
        intermediate_indices: Optional[List[int]] = None,
        load_siglip2_adaptor: bool = True,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if version not in self._DEFAULT_INDICES:
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

    @torch.no_grad()
    def forward_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return three intermediate features as ``(B, embed_dim, H/16, W/16)``.

        Uses RADIO's ``forward_intermediates`` API which returns a list of
        intermediate tokens already reshaped to NCHW when ``output_fmt='NCHW'``.
        """
        radio = self._ensure_radio()
        # RADIO accepts un-normalized [0, 1] images. Phase 2 dataset already
        # produces float [0, 1] inputs, so we just pass through.
        intermediates = radio.forward_intermediates(
            x,
            indices=self.indices,
            return_prefix_tokens=False,
            norm=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        # ``intermediates`` is a list of (B, embed_dim, H/16, W/16) tensors.
        return [feat.float() for feat in intermediates]


# ---------------------------------------------------------------------------
# Neck — DEIMv2-style SpatialPriorModuleV2 + 1x1 projection
# ---------------------------------------------------------------------------


class _ConvBNAct(nn.Module):
    """Conv -> norm -> activation. Norm is BatchNorm; switch to SyncBN
    in distributed training via ``nn.SyncBatchNorm.convert_sync_batchnorm``.
    """

    def __init__(
        self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SPMv2Neck(nn.Module):
    """Spatial Prior Module + per-scale 1x1 projections to ``hidden_dim``.

    Inputs:
        intermediates: list of 3 tensors, each ``(B, embed_dim, H/16, W/16)``
            from ``CRADIOv4Backbone.forward_pyramid``.
        image: ``(B, 3, H, W)`` raw image, used only to extract H, W and feed
            the 4-stage SPM stem.

    Output: list of 3 NCHW tensors at strides 8, 16, 32, each
    ``(B, hidden_dim, H/s, W/s)``.

    Strategy: take the three ViT intermediates (all stride 16) and resize them
    to strides 8, 16, 32 with bilinear interpolation. Concatenate each with the
    matching SPM stage's CNN feature, then 1x1 project + BN to hidden_dim.
    The shallow CNN path adds high-frequency detail that pure ViT features
    typically lack.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        spm_inplanes: int = 16,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Convolutional spatial-prior stem. Produces feature maps at strides
        # 8, 16, 32 with channel dims 2*spm_inplanes, 4*spm_inplanes,
        # 4*spm_inplanes — chosen to roughly balance channel count vs the ViT
        # feature dim (which is 1152 / 1280) at each scale.
        c1, c2, c3 = spm_inplanes * 2, spm_inplanes * 4, spm_inplanes * 4
        self.stem = nn.Sequential(
            _ConvBNAct(3, spm_inplanes, kernel=3, stride=2, padding=1),       # /2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                  # /4
        )
        self.spm_s8 = nn.Sequential(
            _ConvBNAct(spm_inplanes, c1, kernel=3, stride=2, padding=1),       # /8
        )
        self.spm_s16 = nn.Sequential(
            _ConvBNAct(c1, c2, kernel=3, stride=2, padding=1),                 # /16
        )
        self.spm_s32 = nn.Sequential(
            _ConvBNAct(c2, c3, kernel=3, stride=2, padding=1),                 # /32
        )

        # 1x1 projections per scale to hidden_dim.
        self.proj_s8 = nn.Sequential(
            nn.Conv2d(embed_dim + c1, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )
        self.proj_s16 = nn.Sequential(
            nn.Conv2d(embed_dim + c2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )
        self.proj_s32 = nn.Sequential(
            nn.Conv2d(embed_dim + c3, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

    def forward(
        self, intermediates: List[torch.Tensor], image: torch.Tensor
    ) -> List[torch.Tensor]:
        if len(intermediates) != 3:
            raise ValueError(
                f"SPMv2Neck expects 3 intermediates, got {len(intermediates)}"
            )
        # ViT intermediate features at stride 16. Resize copies for the other scales.
        f_low, f_mid, f_high = intermediates  # all (B, embed_dim, H/16, W/16)

        H, W = image.shape[-2:]
        h16, w16 = H // 16, W // 16
        h8, w8 = h16 * 2, w16 * 2
        h32, w32 = max(h16 // 2, 1), max(w16 // 2, 1)

        # Upsample shallowest layer for high-res / stride-8.
        v_s8 = F.interpolate(f_low, size=(h8, w8), mode="bilinear", align_corners=False)
        # Mid layer kept at stride 16.
        v_s16 = f_mid
        # Downsample deepest layer for stride 32.
        v_s32 = F.interpolate(f_high, size=(h32, w32), mode="bilinear", align_corners=False)

        # SPM CNN path.
        s_stem = self.stem(image)               # /4
        s8 = self.spm_s8(s_stem)                # /8
        s16 = self.spm_s16(s8)                  # /16
        s32 = self.spm_s32(s16)                 # /32

        # Match SPM features to the ViT spatial size in case of off-by-one
        # rounding (e.g. odd input sizes). They should already match.
        if s8.shape[-2:] != v_s8.shape[-2:]:
            s8 = F.interpolate(s8, size=v_s8.shape[-2:], mode="bilinear", align_corners=False)
        if s16.shape[-2:] != v_s16.shape[-2:]:
            s16 = F.interpolate(s16, size=v_s16.shape[-2:], mode="bilinear", align_corners=False)
        if s32.shape[-2:] != v_s32.shape[-2:]:
            s32 = F.interpolate(s32, size=v_s32.shape[-2:], mode="bilinear", align_corners=False)

        # Concat ViT + SPM, then 1x1 project to hidden_dim.
        out_s8 = self.proj_s8(torch.cat([v_s8, s8], dim=1))
        out_s16 = self.proj_s16(torch.cat([v_s16, s16], dim=1))
        out_s32 = self.proj_s32(torch.cat([v_s32, s32], dim=1))
        return [out_s8, out_s16, out_s32]


# ---------------------------------------------------------------------------
# Hybrid encoder — RT-DETR-style intra-scale attention + lightweight FPN
# ---------------------------------------------------------------------------


class _TransformerEncoderLayer(nn.Module):
    """Standard pre-norm transformer encoder block (self-attn + FFN)."""

    def __init__(self, d_model: int, n_heads: int = 8, dim_ff: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention. RoPE-style pos enc would be cleaner but
        # standard learned pos enc is fine for this baseline.
        h = self.norm1(x)
        q = k = h if pos is None else h + pos
        h, _ = self.attn(q, k, h, need_weights=False)
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    """Standard 2D sin-cos position embedding, shape ``(h*w, dim)``."""
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sincos, got {dim}")
    grid_h = torch.arange(h, device=device, dtype=torch.float32)
    grid_w = torch.arange(w, device=device, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)  # (2, h, w)

    half = dim // 2
    omega = torch.arange(half // 2, device=device, dtype=torch.float32) / (half // 2)
    omega = 1.0 / (10000 ** omega)

    out: List[torch.Tensor] = []
    for i in range(2):  # h, w
        pos = grid[i].reshape(-1)            # (h*w,)
        out_pe = torch.einsum("m,d->md", pos, omega)
        out.append(torch.cat([out_pe.sin(), out_pe.cos()], dim=-1))
    pe = torch.cat(out, dim=-1)              # (h*w, dim)
    return pe.to(dtype=dtype)


class HybridEncoder(nn.Module):
    """Lightweight cross-modal hybrid encoder.

    Phase 2 starting point — simpler than upstream RT-DETRv2's full hybrid:
        - Intra-scale self-attention only on the smallest (stride-32) feature
          map (AIFI-style) — that's where global context matters most.
        - FPN top-down fusion: project each scale, then add upsampled deeper
          features into shallower ones.
        - No PAN bottom-up path (can be added later if needed).

    Text fusion happens in the *decoder* via cross-attention, not here, to
    keep this encoder agnostic of the prompt list size K.

    Input/output: list of 3 NCHW tensors at strides 8, 16, 32 with channel
    dim ``hidden_dim``.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_levels: int = 3,
        n_heads: int = 8,
        dim_ff: int = 1024,
        n_intra_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_levels != 3:
            raise NotImplementedError("HybridEncoder currently fixed to 3 levels")
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels

        self.intra_attn = nn.Sequential(
            *[
                _TransformerEncoderLayer(hidden_dim, n_heads=n_heads, dim_ff=dim_ff)
                for _ in range(n_intra_layers)
            ]
        )

        # Lightweight FPN refinement convs.
        self.fpn_conv_s16 = _ConvBNAct(hidden_dim, hidden_dim, kernel=3, stride=1, padding=1)
        self.fpn_conv_s8 = _ConvBNAct(hidden_dim, hidden_dim, kernel=3, stride=1, padding=1)

    def _aifi(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention on the smallest scale (stride-32). Standard AIFI."""
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        pe = _build_2d_sincos_pos_embed(H, W, C, x.device, x.dtype).unsqueeze(0).expand(B, -1, -1)
        for layer in self.intra_attn:
            seq = layer(seq, pos=pe)
        return seq.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(pyramid) != 3:
            raise ValueError(f"HybridEncoder expects 3 levels, got {len(pyramid)}")
        s8, s16, s32 = pyramid

        # Intra-scale attention on the smallest map.
        s32 = self._aifi(s32)

        # Top-down FPN fusion.
        s16 = self.fpn_conv_s16(s16 + F.interpolate(s32, size=s16.shape[-2:], mode="bilinear", align_corners=False))
        s8 = self.fpn_conv_s8(s8 + F.interpolate(s16, size=s8.shape[-2:], mode="bilinear", align_corners=False))

        return [s8, s16, s32]


# ---------------------------------------------------------------------------
# Open-vocab transformer decoder
# ---------------------------------------------------------------------------


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class _DETRDecoderLayer(nn.Module):
    """One decoder layer: self-attn(queries) -> text cross-attn -> image
    cross-attn -> FFN. Pre-norm. Standard DETR/Grounding-DINO block."""

    def __init__(self, d_model: int, n_heads: int = 8, dim_ff: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.text_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )

    def forward(
        self,
        q: torch.Tensor,
        q_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Self-attention among queries.
        h = self.norm1(q)
        qpos = h + q_pos
        sa, _ = self.self_attn(qpos, qpos, h, need_weights=False)
        q = q + sa

        # 2. Text cross-attention.
        h = self.norm2(q)
        ca, _ = self.text_attn(h + q_pos, text, text, need_weights=False)
        q = q + ca

        # 3. Image cross-attention.
        h = self.norm3(q)
        ia, _ = self.cross_attn(h + q_pos, memory + memory_pos, memory, need_weights=False)
        q = q + ia

        # 4. FFN.
        h = self.norm4(q)
        q = q + self.ffn(h)
        return q


class OpenVocabDecoder(nn.Module):
    """RT-DETR-style transformer decoder with text cross-attention and
    cosine classification.

    Per layer:
        - self-attention(queries)
        - cross-attention(queries, text_tokens)  # the open-vocab piece
        - cross-attention(queries, image_features)
        - FFN

    Box regression is iterative: each layer predicts a box-delta on top of
    the previous reference point. Final cls logits are computed by cosine
    similarity between (q_proj(query) projected to text_dim) and the per-batch
    text embeddings.

    Outputs:
        ``pred_logits`` : (B, Q, K) — last-layer logits (pre-sigmoid, scaled
                          by ``1/temperature``).
        ``pred_boxes``  : (B, Q, 4) — last-layer boxes, cxcywh in [0, 1].
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_layers: int = 6,
        text_dim: int = 1152,
        n_heads: int = 8,
        dim_ff: int = 1024,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.text_dim = text_dim
        self.temperature = temperature

        # Learnable queries + reference points.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # Initial reference points in cxcywh (logit space, learned).
        self.ref_point = nn.Parameter(torch.randn(num_queries, 4) * 0.02)

        # Project text from text_dim (1152) to model dim for cross-attention,
        # and into the cosine-classification space.
        self.text_kv_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Query → text-space projection used for the cosine logits.
        self.q_text_proj = nn.Linear(hidden_dim, text_dim)

        self.layers = nn.ModuleList(
            [
                _DETRDecoderLayer(hidden_dim, n_heads=n_heads, dim_ff=dim_ff)
                for _ in range(num_layers)
            ]
        )
        # Per-layer box-delta heads (3-layer MLP, common DETR pattern).
        self.box_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 4),
                )
                for _ in range(num_layers)
            ]
        )

    @staticmethod
    def _flatten_pyramid(
        pyramid: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate multi-scale features into (B, S, C) and build matching
        (B, S, C) sin-cos position embeddings.
        """
        flats: List[torch.Tensor] = []
        pes: List[torch.Tensor] = []
        for feat in pyramid:
            B, C, H, W = feat.shape
            seq = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            pe = _build_2d_sincos_pos_embed(H, W, C, feat.device, feat.dtype)
            pe = pe.unsqueeze(0).expand(B, -1, -1)
            flats.append(seq)
            pes.append(pe)
        return torch.cat(flats, dim=1), torch.cat(pes, dim=1)

    def forward(
        self,
        encoded: List[torch.Tensor],
        text_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoded:  list of 3 NCHW tensors from the hybrid encoder,
                      each ``(B, hidden_dim, H_l, W_l)``.
            text_emb: (K, text_dim) per-batch text embeddings (already
                      L2-normalized by the SigLIP2 encoder).
        """
        memory, memory_pos = self._flatten_pyramid(encoded)  # (B, S, C), (B, S, C)
        B = memory.shape[0]

        # Project text embeddings into model space for cross-attention.
        text_kv = self.text_kv_proj(text_emb).unsqueeze(0).expand(B, -1, -1)  # (B, K, C)

        # Initialize queries from the learned embedding bank.
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1).contiguous()
        q_pos = q  # use the same embedding as positional cue, common DETR pattern

        # Iterative reference points (cxcywh in [0,1]) starting from the
        # learned ref_point parameter.
        reference = self.ref_point.sigmoid().unsqueeze(0).expand(B, -1, -1).contiguous()

        last_logits: torch.Tensor
        for layer_idx, layer in enumerate(self.layers):
            q = layer(q, q_pos, memory, memory_pos, text_kv)

            # Iterative box refinement.
            delta = self.box_heads[layer_idx](q)
            new_ref = (delta + _inverse_sigmoid(reference)).sigmoid()
            reference = new_ref

        # Final cls logits via cosine of (proj query, text embedding).
        # text_emb is already L2-normalized by SigLIP2.encode_text(normalize=True).
        q_text = self.q_text_proj(q)
        q_text = q_text / (q_text.norm(dim=-1, keepdim=True) + 1e-6)
        text_n = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-6)
        logits = (q_text @ text_n.T) / self.temperature  # (B, Q, K)

        return {"pred_logits": logits, "pred_boxes": reference}


# ---------------------------------------------------------------------------
# Top-level network
# ---------------------------------------------------------------------------


class LibreVocab1Phase2Network(nn.Module):
    """Composes backbone + neck + encoder + decoder for end-to-end forward.

    Inference call:
        out = network(images, text_emb)  # text_emb is (K, 1152)

    Note ``text_emb`` is precomputed externally (typically in the collator,
    via the cached ``TextEmbeddingCache``). At inference, callers can use
    ``self.backbone.encode_text(prompts, device)`` to get it.
    """

    def __init__(
        self,
        size: str = "s",
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_layers: int = 6,
        n_heads: int = 8,
        dim_ff: int = 1024,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        version = "c-radio_v4-so400m" if size == "s" else "c-radio_v4-h"
        self.backbone = CRADIOv4Backbone(version=version, device=device)
        self.neck = SPMv2Neck(embed_dim=self.backbone.embed_dim, hidden_dim=hidden_dim)
        self.encoder = HybridEncoder(
            hidden_dim=hidden_dim, num_levels=3, n_heads=n_heads, dim_ff=dim_ff
        )
        self.decoder = OpenVocabDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            text_dim=1152,
            n_heads=n_heads,
            dim_ff=dim_ff,
        )

    def forward(self, images: torch.Tensor, text_emb: torch.Tensor) -> Dict[str, Any]:
        intermediates = self.backbone.forward_pyramid(images)
        pyramid = self.neck(intermediates, images)
        encoded = self.encoder(pyramid)
        return self.decoder(encoded, text_emb)

"""LibreYOLOWorld — open-vocabulary YOLO architecture (real).

Text-prompted detection with a YOLOv8-CSPDarknet backbone, a frozen CLIP text
encoder, a RepVL-PAN neck (MaxSigmoidCSPLayerWithTwoConv fusion), and a
BNContrastiveHead classifier. Structurally compatible with the Tencent/
YOLO-World-V2.1 release — a separate state-dict remapper (see
`weight_porting.py`) loads their `.pth` checkpoints into this module.

License note: the official YOLO-World weights (`wondervictor/YOLO-World-V2.1`
on HF) are **GPL-3.0**. LibreYOLO itself is MIT, so we never bundle weights.
Users who opt in to the GPL weights at runtime are bound by GPL for their
downstream code — document this clearly in the README.

Architectural reference: https://github.com/AILab-CVC/YOLO-World (`yolo_world/`).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# CLIP ViT-B/32 text embedding dimension (what YOLO-World-V2.1 uses by default).
CLIP_EMBED_DIM = 512

# YOLOv8 scaling factors per size. (deepen, widen, last_stage_out_channels).
# Channel counts per stage (P3/P4/P5) = (256 * widen, 512 * widen, last_stage_out * widen).
YOLO8_SCALES = {
    "s": (0.33, 0.50, 1024),
    "m": (0.67, 0.75, 768),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.25, 512),
}


# ---------------------------------------------------------------------------
# Basic building blocks — match mmyolo / Ultralytics YOLOv8 for weight portability.
# ---------------------------------------------------------------------------


def _autopad(k: int, p: Optional[int] = None) -> int:
    return k // 2 if p is None else p


class ConvModule(nn.Module):
    """Conv -> BN -> SiLU, with sub-module names `conv`/`bn` matching mmyolo's ConvModule."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None,
                 g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DarknetBottleneck(nn.Module):
    """Standard YOLOv8 bottleneck (2x ConvModule, optional residual)."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5,
                 k: Tuple[int, int] = (3, 3)):
        super().__init__()
        c_ = int(c2 * e)
        self.conv1 = ConvModule(c1, c_, k[0], 1)
        self.conv2 = ConvModule(c_, c2, k[1], 1, g=1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class CSPLayerWithTwoConv(nn.Module):
    """YOLOv8 C2f block — mmyolo's name for this: `CSPLayerWithTwoConv`.

    Split-in-half cross-stage with N bottleneck blocks, concat-then-fuse.
    Sub-module names chosen to match mmyolo state dicts:
        main_conv, final_conv, blocks.{0..n-1}
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5):
        super().__init__()
        self.mid_channels = int(c2 * e)
        self.main_conv = ConvModule(c1, 2 * self.mid_channels, 1, 1)
        self.final_conv = ConvModule((2 + n) * self.mid_channels, c2, 1, 1)
        self.blocks = nn.ModuleList(
            DarknetBottleneck(self.mid_channels, self.mid_channels, shortcut=shortcut, e=1.0,
                              k=(3, 3))
            for _ in range(n)
        )

    def forward(self, x):
        y = self.main_conv(x)
        y1, y2 = y.split(self.mid_channels, dim=1)
        outs = [y1, y2]
        for blk in self.blocks:
            outs.append(blk(outs[-1]))
        return self.final_conv(torch.cat(outs, dim=1))


class SPPFBottleneck(nn.Module):
    """YOLOv8 SPPF: three chained maxpools, concat with input, 1x1 fuse."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.conv1 = ConvModule(c1, c_, 1, 1)
        self.conv2 = ConvModule(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


# ---------------------------------------------------------------------------
# Backbone — YOLOv8CSPDarknet.
# ---------------------------------------------------------------------------


def _make_divisible(x: float, divisor: int = 8) -> int:
    return max(divisor, int(x + divisor / 2) // divisor * divisor)


class YOLOv8CSPDarknet(nn.Module):
    """YOLOv8 CSP-Darknet backbone. Outputs P3 / P4 / P5 feature maps.

    Layer naming chosen to match mmyolo's `YOLOv8CSPDarknet` so state_dict
    keys remap cleanly.
    """

    def __init__(self, size: str = "l"):
        super().__init__()
        if size not in YOLO8_SCALES:
            raise ValueError(f"unknown size {size!r}; supported: {list(YOLO8_SCALES)}")
        deepen, widen, last_out = YOLO8_SCALES[size]

        self.size = size
        self.widen = widen
        self.deepen = deepen

        # Stage channel counts (P3/P4/P5)
        ch_p3 = _make_divisible(256 * widen)
        ch_p4 = _make_divisible(512 * widen)
        ch_p5 = _make_divisible(last_out * widen)

        # Depths (C2f num_blocks) per stage — YOLOv8: [3, 6, 6, 3] scaled by deepen
        d1 = max(1, round(3 * deepen))
        d2 = max(1, round(6 * deepen))
        d3 = max(1, round(6 * deepen))
        d4 = max(1, round(3 * deepen))

        ch_stem = _make_divisible(64 * widen)
        ch_s1 = _make_divisible(128 * widen)

        self.stem = ConvModule(3, ch_stem, k=3, s=2)

        # stage{i} = Sequential(down_conv, C2f)
        self.stage1 = nn.Sequential(
            ConvModule(ch_stem, ch_s1, k=3, s=2),
            CSPLayerWithTwoConv(ch_s1, ch_s1, n=d1, shortcut=True),
        )
        self.stage2 = nn.Sequential(
            ConvModule(ch_s1, ch_p3, k=3, s=2),
            CSPLayerWithTwoConv(ch_p3, ch_p3, n=d2, shortcut=True),
        )
        self.stage3 = nn.Sequential(
            ConvModule(ch_p3, ch_p4, k=3, s=2),
            CSPLayerWithTwoConv(ch_p4, ch_p4, n=d3, shortcut=True),
        )
        self.stage4 = nn.Sequential(
            ConvModule(ch_p4, ch_p5, k=3, s=2),
            CSPLayerWithTwoConv(ch_p5, ch_p5, n=d4, shortcut=True),
            SPPFBottleneck(ch_p5, ch_p5, k=5),
        )

        self.out_channels = (ch_p3, ch_p4, ch_p5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


# ---------------------------------------------------------------------------
# CLIP text encoder (frozen).
# ---------------------------------------------------------------------------


class TextEncoder(nn.Module):
    """Frozen HuggingFace CLIP text encoder with a projection head.

    Matches YOLO-World-V2.1's `HuggingCLIPLanguageBackbone` (frozen, using
    the `CLIPTextModelWithProjection` variant with `text_projection.weight`).
    Outputs L2-normalized 512-D text embeddings.
    """

    _HF_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self, embed_dim: int = CLIP_EMBED_DIM):
        super().__init__()
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(self._HF_MODEL)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(self._HF_MODEL)
        for p in self.text_model.parameters():
            p.requires_grad_(False)
        self.text_model.eval()

        proj_dim = self.text_model.config.projection_dim
        self.proj = nn.Identity() if proj_dim == embed_dim else nn.Linear(proj_dim, embed_dim, bias=False)

    @torch.no_grad()
    def encode(self, prompts: Sequence[str], device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = next(self.text_model.parameters()).device
        tokens = self.tokenizer(list(prompts), padding=True, return_tensors="pt").to(device)
        out = self.text_model(**tokens)
        # CLIPTextModelWithProjection returns a ModelOutput with .text_embeds (already projected).
        x = self.proj(out.text_embeds)
        return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# RepVL-PAN fusion — the core YOLO-World innovation.
# ---------------------------------------------------------------------------


class MaxSigmoidAttnBlock(nn.Module):
    """Per-scale text-visual fusion: max-over-class sigmoid gating.

    Given image feature `x` [B, C, H, W] and text embeddings
    `guide` [B, N_cls, D_txt], compute a [B, H, W] per-pixel attention
    (max over classes, per head, softmax-free), sigmoid it, and gate the
    projected visual features.

    Implementation matches `MaxSigmoidAttnBlock` from
    yolo_world/models/layers/yolo_bricks.py.
    """

    def __init__(self, in_channels: int, out_channels: int, guide_channels: int,
                 embed_channels: int, num_heads: int = 8):
        super().__init__()
        if embed_channels % num_heads != 0:
            raise ValueError(f"embed_channels ({embed_channels}) must be divisible by num_heads ({num_heads})")
        if out_channels % num_heads != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.attn_head_channels = embed_channels // num_heads   # Ch for attention dot-product
        self.proj_head_channels = out_channels // num_heads     # Ch for output projection

        self.embed_conv = (
            ConvModule(in_channels, embed_channels, k=1, act=False)
            if in_channels != embed_channels else nn.Identity()
        )
        self.guide_fc = nn.Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.project_conv = ConvModule(in_channels, out_channels, k=3)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        N = guide.shape[1]
        M = self.num_heads

        # Text embeddings projected and reshaped to (B, N, M, attn_Ch)
        g = self.guide_fc(guide).view(B, N, M, self.attn_head_channels)

        # Image embeddings for attention: (B, M, attn_Ch, H, W)
        e = self.embed_conv(x).view(B, M, self.attn_head_channels, H, W)

        # Attention scores: einsum -> (B, M, H, W, N)
        attn = torch.einsum("bmchw,bnmc->bmhwn", e, g)
        attn = attn.max(dim=-1)[0]  # max over classes -> (B, M, H, W)
        attn = attn / math.sqrt(self.attn_head_channels)
        attn = (attn + self.bias.view(1, M, 1, 1)).sigmoid()

        # Image features for output, gated per-head
        out = self.project_conv(x).view(B, M, self.proj_head_channels, H, W)
        out = out * attn.unsqueeze(2)
        return out.view(B, -1, H, W)


class MaxSigmoidCSPLayerWithTwoConv(nn.Module):
    """C2f block with text-visual fusion via MaxSigmoidAttnBlock (RepVL-PAN).

    The attn_block is applied **in-place on the last block's output** (it
    replaces the last block's output in the concat list, not adds a new
    stream). This matches upstream's `(2 + n) * mid` final_conv input shape.
    """

    def __init__(self, in_channels: int, out_channels: int, guide_channels: int,
                 embed_channels: int, num_heads: int, n: int = 1, shortcut: bool = False,
                 e: float = 0.5):
        super().__init__()
        self.mid = int(out_channels * e)
        self.main_conv = ConvModule(in_channels, 2 * self.mid, 1, 1)
        self.blocks = nn.ModuleList(
            DarknetBottleneck(self.mid, self.mid, shortcut=shortcut, e=1.0, k=(3, 3))
            for _ in range(n)
        )
        self.attn_block = MaxSigmoidAttnBlock(
            in_channels=self.mid, out_channels=self.mid,
            guide_channels=guide_channels, embed_channels=embed_channels,
            num_heads=num_heads,
        )
        self.final_conv = ConvModule((2 + n) * self.mid, out_channels, 1, 1)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        y = self.main_conv(x)
        y1, y2 = y.split(self.mid, dim=1)
        outs = [y1, y2]
        for blk in self.blocks:
            outs.append(blk(outs[-1]))
        # Attn applied IN-PLACE on the last block output (replaces, not appends)
        outs[-1] = self.attn_block(outs[-1], guide)
        return self.final_conv(torch.cat(outs, dim=1))


# ---------------------------------------------------------------------------
# Neck — YOLO-World PAFPN with MaxSigmoid fusion at each merge.
# ---------------------------------------------------------------------------


class YOLOWorldPAFPN(nn.Module):
    """PAFPN (top-down + bottom-up) with text-visual fusion via RepVL-PAN blocks."""

    # YOLO-World-V2.1 neck conventions (verified against the S checkpoint):
    #   - n_blocks=2 across all 4 neck CSP layers (size-independent)
    #   - num_heads = mid_channels // 32 per scale (so heads grow with channels)
    #   - embed_channels = mid_channels (=out_channels // 2)
    #   - num_heads is per-scale (P3, P4, P5) since deeper layers have wider features
    DEFAULT_N_BLOCKS = 2

    def __init__(self, in_channels: Tuple[int, int, int], out_channels: Tuple[int, int, int],
                 guide_channels: int = CLIP_EMBED_DIM,
                 embed_channels: Optional[Sequence[int]] = None,
                 num_heads: Optional[Sequence[int]] = None,
                 n_blocks: int = DEFAULT_N_BLOCKS):
        super().__init__()
        c3, c4, c5 = in_channels
        o3, o4, o5 = out_channels

        # Default embed_channels = mid (out_channels // 2). Matches the V2.1 conv shapes.
        if embed_channels is None:
            embed_channels = (o3 // 2, o4 // 2, o5 // 2)
        # Default num_heads = mid // 32 (V2.1 rule: each head sees 32 channels).
        if num_heads is None:
            num_heads = tuple(max(1, ec // 32) for ec in embed_channels)

        # Top-down path: P5 -> P4 -> P3
        self.top_down_layer_1 = MaxSigmoidCSPLayerWithTwoConv(
            in_channels=c5 + c4, out_channels=o4,
            guide_channels=guide_channels,
            embed_channels=embed_channels[1], num_heads=num_heads[1],
            n=n_blocks,
        )
        self.top_down_layer_2 = MaxSigmoidCSPLayerWithTwoConv(
            in_channels=o4 + c3, out_channels=o3,
            guide_channels=guide_channels,
            embed_channels=embed_channels[0], num_heads=num_heads[0],
            n=n_blocks,
        )

        # Bottom-up path
        self.downsample_1 = ConvModule(o3, o3, k=3, s=2)
        self.bottom_up_layer_1 = MaxSigmoidCSPLayerWithTwoConv(
            in_channels=o3 + o4, out_channels=o4,
            guide_channels=guide_channels,
            embed_channels=embed_channels[1], num_heads=num_heads[1],
            n=n_blocks,
        )
        self.downsample_2 = ConvModule(o4, o4, k=3, s=2)
        self.bottom_up_layer_2 = MaxSigmoidCSPLayerWithTwoConv(
            in_channels=o4 + c5, out_channels=o5,
            guide_channels=guide_channels,
            embed_channels=embed_channels[2], num_heads=num_heads[2],
            n=n_blocks,
        )

    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                text_embeds: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = feats

        # Top-down
        p5_up = F.interpolate(p5, scale_factor=2.0, mode="nearest")
        m4 = self.top_down_layer_1(torch.cat([p5_up, p4], dim=1), text_embeds)
        m4_up = F.interpolate(m4, scale_factor=2.0, mode="nearest")
        m3 = self.top_down_layer_2(torch.cat([m4_up, p3], dim=1), text_embeds)

        # Bottom-up
        m3_d = self.downsample_1(m3)
        m4_out = self.bottom_up_layer_1(torch.cat([m3_d, m4], dim=1), text_embeds)
        m4_d = self.downsample_2(m4_out)
        m5_out = self.bottom_up_layer_2(torch.cat([m4_d, p5], dim=1), text_embeds)

        return m3, m4_out, m5_out


# ---------------------------------------------------------------------------
# Head — YOLOv8 reg head + BNContrastiveHead classifier.
# ---------------------------------------------------------------------------


class BNContrastiveHead(nn.Module):
    """YOLO-World's BNContrastiveHead: BN over image-side embed, L2 norm text side.

    cls_logit = (BN(cls_embed) · L2Norm(text_embeds)) * exp(logit_scale) + bias
    """

    def __init__(self, embed_dim: int = CLIP_EMBED_DIM):
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dim)
        # Both bias and logit_scale are scalars (shape ()) to match V2.1 checkpoints.
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))  # exp -> ~14.3
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, cls_embed: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        cls_embed: (B, embed_dim, H, W)
        text_embeds: (B, N_cls, embed_dim)   L2-normalized
        returns: (B, N_cls, H, W) logits
        """
        x = self.norm(cls_embed)
        logits = torch.einsum("bdhw,bnd->bnhw", x, text_embeds)
        return logits * self.logit_scale.exp() + self.bias


class YOLOWorldHeadModule(nn.Module):
    """Three-scale detection head. Regression via DFL; classification via contrastive."""

    def __init__(self, in_channels: Tuple[int, int, int], embed_dim: int = CLIP_EMBED_DIM,
                 reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.num_levels = 3

        # V2.1 head conventions:
        #   reg intermediate channels = 4 * reg_max  (= 64 for reg_max=16)
        #   cls intermediate channels = embed_dim // 4  (= 128 for embed_dim=512)
        c_reg = 4 * reg_max
        c_cls = embed_dim // 4

        self.reg_preds = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()

        for c in in_channels:
            self.reg_preds.append(nn.Sequential(
                ConvModule(c, c_reg, k=3),
                ConvModule(c_reg, c_reg, k=3),
                nn.Conv2d(c_reg, 4 * reg_max, kernel_size=1),
            ))
            self.cls_preds.append(nn.Sequential(
                ConvModule(c, c_cls, k=3),
                ConvModule(c_cls, c_cls, k=3),
                nn.Conv2d(c_cls, embed_dim, kernel_size=1),
            ))
            self.cls_contrasts.append(BNContrastiveHead(embed_dim))

    def forward(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                text_embeds: torch.Tensor):
        bbox_dists = []
        cls_logits = []
        for i, f in enumerate(feats):
            bbox_dists.append(self.reg_preds[i](f))       # (B, 4*reg_max, H, W)
            emb = self.cls_preds[i](f)                    # (B, embed_dim, H, W)
            logits = self.cls_contrasts[i](emb, text_embeds)  # (B, N_cls, H, W)
            cls_logits.append(logits)
        return bbox_dists, cls_logits


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class LibreYOLOWorldModel(nn.Module):
    """Open-vocabulary YOLO detector: YOLOv8 backbone + CLIP text + RepVL-PAN + BNContrastive head.

    Structure follows YOLO-World-V2.1 so `weight_porting.py` can remap Tencent
    state_dicts directly into this module.
    """

    def __init__(self, size: str = "l", imgsz: int = 640, reg_max: int = 16):
        super().__init__()
        self.size = size
        self.imgsz = imgsz
        self.reg_max = reg_max

        self.backbone = YOLOv8CSPDarknet(size=size)
        self.text_encoder = TextEncoder(embed_dim=CLIP_EMBED_DIM)

        # Neck in/out channels = backbone's P3/P4/P5.
        self.neck = YOLOWorldPAFPN(
            in_channels=self.backbone.out_channels,
            out_channels=self.backbone.out_channels,
            guide_channels=CLIP_EMBED_DIM,
        )

        self.head = YOLOWorldHeadModule(
            in_channels=self.backbone.out_channels,
            embed_dim=CLIP_EMBED_DIM,
            reg_max=reg_max,
        )

        # Strides at each scale (YOLOv8 default: 8, 16, 32)
        self.register_buffer("strides", torch.tensor([8.0, 16.0, 32.0]), persistent=False)

        # Cached prompts
        self.register_buffer("_text_embeds", torch.zeros(0, CLIP_EMBED_DIM), persistent=False)
        self._current_prompts: List[str] = []

    # ------------------------------------------------------------------
    # Prompt API
    # ------------------------------------------------------------------

    def set_prompts(self, prompts: Sequence[str]) -> None:
        if not isinstance(prompts, (list, tuple)) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("prompts must be a list of strings")
        if len(prompts) == 0:
            raise ValueError("prompts list must be non-empty")
        device = next(self.parameters()).device
        embeds = self.text_encoder.encode(list(prompts), device=device)
        self._text_embeds = embeds
        self._current_prompts = list(prompts)

    @property
    def prompts(self) -> List[str]:
        return list(self._current_prompts)

    @property
    def num_prompts(self) -> int:
        return len(self._current_prompts)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> dict:
        if self._text_embeds.shape[0] == 0:
            raise RuntimeError("No text prompts set. Call model.set_prompts([...]) first.")
        B = images.shape[0]

        feats = self.backbone(images)  # (P3, P4, P5)

        # Broadcast text to batch: (N, D) -> (B, N, D)
        text = self._text_embeds.unsqueeze(0).expand(B, -1, -1).contiguous()

        neck_feats = self.neck(feats, text)
        bbox_dists, cls_logits = self.head(neck_feats, text)

        return {
            "bbox_dists": bbox_dists,   # list of 3 tensors (B, 4*reg_max, H, W)
            "cls_logits": cls_logits,   # list of 3 tensors (B, N_cls, H, W)
            "strides": self.strides.tolist(),
            "feature_shapes": [tuple(f.shape[-2:]) for f in neck_feats],
        }

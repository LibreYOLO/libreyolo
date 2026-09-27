"""PP-LiteSeg network: STDC backbone + SPPM context + UAFM decoder.

Ported from the Apache-2.0 SuperGradients implementation (see ``NOTICE`` in
this directory for the pinned revision and the STDC lineage). Module and
attribute names mirror the upstream tree exactly, so an upstream checkpoint
loads after stripping its ``module.`` DDP prefix and nothing else.

The public forward takes RGB in ``[0, 1]`` and applies ImageNet
standardization internally on that raw tensor, matching the semantic house
convention (SegFormer, RF-DETR): the dataset, validator, and preprocessor all
stay ``/255``-only and normalization happens exactly once, inside the graph
that gets exported.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# (height, width) validation canvases of the released Cityscapes recipes. The
# "50"/"75" tokens are the source validation scale factors against Cityscapes'
# native 1024x2048, not width multipliers.
SIZE_CONFIGS: Dict[str, Dict] = {
    "t50": {
        "backbone": "stdc1",
        "projection_channels": [64, 128, 128],
        "decoder_channels": [128, 64, 32],
        "head_mid_channels": 32,
        "imgsz": (512, 1024),
        "train_crop": (512, 1024),
        "rescale_range": (0.125, 1.5),
    },
    "b50": {
        "backbone": "stdc2",
        "projection_channels": [96, 128, 128],
        "decoder_channels": [128, 96, 64],
        "head_mid_channels": 64,
        "imgsz": (512, 1024),
        "train_crop": (512, 1024),
        "rescale_range": (0.125, 1.5),
    },
    "t75": {
        "backbone": "stdc1",
        "projection_channels": [64, 128, 128],
        "decoder_channels": [128, 64, 32],
        "head_mid_channels": 32,
        "imgsz": (768, 1536),
        "train_crop": (768, 768),
        "rescale_range": (0.25, 1.75),
    },
    "b75": {
        "backbone": "stdc2",
        "projection_channels": [96, 128, 128],
        "decoder_channels": [128, 96, 64],
        "head_mid_channels": 64,
        "imgsz": (768, 1536),
        "train_crop": (768, 768),
        "rescale_range": (0.25, 1.75),
    },
}

STDC_BLOCK_COUNTS = {"stdc1": [1, 1, 2, 2, 2], "stdc2": [1, 1, 4, 5, 3]}
STDC_CH_WIDTHS = [32, 64, 256, 512, 1024]
STDC_BLOCK_TYPES = ["conv", "conv", "stdc", "stdc", "stdc"]

AUX_HIDDEN_CHANNELS = [32, 64, 64]
AUX_SCALE_FACTORS = [8, 16, 32]
SPPM_POOL_SIZES = [1, 2, 4]
SPPM_INTER_CHANNELS = 128
SPPM_OUT_CHANNELS = 128
DECODER_UP_FACTORS = [1, 2, 2]
HEAD_SCALE_FACTOR = 8
# Product of the encoder strides; input H and W must both be divisible by it.
STRIDE = 32


class ConvBNReLU(nn.Module):
    """Conv-BN-ReLU with the upstream ``seq.{conv,bn,act}`` child naming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
        use_activation: bool = True,
    ) -> None:
        super().__init__()
        self.seq = nn.Sequential()
        self.seq.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            ),
        )
        self.seq.add_module("bn", nn.BatchNorm2d(out_channels))
        if use_activation:
            self.seq.add_module("act", nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class STDCBlock(nn.Module):
    """Short-Term Dense Concatenate block (4 steps, average-pool downsample)."""

    def __init__(self, in_channels: int, out_channels: int, steps: int, stride: int) -> None:
        super().__init__()
        if steps not in (2, 3, 4):
            raise ValueError(f"STDCBlock supports 2, 3 or 4 steps, got {steps}")
        self.conv_list = nn.ModuleList()
        self.conv_list.append(ConvBNReLU(in_channels, out_channels // 2, kernel_size=1, bias=False))
        # The released recipes use `avg_pool` downsampling; the parameterless
        # identity/avg-pool skip keeps the state dict identical either way.
        if stride == 1:
            self.skip_step1: nn.Module = nn.Identity()
        else:
            self.skip_step1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        channels = out_channels // 2
        mid_channels = channels
        for idx in range(1, steps):
            if idx < steps - 1:
                mid_channels //= 2
            self.conv_list.append(
                ConvBNReLU(channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
            )
            channels = mid_channels

        if stride == 2:
            self.conv_list[1] = nn.Sequential(
                ConvBNReLU(
                    out_channels // 2,
                    out_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_channels // 2,
                    bias=False,
                    use_activation=False,
                ),
                self.conv_list[1],
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_list[0](x)
        out_list = [self.skip_step1(x)]
        for conv in self.conv_list[1:]:
            x = conv(x)
            out_list.append(x)
        return torch.cat(out_list, dim=1)


class STDCBackbone(nn.Module):
    """STDC1 / STDC2 backbone returning the stride 8, 16 and 32 features."""

    def __init__(self, num_blocks: Sequence[int], in_channels: int = 3) -> None:
        super().__init__()
        self.stages = nn.ModuleDict()
        self.out_stage_keys: List[str] = []
        self.out_widths: List[int] = []
        down_ratio = 2
        for block_type, width, blocks in zip(STDC_BLOCK_TYPES, STDC_CH_WIDTHS, num_blocks):
            name = f"block_s{down_ratio}"
            self.stages[name] = self._make_stage(in_channels, width, block_type, blocks)
            if down_ratio in (8, 16, 32):
                self.out_stage_keys.append(name)
                self.out_widths.append(width)
            in_channels = width
            down_ratio *= 2

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, block_type: str, num_blocks: int) -> nn.Sequential:
        if block_type == "conv":
            def build(cin: int, cout: int, stride: int) -> nn.Module:
                return ConvBNReLU(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        elif block_type == "stdc":
            def build(cin: int, cout: int, stride: int) -> nn.Module:
                return STDCBlock(cin, cout, steps=4, stride=stride)
        else:
            raise ValueError(f"Unsupported STDC stage block type: {block_type!r}")

        blocks = [build(in_channels, out_channels, 2)]
        blocks += [build(out_channels, out_channels, 1) for _ in range(num_blocks - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for name, stage in self.stages.items():
            x = stage(x)
            if name in self.out_stage_keys:
                outputs.append(x)
        return outputs


class SPPM(nn.Module):
    """Simple Pyramid Pooling Module over the stride-32 feature."""

    def __init__(
        self,
        in_channels: int,
        inter_channels: int,
        out_channels: int,
        pool_sizes: Sequence[int],
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    ConvBNReLU(in_channels, inter_channels, kernel_size=1, bias=False),
                )
                for pool_size in pool_sizes
            ]
        )
        self.conv_out = ConvBNReLU(inter_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.out_channels = out_channels
        self.align_corners = align_corners
        self.pool_sizes = list(pool_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        input_shape = x.shape[2:]
        for branch in self.branches:
            y = branch(x)
            y = F.interpolate(y, size=input_shape, mode="bilinear", align_corners=self.align_corners)
            out = y if out is None else out + y
        return self.conv_out(out)

    def prep_model_for_conversion(self, input_size: Tuple[int, int], stride_ratio: int = STRIDE) -> None:
        """Swap adaptive pooling for fixed-kernel pooling for tracing exporters.

        ``adaptive_avg_pool2d`` has no ONNX lowering when the input size is not
        statically known. The kernel is derived from the *actual* rectangle
        being exported, so a module prepared for 512x1024 must never be reused
        for 768x1536 -- callers work on a deep copy, never the live model.
        """
        feat_h = input_size[-2] / stride_ratio
        feat_w = input_size[-1] / stride_ratio
        for branch in self.branches:
            pool = branch[0]
            if not isinstance(pool, nn.AdaptiveAvgPool2d):
                continue
            out_size = pool.output_size
            out_size = out_size if isinstance(out_size, (tuple, list)) else (out_size, out_size)
            kernel_size = [int(i / o) for i, o in zip((feat_h, feat_w), out_size)]
            branch[0] = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)


class UAFM(nn.Module):
    """Unified Attention Fusion Module (spatial mean/max attention)."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        up_factor: int,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.conv_atten = nn.Sequential(
            ConvBNReLU(4, 2, kernel_size=3, padding=1, bias=False),
            ConvBNReLU(2, 1, kernel_size=3, padding=1, bias=False, use_activation=False),
        )
        self.proj_skip: nn.Module = (
            nn.Identity()
            if skip_channels == in_channels
            else ConvBNReLU(skip_channels, in_channels, kernel_size=3, padding=1, bias=False)
        )
        self.up_x: nn.Module = (
            nn.Identity()
            if up_factor == 1
            else nn.Upsample(scale_factor=up_factor, mode="bilinear", align_corners=align_corners)
        )
        self.conv_out = ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

    @staticmethod
    def _avg_max_spatial_reduce(x: torch.Tensor) -> List[torch.Tensor]:
        return [torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]]

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_x(x)
        skip = self.proj_skip(skip)
        atten = torch.cat(
            [*self._avg_max_spatial_reduce(x), *self._avg_max_spatial_reduce(skip)], dim=1
        )
        atten = torch.sigmoid(self.conv_atten(atten))
        out = x * atten + skip * (1 - atten)
        return self.conv_out(out)


class PPLiteSegEncoder(nn.Module):
    """STDC backbone plus SPPM context, with the three projection convs."""

    def __init__(self, backbone: STDCBackbone, projection_channels: Sequence[int]) -> None:
        super().__init__()
        self.backbone = backbone
        self.context_module = SPPM(
            in_channels=backbone.out_widths[-1],
            inter_channels=SPPM_INTER_CHANNELS,
            out_channels=SPPM_OUT_CHANNELS,
            pool_sizes=SPPM_POOL_SIZES,
            align_corners=False,
        )
        self.proj_convs = nn.ModuleList(
            [
                ConvBNReLU(feat_ch, proj_ch, kernel_size=3, padding=1, bias=False)
                for feat_ch, proj_ch in zip(backbone.out_widths, projection_channels)
            ]
        )
        self.projection_channels = list(projection_channels)

    def get_output_number_of_channels(self) -> List[int]:
        return self.projection_channels + [self.context_module.out_channels]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)
        context = self.context_module(feats[-1])
        feats = [conv(feat) for conv, feat in zip(self.proj_convs, feats)]
        return feats + [context]


class PPLiteSegDecoder(nn.Module):
    """Three UAFM stages fusing the SPPM stream back down to stride 8."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        up_factors: Sequence[int],
        out_channels: Sequence[int],
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        reversed_channels = list(encoder_channels)[::-1]
        in_channels = reversed_channels[0]
        self.up_stages = nn.ModuleList()
        for skip_ch, up_factor, out_ch in zip(reversed_channels[1:], up_factors, out_channels):
            self.up_stages.append(
                UAFM(
                    in_channels=in_channels,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    up_factor=up_factor,
                    align_corners=align_corners,
                )
            )
            in_channels = out_ch

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        # Reverse a local copy: upstream reverses the caller's list in place,
        # which is a latent aliasing bug we do not reproduce.
        reversed_feats = list(feats)[::-1]
        x = reversed_feats[0]
        for up_stage, skip in zip(self.up_stages, reversed_feats[1:]):
            x = up_stage(x, skip)
        return x


class SegmentationHead(nn.Module):
    """3x3 ConvBNReLU, dropout, 1x1 classifier (no bias)."""

    def __init__(self, in_channels: int, mid_channels: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.seg_head = nn.Sequential(
            ConvBNReLU(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.Dropout(dropout),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(x)

    def replace_num_classes(self, num_classes: int) -> None:
        old = self.seg_head[-1]
        new = nn.Conv2d(old.in_channels, num_classes, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
        self.seg_head[-1] = new


class LibrePPLiteSegNet(nn.Module):
    """PP-LiteSeg t50/b50/t75/b75.

    ``forward`` returns the main logits ``(B, nc, H, W)`` in eval mode. With
    ``use_aux_heads=True`` and training mode it returns the source 4-tuple
    ``(main, aux_s8, aux_s16, aux_s32)``; the auxiliary heads are never part of
    the public inference or export graph.
    """

    def __init__(
        self,
        size: str = "t50",
        num_classes: int = 19,
        dropout: float = 0.0,
        use_aux_heads: bool = True,
    ) -> None:
        super().__init__()
        if size not in SIZE_CONFIGS:
            raise ValueError(f"Unknown PP-LiteSeg size {size!r}; expected one of {tuple(SIZE_CONFIGS)}")
        config = SIZE_CONFIGS[size]
        self.size = size
        self.num_classes = num_classes
        self._use_aux_heads = bool(use_aux_heads)

        backbone = STDCBackbone(STDC_BLOCK_COUNTS[config["backbone"]])
        projection_channels = list(config["projection_channels"])
        self.encoder = PPLiteSegEncoder(backbone, projection_channels)
        self.decoder = PPLiteSegDecoder(
            encoder_channels=self.encoder.get_output_number_of_channels(),
            up_factors=DECODER_UP_FACTORS,
            out_channels=config["decoder_channels"],
            align_corners=False,
        )
        self.seg_head = nn.Sequential(
            SegmentationHead(
                in_channels=config["decoder_channels"][-1],
                mid_channels=config["head_mid_channels"],
                num_classes=num_classes,
                dropout=dropout,
            ),
            nn.Upsample(scale_factor=HEAD_SCALE_FACTOR, mode="bilinear", align_corners=False),
        )
        if self._use_aux_heads:
            self.aux_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        SegmentationHead(backbone_ch, hidden_ch, num_classes, dropout=dropout),
                        nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
                    )
                    for backbone_ch, hidden_ch, scale in zip(
                        projection_channels, AUX_HIDDEN_CHANNELS, AUX_SCALE_FACTORS
                    )
                ]
            )
        self.init_params()

        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    # ------------------------------------------------------------------

    def init_params(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    @property
    def use_aux_heads(self) -> bool:
        return self._use_aux_heads

    def remove_aux_heads(self) -> None:
        """Drop the training-only auxiliary heads (used on export copies)."""
        if hasattr(self, "aux_heads"):
            del self.aux_heads
        self._use_aux_heads = False

    def replace_num_classes(self, num_classes: int) -> None:
        """Rebuild the main and every auxiliary classifier for a new class count."""
        for module in self.modules():
            if isinstance(module, SegmentationHead):
                module.replace_num_classes(num_classes)
        self.num_classes = num_classes

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet-standardize an RGB tensor already scaled to ``[0, 1]``.

        Device deliberately comes from the buffers themselves (they move with
        ``Module.to``) rather than from ``x``: a traced ``.to(x.device)`` bakes
        the *export-time* device into the graph, so a CPU-traced TorchScript
        artifact then fails on a CUDA input.
        """
        return (x - self.pixel_mean.to(dtype=x.dtype)) / self.pixel_std.to(dtype=x.dtype)

    def forward_features(self, x: torch.Tensor):
        """Return ``(projected_feats, sppm, decoder_out)`` for parity probing."""
        feats = self.encoder(x)
        decoded = self.decoder(feats)
        return feats[:-1], feats[-1], decoded

    def forward(self, x: torch.Tensor):
        x = self.normalize(x)
        feats = self.encoder(x)
        enc_feats = feats[:-1]
        decoded = self.decoder(feats)
        main = self.seg_head(decoded)
        if not (self._use_aux_heads and self.training):
            return main
        aux = [head(feat) for feat, head in zip(enc_feats, self.aux_heads)]
        return tuple([main] + aux)


__all__ = [
    "AUX_HIDDEN_CHANNELS",
    "AUX_SCALE_FACTORS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SIZE_CONFIGS",
    "STDC_BLOCK_COUNTS",
    "STRIDE",
    "LibrePPLiteSegNet",
    "SegmentationHead",
    "SPPM",
    "UAFM",
]

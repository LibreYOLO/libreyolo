"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE/)
Copyright (c) 2024 D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# Adapted for LibreYOLO from Intellindust-AI-Lab/GTR
# revision 782e737efe2e6437ac537fbdcee089673d3376c1 (MIT).
# Changes: native construction, local imports and portable attention. See NOTICE.

import torch
from torch import nn
from torch.nn import functional as F

from .utils import get_activation


class _LayerNorm2d(nn.Module):
    """Channel-first LayerNorm: input/output shape [B, C, H, W]."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
        self.normalized_shape = (channels,)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class _ConvBNAct(nn.Module):
    """Conv -> BN(or LN) -> SiLU."""

    def __init__(
        self,
        ch_in,
        ch_out,
        kernel=3,
        stride=1,
        groups=1,
        dilation=1,
        act="silu",
        layer_norm=False,
        force_contiguous=True,
    ):
        super().__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.norm = _LayerNorm2d(ch_out) if layer_norm else nn.BatchNorm2d(ch_out)
        self.act = get_activation(act) if act is not None else nn.Identity()
        self.force_contiguous = force_contiguous

    def forward(self, x):
        # The `.contiguous()` is training-lineage dependent, so it is a per-encoder flag
        # (GTREncoder.force_contiguous) rather than a hard-coded call:
        #   * det / seg / semseg / obb (force_contiguous=True, default): the preceding
        #     _LayerNorm2d returns a permute() view with channels_last strides; without the
        #     copy cuDNN runs NHWC kernels whose rounding differs from the NCHW path every
        #     released checkpoint of those tasks was trained/evaluated with (first visible in
        #     the stride-32 stage, then amplified by the decoder's top-k query selection).
        #   * depth (force_contiguous=False): the released GTRDepth checkpoints were trained
        #     WITHOUT the copy, so forcing it here breaks their bit-exact reproduction.
        # The copy costs a few % in the deploy graph; per-task parity with the training code wins.
        # NOTE: rewriting 1x1/s1 convs as F.linear was tried and is a no-op under
        # torch.compile (inductor already lowers 1x1 conv to the same mm — verified
        # bit-identical and equal speed standalone); it only pessimizes eager runs.
        if self.force_contiguous:
            x = x.contiguous()
        return self.act(self.norm(self.conv(x)))


class _Bottleneck(nn.Module):
    """YOLOv8-style 3x3 -> 3x3 bottleneck with optional shortcut."""

    def __init__(
        self,
        c_in,
        c_out,
        shortcut=True,
        e=1.0,
        act="silu",
        layer_norm=False,
        force_contiguous=True,
    ):
        super().__init__()
        c_hid = int(c_out * e)
        self.cv1 = _ConvBNAct(
            c_in,
            c_hid,
            kernel=3,
            stride=1,
            act=act,
            layer_norm=layer_norm,
            force_contiguous=force_contiguous,
        )
        self.cv2 = _ConvBNAct(
            c_hid,
            c_out,
            kernel=3,
            stride=1,
            act=act,
            layer_norm=layer_norm,
            force_contiguous=force_contiguous,
        )
        self.add = shortcut and c_in == c_out

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class _C2f(nn.Module):
    """YOLOv8 C2f (CSP with 2 convs and n bottlenecks)."""

    def __init__(
        self,
        c_in,
        c_out,
        n=3,
        shortcut=False,
        e=0.5,
        act="silu",
        layer_norm=False,
        force_contiguous=True,
    ):
        super().__init__()
        self.c = int(c_out * e)
        self.cv1 = _ConvBNAct(
            c_in,
            2 * self.c,
            kernel=1,
            stride=1,
            act=act,
            layer_norm=layer_norm,
            force_contiguous=force_contiguous,
        )
        self.cv2 = _ConvBNAct(
            (2 + n) * self.c,
            c_out,
            kernel=1,
            stride=1,
            act=act,
            layer_norm=layer_norm,
            force_contiguous=force_contiguous,
        )
        self.m = nn.ModuleList(
            _Bottleneck(
                self.c,
                self.c,
                shortcut=shortcut,
                e=1.0,
                act=act,
                layer_norm=layer_norm,
                force_contiguous=force_contiguous,
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class GTREncoder(nn.Module):
    """RF-DETR / LW-DETR 风格的 MultiScaleProjector。

    输入：来自 ViT 的 N 个同分辨率特征 [B, in_dim, H, W]（H=W=img/patch_size）。
    输出：len(scale_factors) 个尺度的金字塔 [B, hidden_dim, H', W']。

    每个目标尺度独立一组：
        - 对每个输入层应用 (scale, layer) 专属重采样：
            * scale=4.0  -> 2 次 ConvTranspose2d(↑2，通道 //2)
            * scale=2.0  -> 1 次 ConvTranspose2d(↑2，通道 //2)
            * scale=1.0  -> identity
            * scale=0.5  -> stride-2 ConvBNAct(下采样 ½，通道不变)
        - 沿通道 concat
        - C2f(in_dim_after_resample -> hidden_dim) + LayerNorm2d
    """

    def __init__(
        self,
        in_channels,  # list[int]，长度 N，ViT 各层通道数（一般都相同）
        hidden_dim=256,
        scale_factors=(2.0, 1.0, 0.5),
        feat_strides=(8, 16, 32),  # 每个 scale 对应的下游 stride，喂给 decoder
        num_blocks=3,  # C2f 内 bottleneck 个数
        layer_norm=True,  # C2f 内是否使用 LayerNorm2d (True: 与 RF-DETR 一致)
        act="silu",
        eval_spatial_size=None,  # 仅占位，与 HybridEncoder 接口对齐
        force_contiguous=True,  # _ConvBNAct 是否对 conv 输入强制 .contiguous()（见该类注释）
        # det/seg/semseg/obb=True；depth 发布权重训练时无此拷贝，需置 False
    ):
        super().__init__()
        in_channels = list(in_channels)
        scale_factors = list(scale_factors)
        feat_strides = list(feat_strides)
        assert len(scale_factors) == len(feat_strides), (
            f"len(scale_factors)={len(scale_factors)} != len(feat_strides)={len(feat_strides)}"
        )

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.scale_factors = scale_factors
        self.feat_strides = feat_strides
        self.out_channels = [hidden_dim for _ in scale_factors]
        self.out_strides = feat_strides
        self.eval_spatial_size = eval_spatial_size

        # 每个目标尺度一组 per-layer 重采样模块 + 一个 C2f 融合
        stages_sampling = []
        stages = []
        for scale in scale_factors:
            samp_per_layer = []
            in_dim_after_resample = 0
            for in_dim in in_channels:
                if scale == 4.0:
                    layers = [
                        nn.ConvTranspose2d(
                            in_dim, in_dim // 2, kernel_size=2, stride=2
                        ),
                        _LayerNorm2d(in_dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(
                            in_dim // 2, in_dim // 4, kernel_size=2, stride=2
                        ),
                    ]
                    out_c_layer = in_dim // 4
                elif scale == 2.0:
                    layers = [
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2)
                    ]
                    out_c_layer = in_dim // 2
                elif scale == 1.0:
                    layers = []
                    out_c_layer = in_dim
                elif scale == 0.5:
                    layers = [
                        _ConvBNAct(
                            in_dim,
                            in_dim,
                            kernel=3,
                            stride=2,
                            act=act,
                            layer_norm=layer_norm,
                            force_contiguous=force_contiguous,
                        )
                    ]
                    out_c_layer = in_dim
                else:
                    raise NotImplementedError(f"Unsupported scale_factor: {scale}")
                samp_per_layer.append(
                    nn.Sequential(*layers) if layers else nn.Identity()
                )
                in_dim_after_resample += out_c_layer
            stages_sampling.append(nn.ModuleList(samp_per_layer))

            stages.append(
                nn.Sequential(
                    _C2f(
                        in_dim_after_resample,
                        hidden_dim,
                        n=num_blocks,
                        act=act,
                        layer_norm=layer_norm,
                        force_contiguous=force_contiguous,
                    ),
                    _LayerNorm2d(hidden_dim),
                )
            )

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

    def convert_to_deploy(self):
        # The per-scale stages only read `feats` and are mutually independent, and
        # their conv kernels are too small to fill the GPU one at a time. In deploy
        # mode run stage 0 on the current stream and fork the remaining stages onto
        # side streams; inside an outer CUDA-graph capture this records parallel
        # branches that overlap on idle SMs.
        self._deploy_parallel_stages = True
        self._stage_streams = None

    def _run_stage(self, stage_idx, feats):
        samp_list = self.stages_sampling[stage_idx]
        fuse = self.stages[stage_idx]
        resampled = [samp(feats[j]) for j, samp in enumerate(samp_list)]
        fused = resampled[0] if len(resampled) == 1 else torch.cat(resampled, dim=1)
        return fuse(fused)

    # Dynamo must not trace the stream fork/join (inductor's stream codegen chokes
    # on it); recursive=False so the per-stage compiled submodules still run their
    # compiled artifacts when invoked from this eager frame.
    @torch.compiler.disable(recursive=False)
    def _forward_parallel_stages(self, feats):
        cur = torch.cuda.current_stream()
        if self._stage_streams is None:
            self._stage_streams = [
                torch.cuda.Stream() for _ in range(len(self.stages) - 1)
            ]
        outs = [None] * len(self.stages)
        for i, s in enumerate(self._stage_streams):
            s.wait_stream(cur)
            with torch.cuda.stream(s):
                outs[i + 1] = self._run_stage(i + 1, feats)
        outs[0] = self._run_stage(0, feats)
        capturing = torch.cuda.is_current_stream_capturing()
        for i, s in enumerate(self._stage_streams):
            cur.wait_stream(s)
            if not capturing:
                # Mark cross-stream use so the caching allocator does not hand the
                # block back to the side stream while the main stream still reads it.
                outs[i + 1].record_stream(cur)
        return outs

    def forward(self, feats):
        assert len(feats) == len(self.in_channels), (
            f"expect {len(self.in_channels)} ViT-layer features, got {len(feats)}"
        )

        if (
            getattr(self, "_deploy_parallel_stages", False)
            and not self.training
            and feats[0].is_cuda
            and len(self.stages) > 1
        ):
            return self._forward_parallel_stages(feats)

        outs = []
        for stage_idx, (samp_list, fuse) in enumerate(
            zip(self.stages_sampling, self.stages)
        ):
            resampled = [samp(feats[j]) for j, samp in enumerate(samp_list)]
            fused = resampled[0] if len(resampled) == 1 else torch.cat(resampled, dim=1)
            outs.append(fuse(fused))
        return outs

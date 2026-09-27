"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
"""

# Adapted for LibreYOLO from Intellindust-AI-Lab/GTR
# revision 782e737efe2e6437ac537fbdcee089673d3376c1 (MIT).
# Changes: native construction, local imports and portable attention. See NOTICE.

import math
import warnings
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .attention import GatedLinearAttention


def get_activation(act, inplace=True):
    if act is None:
        return nn.Identity()
    elif isinstance(act, nn.Module):
        return act
    act = act.lower()
    if act in ("silu", "swish"):
        m = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError(f"Unknown activation: {act}")
    if hasattr(m, "inplace"):
        m.inplace = inplace
    return m


class ConvNormLayer_fuse(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        stride,
        g=1,
        padding=None,
        bias=False,
        act=None,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        (
            self.ch_in,
            self.ch_out,
            self.kernel_size,
            self.stride,
            self.g,
            self.padding,
            self.bias,
        ) = ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, "conv_bn_fused"):
            return self.act(self.conv_bn_fused(x))
        return self.act(self.norm(self.conv(x)))

    def convert_to_deploy(self):
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True,
            )
        k, b = self._fuse_bn_tensor()
        self.conv_bn_fused.weight.data = k
        self.conv_bn_fused.bias.data = b
        self.__delattr__("conv")
        self.__delattr__("norm")

    def _fuse_bn_tensor(self):
        k = self.conv.weight
        rm, rv = self.norm.running_mean, self.norm.running_var
        g, b, eps = self.norm.weight, self.norm.bias, self.norm.eps
        std = (rv + eps).sqrt()
        t = (g / std).reshape(-1, 1, 1, 1)
        return k * t, b - rm * g / std

    def get_equivalent_kernel_bias(self):
        return self._fuse_bn_tensor()


class GLAMLP(nn.Module):
    def __init__(self, hidden_size, hidden_ratio=4, intermediate_size=None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        # Use pure PyTorch SwiGLU to avoid Triton autotuner failures in swiglu_linear.
        z = F.silu(gate) * y
        return F.linear(z, self.down_proj.weight, self.down_proj.bias)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class ConvPyramidPatchEmbed(nn.Module):
    def __init__(self, embed_dim=192, patch_size=16, act="relu"):
        super().__init__()
        assert patch_size == 16
        num_stages = int(math.log2(patch_size)) - 1
        ratios = [2**i for i in range(num_stages, 0, -1)]
        channels = [embed_dim // r for r in ratios]
        self.convs = nn.ModuleList(
            [
                ConvNormLayer_fuse(ic, oc, 3, 2, act=act)
                for ic, oc in zip([3] + channels[:-1], channels)
            ]
        )
        self.proj = nn.Conv2d(channels[-1], embed_dim, 3, 2, padding=1)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return self.proj(x).flatten(2).transpose(1, 2)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1).erfinv_()
        tensor.mul_(std * math.sqrt(2.0)).add_(mean).clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_idx=None,
        mode="chunk",
        expand_k=0.5,
        expand_v=1.0,
        use_short_conv=False,
        conv_size=4,
        use_output_gate=True,
        gate_fn="swish",
        gate_logit_normalizer=16,
        gate_low_rank_dim=16,
        fuse_norm=True,
        use_glamlp=True,
        layerscale=False,
        layerscale_init_value=1e-4,
        qk_norm=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GatedLinearAttention(
            mode=mode,
            hidden_size=dim,
            expand_k=expand_k,
            expand_v=expand_v,
            num_heads=num_heads,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            use_output_gate=use_output_gate,
            gate_fn=gate_fn,
            gate_logit_normalizer=gate_logit_normalizer,
            gate_low_rank_dim=gate_low_rank_dim,
            fuse_norm=fuse_norm,
            layer_idx=layer_idx,
            qk_norm=qk_norm,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        if use_glamlp:
            self.mlp = GLAMLP(hidden_size=dim, hidden_ratio=int(mlp_ratio))
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=drop,
            )
        self.layerscale = layerscale
        if layerscale:
            self.gamma1 = nn.Parameter(
                layerscale_init_value * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layerscale_init_value * torch.ones(dim), requires_grad=True
            )

    def forward(self, x):
        attn_output, _, _ = self.attn(self.norm1(x))
        if self.layerscale:
            x = x + self.drop_path(self.gamma1 * attn_output)
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(attn_output)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        return_layers=(3, 7, 11),
        embed_layer=ConvPyramidPatchEmbed,
        norm_layer=None,
        act_layer=None,
        mode="chunk",
        expand_k=0.5,
        expand_v=1.0,
        use_short_conv=False,
        conv_size=4,
        use_output_gate=True,
        gate_fn="swish",
        gate_logit_normalizer=16,
        gate_low_rank_dim=16,
        fuse_norm=True,
        use_glamlp=True,
        layerscale=False,
        layerscale_init_value=1e-4,
        qk_norm=False,
        use_pos_embed=True,
        if_bidirectional=False,
        bid_scan=True,
        if_quad_dir=True,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.depth = depth
        self.return_layers = list(return_layers)
        self.if_bidirectional = if_bidirectional
        self.bid_scan = bid_scan
        self.if_quad_dir = if_quad_dir
        self.use_pos_embed = use_pos_embed
        self.grid_size = img_size // patch_size
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self._model = nn.Module()
        if embed_layer is PatchEmbed:
            self._model.patch_embed = embed_layer(
                img_size, patch_size, in_chans, embed_dim
            )
        else:
            self._model.patch_embed = embed_layer(
                embed_dim=embed_dim, patch_size=patch_size
            )
        self.patch_size = patch_size
        self._model.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self._model.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.use_pos_embed:
            num_patches = self.grid_size * self.grid_size
            self._model.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim)
            )
            self._model.pos_drop = nn.Dropout(p=drop_rate)
        self._model.norm = norm_layer(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self._model.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    layer_idx=i,
                    mode=mode,
                    expand_k=expand_k,
                    expand_v=expand_v,
                    use_short_conv=use_short_conv,
                    conv_size=conv_size,
                    use_output_gate=use_output_gate,
                    gate_fn=gate_fn,
                    gate_logit_normalizer=gate_logit_normalizer,
                    gate_low_rank_dim=gate_low_rank_dim,
                    fuse_norm=fuse_norm,
                    use_glamlp=use_glamlp,
                    layerscale=layerscale,
                    layerscale_init_value=layerscale_init_value,
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )
        self.init_weights()

    def init_weights(self, prenorm_residual_strategy=None, num_residuals_per_layer=2):
        trunc_normal_(self._model.cls_token, std=0.02)
        trunc_normal_(self._model.mask_token, std=0.02)
        if self.use_pos_embed:
            trunc_normal_(self._model.pos_embed, std=0.02)
        self.apply(self._init_vit_weights)
        if prenorm_residual_strategy is not None:
            for m in self.modules():
                p = None
                if hasattr(m, "o_proj"):
                    p = m.o_proj.weight
                elif hasattr(m, "down_proj"):
                    p = m.down_proj.weight
                if p is not None:
                    if prenorm_residual_strategy == "rescale":
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                        with torch.no_grad():
                            p /= math.sqrt(num_residuals_per_layer * self.depth)
                    elif prenorm_residual_strategy == "zero":
                        nn.init.zeros_(p)
                    else:
                        raise ValueError(
                            f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}"
                        )

    def _init_vit_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif hasattr(m, "reset_parameters"):
            m.reset_parameters()

    @torch.jit.ignore
    def no_weight_decay(self):
        base = {"cls_token", "mask_token"}
        if self.use_pos_embed:
            base.add("pos_embed")
        return base

    def get_model(self):
        return self._model

    def feature_dim(self):
        return self.embed_dim

    def interpolate_pos_embed(self, num_patches_h, num_patches_w):
        """Interpolate pos_embed to match (num_patches_h, num_patches_w) grid."""
        pos_embed = self._model.pos_embed
        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]
        gs_train = self.grid_size
        if num_patches_h == gs_train and num_patches_w == gs_train:
            return pos_embed
        D = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, gs_train, gs_train, D).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos.float(),
            size=(num_patches_h, num_patches_w),
            mode="bicubic",
            align_corners=False,
        ).to(patch_pos.dtype)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
            1, num_patches_h * num_patches_w, D
        )
        return torch.cat((cls_pos, patch_pos), dim=1)

    def _apply_bid_scan(self, x, layer_idx):
        # In deploy mode (static eval shape), use a precomputed per-layer index buffer:
        # one ONNX Gather instead of Slice+Flip+Reshape+Transpose+Flatten chain.
        if getattr(self, "_deploy_static_perm", False):
            # Stacked [depth, L+1] buffer, integer-indexed so Dynamo can trace through
            # the loop without recompiling per layer (avoids ``getattr(self, f'..._{i}')``).
            full_perm = self._bid_step_perm_full[layer_idx]
            return x.index_select(1, full_perm)
        cls_tok, patches = x[:, :1, :], x[:, 1:, :]
        patches = patches.flip(1)
        if layer_idx % 2 == 1 and self.if_quad_dir:
            gs = int(patches.shape[1] ** 0.5)
            patches = (
                patches.reshape(patches.shape[0], gs, gs, -1)
                .transpose(1, 2)
                .flatten(1, 2)
            )
        return torch.cat((cls_tok, patches), dim=1)

    def _update_perm(self, perm, layer_idx):
        perm = perm.flip(0)
        if layer_idx % 2 == 1 and self.if_quad_dir:
            gs = int(perm.shape[0] ** 0.5)
            perm = perm.reshape(gs, gs).t().reshape(-1).contiguous()
        return perm

    def random_masking(self, x, mask_ratio):
        N, L, _D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones([N, L], device=x.device, dtype=torch.bool)
        mask[:, :len_keep] = False
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return ids_keep, mask, ids_restore

    def forward_distill(self, x, mask_ratio=0.75, num_stages=4):
        """Masked distillation forward (CLS at front, bid_scan permutation tracked).

        Returns:
            stage_feats: list of K [B, L', D]
            final_feats: [B, L, D] post-norm, original spatial order
            mask:        [B, L] bool
            ids_keep:    [B, L']
        """
        B, _, H_img, W_img = x.shape
        x_embed = self._model.patch_embed(x)
        L, D = x_embed.shape[1], x_embed.shape[2]

        if self.use_pos_embed:
            nH, nW = H_img // self.patch_size, W_img // self.patch_size
            pos_embed = self.interpolate_pos_embed(nH, nW)
            x_embed = x_embed + pos_embed[:, 1:, :]

        ids_keep, mask, _ids_restore = self.random_masking(x_embed, mask_ratio)

        mask_tokens = self._model.mask_token.expand(B, L, -1)
        x_masked = torch.where(mask.unsqueeze(-1), mask_tokens, x_embed)

        cls_token = self._model.cls_token.expand(B, -1, -1)
        if self.use_pos_embed:
            cls_token = cls_token + pos_embed[:, :1, :]
        x = torch.cat([cls_token, x_masked], dim=1)  # CLS at front
        if self.use_pos_embed:
            x = self._model.pos_drop(x)

        stage_size = self.depth // num_stages
        stage_ends = {
            min((k + 1) * stage_size - 1, self.depth - 1) for k in range(num_stages)
        }

        perm = torch.arange(L, device=x.device)

        stage_feats = []
        for i, blk in enumerate(self._model.blocks):
            x = blk(x)
            if self.bid_scan:
                x = self._apply_bid_scan(x, i)
                perm = self._update_perm(perm, i)
            if i in stage_ends:
                patch_feats = x[:, 1:, :]
                perm_inv = torch.argsort(perm)
                visible = torch.gather(
                    patch_feats[:, perm_inv, :],
                    1,
                    ids_keep.unsqueeze(-1).expand(-1, -1, D),
                )
                stage_feats.append(visible)

        perm_inv = torch.argsort(perm)
        patch_orig = x[:, 1:, :][:, perm_inv, :]
        x_orig = torch.cat([x[:, :1, :], patch_orig], dim=1)
        final_feats = self._model.norm(x_orig)[:, 1:, :]

        return stage_feats, final_feats, mask, ids_keep

    def forward(self, x, return_layer_indexes=None, n=1, mask_ratio=0.0, num_stages=0):
        if num_stages > 0:
            return self.forward_distill(x, mask_ratio=mask_ratio, num_stages=num_stages)

        if return_layer_indexes is None:
            rl_list = list(range(self.depth - n, self.depth))
        else:
            rl_list = sorted({int(i) for i in return_layer_indexes})
        rl_set = set(rl_list)

        B, _, H_img, W_img = x.shape
        patches = self._model.patch_embed(x)
        cls_token = self._model.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, patches], dim=1)  # CLS at front
        if self.use_pos_embed:
            nH, nW = H_img // self.patch_size, W_img // self.patch_size
            x = x + self.interpolate_pos_embed(nH, nW)
            x = self._model.pos_drop(x)

        L = x.shape[1] - 1  # patch 数量（去掉 CLS）
        deploy_static = getattr(self, "_deploy_static_perm", False)
        if self.bid_scan and not deploy_static:
            perm = torch.arange(L, device=x.device)
            # bid_scan 的置换具有周期性：quad_dir 每 4 层回到 identity，否则每 2 层。
            # 在 identity 层返回特征可跳过 argsort + gather 的逆置换。
            period = 4 if self.if_quad_dir else 2

        outs_by_idx = {}
        for i, blk in enumerate(self._model.blocks):
            x = blk(x)
            if self.bid_scan:
                x = self._apply_bid_scan(x, i)
                if not deploy_static:
                    perm = self._update_perm(perm, i)
            if i in rl_set:
                if deploy_static:
                    # convert_to_deploy() asserts return layers fall on period boundaries
                    # so the inverse permutation is identity — drop the argsort+gather entirely.
                    patch_out = x[:, 1:, :]
                elif self.bid_scan and ((i + 1) % period != 0):
                    perm_inv = torch.argsort(perm)
                    patch_out = x[:, 1:, :][:, perm_inv, :]
                else:
                    patch_out = x[:, 1:, :]
                outs_by_idx[i] = (patch_out, x[:, 0, :])

        return [outs_by_idx[i] for i in rl_list]


EMBED_LAYER_REGISTRY = {
    "ConvPyramidPatchEmbed": ConvPyramidPatchEmbed,
    "PatchEmbed": PatchEmbed,
}


FFN_LAYER_REGISTRY = {
    "mlp": Mlp,
    # "swigluffn": SwiGLUFFN,  # To be implemented
}


class ViTAdapter(nn.Module):
    def __init__(
        self,
        weights_path=None,
        interaction_indexes=None,
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        proj_dim=None,
        num_levels=3,
        embed_layer="ConvPyramidPatchEmbed",
        ffn_layer="mlp",
        ffn_ratio=4,
        skip_weights_warning=False,
        multi_layer_same_res=False,
        eval_spatial_size=None,
        **kwargs,
    ):
        if interaction_indexes is None:
            interaction_indexes = []
        super().__init__()
        self.eval_spatial_size = eval_spatial_size
        # multi_layer_same_res=True: 直接返回 N 个原生分辨率特征 (1/patch_size)，
        # 不做 bilinear 重采样、不做 1x1 投影；交给下游 RFDETR-style encoder 处理。
        self.multi_layer_same_res = multi_layer_same_res
        if isinstance(embed_layer, str):
            try:
                embed_layer = EMBED_LAYER_REGISTRY[embed_layer]
            except KeyError as exc:
                valid_names = ", ".join(sorted(EMBED_LAYER_REGISTRY))
                raise ValueError(
                    f"Unknown embed_layer '{embed_layer}'. Expected one of: {valid_names}"
                ) from exc
        elif not callable(embed_layer):
            raise TypeError(
                f"embed_layer must be a string registry key or callable, got {type(embed_layer).__name__}"
            )

        # ffn_layer 暂不支持。ffn_ratio 映射为 VisionTransformer 的 mlp_ratio。
        # 注意 ffn_ratio 是显式参数，YAML 传 ffn_ratio 时不会进 kwargs；
        # 这里以显式 ffn_ratio 为准，仅当 kwargs 中显式给出 mlp_ratio 时才覆盖。
        vit_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("ffn_layer", "ffn_ratio", "mlp_ratio")
        }
        vit_kwargs["mlp_ratio"] = float(kwargs.get("mlp_ratio", ffn_ratio))
        self.backbone = self._build_vit(
            embed_dim=embed_dim,
            num_heads=num_heads,
            return_layers=interaction_indexes,
            patch_size=patch_size,
            embed_layer=embed_layer,
            **vit_kwargs,
        )
        if not skip_weights_warning:
            self._load_weights(weights_path)

        self.patch_size = patch_size
        self.num_levels = num_levels
        self.embed_dim = embed_dim

        if self.multi_layer_same_res:
            # 任意 N 层都允许；下游 GTREncoder 自行决定要扩展到多少 scale。
            assert len(interaction_indexes) > 0, "interaction_indexes must be non-empty"
            self.interaction_indexes = sorted(int(i) for i in interaction_indexes)
            self.level_scales = None
            self.proj_dim = [embed_dim]
            self.projector = nn.ModuleList()
        else:
            assert num_levels == 3, NotImplementedError(
                "Only support num_levels=3 for ViTAdapter now."
            )
            assert len(interaction_indexes) == num_levels, (
                f"interaction_indexes must provide exactly {num_levels} layer indices, "
                f"got {interaction_indexes}"
            )
            # 约定：interaction_indexes 升序排列，分别对应 1/8、1/16、1/32 三个尺度。
            self.interaction_indexes = sorted(int(i) for i in interaction_indexes)
            # 目标尺度相对于 backbone 原生 1/patch_size 分辨率的缩放因子。
            self.level_scales = (2.0, 1.0, 0.5)

            self.proj_dim = (
                [proj_dim] * num_levels if proj_dim is not None else [embed_dim]
            )

            self.projector = nn.ModuleList(
                [
                    ConvNormLayer_fuse(embed_dim, dim, kernel_size=1, stride=1)
                    for dim in self.proj_dim
                ]
            )

    @staticmethod
    def _build_vit(**kwargs):
        """Hook for backbone variants: subclasses return a different VisionTransformer."""
        return VisionTransformer(**kwargs)

    def _load_weights(self, weights_path):

        path = Path(weights_path) if weights_path else None
        if path and path.exists():
            state = torch.load(path, weights_only=True, map_location="cpu")
            self.backbone._model.load_state_dict(state, strict=True)
            print(
                "\033[34m" + "=" * 80 + "\n",
                "✅ Pretrained ViT weights loaded successfully!\n"
                f"📦 Weight file: {path}\n",
                "=" * 80 + "\033[0m\n",
                sep="",
            )
        else:
            print(
                "\033[34m" + "=" * 80 + "\n",
                "❌❌❌ WARNING: Pretrained ViT weights NOT loaded! ❌❌❌ \n"
                "The model is running with RANDOMLY INITIALIZED parameters.\n"
                "This will severely degrade performance and convergence!\n"
                "Please check the weights_path.\n" * 2,
                "=" * 80 + "\033[0m\n",
                sep="",
            )

    def convert_to_deploy(self):
        # Stamp the backbone with eval-shape constants so ONNX export emits a static graph:
        #   1. resize pos_embed to the eval grid; interpolate_pos_embed will then early-return
        #   2. precompute the per-layer bid_scan permutation buffers and assert that
        #      ``interaction_indexes`` fall on period boundaries (so argsort+gather drops out)
        if self.eval_spatial_size is None:
            return
        eval_h, eval_w = self.eval_spatial_size
        eval_h_grid = int(eval_h) // self.patch_size
        eval_w_grid = int(eval_w) // self.patch_size
        backbone = self.backbone

        if backbone.use_pos_embed:
            with torch.no_grad():
                new_pos = backbone.interpolate_pos_embed(eval_h_grid, eval_w_grid)
            param_dtype = backbone._model.pos_embed.dtype
            param_device = backbone._model.pos_embed.device
            backbone._model.pos_embed = nn.Parameter(
                new_pos.detach().to(dtype=param_dtype, device=param_device).clone(),
                requires_grad=False,
            )
            backbone.grid_size = eval_h_grid

        if backbone.bid_scan:
            assert eval_h_grid == eval_w_grid, (
                "bid_scan static perm requires square eval grid; "
                f"got {eval_h_grid}x{eval_w_grid}"
            )
            period = 4 if backbone.if_quad_dir else 2
            for i in self.interaction_indexes:
                assert (i + 1) % period == 0, (
                    f"interaction_indexes must fall on bid_scan period boundaries; "
                    f"got i={i}, period={period}"
                )
            L = eval_h_grid * eval_w_grid
            # Spatial SwiGLU backbones have no pos_embed; any parameter gives the right device.
            buf_device = next(backbone.parameters()).device
            identity = torch.arange(L, dtype=torch.long, device=buf_device)
            full_perms = []
            for i in range(backbone.depth):
                p = identity.flip(0)
                if i % 2 == 1 and backbone.if_quad_dir:
                    p = p.view(eval_h_grid, eval_w_grid).t().reshape(-1).contiguous()
                backbone.register_buffer(f"_bid_step_perm_{i}", p, persistent=False)
                # Full-sequence index that maps over [CLS, patches] in one shot.
                full_p = torch.cat(
                    [torch.zeros(1, dtype=torch.long, device=buf_device), p + 1]
                )
                full_perms.append(full_p.contiguous())
            # Stacked tensor: integer indexing in the forward loop is Dynamo-friendly
            # (no per-layer recompiles from f-string getattr).
            backbone.register_buffer(
                "_bid_step_perm_full",
                torch.stack(full_perms, dim=0).contiguous(),
                persistent=False,
            )
            backbone._deploy_static_perm = True

    def forward(self, x):

        H_c, W_c = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs = x.shape[0]

        layer_outs = self.backbone(x, return_layer_indexes=self.interaction_indexes)
        # layer_outs 按 interaction_indexes 升序返回。

        if self.multi_layer_same_res:
            # 仅做 [B, N, C] -> [B, C, H, W] 的 reshape，保持原生分辨率交给下游处理。
            return [
                pt.transpose(1, 2).contiguous().view(bs, -1, H_c, W_c)
                for (pt, _) in layer_outs
            ]

        proj_feats = []
        for (patch_tokens, _), scale in zip(layer_outs, self.level_scales):
            feat = patch_tokens.transpose(1, 2).contiguous().view(bs, -1, H_c, W_c)
            if scale != 1.0:
                tH, tW = round(H_c * scale), round(W_c * scale)
                feat = F.interpolate(
                    feat, size=[tH, tW], mode="bilinear", align_corners=False
                )
            proj_feats.append(feat)

        if len(self.proj_dim) == 1:
            proj_feats[-1] = self.projector[-1](proj_feats[-1])
        else:
            proj_feats = [
                layer(feat) for layer, feat in zip(self.projector, proj_feats)
            ]

        return proj_feats

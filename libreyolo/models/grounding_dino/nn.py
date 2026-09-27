"""Native Grounding DINO open-vocabulary detector.

Clean-room port of the Apache-2.0 transformers Grounding DINO, composing native
components: Swin-T backbone (transformers-convention variant), native BERT text
encoder, a 6-layer cross-modal fusion encoder (text enhancer + bidirectional
vision<->text attention + deformable self-attn), a 6-layer deformable decoder
with text cross-attention, and a contrastive-embedding class head.

Modules read the same config object as transformers to maximize fidelity. Eager
attention throughout for deterministic parity.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...kernels.attention.ms_deform_attn import (
    maybe_ms_deform_attn,
    ms_deform_attn_available,
    spatial_shapes_tensor,
)
from ..bert.nn import BertModel
from ..swin.nn import SwinBackbone, SwinDims
from ...kernels.attention.sdpa import manual_attention_required

SPECIAL_TOKENS = [101, 102, 1012, 1029]  # BERT [CLS], [SEP], '.', '?'


def _act(name):
    return {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}[name]


def encode_sine_pos(pos_tensor, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    coords = pos_tensor.unbind(-1)
    embeddings = [c[..., None] * scale / dim_t for c in coords]
    embeddings = [torch.stack((e[..., 0::2].sin(), e[..., 1::2].cos()), dim=-1).flatten(-2) for e in embeddings]
    if len(embeddings) >= 2:
        embeddings[0], embeddings[1] = embeddings[1], embeddings[0]
    return torch.cat(embeddings, dim=-1).to(pos_tensor.dtype)


def generate_masks_with_special_tokens_and_transfer_map(input_ids):
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    special_mask = torch.isin(input_ids, torch.tensor(SPECIAL_TOKENS, device=device))
    indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    prev_special = torch.cummax(torch.where(special_mask, indices, torch.tensor(-1, device=device)), dim=1)[0]
    next_special = torch.where(special_mask, indices, torch.tensor(seq_len, device=device))
    next_special = torch.flip(torch.cummin(torch.flip(next_special, dims=[1]), dim=1)[0], dims=[1])
    valid_block = (next_special != 0) & (next_special != seq_len - 1) & (next_special != seq_len)
    attention_mask = (next_special.unsqueeze(2) == next_special.unsqueeze(1)) & valid_block.unsqueeze(1)
    identity = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0).expand(batch_size, -1, -1)
    attention_mask = identity | attention_mask
    position_ids = indices - prev_special - 1
    position_ids = torch.where(valid_block, position_ids, torch.zeros_like(position_ids))
    position_ids = torch.clamp(position_ids, min=0).to(torch.long)
    return attention_mask, position_ids


def _msda(value, shapes_list, sampling_locations, attention_weights):
    """Portable multi-scale deformable attention (classic Deformable-DETR layout).

    The per-level loop below is the default and the export path; when the
    optional accelerated ``ms_deform_attn`` slot resolves it takes over
    (see ``libreyolo/kernels/attention/ms_deform_attn.py``).
    """
    if ms_deform_attn_available(value):
        accelerated = maybe_ms_deform_attn(
            value,
            spatial_shapes_tensor(shapes_list, value.device),
            sampling_locations,
            attention_weights,
        )
        if accelerated is not None:
            return accelerated
    b, _, nh, hd = value.shape
    _, nq, _, nl, npnt, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in shapes_list], dim=1)
    grids = 2 * sampling_locations - 1
    out_list = []
    for lid, (h, w) in enumerate(shapes_list):
        v = value_list[lid].flatten(2).transpose(1, 2).reshape(b * nh, hd, h, w)
        g = grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        out_list.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    aw = attention_weights.transpose(1, 2).reshape(b * nh, 1, nq, nl * npnt)
    out = (torch.stack(out_list, dim=-2).flatten(-2) * aw).sum(-1).view(b, nh * hd, nq)
    return out.transpose(1, 2).contiguous()


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class GDMultiheadAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.num_heads = heads
        self.head_size = hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self, q, k, v, attention_mask=None):
        b = q.shape[0]
        def shape(t, lin):
            return lin(t).view(b, -1, self.num_heads, self.head_size).transpose(1, 2)
        ql, kl, vl = shape(q, self.query), shape(k, self.key), shape(v, self.value)
        if manual_attention_required():
            # Graph capture (ONNX, and the jit.trace-based TorchScript /
            # CoreML / NCNN exporters) must see the primitive-op equation;
            # eager inference keeps the fused kernels below.
            scores = (ql @ kl.transpose(-1, -2)) / math.sqrt(self.head_size)
            if attention_mask is not None:
                scores = scores + attention_mask
            ctx = scores.softmax(-1) @ vl
        else:
            mask = attention_mask
            if mask is not None and mask.dtype == torch.bool:
                # The eager path ADDS the mask (True -> +1, False -> +0); SDPA
                # reads a bool mask as "allowed", so cast to stay additive.
                mask = mask.to(ql.dtype)
            ctx = F.scaled_dot_product_attention(
                ql,
                kl,
                vl,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0 / math.sqrt(self.head_size),
            )
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(b, -1, self.num_heads * self.head_size)
        return self.out_proj(ctx)


class GDMSDeformAttn(nn.Module):
    def __init__(self, config, num_heads, n_points):
        super().__init__()
        self.d_model = config.d_model
        self.n_levels = config.num_feature_levels
        self.n_heads = num_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(config.d_model, num_heads * self.n_levels * n_points * 2)
        self.attention_weights = nn.Linear(config.d_model, num_heads * self.n_levels * n_points)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, hidden_states, encoder_hidden_states, attention_mask, reference_points,
                spatial_shapes, spatial_shapes_list, position_embeddings=None):
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings
        b, nq, _ = hidden_states.shape
        _, seq, _ = encoder_hidden_states.shape
        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(~attention_mask[..., None], 0.0)
        value = value.view(b, seq, self.n_heads, self.d_model // self.n_heads)
        off = self.sampling_offsets(hidden_states).view(b, nq, self.n_heads, self.n_levels, self.n_points, 2)
        aw = self.attention_weights(hidden_states).view(b, nq, self.n_heads, self.n_levels * self.n_points)
        aw = F.softmax(aw, -1).view(b, nq, self.n_heads, self.n_levels, self.n_points)
        nc = reference_points.shape[-1]
        if nc == 2:
            normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            loc = reference_points[:, :, None, :, None, :] + off / normalizer[None, None, None, :, None, :]
        else:
            loc = reference_points[:, :, None, :, None, :2] + off / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        out = _msda(value, spatial_shapes_list, loc, aw)
        return self.output_proj(out)


class TextEnhancerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.encoder_attention_heads // 2
        self.self_attn = GDMultiheadAttention(config.d_model, self.num_heads)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim // 2)
        self.fc2 = nn.Linear(config.encoder_ffn_dim // 2, config.d_model)
        self.layer_norm_before = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layer_norm_after = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self._act = _act(config.activation_function)

    def forward(self, hidden_states, attention_masks, position_embeddings):
        if attention_masks.dim() == 3:
            am = attention_masks[:, None, :, :].repeat(1, self.num_heads, 1, 1).to(hidden_states.dtype)
            am = (1.0 - am) * torch.finfo(hidden_states.dtype).min
        else:
            am = attention_masks
        q = k = hidden_states if position_embeddings is None else hidden_states + position_embeddings
        attn = self.self_attn(q, k, hidden_states, attention_mask=am)
        hidden_states = self.layer_norm_before(hidden_states + attn)
        residual = hidden_states
        hidden_states = self.fc2(self._act(self.fc1(hidden_states)))
        return self.layer_norm_after(hidden_states + residual)


class BiMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.encoder_ffn_dim // 2
        self.num_heads = config.encoder_attention_heads // 2
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** (-0.5)
        d = config.d_model
        self.vision_proj = nn.Linear(d, self.embed_dim)
        self.text_proj = nn.Linear(d, self.embed_dim)
        self.values_vision_proj = nn.Linear(d, self.embed_dim)
        self.values_text_proj = nn.Linear(d, self.embed_dim)
        self.out_vision_proj = nn.Linear(self.embed_dim, d)
        self.out_text_proj = nn.Linear(self.embed_dim, d)

    def _reshape(self, t, b):
        return t.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, vision, text, vision_mask=None, text_mask=None):
        b, tgt_len, _ = vision.size()
        vq = self._reshape(self.vision_proj(vision) * self.scale, b).view(b * self.num_heads, -1, self.head_dim)
        tk = self._reshape(self.text_proj(text), b).view(b * self.num_heads, -1, self.head_dim)
        vv = self._reshape(self.values_vision_proj(vision), b).view(b * self.num_heads, -1, self.head_dim)
        tv = self._reshape(self.values_text_proj(text), b).view(b * self.num_heads, -1, self.head_dim)
        attn = torch.bmm(vq, tk.transpose(1, 2))
        attn = attn - attn.max()
        attn = torch.clamp(attn, min=-50000, max=50000)
        attn_t = attn.transpose(1, 2)
        text_attn = attn_t - torch.max(attn_t, dim=-1, keepdim=True)[0]
        text_attn = torch.clamp(text_attn, min=-50000, max=50000)
        if vision_mask is not None:
            vm = vision_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            text_attn.masked_fill_(vm, float("-inf"))
        text_attn = text_attn.softmax(-1)
        if text_mask is not None:
            tm = text_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn.masked_fill_(tm, float("-inf"))
        vision_attn = attn.softmax(-1)
        vout = torch.bmm(vision_attn, tv).view(b, self.num_heads, tgt_len, self.head_dim).transpose(1, 2).reshape(b, tgt_len, self.embed_dim)
        tout = torch.bmm(text_attn, vv).view(b, self.num_heads, -1, self.head_dim).transpose(1, 2).reshape(b, text.shape[1], self.embed_dim)
        return self.out_vision_proj(vout), self.out_text_proj(tout)


class FusionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_vision = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layer_norm_text = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.attn = BiMultiHeadAttention(config)
        self.vision_param = nn.Parameter(1e-4 * torch.ones(config.d_model))
        self.text_param = nn.Parameter(1e-4 * torch.ones(config.d_model))

    def forward(self, vision, text, vision_mask=None, text_mask=None):
        vision = self.layer_norm_vision(vision)
        text = self.layer_norm_text(text)
        delta_v, delta_t = self.attn(vision, text, vision_mask=vision_mask, text_mask=text_mask)
        return vision + self.vision_param * delta_v, text + self.text_param * delta_t


class DeformableLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = GDMSDeformAttn(config, config.encoder_attention_heads, config.encoder_n_points)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self._act = _act(config.activation_function)

    def forward(self, hidden_states, attention_mask, position_embeddings, reference_points,
                spatial_shapes, spatial_shapes_list):
        residual = hidden_states
        h = self.self_attn(hidden_states, hidden_states, attention_mask, reference_points,
                           spatial_shapes, spatial_shapes_list, position_embeddings=position_embeddings)
        hidden_states = self.self_attn_layer_norm(residual + h)
        residual = hidden_states
        h = self.fc2(self._act(self.fc1(hidden_states)))
        return self.final_layer_norm(residual + h)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.text_enhancer_layer = TextEnhancerLayer(config)
        self.fusion_layer = FusionLayer(config)
        self.deformable_layer = DeformableLayer(config)

    def forward(self, vision, text, vision_pos, spatial_shapes, spatial_shapes_list, level_start_index,
                key_padding_mask, reference_points, text_attention_mask, text_self_attention_masks, text_position_ids):
        # text position embedding. NOTE: transformers passes LONG position_ids, and
        # encode_sinusoidal_position_embedding casts the result back to the input dtype
        # -> the sin/cos values are TRUNCATED to integers. Replicate by keeping long.
        if text_position_ids is not None:
            tpe = encode_sine_pos(text_position_ids[..., None], num_pos_feats=self.d_model)
        else:
            seq = text.shape[1]
            arange = torch.arange(seq, device=text.device).float()[None, :, None].repeat(text.shape[0], 1, 1)
            tpe = encode_sine_pos(arange, num_pos_feats=self.d_model)
        vision, text = self.fusion_layer(vision, text, vision_mask=key_padding_mask, text_mask=text_attention_mask)
        text = self.text_enhancer_layer(text, attention_masks=~text_self_attention_masks, position_embeddings=tpe)
        vision = self.deformable_layer(vision, ~key_padding_mask, vision_pos, reference_points,
                                       spatial_shapes, spatial_shapes_list)
        return vision, text


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.decoder_attention_heads
        self.self_attn = GDMultiheadAttention(config.d_model, h)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.encoder_attn_text = GDMultiheadAttention(config.d_model, h)
        self.encoder_attn_text_layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.encoder_attn = GDMSDeformAttn(config, h, config.decoder_n_points)
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self._act = _act(config.activation_function)

    def forward(self, hidden_states, position_embeddings, reference_points, spatial_shapes, spatial_shapes_list,
                vision_hidden, vision_mask, text_hidden, text_mask):
        residual = hidden_states
        q = k = hidden_states + position_embeddings
        h = self.self_attn(q, k, hidden_states)
        hidden_states = self.self_attn_layer_norm(residual + h)
        residual = hidden_states
        q = hidden_states + position_embeddings
        h = self.encoder_attn_text(q, text_hidden, text_hidden, attention_mask=text_mask)
        hidden_states = self.encoder_attn_text_layer_norm(residual + h)
        residual = hidden_states
        h = self.encoder_attn(hidden_states, vision_hidden, vision_mask, reference_points,
                              spatial_shapes, spatial_shapes_list, position_embeddings=position_embeddings)
        hidden_states = self.encoder_attn_layer_norm(residual + h)
        residual = hidden_states
        h = self.fc2(self._act(self.fc1(hidden_states)))
        return self.final_layer_norm(residual + h)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])

    @staticmethod
    def get_reference_points(spatial_shapes_list, valid_ratios, device):
        ref_list = []
        for level, (h, w) in enumerate(spatial_shapes_list):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device), indexing="ij")
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * w)
            ref_list.append(torch.stack((ref_x, ref_y), -1))
        ref = torch.cat(ref_list, 1)
        return ref[:, :, None] * valid_ratios[:, None]

    def forward(self, vision, vision_mask, vision_pos, spatial_shapes, spatial_shapes_list, level_start_index,
                valid_ratios, text, text_attention_mask, text_self_attention_masks, text_position_ids):
        reference_points = self.get_reference_points(spatial_shapes_list, valid_ratios, vision.device)
        for layer in self.layers:
            vision, text = layer(vision, text, vision_pos, spatial_shapes, spatial_shapes_list, level_start_index,
                                 vision_mask, reference_points, text_attention_mask, text_self_attention_masks,
                                 text_position_ids)
        return vision, text


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.decoder_layers)])
        self.reference_points_head = MLPHead(config.query_dim // 2 * config.d_model, config.d_model, config.d_model, 2)
        self.bbox_embed = None

    def forward(self, target, reference_points, vision_hidden, vision_mask, text_hidden, text_mask,
                spatial_shapes, spatial_shapes_list, level_start_index, valid_ratios):
        hidden_states = target
        dtype = text_hidden.dtype
        tmask = text_mask[:, None, None, :].repeat(1, self.config.decoder_attention_heads, self.config.num_queries, 1)
        tmask = tmask.to(dtype) * torch.finfo(dtype).min
        intermediate, intermediate_ref = [], []
        for idx, layer in enumerate(self.layers):
            nc = reference_points.shape[-1]
            if nc == 4:
                ref_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                ref_input = reference_points[:, :, None] * valid_ratios[:, None]
            query_pos = encode_sine_pos(ref_input[:, :, 0, :], num_pos_feats=self.config.d_model // 2)
            query_pos = self.reference_points_head(query_pos)
            hidden_states = layer(hidden_states, query_pos, ref_input, spatial_shapes, spatial_shapes_list,
                                  vision_hidden, vision_mask, text_hidden, tmask)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                new_ref = (tmp + torch.special.logit(reference_points, eps=1e-5)).sigmoid()
                reference_points = new_ref.detach()
            intermediate.append(self.layer_norm(hidden_states))
            intermediate_ref.append(reference_points)
        return torch.stack(intermediate, dim=1), torch.stack(intermediate_ref, dim=1)


class SinePositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.d_model // 2
        self.temperature = config.positional_embedding_temperature
        self.scale = 2 * math.pi

    def forward(self, mask):
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class SwinWrap(nn.Module):
    """GD backbone: transformers-order Swin + per-stage output LayerNorms."""

    def __init__(self, config):
        super().__init__()
        bc = config.backbone_config
        self._backbone = SwinBackbone(SwinDims(
            embed_dim=bc.embed_dim, depths=tuple(bc.depths), num_heads=tuple(bc.num_heads),
            window_size=bc.window_size, out_indices=(1, 2, 3), tf_order=True))
        self.hidden_states_norms = nn.ModuleList(
            [nn.LayerNorm(c) for c in [bc.embed_dim * 2, bc.embed_dim * 4, bc.embed_dim * 8]])

    def forward(self, pixel_values):
        feats = self._backbone(pixel_values)  # NHWC
        return [ln(f).permute(0, 3, 1, 2).contiguous() for ln, f in zip(self.hidden_states_norms, feats)]


def _valid_ratio(mask):
    _, h, w = mask.shape
    vh = torch.sum(mask[:, :, 0], 1).float() / h
    vw = torch.sum(mask[:, 0, :], 1).float() / w
    return torch.stack([vw, vh], -1)


class GroundingDinoDetectionModel(nn.Module):
    """Native equivalent of transformers ``GroundingDinoForObjectDetection``."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        # --- model.* submodules (mirror transformers tree) ---
        self.backbone_conv = SwinWrap(config)             # -> model.backbone.conv_encoder.model
        self.position_embedding = SinePositionEmbedding(config)
        chans = [config.backbone_config.embed_dim * m for m in (2, 4, 8)]
        proj = []
        for c in chans:
            proj.append(nn.Sequential(nn.Conv2d(c, config.d_model, 1), nn.GroupNorm(32, config.d_model)))
        proj.append(nn.Sequential(nn.Conv2d(chans[-1], config.d_model, 3, 2, 1), nn.GroupNorm(32, config.d_model)))
        self.input_proj_vision = nn.ModuleList(proj)
        self.text_backbone = BertModel(config.text_config)
        self.text_projection = nn.Linear(config.text_config.hidden_size, config.d_model)
        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.level_embed = nn.Parameter(torch.zeros(config.num_feature_levels, config.d_model))
        self.enc_output = nn.Linear(config.d_model, config.d_model)
        self.enc_output_norm = nn.LayerNorm(config.d_model, config.layer_norm_eps)
        self.encoder_output_bbox_embed = MLPHead(config.d_model, config.d_model, 4, 3)
        # --- detection heads (ForObjectDetection level) ---
        self.bbox_embed = nn.ModuleList([MLPHead(config.d_model, config.d_model, 4, 3)
                                         for _ in range(config.decoder_layers)])
        self.decoder.bbox_embed = self.bbox_embed
        self.max_text_len = config.max_text_len

    def _contrastive(self, vision_hidden, text_hidden, text_token_mask):
        out = vision_hidden @ text_hidden.transpose(-1, -2)
        out = out.masked_fill(~text_token_mask[:, None, :], float("-inf"))
        new = torch.full((*out.shape[:-1], self.max_text_len), float("-inf"), device=out.device)
        new[..., : out.shape[-1]] = out
        return new

    def _proposals(self, enc_output, padding_mask, spatial_shapes_list):
        b = enc_output.shape[0]
        proposals = []
        pos = 0
        for level, (h, w) in enumerate(spatial_shapes_list):
            m = padding_mask[:, pos:pos + h * w].view(b, h, w, 1)
            vh = torch.sum(~m[:, :, 0, 0], 1)
            vw = torch.sum(~m[:, 0, :, 0], 1)
            gy, gx = torch.meshgrid(torch.linspace(0, h - 1, h, dtype=torch.float32, device=enc_output.device),
                                    torch.linspace(0, w - 1, w, dtype=torch.float32, device=enc_output.device), indexing="ij")
            grid = torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], -1)
            scale = torch.cat([vw.unsqueeze(-1), vh.unsqueeze(-1)], 1).view(b, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(b, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** level)
            proposals.append(torch.cat((grid, wh), -1).view(b, -1, 4))
            pos += h * w
        output_proposals = torch.cat(proposals, 1)
        valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~valid, float("inf"))
        object_query = enc_output.masked_fill(padding_mask.unsqueeze(-1), 0.0).masked_fill(~valid, 0.0)
        object_query = self.enc_output_norm(self.enc_output(object_query))
        return object_query, output_proposals

    def forward(self, pixel_values, input_ids, token_type_ids=None, attention_mask=None, pixel_mask=None):
        b, _, H, W = pixel_values.shape
        device = pixel_values.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if pixel_mask is None:
            pixel_mask = torch.ones((b, H, W), dtype=torch.long, device=device)

        text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(input_ids)
        text_token_mask = attention_mask.bool()
        mtl = self.config.max_text_len
        if text_self_attention_masks.shape[1] > mtl:
            text_self_attention_masks = text_self_attention_masks[:, :mtl, :mtl]
            position_ids = position_ids[:, :mtl]
            input_ids = input_ids[:, :mtl]
            token_type_ids = token_type_ids[:, :mtl]
            text_token_mask = text_token_mask[:, :mtl]

        # text features
        text_features = self.text_backbone(input_ids, text_self_attention_masks[:, None, :, :],
                                           token_type_ids, position_ids)
        text_features = self.text_projection(text_features)

        # vision features + projections
        feature_maps = self.backbone_conv(pixel_values)
        masks = [F.interpolate(pixel_mask[None].float(), size=f.shape[-2:]).to(torch.bool)[0] for f in feature_maps]
        sources = [self.input_proj_vision[i](f) for i, f in enumerate(feature_maps)]
        pos_embeds = [self.position_embedding(m).to(f.dtype) for m, f in zip(masks, sources)]
        # extra 4th level
        src3 = self.input_proj_vision[3](feature_maps[-1])
        mask3 = F.interpolate(pixel_mask[None].float(), size=src3.shape[-2:]).to(torch.bool)[0]
        pos3 = self.position_embedding(mask3).to(src3.dtype)
        sources.append(src3); masks.append(mask3); pos_embeds.append(pos3)

        # flatten
        source_flatten, mask_flatten, lvl_pos, shapes_list = [], [], [], []
        for level, (s, m, p) in enumerate(zip(sources, masks, pos_embeds)):
            _, _, h, w = s.shape
            shapes_list.append((h, w))
            source_flatten.append(s.flatten(2).transpose(1, 2))
            mask_flatten.append(m.flatten(1))
            lvl_pos.append(p.flatten(2).transpose(1, 2) + self.level_embed[level].view(1, 1, -1))
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos = torch.cat(lvl_pos, 1)
        spatial_shapes = torch.as_tensor(shapes_list, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([_valid_ratio(m) for m in masks], 1).float()

        vision_enc, text_enc = self.encoder(
            source_flatten, ~mask_flatten, lvl_pos, spatial_shapes, shapes_list, level_start_index, valid_ratios,
            text_features, ~text_token_mask, ~text_self_attention_masks, position_ids)

        # two-stage query selection
        object_query, output_proposals = self._proposals(vision_enc, ~mask_flatten, shapes_list)
        enc_class = self._contrastive(object_query, text_enc, text_token_mask)
        enc_coord = self.encoder_output_bbox_embed(object_query) + output_proposals
        topk = self.config.num_queries
        topk_idx = torch.topk(enc_class.max(-1)[0], topk, dim=1)[1]
        topk_coords = torch.gather(enc_coord, 1, topk_idx.unsqueeze(-1).repeat(1, 1, 4)).detach()
        reference_points = topk_coords.sigmoid()
        init_reference = reference_points
        target = self.query_position_embeddings.weight.unsqueeze(0).repeat(b, 1, 1)

        inter_hidden, inter_ref = self.decoder(
            target, reference_points, vision_enc, mask_flatten, text_enc, ~text_token_mask,
            spatial_shapes, shapes_list, level_start_index, valid_ratios)

        # detection heads over decoder stages
        logits_list, boxes_list = [], []
        for level in range(inter_hidden.shape[1]):
            reference = init_reference if level == 0 else inter_ref[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-5)
            cls = self._contrastive(inter_hidden[:, level], text_enc, text_token_mask)
            delta = self.bbox_embed[level](inter_hidden[:, level])
            coord = (delta + reference).sigmoid()
            logits_list.append(cls)
            boxes_list.append(coord)
        return {"logits": logits_list[-1], "pred_boxes": boxes_list[-1]}


__all__ = ["GroundingDinoDetectionModel"]

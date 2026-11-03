"""DeepEncoder V2 tail: a Qwen2 decoder used as the DeepSeek-OCR-2 vision encoder.

Ported from ``deepencoderv2.py`` of the DeepSeek-OCR-2 Hugging Face release
(deepseek-ai/DeepSeek-OCR-2 @ aaa02f3811945a91062062994c5c4a3f4c0af2b0,
Apache-2.0). The release instantiates the transformers 4.46 ``Qwen2Model``
(Apache-2.0) with SDPA attention and overrides its mask; the decoder layer,
attention and rotary embedding used on that path are ported from
transformers 4.46.3 ``models/qwen2/modeling_qwen2.py``. See this package's
``NOTICE``.

SAM features (``n`` tokens) are followed by ``n`` learned queries. The SAM
tokens attend to each other bidirectionally; each query attends to every SAM
token and causally to the queries before it. The query outputs are the visual
tokens. Module names match the checkpoint
(``qwen2_model.model.model.layers.*``), so the weights load strictly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .language import DeepseekV2RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2SdpaAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, cos_sin, attention_mask):
        bsz, q_len, _ = hidden_states.size()
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
            1, 2
        )
        cos, sin = cos_sin
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        if query.device.type == "cuda":
            query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=causal_mask, dropout_p=0.0, is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)


class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size, eps):
        super().__init__()
        self.self_attn = Qwen2SdpaAttention(hidden_size, num_heads, num_kv_heads)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size)
        self.input_layernorm = DeepseekV2RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(hidden_size, eps=eps)

    def forward(self, hidden_states, cos_sin, attention_mask):
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), cos_sin, attention_mask
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class _Qwen2Stack(nn.Module):
    """``Qwen2Model`` without token embeddings."""

    def __init__(
        self,
        num_layers=24,
        hidden_size=896,
        num_heads=14,
        num_kv_heads=2,
        intermediate_size=4864,
        rope_theta=1000000.0,
        eps=1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            Qwen2DecoderLayer(hidden_size, num_heads, num_kv_heads, intermediate_size, eps)
            for _ in range(num_layers)
        )
        self.norm = DeepseekV2RMSNorm(hidden_size, eps=eps)
        self.rotary_emb = RotaryEmbedding(hidden_size // num_heads, rope_theta)

    def forward(self, inputs_embeds: torch.Tensor, n_bidirectional: int) -> torch.Tensor:
        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        mask = flow_mask(seq_len, n_bidirectional, inputs_embeds.dtype, inputs_embeds.device)
        mask = mask.expand(inputs_embeds.shape[0], 1, seq_len, seq_len)
        hidden_states = inputs_embeds
        cos_sin = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin, mask)
        return self.norm(hidden_states)


def flow_mask(seq_len: int, n_image: int, dtype, device) -> torch.Tensor:
    """Additive ``(1, 1, L, L)`` mask: image tokens bidirectional, queries causal.

    Rows and columns ``[0, n_image)`` are image tokens; the rest are queries,
    which see every image token and the queries up to themselves.
    """
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
    mask[:n_image, :n_image] = 0.0
    queries = seq_len - n_image
    if queries:
        mask[n_image:, :n_image] = 0.0
        tri = torch.ones(queries, queries, dtype=torch.bool, device=device).tril()
        block = mask[n_image:, n_image:]
        block.masked_fill_(tri, 0.0)
    return mask[None, None]


class _CustomQwen2Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Qwen2Stack()


class Qwen2Decoder2Encoder(nn.Module):
    """SAM tokens plus learned queries through the Qwen2 stack."""

    def __init__(self, hidden_size: int = 896):
        super().__init__()
        self.model = _CustomQwen2Decoder()
        self.query_768 = nn.Embedding(144, hidden_size)
        self.query_1024 = nn.Embedding(256, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        bs, n_query, _ = x.shape
        if n_query == 144:
            param_img = self.query_768.weight
        elif n_query == 256:
            param_img = self.query_1024.weight
        else:
            raise ValueError(
                f"DeepSeek-OCR-2 has learned queries for 768 and 1024 px views only; "
                f"got a {n_query}-token view."
            )
        queries = param_img.unsqueeze(0).expand(bs, -1, -1)
        y = self.model.model(torch.cat([x, queries], dim=1), n_query)
        return y[:, n_query:, :]

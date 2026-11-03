"""DeepSeek-OCR language decoder: a DeepSeek-V2 style MoE with plain attention.

Ported from ``modeling_deepseekv2.py`` of the DeepSeek-OCR Hugging Face
release (deepseek-ai/DeepSeek-OCR @ 9f30c71f441d010e5429c532364a86705536c53a,
Apache-2.0 header, Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team).
The checkpoint disables multi-head latent attention, which makes the release
fall back to the Llama attention and rotary embedding of transformers 4.46
(Apache-2.0); those two pieces are ported from there. See this package's
``NOTICE``.

Only the inference path the checkpoint uses is kept: softmax gating with greedy
top-k routing, no expert parallelism, no auxiliary loss, eager attention with a
plain key/value cache. The arithmetic is unchanged, which is what token-level
parity with the release relies on.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeepSeekOCRTextConfig:
    vocab_size: int = 129280
    hidden_size: int = 1280
    intermediate_size: int = 6848
    moe_intermediate_size: int = 896
    num_hidden_layers: int = 12
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = False
    scoring_func: str = "softmax"
    topk_method: str = "greedy"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    use_mla: bool = False
    bos_token_id: int = 0
    eos_token_id: int = 1

    @classmethod
    def from_json(cls, path: str | Path) -> "DeepSeekOCRTextConfig":
        """Read the text settings from a snapshot ``config.json``.

        The release stores them twice (top level and ``language_config``);
        the top level wins, as it does for the release's own config class.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        merged = {**raw.get("language_config", {}), **raw}
        known = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in merged.items() if k in known and v is not None})
        if cfg.use_mla:
            raise ValueError(
                "This DeepSeek-OCR port implements the plain-attention decoder only; "
                "the snapshot enables multi-head latent attention (use_mla=true)."
            )
        if cfg.scoring_func != "softmax" or cfg.topk_method != "greedy":
            raise ValueError(
                "Unsupported MoE routing in snapshot config: "
                f"scoring_func={cfg.scoring_func!r}, topk_method={cfg.topk_method!r}."
            )
        return cfg


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Default (unscaled) rotary embedding, as in the transformers 4.46 Llama port."""

    def __init__(self, head_dim: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
        )
        # A floating buffer on purpose: casting the module to bf16 rounds it,
        # exactly as casting the release model does.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
                1, 2
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Attention(nn.Module):
    """Eager multi-head attention (no MLA), with an appendable key/value cache."""

    def __init__(self, config: DeepSeekOCRTextConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

    def forward(self, hidden_states, cos_sin, attention_mask, past):
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
        if past is not None:
            key = torch.cat([past[0], key], dim=-2)
            value = torch.cat([past[1], value], dim=-2)
        present = (key, value)
        if self.num_key_value_groups > 1:
            key = torch.repeat_interleave(key, self.num_key_value_groups, dim=1)
            value = torch.repeat_interleave(value, self.num_key_value_groups, dim=1)
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), present


class DeepseekV2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, config: DeepSeekOCRTextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty((config.n_routed_experts, config.hidden_size)))

    def forward(self, hidden_states: torch.Tensor):
        _, _, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(torch.float32), None
        )
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator * self.routed_scaling_factor
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class DeepseekV2MoE(nn.Module):
    """Routed experts plus always-on shared experts."""

    def __init__(self, config: DeepSeekOCRTextConfig):
        super().__init__()
        self.experts = nn.ModuleList(
            DeepseekV2MLP(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        )
        self.gate = MoEGate(config)
        self.shared_experts = None
        if config.n_shared_experts:
            self.shared_experts = DeepseekV2MLP(
                config.hidden_size, config.moe_intermediate_size * config.n_shared_experts
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0).cpu().tolist()
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            outputs.append(self.experts[i](sorted_tokens[start_idx:end_idx]))
            start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )


class DecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekOCRTextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(config)
        use_moe = (
            config.n_routed_experts
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        self.mlp = (
            DeepseekV2MoE(config)
            if use_moe
            else DeepseekV2MLP(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, cos_sin, attention_mask, past):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(hidden_states, cos_sin, attention_mask, past)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states, present


KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


def causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Additive ``(1, 1, L, L)`` causal mask: 0 on and below the diagonal."""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
    cond = torch.arange(seq_len, device=device)
    mask.masked_fill_(cond < (cond + 1).view(seq_len, 1), 0)
    return mask.to(dtype)[None, None, :, :]


class NoRepeatNGram:
    """Bans any token that would repeat an n-gram already in the sequence.

    Incremental form of the standard ``no_repeat_ngram_size`` logits rule:
    every complete n-gram seen so far (prompt included) is indexed by its first
    ``n - 1`` tokens, and the continuation of the current ``n - 1`` suffix is
    banned. Equivalent to recomputing all n-grams at each step.
    """

    def __init__(self, size: int, tokens: List[int]):
        self.size = int(size)
        self.tokens: List[int] = []
        self.index: dict = {}
        for tok in tokens:
            self.append(tok)

    def append(self, token: int) -> None:
        self.tokens.append(int(token))
        n = self.size
        if n > 0 and len(self.tokens) >= n:
            gram = tuple(self.tokens[-n:])
            self.index.setdefault(gram[:-1], set()).add(gram[-1])

    def banned(self) -> List[int]:
        n = self.size
        if n <= 0 or len(self.tokens) + 1 < n:
            return []
        if n == 1:
            return sorted(self.index.get((), ()))
        return sorted(self.index.get(tuple(self.tokens[-(n - 1) :]), ()))


class DeepSeekV2TextModel(nn.Module):
    """Decoder stack; ``forward`` takes embeddings and returns hidden states."""

    def __init__(self, config: DeepSeekOCRTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = DeepseekV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads, config.rope_theta
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, KVCache]:
        seq_len = inputs_embeds.shape[1]
        past_len = past[0][0].shape[-2] if past else 0
        position_ids = torch.arange(
            past_len, past_len + seq_len, dtype=torch.long, device=inputs_embeds.device
        ).unsqueeze(0)
        mask = None
        if seq_len > 1:
            if past_len:
                raise ValueError("Chunked prefill is not supported.")
            mask = causal_mask(seq_len, inputs_embeds.dtype, inputs_embeds.device)
        hidden_states = inputs_embeds
        cos_sin = None
        presents: KVCache = []
        for i, layer in enumerate(self.layers):
            if cos_sin is None:
                # The release computes this inside every attention layer from the
                # value states; their dtype is the same at every layer.
                cos_sin = self.rotary_emb(
                    self._value_dtype_probe(hidden_states, layer), position_ids
                )
            hidden_states, present = layer(
                hidden_states, cos_sin, mask, past[i] if past else None
            )
            presents.append(present)
        return self.norm(hidden_states), presents

    @staticmethod
    def _value_dtype_probe(hidden_states: torch.Tensor, layer: DecoderLayer) -> torch.Tensor:
        """A tensor carrying the dtype the value projection produces.

        Under autocast a linear layer emits the autocast dtype; otherwise it
        follows its weight. Only the dtype and device are read.
        """
        if torch.is_autocast_enabled(hidden_states.device.type):
            dtype = torch.get_autocast_dtype(hidden_states.device.type)
        else:
            dtype = layer.self_attn.v_proj.weight.dtype
        return torch.empty(0, dtype=dtype, device=hidden_states.device)

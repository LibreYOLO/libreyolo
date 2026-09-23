"""Portable GTR gated linear attention, with optional FLA CUDA acceleration.

Projection layout follows GTR revision 782e737efe2e6437ac537fbdcee089673d3376c1
(MIT). The PyTorch recurrence implements its state update directly. See NOTICE.
"""

import torch
from torch import nn
from torch.nn import functional as F


@torch.jit.script
def recurrent_gla(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """Inclusive key-gated recurrence on tensors shaped [B, T, H, D]."""
    dtype = q.dtype
    q, k, v, g = q.float(), k.float(), v.float(), g.float()
    state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    outputs = torch.jit.annotate(list[torch.Tensor], [])
    scale = q.shape[-1] ** -0.5
    for index in range(q.shape[1]):
        state = state * g[:, index].exp().unsqueeze(-1)
        state = state + k[:, index].unsqueeze(-1) * v[:, index].unsqueeze(-2)
        outputs.append(((q[:, index] * scale).unsqueeze(-1) * state).sum(-2))
    return torch.stack(outputs, dim=1).to(dtype)


class RMSNormGated(nn.Module):
    """FP32 RMS normalization and SiLU output gate, with upstream weight keys."""

    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x, gate):
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float() * F.silu(gate.float())).to(x.dtype)


class GatedLinearAttention(nn.Module):
    """The attention configuration used by all four published GTR detectors."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        expand_k=0.5,
        expand_v=1.0,
        gate_logit_normalizer=16,
        gate_low_rank_dim=16,
        norm_eps=1e-5,
        use_short_conv=False,
        qk_norm=False,
        use_output_gate=True,
        gate_fn="swish",
        **kwargs,
    ):
        super().__init__()
        if use_short_conv or qk_norm or not use_output_gate or gate_fn != "swish":
            raise ValueError(
                "GTR supports the published gated attention configuration only"
            )
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.gate_logit_normalizer = gate_logit_normalizer
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim),
        )
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.g_norm_swish_gate = RMSNormGated(self.value_dim // num_heads, norm_eps)

    def forward(self, x):
        shape = (*x.shape[:-1], self.num_heads, -1)
        q, k, v = (
            projection(x).reshape(shape)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        )
        logits = self.gk_proj(x).reshape(shape)
        if torch.onnx.is_in_onnx_export():
            # The legacy exporter lowers logsigmoid to log(sigmoid(x)),
            # which becomes -inf for the large negative gates in real weights.
            g = logits.clamp(max=0) - torch.log1p(torch.exp(-logits.abs()))
        else:
            g = F.logsigmoid(logits)
        g = g / self.gate_logit_normalizer
        if (
            x.is_cuda
            and not torch.onnx.is_in_onnx_export()
            and not torch.jit.is_tracing()
        ):
            try:
                from fla.ops.gla import chunk_gla
            except ImportError:
                output = recurrent_gla(q, k, v, g)
            else:
                output, _ = chunk_gla(q=q, k=k, v=v, g=g, output_final_state=False)
        else:
            output = recurrent_gla(q, k, v, g)
        output = self.g_norm_swish_gate(output, self.g_proj(x).reshape(shape))
        return self.o_proj(output.flatten(-2)), None, None

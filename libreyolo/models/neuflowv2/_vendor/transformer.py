# Vendored from https://github.com/neufieldrobotics/NeuFlow_v2 (NeuFlow/transformer.py) at
# commit 204b5e3744461d90303b9ff82caa7a1bb56a2ca2, Apache-2.0 (see LICENSE).
# LibreYOLO changes: package-relative imports; see _vendor/__init__.py.

import torch
import torch.nn.functional as F

class TransformerLayer(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 ffn=True,
                 ffn_dim_expansion=1
                 ):
        super(TransformerLayer, self).__init__()

        # multi-head attention
        self.q_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.k_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.v_proj = torch.nn.Linear(feature_dim, feature_dim)

        self.merge = torch.nn.Linear(feature_dim, feature_dim)

        # self.multi_head_attn = torch.nn.MultiheadAttention(feature_dim, 2, batch_first=True, device='cuda')

        self.norm1 = torch.nn.LayerNorm(feature_dim)

        self.ffn = ffn

        if self.ffn:
            in_channels = feature_dim * 2
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(in_channels * ffn_dim_expansion, feature_dim, bias=False),
            )

            self.norm2 = torch.nn.LayerNorm(feature_dim)

    def forward(self, source, target):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        message = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)

        message = self.merge(message)

        # message, _ = self.multi_head_attn(query, key, value, need_weights=False)
        message = self.norm1(message)

        if self.ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message

class FeatureAttention(torch.nn.Module):
    def __init__(self, feature_dim, num_layers, ffn=True, ffn_dim_expansion=1, post_norm=False):
        super(FeatureAttention, self).__init__()

        self.layers = torch.nn.ModuleList([
            TransformerLayer(feature_dim, ffn=ffn, ffn_dim_expansion=ffn_dim_expansion
                             )
            for i in range(num_layers)])

        self.post_norm = post_norm

        if self.post_norm:
            self.norm = torch.nn.BatchNorm2d(feature_dim)

    def forward(self, concat_features0):

        b, c, h, w = concat_features0.shape

        concat_features0 = concat_features0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        concat_features1 = torch.cat(concat_features0.chunk(chunks=2, dim=0)[::-1], dim=0)

        for layer in self.layers:
            concat_features0 = layer(concat_features0, concat_features1)
            concat_features1 = torch.cat(concat_features0.chunk(chunks=2, dim=0)[::-1], dim=0)

        # reshape back
        concat_features0 = concat_features0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        if self.post_norm:
            concat_features0 = self.norm(concat_features0)

        return concat_features0


class FlowAttention(torch.nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, feature_dim):
        super(FlowAttention, self).__init__()

        self.q_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.k_proj = torch.nn.Linear(feature_dim, feature_dim)

    def forward(self, feature, flow):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        b, c, h, w = feature.size()

        feature = feature.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]

        flow = flow.flatten(-2).permute(0, 2, 1)

        query = self.q_proj(feature)  # [B, H*W, C]
        key = self.k_proj(feature)  # [B, H*W, C]

        flow = F.scaled_dot_product_attention(query, key, flow)

        flow = flow.view(b, h, w, 2).permute(0, 3, 1, 2)

        return flow

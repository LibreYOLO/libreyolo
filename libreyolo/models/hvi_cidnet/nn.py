"""Native HVI-CIDNet inference graph.

Adapted from ``Fediory/HVI-CIDNet`` at commit
``eb43d7d91e9a336c66856824ff9e4603ae41f408`` under the MIT License.  Module
and parameter names intentionally match the released checkpoint.  The only
structural substitution is expressing the MIT ``einops.rearrange`` calls with
equivalent native ``reshape`` operations, so LibreYOLO needs no extra runtime
dependency.  See the family ``NOTICE`` for the complete provenance record.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Channel-first layer normalization used by the released network."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class NormDownsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, use_norm: bool = False) -> None:
        super().__init__()
        self.use_norm = use_norm
        if use_norm:
            self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.down(x))
        return self.norm(x) if self.use_norm else x


class NormUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, use_norm: bool = False) -> None:
        super().__init__()
        self.use_norm = use_norm
        if use_norm:
            self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.up = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_scale(x)
        x = self.prelu(self.up(torch.cat((x, skip), dim=1)))
        return self.norm(x) if self.use_norm else x


class CAB(nn.Module):
    """Channel-wise cross-attention block."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias
        )
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            dim * 2,
            dim * 2,
            kernel_size=3,
            padding=1,
            groups=dim * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        channels_per_head = channels // self.num_heads
        q = self.q_dwconv(self.q(x)).reshape(
            batch, self.num_heads, channels_per_head, height * width
        )
        key, value = self.kv_dwconv(self.kv(context)).chunk(2, dim=1)
        key = key.reshape(batch, self.num_heads, channels_per_head, height * width)
        value = value.reshape(batch, self.num_heads, channels_per_head, height * width)
        q = F.normalize(q, dim=-1)
        key = F.normalize(key, dim=-1)
        attention = (q @ key.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        out = (attention @ value).reshape(batch, channels, height, width)
        return self.project_out(out)


class IEL(nn.Module):
    """Intensity-enhancement feed-forward layer."""

    def __init__(self, dim: int, expansion: float = 2.66, bias: bool = False) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
            bias=bias,
        )
        self.dwconv1 = nn.Conv2d(
            hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=bias
        )
        self.dwconv2 = nn.Conv2d(
            hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=bias
        )
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        left = torch.tanh(self.dwconv1(left)) + left
        right = torch.tanh(self.dwconv2(right)) + right
        return self.project_out(left * right)


class HV_LCA(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.gdfn = IEL(dim)
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.ffn(self.norm(x), self.norm(context))
        return self.gdfn(self.norm(x))


class I_LCA(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.ffn(self.norm(x), self.norm(context))
        return x + self.gdfn(self.norm(x))


class RGB_HVI(nn.Module):
    """Learned RGB/HVI transform from HVI-CIDNet."""

    def __init__(self) -> None:
        super().__init__()
        self.density_k = nn.Parameter(torch.full((1,), 0.2))
        self.this_k: float = 0.2

    def HVIT(self, image: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        value = image.max(1).values
        image_min = image.min(1).values
        hue = torch.empty_like(value)

        blue_max = image[:, 2] == value
        green_max = image[:, 1] == value
        red_max = image[:, 0] == value
        delta = value - image_min + eps
        hue[blue_max] = (4.0 + (image[:, 0] - image[:, 1]) / delta)[blue_max]
        hue[green_max] = (2.0 + (image[:, 2] - image[:, 0]) / delta)[green_max]
        hue[red_max] = ((image[:, 1] - image[:, 2]) / delta % 6.0)[red_max]
        hue[image_min == value] = 0.0
        hue = hue / 6.0

        saturation = (value - image_min) / (value + eps)
        saturation[value == 0] = 0
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        self.this_k = self.density_k.item()
        color_sensitive = (torch.sin(value * 0.5 * math.pi) + eps).pow(self.density_k)
        horizontal = color_sensitive * saturation * torch.cos(2.0 * math.pi * hue)
        vertical = color_sensitive * saturation * torch.sin(2.0 * math.pi * hue)
        return torch.cat((horizontal, vertical, value), dim=1)

    def PHVIT(
        self,
        image: torch.Tensor,
        *,
        saturation_scale: float = 1.0,
        intensity_scale: float = 1.0,
    ) -> torch.Tensor:
        eps = 1e-8
        horizontal = image[:, 0].clamp(-1, 1)
        vertical = image[:, 1].clamp(-1, 1)
        intensity = image[:, 2].clamp(0, 1)
        color_sensitive = (torch.sin(intensity * 0.5 * math.pi) + eps).pow(self.this_k)
        horizontal = (horizontal / (color_sensitive + eps)).clamp(-1, 1)
        vertical = (vertical / (color_sensitive + eps)).clamp(-1, 1)
        hue = (torch.atan2(vertical + eps, horizontal + eps) / (2 * math.pi)) % 1
        saturation = torch.sqrt(horizontal.square() + vertical.square() + eps)
        saturation = (saturation * saturation_scale).clamp(0, 1)
        value = intensity.clamp(0, 1)

        sector = torch.floor(hue * 6.0)
        fraction = hue * 6.0 - sector
        p = value * (1.0 - saturation)
        q = value * (1.0 - fraction * saturation)
        t = value * (1.0 - (1.0 - fraction) * saturation)
        red = torch.zeros_like(hue)
        green = torch.zeros_like(hue)
        blue = torch.zeros_like(hue)

        choices = (
            (value, t, p),
            (q, value, p),
            (p, value, t),
            (p, q, value),
            (t, p, value),
            (value, p, q),
        )
        for index, (r_value, g_value, b_value) in enumerate(choices):
            active = sector == index
            red[active] = r_value[active]
            green[active] = g_value[active]
            blue[active] = b_value[active]
        rgb = torch.stack((red, green, blue), dim=1)
        return rgb * intensity_scale


class CIDNet(nn.Module):
    """Color-and-intensity decoupling network."""

    def __init__(
        self,
        channels: tuple[int, int, int, int] = (36, 36, 72, 144),
        heads: tuple[int, int, int, int] = (1, 2, 4, 8),
        norm: bool = False,
    ) -> None:
        super().__init__()
        ch1, ch2, ch3, ch4 = channels
        _, head2, head3, head4 = heads
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, bias=False),
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, bias=False),
        )

        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, bias=False),
        )

        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)
        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)
        self.trans = RGB_HVI()

    def forward(
        self,
        x: torch.Tensor,
        *,
        saturation_scale: float = 1.0,
        intensity_scale: float = 1.0,
    ) -> torch.Tensor:
        dtype = x.dtype
        hvi = self.trans.HVIT(x)
        intensity = hvi[:, 2].unsqueeze(1).to(dtype)

        i_enc0 = self.IE_block0(intensity)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0, hv_jump0 = i_enc0, hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        i_jump1, hv_jump1 = i_enc2, hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        i_jump2, hv_jump2 = i_enc3, hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, i_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, i_jump1)

        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        output_hvi = torch.cat((hv_0, i_dec0), dim=1) + hvi
        return self.trans.PHVIT(
            output_hvi,
            saturation_scale=saturation_scale,
            intensity_scale=intensity_scale,
        )


__all__ = ["CIDNet"]

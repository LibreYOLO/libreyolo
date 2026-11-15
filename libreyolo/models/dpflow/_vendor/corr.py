# =============================================================================
# Copyright 2025 Henrique Morimitsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import math

import torch
import torch.nn.functional as F

# LibreYOLO: the compiled spatial_correlation_sampler extension is not used;
# the pure-PyTorch samplers from ptlflow.utils.correlation are vendored as
# .correlation.
from .correlation import (
    IterSpatialCorrelationSampler as SpatialCorrelationSampler,
)
from .correlation import (
    IterTranslatedSpatialCorrelationSampler as TranslatedSpatialCorrelationSampler,
)

try:
    import alt_cuda_corr
except ImportError:
    # alt_cuda_corr is not compiled
    pass

from .utils import bilinear_sampler


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            if min(corr.shape[2:4]) > 2 * radius + 1:
                corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, f1 = coords.shape

        corr_full = []
        for j in range(0, f1, 2):
            corr_pyramid = []
            for i in range(self.num_levels):
                corr_raw = self.corr_pyramid[i]
                dx = torch.linspace(-r, r, 2 * r + 1, dtype=coords.dtype)
                dy = torch.linspace(-r, r, 2 * r + 1, dtype=coords.dtype)
                delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(
                    coords.device
                )

                centroid_lvl = (
                    coords[..., j : j + 2].reshape(batch * h1 * w1, 1, 1, 2) / 2**i
                )
                delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
                coords_lvl = centroid_lvl + delta_lvl

                corr = bilinear_sampler(corr_raw, coords_lvl)
                corr = corr.view(batch, h1, w1, -1)
                corr_pyramid.append(corr)

            corr_pyramid = torch.cat(corr_pyramid, dim=-1)
            corr_full.append(corr_pyramid)

        corr_full = torch.cat(corr_full, -1)
        return corr_full.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim))


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            if coords.dtype == torch.float16:
                fmap1_i = fmap1_i.float()
                fmap2_i = fmap2_i.float()
                coords_i = coords_i.float()
            (corr,) = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            if coords.dtype == torch.float16:
                corr = corr.half()
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / math.sqrt(dim)


class LocalCorrBlock:
    def __init__(self, num_levels=1, radius=4, use_translated_correlation=False):
        self.num_levels = num_levels
        self.radius = radius
        self.use_translated_correlation = use_translated_correlation
        self.side = 2 * radius + 1

        if self.use_translated_correlation:
            self.corr = TranslatedSpatialCorrelationSampler(
                kernel_size=1, patch_size=2 ** (num_levels - 1) * self.side, padding=0
            )
        else:
            self.corr = SpatialCorrelationSampler(
                kernel_size=1, patch_size=2 ** (num_levels - 1) * self.side, padding=0
            )

    def __call__(self, fmap1, fmap2, flow=None):
        if self.use_translated_correlation:
            out_corr = self.corr(fmap1, fmap2, flow)
        else:
            out_corr = self.corr(fmap1, fmap2)

        out_corr = out_corr / math.sqrt(fmap1.shape[1])

        b, h2, w2, h1, w1 = out_corr.shape
        # LibreYOLO: einops rearrange "b c h2 w2 h1 w1 -> (b h1 w1) c h2 w2" in torch.
        out_corr = out_corr[:, None].permute(0, 4, 5, 1, 2, 3).reshape(b * h1 * w1, 1, h2, w2)
        corr_pyr = []
        for i in range(self.num_levels):
            if i > 0:
                out_corr = F.avg_pool2d(out_corr, 2, stride=2)

            hm, wm = out_corr.shape[2] // 2, out_corr.shape[3] // 2
            corr_pyr.append(
                out_corr[
                    :,
                    :,
                    hm - self.radius : hm + self.radius + 1,
                    wm - self.radius : wm + self.radius + 1,
                ]
            )
        for i in range(len(corr_pyr)):
            # LibreYOLO: einops "(b h1 w1) c h2 w2 -> b (c w2 h2) h1 w1" in torch.
            _, c_i, h2_i, w2_i = corr_pyr[i].shape
            corr_pyr[i] = (
                corr_pyr[i]
                .reshape(b, h1, w1, c_i, h2_i, w2_i)
                .permute(0, 3, 5, 4, 1, 2)
                .reshape(b, c_i * w2_i * h2_i, h1, w1)
            )
        out_corr = torch.cat(corr_pyr, dim=1)
        return out_corr

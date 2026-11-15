import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import ConvNextBlock
from .gma import Aggregate


class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_channel: int, dim: int):
        super().__init__()
        cor_planes = corr_channel * 2
        self.convc1 = nn.Conv2d(cor_planes, dim * 2, 1, padding=0)
        self.convc2 = nn.Conv2d(dim * 2, dim + dim // 2, 3, padding=1)
        self.convf1 = nn.Conv2d(2 * 2, dim, 7, padding=3)
        self.convf2 = nn.Conv2d(dim, dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(dim * 2, dim - 2 * 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class GMAUpdateBlock(nn.Module):
    def __init__(self, num_blocks: int, corr_channel: int, hdim: int, cdim: int):
        # net: hdim, inp: cdim
        super().__init__()
        self.encoder = BasicMotionEncoder(corr_channel, cdim)
        self.refine = []
        for i in range(num_blocks):
            self.refine.append(ConvNextBlock(3 * cdim + hdim, hdim))
        self.refine = nn.ModuleList(self.refine)
        self.aggregator = Aggregate(cdim, 1, cdim)

    def forward(self, net, inp, corr, flow, attention):
        motion_features = self.encoder(flow, corr)
        motion_features_global = self.aggregator(attention, motion_features)
        inp = torch.cat([inp, motion_features, motion_features_global], dim=1)
        for blk in self.refine:
            net = blk(torch.cat([net, inp], dim=1))
        return net

# Vendored from https://github.com/neufieldrobotics/NeuFlow_v2 (NeuFlow/refine.py) at
# commit 204b5e3744461d90303b9ff82caa7a1bb56a2ca2, Apache-2.0 (see LICENSE).
# LibreYOLO changes: package-relative imports; see _vendor/__init__.py.

import torch
from . import utils


class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        return self.relu(self.conv(x))

class Refine(torch.nn.Module):
    def __init__(self, context_dim, iter_context_dim, num_layers, levels, radius, inter_dim):
        super(Refine, self).__init__()

        self.radius = radius

        self.conv1 = ConvBlock((radius*2+1)**2*levels+context_dim+iter_context_dim+2+1, context_dim+iter_context_dim, kernel_size=3, stride=1, padding=1)

        self.conv2 = ConvBlock(context_dim+iter_context_dim, inter_dim, kernel_size=3, stride=1, padding=1)

        self.conv_layers = torch.nn.ModuleList([ConvBlock(inter_dim, inter_dim, kernel_size=3, stride=1, padding=1)
                                                for i in range(num_layers)])

        self.conv3 = torch.nn.Conv2d(inter_dim, iter_context_dim+2, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True)

        # self.hidden_act = torch.nn.Tanh()
        self.hidden_act = torch.nn.Hardtanh(min_val=-4.0, max_val=4.0)
        # self.hidden_norm = torch.nn.BatchNorm2d(feature_dim)

    def init_bhwd(self, batch_size, height, width, device, amp):
        self.radius_emb = torch.tensor(self.radius, dtype=torch.half if amp else torch.float, device=device).view(1,-1,1,1).expand([batch_size,1,height,width])

    def forward(self, corrs, context, iter_context, flow0):

        x = torch.cat([corrs, context, iter_context, flow0, self.radius_emb], dim=1)

        x = self.conv1(x)

        x = self.conv2(x)

        for layer in self.conv_layers:
            x = layer(x)

        x = self.conv3(x)

        return self.hidden_act(x[:,2:]), x[:,:2]

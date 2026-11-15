# Vendored from https://github.com/neufieldrobotics/NeuFlow_v2 (NeuFlow/upsample.py) at
# commit 204b5e3744461d90303b9ff82caa7a1bb56a2ca2, Apache-2.0 (see LICENSE).
# LibreYOLO changes: package-relative imports; see _vendor/__init__.py.

import torch
import torch.nn.functional as F
# from spatial_correlation_sampler import SpatialCorrelationSampler

class UpSample(torch.nn.Module):
    def __init__(self, feature_dim, upsample_factor):
        super(UpSample, self).__init__()

        self.upsample_factor = upsample_factor

        self.conv1 = torch.nn.Conv2d(2 + feature_dim, 256, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(256, 512, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(512, upsample_factor ** 2 * 9, 1, 1, 0)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, feature, flow):

        concat = torch.cat((flow, feature), dim=1)

        mask = self.conv3(self.relu(self.conv2(self.relu(self.conv1(concat)))))

        b, _, h, w = flow.shape

        mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
        mask = torch.softmax(mask, dim=2)

        # up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(b, 2, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

        up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
        up_flow = up_flow.reshape(b, 2, self.upsample_factor * h,
                                  self.upsample_factor * w)  # [B, 2, K*H, K*W]

        return up_flow
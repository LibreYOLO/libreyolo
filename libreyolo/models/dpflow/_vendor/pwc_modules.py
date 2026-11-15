from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F


def upsample2d_as(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    if inputs.shape[-2] != h or inputs.shape[-1] != w:
        inputs = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    return inputs


def rescale_flow(flow, width_im, height_im, to_local=True):
    if to_local:
        u_scale = float(flow.size(3) / width_im)
        v_scale = float(flow.size(2) / height_im)
    else:
        u_scale = float(width_im / flow.size(3))
        v_scale = float(height_im / flow.size(2))

    u, v = flow.chunk(2, dim=1)
    u = u * u_scale
    v = v * v_scale

    return torch.cat([u, v], dim=1)

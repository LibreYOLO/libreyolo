# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License
#
# Adapted from FlowSeek (https://github.com/mattpoggi/flowseek, commit
# ba6407370bceeb96244cad4d60d37489f6eb471e, core/flowseek.py and
# core/depth_anything_v2/dpt.py), Copyright (c) 2025 Matteo Poggi and Fabio
# Tosi, Apache License 2.0.
#
# LibreYOLO modifications (inference only, learned parameters unchanged):
# - the Depth Anything V2 encoder reuses LibreYOLO's vendored copy
#   (libreyolo/models/depth_anything/_vendor); FlowSeekDAv2 reproduces the
#   upstream FlowSeek variant whose forward returns the DPT path_1 features
#   alongside the depth map, and keeps upstream's manual attention math
#   (q * scale @ k.T, softmax, @ v) instead of the fused SDPA kernel the shared
#   vendor uses: the recurrent refinement amplifies the kernel's rounding
#   differences to tens of pixels at isolated pixels, while the manual math
#   reproduces upstream bit for bit;
# - the DAv2 weights are no longer read from weights/*.pth at construction:
#   they are part of the FlowSeek checkpoint (dav2.* keys);
# - hard-coded .cuda() calls follow the input device instead;
# - the training-only loss branch (test_mode=False) and the Hugging Face hub
#   mixin are removed; forward returns the final flow only.

import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...depth_anything._vendor.dinov2_layers.attention import Attention
from ...depth_anything._vendor.dpt import DepthAnythingV2
from .corr import CorrBlock
from .extractor import ResNetFPN
from .layer import conv3x3
from .update import BasicUpdateBlock
from .utils import InputPadder, coords_grid

DA_MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
}


def _upstream_attention_forward(self, x, attn_bias=None):
    """DINOv2 attention exactly as in upstream FlowSeek's DAv2 copy."""
    if attn_bias is not None:
        raise ValueError("FlowSeek's DINOv2 attention takes no attention bias")
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class FlowSeekDAv2(DepthAnythingV2):
    """Depth Anything V2 returning (path_1 features, relu depth), as in FlowSeek."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for module in self.pretrained.modules():
            if isinstance(module, Attention):
                module.forward = types.MethodType(_upstream_attention_forward, module)

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder], return_class_token=True
        )
        head = self.depth_head
        out = []
        for i, feat in enumerate(features):
            if head.use_clstoken:
                feat, cls_token = feat[0], feat[1]
                readout = cls_token.unsqueeze(1).expand_as(feat)
                feat = head.readout_projects[i](torch.cat((feat, readout), -1))
            else:
                feat = feat[0]
            feat = feat.permute(0, 2, 1).reshape((feat.shape[0], feat.shape[-1], patch_h, patch_w))
            feat = head.projects[i](feat)
            feat = head.resize_layers[i](feat)
            out.append(feat)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = head.scratch.layer1_rn(layer_1)
        layer_2_rn = head.scratch.layer2_rn(layer_2)
        layer_3_rn = head.scratch.layer3_rn(layer_3)
        layer_4_rn = head.scratch.layer4_rn(layer_4)

        path_4 = head.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = head.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = head.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = head.scratch.refinenet1(path_2, layer_1_rn)

        depth = head.scratch.output_conv1(path_1)
        depth = F.interpolate(depth, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        depth = head.scratch.output_conv2(depth)
        return path_1, F.relu(depth)


class FlowSeek(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.output_dim = args.dim * 2

        self.da_size = args.da_size

        self.args.corr_levels = 4
        self.args.corr_radius = args.radius
        self.args.corr_channel = args.corr_levels * (args.radius * 2 + 1) ** 2
        self.cnet = ResNetFPN(args, input_dim=6, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=False)

        self.da_model_configs = DA_MODEL_CONFIGS
        self.dav2 = FlowSeekDAv2(**self.da_model_configs[args.da_size]).eval()
        for param in self.dav2.parameters():
            param.requires_grad = False

        feats = self.da_model_configs[args.da_size]['features']
        self.merge_head = nn.Sequential(
            nn.Conv2d(feats, feats // 2 * 3, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feats // 2 * 3, feats * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feats * 2, feats * 2, 3, stride=2, padding=1),
        )

        self.bnet = ResNetFPN(args, input_dim=16, output_dim=2 * self.args.dim, norm_layer=nn.BatchNorm2d, init_weight=False)

        # conv for iter 0 results
        self.init_conv = conv3x3(2 * args.dim, 2 * args.dim)

        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(args.dim * 2, args.dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(args.dim * 2, 2 * args.dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * args.dim, 6, 3, padding=1)
        )
        if args.iters > 0:
            self.fnet = ResNetFPN(args, input_dim=3, output_dim=self.output_dim, norm_layer=nn.BatchNorm2d, init_weight=False)
            self.update_block = BasicUpdateBlock(args, hdim=args.dim * 2, cdim=args.dim * 2)

    def create_bases(self, disp):
        B, C, H, W = disp.shape
        assert C == 1
        cx = 0.5
        cy = 0.5

        ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H)
        xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W)
        u, v = torch.meshgrid(xs, ys, indexing='xy')
        u = u - cx
        v = v - cy
        u = u.unsqueeze(0).unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(0)
        u = u.repeat(B, 1, 1, 1).to(disp.device)
        v = v.repeat(B, 1, 1, 1).to(disp.device)

        aspect_ratio = W / H

        Tx = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        Ty = torch.cat([torch.zeros_like(disp), -torch.ones_like(disp)], dim=1)
        Tz = torch.cat([u, v], dim=1)

        Tx = Tx / torch.linalg.vector_norm(Tx, dim=(1, 2, 3), keepdim=True)
        Ty = Ty / torch.linalg.vector_norm(Ty, dim=(1, 2, 3), keepdim=True)
        Tz = Tz / torch.linalg.vector_norm(Tz, dim=(1, 2, 3), keepdim=True)

        Tx = 2 * disp * Tx
        Ty = 2 * disp * Ty
        Tz = 2 * disp * Tz

        R1x = torch.cat([torch.zeros_like(disp), torch.ones_like(disp)], dim=1)
        R2x = torch.cat([u * v, v * v], dim=1)
        R1y = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        R2y = torch.cat([-u * u, -u * v], dim=1)
        Rz = torch.cat([-v / aspect_ratio, u * aspect_ratio], dim=1)

        R1x = R1x / torch.linalg.vector_norm(R1x, dim=(1, 2, 3), keepdim=True)
        R2x = R2x / torch.linalg.vector_norm(R2x, dim=(1, 2, 3), keepdim=True)
        R1y = R1y / torch.linalg.vector_norm(R1y, dim=(1, 2, 3), keepdim=True)
        R2y = R2y / torch.linalg.vector_norm(R2y, dim=(1, 2, 3), keepdim=True)
        Rz = Rz / torch.linalg.vector_norm(Rz, dim=(1, 2, 3), keepdim=True)

        M = torch.cat([Tx, Ty, Tz, R1x, R2x, R1y, R2y, Rz], dim=1)  # Bx(8x2)xHxW
        return M

    def upsample_data(self, flow, info, mask):
        """ Upsample [H/8, W/8, C] -> [H, W, C] using convex combination """
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, 2, 8 * H, 8 * W), up_info.reshape(N, C, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=None):
        """Estimate optical flow from image1 to image2.

        Inputs are RGB float tensors in [0, 255], shape (B, 3, H, W). Returns
        the final (B, 2, H, W) flow at the input resolution.
        """
        N, _, H, W = image1.shape
        if iters is None:
            iters = self.args.iters

        image1_res = F.interpolate(image1, (518, 518), mode="bilinear", align_corners=False) / 255.
        image2_res = F.interpolate(image2, (518, 518), mode="bilinear", align_corners=False) / 255.

        mean = torch.from_numpy(np.array([0.485, 0.456, 0.406])).unsqueeze(0).unsqueeze(2).unsqueeze(2).to(image1.device)
        std = torch.from_numpy(np.array([0.229, 0.224, 0.225])).unsqueeze(0).unsqueeze(2).unsqueeze(2).to(image1.device)
        # Upstream note: should be (x - mean) / std, but the released models
        # were trained with x / mean - std, so it is kept as is.
        image1_res = image1_res / mean - std
        image2_res = image2_res / mean - std

        im1_path1, depth1 = self.dav2.forward(image1_res.float())
        im2_path1, _ = self.dav2.forward(image2_res.float())

        im1_path1 = F.interpolate(im1_path1, (H, W), mode="bilinear", align_corners=False)
        im2_path1 = F.interpolate(im2_path1, (H, W), mode="bilinear", align_corners=False)
        bases1 = self.create_bases(F.interpolate(depth1, (H, W), mode="bilinear", align_corners=False))

        mono1 = self.merge_head(im1_path1)
        mono2 = self.merge_head(im2_path1)

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # padding
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        bases1 = padder.pad(bases1)

        N, _, H, W = image1.shape
        dilation = torch.ones(N, 1, H // 8, W // 8, device=image1.device)

        # run the context network
        cnet_inputs = torch.cat([image1, image2], dim=1)
        cnet = self.cnet(cnet_inputs)
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.args.dim, self.args.dim], dim=1)

        bnet_inputs = bases1[0]
        bnet = self.bnet(bnet_inputs)
        bnet = self.init_conv(bnet)
        netbases, ctxbases = torch.split(bnet, [self.args.dim, self.args.dim], dim=1)

        context = torch.cat((context, ctxbases), 1)
        net = torch.cat((net, netbases), 1)

        # init flow
        flow_update = self.flow_head(net)
        weight_update = .25 * self.upsample_weight(net)
        flow_8x = flow_update[:, :2]
        info_8x = flow_update[:, 2:]
        flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)

        if self.args.iters > 0:
            # run the feature network
            fmap1_8x = self.fnet(image1)
            fmap2_8x = self.fnet(image2)

            fmap1_8x = torch.cat((fmap1_8x, mono1), 1)
            fmap2_8x = torch.cat((fmap2_8x, mono2), 1)

            corr_fn = CorrBlock(fmap1_8x, fmap2_8x, self.args)

        for itr in range(iters):
            N, _, H, W = flow_8x.shape
            flow_8x = flow_8x.detach()
            coords2 = (coords_grid(N, H, W, device=image1.device) + flow_8x).detach()
            corr = corr_fn(coords2, dilation=dilation)
            net = self.update_block(net, context, corr, flow_8x)
            flow_update = self.flow_head(net)
            weight_update = .25 * self.upsample_weight(net)
            flow_8x = flow_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            # upsample predictions
            flow_up, info_up = self.upsample_data(flow_8x, info_8x, weight_update)

        return padder.unpad(flow_up)

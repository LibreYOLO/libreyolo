# Copyright (c) 2025, Princeton Vision & Learning Lab. BSD-3-Clause, see LICENSE.
# Adapted from WAFT model/waft_a1.py, model/waft_a2.py and the encoder wrappers
# in model/backbone/{waftv1_dav2,waftv2_dav2,dinov3,twins}.py, inference only.
#
# Changes from upstream, none of which touch learned parameters or their names:
# - training branches (flow_gt, the mixture-of-Laplace loss) are removed;
# - encoders are built without downloading pretrained weights (the WAFT
#   checkpoint carries every tensor), from the Depth Anything V2 and DINOv3
#   code already vendored in LibreYOLO and a plain-torch Twins copy;
# - timm's ResNet-18 stages (a1) are rebuilt as plain BasicBlocks;
# - ``forward`` returns only the last iteration's flow and info by default.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...depth_anything._vendor.dinov2 import DINOv2
from ...depth_anything._vendor.dpt import DepthAnythingV2
from .head import DPTHead
from .utils import Padder, bilinear_sampler, coords_grid
from .vit import MODEL_CONFIGS, VisionTransformer, dpt_intermediate

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DAV2_CONFIGS = {
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}
INTERMEDIATE_LAYER_IDX = {"vits": [2, 5, 8, 11], "vitb": [2, 5, 8, 11], "vitl": [4, 11, 17, 23]}


class resconv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k // 2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(inp, oup, kernel_size=1, stride=s, padding=0, bias=True)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)


class BasicBlock(nn.Module):
    """ResNet basic block with timm's parameter names (resnet18 stages)."""

    def __init__(self, inp, oup, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(oup, oup, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or inp != oup:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp, oup, 1, stride, bias=False), nn.BatchNorm2d(oup)
            )

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act2(x + shortcut)


def _resnet_stage(inp, oup, stride):
    return nn.Sequential(BasicBlock(inp, oup, stride), BasicBlock(oup, oup, 1))


class ResNet18DeconvA1(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        self.ds1 = resconv(inp, 64, k=7, s=2)
        self.conv1 = _resnet_stage(64, 64, 1)
        self.conv2 = _resnet_stage(64, 128, 2)
        self.conv3 = _resnet_stage(128, 256, 2)
        self.conv4 = _resnet_stage(256, 512, 2)
        self.up_4 = nn.ConvTranspose2d(512, 256, 2, 2, 0, bias=True)
        self.proj_3 = resconv(256, 256)
        self.up_3 = nn.ConvTranspose2d(256, 128, 2, 2, 0, bias=True)
        self.proj_2 = resconv(128, 128)
        self.up_2 = nn.ConvTranspose2d(128, 64, 2, 2, 0, bias=True)
        self.proj_1 = resconv(64, oup)

    def forward(self, x):
        out_1 = self.conv1(self.ds1(x))
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_3 = self.proj_3(out_3 + self.up_4(out_4))
        out_2 = self.proj_2(out_2 + self.up_3(out_3))
        return self.proj_1(out_1 + self.up_2(out_2))


class ResNet18DeconvA2(ResNet18DeconvA1):
    def __init__(self, inp, oup):
        super().__init__(inp, oup)
        self.conv1 = resconv(64, 64, k=3, s=1)
        self.conv2 = resconv(64, 128, k=3, s=2)
        self.conv3 = resconv(128, 256, k=3, s=2)
        self.conv4 = resconv(256, 512, k=3, s=2)


class _WAFTBase(nn.Module):
    """Shared warping iteration of WAFT a1 and a2."""

    factor = 112

    def __init__(self, iter_dim, iters):
        super().__init__()
        self.iters = iters
        self.hidden_conv = nn.Conv2d(iter_dim * 2, iter_dim, 1, 1, 0, bias=True)
        self.warp_linear = nn.Conv2d(3 * iter_dim + 2, iter_dim, 1, 1, 0, bias=True)
        self.refine_transform = nn.Conv2d(iter_dim // 2 * 3, iter_dim, 1, 1, 0, bias=True)
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(iter_dim, 2 * iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * iter_dim, 4 * 9, 1, padding=0, bias=True),
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(iter_dim, 2 * iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * iter_dim, 6, 1, padding=0, bias=True),
        )
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @staticmethod
    def upsample_data(flow, info, mask):
        N, C, H, W = info.shape
        mask = torch.softmax(mask.view(N, 1, 9, 2, 2, H, W), dim=2)
        up_flow = F.unfold(2 * flow, [3, 3], padding=1).view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1).view(N, C, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2).permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2).permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 2 * H, 2 * W), up_info.reshape(N, C, 2 * H, 2 * W)

    def normalize_image(self, img):
        """``img`` (B, 3, H, W) RGB in [0, 255]."""
        return ((img / 255.0 - self.pixel_mean) / self.pixel_std).contiguous()

    def features(self, image1, image2):  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, image1, image2, iters=None, all_iters=False):
        """Flow from ``image1`` to ``image2`` (both (B, 3, H, W), RGB 0-255)."""
        iters = self.iters if iters is None else iters
        image1 = self.normalize_image(image1)
        image2 = self.normalize_image(image2)
        padder = Padder(image1.shape, factor=self.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        N, _, H, W = image1.shape
        fmap1_2x, fmap2_2x = self.features(image1, image2)
        net = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        flow_2x = torch.zeros(N, 2, H // 2, W // 2, device=image1.device, dtype=fmap1_2x.dtype)
        grid = coords_grid(N, H // 2, W // 2, device=image1.device).to(fmap1_2x.dtype)
        flows, infos = [], []
        for itr in range(iters):
            coords2 = grid + flow_2x
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1))
            refine_out = self.refine_net(refine_inp)
            net = self.refine_transform(torch.cat([refine_out, net], dim=1))
            flow_update = self.flow_head(net)
            weight_update = 0.25 * self.upsample_weight(net)
            flow_2x = flow_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            if all_iters or itr == iters - 1:
                flow_up, info_up = self.upsample_data(flow_2x, info_2x, weight_update)
                flows.append(padder.unpad(flow_up))
                infos.append(padder.unpad(info_up))
        if all_iters:
            return {"flow": flows, "info": infos}
        return {"flow": flows[-1], "info": infos[-1]}


class DepthAnythingFeatureA1(nn.Module):
    """WAFT a1 encoder: the full Depth Anything V2 DPT, intermediate output."""

    def __init__(self, encoder="vits"):
        super().__init__()
        self.encoder = encoder
        self.model_configs = DAV2_CONFIGS
        self.depth_anything = DepthAnythingV2(**DAV2_CONFIGS[encoder])

    def forward(self, x):
        h, w = x.shape[-2:]
        da = self.depth_anything
        feats = da.pretrained.get_intermediate_layers(
            x, da.intermediate_layer_idx[self.encoder], return_class_token=True
        )
        p = da.pretrained.patch_size
        out, *_ = dpt_intermediate(da.depth_head, feats, h // p, w // p)
        return out


class WAFTA1(_WAFTBase):
    """WAFT a1 (upstream ``ViTWarpV8``): DAv2 features plus a ResNet-18 U-Net."""

    factor = 112

    def __init__(self, dav2_backbone="vits", network_backbone="vits", iters=5):
        iter_dim = MODEL_CONFIGS[network_backbone]["features"]
        super().__init__(iter_dim, iters)
        self.da_feature = DepthAnythingFeatureA1(dav2_backbone)
        self.pretrain_dim = DAV2_CONFIGS[dav2_backbone]["features"]
        self.network_dim = iter_dim
        self.refine_net = VisionTransformer(network_backbone, iter_dim, patch_size=8)
        self.fnet = ResNet18DeconvA1(self.pretrain_dim // 2 + 3, 64)
        self.fmap_conv = nn.Conv2d(self.pretrain_dim // 2 + 64, iter_dim, 1, 1, 0, bias=True)

    def features(self, image1, image2):
        da1 = self.da_feature(image1)
        da2 = self.da_feature(image2)
        f1 = self.fnet(torch.cat([da1, image1], dim=1))
        f2 = self.fnet(torch.cat([da2, image2], dim=1))
        da1 = F.interpolate(da1, scale_factor=0.5, mode="bilinear", align_corners=True)
        da2 = F.interpolate(da2, scale_factor=0.5, mode="bilinear", align_corners=True)
        return (
            self.fmap_conv(torch.cat([f1, da1], dim=1)),
            self.fmap_conv(torch.cat([f2, da2], dim=1)),
        )


class DepthAnythingFeatureA2(nn.Module):
    """WAFT a2 DAv2 encoder: frozen DINOv2 ViT plus a fresh DPT head."""

    output_dim = 64

    def __init__(self, model_name="vits", lvl=-3):
        super().__init__()
        cfg = DAV2_CONFIGS[model_name]
        self.model_name = model_name
        self.encoder = DINOv2(model_name)
        self.embed_dim = self.encoder.embed_dim
        self.output_dim = cfg["features"]
        self.dpt_head = DPTHead(self.embed_dim, features=self.output_dim, out_channels=cfg["out_channels"], lvl=lvl)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.encoder.get_intermediate_layers(
            x, INTERMEDIATE_LAYER_IDX[self.model_name], return_class_token=True
        )
        p = self.encoder.patch_size
        outs = self.dpt_head(feats, h // p, w // p)
        return F.interpolate(outs[0], size=(h // 2, w // 2), mode="bilinear", align_corners=True)


class DinoV3FeatureA2(nn.Module):
    """WAFT a2 DINOv3 encoder (Meta DINOv3 License, see the DEIMv2 vendor)."""

    def __init__(self, model_name="vits", lvl=-3):
        super().__init__()
        from ...deimv2.engine.backbone.dinov3.vision_transformer import DinoVisionTransformer

        cfg = DAV2_CONFIGS[model_name]
        self.model_name = model_name
        self.encoder = DinoVisionTransformer(name=f"dinov3_{model_name}16")
        self.embed_dim = self.encoder.embed_dim
        self.output_dim = cfg["features"]
        self.dpt_head = DPTHead(self.embed_dim, features=self.output_dim, out_channels=cfg["out_channels"], lvl=lvl)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.encoder.get_intermediate_layers(
            x, n=INTERMEDIATE_LAYER_IDX[self.model_name], return_class_token=True
        )
        p = self.encoder.patch_size
        outs = self.dpt_head(feats, h // p, w // p)
        return F.interpolate(outs[0], size=(h // 2, w // 2), mode="bilinear", align_corners=True)


class WAFTA2(_WAFTBase):
    """WAFT a2 (upstream ``WAFTv2``) with a Twins, DAv2 or DINOv3 encoder."""

    FACTORS = {"twins": 32, "dav2": 112, "dinov3": 16}

    def __init__(self, feature_encoder="dav2", iterative_module="vits", iters=5):
        iter_dim = MODEL_CONFIGS[iterative_module]["features"]
        super().__init__(iter_dim, iters)
        if feature_encoder == "twins":
            from .twins import TwinsFeatureEncoder

            self.encoder = TwinsFeatureEncoder()
        elif feature_encoder == "dav2":
            self.encoder = DepthAnythingFeatureA2("vits", lvl=-3)
        elif feature_encoder == "dinov3":
            self.encoder = DinoV3FeatureA2("vits", lvl=-3)
        else:
            raise ValueError(f"Unknown WAFT feature encoder: {feature_encoder!r}")
        self.factor = self.FACTORS[feature_encoder]
        self.pretrain_dim = self.encoder.output_dim
        self.fnet = ResNet18DeconvA2(3, self.pretrain_dim)
        self.iter_dim = iter_dim
        self.refine_net = VisionTransformer(iterative_module, iter_dim, patch_size=8)
        self.fmap_conv = nn.Conv2d(self.pretrain_dim * 2, iter_dim, 1, 1, 0, bias=True)

    def features(self, image1, image2):
        e1 = self.encoder(image1)
        e2 = self.encoder(image2)
        return (
            self.fmap_conv(torch.cat([e1, self.fnet(image1)], dim=1)),
            self.fmap_conv(torch.cat([e2, self.fnet(image2)], dim=1)),
        )

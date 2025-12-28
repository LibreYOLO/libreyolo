"""
YOLOX neural network architecture for LibreYOLO.

This module implements the YOLOX architecture with layer names matching
the official YOLOX repository for direct weight loading.
"""

import torch
import torch.nn as nn


def get_activation(name="silu", inplace=True):
    """Get activation function by name."""
    if name == "silu":
        return nn.SiLU(inplace=inplace)
    elif name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise ValueError(f"Unsupported activation type: {name}")


class BaseConv(nn.Module):
    """A Conv2d -> BatchNorm -> SiLU/LeakyReLU block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride,
        groups=1,
        bias=False,
        act="silu"
    ):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise Conv + Pointwise Conv (for nano/tiny models)."""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act
        )
        self.pconv = BaseConv(
            in_channels,
            out_channels,
            ksize=1,
            stride=1,
            groups=1,
            act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    """Standard bottleneck block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu"
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling Bottleneck."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=(5, 9, 13),
        activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """CSP Bottleneck with 3 convolutions (C3 in YOLOv5)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu"
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        self.m = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act)
            for _ in range(n)
        ])

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus: Space to depth and then convolution."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # Patch extraction: (B, C, H, W) -> (B, 4C, H/2, W/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right),
            dim=1
        )
        return self.conv(x)


class CSPDarknet(nn.Module):
    """CSPDarknet backbone for YOLOX."""

    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu"
    ):
        super().__init__()
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        # Stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act
            )
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act
            )
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act
            )
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act
            )
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class YOLOPAFPN(nn.Module):
    """YOLO Path Aggregation Feature Pyramid Network."""

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu"
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down path
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

        # Bottom-up path
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act
        )

    def forward(self, input):
        # Backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        # Top-down path
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], 1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_out1)

        # Bottom-up path
        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


class YOLOXHead(nn.Module):
    """YOLOX decoupled detection head."""

    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    Conv(int(256 * width), int(256 * width), 3, 1, act=act),
                    Conv(int(256 * width), int(256 * width), 3, 1, act=act)
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    Conv(int(256 * width), int(256 * width), 3, 1, act=act),
                    Conv(int(256 * width), int(256 * width), 3, 1, act=act)
                )
            )
            self.cls_preds.append(
                nn.Conv2d(int(256 * width), self.num_classes, 1, 1, 0)
            )
            self.reg_preds.append(
                nn.Conv2d(int(256 * width), 4, 1, 1, 0)
            )
            self.obj_preds.append(
                nn.Conv2d(int(256 * width), 1, 1, 1, 0)
            )

    def forward(self, xin):
        """Forward pass for inference."""
        outputs = []

        for k, x in enumerate(xin):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = self.cls_convs[k](cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = self.reg_convs[k](reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # For inference: concat [reg, obj, cls] with sigmoid on obj and cls
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )
            outputs.append(output)

        return outputs


class YOLOXModel(nn.Module):
    """
    Complete YOLOX model combining backbone (PAFPN) and head.

    This class matches the official YOLOX weight structure for direct loading.
    """

    # Model configurations: depth, width, depthwise
    CONFIGS = {
        'nano': {'depth': 0.33, 'width': 0.25, 'depthwise': True},
        'tiny': {'depth': 0.33, 'width': 0.375, 'depthwise': True},
        's': {'depth': 0.33, 'width': 0.50, 'depthwise': False},
        'm': {'depth': 0.67, 'width': 0.75, 'depthwise': False},
        'l': {'depth': 1.00, 'width': 1.00, 'depthwise': False},
        'x': {'depth': 1.33, 'width': 1.25, 'depthwise': False},
    }

    def __init__(
        self,
        config: str = "s",
        nb_classes: int = 80,
        act: str = "silu"
    ):
        """
        Initialize YOLOX model.

        Args:
            config: Model size ('nano', 'tiny', 's', 'm', 'l', 'x')
            nb_classes: Number of classes (default: 80 for COCO)
            act: Activation function (default: 'silu')
        """
        super().__init__()

        if config not in self.CONFIGS:
            raise ValueError(f"Invalid config: {config}. Must be one of: {list(self.CONFIGS.keys())}")

        cfg = self.CONFIGS[config]
        depth = cfg['depth']
        width = cfg['width']
        depthwise = cfg['depthwise']

        in_channels = [256, 512, 1024]

        # Backbone includes PAFPN (matching official YOLOX structure)
        self.backbone = YOLOPAFPN(
            depth=depth,
            width=width,
            in_channels=in_channels,
            depthwise=depthwise,
            act=act
        )

        # Detection head
        self.head = YOLOXHead(
            num_classes=nb_classes,
            width=width,
            in_channels=in_channels,
            depthwise=depthwise,
            act=act
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            List of 3 tensors, one per scale, each with shape:
            (B, 5 + num_classes, H/stride, W/stride)
            where channels are [reg_x, reg_y, reg_w, reg_h, obj, cls...]
        """
        # Get FPN features
        fpn_outs = self.backbone(x)

        # Get detection outputs
        outputs = self.head(fpn_outs)

        return outputs

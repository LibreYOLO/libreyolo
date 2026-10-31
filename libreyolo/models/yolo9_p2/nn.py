"""YOLOv9-P2 network: YOLOv9 with an extra stride-4 detection scale.

The stock YOLOv9 backbone computes a stride-4 feature map (the ``elan1``
output) and discards it. This variant routes it through one additional
top-down/bottom-up neck level and a 4-scale detection head with strides
(4, 8, 16, 32), so objects in the 4-16 px range land on a grid fine enough
to be assigned and localized.

Only the ``t`` and ``s`` sizes are defined; the P2 level roughly doubles the
compute of the base model, which defeats the purpose on the larger sizes.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from ..yolo9.nn import (
    AConv,
    ADown,
    RepNCSPELAN,
    YOLO9_CONFIGS,
    Backbone9,
    LibreYOLO9Model,
    Neck9,
    YOLO9Head,
    default_class_neck,
)

# Extra keys per size on top of the base YOLO9 config. Channel choices
# continue each size's arithmetic progression (t: 32/64/96/128) so the
# existing bottom-up neck modules keep identical shapes and transfer
# weight-for-weight from base checkpoints.
_P2_EXTRAS = {
    "t": {
        "neck_elan_up3": (32, 32),  # out_p2: out=32, part=32
        "neck_down0_out": 32,  # AConv after out_p2
        "neck_elan_down0": (64, 64),  # out_p3: out=64, part=64
        "head_channels": (32, 64, 96, 128),  # P2, P3, P4, P5
    },
    "s": {
        "neck_elan_up3": (64, 64),
        "neck_down0_out": 64,
        "neck_elan_down0": (128, 128),
        "head_channels": (64, 128, 192, 256),
    },
}

YOLO9_P2_CONFIGS = {}
for _size, _extras in _P2_EXTRAS.items():
    _cfg = deepcopy(YOLO9_CONFIGS[_size])
    _cfg.update(_extras)
    YOLO9_P2_CONFIGS[_size] = _cfg


class Backbone9P2(Backbone9):
    """YOLOv9 backbone that also surfaces the stride-4 feature map.

    Identical modules and compute to :class:`Backbone9`; the only change is
    that the ``elan1`` output (stride 4) is returned instead of discarded.
    """

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        p2 = self.elan1(x)

        x = self.down2(p2)
        p3 = self.elan2(x)

        x = self.down3(p3)
        p4 = self.elan3(x)

        x = self.down4(p4)
        x = self.elan4(x)
        p5 = self.spp(x)

        return p2, p3, p4, p5


class Neck9P2(Neck9):
    """YOLOv9 PAN neck extended one level down to stride 4.

    Reuses every module of :class:`Neck9` under its original attribute name
    (so base-checkpoint keys still match) and adds one top-down step onto P2
    plus one bottom-up step back to P3. The bottom-up P3 output keeps the
    base channel count, so ``down1``/``elan_down1`` are unchanged.
    """

    def __init__(self, config="t"):
        super().__init__(config)

        cfg = YOLO9_P2_CONFIGS[config]
        n = cfg["repeat_num"]
        DownBlock = ADown if cfg["down_type"] == "adown" else AConv

        p2_ch = cfg["first_block_out"]  # backbone stride-4 channels
        n3_ch = cfg["neck_elan_up2"][0]  # top-down P3 output channels

        # Top-down: N3 -> P2
        up3_out, up3_part = cfg["neck_elan_up3"]
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.elan_up3 = RepNCSPELAN(
            n3_ch + p2_ch, up3_out, up3_part, csp_args={"repeat_num": n}
        )

        # Bottom-up: P2 -> P3
        self.down0 = DownBlock(up3_out, cfg["neck_down0_out"])
        down0_out, down0_part = cfg["neck_elan_down0"]
        self.elan_down0 = RepNCSPELAN(
            cfg["neck_down0_out"] + n3_ch,
            down0_out,
            down0_part,
            csp_args={"repeat_num": n},
        )

    def forward(self, p2, p3, p4, p5):
        # Top-down path
        x = self.up1(p5)
        x = torch.cat([x, p4], 1)
        n4 = self.elan_up1(x)

        x = self.up2(n4)
        x = torch.cat([x, p3], 1)
        n3 = self.elan_up2(x)

        x = self.up3(n3)
        x = torch.cat([x, p2], 1)
        out_p2 = self.elan_up3(x)

        # Bottom-up path
        x = self.down0(out_p2)
        x = torch.cat([x, n3], 1)
        out_p3 = self.elan_down0(x)

        x = self.down1(out_p3)
        x = torch.cat([x, n4], 1)
        out_p4 = self.elan_down1(x)

        x = self.down2(out_p4)
        x = torch.cat([x, p5], 1)
        out_p5 = self.elan_down2(x)

        return out_p2, out_p3, out_p4, out_p5


class LibreYOLO9P2Model(LibreYOLO9Model):
    """Complete YOLOv9-P2 model: 4-scale detection with strides (4, 8, 16, 32)."""

    def __init__(self, config="t", reg_max=16, nb_classes=80, img_size=640):
        if config not in YOLO9_P2_CONFIGS:
            raise ValueError(
                f"Invalid config: {config}. Must be one of: "
                f"{list(YOLO9_P2_CONFIGS.keys())}"
            )
        # Skip LibreYOLO9Model.__init__: it would build a full 3-scale
        # backbone/neck/head only for all three to be replaced below.
        nn.Module.__init__(self)
        self.config = config
        self.nc = nb_classes
        self.reg_max = reg_max
        self.img_size = img_size
        self.backbone = Backbone9P2(config)
        self.neck = Neck9P2(config)
        head_channels = YOLO9_P2_CONFIGS[config]["head_channels"]
        self.head = YOLO9Head(
            head_channels,
            nb_classes,
            reg_max=reg_max,
            strides=(4, 8, 16, 32),
            class_neck=default_class_neck(head_channels[0], nb_classes),
        )

    def forward(self, x, targets=None):
        p2, p3, p4, p5 = self.backbone(x)
        n2, n3, n4, n5 = self.neck(p2, p3, p4, p5)

        if self.training and targets is not None:
            img_size = (x.shape[3], x.shape[2])  # (W, H)
            return self.head([n2, n3, n4, n5], targets=targets, img_size=img_size)

        output = self.head([n2, n3, n4, n5])

        if self.training:
            return output

        y, x_list = output

        if self.head.export:
            return y

        return {
            "predictions": y,  # (batch, 4+nc, total_anchors)
            "raw_outputs": x_list,
            "x4": {"features": n2},
            "x8": {"features": n3},
            "x16": {"features": n4},
            "x32": {"features": n5},
        }


__all__ = ["YOLO9_P2_CONFIGS", "Backbone9P2", "Neck9P2", "LibreYOLO9P2Model"]

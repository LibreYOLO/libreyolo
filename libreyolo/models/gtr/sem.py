"""GTR semantic segmentation: backbone, one-level encoder and an FCN head.

Adapted from Intellindust-AI-Lab/GTR (MIT), revision
782e737efe2e6437ac537fbdcee089673d3376c1: ``engine/gtr/semseg/gtrsemseg.py``,
``configs/semseg`` and the sliding-window evaluation in
``engine/solver/semseg_solver.py``.
Changes: native construction, ImageNet normalization inside ``forward`` (the
LibreYOLO semantic input contract is RGB in [0, 1]), and sliding windows that
also accept inputs smaller than the window by rescaling. See NOTICE.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ...data.augment.constants import IMAGENET_MEAN, IMAGENET_STD
from .encoder import GTREncoder
from .nn import SIZE_CONFIGS
from .spatial import ViTAdapterSpatialSwiGLU

IGNORE_INDEX = 255
# Upstream trains and evaluates the 1024px square window with a 768px stride.
SEM_WINDOW = 1024
SEM_STRIDE_RATIO = 0.75

CITYSCAPES_NAMES = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}


class SemSegHead(nn.Module):
    """conv3x3-BN-ReLU -> dropout -> conv1x1, with upstream parameter names."""

    def __init__(self, in_channels=256, hidden_dim=256, num_classes=19, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, feat):
        return self.classifier(self.dropout(self.act(self.bn(self.conv(feat)))))


class LibreGTRSemModel(nn.Module):
    """GTR semantic network. Input: RGB in [0, 1]; output: logits at input size.

    A square input whose side is a multiple of 32 runs in one pass. Any other
    input is evaluated with overlapping ``window`` squares (stride
    ``0.75 * window``) whose logits are averaged, as upstream evaluates
    Cityscapes. Inputs shorter than the window are rescaled up first.
    """

    def __init__(self, config="s", nb_classes=19, window=SEM_WINDOW):
        super().__init__()
        embed, heads, ratio, hidden, _ = SIZE_CONFIGS[config]
        self.window = int(window)
        self.backbone = ViTAdapterSpatialSwiGLU(
            embed_dim=embed,
            num_heads=heads,
            ffn_ratio=ratio,
            interaction_indexes=[3, 7, 11],
            multi_layer_same_res=True,
            skip_weights_warning=True,
            eval_spatial_size=(self.window, self.window),
        )
        self.encoder = GTREncoder(
            in_channels=[embed] * 3,
            hidden_dim=hidden,
            scale_factors=[2.0],
            feat_strides=[8],
            eval_spatial_size=(self.window, self.window),
        )
        self.head = SemSegHead(hidden, hidden, nb_classes, dropout=0.1)
        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    @property
    def num_classes(self) -> int:
        return self.head.classifier.out_channels

    def forward_normalized(self, x):
        """Upstream ``GTRSemSeg.forward`` on an ImageNet-normalized square."""
        if x.shape[-2] != x.shape[-1] or x.shape[-1] % 32:
            raise ValueError("GTR expects a square input with side divisible by 32")
        logits = self.head(self.encoder(self.backbone(x))[0])
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

    def slide(self, x):
        """Average overlapping window logits over a normalized input."""
        _, _, height, width = x.shape
        if height == width and height % 32 == 0:
            return self.forward_normalized(x)
        window = self.window
        if min(height, width) < window:
            scale = window / min(height, width)
            size = (
                max(window, round(height * scale)),
                max(window, round(width * scale)),
            )
            resized = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            logits = self.slide(resized)
            return F.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
        stride = int(window * SEM_STRIDE_RATIO)
        rows = (height - window + stride - 1) // stride + 1
        cols = (width - window + stride - 1) // stride + 1
        logits = None
        count = x.new_zeros((1, 1, height, width))
        for row in range(rows):
            for col in range(cols):
                y2 = min(row * stride + window, height)
                x2 = min(col * stride + window, width)
                # Shift the last window back inside instead of shrinking it.
                y1, x1 = y2 - window, x2 - window
                crop = self.forward_normalized(x[:, :, y1:y2, x1:x2])
                if logits is None:
                    logits = x.new_zeros((x.shape[0], crop.shape[1], height, width))
                logits[:, :, y1:y2, x1:x2] += crop
                count[:, :, y1:y2, x1:x2] += 1
        return logits / count

    def loss(self, logits, targets):
        """Per-pixel cross-entropy, sum over valid pixels divided by their count."""
        targets = targets.long()
        if tuple(targets.shape[-2:]) != tuple(logits.shape[-2:]):
            logits = F.interpolate(
                logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )
        loss_sum = F.cross_entropy(
            logits, targets, ignore_index=IGNORE_INDEX, reduction="sum"
        )
        loss = loss_sum / (targets != IGNORE_INDEX).sum().clamp(min=1)
        return {"total_loss": loss, "sem": loss}

    def forward(self, x, targets=None):
        x = (x - self.pixel_mean.to(x.dtype)) / self.pixel_std.to(x.dtype)
        if self.training and targets is not None:
            return self.loss(self.forward_normalized(x), targets)
        return self.slide(x)

    def deploy(self):
        return self.eval()


def preprocess_image(image, input_size, color_format="auto"):
    """Letterbox into the ``(h, w)`` canvas (top-left, grey pad), RGB in [0, 1].

    The same geometry as the semantic dataset's validation letterbox and the
    exported-backend preprocessing, so predict, val and export agree. A
    Cityscapes frame at the default 1024x2048 canvas passes through unchanged.
    """
    import numpy as np

    from ...utils.image_loader import ImageLoader
    from ..segformer.model import preprocess_numpy

    img = ImageLoader.load(image, color_format=color_format)
    chw, ratio = preprocess_numpy(np.asarray(img.convert("RGB")), input_size)
    return torch.from_numpy(chw).unsqueeze(0), img, img.size, ratio


def logits_at(output, original_size, ratio=1.0):
    """Crop the letterboxed content and resize logits to ``original_size`` (w, h)."""
    logits = output.get("semantic_logits") if isinstance(output, dict) else output
    orig_w, orig_h = original_size
    valid_h = min(logits.shape[-2], max(1, int(round(orig_h * ratio))))
    valid_w = min(logits.shape[-1], max(1, int(round(orig_w * ratio))))
    return F.interpolate(
        logits[..., :valid_h, :valid_w].float(),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )


def postprocess(output, original_size, ratio=1.0):
    return {"semantic": logits_at(output, original_size, ratio).argmax(dim=1)[0].cpu()}


def is_semantic_state_dict(state_dict) -> bool:
    return "head.classifier.weight" in state_dict and not any(
        key.startswith("decoder.") for key in state_dict
    )


__all__ = [
    "CITYSCAPES_NAMES",
    "logits_at",
    "postprocess",
    "preprocess_image",
    "IGNORE_INDEX",
    "LibreGTRSemModel",
    "SEM_WINDOW",
    "SemSegHead",
    "is_semantic_state_dict",
]

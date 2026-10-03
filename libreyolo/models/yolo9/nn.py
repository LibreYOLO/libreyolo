"""YOLOv9 network: building blocks, detection head and model assembly.

Provenance: architecture blocks, detection-head towers and bias
initialization are ported from MultimediaTechLab/YOLO
(https://github.com/MultimediaTechLab/YOLO, commit c4cb5f6f, MIT License,
Copyright (c) 2024 Kin-Yiu Wong and Hao-Tang Tsui): ``yolo/model/module.py``
and ``yolo/utils/module_utils.py``. Anchor generation and box decoding follow
``yolo/utils/bounding_box_utils.py`` (``generate_anchors``, ``Vec2Box``)
there. The auxiliary (PGI) branches follow the ``auxiliary`` sections of
``yolo/config/model/v9-{t,s,m,c}.yaml``. Model assembly, checkpoint loading
and export glue are LibreYOLO code.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_2_t

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers (MultimediaTechLab/YOLO yolo/utils/module_utils.py)
# =============================================================================


def auto_pad(kernel_size: _size_2_t, dilation: _size_2_t = 1, **kwargs) -> Tuple[int, int]:
    """Padding that keeps the spatial size for a stride-1 (dilated) kernel."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    pad_h = ((kernel_size[0] - 1) * dilation[0]) // 2
    pad_w = ((kernel_size[1] - 1) * dilation[1]) // 2
    return (pad_h, pad_w)


def create_activation_function(activation: Optional[Union[str, bool]]) -> nn.Module:
    """Return a ``torch.nn`` activation by case-insensitive name.

    ``None``, ``False``, ``"false"`` and ``"none"`` give ``nn.Identity``;
    ``True`` gives the default ``SiLU``.
    """
    if not activation or str(activation).lower() in ("false", "none"):
        return nn.Identity()
    if activation is True:
        activation = "SiLU"

    activation_map = {
        name.lower(): obj
        for name, obj in nn.modules.activation.__dict__.items()
        if isinstance(obj, type) and issubclass(obj, nn.Module)
    }
    name = str(activation).lower()
    if name in activation_map:
        try:
            return activation_map[name](inplace=True)
        except TypeError:  # activations without an in-place variant (GELU, ...)
            return activation_map[name]()
    raise ValueError(f"Activation function '{activation}' is not found in torch.nn")


def round_up(x: Union[int, Tensor], div: int = 1) -> Union[int, Tensor]:
    """Round ``x`` up to the nearest multiple of ``div``."""
    return x + (-x % div)


# =============================================================================
# Building blocks (MultimediaTechLab/YOLO yolo/model/module.py)
# =============================================================================


class Conv(nn.Module):
    """Convolution, batch normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        *,
        activation: Optional[Union[str, bool]] = "SiLU",
        **kwargs,
    ):
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-2)
        self.act = create_activation_function(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Pool(nn.Module):
    """Max or average pooling with automatic padding."""

    def __init__(self, method: str = "max", kernel_size: _size_2_t = 2, **kwargs):
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        pool_classes = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}
        self.pool = pool_classes[method.lower()](kernel_size=kernel_size, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(x)


class RepConv(nn.Module):
    """Parallel kxk and 1x1 convolutions (no activation) summed, then activated."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        *,
        activation: Optional[Union[str, bool]] = "SiLU",
        **kwargs,
    ):
        super().__init__()
        self.act = create_activation_function(activation)
        self.conv1 = Conv(in_channels, out_channels, kernel_size, activation=False, **kwargs)
        self.conv2 = Conv(in_channels, out_channels, 1, activation=False, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv1(x) + self.conv2(x))


class Bottleneck(nn.Module):
    """RepConv followed by a Conv, with an optional residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: Tuple[int, int] = (3, 3),
        residual: bool = True,
        expand: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        neck_channels = int(out_channels * expand)
        self.conv1 = RepConv(in_channels, neck_channels, kernel_size[0], **kwargs)
        self.conv2 = Conv(neck_channels, out_channels, kernel_size[1], **kwargs)
        self.residual = residual

        if residual and (in_channels != out_channels):
            self.residual = False
            logger.warning(
                "Residual connection disabled: in_channels (%d) != out_channels (%d)",
                in_channels,
                out_channels,
            )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.residual else y


class RepNCSP(nn.Module):
    """CSP block: one half through bottlenecks, the other a plain Conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        csp_expand: float = 0.5,
        repeat_num: int = 1,
        neck_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        neck_channels = int(out_channels * csp_expand)
        self.conv1 = Conv(in_channels, neck_channels, kernel_size, **kwargs)
        self.conv2 = Conv(in_channels, neck_channels, kernel_size, **kwargs)
        self.conv3 = Conv(2 * neck_channels, out_channels, kernel_size, **kwargs)

        neck_args = dict(neck_args or {})
        self.bottleneck = nn.Sequential(
            *[Bottleneck(neck_channels, neck_channels, **neck_args) for _ in range(repeat_num)]
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.bottleneck(self.conv1(x))
        x2 = self.conv2(x)
        return self.conv3(torch.cat((x1, x2), dim=1))


class ELAN(nn.Module):
    """ELAN block (first stage of yolo9-t/s)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        if process_channels is None:
            process_channels = part_channels // 2

        self.conv1 = Conv(in_channels, part_channels, 1, **kwargs)
        self.conv2 = Conv(part_channels // 2, process_channels, 3, padding=1, **kwargs)
        self.conv3 = Conv(process_channels, process_channels, 3, padding=1, **kwargs)
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, 1, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return self.conv4(torch.cat([x1, x2, x3, x4], dim=1))


class RepNCSPELAN(nn.Module):
    """ELAN block whose processing branches are RepNCSP + Conv."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        part_channels: int,
        *,
        process_channels: Optional[int] = None,
        csp_args: Optional[Dict[str, Any]] = None,
        csp_neck_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        if process_channels is None:
            process_channels = part_channels // 2
        csp_args = dict(csp_args or {})
        csp_neck_args = dict(csp_neck_args or {})

        self.conv1 = Conv(in_channels, part_channels, 1, **kwargs)
        self.conv2 = nn.Sequential(
            RepNCSP(part_channels // 2, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, 3, padding=1, **kwargs),
        )
        self.conv3 = nn.Sequential(
            RepNCSP(process_channels, process_channels, neck_args=csp_neck_args, **csp_args),
            Conv(process_channels, process_channels, 3, padding=1, **kwargs),
        )
        self.conv4 = Conv(part_channels + 2 * process_channels, out_channels, 1, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = self.conv1(x).chunk(2, 1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        return self.conv4(torch.cat([x1, x2, x3, x4], dim=1))


class AConv(nn.Module):
    """Downsampling: 2x2 average pool (stride 1) then a stride-2 3x3 Conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_layer = {"kernel_size": 3, "stride": 2}
        self.avg_pool = Pool("avg", kernel_size=2, stride=1)
        self.conv = Conv(in_channels, out_channels, **mid_layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x = self.conv(x)
        return x


class ADown(nn.Module):
    """Downsampling: average pool, then half strided Conv, half max pool + 1x1."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        half_in_channels = in_channels // 2
        half_out_channels = out_channels // 2
        mid_layer = {"kernel_size": 3, "stride": 2}
        self.avg_pool = Pool("avg", kernel_size=2, stride=1)
        self.conv1 = Conv(half_in_channels, half_out_channels, **mid_layer)
        self.max_pool = Pool("max", **mid_layer)
        self.conv2 = Conv(half_in_channels, half_out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv1(x1)
        x2 = self.max_pool(x2)
        x2 = self.conv2(x2)
        return torch.cat((x1, x2), dim=1)


class SPPELAN(nn.Module):
    """Spatial pyramid pooling: 1x1 Conv, three chained 5x5 max pools, 1x1 Conv."""

    def __init__(self, in_channels: int, out_channels: int, neck_channels: Optional[int] = None):
        super().__init__()
        neck_channels = neck_channels or out_channels // 2

        self.conv1 = Conv(in_channels, neck_channels, kernel_size=1)
        self.pools = nn.ModuleList([Pool("max", 5, stride=1) for _ in range(3)])
        self.conv5 = Conv(4 * neck_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        features = [self.conv1(x)]
        for pool in self.pools:
            features.append(pool(features[-1]))
        return self.conv5(torch.cat(features, dim=1))


class CBLinear(nn.Module):
    """1x1 convolution (with bias, no norm) whose output is split into several maps.

    ``out_channels`` lists the width of each returned map; the PGI auxiliary
    branch of yolo9-m/c uses one map per pyramid level the feature is fused
    into.
    """

    def __init__(
        self, in_channels: int, out_channels: Sequence[int], kernel_size: int = 1, **kwargs
    ):
        super().__init__()
        kwargs.setdefault("padding", auto_pad(kernel_size, **kwargs))
        self.conv = nn.Conv2d(in_channels, sum(out_channels), kernel_size, **kwargs)
        self.out_channels = list(out_channels)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = self.conv(x)
        return x.split(self.out_channels, dim=1)


class CBFuse(nn.Module):
    """Sum of the last input and one resized map picked from each other input.

    ``x_list`` holds :class:`CBLinear` outputs followed by the target map;
    ``index[i]`` picks the map of ``x_list[i]``, which is interpolated to the
    target's spatial size before the sum. The module has no parameters.
    """

    def __init__(self, index: Sequence[int], mode: str = "nearest"):
        super().__init__()
        self.idx = list(index)
        self.mode = mode

    def forward(self, x_list: Sequence[Any]) -> Tensor:
        target = x_list[-1]
        target_size = target.shape[2:]

        res = [
            F.interpolate(x[pick_id], size=target_size, mode=self.mode)
            for pick_id, x in zip(self.idx, x_list)
        ]
        return torch.stack(res + [target]).sum(dim=0)


# =============================================================================
# Detection head
# =============================================================================


class Anchor2Vec(nn.Module):
    """Expected box-side distance from per-side bin logits (MIT ``Anchor2Vec``).

    Input ``(B, 4 * reg_max, *spatial)`` with side-major, bin-minor channels;
    output ``(B, 4, *spatial)`` distances in grid units: a softmax over the
    ``reg_max`` bins of each side, then the expectation over bin indices.
    MIT keeps the bin indices in a frozen ``Conv3d`` weight; here they are a
    non-persistent buffer, so the module adds nothing to state dicts.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = int(reg_max)
        self.register_buffer(
            "bins",
            torch.arange(self.reg_max, dtype=torch.float32).view(1, 1, self.reg_max, 1),
            persistent=False,
        )

    def forward(self, anchor_x: Tensor) -> Tensor:
        batch = anchor_x.shape[0]
        spatial = anchor_x.shape[2:]
        logits = anchor_x.reshape(batch, 4, self.reg_max, -1)
        dist = (logits.softmax(dim=2) * self.bins).sum(dim=2)
        return dist.reshape(batch, 4, *spatial)


def default_class_neck(first_channels: int, num_classes: int) -> int:
    """Class-tower width LibreYOLO uses for freshly built YOLO9-family heads.

    ``max(P-first channels, min(num_classes, 128))``: every width LibreYOLO has
    produced for fresh builds and published checkpoints so far (e.g. 80 for
    yolo9-t on COCO, 64 for yolo9-t with 40 classes), with the 128 cap of
    MultimediaTechLab/YOLO. :class:`YOLO9Head` on its own defaults to the
    MultimediaTechLab formula (``min(2 * num_classes, 128)``); the LibreYOLO
    assemblies pass this width explicitly so existing training runs and
    checkpoints keep their architecture. A loaded checkpoint's width always
    wins over either rule.
    """
    return max(int(first_channels), min(int(num_classes), 128))


class YOLO9Head(nn.Module):
    """Anchor-free YOLOv9 detection head over several pyramid levels.

    Each level has a box tower (``anchor_convs``) predicting ``4 * reg_max``
    distance-bin logits and a class tower (``class_convs``) predicting
    ``num_classes`` logits, as in MultimediaTechLab/YOLO ``Detection``; the
    towers of all levels are held in two module lists.

    Forward contract:
        * ``targets`` given: the training loss dict (needs ``img_size=(W, H)``).
        * training, no targets: list of raw maps ``(B, 4 * reg_max + nc, H, W)``
          per level, box channels first.
        * eval: ``(decoded, raw)`` where ``decoded`` is ``(B, 4 + nc, N)``:
          xyxy boxes in input pixels, then sigmoid class scores.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        *,
        reg_max: int = 16,
        strides: Sequence[int] = (8, 16, 32),
        use_group: bool = True,
        class_neck: Optional[int] = None,
    ):
        super().__init__()
        in_channels = [int(c) for c in in_channels]
        if len(in_channels) != len(strides):
            raise ValueError(
                f"YOLO9Head needs one stride per input level, got "
                f"{len(in_channels)} levels and strides {tuple(strides)}"
            )

        self.in_channels = tuple(in_channels)
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.strides = tuple(int(s) for s in strides)
        self.export = False
        self.groups = 4 if use_group else 1

        first_neck = in_channels[0]
        self.anchor_neck = max(
            round_up(first_neck // 4, self.groups), 4 * self.reg_max, self.reg_max
        )
        if class_neck is None:
            class_neck = max(first_neck, min(self.num_classes * 2, 128))
        self.class_neck = int(class_neck)

        self.anchor_convs = self.build_anchor_convs(in_channels)
        self.class_convs = self.build_class_convs(
            in_channels, self.class_neck, self.num_classes
        )
        self.anchor2vec = Anchor2Vec(reg_max=self.reg_max)
        self.init_bias()

        self._loss_fn = None
        # Eager-mode anchor grid cache: ((level sizes, device, dtype), anchors, strides).
        self._grid_cache = None
        # Export canvas pinned by freeze_anchor_grid: (level sizes, anchors, strides).
        self._frozen_grid = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def build_anchor_convs(self, in_channels: Sequence[int]) -> nn.ModuleList:
        """Box towers: Conv 3x3, grouped Conv 3x3, grouped 1x1 to ``4 * reg_max``."""
        neck = self.anchor_neck
        return nn.ModuleList(
            nn.Sequential(
                Conv(int(c), neck, 3),
                Conv(neck, neck, 3, groups=self.groups),
                nn.Conv2d(neck, 4 * self.reg_max, 1, groups=self.groups),
            )
            for c in in_channels
        )

    def build_class_convs(
        self, in_channels: Sequence[int], class_neck: int, num_classes: int
    ) -> nn.ModuleList:
        """Class towers: Conv 3x3, Conv 3x3, 1x1 to ``num_classes``."""
        return nn.ModuleList(
            nn.Sequential(
                Conv(int(c), int(class_neck), 3),
                Conv(int(class_neck), int(class_neck), 3),
                nn.Conv2d(int(class_neck), int(num_classes), 1),
            )
            for c in in_channels
        )

    @torch.no_grad()
    def init_bias(self) -> None:
        """MultimediaTechLab bias init: box logits 1.0, class logits -10."""
        for tower in self.anchor_convs:
            tower[-1].bias.fill_(1.0)
        for tower in self.class_convs:
            tower[-1].bias.fill_(-10.0)

    def set_num_classes(self, num_classes: int) -> None:
        """Swap only the final 1x1 class convs for a new class count.

        Hidden tower widths are kept (transfer learning); the new class
        biases are re-initialized and the cached loss is dropped.
        """
        num_classes = int(num_classes)
        for tower in self.class_convs:
            old = tower[-1]
            new = nn.Conv2d(old.in_channels, num_classes, 1)
            tower[-1] = new.to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            for tower in self.class_convs:
                tower[-1].bias.fill_(-10.0)
        self.num_classes = num_classes
        self._loss_fn = None

    # ------------------------------------------------------------------
    # Forward pieces
    # ------------------------------------------------------------------

    def branch_outputs(
        self,
        features: Sequence[Tensor],
        anchor_convs: Optional[nn.ModuleList] = None,
        class_convs: Optional[nn.ModuleList] = None,
    ) -> List[Tensor]:
        """Per level ``cat((box logits, class logits), 1)``: ``(B, 4 * reg_max + nc, H, W)``."""
        anchor_convs = self.anchor_convs if anchor_convs is None else anchor_convs
        class_convs = self.class_convs if class_convs is None else class_convs
        if not (len(features) == len(anchor_convs) == len(class_convs)):
            raise ValueError(
                f"YOLO9Head got {len(features)} feature levels for "
                f"{len(anchor_convs)} box / {len(class_convs)} class towers"
            )
        return [
            torch.cat((anchor_conv(feat), class_conv(feat)), 1)
            for feat, anchor_conv, class_conv in zip(features, anchor_convs, class_convs)
        ]

    def decode(self, raw: Sequence[Tensor]) -> Tensor:
        """Raw level maps -> ``(B, 4 + nc, N)`` xyxy pixel boxes then sigmoid scores.

        Anchors are the level cell centres ``k * stride + stride // 2`` and the
        expected distances are scaled by each anchor's stride, as in the
        MultimediaTechLab ``generate_anchors`` / ``Vec2Box`` pair.
        """
        box_channels = 4 * self.reg_max
        flat = torch.cat([level.flatten(2) for level in raw], dim=2)
        box_logits = flat[:, :box_channels]
        class_logits = flat[:, box_channels:]

        distances = self.anchor2vec(box_logits)  # (B, 4, N) in grid units
        anchors, strides = self._anchor_grid_for(raw, distances)
        lt, rb = (distances * strides).chunk(2, dim=1)
        boxes = torch.cat((anchors - lt, anchors + rb), dim=1)
        return torch.cat((boxes, class_logits.sigmoid()), dim=1)

    # ------------------------------------------------------------------
    # Anchor grid
    # ------------------------------------------------------------------

    def _grid_tensors(
        self, level_sizes: Sequence[Tuple[Any, Any]], device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Integer ``(2, N)`` anchor centres in pixels and ``(1, N)`` strides."""
        anchors, strides = [], []
        for (height, width), stride in zip(level_sizes, self.strides):
            xs = torch.arange(width, device=device) * stride + stride // 2
            ys = torch.arange(height, device=device) * stride + stride // 2
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            anchors.append(torch.stack((grid_x, grid_y)).flatten(1))
            strides.append(torch.full_like(grid_x, stride).flatten())
        return torch.cat(anchors, dim=1), torch.cat(strides).unsqueeze(0)

    def _anchor_grid_for(self, raw: Sequence[Tensor], like: Tensor) -> Tuple[Tensor, Tensor]:
        """Anchor centres and strides for ``raw``, cast to ``like``'s dtype."""
        level_sizes = tuple((level.shape[2], level.shape[3]) for level in raw)
        if self._frozen_grid is not None:
            frozen_sizes, anchors, strides = self._frozen_grid
            live_sizes = tuple((int(h), int(w)) for h, w in level_sizes)
            if live_sizes != frozen_sizes:
                raise ValueError(
                    f"YOLO9Head anchor grid is frozen for level sizes {frozen_sizes}, "
                    f"got {live_sizes}; call unfreeze_anchor_grid() first"
                )
            if anchors.dtype != like.dtype or anchors.device != like.device:
                anchors = anchors.to(device=like.device, dtype=like.dtype)
                strides = strides.to(device=like.device, dtype=like.dtype)
            return anchors, strides

        if self.export or torch.jit.is_tracing() or _is_compiling():
            # Rebuild from live shapes every call so traced graphs follow the
            # input size instead of baking a cached constant.
            anchors, strides = self._grid_tensors(level_sizes, like.device)
            return anchors.to(like.dtype), strides.to(like.dtype)

        key = (tuple((int(h), int(w)) for h, w in level_sizes), like.device, like.dtype)
        cache = self._grid_cache
        if cache is None or cache[0] != key:
            anchors, strides = self._grid_tensors(key[0], like.device)
            cache = (key, anchors.to(like.dtype), strides.to(like.dtype))
            self._grid_cache = cache
        return cache[1], cache[2]

    def freeze_anchor_grid(self, input_hw: Tuple[int, int]) -> None:
        """Pin the anchor grid as constants for a fixed ``(H, W)`` export canvas.

        Level sizes are ``input // stride``. Used by fixed-canvas exporters
        (CoreML, Core AI) whose tracers reject shape-derived grids.
        """
        height, width = (int(v) for v in input_hw)
        level_sizes = tuple((height // s, width // s) for s in self.strides)
        anchors, strides = self._grid_tensors(level_sizes, self._device())
        self._frozen_grid = (level_sizes, anchors.float(), strides.float())

    def unfreeze_anchor_grid(self) -> None:
        """Return to shape-derived anchor grids."""
        self._frozen_grid = None

    def _device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _get_loss_fn(self, device: Union[str, torch.device]):
        """Lazily build (or rebuild on a device change) this head's loss."""
        device = torch.device(device)
        if self._loss_fn is None or self._loss_fn.device != device:
            from .loss import YOLO9Loss

            self._loss_fn = YOLO9Loss(
                num_classes=self.num_classes,
                reg_max=self.reg_max,
                strides=list(self.strides),
                image_size=None,
                device=device,
            )
        return self._loss_fn

    def forward(
        self,
        features: Sequence[Tensor],
        targets: Optional[Tensor] = None,
        img_size: Optional[Tuple[int, int]] = None,
    ):
        raw = self.branch_outputs(features)
        if targets is not None:
            if img_size is None:
                raise ValueError("YOLO9Head needs img_size=(W, H) when targets are given")
            loss_fn = self._get_loss_fn(raw[0].device)
            loss_fn.update_anchors(list(img_size))
            return loss_fn(raw, targets)
        if self.training:
            return raw
        return self.decode(raw), raw


def _is_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    return bool(is_compiling()) if callable(is_compiling) else False


# =============================================================================
# Model assembly (LibreYOLO)
# =============================================================================

# YOLOv9 configurations - exact channel dimensions from official YOLO configs
# Each variant has unique, non-linear channel structures
YOLO9_CONFIGS = {
    "t": {  # Tiny
        # Backbone: Conv(16) -> Conv(32) -> ELAN(32) -> [AConv -> RepNCSPELAN] x3
        "conv0_out": 16,
        "conv1_out": 32,
        "first_block": "elan",  # ELAN for t/s, RepNCSPELAN for m/c
        "first_block_out": 32,
        "down_type": "aconv",  # AConv for t/s/m, ADown for c
        "stages": [  # (down_out, elan_out, elan_part) for stages 2, 3, 4
            (64, 64, 64),  # B3: AConv->64, RepNCSPELAN->64
            (96, 96, 96),  # B4: AConv->96, RepNCSPELAN->96
            (128, 128, 128),  # B5: AConv->128, RepNCSPELAN->128
        ],
        "spp_out": 128,
        "repeat_num": 3,  # RepNCSPELAN repeat for t/s
        # Neck
        "neck_elan_up1": (96, 96),  # N4: out=96, part=96
        "neck_elan_up2": (64, 64),  # P3: out=64, part=64
        "neck_down1_out": 48,  # AConv after P3
        "neck_elan_down1": (96, 96),  # P4: out=96, part=96
        "neck_down2_out": 64,  # AConv after P4
        "neck_elan_down2": (128, 128),  # P5: out=128, part=128
        # Detection head channels
        "head_channels": (64, 96, 128),  # P3, P4, P5
    },
    "s": {  # Small
        # Backbone: Conv(32) -> Conv(64) -> ELAN(64) -> [AConv -> RepNCSPELAN] x3
        "conv0_out": 32,
        "conv1_out": 64,
        "first_block": "elan",
        "first_block_out": 64,
        "down_type": "aconv",
        "stages": [
            (128, 128, 128),  # B3
            (192, 192, 192),  # B4
            (256, 256, 256),  # B5
        ],
        "spp_out": 256,
        "repeat_num": 3,
        # Neck
        "neck_elan_up1": (192, 192),
        "neck_elan_up2": (128, 128),
        "neck_down1_out": 96,
        "neck_elan_down1": (192, 192),
        "neck_down2_out": 128,
        "neck_elan_down2": (256, 256),
        # Detection
        "head_channels": (128, 192, 256),
    },
    "m": {  # Medium
        # Backbone: Conv(32) -> Conv(64) -> RepNCSPELAN(128) -> [AConv -> RepNCSPELAN] x3
        "conv0_out": 32,
        "conv1_out": 64,
        "first_block": "repncspelan",
        "first_block_out": 128,
        "first_block_part": 128,
        "down_type": "aconv",
        "stages": [
            (240, 240, 240),  # B3
            (360, 360, 360),  # B4
            (480, 480, 480),  # B5
        ],
        "spp_out": 480,
        "repeat_num": 1,  # Default repeat for m/c
        # Neck
        "neck_elan_up1": (360, 360),
        "neck_elan_up2": (240, 240),
        "neck_down1_out": 184,
        "neck_elan_down1": (360, 360),
        "neck_down2_out": 240,
        "neck_elan_down2": (480, 480),
        # Detection
        "head_channels": (240, 360, 480),
    },
    "c": {  # Compact (largest)
        # Backbone: Conv(64) -> Conv(128) -> RepNCSPELAN(256) -> [ADown -> RepNCSPELAN] x3
        "conv0_out": 64,
        "conv1_out": 128,
        "first_block": "repncspelan",
        "first_block_out": 256,
        "first_block_part": 128,  # part_channels for first RepNCSPELAN
        "down_type": "adown",
        "stages": [
            (256, 512, 256),  # B3: ADown->256, RepNCSPELAN->512, part=256
            (512, 512, 512),  # B4: ADown->512, RepNCSPELAN->512, part=512
            (512, 512, 512),  # B5: ADown->512, RepNCSPELAN->512, part=512
        ],
        "spp_out": 512,
        "repeat_num": 1,
        # Neck
        "neck_elan_up1": (512, 512),
        "neck_elan_up2": (256, 256),
        "neck_down1_out": 256,
        "neck_elan_down1": (512, 512),
        "neck_down2_out": 512,
        "neck_elan_down2": (512, 512),
        # Detection
        "head_channels": (256, 512, 512),
    },
}


def _elan_stage(in_channels: int, out_channels: int, part_channels: int, repeat_num: int):
    """RepNCSPELAN stage with ``repeat_num`` bottlenecks per CSP branch."""
    return RepNCSPELAN(
        in_channels,
        out_channels,
        part_channels,
        csp_args={"repeat_num": repeat_num},
    )


class Backbone9(nn.Module):
    """YOLOv9 Backbone.

    Supports all variants with their specific architectures:
    - yolo9-t/s: Conv -> Conv -> ELAN -> [AConv -> RepNCSPELAN] x3 -> SPPELAN
    - yolo9-m/c: Conv -> Conv -> RepNCSPELAN -> [AConv/ADown -> RepNCSPELAN] x3 -> SPPELAN
    """

    def __init__(self, config="c"):
        super().__init__()

        cfg = YOLO9_CONFIGS[config]
        self.config = config

        # Stem
        self.conv0 = Conv(3, cfg["conv0_out"], 3, stride=2)
        self.conv1 = Conv(cfg["conv0_out"], cfg["conv1_out"], 3, stride=2)

        # First block (ELAN for t/s, RepNCSPELAN for m/c)
        c_in = cfg["conv1_out"]
        c_out = cfg["first_block_out"]
        if cfg["first_block"] == "elan":
            # t/s: part_channels = out_channels
            self.elan1 = ELAN(c_in, c_out, c_out)
        else:
            part = cfg.get("first_block_part", c_out)
            self.elan1 = _elan_stage(c_in, c_out, part, cfg["repeat_num"])

        # Determine downsampling block type
        DownBlock = ADown if cfg["down_type"] == "adown" else AConv
        n = cfg["repeat_num"]

        # Stage 2 (B3) - stage = (down_out, elan_out, part_channels)
        stage = cfg["stages"][0]
        self.down2 = DownBlock(cfg["first_block_out"], stage[0])
        self.elan2 = _elan_stage(stage[0], stage[1], stage[2], n)

        # Stage 3 (B4)
        stage = cfg["stages"][1]
        self.down3 = DownBlock(cfg["stages"][0][1], stage[0])
        self.elan3 = _elan_stage(stage[0], stage[1], stage[2], n)

        # Stage 4 (B5)
        stage = cfg["stages"][2]
        self.down4 = DownBlock(cfg["stages"][1][1], stage[0])
        self.elan4 = _elan_stage(stage[0], stage[1], stage[2], n)

        # SPP
        self.spp = SPPELAN(cfg["stages"][2][1], cfg["spp_out"])

    def forward(self, x, return_b5=False):
        # Stem
        x = self.conv0(x)
        x = self.conv1(x)

        # First block
        x = self.elan1(x)

        # Stage 2 - B3/P3
        x = self.down2(x)
        p3 = self.elan2(x)

        # Stage 3 - B4/P4
        x = self.down3(p3)
        p4 = self.elan3(x)

        # Stage 4 - B5 (pre-SPP) then SPP -> P5. The PGI aux branch needs the
        # pre-SPP B5; it is returned on request instead of being stored on the
        # module, which would keep a non-leaf tensor alive and break deepcopy.
        x = self.down4(p4)
        b5 = self.elan4(x)
        p5 = self.spp(b5)

        if return_b5:
            return p3, p4, p5, b5
        return p3, p4, p5


class Neck9(nn.Module):
    """YOLOv9 PANet Neck.

    Architecture (varies by config):
    Top-down path:
    - UpSample + Concat(B4) -> RepNCSPELAN (N4)
    - UpSample + Concat(B3) -> RepNCSPELAN (P3)
    Bottom-up path:
    - AConv/ADown + Concat(N4) -> RepNCSPELAN (P4)
    - AConv/ADown + Concat(SPP) -> RepNCSPELAN (P5)
    """

    def __init__(self, config="c"):
        super().__init__()

        cfg = YOLO9_CONFIGS[config]
        self.config = config
        n = cfg["repeat_num"]

        # Backbone output channels used by the concatenations
        b3_ch = cfg["stages"][0][1]  # B3 output channels
        b4_ch = cfg["stages"][1][1]  # B4 output channels
        spp_ch = cfg["spp_out"]  # SPP/P5 output channels

        # Top-down path: Concat(SPP_up, B4) -> N4
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        up1_out, up1_part = cfg["neck_elan_up1"]
        self.elan_up1 = _elan_stage(spp_ch + b4_ch, up1_out, up1_part, n)

        # Concat(N4_up, B3) -> P3
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        up2_out, up2_part = cfg["neck_elan_up2"]
        self.elan_up2 = _elan_stage(up1_out + b3_ch, up2_out, up2_part, n)

        # Bottom-up path
        DownBlock = ADown if cfg["down_type"] == "adown" else AConv

        # P3 -> down -> Concat(N4) -> P4
        self.down1 = DownBlock(up2_out, cfg["neck_down1_out"])
        down1_out, down1_part = cfg["neck_elan_down1"]
        self.elan_down1 = _elan_stage(
            cfg["neck_down1_out"] + up1_out, down1_out, down1_part, n
        )

        # P4 -> down -> Concat(SPP) -> P5
        self.down2 = DownBlock(down1_out, cfg["neck_down2_out"])
        down2_out, down2_part = cfg["neck_elan_down2"]
        self.elan_down2 = _elan_stage(
            cfg["neck_down2_out"] + spp_ch, down2_out, down2_part, n
        )

    def forward(self, p3, p4, p5):
        # Top-down path
        x = self.up1(p5)
        x = torch.cat([x, p4], 1)
        n4 = self.elan_up1(x)

        x = self.up2(n4)
        x = torch.cat([x, p3], 1)
        out_p3 = self.elan_up2(x)

        # Bottom-up path
        x = self.down1(out_p3)
        x = torch.cat([x, n4], 1)
        out_p4 = self.elan_down1(x)

        x = self.down2(out_p4)
        x = torch.cat([x, p5], 1)
        out_p5 = self.elan_down2(x)

        return out_p3, out_p4, out_p5


# PGI auxiliary branch kinds. ``neck`` is :class:`AuxNeck`, ``backbone`` is
# :class:`AuxBackbone`.
AUX_BRANCH_NECK = "neck"
AUX_BRANCH_BACKBONE = "backbone"


def supported_aux_branches(config: str) -> Tuple[str, ...]:
    """Auxiliary branch kinds a size can build, the default first.

    yolo9-t/s have the top-down branch only. yolo9-m/c default to the
    second-backbone branch of their upstream configs and can still build the
    top-down one, which LibreYOLO 1.6.0 trained them with.
    """
    if YOLO9_CONFIGS[config]["first_block"] == "elan":
        return (AUX_BRANCH_NECK,)
    return (AUX_BRANCH_BACKBONE, AUX_BRANCH_NECK)


def aux_branch_from_state_dict(state_dict: Dict[str, Any]) -> Optional[str]:
    """Auxiliary branch kind stored in a LibreYOLO state dict, from its ``aux.*`` keys.

    ``None`` when the dict has no ``aux.*`` tensors (``aux_head.*`` alone does
    not identify the branch).
    """
    for key in state_dict:
        key = str(key)
        if key.startswith("aux.cblinear"):
            return AUX_BRANCH_BACKBONE
        if key.startswith(("aux.spp.", "aux.elan_a")):
            return AUX_BRANCH_NECK
    return None


class AuxNeck(nn.Module):
    """PGI auxiliary top-down branch used only during training.

    Mirrors MultimediaTechLab/YOLO ``auxiliary`` in ``v9-t.yaml`` and
    ``v9-s.yaml``: SPPELAN on pre-SPP B5, then top-down concat with B4/B3.
    It is the auxiliary branch of yolo9-t and yolo9-s. yolo9-m and yolo9-c
    use :class:`AuxBackbone`, the branch of their upstream configs; LibreYOLO
    1.6.0 built this class for them too, so it is still built for m/c when a
    checkpoint carries its ``aux.spp`` / ``aux.elan_a4`` / ``aux.elan_a3``
    tensors. Inference never calls this module, so old single-head
    checkpoints keep their exact graph.
    """

    def __init__(self, config="c"):
        super().__init__()
        cfg = YOLO9_CONFIGS[config]
        n = cfg["repeat_num"]
        b3_ch = cfg["stages"][0][1]
        b4_ch = cfg["stages"][1][1]
        b5_ch = cfg["stages"][2][1]
        spp_out = cfg["spp_out"]

        self.spp = SPPELAN(b5_ch, spp_out)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        a4_out, a4_part = cfg["neck_elan_up1"]
        self.elan_a4 = _elan_stage(spp_out + b4_ch, a4_out, a4_part, n)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        a3_out, a3_part = cfg["neck_elan_up2"]
        self.elan_a3 = _elan_stage(a4_out + b3_ch, a3_out, a3_part, n)

    def forward(self, p3, p4, b5):
        a5 = self.spp(b5)
        x = self.up1(a5)
        x = torch.cat([x, p4], 1)
        a4 = self.elan_a4(x)
        x = self.up2(a4)
        x = torch.cat([x, p3], 1)
        a3 = self.elan_a3(x)
        return a3, a4, a5


class AuxBackbone(nn.Module):
    """PGI auxiliary second backbone of yolo9-m/c, used only during training.

    Provenance: reproduces the ``auxiliary`` section of
    ``yolo/config/model/v9-m.yaml`` and ``v9-c.yaml`` in MultimediaTechLab/YOLO
    (https://github.com/MultimediaTechLab/YOLO, commit c4cb5f6f, MIT License),
    built from the ``CBLinear`` / ``CBFuse`` blocks of ``yolo/model/module.py``.
    Module order matches the upstream layer order:

    * ``cblinear3`` / ``cblinear4`` / ``cblinear5``: CBLinear on the main
      backbone's B3 / B4 / B5 (upstream tags R3 / R4 / R5), giving one map for
      every auxiliary level at or below the source level.
    * ``conv0``, ``conv1``, ``elan1``: a second stem and first block, fed
      with the input image.
    * ``down2`` -> ``cbfuse3`` -> ``elan2`` (A3), ``down3`` -> ``cbfuse4`` ->
      ``elan3`` (A4), ``down4`` -> ``cbfuse5`` -> ``elan4`` (A5): the main
      backbone's stages, each downsampled map summed with the CBLinear maps
      of its level before the RepNCSPELAN.

    The auxiliary head reads ``(A3, A4, A5)``, whose widths are
    :attr:`out_channels` (the backbone stage widths, not the main head's).
    """

    def __init__(self, config="c"):
        super().__init__()
        cfg = YOLO9_CONFIGS[config]
        if cfg["first_block"] != "repncspelan":
            raise ValueError(
                f"AuxBackbone is the auxiliary branch of yolo9-m/c, got config {config!r}"
            )
        n = cfg["repeat_num"]
        stages = cfg["stages"]
        fuse_channels = [stage[0] for stage in stages]  # widths after each downsample
        stage_out = [stage[1] for stage in stages]  # B3, B4, B5 widths
        DownBlock = ADown if cfg["down_type"] == "adown" else AConv

        self.cblinear3 = CBLinear(stage_out[0], fuse_channels[:1])
        self.cblinear4 = CBLinear(stage_out[1], fuse_channels[:2])
        self.cblinear5 = CBLinear(stage_out[2], fuse_channels[:3])

        self.conv0 = Conv(3, cfg["conv0_out"], 3, stride=2)
        self.conv1 = Conv(cfg["conv0_out"], cfg["conv1_out"], 3, stride=2)
        self.elan1 = _elan_stage(
            cfg["conv1_out"],
            cfg["first_block_out"],
            cfg.get("first_block_part", cfg["first_block_out"]),
            n,
        )

        self.down2 = DownBlock(cfg["first_block_out"], stages[0][0])
        self.cbfuse3 = CBFuse([0, 0, 0])
        self.elan2 = _elan_stage(stages[0][0], stages[0][1], stages[0][2], n)

        self.down3 = DownBlock(stages[0][1], stages[1][0])
        self.cbfuse4 = CBFuse([1, 1])
        self.elan3 = _elan_stage(stages[1][0], stages[1][1], stages[1][2], n)

        self.down4 = DownBlock(stages[1][1], stages[2][0])
        self.cbfuse5 = CBFuse([2])
        self.elan4 = _elan_stage(stages[2][0], stages[2][1], stages[2][2], n)

        self.out_channels = tuple(stage_out)

    def forward(self, x, b3, b4, b5):
        """``x`` is the input image; ``b3``/``b4``/``b5`` the main backbone stages (B5 pre-SPP)."""
        r3 = self.cblinear3(b3)
        r4 = self.cblinear4(b4)
        r5 = self.cblinear5(b5)

        x = self.elan1(self.conv1(self.conv0(x)))
        a3 = self.elan2(self.cbfuse3([r3, r4, r5, self.down2(x)]))
        a4 = self.elan3(self.cbfuse4([r4, r5, self.down3(a3)]))
        a5 = self.elan4(self.cbfuse5([r5, self.down4(a4)]))
        return a3, a4, a5


class LibreYOLO9Model(nn.Module):
    """
    Complete LibreYOLO9 model.

    Supports yolo9-t, yolo9-s, yolo9-m, and yolo9-c variants with their specific architectures.
    """

    def __init__(
        self,
        config="c",
        reg_max=16,
        nb_classes=80,
        img_size=640,
    ):
        """
        Initialize YOLOv9 model.

        Args:
            config: Model size ('t', 's', 'm', 'c')
            reg_max: Regression max value for DFL
            nb_classes: Number of classes
            img_size: Input image size
        """
        super().__init__()

        if config not in YOLO9_CONFIGS:
            raise ValueError(
                f"Invalid config: {config}. Must be one of: {list(YOLO9_CONFIGS.keys())}"
            )

        self.config = config
        self.nc = nb_classes
        self.reg_max = reg_max
        self.img_size = img_size

        cfg = YOLO9_CONFIGS[config]

        self.backbone = Backbone9(config)
        self.neck = Neck9(config)

        # Detection head - use exact channels from config
        self.head = self._build_head(cfg["head_channels"], nb_classes)
        # Built only when training with PGI. Inference checkpoints stay
        # single-head so a 1.6 upgrade does not change the exported graph.
        self.aux = None
        self.aux_head = None
        self.aux_weight = 0.0

    def _build_head(self, head_channels: Sequence[int], nb_classes: int) -> YOLO9Head:
        return YOLO9Head(
            head_channels,
            nb_classes,
            reg_max=self.reg_max,
            strides=(8, 16, 32),
            class_neck=default_class_neck(head_channels[0], nb_classes),
        )

    @property
    def aux_branch(self) -> Optional[str]:
        """Kind of the attached PGI branch (``"neck"`` / ``"backbone"``), or ``None``."""
        if self.aux is None:
            return None
        return AUX_BRANCH_BACKBONE if isinstance(self.aux, AuxBackbone) else AUX_BRANCH_NECK

    def enable_aux(self, weight: float = 0.25, branch: Optional[str] = None):
        """Attach the PGI auxiliary branch and head if they are not already present.

        ``branch`` is ``None`` for the size's own branch (:class:`AuxNeck` for
        yolo9-t/s, :class:`AuxBackbone` for yolo9-m/c) and keeps whatever is
        already attached. Passing a kind (see :func:`supported_aux_branches`)
        builds that one, replacing an attached branch of the other kind and
        its head; checkpoint loaders use it to match the stored branch.
        """
        self.aux_weight = float(weight)
        if self.aux is not None and branch in (None, self.aux_branch):
            return self
        supported = supported_aux_branches(self.config)
        if branch is None:
            branch = supported[0]
        elif branch not in supported:
            raise ValueError(
                f"yolo9-{self.config} has no {branch!r} auxiliary branch; "
                f"expected one of {supported}"
            )
        cfg = YOLO9_CONFIGS[self.config]
        if branch == AUX_BRANCH_BACKBONE:
            self.aux = AuxBackbone(self.config)
            self.aux_head = self._build_head(self.aux.out_channels, self.nc)
        else:
            self.aux = AuxNeck(self.config)
            self.aux_head = self._build_head(cfg["head_channels"], self.nc)
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        self.aux.to(device)
        self.aux_head.to(device)
        return self

    def disable_aux(self):
        """Drop the PGI branch so its parameters leave the optimizer and DDP."""
        self.aux = None
        self.aux_head = None
        self.aux_weight = 0.0
        return self

    def aux_features(self, x, b3, b4, b5):
        """Auxiliary pyramid ``(A3, A4, A5)`` from the image and the backbone stages.

        ``b5`` is the pre-SPP stage. :class:`AuxBackbone` also reads the
        image; :class:`AuxNeck` only the backbone features.
        """
        if isinstance(self.aux, AuxBackbone):
            return self.aux(x, b3, b4, b5)
        return self.aux(b3, b4, b5)

    def combine_aux_losses(self, main: dict, aux: dict) -> dict:
        """Add the PGI auxiliary losses to the main ones at ``aux_weight``."""
        combined = dict(main)
        for key in ("total_loss", "box_loss", "dfl_loss", "cls_loss", "box", "dfl", "cls"):
            if key in main and key in aux:
                combined[key] = main[key] + self.aux_weight * aux[key]
        return combined

    def forward(self, x, targets=None):
        """
        Forward pass through backbone, neck, and detection head.

        Args:
            x: Input tensor [B, 3, H, W]
            targets: Optional ground truth [B, max_targets, 5] with [class, x1, y1, x2, y2] normalized
                    Only used during training to compute loss.

        Returns:
            Training with targets: Dict with loss values (total_loss, box_loss, dfl_loss, cls_loss)
            Training without targets: Raw predictions (list of tensors)
            Inference: Dict with decoded predictions and features
        """
        # Backbone. The PGI aux branch also needs the pre-SPP B5 feature.
        use_aux = (
            self.training
            and targets is not None
            and self.aux is not None
            and self.aux_weight > 0
        )
        if use_aux:
            p3, p4, p5, b5 = self.backbone(x, return_b5=True)
        else:
            p3, p4, p5 = self.backbone(x)

        # Neck
        n3, n4, n5 = self.neck(p3, p4, p5)

        # Detection head
        if self.training and targets is not None:
            # Pass image size for anchor generation
            img_size = (x.shape[3], x.shape[2])  # (W, H)
            main = self.head([n3, n4, n5], targets=targets, img_size=img_size)
            if not use_aux:
                return main
            a3, a4, a5 = self.aux_features(x, p3, p4, b5)
            aux = self.aux_head([a3, a4, a5], targets=targets, img_size=img_size)
            return self.combine_aux_losses(main, aux)

        # Normal forward (training without targets or inference)
        output = self.head([n3, n4, n5])

        if self.training:
            # Return raw outputs for loss calculation
            return output

        # Inference mode
        y, x_list = output

        # Export mode: return only the prediction tensor for ONNX/TorchScript
        if self.head.export:
            return y

        return {
            "predictions": y,  # (batch, 4+nc, total_anchors)
            "raw_outputs": x_list,
            "x8": {"features": n3},
            "x16": {"features": n4},
            "x32": {"features": n5},
        }


__all__ = [
    "ADown",
    "AConv",
    "AUX_BRANCH_BACKBONE",
    "AUX_BRANCH_NECK",
    "Anchor2Vec",
    "AuxBackbone",
    "AuxNeck",
    "Backbone9",
    "Bottleneck",
    "CBFuse",
    "CBLinear",
    "Conv",
    "ELAN",
    "LibreYOLO9Model",
    "Neck9",
    "Pool",
    "RepConv",
    "RepNCSP",
    "RepNCSPELAN",
    "SPPELAN",
    "YOLO9_CONFIGS",
    "YOLO9Head",
    "aux_branch_from_state_dict",
    "auto_pad",
    "create_activation_function",
    "default_class_neck",
    "round_up",
    "supported_aux_branches",
]

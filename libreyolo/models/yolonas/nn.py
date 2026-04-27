"""Native YOLO-NAS architecture port.

This file keeps the module tree close to SuperGradients so official
``yolo_nas_{s,m,l}_coco.pth`` checkpoints load with minimal or no remapping.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def autopad(kernel, padding=None):
    if padding is None:
        return kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding


def width_multiplier(original: int, factor: float, divisor: int | None = None) -> int:
    scaled = int(original * factor)
    if divisor is None:
        return scaled
    return math.ceil(scaled / divisor) * divisor


class Residual(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        padding: int | Tuple[int, int],
        activation_type: type[nn.Module],
        stride: int | Tuple[int, int] = 1,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        use_normalization: bool = True,
        activation_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        activation_kwargs = activation_kwargs or {}
        self.seq = nn.Sequential()
        self.seq.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
        )
        if use_normalization:
            self.seq.add_module("bn", nn.BatchNorm2d(out_channels))
        if activation_type is not None:
            self.seq.add_module("act", activation_type(**activation_kwargs))

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class Conv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel: int,
        stride: int,
        activation_type: type[nn.Module],
        padding: Optional[int] = None,
        groups: Optional[int] = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel,
            stride,
            autopad(kernel, padding),
            groups=groups or 1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = activation_type()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvBNReLU(ConvBNAct):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        use_normalization: bool = True,
        use_activation: bool = True,
        inplace: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            activation_type=nn.ReLU if use_activation else None,
            activation_kwargs={"inplace": True} if inplace else None,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            use_normalization=use_normalization,
        )


class QARepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation_type: type[nn.Module] = nn.ReLU,
        activation_kwargs: Optional[dict] = None,
        build_residual_branches: bool = True,
        use_residual_connection: bool = True,
        use_alpha: bool = False,
        use_1x1_bias: bool = True,
        use_post_bn: bool = True,
    ):
        super().__init__()
        activation_kwargs = activation_kwargs or {}

        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.use_residual_connection = use_residual_connection
        self.use_alpha = use_alpha
        self.use_1x1_bias = use_1x1_bias
        self.use_post_bn = use_post_bn

        self.nonlinearity = activation_type(**activation_kwargs)
        self.se = nn.Identity()

        self.branch_3x3 = nn.Sequential()
        self.branch_3x3.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                groups=groups,
                bias=False,
                dilation=dilation,
            ),
        )
        self.branch_3x3.add_module("bn", nn.BatchNorm2d(out_channels))

        self.branch_1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            groups=groups,
            bias=use_1x1_bias,
        )

        if use_residual_connection:
            assert out_channels == in_channels and stride == 1
            self.identity = Residual()
            input_dim = self.in_channels // self.groups
            id_tensor = torch.zeros((self.in_channels, input_dim, 3, 3))
            for i in range(self.in_channels):
                id_tensor[i, i % input_dim, 1, 1] = 1.0
            self.register_buffer("id_tensor", id_tensor, persistent=False)
        else:
            self.identity = None

        if use_alpha:
            noise = torch.randn((1,)) * 0.01
            self.alpha = nn.Parameter(torch.tensor([1.0]) + noise, requires_grad=True)
        else:
            self.alpha = 1.0

        self.post_bn = nn.BatchNorm2d(out_channels) if use_post_bn else nn.Identity()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.branch_3x3.conv.in_channels,
            out_channels=self.branch_3x3.conv.out_channels,
            kernel_size=self.branch_3x3.conv.kernel_size,
            stride=self.branch_3x3.conv.stride,
            padding=self.branch_3x3.conv.padding,
            dilation=self.branch_3x3.conv.dilation,
            groups=self.branch_3x3.conv.groups,
            bias=True,
        )
        self.partially_fused = False
        self.fully_fused = False

        if not build_residual_branches:
            self.fuse_block_residual_branches()

    def forward(self, inputs: Tensor) -> Tensor:
        if self.fully_fused:
            return self.se(self.nonlinearity(self.rbr_reparam(inputs)))
        if self.partially_fused:
            return self.se(self.nonlinearity(self.post_bn(self.rbr_reparam(inputs))))

        id_out = 0.0 if self.identity is None else self.identity(inputs)
        x_3x3 = self.branch_3x3(inputs)
        x_1x1 = self.alpha * self.branch_1x1(inputs)
        branches = x_3x3 + x_1x1 + id_out
        out = self.nonlinearity(self.post_bn(branches))
        return self.se(out)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(
        self, kernel, bias, running_mean, running_var, gamma, beta, eps
    ):
        std = torch.sqrt(running_var + eps)
        fused_bias = beta - gamma * running_mean / std
        scale = (gamma / std).expand_as(kernel.transpose(0, -1)).transpose(0, -1)
        return kernel * scale, bias * (gamma / std) + fused_bias

    def _get_equivalent_kernel_bias_for_branches(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(
            self.branch_3x3.conv.weight,
            0,
            self.branch_3x3.bn.running_mean,
            self.branch_3x3.bn.running_var,
            self.branch_3x3.bn.weight,
            self.branch_3x3.bn.bias,
            self.branch_3x3.bn.eps,
        )
        kernel1x1 = self._pad_1x1_to_3x3_tensor(self.branch_1x1.weight)
        bias1x1 = self.branch_1x1.bias if self.branch_1x1.bias is not None else 0
        kernelid = self.id_tensor if self.identity is not None else 0
        biasid = 0
        eq_kernel = kernel3x3 + self.alpha * kernel1x1 + kernelid
        eq_bias = bias3x3 + self.alpha * bias1x1 + biasid
        return eq_kernel, eq_bias

    def partial_fusion(self):
        if self.partially_fused:
            return
        if self.fully_fused:
            raise NotImplementedError(
                "QARepVGGBlock can't be converted to partially fused from fully fused"
            )

        kernel, bias = self._get_equivalent_kernel_bias_for_branches()
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        del self.branch_3x3
        del self.branch_1x1
        if hasattr(self, "identity"):
            del self.identity
        if hasattr(self, "alpha"):
            del self.alpha
        if hasattr(self, "id_tensor"):
            del self.id_tensor

        self.partially_fused = True
        self.fully_fused = False

    def full_fusion(self):
        if self.fully_fused:
            return
        if not self.partially_fused:
            self.partial_fusion()

        if self.use_post_bn:
            eq_kernel, eq_bias = self._fuse_bn_tensor(
                self.rbr_reparam.weight,
                self.rbr_reparam.bias,
                self.post_bn.running_mean,
                self.post_bn.running_var,
                self.post_bn.weight,
                self.post_bn.bias,
                self.post_bn.eps,
            )
            self.rbr_reparam.weight.data = eq_kernel
            self.rbr_reparam.bias.data = eq_bias

        for para in self.parameters():
            para.detach_()

        if hasattr(self, "post_bn"):
            del self.post_bn

        self.partially_fused = False
        self.fully_fused = True

    def fuse_block_residual_branches(self):
        self.partial_fusion()

    def prep_model_for_conversion(
        self, input_size=None, full_fusion: bool = False, **kwargs
    ):
        if full_fusion:
            self.full_fusion()
        else:
            self.partial_fusion()


class YoloNASBottleneck(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        block_type,
        activation_type: type[nn.Module],
        shortcut: bool,
        use_alpha: bool,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.cv1 = block_type(
            input_channels, output_channels, activation_type=activation_type
        )
        self.cv2 = block_type(
            output_channels, output_channels, activation_type=activation_type
        )
        self.add = shortcut and input_channels == output_channels
        self.shortcut = Residual() if self.add else None
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.alpha = (
            nn.Parameter(torch.tensor([1.0]), requires_grad=True) if use_alpha else 1.0
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.drop_path(self.cv2(self.cv1(x)))
        return self.alpha * self.shortcut(x) + y if self.add else y


class SequentialWithIntermediates(nn.Sequential):
    def __init__(self, output_intermediates: bool, *args):
        super().__init__(*args)
        self.output_intermediates = output_intermediates

    def forward(self, input: Tensor) -> List[Tensor]:
        if self.output_intermediates:
            output = [input]
            for module in self:
                output.append(module(output[-1]))
            return output
        return [super().forward(input)]


class YoloNASCSPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int,
        block_type,
        activation_type: type[nn.Module],
        shortcut: bool = True,
        use_alpha: bool = True,
        expansion: float = 0.5,
        hidden_channels: Optional[int] = None,
        concat_intermediates: bool = False,
        drop_path_rates: Optional[Iterable[float]] = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * num_bottlenecks
        else:
            drop_path_rates = tuple(drop_path_rates)
        if len(drop_path_rates) != num_bottlenecks:
            raise ValueError("drop_path_rates length must equal num_bottlenecks")

        if hidden_channels is None:
            hidden_channels = int(out_channels * expansion)

        self.conv1 = Conv(in_channels, hidden_channels, 1, 1, activation_type)
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1, activation_type)
        self.conv3 = Conv(
            hidden_channels * (2 + int(concat_intermediates) * num_bottlenecks),
            out_channels,
            1,
            1,
            activation_type,
        )
        self.bottlenecks = SequentialWithIntermediates(
            concat_intermediates,
            *[
                YoloNASBottleneck(
                    hidden_channels,
                    hidden_channels,
                    block_type,
                    activation_type,
                    shortcut,
                    use_alpha,
                    drop_path_rate=drop_path_rates[i],
                )
                for i in range(num_bottlenecks)
            ],
        )
        self.dropout = (
            nn.Dropout2d(dropout_rate, inplace=True)
            if dropout_rate > 0.0
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        x = torch.cat((*x_1, x_2), dim=1)
        x = self.dropout(x)
        return self.conv3(x)


class YoloNASStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self._out_channels = out_channels
        self.conv = QARepVGGBlock(
            in_channels,
            out_channels,
            stride=stride,
            use_residual_connection=False,
        )

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class YoloNASStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        activation_type: type[nn.Module],
        hidden_channels: Optional[int] = None,
        concat_intermediates: bool = False,
        drop_path_rates: Optional[Iterable[float]] = None,
        dropout_rate: float = 0.0,
        stride: int = 2,
    ):
        super().__init__()
        self._out_channels = out_channels
        self.downsample = QARepVGGBlock(
            in_channels,
            out_channels,
            stride=stride,
            activation_type=activation_type,
            use_residual_connection=False,
        )
        self.blocks = YoloNASCSPLayer(
            out_channels,
            out_channels,
            num_blocks,
            QARepVGGBlock,
            activation_type,
            shortcut=True,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
            drop_path_rates=drop_path_rates,
            dropout_rate=dropout_rate,
        )

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.downsample(x))


class YoloNASUpStage(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        width_mult: float,
        num_blocks: int,
        depth_mult: float,
        activation_type: type[nn.Module],
        hidden_channels: Optional[int] = None,
        concat_intermediates: bool = False,
        reduce_channels: bool = False,
        drop_path_rates: Optional[Iterable[float]] = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        num_inputs = len(in_channels)
        if num_inputs == 2:
            in_channels, skip_in_channels = in_channels
        else:
            in_channels, skip_in_channels1, skip_in_channels2 = in_channels
            skip_in_channels = skip_in_channels1 + out_channels

        out_channels = width_multiplier(out_channels, width_mult, 8)
        num_blocks = (
            max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks
        )

        if num_inputs == 2:
            self.reduce_skip = (
                Conv(skip_in_channels, out_channels, 1, 1, activation_type)
                if reduce_channels
                else nn.Identity()
            )
        else:
            self.reduce_skip1 = (
                Conv(skip_in_channels1, out_channels, 1, 1, activation_type)
                if reduce_channels
                else nn.Identity()
            )
            self.reduce_skip2 = (
                Conv(skip_in_channels2, out_channels, 1, 1, activation_type)
                if reduce_channels
                else nn.Identity()
            )

        self.conv = Conv(in_channels, out_channels, 1, 1, activation_type)
        self.upsample = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=2, stride=2
        )
        if num_inputs == 3:
            ds_in = out_channels if reduce_channels else skip_in_channels2
            self.downsample = Conv(ds_in, out_channels, 3, 2, activation_type)

        self.reduce_after_concat = (
            Conv(num_inputs * out_channels, out_channels, 1, 1, activation_type)
            if reduce_channels
            else nn.Identity()
        )
        after_concat_channels = (
            out_channels if reduce_channels else out_channels + skip_in_channels
        )
        self.blocks = YoloNASCSPLayer(
            after_concat_channels,
            out_channels,
            num_blocks,
            QARepVGGBlock,
            activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
            drop_path_rates=drop_path_rates,
            dropout_rate=dropout_rate,
        )
        self._out_channels = [out_channels, out_channels]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        if len(inputs) == 2:
            x, skip_x = inputs
            skip_x = [self.reduce_skip(skip_x)]
        else:
            x, skip_x1, skip_x2 = inputs
            skip_x1 = self.reduce_skip1(skip_x1)
            skip_x2 = self.reduce_skip2(skip_x2)
            skip_x = [skip_x1, self.downsample(skip_x2)]
        x_inter = self.conv(x)
        x = self.upsample(x_inter)
        x = torch.cat([x, *skip_x], 1)
        x = self.reduce_after_concat(x)
        x = self.blocks(x)
        return x_inter, x


class YoloNASDownStage(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        width_mult: float,
        num_blocks: int,
        depth_mult: float,
        activation_type: type[nn.Module],
        hidden_channels: Optional[int] = None,
        concat_intermediates: bool = False,
        drop_path_rates: Optional[Iterable[float]] = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        in_channels, skip_in_channels = in_channels
        out_channels = width_multiplier(out_channels, width_mult, 8)
        num_blocks = (
            max(round(num_blocks * depth_mult), 1) if num_blocks > 1 else num_blocks
        )

        self.conv = Conv(in_channels, out_channels // 2, 3, 2, activation_type)
        after_concat_channels = out_channels // 2 + skip_in_channels
        self.blocks = YoloNASCSPLayer(
            in_channels=after_concat_channels,
            out_channels=out_channels,
            num_bottlenecks=num_blocks,
            block_type=partial(Conv, kernel=3, stride=1),
            activation_type=activation_type,
            hidden_channels=hidden_channels,
            concat_intermediates=concat_intermediates,
            drop_path_rates=drop_path_rates,
            dropout_rate=dropout_rate,
        )
        self._out_channels = out_channels

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, inputs):
        x, skip_x = inputs
        x = self.conv(x)
        x = torch.cat([x, skip_x], 1)
        return self.blocks(x)


class SPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        k: Tuple[int, ...],
        activation_type: type[nn.Module],
    ):
        super().__init__()
        self._output_channels = output_channels
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = Conv(
            hidden_channels * (len(k) + 1), output_channels, 1, 1, activation_type
        )
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

    @property
    def out_channels(self):
        return self._output_channels


class YoloNASBackbone(nn.Module):
    def __init__(self, config: dict, in_channels: int = 3):
        super().__init__()
        activation_type = nn.ReLU
        self.stem = YoloNASStem(in_channels, 48)
        stage_out = [96, 192, 384, 768]
        stage_blocks = [2, 3, 5, 2]
        stage_hidden = config["stage_hidden"]
        stage_concat = config["stage_concat"]

        self.stage1 = YoloNASStage(
            48,
            stage_out[0],
            stage_blocks[0],
            activation_type,
            hidden_channels=stage_hidden[0],
            concat_intermediates=stage_concat[0],
        )
        self.stage2 = YoloNASStage(
            stage_out[0],
            stage_out[1],
            stage_blocks[1],
            activation_type,
            hidden_channels=stage_hidden[1],
            concat_intermediates=stage_concat[1],
        )
        self.stage3 = YoloNASStage(
            stage_out[1],
            stage_out[2],
            stage_blocks[2],
            activation_type,
            hidden_channels=stage_hidden[2],
            concat_intermediates=stage_concat[2],
        )
        self.stage4 = YoloNASStage(
            stage_out[2],
            stage_out[3],
            stage_blocks[3],
            activation_type,
            hidden_channels=stage_hidden[3],
            concat_intermediates=stage_concat[3],
        )
        self.context_module = SPP(
            stage_out[3], stage_out[3], (5, 9, 13), activation_type
        )
        self._out_channels = (stage_out[0], stage_out[1], stage_out[2], stage_out[3])

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.stem(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        c5 = self.context_module(c5)
        return c2, c3, c4, c5


class YoloNASPANNeckWithC2(nn.Module):
    def __init__(self, in_channels: List[int], config: dict):
        super().__init__()
        c2_out, c3_out, c4_out, c5_out = in_channels
        activation_type = nn.ReLU

        self.neck1 = YoloNASUpStage(
            [c5_out, c4_out, c3_out],
            out_channels=192,
            num_blocks=config["up_blocks"][0],
            hidden_channels=config["up_hidden"][0],
            width_mult=1.0,
            depth_mult=1.0,
            activation_type=activation_type,
            reduce_channels=True,
        )
        self.neck2 = YoloNASUpStage(
            [self.neck1.out_channels[1], c3_out, c2_out],
            out_channels=96,
            num_blocks=config["up_blocks"][1],
            hidden_channels=config["up_hidden"][1],
            width_mult=1.0,
            depth_mult=1.0,
            activation_type=activation_type,
            reduce_channels=True,
        )
        self.neck3 = YoloNASDownStage(
            [self.neck2.out_channels[1], self.neck2.out_channels[0]],
            out_channels=192,
            num_blocks=config["down_blocks"][0],
            hidden_channels=config["down_hidden"][0],
            width_mult=1.0,
            depth_mult=1.0,
            activation_type=activation_type,
        )
        self.neck4 = YoloNASDownStage(
            [self.neck3.out_channels, self.neck1.out_channels[0]],
            out_channels=384,
            num_blocks=config["down_blocks"][1],
            hidden_channels=config["down_hidden"][1],
            width_mult=1.0,
            depth_mult=1.0,
            activation_type=activation_type,
        )

        self._out_channels = [
            self.neck2.out_channels[1],
            self.neck3.out_channels,
            self.neck4.out_channels,
        ]

    @property
    def out_channels(self):
        return self._out_channels

    def forward(
        self, inputs: Tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        c2, c3, c4, c5 = inputs
        x_n1_inter, x = self.neck1([c5, c4, c3])
        x_n2_inter, p3 = self.neck2([x, c3, c2])
        p4 = self.neck3([p3, x_n2_inter])
        p5 = self.neck4([p4, x_n1_inter])
        return p3, p4, p5


@torch.no_grad()
def generate_anchors_for_grid_cell(
    feats: Tuple[Tensor, ...],
    fpn_strides: Tuple[int, ...],
    grid_cell_size: float = 5.0,
    grid_cell_offset: float = 0.5,
    dtype: torch.dtype = torch.float,
) -> Tuple[Tensor, Tensor, List[int], Tensor]:
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    device = feats[0].device

    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
        shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
        if torch.__version__ >= "1.10":
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        else:
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)

        anchor = torch.stack(
            [
                shift_x - cell_half_size,
                shift_y - cell_half_size,
                shift_x + cell_half_size,
                shift_y + cell_half_size,
            ],
            dim=-1,
        ).to(dtype=dtype)
        anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)

        anchors.append(anchor.reshape([-1, 4]))
        anchor_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(
            torch.full([num_anchors_list[-1], 1], stride, dtype=dtype, device=device)
        )

    anchors = torch.cat(anchors).to(device)
    anchor_points = torch.cat(anchor_points).to(device)
    stride_tensor = torch.cat(stride_tensor).to(device)
    return anchors, anchor_points, num_anchors_list, stride_tensor


def batch_distance2bbox(
    points: Tensor, distance: Tensor, max_shapes: Optional[Tensor] = None
) -> Tensor:
    lt, rb = torch.split(distance, 2, dim=-1)
    x1y1 = points - lt
    x2y2 = rb + points
    out_bbox = torch.cat([x1y1, x2y2], dim=-1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))
    return out_bbox


class YoloNASDFLHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        inter_channels: int,
        width_mult: float,
        first_conv_group_size: int,
        num_classes: int,
        stride: int,
        reg_max: int,
        cls_dropout_rate: float = 0.0,
        reg_dropout_rate: float = 0.0,
    ):
        super().__init__()
        inter_channels = width_multiplier(inter_channels, width_mult, 8)
        if first_conv_group_size == 0:
            groups = 0
        elif first_conv_group_size == -1:
            groups = 1
        else:
            groups = inter_channels // first_conv_group_size

        self.num_classes = num_classes
        self.stem = ConvBNReLU(
            in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

        first_cls_conv = (
            [
                ConvBNReLU(
                    inter_channels,
                    inter_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    bias=False,
                )
            ]
            if groups
            else []
        )
        self.cls_convs = nn.Sequential(
            *first_cls_conv,
            ConvBNReLU(
                inter_channels,
                inter_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        first_reg_conv = (
            [
                ConvBNReLU(
                    inter_channels,
                    inter_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    bias=False,
                )
            ]
            if groups
            else []
        )
        self.reg_convs = nn.Sequential(
            *first_reg_conv,
            ConvBNReLU(
                inter_channels,
                inter_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.cls_pred = nn.Conv2d(inter_channels, self.num_classes, 1, 1, 0)
        self.reg_pred = nn.Conv2d(inter_channels, 4 * (reg_max + 1), 1, 1, 0)
        self.cls_dropout_rate = (
            nn.Dropout2d(cls_dropout_rate) if cls_dropout_rate > 0 else nn.Identity()
        )
        self.reg_dropout_rate = (
            nn.Dropout2d(reg_dropout_rate) if reg_dropout_rate > 0 else nn.Identity()
        )

        self.grid = torch.zeros(1)
        self.stride = stride
        self.prior_prob = 1e-2
        self._initialize_biases()

    def _initialize_biases(self):
        prior_bias = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_pred.bias, prior_bias)

    def replace_num_classes(self, num_classes: int):
        self.cls_pred = nn.Conv2d(self.cls_pred.in_channels, num_classes, 1, 1, 0)
        self.num_classes = num_classes
        self._initialize_biases()

    @property
    def out_channels(self):
        return None

    def forward(self, x: Tensor):
        x = self.stem(x)
        cls_feat = self.cls_dropout_rate(self.cls_convs(x))
        reg_feat = self.reg_dropout_rate(self.reg_convs(x))
        cls_output = self.cls_pred(cls_feat)
        reg_output = self.reg_pred(reg_feat)
        return reg_output, cls_output


class NDFLHeads(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: Tuple[int, int, int],
        width_mult: float,
        grid_cell_scale: float = 5.0,
        grid_cell_offset: float = 0.5,
        reg_max: int = 16,
        eval_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.num_classes = num_classes
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.eval_size = eval_size
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1).reshape(
            [1, self.reg_max + 1, 1, 1]
        )
        self.register_buffer("proj_conv", proj, persistent=False)

        self.head1 = YoloNASDFLHead(
            in_channels[0], 128, width_mult, 0, num_classes, 8, reg_max
        )
        self.head2 = YoloNASDFLHead(
            in_channels[1], 256, width_mult, 0, num_classes, 16, reg_max
        )
        self.head3 = YoloNASDFLHead(
            in_channels[2], 512, width_mult, 0, num_classes, 32, reg_max
        )
        self.num_heads = 3
        self.fpn_strides = (8, 16, 32)
        self._init_weights()

    def replace_num_classes(self, num_classes: int):
        self.head1.replace_num_classes(num_classes)
        self.head2.replace_num_classes(num_classes)
        self.head3.replace_num_classes(num_classes)
        self.num_classes = num_classes

    @torch.jit.ignore
    def cache_anchors(self, input_size: Tuple[int, int]):
        self.eval_size = input_size
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        anchor_points, stride_tensor = self._generate_anchors(
            dtype=dtype, device=device
        )
        self.register_buffer("anchor_points", anchor_points, persistent=False)
        self.register_buffer("stride_tensor", stride_tensor, persistent=False)

    @torch.jit.ignore
    def _init_weights(self):
        if self.eval_size:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            anchor_points, stride_tensor = self._generate_anchors(
                dtype=dtype, device=device
            )
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def _generate_anchors(self, feats=None, dtype=None, device=None):
        anchor_points = []
        stride_tensor = []
        dtype = dtype or feats[0].dtype
        device = device or feats[0].device

        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)

            shift_x = (
                torch.arange(end=w, dtype=torch.float32, device=device)
                + self.grid_cell_offset
            )
            shift_y = (
                torch.arange(end=h, dtype=torch.float32, device=device)
                + self.grid_cell_offset
            )
            if torch.__version__ >= "1.10":
                shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            else:
                shift_y, shift_x = torch.meshgrid(shift_y, shift_x)

            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                torch.full([h * w, 1], stride, dtype=dtype, device=device)
            )

        return torch.cat(anchor_points), torch.cat(stride_tensor)

    @property
    def out_channels(self):
        return None

    def forward(self, feats: Tuple[Tensor, ...]):
        feats = feats[: self.num_heads]
        cls_score_list = []
        reg_distri_list = []
        reg_dist_reduced_list = []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            hw = h * w
            reg_distri, cls_logit = getattr(self, f"head{i + 1}")(feat)
            reg_distri_list.append(torch.permute(reg_distri.flatten(2), [0, 2, 1]))

            reg_dist_reduced = torch.permute(
                reg_distri.reshape([-1, 4, self.reg_max + 1, hw]),
                [0, 2, 3, 1],
            )
            reg_dist_reduced = torch.softmax(reg_dist_reduced, dim=1) * self.proj_conv
            reg_dist_reduced = reg_dist_reduced.sum(dim=1, keepdim=False)

            cls_score_list.append(cls_logit.reshape([b, self.num_classes, hw]))
            reg_dist_reduced_list.append(reg_dist_reduced)

        cls_score_list = torch.cat(cls_score_list, dim=-1)
        cls_score_list = torch.permute(cls_score_list, [0, 2, 1])
        reg_distri_list = torch.cat(reg_distri_list, dim=1)
        reg_dist_reduced_list = torch.cat(reg_dist_reduced_list, dim=1)

        if self.eval_size:
            anchor_points_inference, stride_tensor = (
                self.anchor_points,
                self.stride_tensor,
            )
        else:
            anchor_points_inference, stride_tensor = self._generate_anchors(feats)

        pred_scores = cls_score_list.sigmoid()
        pred_bboxes = (
            batch_distance2bbox(anchor_points_inference, reg_dist_reduced_list)
            * stride_tensor
        )
        decoded_predictions = pred_bboxes, pred_scores

        if torch.jit.is_tracing():
            return decoded_predictions

        anchors, anchor_points, num_anchors_list, raw_stride_tensor = (
            generate_anchors_for_grid_cell(
                feats,
                self.fpn_strides,
                self.grid_cell_scale,
                self.grid_cell_offset,
            )
        )
        raw_predictions = (
            cls_score_list,
            reg_distri_list,
            anchors,
            anchor_points,
            num_anchors_list,
            raw_stride_tensor,
        )
        return decoded_predictions, raw_predictions


_VARIANT_CONFIGS = {
    "s": {
        "stage_hidden": [32, 64, 96, 192],
        "stage_concat": [False, False, False, False],
        "up_blocks": [2, 2],
        "up_hidden": [64, 48],
        "down_blocks": [2, 2],
        "down_hidden": [64, 64],
        "head_width_mult": 0.5,
    },
    "m": {
        "stage_hidden": [64, 128, 256, 384],
        "stage_concat": [True, True, True, False],
        "up_blocks": [2, 3],
        "up_hidden": [192, 64],
        "down_blocks": [2, 3],
        "down_hidden": [192, 256],
        "head_width_mult": 0.75,
    },
    "l": {
        "stage_hidden": [96, 128, 256, 512],
        "stage_concat": [True, True, True, True],
        "up_blocks": [4, 4],
        "up_hidden": [128, 128],
        "down_blocks": [4, 4],
        "down_hidden": [128, 256],
        "head_width_mult": 1.0,
    },
}


class LibreYOLONASModel(nn.Module):
    def __init__(
        self,
        config: str = "s",
        nb_classes: int = 80,
        in_channels: int = 3,
        reg_max: int = 16,
        eval_size: Optional[Tuple[int, int]] = None,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.03,
        inplace_act: bool = True,
    ):
        super().__init__()
        if config not in _VARIANT_CONFIGS:
            raise ValueError(f"Unknown YOLO-NAS config '{config}'")

        variant = _VARIANT_CONFIGS[config]
        self.config = config
        self.nc = nb_classes
        self.reg_max = reg_max

        self.backbone = YoloNASBackbone(variant, in_channels=in_channels)
        self.neck = YoloNASPANNeckWithC2(list(self.backbone.out_channels), variant)
        self.heads = NDFLHeads(
            num_classes=nb_classes,
            in_channels=tuple(self.neck.out_channels),
            width_mult=variant["head_width_mult"],
            reg_max=reg_max,
            eval_size=eval_size,
        )
        self._initialize_weights(bn_eps, bn_momentum, inplace_act)

    @property
    def head(self):
        return self.heads

    def _initialize_weights(self, bn_eps: float, bn_momentum: float, inplace_act: bool):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = bn_eps
                m.momentum = bn_momentum
            elif inplace_act and isinstance(
                m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, nn.Mish)
            ):
                m.inplace = True

    def prep_model_for_conversion(
        self, input_size=None, full_fusion: bool = False, **kwargs
    ):
        for module in self.modules():
            if module is not self and hasattr(module, "prep_model_for_conversion"):
                module.prep_model_for_conversion(
                    input_size=input_size, full_fusion=full_fusion, **kwargs
                )

    def fuse_reparam(self, full_fusion: bool = False):
        self.prep_model_for_conversion(full_fusion=full_fusion)
        return self

    def forward(self, x: Tensor):
        x = self.backbone(x)
        x = self.neck(x)
        return self.heads(x)

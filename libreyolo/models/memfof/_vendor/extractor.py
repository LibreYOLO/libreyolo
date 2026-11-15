from typing import Literal
import torch
import torch.nn as nn
from torchvision.models import (
    WeightsEnum,
    get_model,
)


def init_model_weights(model: nn.Module) -> None:
    """
    Initialize model weights using Kaiming initialization.

    Parameters
    ----------
    model : nn.Module
        Model to initialize weights for
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def init_conv1_weights(
    conv1: nn.Conv2d, pretrained_weights: torch.Tensor, input_dim: int
) -> None:
    """
    Initialize conv1 layer weights based on input dimension.

    Parameters
    ----------
    conv1 : nn.Conv2d
        Conv1 layer to initialize
    pretrained_weights : torch.Tensor
        Pretrained weights from original model
    input_dim : int
        Input dimension, must be divisible by 3
    """
    if input_dim % 3 != 0:
        raise ValueError(f"Input dimension must be divisible by 3, got {input_dim}")

    n_repeats = input_dim // 3
    conv1.weight.data = torch.cat([pretrained_weights] * n_repeats, dim=1)


class ResNetFPN16x(nn.Module):
    """
    ResNet18, output resolution is 1/16.
    Each block has 2 layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model: Literal["resnet18", "resnet34", "resnet50"],
        model_weights: WeightsEnum | None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = model
        self.model_weights = model_weights

        self.resnet = get_model(model, weights=model_weights)

        if model_weights is None:
            init_model_weights(self.resnet)

        pretrained_conv1 = self.resnet.conv1.weight
        self.resnet.conv1 = nn.Conv2d(
            input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        init_conv1_weights(self.resnet.conv1, pretrained_conv1, input_dim)

        del self.resnet.maxpool
        del self.resnet.layer4
        del self.resnet.avgpool
        del self.resnet.fc

        self.final_conv = nn.Conv2d(
            1024 if model == "resnet50" else 256,
            output_dim,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        init_model_weights(self.final_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        return self.final_conv(x)

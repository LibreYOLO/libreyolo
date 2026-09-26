"""Native construction of the published GTR oriented-box (DOTA) architectures.

Configuration source: Intellindust-AI-Lab/GTR (MIT), revision
782e737efe2e6437ac537fbdcee089673d3376c1, configs/obb/dota_finetune.

Only S and X are built: they are the sizes with published weights, and they
share the detection backbone (``ViTAdapterSpatialSwiGLU``). The upstream M and
L OBB configs use a different, plain ``ViTAdapter`` backbone and have no
released checkpoints.
"""

from torch import nn

from .encoder import GTREncoder
from .nn import SIZE_CONFIGS
from .obb_decoder import OBBGTRTransformer
from .spatial import ViTAdapterSpatialSwiGLU

OBB_SIZES = ("s", "x")
OBB_INPUT_SIZE = 1024
DOTA_CLASSES = (
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
)


class LibreGTROBBModel(nn.Module):
    def __init__(
        self,
        config="s",
        nb_classes=15,
        eval_spatial_size=(OBB_INPUT_SIZE, OBB_INPUT_SIZE),
    ):
        super().__init__()
        if config not in OBB_SIZES:
            raise ValueError(
                f"GTR OBB supports sizes {OBB_SIZES}; upstream publishes no "
                f"{config!r} OBB weights"
            )
        embed, heads, ratio, hidden, feedforward = SIZE_CONFIGS[config]
        self.backbone = ViTAdapterSpatialSwiGLU(
            embed_dim=embed,
            num_heads=heads,
            ffn_ratio=ratio,
            interaction_indexes=[3, 7, 11],
            multi_layer_same_res=True,
            skip_weights_warning=True,
            eval_spatial_size=eval_spatial_size,
        )
        self.encoder = GTREncoder(
            in_channels=[embed] * 3,
            hidden_dim=hidden,
            eval_spatial_size=eval_spatial_size,
        )
        self.decoder = OBBGTRTransformer(
            num_classes=nb_classes,
            hidden_dim=hidden,
            feat_channels=[hidden] * 3,
            feat_strides=[8, 16, 32],
            num_levels=3,
            num_layers=4,
            dim_feedforward=feedforward,
            num_points=[3, 6, 3],
            group_detr=3,
            num_denoising=100,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            reg_max=32,
            reg_scale=4,
            layer_scale=1,
            activation="silu",
            eval_spatial_size=eval_spatial_size,
        )

    def forward(self, x, targets=None):
        if x.shape[-2] != x.shape[-1] or x.shape[-1] % 32:
            raise ValueError("GTR expects a square input with side divisible by 32")
        return self.decoder(self.encoder(self.backbone(x)), targets)

    def deploy(self):
        return self.eval()

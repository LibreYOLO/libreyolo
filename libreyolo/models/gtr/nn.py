"""Native construction of the published GTR detection architectures.

Configuration source: Intellindust-AI-Lab/GTR (MIT), revision
782e737efe2e6437ac537fbdcee089673d3376c1, configs/det/coco_finetune.
"""

from torch import nn

from .decoder import GTRTransformer
from .encoder import GTREncoder
from .spatial import ViTAdapterSpatialSwiGLU

SIZE_CONFIGS = {
    "s": (192, 3, 4, 192, 512),
    "m": (256, 4, 4, 256, 1024),
    "l": (384, 6, 4, 256, 1024),
    "x": (384, 6, 6, 256, 2048),
}


class LibreGTRModel(nn.Module):
    def __init__(self, config="s", nb_classes=80, eval_spatial_size=(640, 640)):
        super().__init__()
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
        self.decoder = GTRTransformer(
            num_classes=nb_classes,
            hidden_dim=hidden,
            feat_channels=[hidden] * 3,
            feat_strides=[8, 16, 32],
            num_layers=4,
            dim_feedforward=feedforward,
            num_points=[3, 6, 3],
            group_detr=3,
            eval_spatial_size=eval_spatial_size,
        )

    def forward(self, x, targets=None):
        if x.shape[-2] != x.shape[-1] or x.shape[-1] % 32:
            raise ValueError("GTR expects a square input with side divisible by 32")
        return self.decoder(self.encoder(self.backbone(x)), targets)

    def deploy(self):
        # Keep the portable graph and training state dict intact. Upstream's
        # destructive fusion requires its separately built CUDA plugin.
        return self.eval()

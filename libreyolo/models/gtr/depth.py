"""GTR monocular depth: GTR backbone + encoder + DPT-style depth head.

Ported from Intellindust-AI-Lab/GTR (MIT), revision
782e737efe2e6437ac537fbdcee089673d3376c1, ``engine/gtr/depth/`` and
``configs/depth/pretrain``. The head itself is upstream's adaptation of the
Depth-Anything-V2 metric DPT head (Apache-2.0) to the 3-level GTR pyramid.

The released checkpoints are the ``gtrdepth_{s,m,l,x}`` models pretrained on
a mixed metric-depth corpus with a log-depth head: the raw output is depth in
metres, ``exp(clamp(logit, -4, 5))``. LibreYOLO's depth contract (ADR 0006) is
relative inverse depth, so :meth:`LibreGTRDepthModel.forward` returns
``1 / depth``: higher means closer, and ``1 / value`` recovers upstream's
metre estimate for cameras like the training ones. ImageNet normalization is
in-graph so native and exported runtimes take the same ``[0, 1]`` RGB input.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import GTREncoder
from .nn import SIZE_CONFIGS
from .spatial import ViTAdapterSpatialSwiGLU

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEPTH_HEAD_KEY = "decoder.output_conv2.2.weight"


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.activation = nn.ReLU(False)

    def forward(self, x):
        out = self.conv1(self.activation(x))
        out = self.conv2(self.activation(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    """DPT RefineNet fusion: add the skip branch, refine, then upsample."""

    def __init__(self, features):
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, 1, bias=True)
        self.res_conv_unit1 = ResidualConvUnit(features)
        self.res_conv_unit2 = ResidualConvUnit(features)

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2:
            output = output + self.res_conv_unit1(xs[1])
        output = self.res_conv_unit2(output)
        modifier = {"scale_factor": 2} if size is None else {"size": size}
        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=True)
        return self.out_conv(output)


class DPTDepthHead(nn.Module):
    """Fuse the stride 8/16/32 pyramid into a dense log-decoded metre map."""

    def __init__(self, in_channels, features=128):
        super().__init__()
        self.layer_rn = nn.ModuleList(
            nn.Conv2d(ch, features, 3, padding=1, bias=False) for ch in in_channels
        )
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet1 = FeatureFusionBlock(features)
        self.output_conv1 = nn.Conv2d(features, features // 2, 3, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 1),
        )
        # exp(0.182) ~ 1.2 m keeps early outputs well conditioned (upstream init).
        nn.init.constant_(self.output_conv2[-1].bias, 0.182)

    def forward(self, feats, out_hw):
        layer_1, layer_2, layer_3 = [rn(f) for rn, f in zip(self.layer_rn, feats)]
        path_3 = self.refinenet3(layer_3, size=layer_2.shape[2:])
        path_2 = self.refinenet2(path_3, layer_2, size=layer_1.shape[2:])
        path_1 = self.refinenet1(path_2, layer_1)
        out = self.output_conv1(path_1)
        out = F.interpolate(out, size=out_hw, mode="bilinear", align_corners=True)
        out = self.output_conv2(out)
        return torch.exp(out.clamp(-4.0, 5.0)).squeeze(1)


class LibreGTRDepthModel(nn.Module):
    """GTR depth network taking ``[0, 1]`` RGB and emitting inverse depth."""

    def __init__(self, config="s", eval_spatial_size=(640, 640)):
        super().__init__()
        embed, heads, ratio, hidden, _ = SIZE_CONFIGS[config]
        self.backbone = ViTAdapterSpatialSwiGLU(
            embed_dim=embed,
            num_heads=heads,
            ffn_ratio=ratio,
            interaction_indexes=[3, 7, 11],
            multi_layer_same_res=True,
            skip_weights_warning=True,
            eval_spatial_size=eval_spatial_size,
        )
        # The released depth checkpoints were trained without the contiguous
        # copy the detection encoder inserts before its convs.
        self.encoder = GTREncoder(
            in_channels=[embed] * 3,
            hidden_dim=hidden,
            eval_spatial_size=eval_spatial_size,
            force_contiguous=False,
        )
        self.decoder = DPTDepthHead([hidden] * 3)
        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward_metric(self, x):
        """Upstream ``pred_depth``: ``[B, H, W]`` metres from ``[0, 1]`` RGB."""
        if x.shape[-2] != x.shape[-1] or x.shape[-1] % 32:
            raise ValueError("GTR expects a square input with side divisible by 32")
        x = (x - self.pixel_mean) / self.pixel_std
        return self.decoder(self.encoder(self.backbone(x)), x.shape[-2:])

    def forward(self, x):
        """Relative inverse depth ``[B, 1, H, W]`` (ADR 0006)."""
        return self.forward_metric(x).reciprocal().unsqueeze(1)

    def deploy(self):
        return self.eval()


def is_depth_state_dict(state_dict) -> bool:
    return DEPTH_HEAD_KEY in state_dict and "decoder.refinenet1.out_conv.weight" in (
        state_dict
    )


def preprocess_numpy(img_rgb_hwc, input_size=640):
    """Square bilinear resize to ``[0, 1]`` CHW; normalization is in-graph."""
    import numpy as np
    from PIL import Image

    size = input_size if isinstance(input_size, int) else int(input_size[0])
    resized = Image.fromarray(img_rgb_hwc).resize(
        (size, size), Image.Resampling.BILINEAR
    )
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), 1.0


def preprocess_image(image, input_size, color_format="auto"):
    import numpy as np

    from ...utils.image_loader import ImageLoader

    img = ImageLoader.load(image, color_format=color_format).convert("RGB")
    chw, ratio = preprocess_numpy(np.asarray(img), input_size)
    return torch.from_numpy(chw).unsqueeze(0), img, img.size, ratio


def postprocess(output, original_size):
    depth = torch.as_tensor(output)
    if depth.ndim == 3:
        depth = depth.unsqueeze(1)
    orig_w, orig_h = original_size
    # align_corners=True matches upstream's resize back to the ground-truth grid.
    depth = F.interpolate(
        depth.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=True
    )
    return {"depth": depth[0, 0].cpu()}


def train(wrapper, *, data=None, resume=False, **kwargs):
    """Fine-tune a GTR depth model with the upstream SILog recipe."""
    from pathlib import Path

    from .depth_trainer import GTRDepthConfig, GTRDepthTrainer

    if kwargs.get("lora"):
        raise ValueError("LoRA is not supported for depth models (ADR 0006).")
    kwargs.pop("pretrained", None)
    resume_path, settings = wrapper._resume_settings(
        resume, GTRDepthConfig, {"data": data, **kwargs}
    )
    data = settings.pop("data", None)
    device = settings.pop("device", "") or "auto"
    if settings.get("imgsz") is not None:
        settings["imgsz"] = wrapper._validate_imgsz(settings["imgsz"])
    trainer = GTRDepthTrainer(
        model=wrapper.model,
        wrapper_model=wrapper,
        size=wrapper.size,
        num_classes=1,
        data=data,
        device=device,
        resume=bool(resume_path),
        **settings,
    )
    if resume_path:
        trainer.setup()
        trainer.resume(resume_path)
    results = trainer.train()
    best = results.get("best_checkpoint")
    if best and Path(best).exists():
        wrapper.model_path = best
        wrapper._load_weights(best)
    wrapper.model.to(wrapper.device)
    return results

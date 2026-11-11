"""Appearance (re-identification) embedders for tracking.

Provides the OSNet-AIN feature extractor used by
:class:`~libreyolo.tracking.deepocsort.DeepOCSortTracker` to describe what each
detection *looks like*, so tracks can be re-associated by appearance after
occlusions or crossings.

The network is a port of OSNet-AIN from Torchreid (MIT):

    Zhou et al. Omni-Scale Feature Learning for Person Re-Identification.
    ICCV, 2019.
    Zhou et al. Learning Generalisable Omni-Scale Representations for Person
    Re-Identification. TPAMI, 2021.
    https://github.com/KaiyangZhou/deep-person-reid (MIT license)

Module/parameter names deliberately mirror the original so Torchreid
checkpoints load directly (the unused identity-classifier head is dropped).
Every variant outputs a 512-d embedding; crops are preprocessed exactly like
the Deep OC-SORT reference ("general model" path): RGB, resized to 256x128,
scaled to [0, 1] and ImageNet-normalized, with the output L2-normalized.

Any callable with the :class:`OSNetEmbedder.__call__` signature
``(image, boxes) -> (N, D) float32`` can be used as an embedder, so custom
appearance models plug in without subclassing.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# Per-variant stage widths (from Torchreid's osnet_ain instantiations).
_OSNET_CHANNELS = {
    "osnet_ain_x1_0": [64, 256, 384, 512],
    "osnet_ain_x0_75": [48, 192, 288, 384],
    "osnet_ain_x0_5": [32, 128, 192, 256],
    "osnet_ain_x0_25": [16, 64, 96, 128],
}

# LibreYOLO-hosted mirrors of the Torchreid multi-source (MS+D+C) checkpoints.
_WEIGHT_URL = "https://huggingface.co/LibreYOLO/LibreReID-osnet/resolve/main/{name}.pt"


# ---------------------------------------------------------------------------
# OSNet-AIN building blocks (names mirror Torchreid for state-dict parity)
# ---------------------------------------------------------------------------


class _ConvLayer(nn.Module):
    """Conv + (batch|instance) norm + ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.InstanceNorm2d(out_ch, affine=True) if IN else nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1(nn.Module):
    """1x1 conv + bn + ReLU."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class _Conv1x1Linear(nn.Module):
    """1x1 conv + optional bn, no non-linearity."""

    def __init__(self, in_ch, out_ch, stride=1, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class _LightConv3x3(nn.Module):
    """1x1 (linear) conv followed by a depthwise 3x3 conv + bn + ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, 3, stride=1, padding=1, bias=False, groups=out_ch
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class _LightConvStream(nn.Module):
    """Chain of ``depth`` LightConv3x3 layers (one omni-scale stream)."""

    def __init__(self, in_ch, out_ch, depth):
        super().__init__()
        layers = [_LightConv3x3(in_ch, out_ch)]
        layers += [_LightConv3x3(out_ch, out_ch) for _ in range(depth - 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _ChannelGate(nn.Module):
    """Channel-wise gates conditioned on the input (squeeze-excite style)."""

    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, in_ch // reduction, kernel_size=1, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_ch // reduction, in_ch, kernel_size=1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        inp = x
        x = self.global_avgpool(x)
        x = self.relu(self.fc1(x))
        x = self.gate_activation(self.fc2(x))
        return inp * x


class _OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(self, in_ch, out_ch, reduction=4, T=4):
        super().__init__()
        mid = out_ch // reduction
        self.conv1 = _Conv1x1(in_ch, mid)
        self.conv2 = nn.ModuleList(
            [_LightConvStream(mid, mid, t) for t in range(1, T + 1)]
        )
        self.gate = _ChannelGate(mid)
        self.conv3 = _Conv1x1Linear(mid, out_ch)
        self.downsample = _Conv1x1Linear(in_ch, out_ch) if in_ch != out_ch else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for stream in self.conv2:
            x2 = x2 + self.gate(stream(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class _OSBlockINin(nn.Module):
    """Omni-scale block with instance normalization inside the residual."""

    def __init__(self, in_ch, out_ch, reduction=4, T=4):
        super().__init__()
        mid = out_ch // reduction
        self.conv1 = _Conv1x1(in_ch, mid)
        self.conv2 = nn.ModuleList(
            [_LightConvStream(mid, mid, t) for t in range(1, T + 1)]
        )
        self.gate = _ChannelGate(mid)
        self.conv3 = _Conv1x1Linear(mid, out_ch, bn=False)
        self.downsample = _Conv1x1Linear(in_ch, out_ch) if in_ch != out_ch else None
        self.IN = nn.InstanceNorm2d(out_ch, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for stream in self.conv2:
            x2 = x2 + self.gate(stream(x1))
        x3 = self.IN(self.conv3(x2))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity)


class OSNet(nn.Module):
    """OSNet-AIN feature extractor (inference-only, classifier head dropped).

    Args:
        channels: Stage widths, one of the ``_OSNET_CHANNELS`` entries.
        feature_dim: Output embedding dimensionality (512 for all released
            checkpoints).
    """

    def __init__(self, channels: list[int], feature_dim: int = 512):
        super().__init__()
        # Block layout is identical across all osnet_ain variants.
        blocks = [
            [_OSBlockINin, _OSBlockINin],
            [_OSBlock, _OSBlockINin],
            [_OSBlockINin, _OSBlock],
        ]
        self.conv1 = _ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], channels[0], channels[1])
        self.pool2 = nn.Sequential(
            _Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2)
        )
        self.conv3 = self._make_layer(blocks[1], channels[1], channels[2])
        self.pool3 = nn.Sequential(
            _Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2)
        )
        self.conv4 = self._make_layer(blocks[2], channels[2], channels[3])
        self.conv5 = _Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _make_layer(blocks, in_ch, out_ch):
        layers = [blocks[0](in_ch, out_ch)]
        layers += [b(out_ch, out_ch) for b in blocks[1:]]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        return self.fc(v)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_osnet_state_dict(path: str | Path) -> dict:
    """Load an OSNet state dict, accepting raw Torchreid checkpoints.

    Handles the ``{"state_dict": ...}`` wrapper, the DataParallel ``module.``
    prefix, and drops the identity-classifier head (dataset-specific, unused
    for embedding extraction).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("classifier."):
            continue
        out[k] = v
    return out


def _resolve_weights(name: str) -> Path:
    """Download a named OSNet checkpoint to the local cache if needed."""
    cache_dir = Path.home() / ".cache" / "libreyolo" / "reid"
    dest = cache_dir / f"{name}.pt"
    if dest.exists():
        return dest
    from ..utils.download import _get_hf_token
    from ..utils.hf_hub import apply_hf_endpoint, hf_unreachable_hint

    import requests

    url = apply_hf_endpoint(_WEIGHT_URL.format(name=name))
    logger.info("Downloading ReID weights %s from %s", name, url)
    cache_dir.mkdir(parents=True, exist_ok=True)
    headers = {}
    token = _get_hf_token()
    # Only send the token to huggingface.co, never to a third-party mirror.
    if token and url.startswith("https://huggingface.co/"):
        headers["Authorization"] = f"Bearer {token}"
    # Atomically-created temp file so concurrent cold-cache downloads (threads
    # or processes) never clobber each other's partial file; the final rename
    # is atomic (last writer wins, both files are identical).
    fd, tmp_name = tempfile.mkstemp(suffix=".part", dir=cache_dir)
    tmp = Path(tmp_name)
    try:
        # fdopen first so the descriptor is always closed (Windows cannot
        # unlink a file that is still open).
        with os.fdopen(fd, "wb") as f:
            response = requests.get(
                url, stream=True, headers=headers, timeout=(10, 60)
            )
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(dest)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to download ReID weights from {url}: {e}\n"
            "You can convert the upstream Torchreid checkpoint yourself with "
            "weights/convert_osnet_reid_weights.py and pass it via "
            "OSNetEmbedder(weights=...), or place it at "
            f"{dest}." + hf_unreachable_hint(e, url)
        ) from e
    return dest


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class OSNetEmbedder:
    """Compute L2-normalized OSNet appearance embeddings for detection boxes.

    Args:
        variant: One of ``osnet_ain_x1_0`` / ``x0_75`` / ``x0_5`` /
            ``x0_25`` (smallest, default: ~0.6 MB of weights, 512-d output).
        weights: Optional path to a checkpoint. When None, the named variant
            is downloaded to ``~/.cache/libreyolo/reid/`` on first use.
        device: Torch device string; auto-selects CUDA when available.

    Example::

        embedder = OSNetEmbedder()
        embs = embedder(frame_rgb, boxes_xyxy)  # (N, 512) float32
    """

    crop_size = (128, 256)  # (width, height), as in ReID convention
    _mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
    _std = np.array((0.229, 0.224, 0.225), dtype=np.float32)

    def __init__(
        self,
        variant: str = "osnet_ain_x0_25",
        weights: str | Path | None = None,
        device: str | None = None,
    ):
        if variant not in _OSNET_CHANNELS:
            raise ValueError(
                f"Unknown OSNet variant {variant!r}; "
                f"choose from {sorted(_OSNET_CHANNELS)}."
            )
        self.variant = variant
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = Path(weights) if weights is not None else _resolve_weights(variant)
        self.model = OSNet(_OSNET_CHANNELS[variant])
        self.model.load_state_dict(_load_osnet_state_dict(path))
        self.model.eval().to(self.device)

    def _preprocess(self, image: np.ndarray, boxes: np.ndarray) -> torch.Tensor:
        h, w = image.shape[:2]
        coords = np.round(boxes[:, :4]).astype(np.int64)
        coords[:, 0] = coords[:, 0].clip(0, w)
        coords[:, 1] = coords[:, 1].clip(0, h)
        coords[:, 2] = coords[:, 2].clip(0, w)
        coords[:, 3] = coords[:, 3].clip(0, h)
        crops = []
        for x1, y1, x2, y2 in coords:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                # Degenerate box (fully outside the frame after clipping):
                # feed a black crop so the batch shape stays consistent.
                crop = np.zeros((2, 2, 3), dtype=image.dtype)
            crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - self._mean) / self._std
            crops.append(crop.transpose(2, 0, 1))
        return torch.from_numpy(np.stack(crops))

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Embed each box crop of ``image``.

        Args:
            image: Full frame as an ``(H, W, 3)`` RGB uint8 array.
            boxes: ``(N, 4+)`` array of ``[x1, y1, x2, y2, ...]`` pixel boxes.

        Returns:
            ``(N, 512)`` float32 array of L2-normalized embeddings.
        """
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.shape[0] == 0:
            return np.zeros((0, 512), dtype=np.float32)
        batch = self._preprocess(image, boxes).to(self.device)
        with torch.no_grad():
            embs = self.model(batch)
        embs = F.normalize(embs, dim=-1)
        return embs.cpu().numpy().astype(np.float32)

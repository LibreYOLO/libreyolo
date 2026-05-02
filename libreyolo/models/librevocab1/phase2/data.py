"""LibreVocab1 Phase 2 — open-vocab dataset, text-embedding cache, collator.

Flow:

    raw COCO sample  ─► OpenVocabCOCODataset ─► {image, boxes_xyxy, class_names, image_id}
                                                            │
                                                            ▼
                                              openvocab_collate(batch, K_neg)
                                                            │
                                                            ▼
                  {images: (B, 3, H, W),
                   targets: list of {labels (N,), boxes_cxcywh (N, 4)},
                   prompt_names: list[str] of length K,
                   text_emb: (K, 1152)}

Targets' ``labels`` are indices into ``prompt_names``, which is the per-batch
"vocabulary." That vocabulary = positives present in the batch + K negative
class names sampled from the rest of the dataset's category list.

The ``text_emb`` is built by looking up cached SigLIP2 text embeddings keyed
by class name. The cache is built once with :class:`TextEmbeddingCache` from
the CRADIOv4 ``siglip2-g`` adaptor's text encoder.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text embedding cache
# ---------------------------------------------------------------------------


@dataclass
class TextEmbeddingCache:
    """On-disk cache of SigLIP2 text embeddings keyed by class name.

    The cache file is a small ``.pt`` containing:
        {"prompt_template": str,
         "siglip2_version": str,
         "embed_dim": int,
         "names": list[str],
         "embeddings": Tensor (N, embed_dim)}.

    Lookup in code is by name; ``names`` is paired index-aligned with
    ``embeddings``.
    """

    prompt_template: str
    siglip2_version: str
    embed_dim: int
    names: List[str]
    embeddings: torch.Tensor  # (N, embed_dim)

    @property
    def name_to_index(self) -> Dict[str, int]:
        return {n: i for i, n in enumerate(self.names)}

    def lookup(self, names: Sequence[str]) -> torch.Tensor:
        """Return ``(len(names), embed_dim)`` rows in the order requested."""
        idx_map = self.name_to_index
        try:
            indices = torch.tensor([idx_map[n] for n in names], dtype=torch.long)
        except KeyError as e:
            raise KeyError(
                f"class name {e.args[0]!r} not in cache; rebuild cache to include it"
            )
        return self.embeddings.index_select(0, indices)

    def save(self, path: str | os.PathLike) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prompt_template": self.prompt_template,
                "siglip2_version": self.siglip2_version,
                "embed_dim": self.embed_dim,
                "names": self.names,
                "embeddings": self.embeddings,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "TextEmbeddingCache":
        d = torch.load(path, map_location="cpu", weights_only=False)
        emb = d["embeddings"]
        if not isinstance(emb, torch.Tensor):
            raise ValueError(f"corrupted cache at {path}: embeddings not a Tensor")
        return cls(
            prompt_template=d["prompt_template"],
            siglip2_version=d["siglip2_version"],
            embed_dim=int(d["embed_dim"]),
            names=list(d["names"]),
            embeddings=emb,
        )

    @classmethod
    def build(
        cls,
        names: Sequence[str],
        encode_text_fn: Callable[[List[str], torch.device], torch.Tensor],
        prompt_template: str = "a photo of a {}",
        siglip2_version: str = "siglip2-g-384",
        device: str | torch.device = "cpu",
        batch_size: int = 64,
    ) -> "TextEmbeddingCache":
        """Encode every name through SigLIP2's text encoder and pack the cache.

        ``encode_text_fn(prompts, device) -> (N, embed_dim)`` is the user-supplied
        encoder. In Phase 2 it comes from
        ``CRADIOv4Backbone.encode_text``.
        """
        device = torch.device(device)
        prompts_all = [prompt_template.format(name) for name in names]
        out_chunks: List[torch.Tensor] = []
        for i in range(0, len(prompts_all), batch_size):
            chunk = prompts_all[i : i + batch_size]
            emb = encode_text_fn(chunk, device).detach().cpu()
            out_chunks.append(emb)
        embeddings = torch.cat(out_chunks, dim=0)
        return cls(
            prompt_template=prompt_template,
            siglip2_version=siglip2_version,
            embed_dim=int(embeddings.shape[-1]),
            names=list(names),
            embeddings=embeddings,
        )


# ---------------------------------------------------------------------------
# Open-vocab COCO wrapper
# ---------------------------------------------------------------------------


class OpenVocabCOCODataset(Dataset):
    """Wraps a COCO-format dataset and yields class-*name* targets.

    Output schema per sample:

        {"image":          Tensor (3, H, W) float in [0, 1],
         "boxes_xyxy":     Tensor (N, 4) pixel coords on the *resized* image,
         "class_names":    list[str] of length N,
         "image_id":       int,
         "image_size":     (H, W) of the resized image}

    Augmentations are *intentionally minimal* in this first cut — the Phase 2
    A100 smoke run is about validating engineering, not pushing mAP. We do
    fixed-resize-with-letterbox to a square ``input_size``. Heavier
    augmentations (mosaic, mixup, scale jitter) can be layered on later via a
    ``transform=`` callable on the wrapped dataset before this wrapper runs.
    """

    def __init__(
        self,
        data_dir: str,
        json_file: str = "instances_train2017.json",
        name: str = "train2017",
        input_size: int = 640,
    ) -> None:
        try:
            from pycocotools.coco import COCO
        except ImportError as e:
            raise ImportError(
                "pycocotools is required for OpenVocabCOCODataset; "
                "install with: pip install pycocotools"
            ) from e

        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.input_size = int(input_size)

        ann_file = os.path.join(data_dir, "annotations", json_file)
        self.coco = COCO(ann_file)
        self.image_ids = self.coco.getImgIds()
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._cat_id_to_name: Dict[int, str] = {c["id"]: c["name"] for c in cats}
        # Stable name list (sorted by COCO category id ordering).
        cats_sorted = sorted(cats, key=lambda c: c["id"])
        self.class_names: List[str] = [c["name"] for c in cats_sorted]

    def __len__(self) -> int:
        return len(self.image_ids)

    def _load_image(self, img_info: Dict[str, Any]) -> np.ndarray:
        path = os.path.join(self.data_dir, self.name, img_info["file_name"])
        # cv2 import is lazy so unit tests that don't open images don't pay.
        import cv2
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image at {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _resize_letterbox(
        self, image: np.ndarray, boxes_xyxy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize to ``self.input_size`` square with letterbox padding.

        Boxes are scaled by the same factor and translated by the pad offset.
        """
        import cv2
        h0, w0 = image.shape[:2]
        s = self.input_size
        scale = min(s / w0, s / h0)
        new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad = np.full((s, s, 3), 114, dtype=np.uint8)
        top = (s - new_h) // 2
        left = (s - new_w) // 2
        pad[top : top + new_h, left : left + new_w] = resized
        if boxes_xyxy.size:
            boxes_xyxy = boxes_xyxy * scale
            boxes_xyxy[:, [0, 2]] += left
            boxes_xyxy[:, [1, 3]] += top
        return pad, boxes_xyxy

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image = self._load_image(img_info)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        # COCO bbox = [x, y, w, h] in pixels; convert to xyxy.
        if anns:
            xywh = np.array([a["bbox"] for a in anns], dtype=np.float32)
            xyxy = xywh.copy()
            xyxy[:, 2] = xywh[:, 0] + xywh[:, 2]
            xyxy[:, 3] = xywh[:, 1] + xywh[:, 3]
            class_names = [self._cat_id_to_name[a["category_id"]] for a in anns]
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            class_names = []

        image, xyxy = self._resize_letterbox(image, xyxy)
        # Drop boxes that were fully clipped or have zero area after resize.
        keep = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        xyxy = xyxy[keep]
        class_names = [n for k, n in zip(keep.tolist(), class_names) if k]

        image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0
        return {
            "image": image_t,
            "boxes_xyxy": torch.from_numpy(xyxy.astype(np.float32)),
            "class_names": class_names,
            "image_id": int(img_id),
            "image_size": (self.input_size, self.input_size),
        }


# ---------------------------------------------------------------------------
# Per-batch open-vocab collator
# ---------------------------------------------------------------------------


def _xyxy_to_cxcywh_normalized(
    xyxy: torch.Tensor, hw: Tuple[int, int]
) -> torch.Tensor:
    """Pixel xyxy on a (H, W) canvas → cxcywh normalized to [0, 1]."""
    if xyxy.numel() == 0:
        return xyxy.new_zeros((0, 4))
    h, w = hw
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 / w
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2 / h
    bw = (xyxy[:, 2] - xyxy[:, 0]) / w
    bh = (xyxy[:, 3] - xyxy[:, 1]) / h
    return torch.stack([cx, cy, bw, bh], dim=1)


class OpenVocabCollator:
    """Builds the per-batch open-vocab batch dict.

    Args:
        text_emb_cache: pre-computed text embeddings.
        full_class_names: every class name in the dataset's vocabulary.
            Used as the negative-sample pool.
        num_negatives: how many negative class names to add to the per-batch
            prompt list, on top of the positives present in the batch.
        rng_seed: optional fixed seed for negative sampling (reproducible
            tests). When None, samples without seeding.
    """

    def __init__(
        self,
        text_emb_cache: TextEmbeddingCache,
        full_class_names: Sequence[str],
        num_negatives: int = 32,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.cache = text_emb_cache
        self.full_class_names = list(full_class_names)
        self.num_negatives = int(num_negatives)
        self._rng = random.Random(rng_seed)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Stack images. All images come from the same OpenVocabCOCODataset and
        # were resized to the same canvas, so a plain stack works.
        images = torch.stack([s["image"] for s in samples], dim=0)
        hw = samples[0]["image_size"]

        # Collect positives (preserve order, deduplicate).
        positives: List[str] = []
        seen = set()
        for s in samples:
            for n in s["class_names"]:
                if n not in seen:
                    positives.append(n)
                    seen.add(n)

        # Sample negatives from the rest of the vocabulary.
        candidates = [n for n in self.full_class_names if n not in seen]
        n_neg = min(self.num_negatives, len(candidates))
        negatives = self._rng.sample(candidates, n_neg) if n_neg > 0 else []

        prompt_names = positives + negatives
        # Empty edge case: if no positives in batch, still keep at least one
        # prompt so K > 0 and the decoder has something to score against.
        if not prompt_names:
            prompt_names = list(self.full_class_names[:1])
        text_emb = self.cache.lookup(prompt_names)

        name_to_idx = {n: i for i, n in enumerate(prompt_names)}
        targets: List[Dict[str, torch.Tensor]] = []
        for s in samples:
            labels = torch.tensor(
                [name_to_idx[n] for n in s["class_names"]], dtype=torch.long
            )
            boxes_cxcywh = _xyxy_to_cxcywh_normalized(s["boxes_xyxy"], hw)
            targets.append({"labels": labels, "boxes_cxcywh": boxes_cxcywh})

        return {
            "images": images,
            "targets": targets,
            "prompt_names": prompt_names,
            "text_emb": text_emb,
            "image_ids": [s["image_id"] for s in samples],
            "image_size": hw,
        }

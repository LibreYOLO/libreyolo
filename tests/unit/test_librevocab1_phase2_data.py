"""LibreVocab1 Phase 2 — open-vocab dataset / collator / text-cache tests.

Goes deep on the collator + cache because that's where the open-vocab
plumbing lives. The COCO dataset wrapper itself is exercised via a tiny
synthetic in-memory COCO instance (no real images downloaded).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


_REQUIRED = ("scipy",)
_MISSING = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        bool(_MISSING),
        reason=f"libreyolo[vocab1] extras missing: {_MISSING!r}",
    ),
]


def _fake_encode(prompts, device):
    """Deterministic stand-in for SigLIP2 encode_text: hashes -> 8-dim vectors."""
    import torch
    out = torch.zeros(len(prompts), 8)
    for i, p in enumerate(prompts):
        h = abs(hash(p)) % (2**31)
        for j in range(8):
            out[i, j] = ((h >> (j * 3)) & 0x7) / 7.0
    return out


def test_text_emb_cache_build_lookup_save_load(tmp_path):
    import torch
    from libreyolo.models.librevocab1.phase2.data import TextEmbeddingCache

    names = ["dog", "cat", "stop sign"]
    cache = TextEmbeddingCache.build(
        names=names,
        encode_text_fn=_fake_encode,
        prompt_template="a photo of a {}",
        siglip2_version="fake",
        batch_size=2,
    )
    assert cache.embeddings.shape == (3, 8)
    assert cache.lookup(["cat", "dog"]).shape == (2, 8)
    # Round-trip via disk.
    p = tmp_path / "cache.pt"
    cache.save(p)
    loaded = TextEmbeddingCache.load(p)
    assert loaded.names == names
    assert torch.allclose(loaded.embeddings, cache.embeddings)


def test_text_emb_cache_lookup_raises_on_missing():
    from libreyolo.models.librevocab1.phase2.data import TextEmbeddingCache

    cache = TextEmbeddingCache.build(["dog"], _fake_encode)
    with pytest.raises(KeyError):
        cache.lookup(["unknown_class"])


def _build_synthetic_coco(tmp_path: Path) -> Path:
    """Write a tiny in-memory COCO instance to disk with 4 images,
    3 categories, and a few annotations. Returns the dataset root."""
    import cv2
    root = tmp_path / "tinycoco"
    (root / "annotations").mkdir(parents=True)
    (root / "train").mkdir(parents=True)

    # 4 random color images, 256x256.
    for i in range(4):
        img = (np.random.RandomState(i).rand(256, 256, 3) * 255).astype(np.uint8)
        cv2.imwrite(str(root / "train" / f"img{i}.jpg"), img)

    coco_json = {
        "images": [
            {"id": i, "file_name": f"img{i}.jpg", "width": 256, "height": 256}
            for i in range(4)
        ],
        "categories": [
            {"id": 1, "name": "dog"},
            {"id": 2, "name": "cat"},
            {"id": 3, "name": "tree"},
        ],
        "annotations": [
            # img 0: one dog.
            {"id": 100, "image_id": 0, "category_id": 1,
             "bbox": [10, 20, 80, 100], "area": 8000, "iscrowd": 0},
            # img 1: one dog + one cat.
            {"id": 101, "image_id": 1, "category_id": 1,
             "bbox": [50, 50, 60, 60], "area": 3600, "iscrowd": 0},
            {"id": 102, "image_id": 1, "category_id": 2,
             "bbox": [120, 120, 80, 80], "area": 6400, "iscrowd": 0},
            # img 2: one tree.
            {"id": 103, "image_id": 2, "category_id": 3,
             "bbox": [30, 30, 100, 100], "area": 10000, "iscrowd": 0},
            # img 3: empty (no annotations).
        ],
    }
    with open(root / "annotations" / "instances_train.json", "w") as f:
        json.dump(coco_json, f)
    return root


def test_openvocab_coco_dataset_yields_class_names(tmp_path):
    pytest.importorskip("pycocotools")
    pytest.importorskip("cv2")
    import torch
    from libreyolo.models.librevocab1.phase2.data import OpenVocabCOCODataset

    root = _build_synthetic_coco(tmp_path)
    ds = OpenVocabCOCODataset(
        data_dir=str(root),
        json_file="instances_train.json",
        name="train",
        input_size=320,
    )
    assert len(ds) == 4
    assert ds.class_names == ["dog", "cat", "tree"]

    # img 1 has dog + cat.
    s1 = ds[1]
    assert isinstance(s1["image"], torch.Tensor)
    assert s1["image"].shape == (3, 320, 320)
    assert s1["image"].dtype == torch.float32
    assert s1["boxes_xyxy"].shape == (2, 4)
    assert set(s1["class_names"]) == {"dog", "cat"}
    assert s1["image_id"] == 1

    # img 3 is empty.
    s3 = ds[3]
    assert s3["boxes_xyxy"].shape == (0, 4)
    assert s3["class_names"] == []


def test_openvocab_collator_builds_per_batch_vocab(tmp_path):
    pytest.importorskip("pycocotools")
    pytest.importorskip("cv2")
    import torch
    from libreyolo.models.librevocab1.phase2.data import (
        OpenVocabCOCODataset,
        OpenVocabCollator,
        TextEmbeddingCache,
    )

    root = _build_synthetic_coco(tmp_path)
    ds = OpenVocabCOCODataset(
        data_dir=str(root),
        json_file="instances_train.json",
        name="train",
        input_size=320,
    )
    cache = TextEmbeddingCache.build(ds.class_names, _fake_encode)
    collator = OpenVocabCollator(
        text_emb_cache=cache,
        full_class_names=ds.class_names,
        num_negatives=2,  # we have 3 categories total; max 2 negatives possible
        rng_seed=0,
    )

    # Batch with images 0 and 1 → positives = {dog, cat}, negatives drawn from
    # the rest (just "tree" remains).
    batch = collator([ds[0], ds[1]])

    assert batch["images"].shape == (2, 3, 320, 320)
    assert isinstance(batch["prompt_names"], list)
    # positives "dog" and "cat" must be present.
    assert "dog" in batch["prompt_names"]
    assert "cat" in batch["prompt_names"]
    # text_emb dim = K x 8 (our fake encoder outputs 8-d).
    K = len(batch["prompt_names"])
    assert batch["text_emb"].shape == (K, 8)

    # Each target's labels are indices into prompt_names.
    name_to_idx = {n: i for i, n in enumerate(batch["prompt_names"])}
    for sample, target in zip([ds[0], ds[1]], batch["targets"]):
        expected = [name_to_idx[n] for n in sample["class_names"]]
        assert target["labels"].tolist() == expected
        assert target["boxes_cxcywh"].shape == (len(sample["class_names"]), 4)
        # cxcywh values normalized to [0, 1].
        if target["boxes_cxcywh"].numel() > 0:
            assert (target["boxes_cxcywh"] >= 0).all()
            assert (target["boxes_cxcywh"] <= 1).all()


def test_collator_handles_all_empty_batch(tmp_path):
    """Pure no-annotation batch must still yield K >= 1 prompts."""
    pytest.importorskip("pycocotools")
    pytest.importorskip("cv2")
    from libreyolo.models.librevocab1.phase2.data import (
        OpenVocabCOCODataset,
        OpenVocabCollator,
        TextEmbeddingCache,
    )

    root = _build_synthetic_coco(tmp_path)
    ds = OpenVocabCOCODataset(
        data_dir=str(root),
        json_file="instances_train.json",
        name="train",
        input_size=320,
    )
    cache = TextEmbeddingCache.build(ds.class_names, _fake_encode)
    # num_negatives = 0 to make the test deterministic; only image 3 (empty).
    collator = OpenVocabCollator(
        text_emb_cache=cache,
        full_class_names=ds.class_names,
        num_negatives=0,
        rng_seed=0,
    )
    batch = collator([ds[3]])
    assert len(batch["prompt_names"]) >= 1
    assert batch["text_emb"].shape[0] == len(batch["prompt_names"])
    assert batch["targets"][0]["labels"].numel() == 0


def test_collator_output_feeds_loss_pipeline_end_to_end(tmp_path):
    """Wire collator output through the matcher + loss to confirm shapes."""
    pytest.importorskip("pycocotools")
    pytest.importorskip("cv2")
    import torch
    from libreyolo.models.librevocab1.phase2.data import (
        OpenVocabCOCODataset,
        OpenVocabCollator,
        TextEmbeddingCache,
    )
    from libreyolo.models.librevocab1.phase2.matcher import OpenVocabHungarianMatcher
    from libreyolo.models.librevocab1.phase2.loss import OpenVocabSetCriterion

    root = _build_synthetic_coco(tmp_path)
    ds = OpenVocabCOCODataset(
        data_dir=str(root),
        json_file="instances_train.json",
        name="train",
        input_size=320,
    )
    cache = TextEmbeddingCache.build(ds.class_names, _fake_encode)
    collator = OpenVocabCollator(
        text_emb_cache=cache,
        full_class_names=ds.class_names,
        num_negatives=1,
        rng_seed=0,
    )
    batch = collator([ds[0], ds[1], ds[2]])

    B, K = batch["images"].shape[0], len(batch["prompt_names"])
    Q = 12
    # Synthesize decoder outputs of the right shape.
    pred_logits = torch.randn(B, Q, K, requires_grad=True)
    pred_boxes = torch.rand(B, Q, 4) * 0.5 + 0.25
    pred_boxes = pred_boxes.detach().requires_grad_(True)

    criterion = OpenVocabSetCriterion(OpenVocabHungarianMatcher())
    ld = criterion(
        {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
        batch["targets"],
    )
    total = ld["loss_cls"] + ld["loss_bbox"] + ld["loss_giou"]
    total.backward()
    assert torch.isfinite(total)
    assert pred_logits.grad is not None
    assert pred_boxes.grad is not None

# LibreVocab1 — Phase 2 plan

**Status**: Phase 1 (SAM3 head) shipped on branch `160-investigate-the-viability-of-creating-our-first-custom-model-libredet1`, commit `52da9a09`. Inference parity confirmed on CPU (parkour.jpg → two person boxes, ~48 s).

**Goal of Phase 2**: replace the SAM3 decoder + text encoder + mask head with a LibreYOLO-owned, permissively licensed open-vocab detection head trained from scratch. Vision encoder stays CRADIOv4 (frozen).

**Why**: Phase 1 proved the encoder works. Phase 2 is what makes the model a *flagship* rather than a "wrapper around SAM3 with a swapped trunk."

## Architectural target

```
Image ─► CRADIOv4 (frozen)
            │
            ├─► forward_intermediates([L1, L2, L3])
            │       ↓ (3 stride-16 spatial maps, dim 1152 SO400M / 1280 H)
            │       ↓ DEIMv2 SpatialPriorModulev2 + 1×1 conv + SyncBN
            │       ↓
            │   3-scale pyramid (1/8, 1/16, 1/32)
            │       ↓
            │   RT-DETRv2 HybridEncoder
            │       ↓
            │   RT-DETRv2 TransformerDecoder (300 queries, 6 layers, text cross-attn)
            │
            └─► siglip2-g adaptor (frozen)
                       ↓
                  encode_text(["a photo of a {class}"]) ─► (K, 1152)
                       ↓
                  text_proj                              ─► (K, 256)
                       ↓
                       └─► fed to decoder cross-attn + classification head

Decoder query qᵢ:
    cls_logitᵢ,k = (qproj(qᵢ) @ text_emb_k) / temperature
    boxᵢ         = box_head(qᵢ)  (sigmoid cxcywh)
```

**License composition target (Phase 2 final state):**
- CRADIOv4 weights: NVIDIA Open Model License (commercial-permissive)
- SigLIP2 text encoder: Apache 2.0
- DEIMv2-derived neck/encoder/decoder: Apache 2.0
- LibreYOLO glue + trained head weights: MIT
- **No SAM License surface in Phase 2**.

## Validation gate: A100 6-hour smoke run

Before committing the 8×H200 week, run a 6-hour A100 engineering smoke test on COCO. Goal is not mAP, it's "does the pipeline run cleanly end-to-end."

### Validation criteria

| Check | Pass criterion |
|---|---|
| Forward pass on synthetic input | shapes correct, no NaN |
| Backward pass + optimizer step | gradient norm finite, optimizer state updates |
| Loss decreases monotonically over ~1k steps | loss[1000] < loss[100] · 0.7 |
| GPU utilization | >70% sustained (otherwise dataloader bound) |
| Validation on COCO val5k after 1 epoch | closed-vocab mAP ≥ 10 (just "not random") |
| Visual eval on 10 held-out images | most produce ≥1 sane box per labeled class |
| Checkpoint save/load round-trip | reload + forward matches pre-save output bit-exactly |

If all 7 pass, Phase 2 engineering is sound and we proceed to H200 week.

If any fail, debug locally, re-run on the A100 (or a remaining slot in the same allocation) before scaling up.

### A100 plan (6 hours wall-clock)

| Hour | Task |
|---|---|
| 0:00–0:30 | Single-batch forward sanity check. Shapes, no NaN. |
| 0:30–3:30 | 1 epoch COCO train2017, batch 8, 640 px, frozen backbone, AdamW lr 1e-4. |
| 3:30–4:00 | Validate on COCO val5k. mAP. |
| 4:00–4:30 | Visual eval on 10 held-out images. Save with predicted boxes. |
| 4:30–6:00 | Spare slot for debugging / second epoch / config sweep. |

## H200 production plan (week of [TBD])

Compute: 8× H200 (141 GB HBM3e each), available for ~1 week.

### Round 1 — Single config, fast turnaround
- Backbone: CRADIOv4-SO400M, frozen
- Decoder: DEIMv2-derived (RT-DETRv2 hybrid encoder + 6-layer transformer decoder)
- Text encoder: SigLIP2-g (loaded via CRADIOv4's siglip2-g adaptor — no extra download)
- Dataset: Objects365 v2, full
- Schedule: 12 epochs, batch 32/GPU = effective batch 256, 800 px input
- Optimizer: AdamW lr 1e-4, weight decay 1e-4, betas (0.9, 0.999)
- Loss weights: VFL 1.0 + L1 5.0 + GIoU 2.0 (DEIMv2 defaults)
- Wall-clock estimate: 24–36 h
- Eval: COCO val (closed-vocab mAP), LVIS mini-val (rare-class AP_r)

**Round 1 decision rule:**
| COCO mAP | LVIS-rare AP_r | Outcome |
|---|---|---|
| ≥ 45 | ≥ 22 | Phase 2 is the flagship. Ship as `LibreVocab1` v2 (or rename Phase 1 to `LibreVocab1Sam3`). |
| ≥ 45 | < 22 | Backbone fine, open-vocab story needs more work. Add Phase 2.5 (GRiT-style grounding pretrain). |
| < 45 | * | Recipe wrong. Inspect loss curves, try LoRA on backbone, retry with same time budget. |

### Round 2 — Ablation grid (if time + Round 1 passes)
With 8× H200 and ~3-day Round 1 wall-clock, the rest of the week buys 1–2 ablation runs in parallel pairs (4 GPUs each):

| Variant | Hypothesis tested |
|---|---|
| SO400M LoRA r=16 | Does mild backbone unfreezing close the gap to fine-tuned baselines? |
| CRADIOv4-H frozen | Bigger backbone enough to compensate for whatever LoRA fixes? |
| 1024 px input | High-res frontier — small/dense object benefit vs latency cost. |
| ViTDet-16 enabled | Throughput improvement at high-res, accuracy parity check. |

### Round 3 — Polish (only if Round 2 produces a clear winner)
- Take the best variant, longer schedule (24 epochs), full GRiT pretrain (if time permits).
- This is the "publishable numbers" run, and the candidate flagship checkpoint.

## Engineering work to do *before* H200 week

All on local CPU/laptop. Estimated 4 days. Goal: by Friday, the H200 box can run `pytest tests/e2e/test_librevocab1_phase2.py` and start training Monday morning.

### Day 1 — Phase 2 model file
**Branch**: new branch off `160-...`, e.g. `160-librevocab1-phase2`.

**Files**:
- `libreyolo/models/librevocab1/phase2/__init__.py`
- `libreyolo/models/librevocab1/phase2/model.py` — `LibreVocab1Phase2(BaseModel)`
- `libreyolo/models/librevocab1/phase2/nn.py` — assembled architecture: CRADIOv4 backbone + DEIMv2 SPMv2 + RT-DETRv2 HybridEncoder + open-vocab decoder

**Reuse**:
- CRADIOv4 loader and `forward_intermediates` from `librevocab1/sam3/sam3_radio_utils.py` (or equivalent)
- DEIMv2 RT-DETRv2 decoder + SPMv2 from `libreyolo/models/deimv2/` (branch `151-add-deimv2`, currently merged or to-be-merged)
- SigLIP2 text encoder from CRADIOv4's `adaptors['siglip2-g'].text_model`

**Tests**:
- `tests/unit/test_librevocab1_phase2_arch.py` — synthetic forward pass on CPU, no weight downloads, like Phase 1's `test_librevocab1_arch.py`.

### Day 2 — Open-vocab loss + matcher
**Files**:
- `libreyolo/models/librevocab1/phase2/loss.py`
- `libreyolo/models/librevocab1/phase2/matcher.py`

**Pieces**:
- `OpenVocabHungarianMatcher`: Hungarian assignment with cost = focal-on-cosine + L1 + GIoU. Cosine cost replaces standard BCE class cost.
- `OpenVocabSetCriterion`: VFL on `(query_proj @ text_emb.T) / temperature` matched targets, L1 + GIoU on boxes.
- Per-image text-embedding pre-encode: cache text embeddings keyed by `(class_name, prompt_template)`.

**Tests**:
- `tests/unit/test_librevocab1_phase2_loss.py` — synthetic targets, verify the matcher returns a valid permutation, the loss is finite and decreasing over a few gradient steps on a single batch.

### Day 3 — Open-vocab dataset wrapper
**Files**:
- `libreyolo/data/openvocab_coco.py` — wraps `COCODataset`
- `libreyolo/data/openvocab_objects365.py` — `Objects365OpenVocabDataset` (used in H200 week, but written now)

**Output schema** per sample:
```python
{
    "image": Tensor (3, H, W) normalized,
    "boxes_xyxy": Tensor (N, 4),
    "class_names": list[str] of length N,
    "image_id": int,
}
```

**Per-batch sampler** for open-vocab training:
- Collect class names present in batch as positives.
- Sample `K` negatives (default 32) from the rest of the dataset's class set.
- Build a list `prompt_classes = positives + negatives` (deduplicated).
- This is the per-batch "vocabulary" the matcher matches against.

**Cached text embeddings**:
- One-time precompute: encode every class name with SigLIP2 text encoder, save to `cache/siglip2_text_embeddings_{dataset}.pt`. Avoids re-encoding 80–365 strings per batch.
- Lookup: dict[class_name, Tensor(1152)].

### Day 4 — Trainer + training script
**Files**:
- `libreyolo/models/librevocab1/phase2/trainer.py` — `LibreVocab1Phase2Trainer(BaseTrainer)`
- `libreyolo/training/config.py` — append `LibreVocab1Phase2Config(TrainConfig)`
- `tools/train_librevocab1_phase2.py` — CLI entry point
- Single config YAML for the A100 6-hour run: `configs/librevocab1_phase2_so400m_coco_smoketest.yaml`
- Single config YAML for the H200 Round 1 run: `configs/librevocab1_phase2_so400m_o365.yaml`

**Smoke tests** (CPU):
- `tests/e2e/test_librevocab1_phase2_overfit.py` — overfit a single image with one class for 200 steps; verify loss → 0 and the model produces correct boxes for that image. Gated on `LIBREVOCAB1_PHASE2_OVERFIT=1`.

## Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Frozen CRADIOv4 underfits the detection objective | medium | Add LoRA (r=16) on backbone as a Round-2 ablation. |
| SigLIP2 text features not crisp enough for patch grounding | medium | Add CLIP-L text path as an alternative; bench head-to-head in Round 2. |
| Objects365 download/storage too slow on H200 box | medium | Pre-stage data the weekend before the H200 week. |
| Dataloader CPU-bound, GPU starved | medium | Validate on A100 smoke run; fix `num_workers`, prefetch_factor, persistent_workers if util < 70%. |
| Class-name text encoding redundancy at training time | low | Pre-cache embeddings to disk. |
| Hungarian matcher slow with 365 classes | low | Standard issue; use scipy LAP or torchvision's matcher; not a real bottleneck. |
| First trained Phase 2 model produces nothing useful | high | A100 smoke run catches this *before* H200 week. Decision rule says abandon if COCO mAP < 45. |

## Decision points sequencing

1. **Now**: approve Phase 2 scope and start Day 1 engineering.
2. **End of week (Friday)**: all Phase 2 code lands on branch. Local CPU smoke tests green. Ready to run on A100.
3. **A100 6-hour run** (this week if available, else early next week): validation gate.
4. **H200 Round 1 (next week)**: production pretraining run. ~24–36 h wall-clock.
5. **End of H200 week**: Round 1 results + at most 1–2 ablations.
6. **Decision after H200 week**: ship Phase 2 as `LibreVocab1` v2, ship as ablation, or extend with Phase 2.5.

## What this plan does *not* commit to

- No promises on COCO/LVIS numbers until after H200 Round 1.
- No commitment to ship Phase 2 if Round 1 results are weak — Phase 1 is independently shippable.
- No CLI / API surface changes for end users until the model is validated.
- No checkpoint redistribution until the model is validated.
- No fine-tuning / training UX for end users in Phase 2; that's later.

## What we know works today

- Phase 1 inference on CPU, ~48 s/image, real boxes on parkour.jpg.
- CRADIOv4-SO400M downloads cleanly from torch.hub (~1.6 GB, ~85 MB/s).
- SAM3 weights download cleanly from HF after access approval.
- LibreYOLO core not affected by LibreVocab1 inclusion (factory still routes correctly to existing families).
- 5 unit tests + 1 e2e test, all green, on the working CPU pipeline.

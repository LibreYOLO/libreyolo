# VLM detection fine-tuning

Fine-tune a LibreVLM detector on a standard detect dataset. First trainable
family: Qwen3-VL. Everything below is the shipped contract; the wider design
rationale lives in `docs/adr/0002-librevlm-contract.md` (tier contract) and the
training ADR draft.

## Use

```python
from libreyolo import LibreVLM

model = LibreVLM("qwen3-vl-2b")
results = model.train(data="strawberries.yaml", epochs=10)

model = LibreVLM(results["best"])       # load the fine-tune
model.predict("field.jpg")              # vocabulary comes from the dataset
```

Needs the optional extra:

```
pip install "libreyolo[vlm-train]"
```

- `data` is the standard detect dataset (`docs/dataset_schema.md`): YAML with
  `train`/`val` image sources and txt label rows. Native COCO JSON datasets
  are not supported yet and raise with a clear message.
- The `names` in the YAML become the training vocabulary. Names may be any
  phrases ("ripe strawberry", "person wearing a helmet"); they are rendered
  into the model's own grounding prompt and answer format automatically, using
  the same `BBOX_KEY` / `COORD_DIVISOR` / `BOX_FORMAT` convention the family's
  `predict()` parser uses. Train and predict cannot drift apart, by
  construction.
- `lora=True` is the default and the recommended path. The recipe (rank 16,
  alpha 32, language-model attention and MLP projections, vision tower frozen)
  is fixed per family in `libreyolo/models/vlm/training/recipes.py`, not a
  user-facing knob. `lora=False` full fine-tuning exists behind a VRAM
  preflight.
- Defaults reflect VLM reality: `batch=1`, `accumulate=8` (effective batch 8),
  gradient checkpointing on, cosine schedule with warmup, bf16 autocast on
  CUDA. There is no `imgsz`; the family's processor owns resolution.
- Augmentation is geometric-safe only (`hflip=0.5`), because every geometric
  transform re-renders the target text. Mosaic-style compositing does not
  apply to generative targets.
- `resume=` continues from a prior adapter checkpoint (weights only; the
  optimizer state starts fresh, and the log says so). `resume=True` on a
  model loaded from `<run>/weights/last` continues and writes into that run.
- `callbacks=` and `loggers=` are the standard training layers; TensorBoard,
  MLflow, and W&B loggers work unchanged.

## Checkpoints

A VLM checkpoint is a directory, not a `.pt`:

```
runs/vlm/train/weights/best/
  adapter_model.safetensors    # LoRA adapter (megabytes)
  adapter_config.json
  ...processor files...
  libreyolo_vlm.json           # the contract file
```

`libreyolo_vlm.json` records family, size, the exact base repo and revision
the adapter was trained on, the ordered vocabulary, the coordinate convention,
and the run metrics. `LibreVLM(path)` recognizes the contract, downloads the
recorded base (never a drifted newer snapshot), merges the adapter for
inference speed, and pre-applies the saved vocabulary. Base weights are never
copied into checkpoints, so LibreYOLO never redistributes upstream weights.

Full fine-tunes (`lora=False`) save a self-contained model directory instead;
the same `LibreVLM(path)` call loads either kind.

## Best/last selection

`best` tracks validation loss when the dataset has a `val` split, else
training loss. Validation mAP for VLM checkpoints lands together with the
tier's `val()` support (blocked on real per-box confidence; see the ADR).

## Trainable families

| Family | train() | Why |
|---|---|---|
| qwen3-vl | yes | grounding-pretrained, Apache-2.0, official upstream recipe mirrored |
| lfm2-vl, florence-2, north-micro-vision, internvl3, gemma-4, moondream | not yet | planned; see the training ADR draft |
| smolvlm2 | no | not grounding-pretrained; cannot reliably learn box emission |
| kosmos-2 | no | no established recipe; 224px 32x32 grid caps localization |
| locate-anything | no | research-only non-commercial upstream weights |
| sensenova-vision | no | multi-node-scale training only, no parameter-efficient path |
| libremodus | no | upstream ships a pretraining pipeline; gated research-only weights |

The refusal messages on the untrainable families state these reasons.

## Implementation notes

- Trainer: `libreyolo/models/vlm/training/` (dataset rendering, chat-template
  collation with longest-common-prefix label masking, PEFT LoRA injection with
  scope assertions, compact loop). Not a `BaseTrainer` subclass: stacked-tensor
  batching, EMA, and mAP-based epoch logic do not apply to a generative
  fine-tune.
- Label masking tolerates boundary retokenization (a template ending in a bare
  newline can merge that newline into the first answer token) but fails loudly
  if a chat template rewrites the user turn when an answer is appended.
- Tests: `tests/unit/test_vlm_training.py` (offline: serialization round-trip
  through the inference parser, collator masking, contract, gating) and
  `tests/e2e/test_vlm_train_qwen3vl.py` (CPU end-to-end on a pinned tiny-random
  Qwen3-VL: train, checkpoint, factory reload, predict).

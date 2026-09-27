---
name: use-libreyolo-zero-shot
description: >-
  Use LibreYOLO's zero-shot and promptable tiers: open-vocabulary detection
  with text vocabularies (LibreOpenVocab: Grounding DINO, OWLv2), promptable
  segmentation with points/boxes or concept text (LibreSAM: SAM-1, SAM-2,
  SAM 3, MobileSAM),
  zero-shot classification (LibreCLIP / LibreSigLIP2 set_classes), VLM-as-detector
  (LibreVLM), and instruction-to-click grounding (LibreGround). Use when someone
  wants to detect arbitrary text-described
  classes without training ("find the forklifts", custom vocabulary),
  click a UI element from a screenshot ("where is Bluetooth"),
  click-to-segment / box-to-mask, segment-everything, open-set
  classification, or asks which zero-shot model to pick. Covers choosing a
  tier, each tier's API and gotchas, extras to install, and honest guidance
  on speed/calibration. For the standard trained detectors see
  use-libreyolo.
---

# Zero-shot and promptable LibreYOLO

Five tiers answer "no training data, still want results". They are separate
factories from `LibreYOLO(...)`, with snapshot-style weights (a downloaded
directory, not a `Libre*<size>.pt` checkpoint).

## Pick the tier by question

| You want | Tier | Install |
|---|---|---|
| Boxes for classes described in text | `LibreOpenVocab` | `libreyolo[openvocab]` |
| Mask from a click / box / concept text, or segment-everything | `LibreSAM` | `libreyolo[sam]` |
| Whole-image label from your own label set | `LibreCLIP` / `LibreSigLIP2` | `libreyolo[clip]` / `libreyolo[siglip2]` |
| Ask an instruction-following model to find things (slow, flexible) | `LibreVLM` | `libreyolo[vlm]` |
| Click coordinates from a screenshot + instruction | `LibreGround` | `libreyolo[vlm]` |

Combinations are normal: LibreOpenVocab boxes fed to LibreSAM as box prompts
gives text-to-mask ("segment the forklifts") without training anything.

If the classes are known in advance and latency matters, a fine-tuned
regular detector beats all of these at runtime; zero-shot is for open or
changing vocabularies and for prototyping before labeling.

## LibreOpenVocab (text-vocabulary detection)

Discriminative text-conditioned detectors (not VLMs): real boxes, real
calibrated-ish scores, no text generation. Contract: `docs/adr/0008` and
`docs/openvocab_design.md`.

```python
from libreyolo import LibreOpenVocab

model = LibreOpenVocab("grounding-dino")      # or "owlv2"; aliases in
model.set_classes(["cat", "dog", "remote control"])   # sticky vocabulary
r = model.predict("a.jpg", conf=0.25)         # standard Results: r.boxes, r.names
```

- Vocabulary is **sticky** across predicts; defaults to COCO-80 when
  `set_classes` was never called. `set_classes` rejects a bare string,
  empties, and case-insensitive duplicates.
- Class names in prompts work best as short noun phrases; "remote control"
  outperforms "remote".
- **Grounding DINO** extras: `text_threshold=` (token threshold) on predict;
  long vocabularies are automatically chunked into multiple prompts and
  merged, so big label lists work but cost one forward per chunk. Phrase to
  class mapping is deliberately conservative: unmatched/ambiguous phrases
  are dropped rather than guessed, so a missing detection can be a mapping
  drop, lower `conf`/`text_threshold` and check labels before blaming the
  detector.
- **OWLv2** has no `text_threshold` and rejects it. Ensemble models, strong
  on rare-object recall.
- Scores across the two families are not comparable; tune `conf` per
  family, per vocabulary.
- Sizes: `grounding-dino` t/b; `owlv2` b16/l14. Weights auto-download from
  the LibreYOLO HF mirrors into `weights/Libre<Family><size>/` directories.

## LibreSAM (promptable segmentation)

Contract: `docs/adr/0007`. Point/box prompts to masks, or segment-everything.

```python
from libreyolo import LibreSAM

model = LibreSAM("base")                            # SAM-1 (Apache-2.0), default tier
r = model.predict("img.jpg", points=[900, 370], labels=[1])   # 1=foreground, 0=background
r = model.predict("img.jpg", bboxes=[100, 100, 200, 200])
r = model.predict("img.jpg")                        # segment everything (slow)
r.masks.xy; r.boxes.xyxy                            # boxes derived from masks

sam3 = LibreSAM("sam3")
matches = sam3.predict("img.jpg", text="yellow school bus", conf=0.3)

model.set_image("img.jpg")                          # encode once (the expensive part)...
a = model.predict(points=[500, 375], labels=[1])    # ...prompt many, cheap
model.reset_image()
```

- Family pick: `LibreSAM("base"/"large"/"huge")` = SAM-1; `LibreSAM2`
  aliases for SAM-2 (better masks, video-capable lineage);
  `LibreSAM("sam3")` = SAM 3 visual prompts plus concept `text=` prompts;
  `LibreMobileSAM` (tiny encoder, edge/CPU-friendly, same prompt API).
  The alias table in `libreyolo/models/sam/model.py` is authoritative.
- SAM 3 weights come directly from the gated `facebook/sam3` repository under
  Meta's custom SAM License, not MIT or Apache-2.0. Accept its terms and run
  `hf auth login` (or set `HF_TOKEN`) before first use. The first `text=` call
  lazily loads a second model instance and therefore raises peak RAM/VRAM.
  In a reference CPU/fp32 run at the native 1008 px frame, RSS was 3.0 GB after
  visual inference and 5.9 GB with both models resident, with a 9.0 GB peak
  while loading/running the first text prompt. An 8 GB host may exhaust memory
  on the text path even when visual prompting works comfortably.
- `text=` returns every instance matching the concept and cannot be combined
  with points or boxes. Its `conf` is a PCS detection score; visual-prompt
  `conf` remains predicted mask IoU. Text prompts default to `conf=0.3`; pass
  `conf=0.0` explicitly to keep all candidates. Image exemplars are reserved
  but not yet implemented.
- Interactive loops: always `set_image` once, then prompt; re-passing the
  image per predict re-encodes and dominates latency.
- Prompt coordinates are pixels on the original image. Multiple points with
  mixed labels refine one object; foreground-only points on different
  objects need separate predicts.

## LibreCLIP / LibreSigLIP2 (zero-shot classify)

```python
from libreyolo import LibreYOLO
model = LibreYOLO("LibreCLIPb32-cls.pt")        # or "LibreSigLIP2b16-cls.pt"
model.set_classes(["a forklift", "an empty aisle", "a spill"])
r = model("frame.jpg"); r.probs.top1
```

Same `Results.probs` contract as trained classifiers; defaults to
ImageNet-1k labels until `set_classes`. Descriptive phrases ("a photo of a
...") often score better than bare nouns; the model re-derives text
embeddings on each `set_classes`, so set once, not per frame.

LibreSigLIP2 (sizes `b16` at 256 px, `so400m` at 384 px; extra
`libreyolo[siglip2]` for the SentencePiece tokenizer) is the accuracy upgrade
tier next to LibreCLIP and a drop-in swap of the weights path. Two
SigLIP-specific capabilities:

- **Multilingual class names.** The tokenizer is multilingual (Gemma
  vocabulary); Spanish/German/French labels work out of the box.
- **`set_classes([...], multi_label=True)`** switches from the default
  softmax-over-classes to independent per-class sigmoid probabilities
  (SigLIP's native calibrated scoring) for images where several labels can be
  true at once. CLIP cannot offer this.

Prompt sensitivity differs between the two: SigLIP over-triggers on verbose
label phrasings more than CLIP, so prefer concise class names ("English
springer" rather than "English Springer Spaniel").

## LibreGround (instruction → click)

`LibreGround` is the factory for GUI / referring grounding. Image +
instruction in, `Results.points` out. Not a new task: the output contract
is `point`. Inference-only, same `libreyolo[vlm]` extra as `LibreVLM`.
Contract: `docs/adr/0020` and `docs/libreground_design.md`.

```python
from libreyolo import LibreGround

r = LibreGround()("screen.png", prompt="Bluetooth")   # ShowUI-2B default
x, y = r.points.xy[0].tolist()                        # original-canvas pixels

model = LibreGround("florence-2-base")                # or qwen3-vl-2b
model.set_query("the red Save button")
r = model.predict("folder/")
```

- There is no default COCO vocabulary. A call without `prompt=` / `set_query`
  raises. A per-call `prompt=` does not become sticky.
- `query=` is an alias of `prompt=`. `set_classes` forwards to `set_query`.
- A list of queries on one image runs one generate per query and merges
  points (class id = query index). A list of queries on a folder raises.
- `conf` is a placeholder. `train()` / `val()` / `export()` raise.
- Shipped aliases are ShowUI-2B (default; MIT weights, Apache-2.0 code/base),
  Florence-2, and Qwen3-VL. Each call keeps one click per query.
  Do not promise ScreenSpot-Pro pixel-perfect clicks from the 2B models.
- The library returns coordinates. It does not move the OS mouse.

## LibreVLM (generative, last resort for detection)

`LibreVLM(<alias>)` prompts a vision-language model and parses its text into
boxes. Flexible (arbitrary instructions via `prompt=`), but slow,
uncalibrated, and parse-dependent; treat outputs as suggestions, not
metrics-grade detections, and prefer LibreOpenVocab when the task is "boxes
for named classes". Zero-shot vocab via `set_classes` works by prompting,
so any string "works", with matching honesty caveats.

## Honest limits (say these to users)

- Zero-shot mAP is well below a fine-tuned detector on a fixed class list;
  these tiers trade accuracy and speed for vocabulary freedom.
- The VLM / ground / open-vocab / SAM snapshot tiers are inference-only
  in LibreYOLO: no `train()`/fine-tuning,
  and `val()` support varies (CLIP and SigLIP2 have classify validators; open-vocab
  eval runs through the standard detect path with a fixed vocabulary).
- Everything runs through the `transformers` extra stack; first use
  downloads snapshot weights (hundreds of MB to GB). `libreyolo checks`
  confirms the extra is present.
- Export of these tiers is not supported like factory families; don't
  promise ONNX/TensorRT for them without checking current e2e coverage.

## Related

- `skills/use-libreyolo/`: the trained-model core library guide.
- `docs/adr/0007-libresam-contract.md`, `docs/adr/0008-open-vocab-detector-contract.md`,
  `docs/adr/0020-libreground-contract.md`, `docs/openvocab_design.md`,
  `docs/librevlm_design.md`, `docs/libreground_design.md`: the contracts.
- `skills/libreyolo-license-audit/`: weight licenses per tier if rehosting
  questions come up.

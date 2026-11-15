# LibreVLM Design Decisions

This document records the user-facing API and internal-contract decisions for
the `LibreVLM` tier, and the reasoning behind them. It is the companion to the
formal contract in [`adr/0002-librevlm-contract.md`](adr/0002-librevlm-contract.md).

## What LibreVLM is

LibreVLM lets you use a general multimodal model as an open-vocabulary object
detector. Most families are autoregressive chat models: you give one an image
and a text prompt, and it generates text back. Structured boxes are parsed into
the same `Results` object every LibreYOLO model returns. LibreMODUS is the
analysis-only exception: it also uses a flow decoder for dense modalities and
loads a pinned external snapshot through a vendored permissive runtime.

There is no detection head and no fixed class set. The "vocabulary" is just a
list of words you supply, so any label works ("pink car", "island", "wheel"),
and a new label costs nothing.

This tier is deliberately for *general* VLMs, not for purpose-built open-vocab
detectors. The bet is that general VLMs keep improving at grounding, and that
integrating them as-is keeps LibreYOLO current with that progress.

## Available models

Pass any of these names to `LibreVLM(name)`. A bare family name resolves to the
size marked with `*`. The authoritative alias tables are in
`libreyolo/models/vlm/__init__.py`, and passing an unknown name raises a
`ValueError` listing every alias.

| Aliases                                      | Family    | License             | Notes                                    |
|----------------------------------------------|-----------|---------------------|------------------------------------------|
| `qwen3-vl`, `-2b`, `-4b`*, `-8b`             | Qwen3-VL  | Apache-2.0          | default model; strongest detector here   |
| `lfm2-vl`, `-450m`*, `-1.6b`, `-3b`          | LFM2-VL   | LFM Open License v1.0 | edge VLM; non-permissive, logs notice   |
| `north-micro-vision`, `-2.4b`*               | North Micro Vision | Apache-2.0    | Cohere Labs native-resolution VLM; needs transformers>=5.16 |
| `internvl3`, `-1b`, `-2b`*, `-8b`            | InternVL3 | Qwen License        | Qwen-backbone weights, logs notice; weak small |
| `florence-2`, `-base`*, `-large`             | Florence-2 | MIT                | small purpose-built detector; tight boxes |
| `kosmos-2`*                                   | Kosmos-2  | MIT                 | 2023 grounder; loads clean, coarse boxes |
| `smolvlm2`, `-2.2b`*, `-500m`                | SmolVLM2  | Apache-2.0          | tiny; weak detector, zero-code family    |
| `locate-anything`, `-3b`*                     | LocateAnything | NVIDIA non-commercial | remote-code grounder; boxes and points |
| `gemma-4`, `-e2b`, `-e4b`*                    | Gemma 4   | Apache-2.0          | native `box_2d` y-first 0-1000; needs transformers>=5.10 |
| `moondream`, `-2`*, `-3`                      | Moondream | Apache-2.0 (`-2`); BSL 1.1 (`-3`) | native detect/point skills; `-3` logs a notice; both mirrored |
| `molmo2`, `-4b`*, `-8b`, `-o-7b` | Molmo2 | Apache-2.0 | point-only; pinned LibreYOLO mirrors; separate `molmo2` extra |
| `sensenova-vision`, `-7b`*                    | SenseNova-Vision | Apache-2.0 code, CC BY-NC 4.0 weights | unified multimodal; 7 tasks, vendored port, heavy |
| `wevisone-vl`, `-2b`*, `-4b`                  | WeVisOne-VL | Apache-2.0        | Qwen3-VL perception fine-tune; 5 tasks; ported pixel decoder |
| `paddleocr-vl`, `-0.9b`*, `paddleocrvl`       | PaddleOCR-VL | Apache-2.0       | `ocr` only: layout regions plus per-region recognition and page markdown; needs transformers>=5.3 |
| `glm-ocr`, `-0.9b`*, `glmocr`                 | GLM-OCR | MIT weights, Apache-2.0 SDK and layout | `ocr` only: PP-DocLayoutV3 regions plus per-region recognition and page markdown; needs transformers>=5.3 |
| `falcon-perception`, `-600m`*, `-300m`        | Falcon Perception | Apache-2.0 | text-query detect and segment; vendored network; `-300m` is detect only |
| `libremodus`, `-14b-a7b`*, `modus`            | MODUS | Apache-2.0 code, external custom-term weights | analysis-only; four standard tasks plus `any2any()` |
| `deepseek-ocr`*, `deepseekocr`, `deepseek-ocr-2` | DeepSeek-OCR | MIT (`1`), Apache-2.0 (`2`) | document OCR, `ocr` task only; vendored port, mirrored weights |
| `qwen3-vl-embedding`, `-2b`*                 | Qwen3-VL-Embedding | Apache-2.0  | `embed` only: image and text vectors in one space; mirrored weights |
| `falcon-ocr`*, `falconocr`, `falcon-ocr-270m` | Falcon OCR | Apache-2.0 | document OCR, `ocr` task only; 0.27B early-fusion; vendored port, mirrored weights |

Florence-2 and Kosmos-2 do not use a chat template: they are driven by task /
grounding prompts and decode boxes via the processor's `post_process_generation`,
so their families override the three inference hooks (and Florence-2's boxes come
back in pixels, no scaling needed). Use the `florence-community/*` Florence-2
checkpoints; the original `microsoft/*` ones do not load on current transformers.

LibreMODUS is not a raw-chat family. It exposes depth, normals, edges, COCO
detection, grounding, and image-conditioned composition; RGB generation and
VQA are intentionally unavailable. See [`libremodus.md`](libremodus.md).

Larger Qwen3-VL tiers (30B and up) and Qwen2.5-VL are not included: the big ones
do not fit a single consumer GPU, and Qwen2.5-VL uses a different coordinate
convention that would need its own family. Some strong models are deliberately
left out for being remote-code without enough payoff (Ovis2.5, MiniCPM-V),
gated (PaliGemma2, Gemma 3), too large for ~16 GB (GLM-4.1V-9B), or
not a clean drop-in (Rex-Omni crashes on the standard Qwen2.5-VL path despite
being the strongest generative detector). Gemma 4 replaces Gemma 3 here:
Apache-2.0 weights and a documented `box_2d` detection format.
`LibreVLM()` defaults to `qwen3-vl-4b`. Detection quality varies a lot by family
and size; Qwen3-VL, LFM2-VL, Florence-2, Gemma 4, and Moondream are the strong ones.

Molmo2 defaults to `task="point"`: `LibreVLM("molmo2", names=["boat"])`
returns `Results.points` in original-image pixels, with synthetic confidence
1.0. It generates once per class. Its parser supports current single-image
`<points coords="1 id x y ...">` markup (0-1000) and legacy `x`/`y` percentage
attributes. Its decoder belongs in the pure `models/vlm/parsing.py` module
under ADR 0002: string and scalar coordinate validation stays separate from
model loading, generation and tensor result construction. Only Molmo2 calls
this new helper; existing JSON box parsers and other VLM workflows are unchanged.
It rejects video/multi-image coordinate groups. A custom `prompt`
must contain `{label}`, replaced with each vocabulary entry; arbitrary raw
prompts remain available through `chat()`. Box detection, point tracking,
training, validation and export are unsupported.

Install Molmo2 in a separate environment with `pip install 'libreyolo[molmo2]'`.
The pinned remote code needs Transformers 4.57.1 and fails on 5.16.1 in both
processor construction and rotary-embedding initialization. Other VLM extras
require Transformers 5. The 8B and O-7B variants need more memory than 4B;
a 16 GB peak-memory claim has not been measured.

Validation: 4B passed a CPU synthetic-point check and matched direct upstream
generation under Transformers 4.57.1. The 8B/O-7B processors and model graphs
were checked without loading weights; their pretrained inference and CUDA
execution remain unverified.

## Decision 1: two layers, raw chat under a detection convenience

A VLM is fundamentally a chat model that can also draw boxes. The API reflects
that honestly with two layers:

```python
# Layer 1, the raw model:
text = model.chat("image.jpg", "Describe the boats and count them.")

# Layer 2, the detection convenience:
model.set_classes(["boat"])
results = model.predict("image.jpg")     # -> Results(boxes, cls, conf)
```

`predict()` is `chat()` with a canonical detection prompt and a parser bolted
on. Keeping `chat()` first-class means power users are never boxed in by the
detection wrapper: free-form questions, custom output formats, counting, and
reasoning are all one call away. This is the property that makes the tier
future-proof, so it is a first-class method rather than an internal detail.

`chat()` applies to the chat-template families (Qwen3-VL, LFM2-VL, SmolVLM2,
InternVL3, Gemma 4) and to Moondream via its native `query` skill. The
task-prompt families (Florence-2, Kosmos-2) are not chat models:
they are driven by fixed task / grounding tokens, so their `chat()` raises
`NotImplementedError` and only `predict()` is supported. `predict()` (the
detection layer) works on every family.

## Decision 2: `set_classes()` is the open-vocabulary surface, and it is sticky

The vocabulary is set with a method, not baked only into the constructor and not
passed on every `predict()` call:

```python
model = LibreVLM()                      # default model, vocabulary defaults to COCO-80
model.set_classes(["pink car", "wheel"])
model.predict("a.jpg")                   # uses the vocabulary
model.predict("b.jpg")                   # still uses it, no need to repeat
model.set_classes(["person", "dog"])     # change it whenever
```

If you never call `set_classes`, the vocabulary defaults to the COCO-80 labels
(like the detector tier), so a bare `predict()` asks the model for the 80 COCO
classes. `set_classes` replaces that with your own open-vocabulary list.

Rationale:

- It keeps `predict()` signature-compatible with the closed-vocab detectors, so
  the two tiers feel the same from the caller's side.
- The vocabulary is conceptually a property of the configured model, not of a
  single image, so it belongs on the object and should persist.
- `names=[...]` at construction is kept as a convenience that simply calls
  `set_classes` for you.

Per-image, free-form queries are served by `chat()`, which is the right place
for genuinely per-call prompts.

## Decision 3: the output is `Results`, and confidence is honestly soft

The tier returns the standard `Results` (`boxes.xyxy`, `boxes.cls`,
`boxes.conf`, `.plot()`, `.save()`), so folders, video, tracking, and drawing
all work unchanged. No new output type is invented.

But these models emit no calibrated per-box score, so `conf` is a placeholder.
We do not pretend otherwise:

- `conf=` filtering and ranking are soft, not calibrated.
- `val()` (mAP) is intentionally unsupported, because it would be misleading.
- `_score_detections()` is the documented hook for a real signal later (decoder
  token log-probabilities or self-consistency).

This is the honest boundary of the tier: it gives you boxes and labels, not a
calibrated detector contract. For calibrated scores and tight boxes, the
detector families behind `LibreYOLO(...)` remain the right tool.

## Decision 4: each model keeps its own output format; we adapt per family

Every model writes boxes in its own scheme, learned from its training data. We
do not try to force one universal format on them. Instead, each family declares
its convention and the shared parser handles the rest. A family is a small
adapter:

```python
class LibreQwen3VL(LibreVLMModel):
    FAMILY = "qwen3vl"
    FILENAME_PREFIX = "LibreQwen3VL"
    HF_REPOS = {"4b": "Qwen/Qwen3-VL-4B-Instruct", ...}
    # HF_REVISIONS required if TRUST_REMOTE_CODE=True
    INPUT_SIZES = {"4b": 1024, ...}
    BBOX_KEY = "bbox_2d"        # this model's JSON key
    COORD_DIVISOR = 1000.0      # this model's coordinate scale
    # _detection_prompt() overridden to ask in this model's expected style
```

The tolerant parser (`libreyolo/models/vlm/parsing.py`) absorbs the rest of the
variation: markdown fences, prose around the JSON, single quotes, truncated
arrays, duplicate boxes from a generation loop, and out-of-vocabulary labels.

### Always verify the coordinate convention empirically

Documentation is often ambiguous about whether a model emits `[0,1]`, `0-1000`,
or absolute pixels. Before trusting a new family's parser, feed the model a
synthetic image with a known box and read back the numbers. Verified so far:

| Model      | Box key   | Scale  | Layout | Knobs                                   |
|------------|-----------|--------|--------|-----------------------------------------|
| Qwen3-VL   | `bbox_2d` | 0-1000 | xyxy   | `COORD_DIVISOR=1000`                    |
| LFM2-VL 450m/1.6b | `bbox` | [0, 1] | xyxy | defaults                              |
| LFM2.5-VL-3B | `bbox`/`bbox_2d` | 0-1000 | xyxy | per-size divisor; the 3B keeps its trained 0-1000 scale no matter what the prompt asks, and drifts between the two key names |
| SmolVLM2   | `bbox`    | [0, 1] | xyxy   | defaults                                |
| InternVL3  | `bbox`    | 0-1000 | xyxy   | `COORD_DIVISOR=1000` + flatten override |
| LocateAnything | `bbox` | 0-1000 | xyxy   | remote-code adapter, boxes + points     |
| Gemma 4    | `box_2d`  | 0-1000 | yxyx   | official Gemini-style `[ymin, xmin, ymax, xmax]` |
| Moondream  | named `x_min..y_max` | [0, 1] | xyxy | native `detect`/`point` skills; one call per class |
| North Micro Vision | bare `[[x1,y1,x2,y2], ...]` | 0-1000 | xyxy | per-class grounding queries; labels assigned by the adapter |
| WeVisOne-VL | `<ref>label</ref><box><x1><y1><x2><y2></box>` | 0-1000 | xyxy | tagged-text parser; `<point><x><y></point>`; `None` when absent |

Three knobs cover the output variation without touching the parser:

- `BBOX_KEY` : the JSON key holding the box (`bbox`, `bbox_2d`, ...).
- `COORD_DIVISOR` : the numeric scale (1.0 for [0,1], 1000.0 for 0-1000).
- `BOX_FORMAT` : the box layout, `xyxy` (default), `xywh`, `cxcywh`, or `yxyx`.

A model that already emits the default shape (a `bbox` key, [0,1], xyxy, through
a chat template) needs no code at all; SmolVLM2 is such a case and its family is
just a repo table. A model that only differs by scale/key/layout sets the knobs
(Qwen3-VL). A model whose output shape is genuinely different overrides a hook:
InternVL3 wraps each object's boxes in an extra list, so its family overrides
`_postprocess` to flatten before the shared builder runs; a grounding-token
model whose boxes come from a processor `post_process_generation` call overrides
`_preprocess`/`_forward`/`_postprocess` (Florence-2 and Kosmos-2 do exactly this).
North Micro Vision is a fourth style: a chat model that refuses labeled-JSON
output but grounds single-class queries extremely well, so its family runs one
"Locate every <label> ..." generation per vocabulary entry and assigns the
labels itself (which also means `predict()` cost scales with the vocabulary
size, and a query for an absent but plausible label can hallucinate a box).
The tier ships eight distinct families (Qwen3-VL, LFM2-VL, SmolVLM2, InternVL3,
Florence-2, Kosmos-2, LocateAnything, North Micro Vision) spanning all these
integration styles, confirming it is
genuinely model-agnostic. Detection *quality* varies a lot by model and size
(Qwen3-VL, LFM2-VL, and Florence-2 are the strong ones); the framework is what is
general, not every model's accuracy.

## Decision 5: default model and licensing

The default model is **Qwen3-VL-4B** (`LibreVLM()` with no arguments), chosen
because it is the strongest general open-weight VLM that runs on a single
consumer GPU and is **Apache-2.0**, so it is clean for LibreYOLO to ship and
needs no license notice.

Weights autodownload on construction (first use) into
`weights/<FILENAME_PREFIX><size>/`, resolved relative to the current working
directory, matching LibreYOLO's existing `weights/` convention. Note that
`FILENAME_PREFIX` here is a weights-directory prefix, not a LibreYOLO `.pt`
checkpoint name: VLM families download Hugging Face repos rather than emitting
`Libre<FAMILY><size>.pt` checkpoints, so the checkpoint-filename nomenclature
does not apply and upstream brand casing (`LibreQwen3VL`, `LocateAnything`) is
kept.
Models under non-permissive licenses log a one-time license notice before
loading/downloading (following the existing download-notice pattern in the repo);
LFM2-VL (LFM Open License v1.0, with a revenue threshold) and InternVL3 (Qwen
License, on its `-hf` weights) are examples. LocateAnything-3B also logs a
notice because its Hugging Face model repository is distributed under NVIDIA's
non-commercial model license and must be loaded through HF remote code.
Remote-code families additionally pin `HF_REVISIONS` to a commit SHA so runtime
downloads do not execute mutable upstream code.

LibreYOLO contributes only adapter code. It does not redistribute VLM weights or
remote-code model repositories; those are downloaded at runtime under the
upstream model repository's terms when a user chooses that model.

## Multi-task families

### SenseNova-Vision

The tier's "VLM as detector" framing was v1 scoping, not a ceiling. The
LocateAnything family already served two tasks (`detect`, `point`); the
SenseNova-Vision family extends the same mechanics to seven: `detect`,
`point`, `pose`, `ocr`, `depth`, `segment` (referring segmentation with
free-text queries), and `panoptic`. Symbolic tasks come back as tagged text
the family parses; dense tasks come back as generated images a frozen VAE
decodes. Every task returns the same `Results` payload the specialist
families return, so the generalist and the specialists are interchangeable
call sites.

```python
model = LibreVLM("sensenova-vision", task="depth")
result = model.predict("image.jpg")            # Results.depth_map
model.set_task("segment").set_classes(["person furthest to the right"])
result = model.predict("image.jpg")            # Results.masks, one referred mask
model.set_task("panoptic")                      # COCO-panoptic vocabulary default
```

`set_task()` (on the tier base) switches the active task without reloading
weights, which matters at 29.6 GB. Two family-specific notes: this is the
one family whose architecture is vendored rather than loaded through
transformers (no upstream remote code exists; the port carries per-file
Apache-2.0 provenance and an SDPA fallback so flash-attn is optional), and
its weights are CC BY-NC 4.0 (non-commercial), auto-downloaded from the
byte-identical LibreYOLO mirror (`LibreYOLO/SenseNovaVision7b`) under a
one-time license notice. Capabilities without a canonical LibreYOLO task
(surface normals, grounded-conversation segmentation, editing, multi-view
reconstruction) stay behind `chat()`/`generate()`. See
[`adr/0012-sensenova-unified-multimodal.md`](adr/0012-sensenova-unified-multimodal.md).

### LibreMODUS

LibreMODUS adds `detect`, `depth`, `normal`, and `edge` to the standard task
surface and returns the corresponding `Boxes`, `DepthMap`, `NormalMap`, or
`EdgeMap`. Its `any2any()` method accepts one to three image-derived modalities
plus optional text, can feed chained outputs back as conditions, and can rank
multiple candidates with the model's own yes/no consistency logits.

Unlike ordinary VLM snapshots, its architecture is built from the audited
Bagel/Qwen2/SigLIP/FLUX modules in tree and dispatched with Accelerate. The
checkpoint stays external and is never mirrored. The supported matrix,
precision tiers, and terms distinction are documented in
[`libremodus.md`](libremodus.md) and
[`adr/0016-libremodus-analysis-contract.md`](adr/0016-libremodus-analysis-contract.md).

### Falcon Perception

Falcon Perception (TII, Apache-2.0 code and weights) is an early-fusion
transformer: image patches and text share one sequence, and for each matching
instance the model emits `<|coord|>`, `<|size|>` and `<|seg|>` tokens. Small
heads on the hidden state decode the box centre and extent; the `<|seg|>`
embedding is dotted with AnyUp-upsampled image features to give the mask. The
network is vendored in `libreyolo/models/falcon_perception/` (no remote code,
pinned `HF_REVISIONS`); see that package's `NOTICE`.

```python
model = LibreVLM("falcon-perception", names=["cat", "the dog on the left"])
model.predict("image.jpg")                    # Results.boxes
model.set_task("segment").predict("image.jpg")  # Results.boxes + Results.masks
LibreVLM("falcon-perception-300m", names=["cat"])  # detect only, no mask head
```

Each vocabulary entry is one query and one generation, so cost grows with the
number of entries; every instance found for an entry gets that entry's class
id. `set_classes` (or `names=`) is required. Confidence is a constant 1.0 and
there is no NMS (upstream's coordinate de-duplication runs inside decoding).
On MPS the family runs on CPU because the RoPE tables are complex64.

### WeVisOne-VL

WeVisOne-VL (Tencent, Apache-2.0 code and weights) is Qwen3-VL fine-tuned to
answer perception requests as tagged text on a 0-1000 grid, with `None` for an
absent target. An `<OBJ>` token after a box asks a separate pixel decoder
(`mask_decoder.pt`, about 9.5M parameters for 2B and 9.6M for 4B) for that
instance's alpha matte. The backbone loads through the native transformers
Qwen3-VL classes (no remote code, pinned `HF_REVISIONS`). The decoder is a
minimal Apache-2.0 port in `libreyolo/models/wevisone/` whose deformable
cross-attention reuses the in-tree Deformable-DETR `MSDeformAttn`; see that
package's `NOTICE`.

```python
model = LibreVLM("wevisone-vl", names=["person", "dog"])
model.predict("image.jpg")                        # Results.boxes
model.set_task("point").predict("image.jpg")      # Results.points
model.set_classes(["dog"]).set_task("segment")    # Results.boxes + Results.masks
model.set_task("matte").predict("image.jpg")      # Results.matte
model.set_task("ocr").predict("image.jpg")        # Results.ocr
LibreVLM("wevisone-vl", names=["the man in red"], referring=True)
```

| Task | Generations | Output |
|---|---|---|
| `detect` | one for the whole vocabulary; with `referring=True`, one per phrase using the referring instruction, which answers `None` for an absent target | `Results.boxes` |
| `point` | one per label | `Results.points` |
| `segment` | one per phrase, then one teacher-forced pass for the pixel decoder | `Results.boxes` and binary `Results.masks` (`alpha >= mask_threshold`) |
| `matte` | as `segment` | `Results.matte`, the pixelwise max of the instance mattes |
| `ocr` | one, with the released OCR instruction | `Results.ocr`, axis-aligned quads with transcripts |

- `segment`, `matte` and `referring=True` treat each vocabulary entry as a
  phrase, so they need an explicit `names=` or `set_classes()`.
- Mask decoding reads four ViT levels before the spatial merge and the last
  decoder layer's hidden state at each `<OBJ>`, then runs the decoder in FP32
  on a 1024x1024 canvas. All instances of a response share one
  query-transformer pass, whose self-attention mixes them. Only the
  full-canvas refinement, about 2 GB of activations per instance, is chunked
  by `decoder_batch_size` (default 1), so the batch size does not change
  the masks.
- Counting is `len(result.boxes)` or `len(result.points)`. Visual-prompt
  detection and OCR grounding are not wrapped as tasks; use `chat()`.
- `min_pixels` / `max_pixels` default to the upstream budget (16x32x32 to
  2025x32x32 pixels). Lowering `max_pixels` trades accuracy for speed.
- Confidence is a constant 1.0. `train()`, `val()` and `export()` are
  unsupported, and the vLLM backend of the upstream engine is not ported.

Upstream-reported results (README of `Tencent/WeVisOne-VL` at commit
`9a6bcd85`, not reproduced in LibreYOLO). F1 Mean averages F1 over IoU
0.5:0.95; greedy decoding with at most 1,960 visual tokens per image. The
upstream detection benchmarks prompt one category per generation.

| Benchmark (metric) | 2B | 4B |
|---|---|---|
| COCO detection (F1 Mean) | 55.2 | 57.1 |
| LVIS detection (F1 Mean) | 53.3 | 55.3 |
| RefCOCOg test referring (F1 Mean) | 78.2 | 81.8 |
| WeRef-Bench rejection rate (%) | 92.9 | 96.4 |
| PixMo-Count test counting (accuracy %) | 88.1 | 89.6 |
| PointBench pointing (accuracy) | 62.9 | 71.9 |
| TotalText OCR (F1 Mean) | 42.5 | 52.0 |
| ReasonSeg val referring segmentation (gIoU) | 60.9 | 71.2 |
| P3M-500-NP box-prompted matting (MAD x1000, lower is better) | 6.6 | 7.3 |

Validation: the 2B checkpoint ran all five tasks through `predict()` on CPU in
FP32, and the ported decoder loads the released sidecar strictly and matches
the upstream module's logits to within 1e-4. The 4B decoder sidecar loads
strictly against the 4B config and tokenizer, but the 4B backbone and CUDA
were not run.

### DeepSeek-OCR

DeepSeek-OCR (deepseek-ai, MIT code and weights) is a 3B mixture-of-experts
document OCR model: a SAM ViT-B plus CLIP-L encoder compresses the page into a
few hundred visual tokens and a DeepSeek-V2 style decoder (about 570M active
parameters) writes the text. LibreYOLO runs a vendored PyTorch port of the
release's inference path (`libreyolo/models/deepseekocr/`, no remote code),
loaded strictly from the byte-identical mirror `LibreYOLO/LibreDeepSeekOCR1`
at a pinned revision. It serves the `ocr` task only.

```python
model = LibreVLM("deepseek-ocr")                       # resolution="gundam"
result = model.predict("page.png")                    # Results.ocr, one region per line
LibreVLM("deepseek-ocr", prompt="markdown").predict("page.png").ocr.markdown
model.set_prompt("free").predict("page.png").ocr.markdown   # plain text, no regions
model.chat("page.png", "Parse the figure.")           # any prompt, raw text
model.val(data="ocr_dataset/")                        # det hmean, e2e F1, 1-NED
```

| `prompt` | Release prompt | `Results.ocr` |
|---|---|---|
| `"ocr"` (default) | `<|grounding|>OCR this image.` | one region per text line, `labels` all `text`, `markdown` empty |
| `"markdown"` | `<|grounding|>Convert the document to markdown.` | one region per layout block (block markdown as transcript, layout category in `labels`; figures dropped) and the page in `markdown` |
| `"free"` | `Free OCR.` | no regions, the page text in `markdown` |
| any other string | as given (`<image>` is prepended if missing) | whatever grounding it emits, plus `markdown` |

- Boxes come on a 0 to 999 grid and are scaled with `x / 999 * width` to
  original-image pixels as axis-aligned quads, then sorted top to bottom and
  left to right as the OCR contract specifies. `markdown` keeps the model's
  own reading order.
- `resolution`: `tiny` (512), `small` (640), `base` (1024), `large` (1280),
  or `gundam` (default: a 1024 global view plus 2 to 9 local 640 crops for
  pages larger than 640 px), as documented by the release.
- Decoding follows the release: greedy, up to 8192 new tokens,
  `no_repeat_ngram_size=20` (constructor option).
- Confidence is a constant 1.0. `train()` and `export()` raise.
- CUDA runs bf16 weights under `torch.autocast`, as the release does (fp16 on
  GPUs without bf16). CPU and MPS run float32 with eager attention (about 13
  GB of memory); MPS has not been run on hardware.
- DeepSeek-OCR-2 (`LibreVLM("deepseek-ocr-2")`, size `2`, Apache-2.0 code and
  weights, mirror `LibreYOLO/LibreDeepSeekOCR2`) keeps the decoder and the SAM
  stage and replaces CLIP with a 24-layer Qwen2 encoder that reads the SAM
  tokens bidirectionally and emits one learned query per visual token
  causally. Its views are 1024 px global plus up to 6 local 768 px crops
  (`resolution="gundam"`, default) or a single 1024 view (`"base"`); the
  other resolutions have no learned queries and raise. Same prompts, same
  outputs.
- EXIF orientation is not applied (LibreYOLO's image loader convention); the
  release transposes EXIF-rotated photos first.

Validation: on an RTX 3090 the port reproduces the release (transformers
4.46.3 remote code, bf16 autocast, greedy) token for token on 17 cases: six
synthetic pages with the OCR and markdown prompts in Gundam mode, one page at
each of the other four resolutions, and the free prompt; the parsed boxes are
identical. DeepSeek-OCR-2 reproduces its release the same way on 14 cases (the
six pages with both prompts in Gundam mode, one Base page, the free prompt).
Upstream-reported benchmark numbers are in the release papers (arXiv
2510.18234, 2601.20552) and are not reproduced here.

### PaddleOCR-VL

PaddleOCR-VL (PaddlePaddle, Apache-2.0 code and weights) is a two-stage
document parser serving the `ocr` task only. PP-DocLayoutV2 finds 25 kinds of
layout region and their reading order; PaddleOCR-VL-0.9B (NaViT-style vision
encoder plus ERNIE-4.5-0.3B) then reads each region from a task prompt:
`OCR:` for text and titles, `Table Recognition:` (OTSL, returned as HTML),
`Formula Recognition:` (LaTeX) and, with `use_chart_recognition=True`,
`Chart Recognition:`. Both networks load through native transformers classes
(no remote code) from one pinned mirror, `LibreYOLO/LibrePaddleOCRVL0.9b`. The
pipeline around them (layout thresholds, NMS and merges, overlap filter,
prompt routing, pixel budgets, content clean-up, markdown) is ported from
PaddleX in `libreyolo/models/paddleocrvl/`; see that package's `NOTICE`.

```python
model = LibreVLM("paddleocr-vl")
result = model.predict("page.png")
result.ocr.texts, result.ocr.labels   # one transcript and layout class per region
result.ocr.markdown                   # the page as markdown, reading order
model.val(data="ocr_dataset/")        # OCR validator: hmean, end-to-end F1, 1-NED
LibreVLM("paddleocr-vl", use_layout_detection=False, prompt_label="formula")
```

- Regions are axis-aligned layout boxes written as 4-point quads, one per
  recognized block (a paragraph, a title, a table), not one per text line.
  Pictures, and charts and seals by default, are not recognized and are left
  out of `Results.ocr` and the markdown.
- `Results.ocr.conf` is the geometric mean of the generated tokens'
  probabilities; `det_conf` is the layout score. `conf=` does not gate
  regions; the layout keeps the upstream per-class thresholds (override with
  `layout_threshold=`).
- Recognition is greedy with up to 4096 new tokens and no repetition penalty,
  as upstream runs it locally, and each region is resized within
  `min_pixels`/`max_pixels` (default 112896 / 1003520). The image processor
  uses the PIL backend, which matches upstream's resize exactly.
- Not ported: adjacent text-block merging, figures embedded in tables, seal
  and spotting recognition, polygon layout shapes, document unwarping and
  image export. The markdown is the plain variant (no HTML centering).
- `train()` and `export()` raise.

### Qwen3-VL-Embedding

Qwen3-VL-Embedding-2B (Qwen, Apache-2.0) is Qwen3-VL-2B-Instruct fine-tuned
into an embedder. It serves the `embed` task only (ADR 0015) and is the first
VLM-based embed family. It joins this tier rather than the checkpoint factory
because it shares the Qwen3-VL snapshot download and transformers loader; its
output is the same embed contract as CLIP, SigLIP2 and PE.

```python
model = LibreVLM("qwen3-vl-embedding-2b")
pages = model.embed("pages/")                  # (N, 2048), one row per page image
r = model.predict("page.png")                  # r.embeddings is (1, 2048)
q = model.embed_text(
    ["total due on the invoice"],
    prompt="Retrieve images or text relevant to the user's query.",
)                                              # (1, 2048)
scores = q @ pages.T                           # rows are unit length
```

- Each input is a chat: a system instruction, then the user's image or text.
  The model's hidden state at the last token, L2-normalized, is the vector.
- `prompt` is that instruction. The default is upstream's
  `"Represent the user's input."`; set `prompt=` at construction to change it
  for images and texts, or pass `prompt=` to `embed_text` per call. For
  retrieval, give queries a task instruction and keep the default for
  documents. A missing final punctuation mark gets a period, as upstream does.
- Images are resized on a 32-pixel grid to between 4,096 and 1,843,200 pixels
  with PIL bicubic, matching upstream's reference embedder
  (`scripts/qwen3_vl_embedding.py`). The sentence-transformers path in the
  model card caps images at 1,310,720 pixels instead, so its vectors for
  larger images differ slightly (cosine 0.989 on a 1.8 MP photo).
- Weights download from `LibreYOLO/LibreQwen3VLEmbedding2b`, a mirror of the
  upstream snapshot with unchanged parameters. Install `libreyolo[vlm]`.
- Not supported: `chat()`, `set_classes()`, `train()`, `val()`, `export()`,
  combined image-plus-text inputs, video clips (a video is embedded frame by
  frame), and Matryoshka truncation as an option (slice the first `k`
  columns and re-normalize). The 8B size is not wrapped.

### GLM-OCR

GLM-OCR (Z.ai; MIT weights per the model card, Apache-2.0 SDK) is a two-stage
document parser serving the `ocr` task only. PP-DocLayoutV3 finds 25 kinds of
layout region and their reading order; GLM-OCR-0.9B (CogViT encoder plus a
GLM-0.5B decoder) then reads each region from a task prompt:
`Text Recognition:` for text, titles and seals, `Table Recognition:` (HTML)
and `Formula Recognition:` (LaTeX). Both networks load through native
transformers classes (no remote code) from one pinned mirror,
`LibreYOLO/LibreGLMOCR0.9b`. The pipeline around them (layout NMS and merges,
label routing, crops, resize and JPEG round trip, decoding settings, content
clean-up, formula-number and hyphen merges, markdown) is ported from the
official GLM-OCR SDK in `libreyolo/models/glmocr/`; see that package's
`NOTICE`.

```python
model = LibreVLM("glm-ocr")
result = model.predict("page.png")
result.ocr.texts, result.ocr.labels   # one transcript and layout class per block
result.ocr.markdown                   # the page as markdown, reading order
LibreVLM("glm-ocr", use_layout_detection=False, prompt_label="table")
model.chat("id.png", 'Extract as JSON: {"name": "", "date": ""}')  # information extraction
```

- Regions are axis-aligned layout boxes written as 4-point quads, one per
  recognized block. Headers, footers, page numbers, footnotes and side text
  are dropped, as upstream does; pictures and charts are not read and are
  left out of `Results.ocr` and the markdown.
- Decoding follows the SDK request: greedy, repetition penalty 1.1, up to
  8192 new tokens. Each crop is resized to multiples of 28 pixels and
  round-tripped through JPEG before recognition, as the SDK sends it
  (`image_format=None` skips the round trip).
- `Results.ocr.conf` is the geometric mean of the generated tokens'
  probabilities; `det_conf` is the layout score. `conf=` does not gate
  regions; the layout uses the upstream threshold 0.3 (override with
  `layout_threshold=`).
- Not ported: PDF input, polygon crops, picture export and the hosted-API
  client. `use_layout_detection=False` is a LibreYOLO convenience with no SDK
  counterpart.
- `train()` and `export()` raise.

### Falcon OCR

Falcon OCR (TII, Apache-2.0 code and weights, version 1.5) is a 0.27B
early-fusion model: one 22-layer transformer reads 16-pixel image patches
and text tokens together, patches attending to each other bidirectionally
and text decoding causally. It serves the `ocr` task only and writes the page
as text, with LaTeX for formulas and HTML for tables. The release runs on
FlexAttention and a Triton kernel through remote code; LibreYOLO runs a
PyTorch port in `libreyolo/models/falconocr/` (dense masks, float32
attention-sink math) from the byte-identical mirror
`LibreYOLO/LibreFalconOCR270m`. See that package's `NOTICE`.

```python
model = LibreVLM("falcon-ocr")
result = model.predict("page.png")
result.ocr.markdown                   # the page text
LibreVLM("falcon-ocr", prompt="table").predict("table.png")
LibreVLM("falcon-ocr", layout=True).predict("page.png").ocr.labels
```

- Default (upstream `generate`): one pass over the whole page, one
  `Results.ocr` region covering the image with label `plain`, and the same
  text in `markdown`.
- `layout=True` (upstream `generate_with_layout`): LibreYOLO's PP-DocLayoutV3
  port at `layout_threshold=0.3`, nested boxes dropped, each crop read with
  its category prompt; regions keep the layout's reading order, `det_conf` is
  the layout score. Classes without an upstream prompt category
  (`display_formula`, `inline_formula`, `vertical_text`, `header_image`,
  `footer_image`) and pictures are skipped, as upstream skips them.
- `prompt=` takes a category (`plain`, `text`, `table`, `formula`, ...) or an
  instruction. Upstream `generate` ignores its `category` argument and always
  sends `plain`; here the category is applied.
- Pages are resized so both sides fall in `[64, 1536]` and snapped to
  multiples of 16. Greedy decoding, budget `max_new_tokens` (8192) rounded up
  to a 128-token block as upstream. `conf` is a constant 1.0.
- CUDA runs bf16 weights under `torch.autocast`; CPU and MPS run float32
  (about 1.1 GB of weights). Batches of one; upstream's batched left-padded
  engine and temperature/top-k sampling are not ported. `train()` and
  `export()` raise.

Validation: on CPU in float32 the port reproduces the release code
(tiiuae/Falcon-OCR @ 0bc28f89, run eagerly with the Triton kernel replaced by
its PyTorch expression) token for token on six OCR fixtures and a synthetic
1100x1400 report page; token ids, positions and pixel patches are
bit-identical. In float64 both implementations agree on prefill logits to
1e-6. CUDA bf16 and the layout mode were not compared against upstream.

## Known limitations (v1)

These are deliberate v1 scoping choices, called out so behavior matches expectations:

- **Confidence is usually synthetic.** Chat-model adapters assign every box
  `1.0`; `conf=` filtering is mechanical, and ByteTrack's score-stratified
  association is inert. LibreMODUS derives a sequence score from constrained
  token probabilities, but it is still not calibrated detector confidence.
  Generic `val()`/mAP remains unsupported.
- **`batch=` does not speed up VLMs.** `predict("folder/")` works, but generation
  runs one image at a time, so a larger `batch=` gives no throughput gain in v1.
- **Python-API only.** The `libreyolo` CLI does not resolve VLM aliases yet; use
  `LibreVLM(...)` from Python. `predict`/`track` parity is at the API level.
- **`chat()`** applies to chat-template families only. Florence-2, Kosmos-2,
  and LibreMODUS are task-driven and raise; `prompt=` is ignored by the first
  two and is a one-phrase grounding convenience for LibreMODUS.
- **Point support is family-specific.** LocateAnything supports `task="point"`
  and returns `Results.points`; point tracking, point validation, and exported
  backend point decoding remain out of scope.

## Adding a new model: checklist

1. Create `libreyolo/models/vlm/<family>.py` subclassing `LibreVLMModel`.
2. Set `FAMILY`, `FILENAME_PREFIX`, `HF_REPOS`, `INPUT_SIZES`.
3. Probe the model on a known box; set `BBOX_KEY` and `COORD_DIVISOR`, and
   override `_detection_prompt()` if its expected ask differs.
4. Add an alias in `libreyolo/models/vlm/__init__.py` (and to the top-level lazy
   exports if it should be importable as `libreyolo.Libre<Name>`).
5. Add a `_LICENSE_NOTICE` only if the weights are non-permissive.
6. Verify with a real inference; the parser and the predict/track surface are
   shared, so there is usually no other code to write.

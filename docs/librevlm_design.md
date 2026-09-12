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
| `libremodus`, `-14b-a7b`*, `modus`            | MODUS | Apache-2.0 code, external custom-term weights | analysis-only; four standard tasks plus `any2any()` |

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

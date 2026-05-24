---
name: libreyolo-upload-hf-model
description: Prepare and upload a LibreYOLO weight repo to the HuggingFace LibreYOLO org. Use when publishing new weights (new family, new size, or new task like -seg). Covers filename, README, LICENSE, NOTICE, and collection membership.
---

# Upload a LibreYOLO weight repo to HuggingFace

Use this skill when publishing model weights to `https://huggingface.co/LibreYOLO/<repo>`.

Scope: **weight-only repos** (one `.pt`, one canonical filename, matching a family defined in `libreyolo/models/<family>/model.py`). Not product repos (`face-*`, `libreyolo-web`) — those are bespoke and out of scope.

## The 5-file contract

Every weight repo contains exactly these 5 files. No more, no less.

```
<repo>/
├── .gitattributes       # LFS rules (copy from any existing LibreYOLO weight repo, 1519 bytes)
├── README.md            # YAML frontmatter + Source / Modifications / License
├── LICENSE              # upstream license text, verbatim
├── NOTICE               # attribution block (required for Apache-2.0 upstreams)
└── Libre<Family><size>[-<task>].pt   # the canonical weight file
```

Do **not** upload:

- Lowercase or legacy filenames (`libreyolo9s.pt`, `rf-detr-nano.pth`).
- Raw upstream checkpoints alongside the converted weight.
- Both `.pt` and `.pth` of the same weights.

## Canonical filename

Derived from code, not invented:

```
name = FILENAME_PREFIX + size + ("-" + task if task else "")
file = name + ".pt"
```

`FILENAME_PREFIX` per family — read from `libreyolo/models/<family>/model.py`:

| Family | Prefix | Example |
|---|---|---|
| YOLOX | `LibreYOLOX` | `LibreYOLOXs.pt` |
| YOLO9 | `LibreYOLO9` | `LibreYOLO9m.pt` |
| RFDETR | `LibreRFDETR` | `LibreRFDETRn.pt`, `LibreRFDETRn-seg.pt` |
| RTDETR | `LibreRTDETR` | `LibreRTDETRr50.pt` |
| YOLONAS | `LibreYOLONAS` | `LibreYOLONASs.pt` |

**Ask the user** if: the size code isn't obvious, the family isn't one of the above, or the filename doesn't match what the loader at `libreyolo/models/base/model.py:get_download_url` builds. Do not guess.

## Canonical filename whitelist

Before creating the HF repo or uploading, **verify that the `.pt` filename you are about to ship appears verbatim in this list**. If it doesn't, stop and ask the user — either a new family/size/task is being introduced (skill should be updated) or the name is wrong.

Authoritative list of all valid weight filenames (matches the schema enforced by `BaseModel._filename_regex` and the family table in `docs/nomenclature.md`):

```
LibreYOLOXn.pt, LibreYOLOXt.pt, LibreYOLOXs.pt, LibreYOLOXm.pt,
LibreYOLOXl.pt, LibreYOLOXx.pt,

LibreYOLO9t.pt, LibreYOLO9s.pt, LibreYOLO9m.pt, LibreYOLO9c.pt,

LibreYOLO9E2Et.pt, LibreYOLO9E2Es.pt, LibreYOLO9E2Em.pt,
LibreYOLO9E2Ec.pt,

LibreYOLONASs.pt, LibreYOLONASm.pt, LibreYOLONASl.pt,
LibreYOLONASn-pose.pt, LibreYOLONASs-pose.pt,
LibreYOLONASm-pose.pt, LibreYOLONASl-pose.pt,

LibreDFINEn.pt, LibreDFINEs.pt, LibreDFINEm.pt, LibreDFINEl.pt,
LibreDFINEx.pt,

LibreDEIMn.pt, LibreDEIMs.pt, LibreDEIMm.pt, LibreDEIMl.pt,
LibreDEIMx.pt,

LibreDEIMv2atto.pt, LibreDEIMv2femto.pt, LibreDEIMv2pico.pt,
LibreDEIMv2n.pt, LibreDEIMv2s.pt, LibreDEIMv2m.pt,
LibreDEIMv2l.pt, LibreDEIMv2x.pt,

LibrePICODETs.pt, LibrePICODETm.pt, LibrePICODETl.pt,

LibreRTDETRr18.pt, LibreRTDETRr34.pt, LibreRTDETRr50.pt,
LibreRTDETRr50m.pt, LibreRTDETRr101.pt, LibreRTDETRl.pt,
LibreRTDETRx.pt,

LibreRFDETRn.pt, LibreRFDETRs.pt, LibreRFDETRm.pt,
LibreRFDETRl.pt, LibreRFDETRn-seg.pt, LibreRFDETRs-seg.pt,
LibreRFDETRm-seg.pt, LibreRFDETRl-seg.pt,

LibreECs.pt, LibreECm.pt, LibreECl.pt, LibreECx.pt,
LibreECs-pose.pt, LibreECm-pose.pt, LibreECl-pose.pt,
LibreECx-pose.pt, LibreECs-seg.pt, LibreECm-seg.pt,
LibreECl-seg.pt, LibreECx-seg.pt
```

Common rule violations to reject before upload:

- Wrong casing on the size — `LibreDEIMv2Atto.pt` (PascalCase). Sizes are always lowercase.
- Old class-name forms in the file — `LibreYOLORTDETR*.pt` or `LibreECDET*.pt`. The library renamed those.
- Detect carrying an explicit `-det` suffix — `LibreECs-det.pt`. Detect is implicit, no suffix.
- Lowercase prefix — `librerfdetrn.pt` (Hugging Face is case-insensitive on lookup but the file inside the repo must use the canonical case).

If a candidate filename isn't in the list and isn't a brand-new family being introduced now, **stop and ask** rather than uploading.

## README template

```markdown
---
license: <apache-2.0 | mit | ...>
library_name: libreyolo
tags:
  - object-detection
  - <family-tag>          # yolox | yolov9 | rf-detr | rt-detr | yolo-nas
---

# <RepoName>

<One sentence: what architecture, what size, repackaged for LibreYOLO.>

## Source

Derived from [<upstream-org>/<upstream-repo>](https://github.com/<upstream-org>/<upstream-repo>)
at <tag-or-commit>.
Copyright (c) <years> <upstream-authors>. Licensed under the <License> License.

<If a backbone has its own upstream, add a second paragraph for it.>

## Modifications

State-dict key remapping only. Learned parameters are unchanged.
See `weights/convert_<family>_weights.py` in the
[LibreYOLO source repository](https://github.com/LibreYOLO/libreyolo).

## License

<Apache License 2.0 | MIT License>. See the [`LICENSE`](./LICENSE)
and [`NOTICE`](./NOTICE) files in this repository.
```

## LICENSE + NOTICE

- **LICENSE**: copy the upstream `LICENSE` file **verbatim**. Do not synthesize, do not template.
- **NOTICE**: required when upstream is Apache-2.0. Short attribution block:

```
Libre<Family> weights
---------------------

This product contains weights derived from <Upstream>
(https://github.com/<upstream-org>/<upstream-repo>).
Copyright (c) <years> <upstream-authors>.
Licensed under the Apache License, Version 2.0.

<Second paragraph if there's a separately-licensed backbone.>
```

For MIT upstreams (e.g. YOLOv9): NOTICE is not legally required. For consistency with existing YOLOX/RFDETR/RTDETR repos, include one anyway.

## Collection membership

After the repo is uploaded, add it to a collection:

| Repo type | Collection |
|---|---|
| Detection weights | `LibreYOLO/libreyolo-models-698875bf2b5f695708415169` |
| RF-DETR segmentation | `LibreYOLO/rf-detr-instance-segmentation-69bde2744d6c285366a69603` |
| New seg family (e.g. YOLOX-seg) | **Ask the user** — create a new collection or extend existing |
| New detection family with no siblings yet | Add to `LibreYOLO Models` |

Add via HF UI or `huggingface_hub.add_collection_item(collection_slug, item_id=<repo>, item_type="model")`.

## Upload workflow

1. Build the 5 files locally in a clean directory.
2. Verify canonical filename matches `BaseModel.get_download_url()` output for this family + size.
3. **Cross-check the filename against the whitelist above.** If it isn't in the list, halt and ask the user — don't paper over it with a manual override.
4. Validate the `.pt` against the current LibreYOLO checkpoint metadata schema before upload. The source of truth is `docs/checkpoint_schema.md` and the helpers in `libreyolo/utils/serialization.py`; do not duplicate the schema in this skill. A simple load smoke test is not enough.
5. Create the HF repo (skip if it exists): `huggingface-cli repo create LibreYOLO/<RepoName> --type model`.
6. Upload: `huggingface-cli upload LibreYOLO/<RepoName> <local-dir> . --commit-message "Initial upload"`.
7. Smoke test: `YOLO.from_pretrained("LibreYOLO/<RepoName>")` on a fresh machine / cleared cache.
8. Add to the matching collection.

One commit per file if iterating — easier to revert than a batch commit.

## Ask the user when

- The upstream release / commit pin isn't known (reproducibility needs it in README).
- The family isn't in the code yet (the skill can't derive canonical filename).
- A file with the same name already exists on the target repo (overwrite is destructive).
- The repo is a new task type and no collection fits.
- The upstream has a non-standard license (neither Apache-2.0 nor MIT).

## Common traps

- Relying only on `YOLO.from_pretrained(...)`; it can pass even when required checkpoint metadata is missing or stale.
- Uploading both `.pt` and `.pth` of the same weights (wastes HF storage, no canonical filename).
- Copying a lowercase filename from an old release — the loader only fetches the `FILENAME_PREFIX`-cased `.pt`.
- Writing `license: mit` in README YAML for a repo whose weights derive from an Apache-2.0 upstream — MIT re-licensing is not legal without explicit permission.
- Forgetting `.gitattributes` — weights upload as raw blobs instead of LFS and the repo becomes huge.
- Adding to the wrong collection (seg → detection collection).

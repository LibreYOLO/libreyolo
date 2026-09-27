---
name: libreyolo-verify-training
description: >-
  Prove that a LibreYOLO model actually trains correctly, not just that
  train() runs without crashing. Use when adding or changing a trainer, loss,
  augmentation, scheduler, or DDP path; when someone asks "does training work
  for family X?", "is this model trainable?", or reports bad fine-tune
  results; or before making a validation claim about a new family's training.
  Covers the confidence ladder (overfit gate, RF1 marbles floor, regression
  and RF5 tiers, full-run spot checks), the objective definition of
  training-evidence tiers, the recurring silent-training-bug classes and how
  to hunt each one, and how to watch a live run. Speed problems are
  libreyolo-profiling; running the suites is libreyolo-run-e2e-tests.
---

# Verify LibreYOLO training

Training bugs ship silently because `train()` almost never crashes: a dead
augmentation knob, a dropped label source, a mis-scaled LR, or a randomly
initialized backbone all still produce falling loss curves. Real examples that
reached users: an augmentation stage silently skipping the affine transform
(its degrees/translate/shear/scale knobs were no-ops), mixup dropping the
second image's labels, a `from_pretrained` that silently no-oped and trained a
classifier on a random backbone for weeks, and an eval interval that deleted
`best.pt` on short runs. "Loss goes down" proves nothing; climb the ladder.

## The confidence ladder

Climb until the claim you want to make is covered. Each rung is cheap
relative to the one above it.

**Rung 0: it overfits a tiny fixture (minutes, local GPU or CPU).**
Train on `coco8.yaml` (or `coco8-pose.yaml` for pose) for ~50-100 epochs and
validate **on the training set**. A correct pipeline memorizes 8 images:
expect mAP to climb toward ~0.9+. Loss falling but train-set mAP staying near
zero is the signature of a broken label path, target assigner, or decode.
This is the single highest-value check per invested minute; run it for any
new trainer before anything else.

```bash
libreyolo train model=Libre<X>.pt data=coco8.yaml epochs=100 imgsz=640 batch=8
libreyolo val model=runs/train/<name>/weights/best.pt data=coco8.yaml split=train  # the save_dir train printed
```

**Rung 1: the RF1 floor (the repo's objective bar).**
`tests/e2e/test_rf1_training.py` fine-tunes every trainable family on the
marbles dataset and asserts `MIN_MAP = 0.05` plus a save/reload check.
The `_RF1_NOT_APPLICABLE` and `_RF1_VALIDATION_GAPS` maps distinguish families
without a training implementation from trainable families that have not yet
cleared RF1. Each entry must describe completed checks and known limits.
Removing a validation-gap entry means making the family pass RF1, not hiding
the evidence gap.

```bash
PYTHONPATH=. .venv/Scripts/python.exe -m pytest tests/e2e/test_rf1_training.py \
  -m "e2e and not rf5" -k "<family>" -v
```

**Rung 2: regression + behavior tests.** `test_training_regression.py`
(training-specific regressions) and the unit-level trainer/loss/DDP tests
(`tests/unit/test_ddp_*`, `test_*_trainer*`). Run these whenever shared
training code changes, not just the family you touched.

**Rung 3: RF5 benchmark.** `make test_rf5` trains across the RF100-style
suite (needs `ROBOFLOW_API_KEY`). This is the "does it fine-tune well, not
just at all" tier; expensive, run when claiming training quality.

**Rung 4: a real run at scale.** Full-dataset training on a rented GPU (use
`launch-serverless-gpu-job`), then compare val mAP against the published
number for that family/size. `scripts/spot_check_val_map.py` manages
baseline-vs-current comparisons on COCO val. Only this rung supports claims
like "training reproduces the paper/upstream recipe".

## The silent-bug classes (and the hunt for each)

Check these deliberately whenever touching training code; none of them crash.

1. **Dead augmentation knobs.** A config key that no code consumes, or a
   pipeline stage that never runs. Hunt: set the knob to an extreme value and
   diff output pixels/labels across a fixed seed; every documented knob must
   visibly change the sample. For refactors, `scripts/augment_diff_sweep.py`
   runs the parity cases across many seeds against two checkouts and asserts
   byte-identical outputs; the golden fixtures in
   `tests/unit/fixtures/augment_golden/` pin single-seed behavior.
2. **Label loss in multi-image augs.** Mosaic/mixup must carry *all* source
   images' labels into the composite. Hunt: synthetic images with one box
   each, assert the merged target count.
3. **LR and loss scaling under DDP.** Mean-normalized losses need no
   world-size scaling (gradient averaging already handles it); scaling them
   again gives an effective LR multiplied by world size. Hunt: single-GPU vs
   2-process DDP on the same seed, compare loss magnitude and update norms
   (`tests/unit/test_ddp_loss_parity.py` is the pattern).
4. **Checkpoint lifecycle.** `best.pt` must exist after short runs (eval
   cadence vs epochs interplay), `last.pt` must resume, and a save/reload
   must produce identical val metrics (RF1 already asserts this).
5. **Warm-start that isn't.** Loading pretrained weights must actually
   transfer tensors; a silently-empty load trains from scratch and looks
   like "slow convergence". Hunt: compare a few backbone tensor checksums
   before/after load, and expect epoch-1 val mAP well above random for a
   warm start.
6. **Val-side bugs masquerading as training bugs.** A preprocessing mismatch
   in the validator (letterbox scaling, class maps) makes a healthy model
   look broken. Before blaming training, run val on the *pretrained*
   checkpoint: if that number is already wrong, the bug is in val.

## Family caveats worth knowing

- RF-DETR ignores the generic YOLO augmentation knobs and takes an absolute
  `lr` (not `lr0`); its trainer has its own recipe. Do not "fix" that.
- Trainers orchestrate, families own recipes (`libreyolo/training/trainer.py`
  `BaseTrainer` + per-family subclasses). A shared-trainer change needs rung
  1 across *several* families, flagship (YOLO9, RF-DETR) at minimum.
- Inference-only families (`l2cs`, `pidnet`, `depth_anything`, the legacy
  Darknet lineage, open-vocab tier) have no training to verify; check
  `SUPPORTED_TASKS`/docs before promising trainability.

## Watching a run

Every run writes live monitoring files into its `save_dir`: `status.json`
(state, epoch, ETA, latest/best metrics, error on failure), `metrics.jsonl`
(per-epoch history), `train.log` (console tee). Read `status.json` instead of
tailing logs; `libreyolo monitor [root]` serves a browser dashboard over any
number of runs, live or finished.

## Reporting the result

State the rung reached, per family: "YOLO9-t passes rung 0 and RF1;
regression suite green; no rung-4 claim made." Never say "training works"
from a completed `train()` alone, and never present a family on the
RF1 skip list as having validated convergence without saying so.

## Related

- `skills/libreyolo-run-e2e-tests/`: mechanics of running RF1/RF5 correctly.
- `skills/libreyolo-profiling/`: when training is *slow* rather than wrong.
- `skills/launch-serverless-gpu-job/`: rung-4 runs on rented GPUs.
- `docs/testing.md`: where each training tier lives in CI.

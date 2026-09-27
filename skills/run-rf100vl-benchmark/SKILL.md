---
name: run-rf100vl-benchmark
description: >-
  Run the RF100-VL (Roboflow 100-VL) detection benchmark on LibreYOLO models
  end to end: dataset download with version locking, per-dataset fine-tuning,
  protocol-conformant pycocotools evaluation at maxDets 500, artifact
  publishing, and multi-GPU execution on vast.ai (the default compute). Use
  when someone wants to "run RF100-VL", "benchmark on Roboflow 100",
  reproduce or extend published RF100-VL numbers, add a model family to the
  campaign, asks why campaign GPUs look underutilized or the ETA looks wrong,
  or needs to decide stop-vs-destroy / packing depth. Training and eval run
  in the vision-analysis-benchmark harness; this skill holds the protocol,
  locked decisions, compute playbook, and the operational lessons that
  change money and wall-clock.
---

# Run the RF100-VL benchmark

RF100-VL: 100 real-world detection datasets from Roboflow Universe (164,149
images, 564 classes, 7 domains), paper arXiv 2505.20612 (NeurIPS 2025
Datasets and Benchmarks). Each dataset ships fixed `train`/`valid`/`test`
splits in COCO JSON. There is no official runner; the harness referenced
below is the runner.

Exact install/launch commands: harness
`docs/rf100vl-operator-runbook.md`. Vast rental mechanics (2FA/TOTP, launch,
destroy): `skills/launch-serverless-gpu-job` (Vast section) or the personal
`vast-launch` skill. This skill is the protocol plus the operational
knowledge those two do not cover.

## Protocol (locked decisions)

| Decision | Value | Authority |
|---|---|---|
| Scoring | pycocotools, `maxDets=500`, per-dataset `test` split | paper |
| Headline metric | AP 0.50:0.95, unweighted mean over the 100 datasets | paper |
| Per-domain means | published alongside (7 domains) | our addition |
| Split discipline | train on `train`, select on `valid`, report on `test` | Roboflow reference code |
| Selection | validate every epoch, EMA weights, keep best AP50:95 | Roboflow reference code |
| Epoch budget | fixed 100, early stopping DISABLED (`patience: 0`) | Roboflow reference code; the harness enforces it |
| Effective batch | 16 (physical batch x gradient accumulation) | Roboflow reference code |
| Precision | whatever the family's upstream reference trains at, declared in the recipe and recorded in every submission. fp32 for the YOLO trio, fp16 for `ec`, bf16 for `rfdetr`. Never a precision its authors do not use, in either direction | harness #10; measured 2026-08 |
| Eval thresholds | conf 0.001; NMS IoU 0.65 for NMS families, identical at selection and final eval; DETR families are NMS-free top-k | LibreYOLO policy |
| Seed | 0 | LibreYOLO policy |
| Recipes | one pinned JSON per family in the harness (`va_bench/recipes/rf100vl/`), sha recorded in every run, same recipe for all 100 datasets | LibreYOLO policy |
| Execution (YOLO) | `cuda_graph: true` and `cache: "disk"` in the recipe protocol when the installed LibreYOLO supports them (bit-identical; covered by recipe hash). Verify capture per family, per commit and per card: see "CUDA graphs and cache" | measured 2026-08 |
| COCO evaluator | faster-coco-eval (Apache-2.0), now the LibreYOLO default; record the backend in artifacts | measured 2026-08 |

**"Roboflow reference code" means what they RAN, not what the library
defaults to.** For RF-DETR those differ, and the difference is invisible in a
result table. Their maintainer states it in
[rf-detr#266](https://github.com/roboflow/rf-detr/issues/266): the published
RF100-VL numbers used "the default arguments but with batch size 16 and grad
accum 1", posted three hours after the 1.2.0 release, so read the defaults off
that tag rather than off `develop`, which has since drifted (`batch_size` now
takes `"auto"`, `amp_dtype` became a selector). The library default is batch 4
with accum 4. Both give effective batch 16, but `multi_scale` draws a fresh
resolution per forward, so accum 4 averages four scales into an optimizer step
where they averaged one. Two things in their numbers are not reproducible at
all and belong in the recipe as disclosed deviations: they initialised from
private Objects365 checkpoints that were never released (they estimate ~0.1
mAP), and their table predates their own fix for an incorrectly initialised
head.

Never report toolkit-native trapezoidal mAP; it inflates up to 2.7 AP on
RF100-VL versus pycocotools (paper, App B). LibreYOLO validation is
pycocotools-based already; the 500 cap is the opt-in `eval_max_det` kwarg
(`model.val(data=..., split="test", eval_max_det=500)`). Defaults for normal
users are unchanged (AP at maxDets 100) and test-locked.

**Scoring can cost more than training, and it scales with box density, not
image count.** RF100-VL's `gwhd2021` has 43.6 boxes/image against a median of
4.9; at conf 0.001 and maxDets 500, stock pycocotools spent **496 s/epoch
scoring against 112 s training** — 4.4×, and 6.9 of that dataset's 9.6 h.
faster-coco-eval removes it: verified across all 100 test splits at
max_det=500, **1381/1400 metric values bit-identical, max deviation 2.22e-16
(one float64 ULP), headline mean AP delta exactly 0**, wall 131.4 s → 8.4 s
(15.6× overall, 56× on gwhd2021). Scoring cost tracks the MEAN boxes/image, so
the datasets to expect it on are `uavdet-small` (75.9), `recode-waste` (44.3)
and `gwhd2021` (42.9), against a median of 4.9. Keep stock pycocotools as the
reference implementation for audit; do not silently swap backends without
recording which one ran.

Validation frequency is protocol-mandated every epoch (gdino
`val_interval=1`; rt-detr / d-fine / lw-detr call `evaluate()` unconditionally,
lw-detr twice). The knob that exists is validation COST (image cache, hoisted
validator, graphed val forward), not frequency. LibreYOLO's own
`TrainConfig.eval_interval` defaults to 10 and the final epoch always
validates, so a short smoke validates once at the end; pass
`eval_interval=1` to validate every epoch.

## Where the work happens

Repo `LibreYOLO/vision-analysis-benchmark` (harness branch `rf100vl-harness`).

```bash
# per-dataset fine-tuning: one worker per lane, subprocess children,
# atomic status files, resume, timeouts, dense-dataset OOM fallback
va-bench rf100vl-train --data-dir ./rf100-vl --weights-root ./rf100vl-weights \
  --gpus 0,1,2,3,4,5,6,7 --jobs-per-gpu 3

# per-dataset test-split eval
va-bench rf100vl --all --data-dir ./rf100-vl --weights-root ./rf100vl-weights

# both plus checks and rendering as ONE resumable command
va-bench rf100vl-campaign --model yolov9t --data-dir ./rf100-vl \
  --weights-root ./rf100vl-weights --gpus 0,1,2,3,4,5,6,7 --jobs-per-gpu 3
```

`--jobs-per-gpu` is the biggest throughput lever, **but throughput is not the
makespan**. Size it against the LONGEST dataset, not just VRAM. Measured
2026-08: going 8 → 16 lanes on one box made every lane 1.74× slower (uniform
across all 16 datasets, so it is contention, not any one model) and therefore
*lengthened* the serial long pole, finishing later overall. When one dataset
sets the makespan, packing deeper is strictly harmful. Pack when total work
dominates; do not pack when a single dataset does.

Pin `OMP_NUM_THREADS` (and `MKL_/OPENBLAS_`) whenever you pack. Left unset,
torch takes 64 intra-op + 128 inter-op threads *per process*: 16 lanes gave
~124 threads per dataloader worker, 11,890 threads on 128 cores, and **67.6%
system time with 6% idle** — two thirds of the CPU spent on scheduling, not
work. Pinning to 2 took system time to 7.4%.

CUDA graphs improve packing because they remove host launch contention between
lanes sharing a GPU. They also *add* persistent static buffers, so measure
VRAM with graphs in the configuration you will actually run.

Supporting verbs: `rf100vl-preflight`, `rf100vl-dash` (pass `--data-dir` or
queued datasets show no image counts and the ETA is size-blind),
`rf100vl-report`.

- **Dataset, fast path.** Pull [`LibreYOLO/rf100-vl`](https://huggingface.co/datasets/LibreYOLO/rf100-vl):
  100 per-dataset tars + lock files. Prefer `max_workers=32` and stay logged
  in (`HF_TOKEN`) for the authenticated rate limit. Do not cargo-cult
  `huggingface_hub[hf_transfer]` / `HF_HUB_ENABLE_HF_TRANSFER=1` on hub 1.x.
- **Dataset, canonical path.** `--download` wraps the `rf100vl` pip package
  (`ROBOFLOW_API_KEY`); use it to rebuild/verify the HF copy. Licensing is in
  `va_bench/data/rf100vl_licenses.json`.
- Weights: `best.pt` at `<weights_root>/<dataset>/<weight_file>`.
- A capability guard aborts on builds without `eval_max_det`/`amp_dtype`.
  `cuda_graph` / `cache` are reported, not required — missing them runs the
  protocol correctly, just slower.
- Flag lists: harness README / `--help` win over this skill.

## Decisions BEFORE dataset one (one-way doors)

The run signature hashes the recipe. **Any recipe change
(`cuda_graph`, `cache`, epochs, imgsz, …) means a fresh campaign** — banked
checkpoints under the old recipe cannot be resumed. Enabling `cuda_graph`
mid-campaign once orphaned 973 banked epochs.

1. Recipe final? (`va_bench/recipes/rf100vl/*.json`)
2. LibreYOLO commit pinned and *actually* installed?
   `pip install --upgrade` on a git URL **silently no-ops when the version
   string is unchanged** and reports success. Use
   `pip install -q --force-reinstall --no-deps "git+..."`, then prove the
   installed `TrainConfig` has the fields the recipe sets.
3. Vast TOTP seed saved (`~/.config/vastai/vast_totp_seed`)? A 2FA session
   expiring mid-campaign once forced stopping a billing box through an
   unverifiable path.
4. Shakedown done on the current stack? (below)

## Compute: vast.ai (default)

Account setup, 2FA, launch, exec, pull, destroy: follow
`skills/launch-serverless-gpu-job` (Vast section). RF100-VL specifics:

- **Image, tested end to end:**
  `vastai/pytorch:2.11.0-cu128-cuda-12.9-mini-py312-2026-06-15`. Vast's own
  image takes their key-injection path; plain `pytorch/pytorch` sometimes
  leaves sshd rejecting keys. cu128 is required for 5090 (`sm_120`).
  Interpreter: `/venv/main/bin/python`, not bare `python`.
- **Accept or reject in 60 seconds** with harness
  `deploy/vast/accept-box.sh` (GPUs, kernels, matmul, HF, PyPI, disk ≥ 120
  GB free). Destroy duds immediately. **`loading` is essentially unbilled**
  (meter starts at `running`); a wedged 15-minute pull destroyed after
  measured ~$0.02. An older estimate of "~$1 for the host search" overstated
  this by ~10x — the real cost is operator attention. Never nurse a doubtful
  host because destroying it "feels wasteful."
- **Disk allocate ~120 GB without image cache, ~250 GB with
  `cache: "disk"`.** The offer filter "machine has ≥ 300 GB free" and the
  `--disk` you rent are different; disk bills on allocated GB. One campaign
  needs roughly 70 GB (image + pip + 49 GB dataset + weights); disk-cache
  `.npy` sidecars add **~108 GB at 416px and ~159 GB at 640px** (see the disk
  projection above — 640px does not fit 250 GB without purging as you go).
- Workload is **CPU / host-bound**, priced GPU-centric. Measured
  [pre-cache]: 46 ms GPU vs 507 ms CPU per step; 8 cores/lane still ~94%
  CPU-saturated. `pick_box.py`'s old `MIN_CORES_PER_LANE = 3.0` was far too
  low. Weight core count and single-core clock heavily (a 2.6 GHz EPYC 7K62
  lost to a consumer Ryzen). Size cores for epoch 1 (cache fill + all lanes
  cold), not the steady-state average. Prefer high-clock CPUs even if the
  GPU $/hr is slightly higher.
- Primary target: one 8× RTX 5090 interruptible box with strong CPU; pack
  with `--jobs-per-gpu` after a VRAM/lane re-measure. Fallback: several
  1–4× 5090/4090 boxes.
- Offer filter: verified, reliability > 0.99, **≥ 8 vCPU per lane after
  packing**, machine disk ≥ 300 GB free (so 120–250 allocated fits), ≥ 500
  Mbps down, download < 0.01 USD/GB, host driver CUDA 12.8+, distinct
  egress IP from known-bad NATs. Bid 20–30% above minimum.
- Interruptible: outbid pauses the box; disk persists and bills; destroy
  deletes. Harness resumes at dataset (status files) and epoch (`last.pt`)
  level. Sync weights/results off-box at milestones; always pull before
  destroy.
- Local-first: one dataset, then rf20vl pilot, then rent. When the stack
  changes (new LibreYOLO commit, recipe, packing, image), shake out on ONE
  cheap GPU first (~$1).
- **Select on `$/GPU-hour`, not `$/hour`.** Measured 2026-08: 8× RTX 5060 Ti
  (16 GB, 128 cores) at $0.79/hr is **$0.099/GPU-h**; an 8× 5090 box is
  $0.467/GPU-h for perhaps 2.5–3× the throughput. On a host-bound workload the
  cheap many-core box usually wins. Check VRAM separately (see sizing above) —
  16 GB is the constraint that actually rules boxes out.

### Running several boxes at once

Each box is fully independent: the harness has no cross-box coordination, so
"multi-node" is really N single-node campaigns plus your attention. That makes
the failure modes operational, not algorithmic.

- **One chain script per box** (wait → campaign → upload → stop), launched
  detached with `setsid nohup ... < /dev/null &`. Put a **gate at the top** that
  asserts the stack invariant you care about and aborts before spending a
  night — e.g. build the model and assert BN eps, or assert the eval backend.
- **One monitor per box**, and make its filter cover failure, not just
  progress: a monitor that only greps the happy path is silent through a
  crashloop, which looks identical to "still running".
- Crash recovery is genuinely good and does not need you: dataset+epoch level
  resume, atomic status files, per-dataset timeouts, OOM → grad-accum fallback.
  ~300 dataset-runs completed with 0 failures unattended.
- **Silent wrongness is what needs you**, and it scales badly with box count.
  Every serious problem in the 2026-08 campaigns was silent: wrong BN eps,
  uploads skipping checkpoints, disk projections, a bad ETA. Before fanning
  out, make sure each of those has a loud check; otherwise N boxes produce N
  results you must hand-verify anyway.

### What healthy looks like (do not "fix" this)

- Launch/host-bound: [pre-cache] healthy meant GPUs at **9–35% util and
  ~170 W of 575 W** with everything fine. Low GPU numbers are the signature
  of this workload, not a fault. The runbook once claimed 60–100% util as
  healthy; that reading burned an expensive detour.
- **Re-baselined 2026-08 with disk cache + faster eval**: dataload fell to
  **0.2 ms of a 350 ms step**, so the pre-cache "507 ms CPU per step" figure
  no longer describes the loop. Util now runs ~39% mean exclusive and 74–94%
  when packed. Both bands are healthy; read util next to power, and treat a
  *change* against your own shakedown as the signal rather than any absolute
  number.
- `nvidia-smi` util is time-with-a-kernel-resident on the CARD, not die
  occupancy per job. With 3 lanes sharing a GPU the row describes the card.
- Datasets are heterogeneous: **92 to 8,791** train images, **0.31 to 12+
  MP**. Epoch times span ~40×. Longest-first scheduling; one dataset sets
  the makespan.
- Epoch 1 costs 1.3–2.1× a steady epoch [pre-cache]; with post-resize cache
  the ratio **widens** (epoch 1 fills the cache). **Every ETA in the first
  hour is garbage** — do not make money decisions off it. A "16.4 h" ETA
  was once an artifact of averaging epoch 1 into a two-epoch mean.

### Sizing a box: measure VRAM, never predict it

Do not derive VRAM from parameter count or GFLOPs. Measured 2026-08 at 640px,
batch 16: **yolox-s (8.97M) = 6.2 GB/lane, yolov9s (~9M) = 10.8 GB/lane** —
same size, 1.7× apart. Activation memory, cuDNN workspace autotuning,
allocator behaviour and CUDA-graph static buffers all dominate, and none of
them follow parameter count.

Measure instead, with a 2-epoch smoke campaign rather than the profiler:

```bash
va-bench rf100vl-campaign --model <M> --recipe <R> \
  --datasets <one-dense-dataset> --gpus 0 --jobs-per-gpu 1 --smoke-epochs 2
```

**Do not size a lane with `libreyolo profile run`.** It does not capture CUDA
graphs and does not run EMA or validation, so it under-reports the real lane.
It measured yolo9-m at 15721 MB, which looks survivable on a 16 GB card; the
graphed campaign then OOM'd on **all 100 datasets** with 13.61 GiB sitting in
graph private pools. The profiler is the right tool for *where time goes*, not
for how much memory a lane needs.

Pick a **dense** dataset for the smoke, not a convenient one. `actions` is
sparse and gave ec-m 13939 MB; the real campaign peaked at 14787 MB on
`gwhd2021` (43.6 boxes/image).

**Dense for memory and dense for scoring are different lists, and this skill
used to give the scoring one for both jobs.** Peak VRAM is set by the WORST
SINGLE IMAGE in a batch, not the average, so size a lane against max
boxes/image: `exploratorium-daphnia` (349), `all-elements` (279),
`penguin-finder-seg` (246). The scoring-cost trio above is a poor proxy for it:
`gwhd2021` maxes at 190 and `uavdet-small`, densest of all by mean, at only
109. Measured on the train splits, which is what the harness reads when it
builds the batch plan.

The same asymmetry sets the OOM fallback: `dense_oom_fallback` triggers on
`max_annotations_per_image`, so at the recipes' threshold of 200 exactly those
three datasets drop to the fallback batch, on every family. That is a protocol
deviation on 3 of 100 datasets which fires whether or not the lane would
actually have OOM'd, so check it is the trade you want before dataset one.

Take ~20% headroom. The registry's `params_millions`/`gflops` fields are `0.0`
for most non-flagship specs, and parameter count does not predict memory
anyway: yolo9-m at 20.1M needs 15207 MB while yolonas-s at 19.05M needs 7615,
2x apart at nearly identical size.

Being wrong is survivable but not free: for `rfdetr` the harness falls back to
grad-accum and **restarts that dataset from epoch 0**. For every other family
the fallback is not wired up (`_batch_plan` gates it on
`spec.family == "rfdetr"`), so a mid-run OOM is fatal for that dataset even
though the scheduler asks for a fallback.

### Packing: decide by utilisation, not by free memory

Two lanes fit is not the same as two lanes help. Run one lane and read
`utilization.gpu`:

| model | 1 lane VRAM | 1 lane util | verdict |
|---|---|---|---|
| ec-s | 7.0 GB | 44% | 2 lanes: **~2x throughput**, second lane nearly free |
| ec-m | 10.3 GB | ~100% | 1 lane: packing buys nothing, and 2 would not fit |
| yolonas-s | 9-11 GB | 88-100% | 1 lane |

Measured on ec-s: 78.0 s/epoch alone against 80.3 s with a neighbour, so the
second lane cost ~2% and doubled the work. On a saturated model there is no
idle capacity to sell, and packing only adds OOM risk. Sample utilisation over
30 s or so, because ~20% of each epoch is validation, where a single sample
legitimately reads 0%.

### Disk: project it, do not eyeball it

Two campaigns nearly died on this. Post-resize cache measured
**~953 KB/image at 640px** and **~660 KB at 416px**, so:

    cache_GB ≈ total_images × bytes_per_image
    RF100-VL (163,151 images): ~159 GB at 640, ~108 GB at 416

That is far above the "~105 GB of sidecars" this skill used to quote. Add the
dataset (~49 GB) and checkpoints (measured 108 MB each for a ~9M-param model,
so ~30-56 GB across 100 datasets) and 640px does **not** fit a 250 GB disk.

Fix: purge each dataset's cache when it completes. The cache is reused across
that dataset's 100 epochs and is dead weight afterwards, so purging on
`state == done` bounds it to the active working set (~54 GB) instead of the
full ~159 GB. Never delete anything but `*.r<W>x<H>.npy`.

### When something looks slow: profile, do not theorize

`libreyolo profile` answers "where is the time going" in under a minute.
An hour of py-spy (blocked by container caps), `ps` aggregation, and
log-timestamp forensics produced a confidently wrong answer that
`libreyolo profile run` corrected in 52 seconds. The campaign should print
a Profile hint next to the Monitor hint; if it does not, still run:

```bash
libreyolo profile run ...      # one profiled training epoch
libreyolo profile phases ...   # train / validation / save split
libreyolo profile what-if ...  # projected gain from a fix
```

High self-CPU on an op like `aten::max` with near-zero self-CUDA usually
marks a GPU→CPU sync absorbing device wait, not a CPU bottleneck in that
op. Total self-CUDA ≪ total self-CPU ⇒ launch-bound (CUDA graphs).

**First check that the GPU counters attached at all.** On a rented box where
CUPTI fails to load, the profiler still prints a full report, but every
device-side number is zero and the VERDICT line is then an artifact rather
than a measurement. The cause is a version mismatch, not a permissions
problem, so the NVIDIA doc link the error prints is a red herring:

```
CUPTI initialization failed: CUPTI_ERROR_INVALID_DEVICE (2)
```

CUPTI has to match the DRIVER, not the toolkit. A Vast host on driver
595.58.03 (CUDA 13.2) with an sm_120 card meets the `cu128` wheel's bundled
`nvidia-cuda-cupti-cu12 12.8.90` and cannot enumerate the device. There is no
cheap fix: `nvidia-cuda-cupti-cu13` on PyPI is a 0.0.1 stub, the image ships
no system CUPTI, and a real fix needs torch built for CUDA 13. **Rent a host
whose driver CUDA major matches the wheel's CUPTI major, or accept
wall-clock-only profiling.** Measured 2026-08 on an 8x RTX 5060 Ti box:

```
REAL step 362.3 ms = dataload 0.1 ms + compute 362.3 ms  ->  44.2 img/s
GPU util 0%  (0 ms GPU-busy / 362 ms step)  |  0 kernels/step @ ~0us
>> VERDICT: HOST/LAUNCH-BOUND - GPU only ~0% busy
```

That verdict was wrong: enabling CUDA graphs on the same model and batch
changed the epoch time by nothing. `0 kernels/step` is the tell. When you see
it, ignore the verdict and the "self-CUDA vs self-CPU" rule above, both of
which are derived from the dead counters, and read the per-phase wall table
instead, which is host-side timing and stays valid:

```
phase         gpu_ms  wall_ms  kernels     ops
to_device        0.0      5.9        0       8
forward          0.0    127.9        0    4562
backward         0.0    222.9        0    5093
optimizer        0.0      6.3        0    2800
```

### CUDA graphs and cache: prefer them, then verify both

Keep `cuda_graph: true` and `cache: "disk"` as the default recipe protocol.
Both are perf-only and covered by the recipe hash. Two things to verify
before trusting either, because each has cost a campaign.

**Capture is gated by family and by commit.** At an older pinned LibreYOLO
only `yolo9` and `rfdetr` implemented the training capture hook; every other
family took the eager fallback, so setting the flag did nothing but log a
warning. Current dev covers far more families (`docs/training_cuda_graphs.md`
holds the table and the per-family speedups). Check that table for the commit
you actually pinned, not for `dev`, and confirm the run log says
`cuda_graph: captured training forward/backward at input shape (...)`. Absent
that line, the flag is decorative.

**The speedup is batch-dependent, and can be zero.** The documented gains are
measured at batch 8, where launch overhead is a large share of the step. At
the protocol's batch 16 the same model can show nothing: yolonas-s measured
83.4 / 78.9 s per epoch eager against 83.6 / 78.4 s graphed, with capture
confirmed in the log. yolo9 still gains at batch 16. Measure before assuming.

**Capture inflates VRAM, and `libreyolo profile run` will not show it**
because it does not capture graphs. yolo9-m at 640 and batch 16 profiled at
15721 MB, which looks survivable on a 16 GB card. The real graphed run OOM'd
on **all 100 datasets**, with the failure naming the cause:

```
13.61 GiB allocated in private pools (e.g., CUDA Graphs)
```

Disabling capture on that box dropped the same run to 15207 MB and it trained
fine at full protocol batch. So size the lane with a 2-epoch smoke campaign,
which exercises the real path including capture, EMA and validation:

```bash
va-bench rf100vl-campaign --model <M> --recipe <R> \
  --datasets <one-dataset> --gpus 0 --jobs-per-gpu 1 --smoke-epochs 2
```

For a NMS-family model that cannot hold capture on the target card, dropping
`cuda_graph` is the cheap deviation: the harness treats graph replay as
outside the run signature and bit-identical on loss. Dropping physical batch
is the expensive one, because it changes BatchNorm statistics even though
`nbs` keeps the effective batch at 16.

### Contribute the speedup back

The campaign is a load test of LibreYOLO training, so treat what the profiler
finds as a LibreYOLO bug list rather than a campaign workaround list. The
scoring cost above is the model: it was found by profiling a campaign, fixed
in the library as a default, and now every user benefits. When a fix belongs
in the library, send it there and pin the campaign to the commit that carries
it, rather than carrying a private patch on the box.

### Shakedown (before any full campaign)

1. Rent one cheap high-clock-CPU GPU.
2. Install exactly as the campaign would (`--force-reinstall --no-deps`);
   prove recipe fields exist on the installed `TrainConfig`.
3. Run 2–3 representative datasets (tiny ~100 imgs, huge ~8k, large-image
   12 MP class), 10–15 epochs, `eval_interval=1`, plus one full completion.
4. Kill-and-resume between checkpoints (not mid-write).
5. Record VRAM/lane, cores/lane at epoch 1 and steady state, epoch-1 ratio,
   validation share, healthy util band. Size the real box from those numbers.

### Stopping and resuming

- **Ctrl-C mid-checkpoint can corrupt `last.pt`** and poison resume for that
  dataset. Stop gracefully: `tmux send-keys -t bench C-c`, wait for the
  orchestrator to exit, then `vastai stop instance` if keeping the box.
- **`pkill` on live workers does the same thing, and an agent restarting a
  campaign will reach for it.** One dataset out of 33 came back as a 0.5 MB
  `last.pt` after a routine restart, and the next attempt died on
  `PytorchStreamReader ... failed finding central directory`. The recovery is
  to delete the corrupt file so the dataset restarts clean; the retry pass
  then picks it up. Worth sweeping every `last.pt` with a `torch.load` after
  any hard kill, because the corruption is silent until the resume.
- A **stopped** box bills disk only; restart needs those exact GPUs still
  free. Decide stop-vs-destroy from disk $/day, re-stage cost (~35 min,
  nearly free bandwidth), and whether banked checkpoints are usable
  (recipe change ⇒ they are not).
- **Stopped is not restartable on demand.** 2026-08 a restart returned
  `resources_unavailable, state change queued` and `intended_status` reverted
  to `stopped`; the GPUs came back only hours later. Treat a stop as
  "possibly forever". **Sync every box-local file you cannot regenerate BEFORE
  stopping** — recipes, chain scripts, any hand-written config. `sync-artifacts`
  does not upload the recipe, and a run whose `recipe_sha256` points at a file
  that exists nowhere is not reproducible.
- Instance ops need a live 2FA session (~7 days) and `ssh<N>.vast.ai` DNS can
  fail transiently. Keep the dashboard tunnel up: an established tunnel
  survives a DNS outage and is the only health channel left when ssh will not
  resolve.
- Destroy ends spend. Always pull/sync first.

## Decisions to take per campaign

1. Model list and sizes. Flagships deep (yolo9 and rfdetr); other families
   start with their one or two smallest variants.
2. Pilot first: `rf20vl` end to end. Graduate to 100 only if all 20 complete,
   kill-and-resume reproduces within noise, and AP ordering is sane.
3. Budget and waves: price from live offers (`deploy/vast/pick_box.py`),
   publish after flagships land, append families as they finish.
4. Recipe deviations: live in the recipe JSON, hash-recorded, disclosed.
   No silent knobs. Finalize before dataset one.

## Artifacts and publishing

Keep, per model: per-dataset eval JSONs, raw predictions (COCO detections),
per-run `stats.json` (recipe sha, best epoch, seed, dataset version, wall
time), the recipe JSON, the dataset version lock, and the final submission
JSON. Predictions are on by default for publishable runs.

- Upload with `va-bench sync-artifacts` to a LibreYOLO HF dataset repo
  (create the repo by hand; fine-grained write token scoped to that repo).
  Pass `--eval-dir` so predictions are collected. Stock-pycocotools rescore
  of a saved dump reproduced harness AP to four decimals with no harness
  code imported.
- **`sync-artifacts` is silent about what it does not upload.** It ships the
  paths it knows (`eval/`, `stats/`, `submissions/`, and `runs/*/weights` when
  the weights root has the `.runs` layout) and ignores everything else, while
  printing a cheerful `uploaded N, skipped 0`. It does **not** ship the recipe
  JSON, nor arbitrary files placed in the weights root, nor checkpoints from a
  flat `<root>/<dataset>/<file>` layout. **Verify after every publish by
  listing the repo** (`HfApi().list_repo_files`), not by reading the uploader's
  log — and note the tree API paginates at 1000 entries, so a raw count there
  silently truncates.
- Leaderboard submission: `vision-analysis` via `submit-benchmark-results`
  (see `benchmark-on-visionanalysis`).
- Never hand-edit result JSONs. Regenerate them.

### Always check: does selection agree with the test score?

The cheapest possible guard, and the one whose absence cost the most. Compare
each run's `valid_mAP50_95` (from `stats.json`) against the published test AP.
For a healthy run they track within noise; a large one-sided gap means the two
paths disagree about the model, not that the model generalises badly.

2026-08: yolox-nano reported valid 0.5663 and test 0.1620 on `ball` while
yolox-tiny agreed to 0.002 on the same datasets. The cause was BatchNorm eps
(models trained at 1e-5, evaluated at 1e-3, ruinous for the depthwise nano
only), fixed in libreyolo #700. A whole 100-dataset campaign published a
headline **0.3601 that should have been 0.4853** and nothing flagged it.

The corollary: a **16-point gap between adjacent sizes of one family is not a
result, it is a bug report.** Adjacent sizes land ~3 points apart.

## Traps

- Stock pycocotools `summarize()` breaks with a non-default maxDets list
  (headline AP becomes -1). Never call it with a modified list.
- Package-cleaned data only. Raw Universe exports keep the dummy class and
  score near zero.
- Dataset named `-grccs`: `--datasets=-grccs` (space form is eaten by argparse).
  The same leading hyphen breaks any shell tooling you write around the
  campaign: `grep -qxF "$d"` parses it as options and exits 2, so a cleanup
  guard written that way fails open and deletes the cache of a dataset that is
  actively training. Use `grep -qxF -- "$d"`.
- **Anything that fails at worker startup destroys the most expensive datasets
  first.** Longest-first scheduling puts the biggest datasets in the opening
  batch, which is exactly the batch that races on a shared resource. A weight
  auto-conversion race took out `flir-camera-objects` (8792 images),
  `paper-parts` (8423) and `underwater-objects` (5253) on every launch, while
  the small datasets sailed through. If the same names keep failing, suspect
  startup contention rather than the datasets.
- **A single `nvidia-smi` sample is not a health check.** Roughly 20% of each
  epoch is validation, where utilisation legitimately reads 0% while ~12 GB
  stays allocated. Sample over 30 s, or read epoch progress instead.
- Never resume after changing physical batch, accumulation, **or recipe
  fields covered by the run signature** (including `cuda_graph` / `cache`).
- Dense datasets can OOM rfdetr; harness falls back to grad-accum and
  restarts that dataset from epoch 0. Expected.
- ec family: AdamW, no mosaic (mosaic triggers a degenerate-box assertion).
- Largest datasets take hours; raise timeouts for slow families, never remove.
- Differences under 0.5 mAP on the 100-dataset mean are noise. Replicate.
- `ROBOFLOW_API_KEY` and vast credentials: env/local config only. Never commit.

## Published numbers to beat (fully supervised, AP 0.50:0.95)

RF-DETR N/S/M/L/XL/2XL: 57.7 / 60.2 / 61.2 / 62.2 / 62.9 / 63.2. LW-DETR
T to X: 57.1 to 62.1. D-FINE N to X: 58.2 to 62.2. YOLO11 N to X: 55.3 to
56.5. YOLO26 N to X: 52.0 to 60.0. Sources: the rf-detr repo README
(develop) and paper v4 tables. No YOLO9-lineage numbers exist anywhere yet.

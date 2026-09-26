"""Inference profiler: latency + GPU-truth for the predict path.

Companion to the training profiler. Drives a model's
``_preprocess -> _forward -> _postprocess`` stages directly over a fixed source
for a short measured window and reports:

* **Latency** (unsynced, end-to-end): p50/p90/p99/mean per-iteration latency and
  throughput (img/s) at the requested batch.
* **Stage composition** (synced): preprocess / forward / postprocess(NMS) ms.
* **GPU truth**: the same kernel / op / category / phase breakdown as training,
  by reusing :func:`libreyolo.training.profiler.analyze_trace` on an emitted
  Chrome trace — so ``libreyolo profile summary/kernels/ops/phases/get/compare``
  all work on inference profiles unchanged.

It never edits the predict path: it calls the model's public stage methods, so
there is zero overhead on normal inference. Splitting the window matters for the
same reason it does in training — the unsynced half gives honest latency, the
synced half attributes per-stage time; we never derive the verdict from synced
numbers.

Targets detection models (the flagship predict path). Other tasks work
best-effort through the same stage methods.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.profiler import record_function

from libreyolo.utils.amp import normalize_amp_dtype, torch_amp_dtype

# Inference phase order for the shared analysis (`analyze_trace`'s _ORDER also
# lists these so per-phase tables read pre -> forward -> post).
_STAGES = ("preprocess", "forward", "postprocess")


class InferenceProfiler:
    """Profiles a short window of inference steps; reuses ``analyze_trace``.

    Parameters mirror the training profiler where they overlap. ``model`` is a
    loaded LibreYOLO model exposing ``_preprocess``/``_forward``/``_postprocess``
    (i.e. any ``BaseModel``); it is duck-typed so tests can pass a stub.
    """

    def __init__(
        self,
        model,
        *,
        warmup: int = 20,
        runs: int = 100,
        batch: int = 1,
        imgsz: Optional[int] = None,
        half: bool = False,
        amp_dtype: str = "float16",
        trace: bool = True,
        save_dir: Optional[Path] = None,
        logger=None,
        meta: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.warmup = max(0, int(warmup))
        self.runs = max(2, int(runs))
        self.batch = max(1, int(batch))
        self.imgsz = imgsz
        self.half = bool(half)
        self.amp_dtype = normalize_amp_dtype(amp_dtype)
        self.trace = bool(trace)
        self.save_dir = Path(save_dir) if save_dir else None
        self.logger = logger
        self.meta = dict(meta or {})

        dev = getattr(model, "device", None)
        self.device = torch.device(dev) if dev is not None else torch.device("cpu")
        self._is_cuda = self.device.type == "cuda"
        # First half of the measured window is unsynced (honest latency); the
        # second half syncs each stage to attribute composition.
        self._half = max(1, self.runs // 2)
        self.summary: Optional[dict] = None

    # -- stage drivers (mirror InferenceRunner._predict_single/_predict_batch) --

    def _preprocess(self, images, color_format, imgsz):
        tensors, sizes, ratios = [], [], []
        for im in images:
            t, _orig, size, ratio = self.model._preprocess(
                im, color_format, input_size=imgsz
            )
            tensors.append(t)
            sizes.append(size)
            ratios.append(ratio)
        stacked = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
        return stacked, sizes, ratios

    def _forward(self, stacked):
        x = stacked.to(self.device)
        with torch.no_grad():
            if self.half and self._is_cuda:
                with torch.autocast(
                    "cuda", dtype=torch_amp_dtype(self.amp_dtype)
                ):
                    return self.model._forward(x)
            return self.model._forward(x)

    def _postprocess(self, output, sizes, ratios, conf, iou, max_det, classes, imgsz):
        if len(sizes) == 1:
            return self.model._postprocess(
                output, conf, iou, sizes[0], max_det=max_det, ratio=ratios[0],
                classes=classes, input_size=imgsz,
            )
        from libreyolo.postprocess.slicing import slice_batch_outputs

        for off, (size, ratio) in enumerate(zip(sizes, ratios)):
            self.model._postprocess(
                slice_batch_outputs(output, off), conf, iou, size,
                max_det=max_det, ratio=ratio, classes=classes, input_size=imgsz,
            )
        return None

    def _sync(self) -> None:
        if self._is_cuda:
            torch.cuda.synchronize(self.device)

    # -- run -----------------------------------------------------------------

    def run(
        self,
        images: Sequence,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        classes=None,
        color_format: str = "auto",
    ) -> dict:
        """Profile inference over ``images`` (cycled to fill batch+iterations)."""
        if not images:
            raise ValueError("no images to profile")
        imgsz = self.imgsz or self.model._get_input_size()
        if not isinstance(imgsz, (tuple, list)):
            imgsz = int(imgsz)
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "eval"):
            inner.eval()
        if self._is_cuda:
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

        prof = self._start_torch_profiler() if self.trace else None

        latencies: list[float] = []
        stage_ms = {s: [] for s in _STAGES}
        n = len(images)
        for i in range(self.warmup + self.runs):
            batch_imgs = [images[(i * self.batch + j) % n] for j in range(self.batch)]
            measuring = i >= self.warmup
            m = i - self.warmup
            if measuring and m >= self._half:
                # synced: attribute per-stage time
                t0 = time.perf_counter()
                with record_function("step/preprocess"):
                    stacked, sizes, ratios = self._preprocess(batch_imgs, color_format, imgsz)
                self._sync()
                t1 = time.perf_counter()
                with record_function("step/forward"):
                    out = self._forward(stacked)
                self._sync()
                t2 = time.perf_counter()
                with record_function("step/postprocess"):
                    self._postprocess(out, sizes, ratios, conf, iou, max_det, classes, imgsz)
                self._sync()
                t3 = time.perf_counter()
                stage_ms["preprocess"].append((t1 - t0) * 1000.0)
                stage_ms["forward"].append((t2 - t1) * 1000.0)
                stage_ms["postprocess"].append((t3 - t2) * 1000.0)
            else:
                # unsynced end-to-end latency (one sync at the very end)
                t0 = time.perf_counter()
                with record_function("step/preprocess"):
                    stacked, sizes, ratios = self._preprocess(batch_imgs, color_format, imgsz)
                with record_function("step/forward"):
                    out = self._forward(stacked)
                with record_function("step/postprocess"):
                    self._postprocess(out, sizes, ratios, conf, iou, max_det, classes, imgsz)
                self._sync()
                if measuring:
                    latencies.append((time.perf_counter() - t0) * 1000.0)
            if prof is not None:
                try:
                    prof.step()
                except Exception:
                    pass

        trace_path = self._finish_profiler(prof)
        return self._build(
            latencies, stage_ms, trace_path,
            imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
        )

    # -- torch profiler ------------------------------------------------------

    def _start_torch_profiler(self):
        try:
            from torch.profiler import ProfilerActivity, profile, schedule

            acts = [ProfilerActivity.CPU]
            if self._is_cuda:
                acts.append(ProfilerActivity.CUDA)
            p = profile(
                activities=acts,
                schedule=schedule(wait=0, warmup=self.warmup, active=self.runs, repeat=1),
                record_shapes=False, with_stack=False, profile_memory=False,
            )
            p.start()
            return p
        except Exception:
            return None  # trace is best-effort; never break profiling

    def _finish_profiler(self, prof):
        if prof is None:
            return None
        try:
            prof.stop()
            if self.save_dir is not None:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                tp = self.save_dir / "profile_trace.json"
                prof.export_chrome_trace(str(tp))
                return tp
        except Exception:
            return None
        return None

    # -- verdict + build -----------------------------------------------------

    def _verdict(self, pre, fwd, post, util=None):
        """Inference bottleneck from the stage split (+ GPU util when traced)."""
        step = pre + fwd + post
        if step <= 0:
            return "inference", "insufficient timing to classify"
        fr_post, fr_pre = post / step, pre / step
        if post >= fwd and fr_post >= 0.35:
            return "nms / postprocess", (
                f"postprocess (NMS) is ~{fr_post * 100:.0f}% of the step "
                f"({post:.2f} ms) — try lower max_det, higher conf, fewer classes, "
                "or an NMS-free / end-to-end model"
            )
        if pre >= fwd and fr_pre >= 0.35:
            return "preprocess", (
                f"preprocess is ~{fr_pre * 100:.0f}% of the step ({pre:.2f} ms) — "
                "CPU resize/letterbox dominates; try a larger batch, faster decode, "
                "or GPU preprocessing"
            )
        # forward-dominated. GPU starvation ("host / launch") is only meaningful
        # on CUDA; on CPU the forward IS the compute.
        if not self._is_cuda:
            return "compute", f"forward dominates the step ({fwd:.2f} ms) — CPU-bound"
        if util is not None and util < 0.8:
            return "host / launch", (
                f"forward dominates but the GPU is only ~{util * 100:.0f}% busy — "
                "small model / tiny kernels; try a larger batch or a bigger model"
            )
        return "compute", (
            f"forward dominates the step ({fwd:.2f} ms)"
            + (f"; GPU ~{util * 100:.0f}% busy" if util is not None else "")
        )

    def _build(self, latencies, stage_ms, trace_path, *, imgsz, conf, iou, max_det):
        def pct(xs, q):
            if not xs:
                return None
            xs = sorted(xs)
            k = min(len(xs) - 1, int(round((q / 100.0) * (len(xs) - 1))))
            return round(xs[k], 3)

        pre_ms = round(statistics.fmean(stage_ms["preprocess"]), 3) if stage_ms["preprocess"] else 0.0
        fwd_ms = round(statistics.fmean(stage_ms["forward"]), 3) if stage_ms["forward"] else 0.0
        post_ms = round(statistics.fmean(stage_ms["postprocess"]), 3) if stage_ms["postprocess"] else 0.0
        if latencies:
            mean_lat = round(statistics.fmean(latencies), 3)
        else:
            mean_lat = round(pre_ms + fwd_ms + post_ms, 3)
        img_s = round(self.batch / (mean_lat / 1000.0), 2) if mean_lat > 0 else 0.0
        latency = {
            "p50_ms": pct(latencies, 50), "p90_ms": pct(latencies, 90),
            "p99_ms": pct(latencies, 99), "mean_ms": mean_lat,
            "min_ms": round(min(latencies), 3) if latencies else None,
            "max_ms": round(max(latencies), 3) if latencies else None,
            "n": len(latencies),
        }
        stages = {"preprocess": pre_ms, "forward": fwd_ms, "postprocess": post_ms}

        peak_vram = None
        if self._is_cuda:
            try:
                peak_vram = round(torch.cuda.max_memory_allocated(self.device) / 1e6, 1)
            except Exception:
                peak_vram = None

        meta = dict(self.meta)
        meta.update({
            "batch": self.batch, "imgsz": imgsz, "half": self.half,
            "amp_dtype": self.amp_dtype,
            "device": str(self.device), "conf": conf, "iou": iou,
            "max_det": max_det, "mode": "inference",
        })
        summary = {
            "meta": meta,
            "real": {"step_ms": mean_lat, "img_per_s": img_s,
                     "dataload_ms": 0.0, "dataload_frac": 0.0},
            "latency": latency, "stages_ms": stages,
            "bound": None, "bound_why": "",
        }
        if peak_vram is not None:
            summary["analysis"] = {"peak_vram_mb": peak_vram}
        if self.save_dir is not None:
            try:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                (self.save_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2))
            except Exception:
                pass

        # Full analysis via the shared analyzer, then overlay the verdict + the
        # inference-specific blocks (latency / stages / metrics).
        analysis = None
        if trace_path is not None and Path(trace_path).exists():
            try:
                from libreyolo.training.profiler import analyze_trace

                sp = (self.save_dir / "profile_summary.json") if self.save_dir else None
                analysis = analyze_trace(trace_path, summary_path=sp)
            except Exception:
                analysis = None

        if analysis is not None:
            util = analysis.get("gpu_util")
            if analysis.get("memory_pressure"):
                bound, why = analysis["bound"], analysis["bound_why"]
            else:
                bound, why = self._verdict(pre_ms, fwd_ms, post_ms, util=util)
            analysis["bound"] = bound
            analysis["bound_why"] = why
            self._overlay_inference(analysis, latency, stages, img_s, mean_lat)
            out = analysis
        else:
            bound, why = self._verdict(pre_ms, fwd_ms, post_ms, util=None)
            out = self._fallback_profile(meta, latency, stages, img_s, mean_lat,
                                         bound, why, peak_vram)

        summary["bound"], summary["bound_why"] = bound, why
        if self.save_dir is not None:
            try:
                (self.save_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2))
                (self.save_dir / "profile.json").write_text(json.dumps(out))
            except Exception:
                pass
        self.summary = summary
        return out

    @staticmethod
    def _overlay_inference(a, latency, stages, img_s, mean_lat):
        a["mode"] = "inference"
        a["latency"] = latency
        a["stages_ms"] = stages
        a.setdefault("metrics", {})
        a["metrics"].update({
            "latency_p50_ms": latency["p50_ms"], "latency_p90_ms": latency["p90_ms"],
            "latency_p99_ms": latency["p99_ms"], "throughput_img_s": img_s,
            "preprocess_ms": stages["preprocess"], "forward_ms": stages["forward"],
            "nms_ms": stages["postprocess"], "bound": a["bound"],
        })

    @staticmethod
    def _fallback_profile(meta, latency, stages, img_s, mean_lat, bound, why, peak_vram):
        """A valid, CLI-consumable profile.json when no GPU trace is available.

        Populated with zeros for the GPU-truth fields (no kernels were traced)
        and the measured latency / stage split, so ``summary``/``get``/``phases``
        still work on a trace-less (e.g. ``--no-trace``) inference profile.
        """
        phases = [
            {"phase": s, "gpu_ms_per_step": 0.0, "wall_ms_per_step": stages[s],
             "kernels_per_step": 0, "ops_per_step": 0, "unique_kernels": 0}
            for s in _STAGES
        ]
        metrics = {
            "img_per_s": img_s, "step_ms": mean_lat, "gpu_util": 0.0,
            "gpu_busy_ms": 0.0, "mean_kernel_us": 0.0, "kernels_per_step": 0,
            "unique_kernels": 0, "memcpy_ms": 0.0, "tensorcore_pct": 0.0,
            "peak_vram_mb": peak_vram, "host_overhead_ms": 0.0,
            "launches_per_step": 0, "cuda_mallocs_per_step": 0,
            "memory_pressure": False, "bound": bound,
            "latency_p50_ms": latency["p50_ms"], "latency_p90_ms": latency["p90_ms"],
            "latency_p99_ms": latency["p99_ms"], "throughput_img_s": img_s,
            "preprocess_ms": stages["preprocess"], "forward_ms": stages["forward"],
            "nms_ms": stages["postprocess"],
        }
        for p in phases:
            metrics[f"{p['phase']}_wall_ms"] = p["wall_ms_per_step"]
        return {
            "schema": "libreyolo.profile.analysis/v1", "mode": "inference",
            "trace": None, "model": meta.get("model"), "config": meta,
            "bound": bound, "bound_why": why,
            "step_ms": mean_lat, "img_per_s": img_s,
            "dataload_ms": 0.0, "dataload_frac": 0.0,
            "gpu_util": 0.0, "gpu_util_raw": 0.0, "memory_pressure": False,
            "cuda_mallocs_per_step": 0, "host_overhead_ms_per_step": 0.0,
            "launches_per_step": 0, "gpu_busy_ms_per_step": 0.0,
            "mean_kernel_us": 0.0, "kernels_per_step": 0, "unique_kernels": 0,
            "memcpy_ms_per_step": 0.0, "tensorcore_pct": 0.0, "peak_vram_mb": peak_vram,
            "window": {"wall_ms": 0.0, "real_steps": latency["n"]},
            "latency": latency, "stages_ms": stages,
            "metrics": metrics, "categories": [], "phases": phases,
            "phases_gpu": [], "kernels_by_phase": {}, "ops_by_phase": {},
            "top_kernels": [], "kernels": [], "ops": [],
        }

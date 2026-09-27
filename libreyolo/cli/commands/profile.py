"""profile: analyse training profiles from ``model.train(profile=True)``.

Built for agents. Every subcommand prints results to stdout and supports
``--json``, so an LLM can drive the loop:

    libreyolo profile run   coco1000 --weights LibreYOLO9t.pt --size t   # profile TRAINING
    libreyolo profile infer path/to/img --weights LibreYOLO9t.pt         # profile INFERENCE
    libreyolo profile summary <trace>                # what's the bottleneck?
    libreyolo profile kernels <trace> --top 20       # drill to the kernels
    libreyolo profile ops     <trace> --top 20       # framework/host ops
    libreyolo profile compare <before> <after>       # did my change help?

``run`` and ``infer`` emit the same self-contained ``profile.json`` (schema
``libreyolo.profile.analysis/v1``), so every lens below works on either. The
profiler only measures: read the verdict, change the training/config/code
yourself, re-run, ``compare``, repeat until images/sec is maxed (or latency is
minimised).
"""

from __future__ import annotations

import json as _json
import re as _re
from pathlib import Path
from typing import Optional

import typer

profile_app = typer.Typer(
    name="profile",
    help="Analyse training profiles (model.train(profile=True)) for speed tuning.",
    no_args_is_help=True,
    add_completion=False,
)


def _load(trace: str) -> dict:
    from libreyolo.training.profiler import analyze_trace

    p = Path(trace)
    if not p.exists():
        typer.echo(f"not found: {trace}", err=True)
        raise typer.Exit(2)
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = None
    # A self-contained profile.json (the recommended artifact to copy/compare)
    # is used as-is; a raw profile_trace.json is analysed on the fly.
    if isinstance(data, dict) and str(data.get("schema", "")).startswith(
        "libreyolo.profile.analysis"
    ):
        return data
    try:
        return analyze_trace(p)
    except Exception as exc:  # pragma: no cover - defensive
        typer.echo(f"failed to analyse {trace}: {exc}", err=True)
        raise typer.Exit(1)


def _pct(before, after) -> str:
    if not before:
        return ""
    return f" ({(after - before) / before * 100:+.0f}%)"


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _resolve_infer_source(source: str) -> list[str]:
    """Resolve a file or directory to a list of image paths (empty = not found)."""
    p = Path(source)
    if p.is_dir():
        return sorted(str(f) for f in p.iterdir() if f.suffix.lower() in _IMG_EXTS)
    if p.exists():
        return [str(p)]
    return []


@profile_app.command("run")
def run_cmd(
    data: str = typer.Argument(..., help="Dataset yaml/name (e.g. coco128)"),
    weights: str = typer.Option("LibreYOLO9t.pt", "--weights", help="Model weights file / name"),
    size: str = typer.Option("t", "--size", help="Model size variant"),
    batch: int = typer.Option(16, "--batch", help="Micro-batch (-1 auto-fits ~70%% VRAM)"),
    imgsz: int = typer.Option(640, "--imgsz"),
    workers: int = typer.Option(8, "--workers"),
    amp: bool = typer.Option(True, "--amp/--no-amp", help="Use the family's AMP path"),
    steps: int = typer.Option(20, "--steps", help="Profiled (measured) steps"),
    warmup: int = typer.Option(5, "--warmup", help="Warmup steps before measuring"),
    repeat: int = typer.Option(1, "--repeat", help="Repeat N times for mean +/- stdev (a single run lies when launch-bound)"),
    device: str = typer.Option("0", "--device"),
    project: str = typer.Option("runs/profile", "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Launch a short profiled training and emit a self-contained profile.json."""
    import statistics as _st

    import torch
    from libreyolo import LibreYOLO

    # Enough total iterations to fill warmup+steps even on a tiny dataset;
    # profile_then_stop ends the run once the window is full, so extra epochs
    # never run.
    epochs = warmup + steps + 5
    trials, trial_profiles, trial_paths, last_dir = [], [], [], None
    for i in range(max(1, repeat)):
        name = f"prof_{i}" if repeat > 1 else "prof"
        model = LibreYOLO(model_path=weights, size=size, device=device)
        model.train(
            data=data, epochs=epochs, batch=batch, imgsz=imgsz, workers=workers,
            amp=amp, device=device, profile=True, profile_then_stop=True,
            profile_steps=steps,
            profile_warmup=warmup, profile_open=False, no_aug_epochs=0,
            project=project, name=name, exist_ok=True,
        )
        last_dir = Path(project) / name
        pj = last_dir / "profile.json"
        if not pj.exists():
            typer.echo(
                f"no profile produced for run {i}: the window (warmup {warmup} + "
                f"steps {steps}) never filled. Use a larger dataset, fewer --steps, "
                "or a smaller --batch.", err=True)
            raise typer.Exit(3)
        a = _load(str(pj))
        trial_profiles.append(a)
        trial_paths.append(str(pj))
        if a.get("img_per_s") is not None:
            trials.append(a["img_per_s"])
        if i < repeat - 1:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pj = last_dir / "profile.json"
    agg = _load(str(pj))
    if len(trials) > 1:
        agg = dict(trial_profiles[-1])
        mean, std = round(_st.fmean(trials), 2), round(_st.pstdev(trials), 2)
        agg["img_per_s"] = agg["metrics"]["img_per_s"] = mean
        agg["img_per_s_std"] = agg["metrics"]["img_per_s_std"] = std
        agg["img_per_s_n"] = agg["metrics"]["img_per_s_n"] = len(trials)
        agg["img_per_s_trials"] = [round(t, 2) for t in trials]
        agg["aggregation"] = {
            "n": len(trial_profiles),
            "profiles": trial_paths,
            "representative_profile": trial_paths[-1],
            "note": "scalar metrics are averaged; kernel lists and trace view use the final trial.",
        }
        scalar_keys = (
            "step_ms", "gpu_util", "gpu_util_raw", "gpu_busy_ms_per_step",
            "mean_kernel_us", "kernels_per_step", "memcpy_ms_per_step",
            "host_overhead_ms_per_step", "launches_per_step", "dataload_ms",
            "dataload_frac", "tensorcore_pct", "cuda_mallocs_per_step",
        )
        metric_aliases = {
            "gpu_busy_ms_per_step": "gpu_busy_ms",
            "memcpy_ms_per_step": "memcpy_ms",
            "host_overhead_ms_per_step": "host_overhead_ms",
        }
        metrics = dict(agg.get("metrics") or {})
        for key in scalar_keys:
            values = [
                p.get(key) for p in trial_profiles
                if isinstance(p.get(key), (int, float)) and p.get(key) is not None
            ]
            if not values:
                continue
            value = round(_st.fmean(values), 3)
            if all(isinstance(v, int) for v in values):
                value = int(round(value))
            agg[key] = value
            metrics[metric_aliases.get(key, key)] = value
        bounds = [p.get("bound") for p in trial_profiles if p.get("bound")]
        if bounds and len(set(bounds)) > 1:
            agg["bound"] = metrics["bound"] = "mixed"
            agg["bound_why"] = "repeat trials produced mixed bottleneck verdicts"
        agg["metrics"] = metrics
        pj = Path(project) / "profile_repeat.json"
        pj.write_text(_json.dumps(agg))

    if json_output:
        print(_json.dumps({
            "profile": str(pj), "trace": str(last_dir / "profile_trace.json"),
            "img_per_s": agg.get("img_per_s"), "img_per_s_std": agg.get("img_per_s_std"),
            "img_per_s_n": agg.get("img_per_s_n", 1), "bound": agg.get("bound"),
            "gpu_util": agg.get("gpu_util"),
        }, indent=2))
    else:
        v = str(agg.get("img_per_s"))
        if agg.get("img_per_s_n", 1) > 1:
            v += f" +/- {agg.get('img_per_s_std')} (n={agg['img_per_s_n']})"
        print(f"img/s: {v}  |  {agg.get('bound')}")
        print(f"profile: {pj}")
        print(f"next:    libreyolo profile summary {pj}")


@profile_app.command("infer")
def infer_cmd(
    source: Optional[str] = typer.Argument(None, help="Image or directory (default: bundled sample image)"),
    weights: str = typer.Option("LibreYOLO9t.pt", "--weights", help="Model weights file / name"),
    size: str = typer.Option("t", "--size", help="Model size variant"),
    batch: int = typer.Option(1, "--batch", help="Images per forward pass"),
    imgsz: int = typer.Option(640, "--imgsz"),
    half: bool = typer.Option(False, "--half/--no-half", help="Autocast forward (CUDA only)"),
    amp_dtype: str = typer.Option(
        "float16",
        "--amp-dtype",
        help="CUDA autocast dtype: float16 or bfloat16",
    ),
    warmup: int = typer.Option(20, "--warmup", help="Warmup iterations before measuring"),
    runs: int = typer.Option(100, "--runs", help="Measured iterations"),
    repeat: int = typer.Option(1, "--repeat", help="Repeat N times for mean +/- stdev throughput (needed for a significant compare)"),
    conf: float = typer.Option(0.25, "--conf", help="Confidence threshold (affects NMS work)"),
    iou: float = typer.Option(0.45, "--iou", help="NMS IoU threshold"),
    max_det: int = typer.Option(300, "--max-det", help="Max detections/image (affects NMS work)"),
    device: str = typer.Option("0", "--device"),
    trace: bool = typer.Option(True, "--trace/--no-trace", help="Emit a Chrome trace for kernel/op drill-down"),
    project: str = typer.Option("runs/profile", "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Profile the inference path: latency p50/p90/p99, stage split (preprocess/forward/NMS), GPU truth."""
    import statistics as _st

    import torch
    from libreyolo import SAMPLE_IMAGE, LibreYOLO
    from libreyolo.profiling import InferenceProfiler

    images = _resolve_infer_source(source or SAMPLE_IMAGE)
    if not images:
        typer.echo(f"no images found at: {source}", err=True)
        raise typer.Exit(2)

    trials: list[float] = []
    last = None
    for i in range(max(1, repeat)):
        name = f"infer_{i}" if repeat > 1 else "infer"
        model = LibreYOLO(model_path=weights, size=size, device=device)
        save_dir = Path(project) / name
        prof = InferenceProfiler(
            model, warmup=warmup, runs=runs, batch=batch, imgsz=imgsz,
            half=half, amp_dtype=amp_dtype, trace=trace, save_dir=save_dir,
            meta={"model": weights},
        )
        a = prof.run(images, conf=conf, iou=iou, max_det=max_det)
        last = (a, save_dir)
        if a.get("img_per_s") is not None:
            trials.append(a["img_per_s"])
        if i < repeat - 1:
            import gc

            del model, prof
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    a, save_dir = last
    pj = save_dir / "profile.json"
    if len(trials) > 1:
        mean, std = round(_st.fmean(trials), 2), round(_st.pstdev(trials), 2)
        a["img_per_s"] = mean
        a.setdefault("metrics", {})["throughput_img_s"] = mean
        a["img_per_s_std"] = std
        a["img_per_s_n"] = len(trials)
        a["img_per_s_trials"] = [round(t, 2) for t in trials]
        pj = Path(project) / "infer_repeat.json"
        pj.write_text(_json.dumps(a))

    lat = a.get("latency") or {}
    if json_output:
        print(_json.dumps({
            "profile": str(pj), "mode": "inference",
            "img_per_s": a.get("img_per_s"), "img_per_s_std": a.get("img_per_s_std"),
            "img_per_s_n": a.get("img_per_s_n", 1),
            "latency_p50_ms": lat.get("p50_ms"), "latency_p90_ms": lat.get("p90_ms"),
            "latency_p99_ms": lat.get("p99_ms"), "bound": a.get("bound"),
            "stages_ms": a.get("stages_ms"),
        }, indent=2))
    else:
        thr = str(a.get("img_per_s"))
        if a.get("img_per_s_n", 1) > 1:
            thr += f" +/- {a.get('img_per_s_std')} (n={a['img_per_s_n']})"
        print(f"latency  p50 {lat.get('p50_ms')} ms | p90 {lat.get('p90_ms')} ms | "
              f"p99 {lat.get('p99_ms')} ms  |  {thr} img/s")
        st = a.get("stages_ms") or {}
        print(f"stages   preprocess {st.get('preprocess')} ms | forward {st.get('forward')} ms | "
              f"postprocess/NMS {st.get('postprocess')} ms")
        print(f">> {str(a.get('bound')).upper()}: {a.get('bound_why')}")
        print(f"profile: {pj}")
        print(f"next:    libreyolo profile summary {pj}")


@profile_app.command("summary")
def summary_cmd(
    trace: str = typer.Argument(..., help="Path to profile_trace.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """High-level diagnosis: utilisation, verdict, kernel mix, top kernels."""
    a = _load(trace)
    if json_output:
        keep = ("trace", "mode", "model", "config", "bound", "bound_why", "step_ms",
                "img_per_s", "latency", "stages_ms", "gpu_util", "gpu_busy_ms_per_step",
                "mean_kernel_us", "kernels_per_step", "unique_kernels",
                "memcpy_ms_per_step", "tensorcore_pct", "peak_vram_mb",
                "dataload_ms", "dataload_frac", "host_overhead_ms_per_step",
                "launches_per_step", "memory_pressure", "cuda_mallocs_per_step",
                "gpu_util_raw", "window", "categories", "phases_gpu", "top_kernels")
        print(_json.dumps({k: a[k] for k in keep if k in a}, indent=2))
        return
    print(f"model {a.get('model') or '?'}  |  {a.get('img_per_s') or '?'} img/s  |  "
          f"step {a['step_ms']} ms  |  {a['kernels_per_step']} kernels/step @ ~{a['mean_kernel_us']:.0f}us")
    if a.get("mode") == "inference":
        lat, st = a.get("latency") or {}, a.get("stages_ms") or {}
        print(f"latency  p50 {lat.get('p50_ms')} ms | p90 {lat.get('p90_ms')} ms | "
              f"p99 {lat.get('p99_ms')} ms  ({lat.get('n')} samples)")
        print(f"stages   preprocess {st.get('preprocess')} ms | forward {st.get('forward')} ms | "
              f"postprocess/NMS {st.get('postprocess')} ms")
    print(f"GPU util {a['gpu_util'] * 100:.0f}%  ({a['gpu_busy_ms_per_step']} ms busy)  |  "
          f"Tensor Cores {a['tensorcore_pct']:.0f}%  |  peak VRAM {a.get('peak_vram_mb') or '?'} MB  |  "
          f"memcpy {a['memcpy_ms_per_step']} ms")
    print(f"host overhead {a['host_overhead_ms_per_step']} ms/step  |  "
          f"{a['launches_per_step']} kernel launches/step")
    if a.get("memory_pressure"):
        print("** MEMORY-PRESSURE: VRAM thrash, utilisation/throughput here are unreliable **")
    print(f">> {str(a['bound']).upper()}: {a['bound_why']}")
    print("kernel mix:")
    for c in a["categories"]:
        print(f"  {c['name']:<17} {c['pct']:5.1f}%   {c['ms_per_step']:.1f} ms/step")
    print("top kernels/step:")
    for k in a["top_kernels"]:
        tc = " [TC]" if k["tensorcore"] else ""
        print(f"  {k['pct_of_gpu']:5.1f}%  {k['ms_per_step']:6.3f} ms  x{k['count_per_step']:<4} {k['kernel']}{tc}")


@profile_app.command("get")
def get_cmd(
    trace: str = typer.Argument(..., help="Path to profile_trace.json"),
    field: Optional[str] = typer.Argument(None, help="Metric name (omit to list metrics)"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Print ONE metric, minimal output for tight loops (img_per_s, forward_ms, gpu_util...)."""
    a = _load(trace)
    m = a["metrics"]
    if not field:
        keys = sorted(m.keys())
        print(_json.dumps(keys) if json_output else "available metrics:\n  " + "\n  ".join(keys))
        return
    if field not in m:
        typer.echo(f"unknown metric '{field}'. run 'profile get {trace}' to list metrics.", err=True)
        raise typer.Exit(2)
    print(_json.dumps({field: m[field]}) if json_output else m[field])


@profile_app.command("phases")
def phases_cmd(
    trace: str = typer.Argument(..., help="Path to profile_trace.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Per-phase breakdown: GPU ms, wall ms, kernel & op counts (forward/backward/dataload/...)."""
    a = _load(trace)
    if json_output:
        print(_json.dumps({"trace": a["trace"], "phases": a["phases"]}, indent=2))
        return
    print(f"{'phase':<11} {'gpu_ms':>8} {'wall_ms':>8} {'kernels':>8} {'ops':>7}")
    for p in a["phases"]:
        wall = "-" if p["wall_ms_per_step"] is None else f"{p['wall_ms_per_step']:.1f}"
        print(f"{p['phase']:<11} {p['gpu_ms_per_step']:8.1f} {wall:>8} "
              f"{p['kernels_per_step']:8d} {p['ops_per_step']:7d}")


@profile_app.command("kernels")
def kernels_cmd(
    trace: str = typer.Argument(..., help="Path to profile_trace.json"),
    top: int = typer.Option(20, "--top", help="Show top N by GPU time"),
    category: Optional[str] = typer.Option(None, "--category", help="Filter by category substring (gemm, layout, norm, elementwise)"),
    grep: Optional[str] = typer.Option(None, "--grep", help="Filter by kernel-name regex"),
    tensorcore: bool = typer.Option(False, "--tensorcore", help="Only Tensor-Core kernels"),
    sort: str = typer.Option("time", "--sort", help="time | count | name"),
    phase: Optional[str] = typer.Option(None, "--phase", help="Restrict to a phase (forward/backward/dataload/to_device/optimizer)"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Drill to individual GPU kernels, the bottom of the analysis."""
    a = _load(trace)
    if phase:
        ks = a["kernels_by_phase"].get(phase)
        if ks is None:
            typer.echo(f"unknown phase '{phase}'. available: {', '.join(a['kernels_by_phase'])}", err=True)
            raise typer.Exit(2)
    else:
        ks = a["kernels"]
    if category:
        ks = [k for k in ks if category.lower() in k["category"].lower()]
    if grep:
        rx = _re.compile(grep, _re.I)
        ks = [k for k in ks if rx.search(k["raw_name"]) or rx.search(k["kernel"])]
    if tensorcore:
        ks = [k for k in ks if k["tensorcore"]]
    keyf = {"time": lambda k: -k["pct_of_gpu"], "count": lambda k: -k["count_per_step"],
            "name": lambda k: k["kernel"]}.get(sort, lambda k: -k["pct_of_gpu"])
    ks = sorted(ks, key=keyf)[:top]
    if json_output:
        print(_json.dumps({"trace": a["trace"], "matched": len(ks), "kernels": ks}, indent=2))
        return
    print(f"{'%GPU':>6} {'ms/step':>9} {'x/step':>7} TC  kernel  [category]")
    for k in ks:
        print(f"{k['pct_of_gpu']:6.2f} {k['ms_per_step']:9.3f} {k['count_per_step']:7d}  "
              f"{'Y' if k['tensorcore'] else ' '}  {k['kernel']}  [{k['category']}]")


@profile_app.command("ops")
def ops_cmd(
    trace: str = typer.Argument(..., help="Path to profile_trace.json"),
    top: int = typer.Option(20, "--top"),
    phase: Optional[str] = typer.Option(None, "--phase", help="Restrict to a phase (forward/backward/...)"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Framework view: aten/autograd ops by CPU time (host-launch culprits)."""
    a = _load(trace)
    if phase:
        ops_all = a["ops_by_phase"].get(phase)
        if ops_all is None:
            typer.echo(f"unknown phase '{phase}'. available: {', '.join(a['ops_by_phase'])}", err=True)
            raise typer.Exit(2)
    else:
        ops_all = a["ops"]
    ops = ops_all[:top]
    if json_output:
        print(_json.dumps({"trace": a["trace"], "ops": ops}, indent=2))
        return
    print(f"{'%cpu':>6} {'ms/step':>9} {'x/step':>7}  op")
    for o in ops:
        print(f"{o['pct_of_cpu_ops']:6.2f} {o['ms_per_step']:9.3f} {o['count_per_step']:7d}  {o['op']}")


@profile_app.command("compare")
def compare_cmd(
    before: str = typer.Argument(..., help="baseline profile_trace.json"),
    after: str = typer.Argument(..., help="new profile_trace.json"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Diff two profiles: did the change help? (the optimise-loop closer)."""
    a, b = _load(before), _load(after)

    def per_img(x):
        bs = (x.get("config") or {}).get("batch") or 0
        return round(x["step_ms"] / bs, 3) if bs else None

    def significance(x, y):
        import math
        ma, mb = x.get("img_per_s"), y.get("img_per_s")
        if ma is None or mb is None:
            return None, "img/s unavailable"
        sa, sb = x.get("img_per_s_std"), y.get("img_per_s_std")
        na, nb = x.get("img_per_s_n", 1), y.get("img_per_s_n", 1)
        if na < 2 or nb < 2 or sa is None or sb is None:
            return None, "single run, use 'run --repeat N' for a significance call"
        se = math.sqrt(sa ** 2 / na + sb ** 2 / nb)
        return (abs(mb - ma) > 2 * se), f"|d|={abs(mb - ma):.1f} vs 2*SE={2 * se:.1f}"

    sig, sig_why = significance(a, b)
    res = {
        "before": before, "after": after,
        "img_per_s": {"before": a.get("img_per_s"), "after": b.get("img_per_s"),
                      "std": [a.get("img_per_s_std"), b.get("img_per_s_std")],
                      "n": [a.get("img_per_s_n", 1), b.get("img_per_s_n", 1)],
                      "significant": sig, "significance": sig_why},
        "ms_per_image": {"before": per_img(a), "after": per_img(b)},
        "gpu_util": {"before": a["gpu_util"], "after": b["gpu_util"]},
        "host_overhead_ms_per_step": {"before": a.get("host_overhead_ms_per_step"),
                                      "after": b.get("host_overhead_ms_per_step")},
        "launches_per_step": {"before": a.get("launches_per_step"),
                              "after": b.get("launches_per_step")},
        "bound": {"before": a["bound"], "after": b["bound"]},
    }
    if json_output:
        print(_json.dumps(res, indent=2))
        return
    tag = {True: "  [significant]", False: "  [NOT significant]"}.get(sig, f"  [{sig_why}]")
    ia, ib = a.get("img_per_s") or 0, b.get("img_per_s") or 0
    print(f"img/s          {a.get('img_per_s')} -> {b.get('img_per_s')}{_pct(ia, ib)}{tag}")
    pa, pb = per_img(a), per_img(b)
    if pa and pb:
        print(f"ms/image       {pa} -> {pb}{_pct(pa, pb)}   (batch-independent speed)")
    print(f"GPU util       {a['gpu_util'] * 100:.0f}% -> {b['gpu_util'] * 100:.0f}%")
    print(f"host overhead  {a.get('host_overhead_ms_per_step')} -> "
          f"{b.get('host_overhead_ms_per_step')} ms/step")
    print(f"launches/step  {a.get('launches_per_step')} -> {b.get('launches_per_step')}")
    print(f"verdict        {a['bound']} -> {b['bound']}")


@profile_app.command("what-if")
def whatif_cmd(
    trace: str = typer.Argument(..., help="Path to profile.json / profile_trace.json"),
    remove_category: Optional[str] = typer.Option(None, "--remove-category", help="Project removing a kernel category (gemm, layout, norm, elementwise)"),
    remove_launches: Optional[int] = typer.Option(None, "--remove-launches", help="Project removing N kernel launches/step (e.g. an op-fusion win)"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Estimate a change's payoff BEFORE editing code (rough projection)."""
    a = _load(trace)
    if not remove_category and remove_launches is None:
        typer.echo("specify --remove-category or --remove-launches", err=True)
        raise typer.Exit(2)
    step = a["step_ms"]
    bs = (a.get("config") or {}).get("batch") or 0
    host = a.get("host_overhead_ms_per_step") or 0.0
    launches = a.get("launches_per_step") or 1
    gpu = a.get("gpu_busy_ms_per_step") or 0.0
    util = a.get("gpu_util") or 0.0
    per_launch_ms = host / launches if launches else 0.0

    removed_gpu = 0.0
    removed_launches = remove_launches or 0
    if remove_category:
        cat = next((c for c in a["categories"]
                    if remove_category.lower() in c["name"].lower()), None)
        if cat is None:
            have = ", ".join(c["name"] for c in a["categories"])
            typer.echo(f"no category matching '{remove_category}'. have: {have}", err=True)
            raise typer.Exit(2)
        removed_gpu = cat["ms_per_step"]
        removed_launches += int(launches * (cat["ms_per_step"] / (gpu or 1)))

    if util < 0.8:
        new_step = max(step - removed_launches * per_launch_ms,
                       gpu + (a.get("dataload_ms") or 0.0))
        mechanism = (f"host/launch-bound: ~{removed_launches} fewer launches x "
                     f"~{per_launch_ms * 1000:.0f}us host cost")
    else:
        new_step = max(step - removed_gpu, host)
        mechanism = f"compute-bound: ~{removed_gpu:.1f} ms less GPU work"
    cur_imgs = a.get("img_per_s")
    new_imgs = round(bs / (new_step / 1000.0), 1) if (bs and new_step) else None
    res = {
        "trace": a["trace"], "bound": a["bound"], "mechanism": mechanism,
        "removed": {"gpu_ms": round(removed_gpu, 2), "launches": removed_launches},
        "step_ms": {"now": step, "projected": round(new_step, 1)},
        "img_per_s": {"now": cur_imgs, "projected": new_imgs},
        "caveat": "rough estimate (per-launch host cost is approximate); verify by profiling.",
    }
    if json_output:
        print(_json.dumps(res, indent=2))
        return
    print(f"what-if [{a['bound']}]: {mechanism}")
    print(f"  step   {step} -> ~{new_step:.0f} ms")
    if new_imgs and cur_imgs:
        print(f"  img/s  {cur_imgs} -> ~{new_imgs:.0f}{_pct(cur_imgs, new_imgs)}")
    print("  (rough, verify by actually profiling the change)")

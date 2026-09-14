"""Static HTML for the ``libreyolo monitor`` web UI.

A single self-contained page (inline CSS + vanilla JS, no external assets, no
chart library) so it works offline and adds zero dependencies. One app serves
two views, chosen by the ``?run=`` query parameter:

- no ``run`` -> an **index** of every discovered run with its live state, and
- ``?run=<id>`` -> the per-run **dashboard** (cards, metric charts, log,
  images), deep-linkable so each run can live in its own browser tab.

Everything is fetched from the read-only JSON endpoints served by
:mod:`libreyolo.ui.train_monitor`.
"""

from __future__ import annotations

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LibreYOLO Training Monitor</title>
<style>
  :root {
    --bg: #0f1216; --panel: #171b21; --panel2: #1e242c; --border: #2a323c;
    --text: #e6edf3; --muted: #8b98a5; --accent: #4cc38a; --accent2: #58a6ff;
    --warn: #e3b341; --err: #f85149;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f6f8fa; --panel: #ffffff; --panel2: #f0f3f6; --border: #d0d7de;
      --text: #1f2328; --muted: #656d76; --accent: #1a7f4b; --accent2: #0969da;
      --warn: #9a6700; --err: #cf222e;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  header {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    padding: 14px 20px; background: var(--panel); border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 5;
  }
  .brand { font-weight: 700; font-size: 16px; letter-spacing: .2px; }
  .brand span { color: var(--accent); }
  a { color: var(--accent2); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .back { font-size: 13px; }
  .badge {
    padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .5px; border: 1px solid var(--border);
  }
  .badge.running { color: var(--accent2); border-color: var(--accent2); }
  .badge.completed { color: var(--accent); border-color: var(--accent); }
  .badge.failed { color: var(--err); border-color: var(--err); }
  .badge.missing, .badge.unknown, .badge.unreadable { color: var(--muted); }
  .spacer { flex: 1; }
  .run-path { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
  main { padding: 20px; max-width: 1200px; margin: 0 auto; display: grid; gap: 16px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px;
  }
  .card .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .4px; }
  .card .value { font-size: 22px; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .card .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .progress { height: 8px; background: var(--panel2); border-radius: 999px; overflow: hidden; margin-top: 10px; }
  .progress > div { height: 100%; background: var(--accent2); width: 0; transition: width .4s ease; }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
  }
  .panel h2 {
    margin: 0; padding: 12px 16px; font-size: 13px; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: .5px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px;
  }
  .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
  .chart-box { padding: 12px 16px 16px; }
  canvas { width: 100%; height: 200px; display: block; }
  .log {
    margin: 0; padding: 14px 16px; max-height: 320px; overflow: auto; background: #0b0e12;
    color: #d1d9e0; font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace;
    white-space: pre-wrap; word-break: break-word;
  }
  .images { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; padding: 16px; }
  .images figure { margin: 0; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: var(--panel2); }
  .images img { width: 100%; display: block; cursor: zoom-in; }
  .images figcaption { padding: 6px 8px; font-size: 11px; color: var(--muted); font-family: ui-monospace, monospace; word-break: break-all; }
  .empty { padding: 16px; color: var(--muted); font-size: 13px; }
  .err-box { padding: 12px 16px; color: var(--err); font-family: ui-monospace, monospace; font-size: 12px; white-space: pre-wrap; }
  /* run index */
  .runlist { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }
  .runcard {
    display: block; background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px; color: inherit;
  }
  .runcard:hover { border-color: var(--accent2); text-decoration: none; }
  .runcard .top { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .runcard .name { font-weight: 600; font-family: ui-monospace, monospace; font-size: 13px; word-break: break-all; }
  .runcard .meta { color: var(--muted); font-size: 12px; display: flex; justify-content: space-between; gap: 8px; }
  #lightbox {
    position: fixed; inset: 0; background: rgba(0,0,0,.85); display: none;
    align-items: center; justify-content: center; z-index: 20; cursor: zoom-out;
  }
  #lightbox img { max-width: 94vw; max-height: 94vh; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); display: inline-block; }
  .dot.live { background: var(--accent); animation: pulse 1.4s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }
</style>
</head>
<body>
<header>
  <div class="brand">Libre<span>YOLO</span> Training Monitor</div>
  <a class="back" id="back" href="/" style="display:none">&larr; all runs</a>
  <span id="state" class="badge missing" style="display:none">-</span>
  <div class="spacer"></div>
  <span class="dot" id="livedot"></span>
  <span class="run-path" id="runpath"></span>
</header>

<!-- INDEX VIEW -->
<main id="index-view" style="display:none">
  <div class="panel">
    <h2>Runs <span class="run-path" id="root-path" style="text-transform:none"></span></h2>
    <div style="padding:16px"><div class="runlist" id="runlist"></div>
    <div class="empty" id="runs-empty" style="display:none">No training runs found.</div></div>
  </div>
</main>

<!-- RUN VIEW -->
<main id="run-view" style="display:none">
  <section class="cards" id="cards"></section>
  <div class="panel">
    <h2>Metrics</h2>
    <div class="charts" id="charts">
      <div class="chart-box"><div class="run-path" id="chart-loss-title">train/loss</div><canvas id="chart-loss"></canvas></div>
      <div class="chart-box"><div class="run-path" id="chart-metric-title">metric</div><canvas id="chart-metric"></canvas></div>
    </div>
    <div class="empty" id="metrics-empty" style="display:none">No metrics logged yet.</div>
  </div>
  <div class="panel">
    <h2>Log <span class="run-path" style="text-transform:none">train.log</span></h2>
    <pre class="log" id="log">No log output yet.</pre>
  </div>
  <div class="panel">
    <h2>Images</h2>
    <div class="images" id="images"></div>
    <div class="empty" id="images-empty">No plot or sample images yet.</div>
  </div>
</main>
<div id="lightbox"><img alt=""></div>

<script>
const $ = (id) => document.getElementById(id);
const RUN = new URLSearchParams(location.search).get("run");
let logStuck = true;

function fmtSecs(s) {
  if (s == null || !isFinite(s)) return "-";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${sec}s`;
  return `${sec}s`;
}
function fmtNum(v, d = 4) {
  if (v == null || v === "" || !isFinite(v)) return "-";
  return Number(v).toFixed(d).replace(/\.?0+$/, "");
}
function card(label, value, sub) {
  return `<div class="card"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}
async function j(url) { const r = await fetch(url); return r.json(); }
const withRun = (url) => url + (url.includes("?") ? "&" : "?") + "run=" + encodeURIComponent(RUN);

/* ============================ INDEX VIEW ============================ */
async function renderIndex() {
  $("index-view").style.display = "grid";
  $("livedot").style.display = "none";
  let data;
  try { data = await j("/api/runs"); } catch { return; }
  $("root-path").textContent = data.root || "";
  const runs = data.runs || [];
  // Convenience: a lone run skips the index entirely.
  if (runs.length === 1 && !sessionStorage.getItem("noskip")) {
    location.replace("/?run=" + encodeURIComponent(runs[0].id));
    return;
  }
  $("runs-empty").style.display = runs.length ? "none" : "block";
  $("runlist").innerHTML = runs.map(r => {
    const st = (r.state || "unknown").toLowerCase();
    const ep = r.total_epochs != null
      ? `${r.current_epoch != null ? r.current_epoch : (r.state === 'completed' ? r.total_epochs : 0)}/${r.total_epochs}` : "-";
    const best = r.best_metric != null ? `${r.best_metric_name || "best"} ${fmtNum(r.best_metric)}` : "";
    return `<a class="runcard" href="/?run=${encodeURIComponent(r.id)}">
      <div class="top"><span class="badge ${st}">${st}</span>
        ${st === "running" ? '<span class="dot live"></span>' : ""}</div>
      <div class="name">${r.name}</div>
      <div class="meta"><span>${r.model || ""} ${r.task || ""}</span><span>epoch ${ep}</span></div>
      <div class="meta"><span>${best}</span><span>${((r.progress || 0) * 100).toFixed(0)}%</span></div>
    </a>`;
  }).join("");
  setTimeout(renderIndex, 3000);
}

/* ============================ RUN VIEW ============================ */
async function refreshStatus() {
  let s;
  try { s = await j(withRun("/api/status")); } catch { return; }
  const state = (s.state || "unknown").toLowerCase();
  const badge = $("state");
  badge.style.display = "inline-block";
  badge.className = "badge " + state;
  badge.textContent = state;
  $("livedot").className = "dot" + (state === "running" ? " live" : "");
  $("runpath").textContent = s.save_dir || RUN;

  const cards = [];
  const cur = s.current_epoch != null ? s.current_epoch : (s.completed_epochs || 0);
  cards.push(card("Epoch", `${cur} / ${s.total_epochs ?? "-"}`,
    s.mean_epoch_seconds ? `${fmtSecs(s.mean_epoch_seconds)}/epoch` : ""));
  cards.push(card("ETA", state === "running" ? fmtSecs(s.eta_seconds) : "-",
    `elapsed ${fmtSecs(s.elapsed_seconds)}`));
  const metricName = s.best_metric_name || s.current_metric_name || "metric";
  cards.push(card(`Best ${metricName}`, fmtNum(s.best_metric),
    s.best_epoch != null ? `epoch ${s.best_epoch}` : ""));
  cards.push(card("Train loss", fmtNum(s.train_loss, 4),
    s.model_family ? `${s.model_family}${s.model_size || ""} ${s.task || ""}` : ""));
  if (s.metrics && s.metrics.loss != null) {
    cards.push(card("Val loss", fmtNum(s.metrics.loss, 4), "latest validation"));
  }

  let cardsHtml = cards.join("");
  cardsHtml += `<div class="card" style="grid-column:1/-1">
    <div class="label">Progress</div>
    <div class="progress"><div style="width:${((s.progress || 0) * 100).toFixed(1)}%"></div></div>
    <div class="sub">${((s.progress || 0) * 100).toFixed(1)}%</div></div>`;
  $("cards").innerHTML = cardsHtml;

  if (state === "failed" && s.error) {
    let box = $("errbox");
    if (!box) { box = document.createElement("div"); box.id = "errbox"; box.className = "err-box"; $("cards").after(box); }
    box.textContent = `${s.error.type || "Error"}: ${s.error.message || ""}`;
  }
  return state;
}

function drawChart(canvas, series, xs) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const border = css.getPropertyValue("--border").trim();
  const muted = css.getPropertyValue("--muted").trim();
  const colors = [css.getPropertyValue("--accent2").trim(), css.getPropertyValue("--accent").trim(), css.getPropertyValue("--warn").trim()];
  const pad = { l: 46, r: 10, t: 10, b: 22 };
  const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
  const all = series.flatMap(s => s.data.filter(v => v != null && isFinite(v)));
  if (!all.length || xs.length < 2) {
    ctx.fillStyle = muted; ctx.font = "12px sans-serif";
    ctx.fillText("no data yet", pad.l, pad.t + plotH / 2); return;
  }
  let min = Math.min(...all), max = Math.max(...all);
  if (min === max) { min -= 1; max += 1; }
  const xmin = xs[0], xmax = xs[xs.length - 1];
  const X = (x) => pad.l + (xmax === xmin ? 0 : (x - xmin) / (xmax - xmin)) * plotW;
  const Y = (y) => pad.t + plotH - ((y - min) / (max - min)) * plotH;
  ctx.strokeStyle = border; ctx.fillStyle = muted;
  ctx.font = "11px ui-monospace, monospace"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const gy = pad.t + (plotH / 4) * i;
    ctx.globalAlpha = .5; ctx.beginPath(); ctx.moveTo(pad.l, gy); ctx.lineTo(w - pad.r, gy); ctx.stroke(); ctx.globalAlpha = 1;
    ctx.fillText((max - ((max - min) / 4) * i).toPrecision(3), 4, gy + 3);
  }
  ctx.fillText("epoch " + xmin, pad.l, h - 6);
  ctx.fillText("" + xmax, w - pad.r - 20, h - 6);
  series.forEach((s, si) => {
    ctx.strokeStyle = colors[si % colors.length]; ctx.lineWidth = 2;
    ctx.beginPath(); let started = false;
    s.data.forEach((v, i) => {
      if (v == null || !isFinite(v)) return;
      const px = X(xs[i]), py = Y(v);
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    });
    ctx.stroke();
  });
  ctx.font = "11px sans-serif";
  series.forEach((s, si) => { ctx.fillStyle = colors[si % colors.length]; ctx.fillText("● " + s.name, pad.l + si * 120, pad.t + 4); });
}

const PREFERRED_METRICS = ["mAP50-95", "mAP50", "mAP", "top1", "accuracy", "top5", "PR"];

async function refreshMetrics() {
  let m;
  try { m = await j(withRun("/api/metrics")); } catch { return; }
  const rows = m.rows || [];
  if (!rows.length) { $("metrics-empty").style.display = "block"; return; }
  $("metrics-empty").style.display = "none";
  const xs = rows.map(r => r.epoch != null ? r.epoch : 0);
  const lossCols = (m.columns || []).filter(c => c.startsWith("train/") && c.includes("loss"));
  const primaryLoss = lossCols.includes("train/loss") ? "train/loss" : lossCols[0];
  const valLoss = (m.columns || []).includes("metrics/loss") ? "metrics/loss" : null;
  const lossSeries = [];
  if (primaryLoss) lossSeries.push({ name: primaryLoss, data: rows.map(r => r[primaryLoss]) });
  if (valLoss) lossSeries.push({ name: "val/loss", data: rows.map(r => r[valLoss]) });
  drawChart($("chart-loss"), lossSeries, xs);
  $("chart-loss-title").textContent = valLoss ? "train/loss · val/loss" : (primaryLoss || "train/loss");
  const metricCols = (m.columns || []).filter(c => c.startsWith("metrics/"));
  let chosen = null;
  for (const p of PREFERRED_METRICS) { const hit = metricCols.find(c => c.replace("metrics/", "") === p); if (hit) { chosen = hit; break; } }
  if (!chosen && metricCols.length) chosen = metricCols[0];
  const metricSeries = chosen ? [{ name: chosen.replace("metrics/", ""), data: rows.map(r => r[chosen]) }] : [];
  drawChart($("chart-metric"), metricSeries, xs);
  $("chart-metric-title").textContent = chosen ? chosen.replace("metrics/", "") : "metric";
}

async function refreshLog() {
  let d;
  try { d = await j(withRun("/api/log")); } catch { return; }
  const el = $("log");
  const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 40;
  el.textContent = d.log && d.log.length ? d.log : "No log output yet.";
  if (atBottom || logStuck) { el.scrollTop = el.scrollHeight; }
  logStuck = false;
}

async function refreshImages() {
  let d;
  try { d = await j(withRun("/api/images")); } catch { return; }
  const imgs = d.images || [];
  const wrap = $("images");
  $("images-empty").style.display = imgs.length ? "none" : "block";
  wrap.innerHTML = imgs.map(im =>
    `<figure><img loading="lazy" src="${im.url}&t=${Math.floor(im.mtime)}" alt="${im.name}">
     <figcaption>${im.path}</figcaption></figure>`).join("");
  wrap.querySelectorAll("img").forEach(img => img.addEventListener("click", () => {
    const lb = $("lightbox"); lb.querySelector("img").src = img.src; lb.style.display = "flex";
  }));
}
$("lightbox").addEventListener("click", () => { $("lightbox").style.display = "none"; });

async function runTick() {
  const state = await refreshStatus();
  await Promise.all([refreshMetrics(), refreshLog(), refreshImages()]);
  const done = state === "completed" || state === "failed";
  setTimeout(runTick, done ? 8000 : 2000);
}

/* ============================ BOOT ============================ */
if (RUN) {
  $("run-view").style.display = "grid";
  $("back").style.display = "inline-block";
  sessionStorage.setItem("noskip", "1"); // once inside a run, index won't auto-skip
  runTick();
  window.addEventListener("resize", refreshMetrics);
} else {
  renderIndex();
}
</script>
</body>
</html>
"""

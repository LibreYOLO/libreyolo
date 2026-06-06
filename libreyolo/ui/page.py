"""Static HTML page for the LibreYOLO UI.

Embedded as a string so it ships in the wheel with no package-data wiring and
no runtime file lookups. Styling mirrors the LibreYOLO website palette
(libre cyan + slate surfaces, Outfit / JetBrains Mono).
"""

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LibreYOLO</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --libre-300: #67e8f9; --libre-400: #06b6d4; --libre-500: #0891b2;
    --libre-600: #0e7490; --emerald: #10b981; --emerald-d: #059669;
    --bg: #fafbfd; --panel: #ffffff; --panel-2: #f1f5f9;
    --border: #e2e8f0; --border-2: #cbd5e1; --text: #1e293b; --muted: #64748b;
    --red: #dc2626;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body {
    background-color: var(--bg); color: var(--text);
    font-family: 'Outfit', system-ui, sans-serif; -webkit-font-smoothing: antialiased;
    display: flex; flex-direction: column;
    background-image:
      radial-gradient(at 20% 20%, rgba(8, 145, 178, 0.08) 0%, transparent 50%),
      radial-gradient(at 80% 80%, rgba(16, 185, 129, 0.06) 0%, transparent 50%),
      radial-gradient(at 50% 50%, rgba(8, 145, 178, 0.03) 0%, transparent 70%);
    background-attachment: fixed;
  }
  .mono { font-family: 'JetBrains Mono', monospace; }
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: #f1f5f9; }
  ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
  ::selection { background: rgba(8, 145, 178, 0.2); color: #0f172a; }

  header {
    display: flex; align-items: center; gap: 12px; padding: 14px 24px;
    border-bottom: 1px solid var(--border); background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(10px);
  }
  .logo { width: 30px; height: 30px; border-radius: 9px; box-shadow: 0 0 20px rgba(8, 145, 178, 0.25); display: block; }
  header h1 { font-size: 17px; margin: 0; font-weight: 700; letter-spacing: .2px; }
  header h1 .lo { color: var(--libre-500); }
  header .tag { color: var(--muted); font-size: 12px; padding: 3px 9px; border: 1px solid var(--border-2); border-radius: 999px; }
  .spacer { flex: 1; }
  .controls { display: flex; align-items: center; gap: 10px; }
  label.ctl { color: var(--muted); font-size: 12px; }
  select {
    background: var(--panel); color: var(--text); border: 1px solid var(--border-2);
    border-radius: 9px; padding: 8px 12px; font-size: 13px; cursor: pointer; font-family: inherit;
  }
  select:hover { border-color: var(--libre-500); }

  .stage { flex: 1; padding: 24px; overflow: auto; display: flex; flex-direction: column; gap: 16px; }

  .toolbar {
    border: 2px dashed var(--border-2); border-radius: 16px; padding: 14px 16px;
    background: rgba(255, 255, 255, 0.6); display: flex; align-items: center; gap: 12px;
    color: var(--muted); font-size: 13px; transition: all .2s ease;
  }
  .toolbar.drag { border-color: var(--libre-500); background: rgba(8,145,178,.06); box-shadow: 0 10px 40px rgba(8,145,178,.15); }
  .toolbar b { color: var(--text); }
  .toolbar kbd {
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--muted);
    border: 1px solid var(--border-2); border-bottom-width: 2px; border-radius: 5px; padding: 1px 6px; background: var(--panel);
  }
  .btn {
    font-family: inherit; font-size: 13px; font-weight: 600; cursor: pointer;
    padding: 9px 16px; border-radius: 10px; color: #fff; border: none;
    background: linear-gradient(135deg, var(--libre-400), var(--libre-500)); transition: all .25s ease;
    display: inline-flex; align-items: center; gap: 8px;
  }
  .btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 10px 40px rgba(8,145,178,.25); }
  .btn:disabled { opacity: .45; cursor: not-allowed; }
  .btn.ghost { background: var(--panel); color: var(--text); border: 1px solid var(--border-2); }
  .btn.ghost:hover:not(:disabled) { border-color: var(--libre-500); box-shadow: none; }
  .btn svg { width: 15px; height: 15px; }
  .spin { width: 14px; height: 14px; border: 2px solid rgba(255,255,255,.4); border-top-color: #fff; border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .resbar {
    display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-radius: 12px;
    background: linear-gradient(135deg, rgba(16,185,129,.08), rgba(8,145,178,.08));
    border: 1px solid rgba(16,185,129,.3); font-size: 13px;
  }
  .resbar .check { width: 22px; height: 22px; border-radius: 50%; background: var(--emerald-d); color: #fff; display: grid; place-items: center; flex: none; }
  .resbar .path { font-family: 'JetBrains Mono', monospace; color: var(--libre-600); font-size: 12.5px; word-break: break-all; }
  .resbar b { color: var(--text); }

  .empty {
    flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 14px; color: var(--muted); text-align: center; border-radius: 16px;
    border: 2px dashed var(--border); min-height: 340px;
  }
  .empty .big { font-size: 17px; color: var(--text); font-weight: 600; }
  .empty svg { opacity: .35; }

  .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 16px; }
  .card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
    box-shadow: 0 10px 30px rgba(0,0,0,.04); transition: all .3s cubic-bezier(.4,0,.2,1); display: flex; flex-direction: column;
  }
  .card:hover { transform: translateY(-4px); box-shadow: 0 20px 40px rgba(0,0,0,.08), 0 0 30px rgba(8,145,178,.06); }
  .card .imgbox { position: relative; background: #eef2f6; line-height: 0; aspect-ratio: 4 / 3; }
  .card .imgbox img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .card .imgbox.queued img { filter: grayscale(.4) brightness(.97); opacity: .8; }
  .card .state {
    position: absolute; top: 8px; left: 8px; font-size: 10px; font-weight: 700; letter-spacing: .4px; text-transform: uppercase;
    padding: 3px 8px; border-radius: 999px; backdrop-filter: blur(4px);
  }
  .state.queued { background: rgba(100,116,139,.15); color: #475569; border: 1px solid rgba(100,116,139,.3); }
  .state.done { background: rgba(8,145,178,.12); color: var(--libre-600); border: 1px solid rgba(8,145,178,.35); }
  .state.busy { background: rgba(8,145,178,.12); color: var(--libre-600); border: 1px solid rgba(8,145,178,.35); display: inline-flex; align-items: center; gap: 5px; }
  .state.err { background: rgba(220,38,38,.1); color: var(--red); border: 1px solid rgba(220,38,38,.35); }
  .state .minispin { width: 9px; height: 9px; border: 2px solid rgba(8,145,178,.3); border-top-color: var(--libre-600); border-radius: 50%; animation: spin .7s linear infinite; }
  .card .cap { padding: 10px 12px; display: flex; align-items: center; gap: 8px; }
  .card .cap .nm { font-size: 13px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .card .cap .ct { margin-left: auto; flex: none; font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--muted); }
  .card .cap .ct b { color: var(--libre-600); }

  .toast {
    position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%) translateY(20px);
    background: #0f172a; color: #fff; padding: 11px 18px; border-radius: 11px; font-size: 13px;
    box-shadow: 0 10px 40px rgba(0,0,0,.25); opacity: 0; pointer-events: none; transition: all .3s ease; z-index: 50;
    display: flex; align-items: center; gap: 9px; max-width: 80vw;
  }
  .toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
  .toast .pp { font-family: 'JetBrains Mono', monospace; color: var(--libre-300); word-break: break-all; }

  /* read-only terminal that streams the real CLI run */
  .terminal {
    border-radius: 12px; overflow: hidden; border: 1px solid #1e293b;
    box-shadow: 0 20px 40px rgba(0,0,0,.12), 0 0 30px rgba(8,145,178,.05);
    background: #0b1220;
  }
  .term-head {
    display: flex; align-items: center; gap: 7px; padding: 9px 13px;
    background: #0f172a; border-bottom: 1px solid #1e293b;
  }
  .term-head .dot { width: 11px; height: 11px; border-radius: 50%; }
  .term-head .r { background: #ff5f56; } .term-head .y { background: #ffbd2e; } .term-head .g { background: #27c93f; }
  .term-head .ttl { margin-left: 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #64748b; }
  .term-head .tbtn {
    margin-left: auto; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #64748b;
    background: transparent; border: 1px solid #1e293b; border-radius: 6px; padding: 3px 9px; cursor: pointer;
  }
  .term-head .tbtn:hover { color: #cbd5e1; border-color: #334155; }
  .term-body {
    padding: 12px 15px; height: 280px; min-height: 120px; max-height: 70vh; overflow: auto; resize: vertical;
    font-family: 'JetBrains Mono', monospace; font-size: 12.5px; line-height: 1.65;
    color: #cbd5e1; white-space: pre-wrap; word-break: break-word;
  }
  .term-body::-webkit-scrollbar-track { background: #0b1220; }
  .term-body::-webkit-scrollbar-thumb { background: #1e293b; }
  .term-body::-webkit-scrollbar-thumb:hover { background: #334155; }
  .term-body .tline { display: block; }
  .term-body .cmd { color: #e2e8f0; }
  .term-body .cmd .prompt, .term-body .live .prompt { color: #34d399; font-weight: 600; }
  .term-body .log { color: #93a4b8; }
  .term-body .ok { color: #34d399; }
  .term-body .err { color: #f87171; }
  .term-body .live { color: #e2e8f0; }
  .cursor {
    display: inline-block; width: 8px; height: 14px; background: #34d399;
    margin-left: 1px; vertical-align: text-bottom; animation: blink 1s steps(1) infinite;
  }
  @keyframes blink { 50% { opacity: 0; } }
</style>
</head>
<body>
  <header>
    <svg class="logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" aria-label="LibreYOLO">
      <rect width="32" height="32" rx="6" fill="#0891b2"/>
      <g transform="translate(4,4) scale(0.75)">
        <path fill="white" d="M16 0C16 0 10.3 0 8 2.3C5.7 4.6 5.7 6 5.7 8C5.7 8 0 8 0 16C0 24 5.7 24 5.7 24C5.7 26 5.7 27.4 8 29.7C10.3 32 16 32 16 32C16 32 21.7 32 24 29.7C26.3 27.4 26.3 26 26.3 24C26.3 24 32 24 32 16C32 8 26.3 8 26.3 8C26.3 6 26.3 4.6 24 2.3C21.7 0 16 0 16 0Z"/>
        <path fill="#0891b2" d="M16 5C16 5 12.2 5 10.6 6.6C9 8.2 9 9.2 9 10.5C9 10.5 5 10.5 5 16C5 21.5 9 21.5 9 21.5C9 22.8 9 23.8 10.6 25.4C12.2 27 16 27 16 27C16 27 19.8 27 21.4 25.4C23 23.8 23 22.8 23 21.5C23 21.5 27 21.5 27 16C27 10.5 23 10.5 23 10.5C23 9.2 23 8.2 21.4 6.6C19.8 5 16 5 16 5Z"/>
      </g>
    </svg>
    <h1>Libre<span class="lo">YOLO</span></h1>
    <span class="tag mono" id="addr">localhost</span>
    <div class="spacer"></div>
    <div class="controls">
      <label class="ctl">Model</label>
      <select id="model"><option>loading...</option></select>
      <label class="ctl">Conf</label>
      <select id="conf"><option>0.25</option><option>0.40</option><option>0.50</option></select>
    </div>
  </header>

  <section class="stage">
    <div class="toolbar" id="toolbar">
      <span style="font-size:18px">&#128193;</span>
      <span><b>Drop images or a folder</b>, paste with <kbd>Ctrl</kbd>+<kbd>V</kbd>, or browse</span>
      <span class="spacer"></span>
      <button class="btn ghost" id="pickFiles">Choose files</button>
      <button class="btn ghost" id="pickFolder">Add folder</button>
      <button class="btn" id="runBtn" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M5 3l14 9-14 9V3z"/></svg>
        Run inference
      </button>
    </div>

    <input type="file" id="fileInput" accept="image/*" multiple hidden>
    <input type="file" id="folderInput" accept="image/*" webkitdirectory directory multiple hidden>

    <div class="terminal" id="terminal" style="display:none">
      <div class="term-head">
        <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
        <span class="ttl">libreyolo &#8212; inference</span>
        <button class="tbtn" id="termClearBtn" title="Clear scrollback">clear</button>
      </div>
      <div class="term-body" id="termBody">
        <div class="live" id="termLive"><span class="prompt">$</span> <span class="typed"></span><span class="cursor"></span></div>
      </div>
    </div>

    <div class="resbar" id="resbar" style="display:none">
      <span class="check">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M20 6L9 17l-5-5"/></svg>
      </span>
      <span><b id="resCount">0</b> rendered images saved to <span class="path" id="resPath">-</span></span>
      <span class="spacer"></span>
      <button class="btn ghost" id="openFolder">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7a2 2 0 012-2h4l2 2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V7z"/></svg>
        Open results folder
      </button>
    </div>

    <div class="empty" id="empty">
      <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.8"/><path d="M21 15l-5-5L5 21"/>
      </svg>
      <div class="big">No images yet</div>
      <div>Drop images or a folder, then hit Run inference</div>
    </div>

    <div class="gallery" id="gallery"></div>
  </section>

  <div class="toast" id="toast"></div>

<script>
(function () {
  "use strict";
  var images = []; // {name, file, srcUrl, w, h, status, renderedUrl, n, error}
  var $ = function (id) { return document.getElementById(id); };

  $("addr").textContent = location.host;

  // ---- populate model dropdown from the real registry ----
  fetch("/api/models").then(function (r) { return r.json(); }).then(function (j) {
    var sel = $("model");
    sel.innerHTML = "";
    (j.models || []).forEach(function (m) {
      var o = document.createElement("option");
      o.value = m; o.textContent = m;
      if (m === j.default) o.selected = true;
      sel.appendChild(o);
    });
  }).catch(function () { $("model").innerHTML = '<option>yolo9-t</option>'; });

  function loadImage(url) {
    return new Promise(function (res) { var im = new Image(); im.onload = function () { res(im); }; im.onerror = function () { res(null); }; im.src = url; });
  }

  function addFiles(fileList) {
    var files = Array.prototype.slice.call(fileList).filter(function (f) { return f.type && f.type.indexOf("image/") === 0; });
    if (!files.length) return;
    files.forEach(function (f) {
      var url = URL.createObjectURL(f);
      var entry = { name: f.name || "pasted.png", file: f, srcUrl: url, w: 0, h: 0, status: "queued", renderedUrl: url, n: 0, error: null };
      images.push(entry);
      loadImage(url).then(function (im) { if (im) { entry.w = im.naturalWidth; entry.h = im.naturalHeight; } renderGallery(); });
    });
    $("empty").style.display = "none";
    $("runBtn").disabled = false;
    renderGallery();
  }

  // ---- terminal helpers ----
  var TERM_MAX_LINES = 800;  // scrollback cap
  function termAtBottom() {
    var b = $("termBody");
    return (b.scrollHeight - b.scrollTop - b.clientHeight) < 28;
  }
  function termStick(wasBottom) {
    if (wasBottom) { var b = $("termBody"); b.scrollTop = b.scrollHeight; }
  }
  function termClear() {
    var body = $("termBody");
    Array.prototype.slice.call(body.querySelectorAll(".tline")).forEach(function (n) { n.remove(); });
  }
  function termLine(text, cls) {
    var body = $("termBody"), live = $("termLive");
    var stick = termAtBottom();
    var d = document.createElement("div");
    d.className = "tline " + (cls || "log");
    if (cls === "cmd") { d.innerHTML = '<span class="prompt">$</span> '; d.appendChild(document.createTextNode(text)); }
    else { d.textContent = text; }
    body.insertBefore(d, live);
    // trim scrollback so very long sessions stay light
    var lines = body.querySelectorAll(".tline");
    if (lines.length > TERM_MAX_LINES) {
      for (var k = 0; k < lines.length - TERM_MAX_LINES; k++) lines[k].remove();
    }
    termStick(stick);
  }
  function typeCommand(cmd) {
    return new Promise(function (resolve) {
      var typed = $("termLive").querySelector(".typed");
      typed.textContent = "";
      var i = 0;
      (function step() {
        if (i < cmd.length) {
          var stick = termAtBottom();
          typed.textContent += cmd.charAt(i++);
          termStick(stick);
          setTimeout(step, 13);
        } else {
          setTimeout(function () {           // brief pause, like hitting Enter
            termLine(cmd, "cmd");
            typed.textContent = "";
            resolve();
          }, 180);
        }
      })();
    });
  }

  async function runInference() {
    var pending = images.filter(function (e) { return e.status !== "done"; });
    if (!pending.length) return;
    var btn = $("runBtn");
    btn.disabled = true;
    btn.innerHTML = '<span class="spin"></span> Running...';

    var model = $("model").value || "yolo9-t";
    var conf = $("conf").value || "0.25";
    var outdir = "-";

    $("terminal").style.display = "block";
    // keep scrollback across runs; just separate them
    if ($("termBody").querySelectorAll(".tline").length) termLine("", "log");

    for (var i = 0; i < pending.length; i++) {
      var e = pending[i];
      e.status = "busy"; renderGallery();
      await typeCommand("libreyolo predict --model " + model + " --source " + e.name + " --conf " + conf + " --save");

      try {
        var resp = await fetch("/api/infer?model=" + encodeURIComponent(model) + "&conf=" + encodeURIComponent(conf),
          { method: "POST", headers: { "X-Filename": e.name }, body: e.file });
        if (!resp.body) throw new Error("streaming not supported");
        var reader = resp.body.getReader(), dec = new TextDecoder(), buf = "";
        var result = null, errMsg = null;
        while (true) {
          var ch = await reader.read();
          if (ch.done) break;
          buf += dec.decode(ch.value, { stream: true });
          var parts = buf.split("\n"); buf = parts.pop();
          for (var p = 0; p < parts.length; p++) {
            var ln = parts[p].trim(); if (!ln) continue;
            var obj; try { obj = JSON.parse(ln); } catch (_) { continue; }
            if (obj.type === "log") termLine(obj.line, "log");
            else if (obj.type === "result") result = obj;
            else if (obj.type === "error") errMsg = obj.error;
          }
        }
        if (errMsg) throw new Error(errMsg);
        if (!result) throw new Error("no result returned");
        e.renderedUrl = result.rendered; e.n = result.count; e.status = "done"; e.error = null;
        if (result.dir) outdir = result.dir;
        termLine("✓ " + e.name + ": " + result.count + " object" + (result.count === 1 ? "" : "s") +
                 "  →  " + (result.dir || ""), "ok");
      } catch (err) {
        e.status = "error"; e.error = err.message;
        termLine("✗ " + e.name + ": " + err.message, "err");
      }
      renderGallery();
    }

    var done = images.filter(function (x) { return x.status === "done"; }).length;
    $("resCount").textContent = done;
    $("resPath").textContent = outdir;
    if (done > 0) $("resbar").style.display = "flex";

    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M5 3l14 9-14 9V3z"></path></svg> Run inference';
    btn.disabled = images.length === 0;
  }

  // Changing model/conf invalidates prior results: mark them re-runnable.
  function markStale() {
    var changed = false;
    images.forEach(function (e) { if (e.status === "done") { e.status = "queued"; changed = true; } });
    if (changed) renderGallery();
    $("runBtn").disabled = images.length === 0;
  }

  function renderGallery() {
    var g = $("gallery");
    g.innerHTML = "";
    images.forEach(function (e) {
      var card = document.createElement("div"); card.className = "card";
      var stateHtml, ctHtml;
      if (e.status === "done") {
        stateHtml = '<span class="state done">rendered</span>';
        ctHtml = '<b>' + e.n + '</b> obj';
      } else if (e.status === "busy") {
        stateHtml = '<span class="state busy"><span class="minispin"></span>running</span>';
        ctHtml = '...';
      } else if (e.status === "error") {
        stateHtml = '<span class="state err">error</span>';
        ctHtml = 'failed';
      } else {
        stateHtml = '<span class="state queued">queued</span>';
        ctHtml = e.w ? (e.w + "x" + e.h) : "...";
      }
      card.innerHTML =
        '<div class="imgbox ' + (e.status === "done" ? "" : "queued") + '">' + stateHtml +
          '<img src="' + e.renderedUrl + '" alt=""></div>' +
        '<div class="cap"><span class="nm"></span><span class="ct">' + ctHtml + '</span></div>';
      card.querySelector(".nm").textContent = e.name;
      if (e.error) card.querySelector(".nm").title = e.error;
      g.appendChild(card);
    });
  }

  function toast(msg, path) {
    var t = $("toast");
    t.innerHTML = msg + (path ? ' <span class="pp">' + path + '</span>' : "");
    t.classList.add("show");
    setTimeout(function () { t.classList.remove("show"); }, 3600);
  }

  // ---- controls ----
  $("pickFiles").addEventListener("click", function () { $("fileInput").click(); });
  $("pickFolder").addEventListener("click", function () { $("folderInput").click(); });
  $("fileInput").addEventListener("change", function (ev) { addFiles(ev.target.files); ev.target.value = ""; });
  $("folderInput").addEventListener("change", function (ev) { addFiles(ev.target.files); ev.target.value = ""; });
  $("runBtn").addEventListener("click", runInference);
  $("termClearBtn").addEventListener("click", termClear);
  $("model").addEventListener("change", markStale);
  $("conf").addEventListener("change", markStale);
  $("openFolder").addEventListener("click", function () {
    fetch("/api/open-folder", { method: "POST" }).then(function (r) { return r.json(); }).then(function (j) {
      if (!j.ok) toast("Results folder:", j.dir || "");
    }).catch(function () { toast("Could not open the folder.", ""); });
  });

  // ---- folder-aware drag & drop ----
  var tb = $("toolbar");
  ["dragenter", "dragover"].forEach(function (ev) { window.addEventListener(ev, function (e) { e.preventDefault(); tb.classList.add("drag"); }); });
  ["dragleave", "drop"].forEach(function (ev) { window.addEventListener(ev, function (e) { e.preventDefault(); if (ev === "dragleave" && e.relatedTarget) return; tb.classList.remove("drag"); }); });

  function collectEntries(entries, done) {
    var files = [], pending = 1;
    function finish() { pending--; if (pending === 0) done(files); }
    function walk(entry) {
      if (!entry) return;
      if (entry.isFile) {
        pending++;
        entry.file(function (f) { if (f.type && f.type.indexOf("image/") === 0) files.push(f); finish(); }, finish);
      } else if (entry.isDirectory) {
        var reader = entry.createReader();
        pending++;
        (function readBatch() {
          reader.readEntries(function (batch) { if (batch.length) { batch.forEach(walk); readBatch(); } else finish(); }, finish);
        })();
      }
    }
    entries.forEach(walk); finish();
  }

  window.addEventListener("drop", function (e) {
    e.preventDefault();
    var dt = e.dataTransfer; if (!dt) return;
    var items = dt.items;
    if (items && items.length && items[0].webkitGetAsEntry) {
      var entries = [];
      for (var i = 0; i < items.length; i++) { var en = items[i].webkitGetAsEntry(); if (en) entries.push(en); }
      if (entries.length) { collectEntries(entries, function (files) { addFiles(files); }); return; }
    }
    if (dt.files) addFiles(dt.files);
  });

  // ---- Ctrl+V paste ----
  window.addEventListener("paste", function (e) {
    var items = (e.clipboardData || window.clipboardData).items, files = [];
    for (var i = 0; i < items.length; i++) { if (items[i].kind === "file") { var f = items[i].getAsFile(); if (f) files.push(f); } }
    if (files.length) { e.preventDefault(); addFiles(files); }
  });
})();
</script>
</body>
</html>
"""

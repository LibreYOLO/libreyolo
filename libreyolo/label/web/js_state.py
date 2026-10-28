"""JS: client state, theme, colors, canvas transforms."""

PART = r""""use strict";
const $ = s => document.querySelector(s);
const cv = $("#cv"), ctx = cv.getContext("2d");
let DS = null, IMAGES = [], idx = -1;
let ceRows = [];              // class-editor working rows
let autosaveTimer = null, saveInFlight = false;   // crash-proof periodic flush of the manual loop
let img = new Image(), imgOk = false;
let boxes = [], editable = true, dirty = false;
let active = 0, sel = -1;
let view = {scale:1, ox:0, oy:0};
let VW = 1, VH = 1;
let loadSeq = 0;
let undoStack = [], redoStack = [];
let savedSnap = "";   // snapshot of the last saved/loaded state (undo-back-to-clean baseline)
let gestureSnap = null;
let cursor = null;
let hover = -1;
let stageMsg = "";
let progSig = "";
let assist = null, assistModel = null, conf = 0.35;
let ghosts = [];
let polys = [];              // polygon annotations (image-px pts), from SAM or file
let selPoly = -1;
let selBoxes = new Set();   // multi-selection of box indices (shift-click / Ctrl+A); always cleared on image/project change so indices never go stale
let tool = "box";            // "box" (draw/select) or "seg" (SAM click/box-to-mask)
let segBusy = false;
let segRect = null;          // box being dragged in segment mode (image px)
let polyDraft = null;        // in-progress manual polygon: flat [x1,y1,x2,y2,...] image px
let obbRect = null;          // in-progress oriented-box drag (image px)
let curRev = null;           // the loaded label file's revision (mtime), for stale-save conflict detection
let suggestedIds = new Set();
let listFilter = "all";
let imgQuery = "";             // sidebar filename filter
let selSet = null;             // ids selected from the embedding map (sidebar filter)
let radarFindings = [];        // Radar disagreements overlaid on the current image
let radarDeck = [];            // worst-first deck from the last Radar scan
let radarLast = null;          // last scan's result, so the Radar button reopens it without rescanning
let loupeOn = false;           // magnifier toggled with L (opt-in; no auto-show while drawing)
let gradData = null, gradW = 0, gradH = 0, gradScale = 1;  // edge map for Tighten/magnetic
let mapPoints = [], mapFit = null, mapLasso = null;        // embedding scatter + lasso
let boostTimer = null;
let _cssCache = {};
const HANDLES = ["nw","n","ne","e","se","s","sw","w"];
const HR = 6;
let CLSCOL = [];   // per-class custom colors (hex) from the project sidecar; falls back to the golden-angle palette
function _hexToRgb(h){ h=String(h).replace('#',''); if(h.length===3) h=h.split('').map(c=>c+c).join(''); const n=parseInt(h,16); return [(n>>16)&255,(n>>8)&255,n&255]; }
const color = i => { const c=CLSCOL[i]; if(c) return c; const h=(i*137.508)%360; const l=62 - 16*Math.cos((h-50)*Math.PI/180); return 'hsl('+h+' 70% '+l+'%)'; };
const colorA = (i,a) => { const c=CLSCOL[i]; if(c && /^#/.test(c)){ const [r,g,b]=_hexToRgb(c); return 'rgba('+r+','+g+','+b+','+a+')'; } const h=(i*137.508)%360; const l=62 - 16*Math.cos((h-50)*Math.PI/180); return 'hsl('+h+' 70% '+l+'% / '+a+')'; };
const clamp01 = v => v<0?0:v>1?1:v;
const ICO_CHECK = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 6"/></svg>';
const ICO_COPY = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';
const ICO_X = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>';
const ICO_WARN = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l9 16H3z"/><path d="M12 10v4M12 17h.01"/></svg>';
const ICO_SUN = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const ICO_MOON = '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
function applyTheme(t){
  document.documentElement.classList.toggle("light", t==="light");
  _cssCache = {};   // theme changed -> re-read CSS vars used by canvas overlays
  try{ localStorage.setItem("ll-theme", t); }catch(e){}
  const b=document.querySelector("#themebtn"); if(b) b.innerHTML = (t==="light") ? ICO_MOON : ICO_SUN;
  const h=document.querySelector("#hometheme"); if(h) h.innerHTML = (t==="light") ? ICO_MOON : ICO_SUN;
  if(typeof imgOk!=="undefined" && imgOk) draw();
}
function toggleTheme(){ applyTheme(document.documentElement.classList.contains("light") ? "dark" : "light"); }
(function(){ let t; try{ t=localStorage.getItem("ll-theme"); }catch(e){}
  if(location.hash==="#light") t="light"; else if(location.hash==="#dark") t="dark";
  if(!t) t=(window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches)?"light":"dark";
  document.documentElement.classList.toggle("light", t==="light"); })();

// ---- remappable hotkeys (persisted in localStorage; edited in the ? modal) ----
const KEY_ACTIONS = [
  {id:"prev",          def:"a", label:"Previous image"},
  {id:"next",          def:"d", label:"Next image"},
  {id:"nextunlabeled", def:"e", label:"Next unlabeled"},
  {id:"carry",         def:"c", label:"Copy previous labels"},
  {id:"prelabel",      def:"r", label:"Auto-label this image"},
  {id:"toolbox",       def:"b", label:"Box tool"},
  {id:"toolpoly",      def:"p", label:"Polygon tool"},
  {id:"toolsam",       def:"s", label:"Smart segment (SAM)"},
  {id:"toolobb",       def:"o", label:"Oriented-box tool"},
  {id:"tighten",       def:"t", label:"Tighten box to edges"},
  {id:"loupe",         def:"l", label:"Loupe magnifier"},
  {id:"radar",         def:"y", label:"Label-Error Radar"},
  {id:"flagged",       def:"n", label:"Next flagged"},
  {id:"map",           def:"m", label:"Embedding map"},
  {id:"fit",           def:"f", label:"Fit to view"},
];
let KEYMAP = {};
(function(){ try{ const o=JSON.parse(localStorage.getItem("ll-keys")||"{}");
  if(o && typeof o==="object") KEYMAP=o; }catch(e){} })();
function keyFor(id){ if(KEYMAP[id]) return KEYMAP[id];
  const a=KEY_ACTIONS.find(x=>x.id===id); return a?a.def:""; }
function setKey(id,k){ KEYMAP[id]=k; try{ localStorage.setItem("ll-keys", JSON.stringify(KEYMAP)); }catch(e){} }
function resetKeys(){ KEYMAP={}; try{ localStorage.removeItem("ll-keys"); }catch(e){} }

const esc = s => String(s).replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// ---- transforms ----
const sx = x => view.ox + x*view.scale;
const sy = y => view.oy + y*view.scale;
const ix = px => (px - view.ox)/view.scale;
const iy = py => (py - view.oy)/view.scale;
function rr(x,y,w,h,r){ ctx.beginPath(); if(ctx.roundRect) ctx.roundRect(x,y,w,h,r); else ctx.rect(x,y,w,h); ctx.fill(); }
// A readable label tag: dark pill + class-color dot + light text (never neon text on the photo).
function drawChip(x, y, text, ci){
  ctx.font = "600 11.5px ui-sans-serif,system-ui,sans-serif";
  const padL=18, padR=9, h=18, tw=ctx.measureText(text).width, w=padL+tw+padR;
  let cy = y - h - 3; if(cy < 1) cy = y + 2;          // flip below if it'd clip the top edge
  ctx.save();
  ctx.shadowColor="rgba(2,6,23,.5)"; ctx.shadowBlur=5; ctx.shadowOffsetY=1;
  ctx.fillStyle="rgba(13,17,28,.92)"; rr(x, cy, w, h, 5);
  ctx.shadowColor="transparent"; ctx.shadowBlur=0; ctx.shadowOffsetY=0;
  ctx.fillStyle=color(ci); ctx.beginPath(); ctx.arc(x+9, cy+h/2, 3.4, 0, 6.2832); ctx.fill();
  ctx.fillStyle="#e8edf6"; ctx.textBaseline="middle"; ctx.fillText(text, x+padL, cy+h/2+0.5);
  ctx.restore();
}

function fit(){
  if(!imgOk) return;
  const pad = 40, W = VW, H = VH;
  const s = Math.min((W-pad)/img.naturalWidth, (H-pad)/img.naturalHeight);
  view.scale = s>0 ? s : 1;
  view.ox = (W - img.naturalWidth*view.scale)/2;
  view.oy = (H - img.naturalHeight*view.scale)/2;
}
function resizeCanvas(){
  const r = $("#stage").getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio||1, 2);
  VW = Math.max(1, Math.floor(r.width));
  VH = Math.max(1, Math.floor(r.height));
  cv.width = Math.round(VW*dpr); cv.height = Math.round(VH*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}
function zoomBy(f){
  const cx=VW/2, cy=VH/2, bx=ix(cx), by=iy(cy);
  view.scale = Math.max(0.02, Math.min(64, view.scale*f));
  view.ox = cx-bx*view.scale; view.oy = cy-by*view.scale; draw();
}

"""

"""JS: undo, label load/save/autosave, navigation, AI assist + ghosts + SAM."""

PART = r"""// ---- undo ----
function snap(){ return JSON.stringify({b:boxes, p:polys}); }
function applyUndo(s){ const o=JSON.parse(s); boxes=o.b||[]; polys=o.p||[]; }
function pushUndo(){ undoStack.push(snap()); if(undoStack.length>50) undoStack.shift(); redoStack = []; }
function snapStart(){ gestureSnap = snap(); }
function snapCommit(){
  if(gestureSnap!==null && gestureSnap!==snap()){
    undoStack.push(gestureSnap); if(undoStack.length>50) undoStack.shift(); redoStack = [];
  }
  gestureSnap = null;
}
// Undo/redo move one snapshot between the two stacks; both reschedule autosave so a
// restored state is just as crash-safe as a freshly drawn one.
function applyHistory(fromStack, toStack){
  if(!fromStack.length) return;
  toStack.push(snap()); applyUndo(fromStack.pop()); sel=-1; selPoly=-1;
  dirty = (snap()!==savedSnap); setSave(dirty?"unsaved":(editable?"saved":"read-only"));
  if(dirty) scheduleAutosave();
  draw();
}

// ---- load / save ----
async function load(i){
  if(i<0||i>=IMAGES.length) return;
  if(IMAGES[i] && IMAGES[i].status==="deleted") return;   // never open a quarantined image
  const myGen = ++loadSeq;
  if(dirty && idx>=0 && !(await save())){
    banner("Save failed - staying on this image so you don't lose work."); return;
  }
  if(myGen !== loadSeq) return;   // a newer load() started during the await -> bail before mutating idx
  if(dirty){   // edits landed *during* the save (snap changed mid-POST) -> still unsaved; don't discard them
    banner("You edited while saving - press → again to save those changes."); return;
  }
  idx = i; sel = -1; selPoly = -1; selBoxes.clear(); hover = -1; undoStack = []; redoStack = []; gestureSnap = null; boxes = []; polys = []; ghosts = [];
  radarFindings = []; gradData = null;
  imgOk = false; editable = false; curRev = null;   // canvas stays inert until metadata loads; a failed fetch must not leave it editable on the previous image
  let lab;
  try{ lab = await jget(`/api/label/${i}`); }
  catch(e){ if(myGen!==loadSeq) return; stageMsg = "Couldn't load this image's labels - pick another image."; draw(); return; }
  if(myGen !== loadSeq) return;
  editable = lab.editable;
  curRev = (lab.rev!=null) ? lab.rev : null;
  stageMsg = "Loading…"; draw();
  img = new Image();
  img.onload = ()=>{
    if(myGen !== loadSeq) return;
    imgOk = true;
    const anns = lab.annotations||[]; const iw=img.naturalWidth, ih=img.naturalHeight;
    boxes = anns.filter(a=>a.type==="box").map(b=>({
      cls:b.cls, x:(b.cx-b.w/2)*iw, y:(b.cy-b.h/2)*ih, w:b.w*iw, h:b.h*ih}));
    polys = anns.filter(a=>a.type==="poly").map(p=>({
      cls:p.cls, pts:p.points.map((v,k)=> k%2===0? v*iw : v*ih)}));
    dirty = false; savedSnap = snap(); setSave(editable?"saved":"read-only");
    stageMsg = ""; fit(); draw();
    if(assist && assist.available && editable && suggestedIds.has(i)){
      fetch(`/api/assist/pending/${i}`).then(r=>r.json()).then(d=>{
        if(myGen!==loadSeq) return;
        if((d.suggestions||[]).length) showGhosts(d.suggestions);
      }).catch(()=>{});
    }
  };
  img.onerror = ()=>{ if(myGen !== loadSeq) return; imgOk=false; stageMsg="Could not load image"; setSave("image error"); draw(); };
  img.src = `/api/image/${i}`;
  $("#banner").style.display = "none";
  if(!editable) banner("Read-only: this image has polygon/OBB labels (box-only mode won't overwrite them).");
  else if(DS && !DS.writable) banner(DS.reason);
  markRow(); updateProgress();
}
function pxToNorm(b){
  const iw=img.naturalWidth, ih=img.naturalHeight;
  let x=b.x, y=b.y, w=b.w, h=b.h;
  if(w<0){x+=w;w=-w;} if(h<0){y+=h;h=-h;}
  return {cls:b.cls, cx:clamp01((x+w/2)/iw), cy:clamp01((y+h/2)/ih), w:clamp01(w/iw), h:clamp01(h/ih)};
}
function polyToNorm(p){
  const iw=img.naturalWidth, ih=img.naturalHeight, out=[];
  for(let k=0;k<p.pts.length;k+=2){ out.push(clamp01(p.pts[k]/iw)); out.push(clamp01(p.pts[k+1]/ih)); }
  return out;
}
async function save(){
  clearTimeout(autosaveTimer); autosaveTimer = null;   // a real save supersedes any pending autosave
  if(!imgOk || !editable || (DS && !DS.writable)){ return true; }
  const totalShapes = boxes.length + polys.length;   // everything on the canvas
  const anns = boxes.map(pxToNorm).filter(b=>b.w>0&&b.h>0)
    .map(b=>({type:"box", cls:b.cls, cx:b.cx, cy:b.cy, w:b.w, h:b.h}));
  polys.forEach(p=>{ const pts=polyToNorm(p); if(pts.length>=6) anns.push({type:"poly", cls:p.cls, points:pts}); });
  const cur = idx, sent = snap();   // snapshot of exactly what we're sending
  try{
    const q = `epoch=${(DS&&DS.epoch)||0}` + (curRev!=null ? `&rev=${encodeURIComponent(curRev)}` : "");
    const r = await fetch(`/api/label/${cur}?${q}`,{method:"POST",
      headers:{"Content-Type":"application/json"}, body:JSON.stringify({annotations:anns})});
    if(!r.ok){ setSave("save failed"); banner((await r.json()).error||"save failed"); return false; }
    const d = await r.json().catch(()=>({}));
    if(d && d.rev!=null) curRev = d.rev;   // adopt the new revision so our own next save doesn't self-conflict
    const saved = (d && typeof d.count==="number") ? d.count : anns.length;
    // Compare against EVERY shape on the canvas, not just what we sent: a box clipped
    // to zero area (or a <3-pt polygon) is filtered out client-side and would
    // otherwise look like a clean save while silently vanishing on reload.
    const dropped = totalShapes - saved;
    if(dropped > 0){
      // The shape didn't reach disk (client-filtered zero-area, or server-sanitized
      // degenerate/collinear). Keep the edit DIRTY/failed so close & navigation warn
      // and it can't silently vanish on the next load -- the user must fix or delete it.
      dirty = true; setSave("unsaved");
      suggestedIds.delete(cur); setRowStatus(cur, saved? "labeled":"empty"); scheduleStats();
      banner(`${dropped} invalid shape${dropped>1?"s":""} dropped (degenerate) - fix or delete it to save.`);
      return false;
    }
    savedSnap = sent; dirty = (snap()!==sent);   // edits made during the POST keep it unsaved
    setSave(dirty?"unsaved":"saved");
    const el=$('#save'); el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
    suggestedIds.delete(cur);
    setRowStatus(cur, saved? "labeled":"empty");
    scheduleStats();
    return true;
  }catch(e){ setSave("save failed"); return false; }
}
function markDirty(){ dirty = true; setSave("unsaved"); scheduleAutosave(); }
// Crash-proof the manual loop: flush dirty edits to disk on a steady cadence so a
// browser/tab/server crash can't lose work drawn since the last navigation. Reuses
// the proven save() path; never fires mid-gesture or while a save is already on the wire.
function scheduleAutosave(){ if(!autosaveTimer) autosaveTimer = setTimeout(autosaveTick, 1500); }
async function autosaveTick(){
  autosaveTimer = null;
  if(mode){ scheduleAutosave(); return; }                       // a drag/resize is in progress
  if(!dirty || saveInFlight || !imgOk || !editable || (DS && !DS.writable)) return;
  saveInFlight = true;
  try{ await save(); } finally{ saveInFlight = false; }
  if(dirty) scheduleAutosave();                                 // edits made during the save -> flush again
}
function setSave(t){
  const e = $("#save"); e.textContent = t;
  const fail = (t==="save failed" || t==="image error" || t==="no images");
  e.className = "save" + (t==="unsaved"?" dirty": t==="saved"?" saved": fail?" fail":"");
}
function banner(msg){ const b=$("#banner"); b.textContent=msg; b.style.display="flex"; }
function updateProgress(){
  if(idx<0 || !IMAGES[idx]) return;
  const live = IMAGES.filter(im=>im.status!=='deleted');   // tombstones out of the count
  const total = live.length;
  const done = live.filter(im=>im.status==='labeled').length;  // only accepted labels, matches Dataset Health
  const n = boxes.length + polys.length;
  const sig = done+"|"+total+"|"+suggestedIds.size+"|"+idx+"|"+n;
  if(sig===progSig) return; progSig = sig;
  const rev = suggestedIds.size ? ` &middot; <b style="color:var(--ai)">${suggestedIds.size}</b> to review` : "";
  $("#counter").innerHTML = `<b>${done}</b>/${total} labeled${rev}`;
  const hud = $("#hud");
  if(hud) hud.innerHTML = `${idx+1} / ${total} &nbsp;&middot;&nbsp; ${n} box${n===1?'':'es'}`
    + ` &nbsp;&middot;&nbsp; <span style="color:var(--tx3)">${esc(IMAGES[idx].name)}</span>`;
}
function visibleIds(){ return IMAGES.filter(passFilter).map(im=>im.id); }  // ordered, filter-aware, excludes deleted
function step(dir){
  const vis = visibleIds(); const L=vis.length; if(!L) return;
  const p = vis.indexOf(idx);
  load(p<0 ? vis[dir>0?0:L-1] : vis[(p+dir+L)%L]);
}
function nextUnlabeled(dir){
  const vis = visibleIds(); const L=vis.length; if(!L) return;
  let p = vis.indexOf(idx); if(p<0) p = dir>0 ? -1 : L;
  for(let n=1;n<=L;n++){ const id=vis[((p+dir*n)%L+L)%L];
    if(IMAGES[id] && IMAGES[id].status==="unlabeled"){ load(id); return; } }
  banner("No more unlabeled images");
}
async function carryForward(){
  if(!imgOk || !editable || (DS && !DS.writable)) return;
  if(boxes.length + polys.length > 0){ banner("This image already has labels - carry-forward only fills an empty image."); return; }
  const vis = visibleIds(), p = vis.indexOf(idx);
  const prevId = p>0 ? vis[p-1] : -1;
  if(prevId<0){ banner("No previous image to copy from."); return; }
  const myGen = loadSeq, srcIdx = idx;
  let lab; try{ lab = await jget(`/api/label/${prevId}`); }catch(e){ banner("Couldn't read the previous image's labels."); return; }
  if(myGen!==loadSeq || idx!==srcIdx) return;   // navigated during the fetch -> don't paste onto the wrong image
  if(boxes.length + polys.length > 0){ banner("This image already has labels - carry-forward only fills an empty image."); return; }   // drawn during the fetch -> don't merge stale labels
  if(lab.editable===false){ banner("The previous image's labels are read-only (keypoints/unsupported) - can't carry them forward."); return; }   // partial view: don't drop the unparsed rows
  const anns = lab.annotations||[];
  if(!anns.length){ banner("The previous image has no labels to copy."); return; }
  const iw=img.naturalWidth, ih=img.naturalHeight; let n=0;
  pushUndo();
  anns.forEach(a=>{
    if(a.type==="box"){ const b={cls:a.cls, x:(a.cx-a.w/2)*iw, y:(a.cy-a.h/2)*ih, w:a.w*iw, h:a.h*ih};
      clipToImage(b); if(b.w>0 && b.h>0){ boxes.push(b); n++; } }
    else if(a.type==="poly"){ const q={cls:a.cls, pts:a.points.map((v,k)=> k%2===0? v*iw : v*ih)}; clipPoly(q); polys.push(q); n++; }
  });
  if(!n){ undoStack.pop(); return; }
  markDirty(); draw();
  banner(`Copied ${n} annotation${n===1?"":"s"} from the previous image - nudge them into place, then save.`);
}
function duplicateSelected(){
  if(sel<0 || !imgOk || !editable || (DS && !DS.writable)) return;
  const b=boxes[sel]; pushUndo();
  const off=Math.max(6, Math.abs(b.w)*0.06);
  boxes.push({cls:b.cls, x:b.x+off, y:b.y+off, w:b.w, h:b.h});
  selBoxes=new Set([boxes.length-1]); sel=boxes.length-1; selPoly=-1; clipToImage(boxes[sel]); markDirty(); draw();
}

// ---- AI auto-label (suggest -> review -> accept; nothing written unverified) ----
async function initAssist(){
  try{ assist = await jget("/api/assist/status"); }catch(e){ assist = null; }
  const bar = $("#assistbar"), toolAi = $("#toolAi");
  if(!assist || !assist.available){ if(bar) bar.style.display="none"; if(toolAi) toolAi.style.display="none"; return; }
  if(DS && DS.task==="obb"){
    // The server refuses prelabel/autolabel/SAM/Radar/Boost for OBB projects (their
    // box/mask outputs would corrupt oriented-box labels); offer only the Map.
    assist.sam=false; assist.boost=false;
    if(bar) bar.style.display="none"; if(toolAi) toolAi.style.display="none";
    const mb=$("#mapbtn"); if(mb && assist.embed) mb.style.display="grid";
    return;
  }
  assistModel = assist.default;
  const sel = $("#amodel"); sel.innerHTML = "";
  if(assist.locate){ const o=document.createElement("option"); o.value="__locate__"; o.textContent="Locate Anything (text)"; sel.appendChild(o); }
  assist.models.forEach(m=>{ const o=document.createElement("option");
    o.value=m; o.textContent=m; if(m===assistModel) o.selected=true; sel.appendChild(o); });
  sel.onchange = ()=>{ assistModel = sel.value; updateEngineUI(); };
  if(assist.boosted) addBoostedOption(false);
  const cs = $("#aconf"); cs.value = conf; $("#aconfval").textContent = conf.toFixed(2);
  cs.oninput = ()=>{ conf = parseFloat(cs.value); $("#aconfval").textContent = conf.toFixed(2); };
  $("#aprelabel").onclick = ()=> prelabelCurrent();
  $("#aautolabel").onclick = ()=> autolabelAll();
  const asam = $("#asam");
  if(assist.sam && asam){ asam.style.display="inline-flex"; asam.onclick = ()=> setTool("seg"); }
  bar.style.display = "flex";
  if(assist.sam){ const ts=$("#toolSeg"); if(ts) ts.style.display="grid"; }
  const ar=$("#aradar"); if(ar){ ar.style.display="inline-flex"; ar.onclick=()=>showRadar(); }
  const mb=$("#mapbtn"); if(mb && assist.embed) mb.style.display="grid";
  const bb=$("#boostbtn"); if(bb && assist.boost) bb.style.display="grid";
  if(DS && DS.task==="segment"){
    // Segment projects: the server refuses the BOX producers (prelabel /
    // autolabel / Boost); keep SAM, Radar, the Map, and the conf/model fields.
    assist.boost=false;
    const hide=["#aautolabel","#aprelabel","#boostbtn"];
    hide.forEach(s=>{ const el=$(s); if(el) el.style.display="none"; });
    if(toolAi) toolAi.style.display="none";
  }
  updateEngineUI();
}
function updateEngineUI(){
  const la = assistModel==="__locate__";
  const lp=$("#laprompt"), cf=$("#aconffield");
  if(lp) lp.style.display = la? "inline-block":"none";
  if(cf) cf.style.display = la? "none":"inline-flex";
  if(la){ banner("Locate Anything - NVIDIA non-commercial model (downloads a 3B model + runs remote code). Type the objects to find, then Auto-label."); if(lp) lp.focus(); }
  else { $("#banner").style.display="none"; }
}
function laQuery(){
  const ep = "&epoch="+((DS&&DS.epoch)||0);   // let the server reject a stale tab's run after a project switch
  if(assistModel==="__locate__") return "engine=locate&classes="+encodeURIComponent(($("#laprompt").value||"").trim())+ep;
  return "model="+encodeURIComponent(assistModel)+"&conf="+conf+ep;
}
function restoreSave(){ setSave(dirty?"unsaved":(editable?"saved":"read-only")); }
function ghostsFromNorm(list){
  if(!imgOk) return [];
  const iw=img.naturalWidth, ih=img.naturalHeight;
  return list.map(s=>({cls:s.cls, name:s.name, conf:s.conf, mapped:s.mapped,
    x:(s.cx-s.w/2)*iw, y:(s.cy-s.h/2)*ih, w:s.w*iw, h:s.h*ih}));
}
function showGhosts(list){
  ghosts = ghostsFromNorm(list); draw();
  if(ghosts.length){
    const unm = ghosts.filter(g=>!g.mapped).length;
    banner(`${ghosts.length} AI suggestion${ghosts.length===1?"":"s"} - `
      + "Enter: accept all · click one · Alt+click rejects · Esc clears"
      + (unm? ` · ${unm} unmatched (grey) - set a class to accept`:""));
  } else banner("No objects found above the confidence threshold");
}
async function prelabelCurrent(){
  if(!assist || !assist.available || idx<0 || !imgOk) return;
  if(DS && (DS.task==="obb" || DS.task==="segment")) return;   // box suggestions would corrupt these labels
  if(!editable || (DS && !DS.writable)){ banner("This image/dataset is read-only - auto-label is disabled."); return; }
  const myGen = loadSeq; setSave("running model…");
  try{
    const r = await fetch(`/api/assist/prelabel/${idx}?${laQuery()}`, {method:"POST"});
    if(myGen!==loadSeq) return;
    if(!r.ok){ const e=await r.json().catch(()=>({})); banner("Auto-label failed: "+(e.error||r.status)); restoreSave(); return; }
    const data = await r.json();
    showGhosts(data.suggestions||[]);
    if((data.suggestions||[]).length){
      suggestedIds.add(idx);
      setRowStatus(idx, "suggested");   // make manual prelabels findable via the Review filter, like bulk autolabel
    } else {
      clearReviewState();   // no ghosts -> don't leave a phantom 'suggested' row in the Review filter
    }
  }catch(e){ banner("Auto-label failed"); }
  restoreSave();
}
function pointInPoly(x,y,pts){
  let inside=false; const n=pts.length/2;
  for(let i=0, j=n-1; i<n; j=i++){
    const xi=pts[2*i], yi=pts[2*i+1], xj=pts[2*j], yj=pts[2*j+1];
    if(((yi>y)!==(yj>y)) && (x < (xj-xi)*(y-yi)/(yj-yi)+xi)) inside=!inside;
  }
  return inside;
}
function hitPoly(mx,my){
  const x=ix(mx), y=iy(my);
  for(let i=polys.length-1;i>=0;i--){ if(pointInPoly(x,y,polys[i].pts)) return i; }
  return -1;
}
function hitVertex(p, mx, my){
  for(let k=0;k<p.pts.length;k+=2){
    if(Math.abs(sx(p.pts[k])-mx)<=HR+2 && Math.abs(sy(p.pts[k+1])-my)<=HR+2) return k/2;
  }
  return -1;
}
async function segmentAt(mx,my){
  if(segBusy || !assist || !assist.sam || idx<0 || !imgOk || !editable || (DS && !DS.writable)) return;   // view-only dataset: no canvas mutation
  if(!(DS.names||[]).length){ banner("Add a class before segmenting - every annotation needs one."); openClassEdit(); return; }   // nc=0 -> server would drop the polygon
  const iw=img.naturalWidth, ih=img.naturalHeight, X=ix(mx), Y=iy(my);
  if(X<0||Y<0||X>iw||Y>ih) return;
  segBusy=true; const myGen=loadSeq; banner("Segmenting… (SAM, on your machine)"); cv.style.cursor="wait";
  try{
    const r = await fetch(`/api/assist/segment/${idx}`, {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({x:X/iw, y:Y/ih})});
    if(myGen!==loadSeq) return;
    if(!r.ok){ const e=await r.json().catch(()=>({})); banner("Segment failed: "+(e.error||r.status)); return; }
    const d = await r.json();
    if(!d.polygon || d.polygon.length<6){ banner("No object found there - try clicking on an object"); return; }
    pushUndo();
    polys.push({cls:active, pts:d.polygon.map((v,k)=> k%2===0? v*iw : v*ih)});
    selPoly=polys.length-1; sel=-1; markDirty(); $("#banner").style.display="none"; draw();
  }catch(e){ banner("Segment failed"); }
  finally{ segBusy=false; cv.style.cursor="crosshair"; }
}
async function segmentBox(r){
  if(segBusy || !assist || !assist.sam || idx<0 || !imgOk || !editable) return;
  if(!(DS.names||[]).length){ banner("Add a class before segmenting - every annotation needs one."); openClassEdit(); return; }   // nc=0 -> server would drop the polygon
  const iw=img.naturalWidth, ih=img.naturalHeight;
  const x1=Math.max(0,Math.min(r.x0,r.x1)), y1=Math.max(0,Math.min(r.y0,r.y1));
  const x2=Math.min(iw,Math.max(r.x0,r.x1)), y2=Math.min(ih,Math.max(r.y0,r.y1));
  if(x2-x1<4 || y2-y1<4) return;
  segBusy=true; const myGen=loadSeq; banner("Segmenting… (SAM box prompt)"); cv.style.cursor="wait";
  try{
    const rr = await fetch(`/api/assist/segment/${idx}`, {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({box:[x1/iw, y1/ih, x2/iw, y2/ih]})});
    if(myGen!==loadSeq) return;
    if(!rr.ok){ const e=await rr.json().catch(()=>({})); banner("Segment failed: "+(e.error||rr.status)); return; }
    const d = await rr.json();
    if(!d.polygon || d.polygon.length<6){ banner("No object found in that box"); return; }
    pushUndo();
    polys.push({cls:active, pts:d.polygon.map((v,k)=> k%2===0? v*iw : v*ih)});
    selPoly=polys.length-1; sel=-1; markDirty(); $("#banner").style.display="none"; draw();
  }catch(e){ banner("Segment failed"); }
  finally{ segBusy=false; cv.style.cursor="crosshair"; }
}
function hitGhost(mx,my){
  const x=ix(mx), y=iy(my);
  for(let i=ghosts.length-1;i>=0;i--){ const g=ghosts[i];
    if(x>=g.x && x<=g.x+g.w && y>=g.y && y<=g.y+g.h) return i; }
  return -1;
}
function acceptGhost(i){
  const g=ghosts[i]; if(!g) return;
  if(!editable || (DS && !DS.writable)){ banner("This image/dataset is read-only - suggestions can't be accepted."); return; }
  if(g.cls==null && !(DS.names||[]).length){ banner("Add a class first - unmatched suggestions take the active class."); openClassEdit(); return; }
  // Unmatched suggestion (no dataset class): honour the UI's promise and apply the
  // active palette class the user selected, so open-vocab / custom-name detections
  // are acceptable instead of impossible to take without redrawing.
  const cls = (g.cls==null) ? active : g.cls;
  pushUndo();
  boxes.push({cls:cls, x:g.x, y:g.y, w:g.w, h:g.h});
  ghosts.splice(i,1); markDirty(); draw();
  if(!ghosts.length) $("#banner").style.display="none";
}
function clearReviewState(){   // a fully-dismissed image leaves the review queue (no phantom 'suggested' row)
  if(idx>=0 && suggestedIds.has(idx)){
    suggestedIds.delete(idx);
    // A present-but-empty label file (rev != "0") is a reviewed *background* -> keep it
    // 'empty' (done), don't bounce it back into To-do as 'unlabeled'.
    const fileExists = curRev!=null && String(curRev)!=="0";
    setRowStatus(idx, (boxes.length + polys.length) ? "labeled" : (fileExists ? "empty" : "unlabeled"));
  }
}
function rejectGhost(i){ if(ghosts[i]){ ghosts.splice(i,1); draw(); if(!ghosts.length){ $("#banner").style.display="none"; clearReviewState(); } } }
function acceptAllGhosts(){
  if(!editable || (DS && !DS.writable)){ banner("This image/dataset is read-only - suggestions can't be accepted."); return; }
  if(!ghosts.length) return;
  if(ghosts.some(g=>g.cls==null) && !(DS.names||[]).length){ banner("Add a class first - unmatched suggestions take the active class."); openClassEdit(); return; }
  // Matched ghosts keep their mapped class; unmatched ones take the active palette
  // class (same fallback as acceptGhost), so keyboard accept-all works on
  // open-vocab / custom-name datasets where no suggestion maps by name.
  pushUndo();
  ghosts.forEach(g=> boxes.push({cls:(g.cls==null?active:g.cls), x:g.x, y:g.y, w:g.w, h:g.h}));
  ghosts = [];
  markDirty(); draw();
  $("#banner").style.display="none";
}
function clearGhosts(){ if(ghosts.length){ ghosts=[]; draw(); $("#banner").style.display="none"; clearReviewState(); } }
async function autolabelAll(){
  if(!assist || !assist.available) return;
  if(dirty && idx>=0 && !(await save())){ banner("Couldn't save the current image; fix that first."); return; }
  if(dirty){ banner("You edited while saving - press → to save before auto-labeling all."); return; }   // in-flight edits: don't queue suggestions against stale labels
  const ov=$("#progress"), bar=$("#pbar"), txt=$("#ptxt");
  ov.style.display="flex"; bar.style.width="0%"; txt.textContent="Starting… (first run loads your model)";
  IMAGES.forEach(im=>{ if(im.status==="suggested") setRowStatus(im.id, "unlabeled"); });  // clear last run's stale review rows
  suggestedIds = new Set(); let suggested=0, totalBoxes=0, classes=[], failed=null; const t0=Date.now();
  const myEpoch = DS && DS.epoch;   // project-scoped: A/D image navigation must NOT abort a dataset-wide run
  try{
    const r = await fetch(`/api/assist/autolabel?${laQuery()}`, {method:"POST"});
    if(!r.ok){ const e=await r.json().catch(()=>({})); ov.style.display="none"; banner("Auto-label unavailable: "+(e.error||r.status)); return; }   // e.g. admin-only 403 on a shared server -> don't stream a JSON error as NDJSON
    const reader=r.body.getReader(), dec=new TextDecoder(); let buf="";
    for(;;){
      const {value,done}=await reader.read(); if(done) break;
      if(!DS || DS.epoch!==myEpoch){ ov.style.display="none"; return; }   // project switched -> stop
      buf += dec.decode(value,{stream:true}); let nl;
      while((nl=buf.indexOf("\n"))>=0){
        const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1);
        if(!line) continue; let o; try{o=JSON.parse(line);}catch(e){ continue; }
        if(o.type==="progress"){
          bar.style.width = Math.round(100*o.i/Math.max(1,o.total))+"%";
          txt.textContent = `${o.i} / ${o.total} - ${o.name}` + (o.count? `  (+${o.count})`:"");
          if(o.count>0){ suggestedIds.add(o.id); setRowStatus(o.id, "suggested"); }
        } else if(o.type==="done"){ suggested=o.suggested; totalBoxes=o.boxes; classes=o.classes||[]; }
        else if(o.type==="error"){ failed = o.error || "auto-label failed"; banner("Auto-label failed: "+failed); }
      }
    }
    if(failed){   // systemic failure (e.g. missing local weights) -> show it, don't claim a clean "Done"
      $(".ptitle").textContent = "Auto-label failed";
      txt.textContent = failed;
      bar.style.width="0%";
      setTimeout(()=>{ ov.style.display="none"; $(".ptitle").textContent="Auto-labeling with your model"; }, 3500);
      return;
    }
    bar.style.width="100%";
    const secs = ((Date.now()-t0)/1000).toFixed(1);
    const top = classes.slice(0,5).map(c=>`${c[1]} ${esc(c[0])}`).join("  ·  ");
    $(".ptitle").textContent = "Done - fully offline, your own model";
    txt.innerHTML = `<b style="color:var(--tx);font-size:16px">${totalBoxes} boxes</b> across `
      + `<b style="color:var(--tx)">${suggested}</b> images in <b style="color:var(--tx)">${secs}s</b>`
      + (top? `<div style="margin-top:9px;color:var(--tx2)">${top}</div>`:"")
      + `<div style="margin-top:9px;color:var(--tx3)">Review &amp; accept - nothing is saved until you confirm →</div>`;
    setTimeout(()=>{ ov.style.display="none"; $(".ptitle").textContent="Auto-labeling with your model"; }, 2800);
    progSig = "";
    const first = [...suggestedIds].sort((a,b)=>a-b)[0];
    if(first!=null) await load(first);
    else banner("No objects found in the unlabeled images");
  }catch(e){ banner("Auto-label failed"); ov.style.display="none"; }
}
function nextSuggested(){
  if(!suggestedIds.size) return;
  const ids=[...suggestedIds].sort((a,b)=>a-b);
  const nxt = ids.find(j=>j>idx);
  load(nxt!=null? nxt : ids[0]);
}

"""

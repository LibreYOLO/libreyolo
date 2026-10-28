"""JS: Radar, Tighten/loupe, Boost, embedding map."""

PART = r"""// ===== Data-quality + AI superpowers: Radar · Tighten · Loupe · Boost · Map =====
const ICO_SPIN = '<svg class="ic spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M12 3a9 9 0 1 0 9 9"/></svg>';
function getCss(v){ if(_cssCache[v]!=null) return _cssCache[v]; const c=(getComputedStyle(document.documentElement).getPropertyValue(v)||"").trim()||"#06b6d4"; _cssCache[v]=c; return c; }
function dsname(c){ return (DS && DS.names && DS.names[c]!=null) ? DS.names[c] : c; }

// ---- Label-Error Radar: audit accepted labels with the model ----
function radarQuery(){ const m=(assistModel==="__locate__")?(assist.default||""):assistModel; return "model="+encodeURIComponent(m||"")+"&conf="+conf+"&epoch="+((DS&&DS.epoch)||0); }
function openRadar(){ $("#radar").classList.add("show"); }
function showRadar(){ if(radarLast){ renderRadarDeck(radarLast); openRadar(); } else runRadar(); }   // reopen the last scan; Rescan inside the dialog refreshes it
function closeRadar(){ $("#radar").classList.remove("show"); }
async function runRadar(){
  if(!assist || !assist.available) return;
  if(DS && DS.task==="obb"){ banner("Radar compares box predictions - it is disabled for oriented-box projects."); return; }
  if(dirty && idx>=0 && !(await save())){ banner("Save the current image first, then run Radar."); return; }
  if(dirty){ banner("You edited while saving - press → to save before running Radar."); return; }   // in-flight edits: don't audit stale on-disk labels
  const myEpoch = DS && DS.epoch;   // project-scoped: image navigation must not abort the dataset-wide scan
  openRadar();
  $("#radarbody").innerHTML = `<div class="iload">Auditing your accepted labels with your model…<div class="ptrack" style="width:240px;margin:14px auto 0"><div class="pbar" id="rbar"></div></div></div>`;
  radarDeck = []; radarLast = null;
  try{
    const r = await fetch(`/api/assist/radar?${radarQuery()}`, {method:"POST"});
    if(!r.ok){ const e=await r.json().catch(()=>({})); $("#radarbody").innerHTML=`<div class="iload">Radar unavailable: ${esc(e.error||r.status)}</div>`; return; }
    const reader=r.body.getReader(), dec=new TextDecoder(); let buf="";
    for(;;){ const {value,done}=await reader.read(); if(done) break;
      buf += dec.decode(value,{stream:true}); let nl;
      while((nl=buf.indexOf("\n"))>=0){ const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1);
        if(!line) continue; let o; try{o=JSON.parse(line);}catch(e){ continue; }
        if(o.type==="progress"){ const bar=$("#rbar"); if(bar) bar.style.width=Math.round(100*o.i/Math.max(1,o.total))+"%"; }
        else if(o.type==="done"){ if(!DS || DS.epoch!==myEpoch) return; radarDeck=o.deck||[]; radarLast={...o, at:Date.now()}; renderRadarDeck(radarLast); }
        else if(o.type==="error"){ $("#radarbody").innerHTML=`<div class="iload">Radar failed: ${esc(o.error)}</div>`; }
      }
    }
  }catch(e){ $("#radarbody").innerHTML=`<div class="iload">Radar failed.</div>`; }
}
function renderRadarDeck(o){
  const deck=o.deck||[];
  const when=new Date(o.at||Date.now()).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
  const last=`<div class="rlast"><span>Results from the scan at ${when} - edits made since are not reflected.</span><button class="ifix" id="rrescan">Rescan</button></div>`;
  const show=html=>{ $("#radarbody").innerHTML=last+html; $("#rrescan").onclick=()=>runRadar(); };
  if(!o.scanned){ show(`<div class="iok">${ICO_CHECK}No labelled images to audit yet - accept some boxes, then run Radar to check them against your model.</div>`); return; }
  if(!deck.length){ show(`<div class="iok">${ICO_CHECK}No disagreements - your model agrees with your labels across ${o.scanned} labelled image${o.scanned===1?"":"s"}.</div>`); return; }
  const totalIssues=deck.reduce((a,d)=>a+d.issues,0);
  const order={class:0,miss:1,phantom:2};
  const sum=`<div class="rsummary"><div class="iwarn">${deck.length} image${deck.length>1?"s":""} · ${totalIssues} likely issue${totalIssues>1?"s":""} - worst first. Click one to review; nothing changes until you edit by hand.</div></div>`;
  const maxS = deck[0] ? (deck[0].score||1) : 1;
  const rows=deck.slice(0,80).map(d=>{
    const badges=Object.entries(d.counts||{}).sort((a,b)=>(order[a[0]]||9)-(order[b[0]]||9))
      .map(([t,c])=>`<span class="rbadge ${t}">${c} ${t==="class"?"class slip":t}</span>`).join("");
    const ratio=Math.max(0.1,(d.score||0)/maxS);
    const sevCol=ratio>0.66?"var(--danger)":ratio>0.33?"var(--warn)":"var(--ai)";
    return `<button class="rrow${d.id===idx?" cur":""}" data-id="${d.id}"><img class="rthumb" loading="lazy" src="/api/thumb/${d.id}?e=${(DS&&DS.epoch)||0}"><span class="rmeta"><span class="rfn">${esc(d.name)}</span><span class="rb">${badges}</span></span><span class="rsev"><span style="width:${Math.round(ratio*100)}%;background:${sevCol}"></span></span><span class="rscore">${d.score}</span></button>`;
  }).join("");
  show(sum+rows);
  const cur=$("#radarbody .rrow.cur"); if(cur) cur.scrollIntoView({block:"center"});
  document.querySelectorAll("#radarbody .rrow").forEach(b=> b.onclick=()=>{ closeRadar(); reviewFinding(+b.dataset.id); });
}
async function reviewFinding(id){ await load(id); if(idx===id) await loadRadarFindings(id); }   // only overlay if load() actually landed on this image
async function loadRadarFindings(id){
  const myGen=loadSeq;
  try{ const d=await jget(`/api/assist/radar/${id}`);
    if(myGen!==loadSeq) return;            // navigated away mid-fetch -> drop stale findings
    radarFindings=d.findings||[]; draw();
    if(radarFindings.length) banner(`Radar flagged ${radarFindings.length} here - fix by hand, then save. N: next flagged · Esc: clear.`);
  }catch(e){ if(myGen===loadSeq) radarFindings=[]; }
}
// Step through the deck in its worst-first order (not by image id), wrapping at the end;
// from an image outside the deck, start at the top.
function radarNextId(deck, cur){ if(!deck.length) return null; const i=deck.findIndex(d=>d.id===cur); return deck[(i+1)%deck.length].id; }
function nextFlagged(){ const nxt=radarNextId(radarDeck, idx); if(nxt!=null) reviewFinding(nxt); }
function drawRadarFindings(){
  if(!radarFindings.length || !imgOk) return;
  const iw=img.naturalWidth, ih=img.naturalHeight;
  ctx.save(); ctx.lineWidth=2.5; ctx.font="600 11.5px ui-sans-serif,system-ui,sans-serif"; ctx.textBaseline="bottom";
  radarFindings.forEach(f=>{
    const col = f.type==="class"? getCss("--warn") : f.type==="miss"? getCss("--ai") : getCss("--danger");
    const b=f.box, x=sx((b[0]-b[2]/2)*iw), y=sy((b[1]-b[3]/2)*ih), w=b[2]*iw*view.scale, h=b[3]*ih*view.scale;
    ctx.save(); ctx.setLineDash([7,4]); ctx.strokeStyle=col; ctx.shadowColor=col; ctx.shadowBlur=8; ctx.strokeRect(x,y,w,h); ctx.restore();
    const lab = f.type==="class"? `you: ${esc(String(dsname(f.label_cls)))} · model: ${esc(String(f.pred_name||dsname(f.pred_cls)))}`
              : f.type==="miss"? `missed? ${esc(String(f.pred_name||dsname(f.pred_cls)))} ${Math.round((f.conf||0)*100)}%`
              : `no object here? (${esc(String(dsname(f.label_cls)))})`;
    const tw=ctx.measureText(lab).width+12;
    ctx.save(); ctx.shadowColor="rgba(0,0,0,.4)"; ctx.shadowBlur=5; ctx.shadowOffsetY=1; ctx.globalAlpha=.95; ctx.fillStyle=col; rr(x,y-17,tw,16,4); ctx.restore();
    ctx.fillStyle="#0a0b0e"; ctx.fillText(lab,x+6,y-3);
  });
  ctx.restore();
}

// ---- Magnetic edges + one-key Tighten (T): snap box edges to image gradients ----
function ensureGrad(){
  if(gradData || !imgOk) return gradData;
  try{
    const iw=img.naturalWidth, ih=img.naturalHeight, cap=1280;
    const s=Math.min(1, cap/Math.max(iw,ih));
    const w=Math.max(1,Math.round(iw*s)), h=Math.max(1,Math.round(ih*s));
    const oc=document.createElement("canvas"); oc.width=w; oc.height=h;
    const octx=oc.getContext("2d",{willReadFrequently:true});
    octx.drawImage(img,0,0,w,h);
    const d=octx.getImageData(0,0,w,h).data, lum=new Float32Array(w*h);
    for(let i=0,p=0;i<w*h;i++,p+=4) lum[i]=0.299*d[p]+0.587*d[p+1]+0.114*d[p+2];
    const g=new Float32Array(w*h);
    for(let y=1;y<h-1;y++) for(let x=1;x<w-1;x++){ const i=y*w+x; g[i]=Math.abs(lum[i+1]-lum[i-1])+Math.abs(lum[i+w]-lum[i-w]); }
    gradData=g; gradW=w; gradH=h; gradScale=s;
  }catch(e){ gradData=null; }   // cross-origin / oversize -> graceful no-op
  return gradData;
}
function snapBox(b, m){
  const g=ensureGrad(); if(!g) return false;
  const s=gradScale, W=gradW, H=gradH;
  normalizeRect(b); clipToImage(b);
  let x1=Math.round(b.x*s), y1=Math.round(b.y*s), x2=Math.round((b.x+b.w)*s), y2=Math.round((b.y+b.h)*s);
  x1=Math.max(1,Math.min(W-2,x1)); x2=Math.max(1,Math.min(W-2,x2));
  y1=Math.max(1,Math.min(H-2,y1)); y2=Math.max(1,Math.min(H-2,y2));
  if(x2-x1<3 || y2-y1<3) return false;
  m=Math.max(2, Math.min(m, Math.floor(Math.min(x2-x1,y2-y1)/2)));
  const colSum=(x,a,bb)=>{ let v=0; for(let y=a;y<=bb;y++) v+=g[y*W+x]; return v; };
  const rowSum=(y,a,bb)=>{ let v=0; for(let x=a;x<=bb;x++) v+=g[y*W+x]; return v; };
  const bestCol=(cx,a,bb)=>{ let bx=cx,bv=-1; for(let x=Math.max(1,cx-m);x<=Math.min(W-2,cx+m);x++){ const v=colSum(x,a,bb); if(v>bv){bv=v;bx=x;} } return bx; };
  const bestRow=(cy,a,bb)=>{ let by=cy,bv=-1; for(let y=Math.max(1,cy-m);y<=Math.min(H-2,cy+m);y++){ const v=rowSum(y,a,bb); if(v>bv){bv=v;by=y;} } return by; };
  const nx1=bestCol(x1,y1,y2), nx2=bestCol(x2,y1,y2), ny1=bestRow(y1,x1,x2), ny2=bestRow(y2,x1,x2);
  const X1=Math.min(nx1,nx2)/s, X2=Math.max(nx1,nx2)/s, Y1=Math.min(ny1,ny2)/s, Y2=Math.max(ny1,ny2)/s;
  if(X2-X1<2 || Y2-Y1<2) return false;
  b.x=X1; b.y=Y1; b.w=X2-X1; b.h=Y2-Y1; return true;
}
function tightenSelected(){
  if(sel<0 || !imgOk) return;
  if(!editable || (DS && !DS.writable)) return;   // view-only: never reshape a protected box
  if(!ensureGrad()){ banner("Tighten needs the image pixels (unavailable for this image)."); return; }
  const b=boxes[sel], m=Math.max(10, Math.round(Math.min(b.w,b.h)*gradScale*0.22));
  gestureSnap=snap();
  if(snapBox(b,m)){ snapCommit(); markDirty(); draw(); banner("Tightened to the nearest edges (T)"); }
  else gestureSnap=null;
}

// ---- Loupe magnifier (L pins it; auto-shows while drawing/resizing) ----
function drawLoupe(){
  if(!imgOk || !cursor) return;
  if(!loupeOn) return;   // opt-in only (press L); no auto-zoom-on-background while drawing
  const R=64, zoom=3.4, pad=22;
  let lx=cursor.x+pad+R, ly=cursor.y+pad+R;
  if(lx+R>VW) lx=cursor.x-pad-R;
  if(ly+R>VH) ly=cursor.y-pad-R;
  if(lx-R<0) lx=cursor.x+pad+R;
  if(ly-R<0) ly=cursor.y+pad+R;
  lx=Math.max(R,Math.min(VW-R,lx)); ly=Math.max(R,Math.min(VH-R,ly));   // never clip off the stage
  const bx=ix(cursor.x), by=iy(cursor.y), eff=view.scale*zoom;
  ctx.save();
  ctx.beginPath(); ctx.arc(lx,ly,R,0,6.2832); ctx.closePath();
  ctx.fillStyle=getCss("--bg2"); ctx.fill();
  ctx.save(); ctx.clip();
  ctx.imageSmoothingEnabled=false;
  ctx.drawImage(img, lx-bx*eff, ly-by*eff, img.naturalWidth*eff, img.naturalHeight*eff);
  ctx.imageSmoothingEnabled=true;
  ctx.strokeStyle="rgba(6,182,212,.85)"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(lx-R,ly); ctx.lineTo(lx+R,ly); ctx.moveTo(lx,ly-R); ctx.lineTo(lx,ly+R); ctx.stroke();
  ctx.restore();
  ctx.lineWidth=2.5; ctx.strokeStyle=getCss("--ac"); ctx.shadowColor="rgba(2,6,23,.5)"; ctx.shadowBlur=10;
  ctx.beginPath(); ctx.arc(lx,ly,R,0,6.2832); ctx.stroke();
  ctx.restore();
}

// ---- Boost: train-in-the-loop (background fine-tune + agreement delta) ----
async function runBoost(){
  if(!assist || !assist.boost) return;
  if(dirty && idx>=0 && !(await save())){
    banner("Save the current image first - Boost trains on your accepted labels."); return;
  }
  if(dirty){ banner("You edited while saving - press → to save before boosting."); return; }   // in-flight edits: don't train on stale labels
  setBoostChip("run", "Boosting - warming up…");
  try{
    const r=await fetch("/api/boost",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});
    const d=await r.json();
    if(!d.started){ setBoostChip("bad", d.reason||"Couldn't start Boost"); return; }
    pollBoost();
  }catch(e){ setBoostChip("bad","Boost failed to start"); }
}
function pollBoost(){
  clearTimeout(boostTimer);
  // Guard on the PROJECT generation (epoch), not loadSeq -- loadSeq bumps on every
  // image navigation, which would kill the poll the moment the user moves images
  // mid-boost. The epoch only changes on a project switch.
  const myEpoch = DS && DS.epoch;
  const live = ()=> DS && DS.epoch===myEpoch;
  boostTimer=setTimeout(async ()=>{
    if(!live()) return;        // project switched -> stop polling the old boost
    let s; try{ s=await jget("/api/boost/status"); }catch(e){ if(live()) pollBoost(); return; }
    if(!live()) return;
    if(s.state==="running"){ setBoostChip("run", s.phase||"Boosting…"); pollBoost(); }
    else if(s.state==="done"){
      const a=Math.round((s.boosted_agreement||0)*100), b=Math.round((s.base_agreement||0)*100), dl=Math.round((s.delta||0)*100);
      setBoostChip(dl<0?"bad":"good", `model agrees ${a}% (was ${b}% · ${dl>=0?"+":""}${dl}pts)`);
      if(s.usable){ addBoostedOption(true); banner("Boosted model ready & selected - press R to auto-label with it, or use Auto-label all."); }
    } else if(s.state==="error"){ setBoostChip("bad", s.error||"Boost failed"); }
  }, 1600);
}
function setBoostChip(cls, txt){
  const c=$("#boostchip"); if(!c) return;
  const ic = cls==="run"? ICO_SPIN : cls==="good"? ICO_CHECK : "";
  c.className="chip show "+(cls||"");
  c.innerHTML=`${ic}<span>${esc(txt)}</span><span class="x" id="boostx" title="dismiss">&times;</span>`;
  const x=$("#boostx"); if(x) x.onclick=()=>{ c.className="chip"; };
}
function addBoostedOption(select){
  const sel=$("#amodel"); if(!sel) return;
  if(!sel.querySelector('option[value="boosted"]')){
    const o=document.createElement("option"); o.value="boosted"; o.textContent="✦ boosted (your labels)";
    sel.insertBefore(o, sel.firstChild);
  }
  if(select){ sel.value="boosted"; assistModel="boosted"; updateEngineUI(); }
}

// ---- Embedding map: the whole dataset as a 2-D scatter you can lasso ----
function openMap(){ if(!assist || !assist.embed) return; $("#mapmodal").classList.add("show"); resizeMapCanvas(); if(mapPoints.length){ drawMap(); } else { runEmbeddings(); } }
function closeMap(){ $("#mapmodal").classList.remove("show"); mapLasso=null; }
function resizeMapCanvas(){
  const cv2=$("#mapcv"); if(!cv2) return;
  const r=cv2.getBoundingClientRect(), dpr=Math.min(window.devicePixelRatio||1,2);
  cv2.width=Math.max(1,Math.round(r.width*dpr)); cv2.height=Math.max(1,Math.round(r.height*dpr));
  cv2.getContext("2d").setTransform(dpr,0,0,dpr,0,0);
  if(mapPoints.length) drawMap();
}
async function runEmbeddings(){
  const myEpoch = DS && DS.epoch;   // project-scoped: image navigation must not abort the dataset-wide embed
  const hint=$("#maphint"); hint.textContent="Embedding your images… (first run loads a tiny model)";
  try{
    const r=await fetch("/api/embeddings?epoch="+((DS&&DS.epoch)||0),{method:"POST"});
    if(!r.ok){ const e=await r.json().catch(()=>({})); hint.textContent="Embedding unavailable: "+(e.error||r.status); return; }
    const reader=r.body.getReader(), dec=new TextDecoder(); let buf="";
    for(;;){ const {value,done}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true}); let nl;
      while((nl=buf.indexOf("\n"))>=0){ const line=buf.slice(0,nl).trim(); buf=buf.slice(nl+1);
        if(!line) continue; let o; try{o=JSON.parse(line);}catch(e){ continue; }
        if(o.type==="progress"){ hint.textContent=`Embedding ${o.i} / ${o.total}…`; }
        else if(o.type==="done"){ if(!DS || DS.epoch!==myEpoch) return; mapPoints=o.points||[]; fitMap(); drawMap(); hint.textContent=mapPoints.length? "Drag a lasso around a region → jump to those images" : "No images to embed."; }
        else if(o.type==="error"){ hint.textContent="Embedding failed: "+o.error; }
      }
    }
  }catch(e){ hint.textContent="Embedding failed."; }
}
function fitMap(){
  if(!mapPoints.length){ mapFit=null; return; }
  let a=1e18,b=-1e18,c=1e18,d=-1e18;
  mapPoints.forEach(p=>{ if(p.x<a)a=p.x; if(p.x>b)b=p.x; if(p.y<c)c=p.y; if(p.y>d)d=p.y; });
  mapFit={minx:a,maxx:b,miny:c,maxy:d};
}
function mapProject(p, W, H){
  const pad=34, f=mapFit||{minx:0,maxx:1,miny:0,maxy:1};
  const dx=(f.maxx-f.minx)||1, dy=(f.maxy-f.miny)||1;
  return { x: pad + (p.x-f.minx)/dx*(W-2*pad), y: pad + (p.y-f.miny)/dy*(H-2*pad) };
}
function drawMap(){
  const cv2=$("#mapcv"); if(!cv2) return; const c2=cv2.getContext("2d");
  const r=cv2.getBoundingClientRect(); c2.clearRect(0,0,r.width,r.height);
  mapPoints.forEach(p=>{
    const im=IMAGES[p.id], st=im?im.status:"unlabeled";
    if(st==="deleted") return;   // skip points for quarantined images
    const col = st==="labeled"? "#10b981" : st==="suggested"? getCss("--ai") : "#64748b";
    const q=mapProject(p, r.width, r.height);
    c2.beginPath(); c2.arc(q.x,q.y,p.id===idx?5:3.2,0,6.2832);
    c2.fillStyle=col; c2.globalAlpha=p.id===idx?1:.85; c2.fill();
    if(p.id===idx){ c2.globalAlpha=1; c2.lineWidth=2; c2.strokeStyle=getCss("--ac"); c2.stroke(); }
  });
  c2.globalAlpha=1;
  if(mapLasso && mapLasso.length>3){
    c2.beginPath(); c2.moveTo(mapLasso[0],mapLasso[1]);
    for(let i=2;i<mapLasso.length;i+=2) c2.lineTo(mapLasso[i],mapLasso[i+1]);
    c2.closePath(); c2.fillStyle="rgba(6,182,212,.12)"; c2.fill();
    c2.lineWidth=1.5; c2.setLineDash([5,4]); c2.strokeStyle=getCss("--ac"); c2.stroke(); c2.setLineDash([]);
  }
}
function nearestMapPoint(x,y){
  const r=$("#mapcv").getBoundingClientRect(); let best=null,bd=225;
  mapPoints.forEach(p=>{ if(IMAGES[p.id] && IMAGES[p.id].status==="deleted") return;
    const q=mapProject(p,r.width,r.height); const dd=(q.x-x)*(q.x-x)+(q.y-y)*(q.y-y); if(dd<bd){bd=dd;best=p.id;} });
  return best;
}
function wireMap(){
  const cv2=$("#mapcv"); if(!cv2) return; let drawing=false;
  cv2.addEventListener("pointerdown", e=>{ if(!mapPoints.length) return; cv2.setPointerCapture(e.pointerId); drawing=true; mapLasso=[e.offsetX,e.offsetY]; });
  cv2.addEventListener("pointermove", e=>{ if(!drawing || !mapLasso) return; const L=mapLasso; if(Math.hypot(e.offsetX-L[L.length-2], e.offsetY-L[L.length-1])>3){ L.push(e.offsetX,e.offsetY); drawMap(); } });
  cv2.addEventListener("pointerup", e=>{ if(!drawing) return; drawing=false; const L=mapLasso; mapLasso=null;
    if(!L || L.length<8){ const hit=nearestMapPoint(e.offsetX,e.offsetY); drawMap(); if(hit!=null){ closeMap(); load(hit); } return; }
    const r=cv2.getBoundingClientRect();
    const ids=mapPoints.filter(p=>{ if(IMAGES[p.id] && IMAGES[p.id].status==="deleted") return false;
      const q=mapProject(p,r.width,r.height); return pointInPoly(q.x,q.y,L); }).map(p=>p.id);
    drawMap(); mapSelect(ids);
  });
}
function mapSelect(ids){
  if(!ids.length){ $("#maphint").textContent="Nothing in the lasso - try again."; return; }
  if(ids.length===1){ closeMap(); load(ids[0]); return; }
  selSet=new Set(ids); listFilter="all";
  document.querySelectorAll("#filter button").forEach(x=>x.classList.toggle("on", x.dataset.f==="all"));
  renderList(); closeMap();
  banner(`${ids.length} images selected from the map - sidebar filtered to them. Click a filter tab to clear.`);
}

// ===== Project home: list projects, open/switch, reset per-project state =====
"""

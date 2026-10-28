"""JS: canvas rendering, pointer interactions (draw/move/resize/OBB), keyboard."""

PART = r"""// ---- drawing ----
function draw(){
  ctx.clearRect(0,0,VW,VH);
  if(!imgOk){
    if(stageMsg){ ctx.fillStyle="#6a7280"; ctx.font="15px system-ui,sans-serif";
      ctx.textAlign="center"; ctx.textBaseline="middle";
      ctx.fillText(stageMsg, VW/2, VH/2); ctx.textAlign="left"; ctx.textBaseline="bottom"; }
    return;
  }
  ctx.drawImage(img, view.ox, view.oy, img.naturalWidth*view.scale, img.naturalHeight*view.scale);
  ctx.fillStyle="rgba(2,6,23,.08)";  // subtle scrim so annotations sit above the photo
  ctx.fillRect(view.ox, view.oy, img.naturalWidth*view.scale, img.naturalHeight*view.scale);
  ctx.lineWidth = 2; ctx.font = "600 11.5px ui-sans-serif,system-ui,sans-serif"; ctx.textBaseline="bottom";
  // AI ghost suggestions (dashed/translucent, under real boxes)
  ghosts.forEach(g=>{
    const c = g.mapped ? color(g.cls) : "#9aa3b2";
    const x=sx(g.x), y=sy(g.y), w=g.w*view.scale, h=g.h*view.scale;
    const a = 0.5 + 0.45*Math.min(1, g.conf||0);   // higher confidence -> more solid
    ctx.save();
    ctx.fillStyle = g.mapped ? "rgba(34,211,238,.10)" : "rgba(154,163,178,.10)"; ctx.fillRect(x,y,w,h);
    ctx.setLineDash([6,4]); ctx.globalAlpha=a; ctx.lineWidth=2; ctx.strokeStyle=c; ctx.strokeRect(x,y,w,h);
    ctx.globalAlpha=1; ctx.setLineDash([]);
    const lab = `${g.name}${g.mapped?"":" ?"} ${Math.round((g.conf||0)*100)}%`;
    const tw=ctx.measureText(lab).width+12; const tx=Math.min(x,x+w), ty=Math.min(y,y+h);
    ctx.shadowColor="rgba(0,0,0,.4)"; ctx.shadowBlur=5; ctx.shadowOffsetY=1;
    ctx.globalAlpha=.92; ctx.fillStyle=c; rr(tx,ty-17,tw,16,4); ctx.globalAlpha=1;
    ctx.shadowColor="transparent";
    ctx.fillStyle="#0a0b0e"; ctx.fillText(lab,tx+6,ty-3);
    ctx.restore();
  });
  boxes.forEach((b,i)=>{
    const c = color(b.cls);
    const x=sx(b.x), y=sy(b.y), w=b.w*view.scale, h=b.h*view.scale;
    const isSel = (i===sel) || selBoxes.has(i);
    const dim = (selPoly>=0) || ((sel>=0 || selBoxes.size>0) && !isSel);   // dim others when anything is selected
    const isHover = i===hover && mode===null;
    ctx.save();
    ctx.globalAlpha = dim ? 0.42 : 1;
    ctx.fillStyle = colorA(b.cls, isSel?0.22:0.13); ctx.fillRect(x,y,w,h);
    if(isSel||isHover){ ctx.shadowColor=c; ctx.shadowBlur=isSel?10:5; }
    ctx.strokeStyle = c; ctx.lineWidth = isSel?2.6:(isHover?2.2:1.8); ctx.strokeRect(x,y,w,h);
    ctx.shadowColor="transparent"; ctx.shadowBlur=0;
    const nm = (DS.names&&DS.names[b.cls])!=null ? DS.names[b.cls] : b.cls;
    drawChip(Math.min(x,x+w), Math.min(y,y+h), String(nm), b.cls);
    ctx.restore();
  });
  polys.forEach((p,i)=>{
    const c = color(p.cls), pts = p.pts;
    const selP = i===selPoly, dim = (sel>=0) || (selPoly>=0 && !selP);
    ctx.save();
    ctx.globalAlpha = dim ? 0.42 : 1;
    ctx.beginPath();
    for(let k=0;k<pts.length;k+=2){ const X=sx(pts[k]), Y=sy(pts[k+1]); if(k===0) ctx.moveTo(X,Y); else ctx.lineTo(X,Y); }
    ctx.closePath();
    ctx.fillStyle = colorA(p.cls, selP?0.22:0.14); ctx.fill();
    if(selP){ ctx.shadowColor=c; ctx.shadowBlur=10; }
    ctx.lineWidth = selP?2.6:1.8; ctx.strokeStyle=c; ctx.stroke();
    ctx.shadowColor="transparent"; ctx.shadowBlur=0;
    if(selP){ ctx.fillStyle="#fff"; ctx.strokeStyle="#06b6d4"; ctx.lineWidth=1.2;
      for(let k=0;k<pts.length;k+=2){ const X=sx(pts[k]),Y=sy(pts[k+1]); ctx.beginPath(); ctx.arc(X,Y,3,0,6.2832); ctx.fill(); ctx.stroke(); } }
    const nm = (DS.names&&DS.names[p.cls])!=null ? DS.names[p.cls] : p.cls;
    let mnx=1e9,mny=1e9; for(let k=0;k<pts.length;k+=2){ if(pts[k]<mnx)mnx=pts[k]; if(pts[k+1]<mny)mny=pts[k+1]; }
    drawChip(sx(mnx), sy(mny), String(nm), p.cls);
    ctx.restore();
  });
  if(polyDraft && polyDraft.length){
    ctx.save();
    ctx.strokeStyle=color(active); ctx.lineWidth=2; ctx.setLineDash([5,4]);
    ctx.beginPath(); ctx.moveTo(sx(polyDraft[0]), sy(polyDraft[1]));
    for(let k=2;k<polyDraft.length;k+=2) ctx.lineTo(sx(polyDraft[k]), sy(polyDraft[k+1]));
    if(cursor) ctx.lineTo(cursor.x, cursor.y);
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle="#fff"; ctx.strokeStyle=color(active); ctx.lineWidth=1.4;
    for(let k=0;k<polyDraft.length;k+=2){ ctx.beginPath(); ctx.arc(sx(polyDraft[k]),sy(polyDraft[k+1]),4,0,6.2832); ctx.fill(); ctx.stroke(); }
    if(polyDraft.length>=6 && cursor && Math.hypot(sx(polyDraft[0])-cursor.x, sy(polyDraft[1])-cursor.y)<12){
      ctx.beginPath(); ctx.arc(sx(polyDraft[0]),sy(polyDraft[1]),7,0,6.2832); ctx.strokeStyle="#06b6d4"; ctx.lineWidth=2.5; ctx.stroke();
    }
    ctx.restore();
  }
  drawRadarFindings();
  if(sel>=0){ drawHandles(boxes[sel]); if(editable && !(DS && !DS.writable)) drawDelBadge(boxes[sel]); }
  if(selPoly>=0 && polys[selPoly] && editable && !(DS && !DS.writable)) drawRotHandle(polys[selPoly]);
  if(mode==="obbnew" && obbRect){
    const x=sx(Math.min(obbRect.x0,obbRect.x1)), y=sy(Math.min(obbRect.y0,obbRect.y1));
    const w=Math.abs(obbRect.x1-obbRect.x0)*view.scale, h=Math.abs(obbRect.y1-obbRect.y0)*view.scale;
    ctx.save(); ctx.setLineDash([5,4]); ctx.strokeStyle=color(active); ctx.lineWidth=2; ctx.strokeRect(x,y,w,h);
    ctx.fillStyle=colorA(active,.12); ctx.fillRect(x,y,w,h); ctx.restore();
  }
  if(cursor && (mode===null||mode==='new')){
    ctx.save(); ctx.strokeStyle='rgba(6,182,212,.4)'; ctx.lineWidth=1;
    ctx.beginPath();
    ctx.moveTo(cursor.x+0.5,0); ctx.lineTo(cursor.x+0.5,VH);
    ctx.moveTo(0,cursor.y+0.5); ctx.lineTo(VW,cursor.y+0.5);
    ctx.stroke(); ctx.restore();
  }
  if(mode==="segbox" && segRect){
    const x=sx(Math.min(segRect.x0,segRect.x1)), y=sy(Math.min(segRect.y0,segRect.y1));
    const w=Math.abs(segRect.x1-segRect.x0)*view.scale, h=Math.abs(segRect.y1-segRect.y0)*view.scale;
    ctx.save(); ctx.setLineDash([5,4]); ctx.strokeStyle="#22d3ee"; ctx.lineWidth=1.5; ctx.strokeRect(x,y,w,h);
    ctx.fillStyle="rgba(34,211,238,.10)"; ctx.fillRect(x,y,w,h); ctx.restore();
  }
  drawLoupe();
  updateProgress();
  renderRegions();
}
function handlePts(b){
  const x=b.x,y=b.y,w=b.w,h=b.h;
  return {nw:[x,y], n:[x+w/2,y], ne:[x+w,y], e:[x+w,y+h/2],
          se:[x+w,y+h], s:[x+w/2,y+h], sw:[x,y+h], w:[x,y+h/2]};
}
function drawHandles(b){
  const pts = handlePts(b);
  ctx.fillStyle = "#fff"; ctx.strokeStyle = "#06b6d4"; ctx.lineWidth=1.5;
  HANDLES.forEach(k=>{
    const [hx,hy]=pts[k]; const px=sx(hx), py=sy(hy);
    ctx.beginPath(); ctx.rect(px-4,py-4,8,8); ctx.fill(); ctx.stroke();
  });
}
// A red "x" badge just outside the top-right corner of the selected box -> click to delete.
function delBadgePos(b){ return {x: sx(Math.max(b.x,b.x+b.w))+11, y: sy(Math.min(b.y,b.y+b.h))-11}; }
function drawDelBadge(b){
  const p=delBadgePos(b);
  ctx.save();
  ctx.beginPath(); ctx.arc(p.x,p.y,9,0,6.2832);
  ctx.shadowColor="rgba(2,6,23,.5)"; ctx.shadowBlur=5; ctx.fillStyle="#ef4444"; ctx.fill();
  ctx.shadowColor="transparent"; ctx.strokeStyle="#fff"; ctx.lineWidth=1.8;
  ctx.beginPath();
  ctx.moveTo(p.x-3.4,p.y-3.4); ctx.lineTo(p.x+3.4,p.y+3.4);
  ctx.moveTo(p.x+3.4,p.y-3.4); ctx.lineTo(p.x-3.4,p.y+3.4); ctx.stroke();
  ctx.restore();
}
function hitHandle(b, mx, my){
  const pts = handlePts(b);
  for(const k of HANDLES){
    const [hx,hy]=pts[k];
    if(Math.abs(sx(hx)-mx)<=HR+2 && Math.abs(sy(hy)-my)<=HR+2) return k;
  }
  return null;
}
function hitBox(mx, my){
  const x=ix(mx), y=iy(my);
  for(let i=boxes.length-1;i>=0;i--){
    const b=boxes[i];
    const bx=Math.min(b.x,b.x+b.w), by=Math.min(b.y,b.y+b.h);
    if(x>=bx && x<=bx+Math.abs(b.w) && y>=by && y<=by+Math.abs(b.h)) return i;
  }
  return -1;
}
// ---- manual polygon creation (no SAM needed): click vertices, close to commit ----
function commitPolyDraft(){
  if(polyDraft && polyDraft.length>=6){   // >=3 points
    pushUndo(); polys.push({cls:active, pts:polyDraft.slice()});
    selPoly=polys.length-1; sel=-1; selBoxes.clear(); markDirty();
  }
  polyDraft=null; draw();
}
function cancelPolyDraft(){ polyDraft=null; draw(); }
// ---- oriented boxes: a 4-corner quad (stored as a 4-vertex polygon) + a rotate handle ----
function polyCentroid(pts){ let cx=0,cy=0; const n=pts.length/2||1; for(let k=0;k<pts.length;k+=2){cx+=pts[k];cy+=pts[k+1];} return [cx/n, cy/n]; }
function rotHandleScreen(p){
  let mnx=1e9,mxx=-1e9,mny=1e9;
  for(let k=0;k<p.pts.length;k+=2){ if(p.pts[k]<mnx)mnx=p.pts[k]; if(p.pts[k]>mxx)mxx=p.pts[k]; if(p.pts[k+1]<mny)mny=p.pts[k+1]; }
  return {x:sx((mnx+mxx)/2), y:sy(mny)-22};
}
function drawRotHandle(p){
  const h=rotHandleScreen(p);
  ctx.save();
  ctx.strokeStyle="#06b6d4"; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(h.x, h.y+6); ctx.lineTo(h.x, h.y+16); ctx.stroke();
  ctx.fillStyle="#fff"; ctx.beginPath(); ctx.arc(h.x,h.y,6,0,6.2832); ctx.fill(); ctx.stroke();
  ctx.strokeStyle="#06b6d4"; ctx.lineWidth=1.4; ctx.beginPath(); ctx.arc(h.x,h.y,3.2,0.5,5.6); ctx.stroke();
  ctx.restore();
}

// ---- mouse ----
let mode=null, drag=null, spaceDown=false;
cv.addEventListener("pointerdown", e=>{
  cv.setPointerCapture(e.pointerId);
  const mx=e.offsetX, my=e.offsetY;
  if(spaceDown || e.button===1){ mode="pan"; drag={mx,my,ox:view.ox,oy:view.oy}; return; }
  if(!imgOk) return;
  if(!editable || (DS && !DS.writable)){   // view-only: select to inspect, never mutate
    const hb=hitBox(mx,my);
    if(hb>=0){ sel=hb; selPoly=-1; active=boxes[hb].cls; markPalette(); draw(); return; }
    const hp=hitPoly(mx,my);
    if(hp>=0){ selPoly=hp; sel=-1; active=polys[hp].cls; markPalette(); draw(); return; }
    sel=-1; selPoly=-1; draw(); return;
  }
  if(tool==="poly"){   // manual polygon: each click drops a vertex; click the first to close
    if(!(DS.names||[]).length){ banner("Add a class before drawing - every annotation needs one."); openClassEdit(); return; }
    if(polyDraft && polyDraft.length>=6 && Math.hypot(sx(polyDraft[0])-mx, sy(polyDraft[1])-my)<12){ commitPolyDraft(); return; }
    if(!polyDraft) polyDraft=[];
    polyDraft.push(ix(mx), iy(my)); draw(); return;
  }
  if(sel>=0 && boxes[sel]){   // click the red x badge on the selected box to delete it
    const bp=delBadgePos(boxes[sel]);
    if(Math.hypot(mx-bp.x, my-bp.y)<=11){ pushUndo(); boxes.splice(sel,1); sel=-1; selBoxes.clear(); markDirty(); draw(); return; }
  }
  if(sel>=0){
    const k = hitHandle(boxes[sel], mx, my);
    if(k){ snapStart(); mode="resize"; drag={k, b:boxes[sel]}; return; }
  }
  if(selPoly>=0 && polys[selPoly]){   // grab the round handle above a selected polygon/OBB to rotate it
    const rh=rotHandleScreen(polys[selPoly]);
    if(Math.hypot(mx-rh.x, my-rh.y)<=9){
      snapStart(); mode="rotpoly";
      const c=polyCentroid(polys[selPoly].pts);
      drag={cx:c[0], cy:c[1], start:Math.atan2(iy(my)-c[1], ix(mx)-c[0]), pts:polys[selPoly].pts.slice()};
      return;
    }
  }
  if(selPoly>=0){
    const vi = hitVertex(polys[selPoly], mx, my);
    if(vi>=0){
      if(e.altKey){ if(!(DS&&DS.task==="obb") && polys[selPoly].pts.length>6){ pushUndo(); polys[selPoly].pts.splice(vi*2,2); markDirty(); draw(); } return; }
      snapStart(); mode="vertex"; drag={vi}; return;
    }
  }
  const hit = hitBox(mx,my);
  if(hit>=0){
    if(e.shiftKey){   // toggle this box in/out of the multi-selection (no move)
      selBoxes.has(hit) ? selBoxes.delete(hit) : selBoxes.add(hit);
      sel = selBoxes.size===1 ? [...selBoxes][0] : -1; selPoly=-1;
      if(selBoxes.size) banner(`${selBoxes.size} box${selBoxes.size>1?"es":""} selected`);
      draw(); return;
    }
    snapStart(); selBoxes = new Set([hit]); sel=hit; selPoly=-1; mode="move";
    drag={mx, my, x:boxes[hit].x, y:boxes[hit].y};
    setActive(boxes[hit].cls); draw(); return;
  }
  const ph = hitPoly(mx,my);
  if(ph>=0){
    snapStart(); selBoxes.clear(); selPoly=ph; sel=-1; mode="movepoly";
    drag={mx, my, pts:polys[ph].pts.slice()};
    setActive(polys[ph].cls); draw(); return;
  }
  if(ghosts.length){
    const g = hitGhost(mx,my);
    if(g>=0){ if(e.altKey) rejectGhost(g); else acceptGhost(g); mode=null; drag=null; return; }
  }
  if(!editable || (DS && !DS.writable)) return;
  if(!(DS.names||[]).length){ banner("Add a class before drawing - every box needs one."); openClassEdit(); return; }
  if(tool==="seg"){ mode="segbox"; drag={mx, my, x0:ix(mx), y0:iy(my)}; segRect=null; return; }
  if(tool==="obb"){ snapStart(); mode="obbnew"; drag={}; obbRect={x0:ix(mx), y0:iy(my), x1:ix(mx), y1:iy(my)}; draw(); return; }
  if(tool!=="box") return;   // poly/seg/obb were handled above; only the box tool free-draws on empty
  const x=ix(mx), y=iy(my);
  snapStart();
  boxes.push({cls:active, x, y, w:0, h:0});
  selBoxes.clear(); sel=boxes.length-1; selPoly=-1; mode="new"; drag={};
  draw();
});
cv.addEventListener("pointermove", e=>{
  const mx=e.offsetX, my=e.offsetY;
  cursor={x:mx,y:my};
  if(mode==="pan"){ view.ox=drag.ox+(mx-drag.mx); view.oy=drag.oy+(my-drag.my); draw(); return; }
  if(mode==="new"){ const b=boxes[sel]; b.w=ix(mx)-b.x; b.h=iy(my)-b.y; draw(); return; }
  if(mode==="move"){ const dx=ix(mx)-ix(drag.mx), dy=iy(my)-iy(drag.my);
    boxes[sel].x=drag.x+dx; boxes[sel].y=drag.y+dy; markDirty(); draw(); return; }
  if(mode==="resize"){ resizeBox(drag.b, drag.k, ix(mx), iy(my)); markDirty(); draw(); return; }
  if(mode==="movepoly"){ const dx=ix(mx)-ix(drag.mx), dy=iy(my)-iy(drag.my); const p=polys[selPoly];
    for(let k=0;k<p.pts.length;k+=2){ p.pts[k]=drag.pts[k]+dx; p.pts[k+1]=drag.pts[k+1]+dy; } markDirty(); draw(); return; }
  if(mode==="vertex"){
    const p=polys[selPoly];
    if(DS && DS.task==="obb" && p.pts.length===8){
      // Oriented boxes must STAY rectangles (the 9-field contract): resize along
      // the rect's own axes from the fixed opposite corner instead of free-dragging.
      const vi=drag.vi, o=(vi+2)%4, a=(vi+1)%4, b=(vi+3)%4;
      const ox=p.pts[o*2], oy=p.pts[o*2+1];
      let ux=p.pts[a*2]-ox, uy=p.pts[a*2+1]-oy;
      let vx=p.pts[b*2]-ox, vy=p.pts[b*2+1]-oy;
      const ul=Math.hypot(ux,uy), vl=Math.hypot(vx,vy);
      if(ul>1e-6 && vl>1e-6){
        ux/=ul; uy/=ul; vx/=vl; vy/=vl;
        const dx=ix(mx)-ox, dy=iy(my)-oy;
        const su=dx*ux+dy*uy, sv=dx*vx+dy*vy;   // drag point in the rect's local frame
        p.pts[a*2]=ox+ux*su; p.pts[a*2+1]=oy+uy*su;
        p.pts[b*2]=ox+vx*sv; p.pts[b*2+1]=oy+vy*sv;
        p.pts[vi*2]=ox+ux*su+vx*sv; p.pts[vi*2+1]=oy+uy*su+vy*sv;
      }
      markDirty(); draw(); return;
    }
    p.pts[drag.vi*2]=ix(mx); p.pts[drag.vi*2+1]=iy(my); markDirty(); draw(); return;
  }
  if(mode==="segbox"){ segRect={x0:drag.x0, y0:drag.y0, x1:ix(mx), y1:iy(my)}; draw(); return; }
  if(mode==="obbnew"){ obbRect.x1=ix(mx); obbRect.y1=iy(my); draw(); return; }
  if(mode==="rotpoly"){
    let ang=Math.atan2(iy(my)-drag.cy, ix(mx)-drag.cx) - drag.start;
    if(e.shiftKey) ang=Math.round(ang/(Math.PI/12))*(Math.PI/12);   // snap to 15°
    const ca=Math.cos(ang), sa=Math.sin(ang), p=polys[selPoly];
    for(let k=0;k<drag.pts.length;k+=2){ const dx=drag.pts[k]-drag.cx, dy=drag.pts[k+1]-drag.cy;
      p.pts[k]=drag.cx+dx*ca-dy*sa; p.pts[k+1]=drag.cy+dx*sa+dy*ca; }
    markDirty(); draw(); return;
  }
  let hb=-1;
  if(imgOk && sel>=0 && hitHandle(boxes[sel],mx,my)){ cv.style.cursor="pointer"; }
  else { hb = imgOk?hitBox(mx,my):-1; cv.style.cursor = spaceDown?"grab":(hb>=0?"move":"crosshair"); }
  hover = hb;
  draw();
});
cv.addEventListener("pointerleave", ()=>{ cursor=null; hover=-1; draw(); });
cv.addEventListener("pointerup", e=>{
  if(mode==="segbox"){
    const dmx=drag.mx, dmy=drag.my, r=segRect; segRect=null; mode=null; drag=null; draw();
    if((Math.abs(e.offsetX-dmx)>5 || Math.abs(e.offsetY-dmy)>5) && r) segmentBox(r); else segmentAt(dmx, dmy);
    return;
  }
  if(mode==="new"){
    const b=boxes[sel];
    if(Math.abs(b.w)*view.scale<6 || Math.abs(b.h)*view.scale<6){ boxes.pop(); sel=-1; }   // ignore tiny accidental drags (a click shouldn't spawn a box)
    else { normalizeRect(b); clipToImage(b); markDirty(); }  // WYSIWYG: keep the box exactly where drawn; press T to magnet-tighten on demand
  } else if(mode==="resize"){ normalizeRect(drag.b); clipToImage(drag.b); }
  else if(mode==="move"){ clipToImage(boxes[sel]); }
  else if(mode==="movepoly"){ clipPoly(polys[selPoly]); }
  else if(mode==="vertex"){ clipPoly(polys[selPoly]); }
  else if(mode==="obbnew"){
    const r=obbRect; obbRect=null;
    if(r){ const x1=Math.min(r.x0,r.x1), y1=Math.min(r.y0,r.y1), x2=Math.max(r.x0,r.x1), y2=Math.max(r.y0,r.y1);
      if((x2-x1)*view.scale>=6 && (y2-y1)*view.scale>=6){
        polys.push({cls:active, pts:[x1,y1, x2,y1, x2,y2, x1,y2]}); selPoly=polys.length-1; sel=-1; selBoxes.clear(); markDirty(); } }
  }
  else if(mode==="rotpoly"){ clipPoly(polys[selPoly]); }
  snapCommit();
  mode=null; drag=null; draw();
});
cv.addEventListener("dblclick", e=>{
  if(selPoly<0 || !imgOk) return;
  if(!editable || (DS && !DS.writable)) return;   // view-only: select to inspect, never insert a vertex
  if(DS && DS.task==="obb") return;   // oriented boxes stay 4-corner: no vertex insert
  const mx=e.offsetX, my=e.offsetY, p=polys[selPoly], X=ix(mx), Y=iy(my), n=p.pts.length/2;
  let best=-1, bestD=1e18, bx=0, by=0;
  for(let i=0;i<n;i++){ const a=2*i, b=2*((i+1)%n);
    const ax=p.pts[a], ay=p.pts[a+1], cx=p.pts[b], cy=p.pts[b+1];
    const dx=cx-ax, dy=cy-ay, L=dx*dx+dy*dy||1;
    let t=((X-ax)*dx+(Y-ay)*dy)/L; t=Math.max(0,Math.min(1,t));
    const px=ax+t*dx, py=ay+t*dy, d=(X-px)*(X-px)+(Y-py)*(Y-py);
    if(d<bestD){ bestD=d; best=i; bx=px; by=py; } }
  if(best>=0 && Math.hypot(sx(bx)-mx, sy(by)-my)<14){ pushUndo(); p.pts.splice(2*(best+1),0, bx, by); markDirty(); draw(); }
});
function clipPoly(p){ if(!imgOk||!p) return; const iw=img.naturalWidth, ih=img.naturalHeight;
  if(DS && DS.task==="obb" && p.pts.length===8){
    // Clamping corners individually would shear the rectangle; translate the whole
    // quad back inside instead (per-corner clamp only if it simply can't fit).
    let mnx=1e18,mxx=-1e18,mny=1e18,mxy=-1e18;
    for(let k=0;k<p.pts.length;k+=2){ mnx=Math.min(mnx,p.pts[k]); mxx=Math.max(mxx,p.pts[k]);
      mny=Math.min(mny,p.pts[k+1]); mxy=Math.max(mxy,p.pts[k+1]); }
    if(mxx-mnx<=iw && mxy-mny<=ih){
      const dx=Math.max(0,-mnx)-Math.max(0,mxx-iw), dy=Math.max(0,-mny)-Math.max(0,mxy-ih);
      if(dx||dy) for(let k=0;k<p.pts.length;k+=2){ p.pts[k]+=dx; p.pts[k+1]+=dy; }
      return;
    }
  }
  for(let k=0;k<p.pts.length;k+=2){ p.pts[k]=Math.max(0,Math.min(p.pts[k],iw)); p.pts[k+1]=Math.max(0,Math.min(p.pts[k+1],ih)); } }
function normalizeRect(b){ if(b.w<0){b.x+=b.w;b.w=-b.w;} if(b.h<0){b.y+=b.h;b.h=-b.h;} }
function clipToImage(b){
  if(!imgOk) return;
  const iw=img.naturalWidth, ih=img.naturalHeight;
  const x1=Math.max(0,Math.min(b.x,iw)),     y1=Math.max(0,Math.min(b.y,ih));
  const x2=Math.max(0,Math.min(b.x+b.w,iw)), y2=Math.max(0,Math.min(b.y+b.h,ih));
  b.x=x1; b.y=y1; b.w=x2-x1; b.h=y2-y1;
}
function resizeBox(b, k, mx, my){
  if(k.includes("n")){ b.h += b.y-my; b.y=my; }
  if(k.includes("s")){ b.h = my-b.y; }
  if(k.includes("w")){ b.w += b.x-mx; b.x=mx; }
  if(k.includes("e")){ b.w = mx-b.x; }
}
cv.addEventListener("wheel", e=>{
  e.preventDefault();
  const f = e.deltaY<0 ? 1.1 : 1/1.1;
  const mx=e.offsetX, my=e.offsetY, bx=ix(mx), by=iy(my);
  view.scale = Math.max(0.02, Math.min(64, view.scale * f));
  view.ox = mx - bx*view.scale; view.oy = my - by*view.scale;
  draw();
}, {passive:false});

// ---- keyboard ----
const KEY_DISPATCH = {   // actions for the remappable keys (tools self-gate by task)
  prev: e=>{ e.preventDefault(); step(-1); },
  next: e=>{ e.preventDefault(); step(1); },
  nextunlabeled: e=>{ e.preventDefault(); nextUnlabeled(e.shiftKey?-1:1); },
  carry: e=>{ e.preventDefault(); carryForward(); },
  prelabel: e=>{ e.preventDefault(); prelabelCurrent(); },
  toolbox: ()=> setTool("box"),
  toolpoly: ()=> setTool("poly"),
  toolobb: ()=> setTool("obb"),
  toolsam: ()=>{ if(assist && assist.sam) setTool("seg"); },
  tighten: e=>{ e.preventDefault(); tightenSelected(); },
  loupe: ()=>{ loupeOn=!loupeOn; draw(); },
  radar: e=>{ e.preventDefault(); showRadar(); },
  flagged: e=>{ e.preventDefault(); nextFlagged(); },
  map: e=>{ e.preventDefault(); openMap(); },
  fit: ()=>{ fit(); draw(); },
};
// ---- shortcuts modal: rendered from KEY_ACTIONS so rebinding stays in sync ----
let rebindTarget = null;   // action id currently capturing a new key
const HELP_LAYOUT = [
  ["Drawing & tools", [
    {label:"Draw a box (active class)", keys:["drag"]},
    {act:"toolbox"}, {act:"toolpoly"}, {act:"toolsam"}, {act:"toolobb"},
    {act:"tighten"}, {act:"loupe"},
  ]],
  ["Classes", [
    {label:"Set active class", keys:["1","9","0"]},
    {label:"Open class search", keys:["/"]},
  ]],
  ["Select & edit", [
    {label:"Select / move / resize", keys:["click"]},
    {label:"Add / remove from selection", keys:["Shift","click"]},
    {label:"Select all", keys:["Ctrl","A"]},
    {label:"Delete selected", keys:["Del"]},
    {label:"Undo / redo", keys:["Ctrl","Z","Ctrl","Y"]},
    {label:"Duplicate box", keys:["Ctrl","D"]},
  ]],
  ["Navigate", [
    {act:"prev"}, {act:"next"}, {act:"nextunlabeled"}, {act:"carry"},
    {label:"Submit & next unlabeled", keys:["Enter"]},
  ]],
  ["View", [
    {label:"Pan", keys:["Space","drag"]},
    {label:"Zoom", keys:["wheel","+","−"]},
    {act:"fit"},
  ]],
  ["AI assist", [
    {act:"prelabel"}, {act:"radar"}, {act:"flagged"}, {act:"map"},
  ]],
  ["Save", [
    {label:"Save", keys:["Ctrl","S"]},
    {label:"Cancel / clear / close", keys:["Esc"]},
  ]],
];
function renderHelp(msg, isErr){
  const grid=$("#kgrid"); if(!grid) return;
  grid.innerHTML = HELP_LAYOUT.map(([title, rows])=>{
    const body = rows.map(r=>{
      if(r.act){
        const a=KEY_ACTIONS.find(x=>x.id===r.act); if(!a) return "";
        const live = rebindTarget===a.id;
        const shown = live ? "…" : keyFor(a.id).toUpperCase();
        return `<div class="krow"><span>${esc(a.label)}</span><span class="keys">`+
          `<kbd class="k-edit${live?" k-live":""}" data-act="${a.id}" title="Click, then press a new key">${esc(shown)}</kbd></span></div>`;
      }
      return `<div class="krow"><span>${esc(r.label)}</span><span class="keys">${r.keys.map(k=>`<kbd>${esc(k)}</kbd>`).join("")}</span></div>`;
    }).join("");
    return `<div class="kgroup"><h4>${esc(title)}</h4>${body}</div>`;
  }).join("");
  grid.querySelectorAll(".k-edit").forEach(el=> el.onclick=()=>{
    rebindTarget = (rebindTarget===el.dataset.act) ? null : el.dataset.act;
    renderHelp(rebindTarget ? "Press a letter key (Esc cancels)." : "");
  });
  const h=$("#khint");
  if(h){ h.textContent = msg || "Click a key to change it. Arrow keys always step between images.";
         h.className = "khint" + (isErr ? " err" : ""); }
}
// Close the top-most open overlay; returns false when none was open. Keeping this
// in one place means Escape always works and no overlay can be left unclosable.
function closeTopOverlay(){
  if($("#wizard").classList.contains("show")){ closeWizard(); return true; }
  if($("#settingsmodal").classList.contains("show")){ closeSettings(); return true; }
  if($("#exportmodal").classList.contains("show")){ closeExport(); return true; }
  if($("#renamemodal").classList.contains("show")){ closeRename(); return true; }
  if($("#deletemodal").classList.contains("show")){ closeDelete(); return true; }
  if($("#classedit").classList.contains("show")){ closeClassEdit(); return true; }
  if($("#radar").classList.contains("show")){ closeRadar(); return true; }
  if($("#mapmodal").classList.contains("show")){ closeMap(); return true; }
  if($("#insights").classList.contains("show")){ closeInsights(); return true; }
  if($("#sharepop").classList.contains("show")){ $("#sharepop").classList.remove("show"); return true; }
  if($("#picker").classList.contains("show")){ closePicker(); return true; }
  if($("#help").style.display==="flex"){ $("#help").style.display="none"; return true; }
  return false;
}
function overlayOpen(){
  return !!document.querySelector(".modal.show") || $("#wizard").classList.contains("show")
    || $("#home").classList.contains("show") || $("#help").style.display==="flex";
}
window.addEventListener("keydown", e=>{
  if(rebindTarget){   // the ? modal is capturing a new key for an action
    e.preventDefault(); e.stopPropagation();
    const k=(e.key||"").toLowerCase();
    if(k==="escape"){ rebindTarget=null; renderHelp(); return; }
    if(k.length===1 && k>="a" && k<="z"){
      const clash = KEY_ACTIONS.find(a=>a.id!==rebindTarget && keyFor(a.id)===k);
      if(clash){ renderHelp(k.toUpperCase()+" is already used by “"+clash.label+"” - pick another key.", true); return; }
      setKey(rebindTarget, k); rebindTarget=null; renderHelp("Saved.");
    }
    return;
  }
  const t=(e.target&&e.target.tagName)||"";
  if(t==="INPUT"||t==="SELECT"||t==="TEXTAREA"){ if(e.key==="Escape"){ e.target.blur(); closePicker(); closeTopOverlay(); } return; }
  if(overlayOpen()){   // a dialog (or Home) is up: canvas hotkeys must not fire behind it
    if(e.key==="Escape") closeTopOverlay();
    return;
  }
  if(e.key===" "){ spaceDown=true; cv.style.cursor="grab"; e.preventDefault(); return; }
  if((e.ctrlKey||e.metaKey) && (e.key==="s"||e.key==="S")){ e.preventDefault(); save(); return; }
  if((e.ctrlKey||e.metaKey) && !e.shiftKey && (e.key==="z"||e.key==="Z")){ e.preventDefault(); applyHistory(undoStack, redoStack); return; }
  if((e.ctrlKey||e.metaKey) && ((e.key==="y"||e.key==="Y") || (e.shiftKey && (e.key==="z"||e.key==="Z")))){ e.preventDefault(); applyHistory(redoStack, undoStack); return; }
  if((e.ctrlKey||e.metaKey) && (e.key==="d"||e.key==="D")){ e.preventDefault(); duplicateSelected(); return; }
  if((e.ctrlKey||e.metaKey) && (e.key==="a"||e.key==="A")){ e.preventDefault(); selectAllBoxes(); return; }
  if(e.key==="Enter"){
    e.preventDefault();
    if(polyDraft){ commitPolyDraft(); return; }   // close the in-progress polygon
    if(ghosts.length){ const adv=e.shiftKey; acceptAllGhosts();
      if(adv){ save().then(()=> nextSuggested()); } return; }
    submitImage();   // Enter / Ctrl+Enter: save this image and advance to the next unlabeled one
    return;
  }
  if(e.key>="0" && e.key<="9"){ const i = e.key==="0"?9:(+e.key-1); setActive(i); return; }
  if(e.key==="/"){ e.preventDefault(); togglePicker(); return; }
  if(e.key==="ArrowRight"){ e.preventDefault(); step(1); return; }
  if(e.key==="ArrowLeft"){ e.preventDefault(); step(-1); return; }
  if(!e.ctrlKey && !e.metaKey && !e.altKey){   // remappable single-letter actions (see KEY_ACTIONS)
    const k=(e.key||"").toLowerCase();
    if(k.length===1 && k>="a" && k<="z"){
      const hit = KEY_ACTIONS.find(a=>keyFor(a.id)===k);
      if(hit && KEY_DISPATCH[hit.id]){ KEY_DISPATCH[hit.id](e); return; }
    }
  }
  if(e.key==="Delete"||e.key==="Backspace"){
    if(!editable || (DS && !DS.writable)) return;
    if(polyDraft){ if(polyDraft.length>=2) polyDraft.splice(-2,2); draw(); return; }   // undo last vertex
    if(selBoxes.size){   // bulk delete: splice high indices first so the rest stay valid
      pushUndo(); [...selBoxes].sort((a,b)=>b-a).forEach(bi=>boxes.splice(bi,1));
      selBoxes.clear(); sel=-1; markDirty(); draw(); }
    else if(selPoly>=0){ pushUndo(); polys.splice(selPoly,1); selPoly=-1; markDirty(); draw(); }
    else if(sel>=0){ pushUndo(); boxes.splice(sel,1); sel=-1; markDirty(); draw(); }
    return; }
  if(e.key==="+"||e.key==="="){ e.preventDefault(); zoomBy(1.25); return; }
  if(e.key==="-"||e.key==="_"){ e.preventDefault(); zoomBy(1/1.25); return; }
  if(e.key==="f"||e.key==="F"){ fit(); draw(); return; }
  if(e.key==="?"){ toggleHelp(); return; }
  if(e.key==="Escape"){
    if(polyDraft){ cancelPolyDraft(); return; }
    if(closeTopOverlay()){ }
    else if(radarFindings.length){ radarFindings=[]; $("#banner").style.display="none"; draw(); }
    else if(ghosts.length){ clearGhosts(); }
    else if(mode==="new"){ boxes.pop(); sel=-1; mode=null; gestureSnap=null; draw(); }
    else { selBoxes.clear(); sel=-1; selPoly=-1; draw(); }
  }
});
window.addEventListener("keyup", e=>{ if(e.key===" "){ spaceDown=false; cv.style.cursor="crosshair"; } });
window.addEventListener("resize", ()=>{ resizeCanvas(); if($("#mapmodal").classList.contains("show")) resizeMapCanvas(); });
window.addEventListener("beforeunload", e=>{ if(dirty){ e.preventDefault(); e.returnValue=""; } });

"""

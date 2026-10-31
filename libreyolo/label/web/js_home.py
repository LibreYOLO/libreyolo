"""JS: home screen, New Project wizard, export dialog, project actions, share."""

PART = r"""function wireHome(){
  // Landing is New project + your projects grid; the old
  // inline open-row / create-panel were removed, so every binding is guarded.
  const ht=$("#hometheme"); if(ht) ht.onclick = toggleTheme;
  wireWizard();
  wireProjectActions();
}

// ===== New Project wizard =====
const WZPRESET = ["#06b6d4","#10b981","#f59e0b","#ef4444","#a855f7","#ec4899","#14b8a6","#3b82f6"];
let WZ = null;
function openWizard(){
  WZ = {step:1, mode:"upload", task:"detect", link:false,
        dst:"", uploaded:0, files:[], classes:[{name:"",color:WZPRESET[0]}]};
  $("#wzname").value=""; $("#wzdst").value=""; $("#wzexist").value="";
  document.querySelectorAll("#wztasks .wz-task").forEach(b=>b.classList.toggle("on", b.dataset.task==="detect"));
  document.querySelectorAll("#wzstorage .wz-task").forEach(b=>b.classList.toggle("on", b.dataset.link==="0"));
  renderWzClasses(); renderWzFiles(); wzSetTab("upload"); wzGoto(1);
  $("#wizard").classList.add("show"); setTimeout(()=>{ const n=$("#wzname"); if(n) n.focus(); },30);
}
function closeWizard(){ const w=$("#wizard"); if(w) w.classList.remove("show"); WZ=null; }
function wzErr(m){ const e=$("#wzerr"); if(e) e.textContent=m||""; }
function wzGoto(s){
  WZ.step=s;
  document.querySelectorAll("#wizard .wz-pane").forEach(p=>p.hidden=(+p.dataset.pane!==s));
  document.querySelectorAll("#wizard .wz-step").forEach(el=>{ const n=+el.dataset.s;
    el.classList.toggle("on", n===s); el.classList.toggle("done", n<s); });
  $("#wzback").hidden = s===1;
  $("#wznext").hidden = s===2;
  $("#wzcreate").hidden = s!==2;
  wzErr("");
}
function renderWzClasses(){
  const wrap=$("#wzclasses"); if(!wrap) return;
  wrap.innerHTML = WZ.classes.map((c,i)=>
    `<div class="wz-clsrow"><input type="color" value="${c.color}" data-i="${i}" data-k="color">`
    +`<input class="wz-in" placeholder="class name" value="${esc(c.name)}" data-i="${i}" data-k="name" autocomplete="off">`
    +`<button class="wz-rm" data-i="${i}" title="Remove class">&times;</button></div>`).join("");
  wrap.querySelectorAll("input").forEach(inp=>inp.oninput=()=>{ WZ.classes[+inp.dataset.i][inp.dataset.k]=inp.value; });
  wrap.querySelectorAll(".wz-rm").forEach(b=>b.onclick=()=>{ WZ.classes.splice(+b.dataset.i,1);
    if(!WZ.classes.length) WZ.classes.push({name:"",color:WZPRESET[0]}); renderWzClasses(); });
}
function wzAddClass(){
  WZ.classes.push({name:"",color:WZPRESET[WZ.classes.length%WZPRESET.length]}); renderWzClasses();
  const rows=$("#wzclasses").querySelectorAll('.wz-clsrow input[data-k="name"]'); const last=rows[rows.length-1]; if(last) last.focus();
}
function wzSetTab(t){
  WZ.mode=t;
  document.querySelectorAll("#wizard .wz-tab").forEach(b=>b.classList.toggle("on", b.dataset.tab===t));
  document.querySelectorAll("#wizard .wz-tabpane").forEach(p=>p.hidden = p.dataset.tabpane!==t);
  wzErr("");
}
function renderWzFiles(){
  const wrap=$("#wzfilelist"); if(!wrap) return;
  wrap.innerHTML = WZ.files.map(f=>`<div class="wz-frow ${f.status}"><span class="wz-fn">${esc(f.name)}</span><span class="wz-fs">${f.status==="ok"?"uploaded":f.status==="err"?"failed":"…"}</span></div>`).join("");
  const drop=$("#wzdrop"); if(drop) drop.classList.toggle("disabled", !($("#wzdst").value||"").trim());
}
async function wzUpload(fileList){
  const dst=($("#wzdst").value||"").trim();
  if(!dst){ wzErr("Choose a destination folder first."); return; }
  wzErr("");
  for(const file of fileList){
    const rec={name:file.name, status:"up"}; WZ.files.push(rec); renderWzFiles();
    try{
      const r=await fetch(`/api/upload?dst=${encodeURIComponent(dst)}&name=${encodeURIComponent(file.name)}`,{method:"POST",body:file});
      const d=await r.json().catch(()=>({}));
      if(r.ok&&d.ok){ rec.status="ok"; WZ.uploaded++; } else { rec.status="err"; }
    }catch(e){ rec.status="err"; }
    renderWzFiles();
  }
}
async function wzNext(){
  if(WZ.step!==1) return;
  if(WZ.mode==="upload"){
    if(!($("#wzdst").value||"").trim()){ wzErr("Choose a destination folder."); return; }
    if(WZ.uploaded<1){ wzErr("Upload at least one image."); return; }
    WZ.dst=$("#wzdst").value.trim(); wzGoto(2); return;
  }
  const p=($("#wzexist").value||"").trim();
  if(!p){ wzErr("Paste a folder or a data.yaml."); return; }
  if(/\.ya?ml$/i.test(p)){ closeWizard(); return openProject(p); }
  wzErr("Checking…");
  let info;
  try{ const r=await fetch("/api/projects/inspect",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({folder:p})}); info=await r.json(); if(!r.ok) throw 0; }
  catch(e){ wzErr("Couldn't read that path."); return; }
  if(info.has_yaml){ closeWizard(); return openProject(info.yaml||p); }
  if(!info.is_dir){ wzErr("That isn't a folder or a data.yaml."); return; }
  if(!info.images){ wzErr("No images found in that folder."); return; }
  WZ.dst=p; wzErr(""); wzGoto(2);
}
function wzClassesPayload(){
  const seen=new Set(), names=[], colors=[];
  for(const c of WZ.classes){ const n=(c.name||"").trim(); if(!n) continue;
    const k=n.toLowerCase(); if(seen.has(k)) return {dup:true}; seen.add(k); names.push(n); colors.push(c.color||""); }
  return {names, colors};
}
async function wzCreate(){
  const cp=wzClassesPayload();
  if(cp.dup){ wzErr("Class names must be unique."); return; }
  const task=WZ.task;
  const btn=$("#wzcreate"), t=btn.textContent; btn.disabled=true; btn.textContent="Creating…"; wzErr("");
  try{
    let d;
    if(WZ.mode==="upload"){
      const body={dst:WZ.dst, name:$("#wzname").value.trim(), description:"",
        color:"", task, classes:cp.names, colors:cp.colors, make_val:false};
      const r=await fetch("/api/projects/new",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
      d=await r.json(); if(!r.ok||!d.open){ wzErr((d&&d.error)||"Could not create the project."); return; }
    } else {
      const r=await fetch("/api/projects/create",{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({folder:WZ.dst, classes:cp.names, colors:cp.colors, task,
                             link:!!WZ.link, name:$("#wzname").value.trim()})});
      d=await r.json(); if(!r.ok||!d.open){ wzErr((d&&d.error)||"Could not create the project."); return; }
    }
    closeWizard(); resetClientState(); await enterLabeler(d);
  }catch(e){ wzErr("Could not create the project."); }
  finally{ btn.disabled=false; btn.textContent=t; }
}
async function wzPickFolder(){
  try{ const r=await fetch("/api/pick-folder",{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});
    const d=await r.json().catch(()=>({})); if(d&&d.folder) return d.folder; }catch(e){}
  return null;
}
function wireWizard(){
  const nb=$("#homenew"); if(nb) nb.onclick=openWizard;
  const w=$("#wizard"); if(!w) return;
  $("#wzclose").onclick=closeWizard;
  w.addEventListener("click",e=>{ if(e.target.id==="wizard") closeWizard(); });
  $("#wzback").onclick=()=>wzGoto(Math.max(1,WZ.step-1));
  $("#wznext").onclick=wzNext;
  $("#wzcreate").onclick=wzCreate;
  $("#wzaddcls").onclick=wzAddClass;
  document.querySelectorAll("#wizard .wz-tab").forEach(b=>b.onclick=()=>wzSetTab(b.dataset.tab));
  document.querySelectorAll("#wztasks .wz-task").forEach(b=>b.onclick=()=>{ if(b.disabled) return; WZ.task=b.dataset.task;
    document.querySelectorAll("#wztasks .wz-task").forEach(x=>x.classList.toggle("on",x===b)); });
  document.querySelectorAll("#wzstorage .wz-task").forEach(b=>b.onclick=()=>{ WZ.link = b.dataset.link==="1";
    document.querySelectorAll("#wzstorage .wz-task").forEach(x=>x.classList.toggle("on",x===b)); });
  $("#wzbrowse").onclick=async()=>{ const f=await wzPickFolder(); if(f){ $("#wzdst").value=f; renderWzFiles(); } };
  $("#wzexistbrowse").onclick=async()=>{ const f=await wzPickFolder(); if(f) $("#wzexist").value=f; };
  $("#wzdst").oninput=renderWzFiles;
  const fi=$("#wzfiles"); if(fi) fi.onchange=()=>{ if(fi.files&&fi.files.length) wzUpload(Array.from(fi.files)); fi.value=""; };
  $("#wzpick").onclick=(e)=>{ e.stopPropagation(); if(!($("#wzdst").value||"").trim()){ wzErr("Choose a destination folder first."); return; } $("#wzfiles").click(); };
  const drop=$("#wzdrop");
  if(drop){
    drop.onclick=(e)=>{ if(e.target.closest(".wz-link")) return; if(!($("#wzdst").value||"").trim()){ wzErr("Choose a destination folder first."); return; } $("#wzfiles").click(); };
    drop.addEventListener("dragover",e=>{ e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave",()=>drop.classList.remove("over"));
    drop.addEventListener("drop",e=>{ e.preventDefault(); drop.classList.remove("over");
      const fs=Array.from((e.dataTransfer&&e.dataTransfer.files)||[]).filter(f=>/^image\//.test(f.type)||/\.(jpe?g|png|bmp|webp|tiff?)$/i.test(f.name));
      if(fs.length) wzUpload(fs); });
  }
}
let homeOpenMeta = null;   // the live session, for the teammate Resume escape hatch
function showHome(openMeta){
  $("#home").classList.add("show");
  homeError("");
  homeOpenMeta = (openMeta && openMeta.open) ? openMeta : null;
  const rz=$("#homeresume"); if(rz) rz.style.display="none";   // renderProjects decides
  renderProjects();
}
// "Resume current project" appears ONLY when there is a live session but no project
// list to click - i.e. a LAN teammate on a --share server (the registry is
// admin-only), who otherwise gets stranded on Home. The host has its project cards.
function maybeOfferResume(projectCount){
  const rz=$("#homeresume"), btn=$("#homeresumebtn");
  if(!rz || !btn) return;
  if(homeOpenMeta && projectCount===0){
    rz.style.display="";
    btn.onclick = async ()=>{ try{ sessionStorage.removeItem("ll-home"); }catch(e){} hideHome(); await enterLabeler(homeOpenMeta); };
  } else rz.style.display="none";
}
function hideHome(){ $("#home").classList.remove("show"); }
async function backToHome(){
  if(dirty && idx>=0 && !(await save())){ banner("Save failed - fix it before leaving this image."); return; }
  if(dirty){ banner("You edited while saving - press → to save before going home."); return; }   // in-flight edits: don't drop them on Home
  try{ sessionStorage.setItem("ll-home","1"); }catch(e){}
  showHome(DS);   // pass the open session so "Resume current project" is offered
}
function homeError(msg){ const e=$("#homeerr"); if(e) e.textContent = msg||""; }
async function renderProjects(){
  const grid=$("#homegrid"); if(!grid) return;
  grid.innerHTML = `<div class="home-empty">Loading…</div>`;
  let d; try{ d = await jget("/api/projects"); }
  catch(e){ grid.innerHTML = `<div class="home-empty">Could not load projects.</div>`; return; }
  const list = d.projects||[];
  maybeOfferResume(list.length);
  if(!list.length){ grid.innerHTML = `<div class="home-empty">No projects yet - click <b>New project</b> to get started.</div>`; return; }
  const dots = `<svg class="ic" viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.7"/><circle cx="12" cy="12" r="1.7"/><circle cx="19" cy="12" r="1.7"/></svg>`;
  grid.innerHTML = list.map(p=>{
    const n=p.count||0, l=p.labeled||0, pct = n? Math.round(100*l/n) : 0; const dd=esc(p.data);
    return `<div class="prj" data-data="${dd}" tabindex="0">`
      + `<span class="prj-menu" title="Project actions">${dots}</span>`
      + `<div class="prj-actions">`
      +   `<button class="prj-act" data-act="open">Open</button>`
      +   `<button class="prj-act" data-act="rename">Rename</button>`
      +   `<button class="prj-act" data-act="classes">Edit classes</button>`
      +   `<button class="prj-act" data-act="forget">Forget</button>`
      +   `<button class="prj-act danger" data-act="delete">Delete dataset</button>`
      + `</div>`
      + `<div class="prj-name">${esc(p.name||"dataset")}</div>`
      + `<div class="prj-path" title="${dd}">${dd}</div>`
      + `<div class="prj-barwrap"><div class="prj-bar" style="width:${pct}%"></div></div>`
      + `<div class="prj-meta"><span>${n} images · ${l} labeled</span><span>${esc(relTime(p.last_opened))}</span></div>`
      + `</div>`;
  }).join("");
  grid.querySelectorAll(".prj").forEach(card=> card.onclick=(e)=>{
    if(e.target.closest(".prj-menu")||e.target.closest(".prj-actions")) return;
    openProject(card.dataset.data);
  });
  grid.querySelectorAll(".prj-menu").forEach(m=> m.onclick=(e)=>{ e.stopPropagation();
    const acts=m.parentElement.querySelector(".prj-actions"); const wasOpen=acts.classList.contains("show");
    closeAllPrjMenus(); if(!wasOpen) acts.classList.add("show");
  });
  grid.querySelectorAll(".prj-act").forEach(btn=> btn.onclick=(e)=>{ e.stopPropagation();
    const card=btn.closest(".prj"), data=card.dataset.data, act=btn.dataset.act; closeAllPrjMenus();
    if(act==="open") openProject(data);
    else if(act==="rename") openRename(data, card.querySelector(".prj-name").textContent);
    else if(act==="classes") editClassesFromCard(data);
    else if(act==="forget") forgetProject(data);
    else if(act==="delete") openDeleteConfirm(data);
  });
}
function closeAllPrjMenus(){ document.querySelectorAll(".prj-actions.show").forEach(x=>x.classList.remove("show")); }
async function forgetProject(data){
  try{ await fetch("/api/projects/forget",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({data})}); }catch(_e){}
  renderProjects();
}
let renameTarget=null, deleteTarget=null;
function openRename(data, current){
  renameTarget=data; const i=$("#reninput"); if(i) i.value=current||""; $("#renerr").textContent="";
  $("#renamemodal").classList.add("show"); setTimeout(()=>{ if(i){ i.focus(); i.select(); } },30);
}
function closeRename(){ const m=$("#renamemodal"); if(m) m.classList.remove("show"); renameTarget=null; }
async function doRename(){
  const name=(($("#reninput").value)||"").trim(); if(!name){ $("#renerr").textContent="Enter a name."; return; }
  if(!renameTarget){ closeRename(); return; }
  const btn=$("#rensave"), t=btn.textContent; btn.disabled=true; btn.textContent="Saving…";
  try{
    const r=await fetch("/api/projects/rename",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({data:renameTarget, name})});
    const dd=await r.json().catch(()=>({})); if(!r.ok||!dd.ok){ $("#renerr").textContent=(dd&&dd.error)||"Rename failed."; return; }
    closeRename(); renderProjects();
  }catch(e){ $("#renerr").textContent="Rename failed."; }
  finally{ btn.disabled=false; btn.textContent=t; }
}
function openDeleteConfirm(data){
  deleteTarget=data; $("#delpath").textContent=data; $("#delerr").textContent="";
  $("#deletemodal").classList.add("show");
}
function closeDelete(){ const m=$("#deletemodal"); if(m) m.classList.remove("show"); deleteTarget=null; }
async function doDelete(){
  if(!deleteTarget){ closeDelete(); return; }
  const btn=$("#delconfirm"), t=btn.textContent; btn.disabled=true; btn.textContent="Moving…";
  try{
    const wasOpen = DS && (deleteTarget===DS.yaml || deleteTarget===DS.root);
    const r=await fetch("/api/projects/delete",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({data:deleteTarget})});
    const dd=await r.json().catch(()=>({})); if(!r.ok||!dd.ok){ $("#delerr").textContent=(dd&&dd.error)||"Delete failed."; return; }
    closeDelete();
    if(wasOpen){   // deleted the live project (e.g. from Settings): the session is gone
      resetClientState(); DS=null;
      try{ sessionStorage.setItem("ll-home","1"); }catch(e){}
      showHome();
    } else renderProjects();
  }catch(e){ $("#delerr").textContent="Delete failed."; }
  finally{ btn.disabled=false; btn.textContent=t; }
}
async function editClassesFromCard(data){ if(await openProject(data)) openClassEdit(); }
// ===== Project settings (open project only: name / description / instructions) =====
function openSettings(){
  if(!DS){ banner("Open a project first."); return; }
  $("#setname").value = DS.name || "";
  $("#setdesc").value = DS.description || "";
  $("#setinstr").value = DS.instructions || "";
  $("#seterr").textContent = "";
  $("#settingsmodal").classList.add("show");
  setTimeout(()=>{ const n=$("#setname"); if(n) n.focus(); }, 30);
}
function closeSettings(){ const m=$("#settingsmodal"); if(m) m.classList.remove("show"); }
async function saveSettings(){
  const name=($("#setname").value||"").trim();
  if(!name){ $("#seterr").textContent="Enter a project name."; return; }
  const btn=$("#setsave"), t=btn.textContent; btn.disabled=true; btn.textContent="Saving…";
  try{
    const r=await fetch("/api/projects/meta",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({name, description:$("#setdesc").value, instructions:$("#setinstr").value,
                           epoch:(DS&&DS.epoch)||0})});
    const d=await r.json().catch(()=>({}));
    if(!r.ok){ $("#seterr").textContent=(d&&d.error)||"Could not save settings."; return; }
    DS.name=d.name; DS.description=d.description; DS.instructions=d.instructions;
    $("#dsname").textContent = DS.name || (DS.root||"").split(/[\\/]/).filter(Boolean).pop() || "dataset";
    closeSettings();
  }catch(e){ $("#seterr").textContent="Could not save settings."; }
  finally{ btn.disabled=false; btn.textContent=t; }
}
// ===== Export (YOLO / COCO / Pascal VOC, with split) =====
function openExport(){
  if(!DS){ banner("Open a project first."); return; }
  $("#experr").textContent=""; $("#expresult").textContent="";
  const root=((DS.root||"")+"").replace(/[\\/]+$/,"");
  $("#expdst").value = root ? root+"_export" : "";
  $("#expinplace").checked=false; expSyncMode(); expSetSplit("trainval");
  $("#exportmodal").classList.add("show");
}
function closeExport(){ const m=$("#exportmodal"); if(m) m.classList.remove("show"); }
function expSetSplit(s){
  document.querySelectorAll("#expsplit button").forEach(b=>b.classList.toggle("on", b.dataset.s===s));
  const tw=$("#exptestwrap"); if(tw) tw.hidden = (s!=="trainvaltest");
  const r=$("#expratios"); if(r) r.style.display = (s==="none")?"none":"flex";
}
function expSyncMode(){
  const ip=$("#expinplace").checked;
  const co=$("#expcopyopts"); if(co) co.style.display = ip?"none":"";
  const w=$("#expinplacewarn"); if(w) w.hidden = !ip;
}
async function doExport(){
  if(!DS){ return; }
  // /api/export reads label files from disk: flush the canvas first so an export
  // right after drawing can't silently omit the newest annotations.
  if(dirty && idx>=0 && !(await save())){ $("#experr").textContent="Couldn't save the current image - fix that first."; return; }
  if(dirty){ $("#experr").textContent="You edited while saving - save before exporting."; return; }
  const inplace=$("#expinplace").checked;
  const splitEl=document.querySelector("#expsplit button.on"); const split=(splitEl&&splitEl.dataset.s)||"trainval";
  const body={ split, val_frac:((+$("#expval").value)||20)/100, test_frac:((+$("#exptest").value)||0)/100,
    seed:(+$("#expseed").value)||1234, in_place:inplace, epoch:(DS&&DS.epoch)||0 };
  const yamlp=(DS&&DS.yaml)||"";
  if(!inplace){
    const fmts=[]; if($("#expyolo").checked)fmts.push("yolo"); if($("#expcoco").checked)fmts.push("coco"); if($("#expvoc").checked)fmts.push("voc");
    if(!fmts.length){ $("#experr").textContent="Pick at least one format."; return; }
    const dst=($("#expdst").value||"").trim(); if(!dst){ $("#experr").textContent="Choose a destination folder."; return; }
    body.formats=fmts; body.dst=dst; body.make_zip=$("#expzip").checked;
  }
  const btn=$("#expgo"), t=btn.textContent; btn.disabled=true; btn.textContent="Exporting…"; $("#experr").textContent=""; $("#expresult").textContent="";
  try{
    const r=await fetch("/api/export",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    const d=await r.json(); if(!r.ok||!d.ok){ $("#experr").textContent=(d&&d.error)||"Export failed."; return; }
    const c=d.counts||{}; const sp=`train ${c.train||0}, val ${c.val||0}${c.test?(", test "+c.test):""}`;
    if(d.in_place){ banner("Re-split in place ("+sp+")."); closeExport(); const y=d.yaml||yamlp; if(y) openProject(y); }
    else { $("#expresult").innerHTML=`Exported (${sp}) to<br><b>${esc(d.out)}</b>`+(d.zip?`<br>zip: ${esc(d.zip)}`:""); }
  }catch(e){ $("#experr").textContent="Export failed."; }
  finally{ btn.disabled=false; btn.textContent=t; }
}
function wireProjectActions(){
  document.addEventListener("click", ()=>closeAllPrjMenus());
  $("#renclose").onclick=closeRename; $("#rencancel").onclick=closeRename; $("#rensave").onclick=doRename;
  $("#renamemodal").addEventListener("click",e=>{ if(e.target.id==="renamemodal") closeRename(); });
  $("#reninput").addEventListener("keydown",e=>{ if(e.key==="Enter"){ e.preventDefault(); doRename(); } });
  $("#delclose").onclick=closeDelete; $("#delcancel").onclick=closeDelete; $("#delconfirm").onclick=doDelete;
  $("#deletemodal").addEventListener("click",e=>{ if(e.target.id==="deletemodal") closeDelete(); });
  $("#helpclose").onclick=toggleHelp;
  $("#help").addEventListener("click",e=>{ if(e.target.id==="help") toggleHelp(); });
  const kr=$("#kreset"); if(kr) kr.onclick=()=>{ resetKeys(); rebindTarget=null; renderHelp("Shortcuts reset to defaults."); };
  const sb=$("#settingsbtn"); if(sb) sb.onclick=openSettings;
  $("#setclose").onclick=closeSettings; $("#setcancel").onclick=closeSettings; $("#setsave").onclick=saveSettings;
  $("#settingsmodal").addEventListener("click",e=>{ if(e.target.id==="settingsmodal") closeSettings(); });
  $("#setclasses").onclick=()=>{ closeSettings(); openClassEdit(); };
  $("#setforget").onclick=async()=>{
    if(!DS) return; closeSettings();
    try{ await fetch("/api/projects/forget",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({data:DS.yaml||DS.root})}); }catch(e){}
    backToHome();
  };
  $("#setdelete").onclick=()=>{ if(!DS) return; closeSettings(); openDeleteConfirm(DS.yaml||DS.root||""); };
  const eb=$("#exportbtn"); if(eb) eb.onclick=openExport;
  $("#expclose").onclick=closeExport; $("#expcancel").onclick=closeExport; $("#expgo").onclick=doExport;
  $("#exportmodal").addEventListener("click",e=>{ if(e.target.id==="exportmodal") closeExport(); });
  document.querySelectorAll("#expsplit button").forEach(b=>b.onclick=()=>expSetSplit(b.dataset.s));
  $("#expinplace").addEventListener("change", expSyncMode);
  $("#expbrowse").onclick=async()=>{ const f=await wzPickFolder(); if(f) $("#expdst").value=f; };
}
async function openProject(data){
  if(!data){ homeError("Enter a path to a data.yaml or a dataset folder."); return false; }
  // Persist the open image's edits before swapping projects: resetClientState()
  // zeroes `dirty` and clears boxes/polys, so without this the work is lost silently
  // (not even the beforeunload warning fires). Same guard as backToHome/fixDuplicate.
  if(dirty && idx>=0 && !(await save())){ homeError("Save the current image before switching projects."); return false; }
  if(dirty){ homeError("You edited while saving - save before switching projects."); return false; }
  homeError("");
  try{
    const r = await fetch("/api/projects/open",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({data})});
    const d = await r.json();
    if(!r.ok || !d.open){ homeError(d.error||"Could not open that dataset"); return false; }
    resetClientState();
    await enterLabeler(d);
    return true;
  }catch(e){ showHome(); homeError("Could not open that dataset"); return false; }
}
function resetClientState(){
  loadSeq++;            // invalidate any in-flight load()/poll/stream from the previous project
  clearTimeout(boostTimer); boostTimer=null;
  clearTimeout(autosaveTimer); autosaveTimer=null;
  idx=-1; sel=-1; selPoly=-1; selBoxes.clear(); hover=-1; active=0; boxes=[]; polys=[]; ghosts=[]; radarFindings=[];
  IMAGES=[]; suggestedIds=new Set(); selSet=null; listFilter="all"; radarDeck=[]; radarLast=null; mapPoints=[]; mapFit=null; progSig="";
  undoStack=[]; redoStack=[]; gestureSnap=null; dirty=false; imgOk=false; gradData=null; assist=null; assistModel=null;
  CLSCOL=[];
  setTool("box");
  imgQuery=""; listLimit=400; const si=$("#imgsearch"); if(si) si.value="";
  document.querySelectorAll("#filter button").forEach(x=>x.classList.toggle("on", x.dataset.f==="all"));
  const bc=$("#boostchip"); if(bc) bc.className="chip";
  ["#radar","#mapmodal","#insights","#sharepop","#picker"].forEach(s=>{ const m=$(s); if(m) m.classList.remove("show"); });
  const _help=$("#help"); if(_help) _help.style.display="none";
  const b=$("#banner"); if(b) b.style.display="none";
}
function relTime(epoch){
  if(!epoch) return "";
  const s = Math.max(0, (Date.now()/1000) - epoch);
  if(s<60) return "just now";
  if(s<3600) return Math.floor(s/60)+"m ago";
  if(s<86400) return Math.floor(s/3600)+"h ago";
  if(s<2592000) return Math.floor(s/86400)+"d ago";
  return Math.floor(s/2592000)+"mo ago";
}

// ---- teammate share ----
function urlRow(u){ return `<div class="urlrow"><code>${esc(u)}</code><button class="copybtn" data-u="${esc(u)}" title="Copy">${ICO_COPY}</button></div>`; }
async function toggleShare(){
  const pop=$("#sharepop"); if(!pop) return;
  if(pop.classList.contains("show")){ pop.classList.remove("show"); return; }
  pop.classList.add("show"); pop.innerHTML = `<p>Loading…</p>`;
  let s; try{ s=await jget("/api/server"); }catch(e){ pop.innerHTML=`<p>Could not read server info.</p>`; return; }
  if(s.shareable && s.lan_url){
    pop.innerHTML = `<h4>Label together</h4><p>Send this link to teammates on your network - they open the same dataset and you split the images (each save writes its own file).</p>` + urlRow(s.lan_url);
  } else {
    pop.innerHTML = `<h4>Running locally</h4><p>Only this machine can reach it. To let teammates on your network join, restart with <code>--share</code>.</p>` + urlRow(s.local_url);
  }
  pop.querySelectorAll(".copybtn").forEach(b=> b.onclick=()=>{
    try{ navigator.clipboard.writeText(b.dataset.u); }catch(e){}
    b.classList.add("copied"); b.innerHTML=ICO_CHECK;
    setTimeout(()=>{ b.classList.remove("copied"); b.innerHTML=ICO_COPY; }, 1200);
  });
}

init().catch(err=>{ document.body.insertAdjacentHTML("afterbegin",
  `<div style="padding:14px;color:#f5b13d">LibreLabel failed to start: ${err.message}</div>`); });
"""

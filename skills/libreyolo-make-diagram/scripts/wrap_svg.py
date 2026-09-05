#!/usr/bin/env python3
"""Wrap an original LibreYOLO SVG in a standalone interactive viewer."""
import argparse
import html
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

TEMPLATE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>__TITLE__</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#edf2f5;color:#253740;font:14px system-ui,sans-serif}header{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:20px 28px;background:#fff;border-bottom:1px solid #d1dde3}h1{font-size:20px;margin:0;font-weight:600}.tools{display:flex;flex-wrap:wrap;gap:8px}button{font:inherit;color:#253740;background:#fff;border:1px solid #bcccd4;border-radius:4px;padding:9px 12px;cursor:pointer}button:hover{background:#eefbff}button:focus-visible,a:focus-visible{outline:3px solid #087e98;outline-offset:3px}main{max-width:2080px;padding:24px;margin:auto;display:grid;grid-template-columns:minmax(0,1fr) 270px;gap:16px}.hint{grid-column:1/-1;margin:0;color:#536e7c}#viewport{overflow:auto;background:#fff;border:1px solid #ccdae2}#poster{display:block;width:100%;height:auto}aside{align-self:start;position:sticky;top:16px;background:#fff;border:1px solid #ccdae2;padding:16px;line-height:1.6}aside strong{color:#087e98}aside a{color:#087e98}.inspectable{cursor:pointer}.inspectable:focus{outline:none}.inspectable:hover>.outline,.inspectable:focus>.outline,.inspectable.selected>.outline{stroke:#087e98;stroke-width:3}.wire.selected{stroke:#087e98;stroke-width:3}@media(max-width:1000px){main{display:flex;flex-direction:column}aside{position:static}header{align-items:flex-start;flex-direction:column}}@media print{header,aside,.hint{display:none}main{display:block;padding:0}#viewport{border:0;overflow:visible}#poster{width:100%!important}@page{size:A3 portrait;margin:8mm}}
</style></head><body>
<header><h1>__TITLE__</h1><div class="tools"><button id="fit">Fit diagram</button><button id="zoom">Read at 100%</button><button id="reset">Clear selection</button><button id="saveSvg">Download SVG</button><button id="savePng">Download PNG</button></div></header>
<main><p class="hint">Click a block to read its description, or select it with Tab and Enter.</p><div id="viewport">__SVG__</div><aside id="inspector" aria-live="polite"><strong>Select a block</strong><p>The description and source link appear here.</p></aside></main>
<script>
const svg=document.querySelector('#poster'),inspector=document.querySelector('#inspector'),filename=__FILENAME__;
function clearSelection(){svg.querySelectorAll('.selected').forEach(e=>e.classList.remove('selected'))}
function inspect(node){clearSelection();node.classList.add('selected');const block=node.dataset.block;if(block)svg.querySelectorAll('.inspectable').forEach(e=>{if(e.dataset.block===block)e.classList.add('selected')});svg.querySelectorAll('.wire').forEach(e=>{if(e.dataset.from===node.dataset.node||e.dataset.to===node.dataset.node)e.classList.add('selected')});inspector.replaceChildren();const title=document.createElement('strong');title.textContent=node.dataset.label||node.getAttribute('aria-label');const p=document.createElement('p');p.textContent=node.dataset.description||'See the operation labels and connected block definition.';inspector.append(title,p);const source=node.dataset.source;if(source&&/^https?:\\/\\//.test(source)){const a=document.createElement('a');a.href=source;a.target='_blank';a.rel='noopener noreferrer';a.textContent='View source';inspector.append(a)}}
svg.querySelectorAll('.inspectable').forEach(node=>{node.addEventListener('click',e=>{e.stopPropagation();inspect(node)});node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();inspect(node)}})});
document.querySelector('#fit').onclick=()=>svg.style.width='100%';document.querySelector('#zoom').onclick=()=>svg.style.width=svg.viewBox.baseVal.width+'px';document.querySelector('#reset').onclick=()=>{clearSelection();inspector.innerHTML='<strong>Select a block</strong><p>The description and source link appear here.</p>'};
function serialized(){const clone=svg.cloneNode(true);clone.removeAttribute('style');clone.setAttribute('width',svg.viewBox.baseVal.width);clone.setAttribute('height',svg.viewBox.baseVal.height);clone.querySelectorAll('.selected').forEach(e=>e.classList.remove('selected'));return new XMLSerializer().serializeToString(clone)}
function download(blob,ext){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=filename+'.'+ext;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000)}
document.querySelector('#saveSvg').onclick=()=>download(new Blob([serialized()],{type:'image/svg+xml'}),'svg');
document.querySelector('#savePng').onclick=async()=>{const b=document.querySelector('#savePng');b.disabled=true;b.textContent='Exporting PNG';let url;try{url=URL.createObjectURL(new Blob([serialized()],{type:'image/svg+xml'}));const img=new Image();img.src=url;await img.decode();const canvas=document.createElement('canvas');canvas.width=svg.viewBox.baseVal.width*2;canvas.height=svg.viewBox.baseVal.height*2;canvas.getContext('2d').drawImage(img,0,0,canvas.width,canvas.height);const png=await new Promise(resolve=>canvas.toBlob(resolve,'image/png'));if(!png)throw new Error('The browser did not produce an image.');download(png,'png')}catch(e){inspector.textContent='PNG export failed: '+e.message}finally{if(url)URL.revokeObjectURL(url);b.disabled=false;b.textContent='Download PNG'}};
new ResizeObserver(()=>{if(parent!==window)parent.postMessage({type:'libreyolo-architecture-height',height:Math.ceil(document.body.getBoundingClientRect().height)+2},location.origin)}).observe(document.body);
</script></body></html>'''


def wrap(svg_path, output):
    svg_path, output = Path(svg_path), Path(output)
    raw = svg_path.read_text(encoding="utf-8")
    root = ET.fromstring(raw)
    if root.tag != '{http://www.w3.org/2000/svg}svg':
        raise ValueError('Input must be an SVG with its XML namespace')
    viewbox = [float(v) for v in root.attrib.get('viewBox', '').split()]
    if len(viewbox) != 4 or min(viewbox[2:]) <= 0:
        raise ValueError('SVG needs a positive viewBox width and height')
    title_node = root.find('{http://www.w3.org/2000/svg}title')
    title = (title_node.text if title_node is not None else None) or svg_path.stem
    root.set('id', 'poster')
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    svg = ET.tostring(root, encoding='unicode')
    safe_name = re.sub(r'[^A-Za-z0-9_-]', '-', svg_path.stem)
    # Replace only the template markers, never markers occurring inside SVG text.
    replacements = {'__TITLE__': html.escape(title), '__SVG__': svg,
                    '__FILENAME__': json.dumps(safe_name)}
    page = re.sub(r'__TITLE__|__SVG__|__FILENAME__', lambda m: replacements[m.group()], TEMPLATE)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('svg')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(wrap(args.svg, args.output))

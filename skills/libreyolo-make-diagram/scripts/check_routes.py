#!/usr/bin/env python3
"""Report long coincident or closely parallel SVG wire segments for visual review.

Understands the helper's M/L polylines and translate-only groups. This is a
geometry aid, not a graph-equivalence proof or automatic layout engine.
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET

NUMBER = r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?'


def segments(root):
    result, skipped = [], []
    def walk(e, tx=0., ty=0., unsupported=False):
        transform = e.get('transform', '').strip()
        if transform:
            match = re.fullmatch(r'translate\(\s*('+NUMBER+r')[,\s]+('+NUMBER+r')\s*\)', transform)
            if match:
                tx += float(match[1]); ty += float(match[2])
            else:
                unsupported = True
        cls = set(e.get('class', '').split())
        if e.tag.endswith('}path') and cls.intersection({'wire', 'edge'}):
            name = e.get('id') or f"{e.get('data-from', '?')} / {e.get('data-to', '?')}"
            d = e.get('d', '')
            if unsupported or re.search(r'[A-KN-Za-kn-z]', re.sub(NUMBER, '', d)):
                skipped.append(name)
            else:
                tokens = re.findall(r'[MLml]|'+NUMBER, d)
                points, i, command = [], 0, None
                try:
                    while i < len(tokens):
                        if tokens[i] in ('M','L','m','l'):
                            command = tokens[i]; i += 1
                        x,y = float(tokens[i]),float(tokens[i+1]); i += 2
                        if command not in ('M','L'):
                            raise ValueError('Only absolute M/L is supported')
                        points.append((x+tx,y+ty))
                    wire_id = id(e)
                    for (x1,y1),(x2,y2) in zip(points,points[1:]):
                        if abs(x1-x2)<0.01 and abs(y1-y2)>0.01:
                            axis,pos,lo,hi='v',x1,min(y1,y2),max(y1,y2)
                        elif abs(y1-y2)<0.01 and abs(x1-x2)>0.01:
                            axis,pos,lo,hi='h',y1,min(x1,x2),max(x1,x2)
                        else:
                            continue
                        result.append(dict(wire=wire_id,name=name,source=e.get('data-from',''),
                                           axis=axis,pos=pos,lo=lo,hi=hi))
                except (ValueError,IndexError):
                    skipped.append(name)
        for child in e:
            walk(child,tx,ty,unsupported)
    walk(root)
    return result, skipped


def check(root, min_gap=20, min_run=60):
    parts, skipped = segments(root)
    findings=[]
    for i,a in enumerate(parts):
        for b in parts[i+1:]:
            if a['wire']==b['wire'] or a['axis']!=b['axis']:
                continue
            gap=abs(a['pos']-b['pos'])
            run=min(a['hi'],b['hi'])-max(a['lo'],b['lo'])
            if gap>=min_gap or run<min_run:
                continue
            shared=bool(a['source']) and a['source']==b['source']
            findings.append(dict(kind='coincident' if gap<0.01 else 'close parallel',
                                 first=a['name'],second=b['name'],axis=a['axis'],
                                 gap=round(gap,2),overlap=round(run,2),
                                 same_source=shared,
                                 note='Check whether this is a legitimate shared-tensor trunk.' if shared else 'Keep distinct tensors visually separate.'))
    return dict(findings=findings,skipped=skipped,segments=len(parts))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('svg');p.add_argument('--min-gap',type=float,default=20)
    p.add_argument('--min-run',type=float,default=60)
    p.add_argument('--max-findings',type=int,default=30)
    a=p.parse_args()
    report=check(ET.parse(a.svg).getroot(),a.min_gap,a.min_run)
    total=len(report['findings']);report['findings']=report['findings'][:a.max_findings]
    report['total_findings']=total
    print(json.dumps(report,indent=2))

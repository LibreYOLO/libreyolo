"""Original LibreYOLO diagram primitives. Python standard library only.

Explicit geometry is intentional: the author chooses routes after reading the
model, and checks the rendered result. This helper does not infer architecture.
"""
from __future__ import annotations

import base64
from pathlib import Path
import re
import xml.etree.ElementTree as ET

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
# Palette selected by Xuban from the approved YOLO9-T diagram.
REFERENCE_PALETTE = {
    "conv": "#8cdef5", "conv2d": "#f7c9a7", "bottleneck": "#f6c9a7",
    "norm": "#d4f4a6", "activation": "#a8ece2", "concat": "#ffd448",
    "split": "#c6dfa9", "pool": "#e8d9f7", "aggregate": "#ffe49b",
    "spp": "#a8cdf4", "attention": "#a8cdf4", "linear": "#f7c9a7",
    "plain": "#ffffff",
}
PALETTE = REFERENCE_PALETTE

REFERENCE_TINTS = {"conv": "#eefbff", "pool": "#f6f2fc", "bottleneck": "#fff3e9",
                   "aggregate": "#fff6d9", "attention": "#e9f3ff", "plain": "#ffffff"}
TINTS = dict(REFERENCE_TINTS)


def checked(text):
    text = str(text)
    if re.search(r"[\u2190-\u21ff\u27f0-\u27ff\u2900-\u297f·•—–]", text):
        raise ValueError(f"Decorative separator or Unicode arrow in diagram text: {text!r}")
    return text


def element(parent, tag, **attrs):
    attrs = {k.rstrip("_").replace("_", "-"): str(v) for k, v in attrs.items() if v is not None}
    return ET.SubElement(parent, f"{{{NS}}}{tag}", attrs)


def label(parent, x, y, text, size=16, fill="#0f172a", weight=400, anchor="start"):
    e = element(parent, "text", x=x, y=y, font_size=size, fill=fill,
                font_weight=weight, text_anchor=anchor,
                font_family="Arial, Helvetica, sans-serif")
    e.text = checked(text)
    return e


class Diagram:
    def __init__(self, title, subtitle, *, width=1800, height=2550,
                 source_url="", source_label="", revision="", logo=None, theme="reference"):
        self.width, self.height = width, height
        if theme != "reference":
            raise ValueError("Use the approved reference palette")
        self.palette = REFERENCE_PALETTE
        self.tints = REFERENCE_TINTS
        self.source_url, self.source_label, self.revision = source_url, source_label, revision
        self.nodes = {}
        self.root = ET.Element(f"{{{NS}}}svg", {
            "viewBox": f"0 0 {width} {height}", "width": str(width), "height": str(height),
            "role": "group", "aria-label": checked(title),
        })
        element(self.root, "title").text = checked(title)
        element(self.root, "desc").text = checked(subtitle)
        defs = element(self.root, "defs")
        marker = element(defs, "marker", id="wire-arrow", viewBox="0 0 10 10", refX=9,
                         refY=5, markerWidth=5, markerHeight=5, orient="auto")
        element(marker, "path", d="M0,1 L9,5 L0,9 Z", fill="#475569")
        element(self.root, "rect", x=0, y=0, width=width, height=height, fill="#fff")
        logo = Path(logo) if logo else Path(__file__).resolve().parent.parent / "assets/logo.png"
        encoded = base64.b64encode(logo.read_bytes()).decode("ascii")
        element(self.root, "image", x=45, y=35, width=67, height=67,
                href=f"data:image/png;base64,{encoded}")
        label(self.root, 126, 75, "LibreYOLO", 32, weight=700)
        label(self.root, 50, 150, title, 42, weight=700)
        label(self.root, 50, 185, subtitle, 17, fill="#536e7c")
        self.global_wires = ET.Element(f"{{{NS}}}g")

    def panel(self, id, title, x, y, w, h, *, kind="plain", dashed=False,
              description="", block_type=""):
        return Panel(self, id, title, x, y, w, h, kind, dashed, description, block_type)

    def text(self, x, y, text, size=16, **kwargs):
        return label(self.root, x, y, text, size, **kwargs)

    def port(self, id, side="bottom"):
        n = self.nodes[id]
        x, y, w, h = n
        return {"top": (x+w/2, y), "bottom": (x+w/2, y+h),
                "left": (x, y+h/2), "right": (x+w, y+h/2)}[side]

    def connect(self, start, end, *, via=(), from_port="bottom", to_port="top"):
        points = [self.port(start, from_port), *via, self.port(end, to_port)]
        return self._wire(self.global_wires, points, start, end)

    @staticmethod
    def _wire(parent, points, start="", end="", arrow=True):
        if len(points) < 2:
            raise ValueError("A wire needs at least two points")
        return element(parent, "path", d=" ".join(("M" if i == 0 else "L") + f"{x},{y}"
                       for i, (x, y) in enumerate(points)), fill="none", stroke="#475569",
                       stroke_width=2, marker_end="url(#wire-arrow)" if arrow else None,
                       class_="wire", data_from=start, data_to=end)

    def save(self, path):
        # Global routes are explicit and drawn above panel fills. They must avoid boxes.
        if self.global_wires not in self.root:
            self.root.append(self.global_wires)
        footer = element(self.root, "g", id="provenance")
        source = self.source_label or "LibreYOLO model implementation"
        label(footer, 50, self.height-55, f"Source: {source}. Revision {self.revision[:12]}.",
              13, fill="#587783")
        label(footer, self.width-50, self.height-26, "libreyolo.com", 20,
              fill="#087e98", weight=700, anchor="end")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(self.root).write(path, encoding="utf-8", xml_declaration=True)
        self.root.remove(footer)
        return path


class Panel:
    def __init__(self, diagram, id, title, x, y, w, h, kind, dashed, description, block_type):
        self.diagram, self.x, self.y = diagram, x, y
        self.root = element(diagram.root, "g", id=id, transform=f"translate({x},{y})")
        if block_type:
            self._inspect(self.root, id, title, description, block_type)
        element(self.root, "rect", x=0, y=0, width=w, height=h,
                fill=diagram.tints.get(kind, "#fff"), stroke="#9caab0", stroke_width=1,
                stroke_dasharray="5 4" if dashed else None, class_="outline")
        label(self.root, 16, 29, title, 21, weight=700)
        self.wires = element(self.root, "g")
        self.ops = element(self.root, "g")

    def _inspect(self, e, id, title, description, block_type="", source_url=None):
        attrs = {"class": "inspectable", "tabindex": "0", "role": "button",
                 "aria-label": checked(title), "data-label": checked(title),
                 "data-description": checked(description), "data-node": id,
                 "data-block": block_type, "data-source": source_url or self.diagram.source_url}
        e.attrib.update(attrs)

    def box(self, id, x, y, w, label_text, *, h=49, detail="", kind="plain",
            description="", block_type="", center=False, font_size=16, source_url=None):
        if id in self.diagram.nodes:
            raise ValueError(f"Duplicate node id: {id}")
        self.diagram.nodes[id] = (self.x+x, self.y+y, w, h)
        g = element(self.ops, "g", id=id)
        self._inspect(g, id, label_text, description, block_type, source_url)
        element(g, "rect", x=x, y=y, width=w, height=h, rx=2,
                fill=self.diagram.palette.get(kind, kind if kind.startswith("#") else "#fff"),
                stroke="#94a3b8", stroke_width=1, class_="outline")
        tx, anchor = (x+w/2, "middle") if center else (x+12, "start")
        label(g, tx, y+(19 if detail else h/2+5), label_text, font_size, weight=600, anchor=anchor)
        if detail:
            label(g, tx, y+min(h-8, 37), detail, 12, fill="#435d67", anchor=anchor)
        return id

    def sum(self, id, x, y, *, description="Elementwise addition"):
        if id in self.diagram.nodes:
            raise ValueError(f"Duplicate node id: {id}")
        self.diagram.nodes[id] = (self.x+x-13, self.y+y-13, 26, 26)
        g = element(self.ops, "g", id=id)
        self._inspect(g, id, "Add", description)
        element(g, "circle", cx=x, cy=y, r=13, fill="#fff", stroke="#475569",
                stroke_width=1.5, class_="outline")
        label(g, x, y+6, "+", 21, anchor="middle")
        return id

    def text(self, x, y, text, size=14, **kwargs):
        return label(self.ops, x, y, text, size, **kwargs)

    def port(self, id, side="bottom"):
        x, y = self.diagram.port(id, side)
        return x-self.x, y-self.y

    def connect(self, start, end, *, via=(), from_port="bottom", to_port="top"):
        points = [self.port(start, from_port), *via, self.port(end, to_port)]
        return self.wire(points, start=start, end=end)

    def wire(self, points, *, start="", end="", arrow=True):
        return self.diagram._wire(self.wires, points, start, end, arrow)

    def dot(self, x, y):
        return element(self.wires, "circle", cx=x, cy=y, r=2.7, fill="#475569")

# Authoring and packaging

The helper uses the Python standard library. It uses the earlier palette explicitly selected by Xuban. It writes an SVG; it does not run models, fetch weights or guess architecture. It is optional. Read it if a layout needs more than the methods below. A custom SVG is equally valid when it follows the visual rules and carries the metadata needed by the viewer.

## Minimal drawing API

```python
import sys
sys.path.insert(0, '/path/to/libreyolo/skills/libreyolo-make-diagram/scripts')
from svg_diagram import Diagram

d = Diagram('Model variant', 'Classification, 224 × 224 input',
            width=1800, height=1200, revision='actual library commit',
            source_label='libreyolo/models/family/nn.py',
            source_url='https://github.com/LibreYOLO/libreyolo/blob/COMMIT/libreyolo/models/family/nn.py')
p = d.panel('stem', 'Stem', 50, 220, 410, 450)
p.box('input', 90, 65, 230, 'Input', detail='3 × 224 × 224')
p.box('conv', 90, 160, 230, 'Conv 7×7', detail='64 × 112 × 112',
      kind='conv', block_type='Conv', description='Convolution with stride 2.')
p.connect('input', 'conv')
d.save('/outside/library/model.svg')
```

This is an API illustration, not a complete diagram. Replace all example scope and provenance. Real outputs need the complete model and expanded block definitions.

- `Diagram.panel(id, title, x, y, w, h, kind='plain', dashed=False, description='', block_type='')` returns a panel. Give block-definition panels a meaningful `block_type` so selecting them can highlight their occurrences.
- `panel.box(id, x, y, w, label_text, h=49, detail='', kind='plain', description='', block_type='', center=False, font_size=16, source_url=None)` uses **panel-local** coordinates. IDs are global and unique. Typical mini-operation boxes use `h=32..44`, `font_size=14`, `center=True`.
- `panel.sum(id, x, y)` draws a circular residual addition. `panel.dot(x,y)` draws a true wire junction, not a textual separator.
- `panel.connect(start, end, via=(), from_port='bottom', to_port='top')` uses **panel-local** bend points. All ports are the side midpoint. For multiple concat inputs, use `panel.wire([...], start='...', end='...')` to enter separate positions.
- `diagram.connect(...)` uses **absolute** bend points for cross-panel edges. These paths draw above panel backgrounds, so route them through open corridors and check they do not cross text or boxes.
- `panel.wire(points, start='', end='', arrow=True)` draws an explicit polyline. Include `start`/`end` IDs when the wire connects interactive nodes, so highlighting works. `arrow=False` is useful for a junction trunk.
- `panel.text(x,y,text,size=14,fill=...,weight=...,anchor='start')` adds labels and notes. `diagram.text(...)` uses absolute coordinates.
- `diagram.port(id, side)` gives absolute coordinates; `panel.port(id, side)` gives coordinates relative to that panel.
- `diagram.save(path)` embeds the logo and writes a standalone XML SVG. Header occupies y=0..200; reserve 90 units at the bottom for provenance.

`C`, `H`, `W`, `n`, heads, queries and channel symbols must be defined. For a table, place each column at its own x coordinate; SVG collapses runs of spaces. Keep captions outside wire corridors. A straight `connect` between misaligned ports is diagonal: align the boxes or supply explicit orthogonal bend points. At a merge, use separate target ports so the inputs do not look like one already-joined wire.

The helper rejects decorative separators and Unicode arrows in rendered text. It does not measure fonts or detect line crossings. Browser inspection is still required.

Useful operation kinds: `conv`, `conv2d`, `bottleneck`, `norm`, `activation`, `concat`, `split`, `pool`, `aggregate`, `spp`, `attention`, `linear`, `plain`. A custom hex fill is allowed for a new operation type. Panel tints: `conv`, `pool`, `bottleneck`, `aggregate`, `attention`, `plain`.

## Wrap the SVG

```sh
python3 /path/to/skill/scripts/wrap_svg.py /output/model.svg --output /output/index.html
```

The HTML embeds the exact SVG. It works as a local file or through a static server. Keep the `.svg`, `.html`, model-specific builder and validation record together in an artifact directory outside the library. Use a localhost-only server during review. The viewer provides PNG download; verify the saved file, not just the button click. Browsers may restrict several automatic downloads from one origin.

For hand-authored SVG, add `class="inspectable"`, `tabindex="0"`, `role="button"`, `data-label`, `data-description`, `data-source`, `data-node` and optionally `data-block` to clickable groups. Put `class="outline"` on the group's own outline. Edge paths use `class="wire"` and `data-from`/`data-to`. Text and node outlines should use inline SVG attributes so static exports do not depend on page CSS.

## Evidence to keep

Record the library revision, scope, source paths, shapes observed by execution, and whether the model ran without external weights. Include the command that rebuilds the diagram. Do not copy the library implementation into the output record.

A browser screenshot and PNG check should establish that all the block definitions fit and connect correctly. A runtime forward pass establishes shape facts; it does not prove the hand-drawn topology. Check both.

## Crowded-route check

Run `python3 scripts/check_routes.py /output/model.svg` to find long coincident or closely parallel wire segments. Defaults are 20-unit lane spacing over at least a 60-unit shared run. It reports geometry to review; it is not a pass/fail proof. A same-source trunk may be legitimate, but should use one explicit trunk and a junction rather than accidentally coincident duplicate strokes. Distinct tensors must not be visually merged. The checker supports M/L polylines and translate-only groups and reports skipped paths. Review skipped paths and the actual fitted view too.

During the initial trials, both tiny lane spacing and exactly overlapping horizontal fan-in segments caused the user's complaint. Check both axes and the whole route, not just the last few pixels at a receiving port.

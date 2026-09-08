# Diagram rules from the YOLO9-T experiment

These are the implementation rules distilled by the agent from the 5 September 2026 design session. The human's exact wording is in [HUMAN_SPECS.md](HUMAN_SPECS.md). The approved result is [the YOLO9-T SVG](../assets/approved-yolo9-t.svg).

## What the human asked for

- A branded diagram at the bottom of each model's docs, with clickable blocks.
- All the information still present in a screenshot, as in the familiar YOLOv8 architecture PNG.
- Block internals resembling that reference's actual schematics, not prose cards.
- No decorative dots between names and feature levels.
- Colors closer to the YOLOv8 reference.
- Removal of AI-sounding presentation and wording, using the humanizer discipline.
- No Unicode arrows in text. Drawn diagram arrows are explicitly wanted.

This is a style preference for LibreYOLO diagrams. It is not a claim that punctuation proves who authored a text. Technical mathematical notation and genuine graph symbols remain appropriate.

## Drawing

Use compact rectangular operation boxes with flat fills and thin borders. Slight corner rounding (about 2 SVG units) is fine. Plain white main stage areas and lightly tinted block-definition panels separate content. Dashed thin outlines identify compound block boundaries. Avoid dashboard cards, shadows, glowing borders, gradients, decorative badges and ornamental numbering.

A compound block is a **miniature directed graph**, not a list with textual arrows:

- Draw every split branch separately. Label split features or widths when helpful.
- Route skip connections around operations they bypass.
- Draw concat inputs as separate paths to separate ports. A wire crossing is not a connection; use a small junction dot only for an actual branch.
- Draw residual additions as a circle with `+` and explicit inputs.
- Show sequential pooling as sequential operations with taps. Do not turn sequential pools into parallel input branches.
- State repeat counts without hiding the structure of the repeated unit.
- Draw both box and class towers, or the appropriate output mechanism for the task. Keep train-only and inference paths distinct and visibly labeled.

Use consistent alignment, clear arrowheads and orthogonal routes. Avoid unnecessary long detours, but do not confuse this with the human's complaint about **Unicode** arrows. A long drawn skip connection may be necessary. Never shorten it by changing the graph's meaning.

## Settled palette

Xuban explicitly selected the earlier colors after comparing them with the blue/orange trial. Use the palette in the approved YOLO9-T PNG/SVG for new diagrams. The monochrome and complementary trials are historical experiments, not active alternatives. Reopen the palette only if the human asks.

| Operation | Fill |
|---|---|
| Conv | `#8cdef5` cyan |
| Conv2d, Linear | `#f7c9a7` peach |
| Bottleneck / RepNCSP / RepConvN | `#f6c9a7` peach |
| Normalization | `#d4f4a6` green |
| Activation | `#a8ece2` mint |
| Concat | `#ffd448` yellow |
| Split | `#c6dfa9` green |
| Pooling | `#e8d9f7` lavender |
| Aggregation stage | `#ffe49b` pale yellow |
| Attention / SPP | `#a8cdf4` blue |

Use the same color for an operation across models and insets. Keep the reference's subtle panel tints. For a genuinely new operation, reuse a suitable existing category rather than inventing a new palette. Labels and wires carry the exact meaning; color is a reading aid. Do not assign colors by per-model frequency or flatten everything into pale blue.

## Parallel connections and zoom

Closely packed thin parallel wires are a demonstrated failure: they merge into one apparent line in fit view and separate erratically on zoom. Check both views.

- Start with visibly separated orthogonal lanes, roughly 20-24 SVG units apart on an 1800-2000-unit poster, with about 2-unit strokes. At the intended fitted size, aim for at least 5-6 screen pixels between independent lanes. These are layout defaults, not model facts.
- Allocate a real routing corridor. Widen column gaps or move a collection node nearer the sources rather than squeezing four lanes into a narrow gutter.
- Give a four-input node enough height and separate input ports. Do not bunch distinct inputs into one tiny attachment point.
- Check horizontal collector segments as well as vertical lanes. Separate input routes must not coincide on an unlabeled segment.
- Never merge separate tensors into a single shared wire with a junction dot unless an operation actually combines them. A deliberately grouped connection must say which separate tensors it carries and where the actual Concat occurs.
- For crowded graphs, compare an expanded-lane layout, an explicitly labeled collection/bundle, and named continuation connectors. Continuation labels must match visibly at both ends and keep the full information on the poster. Prefer the simplest option that preserves traceability.
- Do not depend on hover, selection or deep zoom to reveal that four wires exist. A screenshot of the fitted diagram should make the input count unambiguous.

## Labels and prose

Use ordinary technical labels: `RepNCSPELAN (P3)`, `64 × 80 × 80 (n=3)`, `Concat with B4`. Define notation such as `g`, `C` and token counts where it is first needed. Check that parenthetical labels fit their boxes.

Do not use:

- The middle dot character as a textual separator, such as between a block name and `P3`.
- Unicode arrow characters in **text**, including captions, headers, footers, prose or operation sequences. Rewrite as `then`, `with`, `from ... to ...`, or use separate drawn nodes. Do not substitute ASCII arrow strings or chevron chains.
- Em/en dashes as decorative punctuation, emoji, slogans, theatrical hooks or sales language.
- Invented names such as “architecture atlas,” numbered all-caps section titles, repeated promises that everything is complete/visible, or a heading followed by a sentence repeating it.

Keep proper model/module names, kernel sizes, repeat counts, mathematical `×` and residual `+` symbols. Factual scope, provenance and validation limits are useful. Use short neutral prose, not a forced casual voice. If a humanizer skill is available, apply it to the captions; these explicit rules make this skill usable without a machine-specific dependency. Never humanize quoted human requirements or change technical facts for style.

## Branding and export

Use the actual LibreYOLO logo and wordmark with a plain model title. Keep a small `libreyolo.com` footer. Include the exact variant, input geometry and source revision. Branding should identify the diagram, not compete with it.

Use SVG shapes and text, with the logo embedded as a data URI. An SVG export should be useful offline. PNG export must come from that same SVG and retain all panels. Clicks may explain a block or highlight paths, but no hover, accordion, click, zoom level or animation should be required to expose core architectural information.

The inspection panel must not cover the poster. Put it beside the drawing on wide screens and outside the drawing on narrow screens. A small screen cannot make a dense poster readable at once: provide fit view, full-size zoom and export while keeping the whole graph present. For a docs embed, adapt the frame height and provide a full-diagram link.

## Why the first attempt failed

The rejected attempt used eight equal text cards with textual flow recipes, muted dashboard styling, a slogan and repetitive explanatory copy. The accepted revisions replaced these with explicit nested graphs, operation colors, ordinary labels and visible branching. Removing those actual patterns improved it; simply calling a diagram “human” would not.

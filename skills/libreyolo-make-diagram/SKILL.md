---
name: libreyolo-make-diagram
description: Create branded, technically verified LibreYOLO model architecture diagrams with expanded block internals, an interactive SVG viewer, and matching SVG/PNG exports. Use for model diagrams, architecture posters, or a diagram at the bottom of a model docs page.
---

# Make a LibreYOLO model diagram

Make a technical diagram a reader can follow and reuse in a paper. The entire graph and all block definitions must remain visible without interaction. Clicks add explanation and connection highlighting; they do not reveal essential architecture.

This skill belongs to the **library**, alongside the model implementations that determine its facts. Website integration belongs to `LibreYOLO/libreyolo-website`. Generated experiments go outside the library working tree. Creating a diagram does not imply permission to publish the website or open a PR.

## Before drawing

- Read [the visual rules](references/design-rules.md) and **view** [the approved YOLO9-T poster](assets/approved-yolo9-t.png). The [SVG](assets/approved-yolo9-t.svg) supplies its geometry. The earlier colors are the settled palette. Apply the later parallel-wire routing rules while preserving that visual style. Use this approved LibreYOLO example as the reference; do not fetch third-party source code or copy another project's diagram.
- The human's dated, verbatim preferences live in [HUMAN_SPECS.md](references/HUMAN_SPECS.md). They are the source of the style decisions, not text to place on the diagram. Read them when resolving a design choice or changing this skill. Do not rewrite the quotes.
- Select **one family, size, task, input shape and execution mode**. Give the exact scope in the diagram. Do not relabel another variant's graph. A request for a family can start with one clearly identified representative variant.
- Read the in-tree implementation, config, and relevant contracts. Respect the repository's licensing policy. Pin the source revision and paths. Distinguish library behavior from the original paper, optional training paths, and fused/exported behavior.

## Families and variants

For a family with a shared topology and variable widths/depths, provide **both** a symbolic family diagram (defined variables plus a variant table) and concrete variant diagrams with all numbers resolved. The concrete variant should be the default docs view: readers should not have to substitute width/depth multipliers mentally. Read [the variant rules](references/variants.md) before drawing a family.

Generate these from the same architecture description and variant data. A selector chooses a complete, precomputed drawing; export includes the selected variant name and its exact values. Do not pretend a topology change is only a channel multiplier. State which variants share the symbolic graph and give structurally different variants their own graphs.

## Establish the architecture

Record the stage sequence, every branch and merge, repeats, kernel/stride settings, tensor shapes and outputs. List the compound block types and recursively expand them into smaller operations. Stop at standard primitives such as Conv2d, normalization, activation, pooling, attention matmul/softmax and linear projection. Every nontrivial compound name needs a visible definition, including nested blocks.

For repeated identical blocks, `n=3` is acceptable if the repeated unit and its connections are explicit. Show residual additions, split outputs and individual concat inputs as drawn paths. For transformers, include tokenization, positional information, attention, MLPs, residuals and the actual decoder/query path; do not invent a CNN neck. For a classifier, show its pooling/classification output rather than a detection head.

When feasible, instantiate the **in-tree model without pretrained weights**, run a CPU forward pass and inspect shape hooks. Match the model's supported input shape; do not assume 640 everywhere. Test shape-dependent concatenations and head dimensions. If execution is unavailable, state what was checked from source and what remains unverified. Do not claim a runtime check based on comments or expected shapes.

## Draw and package

Use an SVG as the drawing source. Keep text as SVG text and the logo embedded. Prefer an intentionally laid-out poster over an automatic node editor. Adapt the layout to the architecture; do not preserve empty space just to match a template's grid.

The optional [Python helper](scripts/svg_diagram.py) supplies the approved palette, panels, boxes, explicit wire routes and metadata. See [its usage](references/authoring.md). It does not infer architecture or choose a layout. Author a model-specific build script outside the library tree so the diagram can be reproduced.

Produce:

1. A standalone `.svg` containing the full drawing, dimensions, logo, source revision and scope.
2. A self-contained `.html` using the same SVG. [wrap_svg.py](scripts/wrap_svg.py) adds mouse/keyboard inspection, fit/100% views and SVG/PNG download. It also reports its height to a same-origin docs iframe.
3. A `.png` exported from that SVG, normally at 2× resolution. Keep the entire drawing and branding. Never use generative image tools for labels or architecture graphs.
4. A short validation record outside the poster with source files/revision, observed shapes, checks actually run, unresolved limits and reproduction commands.

The interactive page and image are two uses of **one drawing**, not separately maintained designs. The default is both outputs; honor a user's request for only one. Output format is a recommended implementation choice, not a human quote establishing an exclusive format.

## Check before handing over

- Visually inspect the complete poster and the dense block insets. Fix clipped labels, wires through text, missing branches and arrows that touch the wrong port. Use [check_routes.py](scripts/check_routes.py) to flag long close/coincident segments, then inspect parallel lanes at fitted and full sizes: separate tensors must not visually collapse into one wire. A parsed SVG or HTTP 200 is not visual verification.
- Inspect the actual PNG export. It must include the same block definitions and source details as the SVG, with no sidebar or toolbar covering it.
- Click a stage and a nested block. Test Tab/Enter selection, reset and zoom. Highlights must keep all other content legible. Check the browser console.
- Scan **rendered text** for decorative middle dots, Unicode arrows, slogans and repetitive filler. Graphical arrow paths and junction dots are correct and must stay.
- If embedding in docs, preserve the full diagram height and supply a link to the standalone view. Do not leave essential blocks clipped by a fixed-height scrolling iframe. Use the website repo's rules, including localization and route indexing; experiments should be noindex.
- Report the artifact links, exact model scope and actual validation. Do not call the result published unless it was deployed and checked.

If the user requests multiple agents, give each one a distinct model and output directory, this skill, and the same approved reference. Review their actual artifacts for both architecture and visual consistency before presenting a gallery. Do not delegate merely because this paragraph exists.

## Generalization and evidence

This is an agent-guided architecture-diagram workflow, not an automatic model-to-layout converter. The initial independent trials covered YOLO9-C detection, RF-DETR Nano detection, and ResNet-18/34 classification, including symbolic and resolved family views. CPU shape checks passed; manual visual review caught routing and caption problems that required correction.

For another architecture, reuse the drawing grammar and palette, then establish its actual topology from source and verify it. Adapt the layout and primitive definitions instead of forcing the model into a CNN template. The route checker only detects some geometric problems; it does not prove graph completeness, tensor semantics, text fit or aesthetics. Large, unusual or dynamic graphs still require careful visual and technical review. Do not promise arbitrary correct diagrams without that process.

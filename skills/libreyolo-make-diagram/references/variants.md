# Family overview and resolved variants

The human wants both: the family pattern with variables and a usable diagram of each specific size without mental substitution. The concrete view is the recommended default; the family overview is a separate option.

## Shared topology

Use one source of architecture structure plus a checked variant table. Resolve both dimensions and repetitions; preserve the family's actual rounding rules, channel floors, attention widths, query counts and input geometry. A symbolic label can say `C3 × H/8 × W/8`; the corresponding concrete label must contain actual numbers. Define every variable and give each variant's values in the family overview.

A variant selector must update the whole SVG, including expanded blocks, repeat labels, input/output geometry and provenance. Downloaded filenames and titles identify the selected variant. Do not change only the title or top-level channel labels. A concrete diagram can use clearly defined local shorthand for repeated equal dimensions, but should display the resolved number rather than leave width/depth multipliers unevaluated.

Do not multiply blocks indiscriminately. YOLO9 has nonlinear per-size configs. Some variants change ELAN/RepNCSPELAN or AConv/ADown. ResNet-18/34 share BasicBlock structure with different repeat counts; ResNet-50 uses a different block. A family view should either clearly show the structural alternatives or be restricted to the true shared-topology subset.

## Scope of a trial

A user asking to practice the skill with a few models has not asked for an exhaustive model catalogue. For a trial, select a useful shared-topology pair, deliver its symbolic overview plus both resolved diagrams, and identify which family variants remain unrendered. For an actual full family request, include all requested variants.

## Verification

Check the table against the in-tree configurations. Run representative concrete variants when feasible, including both ends of a changed repeat count or shape rule. Check that the symbolic view resolves to the concrete graph, including block internals. A family overview is a comparison aid, not a replacement for the concrete views.

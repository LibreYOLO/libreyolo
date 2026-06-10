"""Centralized postprocessing for all LibreYOLO model families.

One module per family. Each module is a collection of pure functions
(model outputs in -> detections dict out) moved verbatim from the
family's ``models/<family>/utils.py``; the old import paths keep
working via re-exports there.

Design rule: modules in this package must not import from
``libreyolo.models`` at module level. ``libreyolo/models/__init__.py``
eagerly imports every model class, and model modules import from this
package, so a module-level import in the other direction creates a
circular import whose outcome depends on which side is imported first.
Where a postprocess function needs a helper that lives under
``libreyolo.models`` (the D-FINE ``box_cxcywh_to_xyxy``), import it
inside the function body.
"""

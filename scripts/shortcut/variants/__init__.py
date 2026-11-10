"""User-designed LIBERO variants (T020): YAML spec → BDDL + init states → feasibility → preview → eval.

`spec` (YAML, catalog) needs only PyYAML; `generate`, `check`, `preview` need LIBERO (CPU).
See README.md.
"""

from .spec import list_variants, load_manifest, load_spec, variant_rows

__all__ = ["list_variants", "load_manifest", "load_spec", "variant_rows"]

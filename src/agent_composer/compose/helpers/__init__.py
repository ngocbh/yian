"""Shared, kind-agnostic helpers for the Compose front end.

The bottom layer of `compose/`: the primitives every other compose module builds on,
none of which knows about a specific node kind or a specific graph-stage pass. Collected
here so the layering is legible — `helpers` imports only the engine core (`typesys` /
`expr` / `nodes` / `compile`), and everything else in `compose` imports DOWN into it.

Modules:
  - `errors`      the `LoadError` carried by every load-path failure (message + optional
                  `SourceRange`), the one exception type the CLI boxes as a diagnostic.
  - `reader`      `NodeReader`, the located cursor over a parsed `nodes:` mapping: reads a
                  field's value AND records its `SourceRange` from the span tree.
  - `spanloader`  the YAML-text -> located span tree front door (line/col-aware parse).
  - `source`      `SourceFrame` + the parse-free span-map helpers (`_boundary_field_ranges`
                  / `_section_lines`): render-only source-location metadata built on
                  `load_with_spans` + `NodeReader`, never the parser.
  - `types`       `read_type` / `read_flow_inputs` / `InputDecl`: the `outputs:`/`inputs:`
                  source side read into one recursive `Type` (nested records native).
  - `uses`        `UsesRef` / `parse_uses_ref`: the `uses:` child-flow reference parser.

Layer: bottom of `compose`. Imports the engine core only; imported by `compose.nodes`,
`compose.stages`, `compose.loader`, and the package `__init__` re-exports.
"""

from __future__ import annotations

"""Shared, kind-agnostic helpers for the Compose front end.

The shared lower layer above `compose.source`: primitives later Compose phases build on,
none of which knows about a specific node kind or a specific graph-stage pass. Collected
here so the layering is legible: `helpers` imports the engine core (`typesys` / `expr` /
`nodes` / `compile`) and the lower `compose.source` boundary; later Compose phases import
down into it.

Modules:
  - `errors`      the `LoadError` carried by every load-path failure (message + optional
                  `SourceRange`), the one exception type the CLI boxes as a diagnostic.
  - `source`      `SourceFrame` + parse-free location helpers over an existing
                  `SourceCursor`; render-only metadata, never a YAML parser.
  - `types`       `read_type` / `read_flow_inputs`: the `outputs:`/`inputs:` source
                  side read into recursive `Type` / `Parameter` values.
  - `uses`        `UsesRef` / `parse_uses_ref`: the `uses:` child-flow reference parser.

The sibling `compose.source` package exclusively owns YAML parsing and its private
source graph. Layer: `source <- helpers <- nodes <- parse <- desugar <- edges <-
validate <- build <- loader`. The package `__init__` only re-exports the public surface.
"""

from __future__ import annotations

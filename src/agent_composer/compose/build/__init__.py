"""The build/conductor phase: boundary synthesis + the post-parse assembly pipeline.

The LAST compose phase before the loader. Two modules:

- `boundary.py` — `synthesize_boundary_graph`: mint the `START_ID`/`END_ID` boundary
  nodes + their edges (END_ID producer edges + START_ID root edges) over the FULL
  reconciled body-edge set, plus parity-clean boundary wiring.
- `pipeline.py` — `_assemble`, the conductor that drives the whole post-parse pipeline
  (build every node kind-blind -> type/exhaustiveness checks -> edge inference ->
  reference/assert validation -> boundary synthesis -> cycle/handle checks ->
  `CompiledFlow.from_parts`), plus `ComposedFlow` (the compiled+validated result) and the
  flow `outputs:` reader.

Ladder position: `build/` may import every earlier phase (helpers/nodes/parse/desugar/
edges/validate) and the engine core; it is imported by `compose.loader`, never the
other way around. This `__init__` deliberately carries NO re-exports (import from the
`boundary`/`pipeline` submodules directly) to avoid any import-order surprise.
"""

from __future__ import annotations

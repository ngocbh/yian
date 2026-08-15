"""The build/conductor phase: boundary synthesis + the post-parse assembly pipeline.

The LAST compose phase before the loader. Three modules:

- `boundary.py` — `synthesize_boundary_graph`: mint the `START_ID`/`END_ID` boundary
  nodes + their edges (END_ID producer edges + START_ID root edges) over the FULL
  reconciled body-edge set, plus parity-clean boundary wiring.
- `pipeline.py` — `_assemble`, the conductor that drives the whole post-parse pipeline
  (build every node kind-blind -> type/exhaustiveness checks -> edge inference ->
  reference/assert validation -> boundary synthesis -> cycle/handle checks ->
  `CompiledFlow.from_parts`), plus `ComposedFlow` (the compiled+validated result) and the
  flow `outputs:` parser. It associates retained inline bodies with their final
  descriptors, registers lazy builders through the loader-owned callable-scope seam,
  and uses one success-only descriptor cache for on-demand signature dependencies.
- `inline_inputs.py` — strict structural type inference for parameters of generated
  inline-composite child flows. It distinguishes top-level binding strings from nested
  literal strings, walks nullable record paths, proves MAP item types only on demand,
  and refuses unproven types rather than inventing `Any`.

Ladder position: `build/` may import every earlier phase (helpers/nodes/parse/desugar/
edges/validate) and the engine core; it is imported by `compose.loader`, never the
other way around. This `__init__` deliberately carries NO re-exports (import from the
`boundary`/`pipeline` submodules directly) to avoid any import-order surprise.
"""

from __future__ import annotations

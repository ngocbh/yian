"""The build/conductor phase: boundary synthesis + the post-parse assembly pipeline.

The LAST compose phase before the loader. Three modules:

- `boundary.py` — `synthesize_boundary_graph`: mint the `START_ID`/`END_ID` boundary
  nodes + their edges (END_ID producer edges + START_ID root edges) over the FULL
  reconciled body-edge set, plus parity-clean boundary wiring.
- `pipeline.py` — `_assemble`, the conductor that drives the whole post-parse pipeline
  (build every node kind-blind -> type/exhaustiveness checks -> edge inference ->
  reference/assert validation -> boundary synthesis -> cycle/handle checks ->
  `CompiledFlow.from_parts`), plus `ComposedFlow` (the compiled+validated result) and the
  flow `outputs:` parser. It stamps authored node-header origins, then invokes each
  descriptor's immutable source-origin hook while its located cursor is available. CODE uses
  that hook for its whole `code:` value and ordered exact literal-line sites; other descriptors
  currently keep the no-op default. After attachment, the shared `BuildContext` exposes the
  builder's read-only registered-site lookup so a CODE syntax error can resolve the same
  source-neutral line selector used at runtime. This neither reparses the document nor freezes
  the still-growing catalog. It also stamps source-backed ordinary `input:` keys and
  values, and source-backed reserved MAP/WAIT wiring values through the file-owned catalog
  builder. Only ordinary input keys stamp `Parameter` origins; each of
  those retained wiring values stamps its canonical `Binding`. It aligns retained assertion
  item cursors with source order. Flow desugaring receives an origin callback while the
  conductor derives its aligned temporary ranges from that same cursor sequence; node
  validation receives one site callback that returns origin and range together.
  Full root and multi-node-definition flow assertions are then reused by START/END, while
  compact-root and flat single-node-definition assertions stay node-local through their
  projected node cursor. After output rewrites it stamps source-backed bare/mapped root,
  multi-node-definition, and inline-child output values on the final
  `FlowOutput.binding`; boundary synthesis reuses that exact binding, and output mapping
  keys stay origin-free. Generated compact-root and single-node-definition
  reexports remain origin-free because their authored output declaration is a contract site,
  not a retained binding occurrence; contract-origin stamping remains deferred. The conductor
  also associates retained inline bodies with their final descriptors, registers lazy
  builders through the loader-owned callable-scope seam, and uses one success-only
  descriptor cache for on-demand signature dependencies.
  Private assembly leaves `ComposedFlow.origin_resolver` unset because the shared
  document catalog may still grow. The loader alone freezes and composes the exact
  returned-flow resolver after the physical-file root is complete.
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

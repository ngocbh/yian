"""The edges phase: derive the flow's graph edges from descriptors + wiring.

Given the parsed/desugared descriptors and the flow-owned wiring, this package
derives the graph's edges:

- `producers.py` — the SHARED producer helpers (`_ref_producer`,
  `_binding_producers`): the node ids a `${...}` reference reads from.
- `data.py` — `infer_data_edges`: a data `Edge` per `${<id>.output[.…]}` ref.
- `ordering.py` — `infer_ordering_edges`: `depends_on:`/`runs_after:` ordering gates.

Ladder position: `edges/` imports helpers/nodes/parse/desugar + the engine core
(`compile.model`, `expr`). It NEVER imports `validate`/`build`/`loader`. The LATER
`build/boundary.py` imports `edges/producers` — that is a legal later->earlier edge;
`edges/` never imports `build`.
"""

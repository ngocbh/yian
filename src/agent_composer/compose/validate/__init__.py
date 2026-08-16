"""The validate phase: cross-node checks over the assembled flow.

Ordered AFTER edges in the loader pipeline (`helpers <- nodes <- parse <-
desugar <- edges <- validate <- build <- loader`). Given the Compose-owned build
results, the derived edges, and the flow-owned wiring, this package runs the passes that
reject an ill-formed flow before it is compiled:

- `references.py` — `validate_references`: every `${...}` ref resolves to a real
  producer/input + a valid record field.
- `cycles.py` — `reject_cycles`: the graph is acyclic.
- `wiring_parity.py` — `check_wiring_parity`: the flow-owned wiring matches the
  built graph.
- `human_questions.py` — `validate_human_questions`: the `human_input` questions
  surface is well-formed.
- `call_types.py` — `check_ref_map_types`: each `call`/`map` binding's source
  Type is compatible with the child input's declared Type.
- `loop_types.py` — `check_loop_type_contract`: a `loop`'s carried record obeys
  the `'a -> 'a` type contract by comparing each already-retained proven body field
  with a statically proven carried source.
- `asserts.py` — `LoweredAsserts` + `lower_flow_asserts` (flow-level `asserts:`,
  lowered onto the synthesized START/END boundary nodes without replacing the
  originated carriers) and `validate_node_asserts` + `_POOL_HEADS` (the per-node
  `asserts:` contract). Node validation accepts a source-neutral callback that returns
  each authored occurrence's origin and temporary range together by source-order index;
  it never navigates Compose cursors itself.
- `handles.py` — `check_case_handles`: a `CaseNode`'s handles align with its
  outgoing control edges.

Post-build passes may consult `BuildResult.node_range` / `field_ranges`, and assertion
validation may receive a temporary descriptor/callback range. Those coordinates stay in
Compose and are discarded before `CompiledFlow`; field-local output evidence needed at
runtime lives only in the source-neutral `Node.output_contract` sum, while semantic nodes
and assertion carriers keep only opaque origins.

Ladder position: `validate/` MAY import `helpers`/`nodes`/`parse`/`desugar`/
`edges` and the engine core (`compile.model`, `compile.validation`, `expr`,
`typesys`). Shared output-contract inference lives in `helpers`, so validation
does not reach through the child resolver for it. The `validate -> desugar` edge
(`desugar.inline_calls.desugar_call_directives`) is an acknowledged legal
later->earlier edge. NOTHING in helpers/nodes/parse/
desugar/edges imports `validate`. Only `build`/`loader` (and the package facade)
import this back.
"""

from __future__ import annotations

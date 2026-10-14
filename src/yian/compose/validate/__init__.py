"""The validate phase: cross-node checks over the assembled flow.

Ordered AFTER edges in the loader pipeline (`helpers <- nodes <- parse <-
desugar <- edges <- validate <- build <- loader`). Given the Compose-owned build
results, the derived edges, and the flow-owned wiring, this package runs the passes that
reject an ill-formed flow before it is compiled:

- `value_nesting.py` — `check_written_values`: no node binding, case branch, or flow
  output is written out nested deeper than a yian value may (a record output's fields
  one level in); runs on authored values first, before any pass walks them recursively.
- `case_branches.py` — `check_case_branches`: a `case` branch names a node or calls a
  flow, and a `case` has no `output:`; a value branch, an unknown bare word, and a case
  `output:` are load errors that show the value entry or tag to write. Runs before the
  desugars, on the authored branches.
- `references.py` — `validate_references`: every `${...}` ref resolves to a real
  producer/input + a valid record field.
- `cycles.py` — `reject_cycles`: the graph is acyclic.
- `wiring_parity.py` — `check_wiring_parity`: the flow-owned wiring matches the
  built graph.
- `human_questions.py` — `validate_human_questions`: the `human_input` questions
  surface is well-formed.
- `call_types.py` — `check_call_map_types`: each `call`/`map` binding and `loop` seed
  is compatible with the child input's declared Type.
- `loop_types.py` — `check_loop_type_contract`: a `loop`'s carried record obeys
  the `'a -> 'a` type contract by comparing each proven field the body returns
  with a statically proven carried source.
- `asserts.py` — `LoweredAsserts` + `lower_flow_asserts` (flow-level `asserts:`,
  lowered onto the synthesized START/END boundary nodes without replacing the
  originated carriers) and `validate_node_asserts` + `_POOL_HEADS` (the per-node
  `asserts:` contract). Node validation accepts a source-neutral callback that returns
  each authored occurrence's origin and temporary range together by source-order index;
  it never navigates Compose cursors itself.
- `handles.py` — `check_case_handles`: a `CaseNode`'s handles align with its
  outgoing control edges.
- `narrowing.py` — `compute_narrowing`: the union narrowing facts each node gets from
  the CASE branches that route it, and the per-node views of the producer and
  flow-input contracts with each narrowed base rebuilt (`Narrowing.for_node`).
- `typed_bindings.py` — `check_typed_inputs` / `check_typed_outputs`: a typed node
  input or flow output can fit its declared type (literal text exactly; a reference
  unless its inferred type is disjoint, leaving a narrower one to the runtime check);
  `infer_binding_type` returns the type one binding is proven to produce.
- `reserved_sources.py` — `check_reserved_sources`: a MAP `over:` reference infers a
  list and a WAIT `until:` reference a type that overlaps `date` or `datetime`; a
  literal is left to the parse stage and an unproven type to the run-time check.
- `constructors.py` — `constructor_errors`: a `${T(...)}` constructor call that can never
  succeed (mixed keyword and positional arguments or several positional ones, a field
  `T` doesn't have, a missing required field, or an argument whose inferred type is
  disjoint from the declared one).
- `skips.py` — `SkipModel`: which nodes of a flow level the CASE routing may skip, and
  whether a set of them may skip together; `null_skipped_outputs` gives a flow output whose
  producers may all be skipped the explicit null default; `check_skipped_arithmetic` rejects
  arithmetic over a value that may be null from a skip while its reader runs.
- `tool_inputs.py` — `stamp_tool_input_types`: each TOOL node records the proven
  inferred type of each input binding, for the run-start check against its tool.

Post-build passes may consult `BuildResult.node_range` / `field_ranges` / `field_labels`,
and assertion validation may receive a temporary descriptor/callback range. Those
coordinates and labels stay in Compose and are discarded before `CompiledFlow`; field-local
output evidence needed at runtime lives only in the source-neutral `Node.output_contract`
sum, while semantic nodes and assertion carriers keep only opaque origins.

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

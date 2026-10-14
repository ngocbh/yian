"""The desugar phase: FLOW-LEVEL rewrites that run BEFORE the build loop.

A desugar pass takes a flow's parsed descriptors (and its `output:`/`asserts:`
sections) and rewrites them into ordinary descriptors + bindings that the build
loop then sees as author-written — pure load-time sugar, no new runtime kind. The
passes:

- `value_entries` — a tagged `nodes:` entry (`total: !int ${a.output + b.output}`) is a
  value, read at parse time through the reader `parse_nodes` is given: it lowers to a CODE
  node with the entry's id whose one input, named after the entry, holds the value typed
  by its tag, so the typed-binding checks, edges, and origins apply to it unchanged.
- `inline_bodies` — occurrence-local located overlays isolate YAML aliases and
  merged mappings, then retain and mask inline CALL/MAP/LOOP child bodies without
  editing authored anchors or values. The retained child output keeps its source cursor so
  final assembly can stamp the post-rewrite `FlowOutput.binding` value, never its mapping
  key. It assigns each retained child a stable
  reserved identity from its complete lexical owner path; authored private-prefix
  targets are rejected before registration.
- `inline_calls` — whole-value `call(<flow-id>, kw=...)` directives in any binding
  position mint a synth `call` node and rewrite the binding to `${<synth>.output}`.
  The same pass crosses raw flow assertions into ordered `Assertion` carriers; an
  injected source-neutral callback stamps each source-order occurrence, and a rewrite
  changes only `.expression` while preserving `.display` and `.origin`.
- `adaptive_questions` — a `human_input` `adaptive_questions:` block lowers to a
  synth composer agent + a rewritten gate.
- `case_targets` — a `case` then:/else: whole-value `call(...)` route target mints a
  synth call node and rewrites the route target to its id.
- `case_outputs` — every `${<case>.output}` binding ref expands to a coalesce over
  the case's branch leaf targets.

A pass that lifts a node out of an authored node (an inline call in a node's binding, the
inner calls of a nested branch call, the adaptive composer) records that node as the synth
descriptor's `host`; after the build, `compose.edges.hosts.host_control_edges` gives every
lifted node its host's CASE routes, `depends_on`/`runs_after` edges, and data gates on the
host's other inputs in one place. A branch target (`then: call(...)`), a value entry, and
a flow `output:`/`asserts:` call have no host.

Ladder position: desugar imports DOWN/peer only — `helpers`, `nodes`, `parse`,
`source`, and the engine core (`expr`, `events`, `compile`, `nodes.*`). It NEVER
imports the later phases (`edges`, `validate`, `build`, `loader`, `cases`); nothing
earlier imports it back. Only the loader (and the later phases) import this package.
The file-owned callable table and one-build cache stay in the loader; desugar only
produces retained private-child declarations.
"""

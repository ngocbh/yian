---
name: engine
description: >-
  Use when implementing, extending, or changing the Yian flow engine —
  anything under src/yian/ (node kinds, runtime/scheduling, typed state,
  run-log recovery, compile/loader, expressions) or its contract. Enforces the
  workflow: design from the functional model (OCaml as reference), fit the engine
  invariants, plan, then implement in small tested steps. Trigger on requests like
  "add a … node kind", "implement … in the engine", "change how the runtime …".
---

# Engine development workflow

The goal is a *functional language of agents* — a small, general language for
data transformation whose leaf computations may be LLM-backed AGENT nodes.
Every engine change follows the same loop — **never jump straight to code.**

**Companions:** [`reference.md`](reference.md) — the node contract
(`Output | Route | Grow`), the `NodeKind` table, the OCaml-analogue map, the
non-negotiable invariants, the layer ladder, and a design-note template.
[`templates/node_kind/`](templates/node_kind/) — a copy-paste skeleton + a
file-by-file [`WIRING.md`](templates/node_kind/WIRING.md) for adding a new
authorable node kind.

## The mental model: you're extending a functional language

The engine is a **small functional language for data transformation** (full
framing + OCaml concept map in [`src/yian/README.md`](../../../src/yian/README.md)).
The public vocabulary lives in [`docs/agent.md`](../../../docs/agent.md). Hold
this model whenever you design anything:

- A **flow is a function** — typed inputs in and one typed output value out. An
  **agent** is the conceptual umbrella for a composable transformation represented
  by a node, a flow, or a child flow. The concrete **AGENT node** (`kind: agent`)
  is the LLM-backed leaf.
- Flows **compose**: CALL/MAP/LOOP apply child flows recursively. Their `nodes:` +
  `output:` form is front-end sugar for the same application: Compose lowers it
  to a file-private child flow; it does not introduce a runtime kind.
- Computation is **pure at the boundary**: a node returns one `Output | Route |
  Grow`; the engine records and applies that outcome. Only `Output` binds a value.
- Source-neutral semantic operations live in `nodes/contracts.py`: `Binding`,
  `Parameter`, `Assertion`, `ValueContract`, and `PartialRecordContract` carry
  optional opaque origins; `Binding` also carries an optional declared
  `ValueContract` for a tagged slot (`qty: !int ${size.output.qty}`), whose origins
  travel with the binding's and whose type is manifest-backed. `ValueTemplate` is
  the one container source the binder evaluates: an authored mapping or list whose
  leaf strings are templates and which may hold nested typed `Binding`s; each leaf
  string is its own input group (`<param>/<JSON pointer>`), so edges, skips, and
  co-skip stances treat each nested span as its own input. A plain dict or list
  source stays a literal;
  `iter_origins()` and `map_origins()` touch only declared provenance slots, and
  `typesys.Type` remains provenance-free while arbitrary payloads stay opaque.
- Numeric `Type.bounds` is one optional inclusive pair on INTEGER/NUMBER; an open
  end is `None`, never both. Validate shape and values through the existing type
  system. Integer endpoints are exact, up to `YIAN_MAX_INT_DIGITS`; NUMBER endpoints need
  finite, lossless float representation. Compatibility requires interval
  containment (through `typesys.values.numeric_interval`, an open end as
  infinite), and inline type copies, nested schemas, and labels retain the
  refinement. Omit absent bounds from manifests so unbounded declarations keep
  their identity.
- NaN, infinity, and complex numbers are not yian values.
  `typesys.values.reject_nonfinite(value, where)` finds one at any depth and
  raises `TypeCheckError` with its path; the first boundary rejects it. The
  binder fails a resolved value or filled default with `binding.evaluate` (temporaries inside an expression are not
  checked); `run_flow` raises `ValueError` for a run input before the log exists;
  `LLMConfig` rejects a non-finite value. Every run-log codec (`canonical_typed`,
  `canonical_seed`, `encode_pool`) rejects one, with no opt-in, so a node output
  fails `contract.output` at the field path and a growth seed fails
  `contract.growth`. Never record null in its place. An int too large for a
  float slot fails the type check at its path. A model reply is a boundary too:
  `run_agent`'s `check_reply` fails a tool call's non-finite arguments with
  `node.execute` before any call runs; structured `float` slots set
  `allow_inf_nan=False` and freeform values are checked after `_unwrap`, so the
  reply self-corrects under `retries:`; with a transcript `invoke_model` rejects a
  non-finite raw reply (metadata included) before storing it and observes the
  effect `failed` instead of crashing the codec.
- An int longer than `YIAN_MAX_INT_DIGITS` (default 4300) digits is not a yian
  value. `reject_nonfinite` finds one too, so every boundary above rejects it
  (`<where>[ at <path>]: integer with N digits is larger than yian allows (4300
  digits)`); `reject_long_ints` checks ints only. The message never holds the
  digits. Never call `sys.set_int_max_str_digits`. Parse int text with
  `parse_int_text` (raises `IntTooLongError`), and read JSON with
  `parse_int=parse_json_int` where values cross a boundary, so a long token becomes
  an `OverLimitInt` marker the following check reports. Count digits with
  `int_digits`/`long_int_digits`, never `str()`; show a long int with `value_repr`
  (`<int with N digits>`; any other repr is cut to `YIAN_MAX_SHOWN_VALUE_CHARS`, 200,
  by `shown_text`, and an error's path stays exact, cut only when rendered); pass failure text through `rewrite_digit_limit_text` so
  Python's hint never reaches a message or the model. Temporaries inside an
  expression are not checked; writing one as text is an `ExpressionError` with the
  same message. `decode_record` fails a run record holding one with a clear error.
- A yian value nests at most `YIAN_MAX_VALUE_NESTING` (default 100) levels,
  counting itself; each dict, list, tuple, model, or dataclass is a level and a
  `TypedValue` adds none. `reject_deep_nesting(value, where, *,
  enclosing_levels=0)` checks iteratively, before the non-finite walk, at the same
  entries: binder (`binding.evaluate`, MAP `over` included), AGENT variable
  default (`node.execute`), `run_flow` inputs and `RunResult.backward` feedback
  (`ValueError`), and the model reply: `check_reply` fails deeper tool-call
  arguments with `node.execute` before any call runs, and a deeper structured
  value self-corrects under `retries:` (with a transcript it is rejected in
  `check_reply`, so it is never stored). Live success admission checks a node
  output whole (`contract.output` at the path); a record or list built around a
  value is a level deeper, so a MAP's result list or a record flow output can
  fail at its END. Only the kind-blind `commits_input_record` trait (START) may go one level
  over (the same trait makes a node with no output contract read its committed files
  through its typed parameters).
  Replay and the run-log codecs never apply the limit, so older logs replay.
  A value written out in a flow file, `env:` values included (`check_env_values`), is
  checked at load first (`compose.helpers.value_nesting.check_written_value`, located
  `LoadError`), before any recursive load walker, so YAML aliases can't crash the load;
  `check_env_values` also rejects an `env:` value that contains itself through a
  recursive alias (`typesys.values.recursive_entry_path`), which no fingerprint encodes;
  the run-time checks remain for computed values and flows built in Python. A record
  type written in `output:` counts each record plus its fields' `type_levels` (records
  and lists of the resolved type) against `current().max_type_nesting` in
  `compose/helpers/types.py`, before descending. `read_typedefs` measures each typedef
  the same way in dependency order (`_measure_typedef`) and bounds a chain of typedefs
  naming each other at the same limit (`bad typedefs:` at the typedef and field); its
  walks are iterative, so long chains and cycles never raise `RecursionError`. `type_for`
  bounds the resolved type's `type_levels` too (`... once its typedef names are expanded`).
  An authored AGENT `output:` is also held to `current().max_agent_output_nesting`
  (`YIAN_MAX_AGENT_OUTPUT_NESTING`, 16, lower-only), counted with
  `typesys.values.type_levels` and checked in `compose/nodes/agent.py` before
  `type_to_schema`, since pydantic's schema generation recurses per level.
  An expression has the same bound (`current().max_value_nesting`), which
  `parse_expr` checks iteratively before any other walk; an operator chain is one level.
- Documents yian reads nest at most `YIAN_MAX_YAML_NESTING` (default 200)
  levels; `_FlowLoader.get_event` counts collection events and raises a located
  `ComposerError` past it, before PyYAML's recursive composer runs out of stack.
  They expand to at most `YIAN_MAX_YAML_NODES` (default 100,000) nodes;
  `_FlowLoader.compose_document` counts the composed graph iteratively before
  construction, each node once per place it appears (aliases and `<<` merge sources
  in full), and raises a located `ComposerError` at the first mapping or list, in
  post-order, past it, since every later walker, the fingerprint, and the run log cost
  per expanded node. A recursive alias counts 1; once the running count passes the
  limit under one, it raises at the deepest node on the path already over the limit,
  up to the document root. `read_plain_scalar` is not counted.
- A flow's CALL, MAP, and LOOP bodies nest at most
  `YIAN_MAX_CALL_NESTING` (default 50) levels, checked at load in
  `_resolve_child`: before resolving, against the calls active in the file being
  loaded (`call_nesting_scope` per `_load_flow`), and after, adding the built callee's
  depth (`call_nesting`, iterative). A chain of `uses:` imports is held to the same
  limit in `_CallableScope.resolve`, counted apart: before loading against
  `_ACTIVE_IMPORTS` (spans files, never reset by `load_flow`), and after, adding the
  loaded flow's `ComposedFlow.import_depth`. Load, run setup, and replay walk the baked tree
  recursively (about 10 frames per level; a flow copy shares its `Type`s, so record depth
  adds none), so the cap keeps a run inside the default recursion limit. Runtime growth
  depth (`state.depth`) is unrelated and uncapped.
- `AgentNode.variables` reuses ordered `Parameter` declarations, each with an
  exact contract and explicit typed default. Reject duplicate names and input
  collisions. Resolve a detached default record once per invocation, then render
  inputs plus variables without evaluating default strings as bindings.
  `Node.declared_variables()` exposes the local declaration inventory, empty by
  default. Ownership is authored execution `static_path` plus variable name;
  rendering requires no scope. Include nonempty declarations in manifests and
  their origins in AGENT's existing traversal hooks. This inventory does not
  activate backward or modify model configuration.
- `AgentNode.system_prompt` is an optional template rendered from the same
  record as the prompt; it replaces `DEFAULT_SYSTEM` on every model path, with
  skills and then memory appended after it. Keep it out of manifests when unset.
- Nodes own local origin enumeration/rewrite and expose static children through
  `origin_children()`. `compile.origins` combines those hooks with flow wiring
  and outputs, rewriting an isolated copy by identity and returning a finalized
  copy whose hash containers are rebuilt. Shared or cyclic graphs remain safe
  without a `NodeKind` branch.
- Compose descriptors have a kind-blind `with_source_origins(...)` attachment
  hook for semantic sites beyond the common node header. Most descriptors keep
  the no-op default. CODE registers one whole-`code:` fallback plus an ordered
  tuple of exact literal-line origins, transfers only those opaque references to
  `CodeNode`, and extends its local origin traversal/rewrite accordingly. Inline
  CODE compiles an unpadded wrapper once, retains its recursive code objects, and
  attributes the deepest traceback frame owned by identity (`is`), never filename
  or structural equality. Wrapped lines map through the exact-line tuple, then the
  whole scalar and node fallback. Compose uses that same selector for syntax errors.
  The origins and retained code-object cache are manifest-neutral.
- CASE registers every indexed `cases[N].when` plus optional `on:` site before
  lowering. Each ordered `Case` retains its predicate origin; generated wiring
  bindings retain the source occurrence that produced them. `CaseNode` attributes
  a predicate exception by its own `forward` frame and active `Case` identity,
  with node fallback outside predicate evaluation. Extend CASE's local origin
  traversal/rewrite for these carriers; they remain manifest-neutral.
- AGENT node schema selection leaves only a plain `str` on the free-text path. An enum
  string (`Literal(...)`) uses the generated scalar `value` wrapper, structured retries,
  and unwrapping to the bare tag. Structured-output failures reuse
  `Node.output_contract.origins`; do not add another carrier. On retry exhaustion, accept
  a Pydantic location (including an immediate native-parser cause, but no deeper cause
  chain) only when exact generator and generated-schema validation frames prove
  ownership. Normalize only the first error: record paths stay unchanged, scalar `value`
  and list `items` are generated wrappers, and the result must be valid for the expected
  `Type`. Resolve through `ContractOrigins.at()` and retain node fallback for
  tool/provider errors, non-validation failures, or source-free contracts. Never change
  the exception or retry policy for diagnostic attribution.
- `diagnostics.ResolverTable` composes opaque namespaces through exact
  new-origin-to-delegate/old-origin routes without parsing keys or resolver
  fallback. `compile.rebase_flow_origins` maps each distinct reachable old key
  once and returns the rewritten graph with its exact table; source-free graphs
  must carry no resolver. Lexical key construction remains a front-end
  responsibility.
- `diagnostics.Failure` is the frozen source-neutral semantic failure product:
  stable code/message, optional execution node, ordered outer-to-inner opaque
  origins, and optional exception-name/traceback debug data. Its exact closed
  codes are `binding.evaluate`, `binding.type`, `contract.input`,
  `assertion.failed`, `node.execute`, `contract.node_result`, `contract.output`,
  `contract.route`, `contract.growth`, `graph.terminal_unreachable`,
  `loop.max_exceeded`, `agent.max_tool_iterations`, and `provider.auth`. Exception
  class names and traceback text must never decide behavior. The evaluator assigns
  `binding.evaluate`, `binding.type`, `contract.input`, `assertion.failed`,
  `node.execute`, and `contract.node_result` by operation stage; a node-body failure
  takes its code from the node's `exception_code` hook (default `node.execute`; LOOP
  and AGENT name their own codes). The dispatcher assigns `contract.output`,
  `contract.route`, and `contract.growth` from the closed `Output`, `Route`, and
  `Grow` arms, never from an exception class. A
  `TypeCheckError.path` is a contract-relative tuple of record names and actual
  list indices. Supplied-input and output type rejection resolve it through
  `ContractOrigins` (exact, nearest ancestor, then default); output uses the
  effective commit target's contract. Missing contract provenance falls back to
  the parameter or executing node, while invalid defaults retain their default
  origin. Non-type output canonicalization uses the contract default. Route and
  growth use the executing node origin. All paths feed one
  `NodeFailedRecord` boundary carrying the failure, optional evaluation evidence,
  and forward capture when retention is enabled;
  reducer state, `NodeFailed`, `RunFailed`, and `RunResult.failure` retain that
  same semantic authority without flattened aliases. That durable boundary
  collapses any valid `OriginRef` subclass to exact `OriginRef(key)`, because
  subclass identity is not part of the single-key wire. Host control, stream
  callbacks, unexpected binder faults, malformed compiled/setup state,
  unsupported growth kinds, and
  arbitrary builder/compiler/spawner-or-fragment-copy/digest/materialization/reducer/log faults
  escape. The first route that skip-floods END latches a causal
  `graph.terminal_unreachable` failure on its `NodeRoutedRecord`; replay
  structurally revalidates the route and restores that recorded value, while an
  explicit node failure wins immediately. A loaded flow's outputs never trigger it:
  Compose (`compose/validate/skips.py`) null-defaults an output whose producers a CASE
  may skip; the runtime path stays for hand-built graphs.
- `diagnostics.Diagnostic(code, message, frames, notes, node_id, cause_type,
  traceback)` is frozen, resolved presentation-only data. For Compose-authored
  flows, `ComposedFlow.diagnose(failure)` is the sole authoring-front-end failure
  resolution seam: it copies `code`, `message`, `node_id`, `cause_type`, and
  `traceback`, resolves the complete trace in existing
  outermost-to-innermost order, and preserves duplicate origins. Empty traces
  remain unlocated without a warning. A missing, incomplete, malformed, or
  raising resolver for a non-empty trace discards every frame and adds exactly
  `Source presentation was unavailable.` Runtime traces are built only through
  `Node.failure_trace(operation_origin=None)`: generated outer-to-inner
  `origin_context`, then the supplied operation origin or the node origin,
  omitting only an absent local suffix and preserving order/duplicates. The
  evaluator, dispatcher, and terminal-unreachable producer use that node-owned
  rule without a kind branch. Durable records preserve the complete trace, and
  replay restores it even when current origins were rebased. Runtime CLI failures
  must call `ComposedFlow.diagnose()` and pass its result to the shared diagnostic
  renderer; never infer source from runtime ids or graph topology.
  `ComposedFlow` stores no parallel source snapshot.
- **Compose load boundary:** `LoadError` carries exactly one already-resolved
  `Diagnostic`. Compose owns the source label/text/range and builds every frame
  before raising; definition/import context is prepended immutably. A load-error
  renderer consumes only `error.diagnostic` and never receives YAML text or a
  path separately. Retained `SourceRange` values are the only Compose location
  authority; Compose imports them directly from `diagnostics`, and `events` does
  not re-export them. Do not introduce line-map fallbacks. This is distinct from
  runtime `Failure.trace`, which resolves through `ComposedFlow.diagnose()`.
- `Node.origin` is the optional source-neutral identity of an authored node
  context. It has no execution meaning, is unrelated to LOOP's `origin_id`, and
  stays outside manifests/fingerprints. Compose stamps physical node headers;
  source-free/generated nodes default to `None` and clones preserve a value.
- `Node.origin_context` is generated diagnostic ancestry, ordered outermost to
  innermost. Direct/static nodes start empty. It is excluded from authored-origin
  traversal/rewriting and from manifests/fingerprints/expansion digests. The
  compile-owned CALL/MAP/LOOP builders prefix child clones with
  `spawner.failure_trace()`; MAP boundaries receive it too. LOOP respawn copies
  only the driver's outer context, so `failure_trace()` appends the loop origin
  once. New spawners propagate this context while constructing their fragment;
  runtime never gains a kind branch for it.
- Compiled flow-owned wiring has one shape: `Flow` construction wraps each
  direct source as `Binding`, and `FlowOutput.binding` owns declared outputs.
  Python builders may pass raw values at construction, but must not write raw
  values into an already-constructed flow. Compose stamps source-backed bare/mapped
  root, multi-node-definition, and inline-child output values on the final binding after
  rewrites, and END reuses that exact carrier; output mapping keys stay origin-free.
  Generated compact-root and single-node-definition reexports remain origin-free because
  they have no authored binding occurrence. Keep that flow-output binding rule distinct
  from node-output contract provenance: the same physical top-level `output:` is projected
  onto the synthesized leaf and owns its contract origins.
- Compiled node assertions also have one shape: ordered `Assertion` carriers.
  Compose creates one carrier per authored flow/node list position and derives
  its `.origin` (`OriginRef`) from the retained item cursor; duplicate text
  therefore remains occurrence-distinct. Evaluate
  `.expression`, report `.display`, and preserve `.origin`: inline-call and
  flow/CALL lowering replace only the expression, while child cloning and LOOP
  respawn retain the carrier fields. START/END reuse the lowered flow carriers.
  `Node.run` diagnoses a failed assertion through that exact carrier's origin.
  An expression that doesn't parse, or calls an unknown function or a builtin with
  arguments it doesn't take, is malformed setup, not a failed assertion:
  `run_flow` and `resume_flow` (unfinished run) raise `ExpressionError` before any
  record is written, and `Node.run` raises it from setup validation before the body.
  The same holds for a CASE `when` and a LOOP `while`/`until` set in Python.
  A binding, flow output, or prompt set in Python gets the same check from `run_flow`
  and `resume_flow` before any record is written (`Node.validate_prompts` checks a
  node's prompts), but not from `Node.run`.
  Do not add another assertion text/origin map or positional range list.
- Compiled `Node.params` has one shape: ordered exact `Parameter` carriers. The
  flow owns their `Binding` sources; parameter contracts/default presence/
  coercion policy stay source-neutral, and rewrites preserve all origins.
  `compose/helpers/types.py` is the one source-aware recursive type compiler. For
  authored root/definition flow inputs, the declaration key owns `Parameter.origin`,
  the type value and authored nested record fields own `ContractOrigins`,
  and the default after the tag owns `default_origin`; START reuses those exact carriers.
  Scalar alias/list type expressions fall back to the contract root. Python/source-free
  declarations and inferred inline parameters stay source-free, while external aliases
  use ordinary lexical rebasing. Keep that separate from ordinary node-input wiring:
  final assembly stamps its key on the sink `Parameter` and its bound value on the
  corresponding `Binding`. Source-backed reserved MAP `over:`
  and WAIT `until:` values are stamped only on their `Binding` because they have no
  formal parameter. CASE's kind-owned lowering stamps each generated searched
  `Binding` with the first predicate origin that introduced the read, or stamps
  `__on` from `on:`; its generated `Parameter` carriers remain source-free.
  A present typed `Binding` is checked after its source resolves and before the
  parameter contract: `build_value_as` against `Binding.contract`, then the payload is
  delivered with only its file and object representation converted (`normalize_files`:
  a file's JSON form at a `file` position becomes a `FileRef`, a nested `FileRef` at an
  `object` position becomes its mapping; an `int` bound `!float` stays an `int`), as
  `T(...)` constructors and nested `ValueTemplate` bindings do; a miss is `binding.type`.
  The binder never converts a string: only `run_flow`'s `coerce_inputs` converts the
  top-level flow's supplied inputs, so a CALL/MAP/LOOP edge carries values as is.
  Supplied contract failures select
  `Parameter.contract.origins.at(TypeCheckError.path)` before the parameter
  fallback; default failures keep default-site ownership. Never add a parallel
  source map.
- Compiled `Node.output_contract` is one closed sum: an exact `ValueContract`, a
  closed `PartialRecordContract` with optional field contracts, or `None` for an
  unenforced output. Runtime success and replay enforce the same sum; behavior that
  needs a whole semantic type uses only the exact arm. The run log stores nested files
  as their JSON form, so committed values (pool, `NodeSucceeded` observer, trace writer
  value) are decoded through the commit target's contract (`output_contract_file_type`),
  computed once when the commit is prepared: each `file` position holds a `FileRef`
  live and on replay, and the record bytes don't change. The run result reads through
  `RunState.result_contract` (the flow's inferred output contract); LOOP reads its
  carried record through its output contract and a partial loop's omitted field through
  the body input. Backward contexts and `lookup_source` keep canonical JSON; untyped
  outputs and seeds keep nested files as mappings. Dotted reads go through `field_of`
  (a dict key or a file's `uri`/`mime`/`name`, never `getattr`). Manifests exclude
  origins. Compose descriptors, child signatures, producer maps, and validators
  retain the contract and project `.type` only inside type
  algorithms. Compose and runtime growth reference validation share the source-neutral
  closed-field walk from `nodes/contracts.py` (`output_contract_field_error`): it
  rejects a provably absent record field and any field on a scalar or list type;
  an opaque `object` (no declared fields) and `Any` stay unchecked; a `file` reads only
  `uri` (`str`), `mime` and `name` (`Optional(str)`), typed by
  `typesys.values.file_field_type`, which inline-binding inference,
  `project_output_contract`, and expression typing share. On a union it
  reads a field every member has and rejects a member-only field; runtime growth passes
  `partial_union_fields=True` because loader-time narrowing is not recorded on the
  fragment. The walk also checks prompt and node-assert local names, each typed by its
  binding's contract under the node's narrowing view (`local_input_contracts`), a flow
  assert's `${input.X...}`, and a builtin's result (`builtin_result_type`, `str` for each
  shipped builtin; `builtin_field_errors`). Compose's narrowing pass
  (`compose/validate/narrowing.py`) gives each node a view of the producer/flow-input
  contracts under the CASE branches that route it (an `on:` tag branch, or what a
  searched `when:` proves true; a true `isinstance` keeps the base members `T` shares if
  it shares some and no other member overlaps `T`, else is `Exact(T)`, and is none if the
  base already fits `T`; `isinstance_members`; a true `<ref> != null` gives the
  non-null type, and makes each record above a nested path non-null and present; two
  `Exact` facts meet to the narrower); the reference, call-type, and
  exhaustiveness checks read `Narrowing.for_node` (the reference check uses `for_param`
  for guarded searched-CASE params). Every condition (node and flow asserts, searched
  `when:`, LOOP `while:`/`until:`) narrows by short-circuit logic through two shared
  helpers: `compose/helpers/condition_guards.py` (syntax only: `isinstance`, tag and
  null checks, guards over them through `and`/`or`/`not`, `proves`, `guarded_refs`) and
  `compose/helpers/narrowing_facts.py` (the fact algebra, `guard_env`,
  `guarded_read_types`: a guarded read is fine when any narrowed candidate allows it;
  `impossible_checks`: a check that can never be true is a located load error, though a
  null check on a never-null value is not). A LOOP's seed and returned-field checks
  read a named body's inputs as `loop_entry_inputs` gives them: nullable at each path a
  `while:` null check proves (`while_non_null_paths`, `and` parts only) and each record
  above it; a seed only that admits must also fit the carried field type, since a
  zero-iteration run outputs the seeds.
  Runtime routing does not change. A value-producing CASE
  (read as `${case.output}`, or routed to by one) must be exhaustive (an
  `else:`, every tag of a non-nullable enum `on:`, or `true` and `false` for a
  non-nullable `bool` `on:`, compared as typed values), with a flow-input `on:` resolved
  for it. `case_exhaustive.py`'s `check_case_when_values` runs right after narrowing: an
  `on:` CASE's `when:` values are typed literals, and one the known `on:` type can't hold
  is a located load error, even next to an `else:`. It also rejects a `when:` equal to
  an earlier one (typed equality: `1` and `1.0` are equal), at the later one, for any
  `on:`. Routing-only enum coverage reads `Literal` flow inputs as well as node outputs.
  A CASE only routes (a branch names a node or calls a flow; a value branch or CASE
  `output:` is a load error). A tagged `nodes:` entry (a value entry) lowers at load to a
  CODE node with the entry's id; the runtime is unchanged. Python-authored nodes pass
  `output_contract=`; there is no parallel output-Type API. Authored `output:` syntax
  (the descriptor's normalized `outputs` field) stays unchanged. Compose injects an
  optional source-aware reader through `BuildContext`; `NodeDescriptor.output_contract()`
  invokes it only at each output-authoring leaf's established build position, so earlier
  kind-owned validation and error ordering do not move. A physical declaration supplies
  the contract root/default and exact nested record-field origins; scalar alias/list
  interiors fall back through the nearest authored ancestor to the root. Compact roots
  and flat definitions project their top-level declaration onto the synthesized leaf.
  Direct/source-free builds and generated contracts without an authorable type occurrence
  keep empty origins, while derived child contracts preserve their carriers and external
  alias views use ordinary lexical rebasing.
- `isinstance(x, T)` is an evaluator special form, not a `TEMPLATE_FNS` builtin: `T` is
  a bare type name looked up in the evaluating node's `Node.types` table, never a
  reference. Ref walks, rewrites, inference (`bool`), and flow-callee detection skip
  `T`. Every evaluation entry point takes `types=None`; node sites pass `self.types`,
  and `LoopNode.respawn` copies it. Compose fills the tables after the boundary is
  synthesized; the manifest emits `types` only when non-empty. Prompts reject it (no
  type table), as do reserved wiring (WAIT `until:`, MAP `over:`) and spawner asserts,
  which lower onto the child's END.
- Static expression typing is source-neutral: `expr.infer_expr_type` mirrors only
  result types the evaluator proves (literal/direct-ref and fixed boolean roots).
  Compose's `helpers/output_contracts.py` alone adds template shape, reference
  projection, and provenance. Unsupported computed outputs remain opaque instead
  of inheriting a reference leaf's type; inline-composite input inference stays
  strict and separate, but may project an exact known field from a partial producer.
  Direct whole references and `ChildSignature.output` preserve either output-contract
  arm. END has three modes: value (`output: ${x}` returns its one input; no
  `output:` returns `None`), record (`output: {...}` returns a dict keyed by every
  name, even one or none), or list (a MAP fan-in). `ChildSignature.returns_record`
  says whether the child's END is a record; `child_signature` shapes `output` from
  it, CALL keeps that `output`, and LOOP takes a record body's field names from that
  `output`. A typed output binding resolves to its declared contract, and END's own
  `output_contract` comes from those declared output types (exact record when every
  field is typed, else partial; `None` when none is), so commit and replay check them. Record assembly creates an exact record when possible and otherwise a
  closed `PartialRecordContract`; unknown type evidence is never concrete `None`.
  CALL preserves that carrier; a LOOP's carrier is the merged carried record (returned
  fields from the body, omitted fields from their seeds); MAP wraps only an exact child
  type. Live commit,
  replay, validators, and manifests therefore share one authority. A `None` source is
  compatible with any nullable sink, including recursively inside lists, matching
  runtime coercion.
- The **author fixes the structure** (the call graph); the LLM fills leaf boxes,
  it does not rewrite the graph at runtime.

So a new construct isn't merely “a flow feature”—it's a language construct, held
to a language's bar: **does it have a clear type, does it compose, is it pure
where it should be?** If a proposed primitive only works at the top level, or
only for one flow shape, or needs to mutate the pool — stop; that's a smell.

## 0. Orient
Read what's relevant to the change:
- [`src/yian/README.md`](../../../src/yian/README.md) — what the engine is, the layout, + the layer ladder.
- [`docs/syntax.md`](../../../docs/syntax.md) — the flow-authoring surface (YAML shape, `${...}` refs, node kinds).
- GitHub issues (`gh issue list`) — the TODO tracker (keep it current per CLAUDE.md "Zeroth rule").

## 1. Design from the functional model — look to OCaml first
Our north star is a *functional language of agents*. Design the feature as a
**language designer** would, not by porting a mechanism. Reach for OCaml's
*design philosophy and conventions*, then adapt them to our engine:

- **What's the type?** Every construct has a clear type. A flow is `'a -> 'b`; a
  new primitive must say what it consumes and what it produces. If you can't write
  its signature, the design isn't done.
- **Does it compose?** OCaml builds everything from small composable pieces
  (functions, modules, variants). Prefer the general combinator that composes over
  a one-off feature. A primitive that only works at the top level is a smell.
- **Is it pure / explicit about effects?** OCaml keeps evaluation referentially
  transparent and makes effects explicit (`option`/`result` and explicit state
  transitions instead of hidden failure). Map our durable records, recovery,
  failure, and binding onto those shapes rather than inventing ad-hoc control flow.
- **Sum types + exhaustive match.** OCaml models a closed set of cases as a
  variant and forces an exhaustive `match`. Our closed sets are `NodeKind` (kept
  closed, no registry/metaclass), `NodeResult`, the evaluator's worker result, and
  `RunRecord`. Note the runtime *core* matches on result/record variants, not on
  `NodeKind`: ordinary behavior belongs in a node's own `forward` or node-owned
  traits/hooks. Growth construction has a
  sanctioned compile-layer match: the closed `build_grow` interpreter
  interprets node-produced seeds into fragments before the kind-blind runtime
  splices them.
- **Modules with explicit signatures (`.mli`).** A package exposes a narrow,
  declared interface and hides the rest — which is just the `structure` skill's
  package charters. Honor the boundary.

Write down the OCaml analogue you're borrowing and how it maps to nodes / pool /
runtime. If there's an established functional name for the construct (combinator,
fold, continuation, effect, functor), use it.

When borrowing runtime mechanics from any prior worker engine, **borrow** the
correctness-critical parts (single-writer dispatcher, 3-state edge join,
outputs-before-successors, redirect-aware recursive skip-flood, recorded starts
and outcomes,
append-before-apply transitions, and one reducer path for live execution and
historical replay) and **drop** scale/framework baggage (external DBs, dynamic
worker scaling, plugin registries/metaclasses, heavy layering, multi-tenancy) —
and anything that fights the functional model above.

## 2. Fit it to the engine invariants
Decide keep / simplify / drop against these **non-negotiable engine invariants**.
Most are just the functional model (above) made enforceable:

- **Deterministic authored structure** — the author fixes the call graph; the LLM
  fills leaf boxes. A flow never rewrites itself at runtime. No agentic routing.
- **A flow is a function — clear typed inputs and outputs** — every flow has an
  explicit input/output signature. “Agent” is the conceptual umbrella; the
  concrete AGENT node gets no special graph contract.
- **Hierarchical / composable** *(function composition)* — a composite node can
  apply another flow (a `call:`/`uses:` reference → a child subgraph seeded from
  the parent's pool), recursively. This is how one agent composes another and how
  composite nodes are built. One-off inline composite bodies must
  lower to that same child contract with stable lexical identity and a closed
  generated-input scope. A spawner returns only a canonical
  `Grow(seed, prune)`; the compile-owned `build_grow` interprets that seed after the
  package-root run boundary injects it into runtime. Seed values are resolved data: the
  child START receives them verbatim and never re-evaluates a `${...}` inside a string.
  Never assume a node is a leaf;
  preserve recursion through compile + run and deterministic run-record replay.
- **General / expressive** — the engine **privileges no output type** (report,
  classification, decision, number, structured record — whatever the terminal
  node emits) and no domain. When adding a primitive, prefer the most general form
  that composes over a use-case-specific feature. If a real flow can't be
  expressed, that's an engine gap — fix the primitive, don't special-case the flow.
- **Node never writes the pool** *(purity)* — it returns `Output | Route | Grow`;
  the engine binds an `Output` value under the resolved logical target. No node
  mutates shared state.
- **Typed, losslessly-serializable state** *(the type system)* —
  `TypedValue` / `VariablePool`; the basis for canonical run-record payloads and
  `${...}` references. Within a refined specialized list,
  `Type.element.nullable` permits `None`; bare and non-nullable specialized lists
  still reject it. An authored `list(Any)` / `list(Optional(Any))` is `LIST_ANY` with
  an `Any` / `Optional(Any)` element; a `LIST_ANY` with no element is an unproven list,
  labelled `list(Optional(Any))`, whose items compatibility reads as `Optional(Any)`
  and MAP checks at runtime. `ValueKind.ANY` is type-only (any non-null JSON value,
  stored under its natural kind); `object`, `list(object)` items, and `LIST_ANY` items
  must be JSON values (text keys, no date objects or sets; nested `FileRef`s in an
  `object` value or `list(object)` item become mappings, and `LIST_ANY` items and `Any`
  reject them). A `file` slot reads a file's JSON form (exactly `uri`, `mime`, `name`)
  as a `FileRef`. An untyped
  node output must be JSON values or `FileRef`s, checked at `canonical_typed`. A node
  output is first read once through Python iteration into plain containers
  (`plain_containers`); the checks and the stored value use that copy. In wiring an
  `Any` source fits only `Any`/`Optional(Any)`; `type_relation` treats it as narrower
  than every type except `None` and a type whose every value holds a `file`, which are
  disjoint. Load also treats `file` as disjoint from `object` and a file-shaped record,
  while `types_overlap` counts `file` as overlapping plain `object`, `Any`, and records
  admitting its JSON form, so such an untagged union needs a tag field.
  The specialized `LIST_*` discriminator and carrier preserve
  `None` through canonical round-trips. Cross-flow compatibility recursively
  applies the nullable-source-to-strict-sink rule to list elements. A union is
  `Type(kind=OBJECT, variants=(record, ...))`: non-nullable record members in
  canonical order, compared by `==` (a `Type` is unhashable), told apart by a derived
  tag field (`union_tag_fields`) or by non-overlapping shapes (`types_overlap`). Every
  `Type` walker (value check, compatibility, labels, schemas, manifests, inline copies)
  must handle the `variants` arm; a new walker that ignores it is a bug.
- **Append-only durable authority** — a fresh run records complete initialization,
  then every node start and accepted outcome. A transition is validated and prepared,
  appended to its `RunLog`, and only then applied to `RunState`; observers become
  visible after append + apply (transient `StreamChunk`s and in-loop AGENT
  observers are the only exceptions and are never recorded; TS yian's `LiveChannel`
  frames are a separate channel: records forwarded after their append, and model
  call frames named by `node_id` + `n`, never in the log or `onEvent`). Recovery
  validates a contiguous prefix and folds
  it through the same reducer path before recording an eligible ordered retry
  frontier. Logs use schema 9; readers reject schemas 7 and 8 rather than
  migrating them, and preserve canonical bytes through preparation, replay,
  trace, and append checks. Do not rewrite accepted prefixes. The run log holds
  no conversation records; session history lives only in the backend. TS yian
  (schema 19) writes every pool write as a `pool_commit`: the run start's system
  values and START inputs right after `run_started` (applied at once; a log cut
  before the START inputs isn't resumable), and a node's output before the
  `node_succeeded` that names its seq; the success applies it, and a resume drops
  an orphan commit so the node reruns. `RunState.producers` maps each pool name
  to the seq of the accepted `pool_commit` that last wrote it (pruned with the
  pool, rebuilt by replay); a `derived` node (START, END, LOOP, CALL, MAP) gets an
  in-memory input route from it in a run with no retained trace. A derived node's
  commit (except the top-level START's and END's) holds `from`, `{pool name: seq}`,
  not its value, and the reducer rebuilds the value when its success accepts it
  (`rebind`/`rebuildDerived`, reading the pool through `poolResolver`, which checks
  each seq against `producers`). An expansion first writes one such commit per START
  of its fragment, keyed by the child START; `node_expanded` holds no seed (replay
  rebuilds it from that `from` before the prune) and the child START's success names
  its commit. The rule is kind-blind: the `derived` trait and the commit's key.
- **Defaults are read once and recorded** — every predefined default and limit is an
  entry in `yian.defaults` (`docs/defaults.md`), overridable by `YIAN_<NAME>`.
  `current()` reads the environment once per process; the CLI callback, `load_flow`,
  `run_flow`, and `resume_flow` call it first so a bad value raises `DefaultsError`
  before any work. Read a value from `current()`, hoisted out of loops; never add a
  module constant for one. `run_flow` records `current().non_defaults()` as
  `RunStartedRecord.defaults` (omitted when empty); `RunStartData.defaults` defaults
  to `{}` (built-in values), never `current()`. The reducer keeps the run's values as
  `RunState.run_defaults`, and the growth check reads `max_total_nodes` there, never
  from `current()`. Replay calls `check_recorded` before the fingerprint check and
  raises `DefaultsMismatchError` naming each differing or unknown variable.
- **Every authored value is flow identity** — `flow_manifest`/`flow_fingerprint`
  include every value written in the flow, under any name, so an edited literal
  fails the resume check. Never redact, mask, or drop a value because its name or
  text looks like a credential. No name is special at load either: compose rejects
  no value for its name. Keep secrets out of flow files by convention; credentials
  come from inputs or the host.
- **Blocking authorable effects return ordinary outputs** — an AGENT node's
  explicitly granted `ask_user` tool and HUMAN_INPUT call the run's own handler
  (passed as `tools={"ask_user": ...}`) inside their live invocation. A group
  tool may never be named `ask_user` or `load_skill`; host-filled tool groups (`tools={"box": [...]}`)
  expand before the AGENT grant check, and unresolved grants fail at run start.
  The run boundary also resolves each static TOOL node's `host_tool_call().tool_id`
  (an unknown id fails at run start) and checks the call against its tool's
  declared argument and return types (`tools.types`) before any record;
  a TOOL `output:` is the ordinary output gate, so expected outcomes are variants
  of that type and exceptions still fail the node. In an AGENT's tool loop,
  `invoke_tool_effect` turns a `ReportedToolError` into an error `ToolMessage`
  (recorded `completed`) that the model reads, and the loop continues; a pending
  stop still wins, and any other tool exception fails the AGENT. A TOOL node retries
  a `ReportedToolError` like any failure unless it is also a `PermanentToolError`.
  `FlowEngine` binds the run's `ToolSet` with `for_run(run_id, stop_event=...)`;
  `ToolSet.invoke` exposes that event to each call through a context variable, so a
  waiting tool (an MCP call) stops with its own run even on a shared tool. The
  handler takes six keywords; the model can set only `question` and `options`
  (cleaned leniently), and the rest stay `InjectedToolArg`. The AGENT
  node checks the run-owned stop event between external model/tool effects, and
  WAIT blocks on that event until its absolute UTC deadline.
  All three return `Output`; none creates continuation topology.
- **Dependency-light core (with a deliberate exception)** — keep the engine free
  of a DB and heavy frameworks; external capabilities enter through **injected
  seams** (plain callables). **Exception (by decision):** the AGENT node is *not*
  SDK-free — it runs one loop (`run_agent` in `nodes/agent/loop.py`) over its
  granted tools and skills; with neither, the loop makes one model call. The loop
  imports langchain + `yian.llm_clients` and builds its own model via
  `model_from_config`. Prefer seams for *new* external deps, but model access in
  the AGENT loop is a direct call. MCP servers enter as host `tools=` values
  (`McpServer`, a tool group); `tools/mcp.py` imports the MCP SDK only when a server
  opens, and the run boundary opens and closes the servers a run owns. Host-supplied
  `SKILL.md` skills enter the same way as memory: `run_flow(skills=...)` ->
  `FlowEngine(skills=...)` -> the evaluator caps only nodes with
  `needs_skills = True`; the AGENT serves `load_skill` like a registered tool
  and records each result as evidence. The memory context opens with the
  effective system prompt (rendered `system_prompt` or `DEFAULT_SYSTEM`) without
  the skill listing.
  A host steering inbox follows the same path: `run_flow(steering=...)` ->
  `FlowEngine(steering=...)` -> the evaluator caps only nodes with
  `needs_steering is True` (an AGENT authored `steerable: true`). The
  AGENT loop drains steers into `new_messages` at the top of each
  iteration and checks `interrupted` at safe points; an interrupt ends the node
  with a normal `Output`, so no run-log record kind changes. The host sees it
  only through live observers: a cut stream's `ModelCallFinished` has no error,
  and each skipped call emits `ToolCallFinished(error="interrupted")`.
- **Domain-agnostic** — no domain-specific field names or types in the engine;
  domain lives at the contract/boundary (the loader/builder). The `system`
  namespace is generic.
- **Closed `NodeKind` vocabulary; the core is kind-BLIND** — `NodeKind` stays a closed enum (no
  registry/metaclass), but the engine core (`runtime/engine.py` + `runtime/eval_node.py`) does NOT
  branch on it. Ordinary behavior lives in a node's own `forward`; growth construction's
  sanctioned kind match lives in `compile.expand.build_grow`. The core dispatches only on the
  closed worker-result and run-record sums and consults node-owned
  execution-semantic traits/hooks
  (`is_spawner`, `binds_per_item`, `item_source_key`,
  `bind_reserved`, `reserved_wiring_keys`, `validate_execution_setup`,
  `grow_depth_delta`, `is_loop`, `needs_stop_event`, `needs_tools`,
  `needs_steering`, `needs_steps`, `exception_code`, `skipped_output_target`)
  plus the diagnostic-only projections `exception_origin` and `failure_trace`.
  Those projections are identity-neutral. `skipped_output_target()` is pure, deterministic
  topology authority derived only from manifest-backed node state. Its base
  behavior is baked `commit_as` or the physical id; LOOP returns its origin to
  mirror dynamic `Output.commit_as`, which otherwise affects success only. A
  redirected target must have one unfinished physical owner and be an inert expanded
  spawner or no-incoming placeholder, never executable or already committed;
  every target out-edge must still be unknown. A
  ratchet test (`tests/engine/test_kind_census.py`) holds the core's `NodeKind`/`*Expansion`
  dispatch count at 0. A separate layer ratchet forbids `nodes -> compile` and
  `runtime -> compile.expand`: nodes describe seeds, compile constructs fragments,
  and runtime receives the constructor as an injected callable.
  Compile-side origin traversal follows the separate node-owned
  `iter_local_origins`, `rewrite_local_origins`, and `origin_children` hooks; these
  are not runtime capabilities.
- **Backward behavior belongs to the node** — `Node.backward(context, feedback)`
  takes recorded context and an ordered message tuple, then returns the three
  validated feedback channels. The base implementation stops with empty channels.
  START, END, and stopping LOOP own mechanical routing using captured layouts;
  forward execution never invokes this hook. A reached AGENT uses at most one
  tool-free model call from accepted context and the exact retained effective
  model config; visible tool interactions are inert evidence, never executable
  grants. No ambient/node config fallback, recall, render, or default replay; the
  one live field it reads is a marked agent's host `api_key`, which is never
  recorded, from the replayed node's `llm_config` (a call-time agent's run-local
  `host_credentials`, passed only when its recorded config is marked). A call-time
  agent's config is its bound `__llm_config` from the accepted capture, read at
  `capture_render`; a static agent's is the RunStarted map entry.
- **Reverse traversal follows accepted outcome order** — gather all incoming
  feedback before invoking an occurrence once, and validate every returned
  channel before routing any message. Use trace-owned terminal identity and
  detached contexts. Keep unreached accepted occurrences in the result inventory;
  learning coverage must not depend on feedback reachability. Project flow inputs
  from accepted START results so literals and defaults retain feedback. Durable
  attribution uses occurrence and message references; materialized downstream
  context and sort keys remain transient.
- **Pure transfer rules are shared below drivers** — `runtime.learning` owns
  root seeding/projection, message ordering, route transfer and complete
  attribution validation. `runtime.backward` owns callback invocation and the
  explicit attempt driver. Ready/replay validation must not call a node or
  materialize transcript contexts. `RuntimeBackwardState` is the sole attempt
  authority. `project_owner_contexts` is the only episode projection; forward
  execution keeps no per-memory episode inventory. Memory JSON allows
  `current().memory_json_nesting` levels (value nesting + 28, 128 by default),
  so run material at the value limit fits; a context that still cannot be built
  raises `OwnerContextError` with the owner's `BackwardContext`, and the driver
  records `memory.invalid_context` naming that occurrence and the cause (only
  root fallback feedback has no node). `propose_learning` re-validates a
  `LearnContext`'s frozen feedback and shares it, rebuilding one whose frozen
  feedback is not canonical; never deep-copy one.
- **Binding capture follows the actual read boundary** — optional input capture
  uses the existing expression walk and a paired value/presence/source resolver.
  Never reevaluate a callable or reconstruct provenance from value equality.
  Capture the final input record after wrapper defaults and before assertions or
  body mutation; incomplete binding has no usable record. Reserved hooks use
  `bind(name, source, normalize)` for routed values, while directly returned
  materialized values have literal routes. MAP supplies explicit item indices
  and projects the captured `over` route. Preserve `evaluated_keys` from the same
  template scan; an evaluated expression may still produce a literal route.
- **Feedback shape and message order stay explicit** — use ordinary string,
  dictionary, exact-length list, and `None` values. Keep whole-value text
  unchanged with structural context alongside it, and keep independent messages
  in tuples. Input messages may be structured; self and variable feedback contain
  text messages only. Validate all `BackwardResult` channels against recorded
  input shapes and declared targets before routing. Opaque registered expression functions
  stop transfer and retain local context; do not add a function-backward registry.
  Supply accepted canonical inputs to the existing routing helper when temporary
  snapshots lack their shape. This grants no producer mapping: unavailable opaque
  parents retain selected feedback/path, and literal routes remain silent stops.
- **Execution location is independent of memory ownership** — derive
  root-relative authored node/flow paths and explicit boundary roles through
  `compile.execution_layout`; do not parse runtime IDs or require a memory
  declaration or new flow ID. Apply the layout after existing run setup makes
  private child copies per callsite. Validate the complete layout, target
  identities, aliases, and existing scopes before writing any scope. Derived
  `Node.execution_scope` is explicitly excluded from authored fingerprints;
  memory keys and stacks remain separate.
- **Growth connects recorded inputs explicitly** — source-neutral fragment
  recipes select CALL/LOOP parameter routes or MAP item-index/parameter routes.
  Retain real child START/END pairs; synthetic MAP fan nodes are not child-flow
  boundaries. Include connections and execution topology in expansion validation
  and its digest. Join candidates only into fields present in the accepted child
  capture, keeping the original capture unchanged. `evaluated_keys` wins over a
  parent candidate, including an actual expression that returned null. Consume or
  prune pending connections so reused runtime IDs cannot inherit stale sources.
  A connection may also target a fragment node's defaulted, unwired input (the next
  LOOP driver's field the body omits). A spawner selects next-hop seed routes from its
  effective input route, so an omitted field keeps its lineage across iterations.
  `LoopNode.respawn(k, carried)` gives that driver a default per omitted field from the
  canonical seed, and only returned fields are wired, so live growth and replay merge
  alike; a total loop keeps its identity and digests.
  LOOP templates begin at driver zero; cloning adds outer positions while keeping
  a nested LOOP's initial driver suffix. Continuations replace their own driver
  position. Do not parse runtime IDs or change memory ownership to derive these
  execution positions.
- **Retained history follows accepted records** — the recorded `learnable` flag
  controls retention. Schema 9 requires `ForwardCapture` on every accepted node
  outcome when `learnable=True` and omits it when `learnable=False`. Model/tool evidence
  remains separate from general capture and may have no memory scope; recalled
  sections still follow the scope's recorded memory stack, one per key, in
  order. Identify executions by run ID and accepted start sequence,
  with outcome order recorded separately. Sequence zero is the root input source.
  Read value, presence, and producer from one immutable current writer entry at
  the actual binding lookup. Redirected output names retain the physical producer.
  Prepare trace deltas before append and install them only through accepted
  reducer transitions. Replay uses that same reducer; pruning preserves historical
  entries and reconstructed backward callables. Never serialize callables or
  retain discarded worker outcomes as executions.
- **AGENT capture comes from the actual invocation** — one opt-in dynamic
  snapshot retains authored/rendered prompt, resolved variables and canonical
  domains, plus the selected RunStarted effective config. Keep partial established
  locals on failure and distinguish unavailable `None` from empty values.
  Detach immediately, reject duplicate/late writes, and preserve host fault
  identity. Static capture remains separate. Accepted evidence retains base system
  prompt, ordered recall sections, and visible interactions; the private
  `_evaluation_evidence` backward-context bridge comes only from that record.
  TS yian records no evidence: each backward attempt rebuilds it from the
  occurrence's steps (kept on `TraceOccurrence`), its render with the recorded
  skills and a recall against the recorded snapshots, through the node's
  `learningMaterial` hook; a rebuilt request that misses its step's
  `request_sha256` leaves out the occurrence's prompt, sections and
  interactions, never its backward call or episodes (`ReplayDegraded`
  `rebuilt_request_differs`). `backwardContext` stays synchronous and takes the
  material, since the reducer validates backward results through it.
- **Memory setup and learning are record-authoritative** — authored
  `long_term_memory:` entries become `MemoryDecl(name, type, config, site)`; a
  memory's key is its name prefixed by the site's `uses:` scope.
  `scope_memory_declarations` adds the alias in `_rebase_external_child`, and
  `load_flow` runs `resolve_memory_layout` on the assembled root, turning a
  `MemorySetupError` into a located `LoadError`. The resolver yields one
  `ResolvedMemory` per key and one stack per placement: enclosing flows' keys,
  outermost first, then the AGENT's own, each key once. `RunStarted` records the
  key table (`memories`) and stacks (`memory_stacks`); evidence, recovery, and
  backward check scopes against them. `memory=None` is dormant, while a
  host-supplied mapping opens one run-scoped session per key with pinned opening
  snapshots and supplies one kind-blind memory-context capability, independently
  of `learnable`. SessionMemory reads latest live history after admission, not a
  frozen opening transcript. `learnable=True` retains execution only; forward runs
  no learners or learned-mutation commits. Automatic conversation updates are
  separate. Explicit `RunResult.backward(feedback)` performs the reverse fold,
  then gives every memory key exactly one learn call in sorted key order,
  including empty batches. A key's batch holds the accepted occurrences whose
  recorded stack contains it; unattributed root feedback goes only to keys owned
  by the root flow site `DeclSite((), ("root",), None)`.
- **Hosts build memory; runtime never does** — package-root
  `memory_host.build_memories(flow, root=, storage=, supplied=, factories=)`
  returns a `MemoryRegistry` with one backend per declared key, and closes the
  backends it built when the host exits it. `run_flow`, `resume_flow` and delayed
  backward check a registry's types against the table; none constructs memory. `cli.memory` only parses
  `--memory-root DIR` and repeated `--memory-storage KEY=PATH` and calls the
  builder; `cli/run.py` closes the registry after controller work drains. Nothing
  opens before runtime requests `open_run`; terminal inspection opens no backend.
  The CLI refuses a flow with a custom type. Never construct memory in runtime,
  enable learning implicitly, or change `memory=None` dormancy.
- **Episodic memory appends canonical execution history** — import the built-in
  backend from `yian.memory.episodic`. An episodic memory's config
  accepts positive `max_episodes` and `recall_limit`; an unset limit adopts the
  stored value or the built-in default, and an explicit one that differs raises
  `episodic.config_conflict`. Learning deterministically proposes one append per
  owner-scoped episode. After the attempt reaches Ready, the engine requests the
  key's batch; the backend commits it atomically and idempotently, then evicts
  the oldest entries beyond `max_episodes`. Context-free recall is
  chronological. Contextual recall ranks lexical
  overlap and then recency within its limit. Sessions retain immutable JSON
  snapshots; stale sessions may recall their pinned view but cannot learn or
  commit. Optional host-only `storage=path` preserves complete private state.
- **Profile memory rewrites one document** — import `ProfileMemory` from
  `yian.memory.profile` and a store from
  `yian.memory.storage`. The snapshot pins `profile_version` and
  `text_sha256`, never the text. Every stack holding the key recalls the whole
  document; config accepts `max_chars` and `input_budget_chars`. Learning is
  one tool-free model call over
  feedback, the current document, and budgeted episodes with their visible
  interactions; it returns `<profile>...</profile>` within `max_chars`. Commit
  is a store compare-and-set with no model call. Only `profile.version_mismatch`
  is retryable (nothing was written), and it is retried as a new backward
  attempt, never inside the backend. The store has no commit-id replay: a killed
  attempt stays `unknown`, and hosts find its `commit_id` in store provenance.
  `FileMarkdownStore` takes the same local path rule as SQLite storage.
  `close()` does no I/O; afterwards the backend and its sessions raise
  `profile.closed`.
- **Durable memory belongs to the backend, not runtime** — the episodic,
  reflective, and session backends default to `storage=None` with no I/O. With a
  path, construction only checks its shape and starts PID ownership; storage
  opens on the first `open_run` or `reset`, under the backend's lock. `close()`
  before any open does no I/O, and later use raises `<type>.closed`.
  `memory.storage.SQLiteStorage` supplies one lifetime SQLite owner and finite-JSON
  transactions. Persist validated candidates before RAM publication, including
  reset/adoption and private commit tombstones;
  preserve public snapshot formats and exact run-opening pins. Do not store host
  constructor defaults, callbacks, live handles, queues, or clients. Current host
  settings govern future work after reopening; saved memory limits and recorded
  model settings stay pinned. Session summary caches retain exact prefix proofs;
  ignore coverage that would hide newly protected recent turns. Close rejects
  active operations/queued turns immediately, but permits idle run handles and
  invalidates them. Run-handle close never releases active callback protection.
  Check process identity before locks, including cleanup, and preserve primary
  exceptions. No runtime storage branch or new log field is needed.
- **SessionMemory owns conversation state and FIFO admission** — import
  `SessionMemory` from `yian.memory.session`, not the contracts-only
  facade. `MemorySession.recall()` returns text or `MemoryContent(text, transcript)`;
  SessionMemory supplies its notes and `TranscriptSource` through this same path. API-backed AGENT execution through
  `run_flow` retains conversations when the host supplies the backend.
  The CLI builds declared built-in memories automatically.
  `memory.conversation` owns the pure fold, strict codecs, and digests;
  `memory.session` owns live views, FIFO leases, and atomic publication;
  `memory.summary` owns pure whole-turn selection and cache-prefix proofs. These stay
  below nodes/runtime/providers. Runtime keeps no conversation projection, effect
  ledger, checkpoint callback, or reconciliation channel. It recalls each key of
  the stack once and admits a transcript only from a key whose recorded
  `MemorySpec.conversation_eligible` is true; an ineligible or second transcript
  raises `MemoryRecallError`. It
  then binds authored scope plus accepted run/start identity into one lazy
  evaluator-owned `MemoryContext`. Notes stay ordered outer-to-local. A model
  reply holding a NaN or infinity is never stored: `invoke_model` observes that
  effect `failed` and raises a located error (a structured AGENT retries it,
  spending `retries:` even on a metadata NaN the model can't fix). Use
  `needs_memory_context` and `open_memory_context(system, prompt, compatibility)`;
  do not reintroduce separate recall/conversation injection or backend detection. Capture that identity before
  observers/submission; never recover it from mutable state in a worker. Keep the
  lease through stream cleanup, post-assertions, and result-shape validation;
  later dispatcher acceptance is not part of turn completion. No lease travels
  in worker results. The stateless AGENT adapter owns provider encoding/effects,
  not stored history. The run log holds no conversation records, and schema 9
  rejects older logs that did. Ordinary retry may
  resend requests and repeat tool actions. Do not reconstruct memory from the run
  log or require conversation closure before node success. See
  [the conversation and retry contract](reference.md#session-conversation-foundation).
  API summaries use `history_budget_chars` (default 32_000) and `recent_turns`
  (default 4), set as constructor keywords or YAML config.
  Preserve all raw history and the latest four prior whole admissions plus the
  active turn, even above budget. A concrete optional turn capability owns the
  derived cache; injected projection/current-model callbacks run outside the lock
  under the existing lease. No runtime or mandatory protocol change. Prepare once
  before tool binding/structured generation. Preserve model-failure identity as a
  node failure; other codec/backend faults remain capability faults. Never retry
  or truncate an oversized valid summary, and never persist summary/repair context
  as raw messages. Final structured context keeps the prepared prefix plus a safe
  active-turn projection. Reset clears cache; old snapshots never rewind it.
  One session instance holds one conversation and queue, owned by the single AGENT
  that declares its key; call sites and MAP/LOOP positions do not fork it. FIFO is
  admission order, not MAP input order. Other recalled memories stay active beside
  it. Acquire the FIFO lease before reading live
  history while keeping the run-opening snapshot pinned. Publish requests and
  observations directly in the backend; external calls and stop checks run outside
  its condition lock. Preserve observations before honoring stop; a call that
  raises while a stop is pending is observed `interrupted` (or stays `failed` if
  the stop is honored only after the failure is recorded) and the run stops (a
  host callback fault still escapes). Keep unknown requests pending across later
  invocations, never inventing success. Closing a
  run cancels its waiters but retains an active lease until its owner exits; no
  later turn may overlap the finishing external call. Released turns cannot write.
  Reset is backend-wide, fails unchanged while any turn is active or queued,
  and invalidates old-generation handles. Conversation
  updates through the optional turn capability are independent of `learnable`.
  Explicit API backward self-feedback becomes deterministic learned-note mutations
  through the key's learn/session commit. The memory learns notes only from
  invocations it admitted; join feedback execution sequence to the episode's
  root-relative owner path, never guess from a dynamic node name. No extra model
  call or runtime note projection. Commit validates the entire batch before
  atomic publication and retains idempotency bookkeeping across reset. Contextual
  recall contributes pinned opening notes and the transcript source;
  context-free recall contributes empty text with the source. Notes never rewrite raw conversations. New public snapshots use
  version 3 (one conversation view plus notes); older versions are invalid.
  API compatibility uses provider plus versioned message format; compatible
  model changes retain history. Apply current instructions/recall/tool grants
  on each call. Never dispatch old tool calls while restoring history. Keep raw
  history intact when projecting unknown tool results or inert structured-schema
  calls for provider context. Retain native raw replies before parsing, and keep
  memory/codec faults outside structured-output retry catches.
  OpenAI Responses clients preserve content blocks, including opaque continuation
  data; visible text extraction must not mutate the stored/provider message.
  API history projection v2 preserves Responses function-call blocks only when
  their ordered ids, names, and parsed arguments match canonical tool calls;
  unknown namespaces and asynchronous calls remain inert context. A projection
  change must invalidate derived summary caches without changing raw history.
  Summaries extract visible text from validated text/reasoning replies; tool
  calls, unknown content blocks, and empty visible summaries remain invalid.
  The direct OpenAI subscription construction seam in
  `llm_clients/openai_subscription.py` consumes an injected access source, requires
  a completed response before surfacing tool calls, and closes streams on failure
  or cooperative stop. It owns no login, credential store, tool loop, or history.
  Keep its request/stream hooks covered against the pinned OpenAI integration.
  Fresh-run setup snapshots environment and saved provider-scoped defaults once
  (no built-in provider or model; an agent no layer selects fails at run start,
  naming its path), then binds nonsecret connection identity (saved login or key
  variable) onto every keyed agent of the effective model map. Resolve credentials
  only from that pinned identity (OpenAI per request, others per model build);
  refresh preserves identity, while re-authentication or removal cannot silently
  redirect recovery or backward calls. Old records without connection metadata
  retain environment API-key semantics. Runtime and SessionMemory stay unaware of
  authentication methods.
  A run-layer `llm_config.api_key` (needs `provider`) never enters the effective
  map: `run_flow` pops it before the cascade; the map marks that provider's
  agents `_host_key: true` (OpenAI also records a nonsecret `host` connection),
  and `bind_host_credentials` puts the key and the run layer's header values
  only on marked agents' run-local `llm_config`.
  `resume_flow(api_key=, headers=)` and `RunResult.backward(api_key=, headers=)`
  rebind them on the replayed program; a marked config without the key, or
  whose headers lack the recorded names, fails in `split_model_config` (an
  `AuthError` naming header names only), so agents marked for the key never
  fall back to the environment. A key that marks no agent is dropped from the
  result, with its settings, and ignored by replay.
  Connection settings (`organization`, `project`: OpenAI ids, `provider:
  openai` only; `headers`: extra request headers, `dict[str, SecretStr]`
  excluded from dumps like the key) are host-only and travel with a key:
  `pop_host_credentials` takes them only beside `api_key` and `provider`, and
  authored layers reject them at load. `llm_clients/connection_settings.py`
  holds the shared name/value/id checks and refused names (credential and
  framing headers; `OpenAI-Organization`/`OpenAI-Project` for `openai`).
  `bind(host_settings=)` records `organization`/`project` as plain fields and
  the sorted header names (never values) as `_header_names`, allowed only
  beside `_host_key`. A saved login's settings go in its `_connection` as the
  optional `organization`, `project`, and `header_names` (only for `source:
  store`, `method: api_key`; ids only for `openai`), pinned by its
  `credential_id`; old records without them stay valid. A host key never picks
  up a saved login's settings; an environment key gets none. Resume and
  backward compare given header names with `_header_names` case-insensitively
  and raise `ValueError` before anything is appended; `RunResult.backward`
  without them fails the marked agents' calls with `AuthError`. Clients take
  the values as the yian-only field `connection_headers` (`exclude=True`,
  `repr=False`), added to the built SDK clients' custom headers (genai: its
  http options), never a langchain field, so `repr`, `model_dump()`, and
  `dumpd()` never show them; for `openai` the ids go as `OpenAI-Organization`/
  `OpenAI-Project` there. A model with connection headers refuses redirects:
  OpenAI-family/Azure get yian's cached `follow_redirects=False` httpx clients (never
  the shared langchain ones), and a host `http_client` must pass
  `check_no_redirects`. `call_credentials` lists every header value, however
  short, so errors scrub them like keys. OpenAI-family and Anthropic SDK clients keep only the custom
  headers yian or langchain set (`drop_environment_headers`): headers from
  `OPENAI_CUSTOM_HEADERS`/`ANTHROPIC_CUSTOM_HEADERS` and the org/project from
  `OPENAI_ORG_ID`/`OPENAI_ORGANIZATION`/`OPENAI_PROJECT_ID` are never sent; a
  `UserWarning` names any it drops, once per process (`base_client.warn_once`).
  The google client passes `vertexai=False` and rebuilds the genai client on
  the Gemini Developer API, with the resolved key, if google-genai switched it
  from the environment, so `GOOGLE_GENAI_USE_VERTEXAI` and
  `GOOGLE_GENAI_USE_ENTERPRISE` are ignored; a `UserWarning` names either when
  set, once per process (`warn_once`, shared with the header-drop warning).
  Flow runs send such headers only as connection settings (above); a header
  named in `explicit` replaces the environment's and is not reported dropped.
  The ollama client deletes the `Authorization` header the ollama SDK built
  from `OLLAMA_API_KEY` (`NormalizedChatOllama`), so no key reaches Ollama; a
  `UserWarning` says so, once per process (`warn_once`). Basic auth from
  userinfo in `OLLAMA_BASE_URL` stays.
  Azure passes `azure_ad_token=None`, so `AZURE_OPENAI_AD_TOKEN` is used only
  without a key; Anthropic clears `auth_token` when it holds a key.
  `LLMConfig` keeps the key a `SecretStr` excluded from dumps, rejects
  an empty key, and hides inputs in validation errors. `AgentNode` forward,
  AGENT backward (`backward_agent`), `ProfileMemory.learn`, and
  `ReflectiveMemory.learn` pass provider errors through
  `llm_clients.connection.scrub_call_credentials`, which removes every
  credential the call sends (`call_credentials`: the host key, else a `store`
  connection's saved login read without refresh, else the provider's first set
  key variable, plus `AZURE_OPENAI_AD_TOKEN`/`ANTHROPIC_AUTH_TOKEN` whenever set,
  though only a call given no key sends them) and masked echoes of each before
  the runtime records a failure. Learners scrub the chained cause of their
  fixed-message errors; `yian auth`'s model check scrubs the key it
  sends, its header values, and the provider's token variable. Authored layers
  reject `api_key`, the connection settings, and `_host_key` (flow and def
  levels at load); `ProfileMemory` needs `provider`
  with its key and takes the settings beside it; `ReflectiveMemory` rejects
  `api_key` and the settings and always uses the saved login or
  environment credentials. A run-layer `llm_config.base_url` is where the key goes: it is
  accepted only beside `api_key` and `provider` (`pop_host_credentials`), must be
  `https` (`http` only for local hosts), is checked as the exact stored string
  (`validate_base_url`: printable ASCII, valid port, ≤2048 chars), and is
  rejected for `ollama`. The `anthropic`/`google` clients don't follow
  redirects from it (httpx would re-send their key headers cross-origin).
  `bind(host_base_url=)` records it on the marked agents' effective configs
  (OpenAI `host` connection endpoint too); resume/backward reuse the recorded
  URL. Authored layers reject `base_url`, the cascade fails a `base_url`
  without `_host_key` or one `validate_base_url` refuses (an edited log fails
  at resume start), and `split_model_config` fails one that differs from
  its host connection endpoint. An unset `base_url` is dropped from dumps.
  The guarantee is about authored layers: a host key without `base_url` still
  goes to the provider's default or env endpoint (`OPENAI_BASE_URL`, etc.).
  Exception: `provider: azure` takes `base_url` (the resource endpoint) from
  any layer without a key, records it in the effective map, and fails at run
  start without one (`check_model_config`); `model` is the deployment. Env
  `AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_DEPLOYMENT_NAME` are not read. A host
  Azure key must bring its own `base_url` (`pop_host_credentials`).
  Request settings live in provider blocks (`llm_clients/provider_blocks.py`): one
  `LLMConfig` block per provider, each mirroring the request body of the API that
  provider's calls reach, with reserved paths yian builds. Every block is checked at
  load; the cascade rejects unknown and retired host-layer keys and merges blocks per
  block field, one level deep (`merge_llm_config`, still associative); run start
  re-checks each agent's block against the API actually called (`check_model_config`
  → `block_schema`, e.g. `OpenAISubscriptionBlock` on a ChatGPT subscription). Clients
  send the block as written (`merge_request_body`) with no translation; the only
  default is Anthropic `max_tokens`. Keep `provider`, `_connection`, `_host_key`,
  and `_header_names` top-level in recorded configs (resume reads them there).
  `llm_config` values (`${...}` in a field, block field, whole block, or whole record):
  Compose reads each layer through the tag reader (`ValueTemplate`, or `Binding` for
  a whole record; a plain dict when it holds none), refuses host-only fields from a
  value or a declared whole-record type and `base_url` beside a non-literal provider,
  type-checks sources (`compose/validate/llm_config_sources.py`), and wires an AGENT
  whose values can still contribute with the reserved key `__llm_config`
  (`Binding(ValueTemplate({"node": ..., "flow": ...}))`; `reserved_key_label` names
  it `llm_config` in messages). Input-only root-flow agents resolve at run start
  after `coerce_inputs`, only the inputs they read checked before the log;
  `find_inherited_llm_value` refuses a non-root-input flow value a child agent would
  inherit. Call-time agents get a `CallTimeLLM` (base plus restored selection) from
  `RunStarted.call_time_llm` (optional, omitted when empty; schema stays 9, older
  yian can't read logs that hold it). `bind_reserved` → `call_time_config` resolves,
  checks (`check_resolved_layer`, `llm_clients.layers`), binds the connection,
  marks a host-key provider, runs `check_model_config` on a per-invocation copy with
  secrets joined only when marked, and returns the secret-free canonical config;
  failures are located `BindingError`s (`binding.type`/`contract.input`), a missing
  host key an `AuthError` (host error). The node records it as a `NodeStep`
  `{"effective_llm_config": ...}` before memory and the model; a restart whose
  recomputed config differs raises a host fault. `model_hidden_inputs` and
  `backward_input_names` keep `__llm_config` away from the model and from feedback
  routing (memory recall, episodes, evidence). Resume checks in-flight call-time
  configs with the static map, the CLI refuses a recorded host-key selection (passed
  to `_check_resume_credentials` as `call_time_host`), its up-front key check counts
  a call-time agent whose provider is known (an Azure one with its literal
  `base_url`), and the result keeps the host key when the call-time selection names
  its provider.
- **One attempt barrier precedes commits** — accept Started, complete backward
  results and all memory proposals, then accept Ready. Commit each memory key
  once in sorted key order, even with an empty batch, accepting its positive-attempt
  request before the backend effect and observation afterward. A known failure
  stops later keys; Finished records the separate outcome. No per-key learning
  records or retry frontier.
  Keep `unrequested`/`unknown`/`succeeded`/`failed` commit observations distinct.
  Any finished attempt admits another; an unfinished one is closed as interrupted
  first. Each attempt recomputes everything using fresh sessions and original
  snapshots, and every attempt uses `ac-memory-v3` commit IDs over its own
  `attempt_id`. Never resume Ready proposals or infer retries from backend hints.
- **Delayed backward owns no idle resources** — `run.py` keeps private program
  and accepted data plus memory references. Each call reopens/replays authority,
  uses a nonblocking handle guard and durable locks, restores recorded snapshots,
  and closes resources. Preserve primary host/cleanup errors and the accepted
  prefix; poison an unrefreshable handle. Invalid submissions leave prior observed
  outcome objects unchanged. Terminal resume is inspection only, opening no
  memory and invoking no backward/learning work. Forward fields never change.
- **`Node.env` is engine-opaque** — a per-node `dict[str,Any]` of static config the compiler
  bakes at build (`{**flow_env, **node_env}`, node wins, flow-local). The engine NEVER reads it;
  only a node's own `forward()` interprets known keys (e.g. an AGENT node's `max_tool_iterations`). It is the
  kind-blind knob for "tune this node" — add a key by having the node read it, never by an engine
  branch. An AGENT node stays one live loop, so its node-local `env` remains on that invocation.
- **Single-writer invariant** — workers are pure executors; the dispatcher alone
  consumes the forward ready frontier and commits its reducer transitions. It records each start before
  submission and each accepted outcome before applying it. A pooled drive owns workers
  until its full join; abnormal dispatch exit must signal cooperative stop before joining.
  When an outcome fails the run, signal stop and publish its durable NodeFailed/RunFailed
  before the join, so a host can withdraw a sibling's pending prompt.
- **Worker defaults belong to the run boundary** — fresh CLI and Python runs use
  eight workers unless overridden. Zero selects serial evaluation. Omitted worker
  counts on resume preserve recorded authority, not the fresh-run default.
  Submit each pooled node with its own copy of the host context so nested host
  effects retain cancellation and node-local context changes do not leak.
- **Single-process CLI target** — dispatch and workers remain in-process. Durability
  is the small `RunLog` seam with memory and locked JSONL implementations; no DB,
  server, or remote scheduler is baked into the core.

## 3. Plan
Write a short plan: the **OCaml analogue** you're borrowing and its type; what you
simplify/drop and why; where it lands in the layer map
(`diagnostics` below `events`, with nodes combining `diagnostics`, `typesys`, and
`expr` before `compile ← compose`; runtime owning its scheduler-state vocabulary
in `common.py` and consuming compile's execution contracts but not
`compile.expand`); and the **seam** if it
touches an external dependency. Flag any non-obvious choice and confirm before
coding (CLAUDE.md "ask when uncertain").

## 4. Implement in small, tested steps
Per CLAUDE.md: bottom-up by dependency, **code + at least one test + run green**
before moving on. Tests live in `tests/engine/`; run with
`PYTHONPATH=src pytest tests/engine` (or `pip install -e '.[all]'` first). Keep
the engine import-clean (no new heavy deps in the core). Close the tracking
GitHub issue when the work lands (reference the commit/PR). Commit at green.

### Package layout
Follow the **`structure` skill** for where files and folders go — package charters
in `__init__.py`, `common.py` vs `utils.py`, one-way imports, no cycles. The
engine's layer ladder starts at the dependency-free `diagnostics` value spine.
`events` imports that leaf; nodes combine it with the independent `typesys` and
`expr` leaves before `compile ← compose`, while runtime keeps its scheduling
vocabulary in `runtime/common.py`. This is an instance of that framework.
The package-root `run.py` sits above compose and runtime and wires cross-layer
capabilities: notably it owns log lifecycle/recovery and injects compile's
`build_grow` into the kind-blind runtime.

`diagnostics.Failure` owns semantic failure identity, `diagnostics.Diagnostic`
owns resolved presentation data, and `diagnostics.ResolverTable` owns immutable
exact resolver routing; none knows a source format or runtime state.
`compile/origins.py` owns the
representation-neutral whole-graph walk: it visits flow-owned wiring and
outputs, delegates node-local slots and children to node hooks, and deep-copies
before rewriting. Its rebasing operation precomputes one mapping per distinct
reachable old origin and returns the rewritten graph plus a table containing
exactly those routes. Nodes never import this higher layer; binding sources,
defaults, environments, and render metadata remain opaque.

Compose source ownership is one-way too. `compose/source/` alone owns PyYAML and
the private retained document graph; one loader class reads plain scalars with
YAML 1.2 core rules plus unquoted timestamps and merge keys. Root text is
constructed once as a `SourceDocument`; each uncached external file is constructed
once too. The public cursor interface is immutable and defensive. The default file
resolver's successful resolved-path cache reuses one compiled flow for every alias
or diamond reference to that external file.
Persistent pair/arc ropes keep YAML merge topology compact; completed mapping
summaries are reuse frontiers rather than recursively rescanned descendants.
Defensive value snapshots are iterative and preserve shared/recursive aliases.
Source-aware parsing and inline extraction navigate `SourceCursor`s: physical
cursors combine definition ranges with occurrence paths (and mapping-key ranges),
virtual cursors are defensive source-free snapshots, and projected cursors retain
explicitly selected physical child sources while changing a view. Descriptors
retain derived ranges; `NodeDescriptor._finish()` copies them to Compose-only
`BuildResult.node_range` / `field_ranges`, and a desugar's author-facing names for a
synth node's fields and inputs to `BuildResult.field_labels`, for post-build validation.
Assembly discards all three before `CompiledFlow`, so semantic nodes have no YAML range or
aligned assertion-range fields; only a synth AGENT's `llm_config` label survives, as the
render-only `AgentNode.llm_config_label` for run-start messages. Compose also fills
`Node.binding_labels` (param -> the phrase a runtime binding error starts with, such as
`node 'g' on`, `value entry 'v'`, or `flow output 'r'`; not part of node identity or the
manifest) from those labels and `wiring_labels`, and for END. A synth call records its
authored binding as `CallDescriptor.call_site`; its argument bindings and the node share that
binding's origin, so an argument error reads `<site> call(f) argument 'y'` at that binding.
In a cycle error a labelled synth node with a host, inline calls included, gets a
`<synth> is generated from node '<host>'` note. Only an adaptive composer whose generated
wire into the gate alone closes the cycle is reported as its host; an inline call never is.
Never reparse a source slice or mutate
anchors/aliases/merges.
An unchanged source-backed string cursor exposes `ScalarSource`: its whole range and,
for literal `|` blocks, exact zero-based constructed-line ranges captured from the
real block token during that same parse. Other scalar styles have no exact line tuple;
virtual and value-changing cursors have no scalar provenance.
`SourceCatalog` is Compose's immutable `OriginResolver`: its private builder maps
typed semantic paths to opaque origins and retains document occurrences with key/value
anchors or semantic constructed-line indices. Resolution replays an occurrence and
reselects the range without storing coordinates in the catalog site or origin,
reparsing, or adding a parallel range map; unavailable exact lines leave the site free
for whole-scalar fallback. One builder is
shared by a physical document's root, definitions, and inline bodies.
`compose/helpers/types.py` owns the sole source-aware recursive type compiler. For
retained root/definition flow inputs it stamps the key on `Parameter.origin`, the type
value and authored nested record fields on `ContractOrigins`, and the default written
after the tag (`lookback: !int 30`) on `default_origin`; START reuses those carriers. Scalar alias/list expressions fall back
to the root. Python/source-free and inferred inline parameters remain source-free.
Assembly also supplies a Compose-owned output-contract reader through the optional
`BuildContext` seam. The descriptor calls it only from the leaf's existing contract step;
the reader selects the normalized physical `output:` cursor and delegates to the same
recursive compiler. The whole declaration becomes the default origin, physical record
fields become exact nested entries, and scalar alias/list internals use the nearest
authored ancestor and then the root fallback.
Compact roots and flat single-node definitions retain top-level `output:` through their
source-aligned node projections. A virtual/missing occurrence falls back to source-free
compilation; synthesized contracts without an authorable type site stay source-free.
Assembly separately stamps authored node headers, ordinary node-input keys and bound
values, source-backed reserved
MAP/WAIT wiring values, source-backed flow-output values, and authored flow/node
assertions directly on their canonical carriers. Assertion sites are indexed by source
order; one retained item cursor supplies `Assertion.origin`. Full root flows and
multi-node definitions retain a flow assertion cursor and reuse
the lowered carriers on START/END. Compact roots and flat single-node definitions project
their assertions onto the synthesized node, where they remain node-local. Flow-output
values are stamped after rewrites on `FlowOutput.binding`; END reuses that binding and
output mapping keys remain origin-free. Generated compact-root and
single-node-definition reexports have no authored binding occurrence and stay origin-free;
their physical `output:` remains a separate node-contract site through the projection.
Private assembly keeps the
resolver unset while the shared catalog grows. The loader validates and lexically rebases
each external alias view, then freezes the local catalog once and installs an exact
`ComposedFlow.origin_resolver`; source-free graphs use `None`.
After the common header stamp, assembly calls each descriptor's immutable
`with_source_origins(...)` hook while its located cursor is still available. The
default is a no-op. CODE uses the hook to register the whole `code:` value and,
for literal blocks only, each exact constructed line in order; quoted and folded
scalars keep only the whole-value fallback. Source-free descriptors mint no new
Compose origins but preserve explicit programmatic carriers. The hook returns opaque
`OriginRef` carriers only and never retains
the cursor, coordinates, or a parallel source map. CODE load-time syntax errors
resolve those already-registered sites through the builder's package-private read
seam; it does not reparse or freeze. The loader remains the sole successful-load
catalog freeze.
CASE uses the hook to register each indexed `cases[N].when` and optional `on:`
value. Lowering transfers predicate origins to ordered `Case` carriers and transfers
the first owning predicate (or `on:`) origin to generated input `Binding` carriers;
generated `Parameter` values remain source-free.

Tags are read once, in Compose. `compose/source` records each local YAML tag and
constructs the node as if untagged (YAML 1.2 core values plus dates; `!!` tags keep their YAML
meaning); `SourceCursor.tag`, `tag_definition_id`, `has_value`, and `scalar_text` expose
it. `compose/helpers/tags.py` is the one slot reader (`read_tag_type`,
`read_tagged_value` -> `TaggedSlot`, `suggest_tag` for every message spelling). Every
slot reads through it, and each read records the tag in a ledger so a tag outside a
slot is a load error (`x-*` keys excepted). A tag checks and narrows (load:
`type_relation`; run time: `build_value_as`, `binding.type`) and converts only how files
are held (`normalize_files`: a `file` position reads `{uri, mime, name}` as a `FileRef`, an
`object` position reads a nested `FileRef` as its mapping); never add a fill-in or any
other conversion path. Nested `${...}` stays a `ValueTemplate` evaluated by the binder; do
not lift it into synthetic nodes, which would run on an untaken CASE branch and could not
read MAP's `${item}`. Load-time walkers reach nested strings through
`iter_source_strings` / `map_source_strings` / `binding_input_groups` in
`nodes/contracts.py`, never a private tree walk. Constructors `${T(...)}` (typedef
names only) resolve into `Node.types` like `isinstance`. Authored types use the
parenthesized grammar in `typesys/types.py` (`Optional(str)`, `int(1..12)`); bracket
spellings are migration errors, while `ValueKind` wire names (`"list[str]"`) stay.

Inline extraction intentionally takes one graph-wide nodes snapshot before
projection so unchanged aliases retain sharing; field navigation uses cursors or
their package-private read-only view.
`ComposedFlow` stores no source-text/range snapshot. Its exact `origin_resolver`
is the sole runtime presentation authority; load diagnostics are resolved before
they leave Compose.

Inline composite implementation stays wholly in `compose`: `desugar/` retains
and masks located bodies, the loader owns the file-private child-flow table/cache,
and `build/` proves generated parameter types and assembles the child. The child
then enters the ordinary compiled CALL/MAP/LOOP path; never add private-child-flow
resolution or inline-body handling to runtime.

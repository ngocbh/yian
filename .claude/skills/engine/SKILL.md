---
name: engine
description: >-
  Use when implementing, extending, or changing the Agent Composer flow engine —
  anything under src/agent_composer/ (node kinds, runtime/scheduling, typed state,
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
framing + OCaml concept map in [`src/agent_composer/README.md`](../../../src/agent_composer/README.md)).
The public vocabulary lives in [`docs/agent.md`](../../../docs/agent.md). Hold
this model whenever you design anything:

- A **flow is a function** — typed inputs in and one typed output value out. An
  **agent** is the conceptual umbrella for a composable transformation represented
  by a node, a flow, or a child flow. The concrete **AGENT node** (`kind: agent`)
  is the LLM-backed leaf.
- Flows **compose**: CALL/MAP/LOOP apply child flows recursively within the
  runtime depth limit. Their `nodes:` + `output:` form is front-end sugar for
  the same application: Compose
  lowers it to a file-private child flow; it does not introduce a runtime kind.
- Computation is **pure at the boundary**: a node returns one `Output | Route |
  Grow`; the engine records and applies that outcome. Only `Output` binds a value.
- Source-neutral semantic operations live in `nodes/contracts.py`: `Binding`,
  `Parameter`, `Assertion`, `ValueContract`, and `PartialRecordContract` carry
  optional opaque origins;
  `iter_origins()` and `map_origins()` touch only declared provenance slots, and
  `typesys.Type` remains provenance-free while arbitrary payloads stay opaque.
- Numeric `Type.bounds` is one optional inclusive pair on INTEGER/NUMBER.
  Validate shape and values through the existing type system. Integer endpoints
  remain arbitrary precision; NUMBER endpoints need finite, lossless float
  representation. Compatibility requires interval containment, and inline type
  copies, nested schemas, and labels retain the refinement. Omit absent bounds
  from manifests so unbounded declarations keep their identity.
- `AgentNode.variables` reuses ordered `Parameter` declarations, each with an
  exact contract and explicit typed default. Reject duplicate names and input
  collisions. Resolve a detached default record once per invocation, then render
  inputs plus variables without evaluating default strings as bindings.
  `Node.declared_variables()` exposes the local declaration inventory, empty by
  default. Ownership is authored execution `static_path` plus variable name;
  rendering requires no scope. Include nonempty declarations in manifests and
  their origins in AGENT's existing traversal hooks. This inventory does not
  activate backward or modify model configuration.
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
- AGENT node schema selection leaves only an untagged `str` on the free-text path. A tagged
  string (`Literal[...]`) uses the generated scalar `value` wrapper, structured retries,
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
  codes are `binding.evaluate`, `contract.input`, `assertion.failed`,
  `node.execute`, `contract.node_result`, `contract.output`, `contract.route`,
  `contract.growth`, and `graph.terminal_unreachable`. Exception classes and
  traceback text must never decide behavior. The evaluator assigns
  `binding.evaluate`, `contract.input`, `assertion.failed`, `node.execute`, and
  `contract.node_result` by operation stage. The dispatcher assigns
  `contract.output`, `contract.route`, and `contract.growth` from the closed
  `Output`, `Route`, and `Grow` arms, never from an exception class. A
  `TypeCheckError.path` is a contract-relative tuple of record names and actual
  list indices. Supplied-input and output type rejection resolve it through
  `ContractOrigins` (exact, nearest ancestor, then default); output uses the
  effective commit target's contract. Missing contract provenance falls back to
  the parameter or executing node, while invalid defaults retain their default
  origin. Non-type output canonicalization uses the contract default. Route and
  growth use the executing node origin. All paths feed one schema-7
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
  explicit node failure wins immediately.
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
  Do not add another assertion text/origin map or positional range list.
- Compiled `Node.params` has one shape: ordered exact `Parameter` carriers. The
  flow owns their `Binding` sources; parameter contracts/default presence/
  coercion policy stay source-neutral, and rewrites preserve all origins.
  `compose/helpers/types.py` is the one source-aware recursive type compiler. For
  authored root/definition flow inputs, the declaration key owns `Parameter.origin`,
  the type value and authored nested record fields own `ContractOrigins`,
  and a packed default owns `default_origin`; START reuses those exact carriers.
  Scalar alias/list type expressions fall back to the contract root. Python/source-free
  declarations and inferred inline parameters stay source-free, while external aliases
  use ordinary lexical rebasing. Keep that separate from ordinary node-input wiring:
  final assembly stamps its key on the sink `Parameter` and its bound value on the
  corresponding `Binding`. Source-backed reserved MAP `over:`
  and WAIT `until:` values are stamped only on their `Binding` because they have no
  formal parameter. CASE's kind-owned lowering stamps each generated searched
  `Binding` with the first predicate origin that introduced the reference, or stamps
  `__on` from `on:`; its generated `Parameter` carriers remain source-free.
  Supplied contract failures select
  `Parameter.contract.origins.at(TypeCheckError.path)` before the parameter
  fallback; default failures keep default-site ownership. Never add a parallel
  source map.
- Compiled `Node.output_contract` is one closed sum: an exact `ValueContract`, a
  closed `PartialRecordContract` with optional field contracts, or `None` for an
  unenforced output. Runtime success and replay enforce the same sum; behavior that
  needs a whole semantic type uses only the exact arm. Manifests exclude origins. Compose
  descriptors, child signatures, producer
  maps, and validators retain the contract and project `.type` only inside type
  algorithms. Compose and runtime growth reference validation share the source-neutral
  closed-field walk from `nodes/contracts.py`. Python-authored nodes pass
  `output_contract=`; there is no parallel
  output-Type API. Authored `output:` syntax (the descriptor's normalized `outputs` field)
  stays unchanged. Compose injects an optional source-aware reader through `BuildContext`;
  `NodeDescriptor.output_contract()` invokes it
  only at each output-authoring leaf's established build position, so earlier kind-owned
  validation and error ordering do not move. A physical declaration supplies the contract
  root/default and exact nested record-field origins; scalar alias/list interiors fall back
  through the nearest authored ancestor to the root. Compact roots and flat definitions
  project their top-level declaration onto the synthesized leaf. Direct/source-free builds
  and generated contracts without an
  authorable type occurrence keep empty origins, while derived child contracts preserve
  their carriers and external alias views use ordinary lexical rebasing.
- Static expression typing is source-neutral: `expr.infer_expr_type` mirrors only
  result types the evaluator proves (literal/direct-ref and fixed boolean roots).
  Compose's `helpers/output_contracts.py` alone adds template shape, reference
  projection, and provenance. Unsupported computed outputs remain opaque instead
  of inheriting a reference leaf's type; inline-composite input inference stays
  strict and separate, but may project an exact known field from a partial producer.
  Direct whole references and `ChildSignature.output` preserve either output-contract
  arm. Multi-output assembly creates an exact record when possible and otherwise a
  closed `PartialRecordContract`; unknown type evidence is never concrete `None`.
  CALL/LOOP preserve that carrier, while MAP wraps only an exact child type. Live commit,
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
- [`src/agent_composer/README.md`](../../../src/agent_composer/README.md) — what the engine is, the layout, + the layer ladder.
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
  the parent's pool), recursively within the runtime depth limit. This is how one
  agent composes another and how composite nodes are built. One-off inline composite bodies must
  lower to that same child contract with stable lexical identity and a closed
  generated-input scope. A spawner returns only a canonical
  `Grow(seed, prune)`; the compile-owned `build_grow` interprets that seed after the
  package-root run boundary injects it into runtime. Never assume a node is a leaf;
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
  still reject it, while existing `LIST_ANY` and explicit `None`-element semantics
  remain unchanged. The specialized `LIST_*` discriminator and carrier preserve
  `None` through canonical round-trips. Cross-flow compatibility recursively
  applies the nullable-source-to-strict-sink rule to list elements.
- **Append-only durable authority** — a fresh run records complete initialization,
  then every node start and accepted outcome. A transition is validated and prepared,
  appended to its `RunLog`, and only then applied to `RunState`; observers become
  visible after append + apply. Recovery validates a contiguous prefix and folds
  it through the same reducer path before recording the ordered retry frontier.
- **Blocking authorable effects return ordinary outputs** — an AGENT node's
  explicitly
  granted `ask_user` tool and HUMAN_INPUT call the process-owned handler inside
  their live invocation; the AGENT node checks the run-owned stop event between external
  model/tool effects, and WAIT blocks on that event until its absolute UTC deadline.
  All three return `Output`; none creates continuation topology.
- **Dependency-light core (with a deliberate exception)** — keep the engine free
  of a DB and heavy frameworks; external capabilities enter through **injected
  seams** (plain callables). **Exception (by decision):** the AGENT node is *not*
  SDK-free — it decomposes into a `mode` (the loop, in `nodes/agent/modes/`) +
  skills (tools), and a mode imports langchain + `agent_composer.llm_clients` and
  builds its own model via `model_from_config`. Prefer seams for *new* external
  deps, but model access in agent modes is a direct call.
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
  `grow_depth_delta`, `is_loop`, `needs_stop_event`, `skipped_output_target`)
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
  grants. No ambient/node config fallback, recall, render, or default replay.
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
  authority; `RuntimeLearningState` keeps pure inventory/episode projection only.
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
  `compile.execution_layout`; do not parse runtime IDs or require a memory policy
  or new flow ID. Apply the layout after existing run setup makes private child
  copies per callsite. Validate the complete layout, target identities, aliases,
  and existing scopes before writing any scope. Derived `Node.execution_scope`
  is explicitly excluded from authored fingerprints; memory frame ownership
  remains separate.
- **Growth connects recorded inputs explicitly** — source-neutral fragment
  recipes select CALL/LOOP parameter routes or MAP item-index/parameter routes.
  Retain real child START/END pairs; synthetic MAP fan nodes are not child-flow
  boundaries. Include connections and execution topology in expansion validation
  and its digest. Join candidates only into fields present in the accepted child
  capture, keeping the original capture unchanged. `evaluated_keys` wins over a
  parent candidate, including an actual expression that returned null. Consume or
  prune pending connections so reused runtime IDs cannot inherit stale sources.
  LOOP templates begin at driver zero; cloning adds outer positions while keeping
  a nested LOOP's initial driver suffix. Continuations replace their own driver
  position. Do not parse runtime IDs or change memory ownership to derive these
  execution positions.
- **Retained history follows accepted records** — the recorded `learnable` flag
  controls retention. Schema 7 requires `ForwardCapture` on every accepted node
  outcome when `learnable=True` and omits it when `learnable=False`. Model/tool evidence
  remains separate from general capture and may have no memory scope; recalled
  sections still require matching ordered
  frame membership. Identify executions by run ID and accepted start sequence,
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
- **Memory is host-injected and record-authoritative** — authored
  `long_term_memory:` declarations select policies that resolve to static frames
  owned by flows or AGENT nodes; MAP items and LOOP iterations reuse those frames.
  `memory=None` is dormant, while an injected registry opens snapshot-isolated
  sessions and supplies a kind-blind recall callback, independently of `learnable`.
  `learnable=True` retains execution only; forward returns before learning/commits.
  Explicit `RunResult.backward(feedback)` performs the reverse fold, then every
  enabled frame learner in layout order, including empty/unexecuted histories.
  A receiver writes only its own frame; descendant evidence preserves original
  execution coordinates. No learner mutates an inherited parent frame.
- **Episodic memory appends canonical execution history** — import the built-in
  process-local backend from `agent_composer.memory.episodic`. An authored
  `long_term_memory: episodic` frame accepts positive `max_episodes` and
  `recall_limit` settings. Learning deterministically proposes one append per
  owner-scoped episode. After the attempt reaches Ready, the engine requests the
  policy batch; the backend commits it atomically and idempotently, then evicts
  the oldest entries beyond the frame bound. Context-free recall is
  chronological. Contextual recall ranks lexical
  overlap and then recency within its limit. Sessions retain immutable JSON
  snapshots; stale sessions may recall their pinned view but cannot learn or
  commit. The backend owns no file or database persistence.
- **One attempt barrier precedes commits** — accept Started, complete backward
  results and all frame proposals, then accept Ready. Commit each enabled policy
  once in recorded order, even with an empty batch, accepting its positive-attempt
  request before the backend effect and observation afterward. A known failure
  stops later policies; Finished records the separate outcome. No per-frame
  learning records, nullable legacy attempt IDs, or per-frame retry frontier.
  Keep `unrequested`/`unknown`/`succeeded`/`failed` policy observations distinct.
  An accepted request or successful Finished permanently forbids another attempt;
  a permitted manual precommit retry recomputes everything using fresh sessions
  and original snapshots. Never resume Ready proposals or infer retry policy from
  backend hints.
- **Delayed backward owns no idle resources** — `run.py` keeps private program
  and accepted data plus policy references. Each call reopens/replays authority,
  uses a nonblocking handle guard and durable locks, restores recorded snapshots,
  and closes resources. Preserve primary host/cleanup errors and the accepted
  prefix; poison an unrefreshable handle. Invalid submissions leave prior observed
  outcome objects unchanged. Terminal resume is inspection only, opening no
  policies and invoking no backward/learning work. Forward fields never change.
- **`Node.env` is engine-opaque** — a per-node `dict[str,Any]` of static config the compiler
  bakes at build (`{**flow_env, **node_env}`, node wins, flow-local). The engine NEVER reads it;
  only a node's own `forward()` interprets known keys (e.g. an AGENT node's `max_tool_iterations`). It is the
  kind-blind knob for "tune this node" — add a key by having the node read it, never by an engine
  branch. An AGENT node stays one live loop, so its node-local `env` remains on that invocation.
- **Single-writer invariant** — workers are pure executors; the dispatcher alone
  consumes the forward ready frontier and commits its reducer transitions. It records each start before
  submission and each accepted outcome before applying it. A pooled drive owns workers
  until its full join; abnormal dispatch exit must signal cooperative stop before joining.
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
the private retained document graph. Root text is constructed once as a
`SourceDocument`; each uncached external file is constructed once too. The public
cursor interface is immutable and defensive. The default file resolver's
successful resolved-path cache reuses one compiled flow for every alias or diamond
reference to that external file.
Persistent pair/arc ropes keep YAML merge topology compact; completed mapping
summaries are reuse frontiers rather than recursively rescanned descendants.
Defensive value snapshots are iterative and preserve shared/recursive aliases.
Source-aware parsing and inline extraction navigate `SourceCursor`s: physical
cursors combine definition ranges with occurrence paths (and mapping-key ranges),
virtual cursors are defensive source-free snapshots, and projected cursors retain
explicitly selected physical child sources while changing a view. Descriptors
retain derived ranges; `NodeDescriptor._finish()` copies them to Compose-only
`BuildResult.node_range` / `field_ranges` for post-build validation. Assembly
discards both before `CompiledFlow`, so semantic nodes have no YAML range or
aligned assertion-range fields. Never reparse a source slice or mutate
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
value and authored nested record fields on `ContractOrigins`, and a packed
default on `default_origin`; START reuses those carriers. Scalar alias/list expressions fall back
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

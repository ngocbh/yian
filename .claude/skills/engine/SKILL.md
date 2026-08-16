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
data transformation whose leaf computations may be LLM agents. Every engine
change follows the same loop — **never jump straight to code.**

**Companions:** [`reference.md`](reference.md) — the node contract
(`Output | Route | Grow`), the `NodeKind` table, the OCaml-analogue map, the
non-negotiable invariants, the layer ladder, and a design-note template.
[`templates/node_kind/`](templates/node_kind/) — a copy-paste skeleton + a
file-by-file [`WIRING.md`](templates/node_kind/WIRING.md) for adding a new
authorable node kind.

## The mental model: you're extending a functional language

The engine is a **small functional language for data transformation** (full
framing + OCaml concept map in [`src/agent_composer/README.md`](../../../src/agent_composer/README.md)).
Hold this model whenever you design anything:

- A **flow is a function** — typed inputs in, typed outputs out. An **agent is
  just a flow** whose leaf is an LLM loop. Same contract, same composability.
- Flows **compose** (a node can *be* a flow, via a `call:`/`uses:` reference) and
  nest to any depth. A CALL/MAP/LOOP `nodes:` + `output:` body is front-end sugar
  for the same application: compose lowers it to a file-private child flow; it
  does not introduce a runtime kind.
- Computation is **pure at the boundary**: a node *returns* outputs, the engine
  *binds* them under `(node_id, key)` like `let`-bindings — never a mutation.
- Source-neutral semantic operations live in `nodes/contracts.py`: `Binding`,
  `Parameter`, `Assertion`, and `ValueContract` carry optional opaque origins;
  `iter_origins()` and `map_origins()` touch only declared provenance slots, and
  `typesys.Type` remains provenance-free while arbitrary payloads stay opaque.
- Nodes own local origin enumeration/rewrite and expose static children through
  `origin_children()`. `compile.origins` combines those hooks with flow wiring
  and outputs, rewriting an isolated copy by identity and returning a finalized
  copy whose hash containers are rebuilt. Shared or cyclic graphs remain safe
  without a `NodeKind` branch.
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
  `Output`, `Route`, and `Grow` arms, never from an exception class. Output uses
  the effective target contract's root origin when available and otherwise the
  executing node origin; route and growth use the executing node origin. Both
  paths share one schema-1 projector. Host control, stream callbacks, unexpected
  binder faults, malformed compiled/setup state, unsupported growth kinds, and
  arbitrary builder/compiler/spawner-or-fragment-copy/digest/materialization/reducer/log faults
  escape. Durable/public/terminal cutover remains tracked by #182.
- `Node.origin` is the optional source-neutral identity of an authored node
  context. It has no execution meaning, is unrelated to LOOP's `origin_id`, and
  stays outside manifests/fingerprints. Compose stamps physical node headers;
  source-free/generated nodes default to `None` and clones preserve a value.
- Compiled flow-owned wiring has one shape: `Flow` construction wraps each
  direct source as `Binding`, and `FlowOutput.binding` owns declared outputs.
  Python builders may pass raw values at construction, but must not write raw
  values into an already-constructed flow. Compose stamps source-backed bare/mapped
  root, multi-node-definition, and inline-child output values on the final binding after
  rewrites, and END reuses that exact carrier; output mapping keys stay origin-free.
  Generated compact-root and single-node-definition reexports remain origin-free because
  their authored output declaration is reserved for contract-origin work.
- Compiled node assertions also have one shape: ordered `Assertion` carriers.
  Compose creates one carrier per authored flow/node list position and derives
  its `.origin` (`OriginRef`) plus the temporary positional range from the same retained item
  cursor; duplicate text therefore remains occurrence-distinct. Evaluate
  `.expression`, report `.display`, and preserve `.origin`: inline-call and
  flow/CALL lowering replace only the expression, while child cloning and LOOP
  respawn retain the carrier fields. START/END reuse the lowered flow carriers.
  Do not add another assertion text/origin map; the existing positional range
  lists are a temporary compatibility bridge, not the provenance authority.
- Compiled `Node.params` has one shape: ordered exact `Parameter` carriers. The
  flow owns their `Binding` sources; parameter contracts/default presence/
  coercion policy stay source-neutral, and rewrites preserve all origins. Final
  Compose assembly stamps an ordinary authored `input:` key on its `Parameter`
  and its value on the corresponding `Binding`. Source-backed reserved MAP `over:`
  and WAIT `until:` values are stamped only on their `Binding` because they have no
  formal parameter. Never add a parallel source map.
- Compiled `Node.output_contract` has one shape: an exact `ValueContract`, or
  `None` for an unenforced output. Runtime and node behavior project `.type`, while
  manifests exclude `.origins`. Compose descriptors, child signatures, producer
  maps, and validators retain the contract and project `.type` only inside type
  algorithms. Python-authored nodes pass `output_contract=`; there is no parallel
  output-Type API. Authored `output:` / `outputs:` syntax stays unchanged.
- The **author fixes the structure** (the call graph); the LLM fills leaf boxes,
  it does not rewrite the graph at runtime.

So a new construct isn't "a workflow feature" — it's "a language construct," held
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
  `NodeKind`: ordinary behavior belongs in a node's own `_run_impl` or node-owned
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
outputs-before-successors, recursive skip-flood, recorded starts and outcomes,
append-before-apply transitions, and one reducer path for live execution and
historical replay) and **drop** scale/framework baggage (external DBs, dynamic
worker scaling, plugin registries/metaclasses, heavy layering, multi-tenancy) —
and anything that fights the functional model above.

## 2. Fit it to the engine invariants
Decide keep / simplify / drop against these **non-negotiable engine invariants**.
Most are just the functional model (above) made enforceable:

- **Deterministic workflow engine** *(referential transparency at the structure
  level)* — the author fixes the call graph; the LLM fills leaf boxes. A flow
  never rewrites itself at runtime. No agentic routing.
- **A flow is a function — clear typed inputs and outputs** — every flow has an
  explicit input/output signature. An agent is one kind of flow; it gets no
  special contract.
- **Hierarchical / composable** *(function composition)* — a node can *be*
  another flow (a `call:`/`uses:` reference → a child subgraph seeded from the
  parent's pool), nestable to arbitrary depth. This is how an agent calls another
  agent and how composite nodes are built. One-off inline composite bodies must
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
- **Node never writes the pool** *(purity)* — it returns outputs; the engine binds
  them under `(node_id, key)`, like an immutable `let`. No node mutates shared state.
- **Typed, losslessly-serializable state** *(the type system)* —
  `TypedValue` / `VariablePool`; the basis for canonical run-record payloads and
  `${...}` references.
- **Append-only durable authority** — a fresh run records complete initialization,
  then every node start and accepted outcome. A transition is validated and prepared,
  appended to its `RunLog`, and only then applied to `RunState`; observers become
  visible after append + apply. Recovery validates a contiguous prefix and folds
  it through the same reducer path before recording the ordered retry frontier.
- **Blocking authorable effects return ordinary outputs** — an AGENT's explicitly
  granted `ask_user` tool and HUMAN_INPUT call the process-owned handler inside
  their live invocation; AGENT checks the run-owned stop event between external
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
  branch on it. Ordinary behavior lives in a node's own `_run_impl`; growth construction's
  sanctioned kind match lives in `compile.expand.build_grow`. The core dispatches only on the
  closed worker-result and run-record sums and on node-owned traits/hooks
  (`is_spawner`, `binds_per_item`,
  `bind_reserved`, `reserved_wiring_keys`, `validate_execution_setup`,
  `exception_origin`, `grow_depth_delta`, `is_loop`, `needs_stop_event`). A
  ratchet test (`tests/engine/test_kind_census.py`) holds the core's `NodeKind`/`*Expansion`
  dispatch count at 0. A separate layer ratchet forbids `nodes -> compile` and
  `runtime -> compile.expand`: nodes describe seeds, compile constructs fragments,
  and runtime receives the constructor as an injected callable.
  Compile-side origin traversal follows the separate node-owned
  `iter_local_origins`, `rewrite_local_origins`, and `origin_children` hooks; these
  are not runtime capabilities.
- **`Node.env` is engine-opaque** — a per-node `dict[str,Any]` of static config the compiler
  bakes at build (`{**flow_env, **node_env}`, node wins, flow-local). The engine NEVER reads it;
  only a node's own `_run_impl()` interprets known keys (e.g. AGENT's `max_tool_iterations`). It is the
  kind-blind knob for "tune this node" — add a key by having the node read it, never by an engine
  branch. AGENT stays one live loop, so its node-local `env` remains on that one invocation.
- **Single-writer invariant** — workers are pure executors; the dispatcher is the
  only ready-frontier consumer and state mutator. It durably records each start before
  submission and each accepted outcome before applying it. A pooled drive owns workers
  until its full join; abnormal dispatch exit must signal cooperative stop before joining.
- **Single-process CLI target** — dispatch and workers remain in-process. Durability
  is the small `RunLog` seam with memory and locked JSONL implementations; no DB,
  server, or remote scheduler is baked into the core.

## 3. Plan
Write a short plan: the **OCaml analogue** you're borrowing and its type; what you
simplify/drop and why; where it lands in the layer map
(`diagnostics` below `events`, with nodes combining `diagnostics`, `typesys`, and
`expr` before `compile ← compose`; the internal `_state.py` leaf shared by
compile and runtime; and runtime consuming compile's execution contracts but not
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
`expr` leaves before `compile ← compose`, while `_state.py` feeds compile and
runtime. This is an instance of that framework.
The package-root `run.py` sits above compose and runtime and wires cross-layer
capabilities: notably it owns log lifecycle/recovery and injects compile's
`build_grow` into the kind-blind runtime.

`diagnostics.Failure` owns semantic failure identity and
`diagnostics.ResolverTable` owns immutable exact resolver routing; neither knows
a source format or runtime state. `compile/origins.py` owns the
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
explicitly selected physical child sources while changing a view. Descriptors and
nodes carry derived ranges thereafter. Never reparse a source slice or mutate
anchors/aliases/merges.
`SourceCatalog` is Compose's immutable `OriginResolver`: its private builder maps
typed semantic paths to opaque origins and retains document occurrences with explicit
key/value anchors. Resolution replays an occurrence without reparsing or adding a
parallel range map; virtual or unanchored cursors produce no origin. One builder is
shared by a physical document's root, definitions, and inline bodies. Assembly stamps
authored node headers, ordinary node-input keys and values, source-backed reserved
MAP/WAIT wiring values, source-backed flow-output values, and authored flow/node
assertions directly on their canonical carriers. Assertion sites are indexed by source
order; one retained item cursor supplies both `Assertion.origin` and its aligned temporary
range. Full root flows and multi-node definitions retain a flow assertion cursor and reuse
the lowered carriers on START/END. Compact roots and flat single-node definitions project
their assertions onto the synthesized node, where they remain node-local. Flow-output
values are stamped after rewrites on `FlowOutput.binding`; END reuses that binding and
output mapping keys remain origin-free. Generated compact-root and
single-node-definition reexports have no authored binding occurrence and stay origin-free;
their output declaration is reserved for contract-origin work. Private assembly keeps the
resolver unset while the shared catalog grows. The loader validates and lexically rebases
each external alias view, then freezes the local catalog once and installs an exact
`ComposedFlow.origin_resolver`; source-free graphs use `None`.
Inline extraction intentionally takes one graph-wide nodes snapshot before
projection so unchanged aliases retain sharing; field navigation uses cursors or
their package-private read-only view.
`SourceFrame` is persistent render-only text/range metadata on `ComposedFlow.source`
and composite nodes' `child_source`; it does not own parsing or the source graph.

Inline composite implementation stays wholly in `compose`: `desugar/` retains
and masks located bodies, the loader owns the file-private callable table/cache,
and `build/` proves generated parameter types and assembles the child. The child
then enters the ordinary compiled CALL/MAP/LOOP path; never add private-callable
resolution or inline-body handling to runtime.

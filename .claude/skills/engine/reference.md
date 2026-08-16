# Reference — engine internals

Lookup companion to [`SKILL.md`](SKILL.md). The skill is the *workflow* (design
from the functional model → fit invariants → plan → implement); this is the
quick-reference and the [`templates/`](templates/) index. The canonical design
text is [`src/agent_composer/README.md`](../../../src/agent_composer/README.md);
the authoring surface is [`docs/syntax.md`](../../../docs/syntax.md).

## Templates

| Template | Use |
|----------|-----|
| [`templates/node_kind/node.py.template`](templates/node_kind/node.py.template) | the `Node` subclass skeleton |
| [`templates/node_kind/__init__.py.template`](templates/node_kind/__init__.py.template) | the package charter / re-export |
| [`templates/node_kind/test_node.py.template`](templates/node_kind/test_node.py.template) | unit + load/run test skeleton |
| [`templates/node_kind/WIRING.md`](templates/node_kind/WIRING.md) | the cross-file edits to make `kind: xxx` authorable |

## The node contract (the most portable idea)

A node is a **pure function of its bound input record**. A kind implements
`_run_impl(inputs, **caps) -> NodeResult`; the base `Node.run` is a concrete
**template method** the engine drives — it fires the node's asserts around the body:
`apply_defaults(record) → pre_asserts(record) → _run_impl(record, **caps) →
post_asserts({**record, "output": value})`. Assert firing is **record-only** (no pool),
so it preserves node purity, and it happens **exactly once** here in the node (not in
`eval_node`). Both assertion lists contain ordered `Assertion` carriers: the wrapper
evaluates `.expression` and reports/locates `.display`. `_run_impl` returns ONE of the closed sum:

| Return | Meaning |
|--------|---------|
| `Output(value)` | the one produced value. The dispatcher prepares a canonical `NodeSucceededRecord`; after append, the reducer writes it under the resolved success target and fires that target's out-edges. Optional `Output(value, commit_as=<id>)` chooses another success target; `None` uses the node's baked redirect or its own id. |
| `Route(handle)` | a routing-only outcome (a CASE picks an out-edge `handle`); carries no value and writes no pool entry. The dispatcher records the handle, then the reducer takes the selected edge, skip-floods its siblings through their logical output targets, and derives `NodeRouted`. |
| `Grow(seed, prune)` | describe one deterministic graph expansion without constructing it. `seed` is the kind-shaped but runtime-opaque builder input (CALL record, MAP records, or LOOP carried record/index); `prune` names ids to retire in the same step. Compile-owned `build_grow(spawner, seed)` constructs the self-describing `Flow`; the dispatcher prepares a `NodeExpandedRecord`, then the reducer applies the generic splice. The canonical seed is persisted for replay. Only spawner kinds (`is_spawner`) may return it. |

Success and absence resolve their logical output boundaries through distinct
carriers. Success resolves `Output.commit_as`, then baked `Node.commit_as`, then
the physical id. A result-carried redirect is unavailable when the node is
skipped, so it affects success only unless the node's
`skipped_output_target()` hook mirrors it. The base hook returns baked
`commit_as` or id; LOOP returns manifest-backed `origin_id` to mirror its STOP
output. The hook is pure and deterministic, derives only from manifest-backed
execution state, and returns an existing node id.

A redirect is an exclusive single-owner contract, not an OR join: one unfinished
physical node may redirect to a logical target, and the target must be inert—an
expanded spawner or an unknown placeholder with no incoming work, never ready,
in flight, or already committed; all of its outgoing edges must still be unknown. The
reducer validates this topology before mutation, marks the physical node
skipped, deduplicates logical targets, and traverses the logical target's
outgoing adjacency. Settled expansion residue is inert and no longer owns a
future output.

A streaming kind is a generator that yields `StreamChunk` and *returns* a
`NodeResult`. **Failure is not a variant** — a node `raise`s and the engine
evaluation boundary returns `EvaluationFailure(Failure)` for expected rejection;
host control, stream callbacks, unexpected binder faults, and malformed setup
escape. Result preparation constructs the same canonical value for expected
output, route, and growth rejection. One schema-1 projector handles both paths,
and the reducer derives `NodeFailed`/`RunFailed` observers.
`Node.on_failure` is a dormant future seam, not currently driven by `eval_node`.

The dependency-free `diagnostics` leaf defines the canonical semantic value for
that pipeline:

```python
@dataclass(frozen=True, slots=True)
class Failure:
    code: FailureCode
    message: str
    node_id: str | None
    trace: tuple[OriginRef, ...] = ()
    cause_type: str | None = None
    traceback: str | None = None
```

The trace is authored identity ordered outermost to innermost; its final item is
the most precise available operation, while `node_id` remains runtime execution
identity. The closed plain-string codes are `binding.evaluate`,
`contract.input`, `assertion.failed`, `node.execute`, `contract.node_result`,
`contract.output`, `contract.route`, `contract.growth`, and
`graph.terminal_unreachable`. Exception names and traceback text are debug data,
not semantic categories. The contract reserves these codes for expected
flow-owned rejection. The evaluator structurally assigns `binding.evaluate`,
`contract.input`, `assertion.failed`, `node.execute`, and
`contract.node_result`; it never classifies by exception type. `EvaluationFailure`
owns that single `Failure` plus temporary locator/range metadata. Dispatcher
classification is structural too:

| Result rejected during preparation | Code | Best current origin |
|---|---|---|
| `Output` | `contract.output` | effective target contract root, then executing node |
| `Route` | `contract.route` | executing node |
| `Grow` | `contract.growth` | executing node |

`FlowEngine` projects both sources into flattened schema-1 fields, dropping
semantic `code` and opaque `trace`. Durable/public/terminal cutover remains
tracked by #182.

**Invariants:** a node never receives the pool (the `eval_node` seam binds its
inputs); a node never writes the pool (it *describes* `Output(value)`, the engine
records it, and the reducer performs the write). Nodes stay pure; the dispatcher
is the sole live state writer. `Node.origin` is optional opaque diagnostic
provenance for the authored node context, not execution state; it is unrelated to
the LOOP-only `origin_id` attribution field.

Growth uses the same inversion boundary. Node modules never import `compile`: a
spawner returns a canonical seed only. `compile.expand.build_grow` is the sole
kind-aware constructor; package-root `run.py` injects it into `FlowEngine` for live
growth and into reducer replay for recovery. Runtime may consume compile's execution
contracts but never imports `compile.expand`, so the splice remains kind-blind.
`compile.expand.build_grow` raises `compile.model.GrowthRejected` only for an
explicit malformed seed after selecting a legal CALL/MAP/LOOP kind. Unsupported
kinds and arbitrary builder, compiler, spawner/fragment-copy, digest,
topology-materialization, reducer, append, or log faults propagate raw. Runtime
classifies only explicit seed, validated-fragment, budget, depth, and
prune-policy rejection as `contract.growth`.

**Capabilities (`caps`)** are narrow, engine-owned effects passed as keyword args to
`run`; each is gated by a node trait. A mapped call gets `caps['bind_item']`
(per-element bind, gated on `binds_per_item`). AGENT and WAIT get the exact run-owned
`caps['stop_event']` (gated on `needs_stop_event`): AGENT checks it between external
model/tool effects, while WAIT blocks on it so host shutdown or sibling failure wakes
the deadline wait. AGENT is the deliberate model-access exception: its mode imports
`llm_clients` and calls `model_from_config` directly, while its granted tools use the
ordinary tool registry.

## Node-owned traits/hooks (how the core stays kind-blind)

The engine core reads these node members instead of branching on `node.kind`. Defaults sit on the
base `Node`; a kind overrides only what it needs.

| Member | Where the core reads it | Default → overrides |
|--------|-------------------------|---------------------|
| `is_spawner: bool` | result boundary: permit `Grow` only from a graph-expanding node | `False` → `True` on call/map/loop |
| `binds_per_item: bool` | read boundary (`eval_node`): bind PER ELEMENT via a `bind_item` cap vs bind `params` once | `False` → `True` on `map` |
| `bind_reserved(wiring, pool) -> dict` | read boundary: reserved keys pre-resolved into the record before `run` | `{}` → `wait` `{"until": ts}`, `map` `{"over": [...]}` |
| `reserved_wiring_keys() -> set` | load-time (compile passes: `check_wiring_parity`, ref-scan): the NAMES of those reserved keys so validation gates on traits, not `node.kind` | `∅` → `wait` `{"until"}`, `map` `{"over"}` |
| `validate_execution_setup()` | base run wrapper: reject malformed node-owned setup before assertions/body | concrete `_run_impl` → baked child on call/map/loop; complete predicates on case |
| `exception_origin(exc)` | evaluator body boundary: select the best operation origin | authored node origin |
| `skipped_output_target() -> str` | route reducer and growth validation: select the logical output whose out-edges receive this physical node's absence | baked `commit_as` or own id → LOOP `origin_id` |
| `grow_depth_delta: int\|None` | growth core: REF-depth increment stamped on the spliced spawners + terminal (positive bounded by `MAX_REF_DEPTH`) | `None` non-spawner → `0` loop (inherit) → `1` call/map |
| `is_loop: bool` | growth reducer: gate the origin-keyed single-live-expansion invariant (the driver self-respawns, so each iteration's grow is attributed to `origin_id` and supersedes the prior expansion) | `False` → `True` on `loop` |
| `needs_stop_event: bool` | read boundary (`eval_node`): pass the exact cooperative run-stop event | `False` → `True` on agent/wait |

These are runtime execution capabilities. The source-neutral
`iter_local_origins`, `rewrite_local_origins`, and `origin_children` hooks are
consumed by `compile.origins`, not by the engine core.

The rest of the splice (add subgraph, enforce `MAX_TOTAL_NODES`, mint one uniform `NodeExpandedRecord`,
finish/mark the spawner, apply the origin `commit_as` to the derived terminal, schedule the
subgraph's `start_id`) is uniform. A generated terminal's baked `Node.commit_as`
redirects both paths: success resolves it after `Output.commit_as`, while the
inherited `skipped_output_target()` propagates absence through the same spawner
anchor. CASE selection still rides `Route.handle`; dynamic `Output.commit_as`
redirects only success unless a node-owned hook mirrors it. A
spawner's own `post_asserts` (a `call` node's `asserts:`) are **lowered onto the cloned child
`END`** at grow time (`expand._lower_spawner_post_asserts`): `${output}` stays; a declared input
`${X}` becomes a synth END param wired off the cloned child START's coerced output, so the assert
fires through the child END's ordinary `run` wrapper — not re-checked at any commit site. Live
growth and durable replay both pass the canonical seed persisted in `NodeExpandedRecord` through
the same injected `compile.expand.build_grow`, so lowering is identical on resume. The moved
assert's display/origin and temporary positional range stay aligned; its failure names the
**child END id**, not the spawner.

AGENT is now an ordinary non-spawner leaf. Its model/tool loop, including an
explicitly granted `ask_user`, stays inside one live `_run_impl`; post-asserts fire
on the final `Output` through the base `Node.run` template. HUMAN_INPUT and WAIT
are likewise live leaves returning `Output`.

## `NodeKind` (closed vocabulary)

`NodeKind` is a closed enum (no registry/metaclass), but the engine **core does not dispatch on
it** — `runtime/engine.py` + `runtime/eval_node.py` branch only on the closed
worker-result and run-record sums and on node-owned traits/hooks (below). Ordinary
behavior lives in each node's own `_run_impl`; growth construction's sanctioned
closed kind match lives in `compile.expand.build_grow`, which
interprets node-produced seeds outside the runtime core. A ratchet test
(`tests/engine/test_kind_census.py`) holds the core's `NodeKind`/`*Expansion` dispatch count at 0.
Kind-specific diagnostics stay with their kind. CODE owns
`nodes/code/watchdog.py`, which wraps only its reference/inline in-process Python
body and logs one soft-budget warning without killing it. AGENT, HUMAN_INPUT, and
WAIT are never timed by that watchdog.

| Authorable kinds | Internal-only (loader-synthesized) |
|------------------|------------------------------------|
| `AGENT`, `CODE`, `MODEL`*, `TOOL`, `CASE`, `HUMAN_INPUT`, `WAIT`, `CALL`, `MAP`, `LOOP` | `START`, `END` |

\* `MODEL` parses but `run` raises — the ML-serving seam was removed as dead
plumbing; re-add when real serving lands.

## Durable execution contract

`runtime.log` owns the closed durable vocabulary and its canonical codecs:

| Record | Transition |
|--------|------------|
| `RunStartedRecord` | complete initial pool, flow fingerprint, START override, resolved LLM map, and worker count |
| `RunResumedRecord` | ordered unfinished starts to retry plus the selected worker count |
| `NodeStartedRecord` | move the exact ready head into the ordered in-flight set before evaluation/submission |
| `NodeSucceededRecord` | commit one canonical typed value under an already-resolved target |
| `NodeRoutedRecord` | select one named outgoing handle and derive its redirect-aware skip closure |
| `NodeExpandedRecord` | rebuild and verify one canonical seed/digest, then splice its fragment |
| `NodeFailedRecord` | make the run terminal-failed with its captured diagnostic |

`FlowEngine` borrows one initialized `RunState` plus a matching open `RunLog`.
Workers only evaluate; the dispatcher prepares a total transition, appends its
record, applies it, then publishes transient observers. `StreamChunk` callbacks are
explicitly non-durable. The engine never closes its log: package-root `run.py` owns
`MemoryRunLog`/`JsonlRunLog` creation, locking, recovery, tail repair, and closure.

`run_flow(..., log=None)` uses process-local authority; a path selects a durable
JSONL log. `resume_flow(flow, *, log=..., num_workers=None)` validates the authored
flow fingerprint, silently folds the locked contiguous prefix through the same
reducer, and either returns an already-terminal result unchanged or appends one
`RunResumedRecord` before retrying unfinished starts in original dispatch order.
`RunResult` exposes exactly `input`, `status`, `run_id`, `log`, `output`, `error`,
`locator`, `range`, `traceback`, and this invocation's post-durable `events`.

## OCaml analogue map (design from this)

| Our construct | OCaml concept | Why it holds |
|---------------|---------------|--------------|
| a flow | a function `'a -> 'b` | typed in, typed out; composes |
| an agent | a flow whose leaf is an LLM loop | no special contract |
| `call:` / `uses:` | function application / a module ref | nests to any depth |
| the variable pool bind | `let (node_id, key) = ...` | immutable, no mutation |
| `NodeKind` + `match` | a variant + exhaustive `match` | closed set, no registry; runtime matches on worker-result/record variants and traits, while compile's `build_grow` interprets spawner seeds by kind |
| `Output \| Route \| Grow` | a sum type (the node result) | failure is a `raise`, not a case |
| append-only `RunRecord` + reducer | an event history folded into state | one explicit transition vocabulary drives live execution and replay |
| a package charter (`__init__`) | a module signature (`.mli`) | a narrow declared interface |

When borrowing runtime mechanics from a prior worker engine: **borrow** the
correctness-critical parts (single-writer dispatcher, 3-state edge join,
outputs-before-successors, redirect-aware recursive skip-flood, recorded starts
and outcomes,
append-before-apply transitions, and one reducer path for live execution and replay);
**drop** scale/framework baggage (external DBs, dynamic worker scaling, plugin
registries, heavy layering, multi-tenancy).

## Non-negotiable invariants (the keep/simplify/drop bar)

A change is a design smell if it breaks any of these — most are the functional
model made enforceable.

- **Deterministic structure** — the author fixes the call graph; the LLM fills
  leaf boxes. A flow never rewrites itself. No agentic routing.
- **A flow is a function** — explicit typed input/output signature; an agent gets
  no special contract.
- **Composable / recursive** — a node can *be* a flow (`call:`/`uses:`), nestable
  to any depth. Never assume a node is a leaf; preserve recursion through
  compile + run; run-record replay must rebuild and verify recorded growth.
- **Privileges no output type or domain** — prefer the most general primitive that
  composes over a use-case-specific feature.
- **Node never writes the pool** (purity); **typed, losslessly-serializable state**
  (`TypedValue` / `VariablePool` — the basis for canonical record payloads + `${...}` refs).
- **Append-only durable authority** — `RunStartedRecord` captures complete initial
  state/configuration, every dispatched node gets a `NodeStartedRecord`, and every
  accepted outcome gets exactly one success/route/expansion/failure record. A prefix
  may end after a start when host control interrupts evaluation. Preparation is
  side-effect-free; `commit_prepared` appends before applying the frozen reducer plan
  and exposes observers only afterward. Recovery validates a contiguous prefix, folds
  it through that same path, and records the ordered retry frontier in
  `RunResumedRecord` before continuing.
- **Blocking authorable effects are live** — AGENT `ask_user`, HUMAN_INPUT, and
  WAIT return `Output` from the same invocation and never create continuation topology.
- **Dependency-light core** — no DB / heavy frameworks; external capabilities enter
  through injected seams (plain callables). *Exception:* the AGENT node imports
  langchain + `llm_clients` and builds its model via `model_from_config`.
- **`llm_config` cascade resolved once at run start** — `resolve_llm_cascade`
  (`compile/`) walks the static call tree top-down, per-field fill-the-gap
  (most-specific wins), deep-copying each CALL/MAP/LOOP child for per-callsite isolation,
  and bakes the effective dict onto every `AgentNode`. It also returns a complete map
  keyed by stable slash-separated static paths. The CLI config is the outermost
  authored layer; the run boundary reads process provider/model defaults exactly once
  at initialization and passes them as the explicit final fallback. `inherit: false`
  skips authored flow/parent/CLI layers but still receives those process defaults.
  `apply_effective_llm_map` replays a recorded complete map without reading ambient
  settings and deep-copies every callsite again. `RunStartedRecord` stores that complete
  map; recovery applies it to a private copy of the supplied static graph before folding
  later records.
- **Stable program and expansion identity** — compile-owned `flow_manifest`
  canonicalizes execution semantics, `flow_fingerprint` hashes it, and
  `expansion_digest` covers an expanded fragment's entry/end ids and topology.
  Assertion expression/display/order/multiplicity and an output contract's semantic
  `Type` affect identity. Every field consulted by an execution hook must likewise
  be manifest-backed; `skipped_output_target()` currently derives from `commit_as`
  or LOOP's `origin_id`. Node/carrier origins and render-only ranges, comments,
  and object identity do not affect identity.
- **Closed `NodeKind`; the core is kind-blind** — `NodeKind` stays a closed enum, but the engine
  core dispatches on worker-result/run-record variants + node-owned traits/hooks, never on
  `node.kind` (ratchet: census 0).
- **Single writer** — workers are pure executors; the dispatcher is the only
  ready-frontier consumer and mutator. A pooled drive records starts before submission,
  records accepted outcomes before applying them, and signals cooperative stop before
  fully joining workers after an abnormal dispatch exit.
- **Single-process CLI target** — dispatch and workers stay in-process. The durable
  seam is `RunLog` with memory and locked JSONL implementations, not a DB, server, or
  remote scheduler.
- **AGENT structured output** — a non-text `output:` Type switches an agent from text
  producer to structured generation: `type_to_schema` (`nodes/agent/structured.py`)
  derives a pydantic schema, the mode generates a conforming value (native
  `with_structured_output`, or a JSON prompt-injection fallback gated by
  `supports_native_structured(provider, model)`), with `retries:`-capped self-correction.
  Three-part contract: **generate-tries** (the schema asks), **boundary-enforces**
  (success-record preparation runs `canonical_typed` against the effective target's
  declared type on both ordinary and `commit_as` paths), **retry-catches** (a deviation
  is fed back and re-asked). A bare `str`/`Literal[...]` keeps the text path.

## Layer ladder (where code goes)

```
diagnostics  <-  events
      ^
      └────────  nodes  <-  compile  <-  compose
                   ^   ^
      typesys + expr    └──────  llm_clients

_state + compile runtime contracts <── runtime
compose + compile.expand + runtime <── package-root run.py
```

`diagnostics.py` owns source-neutral `Failure`, `SourceRange`, opaque `OriginRef`,
resolved `DiagnosticFrame`, the `OriginResolver` protocol, and immutable
exact-route `ResolverTable`. It imports only the standard library. `events.py`
imports this leaf and temporarily re-exports `SourceRange` while existing
consumers migrate.

Arrows never reverse: a package imports only lower-level or peer packages. See the
`structure` skill. An upward import means the code is in the wrong package (extract
the shared contract to `common.py` / a leaf, or invert via a seam). Dynamic growth is
the concrete example: nodes return seeds, compile owns `build_grow`, `run.py` injects
it, and runtime never imports `compile.expand`. Spawner seed schemas live in their
owning node packages. The scheduling-state vocabulary sits in the internal
`_state.py` leaf shared by compile and runtime. Runtime owns records, log codecs,
append-before-apply reducer state, and dispatch; the package-root run boundary owns
fresh-log creation, locked prefix recovery, and log closure.

The same direction governs provenance: nodes describe their local origin slots
and static children, `diagnostics` owns source-neutral resolver routing, and
`compile.origins` owns flow wiring, declared outputs, identity guards, copying,
recursion, and graph-wide rebasing. No node imports the compile walker.

### Source-neutral semantic carriers

`nodes/contracts.py` is the dependency-low semantic IR vocabulary. `Binding`
owns one input source, `Parameter` owns one formal/default policy, `Assertion`
owns executable and display text, and `ValueContract` pairs a provenance-free
`Type` with `ContractOrigins`. Origins are optional and exposed through each
carrier's `iter_origins()` / pure `map_origins()` methods. Those operations visit
declared origin slots only; sources, defaults, and other payload values remain
opaque. Do not put origin fields into `Type`, or recreate them in parallel maps.
Stored flow-owned bindings have completed their
cutover: `Flow.__init__`, `CompiledFlow.from_parts`, and `add_subgraph` normalize
every direct wiring value exactly once, and `FlowOutput.binding` is canonical.
Raw Python values remain accepted only at those construction boundaries;
pre-IR Compose dictionaries and runtime overrides are deliberately raw. A
growth clone creates a new carrier for a rewritten source and retains its origin.
The base `Node` separately carries one optional `OriginRef` for the authored node
context. Compose stamps physical root/definition/inline headers; direct Python and
generated nodes default to `None`, and cloning preserves an existing value. It is
manifest-neutral and unrelated to LOOP's execution-only `origin_id`.
`Node.pre_asserts` / `post_asserts` are likewise canonical ordered `Assertion`
lists. Compose crosses raw YAML text into one carrier per authored flow/node list
position. That item's retained cursor supplies both the carrier origin and the
temporary positional range; source-order indexing keeps duplicate text distinct.
Inline-call desugaring, flow lowering, child cloning/CALL relocation, and LOOP
respawn replace only the executable expression or copy the carrier, so display and
origin survive. START/END reuse the lowered flow carriers. The manifest projects
`{expression, display}` and ignores origin. `Node.params` is
an ordered exact `Parameter` list: `bind_params` reads its optional resolved
contract, explicit default-presence bit, and string-coercion policy while the
flow owns the corresponding `Binding` sources. Parameter/default/contract
origins are manifest-neutral. A bound null shadows a default but still undergoes
contract validation, so it is accepted only when the parameter is untyped or its
contract accepts null (a nullable or `None`-kind type). Compose `input:` crosses
directly into this exact `Parameter` shape; `ComposedFlow.input` and START use it
without a parallel front-end carrier. For ordinary source-backed node inputs, final
assembly stamps the selecting key on `Parameter.origin` and the value on the
corresponding `Binding.origin`. Source-backed reserved MAP `over:` and WAIT `until:`
wiring receives a `Binding.origin` only; neither field is a formal `Parameter`.
Desugaring may replace `Binding.source` but retains that value origin. Without a retained
source occurrence, Compose mints no origin and preserves any explicit programmatic origin;
source-backed projections inherit their backing site. CASE wiring, synthesized boundary
parameters, contracts, external rebasing, and runtime diagnostic consumption are deferred.
Source-backed bare/mapped root, multi-node-definition, and inline-child
output values stamp the final `FlowOutput.binding` after rewrites, and the synthesized END
boundary reuses that exact binding. Output mapping keys receive no origin. Compact-root and
single-node-definition reexports are generated without an authored binding occurrence and
remain origin-free; their authored output type is reserved for contract-origin work.

`Node.output_contract` is the sole stored output declaration: an exact
`ValueContract`, or `None` for an unenforced output. Runtime success/replay
validation and structured agent generation read `output_contract.type`; growth
clones retain the carrier so its origins survive lexical rewrites. Manifests encode
the semantic type but deliberately omit `ContractOrigins`, keeping provenance
changes identity-neutral.

Compose output derivation, builders, producer maps, and `ChildSignature.output`
retain exact `ValueContract` carriers. Reference, assert, CASE, CALL, and LOOP
algorithms project `.type` only at the point where a semantic type is required.
Whole direct references preserve contract identity; nested references rebase their
origin paths. Python-authored nodes pass `output_contract=`; there is no
parallel output-Type compatibility API. The authored `output:` / `outputs:` YAML
surface is unchanged.

### Compiled origin graph operations

The base `Node` visits its node origin, output contract, parameters, and ordered
assertions through `iter_local_origins()`. `rewrite_local_origins()` mutates only
a compile-owned copied node and shares a frozen-carrier identity guard across
the graph. CALL/MAP/LOOP extend those hooks with their cached `child_inputs` and
return a non-`None` baked child from `origin_children()`; ordinary nodes return no
children. CASE and CODE currently need only the common hook.

`compile.origins.iter_flow_origins(flow)` visits wiring, `CompiledFlow.outputs`,
nodes, and their static children in deterministic structural order. Identity
sets for flows, nodes, and frozen carriers make aliases and deliberate child
cycles finite. `rewrite_flow_origins(flow, mapper)` rewrites provenance fields
on an isolated working copy without replacing frozen carrier or `FlowOutput`
objects, then copies that finalized graph for return. The final copy rebuilds
payload dict/set hashes after frozen values change. The input graph stays unchanged,
shared carriers, payload back-references, and children remain shared, and invalid
mapper results fail loudly. Equal origins stored in distinct
carriers remain distinct occurrences. Neither operation inspects binding
sources, parameter defaults, node environments, or render metadata.

`diagnostics.ResolverTable` is an immutable collision-safe routing value. Each
public `OriginRef.key` selects exactly one `(delegate resolver, old OriginRef)`
pair; construction and merge reject duplicate public keys, unknown origins return
`None`, delegate exceptions propagate, and malformed delegate results are
rejected. It does not parse keys, try resolvers in order, or add fallback
precedence.

`compile.origins.rebase_flow_origins(flow, mapper, resolver=...)` first
ordered-deduplicates reachable old origins by opaque key. It calls `mapper` once
per distinct old key, rejects collisions between distinct mapped keys, and
uses that precomputed mapping to return an independent same-type graph plus a
table containing exactly its reachable routes. An originful graph requires an
`OriginResolver`; a source-free graph requires `None` and returns an empty table.
Traversal and rewriting preserve their existing alias/cycle guarantees.

Origin changes remain manifest-, fingerprint-, and expansion-digest-neutral.
None of these operations defines a source-key format or imports Compose; lexical
key construction and returned-flow resolver integration remain front-end work.

## Compose source ownership

`compose/source/` is the sole production Compose package that imports PyYAML. It
constructs root text once as a `SourceDocument` with an immutable, defensive public
cursor interface and keeps the composed and constructed YAML graph private. Each
uncached external file gets one document too. The document owns the original text,
label, semantic value, occurrence topology, and source ranges. Alias and merge
semantics are retained without later mutation of PyYAML nodes or constructed
containers. Persistent pair/arc ropes retain merge topology without expansion,
and completed mapping summaries stop repeated parent DAG scans. Public defensive
snapshots use a stack-safe iterative copier that preserves shared and recursive
aliases.

Source-aware parsing and inline extraction use `SourceCursor`:

- A **physical cursor** navigates one document occurrence. Its value range belongs
  to the shared definition, its path identifies the occurrence, and a mapping child
  also retains its selecting-key range.
- A **virtual cursor** owns a defensive, source-free snapshot for programmatic or
  isolated values.
- A **projected cursor** presents replacement/renamed values while retaining the
  explicitly supplied physical source for each selected child. Desugaring uses
  projections so occurrence-local changes never alter anchors, aliases, or merges.

Inline-body extraction takes one graph-wide nodes snapshot before projection so
unchanged aliases remain shared. Other source-aware field navigation uses cursors
or their package-private read-only view.

`SourceCatalog` is Compose's immutable `OriginResolver`. Its private builder maps
typed semantic paths to opaque origins while retaining each physical document
occurrence and an explicit key/value anchor. Resolution replays that occurrence
into a `DiagnosticFrame`; virtual or unanchored cursors produce no origin, and no
YAML is reparsed or parallel range map created. The loader creates one builder per
physical document and shares it through root, `defs:`, and inline-body assembly.
Authored node headers, ordinary node-input keys and values, and source-backed reserved
MAP/WAIT wiring values, plus source-backed flow-output values and flow/node assertion
occurrences, are registered under their complete lexical owner paths and carried by the
semantic IR rather than a parallel map. Assertion paths include their authored list index,
and each retained item cursor supplies both `Assertion.origin` and the aligned temporary
range. Full roots and multi-node definitions retain a flow assertion cursor, while compact
roots and flat single-node definitions project `asserts:` onto their synthesized node.
Flow-output values are registered after rewrites on `FlowOutput.binding`; END reuses that
binding, and output mapping keys are not origin sites. Generated compact-root and
single-node-definition reexports stay origin-free because their output declaration is a
contract site whose origin remains deferred. Without a retained source occurrence,
Compose mints no origin and preserves any explicit programmatic origin. Private assembly
leaves the resolver unset. At the physical-file boundary, the loader validates and
lexically rebases canonical external children per alias, freezes the shared local catalog,
and installs an exact `ComposedFlow.origin_resolver`; source-free graphs use `None`.

After cursor-based extraction and parsing, descriptors and nodes carry the derived
ranges needed by later phases; no phase reparses source fragments or inspects the
private document graph. The default file resolver adds a separate successful
`resolved Path -> ComposedFlow` cache, so aliases and diamond imports read, parse,
and compile each external physical file once within a successful load graph.

`SourceFrame` is persistent render-only diagnostic metadata. The loader derives its
node/field range maps from descriptors parsed over the existing cursors, stores the
top frame on `ComposedFlow.source`, and gives composite nodes their child's frame as
`child_source` for CLI tracebacks. It carries text and ranges, not parser state, and
is not another source owner.

## Inline composite bodies (compose-only sugar)

An authored CALL/MAP/LOOP may provide a non-empty `nodes:` plus `output:` instead
of a named `call:`. The implementation boundary is deliberately above compile
and runtime:

- `compose.desugar.inline_bodies` makes occurrence-local cursor projections for the
  enclosing node map, retains the original located child cursors, and masks each
  inline occurrence to an ordinary private `call:`. It never mutates authored
  YAML anchors/aliases/merge values.
- The private identity is stable: `__inline_flow_` + SHA-256 of the complete
  lexical owner path. Authored defs/uses/call targets cannot enter that reserved
  namespace, while diagnostics use the friendly enclosing-node label rather than
  the digest.
- One loader-owned callable scope contains authored `defs:`/`uses:` plus a
  private table. Private builders register during assembly and run lazily;
  private/def/uses resolution shares one success cache. Private children and
  defs share the scope's in-progress guard, while the default file resolver's
  load context guards cross-file recursion. An injected resolver is responsible
  for any equivalent protection, so the baked call graph remains finite when
  that seam honors the contract.
- `compose.build.pipeline` associates retained bodies with final descriptors.
  Outer `input:` bindings become generated `Parameter` values through
  `build.inline_inputs`' strict structural proof. Parent descriptors are built
  through a lazy success-only descriptor cache when a direct producer reference
  needs its output type. MAP has a separate one-proof cache: `over:` is inspected
  only when a lone whole-value `${item...}` binding asks for its element type.
- The child assembly receives only those generated inputs and its own node map,
  which prevents parent-pool capture; the ordinary run-global `system` namespace
  remains available. Nested strings inside literal containers are literals, not
  wiring. Inline-only child flow config/assert sections do not exist; existing
  outer-node and enclosing-flow configuration semantics remain unchanged.

The output is the same `CompiledFlow` baked onto the existing CallNode/MapNode/
LoopNode as a named child. Generated IDs remain ordinary `flow_id`/manifest data,
but runtime never interprets their prefix or sees the resolver table/inline source
shape; live growth and replay keep using the kind-blind injected `build_grow` path.

## The `expr` layer (one grammar, everywhere `${...}` appears)

Bindings, conditions, and prompts share ONE expression grammar — the three
divergent dialects the engine grew (binding coalesce/default, `when:`
boolean/arithmetic, prompt builtin-call) are unified. Three modules, one pipeline
(parse → AST → evaluate / ref-walk):

- **`grammar.py` — parse only.** `parse_expr` compiles a span into a Lark tree
  (arithmetic `+ - * / % **`, comparisons, `and`/`or`/`not`, `in`/`not in`, list
  literals, pure builtin calls with dotted access, `:-` default / `:?` required /
  `|` coalesce). No evaluation here. The evaluator walks this **restricted AST** —
  there is no Python `eval`; a builtin call resolves only against the
  `TEMPLATE_FNS` whitelist, so a span can compute but cannot reach arbitrary code.
- **`expressions.py` — evaluate + ref-walk.** `eval_expr` walks the tree over a
  resolver; `expr_refs` collects the reference paths a span reads (the compile-time
  companion, for edge inference + scope checks); `evaluate_when` is the `when:`
  boolean. `RequiredError` (a `${x:?"msg"}` over an unbound ref) lives here, its
  natural home now that the evaluator raises it.
- **`template.py` — the string surface.** `scan_template` splits literal text from
  `${...}` spans; `eval_binding` renders a binding value; `render_template_record`
  renders a strict prompt; `$$` is the scanner's universal escape (a prompt `$$`
  renders a single `$`).

**Three resolve modes** decide what a MISSING reference does, one per context:
`BINDING_NONE` (a binding → `None`, so `|`/`:-` can coalesce), `CONDITION_FALSY`
(a `when:` → falsy), `STRICT_RAISE` (a prompt → raise; no silent blank).

**`call(...)` desugar — a flow call is NOT an expression.** A child-flow call is a
compile-time directive recognized at TWO sites, by ONE whole-value rule (only the
entire trimmed value, never embedded in a span): a binding's whole value, and a
`case` `then:`/`else:` target. At load it desugars into an anonymous `call` node
(`__call_N`) and the host is rewritten to `${<synth>.output}`. A flow call inside
a span (`${flow(args)}`) is a hard `LoadError` — hoist an embedded call to a named
node; a coalesce-of-calls (`${a() | b()}`) is likewise rejected. Pure builtins
(`${upper(x)}`) stay legal inside `${}`.

**Raw-string public API** (what callers outside `expr` use, re-exported from the
`expr` package `__init__`): `eval_binding` (render a binding), `expr_refs_of` (its
reference paths), `evaluate_when` (a condition), plus `render_template_record` /
`prompt_refs` for prompts. Two families back the reference walkers. Extraction:
`expr_refs_of` reads only `${...}` spans (bindings, templates, prompts), while
`condition_refs` parses a WHOLE expression (a `when:`/`on:`/loop predicate/assert
written bare, mixed, or whole-span) so the bare spelling's refs are seen — using
the template walker on a bare condition finds nothing. Rewriting (spelling-
preserving, for the `case` desugar): `rewrite_expr_refs` / `rewrite_template_refs`
(span walkers) and `rewrite_condition_refs` (whole-expression). Callers pass
source strings + a resolver; they never touch the grammar or AST.

## Located errors (when source provenance is available)

A runtime failure with resolvable origin/range data can point at the **exact YAML
line** it originates from, not just the node header. Source-free Python graphs and
generated operations may have no location. Resolution uses structured provenance,
never a text heuristic.

- **`SourceSpan(node, kind, key)`** (in the low-level `events` observer module) —
  `kind ∈ {input, assert, field}`; `key` is the input name, assertion display, or
  field name; `node` is the node id. A binder uses `None` until the evaluation
  funnel supplies the id; a genuinely flow-level location may remain `None`.
- **Carriers:** `NodeFailed.locator` (node-level) and `RunFailed.locator` /
  `RunResult.locator` (flow-level). All default `None`.
- **Evaluator producers:** `BindingError` carries `binding.evaluate` or
  `contract.input`, the best operation origin, and a temporary node-less `input`
  span. Base-wrapper `NodeAssertFailed` carries its exact `Assertion` and becomes
  `assertion.failed`; the same class raised by node code remains `node.execute`.
  Body exceptions become `node.execute`; invalid ordinary/generator terminal
  values and a non-spawner `Grow` become `contract.node_result`. The current
  `Failure.trace` contains one operation origin, or the node origin as fallback.
  `eval_node` fills a legacy locator's missing node id via
  `replace(loc, node=node.id)`;
  dispatcher preparation assigns `contract.output`, `contract.route`, or
  `contract.growth` from the result arm. Output uses the effective target
  contract's root origin when available and otherwise the executing node origin;
  route and growth use the executing node origin. Schema-1 output projection
  retains the original error type and fallback `field` span (`key="output"`).
- **Concrete range (when resolved at the failure site):** ALONGSIDE the symbolic `locator`, a
  **`SourceRange`** (start/end line+col, 1-based; owned by `diagnostics` and temporarily
  re-exported by `events`) may be attached. For an `input` / `field` locator
  produced during evaluation, the funnel keys the node's recorded `field_ranges`
  on the locator's **`(kind, key)`** tuple — kind-blind (on the locator, never `node.kind`), and
  `(kind, key)`-tagged so an input never collides with a like-named field. `field_ranges` is a
  **`FieldRanges`** (`nodes/base.py`), a dict subclass whose `get` **raises `TypeError` on a
  non-tuple key**: a bare-string lookup can never match, and every caller falls back
  (`or node.range`), so without the guard a wrong key silently yields a caret on the wrong line
  — which is exactly how two dead lookups survived review and a green suite. Any new code path
  that builds the map must keep the type (the parse stamp, `_finish`'s copy onto the runtime
  node, the START boundary map, the inline-call desugar all do). An **assert** is located
  BY POSITION: the `run` wrapper resolves the failing assert's own `pre_assert_ranges[i]` /
  `post_assert_ranges[i]` (each aligned 1:1 with `pre_asserts` / `post_asserts`) and raises
  `NodeAssertFailed(range=…)` — so two identical assert texts get distinct carets (asserts are NOT
  in `field_ranges`). It rides `NodeFailed.range` and its flow-level twin `RunResult.range` /
  `RunFailed.range` (all default `None`; the flow twin is the boundary node's resolved range —
  `StartNode.field_ranges[("input", name)]` for a flow input failure,
  `StartNode.post_assert_ranges` / `EndNode.post_assert_ranges` for a boundary / post assert).
  Output-type failures occur after evaluation, so their durable failure may carry only the
  symbolic `field` locator; the CLI resolves it from the supplied flow when no concrete range
  was recorded.
- **Migration boundary:** evaluator and dispatcher rejection now construct one
  canonical `Failure`, while `FlowEngine` projects only its
  message/cause/traceback into schema-1 records and retains locator/range beside
  it. Durable records and CLI rendering therefore still consume the legacy
  projection; durable/public/terminal cutover remains tracked by #182. Authored
  nodes and assertions carry opaque origins, and every
  flow returned by `load_flow` carries one complete source-neutral resolver (or `None` when
  source-free). Runtime and CLI failure rendering still use the legacy locator/range path
  above. For assertions, Compose derives that temporary positional range and
  `Assertion.origin` from the same retained item cursor before rewrites preserve both.
- **Resolution (CLI boundary):** `cli/run.py` prefers the pre-resolved `nf.range` →
  `result.range`; only if both are absent does `_resolve_range` map the span itself, via
  `_top_ranges` (the loaded flow's `SourceFrame` maps, or the defensive
  `loader.top_frame_ranges(document)` fallback): a
  `field` locator → that field's range, else `_node_range` (node-kind best field, e.g. a code
  node's `code:`, → node HEADER range). The winning `SourceRange` is boxed via a
  **`Diagnostic`** (severity + message + primary range + notes) through `render_diagnostic`.
- **Cross-flow call traceback** — when a failure is inside a *called child*, its node id is
  runtime-NAMESPACED (`gate/approve`, `outer/via/boom`, `gate#0/inner` for a map element).
  A `call`/`map` node bakes its child flow's render-only **`SourceFrame`** (`compose/loader`:
  label + source text + node→`SourceRange` / field→`SourceRange` maps, built by `_frame_ranges`
  off descriptors parsed from the existing source cursors) onto `child_source` at load — for
  a `defs:` child the text is the PARENT file (its projected node cursors retain absolute
  parent-file ranges); for an
  external `uses:` child it is that file's own text, label = its filename. The frame is frozen
  and `__deepcopy__`-returns-self, so `clone_child`'s per-callsite deep-copy shares the one
  instance and it is exempt from the node-purity scan (it carries raw `${...}` YAML, but is
  metadata, not a wiring source). At error time `cli/run.py:_walk_call_frames` splits the
  namespaced id and walks segment-by-segment through the baked IR (`node.child`/
  `node.child_source`), collecting one frame per level; `_render_frame_stack` boxes them
  stacked, most-recent-call-last (Python-traceback style), falling back to the single-frame
  box when fewer than two frames resolve. Each call-chain frame anchors at the node HEADER
  line; the failing leaf at its precise field / kind-fallback line. A frame title names WHERE
  it lives: a top/external file frame is the filename alone, a `defs:` frame is
  filename-qualified (`<file> defs:<name>`) since its nodes physically live in that file.

## Design-note template (step 3 of the workflow)

Before coding any engine change, write this short note and confirm any non-obvious
choice (CLAUDE.md "ask when uncertain"):

```
Construct:    <name> — what it consumes -> what it produces (its type signature)
OCaml analogue: <the concept you're borrowing> — how it maps to nodes/pool/runtime
Keep / drop:  <what you borrow from prior engines; what scale baggage you drop>
Lands in:     <layer/package on the ladder>  (new package? write its charter)
Seam:         <injected callable, if it touches an external dependency> | none
Invariants:   <which of the non-negotiables it touches; how it stays within them>
Tests:        <the tests/engine cases that will prove it>
```

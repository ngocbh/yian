# agent_composer — the engine

The implementation of **Agent Composer**: a from-scratch flow engine. A flow is
authored as **Docker-Compose-shaped YAML**, read by the `compose/` loader into a
compiled IR, and executed by the engine's own scheduler — there is no compilation
down to a third-party graph runtime.

The engine is, at heart, a **small functional programming language** — one whose
leaf computations may be LLM agents. If you know OCaml, you already know the
shape of it:

- A **flow is a function**: typed inputs in, typed outputs out, and nothing
  hidden. An **agent is just a flow** whose leaf computation happens to be an LLM
  loop — same contract, same composability.
- Flows **compose**. A node can be another flow, so a large transformation is
  built from small ones nested to any depth — function composition, not a
  monolith.
- Computation is **pure at the boundary**: a node returns an outcome and the
  engine binds it; a node never receives or mutates the shared pool. Each kind
  implements `_run_impl`; the base `Node.run` template applies defaults and fires
  record-only asserts around that body. A flow's own asserts are lowered onto its
  synthesized START/END boundary nodes, so top-level and nested-flow asserts use
  the same ordinary node path.
- The **program structure is fixed by the author**, as a function's call graph is
  fixed by its source. The LLM fills leaf boxes; it does not rewrite the graph at
  runtime. That structural referential transparency makes execution reproducible
  and makes an interrupted run recoverable from its authoritative record prefix.

## Layout

```
diagnostics.py source-neutral ranges, opaque origins, and resolution seam       [leaf]
typesys/      typed value system (values) + variable pool                 [leaf]
events.py     transient run/node observer vocabulary; imports diagnostics
_state.py     shared scheduler-state vocabulary                          [leaf]
expr/         one ${...} grammar: parse -> AST -> evaluate / ref-walk
nodes/        pure Node contract + per-kind implementations; spawners return seeds
              (start, end, agent, code, model, tool, case, call, map,
               loop, human_input, wait)
compile/      compiled IR + validation; identity-safe origin traversal;
              growth builder; stable identity; static-path LLM config maps
compose/      Compose-YAML front end: text -> CompiledFlow; source/ owns YAML
              construction/navigation and opaque-origin resolution; later phases
              lower file-private children
              (source/nodes/parse/desugar/edges/validate/build/loader)
runtime/      run records/logs, reducer-owned RunState, kind-blind dispatcher,
              and worker pool
run.py        public run boundary: ComposedFlow + inputs/log -> RunResult
llm_clients/  provider clients (lazy-imported) + LLMConfig + model_from_config
tools/        tool registry + process-owned ask_user interaction seam
cli/          `ac` commands + main-thread terminal broker
_settings.py  environment-based default provider/model
```

The main import direction is acyclic: `diagnostics` is the lowest
source-neutral leaf; `events` imports it, and nodes may combine it with the
independent `typesys` and `expr` leaves before `compile <- compose`. Runtime
consumes the event/state leaves plus compile's topology, identity, and validation
contracts, but never imports `compile.expand` or `compose`.
The package facade lazy-loads its declared root exports, so importing a leaf
submodule does not bootstrap those higher layers.
Package-root `run.py` sits above `compose`, `compile.expand`, and `runtime`: it
consumes a `ComposedFlow`, injects the kind-aware growth builder, and owns run-log
creation and recovery. None of those packages imports it back.

## Semantic IR carriers

`nodes/contracts.py` is the source-neutral vocabulary that joins opaque
provenance from `diagnostics` with semantic `typesys.Type` values. Each class has
one responsibility:

| Carrier | Meaning |
| --- | --- |
| `Binding` | one flow-owned input or output expression/literal |
| `Parameter` | one formal input, its optional contract, and default/coercion policy |
| `Assertion` | one executable expression plus stable author-facing display text |
| `ValueContract` | one recursive semantic `Type` plus nested origin selection |
| `ContractOrigins` | immutable root/nested origin lookup for a value contract |

All origins are optional, so Python-authored values default to source-free and
may supply an opaque origin explicitly. The carriers are frozen only at their
fields: arbitrary binding/default payloads are retained by identity.
`iter_origins()` visits only declared provenance slots, while `map_origins()`
transforms those slots without inspecting or copying semantic payloads.
`Flow` construction is the single binding boundary:
every direct `flow.wiring[node_id][parameter]` value and every
`FlowOutput.binding` is canonical `Binding`, while raw Python-authored values are
accepted and wrapped origin-free. Expansion rewrites only `Binding.source` and
retains `Binding.origin`; pre-IR Compose dictionaries and runtime-only overrides
remain raw. `FlowOutput.from_` is a temporary read-only source view during the
remaining migration.

`Node.origin` is the same source-neutral provenance shape for an authored node
context. Compose stamps the authored node-header occurrence for root, `defs:`,
and inline-body nodes; direct Python construction defaults to origin-free, and
loader/desugar-synthesized nodes remain so. Clones preserve an existing origin,
but manifests, fingerprints, and expansion digests ignore it. The legacy
`range` fields remain the active render bridge while diagnostics migrate to
origin resolution.

`Node.pre_asserts` and `Node.post_asserts` are ordered `Assertion` lists. Parsed
YAML remains raw text until the inline-call pass crosses each flow assertion into
one carrier; node-local validation creates one fresh carrier per authored
occurrence. Compose indexes those occurrences by source order and derives both
the carrier's opaque `OriginRef` and its temporary positional `SourceRange` from the
same retained item cursor, so equal assertion text at two positions remains two
distinct sites. Flow carrierization preserves an explicit programmatic origin
instead of replacing it. Inline-call desugaring and flow/CALL lowering replace only
`.expression`; child cloning and LOOP respawn retain every carrier field. Thus
`.display` and `.origin` survive every rewrite. Synthesized START/END nodes reuse
the lowered flow carriers.
`Node.run` evaluates the expression and reports the display text. Manifests
include expression, display, order, and duplicate count, while excluding origin
and the temporary positional range bridge.

`Node.params` has one compiled shape too: an ordered list of exact `Parameter`
carriers (or `None` for a directly constructed no-input node). The flow still owns each
parameter's `Binding`; the node owns the formal name, optional `ValueContract`,
required/default presence, and string-coercion policy. Binding validates stored
defaults without replacing their authored payload, and coerces supplied strings
only when the parameter opts in. A bound null shadows a default but still passes
through contract validation, so it is accepted only when the parameter is untyped
or its contract accepts null (a nullable or `None`-kind type).
Parameter/default/contract origins do not affect manifest identity. Compose
`input:` lowers directly into this same exact carrier shape; `ComposedFlow.input`
and START use it without a parallel front-end declaration type. For an ordinary
source-backed node input, final assembly puts the selecting-key origin on its
`Parameter` and the value origin on its flow-owned `Binding`. Source-backed reserved
wiring carries only the value origin on its canonical `Binding`: MAP `over:` and WAIT
`until:` have no corresponding `Parameter`. A desugar may change `Binding.source`
without moving that value origin. When no source occurrence is retained, Compose
mints no origin and preserves any explicit programmatic origin; source-backed
projections inherit their backing site. CASE wiring, synthesized flow-boundary
parameters, contracts, external rebasing, and runtime diagnostic consumption
remain outside this migration step. Source-backed bare and
mapped flow-output values on root flows, multi-node definitions, and inline children stamp
the final `FlowOutput.binding` after desugaring rewrites; the synthesized END boundary
reuses that exact binding. Output mapping keys carry no origin. Compact-root and
single-node-definition reexports are generated without an authored binding occurrence and
remain origin-free; their authored output type is reserved for contract-origin work.

`Node.output_contract` is the canonical compiled output declaration: exactly one
`ValueContract`, or `None` when the output is unenforced. Runtime enforcement and
node-owned structured generation project its semantic `.type`; nested origins remain
attached to the same carrier for diagnostics. The manifest projects only that semantic
type, so changing or rebasing contract origins does not change flow or expansion
identity.

Compose descriptors derive `ValueContract` directly. Builders, child signatures,
producer maps, and validators retain the carrier; algorithms that only compare or walk
types project `.type` at their boundary. CALL and LOOP preserve a child's output
contract, while MAP deliberately derives a new list contract. There is no parallel
output-Type API: Python-authored nodes pass `output_contract=` explicitly.
The authored `output:` / `outputs:` YAML syntax is unchanged.

### Compiled origin traversal

Nodes own their provenance layout. The base `Node` enumerates and rewrites its
node origin, output contract, parameters, and assertions; CALL/MAP/LOOP also own
their cached `child_inputs` origins and expose the baked child through
`origin_children()`. These hooks remain source-neutral and never import
`compile`.

`compile.origins` joins those hooks with flow-owned wiring and declared outputs.
`iter_flow_origins()` walks wiring, outputs, nodes, and static children in stable
order, guarding flow, node, and frozen-carrier identity so shared or cyclic
graphs terminate. `rewrite_flow_origins()` deep-copies the root once, then uses
identity memos for carrier and output replacements; aliases and deliberate
cycles survive while the authored graph remains unchanged. Binding sources,
parameter defaults, node environments, and render metadata are opaque even when
they contain an `OriginRef`. Origins remain excluded from manifests,
fingerprints, and expansion digests.

## One-pass source ownership

`compose/source/` is the only production Compose package that imports PyYAML.
Root text and each uncached external file are constructed once as
`SourceDocument`s, whose public cursor interface is immutable and defensive. They
retain the original text and label while keeping PyYAML's composed/constructed
graph private. YAML aliases and merge mappings are interpreted without mutating
that graph. Persistent pair/arc ropes keep merge topology compact, and completed
mapping summaries bound repeated merge-DAG traversal. Public value snapshots use
a stack-safe iterative copier that preserves shared and recursive aliases. Within
one successful file-import graph, the loader's resolved-path cache also ensures
that each external physical file is read, parsed, and compiled once; aliases and
diamond imports reuse the cached `ComposedFlow`.

Source-aware parsing and inline extraction navigate stable `SourceCursor` values.
A physical cursor combines a definition range with an occurrence path (and a
mapping-key range when applicable); a virtual cursor is a defensive, source-free
value snapshot; a projected cursor replaces or renames values while explicitly
retaining the physical sources of selected children. Descriptors and nodes carry
derived ranges afterward, so no phase reparses text or touches parser objects.

`SourceCatalog` is Compose's immutable `OriginResolver`. Its private builder maps
typed semantic paths to opaque origins while retaining the physical document
occurrence and an explicit key/value anchor. Resolution replays that occurrence
into a `DiagnosticFrame`; it neither reparses YAML nor creates a parallel range
map. Virtual or unanchored cursors produce no origin. One builder is shared by
the root flow, its `defs:`, and its inline bodies, and each assembled
`ComposedFlow` retains a frozen document-local snapshot as `_local_catalog`.
Final assembly registers authored node headers, ordinary node-input keys and values,
source-backed reserved MAP/WAIT wiring values, assertions, and source-backed flow-output
values directly on their canonical semantic carriers; it does not build a parallel
location map. An assertion's semantic path includes its authored list index, and that
item's cursor supplies both `Assertion.origin` and the aligned temporary range. Full root
flows and multi-node definitions retain a flow assertion cursor; their lowered carriers
are reused by START/END. Compact roots and flat single-node definitions instead project
their `asserts:` into the synthesized node, so those sites are node-local. Flow outputs
are stamped after rewrites on `FlowOutput.binding`, and END reuses that binding; mapping
keys are not origin sites. For sites without retained occurrences, Compose mints no origin
and preserves any explicit programmatic origin. Generated compact-root and
single-node-definition reexports are such source-free output sites; the authored output
type is a contract site whose origin remains deferred.
That field is intentionally private: external child origins still require
lexical rebasing before the returned flow can expose one complete resolver.

`SourceFrame` is persistent render-only diagnostic metadata on
`ComposedFlow.source` and composite nodes' `child_source`. It carries source text
and derived node/field range maps for CLI traceback rendering; it neither owns
parsing nor exposes the private document graph.

## Inline composite lowering

An authored CALL, MAP, or LOOP may contain a non-empty `nodes:` plus a required
`output:` instead of naming a child with `call:`. This is entirely a `compose/`
front-end feature. `desugar.inline_bodies` retains the located child body and
masks that occurrence as an ordinary private `call:` before the normal
parse/build/validate pipeline runs. YAML anchors, aliases, and merged mappings
are not mutated: extraction works on occurrence-local cursor projections while
keeping the original child cursors and source ranges read-only.

Each source file owns one callable scope. Alongside its authored `defs:` and
`uses:` namespace, that scope has a resolver-private table for retained inline
children. A child's stable identity is `__inline_flow_` plus a digest of its
complete lexical owner path (root or def, enclosing node, and nested inline
occurrences), not a process counter or object identity. Authored defs, aliases,
and call targets cannot use that reserved prefix, and author-facing diagnostics
show the lexical child label rather than the opaque digest.

Private child builders are registered during assembly and invoked lazily. The
file scope builds each callable at most once on successful resolution, shares
the same cache across private children/defs/uses, and guards private/def
construction in progress. The default file resolver's load context guards
cross-file recursion; an injected resolver is responsible for any equivalent
protection.
Generated child parameters come from the enclosing composite's `input:`
bindings. Their types are proved structurally; parent producer descriptors are
themselves built on demand through one success cache, and a MAP's `over:` element
is proved and cached only if a lone whole-value `${item...}` binding requests it.
Nested literal container strings are deliberately not scanned as references.

The result is an ordinary compiled child baked onto the existing CallNode,
MapNode, or LoopNode. No private resolver state reaches execution, no runtime
kind or graph-rewrite rule is added, and live execution/recovery continue through
the same kind-blind `Grow`/record/reducer path as named children.

## Node contract

A node is a pure function of its bound input record. It returns exactly one member
of the closed result sum:

| Result | Meaning |
|--------|---------|
| `Output(value, commit_as=None)` | one value for the engine to type-check and bind |
| `Route(handle)` | one named outgoing branch, with sibling skip-flooding |
| `Grow(seed, prune)` | one canonical deterministic expansion description |

Failure is a `raise`, not another result variant. `eval_node` binds inputs and
turns the exception into an evaluation failure for the dispatcher to record. A
streaming kind may yield transient `StreamChunk` values and then return one of the
same three results.

The runtime core is **kind-blind**. It dispatches on this result sum and node-owned
traits (`is_spawner`, `binds_per_item`, `bind_reserved`, `grow_depth_delta`,
`is_loop`, and `needs_stop_event`), never on `NodeKind`. The closed kind-aware
growth match lives in `compile.expand.build_grow`, outside runtime. The kind census
test keeps runtime's `NodeKind`/expansion dispatch count at zero.

## One append-only authority

Every run has exactly one `RunLog` and one matching reducer-owned `RunState`.
`MemoryRunLog` is the process-local implementation; `JsonlRunLog` is the durable
implementation, with a nonblocking exclusive advisory lock, canonical JSONL, and
an fsync per append. Both accept the same closed record vocabulary:

```
RunStartedRecord | RunResumedRecord | NodeStartedRecord |
NodeSucceededRecord | NodeRoutedRecord | NodeExpandedRecord | NodeFailedRecord
```

Runtime handles every transition in one order:

```
prepare and validate -> append -> apply reducer -> publish transient observers
```

Preparation cannot mutate live state. If append fails, its state transition is
not applied and its outcome observer is not published. `FlowEngine` is the sole
ready-frontier consumer and state writer; pooled workers only evaluate durably
started nodes and return raw results to the dispatcher. Stream chunks remain
callback-only and never enter the authoritative log.

The package-root `run_flow` boundary commits the complete initialization record
before lending its state/log pair to `FlowEngine`. A direct runtime caller has the
same obligation: apply `RunStartedRecord` to a matching `RunState` and `RunLog`
before driving the engine. The engine borrows this authority and never closes it;
the boundary or direct caller owns its lifetime.

## Durable recovery

`run_flow(..., log=path)` exclusively creates a new JSONL authority. Its first
record captures the run ID, authored-flow fingerprint, complete initial typed
pool, START wiring, worker count, and fully resolved LLM configuration. Expansion
records retain canonical growth seeds plus topology digests, so replay rebuilds
the same nested CALL/MAP/LOOP graph without re-running the spawner.

`resume_flow(flow, log=path)` locks an existing log, durably removes bytes after
the final complete newline, validates its canonical contiguous prefix, and checks
the supplied flow fingerprint. It folds historical records silently through the
same reducer and injected growth builder used live. A terminal prefix is returned
without a write. A nonterminal prefix gets exactly one `RunResumedRecord`, which
puts durably started nodes with no durable outcome back ahead of the ready
frontier before dispatch continues. Recovery uses the recorded input, run ID, and
complete LLM map rather than current ambient defaults.

This provides deterministic engine-state recovery, not exactly-once external
effects: an interrupted node with a durable start and no outcome is retried.

## Blocking effects and structured drain

Authorable blocking effects stay inside their node calls. An AGENT may invoke the
explicitly granted `ask_user` tool, HUMAN_INPUT always invokes the same host-owned
interaction seam, and WAIT blocks on its absolute UTC deadline. AGENT checks the
run-owned stop signal between model/tool effects; WAIT blocks on that signal. Each
returns `Output` from the same invocation.

CODE's soft-budget watchdog is package-local under `nodes/code/` and times CODE
only. Pooled drives own their executors through a full join: an abnormal dispatch
exit signals cooperative stop before joining, so the boundary never releases its
run-log lock while a worker can still produce an outcome.

## Dynamic growth and stable identity

A spawner node returns `Grow(seed, prune)` without importing `compile`. The
kind-aware `compile.expand.build_grow(spawner, seed)` converts the canonical seed
into a deterministic `Flow`; package-root `run.py` injects that callable into both
live execution and replay. Runtime validates and splices the fragment generically,
remaining kind-blind. Expansion digests cover entry/end IDs and topology, while
loop pruning keeps only one dynamic iteration resident at a time.

Compile also owns stable identity for the static program. `flow_manifest`
canonicalizes execution semantics, `flow_fingerprint` hashes that manifest, and
assertion identity includes executable and display text in list order while
remaining origin/range-neutral. `expansion_digest` applies the same projection to
dynamic fragments. `resolve_llm_cascade` returns a complete map keyed by
slash-separated static agent paths. The run boundary reads process provider/model
defaults once at fresh-run initialization; `apply_effective_llm_map` restores the
recorded map during recovery without consulting the environment.

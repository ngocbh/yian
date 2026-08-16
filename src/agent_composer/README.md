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
diagnostics.py semantic failures, resolved presentation, origins/resolver routing [leaf]
typesys/      typed value system (values) + variable pool                 [leaf]
events.py     transient run/node observer vocabulary; imports diagnostics
_state.py     shared scheduler-state vocabulary                          [leaf]
expr/         one ${...} grammar: parse -> AST -> evaluate / ref-walk
nodes/        pure Node contract + per-kind implementations; spawners return seeds
              (start, end, agent, code, model, tool, case, call, map,
               loop, human_input, wait)
compile/      compiled IR + validation; identity-safe origin traversal/rebasing;
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

`diagnostics.Failure` is the canonical source-neutral semantic failure value. It
is a frozen product of `code`, `message`, optional execution `node_id`, an
outermost-to-innermost `tuple[OriginRef, ...]` trace, optional Python
`cause_type`, and optional debugging `traceback`. The exception class is retained
only as diagnostic data; consumers must branch on the closed semantic code,
never on `cause_type` or traceback text. `Failure` is imported from the leaf
module inside lower layers and is lazy-exported from the package root for public
result consumers.

`diagnostics.Diagnostic` is the frozen, resolved presentation product:
`code`, `message`, ordered `DiagnosticFrame` values, free-standing `notes`, and
the optional `node_id`, `cause_type`, and `traceback` copied from a failure. It
has no execution meaning and owns no source-resolution behavior. For a
Compose-authored graph, `ComposedFlow.diagnose(failure)` is the sole front-end
seam that turns a `Failure` into this value. It reads only the failure and the
flow's `OriginResolver`, resolves the complete trace in its existing
outermost-to-innermost order, and preserves duplicate origins. Resolution is
atomic: an empty trace yields an ordinary unlocated diagnostic, while any
missing, incomplete, malformed, or raising resolver for a non-empty trace yields
no frames and exactly the generic note `Source presentation was unavailable.`;
semantic and debug fields remain unchanged.

Compose load failures use the same presentation product directly. A
`LoadError` contains exactly one `Diagnostic`; a Compose-owned factory resolves
retained document/cursor ranges into source-owning frames before the exception
crosses the front-end boundary. Definition and import wrappers create a new
diagnostic with outer context prepended. They never mutate an inner error or ask
the CLI to combine coordinates with separately supplied source text.

Every node builds failures through one source-neutral rule:
`Node.failure_trace(operation_origin=None)` returns its generated, outermost-to-
innermost `origin_context` followed by the supplied operation origin, or by the
node's own origin when no operation origin is supplied. Only an absent local
suffix is omitted; order and duplicates are preserved. Direct and static nodes
start with an empty context. The context is diagnostic-only generated ancestry,
not authored provenance: origin traversal and rewriting ignore it, and manifests,
fingerprints, and expansion digests exclude it.

Compile-owned CALL/MAP/LOOP expansion prefixes cloned children with the
spawner's complete `failure_trace()`. MAP's generated boundaries receive the
same context. A LOOP respawn instead copies only the driver's outer
`origin_context`, so the loop's own origin is appended exactly once when the
next driver builds a failure. Evaluator, dispatcher, and terminal-unreachable
failure producers remain kind-blind and call the same node-owned rule. Runtime
results still retain their temporary `SourceSpan` locator and concrete range,
while the CLI retains its `SourceFrame` graph walk until the remaining
source-diagnostic cutover lands. During that bridge, the evaluator-resolved concrete
range is authoritative at the deepest current baked node that owns its exact locator
site. The source-frame walk supplies outer call context and keeps current
locator/kind/header fallbacks when a range is caller-owned elsewhere in the chain or is
stale after a source-only edit. Those bridges are not an alternative origin-resolution API.

The closed codes are `binding.evaluate`, `contract.input`, `assertion.failed`,
`node.execute`, `contract.node_result`, `contract.output`, `contract.route`,
`contract.growth`, and `graph.terminal_unreachable`. They cover expected
flow-owned rejection. `EvaluationFailure` owns one `Failure` plus temporary
locator/range presentation metadata. The evaluator assigns its first five codes by
operation stage: expression/default/required binding, supplied-input contracts,
base-wrapper assertions, node execution, and result validation. Each producer
passes its best operation origin, if any, through `Node.failure_trace()`; the
node origin is the local fallback after any generated ancestry. Host control,
stream callbacks, unexpected binder faults, and malformed compiled/setup state
escape this expected-failure sum.

The dispatcher assigns the next three codes structurally from the closed result
sum: rejected `Output`, `Route`, and `Grow` data become `contract.output`,
`contract.route`, and `contract.growth`. Output rejection uses the effective
target contract's root origin when present, then the executing node origin;
route and growth use the executing node origin. The trace is empty only when
generated ancestry is empty and neither the operation nor node fallback supplies
a local origin. Unsupported growth kinds and arbitrary builder, compiler,
spawner/fragment-copy, digest, topology-materialization, reducer, append, or log
faults escape rather than becoming flow failures.

At the schema-2 boundary, every trace item becomes an exact base
`OriginRef(key)`: front-end-specific subclasses are valid in memory, but class
identity is intentionally absent from the single-key wire. The reducer carries
that canonical immutable `Failure` through `NodeFailed`, terminal `RunFailed`,
and `RunResult.failure` without flattened aliases; live carriers share the same
object, while recovery reconstructs an equal canonical value. `locator` and
`range` remain adjacent temporary YAML presentation metadata until operation
origins provide equivalent precision. A route that first skip-floods END latches
`graph.terminal_unreachable` with the causal router and its `failure_trace()`. That
`NodeRoutedRecord` persists the derived failure; replay structurally revalidates
the route and restores the recorded value, including its complete trace, even if
the supplied graph now carries rebased origins. No additional terminal record
is added, and an explicit node failure still supersedes the latch.

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
loader/desugar-synthesized nodes remain so. Clones preserve an existing origin.
`Node.origin_context` separately carries generated outer-to-inner spawner
ancestry for nested failure traces; it is empty on direct/static nodes and is
neither enumerated nor rewritten as local authored provenance. Both fields are
diagnostic-only and excluded from manifests, fingerprints, and expansion
digests. The legacy `range` fields remain the active render bridge while
diagnostics migrate to origin resolution.

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

### Compiled origin traversal and rebasing

Nodes own their provenance layout. The base `Node` enumerates and rewrites its
node origin, output contract, parameters, and assertions; CALL/MAP/LOOP also own
their cached `child_inputs` origins and expose the baked child through
`origin_children()`. These hooks remain source-neutral and never import
`compile`.

`compile.origins` joins those hooks with flow-owned wiring and declared outputs.
`iter_flow_origins()` walks wiring, outputs, nodes, and static children in stable
order, guarding flow, node, and frozen-carrier identity so shared or cyclic
graphs terminate. `rewrite_flow_origins()` rewrites an isolated working copy by
identity, then copies that finalized graph for return so payload dict/set hashes
are rebuilt after frozen carrier values change. Aliases, payload back-references,
and deliberate cycles survive while the authored graph remains unchanged. Binding sources,
parameter defaults, node environments, and render metadata are opaque even when
they contain an `OriginRef`. Origins remain excluded from manifests,
fingerprints, and expansion digests.

`diagnostics.ResolverTable` composes opaque namespaces without owning any source
format. Each public origin key routes to exactly one delegate resolver and one origin
in that delegate's namespace. Duplicate public keys are rejected; unknown
origins return `None`; delegate exceptions propagate; and malformed delegate
results are rejected. The table never parses keys or introduces resolver precedence.

`rebase_flow_origins(flow, mapper, resolver=...)` pairs a pure graph rewrite with
those exact routes. It preserves first structural-visit order while deduplicating
reachable old origins by their opaque key, calls `mapper` once per distinct old key,
and rejects collisions between distinct mapped keys before rewriting through
the precomputed map. The result is an independent graph of the same concrete type
plus a table containing only its reachable routes. An originful graph requires a
resolver; a source-free graph requires `None` and returns an empty table. The
operation defines no lexical key policy and knows nothing about Compose or YAML.

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
diamond imports reuse that canonical `ComposedFlow` as the source for their
lexically rebased views.

Source-aware parsing and inline extraction navigate stable `SourceCursor` values.
A physical cursor combines a definition range with an occurrence path (and a
mapping-key range when applicable); a virtual cursor is a defensive, source-free
value snapshot; a projected cursor replaces or renames values while explicitly
retaining the physical sources of selected children. Descriptors and nodes carry
derived ranges afterward, so no phase reparses text or touches parser objects.

`SourceCatalog` is Compose's immutable document-local `OriginResolver`. Its private
builder maps typed semantic paths to opaque origins while retaining the physical
document occurrence and an explicit key/value anchor. Resolution replays that
occurrence into a `DiagnosticFrame`; it neither reparses YAML nor creates a parallel
range map. Virtual or unanchored cursors produce no origin. One builder is shared by
the root flow, its `defs:`, and its inline bodies. Private assembly leaves the public
resolver unset; the loader freezes the builder once when the physical-file root is
complete.
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
At each `uses:` alias, the loader validates the canonical child's complete resolver,
copies and rebases its compiled origin graph under a collision-free lexical key, and
caches that alias view. The physical resolved-path cache remains canonical and is never
mutated. Finalization filters the document catalog and alias tables to the exact reachable
origin set and installs one immutable `ResolverTable` as
`ComposedFlow.origin_resolver`; a source-free graph carries `None`. Repeated calls through
one alias reuse its view, while distinct aliases and diamond branches remain disjoint.

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
| `Output(value, commit_as=None)` | one value for the engine to type-check and bind under the resolved success target |
| `Route(handle)` | one named outgoing branch, with redirect-aware sibling skip-flooding |
| `Grow(seed, prune)` | one canonical deterministic expansion description |

Success and absence resolve their logical output boundaries through distinct
carriers. Success resolves an explicit `Output.commit_as`, then the baked
`Node.commit_as`, then the physical node id. A skip cannot observe result-carried
`Output.commit_as`, so the reducer instead asks the pure, deterministic
`Node.skipped_output_target()` hook, which may consult only manifest-backed node
state. The base hook returns baked `commit_as` or the node id; LOOP returns its
manifest-backed `origin_id` to mirror its dynamic success redirect. A redirected
target has exactly one unfinished physical owner and must be an inert logical
anchor: an expanded spawner or an unknown placeholder with no incoming work, never a
ready, in-flight, or already-committed node; all of its outgoing edges must still
be unknown. The reducer marks the physical node skipped, deduplicates logical
targets, and floods the target's outgoing edges.

Failure is a `raise`, not another result variant. `eval_node` structurally turns
expected binding, assertion, body, and result rejection into
`EvaluationFailure(Failure)`. Result preparation constructs the same canonical
value for expected output, route, and growth rejection. Schema 2 carries that
value unchanged through the durable record, reducer, observer events, and public
result. Host control, stream callbacks, unexpected binder faults, malformed
setup, and arbitrary host/compiler faults escape. A streaming kind may yield
transient `StreamChunk` values and then return one of the same three results.

The runtime core is **kind-blind**. It dispatches on this result sum and consults
node-owned execution-semantic traits/hooks (`is_spawner`, `binds_per_item`,
`bind_reserved`, `validate_execution_setup`, `grow_depth_delta`, `is_loop`,
`needs_stop_event`, and `skipped_output_target`) plus the diagnostic-only
projections `exception_origin` and `failure_trace`, never on `NodeKind`. The
diagnostic projections do not affect program identity. The closed kind-aware
growth match lives in `compile.expand.build_grow`, outside runtime.
`compile.model.GrowthRejected` crosses that boundary only when a legal
CALL/MAP/LOOP kind supplies a malformed seed. Unsupported kinds and builder or
compiler helper faults remain raw; runtime marks only explicit seed, fragment,
budget, depth, and prune-policy rejection as expected growth data. The kind
census test keeps runtime's `NodeKind`/expansion dispatch count at zero.

## One append-only authority

Every run has exactly one `RunLog` and one matching reducer-owned `RunState`.
`MemoryRunLog` is the process-local implementation; `JsonlRunLog` is the durable
implementation, with a nonblocking exclusive advisory lock, canonical JSONL, and
an fsync per append. Both accept the same closed record vocabulary:

```
RunStartedRecord | RunResumedRecord | NodeStartedRecord |
NodeSucceededRecord | NodeRoutedRecord | NodeExpandedRecord | NodeFailedRecord
```

The exact wire version is schema 2. `NodeFailedRecord` contains one nested
`Failure` plus the temporary locator/range bridge; schema 1 is rejected rather
than migrated. Terminal success remains derived from the END value. The first
route whose skip closure reaches END persists its causal `Failure` on that
`NodeRoutedRecord`; replay revalidates the route before restoring it, and no
additional terminal record is written.

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
remaining kind-blind. During that compile-owned construction, CALL/MAP/LOOP
prefix child clones with `spawner.failure_trace()`; MAP also stamps its generated
boundaries, while a LOOP's next driver keeps only the enclosing
`origin_context`. Expansion digests cover entry/end IDs and topology but exclude
that diagnostic context, while loop pruning keeps only one dynamic iteration
resident at a time.

Compile also owns stable identity for the static program. `flow_manifest`
canonicalizes execution semantics, `flow_fingerprint` hashes that manifest, and
assertion identity includes executable and display text in list order while
remaining origin/range-neutral. `expansion_digest` applies the same projection to
dynamic fragments. `resolve_llm_cascade` returns a complete map keyed by
slash-separated static agent paths. The run boundary reads process provider/model
defaults once at fresh-run initialization; `apply_effective_llm_map` restores the
recorded map during recovery without consulting the environment.

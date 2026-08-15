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
typesys/      typed value system (values) + variable pool                 [leaf]
events.py     transient run/node observer vocabulary + source locations  [leaf]
_state.py     shared scheduler-state vocabulary                          [leaf]
expr/         one ${...} grammar: parse -> AST -> evaluate / ref-walk
nodes/        pure Node contract + per-kind implementations; spawners return seeds
              (start, end, agent, code, model, tool, case, call, map,
               loop, human_input, wait)
compile/      compiled IR + validation; growth builder; stable identity;
              effective static-path LLM configuration maps
compose/      Compose-YAML front end: text -> CompiledFlow; file-private inline
              child lowering (nodes/parse/desugar/edges/validate/build/loader)
runtime/      run records/logs, reducer-owned RunState, kind-blind dispatcher,
              and worker pool
run.py        public run boundary: ComposedFlow + inputs/log -> RunResult
llm_clients/  provider clients (lazy-imported) + LLMConfig + model_from_config
tools/        tool registry + process-owned ask_user interaction seam
cli/          `ac` commands + main-thread terminal broker
_settings.py  environment-based default provider/model
```

The main import direction is acyclic: `events <- typesys <- nodes <- compile <-
compose`, with `expr` feeding nodes. Runtime consumes the internal leaf-state
vocabulary plus compile's topology, identity, and validation contracts, but never
imports `compile.expand` or `compose`.
Package-root `run.py` sits above `compose`, `compile.expand`, and `runtime`: it
consumes a `ComposedFlow`, injects the kind-aware growth builder, and owns run-log
creation and recovery. None of those packages imports it back.

## Inline composite lowering

An authored CALL, MAP, or LOOP may contain a non-empty `nodes:` plus a required
`output:` instead of naming a child with `call:`. This is entirely a `compose/`
front-end feature. `desugar.inline_bodies` retains the located child body and
masks that occurrence as an ordinary private `call:` before the normal
parse/build/validate pipeline runs. YAML anchors, aliases, and merged mappings
are not mutated: extraction works on occurrence-local shallow overlays while
keeping the original value nodes and source ranges read-only.

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
`resolve_llm_cascade` returns a complete map keyed by slash-separated static agent
paths. The run boundary reads process provider/model defaults once at fresh-run
initialization; `apply_effective_llm_map` restores the recorded map during
recovery without consulting the environment.

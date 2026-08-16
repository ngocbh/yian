# The engine — running a `CompiledFlow`

> Phase 2 of the [lifecycle](architecture.md). The [front end](compose.md) produced a
> typed graph; the engine *interprets* it. The core (`runtime/engine.py` +
> `runtime/eval_node.py`) is **kind-blind**: it branches on the closed `NodeResult`
> sum and consults node-owned execution traits plus diagnostic projections, never
> a node's `NodeKind`. The node side of the same contract is in [Nodes](nodes.md).

## Two actors, one narrow contract

The node and dispatcher meet at one narrow method.

```
   ┌────────────── DISPATCHER ──────────────┐      node.run(inputs) → NodeResult
   │ sole ready-frontier consumer + writer  │  ───────────────────────────────▶
   │   • RunState (pool, graph, frontier)    │        ┌──────── NODE ────────┐
   │   • RunLog   (append-only authority)    │        │ a PURE function of   │
   │   • post-durable observer publication   │  ◀──── │ its inputs. Never    │
   └──────────────────────────────────────────┘ inputs │ touches the pool.    │
                                                       └──────────────────────┘
```

The dispatcher owns *state and scheduling*; the node owns *computation*. The node is
**pure at the graph boundary** — handed a plain dict plus declared capabilities,
it returns a result and cannot read or write shared memory. An LLM/human/clock leaf
may perform an external effect, but all graph state still lives in one place the
engine controls. The matching `RunLog` is authoritative: a transition is appended
before the reducer applies it to `RunState` or publishes its observer.

## The loop: read → eval → record → reduce

Running a graph is four questions asked over and over: *what can run now? what does it
produce? which canonical transition describes that result? what state does that
transition derive?* The package-root boundary first commits a complete
`RunStartedRecord`. Each node then follows the same recorded path:

```python
while state.result is None:
    node_id = state.ready[0]
    publish(commit_prepared(log, state, prepare_node_started(state, node_id)))
    result = eval_node(flow.nodes[node_id], flow, state.pool)  # bind + evaluate only
    prepared = prepare_result(state, node_id, result)          # validate, no mutation
    publish(commit_prepared(log, state, prepared))             # append, then reduce

def Node.run(self, inputs, **caps):      # the generic template — IDENTICAL for every kind
    inputs = apply_defaults(self.params, inputs)
    check_assertions(self.pre_asserts, inputs)           # expression executes; display reports
    out = self._run_impl(inputs, **caps)                # the ONLY kind-specific step
    if isinstance(out, Output):
        check_assertions(self.post_asserts, {**inputs, "output": out.value})
    return out
```

This pseudocode is illustrative (`prepare_result` stands for the result-specific
preparation functions). The implementation also preserves pristine post-assert
inputs, source ranges, and streaming-generator cleanup.

`Node.run` is a fixed recipe applied to *every* node. The read boundary is kind-blind
but **node-configured** — a node declares *how* to bind rather than the engine branching
on kind (e.g. `binds_per_item` on `map`, `bind_reserved` for a `wait`'s `until` or a
`map`'s `over`). The only kind-specific step is `_run_impl`.

A flow's own top-level `asserts:` are not a separate engine phase: at build they are
lowered onto the synthesized `__start__`/`__end__` boundary nodes. A boundary assert
(`${input.X}` only) is rewritten to read START's coerced `${output.X}`, then rides
`__start__.post_asserts` through the ordinary inherited `Node.run` recipe. A post assert
(`${node.output}`) similarly rides `__end__.post_asserts`. So `run.py` enforces no asserts
itself; a false assert becomes a durable `NodeFailedRecord`, from which the reducer derives
the `NodeFailed` and `RunFailed` observers.

## `NodeResult` — the whole node result vocabulary

A node returns exactly one of three things, and this closed set is the only result
vocabulary the engine branches on:

```python
NodeResult =
  | Output(value, commit_as=None)   # "here is my result"    → bind at its success target
  | Route(handle)                   # "route only, no value" → branch + redirect-aware skip
  | Grow(seed, prune=∅)             # "I'm really a subflow"  → build + splice more nodes
```

```python
def prepare_result(state, node_id, result):
    match result:                         # closed result match — never on node kind
        case Output(value, commit_as):
            return prepare_node_succeeded(state, node_id, value, commit_as=commit_as)
        case Route(handle):
            return prepare_node_routed(state, node_id, handle)
        case Grow() as grow:
            return prepare_expanded(state, node_id, grow, grow_builder)
```

`Output` covers ordinary steps. `Route` covers a pure router (`case`): it stores nothing
and only selects a live out-edge. `Grow` is how a node that *is really a subflow*
expands. Failure is not a `NodeResult`: `eval_node` turns an exception into an
`EvaluationFailure`, which the dispatcher prepares as a `NodeFailedRecord`.

### Logical output redirects

A successful `Output` resolves an explicit result-carried `commit_as`, then the
node's baked `commit_as`, then its physical id. Result-carried `Output.commit_as`
exists only on success; it does not by itself redirect absence. When a route
skip reaches a physical node, the reducer instead calls the pure, deterministic
`Node.skipped_output_target()` hook. The base implementation returns baked
`commit_as` or the node id. LOOP overrides it with its manifest-backed
`origin_id`, mirroring the dynamic `Output(commit_as=origin_id)` returned when it
stops. Any other dynamic success redirect must be mirrored explicitly by this
hook if skipping the node should advance the same logical boundary.

The hook returns an existing logical target and derives only from execution
state represented in the flow manifest. A redirect is exclusive ownership, not
an OR join or multi-writer rule: exactly one unfinished physical node may
redirect to a target, and that target must be an inert anchor—an expanded
spawner or an unknown placeholder with no incoming work, never ready, in flight,
or already committed.
All of the anchor's outgoing edges must still be unknown. The reducer validates
those rules before mutation, marks the physical node skipped, deduplicates
logical targets, and skip-floods the target's outgoing edges. Settled expansion
residue is inert and no longer owns a future logical output.

## Growth: a node that is really a subflow

`call`, `map`, and `loop` are not single steps — each *is a smaller flow*. A spawner
returns only a canonical, kind-shaped seed; no node imports the compile layer or builds
topology. `compile.expand.build_grow` is the sole kind-aware interpreter that turns the
spawner and seed into a self-describing `Flow`. Package-root `run.py` injects that
callable into the kind-blind runtime for both live growth and historical record replay.

One convention unifies all three constructed fragments: **the subgraph is a
well-formed sub-flow with a single `__start__` and `__end__`**. The engine schedules
the `start`, and the builder bakes `commit_as=<spawner id>` onto the terminal — so
however many inner nodes run, downstream sees one clean output committed under the
spawner's id. The terminal's inherited `skipped_output_target()` follows that
same baked redirect when its output is absent. Runtime may consume
`compile.model` topology but never imports
`compile.expand`.

The splice carries a few extra concerns (stamp recursion depth, loop bookkeeping),
but the growth core reads **node-owned traits**
(`grow_depth_delta`, `is_loop`) — never
`node.kind`. A child's asserts are **not** an engine-side splice concern: a called flow's
`asserts:` are lowered onto its own `__start__`/`__end__` at build (just like the
top-level flow, below), and those boundary nodes are cloned into the parent along with the
rest of the child — so the child's boundary asserts fire through `StartNode.run` and its
post asserts through the same base template, on the ordinary node path, identically to
the top level.
The compile-owned builder also propagates diagnostic ancestry: CALL/MAP/LOOP
prefix child clones with `spawner.failure_trace()`, MAP stamps its generated
fan-out/fan-in boundaries with the same context, and LOOP respawn copies only its
driver's outer context so the authored loop origin appears once. Runtime only
splices the resulting fragment and never branches on kind to construct a trace.
`prune` is growth's generic inverse: a self-respawning loop retires the
iteration it just finished in the same step, so it grows without leaking. Adding a new
spawner kind means setting traits on its node; the core never gains a branch. Details:
[Nodes](nodes.md).

## Failure and durable recovery

Node evaluation may raise (a failed assert, a bad LLM response); the engine catches the
exception at the evaluation boundary. The dispatcher then prepares and commits a
schema-2 `NodeFailedRecord` carrying one nested, immutable `Failure` plus temporary
source `locator`/`range` presentation metadata. Only after the record is appended
does the reducer expose that same `Failure` through `NodeFailed`, terminal
`RunFailed`, and `RunResult.failure`. There are no flattened error, type, node, or
traceback aliases. The single-key wire canonicalizes any valid front-end
`OriginRef` subclass to exact `OriginRef(key)`; subclass identity is not durable.
The `Node.on_failure` method remains reserved for a future
error-strategy seam and is not driven today.

For a Compose-authored graph, `ComposedFlow.diagnose(failure)` converts that
source-neutral value into the canonical presentation-only
`diagnostics.Diagnostic`. It copies `code`, `message`, `node_id`, `cause_type`,
and `traceback`, then resolves the complete ordered origin trace atomically
through the flow's resolver. Each node's diagnostic-only `origin_context` is
generated outer-to-inner spawner ancestry and is empty on direct/static nodes.
It is excluded from authored-origin traversal/rewriting and from manifests,
fingerprints, and expansion digests. `Node.failure_trace(operation_origin=None)`
appends the supplied operation origin, or the node origin when none is supplied,
to that context; it omits only an absent local suffix and preserves order and
duplicates. Evaluator, dispatcher, and terminal-unreachable failure producers
all use this node-owned rule without a kind branch. Durable replay restores the
recorded complete trace even when the current graph carries rebased origins.
Runtime results retain temporary locator/range fields, and the CLI retains its
`SourceFrame` bridge until the remaining source-diagnostic cutover.

Every execution uses one `RunLog`. `MemoryRunLog` gives a process-local run the same
ordering semantics; `JsonlRunLog` holds a nonblocking exclusive lock and fsyncs each
canonical JSONL append. The closed record vocabulary is:

```
RunStartedRecord | RunResumedRecord | NodeStartedRecord |
NodeSucceededRecord | NodeRoutedRecord | NodeExpandedRecord | NodeFailedRecord
```

There is no separate terminal record: success is derived when the terminal node's
success transition reduces. Failure is derived either from `NodeFailedRecord` or
from a reduced route that skip-floods the declared terminal output. The first such
route latches `graph.terminal_unreachable` with the causal router and its complete
`failure_trace()`, even when unrelated pooled work drains later, and persists that
failure on the causal `NodeRoutedRecord`. Replay structurally revalidates the route
before restoring the recorded failure; a later explicit node failure still wins
immediately.

`resume_flow(flow, log=path)` locks and validates an existing contiguous prefix,
checks the authored-flow fingerprint, rebuilds a private graph, and folds every record
through the same reducer used live. A terminal prefix is returned without a write. For
a nonterminal prefix, one `RunResumedRecord` places durably started nodes with no durable
outcome ahead of the ready frontier, then dispatch continues. Original inputs, run ID,
worker count, typed values, growth seeds, and resolved LLM configuration all come from
the log rather than current process defaults. See the [Python API](api.md).

## Serial and parallel, one contract

The pseudocode above is the serial path. In pooled mode the dispatcher durably records
each start before submission; workers only evaluate and return raw results or transient
stream chunks. The dispatcher alone prepares, appends, and applies outcomes, so only the
number of nodes in flight differs. A pooled drive owns its workers through a full join:
abnormal dispatch exit first signals cooperative stop, then joins every submitted worker
before the run boundary releases the log.

## The entire interface the engine sees

| The engine sees | Meaning |
|---|---|
| matching `RunState` + `RunLog` | reducer projection and append-only authority, already initialized to the same sequence |
| `node.id`, `node.params` | physical execution identity; input *names* (the flow owns their sources) |
| `node.pre_asserts` / `node.post_asserts` | conditions the base `run` template checks around the body |
| `node.run(inputs, **caps) → NodeResult` | the shared template the engine drives |
| `node._run_impl(inputs, **caps) → NodeResult` | the one kind-specific body step |
| `node.is_spawner` | may it return `Grow`? |
| read-boundary traits | `binds_per_item`, `bind_reserved`, `reserved_wiring_keys`, `needs_stop_event` |
| growth traits | `grow_depth_delta`, `is_loop` |
| diagnostic-only projections | `exception_origin(exc)` selects the local origin; `failure_trace(operation_origin=None)` combines it with ordered ancestry without affecting program identity |
| `node.skipped_output_target()` | logical output whose successors observe this physical node's absence |

That is the complete list — a small set of traits/hooks plus the run/body pair. No `NodeKind` is
visible to the engine.

Inline CODE implements `exception_origin(exc)` by retaining its recursive compiled
code-object tree and selecting the deepest traceback frame owned by identity (`is`).
The wrapped line chooses an exact literal-line origin, then the whole `code:` scalar
and node fallback. Filenames and structurally equal code objects never establish
ownership; reference-mode CODE keeps the base node-origin behavior.

CASE implements the same hook for ordered predicates. Each compiled `Case` carries
the origin of its exact `cases[N].when` occurrence. The hook identifies this node's
active `_run_impl` frame and `Case` by identity, so duplicate predicate text remains
distinct; failures outside predicate evaluation use the node fallback. The carriers
are diagnostic-only, participate in ordinary graph rebasing, and do not affect flow
identity.

## Why a closed match, on the right axis

An earlier engine branched on node *kind* in many places. That dispatch is now fully
removed from the core; a ratchet test (`tests/engine/test_kind_census.py`) holds the
kind-branch count at **zero**. The design keeps the value of an explicit closed `match`
— but on the **`NodeResult`** (three arms) and durable `RunRecord` variants, not the
kind (a dozen and growing). New kinds extend the node side; the engine never changes.

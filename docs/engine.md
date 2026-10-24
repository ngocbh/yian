# The engine — running a `CompiledFlow`

> Phase 2 of the [lifecycle](architecture.md). The [front end](compose.md) produced a
> typed graph; the engine *interprets* it. The core (`runtime/engine.py` +
> `runtime/eval_node.py`) is **kind-blind for behavior**: it branches on the closed
> `NodeResult` sum and consults node-owned execution traits plus diagnostic
> projections; it never dispatches execution or scheduling on `NodeKind`. The node side of the same contract is in
> [Runtime nodes](nodes.md).

## Two actors, one narrow contract

The node and dispatcher meet at one narrow method.

```
   ┌────────────── DISPATCHER ──────────────┐      node.run(inputs) → NodeResult
   │ sole ready-frontier consumer + writer  │  ───────────────────────────────▶
   │   • RunState (pool, graph, frontier)    │        ┌──────── NODE ────────┐
   │   • RunLog   (append-only authority)    │        │ graph-boundary pure  │
   │   • post-durable observer publication   │  ◀──── │ its inputs. Never    │
   └──────────────────────────────────────────┘ inputs │ touches the pool.    │
                                                       └──────────────────────┘
```

The dispatcher owns *state and scheduling*; the node owns *computation*. The node is
**pure at the graph boundary** — handed a plain dict plus declared capabilities,
it returns a node outcome and cannot read or write engine-owned graph state. An LLM/human/clock leaf
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
    out = self.forward(inputs, **caps)                   # the ONLY kind-specific step
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
`map`'s `over`). The only kind-specific step is `forward`.

A flow's own top-level `asserts:` are not a separate engine phase: at build they are
lowered onto the synthesized `__start__`/`__end__` boundary nodes. A boundary assert
(`${input.X}` only) is rewritten to read START's bound `${output.X}`, then rides
`__start__.post_asserts` through the ordinary inherited `Node.run` recipe. A post assert
(`${node.output}`) similarly rides `__end__.post_asserts`. So `run.py` enforces no asserts
itself; a false assert becomes a durable `NodeFailedRecord`, from which the reducer derives
the `NodeFailed` and `RunFailed` observers. `run_flow`, and `resume_flow` of an
unfinished run, only check that every static node's assertions and CASE/LOOP conditions
parse before they write anything; `Node.run` still evaluates them.

## `NodeResult` — the whole node result vocabulary

A node returns exactly one of three things, and this closed set is the only result
vocabulary the engine branches on:

```python
NodeResult =
  | Output(value, commit_as=None)   # "here is my result"    → bind at its success target
  | Route(handle)                   # "route only, no value" → branch + redirect-aware skip
  | Grow(seed, prune=∅)             # "apply a child flow"     → build + splice more nodes
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
and only selects a live out-edge. `Grow` is how a composite node asks the engine
to expand a child flow. Failure is not a `NodeResult`: `eval_node` turns an exception into an
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

## Growth: expanding a child flow

`call`, `map`, and `loop` are composite nodes that apply a child flow. A spawner
returns only a canonical, kind-shaped seed; no node imports the compile layer or builds
topology. `compile.expand.build_grow` is the sole kind-aware interpreter that turns the
spawner and seed into a self-describing `Flow`. Package-root `run.py` injects that
callable into the kind-blind runtime for both live growth and historical record replay.

CALL and MAP fragments clone a child flow with one `__start__` and `__end__`.
The engine schedules the start, and the builder bakes `commit_as=<spawner id>`
onto the derived terminal. Its inherited `skipped_output_target()` follows the
same redirect when output is absent.

A LOOP continuation instead clones the body without a terminal redirect and
wires its END to the next driver. This is where the next carried record is merged.
The body may return a subset of the carried fields. `LoopNode.respawn(k, carried)`
gives the next driver a parameter default for each omitted field. The default is that
field's value in the canonical growth seed. The builder wires only the returned fields
from the body END. Live growth and durable replay build the same fragment from the same
seed. A body that returns every field keeps the same compiled identity and expansion
digests. When the loop stops, that driver returns
`Output(carried, commit_as=origin_id)`; its `skipped_output_target()` also resolves
to the authored loop origin. Downstream therefore sees one logical loop output.
Runtime may consume `compile.model` topology but never imports `compile.expand`.

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
iteration it just finished in the same step. A pruned id is retired with the live nodes
grown under it (by a nested CALL, MAP, or LOOP, transitively) once it and all of them
are settled (taken, skipped, or expanded; none ready or running). Until then they stay
live, along with any pruned id they read, and each later growth retires them once
settled (`RunState.deferred_prune`); a self-pruning LOOP driver is always retired. The
reducer finds grown nodes from accepted growth records (`RunState.growth_owner`). `NodeExpandedRecord.prune_ids` keeps only the ids the node
asked for; replay re-derives the rest. A growth costs its fragment plus its prunes, not
the run's graph so far: the reducer looks up the live nodes it affects through the
adjacency lists and a derived reverse index (`RunState.growth_index`), then applies the
change in place after the record is appended. Adding a new spawner kind means setting
traits on its node; the core never gains a branch. Details: [Nodes](nodes.md).

## Failure and durable recovery

Node evaluation may raise (a failed assert, a bad LLM response); the engine catches the
exception at the evaluation boundary. The dispatcher then prepares and commits a
`NodeFailedRecord` carrying one nested, immutable `Failure`, optional
model/tool evaluation evidence, and forward capture when retention is enabled.
Only after the record is appended does the
reducer expose that same `Failure` through `NodeFailed`, terminal `RunFailed`, and
`RunResult.failure`. There are no flattened error, type, node, or traceback
aliases. The single-key wire canonicalizes any valid front-end `OriginRef`
subclass to exact `OriginRef(key)`; subclass identity is not durable. The
`Node.on_failure` method remains reserved for a future error-strategy seam and is
not driven today.

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
The CLI renders `loaded.diagnose(result.failure)` through the
same `render_diagnostic` path used for load failures. Failed presentation remains
unlocated rather than triggering runtime-id or graph inference.

Every execution uses one `RunLog`. `MemoryRunLog` gives a process-local run the same
ordering semantics; `JsonlRunLog` holds a nonblocking exclusive lock and fsyncs each
canonical JSONL append. The closed record vocabulary is:

```
RunStartedRecord | RunResumedRecord | NodeStartedRecord | NodeStepRecord |
NodeSucceededRecord | NodeRoutedRecord | NodeExpandedRecord | NodeFailedRecord |
MemoryCommitRequestedRecord |
MemoryCommitSucceededRecord | MemoryCommitFailedRecord |
BackwardStartedRecord | BackwardReadyRecord | BackwardFinishedRecord
```

Records use schema 9, and readers accept only schema 9. A log written with
schema 7 or 8 is rejected on read with "run log schema N predates schema 9 and
is not supported"; other versions are rejected too. Recovery preserves the
accepted prefix's canonical bytes and appends new records without rewriting it.

Changed (TS yian): records use schema 19, with a `RunStoppedRecord`, every payload field
required, the time each record was appended (`at`), a `RunStartedRecord` that holds the
flow files, the skills, tool specs and tool groups the AGENTs grant, the yian version and the
parent run, a `PoolCommitRecord` for every write to the variable pool (a derived value's holding `from`
pointers, not the value), no seed on `NodeExpandedRecord`, and node steps numbered within their
node (`n`), an AGENT's model calls and tool results among them, no `evidence` on outcome
records (learning reads the steps), the inbox and wait records, and forward captures that hold
no bound inputs (learning binds each node again from what its route read) and are left off
derived records (rebuilt from their sources); see
[Authoritative run logs](api.md#authoritative-run-logs).

Changed (TS yian): a node's output is a `pool_commit` written before the `node_succeeded` that
names its seq, and the reducer applies it to the pool only when that success is accepted. A
crash between the two leaves an orphan commit: the resume retries the node, which runs again and
writes a new commit; the orphan stays in the log, unread. The run's inputs and system values are
`pool_commit`s right after `run_started`, applied at once; START is ready only after the START
inputs' commit, so a log that ends before it holds no inputs yet and isn't resumable
(`advance` refuses it and leaves it as it is). The reducer keeps a producer index: each pool
name maps to the seq of the accepted `pool_commit` that last wrote it, pruned with the pool and
rebuilt by replay. A derived node (START, END, LOOP, CALL, MAP, whose value only reads earlier
values through its binding) gets an input route in every run; in a non-learnable run its sources
are those commits, and the route is kept in memory only, so such a run still keeps no retained
trace.

Changed (TS yian): derived values are rebuilt on replay, not read. A derived commit (a non-top
`__end__`'s, the stopping LOOP driver's, and the commits an expansion writes for its child STARTs
before its `node_expanded`) holds `from`, the seq of the commit that wrote each pool name its
binding read. When the reducer accepts it, live or on replay, it rebinds the node from those
values, read from the pool and checked against the producer index (a seq that isn't the name's
producer is a `RangeError`: the log and the flow disagree), and runs its pure `forward`. Records
apply in log order, so the pool then holds exactly what the binding read; no seq-to-value store is
kept. A growth's seed is rebuilt the same way from its commits' `from`, before its prune applies,
and checked by `topology_digest`. A child START names its expansion commit in its success.

`RunStartedRecord` holds the complete effective model map of the agents resolved at
run start. An agent whose `llm_config` reads a value known only when it runs (see
[Values in `llm_config`](syntax.md#values-in-llm_config)) is in the optional
`call_time_llm` instead: its base (the layers below its own) and the nonsecret
connection selection it binds from. A run without such an agent omits the field, so
its log is unchanged. The schema stays 9, but a yian version from before the field
can't read a log that holds it.

A `NodeStepRecord` holds one step an in-flight node recorded, such as a TOOL retry's
failed attempt, or the config a call-time AGENT resolved
(`{"effective_llm_config": {...}}`, without keys or header values), recorded before
it opens memory or calls its model. Resume and backward use that recorded config. A node that declares `needs_steps` gets `caps["steps"]`, whose
`record(step)` returns once the step is appended and applied. In a pooled run the
worker hands the step to the dispatcher and waits; a serial run commits it inline.
The reducer keeps a node's steps and its accepted start count across `RunResumed`
until the node settles, so a resumed node reads them back. The runtime gives a step
no meaning and publishes no observer for it. A log holding one can't be read by a
yian version from before the record existed.

Changed (TS yian): a resumed AGENT replays its steps by `n`, in order (the runtime still gives
them no meaning; the replay is the node's). Each recorded `model_call` stands for its call, once
the recorded purpose and request digest match the request the node rebuilds; a mismatch is a
capability fault, so nothing more is written and the run stays resumable. Each tool call with a
recorded `tool_result` gets it back without running. A replayed call with no result is closed
lazily: the resume writes its `tool_result` (`closed: "crashed"`) once the call's check passes,
rather than the crash or stop writing anything. New steps continue the numbering.

Changed (TS yian): the reducer also accepts `RunStoppedRecord`, whose `node_ids` must be
the in-flight nodes in dispatch order; it changes no frontier, start count or step, so a
resume after a stop retries those nodes first. Like every frontier record, it is refused once the
run has a terminal result.

Changed (TS yian): the reducer counts, per node, the attempts a resume retried that no stop
covered; a stop covers every attempt started before it. The counts are history: they stay after
the node's outcome and after a prune. `resumedAttemptEvents` in `ts/src/runtime/engine.ts` builds
the `NodeAttemptResumed` event per retried node that a resume will publish, and `inspectRun` will
report the counts, once `advance` and `inspectRun` are ported.

Node outcome records require forward capture when the recorded run
has `learnable=True` and omit it when `learnable=False`. Optional model/tool evaluation
evidence is separate. There is no separate forward terminal record: success is derived
from the terminal value once the graph is quiescent. Failure is derived either from
`NodeFailedRecord` or from a reduced route that skip-floods the declared terminal
output. The first such route latches `graph.terminal_unreachable` with the causal
router and its complete `failure_trace()`, even when unrelated pooled work drains
later, and persists that failure on the causal `NodeRoutedRecord`. Replay
structurally revalidates the route before restoring the recorded failure; a later
explicit node failure still wins immediately. A flow from `load_flow` doesn't reach
this through its outputs: the loader gives an output whose producers a `case` may skip
the explicit null default, so END still runs. Hand-built graphs can still reach it.

SessionMemory owns saved conversation state. Runtime does not reconstruct
conversations or restore memory from the log. An AGENT that declares a session
memory gets an optional turn from that memory key. Every call site of that
AGENT shares the key's one conversation, in FIFO admission order. The evaluator
owns the turn's lifetime through wrapped execution and cleanup, not later
dispatcher acceptance. Runtime binds authored scope and accepted run/start
identity without interpreting provider messages. Learned notes reuse the
existing per-memory learning/commit/recall lifecycle; they do not add runtime
conversation state. API summary selection and derived cache also belong to
SessionMemory, with injected adapter/model callbacks. Optional SQLite storage
belongs to the memory backends, which open it on first use. Hosts build memory
with `build_memories`; `run_flow` never constructs it. `yian run` uses the
same builder, in RAM or under `--memory-root`.
See [SessionMemory foundation](api.md#sessionmemory-foundation).

Changed (TS yian): there is no session memory, so no memory holds a conversation and an
AGENT's memory context is only its recalled notes; a `session` memory is a load error. A
chat's history is the value a LOOP carries from turn to turn, recorded as node outputs in
the run log.

When retention is enabled, the trace keeps the actual terminal producer and
accepted invocation contexts through graph pruning. The reverse-feedback fold
walks those outcomes in reverse acceptance order, collecting every incoming
message before calling an occurrence's `backward` hook once. Node code owns the
transformation; recorded input routes carry its validated feedback upstream.
Opaque expression functions stop transfer. Accepted START results preserve flow
input feedback even where an input came from a literal or default.

Each occurrence's result carries source and message references for attribution.
Equal feedback values remain separate contributions. Unreached accepted
occurrences remain in the inventory with empty feedback, so later learning can
use their history independently of feedback reachability. This fold does not
change forward scheduling or the forward result.

## Explicit backward attempts

`FlowEngine` drives forward scheduling only. `learnable=True` retains accepted
execution independently of memory recall, then the public run boundary returns
the forward result. `RunResult.backward(feedback)` later calls the explicit
`runtime.backward` driver against the same record/reducer authority. Ordinary
CODE/TOOL/HUMAN_INPUT/WAIT nodes stop feedback; START, END, and a stopping
LOOP project it mechanically. A reached AGENT uses at most one tool-free model
call from its accepted prompt, locals, evidence, and effective configuration.
No default, render, recall, expression function, or forward node is rerun.

Changed (TS yian): the evidence is rebuilt from the run log for each backward attempt: the
invocation's steps, its prompt rendered from the recorded flow and skills, and a recall against
the recorded snapshots. No node runs and no model is called; see
[explicit backward feedback](api.md#explicit-backward-feedback).

The driver validates root feedback before accepting `BackwardStartedRecord`.
It completes the reverse fold, builds one context per memory key from the
recorded memory stacks, and runs one learner per memory key, in sorted key
order. All proposals must be valid before one `BackwardReadyRecord`; a known
computation/learner failure can finish earlier, and no memory commits occur
before Ready. Each memory key then gets one ordered batch, including an empty
one, with a durable `MemoryCommitRequestedRecord` before the backend call and an
accepted success or failure observation afterward. Commit records require a
positive `attempt_id`, and every attempt uses `ac-memory-v3` commit IDs. A known
commit failure stops later memories. `BackwardFinishedRecord` records
the separate final outcome, never replacing the forward output or failure.

`runtime.learning` owns shared pure message ordering/transfer validation,
root projections, attempt state, and owner-context construction. Ready replay
validates the exact accepted occurrence inventory and routes without invoking
callbacks or materializing transcript contexts. Memory contexts include all
accepted outcome kinds; a memory without executions gets an empty context. Their
feedback keeps original root-relative execution coordinates; each occurrence's
recorded memory stack selects the keys that receive it. Unattributed root
feedback goes only to memories declared on the root flow.

No effective feedback means no AGENT backward model work, but every memory
key's learner still runs. Every explicit call starts a new attempt and recomputes it
with fresh sessions from the original snapshots; Ready proposals are never
resumed. An unfinished earlier attempt is first closed as interrupted. An
unconfirmed request remains `unknown`, including when the
backend returned but its observation could not append. Host/cancellation/log
faults escape with the accepted prefix and never create a completion record.

Changed (TS yian): stops are signals, so a backend's typed `MemoryOperationError` is a
learning failure whatever its `cause`, and a drive finishes only its own attempt: one
overtaken by a later call while it awaits (its opener, a handler, a learner or a commit)
is refused after that await and appends nothing further.

`resume_flow(flow, log=path)` locks and validates an existing contiguous prefix,
checks the authored-flow fingerprint, rebuilds a private graph, and folds every record
through the same reducer used live. After any incomplete-tail repair, a terminal prefix
is returned without appending a record or invoking backward/memory work, even
when its last backward attempt was interrupted. For an eligible nonterminal prefix, one
`RunResumedRecord` places durably started nodes with no durable outcome ahead of the
ready frontier, then dispatch continues. Original inputs, run ID,
worker count, typed values, growth seeds, and resolved LLM configuration all come from
the log rather than the current LLM defaults.

Conversation metadata does not block ordinary node retry. A retried node may
resend a model request or repeat tool actions: recovery provides no exactly-once
guarantee. Each retried start receives a new sequence, and metadata for the old
unfinished start is no longer accepted. Runtime does not reconstruct a transcript
or fabricate completion for an unknown result. Saved conversation history belongs
to the memory backend, not the run log. See the
[Python API recovery contract](api.md#durable-recovery-with-resume_flow).

## Serial and parallel, one contract

The pseudocode above is the serial path. In pooled mode the dispatcher durably records
each start before submission; workers only evaluate and return raw results or transient
stream chunks. The dispatcher alone prepares, appends, and applies outcomes, so only the
number of nodes in flight differs. A pooled drive owns its workers through a full join:
abnormal dispatch exit first signals cooperative stop, then joins every submitted worker
before the run boundary releases the log. When a node outcome fails the run, the
dispatcher signals stop and publishes that durable `NodeFailed` and `RunFailed` before
the join, so a host learns of the failure while siblings are still blocked.
Each pooled node runs in a separate copy of the host's context variables, so
host effects retain their cancellation context and node-local context changes
do not leak into the dispatcher or later worker calls.

Changed (TS yian): workers are async evaluations, not threads: at most `num_workers` are in
flight, each starts only after its `NodeStarted` record is durable and published, and each
settles into a FIFO completion queue (it never rejects) that the dispatcher reads one entry at a
time, so outcomes commit in settle order. Steps commit inline from the node, since a step is
appended and applied before `record` returns in both modes, and stream chunks and in-loop events
go from the node straight to `onStream`/`onEvent` rather than through the dispatcher. On a `RunFailed` the dispatcher
aborts the run signal, publishes the failure, then waits for every running worker and discards
what they produce; any other error aborts the run signal, waits for every worker, and ends the
run with the first error. A CODE body runs on a separate worker thread
(see [`code`](syntax.md#code--tool--the-other-leaves)).

Changed (TS yian): a run stops through an `AbortController` tree. `FlowEngine` takes the host's
`signal`; each node gets its own child signal as `caps.signal`, linked to the run signal by one
`{ once: true }` listener removed when the node settles, so listeners a node leaves behind die
with it. Once the run signal is aborted (the host's signal, a failing host callback, a
run-ending node failure, a host fault) nothing more starts, a step throws the node signal's
reason and appends nothing, every outcome that settles afterwards is discarded, and every
running node is awaited before `run()` settles; `run()` then rejects with the error that started
the stop (the host's `signal.reason`, the callback's error), never a sibling's, while a node
failure stays the run's `RunFailed`. A host stop and a host callback that throws or rejects
(`onEvent`, `onStream`, steering) append one `RunStoppedRecord` (`signal` or `callback`) when
the stop is requested, listing the nodes in flight, before anything winds down; other host
faults append none, so their attempts count as crashed on resume, and nothing is appended
after a terminal result or a second time in one segment. Inside an AGENT turn a tool call that
had already settled when the stop arrived still records its step (`record(step, { settled: true
})`) and publishes its `ToolCallFinished`, as long as the run has no terminal result; other
events after a stop are dropped. Each node's evaluation runs in a fresh `AsyncResource` created
by the dispatcher, so it starts in the `AsyncLocalStorage` context of the `run()` call and a
store a node sets stays in that node's async chain; callbacks the dispatcher calls (`onEvent`
for durable observers) run in the dispatcher's context. A nested run started from a node with
`signal: caps.signal` stops when its parent stops, and a child's failure doesn't stop the
parent. A `RunLog`'s `append` must not call back into the run.

## The core node interface

| The engine sees | Meaning |
|---|---|
| matching `RunState` + `RunLog` | reducer projection and append-only authority, already initialized to the same sequence |
| `node.id`, `node.params` | physical execution identity; input *names* (the flow owns their sources) |
| `node.pre_asserts` / `node.post_asserts` | conditions the base `run` template checks around the body |
| `node.run(inputs, **caps) → NodeResult` | the shared template the engine drives |
| `node.forward(inputs, **caps) → NodeResult` | the one kind-specific body step |
| `node.is_spawner` | may it return `Grow`? |
| read-boundary traits | `binds_per_item`, `bind_reserved`, `reserved_wiring_keys`, `needs_stop_event`, `needs_tools`, `needs_live_events` |
| memory/evidence traits | `needs_memory_context` opts into recalled notes and an optional transcript; `needs_skills` receives the run's host `skills=` mapping; `needs_evaluation_trace` opts into model/tool evidence when `learnable=True` (TS yian has no such trait: learning reads steps, through `learningMaterial`); `memory_scope` supplies optional memory context and validates recalled memory keys |
| forward capture | recorded `learnable=True` retains general binding data independently of memory scope |
| output metadata | `output_contract` validates committed values; `commit_as` selects a logical success target |
| growth traits | `grow_depth_delta`, `is_loop` |
| diagnostic-only projections | `exception_origin(exc)` selects the local origin; `failure_trace(operation_origin=None)` combines it with ordered ancestry without affecting program identity |
| `node.skipped_output_target()` | logical output whose successors observe this physical node's absence |

Changed (TS yian): `run` and `forward` are async, every node gets its stop as `caps.signal`
instead of the `needs_stop_event` trait, and no node returns a streaming generator, so the
streaming-generator cleanup in [the loop](#the-loop-read-eval-record-reduce) is Python only;
see [the node contract](nodes.md#the-contract).

Changed (TS yian): the evaluator (`evalNode` in `ts/src/runtime/eval_node.ts`) resolves to an
`EvaluatedOutcome` and never touches the run state. When a node rejects after its signal
aborted, the evaluation rejects with the signal's reason, whatever the node threw, instead of
classifying it as a failure.
When a capability fails with a host fault (a stream or event sink, steering, the memory opener,
render capture, `steps.record`), the first such fault is what the evaluation rejects with once
the node settles, even if the node caught it and returned; capability promises the node
started are awaited first. A node declaring `needsMemoryContext` gets
`openMemoryContext(baseSystemPrompt, renderedPrompt)`; the memory opener the engine is given
takes `(scope, recall, { signal })`, where `recall` is the node's `RecallContext`, and resolves
to a `MemoryContext` of recalled sections. There is no transcript to close: the capability only
refuses opens once the outcome is decided, as the steps capability and render capture refuse
their calls. A node records nothing for learning while it runs. The backward driver
(`learningMaterials` in `ts/src/runtime/backward.ts`) rebuilds what learning reads of each
occurrence through its node's `Node.learningMaterial(inputs, steps, caps)` hook, from the steps
the node recorded, its render with the recorded skills and a recall against the recorded memory
snapshots; an AGENT checks each rebuilt request against the digest its step recorded.

The pattern is a small set of traits and hooks around the shared run/body pair.
Runtime may read `NodeKind` for diagnostic text, but behavior and scheduling never
branch on it.

Inline CODE implements `exception_origin(exc)` by retaining its recursive compiled
code-object tree and selecting the deepest traceback frame owned by identity (`is`).
The wrapped line chooses an exact literal-line origin, then the whole `code:` scalar
and node fallback. Filenames and structurally equal code objects never establish
ownership; reference-mode CODE uses the whole authored `code:` origin as its
invocation site, then the base node fallback.

CASE implements the same hook for ordered predicates. Each compiled `Case` carries
the origin of its exact `cases[N].when` occurrence. The hook identifies this node's
active `forward` frame and `Case` by identity, so duplicate predicate text remains
distinct; failures outside predicate evaluation use the node fallback. The carriers
are diagnostic-only, participate in ordinary graph rebasing, and do not affect flow
identity.

The AGENT node also specializes `exception_origin(exc)` without adding a new carrier. A
retry-exhausted Pydantic error, or its directly wrapped native-parser cause, is
accepted only when exact traceback frames prove the generated structured-output
schema performed the validation; deeper cause chains are not traversed. The first
error location keeps record fields, removes only the generated scalar/list wrapper,
and selects the existing output contract's exact, nearest-ancestor, or root origin.
Tool/provider validation errors, non-validation failures, and source-free contracts
retain the node fallback. The exception, retry policy, and kind-blind runtime do not
change.

## Why a closed match, on the right axis

The engine core never branches on node *kind*; a ratchet test
(`tests/engine/test_kind_census.py`) holds the
kind-branch count at **zero**. The design keeps the value of an explicit closed `match`
— but on the **`NodeResult`** (three arms) and durable `RunRecord` variants, not the
kind (a dozen and growing). New kinds extend the node side; the engine never changes.

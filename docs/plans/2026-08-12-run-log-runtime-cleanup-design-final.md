# Run-Log Runtime Cleanup, Final Candidate

**Date:** 2026-08-12
**Issues:** [#110](https://github.com/ngocbh/agent-composer/issues/110),
[#111](https://github.com/ngocbh/agent-composer/issues/111), and
[#34](https://github.com/ngocbh/agent-composer/issues/34)

## Changes From Version 2

This revision:

- derives success and `TerminalSkipped` only when the whole graph is quiescent;
- canonicalizes durable values through a wire encode/decode before live use;
- records and verifies an expansion-topology digest and removes node-to-compile imports;
- separates model-safe tool invocation from trusted HUMAN_INPUT metadata injection;
- replaces worker-owned terminal input with a main-thread CLI broker and defines callback,
  streaming, and interruption lifecycles;
- removes durable abort state and treats Ctrl-C as a resumable host interruption;
- defines UTC-based absolute WAIT behavior; and
- makes `/exit` and `/quit` explicit routes in chat flows.

## Goal And Boundary

Replace checkpoint suspension and synthetic agent continuation graphs with a small,
node-atomic runtime. A node runs until it returns or raises. Human input and timed waits
block that node's worker. An append-only run log is the sole durable run state and can
reconstruct an unfinished expanded DAG, pool, dispositions, ready order, and in-flight
order at node boundaries.

Parallel execution through `num_workers` remains first-class. The dispatcher is the
single state and log writer; workers only evaluate nodes. The runtime remains kind-blind,
and nodes never write the variable pool.

Python stacks, active model requests, tool calls, and inline code cannot be serialized.
After process exit, every `NodeStarted` without a durable outcome is retried from the
beginning. Its external effects, including a human question, may happen again. Recovery
is therefore at-least-once for an in-flight node.

The user confirmed UTC-based absolute WAIT interpretation and flow-owned `/exit` and
`/quit` routing before implementation planning.

## Runtime Roles

The implementation keeps three concepts separate:

1. **Worker outcomes** are in-process `Output`, `Route`, `Grow`, or captured failure
   values. They may contain live Python objects and are never logged directly.
2. **Run records** are the closed serializable transition vocabulary. The dispatcher
   alone appends them, and each record maps to one atomic reducer call.
3. **Observer notifications** are public views produced after a durable record has been
   reduced. They may contain decoded values and derived events such as skips or terminal
   summaries.

`StreamChunk` is deliberately transient. It is neither a run record nor reducer input,
is never replayed, and is delivered through `on_stream` rather than `on_event`.

## Authoritative Records And Terminal Semantics

Each JSONL envelope contains the schema version, run ID, contiguous sequence number,
record kind, and payload. The authoritative records are:

| Record | Complete reducer transaction |
| --- | --- |
| `RunStarted` | Initialize immutable run identity, canonical initial state, and ready START node. |
| `RunResumed` | Move the validated unfinished set to the front of ready and record the selected worker count. |
| `NodeStarted` | Remove the expected ready head and append it to ordered in-flight. |
| `NodeSucceeded` | Remove in-flight, commit one canonical typed value, update edges/skips, and enqueue successors. |
| `NodeRouted` | Remove in-flight, take one route, derive the skip flood, and enqueue successors. |
| `NodeExpanded` | Remove in-flight, verify and splice one expansion, prune named IDs, update growth state, and enqueue its entry. |
| `NodeFailed` | Remove in-flight and make the run terminal-failed with diagnostics. |

There is no authoritative `NodeSkipped`, `RunSucceeded`, `RunFailed`, or `RunAborted`
record. Skips and terminal summaries are derived from a successful reducer transaction,
so a crash cannot land between a causal state change and a required summary record.

An END output is not by itself terminal. END may commit while an independent sibling is
ready or running. The dispatcher continues scheduling and committing until both `ready`
and `in_flight` are empty. Only then does the reducer derive:

- success when END committed its output;
- `TerminalSkipped` when END is unreachable.

It then emits the corresponding terminal observer and constructs `RunResult`.

A sibling `NodeFailed` before quiescence wins even if END committed earlier. Node failure
is immediately terminal and starts drain; success and `TerminalSkipped` never start drain
early. The causal record that empties both collections produces the derived run-level
notification after draining is complete.

`NodeSucceeded` records the executing node ID, the effective `commit_target` already
resolved from result/baked redirect/executing ID, and the canonical wire encoding of the
value validated against that target's declared type.

`NodeExpanded` records the executing spawner ID, canonical wire encoding of its seed,
sorted prune IDs, and a SHA-256 digest of the canonical expansion topology.

## Run Initialization And Identity

A fresh run is initialized in this order:

1. Compute the supplied `ComposedFlow`'s stable recursive build fingerprint.
2. Deep-copy its compiled graph; no run operation mutates `loaded.compiled`.
3. Resolve the LLM cascade and environment defaults on the private copy.
4. Capture effective per-agent LLM configuration by static node path.
5. Coerce inputs, seed the complete pool, and construct START's per-run wiring override.
6. Wire-encode and decode the durable initial pool so live and replayed state use the
   same canonical representation.
7. Append, flush, `fsync`, and reduce `RunStarted`.

The copy precedes `resolve_llm_cascade`, because model resolution also mutates its target.

`RunStarted` stores the full canonical typed initial pool, including the `system`
namespace (`today`, `now`, `run_id`, and host ambients), primed START value, and
`VariablePool.start_id`. Replay restores those values and does not read a new clock or
mint a new run ID. It also stores START's narrowed wiring override, the effective agent
LLM map, and initial `num_workers`.

The build fingerprint is SHA-256 over canonical JSON from a representation-neutral build
manifest. The manifest sorts node and wiring keys, preserves only semantically ordered
collections, and recursively includes static child flows. It includes node kinds and
execution fields, authored LLM layers, inputs, outputs, edges, wiring, asserts,
environment, code/reference strings, and nested children. It excludes source ranges,
object identities, caches, credentials, and run-resolved values.

The fingerprint detects a changed authored flow. It does not attest to changed imported
Python functions, tool implementations, model services, credentials, or other external
dependencies. Expansion digests separately detect changed dynamic builder topology.

## Run State

One private `RunState` owns:

- the static graph copy plus chronological expansion overlay;
- the canonical typed variable pool;
- node and edge dispositions;
- ready FIFO and ordered in-flight set;
- terminal status and output or diagnostics;
- reference/growth depth by live node;
- START wiring overrides; and
- deterministic indexes derived from graph and expansion history.

Adjacency, recursive skip closures, observer notifications, and ready additions are
derived. The log stores chronological expansion records rather than graph snapshots,
hidden checkpoint maps, or a nested `GrowRecord` tree.

Depth is recomputed when each `NodeExpanded` is folded. The reducer removes pruned nodes'
graph, disposition, pool, depth, and derived-index entries as one transaction.

## Dispatch And Commit Pipeline

No worker reads or removes work from ready. In serial and pooled modes the dispatcher:

1. selects the next ID from its ready FIFO;
2. prepares `NodeStarted` for that exact head;
3. appends, flushes, and `fsync`s the record;
4. pops the expected head and appends it to in-flight in the reducer;
5. emits the durable observer notification; and
6. only then runs or submits the node.

The dispatcher never submits more evaluations than the available worker capacity. There
is no executor backlog of not-yet-running futures that can begin external work after a
failure. If submission fails, the durable state contains an unfinished start; execution
stops and recovery requeues it.

A completed worker outcome follows one path:

```text
raw outcome
  -> prepare, validate, wire-encode, and wire-decode
  -> build immutable apply plan from the canonical decoded value
  -> append + flush + fsync one authoritative record
  -> total reducer apply
  -> derived observer callbacks
```

Preparation resolves every operation that could normally fail during apply:

- bound output type and wire codec;
- effective commit target and target existence;
- route handle and complete branch/skip closure;
- grow seed codec, sorted prune set, expansion build, and topology digest;
- node/edge/wiring ID collisions and referenced IDs;
- total-node and reference-depth budgets; and
- ready, disposition, depth, and graph deltas.

Outputs and grow seeds are first encoded to their durable wire form and then decoded.
The decoded value is canonical: live pool commit and live expansion building use that
decoded value, exactly as replay does. The record stores the matching wire form. The
live pre-encoding Python object is not retained in the apply plan.

This does **not** claim that arbitrary `Any` interiors are recursively lossless. For
example, JSON may canonicalize a nested tuple to a list. The contract is live/replay
equivalence at the durable boundary, not preservation of every Python container type.
If the codec rejects a value, preparation produces `NodeFailed` before any pool or graph
mutation. Regression coverage includes nested tuple-, date-, and file-shaped values in
both outputs and growth seeds and compares uninterrupted state with replayed state.

Preparation creates an immutable apply plan. After its record is durable, reducer apply
uses only total container/state operations. Replay decodes each record, prepares it
against reconstructed state, and applies only after all compatibility checks pass.

Parallel completions commit in the order accepted by the dispatcher. Record sequence
numbers preserve that order, and each reducer transaction appends its successors to the
ready FIFO. Queue snapshots are unnecessary.

## Expansion Contract And Layer Direction

Growth becomes seed-first:

```python
Grow(seed=builder_input, prune=frozenset(...))
```

CALL, MAP, and LOOP nodes calculate and return this carrier. They do not build graph
fragments and do not import anything from `compile`.

Expansion construction belongs to the higher compile/compose layer. That layer installs
a plain `grow_builder(spawner, canonical_seed)` callable at the run boundary. The
dispatcher/runtime receives it by injection and invokes it generically; it neither
imports `compile.expand` nor branches on node kind or expansion subtype.

The import direction is explicit:

```text
nodes imports typesys / expr
compile imports nodes
compose imports compile
run imports compose + compile + runtime
run injects compile-owned grow_builder into runtime
```

`nodes` never imports `compile`; `runtime` receives the builder as a callable rather
than importing the compile expansion implementation. This removes the current
`compile.model -> nodes -> compile.expand` cycle.

During live preparation, the dispatcher:

1. wire-encodes and decodes the seed;
2. calls the injected builder with the canonical decoded seed;
3. validates the returned graph fragment;
4. emits a canonical topology manifest containing entry/terminal IDs, sorted nodes and
   their build-relevant fields, edges, and wiring; and
5. stores the SHA-256 manifest digest in `NodeExpanded`.

During replay it performs the same build from the recorded canonical seed and compares
the resulting digest with the record before mutating state. A changed but still valid
builder is therefore rejected as log/flow incompatibility instead of silently producing
a different DAG.

Flat records preserve every chronological expansion, including loop iterations. Replay
first reconstructs the clone that performs a later expansion, invokes the builder for
that clone, verifies its topology digest, and then applies recorded prunes.

Agent human input no longer expands the graph, so agent expansion seeds, memo records,
resume-agent nodes, and origin redirect repair disappear.

## Human Input And Tool Trust

Delete `controls:` and the control registry. An agent receives human access as an
ordinary authored tool:

```yaml
tools: [ask_user]
```

The process-global registry contains one reserved `ask_user` entry. Only `question` is
part of its public model-facing `tool_call_schema`. Structured HUMAN_INPUT fields such
as `questions`, `answer_schema`, `title`, and `node_id` are trusted host metadata, not
model arguments.

Before invoking any model-requested tool, the agent loop:

1. verifies that the tool name appears in that node's authored tool list;
2. explicitly rejects every argument key absent from the tool's public
   `tool_call_schema`;
3. validates the remaining arguments with that public schema; and
4. invokes the registry entry through the public path.

The explicit key check is required even when the underlying schema library would ignore
unknown fields. A model attempt to supply an injected `ask_user` metadata key fails the
agent node; it is never stripped and forwarded.

`HumanInputNode` uses a separate trusted internal registry invocation path. It renders
and validates its question specification, then passes both the public question and its
injected structured metadata directly to the same registered implementation. The
separate path is a trust boundary, not a separate registry or tool implementation.

Tool exceptions propagate as node failures rather than becoming model-visible fallback
strings. Model tool calls execute in returned order, and `ask_user` has no special branch
inside the agent loop.

`ask_user` retains startup-only host ownership. Installation reserves the ID, rejects a
second active owner, and removes only its own registration on exit. Concurrent runs in
one process cannot use different handlers; that process-global constraint was accepted.

## WAIT Contract

The recommended blocking WAIT contract is:

- an ISO date means `00:00:00` UTC on that date;
- a naive ISO datetime is interpreted as UTC;
- an offset-aware datetime is normalized to UTC;
- a deadline at or before the current instant returns immediately;
- waiting uses the real UTC wall clock, not a monotonic timestamp stored in the log;
- the worker waits on an engine cancellation `Event` with the remaining timeout rather
  than calling an uninterruptible sleep; and
- crash recovery recomputes the remaining duration from the same absolute deadline, so
  time elapsed while the process was down is honored.

The cancellation event is transient lifecycle machinery. It lets failure drain, callback
failure, and host interruption wake WAIT workers; it is not a pause, flow result, durable
record, or public cancellation API. If woken because execution is stopping, the worker's
outcome is discarded.

Event/watch WAIT has no blocking provider and is deleted with `Pause`.

This is the confirmed public WAIT contract.

## Streaming And Observer Failures

Workers never call `on_stream` or terminal UI directly. They put `StreamChunk` values on
a transient worker-to-dispatcher queue. The dispatcher drains that queue while work is
running and invokes `on_stream`. Chunks are best-effort, are not logged, and are never
replayed. Once stop-dispatch begins, queued and late chunks are discarded.

`on_event` is invoked by the dispatcher only after the corresponding record is durable
and reduced. Historical records are folded with callbacks disabled. On recovery,
observers see `RunResumed` and newly appended records only; terminal-log replay returns
the reconstructed result without replaying old callbacks.

If `on_event` or `on_stream` raises, the dispatcher captures the original exception,
stops new dispatch, signals transient cancellation, drains already-submitted work while
discarding outcomes/chunks, releases the log lock, and then re-raises that exception.
No callback failure is converted into `NodeFailed` or appended as a flow outcome.

## CLI Main-Thread Terminal Broker

The CLI owns terminal interaction on the process main thread:

1. the main thread installs the registry-backed `ask_user` handler and terminal broker;
2. it runs `run_flow` on an engine/controller thread;
3. a node invoking `ask_user` enqueues a broker request and blocks its node worker on
   that request's result;
4. CLI observer adapters enqueue render updates rather than touching Rich directly; and
5. the main thread alone renders Rich output, reads input, handles SIGINT, and fulfills
   or cancels pending requests.

The broker serializes concurrent questions naturally because only the main thread reads
the terminal. There is no worker-held Rich/input lock and no prompt that SIGINT cannot
reach. This works in `num_workers=0` as well: the controller thread may block in node
evaluation while the main thread services its broker request.

Ctrl-C is a host interruption, not a flow abort. The main thread asks the controller to
stop dispatching, cancels all pending broker requests, and signals the engine cancellation
event. The dispatcher discards cancellation outcomes and late streams, performs structured
drain where workers cooperate, releases the log lock, and reports the interruption so the
main thread re-raises `KeyboardInterrupt` (or the CLI's equivalent host exception).

There is no `RunAborted`, `request_abort`, aborted `RunResult` status, resume command, or
public cancellation token. The broker uses only an internal transient stop signal. A
Ctrl-C log remains nonterminal and resumable; every unfinished `NodeStarted` is retried.
If a non-cooperative external call prevents graceful drain, process termination remains
the hard boundary: workers cannot be killed safely, the OS releases the lock, and the
last complete durable prefix remains resumable.

Node failure differs from interruption. A prepared and durable `NodeFailed` immediately
stops dispatch and begins the same structured drain, then returns a failed `RunResult`.
Log append/flush/fsync failure and observer failure are host/runtime exceptions: they
stop dispatch and drain but do not invent a durable terminal flow record.

## Log File Ownership And Recovery

`runtime/log.py` owns the record codec and low-level JSONL reader/writer. `run.py` owns
public path policy and supplies an opened writer to the engine.

A durable writer:

- creates fresh logs with `O_CREAT | O_EXCL` and mode `0600`;
- holds one OS-backed exclusive writer lock for the whole run and drain;
- writes one complete UTF-8 JSON line, flushes, and calls `fsync` before returning;
- locks resumed logs before reading or validation;
- physically truncates and `fsync`s an unterminated final line before appending; and
- rejects malformed complete lines, sequence gaps, schema/run-ID mismatches, flow
  fingerprint mismatches, and expansion digest mismatches.

Per-record `fsync` is intentional: it provides the strongest complete-prefix guarantee
but may dominate runtime for short nodes, especially under parallel execution. This
durability cost must be documented and measured; implementation must not silently weaken
the contract.

Logs may contain inputs, prompts, human answers, model/tool/code outputs, diagnostics,
and tracebacks. They are sensitive data. Restrictive default permissions reflect that.
A library call with `log=None` uses the same records/reducer in memory but cannot recover
across processes.

`resume_flow`:

1. locks and repairs the JSONL tail;
2. validates envelopes and the supplied fresh flow's recursive fingerprint;
3. deep-copies the supplied compiled graph;
4. applies logged effective LLM configuration without reading new defaults;
5. folds records through prepare and the same total reducer, with callbacks disabled;
6. verifies every reconstructed expansion topology digest;
7. returns immediately if the reconstructed state is terminal;
8. otherwise appends and reduces `RunResumed`, whose retry list must equal the ordered
   replay-derived in-flight set; and
9. prepends retries to ready in original dispatch order and continues.

`RunResumed` records the worker count selected for that segment. Omitting an override
reuses the latest recorded value; an explicit `num_workers` changes future scheduling.

## Public API

The run boundary becomes:

```python
run_flow(flow, inputs, *, log=None, num_workers=0, on_event=None, on_stream=None, ...)
resume_flow(flow, *, log, num_workers=None, on_event=None, on_stream=None)
```

`resume_flow` accepts no live engine, checkpoint, command, human answer, fresh inputs, or
new LLM configuration. The log supplies the exact canonical initial pool and effective
run configuration.

`RunResult` keeps inputs, success/failure status, output/error diagnostics, observer
events, and traceback; adds run ID and optional log path; and removes `engine`,
`checkpoint`, and `pause_reasons`. There is no aborted status. Log creation, locking,
submission, append, observer, and host-interruption failures are exceptions rather than
false flow failures.

The CLI always chooses and reports a durable log path. Library callers opt in with
`log=`. Resuming a succeeded or failed log is read-only and returns its existing result.

## `ac chat`

The recommended design makes textual exit commands flow-owned. The bundled/example chat
flow routes HUMAN_INPUT values `/exit` and `/quit` to an `exited=true` turn result before
the reply agent. Other text reaches the reply agent and transcript fold. EOF may be
normalized by the broker to the same flow input. Ctrl-C remains the host interruption
defined above and leaves a resumable nonterminal log.

This rule applies equally to user-supplied chat flows: they must implement the same route
if `/exit` and `/quit` should exit without reaching the model. Without that route, those
strings are ordinary input and may be sent to the model. The CLI does not secretly
intercept them under the flow-owned design.

The user chose this flow-owned behavior specifically to demonstrate author-visible engine
composition. Host-owned textual command interception is not part of this change.

For normal turns, `run_flow` remains active while HUMAN_INPUT blocks a worker. A durably
reduced reply event is sent through the broker and printed before the next prompt.
Failed-turn resilience stays host-owned: the CLI mirrors the last completed transcript,
and after a failed reply reloads the immutable flow and starts a new run/log from that
transcript.

## Package Cleanup

Delete rather than deprecate:

- `suspension/` and checkpoint/restore/command code;
- `Pause`, pause reasons, runtime paused/deferred queues, and checkpoint ledgers;
- `RunAborted`, `request_abort`, and aborted-result handling;
- `nodes/agent/controls/` and the `controls:` compose field;
- `AgentEntry`, `Fresh`, `Resume`, the `entry` constructor argument, pending memo state,
  agent expansion, and synthetic HUMAN_INPUT/resume-agent nodes;
- event/watch WAIT descriptors and builders; and
- CLI pause/resume prompting loops.

The intended layout is:

```text
events.py          public observer notifications; StreamChunk is transient
runtime/
  engine.py        bounded dispatch, worker and drain lifecycle
  state.py         RunState, preparation, and total reducer
  log.py           durable records, codec, JSONL lock and repair
  eval_node.py     bind and evaluate one node into a raw worker outcome
```

`state.py` replaces `state_manager.py`; ready and in-flight ownership are no longer split.
Durable record types remain separate from convenient public notifications. Package
charters and the documented layer ladder are updated in the implementation.

`AgentNode` has one authored constructor and one local model/tool loop. Its conversation
exists only inside the running node stack. A crash retries the entire agent node.

## Verification

Implementation acceptance requires tests for:

- replay after every authoritative record prefix in nested CALL/MAP/LOOP runs;
- equality of graph topology, pool, dispositions, ready/in-flight order, growth depth,
  START overrides, and terminal result after replay;
- END committing before an independent sibling finishes, including a later sibling
  failure that wins, and success/`TerminalSkipped` only at full quiescence;
- exact restoration of system values, effective LLM configuration, and commit target;
- serial and positive-worker execution with bounded submissions and dispatcher order;
- start-before-submit and retry after submit failure;
- route-derived skips without authoritative skip records;
- output and seed wire canonicalization with nested tuple/date/file-shaped values;
- identical canonical values and builder inputs in live execution and replay;
- expansion digest equality and rejection of a changed-but-valid builder topology;
- an import ratchet proving CALL/MAP/LOOP and all `nodes` code import no `compile` code;
- partial-tail repair, corrupt complete lines, identity mismatches, creation permissions,
  and two-process writer exclusion;
- append/fsync failure proving no proposed transition was reduced or observed;
- per-record fsync behavior and its measured short-node cost;
- model-tool rejection of injected/unknown keys and trusted HUMAN_INPUT metadata;
- duplicate `ask_user` owner rejection and registry cleanup;
- broker serialization of concurrent prompts, Rich updates, EOF, and SIGINT;
- callback exceptions causing stop, wake, drain, lock release, and re-raise;
- transient worker-to-dispatcher streaming, silent historical replay, and discarded late
  chunks;
- failure/log-error/Ctrl-C while siblings block on human input and WAIT;
- resumability of a host-interrupted log and absence of durable abort state;
- repeated use of one `ComposedFlow` without graph or LLM mutation;
- the confirmed WAIT timestamp rules and cancellation behavior;
- the confirmed chat `/exit`, `/quit`, EOF, multi-turn, and failed-turn behavior; and
- deletion ratchets for pause, checkpoint, agent entry, controls, synthetic resume,
  event WAIT, abort API, and suspension imports.

The runtime kind census remains at zero. The log records effects at node boundaries
without teaching the engine about agent, human-input, wait, call, map, or loop kinds.

## Deferred Work

This change does not remove streaming generators, the always-failing MODEL node, the
unused failure hook, or the old prompt parser. It does not include parse-once/source
diagnostic cleanup under #81/#91/#103 or a general CLI split. Stream transport changes
only enough to make existing notifications transient and lifecycle-safe.

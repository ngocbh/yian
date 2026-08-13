# Run-Log Runtime Cleanup, Version 2

**Date:** 2026-08-12
**Issues:** [#110](https://github.com/ngocbh/agent-composer/issues/110),
[#111](https://github.com/ngocbh/agent-composer/issues/111), and
[#34](https://github.com/ngocbh/agent-composer/issues/34)

## Changes From Previous Version

This revision makes the log a crash-consistent state machine rather than an event
transcript:

- durable records, raw worker outcomes, and observer notifications are separate roles;
- every durable record is one complete reducer transaction, with skips and terminal
  summaries derived by the reducer;
- the dispatcher owns the ready queue and records `NodeStarted` before submitting work;
- worker outcomes pass through prepare/validate/encode, append and flush, total reduce,
  and only then observer callbacks;
- `RunStarted` stores the full typed initial pool and fully resolved run configuration;
- `NodeSucceeded` stores its effective commit target, not an optional redirect hint;
- expansion uses one node-owned seed builder in live execution and replay, and records
  the executing spawner, typed seed, and sorted prune IDs;
- the run state explicitly accounts for growth depth and START wiring overrides;
- JSONL ownership includes exclusive locking and physical repair of a partial tail;
- failure, abort, and log-write behavior with blocked sibling workers is defined;
- `RunAborted` remains authoritative, while `StreamChunk` remains explicitly transient;
- `ask_user` has one exact registry schema, registration lifecycle, and CLI terminal lock;
  and
- `ac chat` keeps failed-turn recovery and has a proposed flow-owned `/exit` redesign.

## Goal And Boundary

Replace checkpoint suspension and synthetic agent continuation graphs with a small,
node-atomic runtime. A node runs until it returns or raises. Human input and timed waits
block that node's worker. An append-only run log is the sole durable run state and can
reconstruct an unfinished expanded DAG, pool, dispositions, ready order, and in-flight
order at node boundaries.

Parallel execution through `num_workers` remains first-class. The dispatcher is the
single state and log writer; workers only evaluate nodes. The runtime remains kind-blind,
and nodes still never write the variable pool.

Python stacks, active model requests, tool calls, and inline code cannot be serialized.
After a process exit, any `NodeStarted` without a durable outcome is retried from the
beginning. Those external effects, including a human question, may therefore happen
again. This is deliberately at-least-once recovery for an in-flight node.

## Three Runtime Roles

The implementation keeps three small concepts distinct:

1. **Worker outcomes** are in-process values returned by evaluation: `Output`, `Route`,
   `Grow`, or a captured failure. They may contain live Python objects and are never
   logged directly.
2. **Run records** are the closed, serializable transition vocabulary. The dispatcher
   alone writes them, and one record maps to one atomic reducer call.
3. **Observer notifications** are public, convenient views produced after a durable
   record has been reduced. They may expose decoded Python values and derived events
   such as skipped nodes or terminal summaries.

`StreamChunk` is the deliberate exception: it is a transient, best-effort notification
with no state transition and is never written to the run log. Streaming uses a separate
`on_stream` callback rather than `on_event`. It is not replayed, and chunks received
after a terminal or log failure are discarded. All `on_event` calls correspond to a
record that has already been flushed and reduced. A causal node notification may be
emitted immediately after that reduce; a derived run-level terminal notification is held
until structured draining finishes.

## Authoritative Run Records

Each JSONL envelope contains the schema version, run ID, contiguous sequence number,
record kind, and payload. The authoritative records are:

| Record | Complete reducer transaction |
| --- | --- |
| `RunStarted` | Initialize the immutable run identity, initial state, and ready START node. |
| `RunResumed` | Move the validated ordered unfinished set to the front of ready and record the selected worker count. |
| `NodeStarted` | Remove the expected ready head and append it to ordered in-flight. |
| `NodeSucceeded` | Remove in-flight, commit one typed value to the explicit effective target, update edges/skips, and enqueue successors. |
| `NodeRouted` | Remove in-flight, take one route, derive the full skip flood, and enqueue successors. |
| `NodeExpanded` | Remove in-flight, build and splice one expansion, prune named IDs, stamp growth state, and enqueue its entry. |
| `NodeFailed` | Remove in-flight and make the run terminal-failed with its diagnostics. |
| `RunAborted` | Make the run terminal-aborted when an external abort is observed. |

There is no authoritative `NodeSkipped`, `RunSucceeded`, or `RunFailed` record.
`NodeSkipped` notifications, when useful to an observer, come from the skip set returned
by a successful reducer call. A successful END commit makes the state terminal-succeeded;
an END made unreachable by a route/skip transaction makes it terminal-failed. The
reducer derives `RunSucceeded` or `RunFailed` from that same causal record. A separate
summary record is unnecessary, so a crash cannot land between cause and summary.

`NodeSucceeded` carries:

- the executing node ID;
- the **effective `commit_target`**, already resolved from the result, baked node
  redirect, or executing ID; and
- the losslessly encoded `TypedValue` validated against that target's declared type.

`NodeExpanded` carries the actual executing spawner ID, a typed/JSON-safe seed, and a
sorted list of prune IDs. It does not contain a live `Flow` object.

## Run Initialization And Identity

A fresh run is initialized in this order:

1. Compute the supplied `ComposedFlow`'s stable recursive build fingerprint.
2. Deep-copy its compiled graph. No run operation may mutate `loaded.compiled`.
3. Resolve the LLM cascade and environment defaults on the private copy.
4. Capture the fully effective per-agent LLM configuration by static node path.
5. Coerce inputs, seed the complete `VariablePool`, and construct START's per-run wiring
   override.
6. append, flush, and reduce `RunStarted`.

The copy happens before `resolve_llm_cascade`; otherwise model resolution still mutates
the reusable loaded flow even if dynamic growth is isolated.

`RunStarted` stores the full typed initial pool, not just caller inputs. This includes the
entire `system` namespace (`today`, `now`, `run_id`, and any host ambients), the primed
START value, and `VariablePool.start_id`. Replay restores these values exactly and never
reads a new clock or mints a new run ID. It also stores START's narrowed wiring override,
the effective per-agent LLM configuration, and the initial `num_workers`.

The build fingerprint is SHA-256 over canonical JSON emitted from a representation-neutral
build manifest. The manifest uses sorted node/wiring keys and canonical type/literal forms,
preserves order only where order is semantic, and recursively includes each static child
flow. It covers node kinds and execution fields, authored LLM layers, inputs, outputs,
edges, wiring, asserts, environment, code/reference strings, and nested children. It
excludes source ranges, object identities, caches, credentials, and run-resolved values.
The effective LLM map is logged separately and is reapplied on resume instead of consulting
new environment defaults.

The fingerprint detects a changed flow build. It cannot attest to a changed imported
Python function, tool implementation, model service, credential, or other external
dependency; that limitation is explicit in the API documentation.

## Run State

One private `RunState` owns:

- the static graph copy plus its chronological expansion overlay;
- the typed variable pool;
- node and edge dispositions;
- the ready FIFO and ordered in-flight set;
- terminal status and its output or diagnostics;
- reference/growth depth by live node;
- START wiring overrides; and
- deterministic indexes derived from the graph and expansion history.

Adjacency, recursive skip closures, observer notifications, and the ready additions from
a transition are derived. The initial graph is supplied by the fingerprint-matched flow;
the log stores only chronological expansion seeds and prunes. Hidden checkpoint-only
maps and the nested `GrowRecord` tree disappear.

Depth is recomputed and stored when each `NodeExpanded` is folded. The same node-owned
growth traits used live are used during replay, so `MAX_REF_DEPTH`, loop origin metadata,
terminal redirects, and later nested expansion see the same values. The reducer also
removes pruned nodes' graph, disposition, pool, depth, and derived-index entries as one
transaction.

## Dispatch And Commit Pipeline

No worker reads or removes work from the ready queue. In both serial and pooled modes the
dispatcher:

1. selects the next ID from its own ready FIFO without exposing the queue to a worker;
2. prepares `NodeStarted` for that exact head;
3. appends and flushes the record;
4. atomically pops that expected head and appends it to in-flight in the reducer;
5. emits the durable `NodeStarted` observer notification; and
6. only then runs/submits the node.

If submission fails, the durable state is an unfinished start. The engine stops and the
next `resume_flow` requeues it. No worker-side dequeue or worker-emitted start event exists.

A completed worker outcome follows one path:

```text
raw outcome
  -> prepare, validate, and encode without mutating RunState
  -> append + flush one authoritative record
  -> total reducer apply
  -> derived observer callbacks
```

Preparation resolves and validates everything that could normally fail during apply:

- bound output type and typed encoding;
- effective commit target and target existence;
- route handle and complete branch/skip closure;
- grow seed encoding, sorted prune set, and node-owned subgraph build;
- node/edge/wiring ID collisions and referenced IDs;
- total-node and reference-depth budgets; and
- all ready, disposition, depth, and graph deltas.

Preparation produces an immutable apply plan beside the durable record. Applying that
plan consists only of total container/state operations. During replay, each decoded record
is prepared against the reconstructed state before it is applied; a builder or invariant
mismatch is reported as log/flow incompatibility before that record mutates state.

An invalid worker outcome is converted during preparation into a prepared `NodeFailed`
record. A log append failure is different: the proposed transition is not reduced and no
observer is called for it.

Parallel worker completions are committed in the order the dispatcher accepts them. That
order is captured by record sequence numbers. Successors are appended to the ready FIFO
by the corresponding reducer transaction, so replay reconstructs the exact remaining
queue without logging queue snapshots.

## Expansion Contract

The growth outcome becomes seed-first:

```python
Grow(seed=typed_builder_input, prune=frozenset(...))
```

Each spawner exposes one pure node-owned `build_grow(seed) -> Flow` hook. `call`, `map`,
and `loop` use that same hook for live preparation and log replay; there is no separate
live subgraph plus `replay_grow` path. The runtime invokes the hook generically and never
branches on `NodeKind` or expansion subtype.

Flat `NodeExpanded` records preserve every chronological expansion, including loop
iterations. The executing clone ID is logged before it can be pruned. Replaying records in
order first reconstructs the clone that performs a later expansion, then uses that clone's
builder, and finally applies the recorded prune. This replaces the lossy, origin-collapsed,
nested checkpoint ledger with a direct state-transition history.

Agent human input no longer expands the graph, so agent expansion seeds, memo records,
resume-agent nodes, and origin redirect repair disappear.

## Failure, Abort, And Worker Lifetime

The first terminal cause wins. Whenever a reduced record makes the run terminal, including
an END success, an END made unreachable, a prepared node failure, or an external abort, the
dispatcher stops dispatching new work and stops committing worker outcomes. Failure and
abort are represented by the causal `NodeFailed` or `RunAborted`; success and an unreachable
END are derived from their causal node record. On a log append/flush failure the dispatcher
also stops immediately, but cannot claim a durable terminal record; the last complete JSONL
prefix remains the authority.

Already-running Python stacks cannot be killed safely. The engine therefore uses structured
draining:

- it continues collecting but discards all late outcomes and chunks;
- it waits for every already-running worker to return;
- it does not emit the run-level terminal notification, return `RunResult`, release the log
  lock, or raise a log I/O error until the drain completes; and
- it never claims that an arbitrary model, tool, code call, or terminal read was cancelled.

This can wait indefinitely for a held human-input worker or a long sleep. That is the
accepted consequence of the simpler held-worker model. Engine-owned timed waits may check a
cooperative terminal flag to shorten a drain, and a CLI prompt may check before opening,
but correctness does not depend on cancellation and an already-blocked terminal read cannot
be forcibly awakened. Worker threads are not detached as a clean-shutdown strategy.

If the process dies during a failure drain, replay sees the already-durable terminal record
and returns that terminal result; dangling starts are ignored because the run is terminal.
If it dies during a log-write-error drain, replay sees only the last valid nonterminal prefix
and retries all unfinished starts. No late result is ever appended after the first terminal
or log failure.

`request_abort` remains supported. It produces authoritative `RunAborted` rather than a
pause command. A terminal aborted log is idempotent on resume.

## Human Input As A Registered Tool

Delete `controls:` and the control registry. An agent receives human access exactly like an
ordinary tool:

```yaml
tools: [ask_user]
```

The registry contains one reserved `ask_user` tool schema. Only `question` is exposed to
the model; structured HUMAN_INPUT metadata is accepted as injected arguments:

```python
def ask_user(
    question: str,
    questions: Annotated[list[dict] | None, InjectedToolArg] = None,
    answer_schema: Annotated[list[dict] | None, InjectedToolArg] = None,
    title: Annotated[str | None, InjectedToolArg] = None,
    node_id: Annotated[str | None, InjectedToolArg] = None,
) -> Any: ...
```

An agent call supplies only `question` and receives the handler's value stringified into a
normal `ToolMessage`. `HumanInputNode` still renders templates and parses/validates its
question specifications, then invokes the same registry entry with `question=prompt or ""`
and the injected metadata. Its raw handler return becomes `Output` and is validated against
the node's declared output type during dispatcher preparation.

The tools package exposes one `invoke_tool(tool_id, inputs)` path used by both `ToolNode`
and the agent loop. Before invoking any model-requested name, the agent loop verifies that
the name is in that node's authored `ctx.tools`; a hallucinated or globally registered but
ungranted tool fails the node. Tool exceptions propagate as node failures rather than being
converted into model-visible error strings. Multiple tool calls execute in model order;
`ask_user` has no special branch.

`ask_user` has startup-only host ownership. A small install/context API builds the canonical
schema, reserves the ID against generic registration, rejects a second active owner, and
removes only its own registration on exit. The CLI installs its handler before starting a
run and removes it only after the engine has drained. A library host may instead keep one
process-wide installation. Replacing a handler while a run is active is unsupported and
rejected. Consequently, concurrent runs in one process cannot use different human handlers;
the user explicitly accepted the registry's process-global constraint.

The CLI handler and progress renderer share one terminal coordinator lock. Before prompting,
the handler stops Rich live rendering, renders either the simple question or the existing
structured questionnaire, reads the answer, and restores progress before releasing the
lock. Observer rendering uses the same lock. Concurrent human-input workers therefore queue
without interleaving prompts or corrupting the progress display.

## WAIT And Pause Removal

`NodeResult` becomes exactly:

```text
Output | Route | Grow
```

`HumanInputNode` blocks in `invoke_tool("ask_user", ...)` and returns `Output`. Timed
`WaitNode` parses its resolved timestamp, blocks the worker until that time, and returns
`Output(None)`. The unauthorable event/watch WAIT mode has no blocking provider and is
removed in this change rather than left behind without `Pause`.

Delete `Pause`, `PauseRequested`, `RunPaused`, pause reasons, commands, checkpoints,
live-engine resume, and the `suspension/` package. Ordinary prompt cancellation or handler
errors are tool/node failures; there is no parked run to return.

## Log File Ownership And Recovery

`runtime/log.py` owns the record codec and low-level JSONL reader/writer. `run.py` owns the
public path policy and supplies the opened writer to the engine.

A durable writer follows these rules:

- a fresh run creates its file with `O_CREAT | O_EXCL` and restrictive `0600` permissions;
- fresh and resumed runs hold one OS-backed exclusive writer lock for the whole run/drain;
- append writes one complete UTF-8 JSON line, flushes, and `fsync`s before returning;
- resume opens read/write and locks before reading or validating;
- if EOF follows bytes without a newline, resume physically truncates to the previous
  newline and `fsync`s before any new append;
- malformed newline-terminated JSON, a sequence gap, schema/run-ID mismatch, or fingerprint
  mismatch is fatal and is never truncated or guessed around; and
- the lock remains held from validation through the last append, preventing two resumptions
  from choosing the same next sequence number.

Logs can contain inputs, prompts, human answers, model/tool/code outputs, diagnostics, and
tracebacks. Documentation treats the file as sensitive; the default permissions reflect
that. A library call with `log=None` uses the same records and reducer in memory but has no
cross-process recovery.

`resume_flow` performs:

1. lock and repair the JSONL tail;
2. validate envelopes and the supplied fresh flow's recursive fingerprint;
3. deep-copy the supplied compiled graph;
4. apply the logged effective LLM configuration without reading new defaults;
5. fold every record through prepare plus the same total reducer;
6. if terminal, return the reconstructed result without appending or executing;
7. otherwise prepare, append, and reduce `RunResumed`, whose retry list must equal the
   replay-derived ordered in-flight set; then
8. prepend those retries to the existing ready FIFO in original dispatch order and continue.

`RunResumed` records the worker count selected for that resumed segment. An omitted override
reuses the latest recorded value; an explicit `num_workers` changes only future scheduling.

## Public API

The run boundary becomes:

```python
run_flow(flow, inputs, *, log=None, num_workers=0, on_event=None, on_stream=None, ...)
resume_flow(flow, *, log, num_workers=None, on_event=None, on_stream=None)
```

`resume_flow` accepts no live engine, checkpoint, command, human answer, fresh inputs, or new
LLM configuration. The log supplies the exact initial pool and effective run configuration.

`RunResult` keeps input, status, output/error diagnostics, observer events, and traceback;
adds run ID and optional log path; and removes `engine`, `checkpoint`, and `pause_reasons`.
Its statuses are `succeeded`, `failed`, or `aborted`. Log creation, locking, submission, and
append failures are host/runtime exceptions rather than false flow failures. They are raised
only after structured drain.

The CLI always chooses a durable log path and reports it. Library callers opt in with
`log=`. Resuming a succeeded, failed, or aborted log is read-only and returns its existing
result.

## `ac chat` Compatibility

Today `ac chat` uses each `RunPaused` as its turn prompt, intercepts `/exit` outside the
flow, and restarts a fresh flow after a failed reply while preserving a host-mirrored
transcript. Removing pause requires an explicit redesign rather than deleting that loop.

**Recommended, pending user confirmation:** make exit flow-owned. The bundled and example
chat turn reads its blocking HUMAN_INPUT value, routes `/exit` and `/quit` to an `exited=true`
turn result before the reply agent, and routes other text through the reply agent and
transcript fold. EOF/keyboard exit is normalized by the CLI handler to the same exit value.
The LOOP stops normally, so no host exception or synthetic engine control is needed and an
exit command is never sent to the model.

For normal turns, `run_flow` stays active while HUMAN_INPUT prompts inside its worker. A
durably reduced reply `NodeSucceeded` observer prints the reply before the next prompt; the
terminal coordinator orders that output against progress and input.

Failed-turn resilience remains host-owned. The observer continues mirroring the last
successfully completed turn transcript. If the reply agent fails, `run_flow` returns failed;
the CLI prints the error, reloads the immutable flow, starts a new run/log with that transcript,
and prompts for the next turn. `/exit`, `/quit`, EOF behavior and the existing failed-turn
tests remain acceptance criteria.

## Package Cleanup

Delete rather than deprecate:

- `suspension/` and all checkpoint/restore/command code;
- runtime paused/deferred queues and checkpoint expansion ledgers;
- `nodes/agent/controls/` and the `controls:` compose field;
- `AgentEntry`, `Fresh`, `Resume`, the `entry` constructor argument, pending memo state,
  agent-specific expansion, and synthetic HUMAN_INPUT/resume-agent nodes;
- event/watch WAIT descriptors and builders; and
- CLI pause/resume prompting loops.

The intended runtime layout is:

```text
events.py          public observer notifications; StreamChunk is transient
runtime/
  engine.py        dispatcher, workers, drain lifecycle
  state.py         RunState, preparation, total reducer
  log.py           durable records, codec, JSONL lock/repair
  eval_node.py     bind and evaluate one node into a raw worker outcome
```

`state.py` replaces `state_manager.py`; ready and in-flight ownership are no longer split
between an engine queue and the state manager. Durable records are not forced into
`events.py`, preserving its leaf observer role. Package charters and the documented layer
ladder are updated with the implementation.

`AgentNode` has one authored constructor and one ordinary local model/tool loop. Its
conversation exists only inside the running node stack. A crash retries the entire agent
node, which is the accepted recovery boundary.

## Verification

The implementation is accepted only with tests for:

- replay after every authoritative record prefix, including every record in nested
  CALL/MAP/LOOP runs;
- equality of graph nodes/edges/wiring, pool, dispositions, ready order, in-flight order,
  growth depths, START overrides, and terminal result after replay;
- exact restoration of `system.today`, `system.now`, `system.run_id`, custom system values,
  effective LLM configuration, and explicit `commit_target`;
- serial execution and positive `num_workers`, including dispatcher commit order;
- dispatcher-owned start-before-submit and retry after a submit failure;
- route-derived skip flood without authoritative `NodeSkipped` records;
- one seed builder for live and replay growth, sorted prunes, collision/budget rejection
  before append, and flat loop-history replay;
- output/seed typed-codec round trips and no reducer exception after a successful append;
- partial-tail truncation, corrupt complete lines, sequence/schema/run/fingerprint mismatch,
  restrictive creation mode, and two-process writer exclusion;
- append failure proving the proposed transition was not reduced or observed;
- failure/abort/log-error with a sibling blocked on human input and on timed WAIT, proving
  no new dispatch/commit, no early terminal return, and late-outcome discard;
- process death during terminal drain versus log-error drain;
- `RunAborted` durability and `StreamChunk` transience;
- agent `tools: [ask_user]`, hidden injected metadata, allowed-tool enforcement, duplicate
  owner rejection, sequential CLI registration cleanup, and shared HUMAN_INPUT invocation;
- serialized CLI prompts coordinated with Rich progress;
- repeated runs of one `ComposedFlow` leaving its graph and LLM config untouched;
- `ac chat` `/exit`, `/quit`, EOF, multi-turn output, and failed-turn restart behavior; and
- deletion ratchets for `Pause`, checkpoints, `AgentEntry`, controls, synthetic resume
  nodes, event WAIT, and `suspension` imports.

The existing runtime kind-census remains at zero. The run log records effects at node
boundaries without making the engine aware of agent, human-input, wait, call, map, or loop
kinds.

## Deferred Work

This change does not also remove streaming generators, the always-failing MODEL node, the
unused failure hook, or the old prompt parser. It does not perform the parse-once/source
diagnostic cleanup under #81/#91/#103 or a general CLI split. Streaming remains explicitly
transient; the other dormant surfaces remain separate cleanup issues under #110.

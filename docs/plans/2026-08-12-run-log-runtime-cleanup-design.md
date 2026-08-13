# Run-Log Runtime Cleanup

**Date:** 2026-08-12
**Issues:** [#110](https://github.com/ngocbh/agent-composer/issues/110),
[#111](https://github.com/ngocbh/agent-composer/issues/111), and
[#34](https://github.com/ngocbh/agent-composer/issues/34)

## Goal

Replace checkpoint-based suspension and synthetic agent continuation graphs with a
small node-atomic runtime. Human interaction blocks the executing worker. An
append-only run log is the sole durable state and can reconstruct an unfinished DAG,
including its expanded topology, pool, node and edge dispositions, and ready queue.

The resulting code should make the normal execution path obvious: dispatch a node,
record its result, reduce that result into run state, and schedule its successors.
Parallel execution remains a first-class requirement through `num_workers`.

## Decisions

### Node execution is the recovery boundary

A node runs continuously on one worker until it returns or raises. If an agent invokes
`ask_user`, the tool blocks in that same call stack and returns the human answer as a
normal tool result. No worker is released while waiting.

This deliberately trades worker capacity for a much smaller model. If every worker is
waiting for a person or a timer, ready work waits until a worker becomes available. A
host that needs more concurrent waits can increase `num_workers`.

The engine cannot serialize a Python stack, LLM request, or tool invocation. If the
process exits after `NodeStarted` but before a terminal node record, recovery requeues
the whole node from its beginning. LLM, tool, code, and human-input effects may repeat.
Recovery is therefore at-least-once for in-flight nodes, not bit-for-bit continuation.

### Human input is a normal tool

`ask_user` moves into the existing process-global tool registry. Authors grant it like
any other tool:

```yaml
tools: [ask_user]
```

The CLI registers a blocking implementation that serializes terminal access. Library
hosts register their own implementation and own its thread safety. A host cannot select
different `ask_user` implementations for concurrent runs in the same process without a
future run-scoped registry; this is the same process-global constraint as other tools.

The agent loop invokes `ask_user` through the ordinary tool path and appends its result
as a `ToolMessage`. `HUMAN_INPUT` remains the deterministic, structured authoring form
and invokes the same registered tool with its prompt, questions, answer schema, and
title. Tool errors propagate as node failures; the agent path no longer converts tool
exceptions into model-visible fallback strings.

### One log, no checkpoint

The durable representation is an append-only JSONL log. There is no parallel snapshot,
checkpoint, live-engine resume handle, pause reason, or resume command. Runtime records
are also the observer event stream, avoiding separate transient and durable vocabularies.

The initial record identifies the schema version, run ID, flow fingerprint, typed
inputs, and worker count. Subsequent records form a closed transition vocabulary:

```text
RunStarted
RunResumed
NodeStarted
NodeSucceeded
NodeRouted
NodeExpanded
NodeSkipped
NodeFailed
RunSucceeded
RunFailed
```

Each record has a contiguous sequence number. A record is appended and flushed before
the reducer applies its transition to memory. Failure to append stops execution; the
runtime must not continue after durable and in-memory state could diverge. Node outputs
are converted to losslessly serializable typed values before `NodeSucceeded` is written.

The runtime derives queue mutations from the ordered transitions rather than storing
redundant queue snapshots. Parallel completion order is the single dispatcher's commit
order and is therefore preserved by the log.

## Runtime Architecture

Every run starts from a private deep copy of the compiled flow. `ComposedFlow` remains
immutable and reusable; runtime graph expansion cannot modify the loaded artifact. This
also fixes #34.

One `RunState` owns all mutable state:

- the private expanded graph and variable pool;
- node and edge dispositions;
- the ready FIFO and ordered in-flight nodes; and
- terminal run status.

The dispatcher is the only writer. Workers execute nodes and return outcomes; they never
mutate `RunState` or append to the log. The dispatcher appends a transition and then asks
the state reducer to apply it. `num_workers=0` retains serial execution, while positive
values retain the worker pool.

`NodeResult` becomes:

```text
Output | Route | Grow
```

For dynamic `call`, `map`, and `loop` topology, `NodeExpanded` stores the spawner ID,
serializable builder seed, and pruning information. Replay asks the spawner's existing
node-owned hook to rebuild that graph fragment. Expansion records are flat and
chronological; the nested `GrowRecord` tree and origin-keyed checkpoint ledger disappear.
Agent human input no longer grows the graph, so agent-specific seeds and replay hooks
disappear completely.

## Run And Resume API

The public boundary becomes:

```python
run_flow(flow, inputs, *, log=path, num_workers=...)
resume_flow(flow, *, log=path, num_workers=...)
```

The scheduler receives only an injected `append(record)` callable. `run.py` owns the
standard JSONL implementation and filesystem policy. The CLI always supplies a durable
log path. A library call without `log=` uses an in-memory record list and cannot resume
across processes.

`resume_flow` accepts no live engine, checkpoint, command, or human answer. It validates
the schema, contiguous sequence numbers, run identity, and flow fingerprint, folds the
log into a fresh `RunState`, records `RunResumed`, and continues appending to the same
log. A freshly loaded flow must match; the compiled flow itself is not embedded.

After replay, ready nodes retain their logged FIFO order. Nodes that started but did not
finish are requeued first in their original dispatch order. A succeeded or failed log is
already terminal, so `resume_flow` returns its recorded result without executing work.
Ordinary failure remains terminal; retries and failure branches stay outside this change.

`RunResult` retains terminal output and diagnostic information plus run/log identity. It
drops `engine`, `checkpoint`, and `pause_reasons`.

## Package Cleanup

Delete the following mechanisms rather than leaving compatibility wrappers:

- `suspension/`;
- `nodes/agent/controls/` and the `controls:` authoring field;
- `AgentEntry`, `Fresh`, `Resume`, and the `entry` constructor argument;
- segmented `agent_step` continuation state;
- synthetic human-input/resume-agent subgraphs and expansion helpers;
- runtime pause/deferred state and pause event branches;
- checkpoint snapshot, restore, commands, and expansion ledgers; and
- CLI pause/resume interaction loops.

The intended runtime layout is:

```text
runtime/
  engine.py       worker dispatch and result handling
  state.py        graph, pool, dispositions, queues, and reducer
  log.py          JSONL append and replay
  eval_node.py    execution of one node
```

`state.py` replaces `state_manager.py` and brings the ready queue into the single state
owner. `events.py` becomes the serializable transition vocabulary consumed by the
runtime, log, CLI observers, and callers. The tool registry exposes one shared
`invoke_tool` path used by both `ToolNode` and the agent loop.

## Recovery And Corruption

Replay may ignore only an unterminated final JSONL line. A malformed committed record,
sequence gap, incompatible schema, run-ID mismatch, or flow-fingerprint mismatch aborts
recovery rather than guessing.

If the process dies after a terminal transition is flushed but before it is applied in
memory, replay applies it. If it dies after `NodeStarted`, `RunResumed` moves that node
back to ready. Repeated `NodeStarted` records make each retry visible.

There is no implicit retry after `NodeFailed`. Starting again after a terminal success or
failure requires a new run ID and log.

## Alternatives Rejected

**Keep checkpoints and simplify their schema.** This retains two durable state models,
hard version coupling, and topology reconstruction machinery. It does not meet the
deletion goal.

**Suspend a generator and release its worker.** This preserves worker capacity during
human waits, but a Python generator stack cannot be serialized. Supporting both live
generators and durable recovery would require a second explicit continuation protocol.

**Return an effect plus a serialized continuation.** This is recoverable and avoids held
workers, but recreates `Pause`, node resume hooks, request/answer records, and node-owned
continuation schemas. The accepted at-least-once node retry makes that machinery
unnecessary.

**Use global mutable callback state for `ask_user`.** This is unsafe under concurrent
runs. Host registration through the existing tool registry is explicit and follows the
current tool ownership model.

## Verification

Tests will cover:

- uninterrupted execution versus recovery from log prefixes;
- serial and pooled execution, including dispatcher commit ordering;
- exact ready-queue reconstruction and in-flight retry order;
- flat replay of nested `call`, `map`, and `loop` growth;
- duplicate execution of a crash-interrupted node;
- log append failure, final-line truncation, corruption, and identity mismatches;
- typed output round trips and terminal resume idempotence;
- a blocking registry-backed `ask_user` inside an agent;
- structured `HUMAN_INPUT` through the same tool;
- one held worker while independent work completes, and full-pool saturation;
- repeated execution of the same `ComposedFlow` without graph mutation; and
- deletion ratchets for suspension, checkpoint, agent-entry, and synthetic-resume code.

The existing kind-census invariant remains at zero runtime `NodeKind` branches. Nodes
still return outcomes without touching the pool, graph structure remains author-fixed,
and only the dispatcher mutates run state.

## Documentation And Scope

The same change updates the package README, engine and composing-agent skills, package
charters, API/architecture/engine/syntax/typing docs, examples, and templates. Statements
that durable checkpoint suspension is non-negotiable are replaced by the node-atomic log
contract and its at-least-once boundary.

This design does not also remove the dormant streaming path, always-failing `MODEL` node,
unreachable event/watch waits, unused failure hook, or old prompt parser. It also defers
the parse-once/source-diagnostic work under #81/#91/#103 and the broader CLI split. Those
are separate cleanup changes under #110.

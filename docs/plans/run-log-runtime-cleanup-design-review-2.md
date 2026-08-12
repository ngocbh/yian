VERDICT: NEEDS_REVISION

## Summary Assessment

Version 2 resolves most round-1 defects: durable records are reducer transactions, the
dispatcher owns dispatch order, `RunStarted` captures the initial run seed, JSONL ownership
is explicit, growth state is enumerated, and WAIT/stream/abort are no longer silently omitted.
It is not ready for implementation planning yet. The remaining issues can change run results,
make live and replayed executions disagree, or hang the CLI while it owns the log lock.

## Critical Issues (must fix)

1. **An END commit is not a terminal condition; graph quiescence is.** The design says a
   successful END commit, or an END made unreachable, immediately makes the run terminal and
   causes all later worker outcomes to be discarded. A flow may have independent roots that do
   not feed END: boundary synthesis connects START to every root
   (`compose/build/boundary.py:136-146`), while END depends only on declared output producers.
   Under `num_workers > 1`, END can commit while an independent sibling is still ready or
   running and can later fail. The current engine deliberately waits for both ready and
   executing to be empty (`runtime/state_manager.py:206-215`) before deriving its terminal
   event (`runtime/engine.py:221-264,295-307`). Preserve that rule. Node failure and external
   abort may start structured drain immediately; success or `TerminalSkipped` must be derived
   only when the whole DAG is quiescent. The causal record that empties ready/in-flight can
   produce the run-level terminal notification.

2. **The existing typed codec is not lossless for arbitrary nested values.** `ObjectValue` and
   `ListAnyValue` retain `Any` interiors (`typesys/values.py:164-181`). For example, a live pool
   can contain `{"pair": (1, 2)}`, but JSON round-trip restores the tuple as a list. The proposed
   live apply plan currently uses the pre-encoding Python value/builder seed while replay uses
   the decoded value, so downstream code or `build_grow` can observe different inputs. During
   preparation, encode and decode outputs and seeds, then use that canonical decoded value for
   both the live apply/build and the record; alternatively provide a recursively lossless codec.
   Add a nested tuple/date/file-shaped round-trip regression rather than asserting that the
   current `TypedValue` codec is already lossless.

3. **Replay cannot detect a changed but valid growth builder.** `NodeExpanded` stores only the
   spawner, seed, and prunes. CALL/MAP/LOOP topology is reconstructed by implementation code
   (`compile/expand.py:200-444`), but the flow fingerprint covers authored build fields, not that
   builder implementation. A newer `build_grow` can produce a different valid graph and replay
   will silently accept it; the promised "builder mismatch" has nothing to compare. Store a
   canonical digest of the prepared expansion topology and compare it on replay, or include a
   strict runtime/builder compatibility identity. Also assign the build-manifest and growth
   carrier to a concrete layer: the proposed node-owned hook otherwise preserves the existing
   `compile.model -> nodes -> compile.expand` cycle (`compile/model.py:18`,
   `nodes/call/node.py:20`, `nodes/map/node.py:25`, `nodes/loop/node.py:23`) while the project
   contract requires an acyclic ladder.

4. **`InjectedToolArg` hides metadata from the advertised schema but does not protect it.** With
   the installed `langchain-core 1.4.8`, the proposed tool's `tool_call_schema` exposes only
   `question`, but its invocation schema still accepts and forwards `questions`,
   `answer_schema`, `title`, and `node_id`. The agent currently passes model-returned arguments
   directly to tool invocation (`nodes/agent/modes/tool_calling.py:58-69,123-128`). A model can
   therefore spoof trusted HUMAN_INPUT metadata despite the design's claim that it supplies only
   `question`. Split the paths: validate/filter model arguments against the public tool-call
   schema before invocation, while `HumanInputNode` uses a trusted internal call that may inject
   metadata. Test that model-supplied injected fields are rejected or stripped.

5. **Structured drain is not enforceable for all exits that can occur while the log is locked.**
   In a positive-worker run, terminal input executes in a worker, but Python delivers SIGINT and
   `KeyboardInterrupt` to the main dispatcher. The worker may remain blocked in prompt-toolkit or
   `console.input`; the dispatcher then cannot both honor Ctrl-C and wait for the worker as the
   design requires. This contradicts the promised `ac chat` keyboard-exit behavior and can hold
   the JSONL lock indefinitely. A main-thread terminal broker (the node worker may block on its
   result) or another cooperatively wakeable input mechanism is required. The same lifecycle is
   unspecified when `on_event` or `on_stream` raises: define catch, stop-dispatch, drain, lock
   release, and re-raise behavior. Streaming also needs an explicit transient worker-to-dispatcher
   channel; direct worker callbacks violate the terminal coordinator, while buffering until node
   return is not streaming.

6. **Two public behavior changes remain undecided.** Blocking WAIT needs an exact clock contract
   for date-only values, naive versus offset timestamps, past deadlines, and interruption; the
   current node only validates an ISO value and returns `ScheduledPause`
   (`nodes/wait/node.py:27-44,102-108`). Separately, the design labels flow-owned `ac chat` exit as
   "pending user confirmation." It also does not preserve `/exit` for a user-supplied chat flow
   that lacks the proposed route, whereas the current CLI intercepts it host-side
   (`cli/chat/command.py:115-149`). Resolve both contracts with the user before turning this design
   into an implementation plan.

## Suggestions (nice to have)

- State whether historical records are replayed silently. `on_event` should normally observe
  `RunResumed` and new records only; replaying old node callbacks would duplicate CLI output and
  host side effects.
- Bound submitted work to available worker slots, or define queued futures as in-flight and cancel
  those that have not begun during drain. Otherwise a failure can allow a large executor backlog
  to start new external effects after the terminal cause.
- Per-record `fsync` gives the strongest prefix guarantee but can dominate short parallel nodes.
  Measure it and document that durability/performance choice; do not silently weaken the prefix
  contract during implementation.

## Verified Claims

- Round-1 issue 1 is resolved: `NodeSkipped`, `RunSucceeded`, and `RunFailed` are derived views, so
  each authoritative record is intended to be one reducer transaction. The END timing defect
  above is separate from prefix atomicity.
- Round-1 issues 2 and 3 are structurally resolved: v2 specifies prepare/append/total-apply and
  dispatcher-owned ready-to-in-flight transition before submission. Canonicalizing through the
  wire codec is the remaining prepare gap.
- Round-1 issue 4 now has a clear chosen policy, structured draining, but worker-thread Ctrl-C and
  callback exceptions still prevent that policy from covering every actual exit path.
- Round-1 issue 5 is resolved in principle: the complete initial pool, START override, effective
  LLM map, worker count, and recursive build fingerprint are recorded, and the graph copy precedes
  LLM resolution.
- Round-1 issue 6 is resolved: event/watch WAIT is deleted; `RunAborted` is durable; streaming is
  explicitly transient. The blocking WAIT clock contract and streaming transport remain open.
- Round-1 issue 7 is partly resolved: the proposed `StructuredTool` shape does hide injected
  fields from the model-facing schema, and the startup-only registry owner is coherent. Direct
  invocation still accepts those fields, so the trust boundary needs explicit filtering.
- Round-1 issues 8 and 9 are resolved: fresh/create and resume locking, physical partial-tail
  repair, sequence validation, growth depth, START overrides, prune cleanup, and replay-derived
  indexes are all specified.
- Flat chronological growth is feasible for current CALL/MAP/LOOP builders, and the executing
  loop clone exists when its expansion record is replayed before that same transaction prunes it.

The verdict remains **NEEDS_REVISION** until the six critical items above are resolved.

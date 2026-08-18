# Plan: Node-Atomic Runtime With Append-Only Run Logs

**Goal**: Replace checkpoint suspension and synthetic agent continuations with a small, node-atomic, parallel runtime whose append-only log reconstructs an unfinished DAG exactly at committed node boundaries.

**Architecture**: A single-writer dispatcher appends and `fsync`s typed records before a total reducer mutates private per-run state. Workers run nodes atomically; human input and WAIT block a worker. Recovery folds the log, verifies the authored-flow fingerprint and every dynamic-expansion digest, then retries unfinished `NodeStarted` records at least once.

**Tech Stack**: Python 3.11, dataclasses/Pydantic v2, LangChain tools, JSONL, POSIX `fcntl.flock`, Typer/Rich/questionary, pytest, MkDocs.

**Design**: [`docs/plans/2026-08-12-run-log-runtime-cleanup-design-final.md`](2026-08-12-run-log-runtime-cleanup-design-final.md)

**Tracking**: umbrella #110; parent #111; implementation issues #34, #116, #117, #118, and #119.

**Design branch**: `dev/engine/run-log-cleanup`. Implementation uses one branch and PR per child issue, in the dependency order below. Each branch starts from its merged dependency, not from the long-lived design branch.

## Summary

The current engine mutates a supposedly reusable `ComposedFlow`, represents human interaction as pause/checkpoint state, and resumes agents through generated graph fragments. This series first isolates each run, then makes growth seed-first and fingerprintable, introduces an authoritative append-only run log and reducer, changes human interaction and WAIT to blocking node execution, and finally moves CLI terminal ownership to a main-thread broker.

Parallel execution remains first-class through `num_workers`. A process crash retries every started node without a durable outcome, so model, tool, code, and human effects inside that node are explicitly at-least-once.

## Problem Verification

The issue remains present in the current branch:

- `src/agent_composer/compose/build/pipeline.py:58` calls `ComposedFlow` immutable and reusable, but `src/agent_composer/run.py:209` resolves the LLM cascade in place and `src/agent_composer/run.py:223` passes the shared `loaded.compiled` directly to `FlowEngine`.
- `src/agent_composer/compile/model.py:178` exposes mutating `add_subgraph`, and `src/agent_composer/runtime/engine.py:672` applies growth to `self.flow`. Reusing one loaded flow therefore retains runtime-expanded topology.
- `src/agent_composer/nodes/call/node.py:20`, `src/agent_composer/nodes/map/node.py:25`, and the LOOP/AGENT implementations import `compile.expand`, creating the documented `compile -> nodes -> compile` back-edge.
- `src/agent_composer/nodes/agent/node.py:53` defines `Fresh`, `Resume`, and `AgentEntry`; `src/agent_composer/nodes/agent/node.py:131` exposes the `entry` constructor requested for removal.
- `src/agent_composer/nodes/human_input/node.py:81` returns `Pause`, and `src/agent_composer/nodes/wait/node.py:102` returns scheduled/event pauses instead of doing the wait.
- `src/agent_composer/runtime/engine.py:327` snapshots pool, queues, state maps, pause state, and a nested growth ledger; `src/agent_composer/runtime/engine.py:359` restores by replaying that second state representation.
- `src/agent_composer/events.py:15` explicitly says events are transient and never serialized, so the current event stream cannot recover queue order or a complete run.
- In pooled mode workers currently claim IDs from `ready_q` themselves at `src/agent_composer/runtime/engine.py:561`. That prevents a durable dispatcher-owned `NodeStarted` transition from preceding external work.
- `src/agent_composer/run.py:251` returns a live engine, checkpoint, and pause reasons, while `src/agent_composer/run.py:273` accepts live-engine/checkpoint/answer commands. This conflicts with the accepted meaning of `resume_flow`: replay and continue an append-only run log only.
- `src/agent_composer/nodes/agent/modes/tool_calling.py:123` reorders ordinary tools before controls and turns tool errors into model-visible strings. `src/agent_composer/tools/registry.py:23` has lookup/registration but no shared strict invocation boundary.
- `src/agent_composer/cli/run.py:746` owns a pause/resume prompt loop, and `src/agent_composer/cli/chat/command.py:105` intercepts `/exit` and `/quit` in the host instead of demonstrating a flow route.

Reproduction for #34 before its fix: load one flow containing nested CALL/MAP/LOOP growth, capture its recursive node/edge/wiring manifest, call `run_flow` twice on the same object, and compare the manifest and results. The first run mutates the compiled topology and effective agent configuration; the second starts from that dirty graph.

## Options Considered

1. **Keep checkpoints and simplify only agent continuation.** Lowest immediate migration cost, but retains two durable state models, pause/deferred scheduling, snapshot versioning, and synthetic continuation graphs. Rejected because it does not reach the requested small core.
2. **Hold live Python generators across human input.** Avoids explicit agent resume objects while the process lives, but Python stacks cannot be serialized. Exact unfinished-DAG recovery would still require a second continuation representation. Rejected.
3. **Block workers and restore only from periodic snapshots.** Simple live execution, but a snapshot can lag committed work and reintroduces snapshot/log authority conflicts. Rejected.
4. **Append transitions plus optional authoritative checkpoints.** Faster replay, but creates two sources of truth in this cleanup. Rejected; snapshots may be added later only as disposable caches.
5. **Chosen: block workers and use one append-only transition log.** This spends worker capacity while waiting, which the user explicitly accepted, and makes a whole node the only recovery boundary. It preserves `num_workers`, keeps the runtime kind-blind, and removes continuation state.

## Chosen Approach

### Public APIs

Final library boundary:

```python
def run_flow(
    flow: ComposedFlow,
    inputs: dict[str, Any],
    *,
    log: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    num_workers: int = 0,
    on_event: Callable[[Any], None] | None = None,
    on_stream: Callable[[StreamChunk], None] | None = None,
    llm_config: dict[str, Any] | None = None,
) -> RunResult: ...

def resume_flow(
    flow: ComposedFlow,
    *,
    log: str | os.PathLike[str],
    num_workers: int | None = None,
    on_event: Callable[[Any], None] | None = None,
    on_stream: Callable[[StreamChunk], None] | None = None,
) -> RunResult: ...
```

`resume_flow` accepts no engine, checkpoint, commands, answer, fresh inputs, provider/model overrides, or new run ID. An omitted `num_workers` reuses the latest segment's value; an explicit value changes future scheduling only.

`RunResult` keeps canonical original `input`, status/output/failure diagnostics/traceback, and current-segment observer `events`; adds `run_id` and optional `log`; removes `engine`, `checkpoint`, and `pause_reasons`. Replaying a terminal log is silent and returns `events=[]`. A nonterminal recovery starts its segment with the newly appended `RunResumed`; historical callbacks and events are never replayed.

`FlowEngine` stays public for direct in-memory execution and retains its event iterator. Durable path policy and full run identity remain `run_flow`/`resume_flow` responsibilities. Dynamic direct-engine users must inject `grow_builder`; `snapshot`, `restore`, command resume, and abort disappear in #118.

Final CLI surface:

```text
ac run FLOW.yaml                         # ~/.agent_composer/runs/<run_id>.jsonl
ac run FLOW.yaml --log PATH              # fresh run at PATH
ac run FLOW.yaml --resume PATH           # continue PATH
ac run FLOW.yaml --resume PATH -w N      # override future worker count
```

`--resume` rejects `--input`, `--inputs`, `--provider`, `--model`, and `--log`. It skips input prompts and provider/model re-resolution. The CLI creates the default runs directory privately and reports the selected log path.

### Durable Data Shapes

Every UTF-8 JSONL line is one envelope:

```python
@dataclass(frozen=True)
class RecordEnvelope:
    schema: int
    run_id: str
    seq: int
    kind: str
    payload: dict[str, Any]
```

The closed records are `RunStartedRecord`, `RunResumedRecord`, `NodeStartedRecord`, `NodeSucceededRecord`, `NodeRoutedRecord`, `NodeExpandedRecord`, and `NodeFailedRecord`. The wire `kind` values are stable snake-case strings. There are no authoritative skip or run-terminal records.

Key payloads:

```text
RunStarted:
  flow_fingerprint, canonical full initial VariablePool, start wiring override,
  effective_llm_by_static_path, initial num_workers
RunResumed:
  ordered retry node IDs, selected num_workers
NodeStarted:
  exact ready-head node ID
NodeSucceeded:
  executing node ID, resolved commit_target, canonical TypedValue wire object
NodeRouted:
  executing node ID, selected handle
NodeExpanded:
  actual executing spawner ID, canonical seed wire object,
  sorted prune IDs, SHA-256 topology digest
NodeFailed:
  node ID, message/type, symbolic locator, concrete range, traceback
```

`RunStarted` contains the complete canonical pool, including `system.today`, `system.now`, `system.run_id`, host ambients, primed START value, and `VariablePool.start_id`. It also retains canonical original inputs separately in `RunState` before START overwrites its pool entry.

Use `ANY_VALUE_ADAPTER` and `VariablePool.model_dump(mode="json")`/validation for one encode-then-decode boundary. Preparation always uses the decoded value for live pool writes and live expansion building. This intentionally promises live/replay equivalence, not preservation of arbitrary Python types nested inside `Any` (for example a nested tuple canonicalizes to a list).

### Run State And Reducer

`runtime/state.py` defines one private `RunState` containing the private expanded graph, canonical pool, node/edge dispositions, ready `deque`, ordered in-flight IDs, terminal result, growth/reference depth, START overrides, effective LLM map, and derived indexes. It exposes preparation functions that validate without mutation and a total `apply(prepared)` reducer.

The dispatcher pipeline is always:

```text
raw worker result -> prepare/validate/canonicalize -> append+flush+fsync
-> total reducer apply -> observer callbacks
```

The dispatcher logs `NodeStarted` before inline execution or worker submission. It submits at most `num_workers` evaluations and owns the ready FIFO; workers never dequeue ready work. Parallel completion order is whichever completion the dispatcher accepts first, fixed durably by sequence numbers.

Success and `TerminalSkipped` are derived only when both ready and in-flight are empty. END may commit while a sibling remains ready/running; a later sibling failure still wins. `NodeFailed` is immediately terminal and begins structured drain. Callback/log failures are host exceptions, not flow failures.

### Flow Identity And Growth

`compile/manifest.py` creates canonical, representation-neutral manifests. Maps are key-sorted; semantically ordered edges, outputs, params, and similar lists retain order. Fingerprints include node execution fields, types, wiring, asserts, env, authored LLM layers, and recursive static children, while excluding source ranges, object identity, compiled caches/code objects, credentials, and run-resolved values. Unknown noncanonical node fields fail loudly.

Growth becomes:

```python
@dataclass(frozen=True)
class Grow:
    seed: Any
    prune: frozenset[str] = frozenset()

def build_grow(spawner: Node, canonical_seed: Any) -> Flow: ...
```

CALL/MAP/LOOP return only seed/prune and import no compile code. `compile.expand.build_grow` owns the legal closed-kind match and returns the fragment. The run boundary injects it into runtime. On replay the same builder must produce the recorded topology digest before state changes. Flat chronological records store the actual executing LOOP clone (`L~k`), not only its origin.

### Blocking Tools, WAIT, And Terminal Ownership

`ask_user` is a reserved normal registry tool. Its model-visible schema contains only `question`; `questions`, `answer_schema`, `title`, and `node_id` are `InjectedToolArg` metadata accepted only by `invoke_tool_trusted`. Public invocation explicitly rejects unknown keys before Pydantic validation. One process-global owner may install `ask_user` at a time, as accepted.

An AGENT lists `tools: [ask_user]` and runs one ordinary local tool loop. HUMAN_INPUT uses the same registered implementation through the trusted metadata path. Tool calls execute in model order, must be granted by the node, and exceptions fail the node.

WAIT accepts only an absolute date/datetime: date is midnight UTC, naive datetime is UTC, aware datetime is normalized to UTC, and past deadlines return immediately. It waits with `cancel_event.wait(remaining)` so failure/interruption drain can wake it. Recovery recomputes remaining wall time.

The CLI runs `run_flow` on a controller thread even for `num_workers=0`. A main-thread `TerminalBroker` owns one FIFO for prompts and render work, so Rich/questionary/input and SIGINT remain main-thread-only. Ctrl-C sets only an internal stop signal, wakes broker/WAIT work, drains cooperative workers, releases the log lock, and leaves a nonterminal resumable log. `/exit` and `/quit` are ordinary chat flow values; only chat EOF is normalized to `/exit`.

## Dependency And Merge Plan

| Order | Issue / branch | Depends on | Deliverable | Temporary compatibility boundary | Merge gate |
| --- | --- | --- | --- | --- | --- |
| 1 | #34 `dev/engine/per-run-graph-isolation` | `main` | Deep-copy before cascade/growth | Existing Pause/checkpoint APIs unchanged; engine still mutates only its private copy | Full tests green; repeated nested run leaves source graph unchanged |
| 2 | #116 `dev/engine/seed-first-growth` | merged #34 | Seed-first Grow, compile builder, IDs, fingerprints, effective LLM map | Keep Pause, GrowRecord, snapshot/restore, and a compile-owned legacy AGENT continuation builder until #118 | Full tests green; no `nodes` or `runtime` import of `compile.expand`; kind census 0 |
| 3 | #117 `dev/engine/append-only-run-log` | merged #116 | Records, logs, reducer, serial/parallel dispatcher, recovery | Keep old Pause/checkpoint APIs as an adapter over new state. A paused node remains an unfinished durable start; legacy snapshot/answer delivery is maintained only for old callers/tests. The old expansion ledger is mirrored solely for checkpoint compatibility. | Full tests, including all old pause/checkpoint tests, green; new prefix-replay matrix green |
| 4 | #118 `dev/engine/blocking-interaction` | merged #117 | Normal `ask_user`, blocking HUMAN_INPUT/WAIT, delete suspension | Before deletion, add and pass replacement blocking tests. A small locked CLI handler temporarily preserves `ac run`/`ac chat`; text exit may remain host-owned only in this PR. #119 replaces it with the final broker/flow route. | Full tests green after old pause/checkpoint tests are removed or rewritten; deletion ratchets green |
| 5 | #119 `dev/cli/run-log-terminal-broker` | merged #118 | Main-thread broker, CLI log flags, flow-owned chat, final docs | Removes #118's temporary locked terminal handler | Full tests + strict docs build green; default and explicit log/resume CLI E2E green |

# Changes By Issue

## #34: Per-Run Graph Isolation

**Files**: `src/agent_composer/run.py`; new `tests/engine/test_run_reuse.py`; focused updates to `tests/engine/test_llm_cascade_run.py` and `tests/engine/test_run_resume.py`.

The authored `loaded.compiled` is never passed to a mutating operation. Both fresh and checkpoint-compatible resume copy before `resolve_llm_cascade`; a live-engine resume continues its already-private engine.

## #116: Seed-First Growth, Layers, And Identity

**Files**: `src/agent_composer/nodes/base.py`, new `src/agent_composer/nodes/ids.py`, CALL/MAP/LOOP/AGENT node modules, `src/agent_composer/compile/expand.py`, new `src/agent_composer/compile/manifest.py`, `src/agent_composer/compile/llm_cascade.py`, `src/agent_composer/runtime/engine.py`, package charters; tests `test_grow_outcome.py`, `test_node_result.py`, `test_expand_ids.py`, new `test_grow_builder.py`, new `test_flow_identity.py`, `test_llm_cascade.py`, new `test_layer_direction.py`.

LOOP growth carries inherited reference depth with delta `0` rather than dropping it; it does not increment depth. This closes the current nested-call-in-loop hole and is locked by a regression test.

## #117: Append-Only Log, Reducer, And Recovery

**Files**: new `src/agent_composer/runtime/log.py`, new `src/agent_composer/runtime/state.py`, `runtime/engine.py`, `runtime/eval_node.py`, `events.py`, `run.py`, `runtime/__init__.py`; new tests `test_run_records.py`, `test_run_log.py`, `test_run_state.py`, `test_runtime_dispatch.py`, `test_run_recovery.py`, `test_run_lifecycle.py`, and `test_run_log_growth.py`.

`state_manager.py` remains present during the compatibility PR if old checkpoint helpers still import it, but the authoritative drive uses `RunState`. It is deleted only with the old machinery in #118.

## #118: Blocking Interaction And Suspension Deletion

**Files**: `tools/registry.py`, `tools/__init__.py`, TOOL node, AGENT node/modes/compose descriptor, HUMAN_INPUT, WAIT, CODE watchdog placement, `runtime/{engine,eval_node,state}.py`, `run.py`, `events.py`, CLI compatibility handlers; delete `nodes/agent/controls/` and `suspension/`; delete continuation helpers from `compile/expand.py`; rewrite/delete pause-specific tests only after replacements pass.

## #119: CLI Broker, Log Surface, Chat, And Docs

**Files**: new `src/agent_composer/cli/terminal.py`, `cli/run.py`, `cli/chat/command.py`, three chat YAMLs, examples/templates, root/engine/CLI docs, package charters, engine/composing/structure skills; new `tests/cli/test_terminal_broker.py` and expanded CLI/chat tests.

## Invariants Respected

- **Kind-blind runtime**: runtime switches only on raw outcome/record variants and node-owned traits. `tests/engine/test_kind_census.py` remains at zero.
- **Acyclic layer direction**: `nodes` imports only lower/peer layers; `compile` builds expansion fragments; root `run.py` injects the builder into runtime. An AST ratchet forbids `nodes -> compile` and `runtime -> compile.expand` imports.
- **Node purity**: nodes receive bound records/capabilities and return `Output | Route | Grow`; they never receive or mutate the pool.
- **Structural determinism**: the authored graph remains fixed. Dynamic CALL/MAP/LOOP expansion is derived from canonical seeds and verified against a topology digest.
- **Single writer**: only the dispatcher appends records and reduces graph/pool/queue state. Workers return raw outcomes/chunks.
- **Durability order**: every authoritative transition is prepared, appended, flushed, and `fsync`ed before apply or observation.
- **No dual durable state in the final system**: #117's checkpoint adapter is explicitly temporary and removed in #118.
- **Parallelism**: `num_workers=0` remains inline execution on the controller; positive values remain bounded parallel execution with durable dispatcher commit order.

## Test Plan

- Repeat one `ComposedFlow` through nested CALL/MAP/LOOP and assert recursive topology/config identity before/after each run.
- Fingerprint authored semantic changes and ignore ranges/comments/object identity; preserve ordered edge/output/param lists.
- Fold every authoritative prefix of nested serial and parallel logs; compare graph, pool, dispositions, ready/in-flight order, depth, START override, and terminal result.
- Verify output and growth-seed encode/decode equivalence for nested tuple/date/file-shaped values.
- Verify start-before-submit, bounded submissions, submit failure retry, parallel dispatcher commit order, and no post-failure external starts.
- Verify END-before-sibling quiescence, later sibling failure wins, route-derived skips, and `TerminalSkipped` only at full quiescence.
- Verify expansion digest replay, changed-builder rejection, actual LOOP clone IDs, pruning, depth budgets, and import/kind ratchets.
- Verify log mode `0600`, `O_EXCL`, exclusive cross-process lock, contiguous sequences, tail repair/truncation+fsync, corrupt-line rejection, flow/run/schema mismatch, and append failure before mutation/observation.
- Verify historical callbacks are silent, `RunResult.events` is current-segment-only, terminal resume is read-only, and original canonical inputs survive resume.
- Verify stream chunks are transient/on_stream-only and late chunks are discarded; callback failure and Ctrl-C stop dispatch, wake cooperative blockers, drain, unlock, and re-raise.
- Verify strict public/trusted tool invocation, `ask_user` owner cleanup, blocking AGENT/HUMAN_INPUT, UTC WAIT, all-workers-blocked behavior, and retry of an interrupted node.
- Verify broker FIFO/main-thread behavior, CLI option exclusion, default/explicit logs, resume worker override, flow-owned chat exits/EOF, multi-turn ordering, and failed-turn transcript restart.
- Run templates/examples, full tests, active-source deletion grep, and `mkdocs build --strict`.

## Risks And Out Of Scope

- In-flight effects are at-least-once. The log cannot restore Python stacks, active HTTP calls, or tool side effects.
- Logs contain sensitive inputs, prompts, human answers, outputs, diagnostics, and tracebacks. Default file mode is `0600`; applications remain responsible for retention and encryption.
- Per-record `fsync` may dominate short-node workloads. Measure it and publish the result in #117/PR notes, but do not batch or weaken durability in this series.
- `fcntl.flock` is POSIX-specific. Cross-platform locking is out of scope.
- Flow fingerprints do not attest to imported Python function bodies, tool implementations, remote models, credentials, or services. Expansion digests detect changed dynamic builder topology only.
- A non-cooperative external call may prevent graceful in-process drain. Process termination leaves the last complete prefix recoverable.
- Process-global `ask_user` ownership prevents different handlers in concurrent runs in one process; this limitation was accepted.
- Streaming generators remain supported but are not a durable boundary. Removing dormant streaming, MODEL, event/watch WAIT beyond its required deletion, unused failure hooks, the old prompt parser, parse-once/source diagnostics, and broad CLI splitting are separate work under #110/#81/#91/#103 or new child issues.

# Detailed Execution Plan

All commands run from the repository root. Use this exact prefix:

```bash
PYTHONPATH=src conda run -n agentcomposer python -m pytest
```

Before each PR, post that PR's section to its issue, claim it, create the named branch from the merged dependency, and record the baseline:

```bash
git status --short
PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine tests/cli -q
```

## PR 1 / #34: Isolate Every Run

### Slice 1.1: Reproduce graph and cascade mutation

1. **Red test (2-5 min)**: add `tests/engine/test_run_reuse.py::test_repeated_nested_runs_do_not_mutate_composed_flow`. Load a flow with a nested CALL containing MAP or LOOP; recursively snapshot node IDs, ordered edge tuples, wiring, static children, and agent `own_llm_config`/`llm_config`; run twice; assert snapshots remain equal and outputs match.
2. **Red test (2-5 min)**: add `test_run_reuse.py::test_different_llm_layers_do_not_leak_between_runs`, using a fake model factory and two `llm_config` layers on the same loaded flow.
3. **Verify red**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_reuse.py -q
   ```

   Expected failure: runtime-grown IDs/edges or the first resolved LLM config remain on `loaded.compiled`.
4. **Implement (2-5 min)**: import `copy` in `src/agent_composer/run.py`; in `run_flow`, set `compiled = copy.deepcopy(loaded.compiled)` before `resolve_llm_cascade`, and pass `compiled` to `FlowEngine`.
5. **Implement (2-5 min)**: in checkpoint-compatible `resume_flow`, deep-copy before cascade/restore. Do not copy the live `engine=` path.
6. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_reuse.py tests/engine/test_llm_cascade_run.py tests/engine/test_run_resume.py tests/engine/test_graph_growth_e2e.py -q
   ```

7. **Commit**:

   ```bash
   git add src/agent_composer/run.py tests/engine/test_run_reuse.py tests/engine/test_llm_cascade_run.py tests/engine/test_run_resume.py
   git commit -m "fix(run): isolate each run graph"
   ```

### Slice 1.2: Merge gate

1. Run the full suite, inspect the recursive before/after snapshots in the failing-test message, and confirm no test had to reload a flow as a workaround.

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
   git diff --check
   ```

2. Open the #34 PR with `Fixes #34`; merge before branching #116.

## PR 2 / #116: Seed-First Growth, Layers, And Fingerprints

### Slice 2.1: Make `Grow` carry only seed and prune

1. **Red test (2-5 min)**: update `tests/engine/test_grow_outcome.py` and `test_node_result.py` to construct `Grow(seed={...}, prune=frozenset(...))`, assert it has no `subgraph`, and assert `NodeResult` still accepts it.
2. **Verify red**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_grow_outcome.py tests/engine/test_node_result.py -q
   ```

3. **Implement (2-5 min)** in `src/agent_composer/nodes/base.py`:

   ```python
   @dataclass(frozen=True)
   class Grow:
       seed: Any
       prune: frozenset[str] = frozenset()
   ```

   Keep `Pause` in `NodeResult` during this PR.
4. **Green** with the same command.
5. **Commit**: `git commit -am "refactor(nodes): make Grow seed-first"`.

### Slice 2.2: Lower expansion ID helpers

1. **Red test (2-5 min)**: update `tests/engine/test_expand_ids.py` to import `ns` and `map_callsite` from `agent_composer.nodes.ids`; retain exact current ID strings.
2. **Verify red** with `test_expand_ids.py`.
3. **Implement (2-5 min)**: add `src/agent_composer/nodes/ids.py` with the two pure functions; import them into `compile/expand.py` and LOOP; remove their old definitions from `compile/expand.py`.
4. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_expand_ids.py tests/engine/test_clone_child.py tests/engine/test_call_subgraph_builder.py tests/engine/test_map_subgraph_builder.py tests/engine/test_loop_subgraph_builder.py -q
   ```

5. **Commit**: `git commit -am "refactor(nodes): lower expansion id helpers"` (include the new file explicitly).

### Slice 2.3: Move construction to the compile layer

1. **Red test (2-5 min)**: add `tests/engine/test_grow_builder.py` cases for CALL post-assert lowering, MAP with N and zero records, LOOP continuation/respawn/prune, and the temporary AGENT continuation seed.
2. **Red test (2-5 min)**: update `test_call_node.py`, `test_map_node.py`, and `test_loop_node.py` to assert node execution returns seed/prune only.
3. **Verify red**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_grow_builder.py tests/engine/test_call_node.py tests/engine/test_map_node.py tests/engine/test_loop_node.py -q
   ```

4. **Implement (2-5 min)**: add `compile.expand.build_grow(spawner: Node, canonical_seed: Any) -> Flow`. Match the closed `NodeKind` in compile, delegate to `call_subgraph`, `map_subgraph`, and `loop_continue_subgraph`; temporarily delegate AGENT seeds to `agent_segment_subgraph` so pause/checkpoint tests stay green until #118.
5. **Implement (2-5 min)**: make CALL/MAP/LOOP return `Grow(seed=..., prune=...)`; delete their `replay_grow` methods and compile imports. Give LOOP `grow_depth_delta = 0` and test inherited depth without increment.
6. **Implement (2-5 min)**: add keyword-only `grow_builder` to `FlowEngine.__init__` and `FlowEngine.restore`; have `_apply_grow` call it before mutation and have compatibility replay rebuild from each `GrowRecord.seed`. Inject `build_grow` from root `run.py`.
7. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_grow_builder.py tests/engine/test_call_node.py tests/engine/test_map_node.py tests/engine/test_loop_node.py tests/engine/test_apply_grow.py tests/engine/test_durable_replay.py tests/engine/test_agent_continuation.py -q
   ```

8. **Commit**: `git commit -am "refactor(compile): own growth construction"`.

### Slice 2.4: Add stable manifests and digests

1. **Red test (2-5 min)**: add `tests/engine/test_flow_identity.py::test_flow_fingerprint_is_reload_stable` and cases proving comments/ranges/object identity do not change it.
2. **Red test (2-5 min)**: add parametrized semantic mutations for prompt, code/reference string, node kind/fields, types, ordered edge order, wiring, asserts, env, authored LLM config, outputs, and recursive child flow; each must change the fingerprint.
3. **Red test (2-5 min)**: add expansion-digest tests for entry/end IDs, node execution fields, ordered edges, and wiring; dict insertion order must not matter.
4. **Verify red**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_flow_identity.py -q
   ```

5. **Implement (2-5 min)**: add `src/agent_composer/compile/manifest.py` with `_canonical(value)`, `flow_manifest(flow)`, `flow_fingerprint(flow)`, and `expansion_digest(flow)`. Use canonical JSON with compact separators and SHA-256.
6. **Implement (2-5 min)**: whitelist semantic node fields and explicit exclusions. Preserve ordered lists; sort map/set keys; recursively manifest `child`; raise on an unknown noncanonical field rather than silently dropping it.
7. **Green** with `test_flow_identity.py` plus `test_expand.py`.
8. **Commit**: `git commit -am "feat(compile): add stable flow manifests"`.

### Slice 2.5: Capture and restore effective LLM configuration

1. **Red test (2-5 min)**: extend `tests/engine/test_llm_cascade.py` for stable static paths, recursive children, environment provider/model filling even with `inherit: false`, and exact missing/extra path rejection on apply.
2. **Verify red** with `test_llm_cascade.py` and `test_llm_cascade_resume.py`.
3. **Implement (2-5 min)** in `compile/llm_cascade.py`:

   ```python
   def resolve_llm_cascade(
       flow, parent_config: dict, *, defaults: dict, path: tuple[str, ...] = ()
   ) -> dict[str, dict]: ...

   def apply_effective_llm_map(flow, effective: dict[str, dict], *, path=()) -> None: ...
   ```

   Remove per-child deep copies because #34 copies the root first. Capture fully resolved provider/model once; validate exact static path sets on apply.
4. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_llm_cascade.py tests/engine/test_llm_cascade_run.py tests/engine/test_llm_cascade_resume.py tests/engine/test_run_reuse.py -q
   ```

5. **Commit**: `git commit -am "feat(compile): capture effective llm configuration"`.

### Slice 2.6: Ratchet the layer boundary and update charters

1. **Red test (2-5 min)**: add `tests/engine/test_layer_direction.py` using `ast` to reject any import from `agent_composer.compile` below `src/agent_composer/nodes/` and any `runtime` import of `compile.expand`.
2. **Implement (2-5 min)**: remove the remaining AGENT/tool-loop compile imports by making the temporary continuation path return only its seed; build it in compile. Keep the compatibility builder arm until #118.
3. **Implement (2-5 min)**: update `nodes/__init__.py`, `compile/__init__.py`, `runtime/__init__.py`, `src/agent_composer/README.md`, and the engine/structure skill references to describe injection and the real import direction without yet claiming Pause is gone.
4. **Green and merge gate**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_layer_direction.py tests/engine/test_kind_census.py tests -q
   git diff --check
   ```

5. **Commit**: `git commit -am "refactor(structure): enforce growth layer direction"`.
6. Open the #116 PR with `Fixes #116`; merge before #117.

## PR 3 / #117: Append-Only Run Log And Recovery

### Slice 3.1: Define closed records and canonical value codecs

1. **Red test (2-5 min)**: add `tests/engine/test_run_records.py` to round-trip every record/envelope, reject an unknown kind/extra field, and require frozen payloads and contiguous integer sequence values.
2. **Red test (2-5 min)**: add nested tuple/date/`FileRef` output and seed cases. Assert `canonicalize(value)` returns `(wire, decoded)` and that encoding `decoded` produces the identical wire.
3. **Verify red**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_records.py -q
   ```

4. **Implement (2-5 min)** in new `src/agent_composer/runtime/log.py`: frozen record dataclasses, closed `RunRecord` union, `RecordEnvelope`, `encode_record`, and `decode_record`. Keep public observer classes in `events.py`; do not reuse them as durable values.
5. **Implement (2-5 min)**: add `canonical_typed(value, declared) -> tuple[dict, TypedValue]`, `canonical_seed(value) -> tuple[dict, Any]`, and pool encode/decode helpers using `ANY_VALUE_ADAPTER`/Pydantic JSON mode. The live side must use the decoded object.
6. **Green** with the same command.
7. **Commit**: `git commit -am "feat(runtime): define run records and canonical codecs"`.

### Slice 3.2: Implement in-memory and durable logs

1. **Red test (2-5 min)**: add `tests/engine/test_run_log.py` cases for `MemoryRunLog.append/read`, schema/run ID/sequence validation, and append returning only after the envelope is retained.
2. **Red test (2-5 min)**: add JSONL cases for `O_CREAT|O_EXCL`, mode `0600`, one complete line per append, `flush` plus one `os.fsync` per record, and refusal to overwrite an existing fresh path.
3. **Red test (2-5 min)**: add resume cases for lock-before-read, two-process writer exclusion, physical truncation of an unterminated tail followed by file `fsync`, malformed complete-line rejection, gaps, and identity mismatch.
4. **Verify red**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_log.py -q
   ```

5. **Implement (2-5 min)**:

   ```python
   class MemoryRunLog:
       def append(self, record: RunRecord) -> RecordEnvelope: ...
       def records(self) -> tuple[RecordEnvelope, ...]: ...

   class JsonlRunLog:
       @classmethod
       def create(cls, path: Path, run_id: str) -> "JsonlRunLog": ...
       @classmethod
       def resume(cls, path: Path) -> "JsonlRunLog": ...
       def append(self, record: RunRecord) -> RecordEnvelope: ...
       def close(self) -> None: ...
   ```

   Use `os.open(..., O_WRONLY|O_CREAT|O_EXCL, 0o600)`, `fcntl.flock(..., LOCK_EX|LOCK_NB)`, one UTF-8 line write, flush, and `os.fsync` before returning.
6. **Implement (2-5 min)**: while holding the resumed writer lock, find the last newline. Truncate any bytes after it, flush/fsync the truncation, then decode and validate all complete lines. A valid JSON object without its terminating newline is still an incomplete line and is discarded.
7. **Green** with `test_run_log.py`.
8. **Commit**: `git commit -am "feat(runtime): add durable jsonl run log"`.

### Slice 3.3: Reduce initialization, start, and resume

1. **Red test (2-5 min)**: add `tests/engine/test_run_state.py::test_run_started_restores_full_canonical_initial_state`. Assert clock/run ID/system values, START value/start ID/wiring, original canonical input, effective LLM map, and ready `[START_ID]` are exact.
2. **Red test (2-5 min)**: add exact-head `NodeStarted` tests (wrong head rejects before append), ordered in-flight tests, and `RunResumed` tests that require the recorded retry list to equal replay-derived in-flight and prepend it in dispatch order.
3. **Verify red** with `test_run_state.py`.
4. **Implement (2-5 min)** in new `runtime/state.py`: `RunState`, frozen `ApplyPlan`, `prepare_run_started`, `prepare_record`, and `apply`. Move disposition/join logic from `StateManager` without changing its three-state semantics.
5. **Implement (2-5 min)**: make `RunStarted` initialization restore only logged pool/config values; never read the clock/default model/run ID during replay.
6. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_state.py tests/engine/test_readiness.py tests/engine/test_binding_coskip.py -q
   ```

7. **Commit**: `git commit -am "feat(runtime): reduce run initialization and dispatch state"`.

### Slice 3.4: Reduce success, route, skips, failure, and quiescence

1. **Red test (2-5 min)**: add success tests for resolved `commit_target`, declared-type validation before append, canonical decoded pool value, outputs-before-successors, and exact FIFO successor addition.
2. **Red test (2-5 min)**: add route tests deriving branch/recursive skip closures without a durable skip record; add public derived `NodeSkipped` observer assertions.
3. **Red test (2-5 min)**: add `tests/engine/test_run_lifecycle.py` cases where END commits before a running sibling, a later sibling failure wins, and success/`TerminalSkipped` appears only when ready and in-flight are empty.
4. **Verify red**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_state.py tests/engine/test_run_lifecycle.py -q
   ```

5. **Implement (2-5 min)**: preparation resolves type/target/branch/skip closure into total container operations. `apply` assigns already-decoded `TypedValue` directly; no type check or graph lookup may remain after append.
6. **Implement (2-5 min)**: derive `RunSucceeded`, `RunFailed(error_type="TerminalSkipped")`, `NodeSkipped`, and final `RunFailed` observer values after reducer transactions. Do not add terminal/skip records.
7. **Green** with the same command plus `test_coskip_e2e.py`, `test_route.py`, and `test_commit_as.py`.
8. **Commit**: `git commit -am "feat(runtime): reduce node outcomes at quiescence"`.

### Slice 3.5: Prepare and replay flat growth

1. **Red test (2-5 min)**: add `tests/engine/test_run_log_growth.py` covering flat nested CALL/MAP/LOOP chronology, the actual executing LOOP clone ID, seed canonicalization, digest equality, collision/budget/prune validation, atomic prune, and changed-but-valid builder rejection.
2. **Red test (2-5 min)**: replay after each expansion prefix and compare graph nodes, ordered edges, wiring, dispositions, pool, ready/in-flight, and depth with the uninterrupted state.
3. **Verify red** with `test_run_log_growth.py`.
4. **Implement (2-5 min)**: `prepare_expanded(state, node_id, Grow, grow_builder)` canonicalizes seed, invokes builder, validates fragment entry/end/references/collisions/budgets/prune IDs, computes digest, and returns a record plus total splice/prune plan.
5. **Implement (2-5 min)**: reducer splice/register/mark-depth/mark-EXPANDED/prune/enqueue as one transaction. Prune every graph/state/pool/depth/ready/in-flight/index entry named by the plan.
6. **Implement (2-5 min)**: replay invokes the builder with the recorded canonical seed, checks the digest, then applies. Maintain the old nested `GrowRecord` ledger in parallel only for #117 `snapshot` compatibility; it is not consulted by log recovery.
7. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_log_growth.py tests/engine/test_graph_growth_e2e.py tests/engine/test_loop_run.py tests/engine/test_loop_body_spawner_resume.py tests/engine/test_kind_census.py -q
   ```

8. **Commit**: `git commit -am "feat(runtime): reduce and replay flat expansions"`.

### Slice 3.6: Normalize worker evaluation and serial dispatch

1. **Red test (2-5 min)**: add `tests/engine/test_runtime_dispatch.py` asserting `NodeStarted` is appended/reduced/observed before node code runs and append failure prevents execution.
2. **Red test (2-5 min)**: cover raw `Output|Route|Grow|failure`, canonicalization failure becoming `NodeFailed`, and direct `FlowEngine.run()` compatibility.
3. **Implement (2-5 min)** in `runtime/eval_node.py`: stop emitting started/terminal events. Return a raw completion/failure while yielding only transient `StreamChunk`s; keep node assert/bind/LLM-cap behavior.
4. **Implement (2-5 min)** in `runtime/engine.py`: dispatcher selects ready head, commits `NodeStarted`, then runs serial work and commits one prepared outcome. Add `on_event` after reduce and `on_stream` only for chunks.
5. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_runtime_dispatch.py tests/engine/test_engine.py tests/engine/test_eval_node.py tests/engine/test_golden_baseline.py -q
   ```

6. **Commit**: `git commit -am "refactor(runtime): dispatch serial work through records"`.

### Slice 3.7: Add bounded pooled dispatch and structured drain

1. **Red test (2-5 min)**: add positive-worker tests proving the dispatcher owns ready, at most N nodes are submitted, completion commit order follows the single worker-message queue, and submit failure leaves a resumable unfinished start.
2. **Red test (2-5 min)**: add stream/callback tests: chunks go only to `on_stream`, historical chunks never replay, queued/late chunks are discarded after stop, and callback failure preserves/re-raises the original exception after drain/unlock.
3. **Implement (2-5 min)**: replace worker ready consumption with dispatcher assignments. Use one worker-to-dispatcher queue carrying `StreamMessage` and `CompletionMessage`; workers check the stop event before beginning assigned external work.
4. **Implement (2-5 min)**: on failure, append error, callback exception, or private host stop: stop dispatch, set cancellation, wake cooperative nodes, drain already-started work while discarding outcomes/chunks, then release writer and return/re-raise. Do not submit an executor backlog.
5. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_runtime_dispatch.py tests/engine/test_run_lifecycle.py tests/engine/test_parallel.py tests/engine/test_run_flow_workers.py -q
   ```

6. **Commit**: `git commit -am "refactor(runtime): bound parallel dispatch and drain"`.

### Slice 3.8: Integrate fresh run logs and `RunResult`

1. **Red test (2-5 min)**: add fresh `run_flow(log=None)` memory-log equivalence, explicit path creation, full `RunStarted`, `run_id`/log result fields, original canonical input, and current-segment event assertions.
2. **Red test (2-5 min)**: inject append/fsync failure and prove the proposed state transition and observer are absent.
3. **Implement (2-5 min)** in `run.py`: fingerprint before copying; copy before cascade; capture defaults/effective map; coerce/seed/canonicalize the complete pool; create Memory/JSONL writer; append/reduce `RunStarted`; drive; close in `finally` after drain.
4. **Implement (2-5 min)**: factor one terminal fold used by fresh and resumed drives. Add `run_id`, `log`, and current-segment events to `RunResult` while retaining pause/checkpoint fields temporarily for compatibility.
5. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run.py tests/engine/test_run_records.py tests/engine/test_run_log.py tests/engine/test_run_lifecycle.py tests/engine/test_golden_run_results.py -q
   ```

6. **Commit**: `git commit -am "feat(run): persist fresh runs as records"`.

### Slice 3.9: Add log recovery and preserve old pause API temporarily

1. **Red test (2-5 min)**: add `tests/engine/test_run_recovery.py`: validate/lock first; reject fingerprint/schema/run ID/sequence/config mismatches; replay effective LLM without new env defaults; return terminal logs read-only; requeue unfinished starts in order; allow only `num_workers` override.
2. **Red test (2-5 min)**: assert nonterminal `RunResult.input` is original canonical input, events begin at `RunResumed`, callbacks do not see history, and terminal replay has no events/callbacks.
3. **Implement (2-5 min)**: add log arm to `resume_flow`; deep-copy supplied graph, apply logged effective LLM map, fold with callbacks disabled, verify expansions, append/reduce `RunResumed`, then continue the same writer.
4. **Implement (2-5 min)**: retain the old `engine|checkpoint|commands` arms until #118. A legacy `Pause` leaves its durable `NodeStarted` unfinished and emits transient `RunPaused`; `snapshot/restore` adapt new state plus the temporary mirrored growth ledger; answer delivery commits a normal `NodeSucceeded` record. Do not add a pause record.
5. **Green new and old paths**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_recovery.py tests/engine/test_run_resume.py tests/engine/test_run_resumed_event.py tests/engine/test_suspension.py tests/engine/test_durable_replay.py tests/engine/test_child_suspend.py -q
   ```

6. **Commit**: `git commit -am "feat(run): recover unfinished runs from jsonl"`.

### Slice 3.10: Prefix matrix, durability benchmark, and merge gate

1. Add a parametrized prefix matrix over nested CALL/MAP/LOOP for `num_workers=0` and `>1`, including incomplete starts and every authoritative outcome. Compare full reconstructed state and final result.
2. Run the focused matrix and full suite:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_recovery.py tests/engine/test_run_log_growth.py tests/engine/test_runtime_dispatch.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
   ```

3. Benchmark 100 short in-memory records versus 100 JSONL `fsync` records on the development filesystem; paste median/record into #117 and the PR. Keep per-record `fsync` regardless of the result.

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m timeit -n 1 -r 5 -s 'from pathlib import Path; from tempfile import TemporaryDirectory; from agent_composer.runtime.log import JsonlRunLog, NodeStartedRecord' 'd=TemporaryDirectory(); p=Path(d.name)/"r.jsonl"; w=JsonlRunLog.create(p,"bench"); [w.append(NodeStartedRecord(str(i))) for i in range(100)]; w.close(); d.cleanup()'
   ```

4. `git diff --check`; open the #117 PR with `Fixes #117`. Confirm CI still runs all old pause/checkpoint tests before merge.

## PR 4 / #118: Block Interaction And Delete Suspension

### Slice 4.1: Add strict normal-tool invocation and `ask_user` ownership

1. **Red test (2-5 min)**: add `tests/engine/test_tool_registry.py` for public schema (`question` only), public unknown/injected-key rejection, trusted metadata, reserved ID, duplicate active owner, sequential installs, and identity-safe stale-owner cleanup.
2. **Implement (2-5 min)** in `tools/registry.py`: lock registry mutation; add `invoke_tool(tool_id, inputs)`, `invoke_tool_trusted(tool_id, inputs)`, and `install_ask_user(handler)` context manager. Build the reserved `StructuredTool` with `Annotated[..., InjectedToolArg]` metadata.
3. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_tool_registry.py tests/engine/test_tool_calling_structured.py -q
   ```

4. **Commit**: `git commit -am "feat(tools): unify invocation and install ask_user"`.

### Slice 4.2: Route TOOL and AGENT calls through the shared boundary

1. **Red test (2-5 min)**: extend `tests/engine/test_agent.py` for returned-order calls, ungranted globally registered tool rejection, hidden/unknown args, and tool exception propagation to node failure.
2. **Implement (2-5 min)**: `ToolNode.forward` calls public `invoke_tool`; agent validates each requested name against `ctx.tools`, calls `invoke_tool` in returned order, and `str()`s only when constructing `ToolMessage`. Delete catch-to-string `run_tool` behavior, but retain control continuation temporarily.
3. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_tool_registry.py tests/engine/test_agent.py tests/engine/test_eval_node.py -q
   ```

4. **Commit**: `git commit -am "refactor(agent): use normal tool invocation"`.

### Slice 4.3: Write replacement blocking tests before deleting old tests

1. Add replacement tests without deleting pause tests: AGENT calls installed `ask_user` and receives a `ToolMessage`; HUMAN_INPUT passes exact trusted metadata and returns `Output`; independent work finishes while one worker blocks; all blocked workers stall queued work; interrupted started node is retried from the log.
2. Add WAIT tests for date midnight UTC, naive UTC, offset normalization, past immediate return, exact `Event.wait(seconds)`, cancellation wake, and recovery recomputing remaining absolute time. Use a patched `_utc_now` and fake event.
3. Run only replacement tests and confirm they fail for the intended missing blocking behavior:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_agent.py tests/engine/test_human_input_node_questions.py tests/engine/test_human_questions_e2e.py tests/engine/test_resolve_until.py tests/engine/test_run_lifecycle.py -q
   ```

### Slice 4.4: Simplify AGENT and HUMAN_INPUT

1. **Implement (2-5 min)**: remove `Fresh`, `Resume`, `AgentEntry`, `entry`, controls, spawner traits, `_ctx` continuation fields, `replay_grow`, message serialization, `agent_step`, and synthetic segment building. `AgentNode` stores `prompt` directly and runs one local mode loop.
2. **Implement (2-5 min)**: remove `controls` from `AgentDescriptor`; author human access as `tools: [ask_user]`. Do not validate registry availability at load time.
3. **Implement (2-5 min)**: make `HumanInputNode.forward` render/parse its existing surface, call `invoke_tool_trusted("ask_user", metadata)`, and return `Output(answer)`.
4. Run the replacement AGENT/HUMAN_INPUT tests until green before removing any old continuation tests.

### Slice 4.5: Make WAIT blocking and relocate the watchdog

1. **Implement (2-5 min)**: reduce `WaitNode` to timed-only; parse date/datetime into UTC and call injected `cancel_event.wait(max(0, remaining))`; return `Output(None)` on timeout/immediate completion.
2. **Implement (2-5 min)**: add node-owned `needs_cancel` trait and inject the transient engine event kind-blind. When stop is set, evaluation returns a private discarded worker-cancel outcome rather than `NodeFailed`.
3. **Implement (2-5 min)**: move `runtime/watchdog.py` to `nodes/code/watchdog.py` and wrap only CODE execution. Expected HUMAN_INPUT/WAIT blocking must never produce a stuck-node warning.
4. Run the WAIT and watchdog replacements until green.

### Slice 4.6: Remove suspension and its compatibility adapter

1. Now that replacement tests are green, delete `suspension/`, `nodes/agent/controls/`, agent continuation builders, event/watch WAIT fields, `Pause`, pause events/reasons, aborted state, commands, checkpoint/snapshot/restore, old resume arms, mirrored `GrowRecord` ledger, pause/deferred queues, `state_manager.py`, and CLI pause loops.
2. Finalize `NodeResult = Output | Route | Grow`; final `RunResult` fields/signatures; final log-only `resume_flow`; retain `RunResumed` and transient `StreamChunk`.
3. Rewrite recovery-bearing tests rather than simply deleting them: `test_durable_replay.py`, `test_run_resume.py`, `test_child_suspend.py`, `test_loop_body_spawner_resume.py`, and `test_llm_cascade_resume.py`. Delete only continuation/checkpoint-specific files after their replacement coverage is green: `test_suspension.py`, `test_expansions.py`, `test_engine_expansions_ledger.py`, `test_agent_continuation.py`, `test_agent_pause_post_assert.py`, `test_agent_segment_builder.py`, `test_agent_step.py`, `test_resume_agent.py`, and `test_structured_resume.py`.
4. Replace `test_phase2_cleanup.py` with deletion ratchets for every removed symbol/package/import and confirm kind/layer census.
5. Add a temporary locked CLI `ask_user` handler using the existing render helpers so #118 remains fully green. For `ac chat` only, the temporary handler may translate textual exit into the private stop signal. Mark it for unconditional removal in #119.
6. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_tool_registry.py tests/engine/test_agent.py tests/engine/test_human_input_node_questions.py tests/engine/test_human_questions_e2e.py tests/engine/test_resolve_until.py tests/engine/test_run_recovery.py tests/engine/test_phase2_cleanup.py tests/engine/test_kind_census.py tests/engine/test_layer_direction.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
   ```

7. **Commit**: `git add -A && git commit -m "refactor(engine): replace checkpoint suspension with blocking nodes"`.
8. `git diff --check`; open #118 PR with `Fixes #118`. Verify direct `FlowEngine` remains importable/runnable and lacks only the explicitly removed snapshot/restore/abort surface.

## PR 5 / #119: Main-Thread CLI Broker, Log Flags, Chat, And Docs

### Slice 5.1: Build the terminal broker

1. **Red test (2-5 min)**: add `tests/cli/test_terminal_broker.py` for blocking fulfillment, two concurrent prompts serialized FIFO, prompt/render main-thread identity, structured questions, generic EOF cancellation, chat EOF value, and `cancel_all` waking every waiter.
2. **Implement (2-5 min)**: add `src/agent_composer/cli/terminal.py` with typed prompt/render/controller-completion carriers and `TerminalBroker`. Use one FIFO for both render updates and prompt requests; only `serve()` touches Rich/questionary/input.
3. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli/test_terminal_broker.py tests/engine/test_cli_questions_render.py -q
   ```

4. **Commit**: `git commit -am "feat(cli): add main-thread terminal broker"`.

### Slice 5.2: Add CLI log and resume option validation

1. **Red test (2-5 min)** in `tests/cli/test_run.py`: default log under an isolated HOME, `--log PATH`, `--resume PATH`, terminal resume, and worker override. Parametrize each forbidden resume combination (`--input`, `--inputs`, `--provider`, `--model`, `--log`) and assert no input/provider prompt runs.
2. **Implement (2-5 min)**: add `log: Optional[Path]` and `resume: Optional[Path]` options. Make CLI `num_workers` optional so omission is distinguishable; choose `0` for fresh and recorded value for resume. Mint fresh run ID before selecting `~/.agent_composer/runs/<run_id>.jsonl`; create the directory privately; report path.
3. **Implement (2-5 min)**: fresh calls controlled `run_flow`; resume calls controlled `resume_flow`. Do not pass inputs or LLM flags on resume.
4. **Green** with `tests/cli/test_run.py` and `tests/engine/test_cli_llm_flags.py`.
5. **Commit**: `git commit -am "feat(cli): expose durable run logs and resume"`.

### Slice 5.3: Drive all runs through the main-thread broker

1. **Red test (2-5 min)**: add `num_workers=0` blocking HUMAN_INPUT, positive-worker concurrent questions, Rich/callback thread identity, sequential `CliRunner` registry cleanup, Ctrl-C wake/drain/unlock/exit-130, and resumable interrupted-log cases.
2. **Implement (2-5 min)**: install `ask_user` for the controller lifetime; run the private controlled run/resume function on a controller thread even at workers=0; observer adapters enqueue render work; main thread serves broker FIFO.
3. **Implement (2-5 min)**: on SIGINT, set the internal stop event, cancel pending broker requests, wait for structured drain and log unlock, print cancellation, and exit 130. Remove #118's lock/direct prompt compatibility handler.
4. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli/test_terminal_broker.py tests/cli/test_run.py tests/engine/test_cli_progress.py tests/engine/test_cli_questions_render.py -q
   ```

5. **Commit**: `git commit -am "refactor(cli): drive runs through terminal broker"`.

### Slice 5.4: Make chat exits flow-owned

1. **Red test (2-5 min)**: rewrite `tests/cli/test_chat.py` cases so typed `/exit` and `/quit` reach the flow but never the reply model, chat EOF supplies `/exit`, reply renders before the next prompt, two turns work, failed reply restarts from last completed transcript, and Ctrl-C leaves a resumable log.
2. **Red test (2-5 min)**: `tests/cli/test_chat_flow_loads.py` loads bundled, example, and composing-skill template flows.
3. **Implement (2-5 min)** in all three YAMLs: after `ask`, a CASE routes `/exit|/quit` to an exit record (`exited: true`) before `reply`; normal branch runs reply and yields updated transcript with `exited: false`; coalesce branch records at the turn output.
4. **Implement (2-5 min)** in `cli/chat/command.py`: remove `console.input`, pause/resume, and textual-command interception. Use the broker; normalize chat EOF only; mirror transcript from durable completed-turn observers; on failed turn reload immutable flow and start a new run/log from that transcript.
5. **Green**:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli/test_chat.py tests/cli/test_chat_flow_loads.py tests/cli/test_chat_tools.py -q
   ```

6. **Commit**: `git commit -am "refactor(chat): make textual exit flow-owned"`.

### Slice 5.5: Update examples, charters, docs, and skills

1. Replace `controls: [ask_user]` with `tools: [ask_user]` in examples/templates/seeds. Document blocking worker consumption, all-workers-blocked behavior, UTC WAIT, node-atomic retry, log sensitivity/permissions/locking/fsync cost, CLI flags, and terminal-log behavior.
2. Update `CLAUDE.md` (`AGENTS.md` is its symlink), `README.md`, `src/agent_composer/README.md`, `docs/{index,architecture,engine,nodes,typing,syntax,api,cli,examples,reference}.md`, affected package `__init__.py` charters, and `.claude/skills/{engine,composing-agents,structure}` source files/templates. Do not edit `.agents` symlink targets separately.
3. Active docs must say `Output | Route | Grow`, append-only recovery, blocking HUMAN_INPUT/WAIT, and at-least-once unfinished-node retry. Remove every claim that checkpoint suspension is non-negotiable.
4. Validate examples/templates and docs:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_templates_load.py tests/examples/test_examples.py tests/cli/test_chat_flow_loads.py -q
   conda run -n agentcomposer python -m mkdocs build --strict
   ```

5. **Commit**: `git commit -am "docs(engine): document node-atomic run-log recovery"`.

### Slice 5.6: End-to-end verification

1. Search active source/docs (exclude historical `docs/plans` and generated `site`) for removed vocabulary; every hit must be justified or removed:

   ```bash
   git grep -n -E 'RunCheckpoint|PauseRequested|RunPaused|RunAborted|AgentEntry|GrowRecord|replay_grow|controls:' -- ':!docs/plans/**' ':!site/**'
   ```

2. Run structural, recovery, interaction, CLI, and full suites:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_kind_census.py tests/engine/test_layer_direction.py tests/engine/test_phase2_cleanup.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_recovery.py tests/engine/test_run_log_growth.py tests/engine/test_run_lifecycle.py tests/cli/test_run.py tests/cli/test_chat.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
   conda run -n agentcomposer python -m mkdocs build --strict
   git diff --check
   ```

3. Manually smoke-test a fresh logged run, resume an intentionally interrupted run, a blocking `ask_user`, UTC WAIT, and flow-owned `/exit` with both `-w 0` and `-w 2`. Record paths/results in #119.
4. Run the adversarial self-review required by `CLAUDE.md`: truncate/corrupt logs, change flow/builder topology, fail fsync/callback/submit, interrupt all-blocked workers, and inspect for kind/layer/pool-purity regressions.
5. Open the #119 PR with `Fixes #119`. After it merges, close parent #111 with the merged PR links and leave #110 open for its explicitly deferred cleanup children.

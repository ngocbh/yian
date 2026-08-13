# Plan: Node-Atomic Runtime With Append-Only Run Logs (Revision 2)

**Goal**: Replace checkpoint suspension and synthetic agent continuations with a
small, node-atomic, parallel runtime whose append-only log reconstructs an
unfinished DAG at committed node boundaries.

**Architecture**: Workers execute whole nodes. A single dispatcher appends and
`fsync`s typed records before a total reducer mutates private run state. Human
input and timed WAIT block workers. Recovery folds the log and retries every
`NodeStarted` without a durable outcome.

**Tech Stack**: Python 3.11, dataclasses/Pydantic v2, LangChain tools, JSONL,
POSIX `fcntl.flock`, Typer/Rich/questionary, pytest, and MkDocs.

**Design**: [`2026-08-12-run-log-runtime-cleanup-design-final.md`](2026-08-12-run-log-runtime-cleanup-design-final.md)

**Tracking**: umbrella #110; parent #111; implementation issues #34, #116,
#118, #117, and #119.

## Changes From Previous Plan

- Changes the merge order to `#34 -> #116 -> #118 -> #117 -> #119`.
- Makes seed-first growth one atomic green migration: the `Grow` carrier, every
  producer, compile builder, runtime injection, replay, and direct-engine tests
  change in one commit.
- Retains per-callsite child deep copies in the LLM cascade and adds a shared
  child reached through differently configured parents regression.
- Lands `TerminalBroker`, controller-thread execution, normal `ask_user`,
  blocking AGENT/HUMAN_INPUT/WAIT, flow-owned chat exit, examples, syntax, and
  skills together in #118. There is no temporary worker-thread prompt path.
- Leaves generic `Pause` and checkpoint code dormant in #118 only. No
  authorable node can return `Pause` after #118.
- Makes #117 one log-only runtime/API cutover. It has no checkpoint adapter,
  mirrored growth ledger, or dual writer lifecycle.
- Defines the final direct `FlowEngine` contract and tests its migration at each
  relevant issue boundary.
- Splits reducer behavior into named test nodes and places prefix, append,
  submit, callback, and recovery failure tests before their implementations.
- Uses explicit `git add <paths> && git commit` commands, adds an isolated HOME
  fixture before default CLI logging, includes memory and JSONL benchmarks, and
  updates watchdog imports when ownership moves to CODE.
- Requires the design/plan branch to merge through a docs-only PR before any
  implementation branch is cut.

## Verified Problem

- `run_flow` resolves the LLM cascade on `loaded.compiled` and passes that same
  graph to a mutating `FlowEngine`, contradicting `ComposedFlow` reuse.
- CALL, MAP, LOOP, and AGENT construct graph fragments in nodes, creating the
  `compile -> nodes -> compile` cycle.
- AGENT still has `Fresh`, `Resume`, and `AgentEntry`; HUMAN_INPUT and WAIT
  return `Pause`; runtime snapshots duplicate pool, graph, queue, and growth
  state.
- Workers currently claim ready IDs themselves, so no durable dispatcher-owned
  `NodeStarted` can precede external work.
- `run_flow` returns live engine/checkpoint handles, while the accepted final
  `resume_flow` contract is log replay only.
- `ac chat` intercepts `/exit` and `/quit`; the accepted contract routes them in
  the flow.
- `resolve_llm_cascade` deliberately deep-copies each child. Deep-copying only
  the root preserves shared-reference aliasing and cannot replace this guard.

## Chosen Contracts

### Public run boundary

```python
run_flow(flow, inputs, *, log=None, run_id=None, num_workers=0,
         on_event=None, on_stream=None, llm_config=None) -> RunResult

resume_flow(flow, *, log, num_workers=None,
            on_event=None, on_stream=None) -> RunResult
```

`resume_flow` accepts no live engine, checkpoint, command, human answer, fresh
input, new run ID, or new LLM configuration. `RunResult` adds `run_id` and
optional `log`; it removes `engine`, `checkpoint`, and `pause_reasons`.

### Direct `FlowEngine`

`FlowEngine` remains importable as:

```python
from agent_composer.runtime import FlowEngine
```

It is removed from `agent_composer.__init__` and is not a package-root supported
API. After #117 its constructor is:

```python
FlowEngine(
    state: RunState,
    log: RunLog,
    *,
    grow_builder: GrowBuilder,
    num_workers: int,
    on_event: EventHook | None = None,
    on_stream: StreamHook | None = None,
    stop_event: threading.Event | None = None,
)
```

`state` already has `RunStarted` or replayed records applied, and `log` ends at
the matching sequence. The engine only drives that prepared frontier. It has no
legacy `(flow, pool, run_inputs, llm)` constructor, `snapshot`, `restore`,
`resume`, or `request_abort`. Advanced users may assemble `RunState` and a
`MemoryRunLog`; the supported application boundary remains `run_flow` and
`resume_flow`.

During #116 the old constructor remains, gains injected `grow_builder`, and all
direct dynamic tests pass it explicitly. During #118 direct runs exercise the
normal tool registry and blocking nodes. #117 performs the final constructor and
export migration atomically.

### Durable state

`RunStarted`, `RunResumed`, `NodeStarted`, `NodeSucceeded`, `NodeRouted`,
`NodeExpanded`, and `NodeFailed` are the only authoritative records. Skips and
terminal notifications are derived. `RunState` owns the private graph, pool,
dispositions, ready FIFO, ordered in-flight set, growth depth, and terminal
result. A started node without an outcome is requeued in original dispatch
order. In-flight external effects are at-least-once.

## Dependency And Merge Plan

| Order | Issue / branch | Depends on | Merge result |
| --- | --- | --- | --- |
| 0 | docs-only PR from `dev/engine/run-log-cleanup` | current `main` | Accepted design, review, and final plan are on `main` |
| 1 | #34 `dev/engine/per-run-graph-isolation` | docs PR merged | Every run mutates a private graph |
| 2 | #116 `dev/engine/seed-first-growth` | #34 merged | Atomic seed-first growth, correct layers, fingerprints |
| 3 | #118 `dev/engine/blocking-interaction` | #116 merged | Main-thread broker and all authorable blocking interaction |
| 4 | #117 `dev/engine/append-only-run-log` | #118 merged | Authoritative log/reducer and deletion of suspension |
| 5 | #119 `dev/cli/run-log-options` | #117 merged | Default CLI path and `--log`/`--resume` only |

Every issue must be assigned before work starts. Post the corresponding section
below as an issue comment, cut its branch from the freshly merged dependency,
and open a PR with `Fixes #<issue>`. Do not stack branches from the long-lived
design branch.

## Step 0: Merge The Design And Plan

This is a docs-only PR and must merge before implementation branches begin.

1. Add the design, both design reviews, original plan, plan review, and reviewed
   final plan. Do not claim an implementation issue is fixed.
2. Run:

   ```bash
   git diff --check
   conda run -n agentcomposer python -m mkdocs build --strict
   ```

3. Commit any remaining plan artifacts explicitly:

   ```bash
   git add docs/plans/2026-08-12-run-log-runtime-cleanup-*.md
   git commit -m "docs: finalize run-log runtime plan"
   ```

4. Open a docs-only PR with `Refs #111`, merge it, update local `main`, and only
   then cut `dev/engine/per-run-graph-isolation`.

# PR 1 / #34: Per-Run Graph Isolation

## Issue gate

Claim #34, post this PR section, branch from updated `main`, and record:

```bash
git status --short
PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_llm_cascade_run.py tests/engine/test_graph_growth_e2e.py -q
```

## Slice 1.1: Reproduce reuse mutation

1. Add `tests/engine/test_run_reuse.py` with these failing tests:
   - `test_repeated_nested_runs_do_not_mutate_composed_flow`
   - `test_different_llm_layers_do_not_leak_between_runs`
2. Snapshot recursive nodes, ordered edges, wiring, static children,
   `own_llm_config`, and effective `llm_config`; run the same loaded CALL/MAP/LOOP
   flow twice and compare snapshots/results.
3. Verify red:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_reuse.py -q
   ```

4. In `src/agent_composer/run.py`, deep-copy `loaded.compiled` before
   `resolve_llm_cascade` and before passing it to `FlowEngine`. Do the same for a
   checkpoint restore; a live engine continues its own already-private graph.
5. Verify green:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_reuse.py tests/engine/test_llm_cascade_run.py tests/engine/test_run_resume.py tests/engine/test_graph_growth_e2e.py -q
   ```

6. Commit:

   ```bash
   git add src/agent_composer/run.py tests/engine/test_run_reuse.py tests/engine/test_llm_cascade_run.py tests/engine/test_run_resume.py
   git commit -m "fix(run): isolate each run graph"
   ```

## Slice 1.2: PR gate

```bash
PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
git diff --check
```

Open the #34 PR with `Fixes #34`; merge it before claiming #116.

# PR 2 / #116: Atomic Seed-First Growth, Layers, And Identity

## Issue gate

Claim #116, post this PR section, branch from merged #34, and run the Grow,
expansion, replay, and LLM-cascade baseline.

## Slice 2.1: Lower pure ID helpers

1. Change `tests/engine/test_expand_ids.py` to import `ns` and `map_callsite`
   from `agent_composer.nodes.ids`; verify the focused test is red.
2. Add `src/agent_composer/nodes/ids.py`, update `compile/expand.py` and LOOP
   imports, and retain exact current ID strings.
3. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_expand_ids.py tests/engine/test_clone_child.py tests/engine/test_call_subgraph_builder.py tests/engine/test_map_subgraph_builder.py tests/engine/test_loop_subgraph_builder.py -q
   git add src/agent_composer/nodes/ids.py src/agent_composer/compile/expand.py src/agent_composer/nodes/loop/node.py tests/engine/test_expand_ids.py
   git commit -m "refactor(nodes): lower expansion id helpers"
   ```

## Slice 2.2: Atomic Grow migration (large integration gate)

This slice is intentionally larger than 2-5 minutes. The tree must never contain
a commit where the carrier changed but a producer or consumer still expects
`Grow.subgraph`.

1. Add failing replacement tests before source edits:
   - `tests/engine/test_seed_first_growth.py::test_call_map_loop_and_agent_return_seed_only`
   - `tests/engine/test_seed_first_growth.py::test_compile_builder_reconstructs_every_legacy_growth_kind`
   - `tests/engine/test_flow_engine_grow_builder.py::test_dynamic_direct_engine_requires_injected_builder`
   - `tests/engine/test_flow_engine_grow_builder.py::test_checkpoint_replay_uses_injected_builder`
2. Run those exact nodes and confirm failures are about the old carrier/builder.
3. In one working slice:
   - change `Grow` in `nodes/base.py` to `seed` plus `prune` while retaining
     `Pause` in `NodeResult`;
   - change CALL, MAP, LOOP, and temporary AGENT continuation producers;
   - add `compile.expand.build_grow(spawner, canonical_seed)` for the closed legal
     kinds and keep the temporary AGENT builder only through #116;
   - delete producer imports of `compile.expand` and producer `replay_grow` hooks;
   - inject `grow_builder` into `FlowEngine.__init__`, `_apply_grow`, and
     `FlowEngine.restore`, and store/replay canonical seeds in `GrowRecord`;
   - inject `build_grow` from package-root `run.py`;
   - update every direct dynamic engine callsite, including
     `test_apply_grow.py`, `test_expand.py`, `test_graph_growth_e2e.py`,
     `test_map.py`, `test_loop_run.py`, `test_loop_body_spawner_resume.py`,
     `test_durable_replay.py`, `test_engine_expansions_ledger.py`, and AGENT
     continuation tests, to pass `grow_builder=build_grow`.
4. Search for missed consumers before committing:

   ```bash
   grep -R -nE 'Grow\([^)]*subgraph|\.subgraph|replay_grow|FlowEngine\(' src/agent_composer tests/engine --include='*.py'
   ```

5. Run the atomic green gate:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_seed_first_growth.py tests/engine/test_flow_engine_grow_builder.py tests/engine/test_grow_outcome.py tests/engine/test_node_result.py tests/engine/test_call_node.py tests/engine/test_map_node.py tests/engine/test_loop_node.py tests/engine/test_apply_grow.py tests/engine/test_durable_replay.py tests/engine/test_engine_expansions_ledger.py tests/engine/test_agent_continuation.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine -q
   ```

6. Commit the carrier and every producer/consumer together:

   ```bash
   git add src/agent_composer/nodes/base.py src/agent_composer/nodes/call src/agent_composer/nodes/map src/agent_composer/nodes/loop src/agent_composer/nodes/agent src/agent_composer/compile/expand.py src/agent_composer/runtime/engine.py src/agent_composer/run.py src/agent_composer/suspension/expansions.py tests/engine
   git commit -m "refactor(engine): migrate growth atomically to seeds"
   ```

## Slice 2.3: Stable manifests and expansion digests

1. Add failing `tests/engine/test_flow_identity.py` tests for reload stability,
   ignored ranges/comments/object identity, and semantic changes to nodes,
   ordered edges, wiring, types, asserts, env, authored LLM layers, and recursive
   children. Add expansion digest cases for entry/end IDs and topology.
2. Add `src/agent_composer/compile/manifest.py` with explicit canonicalization,
   `flow_manifest`, `flow_fingerprint`, and `expansion_digest`. Unknown
   noncanonical execution fields must fail loudly.
3. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_flow_identity.py tests/engine/test_expand.py -q
   git add src/agent_composer/compile/manifest.py tests/engine/test_flow_identity.py
   git commit -m "feat(compile): add stable flow manifests"
   ```

## Slice 2.4: Effective LLM map without alias leakage

1. Add these failing tests to `tests/engine/test_llm_cascade.py`:
   - `test_effective_map_round_trips_static_paths`
   - `test_effective_map_rejects_missing_or_extra_paths`
   - `test_shared_child_is_dealiased_per_parent_provider`
2. The last test must use two parent paths sharing the same leaf child, one
   inheriting OpenAI and the other Anthropic, and prove both effective providers
   remain distinct after resolution.
3. Extend `resolve_llm_cascade` to return the effective static-path map and add
   `apply_effective_llm_map`. **Keep** `node.child = copy.deepcopy(node.child)`
   at each callsite; the root copy does not de-alias shared descendants.
4. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_llm_cascade.py tests/engine/test_llm_cascade_run.py tests/engine/test_llm_cascade_resume.py tests/engine/test_run_reuse.py -q
   git add src/agent_composer/compile/llm_cascade.py tests/engine/test_llm_cascade.py tests/engine/test_llm_cascade_run.py tests/engine/test_llm_cascade_resume.py
   git commit -m "feat(compile): capture isolated effective llm configuration"
   ```

## Slice 2.5: Ratchet and document the layer boundary

1. Add `tests/engine/test_layer_direction.py` using `ast` to reject
   `nodes -> compile` and `runtime -> compile.expand` imports.
2. Update `nodes/__init__.py`, `compile/__init__.py`, `runtime/__init__.py`,
   `src/agent_composer/README.md`, `.claude/skills/engine/{SKILL.md,reference.md}`,
   and `.claude/skills/structure/{SKILL.md,reference.md}`. Do not yet claim
   checkpoint suspension is removed.
3. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_layer_direction.py tests/engine/test_kind_census.py tests -q
   conda run -n agentcomposer python -m mkdocs build --strict
   git diff --check
   git add src/agent_composer/nodes/__init__.py src/agent_composer/compile/__init__.py src/agent_composer/runtime/__init__.py src/agent_composer/README.md .claude/skills/engine .claude/skills/structure tests/engine/test_layer_direction.py
   git commit -m "refactor(structure): enforce injected growth construction"
   ```

Open and merge the #116 PR before branching #118.

# PR 3 / #118: Blocking Interaction With Main-Thread Terminal Ownership

## Issue gate

Claim #118, post this PR section, and branch from merged #116. Generic
Pause/checkpoint tests remain a compatibility baseline in this PR, but the merge
gate requires that no authorable node return `Pause`.

## Slice 3.1: Normal registry invocation and `ask_user`

1. Add failing `tests/engine/test_tool_registry.py` cases for public schema
   (`question` only), unknown/injected-key rejection, trusted metadata, reserved
   ID, duplicate active owner, sequential owners, and identity-safe cleanup.
2. Add `invoke_tool`, `invoke_tool_trusted`, and `install_ask_user` to
   `tools/registry.py`; export them from `tools/__init__.py`.
3. Route TOOL and ordinary AGENT tool calls through the public invocation path.
   Validate model tool names against the node's authored list and propagate tool
   exceptions as node failures.
4. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_tool_registry.py tests/engine/test_tool_calling_structured.py tests/engine/test_agent.py tests/engine/test_eval_node.py -q
   git add src/agent_composer/tools/registry.py src/agent_composer/tools/__init__.py src/agent_composer/nodes/tool/node.py src/agent_composer/nodes/agent/modes/tool_calling.py tests/engine/test_tool_registry.py tests/engine/test_agent.py
   git commit -m "feat(tools): make ask_user a normal trusted tool"
   ```

## Slice 3.2: Build the broker before blocking any node

1. Add failing `tests/cli/test_terminal_broker.py` nodes for single fulfillment,
   concurrent FIFO prompts, render/prompt main-thread identity, structured
   questions, EOF policy, and `cancel_all` waking every waiter.
2. Add `src/agent_composer/cli/terminal.py`. Worker/controller callers only
   enqueue typed requests and wait; `TerminalBroker.serve()` alone touches
   Rich/questionary/input.
3. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli/test_terminal_broker.py tests/engine/test_cli_questions_render.py -q
   git add src/agent_composer/cli/terminal.py tests/cli/test_terminal_broker.py
   git commit -m "feat(cli): add main-thread terminal broker"
   ```

## Slice 3.3: Install controller-thread execution and cancellation plumbing

1. Add failing exact tests in `tests/cli/test_terminal_controller.py` for a
   controller callable completing while the main thread serves renders/prompts,
   exception propagation, Ctrl-C cancellation, and registry cleanup.
2. Add the private controlled-run helper in `cli/run.py`: install `ask_user`, run
   `run_flow` on a controller thread even when `num_workers=0`, enqueue observer
   rendering, and let the main thread serve the broker.
3. Add one transient engine stop `Event`. Worker outcomes after stop are
   discarded during drain; this is not a new public flow cancellation result.
4. Keep the old pause prompt loop available only for generic dormant compatibility
   until the authorable cutover. Do not add any direct worker-thread input path.
5. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli/test_terminal_controller.py tests/cli/test_terminal_broker.py tests/engine/test_cli_progress.py tests/engine/test_parallel.py -q
   git add src/agent_composer/cli/run.py src/agent_composer/runtime/engine.py src/agent_composer/runtime/eval_node.py tests/cli/test_terminal_controller.py tests/engine/test_cli_progress.py
   git commit -m "refactor(cli): run the engine behind the terminal broker"
   ```

## Slice 3.4: Write blocking replacement tests first

Add new files so each selected command can go green without rewriting old
Pause-oriented suites first:

- `tests/engine/test_blocking_ask_user.py`
  - `test_agent_receives_answer_as_tool_message`
  - `test_tool_calls_keep_model_order`
- `tests/engine/test_blocking_human_input.py`
  - `test_human_input_invokes_trusted_ask_user_metadata`
  - `test_human_input_returns_output`
- `tests/engine/test_blocking_workers.py`
  - `test_independent_node_finishes_while_one_worker_blocks`
  - `test_ready_work_waits_when_all_workers_block`
- `tests/engine/test_blocking_wait.py`
  - date midnight UTC, naive UTC, aware normalization, past immediate return,
    cancellation wake, and remaining absolute duration.

Run each file and confirm it fails for the missing blocking behavior. Do not
commit red tests separately.

## Slice 3.5: Authorable interaction cutover (large integration gate)

1. Remove `Fresh`, `Resume`, `AgentEntry`, `entry`, continuation memo/state,
   AGENT spawner behavior, controls, `agent_step`, and synthetic agent segment
   construction. AGENT executes one local model/tool loop.
2. Remove `controls` from `AgentDescriptor`; author access as
   `tools: [ask_user]`. HUMAN_INPUT invokes `invoke_tool_trusted("ask_user", ...)`
   and returns `Output`.
3. Delete `nodes/agent/controls/` and the AGENT arm from `build_grow`. Keep the
   generic `Pause` type, checkpoint classes, and direct artificial-pause tests
   untouched for #117.
4. Rewrite `cli/chat/command.py` to use the already-installed broker. Change
   `src/agent_composer/cli/chat/chat.yaml`, `examples/chat.yaml`, and
   `.claude/skills/composing-agents/templates/chat.yaml` so CASE routes `/exit`
   and `/quit` before reply. Normalize chat EOF to `/exit`; do not intercept typed
   exit in the host.
5. Update `examples/ask-user.yaml`, `docs/syntax.md`, `docs/nodes.md`,
   `docs/examples.md`, `docs/cli.md`, composing-agent skill SKILL/reference and
   `human-in-loop.yaml`/`human-questions.yaml` templates, plus affected package
   charters. Remove every active claim that authored human input is a durable
   pause. Retain accurate notes about the temporary generic checkpoint API.
6. Rewrite/delete only continuation-specific old tests after the new blocking
   files pass. Validate every example and template in this same gate.
7. Verify:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_blocking_ask_user.py tests/engine/test_blocking_human_input.py tests/engine/test_blocking_workers.py tests/cli/test_chat.py tests/cli/test_chat_flow_loads.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/examples/test_examples.py tests/engine/test_templates_load.py -q
   conda run -n agentcomposer python -m mkdocs build --strict
   ```

8. Commit all affected source and authoring artifacts together:

   ```bash
   git add -A -- src/agent_composer/nodes/agent src/agent_composer/nodes/human_input src/agent_composer/compose/nodes src/agent_composer/compile/expand.py src/agent_composer/cli/chat examples .claude/skills/composing-agents docs/syntax.md docs/nodes.md docs/examples.md docs/cli.md tests/engine tests/cli/test_chat.py tests/cli/test_chat_flow_loads.py tests/examples
   git commit -m "refactor(engine): make human interaction blocking and flow-owned"
   ```

## Slice 3.6: Blocking UTC WAIT and CODE-owned watchdog

1. Run `tests/engine/test_blocking_wait.py` red before implementation.
2. Reduce WAIT to absolute date/datetime only. Interpret date and naive datetime
   as UTC, normalize offsets, return immediately for past deadlines, and wait on
   the injected stop event for the remaining duration. Delete event/watch fields
   from node and compose descriptors.
3. Move `runtime/watchdog.py` to `nodes/code/watchdog.py` and wrap CODE only.
   Update direct imports and logger assertions in
   `tests/engine/test_code_inline.py` from `agent_composer.runtime.watchdog` to
   `agent_composer.nodes.code.watchdog`.
4. Update WAIT syntax/docs and composing templates in this PR. Rewrite old WAIT
   pause tests only after the replacement file is green.
5. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_blocking_wait.py tests/engine/test_resolve_until.py tests/engine/test_code_inline.py tests/engine/test_templates_load.py -q
   git add -A -- src/agent_composer/nodes/wait src/agent_composer/nodes/code src/agent_composer/compose/nodes/wait.py src/agent_composer/runtime/watchdog.py src/agent_composer/runtime/eval_node.py tests/engine/test_blocking_wait.py tests/engine/test_resolve_until.py tests/engine/test_code_inline.py docs/syntax.md docs/nodes.md .claude/skills/composing-agents
   git commit -m "refactor(nodes): block on UTC wait and scope watchdog to code"
   ```

## Slice 3.7: #118 compatibility and documentation gate

1. Add a ratchet proving AGENT, HUMAN_INPUT, and WAIT cannot return `Pause`,
   `controls:` is rejected, and no authorable AGENT constructor has `entry`.
2. Keep `Pause`, checkpoints, `FlowEngine.snapshot/restore/resume`, and
   `state_manager.py` only as dormant legacy compatibility. Do not add a worker
   prompt fallback.
3. Run:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_blocking_ask_user.py tests/engine/test_blocking_human_input.py tests/engine/test_blocking_wait.py tests/engine/test_blocking_workers.py tests/cli/test_terminal_broker.py tests/cli/test_terminal_controller.py tests/cli/test_chat.py tests/engine/test_kind_census.py tests/engine/test_layer_direction.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
   conda run -n agentcomposer python -m mkdocs build --strict
   git diff --check
   ```

4. Commit the ratchet explicitly:

   ```bash
   git add tests/engine/test_phase2_cleanup.py src/agent_composer/README.md docs/syntax.md docs/nodes.md docs/examples.md docs/cli.md .claude/skills/composing-agents .claude/skills/engine
   git commit -m "test(engine): ratchet blocking interaction surface"
   ```

5. Open #118 with `Fixes #118`, and merge before branching #117.

# PR 4 / #117: Authoritative Run Log And Suspension Deletion

## Issue gate

Claim #117, post this PR section, and branch from merged #118. This PR contains
one breaking cutover. It never adapts checkpoints to logs and never writes a
nested growth ledger beside the log.

## Slice 4.1: Closed records and canonical codecs

1. Add failing `tests/engine/test_run_records.py` cases for every record,
   envelope validation, unknown kinds/fields, tuple/date/FileRef output and seed
   canonicalization, and encode-decode-encode stability.
2. Add record dataclasses, `RunRecord`, `RecordEnvelope`, codecs, canonical typed
   value/seed helpers, and pool codecs to new `runtime/log.py`.
3. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_records.py -q
   git add src/agent_composer/runtime/log.py tests/engine/test_run_records.py
   git commit -m "feat(runtime): define canonical run records"
   ```

## Slice 4.2: Memory and durable log failure boundaries

1. Before implementation, add failing `tests/engine/test_run_log.py` nodes for:
   memory ordering; `O_EXCL`; `0600`; lock-before-read; second-writer exclusion;
   one flush/fsync per append; truncated-tail truncate+fsync; corrupt complete
   line; sequence/schema/run-ID mismatch; and append/fsync failure retaining the
   prior complete prefix.
2. Implement `MemoryRunLog` and `JsonlRunLog` in `runtime/log.py`. A valid JSON
   object without its final newline is still truncated as incomplete.
3. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_log.py -q
   git add src/agent_composer/runtime/log.py tests/engine/test_run_log.py
   git commit -m "feat(runtime): add locked jsonl run logs"
   ```

## Slice 4.3: RunState initialization and dispatch transitions

1. Add failing `tests/engine/test_run_state_initialization.py` nodes for complete
   canonical pool/system clock/run ID/START override/effective LLM restoration.
2. Implement the `RunState` container plus `prepare_run_started` and its total
   apply plan in new `runtime/state.py`; run the focused file green and commit.
3. Add failing `tests/engine/test_run_state_dispatch.py` nodes for exact ready
   head, ordered in-flight, and exact `RunResumed` retry list.
4. Implement only `NodeStarted` and `RunResumed` preparation/apply; run green and
   commit separately.

```bash
PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_state_initialization.py -q
git add src/agent_composer/runtime/state.py tests/engine/test_run_state_initialization.py
git commit -m "feat(runtime): reduce run initialization"
PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_state_dispatch.py -q
git add src/agent_composer/runtime/state.py tests/engine/test_run_state_dispatch.py
git commit -m "feat(runtime): reduce dispatch and retry state"
```

## Slice 4.4: Outcome reducers in small green cycles

Use new `tests/engine/test_run_state_outcomes.py`; each bullet is one red test
node, the smallest implementation, focused green command, and explicit commit:

1. `test_success_commits_canonical_value_before_fifo_successors` - resolve
   commit target/type and apply success.
2. `test_route_derives_skip_closure_without_skip_record` - prepare route and
   recursive skip observer deltas.
3. `test_failure_is_terminal_and_preserves_diagnostics` - apply `NodeFailed`.
4. `test_end_output_waits_for_full_quiescence` and
   `test_late_sibling_failure_wins` - derive success/TerminalSkipped only when
   ready and in-flight are empty.

Run and commit after every node, substituting these exact pairs in order:

| Test node | Commit message |
| --- | --- |
| `test_success_commits_canonical_value_before_fifo_successors` | `feat(runtime): reduce successful output` |
| `test_route_derives_skip_closure_without_skip_record` | `feat(runtime): reduce route and skip closure` |
| `test_failure_is_terminal_and_preserves_diagnostics` | `feat(runtime): reduce node failure` |
| `test_end_output_waits_for_full_quiescence test_late_sibling_failure_wins` | `feat(runtime): derive terminal state at quiescence` |

For the final row pass both node IDs to pytest. For each row run:

```bash
PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_state_outcomes.py::<test_name> -q
git add src/agent_composer/runtime/state.py tests/engine/test_run_state_outcomes.py
git commit -m "<commit message from table>"
```

No graph lookup, validation, or codec call may remain in `apply` after append.

## Slice 4.5: Flat growth reducer

1. Add failing `tests/engine/test_run_state_growth.py` nodes for canonical seed,
   actual LOOP clone ID, topology digest, collisions/budgets/prunes, atomic prune,
   inherited depth, and changed-but-valid builder rejection.
2. Implement `prepare_expanded` and total splice/prune apply. Replay rebuilds
   from the flat seed and verifies the digest. Do not update `GrowRecord` or any
   checkpoint ledger.
3. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_state_growth.py tests/engine/test_kind_census.py -q
   git add src/agent_composer/runtime/state.py tests/engine/test_run_state_growth.py
   git commit -m "feat(runtime): reduce flat graph expansions"
   ```

## Slice 4.6: Prefix and recovery tests before recovery implementation

1. Add `tests/engine/test_run_recovery_prefixes.py` before writing recovery.
   Parametrize every authoritative prefix for nested CALL/MAP/LOOP under
   `num_workers=0` and `2`; compare graph, edges, wiring, pool, dispositions,
   ready/in-flight order, depth, effective LLM map, and final result.
2. Add exact failure nodes for schema/run/sequence/fingerprint mismatch,
   expansion-digest mismatch, terminal read-only replay, silent historical
   observers, original input, and incomplete starts requeued in dispatch order.
3. Confirm the selected nodes fail because replay/resume does not exist. Keep
   them uncommitted until Slice 4.7 is green.

## Slice 4.7: Runtime and public API cutover (large integration gate)

This is the only runtime lifecycle cutover. All prior record/reducer work is
standalone; there is no checkpoint adapter or period with two authoritative
writers.

1. Before implementation add failing `tests/engine/test_runtime_recorded_dispatch.py`
   nodes for start-before-code, append failure preventing execution, raw outcome
   preparation, serial observer order, bounded N-worker submission, dispatcher
   completion order, submit failure leaving an unfinished start, callback failure
   drain/unlock/re-raise, and late stream discard.
2. Replace `runtime/engine.py` with dispatcher-owned ready selection and the
   final `FlowEngine(state, log, *, grow_builder, ...)` constructor. Workers never
   dequeue ready work and no executor backlog exceeds available capacity.
3. Change `runtime/eval_node.py` to return raw outcome/failure messages and send
   transient stream chunks to the dispatcher only.
4. Change `run.py` atomically to initialize/replay the state and expose only the
   final log APIs. Fresh setup fingerprints before copying, copies before cascade,
   canonicalizes the complete pool, appends `RunStarted`, applies it, and drives.
   Recovery locks/repairs, folds silently, verifies flow/expansions, returns a
   terminal log read-only or appends `RunResumed` and continues.
5. Delete in this same cutover:
   - `src/agent_composer/suspension/`;
   - `runtime/state_manager.py`;
   - `Pause`, pause/abort events, commands, checkpoints, snapshot/restore/resume,
     paused/deferred queues, and the nested expansion ledger;
   - `RunResult.engine`, `checkpoint`, and `pause_reasons`;
   - package-root `FlowEngine` export.
6. Keep `FlowEngine` exported from `runtime/__init__.py`. Update all direct-engine
   tests to construct prepared `RunState` plus `MemoryRunLog`; dynamic tests pass
   `build_grow`. Add `tests/engine/test_flow_engine_public_surface.py` proving:
   - runtime import succeeds and package-root import is absent;
   - static memory-log execution works;
   - dynamic execution requires and uses `grow_builder`;
   - legacy constructor, `snapshot`, `restore`, `resume`, and `request_abort` are
     absent.
7. Rewrite recovery-bearing tests (`test_durable_replay.py`,
   `test_run_resume.py`, `test_child_suspend.py`,
   `test_loop_body_spawner_resume.py`, `test_llm_cascade_resume.py`) against logs.
   Delete checkpoint/continuation-only files only after replacement tests pass.
8. Run progressively, then commit only when the whole cutover is green:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_runtime_recorded_dispatch.py tests/engine/test_flow_engine_public_surface.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_recovery_prefixes.py tests/engine/test_run_state_growth.py tests/engine/test_run_lifecycle.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine tests/cli -q
   git add -A -- src/agent_composer/runtime src/agent_composer/suspension src/agent_composer/run.py src/agent_composer/events.py src/agent_composer/nodes/base.py src/agent_composer/__init__.py tests/engine tests/cli
   git commit -m "refactor(runtime): cut over to log-only recovery"
   ```

## Slice 4.8: API, engine docs, skills, and charters

Update in this PR: `CLAUDE.md` (through its real target, not the `AGENTS.md`
symlink separately), `README.md`, `src/agent_composer/README.md`,
`docs/{index,architecture,engine,typing,api,reference}.md`, root/runtime/run/events
docstrings and charters, `.claude/skills/engine/` including
`templates/node_kind/WIRING.md`, and `.claude/skills/structure/`.

Active docs must say `Output | Route | Grow`, append-only recovery, blocking
workers, at-least-once unfinished-node retry, log sensitivity, and final
`FlowEngine`/`run_flow`/`resume_flow` contracts. Remove every claim that
checkpoint suspension is non-negotiable.

```bash
git grep -n -E 'RunCheckpoint|PauseRequested|RunPaused|RunAborted|AgentEntry|GrowRecord|replay_grow' -- ':!docs/plans/**' ':!site/**'
PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_phase2_cleanup.py tests/engine/test_kind_census.py tests/engine/test_layer_direction.py -q
conda run -n agentcomposer python -m mkdocs build --strict
git add CLAUDE.md README.md src/agent_composer/README.md src/agent_composer/__init__.py src/agent_composer/runtime/__init__.py src/agent_composer/run.py src/agent_composer/events.py docs/index.md docs/architecture.md docs/engine.md docs/typing.md docs/api.md docs/reference.md .claude/skills/engine .claude/skills/structure tests/engine/test_phase2_cleanup.py
git commit -m "docs(engine): document log-only runtime recovery"
```

## Slice 4.9: Durability benchmark and PR gate

Measure both backends and post median time per record to #117/PR:

```bash
PYTHONPATH=src conda run -n agentcomposer python -m timeit -n 100 -r 5 -s 'from agent_composer.runtime.log import MemoryRunLog, NodeStartedRecord; w=MemoryRunLog("bench")' 'w.append(NodeStartedRecord(node_id="n"))'
PYTHONPATH=src conda run -n agentcomposer python -m timeit -n 1 -r 5 -s 'from pathlib import Path; from tempfile import TemporaryDirectory; from agent_composer.runtime.log import JsonlRunLog, NodeStartedRecord' 'd=TemporaryDirectory(); p=Path(d.name)/"r.jsonl"; w=JsonlRunLog.create(p,"bench"); [w.append(NodeStartedRecord(node_id=str(i))) for i in range(100)]; w.close(); d.cleanup()'
PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
conda run -n agentcomposer python -m mkdocs build --strict
git diff --check
```

Open #117 with `Fixes #117`; merge before branching #119.

# PR 5 / #119: CLI Log Paths And Resume Flags

The broker, controller thread, blocking input, and chat routing already exist.
This PR adds only CLI path policy and option validation.

## Issue gate

Claim #119, post this section, and branch from merged #117.

## Slice 5.1: Isolate HOME before default logging

1. Add `tests/cli/conftest.py` with an autouse fixture that sets `HOME` to a
   per-test `tmp_path` before any CLI invocation.
2. Add `tests/cli/test_run_log_options.py::test_cli_tests_use_isolated_home` and
   run it green before enabling default logs.
3. Commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli/test_run_log_options.py::test_cli_tests_use_isolated_home -q
   git add tests/cli/conftest.py tests/cli/test_run_log_options.py
   git commit -m "test(cli): isolate run-log home directories"
   ```

## Slice 5.2: Validate options before side effects

1. Add failing parametrized tests for `--resume` with each forbidden option:
   `--input`, `--inputs`, `--provider`, `--model`, and `--log`. Assert input and
   provider prompts are never reached. Add missing/unreadable resume and existing
   fresh `--log` path cases.
2. Add `log: Optional[Path]`, `resume: Optional[Path]`, and optional CLI worker
   parsing in `cli/run.py`. Validate combinations before loading inputs/provider
   config. Omitted workers mean `0` fresh and recorded value on resume.
3. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli/test_run_log_options.py -q
   git add src/agent_composer/cli/run.py tests/cli/test_run_log_options.py
   git commit -m "feat(cli): validate run-log options"
   ```

## Slice 5.3: Default path, explicit path, and resume E2E

1. Add failing E2E tests for:
   - `ac run FLOW` creating `~/.agent_composer/runs/<run_id>.jsonl` mode `0600`;
   - `--log PATH` using exactly PATH;
   - `--resume PATH` returning terminal logs or continuing unfinished logs;
   - `--resume PATH -w N` overriding only future scheduling;
   - printed log location and no fresh input/provider prompts on resume.
2. Mint the fresh run ID before selecting the default path, create the private
   run directory, and call `run_flow(..., log=...)` or `resume_flow(..., log=...)`.
3. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli/test_run_log_options.py tests/cli/test_run.py tests/engine/test_cli_llm_flags.py -q
   git add src/agent_composer/cli/run.py tests/cli/test_run_log_options.py tests/cli/test_run.py tests/engine/test_cli_llm_flags.py
   git commit -m "feat(cli): persist and resume run logs"
   ```

## Slice 5.4: CLI docs and final verification

1. Update `README.md`, `docs/cli.md`, `docs/api.md`, CLI docstrings, and relevant
   CLI examples with the default path, `--log`, `--resume`, forbidden
   combinations, permissions, sensitive contents, and worker override.
2. Run:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli tests/engine/test_cli_llm_flags.py tests/engine/test_cli_progress.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
   conda run -n agentcomposer python -m mkdocs build --strict
   git diff --check
   git add README.md docs/cli.md docs/api.md src/agent_composer/cli/run.py
   git commit -m "docs(cli): document run-log paths and resume"
   ```

3. Manually smoke-test fresh/default, explicit path, terminal resume, interrupted
   resume, blocking `ask_user`, UTC WAIT, and flow-owned `/exit` at `-w 0` and
   `-w 2`. Record commands/results in #119.
4. Run the required adversarial self-review: corrupt/truncate logs, alter flow and
   builder topology, fail fsync/callback/submit, interrupt all-blocked workers,
   and inspect kind/layer/pool-purity ratchets.
5. Open #119 with `Fixes #119`. After merge, close #111 with links to all five
   merged PRs and leave #110 open for the explicitly deferred cleanup.

## Risks And Out Of Scope

- Blocking human input and WAIT consume workers; if all workers block, queued
  work waits. This is accepted behavior.
- A process crash retries the whole unfinished node and may repeat model, tool,
  code, or human effects.
- Per-record `fsync` may dominate short-node workloads; this plan measures but
  does not weaken durability.
- Logs contain sensitive inputs, prompts, answers, outputs, diagnostics, and
  tracebacks.
- Streaming generators, the always-failing MODEL node, unused failure hooks, the
  old prompt parser, parse-once/source diagnostics, and broad CLI splitting are
  separate #110/#81/#91/#103 work.

## Final Acceptance

- Reusing one `ComposedFlow` never mutates its graph or LLM configuration.
- Runtime kind census remains zero and no `nodes` module imports `compile`.
- Every durable transition is append+flush+fsync before reducer apply.
- Recovery reproduces graph, pool, dispositions, queues, depth, and final result
  after every record prefix in serial and pooled runs.
- Started-without-outcome nodes retry in dispatch order.
- AGENT/HUMAN_INPUT/WAIT block workers through the normal tool/broker path; no
  authorable node returns `Pause`.
- `FlowEngine` is runtime-importable with its new prepared-state contract and is
  absent from the package root.
- `resume_flow` is log-only; suspension/checkpoint/state-manager code is absent.
- CLI default/explicit/resume paths work under an isolated HOME.
- Active docs, skills, examples, templates, and charters match the code.

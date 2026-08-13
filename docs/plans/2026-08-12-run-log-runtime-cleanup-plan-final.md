# Plan: Node-Atomic Runtime With Append-Only Run Logs (Final)

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

## Changes From Revision 2

- Keeps the reviewed merge order `#34 -> #116 -> #118 -> #117 -> #119` and
  aligns the live issue titles, dependencies, and scopes before implementation.
- Uses `git add -f` for ignored review artifacts and verifies the staged list.
- Defines one private controlled-run seam from the CLI main thread through
  `FlowEngine` and `eval_node` to WAIT. Public run APIs expose no cancellation
  token; a node-owned capability trait keeps the runtime kind-blind.
- Defines broker completion, result/exception transfer, interruption, worker
  wake, controller join, and identity-safe registry cleanup as one lifecycle.
- Moves WAIT's red tests into its own slice and requires a green suite at every
  commit boundary, including affected compose tests and seed flows.
- Resolves provider/model environment defaults before recording the effective
  LLM map and proves replay ignores subsequently changed defaults.
- Defines public path types, `RunLog`, writer/lock ownership, direct-engine
  preparation, and close/unlock behavior on every exit path.
- Adds durable `ac chat` paths, repository-wide HOME isolation, blocked-node
  recovery tests, outcome-append failure recovery, and direct-engine mismatch tests.
- Expands same-PR documentation, skills, charters, and cleanup ratchets to cover
  `README.md`, `docs/index.md`, `CONTRIBUTING.md`, and the structure skill.
- Breaks broad migrations into named test-first microsteps. Carrier and runtime
  cutovers still commit atomically where a partial public contract cannot be green.

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
run_flow(flow, inputs, *, log: str | os.PathLike[str] | None = None,
         run_id=None, num_workers=0,
         on_event=None, on_stream=None, llm_config=None) -> RunResult

resume_flow(flow, *, log: str | os.PathLike[str], num_workers=None,
            on_event=None, on_stream=None) -> RunResult
```

`resume_flow` accepts no live engine, checkpoint, command, human answer, fresh
input, new run ID, or new LLM configuration. `log=None` selects an in-memory log
for a fresh library call; recovery always requires a durable path. `RunResult`
adds `run_id: str` and `log: Path | None`; it removes `engine`, `checkpoint`, and
`pause_reasons`.

The CLI-only private `_run_flow_controlled(..., stop_event)` implementation owns
the same behavior as `run_flow`; public `run_flow` delegates to it with an
unexposed event. The CLI controller may pass an event it can signal. This helper
is not exported or documented as a public cancellation API.

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

`FlowEngine.run()` remains an iterator of newly derived observer notifications.
It records terminal state in `state.result`; it does not construct `RunResult` or
close the borrowed log. The tested direct setup is:

```python
from agent_composer.runtime import (
    FlowEngine, MemoryRunLog, RunState, commit_prepared, prepare_run_started,
)

log = MemoryRunLog(run_id)
state = RunState(private_graph)
prepared = prepare_run_started(state, initial_run_data)
commit_prepared(log, state, prepared)  # append first, then total apply
events = list(FlowEngine(
    state, log, grow_builder=build_grow, num_workers=0,
).run())
assert state.result is not None
log.close()
```

The implementation may use a typed `RunStartData` for `initial_run_data`; the
public-surface test fixes its exact import and construction. An unapplied start,
a different run ID, or a log sequence that does not equal the state's applied
sequence is rejected before any node executes.

During #116 the old constructor remains, gains injected `grow_builder`, and all
direct dynamic tests pass it explicitly. During #118 direct runs exercise the
normal tool registry and blocking nodes; AGENT returns to its sanctioned direct
`model_from_config` lookup, so the old engine's `llm=` cap is removed there.
#117 performs the final state/log constructor and export migration atomically.

### Run log ownership

`RunLog` is a small protocol implemented by `MemoryRunLog` and `JsonlRunLog`:
ordered record iteration, `append(record)`, `run_id`, `next_sequence`, `path`,
and idempotent `close()`, plus context-manager support. JSONL append writes one
complete record, flushes, and `fsync`s before returning.

Package-root `run.py` owns path conversion, exclusive create/open, lock, tail
repair, and `close()` in `finally` after structured drain. `FlowEngine` borrows
an already-open log and never closes it. A direct-engine caller owns and closes
the supplied log. Tests cover close/unlock on success, flow failure, append or
callback failure, host interruption, and read-only terminal resume.

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
| 5 | #119 `dev/cli/run-log-options` | #117 merged | Durable run/chat paths and run `--log`/`--resume` |

Every issue must be assigned before work starts. Post the corresponding section
below as an issue comment, cut its branch from the freshly merged dependency,
and open a PR with `Fixes #<issue>`. Do not stack branches from the long-lived
design branch.

## Step 0: Merge The Design And Plan

This is a docs-only PR and must merge before implementation branches begin.

1. Verify #111 and #116-#119 state the reviewed order and ownership: #118 owns
   broker/blocking/chat behavior but retains dormant generic suspension, #117
   owns log cutover plus suspension deletion, and #119 owns CLI paths/options.
   Correct issue bodies before posting implementation comments.
2. Add the design, design reviews, original/revised plans, both plan reviews, and
   final plan. Do not claim an implementation issue is fixed.
3. Run:

   ```bash
   git diff --check
   conda run -n agentcomposer python -m mkdocs build --strict
   ```

4. Force-add the ignored new plan artifacts explicitly, inspect the index, and
   commit only the documentation:

   ```bash
   git add -f \
     docs/plans/2026-08-12-run-log-runtime-cleanup-plan-review-1.md \
     docs/plans/2026-08-12-run-log-runtime-cleanup-plan-v2.md \
     docs/plans/2026-08-12-run-log-runtime-cleanup-plan-review-2.md \
     docs/plans/2026-08-12-run-log-runtime-cleanup-plan-final.md
   git diff --cached --check
   git diff --cached --name-status
   git commit -m "docs: finalize run-log runtime plan"
   ```

5. Open a docs-only PR with `Refs #111`, merge it, update local `main`, and only
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

## Slice 2.2: Atomic Grow migration

The carrier and all producers/consumers form one green commit: no commit may
contain a mixed `Grow.subgraph`/`Grow.seed` contract. Execute the following
microsteps separately, running the named focused check after each, and stage only
after the complete gate is green.

1. Add failing replacement tests before source edits:
   - `tests/engine/test_seed_first_growth.py::test_call_map_loop_and_agent_return_seed_only`
   - `tests/engine/test_seed_first_growth.py::test_compile_builder_reconstructs_every_legacy_growth_kind`
   - `tests/engine/test_flow_engine_grow_builder.py::test_dynamic_direct_engine_requires_injected_builder`
   - `tests/engine/test_flow_engine_grow_builder.py::test_checkpoint_replay_uses_injected_builder`
2. Run those exact nodes and confirm failures are about the old carrier/builder.
3. Change `Grow` in `nodes/base.py` to `seed` plus `prune`, retaining `Pause`;
   run the two carrier test nodes to confirm their assertions now reach producers.
4. Convert CALL, MAP, LOOP, and temporary AGENT continuation producers; run
   `test_call_node.py`, `test_map_node.py`, `test_loop_node.py`, and the first
   seed-first node.
5. Add `compile.expand.build_grow(spawner, canonical_seed)` for the closed legal
   kinds, keeping the temporary AGENT arm only through #116. Run the builder node.
6. Delete producer imports of `compile.expand` and producer `replay_grow` hooks;
   run the import search and node tests.
7. Inject `grow_builder` into `FlowEngine.__init__`, `_apply_grow`, and
   `FlowEngine.restore`; store/replay canonical seeds in `GrowRecord`; inject
   `build_grow` from package-root `run.py`. Run the two direct-engine nodes.
8. Update every direct dynamic engine callsite, including `test_apply_grow.py`,
   `test_expand.py`, `test_graph_growth_e2e.py`, `test_map.py`,
   `test_loop_run.py`, `test_loop_body_spawner_resume.py`,
   `test_durable_replay.py`, `test_engine_expansions_ledger.py`, and AGENT
   continuation tests, to pass `grow_builder=build_grow`.
9. Search for missed consumers before committing:

   ```bash
   grep -R -nE 'Grow\([^)]*subgraph|\.subgraph|replay_grow|FlowEngine\(' src/agent_composer tests/engine --include='*.py'
   ```

10. Run the atomic green gate:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_seed_first_growth.py tests/engine/test_flow_engine_grow_builder.py tests/engine/test_grow_outcome.py tests/engine/test_node_result.py tests/engine/test_call_node.py tests/engine/test_map_node.py tests/engine/test_loop_node.py tests/engine/test_apply_grow.py tests/engine/test_durable_replay.py tests/engine/test_engine_expansions_ledger.py tests/engine/test_agent_continuation.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine -q
   ```

11. Commit the carrier and every producer/consumer together:

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

## Slice 2.4: Effective LLM map without alias or default leakage

1. Add these failing tests to `tests/engine/test_llm_cascade.py`:
   - `test_effective_map_round_trips_static_paths`
   - `test_effective_map_rejects_missing_or_extra_paths`
   - `test_shared_child_is_dealiased_per_parent_provider`
   - `test_effective_map_resolves_environment_defaults_once`
   - `test_replay_uses_logged_map_after_environment_changes`
2. The last test must use two parent paths sharing the same leaf child, one
   inheriting OpenAI and the other Anthropic, and prove both effective providers
   remain distinct after resolution.
3. Have package-root `run.py` read `default_llm_provider()` and
   `default_llm_model()` once and pass that explicit fallback layer into
   `resolve_llm_cascade`. Resolve every agent to concrete provider/model values,
   including `llm_inherit=False`, before returning the effective static-path map.
4. Add `apply_effective_llm_map`, which applies the recorded complete map without
   reading `_settings` or the current environment. **Keep**
   `node.child = copy.deepcopy(node.child)` at each callsite; the root copy does
   not de-alias shared descendants.
5. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_llm_cascade.py tests/engine/test_llm_cascade_run.py tests/engine/test_llm_cascade_resume.py tests/engine/test_run_reuse.py -q
   git add src/agent_composer/compile/llm_cascade.py src/agent_composer/run.py tests/engine/test_llm_cascade.py tests/engine/test_llm_cascade_run.py tests/engine/test_llm_cascade_resume.py
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
   questions, EOF policy, active-prompt cancellation, a controller-complete
   sentinel, and `cancel_all` waking every waiter.
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

1. Add exact failing tests in `tests/cli/test_terminal_controller.py` for:
   prompt-free controller completion; result and exception transfer; one prompt;
   Ctrl-C during an active prompt; controller/node failure with a sibling blocked
   in `ask_user`; main-thread render/input identity; controller join; and registry
   cleanup after every exit.
2. Add private `_run_flow_controlled(..., stop_event)` in package-root `run.py`.
   Public `run_flow` delegates with an internal event and keeps its exact public
   signature. `cli/run.py` alone calls the private seam with an event created by
   the CLI main thread.
3. Thread that same event through `FlowEngine` and `eval_node`. Add the node-owned
   `needs_stop_event = False` trait and pass `caps["stop_event"]` only when the
   trait is true; WAIT enables it in Slice 3.6. This is capability dispatch, not
   a `NodeKind` branch.
4. The main thread creates broker/event, installs its identity-owned `ask_user`,
   starts the controller, and serves until a completion sentinel. The controller
   publishes exactly one result or exception. Every host/controller/node failure
   sets the event and calls `cancel_all`; the main thread joins the controller,
   then removes only its registry entry. Ctrl-C re-raises only after drain/join.
5. Add an internal host-stop exception used only to unwind/drain. Do not convert
   it to `RunAborted`, expose it in `RunResult`, or use `request_abort`.
6. Keep the old pause prompt loop only for dormant artificial-Pause compatibility
   until #117. Do not add a worker-thread input path.
7. Verify the focused tests, then the complete suite, before committing:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli/test_terminal_controller.py tests/cli/test_terminal_broker.py tests/engine/test_cli_progress.py tests/engine/test_parallel.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
   git add src/agent_composer/run.py src/agent_composer/cli/run.py src/agent_composer/cli/terminal.py src/agent_composer/nodes/base.py src/agent_composer/runtime/engine.py src/agent_composer/runtime/eval_node.py tests/cli/test_terminal_controller.py tests/cli/test_terminal_broker.py tests/engine/test_cli_progress.py
   git commit -m "refactor(cli): run the engine behind the terminal broker"
   ```

## Slice 3.4: Collapse AGENT to one local tool loop

1. Add failing `tests/engine/test_blocking_ask_user.py` nodes for answer-as-tool
   message, model-returned call order, explicit authored grant, tool exceptions,
   and absence of `AgentEntry`/`entry`/continuation growth. Add the matching
   failing AGENT/controls deletion assertions to `test_phase2_cleanup.py` now.
2. Remove `Fresh`, `Resume`, `AgentEntry`, `entry`, memo/pending continuation
   state, AGENT spawner traits, `agent_step`, and synthetic segment construction.
   AGENT executes one local model/tool loop and authors human access only through
   `tools: [ask_user]`. Its mode resolves `agent_composer.llm_clients.model_from_config`
   at call time, preserving monkeypatchability while removing the engine-owned
   `llm` cap and constructor argument as the documented AGENT exception.
3. Delete `nodes/agent/controls/` and the AGENT arm from `build_grow`. Remove
   `controls` from `AgentDescriptor` and update `tests/seeds/14-agent-tools.yaml`,
   `tests/seeds/errors/e40-agent-controls-no-range.yaml`, seed expectations, and
   `tests/compose/test_kind_agent_parse.py`.
4. In 2-5 minute passes, update or delete each continuation-only test file after
   its replacement assertion exists: `test_agent_continuation.py`,
   `test_agent_pause_post_assert.py`, `test_agent_segment_builder.py`,
   `test_agent_step.py`, `test_resume_agent.py`, and `test_structured_resume.py`.
5. Run the focused agent/compose/seed suites and then all tests. Commit only green:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_blocking_ask_user.py tests/engine/test_agent.py tests/engine/test_tool_calling_structured.py tests/engine/test_node_parse.py tests/engine/test_errors.py tests/compose/test_kind_agent_parse.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
   git add -A -- src/agent_composer/nodes/agent src/agent_composer/compose/nodes/agent.py src/agent_composer/compile/expand.py tests/engine tests/compose/test_kind_agent_parse.py tests/seeds
   git status --short
   git commit -m "refactor(agent): keep human tool calls in one live loop"
   ```

## Slice 3.5: Block HUMAN_INPUT and make chat routing flow-owned

1. Add failing `tests/engine/test_blocking_human_input.py` nodes for trusted
   metadata and returned `Output`, plus `test_blocking_workers.py` nodes proving
   an independent node completes while one worker blocks and ready work waits
   when every worker is held.
2. Make HUMAN_INPUT call `invoke_tool_trusted("ask_user", ...)` and return
   `Output`. Rewrite its former Pause-oriented tests only after these nodes pass.
3. Rewrite `cli/chat/command.py` to use the installed broker. Change bundled,
   example, and composing-skill chat flows so CASE routes `/exit` and `/quit`
   before reply. Normalize EOF to `/exit`; never intercept typed exit in the host.
4. Update `examples/ask-user.yaml`, `examples/human-approval.yaml`, authoring docs,
   composing skill/reference/templates, and affected agent/human/tool/CLI/chat
   package charters. Retain an accurate note that generic artificial Pause is
   temporarily available until #117.
5. Run each focused file after its smallest source edit, then the examples,
   templates, compose suite, seed suite, and full test suite. Commit only green:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_blocking_human_input.py tests/engine/test_blocking_workers.py tests/cli/test_chat.py tests/cli/test_chat_flow_loads.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/examples/test_examples.py tests/engine/test_templates_load.py tests/engine/test_node_parse.py tests/engine/test_errors.py tests/compose -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
   conda run -n agentcomposer python -m mkdocs build --strict
   git add -A -- src/agent_composer/nodes/human_input src/agent_composer/nodes/agent src/agent_composer/tools src/agent_composer/cli src/agent_composer/compose/nodes examples .claude/skills/composing-agents docs tests/engine tests/cli tests/compose tests/examples tests/seeds
   git status --short
   git commit -m "refactor(engine): make human interaction blocking and flow-owned"
   ```

## Slice 3.6: Blocking UTC WAIT and CODE-owned watchdog

1. Create `tests/engine/test_blocking_wait.py` now, not in an earlier commit.
   Add red nodes for date midnight UTC, naive UTC, aware normalization, past
   immediate return, remaining absolute duration, host-stop wake, observer-failure
   wake, and sibling-node-failure wake. Add WAIT/event deletion ratchets to
   `test_phase2_cleanup.py` and confirm the selected nodes are red.
2. Set `WaitNode.needs_stop_event = True` and accept the injected keyword-only
   capability. Reduce WAIT to absolute date/datetime only. Interpret date and naive datetime
   as UTC, normalize offsets, return immediately for past deadlines, and wait on
   that event for the remaining duration. A set event raises the private host-stop
   signal; its result is discarded during drain. Delete event/watch fields from
   node and compose descriptors.
3. Move `runtime/watchdog.py` to `nodes/code/watchdog.py` and wrap CODE only.
   Update direct imports and logger assertions in
   `tests/engine/test_code_inline.py` from `agent_composer.runtime.watchdog` to
   `agent_composer.nodes.code.watchdog`.
4. Update WAIT syntax/docs, `tests/seeds/17-effects-human-wait.yaml`, seed docs,
   and composing templates. Rewrite old WAIT assertions in
   `tests/compose/test_kind_wait_parse.py`, `test_node_read_hooks.py`,
   `test_build_effects.py`, `test_reserved_wiring_keys.py`, and Pause/resume suites
   only after the replacement file is green.
5. Run the focused files after each small edit, then the full suite. Commit only green:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_blocking_wait.py tests/engine/test_resolve_until.py tests/engine/test_code_inline.py tests/engine/test_node_read_hooks.py tests/engine/test_build_effects.py tests/engine/test_reserved_wiring_keys.py tests/compose/test_kind_wait_parse.py tests/engine/test_templates_load.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
   git add -A -- src/agent_composer/nodes/base.py src/agent_composer/nodes/wait src/agent_composer/nodes/code src/agent_composer/compose/nodes/wait.py src/agent_composer/runtime/watchdog.py src/agent_composer/runtime/eval_node.py tests/engine tests/compose/test_kind_wait_parse.py tests/seeds/17-effects-human-wait.yaml tests/seeds/README.md docs/syntax.md docs/nodes.md .claude/skills/composing-agents
   git status --short
   git commit -m "refactor(nodes): block on UTC wait and scope watchdog to code"
   ```

## Slice 3.7: #118 compatibility and documentation gate

1. Finish the ratchet started before each cutover: AGENT, HUMAN_INPUT, and WAIT
   cannot return `Pause`; `controls:` is rejected; no authorable AGENT constructor
   has `entry`; no event WAIT or runtime-wide watchdog remains.
2. Keep `Pause`, checkpoints, `FlowEngine.snapshot/restore/resume`, and
   `state_manager.py` only as dormant legacy compatibility. Do not add a worker
   prompt fallback.
3. Update `README.md`, `docs/index.md`, all affected authoring/CLI pages,
   `src/agent_composer/README.md`, CLI/nodes/tools/compose package charters,
   `.claude/skills/{composing-agents,engine,structure}` and their references.
   Describe blocking authorable interaction while retaining an explicit temporary
   note about the dormant generic checkpoint API removed by #117.
4. Run:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_blocking_ask_user.py tests/engine/test_blocking_human_input.py tests/engine/test_blocking_wait.py tests/engine/test_blocking_workers.py tests/cli/test_terminal_broker.py tests/cli/test_terminal_controller.py tests/cli/test_chat.py tests/engine/test_kind_census.py tests/engine/test_layer_direction.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
   conda run -n agentcomposer python -m mkdocs build --strict
   git diff --check
   ```

5. Inspect broad staging and commit the ratchet/docs explicitly:

   ```bash
   git add README.md docs/index.md docs/syntax.md docs/nodes.md docs/examples.md docs/cli.md src/agent_composer/README.md src/agent_composer/cli/__init__.py src/agent_composer/cli/chat/__init__.py src/agent_composer/nodes/agent/__init__.py src/agent_composer/nodes/human_input/__init__.py src/agent_composer/nodes/wait/__init__.py src/agent_composer/nodes/code/__init__.py src/agent_composer/tools/__init__.py src/agent_composer/compose/nodes/__init__.py .claude/skills/composing-agents .claude/skills/engine .claude/skills/structure tests/engine/test_phase2_cleanup.py
   git diff --cached --name-status
   git commit -m "test(engine): ratchet blocking interaction surface"
   ```

6. Open #118 with `Fixes #118`, and merge before branching #117.

# PR 4 / #117: Authoritative Run Log And Suspension Deletion

## Issue gate

Claim #117, post this PR section, and branch from merged #118. This PR contains
one breaking cutover. It never adapts checkpoints to logs and never writes a
nested growth ledger beside the log.

## Slice 4.1: Closed records and canonical codecs

1. Add failing `tests/engine/test_run_records.py` cases for every record,
   envelope validation, unknown kinds/fields, tuple/date/FileRef output and seed
   canonicalization, encode-decode-encode stability, and the `RunLog` protocol.
2. Add record dataclasses, `RunRecord`, `RecordEnvelope`, codecs, canonical typed
   value/seed helpers, pool codecs, and the protocol (`run_id`, `path`,
   `next_sequence`, iteration, `append`, idempotent `close`, context manager) to
   new `runtime/log.py`.
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
   line; sequence/schema/run-ID mismatch; append/fsync failure retaining the
   prior complete prefix; idempotent close; and lock release on every exception.
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
   canonical pool/system clock/run ID/START override/effective LLM restoration,
   `commit_prepared` append-before-apply, and state/log run-ID/sequence mismatch.
2. Implement the `RunState` container, immutable `PreparedTransition`,
   `prepare_run_started`, and `commit_prepared` in new `runtime/state.py`; run the
   focused file green and commit. `commit_prepared` is the only shared
   append-then-total-apply helper used by direct callers and the dispatcher.
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
3. Add unfinished `NodeStarted` cases for AGENT, HUMAN_INPUT, and WAIT; prove each
   whole node retries, WAIT recomputes from its absolute deadline after simulated
   downtime, and retries preserve original dispatch order.
4. Add `tests/engine/test_run_lifecycle.py` now with red cases for success/failure/
   terminal-resume close ownership; start-append and outcome-append failures;
   callback and host interruption; and drain/unlock while siblings block in both
   `ask_user` and WAIT. Assert no durable/public abort record or result exists.
5. Confirm the selected nodes fail because replay/resume does not exist. Keep
   them uncommitted until Slice 4.7 is green.

## Slice 4.7: Runtime and public API cutover

This is the only authority switch. Record/reducer code remains standalone until
the final atomic commit; no adapter writes checkpoints and logs for one run. Work
through these 2-5 minute steps and run each named selection before continuing.

1. Add failing `tests/engine/test_runtime_recorded_dispatch.py` nodes for
   start-before-code, start-append failure preventing execution, outcome-append
   failure leaving an unfinished start, raw-outcome preparation, serial observer
   order, bounded N-worker submission, completion-order acceptance, submit
   failure, callback drain/unlock/re-raise, and late stream discard.
2. Change `runtime/eval_node.py` to return raw outcome/failure messages and send
   transient stream chunks to the dispatcher only. Run its focused tests; keep
   the old engine adapter local and temporary until Step 4 so the suite stays green.
3. Implement dispatcher-owned ready selection and bounded submission in
   `runtime/engine.py`; then implement outcome preparation/commit and structured
   drain. Workers never dequeue ready work. After each substep run the matching
   exact nodes from `test_runtime_recorded_dispatch.py`.
4. Switch `FlowEngine` to the final `(state, log, *, grow_builder, ...)`
   constructor, preserving the private stop event from #118. Validate matching
   applied sequence/run ID before execution. Remove the temporary old-engine
   adapter only when direct tests below are green.
5. Change `run.py` to expose only the final typed log APIs. Fresh setup fingerprints
   before copying, copies before cascade, resolves defaults once, canonicalizes
   the complete pool, commits `RunStarted`, and drives. Recovery opens/locks and
   repairs, folds historical records without callbacks or new defaults, verifies
   flow/expansions, returns a terminal log read-only, or commits `RunResumed`.
   `run.py` closes/unlocks in `finally` after drain on every return/exception.
6. Add `tests/engine/test_flow_engine_public_surface.py` using the exact direct
   setup in Chosen Contracts. Prove runtime import, static/dynamic execution,
   required builder, borrowed-log ownership, unprepared/mismatched state rejection,
   a direct AGENT honoring a monkeypatched `model_from_config`, and absence of
   legacy constructor/snapshot/restore/resume/request_abort.
7. Delete in this same authority-switch commit: `suspension/`,
   `runtime/state_manager.py`, `Pause`, pause/abort events and commands,
   checkpoints, snapshot/restore/resume, paused/deferred queues, nested expansion
   ledgers, `RunResult.engine/checkpoint/pause_reasons`, and package-root
   `FlowEngine`. Keep the runtime-package export.
8. Rewrite recovery-bearing tests (`test_durable_replay.py`, `test_run_resume.py`,
   `test_child_suspend.py`, `test_loop_body_spawner_resume.py`, and
   `test_llm_cascade_resume.py`) against logs. Delete a checkpoint-only test only
   after its replacement prefix/lifecycle assertion passes.
9. Run the dispatcher/direct tests, then recovery/lifecycle, then all engine/CLI
   tests. Inspect broad staging and commit only when the entire authority switch is green:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_runtime_recorded_dispatch.py tests/engine/test_flow_engine_public_surface.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_run_recovery_prefixes.py tests/engine/test_run_state_growth.py tests/engine/test_run_lifecycle.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine tests/cli -q
   git add -A -- src/agent_composer/runtime src/agent_composer/suspension src/agent_composer/run.py src/agent_composer/events.py src/agent_composer/nodes/base.py src/agent_composer/__init__.py tests/engine tests/cli
   git status --short
   git diff --cached --name-status
   git commit -m "refactor(runtime): cut over to log-only recovery"
   ```

## Slice 4.8: API, engine docs, skills, and charters

Update in this PR: `CLAUDE.md` (through its real target, not the `AGENTS.md`
symlink separately), `CONTRIBUTING.md`, `README.md`,
`src/agent_composer/README.md`, `docs/{index,architecture,engine,typing,api,reference,cli}.md`,
root/runtime/run/events docstrings and charters, `.claude/skills/engine/`
including `templates/node_kind/WIRING.md`, `.claude/skills/structure/`, and any
remaining suspension wording in `.claude/skills/composing-agents/`.

Active docs must say `Output | Route | Grow`, append-only recovery, blocking
workers, at-least-once unfinished-node retry, log sensitivity, and final
`FlowEngine`/`run_flow`/`resume_flow` contracts. Remove every claim that
checkpoint suspension is non-negotiable.

```bash
if git grep -n -E 'RunCheckpoint|PauseRequested|RunPaused|RunAborted|AgentEntry|GrowRecord|replay_grow' -- src README.md CLAUDE.md CONTRIBUTING.md docs .claude ':!docs/plans/**' ':!site/**'; then
  echo 'stale suspension contract remains' >&2
  exit 1
fi
PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/engine/test_phase2_cleanup.py tests/engine/test_kind_census.py tests/engine/test_layer_direction.py -q
conda run -n agentcomposer python -m mkdocs build --strict
git add CLAUDE.md CONTRIBUTING.md README.md src/agent_composer/README.md src/agent_composer/__init__.py src/agent_composer/runtime/__init__.py src/agent_composer/run.py src/agent_composer/events.py docs/index.md docs/architecture.md docs/engine.md docs/typing.md docs/api.md docs/reference.md docs/cli.md .claude/skills/composing-agents .claude/skills/engine .claude/skills/structure tests/engine/test_phase2_cleanup.py
git diff --cached --name-status
git commit -m "docs(engine): document log-only runtime recovery"
```

## Slice 4.9: Durability benchmark and PR gate

Measure both backends and post median time per record to #117/PR:

```bash
PYTHONPATH=src conda run -n agentcomposer python - <<'PY'
import statistics
import timeit
from pathlib import Path
from tempfile import TemporaryDirectory
from agent_composer.runtime.log import JsonlRunLog, MemoryRunLog, NodeStartedRecord

def memory():
    with MemoryRunLog("bench") as log:
        for i in range(100):
            log.append(NodeStartedRecord(node_id=str(i)))

def jsonl():
    with TemporaryDirectory() as directory:
        with JsonlRunLog.create(Path(directory) / "run.jsonl", "bench") as log:
            for i in range(100):
                log.append(NodeStartedRecord(node_id=str(i)))

for name, fn in (("memory", memory), ("jsonl", jsonl)):
    samples = timeit.repeat(fn, number=1, repeat=5)
    print(name, statistics.median(samples) / 100, "seconds/record")
PY
PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
conda run -n agentcomposer python -m mkdocs build --strict
git diff --check
```

Open #117 with `Fixes #117`; merge before branching #119.

# PR 5 / #119: CLI Log Paths And Resume Flags

The broker, controller thread, blocking input, and chat routing already exist.
This PR adds CLI path policy and run-option validation; it also gives each chat
run or failed-turn restart its required durable path.

## Issue gate

Claim #119, post this section, and branch from merged #117.

## Slice 5.1: Isolate HOME before default logging

1. Add a repository-wide autouse HOME fixture to `tests/conftest.py`, because
   CLI invocations also live under `tests/engine`. It sets `HOME` to a per-test
   `tmp_path` before any command can choose a default log path.
2. Add `tests/cli/test_run_log_options.py::test_cli_tests_use_isolated_home` and
   a check covering an engine-directory CLI test; run both green before enabling
   default logs.
3. Commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli/test_run_log_options.py::test_cli_tests_use_isolated_home -q
   git add tests/conftest.py tests/cli/test_run_log_options.py
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
   - `ac run FLOW` creating a `0700` run directory and
     `~/.agent_composer/runs/<run_id>.jsonl` mode `0600`;
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

## Slice 5.4: Durable chat runs and failed-turn rotation

1. Add failing `tests/cli/test_chat_logs.py` cases proving the initial chat run
   mints/reports a `0600` default log, normal turns remain in that live run, a
   failed turn closes it and restarts with a newly reported log, and Ctrl-C leaves
   the last log nonterminal and recoverable. Preserve flow-owned `/exit`, `/quit`,
   and EOF behavior.
2. Put the shared private default-directory/path function in `cli/utils.py`.
   `cli/run.py` and `cli/chat/command.py` mint `run_id` before choosing the path
   and pass both `run_id` and `log` to `run_flow`. Do not add chat `--resume` or
   reuse a failed run's path.
3. Verify and commit:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli/test_chat_logs.py tests/cli/test_chat.py tests/cli/test_chat_flow_loads.py -q
   git add src/agent_composer/cli/utils.py src/agent_composer/cli/run.py src/agent_composer/cli/chat/command.py tests/cli/test_chat_logs.py tests/cli/test_chat.py
   git commit -m "feat(cli): persist chat runs"
   ```

## Slice 5.5: CLI docs and final verification

1. Update `README.md`, `docs/cli.md`, `docs/api.md`, CLI docstrings, and relevant
   CLI examples with the default path, `--log`, `--resume`, forbidden
   combinations, permissions, sensitive contents, worker override, and chat log
   rotation/reporting.
2. Run:

   ```bash
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests/cli tests/engine/test_cli_llm_flags.py tests/engine/test_cli_load_errors.py tests/engine/test_cli_progress.py -q
   PYTHONPATH=src conda run -n agentcomposer python -m pytest tests -q
   conda run -n agentcomposer python -m mkdocs build --strict
   git diff --check
   git add README.md docs/cli.md docs/api.md src/agent_composer/cli/run.py src/agent_composer/cli/chat/command.py src/agent_composer/cli/utils.py
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
- Fresh runs record concrete environment-resolved provider/model values; replay
  uses them even if process defaults later change.
- Runtime kind census remains zero and no `nodes` module imports `compile`.
- Every durable transition is append+flush+fsync before reducer apply.
- Recovery reproduces graph, pool, dispositions, queues, depth, and final result
  after every record prefix in serial and pooled runs.
- Started-without-outcome nodes, including AGENT/HUMAN_INPUT/WAIT, retry in
  dispatch order; WAIT recomputes its remaining absolute UTC duration.
- AGENT/HUMAN_INPUT/WAIT block workers through the normal tool/broker path; no
  authorable node returns `Pause`.
- `FlowEngine` is runtime-importable with its new prepared-state contract and is
  absent from the package root.
- `resume_flow` is log-only; suspension/checkpoint/state-manager code is absent.
- Every stop path wakes blocked human/WAIT workers, drains, closes/unlocks, and
  exposes no durable or public abort state.
- CLI default/explicit/resume paths and durable chat log rotation work under an
  isolated HOME with `0700` directories and `0600` files.
- Active docs, skills, examples, templates, and charters match the code.

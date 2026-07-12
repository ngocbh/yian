"""`run_flow(loaded, inputs) -> RunResult` — the run entrypoint.

A host-agnostic run boundary that drives a loaded `LoadedFlow` through the
`FlowEngine`. It owns the run-input concerns the pure graph engine doesn't:

- coerce + default the run arguments against the flow's `InputDecl`s,
- seed the `input` namespace + the host-ambient `system` clock, and
- drive the engine to a terminal, mapping engine events to a `RunResult`.

The flow's `asserts:` are NOT enforced here: they are lowered onto the synthesized
boundary nodes at build (PRE/boundary asserts -> `StartNode.boundary_asserts`, fired
inside `StartNode.run()` post-coercion; POST-terminal asserts -> `end.post_asserts`,
fired uniformly through `eval_node`). A false assert therefore comes back as a
`RunFailed` engine event, exactly like a body node's node-local assert — run.py just
forwards it. This unifies top-level and child (subflow) asserts on one firing path.

`RunResult` (the surface-agnostic run outcome) lives here, beside `run_flow`. The
seeding fns (`coerce_inputs`/`apply_defaults`/`seed_system_clock`) come from
`typesys.seeding`; the run's clock is the `${system.today}`/`${system.now}` ambients
(a flow reads "as of" via `Optional[date]` + `:-`/`${system.today}`).

Never raises on a flow failure — a failed/aborted run, or a false assert, comes back as
a `RunResult` with `status != "succeeded"` (RunFailed is an engine EVENT, not an
exception). Compile-time errors (a bad flow) are surfaced by `load_flow`, not here.

This is the top-level RUN BOUNDARY: it sits at the package root, above both `compose`
(it consumes `compose`'s product type, `LoadedFlow`) and `runtime` (it drives the
`FlowEngine`). It is deliberately NOT a `compose` submodule — it touches no YAML; it
turns an already-compiled flow contract into a run. Imports flow DOWN only: `compose`
(LoadedFlow), `runtime` (FlowEngine), `compile`, `typesys` (the pool + seeding),
`expr`, and `events`. Nothing imports back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from agent_composer.compile.llm_cascade import resolve_llm_cascade
from agent_composer.events import (
    RunAborted,
    RunFailed,
    RunPaused,
    RunSucceeded,
    SourceRange,
    SourceSpan,
)
from agent_composer.runtime.engine import FlowEngine
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.seeding import (
    apply_defaults,
    coerce_inputs,
    default_run_id,
    seed_system_clock,
)
from agent_composer.compose.loader import LoadedFlow

if TYPE_CHECKING:
    from agent_composer.suspension.commands import DeliverAnswerCommand

EventHook = Callable[[Any], None]

_STATUS: Dict[type, str] = {
    RunSucceeded: "succeeded",
    RunFailed: "failed",
    RunPaused: "paused",
    RunAborted: "aborted",
}


@dataclass
class RunResult:
    """
    The outcome of one flow run (or resume) — surface-agnostic.

    Returned by [`run_flow`][agent_composer.run_flow] and
    [`resume_flow`][agent_composer.run.resume_flow] for every terminal,
    including failures and pauses; these functions never raise on a flow failure.

    Attributes:
        input (`dict[str, Any]`):
            The coerced run-argument dict (singular, for symmetry across surfaces).
            Empty `{}` on a resume, which carries no fresh run arguments.
        status (`str`):
            The terminal state: one of `"succeeded"`, `"failed"`, `"paused"`, or
            `"aborted"`.
        output (`Any`, *optional*, defaults to `None`):
            The flow's single terminal value — a scalar or a multi-field object.
            Set only when `status == "succeeded"`.
        error (`str`, *optional*, defaults to `None`):
            Human-readable failure detail. Set only when `status == "failed"`
            (including a false post-terminal assert).
        locator (`SourceSpan`, *optional*, defaults to `None`):
            The precise YAML sub-location a flow-level failure originates from — a false
            post-terminal/boundary assert expr, or a boundary input-coercion error. Set only
            when `status == "failed"` AND no `NodeFailed` carries the location (a node-level
            failure puts its locator on the `NodeFailed` event instead). The CLI resolves it
            to a 1-based source line.
        range (`SourceRange`, *optional*, defaults to `None`):
            The concrete 1-based source range a flow-level failure boxes — the resolved twin
            of `locator`, already looked up against the boundary `StartNode`'s per-field
            ranges (the failing `input:` decl line, or the flow-level `asserts:` expr line).
            Set alongside `locator` when the range was resolvable; `None` when the flow was
            parsed spanless. Lets the CLI box the failing line without re-resolving the span.
        traceback (`str`, *optional*, defaults to `None`):
            The formatted Python traceback of the node that aborted the run, when one was
            captured (a code/tool/agent raise). `None` for failures with no exception behind
            them (a boundary or post assert, input coercion). The CLI prints it only under
            `--engine-trace`.
        events (`list[Any]`):
            The raw engine event objects, in emission order. A host serializes them
            however it sees fit.
        checkpoint (`Any`, *optional*, defaults to `None`):
            A serializable `RunCheckpoint` for durable, cross-process resume. Set
            only when `status == "paused"`.
        engine (`FlowEngine`, *optional*, defaults to `None`):
            The live engine for fast in-process resume. Set only when
            `status == "paused"`.
        pause_reasons (`list[Any]`):
            The self-addressing `PauseReason`s to act on before resuming. Non-empty
            only when `status == "paused"`.
    """

    input: Dict[str, Any]
    status: str
    output: Any = None
    error: Optional[str] = None
    locator: Optional[SourceSpan] = None
    range: Optional[SourceRange] = None
    traceback: Optional[str] = None
    events: List[Any] = field(default_factory=list)
    checkpoint: Optional[Any] = None
    engine: Optional[Any] = None
    pause_reasons: List[Any] = field(default_factory=list)


def run_flow(
    loaded: LoadedFlow,
    inputs: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    on_event: Optional[EventHook] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    num_workers: int = 0,
) -> RunResult:
    """
    Coerce inputs, seed the variable pool, enforce asserts, and drive the flow to a terminal.

    Never raises on a flow failure: a failed, paused, or aborted run is returned as a
    `RunResult` with a non-`"succeeded"` status (a `RunFailed` is an engine event, not an
    exception). Compile-time errors of a bad flow surface earlier, in
    [`load_flow`][agent_composer.load_flow]. Asserts run in two phases — boundary asserts
    (`${input}`/`${system}`-only) fire *before* any node runs and return `status="failed"`
    on a false one; post-terminal asserts (`${<id>.output}`) run *after* the flow reaches a
    terminal and flip an otherwise-succeeded run to `"failed"`.

    Args:
        loaded (`LoadedFlow`):
            A compiled, validated flow from [`load_flow`][agent_composer.load_flow].
            Carries the IR, the declared input schema, and the assert sets.
        inputs (`dict[str, Any]`):
            Run arguments keyed by declared input name. Each value is coerced to its
            declared type; names omitted here fall back to their declared defaults.
        run_id (`str`, *optional*, defaults to `None`):
            Host-injected run id, readable in the flow as `${system.run_id}`. When
            `None`, a fresh id is minted per run.
        on_event (`Callable[[Any], None]`, *optional*, defaults to `None`):
            Called with each engine event as it occurs (`NodeStarted`, `RunSucceeded`,
            `RunPaused`, `RunFailed`, `RunAborted`). Use it for progress reporting.
        llm_config (`dict[str, Any]`, *optional*, defaults to `None`):
            Outermost cascade layer (CLI `--provider`/`--model`); fills only the gaps a
            flow leaves unset, then `model_from_config` applies env/global defaults.
        num_workers (`int`, *optional*, defaults to `0`):
            Engine drive mode. `0` is the single-threaded inline drain (deterministic
            event order); `>=1` spawns a worker pool of that size so independent ready
            nodes (a fan-out) run concurrently. The result is worker-count-independent.

    Returns:
        `RunResult`:
            The run outcome. `status` is one of `"succeeded"`, `"failed"`, `"paused"`,
            or `"aborted"`; `output` is set on success, and `pause_reasons` plus the
            resume handles (`engine`, `checkpoint`) are set on a pause.

    Example:
        ```python
        from agent_composer import load_flow, run_flow

        loaded = load_flow(open("hello.yaml").read(), search_paths=["."])
        result = run_flow(loaded, {"name": "Ada"})
        print(result.status, result.output)  # succeeded ...
        ```
    """
    coerced = apply_defaults(
        loaded.input, coerce_inputs(loaded.input, inputs)
    )

    # Resolve the per-agent effective llm_config before the engine reads the graph: the CLI
    # layer (llm_config) is the outermost gap-fill layer of the cascade. See resolve_llm_cascade.
    resolve_llm_cascade(loaded.compiled, llm_config or {})

    pool = VariablePool()
    seed_system_clock(pool)  # ${system.today}/${system.now} — once per run
    # ${system.run_id} — host-injected, else a freshly minted id; child-inherited like the clock.
    pool.add_system("run_id", run_id if run_id is not None else default_run_id())

    # The engine seeds store[START_ID] at run init (StartNode.run -> coerce/type-check/defaults +
    # FIRE the flow's boundary asserts post-coercion), then advances START_ID. The flow's
    # post-terminal asserts ride the END node (`end.post_asserts`, lowered onto it at build) and
    # fire uniformly through eval_node — so run.py enforces NO asserts itself; the input
    # TypeCheckError + a false boundary/post assert all come back as a RunFailed engine event.
    engine = FlowEngine(
        loaded.compiled, pool,
        num_workers=num_workers,
        run_inputs=coerced,
    )
    events: List[Any] = []
    status = "incomplete"
    output: Any = None
    error: Optional[str] = None
    result_locator: Optional[SourceSpan] = None  # flow-level location (no NodeFailed behind it)
    result_range: Optional[SourceRange] = None  # resolved twin of result_locator (concrete range)
    tb: Optional[str] = None  # Python traceback of the aborting node, if one was captured

    for event in engine.run():
        if on_event is not None:
            on_event(event)
        events.append(event)
        terminal = _STATUS.get(type(event))
        if terminal is not None:
            status = terminal
            if isinstance(event, RunSucceeded):
                output = event.output
            elif isinstance(event, RunFailed):
                error = event.error
                result_locator = event.locator  # boundary assert / input-coercion error
                result_range = event.range  # concrete range the engine already resolved
                tb = event.traceback

    # A paused run carries the resume handles: the live engine (fast in-process),
    # a serializable checkpoint (durable), and the pause reasons.
    # `resume_flow` consumes these.
    if status == "paused":
        paused = [e for e in events if isinstance(e, RunPaused)]
        return RunResult(
            input=coerced,
            status="paused",
            events=events,
            engine=engine,
            checkpoint=engine.snapshot(),
            pause_reasons=paused[0].reasons if paused else [],
        )

    # Post-terminal asserts ride the END node (`end.post_asserts`) and fire through eval_node,
    # so a false one already comes back as a RunFailed event above — no second phase here.
    return RunResult(
        input=coerced, status=status, output=output, error=error,
        locator=result_locator, range=result_range, traceback=tb, events=events,
    )


def resume_flow(
    loaded: LoadedFlow,
    *,
    engine: Optional[FlowEngine] = None,
    checkpoint: Any = None,
    commands: Optional[List[Any]] = None,
    on_event: Optional[EventHook] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    num_workers: Optional[int] = None,
) -> RunResult:
    """
    Drive a suspended run to its next terminal via exactly one resume handle.

    Resume from a live `engine=` (fast, in-process) or a deserialized `checkpoint=`
    (durable, cross-process) — pass exactly one. The `commands` (typically the injected
    human answer or a wait release, built with
    [`resume_command`][agent_composer.run.resume_command]) are applied before the
    run continues. Mirrors [`run_flow`][agent_composer.run_flow]'s terminal-event loop and
    post-terminal asserts; if the run pauses again, the returned result carries fresh handles.

    Args:
        loaded (`LoadedFlow`):
            The same compiled flow the suspended run was started from. Supplies the IR
            (to restore from a checkpoint) and the post-terminal assert set.
        engine (`FlowEngine`, *optional*, defaults to `None`):
            A live engine from a paused `RunResult.engine`. Mutually exclusive with
            `checkpoint`.
        checkpoint (`Any`, *optional*, defaults to `None`):
            A serializable snapshot from a paused `RunResult.checkpoint`, possibly
            deserialized in another process. Mutually exclusive with `engine`.
        commands (`list[Any]`, *optional*, defaults to `None`):
            Engine commands to apply before resuming (e.g. delivered answers, wait
            releases). `None` is treated as an empty list.
        on_event (`Callable[[Any], None]`, *optional*, defaults to `None`):
            Called with each engine event as it occurs. Use it for progress reporting.
        llm_config (`dict[str, Any]`, *optional*, defaults to `None`):
            Outermost cascade layer (CLI `--provider`/`--model`); fills only the gaps a flow
            leaves unset. On a durable resume the CLI config is NOT persisted across
            processes, so the host must re-supply it; it is re-applied to the recompiled flow
            before `restore`.
        num_workers (`int`, *optional*, defaults to `None`):
            Override the drive mode for a DURABLE (`checkpoint=`) resume; `None` keeps the
            checkpointed count. Ignored with a live `engine=` (the live engine keeps its own
            mode).

    Returns:
        `RunResult`:
            The next terminal outcome, with `input={}` (a resume carries no fresh run
            arguments). On a re-pause it carries new `engine`/`checkpoint`/`pause_reasons`.

    Raises:
        `ValueError`:
            If both or neither of `engine` and `checkpoint` are provided.
    """
    if (engine is None) == (checkpoint is None):
        raise ValueError("resume_flow requires exactly one of engine= or checkpoint=")
    if engine is None:
        # Resolve the cascade BEFORE restore: restore's replay re-clones each CALL/MAP child
        # from the static graph, so the effective config must be baked on first (the CLI layer
        # is not persisted across processes — the host re-supplies it here). In-process resume
        # via engine= already carries the resolved configs baked on the live graph.
        resolve_llm_cascade(loaded.compiled, llm_config or {})
        engine = FlowEngine.restore(loaded.compiled, checkpoint, num_workers=num_workers)

    events: List[Any] = []
    status, output, error = "incomplete", None, None
    result_locator: Optional[SourceSpan] = None  # flow-level location (no NodeFailed behind it)
    result_range: Optional[SourceRange] = None  # resolved twin of result_locator (concrete range)
    tb: Optional[str] = None  # Python traceback of the aborting node, if one was captured
    for event in engine.resume(commands or []):
        if on_event is not None:
            on_event(event)
        events.append(event)
        terminal = _STATUS.get(type(event))
        if terminal is not None:
            status = terminal
            if isinstance(event, RunSucceeded):
                output = event.output
            elif isinstance(event, RunFailed):
                error = event.error
                result_locator = event.locator
                result_range = event.range
                tb = event.traceback

    if status == "paused":
        paused = [e for e in events if isinstance(e, RunPaused)]
        return RunResult(
            input={},
            status="paused",
            events=events,
            engine=engine,
            checkpoint=engine.snapshot(),
            pause_reasons=paused[0].reasons if paused else [],
        )

    # Post-terminal asserts ride the END node and fire through eval_node (see run_flow) — a
    # false one already surfaces as a RunFailed event above, so no second phase on resume either.
    return RunResult(
        input={}, status=status, output=output, error=error,
        locator=result_locator, range=result_range, traceback=tb, events=events,
    )


def resume_command(loaded: LoadedFlow, reason: Any, value: Any) -> "DeliverAnswerCommand":
    """
    Map a host-resumable pause reason and an answer to the engine command that delivers it.

    Dispatch is on the reason's *shape*, never a static-graph lookup: the parked node may be
    a runtime-namespaced live id (e.g. `call_sub/w`, `agent/__ask#q1`) that exists only on
    the live engine, where the command is resolved. A reason carrying a `node_id` delivers
    the answer as that leaf's output; a reason without one (a bare external event a watcher
    satisfies) is not host-resumable.

    Args:
        loaded (`LoadedFlow`):
            The compiled flow the paused run belongs to. (Reserved for symmetry and future
            validation; the command is built from `reason` and `value`.)
        reason (`Any`):
            A `PauseReason` from `RunResult.pause_reasons`. Must expose a `node_id` to be
            host-resumable.
        value (`Any`):
            The answer to deliver. For a `human_input` it is the typed answer; for a `wait`
            release pass `None`.

    Returns:
        `DeliverAnswerCommand`:
            The command to pass to
            [`resume_flow`][agent_composer.run.resume_flow] via `commands=`.

    Raises:
        `ValueError`:
            If `reason` has no `node_id` (a bare external event that a watcher, not the
            host, must satisfy).
    """
    from agent_composer.suspension.commands import DeliverAnswerCommand

    node_id = getattr(reason, "node_id", None)
    if node_id is None:
        raise ValueError(f"pause reason {getattr(reason, 'type', '?')!r} is not host-resumable")
    return DeliverAnswerCommand(node_id=node_id, value=value)   # HUMAN_INPUT + WAIT (release: value=None)

"""`run_flow(loaded, inputs) -> RunResult` — the run entrypoint.

A host-agnostic run boundary that drives a loaded `ComposedFlow` through the
`FlowEngine`. It owns the run-input concerns the pure graph engine doesn't:

- coerce + default the run arguments against the flow's `InputDecl`s,
- seed the `input` namespace + the host-ambient `system` clock, and
- drive the engine to a terminal, mapping engine events to a `RunResult`.

Authorable AGENT ``ask_user``, HUMAN_INPUT, and WAIT calls remain inside live node
execution and return ordinary outputs. The ``paused`` result shape and resume helpers
remain only as dormant custom/artificial generic-Pause compatibility.

The flow's `asserts:` are NOT enforced here: they are lowered onto the synthesized
boundary nodes at build (PRE/boundary asserts -> `StartNode.post_asserts` with each
`${input.X}` rewritten to `${output.X}`, fired against START's COERCED output through
`eval_node`; POST-terminal asserts -> `end.post_asserts`, fired uniformly through
`eval_node`). A false assert therefore comes back as a
`RunFailed` engine event, exactly like a body node's node-local assert — run.py just
forwards it. This unifies top-level and child (subflow) asserts on one firing path.

`RunResult` (the surface-agnostic run outcome) lives here, beside `run_flow`. The
seeding fns (`coerce_inputs`/`seed_system_clock`) come from
`typesys.seeding`; the run's clock is the `${system.today}`/`${system.now}` ambients
(a flow reads "as of" via `Optional[date]` + `:-`/`${system.today}`).

Never raises on a flow failure — a failed/aborted run, or a false assert, comes back as
a `RunResult` with `status != "succeeded"` (RunFailed is an engine EVENT, not an
exception). Compile-time errors (a bad flow) are surfaced by `load_flow`, not here.

This is the top-level RUN BOUNDARY: it sits at the package root, above both `compose`
(it consumes `compose`'s product type, `ComposedFlow`) and `runtime` (it drives the
`FlowEngine`). It is deliberately NOT a `compose` submodule — it touches no YAML; it
turns an already-compiled flow contract into a run. Imports flow DOWN only: `compose`
(ComposedFlow), `runtime` (FlowEngine), `compile`, `typesys` (the pool + seeding),
`expr`, and `events`. Nothing imports back.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from agent_composer._settings import default_llm_model, default_llm_provider
from agent_composer.compile.expand import build_grow
from agent_composer.compile.llm_cascade import (
    apply_effective_llm_map,
    resolve_llm_cascade,
)
from agent_composer.compile.manifest import flow_fingerprint
from agent_composer.events import (
    RunAborted,
    RunFailed,
    RunPaused,
    RunSucceeded,
    SourceRange,
    SourceSpan,
)
from agent_composer.runtime.engine import FlowEngine
from agent_composer.runtime.log import (
    RunLog,
    RunStartedRecord,
    decode_record,
    encode_record,
)
from agent_composer.runtime.state import RunState, _replay_records
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.seeding import (
    coerce_inputs,
    default_run_id,
    seed_system_clock,
)
from agent_composer.compose.loader import ComposedFlow

if TYPE_CHECKING:
    from agent_composer.suspension.commands import DeliverAnswerCommand

EventHook = Callable[[Any], None]

_STATUS: Dict[type, str] = {
    RunSucceeded: "succeeded",
    RunFailed: "failed",
    RunPaused: "paused",
    RunAborted: "aborted",
}


def _replay_run_log(loaded: ComposedFlow, log: RunLog) -> RunState:
    """Reconstruct one borrowed authoritative log without changing public APIs.

    The source log is already open, locked, and tail-repaired by its owner.  This
    seam snapshots a stable canonical prefix, verifies the authored program before
    copying it, applies only the recorded effective LLM map, and delegates every
    historical transition to the runtime's live reducer path.  It neither appends
    to nor closes ``log``.

    Args:
        loaded (`ComposedFlow`):
            Fresh authored flow whose fingerprint must match the first record.
        log (`RunLog`):
            Borrowed non-empty historical authority for exactly one run.

    Returns:
        `RunState`:
            Private reconstructed state at the exact end of the historical prefix.
    """
    if not isinstance(loaded, ComposedFlow):
        raise TypeError("_replay_run_log expects a ComposedFlow")
    source_run_id = log.run_id
    source_sequence = log.next_sequence
    if type(source_run_id) is not str or not source_run_id:
        raise ValueError("run log run_id must be a non-empty string")
    if type(source_sequence) is not int or source_sequence < 0:
        raise ValueError("run log next_sequence must be a non-negative integer")

    supplied = tuple(log)
    if log.run_id != source_run_id or log.next_sequence != source_sequence:
        raise ValueError("run log changed while its historical prefix was read")
    if source_sequence != len(supplied):
        raise ValueError(
            "run log sequence mismatch: "
            f"next_sequence is {source_sequence}, prefix length is {len(supplied)}"
        )
    if not supplied:
        raise ValueError("cannot replay an empty run log")

    canonical = []
    for expected_sequence, supplied_envelope in enumerate(supplied):
        encoded = encode_record(supplied_envelope)
        envelope = decode_record(encoded)
        if envelope != supplied_envelope:
            raise ValueError(
                f"run log envelope {expected_sequence} is not in canonical wire form"
            )
        if envelope.seq != expected_sequence:
            raise ValueError(
                "run log sequence mismatch: "
                f"expected {expected_sequence}, found {envelope.seq}"
            )
        if envelope.run_id != source_run_id:
            raise ValueError(
                "run log run_id mismatch: "
                f"expected {source_run_id!r}, found {envelope.run_id!r}"
            )
        canonical.append(envelope)

    first = canonical[0]
    if not isinstance(first.record, RunStartedRecord):
        raise ValueError("the first run log record must be RunStarted")
    actual_fingerprint = flow_fingerprint(loaded.compiled)
    recorded_fingerprint = first.record.flow_fingerprint
    if actual_fingerprint != recorded_fingerprint:
        raise ValueError(
            "flow fingerprint mismatch: "
            f"recorded {recorded_fingerprint!r}, supplied {actual_fingerprint!r}"
        )

    private_flow = copy.deepcopy(loaded.compiled)
    apply_effective_llm_map(
        private_flow,
        first.record.effective_llm_by_static_path,
    )
    state = _replay_records(
        RunState(private_flow),
        tuple(canonical),
        grow_builder=build_grow,
    )
    if log.run_id != source_run_id or log.next_sequence != source_sequence:
        raise ValueError("run log changed while its historical prefix was replayed")
    return state


@dataclass
class RunResult:
    """
    The outcome of one flow run (or resume) — surface-agnostic.

    Returned by [`run_flow`][agent_composer.run_flow] and
    [`resume_flow`][agent_composer.run.resume_flow] for every terminal,
    including failures and pauses; these functions never raise on a flow failure.

    Attributes:
        input (`dict[str, Any]`):
            The caller's run arguments COERCED to declared scalar types (singular, for symmetry
            across surfaces) — but NOT defaulted: an omitted input is absent here, since
            omitted-input defaulting lives in START's wrapper, not the run boundary (so this
            surface reflects what the caller supplied). Empty `{}` on a resume, which carries no
            fresh run arguments.
        status (`str`):
            The terminal state: one of `"succeeded"`, `"failed"`, `"paused"`, or
            `"aborted"`. Authorable nodes do not produce `"paused"`; it is dormant
            custom/artificial compatibility.
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
    loaded: ComposedFlow,
    inputs: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    on_event: Optional[EventHook] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    num_workers: int = 0,
) -> RunResult:
    """
    Coerce inputs, seed the variable pool, and drive the flow to a terminal.

    Never raises on a flow failure: a failed, paused, or aborted run is returned as a
    `RunResult` with a non-`"succeeded"` status (a `RunFailed` is an engine event, not an
    exception). Compile-time errors of a bad flow surface earlier, in
    [`load_flow`][agent_composer.load_flow]. Asserts are NOT enforced here: a flow's
    `asserts:` are lowered at build onto the synthesized boundary nodes — boundary asserts
    (`${input}`-only) ride `StartNode.post_asserts` (rewritten to read `${output.X}`) and fire
    against START's coerced output through `eval_node` when START runs (a false one returns
    `status="failed"` before any body node); post-terminal
    asserts (`${<id>.output}`) ride `end.post_asserts` and fire through `eval_node` when END
    runs (flipping an otherwise-succeeded run to `"failed"`). This unifies top-level and child
    (subflow) asserts on one firing path.

    Args:
        loaded (`ComposedFlow`):
            A compiled, validated flow from [`load_flow`][agent_composer.load_flow].
            Carries the IR (asserts lowered onto its boundary nodes) and the declared
            input schema.
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
            flow leaves unset. The run boundary reads the process provider/model defaults
            once each and bakes a complete effective config onto every agent before execution.
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
    return _run_flow_controlled(
        loaded,
        inputs,
        run_id=run_id,
        on_event=on_event,
        llm_config=llm_config,
        num_workers=num_workers,
        stop_event=threading.Event(),
    )


def _run_flow_controlled(
    loaded: ComposedFlow,
    inputs: Dict[str, Any],
    *,
    run_id: Optional[str] = None,
    on_event: Optional[EventHook] = None,
    llm_config: Optional[Dict[str, Any]] = None,
    num_workers: int = 0,
    stop_event,
) -> RunResult:
    """Drive a fresh run with a host-owned cooperative stop event.

    Private composition seam for the CLI controller. Public `run_flow` deliberately creates
    its own event and retains its established signature.
    """
    # Coerce the caller-passed args to declared scalar types (int/float/bool) but do NOT
    # fill declared defaults here — omitted-input defaulting lives solely in START (its
    # `_run_impl` runs apply_defaults; pre-Step-8 eval_node drives `_run_impl` directly), so
    # the boundary has exactly one defaulting authority.
    coerced = coerce_inputs(loaded.input, inputs)

    # The loaded graph is a reusable program. Cascade resolution and recursive-node expansion
    # both mutate their graph, so every fresh run receives its own live copy.
    flow = copy.deepcopy(loaded.compiled)
    # Snapshot ambient defaults exactly once each. Every agent receives concrete provider/model
    # values before the engine reads the graph, so later model construction cannot observe an
    # environment change. The CLI layer is the outermost authored gap-fill layer.
    llm_defaults = {
        "provider": default_llm_provider().lower(),
        "model": default_llm_model(),
    }
    resolve_llm_cascade(flow, llm_config or {}, defaults=llm_defaults)

    pool = VariablePool()
    seed_system_clock(pool)  # ${system.today}/${system.now} — once per run
    # ${system.run_id} — host-injected, else a freshly minted id; child-inherited like the clock.
    pool.add_system("run_id", run_id if run_id is not None else default_run_id())

    # The engine schedules START_ID through eval_node at run init: START coerces/type-checks/
    # fills defaults inside its `_run_impl`, and the flow's boundary (`${input}`) asserts — lowered
    # onto START's `post_asserts` (rewritten to read `${output.X}`) — FIRE through eval_node's
    # ordinary post-assert seam against START's coerced output. The flow's post-terminal asserts
    # ride the END node (`end.post_asserts`, lowered onto it at build) and fire the same way — so
    # run.py enforces NO asserts itself; the input TypeCheckError + a false boundary/post assert
    # all come back as a RunFailed engine event.
    engine = FlowEngine(
        flow, pool,
        num_workers=num_workers,
        run_inputs=coerced,
        grow_builder=build_grow,
        stop_event=stop_event,
    )
    events: List[Any] = []
    status = "incomplete"
    output: Any = None
    error: Optional[str] = None
    result_locator: Optional[SourceSpan] = None  # flow-level location (no NodeFailed behind it)
    result_range: Optional[SourceRange] = None  # resolved twin of result_locator (concrete range)
    tb: Optional[str] = None  # Python traceback of the aborting node, if one was captured

    event_stream = engine.run()
    for event in event_stream:
        if on_event is not None:
            try:
                on_event(event)
            except BaseException:
                # Observer/controller failure is the initiating cause. Wake cooperative nodes,
                # then explicitly close the suspended engine generator so its pooled `finally`
                # block fully joins workers. Cleanup must not replace the original exception.
                stop_event.set()
                try:
                    event_stream.close()
                except BaseException:
                    pass
                raise
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
    loaded: ComposedFlow,
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

    Resume a custom/artificial compatibility pause from a live `engine=` (fast,
    in-process) or a deserialized `checkpoint=` (durable, cross-process) — pass
    exactly one. The `commands` (temporary generic-pause values built with
    [`resume_command`][agent_composer.run.resume_command]) are applied before the
    run continues. Mirrors [`run_flow`][agent_composer.run_flow]'s terminal-event loop and
    post-terminal asserts; if the run pauses again, the returned result carries fresh handles.

    Args:
        loaded (`ComposedFlow`):
            The same compiled flow the suspended run was started from. Supplies the IR
            (to restore from a checkpoint) and the post-terminal assert set.
        engine (`FlowEngine`, *optional*, defaults to `None`):
            A live engine from a paused `RunResult.engine`. Mutually exclusive with
            `checkpoint`.
        checkpoint (`Any`, *optional*, defaults to `None`):
            A serializable snapshot from a paused `RunResult.checkpoint`, possibly
            deserialized in another process. Mutually exclusive with `engine`.
        commands (`list[Any]`, *optional*, defaults to `None`):
            Engine commands to apply before resuming (delivered compatibility values for
            artificial parked leaves). `None` is treated as an empty list.
        on_event (`Callable[[Any], None]`, *optional*, defaults to `None`):
            Called with each engine event as it occurs. Use it for progress reporting.
        llm_config (`dict[str, Any]`, *optional*, defaults to `None`):
            Outermost cascade layer (CLI `--provider`/`--model`); fills only the gaps a flow
            leaves unset. On a durable resume the CLI config is NOT persisted across
            processes, so the host must re-supply it; it is re-applied with one snapshot of
            the process provider/model defaults before `restore`. A live resume retains its
            already-resolved configs and does not read defaults again.
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
        # Durable restore replays graph growth, so it needs a private clean graph just like a
        # fresh run. A live engine already owns its graph and must keep using that instance.
        flow = copy.deepcopy(loaded.compiled)
        # Resolve the cascade BEFORE restore: restore's replay re-clones each CALL/MAP/LOOP child
        # from the static graph, so the effective config must be baked on first. Snapshot the
        # ambient defaults once each for this durable initialization. In-process resume via
        # engine= already carries the resolved configs and does not read settings again.
        llm_defaults = {
            "provider": default_llm_provider().lower(),
            "model": default_llm_model(),
        }
        resolve_llm_cascade(flow, llm_config or {}, defaults=llm_defaults)
        engine = FlowEngine.restore(
            flow,
            checkpoint,
            num_workers=num_workers,
            grow_builder=build_grow,
        )

    events: List[Any] = []
    status, output, error = "incomplete", None, None
    result_locator: Optional[SourceSpan] = None  # flow-level location (no NodeFailed behind it)
    result_range: Optional[SourceRange] = None  # resolved twin of result_locator (concrete range)
    tb: Optional[str] = None  # Python traceback of the aborting node, if one was captured
    event_stream = engine.resume(commands or [])
    for event in event_stream:
        if on_event is not None:
            try:
                on_event(event)
            except BaseException:
                # Match the fresh-run boundary: observer failure is primary, but cooperative
                # resumed workers must be woken before closing the generator performs full joins.
                engine.stop_event.set()
                try:
                    event_stream.close()
                except BaseException:
                    pass
                raise
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


def resume_command(loaded: ComposedFlow, reason: Any, value: Any) -> "DeliverAnswerCommand":
    """
    Map a host-resumable pause reason and a compatibility value to its delivery command.

    Dispatch is on the reason's *shape*, never a static-graph lookup: the parked node may be
    a runtime-namespaced live id (e.g. `call_sub/w`) that exists only on
    the live engine, where the command is resolved. A reason carrying a `node_id` delivers
    the value as that leaf's output; a reason without one (a bare external event a watcher
    satisfies) is not host-resumable.

    Args:
        loaded (`ComposedFlow`):
            The compiled flow the paused run belongs to. (Reserved for symmetry and future
            validation; the command is built from `reason` and `value`.)
        reason (`Any`):
            A `PauseReason` from `RunResult.pause_reasons`. Must expose a `node_id` to be
            host-resumable.
        value (`Any`):
            The compatibility value to deliver. A synthetic scheduled release
            conventionally uses `None`; authorable WAIT never enters this path.

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
    return DeliverAnswerCommand(node_id=node_id, value=value)

"""Package-root run boundary over one authoritative append-only log.

``run_flow`` initializes a private copy of a loaded flow, commits its complete
canonical start state, and drives only post-durable observer notifications.
``resume_flow`` reconstructs that state from a locked JSONL prefix and either
returns an existing terminal result without changing the log or commits one
``RunResumed`` transition before continuing.

The runtime borrows an already-open log.  This module owns log creation,
recovery, locking, tail repair, and closure after the engine's structured drain.
It also owns the compile/runtime inversion seam by injecting
``compile.expand.build_grow`` into the kind-blind engine and replay reducer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
from typing import Any, Callable

from agent_composer._settings import default_llm_model, default_llm_provider
from agent_composer.compile.expand import build_grow
from agent_composer.compile.llm_cascade import (
    apply_effective_llm_map,
    resolve_llm_cascade,
)
from agent_composer.compile.manifest import flow_fingerprint
from agent_composer.compose.loader import ComposedFlow
from agent_composer.diagnostics import Failure
from agent_composer.events import (
    RunFailed,
    RunSucceeded,
    SourceRange,
    SourceSpan,
    StreamChunk,
)
from agent_composer.nodes.binding import coerce_inputs
from agent_composer.runtime.engine import FlowEngine
from agent_composer.runtime.log import (
    JsonlRunLog,
    MemoryRunLog,
    RunLog,
    RunStartedRecord,
    decode_record,
    encode_record,
)
from agent_composer.runtime.state import (
    RunStartData,
    RunState,
    _replay_records,
    commit_prepared,
    prepare_run_resumed,
    prepare_run_started,
)
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.seeding import default_run_id, seed_system_clock


EventHook = Callable[[Any], None]
StreamHook = Callable[[StreamChunk], None]


@dataclass
class RunResult:
    """The terminal, surface-independent result of a fresh or recovered run.

    ``events`` contains only observers committed during this invocation.  A
    terminal recovery is therefore silent and returns an empty event list while
    preserving the original canonical input, output, and failure diagnostics.

    Attributes:
        input (`dict[str, Any]`):
            Canonical caller-supplied inputs retained by ``RunStarted``.
        status (`str`):
            Either ``"succeeded"`` or ``"failed"``.
        run_id (`str`):
            Durable identity shared by the state and its run log.
        log (`Path | None`):
            Durable JSONL path, or ``None`` for a process-local fresh run.
        output (`Any | None`):
            Terminal flow value on success.
        failure (`Failure | None`):
            Canonical terminal failure on failure; otherwise `None`.
        locator (`SourceSpan | None`):
            Symbolic source location for a failed flow.
        range (`SourceRange | None`):
            Resolved source range for a failed flow.
        events (`list[Any]`):
            Post-durable observers from this fresh or resume segment only.
    """

    input: dict[str, Any]
    status: str
    run_id: str
    log: Path | None
    output: Any = None
    failure: Failure | None = None
    locator: SourceSpan | None = None
    range: SourceRange | None = None
    events: list[Any] = field(default_factory=list)


def run_flow(
    flow: ComposedFlow,
    inputs: dict[str, Any],
    *,
    log: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    num_workers: int = 0,
    on_event: EventHook | None = None,
    on_stream: StreamHook | None = None,
    llm_config: dict[str, Any] | None = None,
) -> RunResult:
    """Start and drive one flow through an in-memory or durable run log.

    Flow failures are returned as ``RunResult(status="failed")``.  Host,
    callback, log, and reducer errors propagate after any active workers have
    been drained and the boundary-owned log has been closed.

    Args:
        flow (`ComposedFlow`):
            Compiled, validated flow to copy and execute.
        inputs (`dict[str, Any]`):
            Caller values keyed by declared input name. Supplied scalars are
            coerced into the initial pool; START applies declared defaults and
            full boundary type checks during its recorded node execution.
        log (`str | os.PathLike[str] | None`, *optional*, defaults to `None`):
            New JSONL authority path. ``None`` uses a process-local
            ``MemoryRunLog``; an existing path is never overwritten.
        run_id (`str | None`, *optional*, defaults to `None`):
            Non-empty identity exposed as ``system.run_id`` and retained by every
            record. ``None`` mints a fresh identity.
        num_workers (`int`, *optional*, defaults to `0`):
            Non-negative worker bound recorded at initialization. Zero selects
            serial evaluation.
        on_event (`Callable[[Any], None] | None`, *optional*, defaults to `None`):
            Callback for observers after their records have been appended and
            reduced.
        on_stream (`Callable[[StreamChunk], None] | None`, *optional*, defaults to `None`):
            Callback for transient stream chunks, which are never durable records.
        llm_config (`dict[str, Any] | None`, *optional*, defaults to `None`):
            Per-run outer model configuration layer. The fully resolved static map
            is captured in the initialization record.

    Returns:
        `RunResult`:
            Terminal success or failure, including this invocation's post-durable
            observers and the durable path when one was requested.

    Raises:
        `TypeError`:
            A public argument has the wrong type.
        `ValueError`:
            Inputs, identity, worker count, model configuration, or prepared run
            state violates its contract.
        `FileExistsError`:
            ``log`` names an existing path.
        `OSError`:
            Durable log creation, locking, writing, syncing, or closure fails.
    """
    return _run_flow_controlled(
        flow,
        inputs,
        log=log,
        run_id=run_id,
        num_workers=num_workers,
        on_event=on_event,
        on_stream=on_stream,
        llm_config=llm_config,
        stop_event=threading.Event(),
    )


def _run_flow_controlled(
    flow: ComposedFlow,
    inputs: dict[str, Any],
    *,
    log: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    num_workers: int = 0,
    on_event: EventHook | None = None,
    on_stream: StreamHook | None = None,
    llm_config: dict[str, Any] | None = None,
    stop_event: threading.Event,
) -> RunResult:
    """Drive a fresh run with one host-owned cooperative stop event."""
    if not isinstance(flow, ComposedFlow):
        raise TypeError("run_flow expects a ComposedFlow")

    # Identity belongs to the authored program, before cascade resolution or any
    # per-run copy can change object topology.
    authored_fingerprint = flow_fingerprint(flow.compiled)
    private_flow = copy.deepcopy(flow.compiled)

    # Ambient defaults are read exactly once each and become part of RunStarted.
    defaults = {
        "provider": default_llm_provider().lower(),
        "model": default_llm_model(),
    }
    effective_llm = resolve_llm_cascade(
        private_flow,
        llm_config or {},
        defaults=defaults,
    )

    canonical_run_id = run_id if run_id is not None else default_run_id()
    # Canonicalize supplied boundary strings before RunStarted persists its input.
    # The generic binder reuses the same idempotent policy and remains the authority
    # for required/default handling, contract validation, and every child boundary.
    coerced = coerce_inputs(flow.input, inputs)
    pool = VariablePool(start_id=private_flow.start_id)
    pool.set(private_flow.start_id, copy.deepcopy(coerced))
    seed_system_clock(pool)
    pool.add_system("run_id", canonical_run_id)
    start_wiring_override = {
        name: f"${{input.{name}}}" for name in coerced
    }

    state = RunState(private_flow)
    prepared_start = prepare_run_started(
        state,
        RunStartData(
            run_id=canonical_run_id,
            flow_fingerprint=authored_fingerprint,
            initial_pool=pool,
            start_wiring_override=start_wiring_override,
            effective_llm_by_static_path=effective_llm,
            num_workers=num_workers,
        ),
    )

    # Preparation precedes durable-file creation, so invalid initialization
    # cannot leave an empty path behind.
    authority: RunLog = (
        MemoryRunLog(canonical_run_id)
        if log is None
        else JsonlRunLog.create(log, canonical_run_id)
    )
    primary_error: BaseException | None = None
    try:
        events: list[Any] = []
        start_observers = commit_prepared(authority, state, prepared_start)
        _publish_boundary_observers(
            start_observers,
            events,
            on_event=on_event,
            stop_event=stop_event,
        )

        engine = FlowEngine(
            state,
            authority,
            grow_builder=build_grow,
            num_workers=num_workers,
            on_event=on_event,
            on_stream=on_stream,
            stop_event=stop_event,
        )
        events.extend(engine.run())
        return _result_from_state(state, authority.path, events)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _close_owned_log(authority, primary_error)


def resume_flow(
    flow: ComposedFlow,
    *,
    log: str | os.PathLike[str],
    num_workers: int | None = None,
    on_event: EventHook | None = None,
    on_stream: StreamHook | None = None,
) -> RunResult:
    """Recover one durable run and drive it only when it remains nonterminal.

    Historical records are folded silently.  Recovery never reads ambient model
    defaults, mints a new identity, or accepts fresh input/configuration.

    Args:
        flow (`ComposedFlow`):
            Compiled flow whose authored fingerprint must match the log's initial
            record. Recovery runs on a private copy.
        log (`str | os.PathLike[str]`):
            Existing JSONL run log to lock, repair to its final complete line,
            validate, and replay.
        num_workers (`int | None`, *optional*, defaults to `None`):
            Non-negative worker bound for a new recovery segment. ``None`` preserves
            the most recently recorded value.
        on_event (`Callable[[Any], None] | None`, *optional*, defaults to `None`):
            Callback for observers committed during this invocation only; historical
            replay and terminal recovery are silent.
        on_stream (`Callable[[StreamChunk], None] | None`, *optional*, defaults to `None`):
            Callback for new transient stream chunks, which are never replayed.

    Returns:
        `RunResult`:
            Reconstructed terminal result, or the terminal result after retrying
            ordered unfinished starts and continuing the frontier.

    Raises:
        `TypeError`:
            A public argument or decoded carrier has the wrong type.
        `ValueError`:
            The prefix is invalid, the flow fingerprint differs, or the selected
            worker count violates its contract.
        `FileNotFoundError`:
            ``log`` does not exist.
        `OSError`:
            Locking, tail repair, appending, syncing, or closure fails.
    """
    return _resume_flow_controlled(
        flow,
        log=log,
        num_workers=num_workers,
        on_event=on_event,
        on_stream=on_stream,
        stop_event=threading.Event(),
    )


def _resume_flow_controlled(
    flow: ComposedFlow,
    *,
    log: str | os.PathLike[str],
    num_workers: int | None = None,
    on_event: EventHook | None = None,
    on_stream: StreamHook | None = None,
    stop_event: threading.Event,
) -> RunResult:
    """Recover and drive a durable run with a host-owned stop event."""
    authority = JsonlRunLog.resume(log)
    primary_error: BaseException | None = None
    try:
        state = _replay_run_log(flow, authority)
        if state.result is not None:
            return _result_from_state(state, authority.path, [])

        selected_workers = state.num_workers if num_workers is None else num_workers
        prepared_resume = prepare_run_resumed(
            state,
            num_workers=selected_workers,
        )
        events: list[Any] = []
        resume_observers = commit_prepared(authority, state, prepared_resume)
        _publish_boundary_observers(
            resume_observers,
            events,
            on_event=on_event,
            stop_event=stop_event,
        )

        engine = FlowEngine(
            state,
            authority,
            grow_builder=build_grow,
            num_workers=selected_workers,
            on_event=on_event,
            on_stream=on_stream,
            stop_event=stop_event,
        )
        events.extend(engine.run())
        return _result_from_state(state, authority.path, events)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _close_owned_log(authority, primary_error)


def _replay_run_log(flow: ComposedFlow, log: RunLog) -> RunState:
    """Silently reconstruct one borrowed, locked authoritative log prefix.

    The authored fingerprint is checked before copying.  The private graph gets
    only the complete effective LLM map recorded by ``RunStarted``; recovery
    never consults current process defaults.  Historical transitions use the
    same reducer and growth constructor as live execution.
    """
    if not isinstance(flow, ComposedFlow):
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
        envelope = decode_record(encode_record(supplied_envelope))
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
    actual_fingerprint = flow_fingerprint(flow.compiled)
    recorded_fingerprint = first.record.flow_fingerprint
    if actual_fingerprint != recorded_fingerprint:
        raise ValueError(
            "flow fingerprint mismatch: "
            f"recorded {recorded_fingerprint!r}, supplied {actual_fingerprint!r}"
        )

    private_flow = copy.deepcopy(flow.compiled)
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


def _publish_boundary_observers(
    observers: tuple[Any, ...],
    events: list[Any],
    *,
    on_event: EventHook | None,
    stop_event: threading.Event,
) -> None:
    """Publish an already-applied segment lead before retaining it in the result."""
    for observer in observers:
        if on_event is not None:
            try:
                on_event(observer)
            except BaseException as exc:
                _set_stop_preserving(stop_event, exc)
                raise
        events.append(observer)


def _result_from_state(
    state: RunState,
    path: Path | None,
    events: list[Any],
) -> RunResult:
    """Project one terminal reducer state into the exact public result shape."""
    terminal = state.result
    if isinstance(terminal, RunSucceeded):
        return RunResult(
            input=copy.deepcopy(state.input),
            status="succeeded",
            run_id=_required_state_run_id(state),
            log=path,
            output=copy.deepcopy(terminal.output),
            events=events,
        )
    if isinstance(terminal, RunFailed):
        return RunResult(
            input=copy.deepcopy(state.input),
            status="failed",
            run_id=_required_state_run_id(state),
            log=path,
            failure=terminal.failure,
            locator=terminal.locator,
            range=terminal.range,
            events=events,
        )
    raise RuntimeError("engine returned without deriving a terminal run result")


def _required_state_run_id(state: RunState) -> str:
    """Return the initialized non-empty run identity used by ``RunResult``."""
    if type(state.run_id) is not str or not state.run_id:
        raise ValueError("terminal run state has no valid run_id")
    return state.run_id


def _set_stop_preserving(
    stop_event: threading.Event,
    primary_error: BaseException,
) -> None:
    """Signal cooperative stop without replacing an initiating callback error."""
    try:
        stop_event.set()
    except BaseException as cleanup_error:
        _note_cleanup_error(primary_error, "stop signaling", cleanup_error)


def _close_owned_log(
    log: RunLog,
    primary_error: BaseException | None,
) -> None:
    """Close one boundary-owned log while preserving an initiating exception."""
    try:
        log.close()
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        _note_cleanup_error(primary_error, "run-log close", cleanup_error)


def _note_cleanup_error(
    primary_error: BaseException,
    operation: str,
    cleanup_error: BaseException,
) -> None:
    """Attach cleanup diagnostics when supported without changing primary control."""
    try:
        primary_error.add_note(
            f"{operation} also raised "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
    except BaseException:
        pass


__all__ = ["RunResult", "resume_flow", "run_flow"]

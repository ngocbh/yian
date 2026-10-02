"""Package-root run boundary over one authoritative append-only log.

``run_flow`` initializes a private copy of a loaded flow, commits its complete
canonical start state, and drives only post-durable observer notifications.
``resume_flow`` reconstructs that state from a locked JSONL prefix and either
returns a terminal result by inspection, or commits one ``RunResumed`` transition
before primary execution. Retained results accept explicit backward attempts
after forward execution has returned and released its opened resources.

The runtime borrows an already-open log.  This module owns log creation,
recovery, locking, tail repair, and closure after the engine's structured drain.
It also owns the compile/runtime inversion seam by injecting
``compile.expand.build_grow`` into the kind-blind engine and replay reducer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping

from langchain_core.tools import BaseTool
from pydantic import SecretStr

from agent_composer.llm_clients.config import LLMConfig, pop_api_key
from agent_composer.llm_clients.connection import connection_snapshot, legacy_connection_map
from agent_composer.compile.execution_layout import apply_execution_layout, resolve_execution_layout
from agent_composer.compile.expand import build_grow
from agent_composer.compile.llm_cascade import (
    apply_effective_llm_map,
    bind_api_key,
    host_key_providers,
    resolve_llm_cascade,
)
from agent_composer.compile.manifest import flow_fingerprint
from agent_composer.compile.memory_layout import (
    apply_memory_layout,
    check_session_compatibility,
    memory_stacks,
    resolve_memory_layout,
)
from agent_composer.compile.model import CompiledFlow
from agent_composer.compose.loader import ComposedFlow
from agent_composer.diagnostics import Failure, OriginRef
from agent_composer.events import (
    RunFailed,
    RunSucceeded,
    StreamChunk,
)
from agent_composer.expr.lineage import FeedbackValue
from agent_composer.memory import (
    LearningResult,
    Memory,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemorySetupError,
    MemorySpec,
    MemoryStack,
)
from agent_composer.nodes.base import Steering
from agent_composer.nodes.binding import coerce_inputs
from agent_composer.runtime.engine import FlowEngine
from agent_composer.runtime.eval_node import require_steering
from agent_composer.runtime.backward import drive_backward
from agent_composer.runtime.contracts import BackwardOutcome
from agent_composer.runtime.log import (
    JsonlRunLog,
    MemoryRunLog,
    RecordEnvelope,
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
from agent_composer.runtime.memory_runtime import MemoryRuntime
from agent_composer.skills import Skill
from agent_composer.tools import ToolSet
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.seeding import default_run_id, seed_system_clock


EventHook = Callable[[Any], None]
StreamHook = Callable[[StreamChunk], None]


@dataclass
class RunResult:
    """The terminal, surface-independent result of a fresh or recovered run.

    ``events`` contains only observers committed during this invocation.  A
    terminal recovery is therefore silent and returns an empty event list while
    preserving the original canonical input, output, and semantic failure.

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
        events (`list[Any]`):
            Post-durable observers from this fresh or resume segment only.
        learning (`LearningResult | None`):
            Learning projection of the last observed backward attempt, initially
            ``None``. It never changes the forward result.
        backward_outcome (`BackwardOutcome | None`):
            Last observed accepted backward disposition, initially ``None``.
    """

    input: dict[str, Any]
    status: str
    run_id: str
    log: Path | None
    output: Any = None
    failure: Failure | None = None
    events: list[Any] = field(default_factory=list)
    learning: LearningResult | None = None
    backward_outcome: BackwardOutcome | None = None
    _backward_handle: _BackwardHandle | None = field(default=None, init=False, repr=False, compare=False)

    def backward(
        self, feedback: FeedbackValue = None, *, api_key: str | SecretStr | None = None,
    ) -> BackwardOutcome:
        """Run one new explicit backward attempt over retained accepted execution.

        Each call appends a new attempt, whatever earlier attempts did. An
        unfinished earlier attempt, for example from a killed process, is first
        recorded as interrupted.

        Args:
            feedback (`FeedbackValue`, *optional*, defaults to `None`): Ordinary
                text or structured feedback selecting the accepted flow output.
            api_key (`str | SecretStr | None`, *optional*, defaults to `None`):
                The provider key of a run started with ``llm_config.api_key``,
                bound to the same agents as in `run_flow` and never recorded.
                ``None`` uses the key this result was created with, if any.
                A run that marked no agent ignores it. Without a key, marked
                agents fail their backward call instead of reading the
                provider's key environment variable.

        Returns:
            `BackwardOutcome`: The accepted attempt result, separate from forward
                status, output, failure, and events. Learning is its projection.

        Raises:
            `TypeError`: Feedback has an invalid ordinary-message carrier.
            `ValueError`: Retention is disabled or feedback is invalid.
            `MemorySetupError`: Enabled policies cannot restore their recorded
                snapshots. Host and log faults also propagate while preserving
                the accepted prefix.
            `RuntimeError`: Another backward call on this result is in progress,
                or its accepted prefix could not be retained after an earlier call.

        Examples:
            `result.backward("Keep the explanation shorter.")` returns a separate
            outcome for a run created with `learnable=True`.
        """
        handle = self._backward_handle
        if handle is None:
            raise ValueError("backward requires a run retained with learnable=True")
        if not handle._guard.acquire(blocking=False):
            raise RuntimeError("a backward call on this result is already in progress")
        try:
            try:
                return handle.backward(
                    feedback, api_key=None if api_key is None else _secret_key(api_key)
                )
            finally:
                self.backward_outcome = handle.outcome
                self.learning = None if self.backward_outcome is None else self.backward_outcome.learning
        finally:
            handle._guard.release()


@dataclass
class _BackwardHandle:
    """Private original program and closed append authority for delayed feedback."""

    program: CompiledFlow
    run_id: str
    path: Path | None
    records: tuple[RecordEnvelope, ...]
    memory: dict[str, Memory] | None
    outcome: BackwardOutcome | None
    # Held in memory for this result's own backward calls; never recorded.
    api_key: SecretStr | None = field(default=None, repr=False)
    _guard: Any = field(default_factory=threading.Lock, init=False, repr=False, compare=False)
    _unusable: bool = field(default=False, init=False, repr=False, compare=False)

    def backward(self, feedback: FeedbackValue, *, api_key: SecretStr | None = None) -> BackwardOutcome:
        """Reconstruct accepted facts and delegate the complete attempt driver."""
        if self._unusable:
            raise RuntimeError("backward handle is unavailable after an accepted-prefix refresh failure")
        authority = MemoryRunLog._from_records(self.records) if self.path is None else JsonlRunLog.resume(self.path)
        state = None
        replay_sequence = None
        primary_error = None
        try:
            state = _replay_compiled_run_log(
                self.program, authority, api_key=self.api_key if api_key is None else api_key
            )
            if state.run_id != self.run_id:
                raise ValueError("backward log identity differs from the retained run")
            replay_sequence = state.next_sequence

            def open_memory() -> MemoryRuntime:
                if self.memory is None:
                    raise MemorySetupError("backward requires this run's enabled memory mapping")
                program = copy.deepcopy(self.program)
                apply_effective_llm_map(program, legacy_connection_map(state.effective_llm_by_static_path))
                layout, origins, memories, stacks = _resolve_runtime_memory_layout(program)
                if layout != state.memory_layout:
                    raise ValueError("recorded memory layout does not match the retained program")
                _check_recorded_memory_table(
                    state.memories, state.memory_stacks, memories, stacks,
                    subject="retained program",
                )
                return MemoryRuntime.open(layout=state.memory_layout, origins=origins,
                                          memory=self.memory, snapshots=state.memory_snapshots,
                                          memories=state.memories)

            return drive_backward(state, authority, feedback, open_memory=open_memory)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                if replay_sequence is not None:
                    try:
                        self._refresh(authority, state, replay_sequence)
                    except BaseException as refresh_error:
                        self._unusable = True
                        if primary_error is None:
                            primary_error = refresh_error
                            raise
                        _note_cleanup_error(primary_error, "backward prefix refresh", refresh_error)
            finally:
                _close_owned_log(authority, primary_error)

    def _refresh(self, authority: RunLog, state: RunState, replay_sequence: int) -> None:
        """Retain accepted facts, replaying an append that outpaced local reduction."""
        sequence = authority.next_sequence
        if type(sequence) is not int or sequence < state.next_sequence:
            raise ValueError("backward log sequence regressed during the call")
        # Merely reading another handle's work must preserve this result's
        # existing observation when its own submission accepted no new record.
        if sequence == replay_sequence:
            return
        if state.next_sequence > replay_sequence:
            self.outcome = state.backward_state.outcome
        records = tuple(authority)
        if authority.next_sequence != sequence or len(records) != sequence:
            raise ValueError("backward log changed during prefix refresh")
        if self.path is None:
            self.records = records
        if sequence > state.next_sequence:
            accepted = _replay_compiled_run_log(self.program, authority)
            if accepted.run_id != self.run_id:
                raise ValueError("backward log identity differs from the retained run")
            self.outcome = accepted.backward_state.outcome


def run_flow(
    flow: ComposedFlow,
    inputs: dict[str, Any],
    *,
    log: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    num_workers: int = 8,
    on_event: EventHook | None = None,
    on_stream: StreamHook | None = None,
    llm_config: dict[str, Any] | None = None,
    memory: dict[str, Memory] | None = None,
    learnable: bool = False,
    tools: Mapping[str, BaseTool | Callable[..., Any] | Sequence[BaseTool | Callable[..., Any]]] | None = None,
    skills: Mapping[str, Skill] | None = None,
    steering: Steering | None = None,
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
        num_workers (`int`, *optional*, defaults to `8`):
            Non-negative worker bound recorded at initialization. Zero selects
            serial evaluation.
        on_event (`Callable[[Any], None] | None`, *optional*, defaults to `None`):
            Callback for observers after their records have been appended and
            reduced, plus transient in-loop AGENT observers (`IN_LOOP_EVENTS`)
            delivered during the node; the latter are never recorded or replayed.
        on_stream (`Callable[[StreamChunk], None] | None`, *optional*, defaults to `None`):
            Callback for transient stream chunks, which are never durable records.
            When set, AGENT chat replies stream their visible text.
        llm_config (`dict[str, Any] | None`, *optional*, defaults to `None`):
            Per-run outer model configuration layer. The fully resolved static map
            is captured in the initialization record. An ``api_key`` entry (a
            `str` or `SecretStr`) is the provider key for this run and needs
            ``provider``. Every static AGENT whose effective provider matches
            uses it in place of the provider's key environment variable or
            saved OpenAI connection; agents of other providers are unaffected.
            An ``inherit: false`` agent of that provider uses it too. The key is
            never recorded; the log marks which agents use it, so pass it again
            to `resume_flow` or `RunResult.backward`, where those agents
            otherwise fail rather than read the environment.
        memory (`dict[str, Memory] | None`, *optional*, defaults to `None`):
            Host-injected policy implementations keyed by authored policy name.
            ``None`` leaves every authored policy dormant. A mapping enables
            recall and optional conversation history, and records exact per-policy
            recovery snapshots. Session history advances during forward execution.
        learnable (`bool`, *optional*, defaults to `False`):
            Retain accepted execution for a later explicit `RunResult.backward`
            call. Forward execution runs no frame learners or learned-mutation
            commits. Memory is optional; an enabled mapping supplies owned learners
            for that call. Automatic session history is independent of retention.
        tools (`Mapping[str, BaseTool | Callable[..., Any] | Sequence[BaseTool | Callable[..., Any]]] | None`, *optional*, defaults to `None`):
            The tools this run sees, keyed by tool id. `None` uses the tools
            registered with `register_tool`. A mapping, even an empty one, is the
            whole set: registered tools are hidden. To keep them, pass
            `{**registered_tools(), ...}`. That copy is taken when you call it, so
            register every tool before the run starts. A function or method
            becomes a tool named by its key and described by its docstring; bind
            values with a closure, not `functools.partial`. A `BaseTool` must use
            its key as its name. The `ask_user` entry is the host handler for
            HUMAN_INPUT nodes and granted AGENT `ask_user` calls. It must be a
            plain callable that accepts `question`, `questions`, `answer_schema`,
            `title`, `node_id`, and `options` as keywords (define it as
            `def handler(**request)`); the model sees only `question` and
            `options`. Runs use
            up to eight workers by default, so several nodes can call it at once;
            it must be thread-safe. An AGENT can call a tool only if its `tools:`
            list grants the id. A list or tuple value is a tool group: an AGENT
            that grants the group's key can call every tool in it. Each item is a
            `BaseTool` named by its `name`, or a function or method named by its
            `__name__`. Each tool name calls one tool: the same tool may sit in
            several groups, but two different tools may not share a name, and
            no group tool may be named `ask_user` or `load_skill`. Every static AGENT grant must name a tool or
            group in this set; this is checked before the log is created.
            Granting skills also binds the built-in ``load_skill`` tool,
            which is not part of this mapping.
        skills (`Mapping[str, Skill] | None`, *optional*, defaults to `None`):
            Skills available to this run, keyed by `Skill.name`; build one with
            `load_skills`. An AGENT lists its granted skills' names and
            descriptions in its system message and reads them on demand with the
            built-in ``load_skill`` tool. ``skills: all`` grants every supplied
            skill. A flow that grants skills needs this mapping (``{}`` grants
            none). Pass the same mapping to `resume_flow`.
        steering (`Steering | None`, *optional*, defaults to `None`):
            The host inbox for AGENT nodes authored ``steerable: true``; other
            nodes never call it. At the top of each tool-loop iteration the
            node calls ``drain(node_id)`` and sends the returned messages to
            the model as user messages, recorded with that model request in
            session memory. When ``interrupted(node_id)`` turns true, the node
            stops its loop at the next safe point and succeeds with the text
            it has so far. Calls come from worker threads. Pass the same inbox
            to `resume_flow`.

    Returns:
        `RunResult`:
            Terminal success or failure, including this invocation's post-durable
            observers and the durable path when one was requested.

    Raises:
        `TypeError`:
            A public argument has the wrong type, including a malformed `tools`
            entry or group item, or a `steering` object without
            `drain`/`interrupted`.
        `ValueError`:
            Inputs, identity, worker count, model configuration, or prepared run
            state violates its contract, ``llm_config.api_key`` lacks
            ``llm_config.provider``, a `BaseTool` in `tools` is named
            differently from its key, a group tool's name clashes or is
            reserved, an AGENT grants a tool or group not in the run's tools,
            or an AGENT grants skills while `skills` is `None` or lacks a
            listed skill.
        `MemorySetupError`:
            Memory is enabled without a required policy implementation, cannot
            open an exact run snapshot, or has an invalid configuration.
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
        memory=memory,
        learnable=learnable,
        tools=tools,
        skills=skills,
        steering=steering,
        stop_event=threading.Event(),
    )


def _secret_key(raw: str | SecretStr) -> SecretStr:
    """Validate a host key into a `SecretStr` without echoing it in errors."""
    return LLMConfig(api_key=raw).api_key


def _run_flow_controlled(
    flow: ComposedFlow,
    inputs: dict[str, Any],
    *,
    log: str | os.PathLike[str] | None = None,
    run_id: str | None = None,
    num_workers: int = 8,
    on_event: EventHook | None = None,
    on_stream: StreamHook | None = None,
    llm_config: dict[str, Any] | None = None,
    memory: dict[str, Memory] | None = None,
    learnable: bool = False,
    tools: Mapping[str, BaseTool | Callable[..., Any] | Sequence[BaseTool | Callable[..., Any]]] | None = None,
    skills: Mapping[str, Skill] | None = None,
    steering: Steering | None = None,
    stop_event: threading.Event,
) -> RunResult:
    """Drive a fresh run with one host-owned cooperative stop event."""
    if not isinstance(flow, ComposedFlow):
        raise TypeError("run_flow expects a ComposedFlow")
    if type(learnable) is not bool:
        raise TypeError("learnable must be a bool")
    run_tools = ToolSet(tools)
    _check_tool_grants(flow.compiled, run_tools)
    skills = _check_skills(flow.compiled, skills)
    require_steering(steering)

    # Identity belongs to the authored program, before cascade resolution or any
    # per-run copy can change object topology.
    authored_fingerprint = flow_fingerprint(flow.compiled)
    private_flow = copy.deepcopy(flow.compiled)
    retained_program = copy.deepcopy(private_flow) if learnable else None

    # The key stays out of the cascade so RunStarted never records it.
    run_llm_config = dict(llm_config or {})
    api_key = pop_api_key(run_llm_config)

    # One selection snapshot determines every static agent before RunStarted.
    snapshot = connection_snapshot()
    effective_llm = snapshot.bind(
        resolve_llm_cascade(
            private_flow,
            run_llm_config,
            defaults=snapshot.defaults,
            provider_models=snapshot.provider_models,
        ),
        host_key_provider=None if api_key is None else run_llm_config["provider"],
    )
    apply_effective_llm_map(private_flow, effective_llm)
    # A key that marks no agent has nothing to authenticate, so the result keeps none.
    if api_key is not None and not host_key_providers(effective_llm):
        api_key = None
    if api_key is not None:
        bind_api_key(private_flow, api_key)

    if learnable:
        apply_execution_layout(private_flow, resolve_execution_layout(private_flow))

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

    memory_runtime: MemoryRuntime | None = None
    memory_table: tuple[MemorySpec, ...] | None = None
    memory_stack_set: tuple[MemoryStack, ...] | None = None
    if memory is not None:
        memory_layout, memory_origins, memory_table, memory_stack_set = (
            _resolve_runtime_memory_layout(private_flow, effective_llm=effective_llm)
        )
        memory_runtime = MemoryRuntime.open(
            layout=memory_layout,
            origins=memory_origins,
            memory=memory,
            memories=memory_table,
        )

    authority: RunLog | None = None
    primary_error: BaseException | None = None
    try:
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
                memory_layout=(
                    None if memory_runtime is None else memory_runtime.layout
                ),
                memory_snapshots=(
                    None if memory_runtime is None else memory_runtime.snapshots
                ),
                learnable=learnable,
                memories=memory_table,
                memory_stacks=memory_stack_set,
            ),
        )

        # Preparation precedes durable-file creation, so invalid initialization
        # cannot leave an empty path behind.
        log_path = None if log is None else Path(log)
        retained_locator = None if log_path is None else log_path.absolute()
        authority = (
            MemoryRunLog(canonical_run_id)
            if log_path is None
            else JsonlRunLog.create(retained_locator, canonical_run_id)
        )
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
            memory_context_opener=(
                None if memory_runtime is None else memory_runtime.open_context
            ),
            tools=run_tools,
            skills=skills,
            steering=steering,
        )
        events.extend(engine.run())
        return _result_from_state(state, log_path, events,
                                  handle=_retained_handle(retained_program, state, authority, memory,
                                                          locator=retained_locator, api_key=api_key))
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _close_run_resources(memory_runtime, authority, primary_error)


def resume_flow(
    flow: ComposedFlow,
    *,
    log: str | os.PathLike[str],
    num_workers: int | None = None,
    on_event: EventHook | None = None,
    on_stream: StreamHook | None = None,
    memory: dict[str, Memory] | None = None,
    tools: Mapping[str, BaseTool | Callable[..., Any] | Sequence[BaseTool | Callable[..., Any]]] | None = None,
    skills: Mapping[str, Skill] | None = None,
    steering: Steering | None = None,
    api_key: str | SecretStr | None = None,
) -> RunResult:
    """Recover primary execution or inspect a terminal run without backward work.

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
            replay and terminal recovery are silent. Re-executed AGENT nodes publish
            fresh in-loop observers.
        on_stream (`Callable[[StreamChunk], None] | None`, *optional*, defaults to `None`):
            Callback for new transient stream chunks, which are never replayed.
        memory (`dict[str, Memory] | None`, *optional*, defaults to `None`):
            Host-injected policy implementations required when the recorded run
            enabled memory and still has unfinished forward execution. Terminal
            inspection opens no policies; supplied references remain available
            for later explicit backward calls.
        tools (`Mapping[str, BaseTool | Callable[..., Any] | Sequence[BaseTool | Callable[..., Any]]] | None`, *optional*, defaults to `None`):
            The tools the resumed segment sees, with the same rules as
            `run_flow`. Tools are not recorded in the log, so pass the same
            mapping (including `ask_user`) that the original run used. Group
            contents may differ: nodes that restart see the groups passed here.
        skills (`Mapping[str, Skill] | None`, *optional*, defaults to `None`):
            The skills passed to the original `run_flow`. Nodes that restart see
            this mapping. An unfinished run of a flow that grants skills needs
            it; terminal inspection does not check it.
        steering (`Steering | None`, *optional*, defaults to `None`):
            The host inbox for steerable AGENT nodes, as in `run_flow`. Finished
            nodes are replayed from the log and never call it.
        api_key (`str | SecretStr | None`, *optional*, defaults to `None`):
            The provider key of a run started with ``llm_config.api_key``. It
            goes to the agents the log marks for it, as in `run_flow`, is never
            recorded, and stays with the result for `RunResult.backward`. A
            log that marks no agent ignores it. Without it, marked agents fail
            when they call their model instead of reading the provider's key
            environment variable.

    Returns:
        `RunResult`:
            Reconstructed terminal result, or the terminal result after retrying
            ordered unfinished starts and continuing the frontier.

    Raises:
        `TypeError`:
            A public argument or decoded carrier has the wrong type, including a
            malformed `tools` entry or group item.
        `ValueError`:
            The prefix is invalid, the flow fingerprint differs, the selected
            worker count violates its contract, a group tool's name clashes or
            is reserved, an unfinished run's AGENT grants a tool or group
            not in `tools` or grants skills while `skills` is `None` or lacks
            a listed skill.
        `MemorySetupError`:
            The recorded run needs memory but its mapping is absent, incomplete,
            or cannot reproduce the exact recorded snapshots.
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
        memory=memory,
        tools=tools,
        skills=skills,
        steering=steering,
        api_key=api_key,
        stop_event=threading.Event(),
    )


def _resume_flow_controlled(
    flow: ComposedFlow,
    *,
    log: str | os.PathLike[str],
    num_workers: int | None = None,
    on_event: EventHook | None = None,
    on_stream: StreamHook | None = None,
    memory: dict[str, Memory] | None = None,
    tools: Mapping[str, BaseTool | Callable[..., Any] | Sequence[BaseTool | Callable[..., Any]]] | None = None,
    skills: Mapping[str, Skill] | None = None,
    steering: Steering | None = None,
    api_key: str | SecretStr | None = None,
    stop_event: threading.Event,
) -> RunResult:
    """Recover and drive a durable run with a host-owned stop event."""
    run_tools = ToolSet(tools)
    api_key = None if api_key is None else _secret_key(api_key)
    require_steering(steering)
    log_path = Path(log)
    retained_locator = log_path.absolute()
    authority = JsonlRunLog.resume(retained_locator)
    memory_runtime: MemoryRuntime | None = None
    primary_error: BaseException | None = None
    try:
        state = _replay_run_log(flow, authority, api_key=api_key)
        if api_key is not None and not host_key_providers(state.effective_llm_by_static_path):
            api_key = None
        retained_program = copy.deepcopy(flow.compiled) if state.learnable else None
        if state.result is not None:
            return _result_from_state(state, log_path, [],
                                      handle=_retained_handle(retained_program, state, authority, memory,
                                                              locator=retained_locator, api_key=api_key))
        _check_tool_grants(flow.compiled, run_tools)
        skills = _check_skills(flow.compiled, skills)

        if state.memory_layout is not None:
            if memory is None:
                raise MemorySetupError(
                    "resuming this run requires its enabled memory mapping"
                )
            memory_layout, memory_origins, memory_table, memory_stack_set = (
                _resolve_authored_memory_layout(flow)
            )
            if memory_layout != state.memory_layout:
                raise ValueError(
                    "recorded memory layout does not match the supplied flow"
                )
            _check_recorded_memory_table(
                state.memories, state.memory_stacks, memory_table, memory_stack_set,
                subject="supplied flow",
            )
            memory_runtime = MemoryRuntime.open(
                layout=state.memory_layout,
                origins=memory_origins,
                memory=memory,
                snapshots=state.memory_snapshots,
                memories=state.memories,
            )

        events: list[Any] = []
        if state.result is None:
            selected_workers = (
                state.num_workers if num_workers is None else num_workers
            )
            prepared_resume = prepare_run_resumed(
                state,
                num_workers=selected_workers,
            )
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
                memory_context_opener=(
                    None if memory_runtime is None else memory_runtime.open_context
                ),
                tools=run_tools,
                skills=skills,
                steering=steering,
            )
            events.extend(engine.run())

        return _result_from_state(state, log_path, events,
                                  handle=_retained_handle(retained_program, state, authority, memory,
                                                          locator=retained_locator, api_key=api_key))
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _close_run_resources(memory_runtime, authority, primary_error)


def _check_tool_grants(flow: CompiledFlow, tools: ToolSet) -> None:
    """Prove every static AGENT grant names a tool or tool group this run supplies.

    Runs before any record is written and covers every static node, including
    nodes that may never run, so an unresolvable grant raises `ValueError` from
    `run_flow` or `resume_flow` instead of failing its node when the node starts.
    """
    for node_id, node in _static_nodes(flow):
        for tool_id in tools.expand(list(node.granted_tools())):
            try:
                tools.resolve([tool_id])
            except ValueError as exc:
                if tool_id == "ask_user":
                    reason = f"'ask_user': {exc}"
                else:
                    reason = (
                        f"unknown tool or group {tool_id!r}; tool groups in this run: "
                        f"{list(tools.groups)}"
                    )
                raise ValueError(f"AGENT node {node_id!r} grants {reason}") from None


def _check_skills(
    flow: CompiledFlow,
    skills: Mapping[str, Skill] | None,
) -> dict[str, Skill] | None:
    """Copy the host's skills and prove every AGENT grant can be served.

    Walks top-level and static child-flow agents before any record is written. A
    flow that grants skills needs a ``skills=`` mapping (``{}`` grants none), so a
    run or resume never silently drops them. A listed skill must be in the mapping.
    """
    if skills is not None:
        if not isinstance(skills, Mapping):
            raise TypeError("skills must be a mapping of name to Skill, or None")
        for name, skill in skills.items():
            if type(skill) is not Skill or skill.name != name:
                raise TypeError(f"skills[{name!r}] must be a Skill named {name!r}")
        skills = dict(skills)
    for node_id, grants in _skill_grants(flow):
        if skills is None:
            raise ValueError(
                f"AGENT node {node_id!r} grants skills, but no skills= mapping was "
                "passed; pass the run's skills (or {} for none)"
            )
        missing = [] if grants == "all" else [name for name in grants if name not in skills]
        if missing:
            raise ValueError(
                f"AGENT node {node_id!r} grants skills not supplied to this run: {missing}"
            )
    return skills


def _skill_grants(flow: Any) -> list[tuple[str, Any]]:
    """Return each static node's non-empty skill grant, keyed by its node path."""
    return [
        (node_id, node.skills)
        for node_id, node in _static_nodes(flow)
        if node.needs_skills and node.skills
    ]


def _static_nodes(flow: Any, path: tuple[str, ...] = ()) -> list[tuple[str, Any]]:
    """Return every static node of `flow` and its child flows, keyed by node path.

    Kind-blind: child flows come from ``Node.origin_children()``, so the run
    boundary imports no node class.
    """
    found: list[tuple[str, Any]] = []
    for node_id, node in flow.nodes.items():
        node_path = (*path, node_id)
        found.append(("/".join(node_path), node))
        for child in node.origin_children():
            found.extend(_static_nodes(child, node_path))
    return found


def _replay_run_log(
    flow: ComposedFlow, log: RunLog, *, api_key: SecretStr | None = None,
) -> RunState:
    """Silently reconstruct one borrowed, locked authoritative log prefix.

    The authored fingerprint is checked before copying.  The private graph gets
    only the complete effective LLM map recorded by ``RunStarted``; recovery
    never consults current process defaults.  Historical transitions use the
    same reducer and growth constructor as live execution.  A host ``api_key``
    goes to the agents the map marks for it.
    """
    if not isinstance(flow, ComposedFlow):
        raise TypeError("_replay_run_log expects a ComposedFlow")
    return _replay_compiled_run_log(flow.compiled, log, api_key=api_key)


def _replay_compiled_run_log(
    flow: CompiledFlow, log: RunLog, *, api_key: SecretStr | None = None,
) -> RunState:
    """Reuse canonical replay for a private original compiled program."""
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
    actual_fingerprint = flow_fingerprint(flow)
    recorded_fingerprint = first.record.flow_fingerprint
    if actual_fingerprint != recorded_fingerprint:
        raise ValueError(
            "flow fingerprint mismatch: "
            f"recorded {recorded_fingerprint!r}, supplied {actual_fingerprint!r}"
        )

    private_flow = copy.deepcopy(flow)
    apply_effective_llm_map(
        private_flow,
        legacy_connection_map(first.record.effective_llm_by_static_path),
    )
    if api_key is not None:
        # A run that marked no agent ignores the key; bind_api_key finds no target.
        bind_api_key(private_flow, api_key)
    if first.record.learnable:
        apply_execution_layout(private_flow, resolve_execution_layout(private_flow))
    if first.record.memory_layout is not None:
        replay_layout, _, replay_memories, replay_stacks = _resolve_runtime_memory_layout(
            private_flow
        )
        if replay_layout != first.record.memory_layout:
            raise ValueError(
                "recorded memory layout does not match the supplied flow"
            )
        _check_recorded_memory_table(
            first.record.memories, first.record.memory_stacks, replay_memories, replay_stacks,
            subject="supplied flow",
        )
    state = _replay_records(
        RunState(private_flow),
        tuple(canonical),
        grow_builder=build_grow,
    )
    if log.run_id != source_run_id or log.next_sequence != source_sequence:
        raise ValueError("run log changed while its historical prefix was replayed")
    return state


def _resolve_runtime_memory_layout(
    flow: CompiledFlow,
    *,
    effective_llm: dict[str, dict] | None = None,
) -> tuple[
    tuple[MemoryFrameSpec, ...],
    dict[MemoryFrameRef, OriginRef | None],
    tuple[MemorySpec, ...],
    tuple[MemoryStack, ...],
]:
    """Derive, canonically order, and atomically stamp one private graph.

    With ``effective_llm``, first check that each session key resolves one
    conversation compatibility across its placements.
    """
    resolved = resolve_memory_layout(flow)
    if effective_llm is not None:
        check_session_compatibility(resolved, effective_llm)
    apply_memory_layout(flow, resolved)
    ordered = tuple(
        sorted(
            resolved.frames,
            key=lambda frame: (
                frame.spec.ref.root_flow_id,
                frame.spec.ref.static_path,
                frame.spec.ref.owner.value,
            ),
        )
    )
    return (
        tuple(frame.spec for frame in ordered),
        {frame.spec.ref: frame.origin for frame in ordered},
        tuple(memory.spec for memory in resolved.memories),
        memory_stacks(resolved),
    )


def _resolve_authored_memory_layout(
    flow: ComposedFlow,
) -> tuple[
    tuple[MemoryFrameSpec, ...],
    dict[MemoryFrameRef, OriginRef | None],
    tuple[MemorySpec, ...],
    tuple[MemoryStack, ...],
]:
    """Resolve memory only on a fresh static copy, never replayed growth."""
    if not isinstance(flow, ComposedFlow):
        raise TypeError("memory layout resolution expects a ComposedFlow")
    return _resolve_runtime_memory_layout(copy.deepcopy(flow.compiled))


def _check_recorded_memory_table(
    recorded_memories: tuple[MemorySpec, ...] | None,
    recorded_stacks: tuple[MemoryStack, ...] | None,
    memories: tuple[MemorySpec, ...],
    stacks: tuple[MemoryStack, ...],
    *,
    subject: str,
) -> None:
    """Reject a recorded key table or stack set that the flow does not resolve.

    A log that recorded neither passes unchecked.
    """
    if recorded_memories is None and recorded_stacks is None:
        return
    if recorded_memories != memories or recorded_stacks != stacks:
        raise ValueError(f"recorded memory table or stacks do not match the {subject}")


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
    *, handle: _BackwardHandle | None = None,
) -> RunResult:
    """Project one terminal reducer state into the exact public result shape."""
    terminal = state.result
    outcome = state.backward_state.outcome if handle is None else handle.outcome
    projected_learning = None if outcome is None else outcome.learning
    if isinstance(terminal, RunSucceeded):
        result = RunResult(
            input=copy.deepcopy(state.input),
            status="succeeded",
            run_id=_required_state_run_id(state),
            log=path,
            output=copy.deepcopy(terminal.output),
            events=events,
            learning=projected_learning,
            backward_outcome=outcome,
        )
    elif isinstance(terminal, RunFailed):
        result = RunResult(
            input=copy.deepcopy(state.input),
            status="failed",
            run_id=_required_state_run_id(state),
            log=path,
            failure=terminal.failure,
            events=events,
            learning=projected_learning,
            backward_outcome=outcome,
        )
    else:
        raise RuntimeError("engine returned without deriving a terminal run result")
    result._backward_handle = handle
    return result


def _retained_handle(
    program: CompiledFlow | None, state: RunState, log: RunLog, memory: dict[str, Memory] | None,
    *, locator: Path | None, api_key: SecretStr | None = None,
) -> _BackwardHandle | None:
    """Detach delayed authority without keeping an opened session or public wrapper."""
    if program is None:
        return None
    return _BackwardHandle(program, state.run_id, locator, tuple(log) if log.path is None else (),
                           memory.copy() if type(memory) is dict else memory, state.backward_state.outcome,
                           api_key=api_key)


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


def _close_run_resources(
    memory_runtime: MemoryRuntime | None,
    log: RunLog | None,
    primary_error: BaseException | None,
) -> None:
    """Close memory then log while preserving one initiating exception."""
    pending = primary_error
    if memory_runtime is not None:
        try:
            memory_runtime.close()
        except BaseException as cleanup_error:
            if pending is None:
                pending = cleanup_error
            else:
                _note_cleanup_error(
                    pending,
                    "memory-session close",
                    cleanup_error,
                )
    if log is not None:
        try:
            log.close()
        except BaseException as cleanup_error:
            if pending is None:
                pending = cleanup_error
            else:
                _note_cleanup_error(pending, "run-log close", cleanup_error)
    if primary_error is None and pending is not None:
        raise pending


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


__all__ = ["RunResult", "Steering", "resume_flow", "run_flow"]

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

from collections.abc import Iterable, Mapping, Sequence
import copy
import difflib
from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping
import uuid

from langchain_core.tools import BaseTool
from pydantic import SecretStr

from yian.llm_clients.config import LLMConfig, pop_host_credentials
from yian.llm_clients.connection_settings import validate_headers
from yian.llm_clients.connection import (
    HEADER_NAMES_KEY,
    HOST_KEY_MARKER,
    ConnectionSnapshot,
    block_problem,
    check_model_config,
    connection_snapshot,
    header_name_diff,
    legacy_connection_config,
    legacy_connection_map,
    refused_block,
    split_model_config,
)
from yian.compile.execution_layout import apply_execution_layout, resolve_execution_layout
from yian.compile.expand import build_grow
from yian.compile.llm_cascade import (
    apply_effective_llm_map,
    bind_host_credentials,
    call_time_selection,
    in_flight_llm_configs,
    known_call_time_providers,
    RunLLM,
    host_key_providers,
    resolve_run_llm,
    static_agent_labels,
    value_config_subjects,
)
from yian.compile.manifest import flow_fingerprint
from yian.compile.memory_layout import (
    ResolvedMemoryLayout,
    apply_memory_layout,
    check_session_compatibility,
    memory_stacks,
    resolve_memory_layout,
)
from yian.compile.model import CompiledFlow
from yian.compile.validation import type_relation
from yian.compose.helpers.tags import CLOSE_MATCH_CUTOFF
from yian.compose.loader import ComposedFlow
from yian.compose.nodes.resolver import child_signature
from yian.defaults import Defaults, check_recorded, current
from yian.diagnostics import Failure, OriginRef
from yian.events import (
    RunFailed,
    RunSucceeded,
    StreamChunk,
)
from yian.expr import ExpressionError, check_template_calls
from yian.expr.lineage import FeedbackValue
from yian.memory import (
    LearningResult,
    Memory,
    MemorySetupError,
    MemorySpec,
    MemoryStack,
)
from yian.memory_host import MemoryRegistry
from yian.nodes.base import Steering, _HostStop
from yian.nodes.binding import BindingError, apply_defaults, coerce_inputs
from yian.nodes.contracts import (
    HostToolCall,
    OutputContract,
    Parameter,
    iter_source_strings,
)
from yian.runtime.engine import FlowEngine
from yian.runtime.eval_node import require_steering
from yian.runtime.backward import drive_backward
from yian.runtime.contracts import BackwardOutcome
from yian.runtime.log import (
    JsonlRunLog,
    MemoryRunLog,
    RecordEnvelope,
    RunLog,
    RunStartedRecord,
    decode_record,
    encode_record,
)
from yian.runtime.state import (
    RunStartData,
    RunState,
    _replay_records,
    commit_prepared,
    prepare_run_resumed,
    prepare_run_started,
)
from yian.runtime.memory_runtime import MemoryRuntime
from yian.skills import Skill
from yian.tools import FunctionTool, McpServer, ToolSet
from yian.tools.mcp import _AlreadyOpen
from yian.tools.types import tool_types
from yian.typesys import Type, ValueKind, canonical_type_label
from yian.typesys.pool import VariablePool
from yian.typesys.seeding import default_run_id, seed_system_clock
from yian.typesys.values import (
    reject_deep_nesting,
    reject_long_ints,
    reject_non_text_keys,
    reject_nonfinite,
)


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
        headers: Mapping[str, str | SecretStr] | None = None,
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
            headers (`Mapping[str, str | SecretStr] | None`, *optional*, defaults to `None`):
                The header values a run started with ``llm_config.headers`` sent
                with its key, bound with the key and never recorded. Their names
                must be the ones the run recorded for the key's agents, compared
                without case. ``None`` uses the headers this result was created
                with, if any. A run that marked no agent ignores them. Without
                them, agents marked for recorded header names fail their
                backward call.

        Returns:
            `BackwardOutcome`: The accepted attempt result, separate from forward
                status, output, failure, and events. Learning is its projection.

        Raises:
            `TypeError`: Feedback has an invalid ordinary-message carrier.
            `ValueError`: Retention is disabled, feedback does not match the
                output's shape or nests deeper than `YIAN_MAX_VALUE_NESTING` levels,
                or ``headers`` has a name or value that can't be sent or names that
                differ from the recorded ones (the message names the header names,
                never a value).
            `TypeCheckError`: Feedback holds an int over `YIAN_MAX_INT_DIGITS` digits;
                the message locates it, e.g. ``feedback at note: integer with N
                digits is larger than yian allows (4300 digits)``.
            `MemorySetupError`: Enabled memories cannot restore their recorded
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
        # Checked before anything walks the feedback recursively.
        reject_deep_nesting(feedback, "feedback")
        reject_long_ints(feedback, "feedback")
        if not handle._guard.acquire(blocking=False):
            raise RuntimeError("a backward call on this result is already in progress")
        try:
            try:
                return handle.backward(
                    feedback, api_key=None if api_key is None else _secret_key(api_key),
                    headers=None if headers is None else _secret_headers(headers),
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
    memory: Mapping[str, Memory] | None
    outcome: BackwardOutcome | None
    # Held in memory for this result's own backward calls; never recorded.
    api_key: SecretStr | None = field(default=None, repr=False)
    headers: dict[str, SecretStr] | None = field(default=None, repr=False)
    _guard: Any = field(default_factory=threading.Lock, init=False, repr=False, compare=False)
    _unusable: bool = field(default=False, init=False, repr=False, compare=False)

    def backward(
        self, feedback: FeedbackValue, *, api_key: SecretStr | None = None,
        headers: dict[str, SecretStr] | None = None,
    ) -> BackwardOutcome:
        """Reconstruct accepted facts and delegate the complete attempt driver."""
        if self._unusable:
            raise RuntimeError("backward handle is unavailable after an accepted-prefix refresh failure")
        authority = MemoryRunLog._from_records(self.records) if self.path is None else JsonlRunLog.resume(self.path)
        state = None
        replay_sequence = None
        primary_error = None
        try:
            state = _replay_compiled_run_log(
                self.program, authority, api_key=self.api_key if api_key is None else api_key,
                headers=self.headers if headers is None else headers,
            )
            if state.run_id != self.run_id:
                raise ValueError("backward log identity differs from the retained run")
            if headers is not None and _uses_host_key(
                state.effective_llm_by_static_path, state.call_time_llm
            ):
                # Before the attempt is appended; kept headers were checked when kept.
                _check_host_header_names(
                    state.effective_llm_by_static_path, headers, state.call_time_llm
                )
            replay_sequence = state.next_sequence

            def open_memory() -> MemoryRuntime:
                if self.memory is None:
                    raise MemorySetupError("backward requires this run's enabled memory mapping")
                _check_registry_types(self.memory, state.memories)
                program = copy.deepcopy(self.program)
                apply_effective_llm_map(
                    program, legacy_connection_map(state.effective_llm_by_static_path),
                    state.call_time_llm,
                )
                memory_table, origins, memory_stack_set = _resolve_runtime_memory_layout(program)
                _check_recorded_memory_table(
                    state.memories, state.memory_stacks, memory_table, memory_stack_set,
                    subject="retained program",
                )
                return MemoryRuntime.open(memories=state.memories, origins=origins,
                                          memory=self.memory, snapshots=state.memory_snapshots)

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
    memory: Mapping[str, Memory] | None = None,
    learnable: bool = False,
    tools: Mapping[str, BaseTool | FunctionTool | Callable[..., Any] | McpServer | Sequence[BaseTool | FunctionTool | Callable[..., Any]]] | None = None,
    skills: Mapping[str, Skill] | None = None,
    steering: Steering | None = None,
    context: Any = None,
) -> RunResult:
    """Start and drive one flow through an in-memory or durable run log.

    Flow failures are returned as ``RunResult(status="failed")``.  Host,
    callback, log, and reducer errors propagate after any active workers have
    been drained and the boundary-owned log has been closed.

    Args:
        flow (`ComposedFlow`):
            Compiled, validated flow to copy and execute.
        inputs (`dict[str, Any]`):
            Caller values keyed by declared input name; any other key is an
            error. Supplied scalars are coerced into the initial pool; START
            applies declared defaults and full boundary type checks during its
            recorded node execution.
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
            Per-run outer model configuration layer, taken literally: ``${...}``
            in it is text. The fully resolved static map is captured in the
            initialization record. An agent whose flow-file ``llm_config`` reads
            a value known only when it runs is recorded instead with the layers
            below its own and the connection selection it resolves from, and
            resolves just before its model call. An ``api_key`` entry (a
            `str` or `SecretStr`) is the provider key for this run and needs
            ``provider``. Every AGENT whose effective provider matches, static or
            resolved when it runs, uses it in place of the provider's saved login
            or key environment variable; agents of other providers are unaffected.
            An ``inherit: false`` agent of that provider uses it too. The key is
            never recorded; the log marks which agents use it, so pass it again
            to `resume_flow` or `RunResult.backward`, where those agents
            otherwise fail rather than read the environment. A ``base_url``
            entry, accepted only beside ``api_key`` in this layer, is the
            endpoint those agents send the key to. The log records it on them,
            so a resume or backward call sends the key there too. An ``azure``
            key needs a ``base_url``. Flow files cannot set ``api_key``, nor
            ``base_url`` except beside ``provider: azure``, where it is the
            resource endpoint and needs no key.
        memory (`Mapping[str, Memory] | None`, *optional*, defaults to `None`):
            Backend for each declared memory key, typically a `MemoryRegistry`
            from `build_memories`. Extra keys are ignored. ``None`` leaves every
            declared memory dormant. A mapping enables recall and optional
            conversation history, and records exact per-memory recovery
            snapshots. Session history advances during forward execution.
        learnable (`bool`, *optional*, defaults to `False`):
            Retain accepted execution for a later explicit `RunResult.backward`
            call. Forward execution runs no memory learners or learned-mutation
            commits. Memory is optional; an enabled mapping supplies owned learners
            for that call. Automatic session history is independent of retention.
        tools (`Mapping[str, BaseTool | FunctionTool | Callable[..., Any] | McpServer | Sequence[BaseTool | FunctionTool | Callable[..., Any]]] | None`, *optional*, defaults to `None`):
            The tools this run sees, keyed by tool id. `None` uses the tools
            registered with `register_tool`. A mapping, even an empty one, is the
            whole set: registered tools are hidden. To keep them, pass
            `{**registered_tools(), ...}`. That copy is taken when you call it, so
            register every tool before the run starts. A function or method
            becomes a tool named by its key and described by its docstring; bind
            values with a closure, not `functools.partial`. A `FunctionTool` or
            `BaseTool` must use its key as its name. The `ask_user` entry is the host handler for
            HUMAN_INPUT nodes and granted AGENT `ask_user` calls. It must be a
            plain callable that accepts `question`, `questions`, `answer_schema`,
            `title`, `node_id`, and `options` as keywords (define it as
            `def handler(**request)`); the model sees only `question` and
            `options`. Runs use
            up to eight workers by default, so several nodes can call it at once;
            it must be thread-safe. An AGENT can call a tool only if its `tools:`
            list grants the id. A list or tuple value is a tool group: an AGENT
            that grants the group's key can call every tool in it. Each item is a
            `FunctionTool` or `BaseTool` named by its `name`, or a function or
            method named by its `__name__`. Each tool name calls one tool: the same tool may sit in
            several groups, but two different tools may not share a name, and
            no group tool may be named `ask_user` or `load_skill`. An `McpServer`
            from `mcp_server` is a group of the tools it lists. One that isn't open
            belongs to this run: it is opened, after the other entries are checked
            and before the run-start checks, and closed when `run_flow` returns or
            raises; the same server under several keys opens once. One opened with
            `with server:` belongs to the host, is used as is, and stays open, so
            several runs can share it. Every static AGENT grant must name a tool or
            group in this set, and every static TOOL node's `tool_id` must name a
            tool in it; this is checked before the log is created.
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
        context (`Any`, *optional*, defaults to `None`):
            The host's app object, such as a database session or a broker
            client. A tool whose first parameter is annotated `ToolContext`
            receives it as `ctx.context`; with `None`, tools see
            `ctx.context is None`. It is never recorded and is not part of the
            flow fingerprint, so pass it again to `resume_flow`.

    Returns:
        `RunResult`:
            Terminal success or failure, including this invocation's post-durable
            observers and the durable path when one was requested.

    Raises:
        `TypeError`:
            A public argument has the wrong type, including `inputs` that is not
            a mapping, a malformed `tools` entry or group item, or a `steering`
            object without `drain`/`interrupted`.
        `ValueError`:
            An `inputs` key names no declared input, an `inputs` value nests
            deeper than `YIAN_MAX_VALUE_NESTING` levels, holds a NaN, infinity, or
            complex number, or an int with more than `YIAN_MAX_INT_DIGITS` digits, at
            any depth, or holds a mapping with a non-text key;
            identity, worker count,
            model configuration, or prepared run state violates its contract,
            an AGENT has no provider or model (no layer, environment default,
            or saved login sets one; there is no built-in default), an input an
            ``llm_config`` value resolved at run start reads is missing or fails
            its declared type, or the config resolved from it is refused (the
            message names the field, as ``node 'chat' llm_config.model: ...``),
            ``llm_config.api_key`` lacks ``llm_config.provider``,
            ``llm_config.base_url`` lacks
            ``llm_config.api_key`` or is not an allowed URL, an ``azure``
            ``api_key`` lacks ``base_url``, an ``azure`` agent has no
            ``base_url``, a `BaseTool` in
            `tools` is named differently from its key, a group tool's name
            clashes or is reserved, an `McpServer` in `tools` is open, or being
            opened, by another run in progress, an AGENT grants a tool or group
            not in the run's tools, a TOOL node names a tool not in the run's
            tools, a TOOL node's inputs or `output:` do not match the names and
            types its tool declares (an input the tool does not take,
            a missing required input, or a type that can never fit), an
            AGENT grants skills while `skills` is `None` or lacks a listed
            skill, a node assertion, CASE `when`, or LOOP `while`/`until`
            set in Python is not a valid boolean expression, or a binding, flow
            output, or prompt set in Python has a `${...}` span that doesn't
            parse or calls a function it can't (a builtin with arguments it
            doesn't take, a name outside the node's type table, or in a prompt
            any non-builtin).
        `ImportError`:
            An `McpServer` in `tools` must be opened and the MCP SDK is not
            installed.
        `McpConnectError`:
            An `McpServer` in `tools` that the run opens doesn't start or doesn't
            list its tools; the servers it already opened are closed.
        `DefaultsError`:
            A `YIAN_<NAME>` variable holds a value its entry does not accept.
        `MemorySetupError`:
            Memory is enabled without a required memory implementation, cannot
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
        context=context,
        stop_event=threading.Event(),
    )


def _secret_headers(raw: Any) -> dict[str, SecretStr]:
    """Return host header values as secrets once every name and value may be sent.

    Args:
        raw (`Any`):
            The host's mapping of header name to `str` or `SecretStr` value.

    Returns:
        `dict[str, SecretStr]`: The headers in their given order.

    Raises:
        `ValueError`: A name or value is refused, or there are too many headers; the
            message never echoes a value.
    """
    return {name: SecretStr(value) for name, value in validate_headers(raw, None).items()}


def _uses_host_key(
    effective: Mapping[str, Mapping[str, Any]], call_time: Mapping[str, Any] | None,
) -> bool:
    """Return whether a run's recorded agents may use the host key.

    Args:
        effective (`Mapping[str, Mapping[str, Any]]`):
            The recorded effective model map, keyed by static agent path.
        call_time (`Mapping[str, Any] | None`):
            The recorded ``RunStarted.call_time_llm``, or `None`.

    Returns:
        `bool`: Whether a static agent is marked for the host key, or the call-time
            agents' selection names the host key's provider.
    """
    return bool(host_key_providers(dict(effective))) or (
        call_time is not None and call_time["selection"]["host"] is not None
    )


def _selection_host(call_time: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the host key the call-time agents' selection names, for a resume check.

    A call-time agent that has not started may still resolve to that key's provider,
    so a resume that can't supply the key is refused up front, as for a marked agent.

    Args:
        call_time (`Mapping[str, Any] | None`):
            The recorded ``RunStarted.call_time_llm``, or `None`.

    Returns:
        `dict[str, Any] | None`: ``{"provider": ..., "header_names": [...]}`` (header
            names only when recorded), or `None` when the selection names no host key.
    """
    host = None if call_time is None else call_time["selection"]["host"]
    if host is None:
        return None
    selected: dict[str, Any] = {"provider": host["provider"]}
    if host.get("header_names"):
        selected["header_names"] = list(host["header_names"])
    return selected


def _check_host_header_names(
    effective: Mapping[str, Mapping[str, Any]], headers: Mapping[str, Any] | None,
    call_time: Mapping[str, Any] | None = None,
) -> None:
    """Check that the host's headers have the names recorded for its key's agents.

    Names compare without case, as HTTP compares them.

    Args:
        effective (`Mapping[str, Mapping[str, Any]]`):
            The recorded effective model map, keyed by static agent path.
        headers (`Mapping[str, Any] | None`):
            The headers given for the host key; `None` gives none.
        call_time (`Mapping[str, Any] | None`, *optional*, defaults to `None`):
            The recorded ``RunStarted.call_time_llm``; the header names its selection
            records for the host key are expected too.

    Raises:
        `ValueError`: The names differ; the message names the missing and unrecorded
            header names, never a value.
    """
    expected = [
        name
        for config in effective.values()
        if config.get(HOST_KEY_MARKER) is True
        for name in config.get(HEADER_NAMES_KEY, ())
    ]
    host = None if call_time is None else call_time["selection"]["host"]
    if host is not None:
        expected.extend(host.get("header_names", ()))
    diff = header_name_diff(expected, headers or ())
    if diff is not None:
        raise ValueError(
            "headers must have the header names this run recorded for its host key: " + diff
        )


def _secret_key(raw: str | SecretStr) -> SecretStr:
    """Validate a host key into a `SecretStr` without echoing it in errors."""
    return LLMConfig(api_key=raw).api_key


def _check_effective_llm(
    effective_llm: Mapping[str, dict[str, Any]],
    api_key: SecretStr | None,
    headers: dict[str, SecretStr] | None,
    labels: Mapping[str, str],
    located: Mapping[str, str] | None = None,
) -> None:
    """Reject any agent config its model call would reject, before the run starts.

    Args:
        effective_llm (`Mapping[str, dict[str, Any]]`):
            The effective model map, keyed by static agent path.
        api_key (`SecretStr | None`):
            The host key, supplied to each config marked for it.
        headers (`dict[str, SecretStr] | None`):
            The header values sent with the host key, supplied beside it.
        labels (`Mapping[str, str]`):
            The authored site of each synthesized agent's ``llm_config``, keyed by
            static agent path; a message names it instead of the path.
        located (`Mapping[str, str] | None`, *optional*, defaults to `None`):
            The ``llm_config`` subject of each agent whose layers carry values
            (`value_config_subjects`), keyed by static agent path; its message names
            the subject, and the provider's block when the block is what is refused.

    Raises:
        `ValueError`: A config would be rejected by its model call; the message names
            its agent path, or the authored site of a synthesized agent.
    """
    for path, config in effective_llm.items():
        # The host key and header values stay out of the map; supply them where the
        # marker expects them.
        probe = config
        if HOST_KEY_MARKER in config:
            probe = {**config, "api_key": api_key}
            if headers:
                probe["headers"] = headers
        try:
            check_model_config(probe)
        except ValueError as exc:
            subject = (located or {}).get(path)
            if subject is not None:
                raise ValueError(
                    block_problem(subject, refused_block(probe)[1:], str(exc))
                ) from exc
            where = labels.get(path) or f"Agent {path!r}"
            raise ValueError(f"{where}: {exc}") from exc


def _run_start_inputs(
    params: list[Parameter], coerced: dict[str, Any]
) -> dict[str, Any]:
    """Return the run's inputs as run start resolves against them, defaults filled.

    START still binds and checks the inputs, so a missing required input or a
    default that fails its check stays START's logged failure: such a default is
    left out here rather than raised before the log exists.

    Args:
        params (`list[Parameter]`):
            The flow's declared inputs.
        coerced (`dict[str, Any]`):
            The supplied inputs, already coerced at the run boundary.

    Returns:
        `dict[str, Any]`: The supplied inputs plus each omitted input's valid default.
    """
    run_inputs = dict(coerced)
    for parameter in params:
        try:
            run_inputs = apply_defaults([parameter], run_inputs)
        except BindingError:
            continue
    return run_inputs


def _run_start_llm(
    compiled: CompiledFlow,
    coerced: Mapping[str, Any],
    llm_config: dict[str, Any],
    snapshot: ConnectionSnapshot,
) -> RunLLM:
    """Resolve a fresh run's llm cascade over ``compiled``, in place, with its inputs.

    Values that read only the run's inputs resolve against the supplied inputs, with
    omitted inputs' defaults filled.

    Args:
        compiled (`CompiledFlow`):
            The run's private flow copy; its agents get their effective configs.
        coerced (`Mapping[str, Any]`):
            The supplied inputs, already coerced at the run boundary.
        llm_config (`dict[str, Any]`):
            The host layer, without the host key and its settings.
        snapshot (`ConnectionSnapshot`):
            The one selection snapshot that determines every agent's defaults.

    Returns:
        `RunLLM`: The resolved configurations, before any connection is bound.

    Raises:
        `ValueError`: As `resolve_run_llm` raises it.
    """
    return resolve_run_llm(
        compiled,
        llm_config,
        defaults=snapshot.defaults,
        provider_models=snapshot.provider_models,
        stray_model=snapshot.stray_model,
        run_inputs=_run_start_inputs(compiled.nodes[compiled.start_id].params, dict(coerced)),
    )


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
    memory: Mapping[str, Memory] | None = None,
    learnable: bool = False,
    tools: Mapping[str, BaseTool | FunctionTool | Callable[..., Any] | McpServer | Sequence[BaseTool | FunctionTool | Callable[..., Any]]] | None = None,
    skills: Mapping[str, Skill] | None = None,
    steering: Steering | None = None,
    context: Any = None,
    stop_event: threading.Event,
) -> RunResult:
    """Drive a fresh run with one host-owned cooperative stop event and its own MCP servers."""
    # A bad YIAN_<NAME> value fails here, before any work.
    run_defaults = current()
    if not isinstance(flow, ComposedFlow):
        raise TypeError("run_flow expects a ComposedFlow")
    if type(learnable) is not bool:
        raise TypeError("learnable must be a bool")
    other_tools, unopened = _split_run_servers(tools)
    # The other entries' errors, and the checks that need no tools, come before any
    # server starts.
    ToolSet(other_tools, context=context)
    _check_conditions(flow.compiled)
    _check_templates(flow.compiled)
    skills = _check_skills(flow.compiled, skills)
    require_steering(steering)
    run_servers = _open_run_servers(unopened, stop_event)
    primary_error: BaseException | None = None
    try:
        return _start_run(
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
            context=context,
            stop_event=stop_event,
            run_defaults=run_defaults,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _close_run_servers(run_servers, primary_error)


def _start_run(
    flow: ComposedFlow,
    inputs: dict[str, Any],
    *,
    log: str | os.PathLike[str] | None,
    run_id: str | None,
    num_workers: int,
    on_event: EventHook | None,
    on_stream: StreamHook | None,
    llm_config: dict[str, Any] | None,
    memory: Mapping[str, Memory] | None,
    learnable: bool,
    tools: Mapping[str, BaseTool | FunctionTool | Callable[..., Any] | McpServer | Sequence[BaseTool | FunctionTool | Callable[..., Any]]] | None,
    skills: Mapping[str, Skill] | None,
    steering: Steering | None,
    context: Any,
    stop_event: threading.Event,
    run_defaults: Defaults,
) -> RunResult:
    """Check the tools, then record and drive a fresh run whose MCP servers are open."""
    run_tools = ToolSet(tools, context=context)
    _check_tool_grants(flow.compiled, run_tools)
    _check_tool_types(flow.compiled, run_tools)

    # Identity belongs to the authored program, before cascade resolution or any
    # per-run copy can change object topology.
    authored_fingerprint = flow_fingerprint(flow.compiled)
    private_flow = copy.deepcopy(flow.compiled)
    retained_program = copy.deepcopy(private_flow) if learnable else None

    # Convert supplied boundary strings before RunStarted persists its input, and
    # before the cascade, so a bad supplied input is reported first. This is the only
    # string conversion: the generic binder converts nothing, so a value crossing a
    # CALL/MAP/LOOP edge is validated as is, and START stays the authority for
    # required/default handling and contract validation.
    coerced = coerce_inputs(flow.input, inputs)
    for name, value in coerced.items():
        reject_deep_nesting(value, f"input {name!r}")
        reject_nonfinite(value, f"input {name!r}")
        reject_non_text_keys(value, f"input {name!r}")

    # The key stays out of the cascade so RunStarted never records it. Its
    # base_url and connection settings stay out too, so no authored layer can pair
    # the key with another.
    run_llm_config = dict(llm_config or {})
    host = pop_host_credentials(run_llm_config)
    api_key, headers = host.api_key, host.headers

    # One selection snapshot determines every static agent before RunStarted, and
    # what each agent resolved only when it runs binds from.
    snapshot = connection_snapshot()
    host_key_provider = None if api_key is None else run_llm_config["provider"]
    run_llm = _run_start_llm(private_flow, coerced, run_llm_config, snapshot)
    effective_llm = snapshot.bind(
        run_llm.effective,
        host_key_provider=host_key_provider,
        host_base_url=host.base_url,
        host_settings=host,
    )
    # Each call-time agent's provider when run start already knows it: it limits the
    # recorded selection and the session check.
    known = known_call_time_providers(private_flow, run_llm.call_time, snapshot.defaults)
    call_time_llm = None
    if run_llm.call_time:
        call_time_llm = {
            "selection": call_time_selection(
                snapshot,
                known=known,
                host_key_provider=host_key_provider,
                host_base_url=host.base_url,
                host_settings=host,
            ),
            "agents": run_llm.call_time,
        }
    apply_effective_llm_map(private_flow, effective_llm, call_time_llm)
    # A key that neither marks an agent nor is in the call-time agents' selection has
    # nothing to authenticate, so the result keeps neither it nor its headers.
    if api_key is not None and not _uses_host_key(effective_llm, call_time_llm):
        api_key = headers = None
    if api_key is not None:
        bind_host_credentials(private_flow, api_key, headers)
    _check_effective_llm(
        effective_llm,
        api_key,
        headers,
        static_agent_labels(private_flow),
        value_config_subjects(private_flow),
    )

    if learnable:
        apply_execution_layout(private_flow, resolve_execution_layout(private_flow))

    canonical_run_id = run_id if run_id is not None else default_run_id()
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
        memory_table, memory_origins, memory_stack_set = _resolve_runtime_memory_layout(
            private_flow, effective_llm=effective_llm,
            call_time_known=known,
        )
        _check_registry_types(memory, memory_table)
        memory_runtime = MemoryRuntime.open(
            memories=memory_table, origins=memory_origins, memory=memory,
        )

    authority: RunLog | None = None
    primary_error: BaseException | None = None
    try:
        # The result reads its files through the contract a CALL of this flow gets.
        state = RunState(private_flow, result_contract=child_signature(flow).output)
        prepared_start = prepare_run_started(
            state,
            RunStartData(
                run_id=canonical_run_id,
                flow_fingerprint=authored_fingerprint,
                initial_pool=pool,
                start_wiring_override=start_wiring_override,
                effective_llm_by_static_path=effective_llm,
                call_time_llm=call_time_llm,
                num_workers=num_workers,
                memory_snapshots=(
                    None if memory_runtime is None else memory_runtime.snapshots
                ),
                learnable=learnable,
                memories=memory_table,
                memory_stacks=memory_stack_set,
                defaults=run_defaults.non_defaults(),
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
                                  handle=_retained_handle(
                                      retained_program, state, authority, memory,
                                      locator=retained_locator, api_key=api_key, headers=headers,
                                  ))
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
    memory: Mapping[str, Memory] | None = None,
    tools: Mapping[str, BaseTool | FunctionTool | Callable[..., Any] | McpServer | Sequence[BaseTool | FunctionTool | Callable[..., Any]]] | None = None,
    skills: Mapping[str, Skill] | None = None,
    steering: Steering | None = None,
    api_key: str | SecretStr | None = None,
    headers: Mapping[str, str | SecretStr] | None = None,
    context: Any = None,
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
        memory (`Mapping[str, Memory] | None`, *optional*, defaults to `None`):
            Backend for each declared memory key, typically a `MemoryRegistry`
            from `build_memories`. Extra keys are ignored. Required when the
            recorded run enabled memory and still has unfinished forward execution. Terminal
            inspection opens no memories; supplied references remain available
            for later explicit backward calls.
        tools (`Mapping[str, BaseTool | FunctionTool | Callable[..., Any] | McpServer | Sequence[BaseTool | FunctionTool | Callable[..., Any]]] | None`, *optional*, defaults to `None`):
            The tools the resumed segment sees, with the same rules as
            `run_flow`. Tools are not recorded in the log, so pass the same
            mapping (including `ask_user`) that the original run used. Group
            contents may differ: nodes that restart see the groups passed here.
            An unopened `McpServer` belongs to this call: it is opened only when
            the run is unfinished, after the log is replayed and before the
            resume checks, and closed when `resume_flow` returns or raises.
            Terminal inspection opens no server.
        skills (`Mapping[str, Skill] | None`, *optional*, defaults to `None`):
            The skills passed to the original `run_flow`. Nodes that restart see
            this mapping. An unfinished run of a flow that grants skills needs
            it; terminal inspection does not check it.
        steering (`Steering | None`, *optional*, defaults to `None`):
            The host inbox for steerable AGENT nodes, as in `run_flow`. Finished
            nodes are replayed from the log and never call it.
        api_key (`str | SecretStr | None`, *optional*, defaults to `None`):
            The provider key of a run started with ``llm_config.api_key``. It
            goes to the agents the log marks for it, at the ``base_url`` the
            log recorded for them if the run had one, as in `run_flow`. It is
            never recorded, and stays with the result for `RunResult.backward`. A
            log that marks no agent ignores it. Without it, marked agents fail
            when they call their model instead of reading the provider's key
            environment variable. An agent that resolves its ``llm_config`` when
            it runs gets it when it resolves to the key's provider; without it,
            such an agent raises `AuthError` before its model call.
        headers (`Mapping[str, str | SecretStr] | None`, *optional*, defaults to `None`):
            The header values a run started with ``llm_config.headers`` sent
            with its key. They go to the agents the log marks for the key, like
            ``api_key``, and are never recorded. Their names must be the ones the
            log recorded for those agents, compared without case; a log that
            marks no agent ignores them.
        context (`Any`, *optional*, defaults to `None`):
            The host's app object, passed to tools as `ctx.context` as in
            `run_flow`. The log does not hold it and it is not part of the flow
            fingerprint, so pass it again here; without it, tools in the resumed
            segment see `ctx.context is None`.

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
            is reserved, an `McpServer` in `tools` is open, or being opened, by
            another run in progress, an unfinished run's AGENT grants a tool or
            group not in `tools` or grants skills while `skills` is `None` or lacks
            a listed skill, an unfinished run's TOOL node names a tool not in
            `tools`, an unfinished run's TOOL node's inputs or
            `output:` do not match the names and types its tool declares, or
            an unfinished run's node assertion, CASE `when`, or LOOP
            `while`/`until` set in Python is not a valid boolean expression, or
            an unfinished run's binding, flow output, or prompt set in Python
            has a `${...}` span that doesn't parse or calls a function it can't
            (a builtin with arguments it doesn't take, a name outside the node's
            type table, or in a prompt any non-builtin). Also when ``headers``
            has a name or value that can't be sent, or ``api_key`` or
            ``headers`` is given for a log that marks agents for a host key and
            the header names differ from the recorded ones (the message names
            the header names, never a value). Also when an agent that resolves
            its ``llm_config`` when it runs, and had started before, resolves to
            a config other than the one it recorded (the flow or yian version
            changed); no model call happens.
        `AuthError`:
            (`yian.llm_clients.auth.AuthError`) An agent resolving its
            ``llm_config`` when it runs resolves to the host key's provider, and
            ``api_key`` (or the recorded header names) was not given.
        `ImportError`:
            An unfinished run's `McpServer` must be opened and the MCP SDK is not
            installed.
        `McpConnectError`:
            An unfinished run's `McpServer` that the call opens doesn't start or
            doesn't list its tools; the servers it already opened are closed.
        `DefaultsError`:
            A `YIAN_<NAME>` variable holds a value its entry does not accept.
        `DefaultsMismatchError`:
            The run started with `yian.defaults` values that differ from this
            process's, or with an entry this version does not know; the message
            names each variable and how to set it to resume.
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
        headers=headers,
        context=context,
        stop_event=threading.Event(),
    )


def _resume_flow_controlled(
    flow: ComposedFlow,
    *,
    log: str | os.PathLike[str],
    num_workers: int | None = None,
    on_event: EventHook | None = None,
    on_stream: StreamHook | None = None,
    memory: Mapping[str, Memory] | None = None,
    tools: Mapping[str, BaseTool | FunctionTool | Callable[..., Any] | McpServer | Sequence[BaseTool | FunctionTool | Callable[..., Any]]] | None = None,
    skills: Mapping[str, Skill] | None = None,
    steering: Steering | None = None,
    api_key: str | SecretStr | None = None,
    headers: Mapping[str, str | SecretStr] | None = None,
    context: Any = None,
    check_llm_configs: Callable[..., None] | None = None,
    stop_event: threading.Event,
) -> RunResult:
    """Recover and drive a durable run with a host-owned stop event.

    ``check_llm_configs`` receives a copy of the recorded effective model map, plus
    the config each started call-time agent recorded keyed by its node id (old OpenAI
    records adapted to their legacy connection), and as ``call_time_host`` the host
    key the call-time agents' selection names (`_selection_host`), after the terminal
    check, before an unfinished run opens memory or appends ``RunResumed``. An exception
    it raises stops recovery and appends nothing. The CLI uses it to check
    provider credentials.
    """
    # A bad YIAN_<NAME> value fails here, before the log is opened.
    current()
    other_tools, unopened = _split_run_servers(tools)
    # The other entries' TypeErrors and ValueErrors come before the log is opened.
    ToolSet(other_tools, context=context)
    api_key = None if api_key is None else _secret_key(api_key)
    host_headers = None if headers is None else _secret_headers(headers)
    require_steering(steering)
    log_path = Path(log)
    retained_locator = log_path.absolute()
    authority = JsonlRunLog.resume(retained_locator)
    memory_runtime: MemoryRuntime | None = None
    run_servers: list[McpServer] = []
    primary_error: BaseException | None = None
    try:
        state = _replay_run_log(flow, authority, api_key=api_key, headers=host_headers)
        if not _uses_host_key(state.effective_llm_by_static_path, state.call_time_llm):
            api_key = host_headers = None
        elif api_key is not None or host_headers is not None:
            # Checked before the terminal return too, since the backward handle sends them.
            _check_host_header_names(
                state.effective_llm_by_static_path, host_headers, state.call_time_llm
            )
        retained_program = copy.deepcopy(flow.compiled) if state.learnable else None
        if state.result is not None:
            return _result_from_state(state, log_path, [],
                                      handle=_retained_handle(
                                          retained_program, state, authority, memory,
                                          locator=retained_locator, api_key=api_key,
                                          headers=host_headers,
                                      ))
        # The checks that need no tools come before any server starts.
        _check_conditions(flow.compiled)
        _check_templates(flow.compiled)
        skills = _check_skills(flow.compiled, skills)
        # A started call-time agent resumes with the config it recorded; one not yet
        # started is checked when it resolves.
        in_flight = in_flight_llm_configs(state.flow.nodes, state.node_steps)
        recorded_llm = {**state.effective_llm_by_static_path, **in_flight}
        _check_recorded_llm_fields(
            recorded_llm, static_agent_labels(flow.compiled),
            _static_path_keys(state.flow.nodes, in_flight),
        )
        if check_llm_configs is not None:
            check_llm_configs(
                legacy_connection_map(recorded_llm),
                call_time_host=_selection_host(state.call_time_llm),
            )
        if state.memories is not None and memory is None:
            raise MemorySetupError(
                "resuming this run requires its enabled memory mapping"
            )
        run_servers = _open_run_servers(unopened, stop_event)
        run_tools = ToolSet(tools, context=context)
        _check_tool_grants(flow.compiled, run_tools)
        _check_tool_types(flow.compiled, run_tools)

        if state.memories is not None:
            _check_registry_types(memory, state.memories)
            memory_table, memory_origins, memory_stack_set = _resolve_authored_memory_layout(flow)
            _check_recorded_memory_table(
                state.memories, state.memory_stacks, memory_table, memory_stack_set,
                subject="supplied flow",
            )
            memory_runtime = MemoryRuntime.open(
                memories=state.memories,
                origins=memory_origins,
                memory=memory,
                snapshots=state.memory_snapshots,
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
                                  handle=_retained_handle(
                                      retained_program, state, authority, memory,
                                      locator=retained_locator, api_key=api_key,
                                      headers=host_headers,
                                  ))
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            _close_run_servers(run_servers, primary_error)
        finally:
            _close_run_resources(memory_runtime, authority, primary_error)


def _check_tool_grants(flow: CompiledFlow, tools: ToolSet) -> None:
    """Prove every static AGENT grant and TOOL `tool_id` names a tool this run supplies.

    An AGENT grant may name a tool group; a TOOL node must name one tool. Runs
    before any record is written and covers every static node, including nodes
    that may never run, so an unresolvable id raises `ValueError` from `run_flow`
    or `resume_flow` instead of failing its node when the node starts.

    Args:
        flow (`CompiledFlow`):
            The flow whose static nodes are checked.
        tools (`ToolSet`):
            The run's tools.

    Raises:
        `ValueError`:
            An AGENT grants a tool or group this run lacks, or `ask_user` with no
            handler; or a TOOL node names a tool this run lacks, a tool group, or
            `ask_user` with no handler.
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
        call = node.host_tool_call()
        if call is None:
            continue
        try:
            tools.resolve([call.tool_id])
        except ValueError as exc:
            if call.tool_id == "ask_user" or call.tool_id in tools.groups:
                reason = f"{call.tool_id!r}: {exc}"
            else:
                known = tools.tool_ids
                reason = (
                    f"tool {call.tool_id!r}, which isn't in this run's tools; "
                    f"known tools: {list(known)}"
                )
                close = difflib.get_close_matches(
                    call.tool_id, known, n=1, cutoff=CLOSE_MATCH_CUTOFF
                )
                if close:
                    reason += f"; did you mean {close[0]!r}?"
            raise ValueError(f"TOOL node {node_id!r} names {reason}") from None


def _check_conditions(flow: CompiledFlow) -> None:
    """Reject a condition that doesn't parse on any static node, before any record is written.

    A condition is an assertion, a CASE `when`, or a LOOP predicate. A YAML flow can't hold one
    that doesn't parse (Compose rejects it at load); this catches conditions set in Python.
    """
    for node_path, node in _static_nodes(flow):
        node.validate_conditions(path=node_path)


def _check_templates(flow: Any, path: tuple[str, ...] = ()) -> None:
    """Reject a binding or prompt span that can't run, before any record is written.

    Checks every static flow level: each template string of a wiring source (END's is
    the flow's output) must parse, pass each builtin arguments it takes, and name only
    constructors in the type table the binder evaluates it with (the reading node's; none
    for a reserved key such as WAIT `until` or MAP `over`); each node checks its prompts.
    A YAML flow can't hold such a span (Compose rejects it at load); this catches values
    set in Python.

    Args:
        flow (`Any`):
            A compiled flow or a static child flow.
        path (`tuple[str, ...]`, *optional*, defaults to `()`):
            The static path of the node that owns `flow`; empty for the top level.

    Raises:
        `ExpressionError`:
            Raised when a span can't run; the message names the node and binding, the flow
            output, or the prompt field.
    """
    owner = f"node {'/'.join(path)!r} " if path else ""
    for node_id, node in flow.nodes.items():
        node_path = "/".join((*path, node_id))
        reserved = node.reserved_wiring_keys()
        for key, binding in flow.wiring.get(node_id, {}).items():
            if node_id == flow.end_id:
                subject = f"{owner}{node.binding_labels.get(key, f'flow output {key!r}')}"
            elif key in node.binding_labels and key not in node.wiring_labels:
                # A desugared param (an inline call's argument, a value entry's input)
                # reads as its authored site; a CASE param keeps its wiring site name.
                subject = f"{owner}{node.binding_labels[key]}"
            elif key in reserved:
                subject = f"node {node_path!r} {node.reserved_key_label(key)}"
            elif node.binds_per_item:
                subject = f"node {node_path!r} map input {key!r}"
            else:
                subject = f"node {node_path!r} {node.wiring_ref_label(key).rstrip(':')}"
            # The binder resolves a reserved key with no type table.
            _check_template_source(binding, subject, {} if key in reserved else node.types)
        node.validate_prompts(path=node_path)
        for child in node.origin_children():
            _check_templates(child, (*path, node_id))


def _check_template_source(binding: Any, subject: str, types: Mapping[str, Any]) -> None:
    """Check each template string of one wiring source, prefixing an error with `subject`."""
    for _, text in iter_source_strings(binding):
        try:
            check_template_calls(text, types)
        except ExpressionError as exc:
            raise ExpressionError(f"{subject}: {exc}") from exc


def _check_tool_types(flow: CompiledFlow, tools: ToolSet) -> None:
    """Prove every static TOOL node's inputs and output agree with its tool's types.

    Runs before any record is written and covers every static node, including
    nodes that may never run.

    Args:
        flow (`CompiledFlow`):
            The flow whose static nodes are checked.
        tools (`ToolSet`):
            The run's tools.

    Raises:
        `ValueError`:
            A TOOL node passes an input its tool does not take, omits one the
            tool requires, binds an input whose type can never fit the tool's, or
            declares an output the tool can never return.
    """
    for node_id, node in _static_nodes(flow):
        call = node.host_tool_call()
        if call is None:
            continue
        tool = tools.resolve([call.tool_id])[0]
        declared = tool_types(tool)
        if declared.inputs is not None:
            _check_tool_inputs(node_id, call, declared.inputs, declared.required)
        if declared.output is not None:
            _check_tool_output(node_id, call, declared.output)


def _check_tool_inputs(
    node_id: str,
    call: HostToolCall,
    inputs: Mapping[str, Type | None],
    required: frozenset[str],
) -> None:
    """Prove one TOOL node's inputs match the names and types its tool declares.

    Args:
        node_id (`str`):
            The node's path, for the error message.
        call (`HostToolCall`):
            The node's tool call.
        inputs (`Mapping[str, Type | None]`):
            The tool's declared input types by name; `None` leaves an input unchecked.
        required (`frozenset[str]`):
            The names of the inputs the tool requires.

    Raises:
        `ValueError`:
            An input is not one the tool takes, a required input is missing, or an
            input's binding type is disjoint from the tool's. Missing inputs are
            all named at once.
    """
    for name in call.inputs:
        if name not in inputs:
            raise ValueError(
                f"TOOL node {node_id!r} input {name!r}: tool {call.tool_id!r} takes no "
                f"such input; it takes {sorted(inputs)}"
            )
    missing = sorted(required - set(call.inputs))
    if missing:
        named = f"input {missing[0]!r}" if len(missing) == 1 else f"inputs {missing}"
        raise ValueError(
            f"TOOL node {node_id!r} is missing {named} that tool {call.tool_id!r} requires"
        )
    for name, inferred in call.input_types.items():
        expected = inputs.get(name)
        # Disjoint is yian's non-coercing relation, so a binding pydantic would
        # coerce (a string of digits into an int) is rejected too.
        if expected is not None and type_relation(inferred, expected) == "disjoint":
            raise ValueError(
                f"TOOL node {node_id!r} input {name!r}: tool {call.tool_id!r} takes "
                f"{canonical_type_label(expected)}, but the binding is "
                f"{_binding_label(inferred)}"
            )


def _check_tool_output(node_id: str, call: HostToolCall, declared_output: Type) -> None:
    """Prove one TOOL node's declared output can hold a value its tool returns.

    Args:
        node_id (`str`):
            The node's path, for the error message.
        call (`HostToolCall`):
            The node's tool call; a call with no output type is not checked.
        declared_output (`Type`):
            The tool's declared return type.

    Raises:
        `ValueError`:
            The tool's return type is disjoint from the node's output type.
    """
    if call.output_type is None:
        return
    if type_relation(declared_output, call.output_type) == "disjoint":
        raise ValueError(
            f"TOOL node {node_id!r} output: declared "
            f"{canonical_type_label(call.output_type)}, but tool {call.tool_id!r} "
            f"returns {canonical_type_label(declared_output)}"
        )


_LITERAL_LABEL_WIDTH = 40


def _binding_label(typ: Type) -> str:
    """Return a binding type's label, showing a one-value string or int by its value."""
    if (
        typ.kind is ValueKind.INTEGER
        and typ.bounds is not None
        and typ.bounds[0] == typ.bounds[1]
        and not typ.nullable
    ):
        return f"{typ.bounds[0]} (int)"
    if (
        typ.kind is ValueKind.STRING
        and typ.tags is not None
        and len(typ.tags) == 1
        and typ.variants is None
        and not typ.nullable
    ):
        (value,) = typ.tags
        if len(value) > _LITERAL_LABEL_WIDTH:
            value = value[: _LITERAL_LABEL_WIDTH - 3] + "..."
        return f'"{value}" (str)'
    return canonical_type_label(typ)


def _static_path_keys(nodes: Mapping[str, Any], node_ids: Iterable[str]) -> dict[str, str]:
    """Map each runtime node id to its encoded static path, read from its execution scope.

    Args:
        nodes (`Mapping[str, Any]`): The run's live nodes, keyed by node id.
        node_ids (`Iterable[str]`): The runtime ids to map.

    Returns:
        `dict[str, str]`: The ``/``-joined static path of each id whose node has an
            execution scope.
    """
    keys: dict[str, str] = {}
    for node_id in node_ids:
        scope = nodes[node_id].execution_scope
        if scope is not None:
            keys[node_id] = "/".join(scope.static_path)
    return keys


def _check_recorded_llm_fields(
    effective: Mapping[str, Mapping[str, Any]], labels: Mapping[str, str],
    static_paths: Mapping[str, str] | None = None,
) -> None:
    """Check the public fields of each recorded agent config before a resume appends.

    A record written before a key was retired may carry it set; resuming would then
    run with a config the current carrier refuses. No key is required here, so the
    host-key marker is left out, and with it the base_url and connection settings
    bound to that key (replay has already checked them; the model call checks the URL
    against its connection and the header names against the headers given).

    Args:
        effective (`Mapping[str, Mapping[str, Any]]`):
            The recorded configs: the effective model map keyed by static agent path,
            and the config of each started call-time agent keyed by its runtime id.
        labels (`Mapping[str, str]`):
            The authored site of each synthesized agent's ``llm_config``, keyed by
            static agent path; a message names it instead of the path.
        static_paths (`Mapping[str, str] | None`, *optional*, defaults to `None`):
            The static path of each runtime id in ``effective``, under which its label
            is looked up; other keys are static paths already.

    Raises:
        `ValueError`: A config has an invalid or retired field; the message names its
            agent path or runtime id, or the authored site of a synthesized agent, and
            never echoes values.
    """
    static_paths = static_paths or {}
    for path, config in effective.items():
        host_only = (
            (HOST_KEY_MARKER, "base_url", "organization", "project", HEADER_NAMES_KEY)
            if HOST_KEY_MARKER in config
            else ()
        )
        public = {key: value for key, value in config.items() if key not in host_only}
        try:
            split_model_config(legacy_connection_config(public))
        except ValueError as exc:
            where = labels.get(static_paths.get(path, path)) or f"Agent {path!r}"
            raise ValueError(f"{where}: {exc}") from exc


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
    headers: dict[str, SecretStr] | None = None,
) -> RunState:
    """Silently reconstruct one borrowed, locked authoritative log prefix.

    The authored fingerprint is checked before copying.  The private graph gets
    only the complete effective LLM map and the call-time agents recorded by
    ``RunStarted``; recovery never consults the current LLM defaults.  Historical
    transitions use the same reducer and growth constructor as live execution.  A
    host ``api_key`` and the ``headers`` sent with it go to the agents the map marks
    for it, and to the call-time agents. The run result reads its files through the
    flow's output contract, as ``run_flow``'s does.
    """
    if not isinstance(flow, ComposedFlow):
        raise TypeError("_replay_run_log expects a ComposedFlow")
    return _replay_compiled_run_log(
        flow.compiled, log, api_key=api_key, headers=headers,
        result_contract=child_signature(flow).output,
    )


def _replay_compiled_run_log(
    flow: CompiledFlow, log: RunLog, *, api_key: SecretStr | None = None,
    headers: dict[str, SecretStr] | None = None,
    result_contract: OutputContract | None = None,
) -> RunState:
    """Reuse canonical replay for a private original compiled program.

    ``result_contract`` is the contract the terminal output is read through to
    restore its files (`RunState.result_contract`); a finished log materializes its
    terminal during replay, so it is set before any record is folded.
    """
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
    # Values like AGENT_RETRIES change the loaded flow, so a mismatch checked after
    # the fingerprint would surface as a fingerprint error instead of naming them.
    check_recorded(first.record.defaults)
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
        first.record.call_time_llm,
    )
    if api_key is not None:
        # A run that marked no agent ignores the key and headers: no static agent is
        # marked, and a call-time agent gets them only when its selection names the
        # key's provider, and uses them only when its resolved config is marked.
        bind_host_credentials(private_flow, api_key, headers)
    if first.record.learnable:
        apply_execution_layout(private_flow, resolve_execution_layout(private_flow))
    if first.record.memories is not None:
        replay_memories, _, replay_stacks = _resolve_runtime_memory_layout(private_flow)
        _check_recorded_memory_table(
            first.record.memories, first.record.memory_stacks, replay_memories, replay_stacks,
            subject="supplied flow",
        )
    state = _replay_records(
        RunState(private_flow, result_contract=result_contract),
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
    call_time_known: Mapping[str, str | None] | None = None,
) -> tuple[
    tuple[MemorySpec, ...],
    dict[str, OriginRef | None],
    tuple[MemoryStack, ...],
]:
    """Derive the key table, origins and stacks, and stamp one private graph.

    With ``effective_llm``, first check that each session key resolves one
    conversation compatibility across its placements; ``call_time_known`` gives the
    provider of each placement resolved only when it runs, when run start knows it
    (`check_session_compatibility`).
    """
    resolved = resolve_memory_layout(flow)
    if effective_llm is not None:
        check_session_compatibility(resolved, effective_llm, call_time_known=call_time_known)
    apply_memory_layout(flow, resolved)
    return _memory_layout_parts(resolved)


def _memory_layout_parts(
    resolved: ResolvedMemoryLayout,
) -> tuple[
    tuple[MemorySpec, ...],
    dict[str, OriginRef | None],
    tuple[MemoryStack, ...],
]:
    """Return one layout's record-facing key table, per-key origins and stacks."""
    return (
        tuple(memory.spec for memory in resolved.memories),
        {memory.spec.key: memory.origin for memory in resolved.memories},
        memory_stacks(resolved),
    )


def _check_registry_types(memory: Mapping[str, Memory], specs: tuple[MemorySpec, ...]) -> None:
    """Reject a registry whose declared types differ from this flow's table."""
    if not isinstance(memory, MemoryRegistry):
        return
    for spec in specs:
        built = memory.types.get(spec.key)
        if built is not None and built != spec.type:
            raise MemorySetupError(
                f"memory {spec.key!r} is declared as {spec.type!r} but the registry "
                f"built it as {built!r}; build the registry from the flow you run"
            )


def _resolve_authored_memory_layout(
    flow: ComposedFlow,
) -> tuple[
    tuple[MemorySpec, ...],
    dict[str, OriginRef | None],
    tuple[MemoryStack, ...],
]:
    """Purely resolve memory on the authored static program, never replayed growth.

    Nothing is stamped, so a def called from several nodes needs no de-aliasing;
    the table, origins and stacks equal those a fresh run records.
    """
    if not isinstance(flow, ComposedFlow):
        raise TypeError("memory layout resolution expects a ComposedFlow")
    return _memory_layout_parts(resolve_memory_layout(flow.compiled))


def _check_recorded_memory_table(
    recorded_memories: tuple[MemorySpec, ...] | None,
    recorded_stacks: tuple[MemoryStack, ...] | None,
    memories: tuple[MemorySpec, ...],
    stacks: tuple[MemoryStack, ...],
    *,
    subject: str,
) -> None:
    """Reject a recorded key table or stack set that the flow does not resolve."""
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
    program: CompiledFlow | None, state: RunState, log: RunLog, memory: Mapping[str, Memory] | None,
    *, locator: Path | None, api_key: SecretStr | None = None,
    headers: dict[str, SecretStr] | None = None,
) -> _BackwardHandle | None:
    """Detach delayed authority without keeping an opened session or public wrapper."""
    if program is None:
        return None
    return _BackwardHandle(program, state.run_id, locator, tuple(log) if log.path is None else (),
                           memory.copy() if type(memory) is dict else memory, state.backward_state.outcome,
                           api_key=api_key, headers=headers)


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


def _other_run_error(key: str) -> ValueError:
    """Return the error for the server under `key`, which another run has open or is opening."""
    return ValueError(
        f"tools[{key!r}]: this MCP server was opened by another run; "
        'open it with "with server:" to share it between runs'
    )


def _split_run_servers(
    tools: Any,
) -> tuple[Any, list[tuple[str, McpServer]]]:
    """Split the run's tools into the entries that need no opening and the unopened servers.

    An unopened server under a key `ToolSet` rejects (not a non-empty string, or
    `ask_user`) stays with the other entries, so `ToolSet` raises its usual error
    before any server starts.

    Args:
        tools (`Any`):
            The `tools=` argument of `run_flow` or `resume_flow`. Anything other than
            a mapping is returned as is, for `ToolSet` to reject.

    Returns:
        `tuple[Any, list[tuple[str, McpServer]]]`:
            `tools` without its unopened `McpServer` entries, and those servers in
            mapping order with the first key each sits under, each once even when it
            sits under several keys.

    Raises:
        `ValueError`:
            Raised when a server entry is open because another run in progress
            opened it.
    """
    if not isinstance(tools, Mapping):
        return tools, []
    others: dict[Any, Any] = {}
    unopened: list[tuple[str, McpServer]] = []
    for key, entry in tools.items():
        if not isinstance(entry, McpServer) or not (type(key) is str and key and key != "ask_user"):
            others[key] = entry
        elif entry.is_open:
            if entry._opened_by_run:
                raise _other_run_error(key)
            others[key] = entry
        elif not any(entry is seen for _, seen in unopened):
            unopened.append((key, entry))
    return others, unopened


def _open_run_servers(
    servers: Sequence[tuple[str, McpServer]], stop_event: threading.Event
) -> list[McpServer]:
    """Open the servers for one run, in order; if one fails, close the ones already open.

    Once `stop_event` is set, no further server opens: the ones already open close
    and the run stops as it would for a stop during its nodes.

    Args:
        servers (`Sequence[tuple[str, McpServer]]`):
            Unopened servers, each listed once with the `tools=` key it sits under.
        stop_event (`threading.Event`):
            The run's stop event.

    Returns:
        `list[McpServer]`:
            The opened servers, owned by a token unique to this run.

    Raises:
        `ValueError`:
            Raised when another caller opened or is opening a server since it was
            found unopened.
        `ImportError`:
            Raised when the MCP SDK is not installed.
        `McpConnectError`:
            Raised when a server doesn't start or doesn't list its tools.
        `_HostStop`:
            Raised when `stop_event` is set before every server is open.
    """
    token = f"run-{uuid.uuid4().hex}"
    opened: list[McpServer] = []
    try:
        for key, server in servers:
            if stop_event.is_set():
                # The same signal the engine raises for a stop it observes; it is
                # set before the run starts, so nothing is recorded.
                raise _HostStop(primary=False)
            try:
                server._open_for_run(token)
            except _AlreadyOpen as exc:
                raise _other_run_error(key) from exc
            opened.append(server)
    except BaseException as exc:
        _close_run_servers(opened, exc)
        raise
    return opened


def _close_run_servers(
    servers: Sequence[McpServer],
    primary_error: BaseException | None,
) -> None:
    """Close the run's servers while preserving one initiating exception.

    Args:
        servers (`Sequence[McpServer]`):
            The servers the run opened.
        primary_error (`BaseException | None`):
            The exception ending the run, which a close error is noted on. `None`
            raises the first close error instead.

    Raises:
        `BaseException`:
            Raised when `primary_error` is `None` and a close fails: the first close
            error, with any later ones noted on it.
    """
    pending = primary_error
    for server in servers:
        # close() never raises; this keeps an unexpected error from skipping the rest.
        try:
            server.close()
        except BaseException as cleanup_error:
            if pending is None:
                pending = cleanup_error
            else:
                _note_cleanup_error(pending, f"MCP server '{server.label}' close", cleanup_error)
    if primary_error is None and pending is not None:
        raise pending


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

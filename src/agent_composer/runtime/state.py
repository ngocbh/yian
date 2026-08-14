"""Authoritative mutable run state and append-before-apply transitions.

Preparation validates and canonicalizes a transition without touching live state.
``commit_prepared`` then makes its record durable before applying a frozen plan
whose remaining work is limited to total assignments and container operations.
The dispatcher will be the sole caller that mutates this state while a run is live.
The caller supplies a valid private graph and relinquishes direct mutation of its
execution fields; chronological reducer transitions are the only topology writers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from agent_composer._state import NodeState
from agent_composer.compile.model import CompiledFlow, Edge
from agent_composer.events import (
    NodeFailed,
    NodeRouted,
    NodeSkipped,
    NodeSucceeded,
    NodeStarted,
    RunFailed,
    RunResumed,
    RunStarted,
    RunSucceeded,
    SourceRange,
    SourceSpan,
)
from agent_composer.runtime.log import (
    SCHEMA_VERSION,
    NodeFailedRecord,
    NodeRoutedRecord,
    NodeSucceededRecord,
    NodeStartedRecord,
    RecordEnvelope,
    RunLog,
    RunRecord,
    RunResumedRecord,
    RunStartedRecord,
    canonical_seed,
    canonical_typed,
    decode_pool,
    decode_record,
    encode_pool,
    encode_record,
)
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.values import ANY_VALUE_ADAPTER, TypedValue


class RunState:
    """All mutable reducer-owned state for one private compiled graph.

    The supplied graph is already the caller's per-run copy and is adopted rather
    than copied again.  The caller must not mutate that graph's nodes, wiring, edges,
    boundaries, or adjacency directly after adoption; reducer transitions own later
    topology changes.  Initialization starts with no durable sequence applied;
    ``prepare_run_started`` constructs the first transition that installs the
    canonical pool, identity, configuration, and sole ready START node.

    Attributes:
        flow (`CompiledFlow`):
            Private static graph plus any later chronological expansion overlay.
        pool (`VariablePool`):
            Canonical typed values visible to node evaluation.
        node_state (`dict[str, NodeState]`):
            Disposition of every live node, initially all ``UNKNOWN``.
        edge_state (`dict[str, NodeState]`):
            Disposition of every live edge, initially all ``UNKNOWN``.
        ready (`deque[str]`):
            Dispatcher-owned FIFO of nodes eligible to start.
        in_flight (`dict[str, None]`):
            Ordered set of durably started nodes without a durable outcome.
        result (`RunSucceeded | RunFailed | None`):
            Derived terminal result, or ``None`` while the run remains live.
        depth (`dict[str, int]`):
            Reference-growth depth keyed by live dynamic node ID.
        wiring_overrides (`dict[str, dict[str, Any]]`):
            Per-run wiring keyed by node ID, beginning with START's narrowed inputs.
        run_id (`str | None`):
            Durable identity after ``RunStarted`` is applied.
        flow_fingerprint (`str | None`):
            Stable authored-flow digest restored from ``RunStarted``.
        next_sequence (`int`):
            Sequence expected for the next authoritative record.
        num_workers (`int`):
            Active dispatcher worker count; zero selects serial execution.
        input (`dict[str, Any]`):
            Canonical caller-supplied START priming record retained across START's commit.
        effective_llm_by_static_path (`dict[str, dict[str, Any]]`):
            Run-resolved agent configuration keyed by stable static node path.
    """

    def __init__(self, flow: CompiledFlow) -> None:
        """Adopt ``flow`` and seed empty reducer structures for its current topology."""
        if not isinstance(flow, CompiledFlow):
            raise TypeError("RunState expects a CompiledFlow")
        self.flow = flow
        self.pool = VariablePool(start_id=flow.start_id)
        self.node_state: dict[str, NodeState] = {
            node_id: NodeState.UNKNOWN for node_id in flow.nodes
        }
        self.edge_state: dict[str, NodeState] = {
            edge.id: NodeState.UNKNOWN for edge in flow.edges
        }
        self.ready: deque[str] = deque()
        self.in_flight: dict[str, None] = {}
        self.result: RunSucceeded | RunFailed | None = None
        self.depth: dict[str, int] = {}
        self.wiring_overrides: dict[str, dict[str, Any]] = {}
        self.run_id: str | None = None
        self.flow_fingerprint: str | None = None
        self.next_sequence = 0
        self.num_workers = 0
        self.input: dict[str, Any] = {}
        self.effective_llm_by_static_path: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True, slots=True)
class RunStartData:
    """Fully resolved inputs used to prepare the first authoritative record.

    Attributes:
        run_id (`str`):
            Non-empty identity shared by the pool, state, and enclosing log.
        flow_fingerprint (`str`):
            Lowercase SHA-256 identity of the authored flow.
        initial_pool (`VariablePool`):
            Complete pool including system values and the primed START record.
        start_wiring_override (`dict[str, Any]`):
            START's narrowed per-run wiring.
        effective_llm_by_static_path (`dict[str, dict[str, Any]]`):
            Complete resolved LLM configuration by stable static path.
        num_workers (`int`):
            Exact non-negative dispatcher worker count.
    """

    run_id: str
    flow_fingerprint: str
    initial_pool: VariablePool
    start_wiring_override: dict[str, Any]
    effective_llm_by_static_path: dict[str, dict[str, Any]]
    num_workers: int


@dataclass(frozen=True, slots=True)
class _RunStartedPlan:
    """Opaque frozen identity anchors and bytes for one initialization apply.

    The two object references bind authority, while every transition payload is
    retained only as immutable bytes or scalar boundary identity.
    """

    originating_state: RunState
    originating_flow: CompiledFlow
    envelope_bytes: bytes
    start_id: str
    start_node_identity: int
    end_id: str
    end_node_identity: int | None


@dataclass(slots=True)
class _RunStartedApply:
    """Detached mutable containers fully materialized before durable append."""

    pool: VariablePool
    node_state: dict[str, NodeState]
    edge_state: dict[str, NodeState]
    ready: deque[str]
    in_flight: dict[str, None]
    depth: dict[str, int]
    wiring_overrides: dict[str, dict[str, Any]]
    run_id: str
    flow_fingerprint: str
    next_sequence: int
    num_workers: int
    input: dict[str, Any]
    effective_llm_by_static_path: dict[str, dict[str, Any]]
    observers: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _FrontierPlan:
    """Opaque authority anchors and one immutable dispatch/recovery delta.

    The private single-writer reducer owns the frontier between records.  State/flow
    identity plus the next durable coordinate therefore make a complete stale-plan
    authority; ``delta`` contains only the already-validated touched coordinates.
    """

    originating_state: RunState
    originating_flow: CompiledFlow
    envelope_bytes: bytes
    start_id: str
    start_node_identity: int
    end_id: str
    end_node_identity: int | None
    delta: _NodeStartedDelta | _RunResumedDelta


@dataclass(frozen=True, slots=True)
class _NodeStartedDelta:
    """Prevalidated in-place operations for dispatching one ready head."""

    node_id: str


@dataclass(frozen=True, slots=True)
class _RunResumedDelta:
    """Prevalidated in-place operations for restoring one retry frontier."""

    retry_node_ids: tuple[str, ...]
    num_workers: int


@dataclass(frozen=True, slots=True)
class _SuccessPlan:
    """Opaque authority anchors and one immutable successful-output delta."""

    originating_state: RunState
    originating_flow: CompiledFlow
    envelope_bytes: bytes
    start_id: str
    start_node_identity: int
    end_id: str
    end_node_identity: int | None
    delta: _SuccessDelta
    terminal: _TerminalDelta | None


@dataclass(frozen=True, slots=True)
class _SuccessDelta:
    """Prevalidated touched coordinates for one successful output commit."""

    node_id: str
    commit_target: str
    value_bytes: bytes
    outgoing_edge_ids: tuple[str, ...]
    newly_ready_ids: tuple[str, ...]
    observer_commit_as: str | None


@dataclass(slots=True)
class _SuccessApply:
    """Independent success value, terminal result, and observers before append."""

    value: TypedValue
    terminal: _TerminalApply | None
    observers: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _RoutePlan:
    """Opaque authority anchors and one immutable route/skip delta."""

    originating_state: RunState
    originating_flow: CompiledFlow
    envelope_bytes: bytes
    start_id: str
    start_node_identity: int
    end_id: str
    end_node_identity: int | None
    delta: _RouteDelta
    terminal: _TerminalDelta | None


@dataclass(frozen=True, slots=True)
class _RouteDelta:
    """Prevalidated ordered operations for one route and its skip closure."""

    node_id: str
    handle: str
    edge_updates: tuple[tuple[str, NodeState], ...]
    skipped_node_ids: tuple[str, ...]
    newly_ready_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TerminalDelta:
    """Immutable derived terminal authority attached to one causal outcome.

    ``output_bytes`` is a canonical tagged typed value for success and ``None``
    for ``TerminalSkipped`` failure.  ``succeeded`` distinguishes a typed null
    output from the failure case; ``skipped_output_name`` preserves the legacy
    ``None`` fallback when no required dead END group can be identified.
    """

    succeeded: bool
    output_bytes: bytes | None
    skipped_output_name: str | None


@dataclass(slots=True)
class _TerminalApply:
    """Independent state result and returned observer materialized before append."""

    result: RunSucceeded | RunFailed
    observer: RunSucceeded | RunFailed


@dataclass(slots=True)
class _RouteApply:
    """Independent route observers and optional terminal result before append."""

    terminal: _TerminalApply | None
    observers: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _FailurePlan:
    """Opaque authority anchors and one immutable terminal-failure delta."""

    originating_state: RunState
    originating_flow: CompiledFlow
    envelope_bytes: bytes
    start_id: str
    start_node_identity: int
    end_id: str
    end_node_identity: int | None
    delta: _FailureDelta


@dataclass(frozen=True, slots=True)
class _FailureDelta:
    """Canonical diagnostic and touched node for one terminal failure."""

    node_id: str
    error: str
    error_type: str
    locator: SourceSpan | None
    range: SourceRange | None
    traceback: str | None


@dataclass(slots=True)
class _FailureApply:
    """Independent terminal result and observers materialized before append."""

    result: RunFailed
    observers: tuple[Any, ...]


@dataclass(slots=True)
class _DataGroupScratch:
    """Incremental counts for one data-input group in a touched join."""

    optional: bool
    total: int = 0
    skipped: int = 0


@dataclass(slots=True)
class _JoinScratch:
    """One incoming-adjacency scan reduced to constant-time edge updates."""

    states: dict[str, NodeState]
    classes: dict[str, tuple[str, str | None]]
    unknown: int
    control_total: int
    control_skipped: int
    required_data_dead: int
    required_ordering_skipped: int
    data_groups: dict[str | None, _DataGroupScratch]


@dataclass(frozen=True, slots=True)
class PreparedTransition:
    """One canonical record plus its already-validated total state-change plan.

    ``_plan`` is deliberately private: callers may inspect the durable record and
    expected authority coordinates, but reducer implementation details are not a
    construction surface.  Observer notifications are returned only after append
    and apply both succeed.
    """

    record: RunRecord
    expected_run_id: str
    expected_sequence: int
    _plan: (
        _RunStartedPlan
        | _FrontierPlan
        | _SuccessPlan
        | _RoutePlan
        | _FailurePlan
    )
    observers: tuple[Any, ...]


def prepare_run_started(state: RunState, data: RunStartData) -> PreparedTransition:
    """Prepare canonical initialization without mutating ``state``.

    Args:
        state (`RunState`):
            A pristine state whose private graph contains its synthesized START node.
        data (`RunStartData`):
            Complete pool, identity, configuration, and worker selection for the run.

    Returns:
        `PreparedTransition`:
            The canonical ``RunStartedRecord``, frozen total plan, and transient observer.

    Raises:
        `TypeError`:
            The carrier or worker count has the wrong exact type.
        `ValueError`:
            State is not pristine, identity disagrees across inputs, START is missing
            or unprimed, or a field violates the public run-record wire contract.
    """
    if type(state) is not RunState:
        raise TypeError("prepare_run_started expects a RunState")
    if not isinstance(data, RunStartData):
        raise TypeError("prepare_run_started expects RunStartData")
    _require_pristine(state)
    start_id = state.flow.start_id
    if start_id not in state.flow.nodes:
        raise ValueError(f"flow START node {start_id!r} is missing")
    if type(data.num_workers) is not int:
        raise TypeError("num_workers must be a non-negative integer")
    if data.num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")

    # Encode then decode before any identity checks so live initialization and replay
    # both reason about the same strict durable pool representation.
    canonical_pool = decode_pool(encode_pool(data.initial_pool))
    if canonical_pool.start_id != start_id:
        raise ValueError(
            "initial pool start_id mismatch: "
            f"expected {start_id!r}, found {canonical_pool.start_id!r}"
        )
    if canonical_pool.resolve("system", ["run_id"]) != data.run_id:
        raise ValueError("initial pool system.run_id must equal RunStartData.run_id")
    primed_start = canonical_pool.get(start_id)
    if not isinstance(primed_start, dict):
        raise ValueError("initial pool primed START value must be a dict")

    candidate = RunStartedRecord(
        flow_fingerprint=data.flow_fingerprint,
        initial_pool=encode_pool(canonical_pool),
        start_wiring_override=data.start_wiring_override,
        effective_llm_by_static_path=data.effective_llm_by_static_path,
        num_workers=data.num_workers,
    )
    # The public envelope codec is the sole payload authority.  Decode its canonical
    # bytes immediately so the prepared record is exactly what either log will retain.
    canonical_bytes = encode_record(
        RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=data.run_id,
            seq=state.next_sequence,
            record=candidate,
        )
    )
    canonical_envelope = decode_record(canonical_bytes)
    record = canonical_envelope.record
    if not isinstance(record, RunStartedRecord):  # exhaustive codec invariant
        raise TypeError("run-start envelope decoded to a different record kind")

    plan = _RunStartedPlan(
        originating_state=state,
        originating_flow=state.flow,
        envelope_bytes=canonical_bytes,
        start_id=start_id,
        start_node_identity=id(state.flow.nodes[start_id]),
        end_id=state.flow.end_id,
        end_node_identity=(
            id(state.flow.nodes[state.flow.end_id])
            if state.flow.end_id in state.flow.nodes
            else None
        ),
    )
    return PreparedTransition(
        record=record,
        expected_run_id=canonical_envelope.run_id,
        expected_sequence=canonical_envelope.seq,
        _plan=plan,
        observers=(RunStarted(),),
    )


def prepare_node_started(state: RunState, node_id: str) -> PreparedTransition:
    """Prepare moving the exact ready-head node into ordered in-flight work.

    Args:
        state (`RunState`):
            Initialized nonterminal state whose ready head is being dispatched.
        node_id (`str`):
            Exact non-empty ID at the head of ``state.ready``.

    Returns:
        `PreparedTransition`:
            Canonical ``NodeStartedRecord`` plus a detached append-before-apply plan.

    Raises:
        `TypeError`:
            ``state`` or ``node_id`` has the wrong exact type.
        `ValueError`:
            State is uninitialized, terminal, or does not expose ``node_id`` as one
            unknown, non-in-flight ready head.
    """
    if type(state) is not RunState:
        raise TypeError("prepare_node_started expects a RunState")
    if type(node_id) is not str:
        raise TypeError("node_id must be a string")
    return prepare_record(state, NodeStartedRecord(node_id=node_id))


def prepare_node_succeeded(
    state: RunState,
    node_id: str,
    output: Any,
    *,
    commit_as: str | None = None,
) -> PreparedTransition:
    """Prepare one canonical output commit and its newly-ready successors.

    Args:
        state (`RunState`):
            Initialized nonterminal state with ``node_id`` in ordered in-flight work.
        node_id (`str`):
            Exact ID of the executing node that produced ``output``.
        output (`Any`):
            Plain or typed value to validate against the effective target's output type.
        commit_as (`str | None`, *optional*):
            Explicit commit target.  ``None`` falls back to the executing node's baked
            target and then to ``node_id`` itself.

    Returns:
        `PreparedTransition`:
            Canonical ``NodeSucceededRecord`` and detached append-before-apply plan.

    Raises:
        `TypeError`:
            A carrier has the wrong exact type or ``output`` cannot be typed.
        `ValueError`:
            The executing node/target is invalid or ``output`` violates its type.
    """
    if type(state) is not RunState:
        raise TypeError("prepare_node_succeeded expects a RunState")
    if type(node_id) is not str:
        raise TypeError("node_id must be a string")
    if commit_as is not None and type(commit_as) is not str:
        raise TypeError("commit_as must be a string or None")
    if commit_as == "":
        raise ValueError("commit_as must be non-empty when supplied")
    _require_initialized_nonterminal(state)
    _require_succeeding_node(state, node_id)

    node_target = state.flow.nodes[node_id].commit_as
    if node_target is not None and (type(node_target) is not str or not node_target):
        raise ValueError(f"node {node_id!r} has an invalid baked commit target")
    target = commit_as if commit_as is not None else node_target or node_id
    if target not in state.flow.nodes:
        raise ValueError(f"success commit target {target!r} does not exist in the flow")
    value_wire, typed = canonical_typed(
        output,
        state.flow.nodes[target].output_type,
    )
    return _prepare_success_transition(
        state,
        NodeSucceededRecord(
            node_id=node_id,
            commit_target=target,
            value=value_wire,
        ),
        typed,
    )


def prepare_node_routed(
    state: RunState,
    node_id: str,
    handle: str,
) -> PreparedTransition:
    """Prepare one selected route and its derived recursive skip closure.

    Args:
        state (`RunState`):
            Initialized nonterminal state with ``node_id`` in ordered in-flight work.
        node_id (`str`):
            Exact non-empty ID of the executing router.
        handle (`str`):
            Exact non-empty handle selected by the router.

    Returns:
        `PreparedTransition`:
            Canonical ``NodeRoutedRecord`` and detached append-before-apply plan.

    Raises:
        `TypeError`:
            A carrier has the wrong exact type.
        `ValueError`:
            The router/frontier is invalid or no outgoing edge matches ``handle``.
    """
    if type(state) is not RunState:
        raise TypeError("prepare_node_routed expects a RunState")
    if type(node_id) is not str:
        raise TypeError("node_id must be a string")
    if not node_id:
        raise ValueError("node_id must be non-empty")
    if type(handle) is not str:
        raise TypeError("handle must be a string")
    if not handle:
        raise ValueError("handle must be non-empty")
    _require_initialized_nonterminal(state)
    _require_routing_node(state, node_id)
    return _prepare_route_transition(
        state,
        NodeRoutedRecord(node_id=node_id, handle=handle),
    )


def prepare_node_failed(
    state: RunState,
    node_id: str,
    error: str,
    *,
    error_type: str = "",
    locator: SourceSpan | None = None,
    range: SourceRange | None = None,
    traceback: str | None = None,
) -> PreparedTransition:
    """Prepare one terminal node failure with its complete durable diagnostic.

    Args:
        state (`RunState`):
            Initialized nonterminal state with ``node_id`` in ordered in-flight work.
        node_id (`str`):
            Exact non-empty ID of the executing node that failed.
        error (`str`):
            Human-readable failure detail, including the empty string when intentional.
        error_type (`str`, *optional*):
            Stable exception or synthesized diagnostic category.
        locator (`SourceSpan | None`, *optional*):
            Symbolic source location retained alongside the failure.
        range (`SourceRange | None`, *optional*):
            Concrete source range retained alongside the failure.
        traceback (`str | None`, *optional*):
            Formatted traceback captured at the node boundary.

    Returns:
        `PreparedTransition`:
            Canonical ``NodeFailedRecord`` and detached append-before-apply plan.

    Raises:
        `TypeError`:
            A scalar or diagnostic carrier has the wrong exact type.
        `ValueError`:
            State or the executing node is not eligible to fail.
    """
    if type(state) is not RunState:
        raise TypeError("prepare_node_failed expects a RunState")
    record = NodeFailedRecord(
        node_id=node_id,
        error=error,
        error_type=error_type,
        locator=locator,
        range=range,
        traceback=traceback,
    )
    _require_initialized_nonterminal(state)
    return _prepare_failure_transition(state, record)


def prepare_run_resumed(
    state: RunState,
    *,
    num_workers: int,
) -> PreparedTransition:
    """Prepare a recovery segment from the state's ordered unfinished starts.

    The retry list is derived exclusively from insertion-ordered ``state.in_flight``;
    callers may choose only the exact non-negative worker count for the new segment.

    Args:
        state (`RunState`):
            Initialized nonterminal state reconstructed through a complete log prefix.
        num_workers (`int`):
            Exact non-negative worker count; zero selects serial execution.

    Returns:
        `PreparedTransition`:
            Canonical ``RunResumedRecord`` whose retries preserve dispatch order.
    """
    if type(state) is not RunState:
        raise TypeError("prepare_run_resumed expects a RunState")
    if type(num_workers) is not int:
        raise TypeError("num_workers must be a non-negative integer")
    if num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    _require_initialized_nonterminal(state)
    return prepare_record(
        state,
        RunResumedRecord(
            retry_node_ids=tuple(state.in_flight),
            num_workers=num_workers,
        ),
    )


def prepare_record(state: RunState, record: RunRecord) -> PreparedTransition:
    """Validate one supported authoritative record against the current frontier.

    This is the shared reducer seam for live dispatch and later historical replay.
    Dispatch, recovery-frontier, successful-output, route, and terminal-failure
    records are supported here; the remaining node outcomes acquire reducers in
    later slices.

    Args:
        state (`RunState`):
            Initialized nonterminal state immediately before ``record``.
        record (`RunRecord`):
            A ``NodeStartedRecord``, ``RunResumedRecord``,
            ``NodeSucceededRecord``, ``NodeRoutedRecord``, or ``NodeFailedRecord``
            to validate and copy.

    Returns:
        `PreparedTransition`:
            Canonically copied record, immutable authority snapshots, and observer.

    Raises:
        `TypeError`:
            State has the wrong type or the record kind is not implemented yet.
        `ValueError`:
            The record is not the exact next transition for this state.
    """
    if type(state) is not RunState:
        raise TypeError("prepare_record expects a RunState")
    _require_initialized_nonterminal(state)

    if isinstance(record, NodeSucceededRecord):
        typed = _validate_success_value(state, record)
        return _prepare_success_transition(state, record, typed)
    if isinstance(record, NodeRoutedRecord):
        _validate_route_record(state, record)
        return _prepare_route_transition(state, record)
    if isinstance(record, NodeFailedRecord):
        return _prepare_failure_transition(state, record)
    if isinstance(record, NodeStartedRecord):
        _validate_node_started(state, record)
    elif isinstance(record, RunResumedRecord):
        _validate_run_resumed(state, record)
    else:
        raise TypeError(
            f"unsupported run record for state preparation: {type(record).__name__}"
        )

    run_id = state.run_id
    if not isinstance(run_id, str):  # established by initialized-state validation
        raise ValueError("run state is not initialized")
    canonical_bytes = encode_record(
        RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=run_id,
            seq=state.next_sequence,
            record=record,
        )
    )
    canonical_envelope = decode_record(canonical_bytes)
    canonical_record = canonical_envelope.record
    if isinstance(record, NodeStartedRecord):
        if not isinstance(canonical_record, NodeStartedRecord):
            raise TypeError("node-start envelope decoded to a different record kind")
        _validate_node_started(state, canonical_record)
        observers: tuple[Any, ...] = (NodeStarted(node_id=canonical_record.node_id),)
    elif isinstance(record, RunResumedRecord):
        if not isinstance(canonical_record, RunResumedRecord):
            raise TypeError("run-resume envelope decoded to a different record kind")
        _validate_run_resumed(state, canonical_record)
        observers = (RunResumed(),)
    else:  # protected by the supported-record dispatch above
        raise TypeError("frontier envelope decoded to a different record kind")

    start_id = state.flow.start_id
    end_id = state.flow.end_id
    common_plan = {
        "originating_state": state,
        "originating_flow": state.flow,
        "envelope_bytes": canonical_bytes,
        "start_id": start_id,
        "start_node_identity": id(state.flow.nodes[start_id]),
        "end_id": end_id,
        "end_node_identity": (
            id(state.flow.nodes[end_id]) if end_id in state.flow.nodes else None
        ),
    }
    if isinstance(canonical_record, NodeStartedRecord):
        plan = _FrontierPlan(
            **common_plan,
            delta=_NodeStartedDelta(node_id=canonical_record.node_id),
        )
    else:
        plan = _FrontierPlan(
            **common_plan,
            delta=_RunResumedDelta(
                retry_node_ids=canonical_record.retry_node_ids,
                num_workers=canonical_record.num_workers,
            ),
        )
    return PreparedTransition(
        record=canonical_record,
        expected_run_id=canonical_envelope.run_id,
        expected_sequence=canonical_envelope.seq,
        _plan=plan,
        observers=observers,
    )


def _prepare_success_transition(
    state: RunState,
    record: NodeSucceededRecord,
    typed: TypedValue,
) -> PreparedTransition:
    """Freeze one fully validated success delta and its canonical public record."""
    delta = _prepare_success_delta(state, record, typed)
    terminal = _prepare_terminal_delta(
        state,
        completing_node_id=delta.node_id,
        newly_ready_ids=delta.newly_ready_ids,
        pending_commit_target=delta.commit_target,
        pending_value_bytes=delta.value_bytes,
        edge_updates={},
    )
    run_id = state.run_id
    if not isinstance(run_id, str):  # established by initialized-state validation
        raise ValueError("run state is not initialized")
    canonical_bytes = encode_record(
        RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=run_id,
            seq=state.next_sequence,
            record=record,
        )
    )
    canonical_envelope = decode_record(canonical_bytes)
    canonical_record = canonical_envelope.record
    if not isinstance(canonical_record, NodeSucceededRecord):
        raise TypeError("node-success envelope decoded to a different record kind")
    if canonical_record != record:
        raise ValueError("NodeSucceededRecord is not in canonical wire form")

    start_id = state.flow.start_id
    end_id = state.flow.end_id
    plan = _SuccessPlan(
        originating_state=state,
        originating_flow=state.flow,
        envelope_bytes=canonical_bytes,
        start_id=start_id,
        start_node_identity=id(state.flow.nodes[start_id]),
        end_id=end_id,
        end_node_identity=(
            id(state.flow.nodes[end_id]) if end_id in state.flow.nodes else None
        ),
        delta=delta,
        terminal=terminal,
    )
    observer_value = ANY_VALUE_ADAPTER.validate_json(
        delta.value_bytes,
        strict=True,
    ).to_object()
    observer = NodeSucceeded(
        node_id=delta.node_id,
        output=observer_value,
        commit_as=delta.observer_commit_as,
    )
    observers: tuple[Any, ...] = (observer,)
    if terminal is not None:
        observers += (_terminal_observer(terminal),)
    return PreparedTransition(
        record=canonical_record,
        expected_run_id=canonical_envelope.run_id,
        expected_sequence=canonical_envelope.seq,
        _plan=plan,
        observers=observers,
    )


def _prepare_route_transition(
    state: RunState,
    record: NodeRoutedRecord,
) -> PreparedTransition:
    """Freeze one fully validated route delta and its canonical public record."""
    delta, edge_updates = _prepare_route_delta(state, record)
    terminal = _prepare_terminal_delta(
        state,
        completing_node_id=delta.node_id,
        newly_ready_ids=delta.newly_ready_ids,
        pending_commit_target=None,
        pending_value_bytes=None,
        edge_updates=edge_updates,
    )
    run_id = state.run_id
    if not isinstance(run_id, str):  # established by initialized-state validation
        raise ValueError("run state is not initialized")
    canonical_bytes = encode_record(
        RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=run_id,
            seq=state.next_sequence,
            record=record,
        )
    )
    canonical_envelope = decode_record(canonical_bytes)
    canonical_record = canonical_envelope.record
    if not isinstance(canonical_record, NodeRoutedRecord):
        raise TypeError("node-route envelope decoded to a different record kind")
    if canonical_record != record:
        raise ValueError("NodeRoutedRecord is not in canonical wire form")

    start_id = state.flow.start_id
    end_id = state.flow.end_id
    plan = _RoutePlan(
        originating_state=state,
        originating_flow=state.flow,
        envelope_bytes=canonical_bytes,
        start_id=start_id,
        start_node_identity=id(state.flow.nodes[start_id]),
        end_id=end_id,
        end_node_identity=(
            id(state.flow.nodes[end_id]) if end_id in state.flow.nodes else None
        ),
        delta=delta,
        terminal=terminal,
    )
    observers = (NodeRouted(delta.node_id, delta.handle),) + tuple(
        NodeSkipped(node_id) for node_id in delta.skipped_node_ids
    )
    if terminal is not None:
        observers += (_terminal_observer(terminal),)
    return PreparedTransition(
        record=canonical_record,
        expected_run_id=canonical_envelope.run_id,
        expected_sequence=canonical_envelope.seq,
        _plan=plan,
        observers=observers,
    )


def _prepare_failure_transition(
    state: RunState,
    record: NodeFailedRecord,
) -> PreparedTransition:
    """Freeze one validated terminal diagnostic and its canonical public record."""
    _validate_failure_record(state, record)
    run_id = state.run_id
    if not isinstance(run_id, str):  # established by initialized-state validation
        raise ValueError("run state is not initialized")
    canonical_bytes = encode_record(
        RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=run_id,
            seq=state.next_sequence,
            record=record,
        )
    )
    canonical_envelope = decode_record(canonical_bytes)
    canonical_record = canonical_envelope.record
    if not isinstance(canonical_record, NodeFailedRecord):
        raise TypeError("node-failure envelope decoded to a different record kind")
    if canonical_record != record:
        raise ValueError("NodeFailedRecord is not in canonical wire form")

    delta = _failure_delta(canonical_record)
    start_id = state.flow.start_id
    end_id = state.flow.end_id
    plan = _FailurePlan(
        originating_state=state,
        originating_flow=state.flow,
        envelope_bytes=canonical_bytes,
        start_id=start_id,
        start_node_identity=id(state.flow.nodes[start_id]),
        end_id=end_id,
        end_node_identity=(
            id(state.flow.nodes[end_id]) if end_id in state.flow.nodes else None
        ),
        delta=delta,
    )
    return PreparedTransition(
        record=canonical_record,
        expected_run_id=canonical_envelope.run_id,
        expected_sequence=canonical_envelope.seq,
        _plan=plan,
        observers=_failure_observers(delta),
    )


def commit_prepared(
    log: RunLog,
    state: RunState,
    prepared: PreparedTransition,
) -> tuple[Any, ...]:
    """Append one prepared record, verify the log result, then apply its total plan.

    Args:
        log (`RunLog`):
            Borrowed single-run authority; this function never closes it.
        state (`RunState`):
            Live state at the sequence and identity captured during preparation.
        prepared (`PreparedTransition`):
            Canonical record and frozen apply plan produced against ``state``.

    Returns:
        `tuple[Any, ...]`:
            Transient observers made visible only after durable append and total apply.

    Raises:
        `ValueError`:
            State, log, preparation, or the log's append result disagrees on identity,
            sequence, or the exact record.
        `TypeError`:
            The transition contains an unsupported private plan.
        `BaseException`:
            The log's append failure, propagated before any state change.

    Notes:
        A log that durably appends but returns a different envelope has violated the
        `RunLog` protocol.  The live state remains unapplied; the caller must discard
        it and reconstruct from the log's durable prefix.  This function cannot and
        does not attempt to roll durable authority back.
    """
    if type(state) is not RunState:
        raise TypeError("commit_prepared expects a RunState")
    if not isinstance(prepared, PreparedTransition):
        raise TypeError("commit_prepared expects a PreparedTransition")
    if isinstance(prepared._plan, _RunStartedPlan):
        return _commit_run_started(log, state, prepared)
    if isinstance(prepared._plan, _FrontierPlan):
        return _commit_frontier(log, state, prepared)
    if isinstance(prepared._plan, _SuccessPlan):
        return _commit_success(log, state, prepared)
    if isinstance(prepared._plan, _RoutePlan):
        return _commit_route(log, state, prepared)
    if isinstance(prepared._plan, _FailurePlan):
        return _commit_failure(log, state, prepared)
    raise TypeError("unsupported prepared transition plan")


def _commit_run_started(
    log: RunLog,
    state: RunState,
    prepared: PreparedTransition,
) -> tuple[Any, ...]:
    """Validate, append, and install one initialization transition."""
    plan = prepared._plan
    if not isinstance(plan, _RunStartedPlan):
        raise TypeError("RunStarted commit requires a RunStarted plan")
    if not isinstance(prepared.record, RunStartedRecord):
        raise TypeError("RunStarted plan requires a RunStartedRecord")
    _require_prepared_origin(state, plan)
    _require_pristine(state)
    _require_authority_coordinates(log, state, prepared, initialized=False)
    _require_unchanged_record(prepared, plan.envelope_bytes)
    apply = _materialize_run_started(state, prepared)

    _append_and_verify(log, prepared)
    _apply_run_started(state, apply)
    return apply.observers


def _commit_frontier(
    log: RunLog,
    state: RunState,
    prepared: PreparedTransition,
) -> tuple[Any, ...]:
    """Validate, append, and apply one sparse dispatch or recovery delta."""
    plan = prepared._plan
    if not isinstance(plan, _FrontierPlan):
        raise TypeError("frontier commit requires a frontier plan")
    if not isinstance(
        prepared.record,
        (NodeStartedRecord, RunResumedRecord),
    ):
        raise TypeError("frontier plan requires a dispatch or resume record")
    _require_prepared_origin(state, plan)
    _require_initialized_nonterminal(state)
    _require_authority_coordinates(log, state, prepared, initialized=True)
    _require_unchanged_record(prepared, plan.envelope_bytes)
    delta = plan.delta
    if isinstance(prepared.record, NodeStartedRecord) and isinstance(
        delta, _NodeStartedDelta
    ):
        observers: tuple[Any, ...] = (NodeStarted(node_id=delta.node_id),)
    elif isinstance(prepared.record, RunResumedRecord) and isinstance(
        delta, _RunResumedDelta
    ):
        observers = (RunResumed(),)
    else:
        raise TypeError("frontier record and prevalidated delta disagree")

    _append_and_verify(log, prepared)
    if isinstance(delta, _NodeStartedDelta):
        state.ready.popleft()
        state.in_flight[delta.node_id] = None
    else:
        state.ready.extendleft(reversed(delta.retry_node_ids))
        state.in_flight.clear()
        state.num_workers = delta.num_workers
    state.next_sequence = prepared.expected_sequence + 1
    return observers


def _commit_success(
    log: RunLog,
    state: RunState,
    prepared: PreparedTransition,
) -> tuple[Any, ...]:
    """Validate, append, and install one successful outcome transition."""
    plan = prepared._plan
    if not isinstance(plan, _SuccessPlan):
        raise TypeError("success commit requires a success plan")
    if not isinstance(prepared.record, NodeSucceededRecord):
        raise TypeError("success plan requires a NodeSucceededRecord")
    _require_prepared_origin(state, plan)
    _require_initialized_nonterminal(state)
    _require_authority_coordinates(log, state, prepared, initialized=True)
    _require_unchanged_record(prepared, plan.envelope_bytes)
    delta = plan.delta
    if not (
        prepared.record.node_id == delta.node_id
        and prepared.record.commit_target == delta.commit_target
    ):
        raise ValueError("success record and prevalidated delta disagree")
    apply = _materialize_success(plan)

    _append_and_verify(log, prepared)
    state.pool.store[delta.commit_target] = apply.value
    state.node_state[delta.node_id] = NodeState.TAKEN
    for edge_id in delta.outgoing_edge_ids:
        state.edge_state[edge_id] = NodeState.TAKEN
    del state.in_flight[delta.node_id]
    state.ready.extend(delta.newly_ready_ids)
    if apply.terminal is not None:
        state.result = apply.terminal.result
    state.next_sequence = prepared.expected_sequence + 1
    return apply.observers


def _commit_route(
    log: RunLog,
    state: RunState,
    prepared: PreparedTransition,
) -> tuple[Any, ...]:
    """Validate, append, and install one route plus derived skip closure."""
    plan = prepared._plan
    if not isinstance(plan, _RoutePlan):
        raise TypeError("route commit requires a route plan")
    if not isinstance(prepared.record, NodeRoutedRecord):
        raise TypeError("route plan requires a NodeRoutedRecord")
    _require_prepared_origin(state, plan)
    _require_initialized_nonterminal(state)
    _require_authority_coordinates(log, state, prepared, initialized=True)
    _require_unchanged_record(prepared, plan.envelope_bytes)
    delta = plan.delta
    if not (
        prepared.record.node_id == delta.node_id
        and prepared.record.handle == delta.handle
    ):
        raise ValueError("route record and prevalidated delta disagree")
    apply = _materialize_route(plan)

    _append_and_verify(log, prepared)
    state.node_state[delta.node_id] = NodeState.TAKEN
    for edge_id, disposition in delta.edge_updates:
        state.edge_state[edge_id] = disposition
    for node_id in delta.skipped_node_ids:
        state.node_state[node_id] = NodeState.SKIPPED
    del state.in_flight[delta.node_id]
    state.ready.extend(delta.newly_ready_ids)
    if apply.terminal is not None:
        state.result = apply.terminal.result
    state.next_sequence = prepared.expected_sequence + 1
    return apply.observers


def _commit_failure(
    log: RunLog,
    state: RunState,
    prepared: PreparedTransition,
) -> tuple[Any, ...]:
    """Validate, append, and install one immediately terminal node failure."""
    plan = prepared._plan
    if not isinstance(plan, _FailurePlan):
        raise TypeError("failure commit requires a failure plan")
    if not isinstance(prepared.record, NodeFailedRecord):
        raise TypeError("failure plan requires a NodeFailedRecord")
    _require_prepared_origin(state, plan)
    _require_initialized_nonterminal(state)
    _require_authority_coordinates(log, state, prepared, initialized=True)
    _require_unchanged_record(prepared, plan.envelope_bytes)
    delta = plan.delta
    if prepared.record != _failure_record(delta):
        raise ValueError("failure record and prevalidated delta disagree")
    apply = _materialize_failure(delta)

    _append_and_verify(log, prepared)
    state.node_state[delta.node_id] = NodeState.TAKEN
    del state.in_flight[delta.node_id]
    state.result = apply.result
    state.next_sequence = prepared.expected_sequence + 1
    return apply.observers


def _require_prepared_origin(
    state: RunState,
    plan: (
        _RunStartedPlan
        | _FrontierPlan
        | _SuccessPlan
        | _RoutePlan
        | _FailurePlan
    ),
) -> None:
    """Require the exact state, adopted flow, and boundary objects from preparation."""
    if state is not plan.originating_state:
        raise ValueError(
            "prepared transition belongs to a different originating RunState"
        )
    _require_originating_flow(state, plan)


def _require_authority_coordinates(
    log: RunLog,
    state: RunState,
    prepared: PreparedTransition,
    *,
    initialized: bool,
) -> None:
    """Require state, prepared transition, and log to share identity and sequence."""
    if prepared.expected_sequence != state.next_sequence:
        raise ValueError(
            "run-state sequence mismatch: "
            f"prepared {prepared.expected_sequence}, state {state.next_sequence}"
        )
    if prepared.expected_sequence != log.next_sequence:
        raise ValueError(
            "run-log sequence mismatch: "
            f"prepared {prepared.expected_sequence}, log {log.next_sequence}"
        )
    if prepared.expected_run_id != log.run_id:
        raise ValueError(
            "run-log run_id mismatch: "
            f"prepared {prepared.expected_run_id!r}, log {log.run_id!r}"
        )
    if initialized:
        if state.run_id != prepared.expected_run_id:
            raise ValueError(
                "run-state run_id mismatch: "
                f"prepared {prepared.expected_run_id!r}, state {state.run_id!r}"
            )
    elif state.run_id is not None:
        raise ValueError("RunStarted requires state.run_id to be None")


def _require_unchanged_record(
    prepared: PreparedTransition,
    envelope_bytes: bytes,
) -> None:
    """Reject mutation or mismatched construction of the public record carrier."""
    current_bytes = encode_record(
        RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=prepared.expected_run_id,
            seq=prepared.expected_sequence,
            record=prepared.record,
        )
    )
    if current_bytes != envelope_bytes:
        raise ValueError("prepared record changed after validation")


def _append_and_verify(log: RunLog, prepared: PreparedTransition) -> None:
    """Append the canonical record and reject any protocol-poisoned result."""
    envelope = log.append(prepared.record)
    if not isinstance(envelope, RecordEnvelope) or (
        envelope.run_id != prepared.expected_run_id
        or envelope.seq != prepared.expected_sequence
        or envelope.record != prepared.record
        or log.next_sequence != prepared.expected_sequence + 1
    ):
        raise ValueError("run-log append result does not match prepared transition")


def _require_initialized_nonterminal(state: RunState) -> None:
    """Check the initialized reducer roots without scanning run-sized contents."""
    exact_containers = (
        type(state.pool) is VariablePool
        and type(state.node_state) is dict
        and type(state.edge_state) is dict
        and type(state.ready) is deque
        and type(state.in_flight) is dict
        and type(state.depth) is dict
        and type(state.wiring_overrides) is dict
        and type(state.input) is dict
        and type(state.effective_llm_by_static_path) is dict
    )
    if not exact_containers:
        raise ValueError("dispatch requires initialized reducer frontier containers")
    if type(state.run_id) is not str or not state.run_id:
        raise ValueError("run state is not initialized")
    if type(state.flow_fingerprint) is not str or not state.flow_fingerprint:
        raise ValueError("run state is not initialized with a flow fingerprint")
    if type(state.next_sequence) is not int or state.next_sequence < 1:
        raise ValueError("run state is not initialized with a valid sequence")
    if type(state.num_workers) is not int or state.num_workers < 0:
        raise ValueError("run state has an invalid worker count")
    if state.result is not None:
        raise ValueError("terminal run state cannot prepare another transition")
    if state.flow.start_id not in state.flow.nodes:
        raise ValueError("initialized run state is missing its START boundary")
    if state.pool.start_id != state.flow.start_id:
        raise ValueError("initialized pool START must match the adopted flow")


def _validate_node_started(state: RunState, record: NodeStartedRecord) -> None:
    """Require ``record`` to dispatch exactly one eligible ready-head node."""
    node_id = record.node_id
    if type(node_id) is not str:
        raise TypeError("NodeStartedRecord node_id must be a string")
    if not state.ready or state.ready[0] != node_id:
        found = state.ready[0] if state.ready else None
        raise ValueError(
            f"NodeStarted requires ready head {node_id!r}, found {found!r}"
        )
    if node_id not in state.flow.nodes:
        raise ValueError(f"NodeStarted node {node_id!r} does not exist in the flow")
    if state.node_state.get(node_id) is not NodeState.UNKNOWN:
        raise ValueError(f"NodeStarted node {node_id!r} must have UNKNOWN disposition")
    if node_id in state.in_flight:
        raise ValueError(f"NodeStarted node {node_id!r} is already in-flight")


def _validate_run_resumed(state: RunState, record: RunResumedRecord) -> None:
    """Require a resume record to name the exact ordered unfinished frontier."""
    if type(record.num_workers) is not int:
        raise TypeError("num_workers must be a non-negative integer")
    if record.num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if type(record.retry_node_ids) is not tuple or any(
        type(node_id) is not str for node_id in record.retry_node_ids
    ):
        raise TypeError("RunResumedRecord retry_node_ids must be a tuple of strings")
    expected_retries = tuple(state.in_flight)
    if record.retry_node_ids != expected_retries:
        raise ValueError(
            "RunResumedRecord retry_node_ids must exactly equal ordered in-flight "
            f"nodes: expected {expected_retries!r}, found {record.retry_node_ids!r}"
        )
    for node_id in record.retry_node_ids:
        if node_id not in state.flow.nodes:
            raise ValueError(
                f"RunResumed retry node {node_id!r} does not exist in the flow"
            )
        if state.node_state.get(node_id) is not NodeState.UNKNOWN:
            raise ValueError(
                f"RunResumed retry node {node_id!r} must have UNKNOWN disposition"
            )
        if state.in_flight[node_id] is not None:
            raise ValueError("in-flight must be an ordered set with None values")


def _require_succeeding_node(state: RunState, node_id: str) -> None:
    """Require an existing unknown node in the ordered in-flight frontier."""
    if node_id not in state.flow.nodes:
        raise ValueError(f"NodeSucceeded node {node_id!r} does not exist in the flow")
    if state.node_state.get(node_id) is not NodeState.UNKNOWN:
        raise ValueError(f"NodeSucceeded node {node_id!r} must have UNKNOWN disposition")
    if node_id not in state.in_flight:
        raise ValueError(f"NodeSucceeded node {node_id!r} is not in-flight")
    if state.in_flight[node_id] is not None:
        raise ValueError("in-flight must be an ordered set with None values")


def _require_routing_node(state: RunState, node_id: str) -> None:
    """Require an existing unknown router in the ordered in-flight frontier."""
    if not node_id:
        raise ValueError("NodeRouted node_id must be non-empty")
    if node_id not in state.flow.nodes:
        raise ValueError(f"NodeRouted node {node_id!r} does not exist in the flow")
    if state.node_state.get(node_id) is not NodeState.UNKNOWN:
        raise ValueError(f"NodeRouted node {node_id!r} must have UNKNOWN disposition")
    if node_id not in state.in_flight:
        raise ValueError(f"NodeRouted node {node_id!r} is not in-flight")
    if state.in_flight[node_id] is not None:
        raise ValueError("in-flight must be an ordered set with None values")


def _require_failing_node(state: RunState, node_id: str) -> None:
    """Require an existing unknown node in the ordered in-flight frontier."""
    if not node_id:
        raise ValueError("NodeFailed node_id must be non-empty")
    if node_id not in state.flow.nodes:
        raise ValueError(f"NodeFailed node {node_id!r} does not exist in the flow")
    if state.node_state.get(node_id) is not NodeState.UNKNOWN:
        raise ValueError(f"NodeFailed node {node_id!r} must have UNKNOWN disposition")
    if node_id not in state.in_flight:
        raise ValueError(f"NodeFailed node {node_id!r} is not in-flight")
    if state.in_flight[node_id] is not None:
        raise ValueError("in-flight must be an ordered set with None values")


def _validate_failure_record(state: RunState, record: NodeFailedRecord) -> None:
    """Validate exact diagnostic carriers and the executing node for replay."""
    if type(record.node_id) is not str:
        raise TypeError("NodeFailedRecord node_id must be a string")
    if type(record.error) is not str:
        raise TypeError("NodeFailedRecord error must be a string")
    if type(record.error_type) is not str:
        raise TypeError("NodeFailedRecord error_type must be a string")
    if record.locator is not None and type(record.locator) is not SourceSpan:
        raise TypeError("NodeFailedRecord locator must be a SourceSpan or None")
    if record.range is not None and type(record.range) is not SourceRange:
        raise TypeError("NodeFailedRecord range must be a SourceRange or None")
    if record.traceback is not None and type(record.traceback) is not str:
        raise TypeError("NodeFailedRecord traceback must be a string or None")
    _require_failing_node(state, record.node_id)


def _validate_route_record(state: RunState, record: NodeRoutedRecord) -> None:
    """Validate the exact scalar carrier and executing router for replay."""
    if type(record.node_id) is not str:
        raise TypeError("NodeRoutedRecord node_id must be a string")
    if not record.node_id:
        raise ValueError("NodeRoutedRecord node_id must be non-empty")
    if type(record.handle) is not str:
        raise TypeError("NodeRoutedRecord handle must be a string")
    if not record.handle:
        raise ValueError("NodeRoutedRecord handle must be non-empty")
    _require_routing_node(state, record.node_id)


def _validate_success_value(
    state: RunState,
    record: NodeSucceededRecord,
) -> TypedValue:
    """Validate one replay value against its explicit commit target exactly once."""
    if type(record.node_id) is not str:
        raise TypeError("NodeSucceededRecord node_id must be a string")
    if type(record.commit_target) is not str:
        raise TypeError("NodeSucceededRecord commit_target must be a string")
    _require_succeeding_node(state, record.node_id)
    if record.commit_target not in state.flow.nodes:
        raise ValueError(
            f"success commit target {record.commit_target!r} does not exist in the flow"
        )
    try:
        typed = ANY_VALUE_ADAPTER.validate_python(record.value, strict=True)
    except ValueError as exc:
        raise ValueError("NodeSucceededRecord value is not a typed value") from exc
    wire, decoded = canonical_typed(
        typed,
        state.flow.nodes[record.commit_target].output_type,
    )
    if wire != record.value:
        raise ValueError("NodeSucceededRecord value is not canonical for its target")
    return decoded


def _prepare_success_delta(
    state: RunState,
    record: NodeSucceededRecord,
    typed: TypedValue,
) -> _SuccessDelta:
    """Validate touched topology and freeze one sparse successful-output delta."""
    if type(state.pool.store) is not dict:
        raise ValueError("success requires an exact variable-pool store")
    target = record.commit_target
    outgoing_edge_ids: list[str] = []
    newly_ready_ids: list[str] = []
    settled_edge_ids: set[str] = set()
    seen_outgoing_edge_ids: set[str] = set()
    remaining_by_successor: dict[str, int] = {}
    edge_ids_by_successor: dict[str, list[str]] = {}
    outgoing = state.flow.outgoing(target)

    for edge in outgoing:
        if not isinstance(edge, Edge) or edge.from_ != target:
            raise ValueError(
                f"success target {target!r} has an invalid outgoing adjacency"
            )
        if type(edge.id) is not str or not edge.id:
            raise ValueError("success target has an invalid outgoing edge ID")
        if edge.id in seen_outgoing_edge_ids:
            raise ValueError(f"success target outgoing edge {edge.id!r} is duplicated")
        if edge.id not in state.edge_state:
            raise ValueError(
                f"success target outgoing edge {edge.id!r} has no disposition"
            )
        if state.edge_state[edge.id] is not NodeState.UNKNOWN:
            raise ValueError(
                f"success target outgoing edge {edge.id!r} must be UNKNOWN"
            )
        if edge.to not in state.flow.nodes or edge.to not in state.node_state:
            raise ValueError(
                f"success target outgoing edge {edge.id!r} has unknown successor "
                f"{edge.to!r}"
            )

        outgoing_edge_ids.append(edge.id)
        seen_outgoing_edge_ids.add(edge.id)
        remaining_by_successor[edge.to] = remaining_by_successor.get(edge.to, 0) + 1
        edge_ids_by_successor.setdefault(edge.to, []).append(edge.id)

    for edge in outgoing:
        settled_edge_ids.add(edge.id)
        successor = edge.to
        remaining = remaining_by_successor[successor] - 1
        remaining_by_successor[successor] = remaining
        if remaining:
            continue
        if (
            _sparse_disposition(
                state.flow,
                state.edge_state,
                settled_edge_ids,
                successor,
                required_edge_ids=frozenset(edge_ids_by_successor[successor]),
            )
            != "ready"
        ):
            continue
        if state.node_state[successor] is not NodeState.UNKNOWN:
            raise ValueError(
                f"newly-ready successor {successor!r} must have UNKNOWN disposition"
            )
        newly_ready_ids.append(successor)

    value_bytes = ANY_VALUE_ADAPTER.dump_json(typed, warnings="error")
    return _SuccessDelta(
        node_id=record.node_id,
        commit_target=target,
        value_bytes=value_bytes,
        outgoing_edge_ids=tuple(outgoing_edge_ids),
        newly_ready_ids=tuple(newly_ready_ids),
        observer_commit_as=(None if target == record.node_id else target),
    )


def _prepare_route_delta(
    state: RunState,
    record: NodeRoutedRecord,
) -> tuple[_RouteDelta, dict[str, NodeState]]:
    """Validate one route and return its frozen delta plus transient edge overlay."""
    _validate_route_record(state, record)
    edge_updates: dict[str, NodeState] = {}
    skipped_node_ids: list[str] = []
    skipped_nodes: set[str] = set()
    newly_ready_ids: list[str] = []
    newly_ready: set[str] = set()
    joins: dict[str, _JoinScratch] = {}

    root_edges = _route_outgoing(state, record.node_id, edge_updates)
    matched = tuple(
        edge
        for edge in root_edges
        if (edge.source_handle or "default") == record.handle
    )
    if not matched:
        raise ValueError(
            f"NodeRouted handle {record.handle!r} matches no outgoing edge of "
            f"{record.node_id!r}"
        )

    # A LIFO stack with reversed pushes exactly reproduces recursive stored-order DFS
    # without depending on Python's recursion limit for deep skip chains.
    work: list[tuple[Edge, NodeState]] = list(
        reversed(
            tuple(
                (
                    edge,
                    NodeState.TAKEN
                    if (edge.source_handle or "default") == record.handle
                    else NodeState.SKIPPED,
                )
                for edge in root_edges
            )
        )
    )
    while work:
        edge, disposition = work.pop()
        _require_route_edge(state, edge, edge_updates)
        scratch = joins.get(edge.to)
        if scratch is None:
            scratch = _scan_join(state, edge.to)
            joins[edge.to] = scratch
        _settle_join_edge(scratch, edge.id, disposition, edge.to)
        edge_updates[edge.id] = disposition

        head = edge.to
        head_disposition = state.node_state[head]
        if head_disposition not in (NodeState.UNKNOWN, NodeState.SKIPPED):
            raise ValueError(
                f"route successor {head!r} must have UNKNOWN or SKIPPED disposition"
            )
        if head in skipped_nodes or head_disposition is NodeState.SKIPPED:
            continue
        join_disposition = _classify_join(scratch)
        if join_disposition == "wait":
            continue
        if join_disposition == "ready":
            if head_disposition is not NodeState.UNKNOWN:
                raise ValueError(
                    f"newly-ready route successor {head!r} must have UNKNOWN disposition"
                )
            if head in newly_ready:
                raise ValueError(
                    f"route successor {head!r} became ready more than once"
                )
            newly_ready.add(head)
            newly_ready_ids.append(head)
            continue

        skipped_nodes.add(head)
        skipped_node_ids.append(head)
        children = _route_outgoing(state, head, edge_updates)
        for child in reversed(children):
            work.append((child, NodeState.SKIPPED))

    return (
        _RouteDelta(
            node_id=record.node_id,
            handle=record.handle,
            edge_updates=tuple(edge_updates.items()),
            skipped_node_ids=tuple(skipped_node_ids),
            newly_ready_ids=tuple(newly_ready_ids),
        ),
        edge_updates,
    )


def _prepare_terminal_delta(
    state: RunState,
    *,
    completing_node_id: str,
    newly_ready_ids: tuple[str, ...],
    pending_commit_target: str | None,
    pending_value_bytes: bytes | None,
    edge_updates: dict[str, NodeState],
) -> _TerminalDelta | None:
    """Derive an optional terminal result from one predicted post-outcome state.

    Quiescence depends only on the two frontier roots and the sparse additions in
    the causal outcome.  A successful END write is read from its pending typed
    bytes; an already-committed END value is re-encoded so the plan never aliases
    the live pool.  Terminal-skip diagnosis consults only END's incoming adjacency.
    """
    if (
        state.result is not None
        or state.ready
        or newly_ready_ids
        or len(state.in_flight) != 1
        or completing_node_id not in state.in_flight
    ):
        return None

    end_id = state.flow.end_id
    if pending_commit_target == end_id:
        if type(pending_value_bytes) is not bytes:
            raise ValueError("pending END output must be canonical typed bytes")
        return _TerminalDelta(
            succeeded=True,
            output_bytes=pending_value_bytes,
            skipped_output_name=None,
        )
    if end_id in state.pool.store:
        output_bytes = ANY_VALUE_ADAPTER.dump_json(
            state.pool.store[end_id],
            warnings="error",
        )
        return _TerminalDelta(
            succeeded=True,
            output_bytes=output_bytes,
            skipped_output_name=None,
        )
    return _TerminalDelta(
        succeeded=False,
        output_bytes=None,
        skipped_output_name=_terminal_skipped_output_name(state, edge_updates),
    )


def _terminal_skipped_output_name(
    state: RunState,
    edge_updates: dict[str, NodeState],
) -> str | None:
    """Return the first stored-order required END group fully skipped.

    ``edge_updates`` is the pending route overlay; an absent edge reads its live
    disposition.  Group insertion order preserves the legacy first-occurrence
    rule, including ``None`` when that is the diagnosed input-group name.
    """
    groups: dict[str | None, tuple[bool, bool]] = {}
    for edge in state.flow.incoming(state.flow.end_id):
        if edge.source_handle is not None or edge.ordering:
            continue
        disposition = edge_updates.get(edge.id, state.edge_state.get(edge.id))
        if edge.input_group not in groups:
            groups[edge.input_group] = (
                edge.optional,
                disposition is NodeState.SKIPPED,
            )
            continue
        optional, all_skipped = groups[edge.input_group]
        groups[edge.input_group] = (
            optional,
            all_skipped and disposition is NodeState.SKIPPED,
        )
    for name, (optional, all_skipped) in groups.items():
        if not optional and all_skipped:
            return name
    return None


def _route_outgoing(
    state: RunState,
    node_id: str,
    edge_updates: dict[str, NodeState],
) -> tuple[Edge, ...]:
    """Return one node's validated outgoing adjacency in stored order."""
    outgoing = state.flow.outgoing(node_id)
    seen: set[str] = set()
    validated: list[Edge] = []
    for edge in outgoing:
        if not isinstance(edge, Edge) or edge.from_ != node_id:
            raise ValueError(f"route node {node_id!r} has an invalid outgoing adjacency")
        if type(edge.id) is not str or not edge.id or edge.id in seen:
            raise ValueError(f"route node {node_id!r} has invalid outgoing edge IDs")
        seen.add(edge.id)
        if edge.id in edge_updates:
            raise ValueError(f"route edge {edge.id!r} is reached more than once")
        if edge.id not in state.edge_state:
            raise ValueError(f"route edge {edge.id!r} has no disposition")
        if state.edge_state[edge.id] is not NodeState.UNKNOWN:
            raise ValueError(f"route edge {edge.id!r} must be UNKNOWN")
        if edge.to not in state.flow.nodes or edge.to not in state.node_state:
            raise ValueError(
                f"route edge {edge.id!r} has unknown successor {edge.to!r}"
            )
        validated.append(edge)
    return tuple(validated)


def _require_route_edge(
    state: RunState,
    edge: Edge,
    edge_updates: dict[str, NodeState],
) -> None:
    """Recheck a queued edge without scanning any adjacency or global frontier."""
    if edge.id in edge_updates:
        raise ValueError(f"route edge {edge.id!r} is reached more than once")
    if edge.id not in state.edge_state:
        raise ValueError(f"route edge {edge.id!r} has no disposition")
    if state.edge_state[edge.id] is not NodeState.UNKNOWN:
        raise ValueError(f"route edge {edge.id!r} must be UNKNOWN")


def _scan_join(state: RunState, node_id: str) -> _JoinScratch:
    """Scan one touched head's incoming adjacency into incremental gate counts."""
    incoming = state.flow.incoming(node_id)
    if not incoming:
        raise ValueError(f"route successor {node_id!r} has no incoming adjacency")
    states: dict[str, NodeState] = {}
    classes: dict[str, tuple[str, str | None]] = {}
    unknown = 0
    control_total = 0
    control_skipped = 0
    required_data_dead = 0
    required_ordering_skipped = 0
    data_groups: dict[str | None, _DataGroupScratch] = {}
    for edge in incoming:
        if not isinstance(edge, Edge) or edge.to != node_id:
            raise ValueError(f"route successor {node_id!r} has invalid incoming adjacency")
        if type(edge.id) is not str or not edge.id or edge.id in states:
            raise ValueError(f"route successor {node_id!r} has invalid incoming edge IDs")
        if edge.id not in state.edge_state:
            raise ValueError(f"route incoming edge {edge.id!r} has no disposition")
        disposition = state.edge_state[edge.id]
        if type(disposition) is not NodeState:
            raise ValueError(f"route incoming edge {edge.id!r} has invalid disposition")
        states[edge.id] = disposition
        unknown += disposition is NodeState.UNKNOWN
        if edge.source_handle is not None:
            classes[edge.id] = ("control", None)
            control_total += 1
            control_skipped += disposition is NodeState.SKIPPED
        elif edge.ordering:
            classes[edge.id] = (
                "ordering" if edge.optional else "required_ordering",
                None,
            )
            required_ordering_skipped += (
                not edge.optional and disposition is NodeState.SKIPPED
            )
        else:
            group = data_groups.get(edge.input_group)
            if group is None:
                group = _DataGroupScratch(optional=edge.optional)
                data_groups[edge.input_group] = group
            group.total += 1
            group.skipped += disposition is NodeState.SKIPPED
            classes[edge.id] = ("data", edge.input_group)
    required_data_dead = sum(
        not group.optional and group.skipped == group.total
        for group in data_groups.values()
    )
    return _JoinScratch(
        states=states,
        classes=classes,
        unknown=unknown,
        control_total=control_total,
        control_skipped=control_skipped,
        required_data_dead=required_data_dead,
        required_ordering_skipped=required_ordering_skipped,
        data_groups=data_groups,
    )


def _settle_join_edge(
    scratch: _JoinScratch,
    edge_id: str,
    disposition: NodeState,
    node_id: str,
) -> None:
    """Apply one UNKNOWN-to-settled update to a cached join in constant time."""
    if edge_id not in scratch.states:
        raise ValueError(
            f"route edge {edge_id!r} is absent from successor {node_id!r} incoming adjacency"
        )
    if scratch.states[edge_id] is not NodeState.UNKNOWN:
        raise ValueError(f"route edge {edge_id!r} must be UNKNOWN in its incoming join")
    scratch.states[edge_id] = disposition
    scratch.unknown -= 1
    if disposition is not NodeState.SKIPPED:
        return
    edge_class, group_name = scratch.classes[edge_id]
    if edge_class == "control":
        scratch.control_skipped += 1
    elif edge_class == "required_ordering":
        scratch.required_ordering_skipped += 1
    elif edge_class == "data":
        group = scratch.data_groups[group_name]
        group.skipped += 1
        if not group.optional and group.skipped == group.total:
            scratch.required_data_dead += 1


def _classify_join(scratch: _JoinScratch) -> str:
    """Classify cached gate counts with eager dead before unresolved wait."""
    if scratch.control_total and scratch.control_skipped == scratch.control_total:
        return "dead"
    if scratch.required_data_dead:
        return "dead"
    if scratch.required_ordering_skipped:
        return "dead"
    if scratch.unknown:
        return "wait"
    return "ready"


def _sparse_disposition(
    flow: CompiledFlow,
    edge_state: dict[str, NodeState],
    taken_edge_ids: set[str],
    node_id: str,
    *,
    required_edge_ids: frozenset[str],
) -> str:
    """Classify one touched join over the live state plus sparse TAKEN overlay."""
    incoming = flow.incoming(node_id)
    if not incoming:
        raise ValueError(
            f"success edges {sorted(required_edge_ids)!r} are absent from successor "
            f"{node_id!r} incoming adjacency"
        )

    found_required: set[str] = set()
    saw_control = False
    all_control_skipped = True
    required_ordering_skipped = False
    any_unknown = False
    data_groups: dict[str | None, tuple[bool, bool]] = {}
    seen_incoming_ids: set[str] = set()
    for edge in incoming:
        if not isinstance(edge, Edge) or edge.to != node_id:
            raise ValueError(f"successor {node_id!r} has an invalid incoming adjacency")
        if type(edge.id) is not str or not edge.id or edge.id in seen_incoming_ids:
            raise ValueError(
                f"successor {node_id!r} has invalid incoming edge IDs"
            )
        seen_incoming_ids.add(edge.id)
        if edge.id in required_edge_ids:
            found_required.add(edge.id)
        if edge.id not in edge_state:
            raise ValueError(
                f"successor incoming edge {edge.id!r} has no disposition"
            )
        disposition = (
            NodeState.TAKEN if edge.id in taken_edge_ids else edge_state[edge.id]
        )
        if type(disposition) is not NodeState:
            raise ValueError(
                f"successor incoming edge {edge.id!r} has an invalid disposition"
            )
        any_unknown = any_unknown or disposition is NodeState.UNKNOWN
        if edge.source_handle is not None:
            saw_control = True
            all_control_skipped = (
                all_control_skipped and disposition is NodeState.SKIPPED
            )
        elif edge.ordering:
            required_ordering_skipped = required_ordering_skipped or (
                not edge.optional and disposition is NodeState.SKIPPED
            )
        else:
            optional, all_skipped = data_groups.get(
                edge.input_group,
                (edge.optional, True),
            )
            data_groups[edge.input_group] = (
                optional,
                all_skipped and disposition is NodeState.SKIPPED,
            )

    if found_required != required_edge_ids:
        raise ValueError(
            f"success edges {sorted(required_edge_ids - found_required)!r} are absent "
            "from successor "
            f"{node_id!r} incoming adjacency"
        )
    if saw_control and all_control_skipped:
        return "dead"
    for optional, all_skipped in data_groups.values():
        if not optional and all_skipped:
            return "dead"
    if required_ordering_skipped:
        return "dead"
    if any_unknown:
        return "wait"
    return "ready"


def _require_pristine(state: RunState) -> None:
    """Reject any state that no longer has its constructor-created empty frontier."""
    exact_containers = (
        type(state.pool) is VariablePool
        and type(state.node_state) is dict
        and type(state.edge_state) is dict
        and type(state.ready) is deque
        and type(state.in_flight) is dict
        and type(state.depth) is dict
        and type(state.wiring_overrides) is dict
        and type(state.input) is dict
        and type(state.effective_llm_by_static_path) is dict
    )
    if not exact_containers:
        raise ValueError("RunStarted requires pristine reducer container types")
    expected_pool = VariablePool(start_id=state.flow.start_id)
    expected_node_ids = set(state.flow.nodes)
    expected_edge_ids = {edge.id for edge in state.flow.edges}
    if not (
        state.pool == expected_pool
        and set(state.node_state) == expected_node_ids
        and all(value is NodeState.UNKNOWN for value in state.node_state.values())
        and set(state.edge_state) == expected_edge_ids
        and all(value is NodeState.UNKNOWN for value in state.edge_state.values())
        and not state.ready
        and not state.in_flight
        and state.result is None
        and not state.depth
        and not state.wiring_overrides
        and state.run_id is None
        and state.flow_fingerprint is None
        and type(state.next_sequence) is int
        and state.next_sequence == 0
        and type(state.num_workers) is int
        and state.num_workers == 0
        and not state.input
        and not state.effective_llm_by_static_path
    ):
        raise ValueError("RunStarted requires a pristine RunState at sequence 0")


def _require_originating_flow(
    state: RunState,
    plan: (
        _RunStartedPlan
        | _FrontierPlan
        | _SuccessPlan
        | _RoutePlan
        | _FailurePlan
    ),
) -> None:
    """Reject replacement of the adopted flow object or either boundary identity.

    The graph is a valid private ``CompiledFlow`` whose execution fields are owned by
    reducer transitions.  This guard enforces that ownership boundary; it deliberately
    does not pretend to fingerprint arbitrary in-place mutation by external code.
    """
    if not (
        state.flow is plan.originating_flow
        and state.flow.start_id == plan.start_id
        and plan.start_id in state.flow.nodes
        and id(state.flow.nodes[plan.start_id]) == plan.start_node_identity
        and state.flow.end_id == plan.end_id
        and (
            id(state.flow.nodes[plan.end_id])
            if plan.end_id in state.flow.nodes
            else None
        )
        == plan.end_node_identity
    ):
        raise ValueError(
            "transition requires its originating flow and boundary identities"
        )


def _materialize_run_started(
    state: RunState,
    prepared: PreparedTransition,
) -> _RunStartedApply:
    """Decode and detach every apply value before the record becomes durable."""
    envelope = decode_record(prepared._plan.envelope_bytes)
    record = envelope.record
    if not (
        envelope.run_id == prepared.expected_run_id
        and envelope.seq == prepared.expected_sequence
        and isinstance(record, RunStartedRecord)
    ):
        raise ValueError("immutable RunStarted plan does not match its transition")

    pool = decode_pool(record.initial_pool)
    start_id = prepared._plan.start_id
    if pool.start_id != start_id:
        raise ValueError("immutable RunStarted pool start_id mismatch")
    if pool.resolve("system", ["run_id"]) != envelope.run_id:
        raise ValueError("immutable RunStarted pool system.run_id mismatch")
    primed_start = pool.get(start_id)
    if not isinstance(primed_start, dict):
        raise ValueError("immutable RunStarted pool START value must be a dict")

    canonical_wiring = canonical_seed(record.start_wiring_override)[1]
    canonical_effective = canonical_seed(record.effective_llm_by_static_path)[1]
    canonical_input = canonical_seed(primed_start)[1]
    if not isinstance(canonical_wiring, dict):
        raise TypeError("canonical START wiring is not a dict")
    if not isinstance(canonical_effective, dict):
        raise TypeError("canonical effective LLM map is not a dict")
    if not isinstance(canonical_input, dict):
        raise TypeError("canonical START input is not a dict")

    return _RunStartedApply(
        pool=pool,
        node_state=dict(state.node_state),
        edge_state=dict(state.edge_state),
        ready=deque((start_id,)),
        in_flight={},
        depth={},
        wiring_overrides={start_id: canonical_wiring},
        run_id=envelope.run_id,
        flow_fingerprint=record.flow_fingerprint,
        next_sequence=envelope.seq + 1,
        num_workers=record.num_workers,
        input=canonical_input,
        effective_llm_by_static_path=canonical_effective,
        observers=(RunStarted(),),
    )


def _apply_run_started(state: RunState, apply: _RunStartedApply) -> None:
    """Install already-detached values using only total replacement assignments."""
    state.pool = apply.pool
    state.node_state = apply.node_state
    state.edge_state = apply.edge_state
    state.ready = apply.ready
    state.in_flight = apply.in_flight
    state.result = None
    state.depth = apply.depth
    state.wiring_overrides = apply.wiring_overrides
    state.run_id = apply.run_id
    state.flow_fingerprint = apply.flow_fingerprint
    state.num_workers = apply.num_workers
    state.input = apply.input
    state.effective_llm_by_static_path = apply.effective_llm_by_static_path
    state.next_sequence = apply.next_sequence


def _materialize_success(plan: _SuccessPlan) -> _SuccessApply:
    """Decode independent pool, observer, and terminal copies before append."""
    value = ANY_VALUE_ADAPTER.validate_json(plan.delta.value_bytes, strict=True)
    observer_value = ANY_VALUE_ADAPTER.validate_json(
        plan.delta.value_bytes,
        strict=True,
    ).to_object()
    terminal = (
        _materialize_terminal(plan.terminal)
        if plan.terminal is not None
        else None
    )
    observers: tuple[Any, ...] = (
        NodeSucceeded(
            node_id=plan.delta.node_id,
            output=observer_value,
            commit_as=plan.delta.observer_commit_as,
        ),
    )
    if terminal is not None:
        observers += (terminal.observer,)
    return _SuccessApply(
        value=value,
        terminal=terminal,
        observers=observers,
    )


def _materialize_route(plan: _RoutePlan) -> _RouteApply:
    """Materialize route observers and independent terminal copies before append."""
    terminal = (
        _materialize_terminal(plan.terminal)
        if plan.terminal is not None
        else None
    )
    observers = (NodeRouted(plan.delta.node_id, plan.delta.handle),) + tuple(
        NodeSkipped(node_id) for node_id in plan.delta.skipped_node_ids
    )
    if terminal is not None:
        observers += (terminal.observer,)
    return _RouteApply(terminal=terminal, observers=observers)


def _terminal_observer(delta: _TerminalDelta) -> RunSucceeded | RunFailed:
    """Materialize one detached terminal event from immutable plan authority."""
    if delta.succeeded:
        if type(delta.output_bytes) is not bytes:
            raise ValueError("successful terminal delta requires typed output bytes")
        output = ANY_VALUE_ADAPTER.validate_json(
            delta.output_bytes,
            strict=True,
        ).to_object()
        return RunSucceeded(output=output)
    if delta.output_bytes is not None:
        raise ValueError("failed terminal delta cannot carry output bytes")
    return RunFailed(
        error=f"terminal output {delta.skipped_output_name!r} skipped",
        error_type="TerminalSkipped",
    )


def _materialize_terminal(delta: _TerminalDelta) -> _TerminalApply:
    """Build independent state and callback terminal values before durable append."""
    return _TerminalApply(
        result=_terminal_observer(delta),
        observer=_terminal_observer(delta),
    )


def _failure_delta(record: NodeFailedRecord) -> _FailureDelta:
    """Detach one canonical record into immutable scalar diagnostic authority."""
    return _FailureDelta(
        node_id=record.node_id,
        error=record.error,
        error_type=record.error_type,
        locator=_copy_source_span(record.locator),
        range=_copy_source_range(record.range),
        traceback=record.traceback,
    )


def _failure_record(delta: _FailureDelta) -> NodeFailedRecord:
    """Construct an independent durable carrier from an immutable failure delta."""
    return NodeFailedRecord(
        node_id=delta.node_id,
        error=delta.error,
        error_type=delta.error_type,
        locator=_copy_source_span(delta.locator),
        range=_copy_source_range(delta.range),
        traceback=delta.traceback,
    )


def _failure_observers(delta: _FailureDelta) -> tuple[Any, ...]:
    """Construct ordered, independent node- and run-level failure observers."""
    return (
        NodeFailed(
            node_id=delta.node_id,
            error=delta.error,
            error_type=delta.error_type,
            locator=_copy_source_span(delta.locator),
            range=_copy_source_range(delta.range),
            traceback=delta.traceback,
        ),
        RunFailed(
            error=delta.error,
            error_type=delta.error_type,
            locator=_copy_source_span(delta.locator),
            range=_copy_source_range(delta.range),
            traceback=delta.traceback,
        ),
    )


def _materialize_failure(delta: _FailureDelta) -> _FailureApply:
    """Materialize detached state and callback values before durable append."""
    return _FailureApply(
        result=RunFailed(
            error=delta.error,
            error_type=delta.error_type,
            locator=_copy_source_span(delta.locator),
            range=_copy_source_range(delta.range),
            traceback=delta.traceback,
        ),
        observers=_failure_observers(delta),
    )


def _copy_source_span(locator: SourceSpan | None) -> SourceSpan | None:
    """Return an independent immutable symbolic source locator."""
    if locator is None:
        return None
    return SourceSpan(node=locator.node, kind=locator.kind, key=locator.key)


def _copy_source_range(range: SourceRange | None) -> SourceRange | None:
    """Return an independent immutable concrete source range."""
    if range is None:
        return None
    return SourceRange(
        start_line=range.start_line,
        start_col=range.start_col,
        end_line=range.end_line,
        end_col=range.end_col,
    )


__all__ = [
    "PreparedTransition",
    "RunStartData",
    "RunState",
    "commit_prepared",
    "prepare_node_failed",
    "prepare_node_routed",
    "prepare_node_started",
    "prepare_node_succeeded",
    "prepare_record",
    "prepare_run_resumed",
    "prepare_run_started",
]

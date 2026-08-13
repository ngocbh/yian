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
from agent_composer.compile.model import CompiledFlow
from agent_composer.events import RunFailed, RunStarted, RunSucceeded
from agent_composer.runtime.log import (
    SCHEMA_VERSION,
    RecordEnvelope,
    RunLog,
    RunRecord,
    RunStartedRecord,
    canonical_seed,
    decode_pool,
    decode_record,
    encode_pool,
    encode_record,
)
from agent_composer.typesys.pool import VariablePool


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
    _plan: _RunStartedPlan
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
    if not isinstance(prepared._plan, _RunStartedPlan):
        raise TypeError("unsupported prepared transition plan")
    if not isinstance(prepared.record, RunStartedRecord):
        raise TypeError("RunStarted plan requires a RunStartedRecord")
    if state is not prepared._plan.originating_state:
        raise ValueError("prepared transition belongs to a different originating RunState")
    _require_originating_flow(state, prepared._plan)
    _require_pristine(state)
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
    if state.run_id is not None:
        raise ValueError("RunStarted requires state.run_id to be None")

    current_bytes = encode_record(
        RecordEnvelope(
            schema=SCHEMA_VERSION,
            run_id=prepared.expected_run_id,
            seq=prepared.expected_sequence,
            record=prepared.record,
        )
    )
    if current_bytes != prepared._plan.envelope_bytes:
        raise ValueError("prepared record changed after validation")

    apply = _materialize_run_started(state, prepared)

    envelope = log.append(prepared.record)
    if not isinstance(envelope, RecordEnvelope) or (
        envelope.run_id != prepared.expected_run_id
        or envelope.seq != prepared.expected_sequence
        or envelope.record != prepared.record
        or log.next_sequence != prepared.expected_sequence + 1
    ):
        raise ValueError("run-log append result does not match prepared transition")

    _apply_run_started(state, apply)
    return prepared.observers


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


def _require_originating_flow(state: RunState, plan: _RunStartedPlan) -> None:
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
        raise ValueError("RunStarted requires its originating flow and boundary identities")


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


__all__ = [
    "PreparedTransition",
    "RunStartData",
    "RunState",
    "commit_prepared",
    "prepare_run_started",
]

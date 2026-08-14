"""Authoritative mutable run state and append-before-apply transitions.

Preparation validates and canonicalizes a transition without touching live state.
``commit_prepared`` then makes its record durable before applying a frozen plan
whose remaining work is limited to total assignments and container operations.
The dispatcher will be the sole caller that mutates this state while a run is live.
The caller supplies a valid private graph and relinquishes direct mutation of its
execution fields; chronological reducer transitions are the only topology writers.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

from agent_composer._state import NodeState
from agent_composer.compile.manifest import expansion_digest
from agent_composer.compile.model import CompiledFlow, Edge, Flow
from agent_composer.compile.validation import _classify_path
from agent_composer.events import (
    NodeExpanded,
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
from agent_composer.expr import ExpressionError, expr_refs_of
from agent_composer.nodes.base import Grow, Node
from agent_composer.runtime.log import (
    SCHEMA_VERSION,
    NodeExpandedRecord,
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


MAX_TOTAL_NODES = 10_000
"""Maximum live topology size before same-transition pruning."""

MAX_REF_DEPTH = 5
"""Maximum positive CALL/MAP reference-growth depth."""

GrowBuilder = Callable[[Node, Any], Flow]
"""Injected kind-aware interpreter from a spawner and canonical seed to a fragment."""


class _NodeDataError(Exception):
    """Private marker for an outcome-owned value rejected before durability."""

    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


class _NodeDataTypeError(_NodeDataError, TypeError):
    """Node-data rejection retaining compatibility with ``TypeError`` callers."""


class _NodeDataValueError(_NodeDataError, ValueError):
    """Node-data rejection retaining compatibility with ``ValueError`` callers."""


def _raise_node_data(original: Exception) -> NoReturn:
    """Raise the built-in-compatible marker matching one outcome-data error."""
    if isinstance(original, TypeError):
        raise _NodeDataTypeError(original) from original
    if isinstance(original, ValueError):
        raise _NodeDataValueError(original) from original
    raise _NodeDataError(original) from original


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
        initial_static_ids (`frozenset[str]`):
            Immutable node IDs adopted before any runtime expansion.  Safe pruning
            may retire only later dynamic-overlay nodes.
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
        self.initial_static_ids = frozenset(flow.nodes)


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


@dataclass(frozen=True, slots=True)
class _GrowthPlan:
    """Opaque authority anchors and one fully materialized expansion plan."""

    originating_state: RunState
    originating_flow: CompiledFlow
    envelope_bytes: bytes
    start_id: str
    start_node_identity: int
    end_id: str
    end_node_identity: int | None
    delta: _GrowthDelta
    topology: _GrowthTopology


@dataclass(frozen=True, slots=True)
class _GrowthDelta:
    """Immutable scalar authority for one flat deterministic expansion."""

    spawner_id: str
    prune_ids: tuple[str, ...]
    topology_digest: str
    fragment_start_id: str
    fragment_end_id: str
    new_node_ids: tuple[str, ...]
    new_edge_ids: tuple[str, ...]
    derived_depth: int | None


@dataclass(frozen=True, slots=True)
class _GrowthTopology:
    """Detached final containers represented by immutable outer tuples."""

    nodes: tuple[tuple[str, Node], ...]
    edges: tuple[Edge, ...]
    wiring: tuple[tuple[str, dict[str, Any]], ...]
    incoming: tuple[tuple[str, tuple[Edge, ...]], ...]
    outgoing: tuple[tuple[str, tuple[Edge, ...]], ...]
    pool_store: tuple[tuple[str, TypedValue], ...]
    node_state: tuple[tuple[str, NodeState], ...]
    edge_state: tuple[tuple[str, NodeState], ...]
    ready: tuple[str, ...]
    in_flight: tuple[str, ...]
    depth: tuple[tuple[str, int], ...]
    wiring_overrides: tuple[tuple[str, dict[str, Any]], ...]


@dataclass(slots=True)
class _GrowthApply:
    """Independent final topology, reducer containers, and observer before append."""

    nodes: dict[str, Node]
    edges: list[Edge]
    wiring: dict[str, dict[str, Any]]
    incoming: dict[str, list[Edge]]
    outgoing: dict[str, list[Edge]]
    pool_store: dict[str, TypedValue]
    node_state: dict[str, NodeState]
    edge_state: dict[str, NodeState]
    ready: deque[str]
    in_flight: dict[str, None]
    depth: dict[str, int]
    wiring_overrides: dict[str, dict[str, Any]]
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
        | _GrowthPlan
    )
    observers: tuple[Any, ...]


class _HistoricalLogCursor:
    """Read-only historical authority presented to the live commit reducer.

    The cursor owns canonical immutable envelope bytes.  Its ``append`` operation
    advances only when the reducer's exact next envelope matches those bytes; it
    never writes to or closes the source log from which the prefix was obtained.
    """

    def __init__(self, envelope_bytes: tuple[bytes, ...]) -> None:
        """Adopt one validated non-empty canonical prefix at sequence zero."""
        first = decode_record(envelope_bytes[0])
        self._envelope_bytes = envelope_bytes
        self._run_id = first.run_id
        self._next_sequence = 0

    @property
    def run_id(self) -> str:
        """Return the sole historical run identity accepted by this cursor."""
        return self._run_id

    @property
    def path(self):
        """Return no path because the cursor borrows already-read authority."""
        return None

    @property
    def next_sequence(self) -> int:
        """Return the historical envelope expected by the next reducer commit."""
        return self._next_sequence

    def append(self, record: RunRecord) -> RecordEnvelope:
        """Match and consume one historical envelope without writing it again."""
        if self._next_sequence >= len(self._envelope_bytes):
            raise ValueError("replay attempted to append beyond the historical prefix")
        candidate = encode_record(
            RecordEnvelope(
                schema=SCHEMA_VERSION,
                run_id=self._run_id,
                seq=self._next_sequence,
                record=record,
            )
        )
        expected = self._envelope_bytes[self._next_sequence]
        if candidate != expected:
            raise ValueError(
                "prepared transition does not match the historical run record"
            )
        envelope = decode_record(expected)
        self._next_sequence += 1
        return envelope


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
        _raise_node_data(TypeError("commit_as must be a string or None"))
    if commit_as == "":
        _raise_node_data(ValueError("commit_as must be non-empty when supplied"))
    _require_initialized_nonterminal(state)
    _require_succeeding_node(state, node_id)

    node_target = state.flow.nodes[node_id].commit_as
    if node_target is not None and (type(node_target) is not str or not node_target):
        raise ValueError(f"node {node_id!r} has an invalid baked commit target")
    target = commit_as if commit_as is not None else node_target or node_id
    if target not in state.flow.nodes:
        if commit_as is not None:
            _raise_node_data(
                ValueError(
                    f"success commit target {target!r} does not exist in the flow"
                )
            )
        raise ValueError(f"success commit target {target!r} does not exist in the flow")
    try:
        value_wire, typed = canonical_typed(
            output,
            state.flow.nodes[target].output_type,
        )
    except (TypeError, ValueError) as exc:
        _raise_node_data(exc)
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
        _raise_node_data(TypeError("handle must be a string"))
    if not handle:
        _raise_node_data(ValueError("handle must be non-empty"))
    _require_initialized_nonterminal(state)
    _require_routing_node(state, node_id)
    return _prepare_route_transition(
        state,
        NodeRoutedRecord(node_id=node_id, handle=handle),
    )


def prepare_expanded(
    state: RunState,
    node_id: str,
    grow: Grow,
    grow_builder: GrowBuilder,
) -> PreparedTransition:
    """Prepare one canonical flat graph expansion without mutating live state.

    Args:
        state (`RunState`):
            Initialized nonterminal state with ``node_id`` in ordered in-flight work.
        node_id (`str`):
            Exact executing spawner ID, including a dynamic LOOP driver clone ID.
        grow (`Grow`):
            Seed-first expansion description and exact dynamic-overlay prune set.
        grow_builder (`GrowBuilder`):
            Injected compile-owned interpreter from the spawner and canonical seed
            to one self-contained `Flow` fragment.

    Returns:
        `PreparedTransition`:
            Canonical ``NodeExpandedRecord`` and detached append-before-apply plan.

    Raises:
        `TypeError`:
            A carrier, prune member, builder, or built fragment has the wrong type.
        `ValueError`:
            The spawner, fragment, budget, depth, or prune set violates growth safety.
    """
    if type(state) is not RunState:
        raise TypeError("prepare_expanded expects a RunState")
    if type(node_id) is not str:
        raise TypeError("node_id must be a string")
    if not node_id:
        raise ValueError("node_id must be non-empty")
    if type(grow) is not Grow:
        raise TypeError("grow must be a Grow")
    if not callable(grow_builder):
        raise TypeError("grow_builder must be callable")
    if type(grow.prune) is not frozenset:
        _raise_node_data(TypeError("Grow.prune must be a frozenset of node IDs"))
    if any(type(prune_id) is not str for prune_id in grow.prune):
        _raise_node_data(TypeError("Grow.prune must contain only strings"))
    if any(not prune_id for prune_id in grow.prune):
        _raise_node_data(ValueError("Grow.prune node IDs must be non-empty"))
    _require_initialized_nonterminal(state)
    _require_expanding_node(state, node_id)

    try:
        seed_wire, builder_seed = canonical_seed(grow.seed)
    except (TypeError, ValueError) as exc:
        _raise_node_data(exc)
    fragment = _build_growth_fragment(
        state.flow.nodes[node_id],
        builder_seed,
        grow_builder,
    )
    validated_fragment = _validate_growth_fragment(state, fragment)
    try:
        digest = expansion_digest(fragment)
    except (TypeError, ValueError) as exc:
        _raise_node_data(exc)
    return _prepare_growth_transition(
        state,
        NodeExpandedRecord(
            spawner_id=node_id,
            seed=seed_wire,
            prune_ids=tuple(sorted(grow.prune)),
            topology_digest=digest,
        ),
        fragment,
        actual_digest=digest,
        validated_fragment=validated_fragment,
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


def prepare_record(
    state: RunState,
    record: RunRecord,
    *,
    grow_builder: GrowBuilder | None = None,
) -> PreparedTransition:
    """Validate one supported authoritative record against the current frontier.

    This is the shared reducer seam for live dispatch and later historical replay.
    Dispatch, recovery-frontier, successful-output, route, graph-expansion, and
    terminal-failure records are supported here.

    Args:
        state (`RunState`):
            Initialized nonterminal state immediately before ``record``.
        record (`RunRecord`):
            A ``NodeStartedRecord``, ``RunResumedRecord``,
            ``NodeSucceededRecord``, ``NodeRoutedRecord``, ``NodeExpandedRecord``,
            or ``NodeFailedRecord`` to validate and copy.
        grow_builder (`GrowBuilder | None`, *optional*):
            Required only for ``NodeExpandedRecord`` replay; rebuilds and verifies
            the recorded fragment through the same injected path used live.

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
    if isinstance(record, NodeExpandedRecord):
        if not callable(grow_builder):
            raise TypeError("NodeExpandedRecord replay requires a callable grow_builder")
        return _prepare_growth_record(state, record, grow_builder)
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


def _prepare_growth_record(
    state: RunState,
    record: NodeExpandedRecord,
    grow_builder: GrowBuilder,
) -> PreparedTransition:
    """Rebuild and verify one historical expansion through the live builder path."""
    _validate_growth_record(state, record)
    seed_wire, builder_seed = canonical_seed(record.seed)
    if seed_wire != record.seed:
        raise ValueError("NodeExpandedRecord seed is not in canonical wire form")
    fragment = _build_growth_fragment(
        state.flow.nodes[record.spawner_id],
        builder_seed,
        grow_builder,
    )
    validated_fragment = _validate_growth_fragment(state, fragment)
    actual_digest = expansion_digest(fragment)
    if actual_digest != record.topology_digest:
        raise ValueError(
            "NodeExpandedRecord topology digest mismatch: "
            f"recorded {record.topology_digest!r}, rebuilt {actual_digest!r}"
        )
    return _prepare_growth_transition(
        state,
        record,
        fragment,
        actual_digest=actual_digest,
        validated_fragment=validated_fragment,
    )


def _prepare_growth_transition(
    state: RunState,
    record: NodeExpandedRecord,
    fragment: Flow,
    *,
    actual_digest: str | None = None,
    validated_fragment: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
) -> PreparedTransition:
    """Freeze one validated fragment, final topology, and canonical public record."""
    _validate_growth_record(state, record)
    validated = (
        _validate_growth_fragment(state, fragment)
        if validated_fragment is None
        else validated_fragment
    )
    digest = expansion_digest(fragment) if actual_digest is None else actual_digest
    if digest != record.topology_digest:
        raise ValueError("NodeExpandedRecord topology digest disagrees with its fragment")
    delta, topology = _prepare_growth_topology(
        state,
        record,
        fragment,
        validated_fragment=validated,
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
    if not isinstance(canonical_record, NodeExpandedRecord):
        raise TypeError("node-expansion envelope decoded to a different record kind")
    if canonical_record != record:
        raise ValueError("NodeExpandedRecord is not in canonical wire form")

    start_id = state.flow.start_id
    end_id = state.flow.end_id
    plan = _GrowthPlan(
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
        topology=topology,
    )
    return PreparedTransition(
        record=canonical_record,
        expected_run_id=canonical_envelope.run_id,
        expected_sequence=canonical_envelope.seq,
        _plan=plan,
        observers=(_growth_observer(canonical_record),),
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
    if isinstance(prepared._plan, _GrowthPlan):
        return _commit_growth(log, state, prepared)
    raise TypeError("unsupported prepared transition plan")


def _replay_records(
    state: RunState,
    envelopes: tuple[RecordEnvelope, ...],
    *,
    grow_builder: GrowBuilder,
) -> RunState:
    """Silently fold one complete historical prefix through the live reducer path.

    The caller supplies a pristine state around an already-private graph.  Every
    envelope is canonicalized and checked for one contiguous run before any
    transition is reduced.  ``commit_prepared`` then receives a read-only cursor,
    so replay exercises the same total apply functions as live dispatch without
    appending to historical authority or exposing historical observers.

    Args:
        state (`RunState`):
            Pristine reducer state around the graph that authored the run.
        envelopes (`tuple[RecordEnvelope, ...]`):
            Non-empty sequence-zero historical prefix beginning with ``RunStarted``.
        grow_builder (`GrowBuilder`):
            Injected compile-owned builder used to reconstruct and verify growth.

    Returns:
        `RunState`:
            The supplied state after every historical record has been applied.
    """
    if type(state) is not RunState:
        raise TypeError("_replay_records expects a RunState")
    if type(envelopes) is not tuple or any(
        not isinstance(envelope, RecordEnvelope) for envelope in envelopes
    ):
        raise TypeError("replay envelopes must be a tuple of RecordEnvelope values")
    if not callable(grow_builder):
        raise TypeError("grow_builder must be callable")
    _require_pristine(state)
    if not envelopes:
        raise ValueError("cannot replay an empty run-record prefix")

    canonical_bytes: list[bytes] = []
    run_id: str | None = None
    for expected_sequence, supplied in enumerate(envelopes):
        encoded = encode_record(supplied)
        envelope = decode_record(encoded)
        if envelope.seq != expected_sequence:
            raise ValueError(
                "replay sequence mismatch: "
                f"expected {expected_sequence}, found {envelope.seq}"
            )
        if run_id is None:
            run_id = envelope.run_id
        elif envelope.run_id != run_id:
            raise ValueError(
                "replay run_id mismatch: "
                f"expected {run_id!r}, found {envelope.run_id!r}"
            )
        canonical_bytes.append(encoded)

    first = decode_record(canonical_bytes[0])
    if not isinstance(first.record, RunStartedRecord):
        raise ValueError("the first historical run record must be RunStarted")
    cursor = _HistoricalLogCursor(tuple(canonical_bytes))
    start = first.record
    prepared = prepare_run_started(
        state,
        RunStartData(
            run_id=first.run_id,
            flow_fingerprint=start.flow_fingerprint,
            initial_pool=decode_pool(start.initial_pool),
            start_wiring_override=start.start_wiring_override,
            effective_llm_by_static_path=start.effective_llm_by_static_path,
            num_workers=start.num_workers,
        ),
    )
    commit_prepared(cursor, state, prepared)

    for encoded in canonical_bytes[1:]:
        record = decode_record(encoded).record
        prepared = prepare_record(state, record, grow_builder=grow_builder)
        commit_prepared(cursor, state, prepared)

    if cursor.next_sequence != len(canonical_bytes):
        raise ValueError("replay did not consume the complete historical prefix")
    return state


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


def _commit_growth(
    log: RunLog,
    state: RunState,
    prepared: PreparedTransition,
) -> tuple[Any, ...]:
    """Validate, append, and install one fully materialized flat expansion."""
    plan = prepared._plan
    if not isinstance(plan, _GrowthPlan):
        raise TypeError("growth commit requires a growth plan")
    if not isinstance(prepared.record, NodeExpandedRecord):
        raise TypeError("growth plan requires a NodeExpandedRecord")
    _require_prepared_origin(state, plan)
    _require_initialized_nonterminal(state)
    _require_authority_coordinates(log, state, prepared, initialized=True)
    _require_unchanged_record(prepared, plan.envelope_bytes)
    delta = plan.delta
    if not (
        prepared.record.spawner_id == delta.spawner_id
        and prepared.record.prune_ids == delta.prune_ids
        and prepared.record.topology_digest == delta.topology_digest
    ):
        raise ValueError("growth record and prevalidated delta disagree")
    apply = _materialize_growth(plan)

    _append_and_verify(log, prepared)
    state.flow.nodes = apply.nodes
    state.flow.edges = apply.edges
    state.flow.wiring = apply.wiring
    state.flow._incoming = apply.incoming
    state.flow._outgoing = apply.outgoing
    state.pool.store = apply.pool_store
    state.node_state = apply.node_state
    state.edge_state = apply.edge_state
    state.ready = apply.ready
    state.in_flight = apply.in_flight
    state.depth = apply.depth
    state.wiring_overrides = apply.wiring_overrides
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
        | _GrowthPlan
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
        and type(state.initial_static_ids) is frozenset
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
    if state.flow.start_id not in state.initial_static_ids:
        raise ValueError("initialized run state START is not an initial static node")


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


def _require_expanding_node(state: RunState, node_id: str) -> None:
    """Require an existing unknown spawner in the ordered in-flight frontier."""
    if not node_id:
        raise ValueError("NodeExpanded node_id must be non-empty")
    if node_id not in state.flow.nodes:
        raise ValueError(f"NodeExpanded node {node_id!r} does not exist in the flow")
    if state.node_state.get(node_id) is not NodeState.UNKNOWN:
        raise ValueError(f"NodeExpanded node {node_id!r} must have UNKNOWN disposition")
    if node_id not in state.in_flight:
        raise ValueError(f"NodeExpanded node {node_id!r} is not in-flight")
    if state.in_flight[node_id] is not None:
        raise ValueError("in-flight must be an ordered set with None values")
    if state.flow.nodes[node_id].is_spawner is not True:
        raise ValueError(f"NodeExpanded node {node_id!r} is not a graph spawner")


def _validate_growth_record(state: RunState, record: NodeExpandedRecord) -> None:
    """Validate exact durable expansion carriers and its executing spawner."""
    if type(record.spawner_id) is not str:
        raise TypeError("NodeExpandedRecord spawner_id must be a string")
    if not record.spawner_id:
        raise ValueError("NodeExpandedRecord spawner_id must be non-empty")
    if type(record.prune_ids) is not tuple or any(
        type(node_id) is not str for node_id in record.prune_ids
    ):
        raise TypeError("NodeExpandedRecord prune_ids must be a tuple of strings")
    if any(not node_id for node_id in record.prune_ids):
        raise ValueError("NodeExpandedRecord prune IDs must be non-empty")
    if record.prune_ids != tuple(sorted(set(record.prune_ids))):
        raise ValueError("NodeExpandedRecord prune IDs must be sorted and unique")
    if type(record.topology_digest) is not str:
        raise TypeError("NodeExpandedRecord topology_digest must be a string")
    if len(record.topology_digest) != 64 or any(
        char not in "0123456789abcdef" for char in record.topology_digest
    ):
        raise ValueError("NodeExpandedRecord topology_digest must be lowercase SHA-256")
    _require_expanding_node(state, record.spawner_id)


def _build_growth_fragment(
    spawner: Node,
    builder_seed: Any,
    grow_builder: GrowBuilder,
) -> Flow:
    """Call the injected builder with detached inputs and detach its fragment."""
    builder_spawner = copy.deepcopy(spawner)
    if not isinstance(builder_spawner, Node):
        raise TypeError("growth spawner deepcopy must remain a Node")
    built = grow_builder(builder_spawner, builder_seed)
    if not isinstance(built, Flow):
        _raise_node_data(TypeError("grow_builder must return a Flow"))
    try:
        fragment = copy.deepcopy(built)
    except Exception as exc:  # noqa: BLE001 - builder-owned fragment detachment
        _raise_node_data(exc)
    if not isinstance(fragment, Flow):  # defensive against a hostile deepcopy hook
        _raise_node_data(
            TypeError("grow_builder fragment deepcopy must remain a Flow")
        )
    return fragment


def _prepare_growth_topology(
    state: RunState,
    record: NodeExpandedRecord,
    fragment: Flow,
    *,
    validated_fragment: tuple[tuple[str, ...], tuple[str, ...]],
) -> tuple[_GrowthDelta, _GrowthTopology]:
    """Validate growth and materialize every final topology/state container."""
    new_node_ids, new_edge_ids = validated_fragment
    if len(state.flow.nodes) + len(new_node_ids) > MAX_TOTAL_NODES:
        _raise_node_data(
            ValueError(
                f"expansion exceeded node budget ({MAX_TOTAL_NODES}) at spawner "
                f"{record.spawner_id!r}"
            )
        )
    derived_depth = _validated_growth_depth(state, record.spawner_id)
    prune_ids = frozenset(record.prune_ids)
    _validate_growth_prune(
        state,
        record.spawner_id,
        prune_ids,
        frozenset(new_node_ids),
    )
    _validate_prune_dependencies(state, fragment, prune_ids)

    combined_nodes = dict(state.flow.nodes)
    combined_nodes.update(fragment.nodes)
    combined_edges = [*state.flow.edges, *fragment.edges]
    combined_wiring = copy.deepcopy(state.flow.wiring)
    combined_wiring.update(copy.deepcopy(fragment.wiring))

    final_nodes = {
        node_id: node
        for node_id, node in combined_nodes.items()
        if node_id not in prune_ids
    }
    final_edges = [
        edge
        for edge in combined_edges
        if edge.from_ not in prune_ids and edge.to not in prune_ids
    ]
    final_wiring = {
        node_id: wiring
        for node_id, wiring in combined_wiring.items()
        if node_id not in prune_ids
    }
    incoming, outgoing = _growth_adjacency(final_nodes, final_edges)

    node_state = dict(state.node_state)
    node_state.update((node_id, NodeState.UNKNOWN) for node_id in new_node_ids)
    node_state[record.spawner_id] = NodeState.EXPANDED
    for node_id in prune_ids:
        node_state.pop(node_id, None)

    edge_state = dict(state.edge_state)
    edge_state.update((edge_id, NodeState.UNKNOWN) for edge_id in new_edge_ids)
    surviving_edge_ids = {edge.id for edge in final_edges}
    edge_state = {
        edge_id: disposition
        for edge_id, disposition in edge_state.items()
        if edge_id in surviving_edge_ids
    }

    pool_store = dict(state.pool.store)
    depth = dict(state.depth)
    wiring_overrides = copy.deepcopy(state.wiring_overrides)
    for node_id in prune_ids:
        pool_store.pop(node_id, None)
        depth.pop(node_id, None)
        wiring_overrides.pop(node_id, None)
    if derived_depth is not None:
        for node_id, node in fragment.nodes.items():
            if node.is_spawner is True or node.commit_as == record.spawner_id:
                depth[node_id] = derived_depth

    ready = tuple(
        node_id for node_id in state.ready if node_id not in prune_ids
    ) + (fragment.start_id,)
    in_flight = tuple(
        node_id
        for node_id in state.in_flight
        if node_id != record.spawner_id and node_id not in prune_ids
    )

    delta = _GrowthDelta(
        spawner_id=record.spawner_id,
        prune_ids=record.prune_ids,
        topology_digest=record.topology_digest,
        fragment_start_id=fragment.start_id,
        fragment_end_id=fragment.end_id,
        new_node_ids=new_node_ids,
        new_edge_ids=new_edge_ids,
        derived_depth=derived_depth,
    )
    topology = _GrowthTopology(
        nodes=tuple(final_nodes.items()),
        edges=tuple(final_edges),
        wiring=tuple(final_wiring.items()),
        incoming=tuple(
            (node_id, tuple(edges)) for node_id, edges in incoming.items()
        ),
        outgoing=tuple(
            (node_id, tuple(edges)) for node_id, edges in outgoing.items()
        ),
        pool_store=tuple(pool_store.items()),
        node_state=tuple(node_state.items()),
        edge_state=tuple(edge_state.items()),
        ready=ready,
        in_flight=in_flight,
        depth=tuple(depth.items()),
        wiring_overrides=tuple(wiring_overrides.items()),
    )
    return delta, topology


def _validate_growth_fragment(
    state: RunState,
    fragment: Flow,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Mark only fragment-shape rejections as node-owned transition data."""
    try:
        return _validate_growth_fragment_data(state, fragment)
    except _NodeDataError:
        raise
    except (TypeError, ValueError) as exc:
        _raise_node_data(exc)


def _validate_growth_fragment_data(
    state: RunState,
    fragment: Flow,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Require a collision-free, self-contained, acyclic fragment."""
    if type(fragment.nodes) is not dict:
        raise TypeError("growth fragment nodes must be a dict")
    if type(fragment.edges) is not list:
        raise TypeError("growth fragment edges must be a list")
    if type(fragment.wiring) is not dict:
        raise TypeError("growth fragment wiring must be a dict")
    if type(fragment.start_id) is not str or not fragment.start_id:
        raise ValueError("growth fragment start_id must be a non-empty string")
    if type(fragment.end_id) is not str or not fragment.end_id:
        raise ValueError("growth fragment end_id must be a non-empty string")
    if fragment.start_id not in fragment.nodes:
        raise ValueError("growth fragment start_id is missing from its nodes")
    if fragment.end_id not in fragment.nodes:
        raise ValueError("growth fragment end_id is missing from its nodes")

    new_node_ids: list[str] = []
    for node_id, node in fragment.nodes.items():
        if type(node_id) is not str:
            raise TypeError("growth fragment node IDs must be strings")
        if not node_id:
            raise ValueError("growth fragment node IDs must be non-empty")
        if not isinstance(node, Node):
            raise TypeError(f"growth fragment node {node_id!r} must be a Node")
        if node.id != node_id:
            raise ValueError(
                f"growth fragment node key {node_id!r} does not equal node.id "
                f"{node.id!r}"
            )
        if node_id in state.flow.nodes:
            raise ValueError(
                f"growth fragment node {node_id!r} collides with live topology"
            )
        new_node_ids.append(node_id)

    if set(fragment.wiring) != set(fragment.nodes):
        raise ValueError("growth fragment wiring keys must exactly equal fragment nodes")
    combined_nodes = dict(state.flow.nodes)
    combined_nodes.update(fragment.nodes)
    combined_node_ids = set(combined_nodes)
    producer_types = {
        node_id: node.output_type
        for node_id, node in combined_nodes.items()
        if node.output_type is not None
    }
    for node_id, node in fragment.nodes.items():
        for attribute in ("commit_as", "origin_id"):
            target = getattr(node, attribute)
            if target is not None and type(target) is not str:
                raise TypeError(
                    f"growth fragment node {node_id!r} {attribute} must be a "
                    "string or None"
                )
            if target == "":
                raise ValueError(
                    f"growth fragment node {node_id!r} {attribute} must be non-empty"
                )
            if target is not None and target not in combined_node_ids:
                raise ValueError(
                    f"growth fragment node {node_id!r} {attribute} target "
                    f"{target!r} does not exist"
                )
        wiring = fragment.wiring[node_id]
        if type(wiring) is not dict or any(type(key) is not str for key in wiring):
            raise TypeError(
                f"growth fragment wiring for {node_id!r} must be a string-keyed dict"
            )
        params = node.params or []
        if type(params) is not list:
            raise TypeError(f"growth fragment node {node_id!r} params must be a list")
        declared: dict[str, bool] = {}
        for param in params:
            if type(param.name) is not str or not param.name:
                raise ValueError(
                    f"growth fragment node {node_id!r} has an invalid param name"
                )
            if param.name in declared:
                raise ValueError(
                    f"growth fragment node {node_id!r} has duplicate param "
                    f"{param.name!r}"
                )
            declared[param.name] = bool(param.required)
        reserved = node.reserved_wiring_keys()
        if type(reserved) is not set or any(
            type(key) is not str or not key for key in reserved
        ):
            raise TypeError(
                f"growth fragment node {node_id!r} reserved wiring keys are invalid"
            )
        actual = set(wiring)
        allowed = set(declared) | reserved
        extras = actual - allowed
        missing_reserved = reserved - actual
        missing_required = {
            name for name, required in declared.items() if required and name not in actual
        }
        if extras or missing_reserved or missing_required:
            raise ValueError(
                f"growth fragment node {node_id!r} wiring parity failed: "
                f"extra={sorted(extras)!r}, missing_reserved={sorted(missing_reserved)!r}, "
                f"missing_required={sorted(missing_required)!r}"
            )
        _validate_growth_wiring_refs(
            node_id,
            node,
            wiring,
            combined_node_ids,
            producer_types,
            reserved,
        )

    live_edge_ids = {edge.id for edge in state.flow.edges}
    new_edge_ids: list[str] = []
    seen_edge_ids: set[str] = set()
    for edge in fragment.edges:
        if type(edge) is not Edge:
            raise TypeError("growth fragment edges must contain only Edge values")
        if type(edge.id) is not str or not edge.id:
            raise ValueError("growth fragment edge IDs must be non-empty strings")
        if edge.id in seen_edge_ids:
            raise ValueError(f"growth fragment edge ID {edge.id!r} is duplicated")
        if edge.id in live_edge_ids:
            raise ValueError(
                f"growth fragment edge {edge.id!r} collides with live topology"
            )
        for attribute in ("source_handle", "input_group"):
            value = getattr(edge, attribute)
            if value is not None and type(value) is not str:
                raise TypeError(
                    f"growth fragment edge {edge.id!r} {attribute} must be a "
                    "string or None"
                )
            if value == "":
                raise ValueError(
                    f"growth fragment edge {edge.id!r} {attribute} must be non-empty"
                )
        for attribute in ("optional", "ordering"):
            if type(getattr(edge, attribute)) is not bool:
                raise TypeError(
                    f"growth fragment edge {edge.id!r} {attribute} must be a bool"
                )
        if type(edge.from_) is not str or edge.from_ not in combined_node_ids:
            raise ValueError(
                f"growth fragment edge {edge.id!r} has unknown source {edge.from_!r}"
            )
        if type(edge.to) is not str or edge.to not in fragment.nodes:
            raise ValueError(
                f"growth fragment edge {edge.id!r} must target a fragment node, "
                f"found {edge.to!r}"
            )
        seen_edge_ids.add(edge.id)
        new_edge_ids.append(edge.id)

    _validate_growth_reachability(fragment)
    _reject_growth_cycle(
        combined_node_ids,
        [*state.flow.edges, *fragment.edges],
    )
    return tuple(new_node_ids), tuple(new_edge_ids)


def _validate_growth_reachability(fragment: Flow) -> None:
    """Require every fragment node to be driven from its sole scheduled entry."""
    adjacency: dict[str, list[str]] = {
        node_id: [] for node_id in fragment.nodes
    }
    for edge in fragment.edges:
        if edge.from_ in adjacency:
            adjacency[edge.from_].append(edge.to)
    reachable: set[str] = set()
    pending = [fragment.start_id]
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        pending.extend(adjacency[node_id])
    unreachable = sorted(set(fragment.nodes) - reachable)
    if unreachable:
        raise ValueError(
            f"growth fragment nodes are unreachable from its start: {unreachable!r}"
        )


def _validate_growth_wiring_refs(
    node_id: str,
    node: Node,
    wiring: dict[str, Any],
    combined_node_ids: set[str],
    producer_types: dict[str, Any],
    reserved: set[str],
) -> None:
    """Apply the compiled language's closed reference rules to fragment wiring."""
    for param, source in wiring.items():
        if not isinstance(source, str):
            continue
        try:
            paths = expr_refs_of(source)
        except ExpressionError as exc:
            raise ValueError(
                f"growth fragment node {node_id!r} wiring {param!r} is invalid: {exc}"
            ) from exc
        for path in paths:
            extra_heads = (
                ("item",)
                if node.binds_per_item is True and param not in reserved
                else ()
            )
            error = _classify_path(
                path,
                combined_node_ids,
                set(),
                extra_heads,
                producer_types,
            )
            if error is not None:
                raise ValueError(
                    f"growth fragment node {node_id!r} wiring {param!r}: {error}"
                )


def _reject_growth_cycle(node_ids: set[str], edges: list[Edge]) -> None:
    """Reject a directed cycle in the predicted combined topology."""
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    in_degree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        if edge.from_ not in node_ids or edge.to not in node_ids:
            raise ValueError(f"combined growth edge {edge.id!r} has an unknown endpoint")
        adjacency[edge.from_].append(edge.to)
        in_degree[edge.to] += 1
    ready = [node_id for node_id, degree in in_degree.items() if degree == 0]
    visited = 0
    while ready:
        node_id = ready.pop()
        visited += 1
        for successor in adjacency[node_id]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                ready.append(successor)
    if visited != len(node_ids):
        stuck = sorted(node_id for node_id, degree in in_degree.items() if degree)
        raise ValueError(f"growth fragment creates a cycle involving {stuck!r}")


def _validated_growth_depth(state: RunState, spawner_id: str) -> int | None:
    """Derive the exact trait-driven reference depth for one expansion."""
    delta = state.flow.nodes[spawner_id].grow_depth_delta
    if delta is None:
        return None
    if type(delta) is not int or delta < 0:
        raise ValueError(
            f"spawner {spawner_id!r} grow_depth_delta must be None, 0, or a "
            "positive integer"
        )
    parent_depth = state.depth.get(spawner_id, 0)
    if type(parent_depth) is not int or parent_depth < 0:
        raise ValueError(f"spawner {spawner_id!r} has an invalid parent depth")
    depth = parent_depth + delta
    if delta > 0 and depth > MAX_REF_DEPTH:
        _raise_node_data(
            ValueError(
                f"expansion exceeded MAX_REF_DEPTH ({MAX_REF_DEPTH}) at "
                f"{spawner_id!r}"
            )
        )
    return depth


def _validate_growth_prune(
    state: RunState,
    spawner_id: str,
    prune_ids: frozenset[str],
    new_node_ids: frozenset[str],
) -> None:
    """Mark only unsafe node-requested prunes as outcome-data errors."""
    try:
        _validate_growth_prune_data(
            state,
            spawner_id,
            prune_ids,
            new_node_ids,
        )
    except _NodeDataError:
        raise
    except (TypeError, ValueError) as exc:
        _raise_node_data(exc)


def _validate_growth_prune_data(
    state: RunState,
    spawner_id: str,
    prune_ids: frozenset[str],
    new_node_ids: frozenset[str],
) -> None:
    """Enforce the exact settled dynamic-overlay prune safety matrix."""
    ready_ids = set(state.ready)
    spawner = state.flow.nodes[spawner_id]
    for node_id in prune_ids:
        if node_id not in state.flow.nodes:
            raise ValueError(f"growth prune node {node_id!r} does not exist")
        if node_id in (state.flow.start_id, state.flow.end_id):
            raise ValueError(f"growth cannot prune top-level boundary {node_id!r}")
        if node_id in new_node_ids:
            raise ValueError(f"growth cannot prune new fragment node {node_id!r}")
        if node_id in state.initial_static_ids:
            raise ValueError(f"growth cannot prune initial static node {node_id!r}")
        if node_id in ready_ids:
            raise ValueError(f"growth cannot prune ready node {node_id!r}")
        if node_id in state.in_flight:
            if node_id != spawner_id:
                raise ValueError(
                    f"growth cannot prune unrelated in-flight node {node_id!r}"
                )
            origin_id = spawner.origin_id
            if not (
                spawner.is_loop is True
                and type(origin_id) is str
                and origin_id
                and origin_id != spawner_id
                and origin_id in state.flow.nodes
                and origin_id not in prune_ids
            ):
                raise ValueError(
                    f"growth spawner {spawner_id!r} is not a self-prunable LOOP clone"
                )
            continue
        if state.node_state.get(node_id) not in (
            NodeState.TAKEN,
            NodeState.SKIPPED,
            NodeState.EXPANDED,
        ):
            raise ValueError(
                f"growth prune node {node_id!r} must be settled dynamic overlay"
            )


def _validate_prune_dependencies(
    state: RunState,
    fragment: Flow,
    prune_ids: frozenset[str],
) -> None:
    """Mark node-requested prune dependency conflicts as outcome-data errors."""
    try:
        _validate_prune_dependencies_data(state, fragment, prune_ids)
    except _NodeDataError:
        raise
    except (TypeError, ValueError) as exc:
        _raise_node_data(exc)


def _validate_prune_dependencies_data(
    state: RunState,
    fragment: Flow,
    prune_ids: frozenset[str],
) -> None:
    """Reject future work whose edge, wiring, commit, or origin reads a pruned ID."""
    if not prune_ids:
        return
    dependent_ids = set(fragment.nodes)
    dependent_ids.update(
        node_id
        for node_id, disposition in state.node_state.items()
        if disposition is NodeState.UNKNOWN and node_id not in prune_ids
    )
    combined_nodes = dict(state.flow.nodes)
    combined_nodes.update(fragment.nodes)
    combined_wiring = dict(state.flow.wiring)
    combined_wiring.update(fragment.wiring)
    for edge in [*state.flow.edges, *fragment.edges]:
        if edge.from_ in prune_ids and edge.to in dependent_ids:
            raise ValueError(
                f"surviving node {edge.to!r} depends on pruned edge source "
                f"{edge.from_!r}"
            )
    for node_id in dependent_ids:
        node = combined_nodes[node_id]
        if node.commit_as in prune_ids:
            raise ValueError(
                f"surviving node {node_id!r} commits to pruned node {node.commit_as!r}"
            )
        if node.origin_id in prune_ids:
            raise ValueError(
                f"surviving node {node_id!r} has pruned origin {node.origin_id!r}"
            )
        for source in combined_wiring.get(node_id, {}).values():
            if not isinstance(source, str):
                continue
            try:
                paths = expr_refs_of(source)
            except ExpressionError as exc:
                raise ValueError(
                    f"surviving node {node_id!r} has invalid wiring: {exc}"
                ) from exc
            for path in paths:
                if path.split(".", 1)[0] in prune_ids:
                    raise ValueError(
                        f"surviving node {node_id!r} wiring depends on pruned node"
                    )


def _growth_adjacency(
    nodes: dict[str, Node],
    edges: list[Edge],
) -> tuple[dict[str, list[Edge]], dict[str, list[Edge]]]:
    """Build detached incoming/outgoing indexes for a fully validated topology."""
    incoming: dict[str, list[Edge]] = {}
    outgoing: dict[str, list[Edge]] = {}
    for edge in edges:
        if edge.from_ not in nodes or edge.to not in nodes:
            raise ValueError(f"final growth edge {edge.id!r} has an unknown endpoint")
        outgoing.setdefault(edge.from_, []).append(edge)
        incoming.setdefault(edge.to, []).append(edge)
    return incoming, outgoing


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
    # The reserved default route may deliberately have no authored edge: a CASE
    # without ``else`` selects no branch and skip-floods every outgoing edge. Any
    # other unknown handle is invalid node-produced data.
    if not matched and record.handle != "default":
        _raise_node_data(
            ValueError(
                f"NodeRouted handle {record.handle!r} matches no outgoing edge of "
                f"{record.node_id!r}"
            )
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
        and type(state.initial_static_ids) is frozenset
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
        and state.initial_static_ids == frozenset(state.flow.nodes)
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
        | _GrowthPlan
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


def _growth_observer(record: NodeExpandedRecord) -> NodeExpanded:
    """Construct one independent canonical expansion observer."""
    _, seed = canonical_seed(record.seed)
    return NodeExpanded(
        node_id=record.spawner_id,
        grow=Grow(seed=seed, prune=frozenset(record.prune_ids)),
    )


def _materialize_growth(plan: _GrowthPlan) -> _GrowthApply:
    """Materialize every final container and fresh observer before durable append."""
    envelope = decode_record(plan.envelope_bytes)
    record = envelope.record
    if not isinstance(record, NodeExpandedRecord):
        raise TypeError("immutable growth plan does not contain a NodeExpandedRecord")
    if not (
        record.spawner_id == plan.delta.spawner_id
        and record.prune_ids == plan.delta.prune_ids
        and record.topology_digest == plan.delta.topology_digest
    ):
        raise ValueError("immutable growth plan record and delta disagree")
    topology = plan.topology
    return _GrowthApply(
        nodes=dict(topology.nodes),
        edges=list(topology.edges),
        wiring={
            node_id: copy.deepcopy(wiring)
            for node_id, wiring in topology.wiring
        },
        incoming={
            node_id: list(edges) for node_id, edges in topology.incoming
        },
        outgoing={
            node_id: list(edges) for node_id, edges in topology.outgoing
        },
        pool_store=dict(topology.pool_store),
        node_state=dict(topology.node_state),
        edge_state=dict(topology.edge_state),
        ready=deque(topology.ready),
        in_flight={node_id: None for node_id in topology.in_flight},
        depth=dict(topology.depth),
        wiring_overrides={
            node_id: copy.deepcopy(wiring)
            for node_id, wiring in topology.wiring_overrides
        },
        observers=(_growth_observer(record),),
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
    "GrowBuilder",
    "MAX_REF_DEPTH",
    "MAX_TOTAL_NODES",
    "PreparedTransition",
    "RunStartData",
    "RunState",
    "commit_prepared",
    "prepare_expanded",
    "prepare_node_failed",
    "prepare_node_routed",
    "prepare_node_started",
    "prepare_node_succeeded",
    "prepare_record",
    "prepare_run_resumed",
    "prepare_run_started",
]

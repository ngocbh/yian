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
from bisect import bisect_left
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, NoReturn

from yian.compile.manifest import expansion_digest
from yian.compile.model import CompiledFlow, Edge, Flow, GrowthRejected, validate_growth_metadata
from yian.compile.validation import _classify_path
from yian.diagnostics import Failure, OriginRef
from yian.events import (
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
)
from yian.expr import ExpressionError, expr_refs_of
from yian.expr.lineage import SourceRef
from yian.memory import (
    EvaluationEvidence,
    MemoryScope,
    MemorySnapshotEnvelope,
    MemorySpec,
    MemoryStack,
)
from yian.nodes.base import Grow, Node
from yian.nodes.contracts import (
    ChildFlowBoundary,
    OutputContract,
    Parameter,
    PartialRecordContract,
    SeedConnection,
    binding_source,
    output_contract_field_error,
    output_contract_origin,
    output_contract_type,
    validate_output_contract,
)
from yian.runtime.contracts import ForwardCapture, _validate_source
from yian.runtime.common import NodeState
from yian.runtime.trace import (
    RetainedTrace, SourceEntry, TraceDelta, TraceOccurrence, TraceTerminal,
    _copy_memory_scope, prepare_trace, select_seed_route,
)
from yian.runtime.log import (
    SCHEMA_VERSION,
    BackwardStartedRecord,
    BackwardReadyRecord,
    BackwardFinishedRecord,
    MemoryCommitFailedRecord,
    MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
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
from yian.runtime.learning import (
    RuntimeBackwardState,
    reduce_backward_record,
)
from yian.typesys.pool import VariablePool
from yian.typesys.values import (
    ANY_VALUE_ADAPTER,
    Type,
    TypeCheckError,
    TypedValue,
)


MAX_TOTAL_NODES = 10_000
"""Maximum live topology size before same-transition pruning."""

MAX_REF_DEPTH = 5
"""Maximum positive graph-growth nesting depth."""

GrowBuilder = Callable[[Node, Any], Flow]
"""Injected kind-aware interpreter from a spawner and canonical seed to a fragment."""


def _node_output_type(node: Node) -> Type | None:
    """Project a whole-value type while validating the closed contract sum."""
    return output_contract_type(node.output_contract)


def _canonical_output_value(
    output: Any,
    contract: OutputContract | None,
    declared_type: Type | None,
) -> tuple[dict[str, Any], TypedValue]:
    """Canonicalize live or replayed output through the same contract path."""
    if type(contract) is PartialRecordContract:
        output = validate_output_contract(contract, output)
    return canonical_typed(output, declared_type)


@dataclass(frozen=True, slots=True)
class _MemoryAuthority:
    """Immutable identity roots established by canonical RunStarted decoding."""

    snapshots: tuple[MemorySnapshotEnvelope, ...] | None
    learnable: bool
    memories: tuple[MemorySpec, ...] | None
    memory_stacks: tuple[MemoryStack, ...] | None
    memory_order: tuple[str, ...]


class _NodeDataError(Exception):
    """Private marker for an outcome-owned value rejected before durability."""

    def __init__(
        self,
        original: Exception,
        *,
        origin: OriginRef | None = None,
    ) -> None:
        super().__init__(str(original))
        self.original = original
        self.origin = origin


class _NodeDataTypeError(_NodeDataError, TypeError):
    """Node-data rejection retaining compatibility with ``TypeError`` callers."""


class _NodeDataValueError(_NodeDataError, ValueError):
    """Node-data rejection retaining compatibility with ``ValueError`` callers."""


def _raise_node_data(
    original: Exception,
    *,
    origin: OriginRef | None = None,
) -> NoReturn:
    """Raise the built-in-compatible marker matching one outcome-data error."""
    if isinstance(original, TypeError):
        raise _NodeDataTypeError(original, origin=origin) from original
    if isinstance(original, ValueError):
        raise _NodeDataValueError(original, origin=origin) from original
    raise _NodeDataError(original, origin=origin) from original


def _raise_growth_rejected(original: TypeError | ValueError) -> NoReturn:
    """Mark one explicit live node-produced growth rejection."""
    raise GrowthRejected(original) from original


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
        accepted_starts (`dict[str, int]`):
            Original accepted start sequence for each unfinished invocation.
        result (`RunSucceeded | RunFailed | None`):
            Derived terminal result, or ``None`` while the run remains live.
        terminal_unreachable_failure (`Failure | None`):
            First route-derived proof that the terminal output is unreachable,
            retained independently of later frontier and topology changes.
        depth (`dict[str, int]`):
            Graph-growth nesting depth keyed by live dynamic node ID.
        growth_owner (`dict[str, str]`):
            Live dynamic node ID to the live node whose accepted growth owns it.
            Pruning a node also retires the nodes it owns, transitively, once it
            and all of them are settled.  A self-pruning LOOP driver's unsettled
            nodes move to its nearest surviving owner.
        deferred_prune (`frozenset[str]`):
            Live node IDs a growth asked to prune while they or their grown nodes
            were unfinished; each later growth retires those that have settled.
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
        memory_snapshots (`tuple[MemorySnapshotEnvelope, ...] | None`):
            Canonical recorded snapshots, one per memory key.
        learnable (`bool`):
            Whether accepted eligible outcomes must project durable learning evidence.
        trace (`RetainedTrace | None`):
            Accepted executions, immutable producer entries and private backward
            handlers, enabled by recorded learning independently of memory scopes.
        backward_state (`RuntimeBackwardState`):
            Immutable accepted backward attempts and irreversible commit facts.
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
        self.accepted_starts: dict[str, int] = {}
        self.result: RunSucceeded | RunFailed | None = None
        self.terminal_unreachable_failure: Failure | None = None
        self.depth: dict[str, int] = {}
        self.growth_owner: dict[str, str] = {}
        self.deferred_prune: frozenset[str] = frozenset()
        self.wiring_overrides: dict[str, dict[str, Any]] = {}
        self.run_id: str | None = None
        self.flow_fingerprint: str | None = None
        self.next_sequence = 0
        self.num_workers = 0
        self.input: dict[str, Any] = {}
        self.effective_llm_by_static_path: dict[str, dict[str, Any]] = {}
        self.memory_snapshots: tuple[MemorySnapshotEnvelope, ...] | None = None
        self.memories: tuple[MemorySpec, ...] | None = None
        self.memory_stacks: tuple[MemoryStack, ...] | None = None
        self.learnable = False
        self.trace: RetainedTrace | None = None
        self.backward_state = RuntimeBackwardState()
        self._memory_authority = _MemoryAuthority(
            snapshots=None,
            learnable=False,
            memories=None,
            memory_stacks=None,
            memory_order=self.backward_state.memory_order,
        )
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
        memory_snapshots (`tuple[MemorySnapshotEnvelope, ...] | None`, *optional*):
            Canonical snapshots, one per memory key, or ``None`` to disable memory.
        learnable (`bool`, *optional*, defaults to `False`):
            Whether eligible accepted outcomes must carry learning evidence.
        memories (`tuple[MemorySpec, ...] | None`, *optional*):
            Sorted memory key table paired with ``memory_snapshots``.
        memory_stacks (`tuple[MemoryStack, ...] | None`, *optional*):
            Each scoped target's key stack, sorted by locator.
    """

    run_id: str
    flow_fingerprint: str
    initial_pool: VariablePool
    start_wiring_override: dict[str, Any]
    effective_llm_by_static_path: dict[str, dict[str, Any]]
    num_workers: int
    memory_snapshots: tuple[MemorySnapshotEnvelope, ...] | None = None
    learnable: bool = False
    memories: tuple[MemorySpec, ...] | None = None
    memory_stacks: tuple[MemoryStack, ...] | None = None


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
    trace_delta: TraceDelta | None = None


@dataclass(slots=True)
class _RunStartedApply:
    """Detached mutable containers fully materialized before durable append."""

    pool: VariablePool
    node_state: dict[str, NodeState]
    edge_state: dict[str, NodeState]
    ready: deque[str]
    in_flight: dict[str, None]
    accepted_starts: dict[str, int]
    terminal_unreachable_failure: Failure | None
    depth: dict[str, int]
    growth_owner: dict[str, str]
    wiring_overrides: dict[str, dict[str, Any]]
    run_id: str
    flow_fingerprint: str
    next_sequence: int
    num_workers: int
    input: dict[str, Any]
    effective_llm_by_static_path: dict[str, dict[str, Any]]
    memory_snapshots: tuple[MemorySnapshotEnvelope, ...] | None
    memories: tuple[MemorySpec, ...] | None
    memory_stacks: tuple[MemoryStack, ...] | None
    learnable: bool
    backward_state: RuntimeBackwardState
    memory_authority: _MemoryAuthority
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
    trace_delta: TraceDelta | None = None


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
    trace_delta: TraceDelta | None = None


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
    trace_delta: TraceDelta | None = None


@dataclass(frozen=True, slots=True)
class _RouteDelta:
    """Prevalidated ordered operations for one route and its skip closure."""

    node_id: str
    handle: str
    edge_updates: tuple[tuple[str, NodeState], ...]
    skipped_node_ids: tuple[str, ...]
    newly_ready_ids: tuple[str, ...]
    terminal_unreachable_failure: Failure | None


@dataclass(frozen=True, slots=True)
class _TerminalDelta:
    """Immutable derived terminal authority attached to one causal outcome.

    ``output_bytes`` is a canonical tagged typed value for success and ``None``
    for failure. ``succeeded`` distinguishes a typed null output from failure;
    ``failure`` is the one source-neutral authority for an unreachable terminal.
    """

    succeeded: bool
    output_bytes: bytes | None
    failure: Failure | None


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
    """Opaque authority anchors and one immutable canonical failure."""

    originating_state: RunState
    originating_flow: CompiledFlow
    envelope_bytes: bytes
    start_id: str
    start_node_identity: int
    end_id: str
    end_node_identity: int | None
    failure: Failure
    trace_delta: TraceDelta | None = None


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
    trace_delta: TraceDelta | None = None


@dataclass(frozen=True, slots=True)
class _BackwardPlan:
    """Canonical record authority and a detached post-terminal attempt delta."""

    originating_state: RunState
    originating_flow: CompiledFlow
    envelope_bytes: bytes
    start_id: str
    start_node_identity: int
    end_id: str
    end_node_identity: int | None
    originating_backward_state: RuntimeBackwardState
    next_backward_state: RuntimeBackwardState
    trace_delta: TraceDelta | None = None


@dataclass(frozen=True, slots=True)
class _GrowthDelta:
    """Immutable scalar authority for one flat deterministic expansion."""

    spawner_id: str
    prune_ids: tuple[str, ...]
    retired_ids: tuple[str, ...]
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
    accepted_starts: tuple[tuple[str, int], ...]
    depth: tuple[tuple[str, int], ...]
    growth_owner: tuple[tuple[str, str], ...]
    deferred_prune: frozenset[str]
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
    accepted_starts: dict[str, int]
    depth: dict[str, int]
    growth_owner: dict[str, str]
    deferred_prune: frozenset[str]
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
        | _BackwardPlan
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
        expected = self._envelope_bytes[self._next_sequence]
        envelope = decode_record(expected)
        candidate = encode_record(
            RecordEnvelope(
                schema=envelope.schema,
                run_id=self._run_id,
                seq=self._next_sequence,
                record=record,
            )
        )
        if candidate != expected:
            raise ValueError(
                "prepared transition does not match the historical run record"
            )
        self._next_sequence += 1
        return envelope


def prepare_run_started(
    state: RunState,
    data: RunStartData,
    *,
    schema: int = SCHEMA_VERSION,
) -> PreparedTransition:
    """Prepare canonical initialization without mutating ``state``.

    Args:
        state (`RunState`):
            A pristine state whose private graph contains its synthesized START node.
        data (`RunStartData`):
            Complete pool, identity, configuration, and worker selection for the run.
        schema (`int`, *optional*, defaults to `SCHEMA_VERSION`):
            Exact supported envelope version; historical replay supplies the
            recorded version without rewriting accepted authority.

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
        memory_snapshots=data.memory_snapshots,
        learnable=data.learnable,
        memories=data.memories,
        memory_stacks=data.memory_stacks,
    )
    # The public envelope codec is the sole payload authority.  Decode its canonical
    # bytes immediately so the prepared record is exactly what either log will retain.
    canonical_bytes = encode_record(
        RecordEnvelope(
            schema=schema,
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
        _plan=replace(plan, trace_delta=_prepare_record_trace(state, canonical_envelope)),
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
    evidence: EvaluationEvidence | None = None,
    forward_capture: ForwardCapture | None = None,
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
        evidence (`EvaluationEvidence | None`, *optional*):
            Frozen visible evidence captured during this exact evaluation.
        forward_capture (`ForwardCapture | None`, *optional*):
            Canonical binding capture, required exactly when learning is enabled.

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
    target_node = state.flow.nodes[target]
    declared_type = _node_output_type(target_node)
    output_contract = target_node.output_contract
    output_origin = output_contract_origin(output_contract, ())
    try:
        value_wire, typed = _canonical_output_value(
            output,
            output_contract,
            declared_type,
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TypeCheckError):
            output_origin = output_contract_origin(output_contract, exc.path)
        _raise_node_data(exc, origin=output_origin)
    return _prepare_success_transition(
        state,
        NodeSucceededRecord(
            node_id=node_id,
            commit_target=target,
            value=value_wire,
            evidence=evidence,
            forward_capture=forward_capture,
        ),
        typed,
    )


def prepare_node_routed(
    state: RunState,
    node_id: str,
    handle: str,
    *,
    evidence: EvaluationEvidence | None = None,
    forward_capture: ForwardCapture | None = None,
) -> PreparedTransition:
    """Prepare one selected route and its derived recursive skip closure.

    Args:
        state (`RunState`):
            Initialized nonterminal state with ``node_id`` in ordered in-flight work.
        node_id (`str`):
            Exact non-empty ID of the executing router.
        handle (`str`):
            Exact non-empty handle selected by the router.
        evidence (`EvaluationEvidence | None`, *optional*):
            Frozen visible evidence captured during this exact evaluation.
        forward_capture (`ForwardCapture | None`, *optional*):
            Canonical binding capture, required exactly when learning is enabled.

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
        NodeRoutedRecord(
            node_id=node_id, handle=handle, evidence=evidence,
            forward_capture=forward_capture,
        ),
        derive_terminal_failure=True,
    )


def prepare_expanded(
    state: RunState,
    node_id: str,
    grow: Grow,
    grow_builder: GrowBuilder,
    *,
    evidence: EvaluationEvidence | None = None,
    forward_capture: ForwardCapture | None = None,
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
        evidence (`EvaluationEvidence | None`, *optional*):
            Frozen visible evidence captured during this exact evaluation.
        forward_capture (`ForwardCapture | None`, *optional*):
            Canonical binding capture, required exactly when learning is enabled.

    Returns:
        `PreparedTransition`:
            Canonical ``NodeExpandedRecord`` and detached append-before-apply plan.

    Raises:
        `GrowthRejected`:
            The node-produced prune set or seed, returned fragment constraints,
            growth budget/depth ceiling, or requested prune policy is invalid.
        `TypeError`:
            A host/setup carrier or builder has the wrong type, or the builder does
            not return a `Flow`.
        `ValueError`:
            The live state, spawner, topology, trait, or replay-independent host
            invariant is invalid.
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
        _raise_growth_rejected(
            TypeError("Grow.prune must be a frozenset of node IDs")
        )
    if any(type(prune_id) is not str for prune_id in grow.prune):
        _raise_growth_rejected(TypeError("Grow.prune must contain only strings"))
    if any(not prune_id for prune_id in grow.prune):
        _raise_growth_rejected(ValueError("Grow.prune node IDs must be non-empty"))
    _require_initialized_nonterminal(state)
    _require_expanding_node(state, node_id)

    try:
        seed_wire, builder_seed = canonical_seed(grow.seed)
    except (TypeError, ValueError) as exc:
        _raise_growth_rejected(exc)
    fragment = _build_growth_fragment(
        state.flow.nodes[node_id],
        builder_seed,
        grow_builder,
    )
    validated_fragment = _validate_growth_fragment(state, fragment)
    digest = expansion_digest(fragment)
    return _prepare_growth_transition(
        state,
        NodeExpandedRecord(
            spawner_id=node_id,
            seed=seed_wire,
            prune_ids=tuple(sorted(grow.prune)),
            topology_digest=digest,
            evidence=evidence,
            forward_capture=forward_capture,
        ),
        fragment,
        actual_digest=digest,
        validated_fragment=validated_fragment,
    )


def prepare_node_failed(
    state: RunState,
    failure: Failure,
    *,
    evidence: EvaluationEvidence | None = None,
    forward_capture: ForwardCapture | None = None,
) -> PreparedTransition:
    """Prepare one terminal node failure with its canonical durable authority.

    Args:
        state (`RunState`):
            Initialized nonterminal state with ``node_id`` in ordered in-flight work.
        failure (`Failure`):
            Immutable semantic failure whose non-empty ``node_id`` identifies
            the executing node.
        evidence (`EvaluationEvidence | None`, *optional*):
            Frozen visible evidence captured before the evaluation failed.
        forward_capture (`ForwardCapture | None`, *optional*):
            Canonical binding capture, including unavailable inputs on early failure.
    Returns:
        `PreparedTransition`:
            Canonical ``NodeFailedRecord`` and detached append-before-apply plan.

    Raises:
        `TypeError`:
            The failure has the wrong exact type.
        `ValueError`:
            State or the executing node is not eligible to fail.
    """
    if type(state) is not RunState:
        raise TypeError("prepare_node_failed expects a RunState")
    if type(failure) is not Failure:
        raise TypeError("prepare_node_failed expects a Failure")
    record = NodeFailedRecord(failure=failure, evidence=evidence, forward_capture=forward_capture)
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
    schema: int = SCHEMA_VERSION,
) -> PreparedTransition:
    """Validate one supported authoritative record against the current frontier.

    This is the shared reducer seam for live dispatch and later historical replay.
    Graph records require an initialized nonterminal state. Backward and
    memory-commit records require an initialized primary terminal with learning
    enabled.

    Args:
        state (`RunState`):
            Initialized reducer state immediately before ``record``.
        record (`RunRecord`):
            One supported graph/frontier or backward-attempt record.
        grow_builder (`GrowBuilder | None`, *optional*):
            Required only for ``NodeExpandedRecord`` replay; rebuilds and verifies
            the recorded fragment through the same injected path used live.
        schema (`int`, *optional*, defaults to `SCHEMA_VERSION`):
            Exact supported envelope version belonging to this transition.

    Returns:
        `PreparedTransition`:
            Canonically copied record, immutable authority snapshots, and observer.

    Raises:
        `TypeError`:
            State has the wrong type or the record type is unsupported for preparation.
        `ValueError`:
            The record is not the exact next transition for this state.
    """
    if type(state) is not RunState:
        raise TypeError("prepare_record expects a RunState")

    if isinstance(record, (BackwardStartedRecord, BackwardReadyRecord, BackwardFinishedRecord,
                           MemoryCommitRequestedRecord, MemoryCommitSucceededRecord, MemoryCommitFailedRecord)):
        return _prepare_backward_transition(state, record, schema=schema)

    _require_initialized_nonterminal(state)

    if isinstance(record, NodeSucceededRecord):
        typed = _validate_success_value(state, record)
        return _prepare_success_transition(state, record, typed, schema=schema)
    if isinstance(record, NodeRoutedRecord):
        _validate_route_record(state, record)
        return _prepare_route_transition(state, record, schema=schema)
    if isinstance(record, NodeExpandedRecord):
        if not callable(grow_builder):
            raise TypeError("NodeExpandedRecord replay requires a callable grow_builder")
        return _prepare_growth_record(state, record, grow_builder, schema=schema)
    if isinstance(record, NodeFailedRecord):
        return _prepare_failure_transition(state, record, schema=schema)
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
            schema=schema,
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
        _plan=replace(plan, trace_delta=_prepare_record_trace(state, canonical_envelope)),
        observers=observers,
    )


def _prepare_success_transition(
    state: RunState,
    record: NodeSucceededRecord,
    typed: TypedValue,
    *,
    schema: int = SCHEMA_VERSION,
) -> PreparedTransition:
    """Freeze one fully validated success delta and its canonical public record."""
    _validate_node_evidence(state, record.node_id, record.evidence)
    delta = _prepare_success_delta(state, record, typed)
    terminal = _prepare_terminal_delta(
        state,
        completing_node_id=delta.node_id,
        newly_ready_ids=delta.newly_ready_ids,
        pending_commit_target=delta.commit_target,
        pending_value_bytes=delta.value_bytes,
        edge_updates={},
        terminal_unreachable_failure=state.terminal_unreachable_failure,
    )
    run_id = state.run_id
    if not isinstance(run_id, str):  # established by initialized-state validation
        raise ValueError("run state is not initialized")
    canonical_bytes = encode_record(
        RecordEnvelope(
            schema=schema,
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
        _plan=replace(plan, trace_delta=_prepare_record_trace(
            state, canonical_envelope,
            terminal_succeeded=terminal.succeeded if terminal is not None else None,
            terminal_output_bytes=terminal.output_bytes if terminal is not None else None,
            terminal_failure=terminal.failure if terminal is not None else None,
        )),
        observers=observers,
    )


def _canonical_wire_failure(failure: Failure) -> Failure:
    """Collapse origin subclasses to the wire's single-key identity shape."""
    return replace(
        failure,
        trace=tuple(OriginRef(origin.key) for origin in failure.trace),
    )


def _prepare_route_transition(
    state: RunState,
    record: NodeRoutedRecord,
    *,
    derive_terminal_failure: bool = False,
    schema: int = SCHEMA_VERSION,
) -> PreparedTransition:
    """Freeze one fully validated route delta and its canonical public record."""
    _validate_node_evidence(state, record.node_id, record.evidence)
    delta, edge_updates, causal_failure = _prepare_route_delta(
        state,
        record,
        derive_terminal_failure=derive_terminal_failure,
    )
    record_for_wire = replace(
        record,
        terminal_failure=(
            _canonical_wire_failure(causal_failure)
            if causal_failure is not None
            else None
        ),
    )
    run_id = state.run_id
    if not isinstance(run_id, str):  # established by initialized-state validation
        raise ValueError("run state is not initialized")
    canonical_bytes = encode_record(
        RecordEnvelope(
            schema=schema,
            run_id=run_id,
            seq=state.next_sequence,
            record=record_for_wire,
        )
    )
    canonical_envelope = decode_record(canonical_bytes)
    canonical_record = canonical_envelope.record
    if not isinstance(canonical_record, NodeRoutedRecord):
        raise TypeError("node-route envelope decoded to a different record kind")
    if canonical_record != record_for_wire:
        raise ValueError("NodeRoutedRecord is not in canonical wire form")
    if causal_failure is not None:
        if canonical_record.terminal_failure is None:
            raise ValueError("causal route lost its terminal failure")
        delta = replace(
            delta,
            terminal_unreachable_failure=canonical_record.terminal_failure,
        )
    terminal = _prepare_terminal_delta(
        state,
        completing_node_id=delta.node_id,
        newly_ready_ids=delta.newly_ready_ids,
        pending_commit_target=None,
        pending_value_bytes=None,
        edge_updates=edge_updates,
        terminal_unreachable_failure=delta.terminal_unreachable_failure,
    )

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
        _plan=replace(plan, trace_delta=_prepare_record_trace(
            state, canonical_envelope,
            terminal_succeeded=terminal.succeeded if terminal is not None else None,
            terminal_output_bytes=terminal.output_bytes if terminal is not None else None,
            terminal_failure=terminal.failure if terminal is not None else None,
        )),
        observers=observers,
    )


def _prepare_failure_transition(
    state: RunState,
    record: NodeFailedRecord,
    *,
    schema: int = SCHEMA_VERSION,
) -> PreparedTransition:
    """Freeze one validated terminal diagnostic and its canonical public record."""
    _validate_failure_record(state, record)
    run_id = state.run_id
    if not isinstance(run_id, str):  # established by initialized-state validation
        raise ValueError("run state is not initialized")
    record_for_wire = replace(
        record,
        failure=_canonical_wire_failure(record.failure),
    )
    canonical_bytes = encode_record(
        RecordEnvelope(
            schema=schema,
            run_id=run_id,
            seq=state.next_sequence,
            record=record_for_wire,
        )
    )
    canonical_envelope = decode_record(canonical_bytes)
    canonical_record = canonical_envelope.record
    if not isinstance(canonical_record, NodeFailedRecord):
        raise TypeError("node-failure envelope decoded to a different record kind")
    if canonical_record != record_for_wire:
        raise ValueError("NodeFailedRecord is not in canonical wire form")

    failure = canonical_record.failure
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
        failure=failure,
    )
    return PreparedTransition(
        record=canonical_record,
        expected_run_id=canonical_envelope.run_id,
        expected_sequence=canonical_envelope.seq,
        _plan=replace(plan, trace_delta=_prepare_record_trace(
            state, canonical_envelope, terminal_succeeded=False, terminal_failure=failure,
        )),
        observers=_failure_observers(failure),
    )


def _prepare_growth_record(
    state: RunState,
    record: NodeExpandedRecord,
    grow_builder: GrowBuilder,
    *,
    schema: int = SCHEMA_VERSION,
) -> PreparedTransition:
    """Rebuild and verify one historical expansion through the live builder path."""
    _validate_growth_record(state, record)
    try:
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
            schema=schema,
        )
    except GrowthRejected as exc:
        original = exc.original
        raise original.with_traceback(original.__traceback__) from None


def _prepare_growth_transition(
    state: RunState,
    record: NodeExpandedRecord,
    fragment: Flow,
    *,
    actual_digest: str | None = None,
    validated_fragment: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
    schema: int = SCHEMA_VERSION,
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
            schema=schema,
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
    if state.learnable and type(canonical_record.forward_capture) is ForwardCapture:
        for connection in fragment.input_connections:
            if select_seed_route(canonical_record.forward_capture, connection) is None:
                _raise_growth_rejected(ValueError(
                    f"seed connection {connection.seed_path!r} was not captured"
                ))
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
        _plan=replace(plan, trace_delta=_prepare_record_trace(
            state, canonical_envelope, input_connections=fragment.input_connections,
            child_boundaries=fragment.child_boundaries, retired_ids=delta.retired_ids,
        )),
        observers=(_growth_observer(canonical_record),),
    )


def _prepare_backward_transition(
    state: RunState, record: RunRecord, *, schema: int = SCHEMA_VERSION,
) -> PreparedTransition:
    """Validate all backward data and prepare its detached state before append."""
    _require_initialized_terminal_backward(state)
    encoded = encode_record(RecordEnvelope(schema, state.run_id, state.next_sequence, record))
    envelope = decode_record(encoded)
    if type(envelope.record) is not type(record) or envelope.record != record:
        raise ValueError("backward record is not in canonical wire form")
    next_backward = reduce_backward_record(state.trace, state.backward_state, envelope.record)
    start_id, end_id = state.flow.start_id, state.flow.end_id
    plan = _BackwardPlan(
        state, state.flow, encoded, start_id, id(state.flow.nodes[start_id]),
        end_id, id(state.flow.nodes[end_id]) if end_id in state.flow.nodes else None,
        state.backward_state, next_backward,
        _prepare_record_trace(state, envelope),
    )
    return PreparedTransition(envelope.record, envelope.run_id, envelope.seq, plan, ())


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
    if not isinstance(prepared._plan, (
        _RunStartedPlan, _FrontierPlan, _SuccessPlan, _RoutePlan,
        _FailurePlan, _GrowthPlan, _BackwardPlan,
    )):
        raise TypeError("unsupported prepared transition plan")
    trace_delta = prepared._plan.trace_delta
    if isinstance(prepared._plan, _BackwardPlan):
        _require_postterminal_trace_delta(state, trace_delta)
    elif isinstance(prepared._plan, _FrontierPlan):
        _require_frontier_trace_delta(state, prepared)
    if trace_delta is not None:
        trace_delta.validate_origin(state.trace)
    if isinstance(prepared._plan, _RunStartedPlan):
        observers = _commit_run_started(log, state, prepared)
    elif isinstance(prepared._plan, _FrontierPlan):
        observers = _commit_frontier(log, state, prepared)
    elif isinstance(prepared._plan, _SuccessPlan):
        observers = _commit_success(log, state, prepared)
    elif isinstance(prepared._plan, _RoutePlan):
        observers = _commit_route(log, state, prepared)
    elif isinstance(prepared._plan, _FailurePlan):
        observers = _commit_failure(log, state, prepared)
    elif isinstance(prepared._plan, _GrowthPlan):
        observers = _commit_growth(log, state, prepared)
    elif isinstance(prepared._plan, _BackwardPlan):
        observers = _commit_backward(log, state, prepared)
    else:
        raise TypeError("unsupported prepared transition plan")
    if trace_delta is not None:
        state.trace = trace_delta.apply()
    return observers


def _prepare_record_trace(
    state: RunState, envelope: RecordEnvelope, *,
    input_connections: tuple[SeedConnection, ...] = (),
    child_boundaries: tuple[ChildFlowBoundary, ...] = (),
    retired_ids: tuple[str, ...] = (),
    terminal_succeeded: bool | None = None,
    terminal_output_bytes: bytes | None = None,
    terminal_failure: Failure | None = None,
) -> TraceDelta | None:
    """Prepare accepted trace data while the actual executing node still exists."""
    record = envelope.record
    node_id = _outcome_node_id(record)
    return prepare_trace(state.trace, envelope, state.flow.nodes.get(node_id),
                         input_connections=input_connections, child_boundaries=child_boundaries,
                         retired_ids=retired_ids,
                         root_end_id=state.flow.end_id if isinstance(record, RunStartedRecord) else None,
                         root_input_names=(
                             state.flow.nodes[state.flow.start_id].backward_input_names()
                             if isinstance(record, RunStartedRecord) and record.learnable else None
                         ),
                         terminal_succeeded=terminal_succeeded,
                         terminal_output_bytes=terminal_output_bytes,
                         terminal_failure=terminal_failure)


def _outcome_node_id(record: RunRecord) -> str | None:
    """Select the physical executing node from canonical outcome authority."""
    if isinstance(record, NodeFailedRecord):
        return record.failure.node_id
    if isinstance(record, NodeExpandedRecord):
        return record.spawner_id
    if isinstance(record, (NodeSucceededRecord, NodeRoutedRecord)):
        return record.node_id
    return None


def _require_outcome_trace_scope(state: RunState, prepared: PreparedTransition) -> None:
    """Match detached scope to the canonical physical node before append/prune.

    Callers have checked the record against its immutable prepared envelope. The
    adopted graph remains authority until apply; neither a commit target nor a
    substituted occurrence can select the historical owner.
    """
    delta = prepared._plan.trace_delta
    if state.trace is None:
        if delta is not None:
            raise ValueError("an unretained outcome cannot carry a trace delta")
        return
    if type(delta) is not TraceDelta or type(delta.occurrence) is not TraceOccurrence:
        raise ValueError("a retained outcome requires its physical trace occurrence")
    node_id = _outcome_node_id(prepared.record)
    node = state.flow.nodes.get(node_id)
    if not isinstance(node, Node):
        raise ValueError("outcome memory scope requires its physical executing node")
    if delta.occurrence.memory_scope != _copy_memory_scope(node.memory_scope):
        raise ValueError("outcome memory scope disagrees with the physical executing node")


def _require_outcome_trace_terminal(state: RunState, prepared: PreparedTransition) -> None:
    """Match physical occurrence, terminal facts, and source latches to authority.

    The terminal disposition was already derived during preparation. Validate
    its trace projection without repeating quiescence or invoking node hooks.
    Writer and route-latch updates also constrain later terminal projections.
    """
    trace = state.trace
    if trace is None:
        return
    plan, record = prepared._plan, prepared.record
    delta = plan.trace_delta
    if (type(delta.sequence) is not int or delta.sequence != prepared.expected_sequence
            or delta.initialized is not None or delta.started is not None or delta.resumed is not False):
        raise ValueError("outcome trace cannot change initialization or dispatch authority")
    node_id = _outcome_node_id(record)
    source = SourceRef(state.run_id, trace.started[node_id])
    occurrence = delta.occurrence
    _validate_source(occurrence.source)
    if (occurrence.source != source or type(occurrence.node_id) is not str or occurrence.node_id != node_id
            or type(occurrence.outcome_seq) is not int or occurrence.outcome_seq != prepared.expected_sequence
            or type(occurrence._record_bytes) is not bytes or occurrence._record_bytes != plan.envelope_bytes):
        raise ValueError("physical trace occurrence disagrees with the accepted outcome")
    root_start_source = source if node_id == trace.start_id and trace.root_start_source is None else None
    if delta.root_start_source is not None:
        _validate_source(delta.root_start_source)
    if delta.root_start_source != root_start_source:
        raise ValueError("terminal trace root START source disagrees with the accepted outcome")
    writer = None
    if isinstance(plan, _SuccessPlan):
        if (type(delta.writer) is not tuple or len(delta.writer) != 2
                or type(delta.writer[0]) is not str or delta.writer[0] != record.commit_target):
            raise ValueError("physical trace writer disagrees with the accepted output target")
        writer = SourceEntry(source, prepared.expected_sequence, plan.delta.value_bytes)
    elif delta.writer is not None:
        raise ValueError("only an accepted output can install a physical trace writer")
    root_writer = writer if writer is not None and record.commit_target == state.flow.end_id else None
    for actual, expected_writer, label in (
        (delta.writer[1] if delta.writer is not None else None, writer, "physical trace writer"),
        (delta.root_writer, root_writer, "terminal trace root writer"),
    ):
        if actual is not None:
            if (type(actual) is not SourceEntry or type(actual.outcome_seq) is not int
                    or type(actual._value_bytes) is not bytes):
                raise ValueError(f"{label} must be an exact accepted entry")
            if actual.source is not None:
                _validate_source(actual.source)
        if actual != expected_writer:
            raise ValueError(f"{label} disagrees with the accepted outcome")

    causal_source = (
        source if isinstance(record, NodeRoutedRecord) and record.terminal_failure is not None
        and trace._causal_route_source is None else None
    )
    if delta.causal_route_source is not None:
        _validate_source(delta.causal_route_source)
    if delta.causal_route_source != causal_source:
        raise ValueError("terminal trace causal source disagrees with the accepted outcome")

    expected = (
        _TerminalDelta(False, None, plan.failure) if isinstance(plan, _FailurePlan)
        else plan.terminal if isinstance(plan, (_SuccessPlan, _RoutePlan)) else None
    )
    terminal = delta.terminal
    if expected is None:
        if terminal is not None:
            raise ValueError("nonterminal outcome cannot install a terminal trace")
        return
    if type(terminal) is not TraceTerminal:
        raise ValueError("terminal outcome requires an exact terminal trace")
    TraceTerminal.__post_init__(terminal)
    if expected.succeeded:
        accepted_writer = root_writer or trace._root_writer
        if accepted_writer is None:
            raise ValueError("terminal trace requires its accepted root writer")
        terminal_source = accepted_writer.source
    else:
        terminal_source = (
            source if isinstance(plan, _FailurePlan)
            else causal_source or trace._causal_route_source
        )
        if terminal_source is None:
            raise ValueError("terminal trace failure requires its accepted causal occurrence")
    if (terminal.succeeded is not expected.succeeded or terminal.source != terminal_source
            or terminal.output_bytes != expected.output_bytes or terminal.failure != expected.failure):
        raise ValueError("terminal trace disagrees with the prepared terminal outcome")


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
            memory_snapshots=start.memory_snapshots,
            learnable=start.learnable,
            memories=start.memories,
            memory_stacks=start.memory_stacks,
        ),
        schema=first.schema,
    )
    commit_prepared(cursor, state, prepared)

    for encoded in canonical_bytes[1:]:
        envelope = decode_record(encoded)
        prepared = prepare_record(
            state, envelope.record, grow_builder=grow_builder, schema=envelope.schema,
        )
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
        if delta.node_id != prepared.record.node_id:
            raise ValueError("frontier start delta disagrees with its record")
        observers: tuple[Any, ...] = (NodeStarted(node_id=delta.node_id),)
    elif isinstance(prepared.record, RunResumedRecord) and isinstance(
        delta, _RunResumedDelta
    ):
        if (delta.retry_node_ids != prepared.record.retry_node_ids
                or type(delta.num_workers) is not int
                or delta.num_workers != prepared.record.num_workers):
            raise ValueError("frontier resume delta disagrees with its record")
        observers = (RunResumed(),)
    else:
        raise TypeError("frontier record and prevalidated delta disagree")

    _append_and_verify(log, prepared)
    if isinstance(delta, _NodeStartedDelta):
        state.ready.popleft()
        state.in_flight[delta.node_id] = None
        state.accepted_starts[delta.node_id] = prepared.expected_sequence
    else:
        state.ready.extendleft(reversed(delta.retry_node_ids))
        state.in_flight.clear()
        state.accepted_starts.clear()
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
    _require_outcome_trace_scope(state, prepared)
    _require_outcome_trace_terminal(state, prepared)
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
    del state.accepted_starts[delta.node_id]
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
    _require_outcome_trace_scope(state, prepared)
    _require_outcome_trace_terminal(state, prepared)
    delta = plan.delta
    if not (
        prepared.record.node_id == delta.node_id
        and prepared.record.handle == delta.handle
    ):
        raise ValueError("route record and prevalidated delta disagree")
    if prepared.record.terminal_failure is not None:
        if delta.terminal_unreachable_failure is not prepared.record.terminal_failure:
            raise ValueError("route record and terminal-failure delta disagree")
    elif delta.terminal_unreachable_failure is not state.terminal_unreachable_failure:
        raise ValueError("noncausal route changed terminal-failure authority")
    apply = _materialize_route(plan)

    _append_and_verify(log, prepared)
    state.node_state[delta.node_id] = NodeState.TAKEN
    for edge_id, disposition in delta.edge_updates:
        state.edge_state[edge_id] = disposition
    for node_id in delta.skipped_node_ids:
        state.node_state[node_id] = NodeState.SKIPPED
    del state.in_flight[delta.node_id]
    del state.accepted_starts[delta.node_id]
    state.ready.extend(delta.newly_ready_ids)
    state.terminal_unreachable_failure = delta.terminal_unreachable_failure
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
    _require_outcome_trace_scope(state, prepared)
    _require_outcome_trace_terminal(state, prepared)
    if prepared.record.failure is not plan.failure:
        raise ValueError("failure record and prevalidated authority disagree")
    node_id = plan.failure.node_id
    if type(node_id) is not str or not node_id:
        raise ValueError("failure plan has no executing node")
    apply = _materialize_failure(plan.failure)

    _append_and_verify(log, prepared)
    state.node_state[node_id] = NodeState.TAKEN
    del state.in_flight[node_id]
    del state.accepted_starts[node_id]
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
    _require_outcome_trace_scope(state, prepared)
    _require_outcome_trace_terminal(state, prepared)
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
    state.accepted_starts = apply.accepted_starts
    state.depth = apply.depth
    state.growth_owner = apply.growth_owner
    state.deferred_prune = apply.deferred_prune
    state.wiring_overrides = apply.wiring_overrides
    state.next_sequence = prepared.expected_sequence + 1
    return apply.observers


def _require_postterminal_trace_delta(
    state: RunState, delta: TraceDelta | None,
) -> None:
    """Require the exact sequence-only trace change before accepting a record."""
    if (type(delta) is not TraceDelta or delta.originating_trace is not state.trace
            or type(delta.sequence) is not int or delta.sequence != state.next_sequence
            or delta.resumed is not False):
        raise ValueError("postterminal trace delta must only advance the accepted sequence")
    if any(value is not None for value in (
        delta.initialized, delta.started, delta.occurrence, delta.handler, delta.writer,
        delta.consumed_connection, delta.boundary_update, delta.root_start_source,
        delta.root_writer, delta.causal_route_source, delta.terminal,
    )) or any(type(values) is not tuple or len(values) != 0 for values in (
        delta.prune_ids, delta.connections, delta.boundaries,
    )):
        raise ValueError("postterminal trace delta must only advance the accepted sequence")


def _require_frontier_trace_delta(state: RunState, prepared: PreparedTransition) -> None:
    """Permit only the canonical dispatch or resume operation before append."""
    delta, record = prepared._plan.trace_delta, prepared.record
    if state.trace is None:
        if delta is not None:
            raise ValueError("an unretained frontier cannot carry a frontier trace delta")
        return
    if (type(delta) is not TraceDelta or delta.originating_trace is not state.trace
            or type(delta.sequence) is not int or delta.sequence != state.next_sequence):
        raise ValueError("frontier trace delta requires its exact accepted sequence")
    if isinstance(record, NodeStartedRecord):
        if (type(delta.started) is not tuple or len(delta.started) != 2
                or type(delta.started[0]) is not str or delta.started[0] != record.node_id
                or type(delta.started[1]) is not int or delta.started[1] != prepared.expected_sequence
                or delta.resumed is not False):
            raise ValueError("frontier trace start disagrees with its canonical record")
    elif isinstance(record, RunResumedRecord):
        if delta.started is not None or delta.resumed is not True:
            raise ValueError("frontier trace resume disagrees with its canonical record")
    else:
        raise ValueError("frontier trace requires a dispatch or resume record")
    if any(value is not None for value in (
        delta.initialized, delta.occurrence, delta.handler, delta.writer,
        delta.consumed_connection, delta.boundary_update, delta.root_start_source,
        delta.root_writer, delta.causal_route_source, delta.terminal,
    )) or any(type(values) is not tuple or len(values) != 0 for values in (
        delta.prune_ids, delta.connections, delta.boundaries,
    )):
        raise ValueError("frontier trace cannot change outcome or terminal history")


def _commit_backward(log: RunLog, state: RunState, prepared: PreparedTransition) -> tuple[Any, ...]:
    """Verify the complete prepared authority, append, then replace attempt state."""
    plan = prepared._plan
    if type(plan) is not _BackwardPlan:
        raise TypeError("backward commit requires its exact plan")
    _require_prepared_origin(state, plan)
    if state.backward_state is not plan.originating_backward_state:
        raise ValueError("prepared backward transition has stale attempt authority")
    _require_initialized_terminal_backward(state)
    _require_authority_coordinates(log, state, prepared, initialized=True)
    _require_unchanged_record(prepared, plan.envelope_bytes)
    envelope = decode_record(plan.envelope_bytes)
    expected = reduce_backward_record(state.trace, state.backward_state, envelope.record)
    if type(plan.next_backward_state) is not RuntimeBackwardState:
        raise TypeError("backward plan requires an exact next state")
    # The reducer just validated ``expected``, and only ``expected`` is kept, so
    # equality is enough; checking all history here would cost O(N) per record.
    if expected != plan.next_backward_state:
        raise ValueError("backward plan state disagrees with its canonical record")
    _append_and_verify(log, prepared)
    state.backward_state = expected
    state.next_sequence = prepared.expected_sequence + 1
    return ()


def _require_prepared_origin(
    state: RunState,
    plan: (
        _RunStartedPlan
        | _FrontierPlan
        | _SuccessPlan
        | _RoutePlan
        | _FailurePlan
        | _GrowthPlan
        | _BackwardPlan
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
    frozen = decode_record(envelope_bytes)
    if (
        frozen.run_id != prepared.expected_run_id
        or frozen.seq != prepared.expected_sequence
    ):
        raise ValueError("prepared coordinates changed after validation")
    current_bytes = encode_record(
        RecordEnvelope(
            schema=frozen.schema,
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
        or encode_record(envelope) != prepared._plan.envelope_bytes
    ):
        raise ValueError("run-log append result does not match prepared transition")


def _require_initialized(state: RunState) -> None:
    """Check initialized reducer roots and recorded memory authority."""
    exact_containers = (
        type(state.pool) is VariablePool
        and type(state.node_state) is dict
        and type(state.edge_state) is dict
        and type(state.ready) is deque
        and type(state.in_flight) is dict
        and type(state.accepted_starts) is dict
        and type(state.depth) is dict
        and type(state.growth_owner) is dict
        and type(state.deferred_prune) is frozenset
        and type(state.wiring_overrides) is dict
        and type(state.input) is dict
        and type(state.effective_llm_by_static_path) is dict
        and type(state.backward_state) is RuntimeBackwardState
        and type(state._memory_authority) is _MemoryAuthority
        and type(state.initial_static_ids) is frozenset
    )
    if not exact_containers:
        raise ValueError("transition requires initialized reducer containers")
    if type(state.run_id) is not str or not state.run_id:
        raise ValueError("run state is not initialized")
    if type(state.flow_fingerprint) is not str or not state.flow_fingerprint:
        raise ValueError("run state is not initialized with a flow fingerprint")
    if type(state.next_sequence) is not int or state.next_sequence < 1:
        raise ValueError("run state is not initialized with a valid sequence")
    if (state.accepted_starts.keys() != state.in_flight.keys()
            or any(type(sequence) is not int or not 0 < sequence < state.next_sequence
                   for sequence in state.accepted_starts.values())
            or len(set(state.accepted_starts.values())) != len(state.accepted_starts)):
        raise ValueError("accepted starts must match the unfinished invocation inventory")
    if type(state.num_workers) is not int or state.num_workers < 0:
        raise ValueError("run state has an invalid worker count")
    if state.terminal_unreachable_failure is not None:
        failure = state.terminal_unreachable_failure
        if type(failure) is not Failure:
            raise TypeError("terminal-unreachable latch must be a Failure or None")
        if failure.code != "graph.terminal_unreachable":
            raise ValueError(
                "terminal-unreachable latch has the wrong failure code"
            )
        if type(failure.node_id) is not str or not failure.node_id:
            raise ValueError(
                "terminal-unreachable latch requires a causal node ID"
            )
    if state.flow.start_id not in state.flow.nodes:
        raise ValueError("initialized run state is missing its START boundary")
    if state.pool.start_id != state.flow.start_id:
        raise ValueError("initialized pool START must match the adopted flow")
    if state.flow.start_id not in state.initial_static_ids:
        raise ValueError("initialized run state START is not an initial static node")
    if state.learnable:
        if (type(state.trace) is not RetainedTrace or state.trace.run_id != state.run_id
                or state.trace.next_sequence != state.next_sequence):
            raise ValueError("run-state trace disagrees with recorded authority")
    elif state.trace is not None:
        raise ValueError("trace requires recorded learning")
    _validate_runtime_memory_configuration(state)


def _require_initialized_nonterminal(state: RunState) -> None:
    """Require initialized graph state before its primary terminal outcome."""
    _require_initialized(state)
    if state.result is not None:
        raise ValueError("terminal run state cannot prepare another graph transition")


def _require_initialized_terminal_backward(state: RunState) -> None:
    """Require the accepted retained terminal, independently from memory declarations."""
    _require_initialized(state)
    if not isinstance(state.result, (RunSucceeded, RunFailed)):
        raise ValueError("backward records require an accepted terminal result")
    if state.learnable is not True or state.trace is None or state.trace.terminal is None:
        raise ValueError("backward records require recorded learning and terminal trace authority")


def _validate_runtime_memory_configuration(state: RunState) -> None:
    """Require constant-time identity roots from canonical RunStarted decoding."""
    authority = state._memory_authority
    if type(authority) is not _MemoryAuthority:
        raise TypeError("run-state memory authority has the wrong type")
    if not (
        state.memory_snapshots is authority.snapshots
        and state.memories is authority.memories
        and state.memory_stacks is authority.memory_stacks
        and type(state.learnable) is bool
        and state.learnable is authority.learnable
        and state.backward_state.memory_order is authority.memory_order
    ):
        raise ValueError("run-state memory roots disagree with RunStarted authority")


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
    _validate_node_evidence(state, record.spawner_id, record.evidence)


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
        raise TypeError("grow_builder must return a Flow")
    fragment = copy.deepcopy(built)
    if not isinstance(fragment, Flow):  # defensive against a hostile deepcopy hook
        raise TypeError("grow_builder fragment deepcopy must remain a Flow")
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
        _raise_growth_rejected(
            ValueError(
                f"expansion exceeded node budget ({MAX_TOTAL_NODES}) at spawner "
                f"{record.spawner_id!r}"
            )
        )
    derived_depth = _validated_growth_depth(state, record.spawner_id)
    _validate_growth_prune(
        state,
        record.spawner_id,
        frozenset(record.prune_ids),
        frozenset(new_node_ids),
    )
    retired_ids, deferred_prune = _retired_ids(
        state, record.spawner_id, frozenset(record.prune_ids)
    )
    _validate_prune_dependencies(state, fragment, retired_ids)

    combined_nodes = dict(state.flow.nodes)
    combined_nodes.update(fragment.nodes)
    combined_edges = [*state.flow.edges, *fragment.edges]
    combined_wiring = copy.deepcopy(state.flow.wiring)
    combined_wiring.update(copy.deepcopy(fragment.wiring))

    final_nodes = {
        node_id: node
        for node_id, node in combined_nodes.items()
        if node_id not in retired_ids
    }
    final_edges = [
        edge
        for edge in combined_edges
        if edge.from_ not in retired_ids and edge.to not in retired_ids
    ]
    final_wiring = {
        node_id: wiring
        for node_id, wiring in combined_wiring.items()
        if node_id not in retired_ids
    }
    incoming, outgoing = _growth_adjacency(final_nodes, final_edges)

    node_state = dict(state.node_state)
    node_state.update((node_id, NodeState.UNKNOWN) for node_id in new_node_ids)
    node_state[record.spawner_id] = NodeState.EXPANDED
    for node_id in retired_ids:
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
    for node_id in retired_ids:
        pool_store.pop(node_id, None)
        depth.pop(node_id, None)
        wiring_overrides.pop(node_id, None)
    if derived_depth is not None:
        for node_id, node in fragment.nodes.items():
            if node.is_spawner is True or node.commit_as == record.spawner_id:
                depth[node_id] = derived_depth
    growth_owner = dict(state.growth_owner)
    growth_owner.update((node_id, record.spawner_id) for node_id in new_node_ids)
    for node_id, owner in growth_owner.items():
        # A self-pruning spawner's new nodes, and a kept subtree under a retired node,
        # move to the nearest owner that survives this transition.
        while owner in retired_ids:
            owner = state.growth_owner.get(owner)
        growth_owner[node_id] = owner
    growth_owner = {
        node_id: owner
        for node_id, owner in growth_owner.items()
        if node_id not in retired_ids and owner is not None
    }

    ready = tuple(
        node_id for node_id in state.ready if node_id not in retired_ids
    ) + (fragment.start_id,)
    in_flight = tuple(
        node_id
        for node_id in state.in_flight
        if node_id != record.spawner_id and node_id not in retired_ids
    )
    _validate_final_redirects(
        nodes=final_nodes,
        node_state=node_state,
        edge_state=edge_state,
        pool_store=pool_store,
        ready=ready,
        in_flight=in_flight,
        incoming=incoming,
        outgoing=outgoing,
    )

    delta = _GrowthDelta(
        spawner_id=record.spawner_id,
        prune_ids=record.prune_ids,
        retired_ids=tuple(sorted(retired_ids)),
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
        accepted_starts=tuple((node_id, state.accepted_starts[node_id]) for node_id in in_flight),
        depth=tuple(depth.items()),
        growth_owner=tuple(growth_owner.items()),
        deferred_prune=deferred_prune,
        wiring_overrides=tuple(wiring_overrides.items()),
    )
    return delta, topology


def _validate_growth_fragment(
    state: RunState,
    fragment: Flow,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate live topology first, then the node-produced fragment constraints."""
    _reject_growth_cycle(
        set(state.flow.nodes),
        state.flow.edges,
        as_growth_rejection=False,
    )
    static_producer_contracts: dict[str, OutputContract] = {}
    for node_id, node in state.flow.nodes.items():
        contract = node.output_contract
        if contract is not None:
            output_contract_type(contract)
            static_producer_contracts[node_id] = contract
    return _validate_growth_fragment_data(
        state,
        fragment,
        static_producer_contracts=static_producer_contracts,
    )


def _validate_growth_fragment_data(
    state: RunState,
    fragment: Flow,
    *,
    static_producer_contracts: dict[str, OutputContract] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Require a collision-free, self-contained, acyclic fragment.

    ``static_producer_contracts`` is the already validated live-topology map.
    When omitted, it is derived after the fragment's basic carrier checks; live
    contract invariant errors still remain outside ``GrowthRejected``.
    """
    if type(fragment.nodes) is not dict:
        _raise_growth_rejected(TypeError("growth fragment nodes must be a dict"))
    if type(fragment.edges) is not list:
        _raise_growth_rejected(TypeError("growth fragment edges must be a list"))
    if type(fragment.wiring) is not dict:
        _raise_growth_rejected(TypeError("growth fragment wiring must be a dict"))
    if type(fragment.start_id) is not str or not fragment.start_id:
        _raise_growth_rejected(
            ValueError("growth fragment start_id must be a non-empty string")
        )
    if type(fragment.end_id) is not str or not fragment.end_id:
        _raise_growth_rejected(
            ValueError("growth fragment end_id must be a non-empty string")
        )
    if fragment.start_id not in fragment.nodes:
        _raise_growth_rejected(
            ValueError("growth fragment start_id is missing from its nodes")
        )
    if fragment.end_id not in fragment.nodes:
        _raise_growth_rejected(
            ValueError("growth fragment end_id is missing from its nodes")
        )

    new_node_ids: list[str] = []
    for node_id, node in fragment.nodes.items():
        if type(node_id) is not str:
            _raise_growth_rejected(
                TypeError("growth fragment node IDs must be strings")
            )
        if not node_id:
            _raise_growth_rejected(
                ValueError("growth fragment node IDs must be non-empty")
            )
        if not isinstance(node, Node):
            _raise_growth_rejected(
                TypeError(f"growth fragment node {node_id!r} must be a Node")
            )
        if node.id != node_id:
            _raise_growth_rejected(
                ValueError(
                    f"growth fragment node key {node_id!r} does not equal node.id "
                    f"{node.id!r}"
                )
            )
        if node_id in state.flow.nodes:
            _raise_growth_rejected(
                ValueError(
                    f"growth fragment node {node_id!r} collides with live topology"
                )
            )
        new_node_ids.append(node_id)

    if set(fragment.wiring) != set(fragment.nodes):
        _raise_growth_rejected(
            ValueError(
                "growth fragment wiring keys must exactly equal fragment nodes"
            )
        )
    try:
        validate_growth_metadata(fragment)
    except (TypeError, ValueError) as exc:
        _raise_growth_rejected(exc)
    combined_nodes = dict(state.flow.nodes)
    combined_nodes.update(fragment.nodes)
    combined_node_ids = set(combined_nodes)
    if static_producer_contracts is None:
        static_producer_contracts = {}
        for node_id, node in state.flow.nodes.items():
            contract = node.output_contract
            if contract is not None:
                output_contract_type(contract)
                static_producer_contracts[node_id] = contract
    fragment_producer_contracts: dict[str, OutputContract] = {}
    for node_id, node in fragment.nodes.items():
        contract = node.output_contract
        if contract is None:
            continue
        try:
            output_contract_type(contract)
        except (TypeError, ValueError) as exc:
            _raise_growth_rejected(exc)
        fragment_producer_contracts[node_id] = contract
    producer_contracts = dict(static_producer_contracts)
    producer_contracts.update(fragment_producer_contracts)
    for node_id, node in fragment.nodes.items():
        for attribute in ("commit_as", "origin_id"):
            target = getattr(node, attribute)
            if target is not None and type(target) is not str:
                _raise_growth_rejected(
                    TypeError(
                        f"growth fragment node {node_id!r} {attribute} must be a "
                        "string or None"
                    )
                )
            if target == "":
                _raise_growth_rejected(
                    ValueError(
                        f"growth fragment node {node_id!r} {attribute} must be non-empty"
                    )
                )
            if target is not None and target not in combined_node_ids:
                _raise_growth_rejected(
                    ValueError(
                        f"growth fragment node {node_id!r} {attribute} target "
                        f"{target!r} does not exist"
                    )
                )
        skipped_target = node.skipped_output_target()
        try:
            _validate_skipped_output_target(
                combined_nodes,
                node_id,
                skipped_target,
            )
        except (TypeError, ValueError) as exc:
            _raise_growth_rejected(exc)
        wiring = fragment.wiring[node_id]
        if type(wiring) is not dict or any(type(key) is not str for key in wiring):
            _raise_growth_rejected(
                TypeError(
                    f"growth fragment wiring for {node_id!r} must be a string-keyed dict"
                )
            )
        params = node.params if node.params is not None else []
        if type(params) is not list:
            _raise_growth_rejected(
                TypeError(
                    f"growth fragment node {node_id!r} params must be a list"
                )
            )
        declared: dict[str, bool] = {}
        for param in params:
            if type(param) is not Parameter:
                _raise_growth_rejected(
                    TypeError(
                        f"growth fragment node {node_id!r} params must contain "
                        f"Parameter values, got {type(param).__name__}"
                    )
                )
            if type(param.name) is not str or not param.name:
                _raise_growth_rejected(
                    ValueError(
                        f"growth fragment node {node_id!r} has an invalid param name"
                    )
                )
            if param.name in declared:
                _raise_growth_rejected(
                    ValueError(
                        f"growth fragment node {node_id!r} has duplicate param "
                        f"{param.name!r}"
                    )
                )
            declared[param.name] = bool(param.required)
        reserved = node.reserved_wiring_keys()
        if type(reserved) is not set or any(
            type(key) is not str or not key for key in reserved
        ):
            _raise_growth_rejected(
                TypeError(
                    f"growth fragment node {node_id!r} reserved wiring keys are invalid"
                )
            )
        actual = set(wiring)
        allowed = set(declared) | reserved
        extras = actual - allowed
        missing_reserved = reserved - actual
        missing_required = {
            name for name, required in declared.items() if required and name not in actual
        }
        if extras or missing_reserved or missing_required:
            _raise_growth_rejected(
                ValueError(
                    f"growth fragment node {node_id!r} wiring parity failed: "
                    f"extra={sorted(extras)!r}, "
                    f"missing_reserved={sorted(missing_reserved)!r}, "
                    f"missing_required={sorted(missing_required)!r}"
                )
            )
        _validate_growth_wiring_refs(
            node_id,
            node,
            wiring,
            combined_node_ids,
            producer_contracts,
            reserved,
        )

    live_edge_ids = {edge.id for edge in state.flow.edges}
    new_edge_ids: list[str] = []
    seen_edge_ids: set[str] = set()
    for edge in fragment.edges:
        if type(edge) is not Edge:
            _raise_growth_rejected(
                TypeError("growth fragment edges must contain only Edge values")
            )
        if type(edge.id) is not str or not edge.id:
            _raise_growth_rejected(
                ValueError("growth fragment edge IDs must be non-empty strings")
            )
        if edge.id in seen_edge_ids:
            _raise_growth_rejected(
                ValueError(f"growth fragment edge ID {edge.id!r} is duplicated")
            )
        if edge.id in live_edge_ids:
            _raise_growth_rejected(
                ValueError(
                    f"growth fragment edge {edge.id!r} collides with live topology"
                )
            )
        for attribute in ("source_handle", "input_group"):
            value = getattr(edge, attribute)
            if value is not None and type(value) is not str:
                _raise_growth_rejected(
                    TypeError(
                        f"growth fragment edge {edge.id!r} {attribute} must be a "
                        "string or None"
                    )
                )
            if value == "":
                _raise_growth_rejected(
                    ValueError(
                        f"growth fragment edge {edge.id!r} {attribute} must be non-empty"
                    )
                )
        for attribute in ("optional", "ordering"):
            if type(getattr(edge, attribute)) is not bool:
                _raise_growth_rejected(
                    TypeError(
                        f"growth fragment edge {edge.id!r} {attribute} must be a bool"
                    )
                )
        if type(edge.from_) is not str or edge.from_ not in combined_node_ids:
            _raise_growth_rejected(
                ValueError(
                    f"growth fragment edge {edge.id!r} has unknown source {edge.from_!r}"
                )
            )
        if type(edge.to) is not str or edge.to not in fragment.nodes:
            _raise_growth_rejected(
                ValueError(
                    f"growth fragment edge {edge.id!r} must target a fragment node, "
                    f"found {edge.to!r}"
                )
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
        _raise_growth_rejected(
            ValueError(
                "growth fragment nodes are unreachable from its start: "
                f"{unreachable!r}"
            )
        )


def _validate_growth_wiring_refs(
    node_id: str,
    node: Node,
    wiring: dict[str, Any],
    combined_node_ids: set[str],
    producer_contracts: dict[str, OutputContract],
    reserved: set[str],
) -> None:
    """Apply closed reference rules to raw or carrier-wrapped fragment wiring."""
    for param, source in wiring.items():
        source = binding_source(source)
        if not isinstance(source, str):
            continue
        try:
            paths = expr_refs_of(source)
        except ExpressionError as exc:
            error = ValueError(
                f"growth fragment node {node_id!r} wiring {param!r} is invalid: {exc}"
            )
            raise GrowthRejected(error) from exc
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
            )
            parts = path.split(".")
            if (
                error is None
                and len(parts) >= 2
                and parts[1] == "output"
                and parts[0] in producer_contracts
            ):
                # Fragments come from flows the loader already checked, where a CASE
                # branch may narrow a union so that a member-only field read is valid.
                # That narrowing is not recorded here, so such reads are allowed.
                error = output_contract_field_error(
                    producer_contracts[parts[0]],
                    parts[2:],
                    path,
                    partial_union_fields=True,
                )
            if error is not None:
                _raise_growth_rejected(
                    ValueError(
                        f"growth fragment node {node_id!r} wiring {param!r}: "
                        f"{error}"
                    )
                )


def _reject_growth_cycle(
    node_ids: set[str],
    edges: list[Edge],
    *,
    as_growth_rejection: bool = True,
) -> None:
    """Reject a cycle as fragment data or as a pre-existing host invariant."""
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
        message = (
            f"growth fragment creates a cycle involving {stuck!r}"
            if as_growth_rejection
            else f"live topology contains a cycle involving {stuck!r}"
        )
        error = ValueError(message)
        if as_growth_rejection:
            _raise_growth_rejected(error)
        raise error


def _validated_growth_depth(state: RunState, spawner_id: str) -> int | None:
    """Derive the exact trait-driven growth depth for one expansion."""
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
        _raise_growth_rejected(
            ValueError(
                f"expansion exceeded MAX_REF_DEPTH ({MAX_REF_DEPTH}) at "
                f"{spawner_id!r}"
            )
        )
    return depth


def _retired_ids(
    state: RunState,
    spawner_id: str,
    prune_ids: frozenset[str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Split the requested and deferred prune ids into those retired now and those kept.

    Each candidate is a requested prune id or one an earlier growth deferred.  It
    is retired with its whole grown subtree when every node in it is settled
    (taken, skipped, or expanded; none ready or running).  Otherwise it stays live
    with that subtree and is deferred, so a later growth retires it once settled;
    a kept node also keeps every candidate it reads.  A spawner that asks to prune
    itself is always retired; its grown nodes go with it only if all are settled,
    else they move to the nearest surviving owner.

    Args:
        state (`RunState`):
            Pre-transition state whose ``growth_owner`` names each node's owner.
        spawner_id (`str`):
            Spawner whose growth record asks for ``prune_ids``.
        prune_ids (`frozenset[str]`):
            Node IDs the growth record asks to prune.

    Returns:
        `tuple[frozenset[str], frozenset[str]]`:
            The node IDs retired by this growth, and the candidates kept live and
            deferred to a later growth.
    """
    owned: dict[str, list[str]] = {}
    for node_id, owner in state.growth_owner.items():
        owned.setdefault(owner, []).append(node_id)
    settled = (NodeState.TAKEN, NodeState.SKIPPED, NodeState.EXPANDED)
    ready_ids = set(state.ready)

    def is_settled(node_id: str) -> bool:
        return (
            state.node_state.get(node_id) in settled
            and node_id not in state.in_flight
            and node_id not in ready_ids
        )

    units: dict[str, list[str]] = {}
    for candidate in prune_ids | state.deferred_prune:
        unit = [candidate]
        pending = [candidate]
        while pending:
            for node_id in owned.get(pending.pop(), ()):
                unit.append(node_id)
                pending.append(node_id)
        units[candidate] = unit

    retired: set[str] = set()
    # A deferred spawner growing now is unsettled and kept like any other candidate.
    spawner_unit = units.pop(spawner_id) if spawner_id in prune_ids else None
    if spawner_unit is not None:
        retired.add(spawner_id)
        if all(is_settled(node_id) for node_id in spawner_unit[1:]):
            retired.update(spawner_unit[1:])
    kept = {
        candidate for candidate, unit in units.items()
        if not all(is_settled(node_id) for node_id in unit)
    }
    # A settled node that kept work still reads must stay with it.  Units overlap
    # when a deferred id lies under another candidate, so a node can be in several.
    units_of: dict[str, list[str]] = {}
    for candidate, unit in units.items():
        for node_id in unit:
            units_of.setdefault(node_id, []).append(candidate)
    pending = [node_id for candidate in kept for node_id in units[candidate]]
    while pending:
        for read in _read_ids(state, pending.pop()):
            for candidate in units_of.get(read, ()):
                if candidate not in kept:
                    kept.add(candidate)
                    pending.extend(units[candidate])
    for candidate, unit in units.items():
        if candidate not in kept:
            retired.update(unit)
    return frozenset(retired), frozenset(kept - retired)


def _read_ids(state: RunState, node_id: str) -> set[str]:
    """Return the node IDs a live node reads through edges, wiring, or output targets."""
    node = state.flow.nodes[node_id]
    reads = {edge.from_ for edge in state.flow.incoming(node_id)}
    reads.update(
        target
        for target in (node.commit_as, node.origin_id, node.skipped_output_target())
        if type(target) is str
    )
    for source in state.flow.wiring.get(node_id, {}).values():
        source = binding_source(source)
        if not isinstance(source, str):
            continue
        try:
            paths = expr_refs_of(source)
        except ExpressionError:
            continue  # the prune dependency check reports invalid wiring
        reads.update(path.split(".", 1)[0] for path in paths)
    return reads


def _validate_growth_prune(
    state: RunState,
    spawner_id: str,
    prune_ids: frozenset[str],
    new_node_ids: frozenset[str],
) -> None:
    """Validate the IDs a growth record asks to prune."""
    _validate_growth_prune_data(
        state,
        spawner_id,
        prune_ids,
        new_node_ids,
    )


def _validate_growth_prune_data(
    state: RunState,
    spawner_id: str,
    prune_ids: frozenset[str],
    new_node_ids: frozenset[str],
) -> None:
    """Reject requested prune IDs that are not dynamic overlay or a self-prunable LOOP clone.

    Unsettled requested IDs are not errors: ``_retired_ids`` keeps them live.
    """
    spawner = state.flow.nodes[spawner_id]
    for node_id in prune_ids:
        if node_id not in state.flow.nodes:
            _raise_growth_rejected(
                ValueError(f"growth prune node {node_id!r} does not exist")
            )
        if node_id in (state.flow.start_id, state.flow.end_id):
            _raise_growth_rejected(
                ValueError(f"growth cannot prune top-level boundary {node_id!r}")
            )
        if node_id in new_node_ids:
            _raise_growth_rejected(
                ValueError(f"growth cannot prune new fragment node {node_id!r}")
            )
        if node_id in state.initial_static_ids:
            _raise_growth_rejected(
                ValueError(f"growth cannot prune initial static node {node_id!r}")
            )
        if node_id == spawner_id:
            origin_id = spawner.origin_id
            if not (
                spawner.is_loop is True
                and type(origin_id) is str
                and origin_id
                and origin_id != spawner_id
                and origin_id in state.flow.nodes
                and origin_id not in prune_ids
            ):
                _raise_growth_rejected(
                    ValueError(
                        f"growth spawner {spawner_id!r} is not a "
                        "self-prunable LOOP clone"
                    )
                )


def _validate_prune_dependencies(
    state: RunState,
    fragment: Flow,
    prune_ids: frozenset[str],
) -> None:
    """Validate explicit node-requested prune dependency constraints."""
    _validate_prune_dependencies_data(state, fragment, prune_ids)


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
            _raise_growth_rejected(
                ValueError(
                    f"surviving node {edge.to!r} depends on pruned edge source "
                    f"{edge.from_!r}"
                )
            )
    for node_id in dependent_ids:
        node = combined_nodes[node_id]
        if node.commit_as in prune_ids:
            _raise_growth_rejected(
                ValueError(
                    f"surviving node {node_id!r} commits to pruned node "
                    f"{node.commit_as!r}"
                )
            )
        if node.origin_id in prune_ids:
            _raise_growth_rejected(
                ValueError(
                    f"surviving node {node_id!r} has pruned origin "
                    f"{node.origin_id!r}"
                )
            )
        skipped_target = node.skipped_output_target()
        try:
            _validate_skipped_output_target(
                combined_nodes,
                node_id,
                skipped_target,
            )
        except (TypeError, ValueError) as exc:
            _raise_growth_rejected(exc)
        if skipped_target in prune_ids:
            _raise_growth_rejected(
                ValueError(
                    f"surviving node {node_id!r} has pruned skipped output target "
                    f"{skipped_target!r}"
                )
            )
        for source in combined_wiring.get(node_id, {}).values():
            source = binding_source(source)
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
                    _raise_growth_rejected(
                        ValueError(
                            f"surviving node {node_id!r} wiring depends on "
                            "pruned node"
                        )
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


def _recorded_stack(stacks: tuple[MemoryStack, ...], locator: tuple[str, ...]) -> MemoryStack | None:
    """Return the recorded stack at ``locator`` by binary search, or ``None``."""
    index = bisect_left(stacks, locator, key=lambda stack: stack.locator)
    if index < len(stacks) and stacks[index].locator == locator:
        return stacks[index]
    return None


def _validate_node_evidence(
    state: RunState,
    node_id: str,
    evidence: EvaluationEvidence | None,
) -> None:
    """Enforce recorded learning authority for one accepted graph outcome."""
    if node_id not in state.flow.nodes:
        raise ValueError(f"evidence node {node_id!r} does not exist in the flow")
    node = state.flow.nodes[node_id]
    scope = node.memory_scope

    if state.learnable is not True or node.needs_evaluation_trace is not True:
        if evidence is not None:
            raise ValueError(f"node {node_id!r} is ineligible for evaluation evidence")
        return
    if scope is not None:
        if type(scope) is not MemoryScope:
            raise ValueError(f"node {node_id!r} has no exact compiled memory scope")
        MemoryScope(scope.static_path, scope.memories, scope.occurrence_path)
        if state.memory_stacks is None:
            expected: tuple[str, ...] = ()
        else:
            recorded = _recorded_stack(state.memory_stacks, scope.static_path)
            if recorded is None:
                raise ValueError(f"node {node_id!r} has no recorded memory stack")
            expected = recorded.memories
        if scope.memories != expected:
            raise ValueError(f"node {node_id!r} memory scope must carry its exact recorded memory stack")
    elif state.memories:
        raise ValueError(f"node {node_id!r} has no exact compiled memory scope")
    if evidence is None:
        raise ValueError(f"eligible node {node_id!r} requires evaluation evidence while learning")
    if type(evidence) is not EvaluationEvidence:
        raise TypeError("node evidence must be an EvaluationEvidence or None")
    if evidence.scope != scope:
        raise ValueError(f"node {node_id!r} evidence scope does not match its compiled scope")


def _validate_failure_record(state: RunState, record: NodeFailedRecord) -> None:
    """Validate exact diagnostic carriers and the executing node for replay."""
    if type(record.failure) is not Failure:
        raise TypeError("NodeFailedRecord failure must be a Failure")
    if record.failure.code == "graph.terminal_unreachable":
        raise ValueError(
            "graph.terminal_unreachable must be derived from a route record"
        )
    node_id = record.failure.node_id
    if type(node_id) is not str:
        raise TypeError("NodeFailedRecord failure node_id must be a string")
    if not node_id:
        raise ValueError("NodeFailedRecord failure node_id must be non-empty")
    _require_failing_node(state, node_id)
    _validate_node_evidence(state, node_id, record.evidence)


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
    if (
        record.terminal_failure is not None
        and type(record.terminal_failure) is not Failure
    ):
        raise TypeError(
            "NodeRoutedRecord terminal failure must be a Failure or None"
        )
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
    target_node = state.flow.nodes[record.commit_target]
    contract = target_node.output_contract
    wire, decoded = _canonical_output_value(
        typed,
        contract,
        _node_output_type(target_node),
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
    *,
    derive_terminal_failure: bool,
) -> tuple[_RouteDelta, dict[str, NodeState], Failure | None]:
    """Validate one route and return its redirect-aware frozen skip closure."""
    _validate_route_record(state, record)
    edge_updates: dict[str, NodeState] = {}
    skipped_node_ids: list[str] = []
    skipped_nodes: set[str] = set()
    propagated_targets: set[str] = {record.node_id}
    redirect_owners: dict[str, tuple[str, ...]] | None = None
    newly_ready_ids: list[str] = []
    newly_ready: set[str] = set()
    joins: dict[str, _JoinScratch] = {}

    root_edges = _route_outgoing(state, record.node_id, edge_updates)
    # Only branch edges carry a handle. A handle-less edge (a `depends_on`/`runs_after`
    # ordering edge) is not a branch: the router settled by routing, so it is TAKEN on
    # every route rather than read as the `default` branch.
    matched = tuple(
        edge for edge in root_edges if edge.source_handle == record.handle
    )
    # The reserved default route may deliberately have no authored edge: a CASE
    # without ``else`` selects no branch and skip-floods every branch edge. Any
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
                    if edge.source_handle in (None, record.handle)
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
        target = _skipped_output_target(state, head)
        if target != head:
            if redirect_owners is None:
                redirect_owners = _skipped_output_owners(state)
            _require_unambiguous_redirect(
                owner_id=head,
                target=target,
                owners=redirect_owners.get(target, ()),
                has_committed_value=target in state.pool.store,
                disposition=state.node_state.get(target),
                is_executable=target in state.ready or target in state.in_flight,
                has_incoming=bool(state.flow.incoming(target)),
            )
        if target in propagated_targets:
            continue
        propagated_targets.add(target)
        children = _route_outgoing(state, target, edge_updates)
        for child in reversed(children):
            work.append((child, NodeState.SKIPPED))

    causal_failure: Failure | None = None
    terminal_unreachable_failure = state.terminal_unreachable_failure
    first_terminal_skip = (
        terminal_unreachable_failure is None
        and state.flow.end_id in skipped_nodes
    )
    if first_terminal_skip:
        if derive_terminal_failure:
            if record.terminal_failure is not None:
                raise ValueError(
                    "live route preparation cannot supply a terminal failure"
                )
            causal_failure = _make_terminal_unreachable_failure(
                state,
                causal_node_id=record.node_id,
                edge_updates=edge_updates,
            )
        else:
            causal_failure = _validate_recorded_terminal_failure(
                state,
                record,
                edge_updates=edge_updates,
            )
        terminal_unreachable_failure = causal_failure
    elif record.terminal_failure is not None:
        raise ValueError(
            "NodeRoutedRecord terminal failure is present on a noncausal route"
        )

    return (
        _RouteDelta(
            node_id=record.node_id,
            handle=record.handle,
            edge_updates=tuple(edge_updates.items()),
            skipped_node_ids=tuple(skipped_node_ids),
            newly_ready_ids=tuple(newly_ready_ids),
            terminal_unreachable_failure=terminal_unreachable_failure,
        ),
        edge_updates,
        causal_failure,
    )


def _skipped_output_target(state: RunState, node_id: str) -> str:
    """Return one skipped node's validated logical output target.

    The hook is node-owned so CALL/MAP's baked completion redirect and LOOP's
    origin redirect remain outside the kind-blind reducer.  Its result is
    topology authority, not node-produced data; malformed setup therefore
    escapes as a raw host/graph error.
    """
    target = state.flow.nodes[node_id].skipped_output_target()
    return _validate_skipped_output_target(state.flow.nodes, node_id, target)


def _validate_skipped_output_target(
    nodes: dict[str, Node],
    node_id: str,
    target: Any,
) -> str:
    """Validate one hook result against an exact topology node mapping."""
    if type(target) is not str:
        raise TypeError(
            f"node {node_id!r} skipped output target must be a string"
        )
    if not target:
        raise ValueError(
            f"node {node_id!r} skipped output target must be non-empty"
        )
    if target not in nodes:
        raise ValueError(
            f"node {node_id!r} skipped output target {target!r} does not exist"
        )
    return target


def _validate_final_redirects(
    *,
    nodes: dict[str, Node],
    node_state: dict[str, NodeState],
    edge_state: dict[str, NodeState],
    pool_store: dict[str, TypedValue],
    ready: tuple[str, ...],
    in_flight: tuple[str, ...],
    incoming: dict[str, list[Edge]],
    outgoing: dict[str, list[Edge]],
) -> None:
    """Require every final growth redirect to have one inert logical anchor."""
    mutable: dict[str, list[str]] = {}
    for node_id, node in nodes.items():
        if node_state.get(node_id) is not NodeState.UNKNOWN:
            continue
        target = node.skipped_output_target()
        try:
            target = _validate_skipped_output_target(nodes, node_id, target)
        except (TypeError, ValueError) as exc:
            _raise_growth_rejected(exc)
        if target == node_id:
            continue
        mutable.setdefault(target, []).append(node_id)
    for target, owner_ids in mutable.items():
        owner_id = owner_ids[0]
        try:
            _require_unambiguous_redirect(
                owner_id=owner_id,
                target=target,
                owners=tuple(owner_ids),
                has_committed_value=target in pool_store,
                disposition=node_state.get(target),
                is_executable=target in ready or target in in_flight,
                has_incoming=bool(incoming.get(target, ())),
            )
        except ValueError as exc:
            _raise_growth_rejected(exc)
        for edge in outgoing.get(target, ()):
            if edge_state.get(edge.id) is not NodeState.UNKNOWN:
                _raise_growth_rejected(
                    ValueError(
                        f"logical output target {target!r} has a settled outgoing "
                        f"edge {edge.id!r}"
                    )
                )


def _skipped_output_owners(state: RunState) -> dict[str, tuple[str, ...]]:
    """Index unfinished logical-output redirects in stable node order.

    Settled dynamic residue cannot produce another value and is ignored.
    Ordinary nodes target themselves and are not redirect owners.  The scan is
    paid only when a route actually skips a redirected completion; it keeps the
    common sparse skip path unchanged while rejecting ambiguous multi-writer
    topology before any edge is settled.
    """
    mutable: dict[str, list[str]] = {}
    for node_id in state.flow.nodes:
        if state.node_state.get(node_id) is not NodeState.UNKNOWN:
            continue
        target = _skipped_output_target(state, node_id)
        if target == node_id:
            continue
        mutable.setdefault(target, []).append(node_id)
    return {target: tuple(owner_ids) for target, owner_ids in mutable.items()}


def _require_unambiguous_redirect(
    *,
    owner_id: str,
    target: str,
    owners: tuple[str, ...],
    has_committed_value: bool,
    disposition: Any,
    is_executable: bool,
    has_incoming: bool,
) -> None:
    """Require one inert logical anchor to have one unfinished redirect owner.

    Skip means the logical output is absent.  That conclusion is sound only
    when no sibling physical completion or executable anchor can still produce
    the same value.  Supporting several producers would require a distinct OR
    join and multi-writer contract; ``commit_as`` intentionally remains a
    single-owner redirect instead.
    """
    if owners != (owner_id,):
        raise ValueError(
            f"logical output target {target!r} has multiple skipped output "
            f"owners: {owners!r}"
        )
    if has_committed_value:
        raise ValueError(
            f"logical output target {target!r} already has a committed value"
        )
    if type(disposition) is not NodeState:
        raise ValueError(
            f"logical output target {target!r} has an invalid disposition"
        )
    if is_executable:
        raise ValueError(
            f"logical output target {target!r} is also executable"
        )
    if disposition is NodeState.UNKNOWN:
        if has_incoming:
            raise ValueError(
                f"logical output target {target!r} is also executable"
            )
        return
    if disposition is not NodeState.EXPANDED:
        raise ValueError(
            f"logical output target {target!r} must be an expanded spawner "
            "or inert placeholder"
        )


def _make_terminal_unreachable_failure(
    state: RunState,
    *,
    causal_node_id: str,
    edge_updates: dict[str, NodeState],
) -> Failure:
    """Build the source-neutral failure for one causal terminal skip."""
    node = state.flow.nodes[causal_node_id]
    output_name = _terminal_skipped_output_name(state, edge_updates)
    return Failure(
        code="graph.terminal_unreachable",
        message=f"terminal output {output_name!r} skipped",
        node_id=causal_node_id,
        trace=node.failure_trace(),
    )


def _validate_recorded_terminal_failure(
    state: RunState,
    record: NodeRoutedRecord,
    *,
    edge_updates: dict[str, NodeState],
) -> Failure:
    """Validate the persisted authority for one structurally causal route."""
    failure = record.terminal_failure
    if failure is None:
        raise ValueError(
            "causal NodeRoutedRecord requires a persisted terminal failure"
        )
    if type(failure) is not Failure:
        raise TypeError("NodeRoutedRecord terminal failure must be a Failure")
    expected_message = (
        f"terminal output {_terminal_skipped_output_name(state, edge_updates)!r} "
        "skipped"
    )
    if failure.code != "graph.terminal_unreachable":
        raise ValueError("route terminal failure has the wrong failure code")
    if failure.node_id != record.node_id:
        raise ValueError("route terminal failure has the wrong causal node ID")
    if failure.message != expected_message:
        raise ValueError("route terminal failure has the wrong message")
    if failure.cause_type is not None or failure.traceback is not None:
        raise ValueError("route terminal failure cannot carry exception diagnostics")
    return failure


def _prepare_terminal_delta(
    state: RunState,
    *,
    completing_node_id: str,
    newly_ready_ids: tuple[str, ...],
    pending_commit_target: str | None,
    pending_value_bytes: bytes | None,
    edge_updates: dict[str, NodeState],
    terminal_unreachable_failure: Failure | None,
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
            failure=None,
        )
    if end_id in state.pool.store:
        output_bytes = ANY_VALUE_ADAPTER.dump_json(
            state.pool.store[end_id],
            warnings="error",
        )
        return _TerminalDelta(
            succeeded=True,
            output_bytes=output_bytes,
            failure=None,
        )
    if terminal_unreachable_failure is None:
        raise ValueError(
            "run quiesced without END output or a causal terminal failure"
        )
    return _TerminalDelta(
        succeeded=False,
        output_bytes=None,
        failure=terminal_unreachable_failure,
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
    if type(outgoing) is not list or type(state.flow.edges) is not list:
        raise TypeError("compiled flow edge storage and adjacency must be lists")
    authoritative: list[Edge] = []
    for edge in state.flow.edges:
        if not isinstance(edge, Edge):
            raise ValueError("compiled flow contains an invalid edge")
        if edge.from_ == node_id:
            authoritative.append(edge)
    if len(outgoing) != len(authoritative) or any(
        indexed is not stored for indexed, stored in zip(outgoing, authoritative)
    ):
        raise ValueError(
            f"route node {node_id!r} has an invalid outgoing adjacency"
        )
    seen: set[str] = set()
    validated: list[Edge] = []
    for edge in outgoing:
        if not isinstance(edge, Edge) or edge.from_ != node_id:
            raise ValueError(f"route node {node_id!r} has an invalid outgoing adjacency")
        if edge.source_handle is not None and type(edge.source_handle) is not str:
            raise TypeError(
                f"route edge {edge.id!r} source_handle must be a string or None"
            )
        if edge.source_handle == "":
            raise ValueError(f"route edge {edge.id!r} source_handle must be non-empty")
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
        and type(state.accepted_starts) is dict
        and type(state.depth) is dict
        and type(state.growth_owner) is dict
        and type(state.deferred_prune) is frozenset
        and type(state.wiring_overrides) is dict
        and type(state.input) is dict
        and type(state.effective_llm_by_static_path) is dict
        and type(state.backward_state) is RuntimeBackwardState
        and type(state._memory_authority) is _MemoryAuthority
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
        and not state.accepted_starts
        and state.result is None
        and state.terminal_unreachable_failure is None
        and not state.depth
        and not state.growth_owner
        and not state.deferred_prune
        and not state.wiring_overrides
        and state.run_id is None
        and state.flow_fingerprint is None
        and type(state.next_sequence) is int
        and state.next_sequence == 0
        and type(state.num_workers) is int
        and state.num_workers == 0
        and not state.input
        and not state.effective_llm_by_static_path
        and state.memory_snapshots is None
        and state.memories is None
        and state.memory_stacks is None
        and state.learnable is False
        and state.trace is None
        and state.backward_state == RuntimeBackwardState()
        and state._memory_authority.snapshots is None
        and state._memory_authority.memories is None
        and state._memory_authority.memory_stacks is None
        and state._memory_authority.learnable is False
        and state._memory_authority.memory_order is state.backward_state.memory_order
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
        | _BackwardPlan
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

    backward_state = RuntimeBackwardState(
        () if record.memories is None else tuple(spec.key for spec in record.memories)
    )
    memory_authority = _MemoryAuthority(
        snapshots=record.memory_snapshots,
        learnable=record.learnable,
        memories=record.memories,
        memory_stacks=record.memory_stacks,
        memory_order=backward_state.memory_order,
    )
    return _RunStartedApply(
        pool=pool,
        node_state=dict(state.node_state),
        edge_state=dict(state.edge_state),
        ready=deque((start_id,)),
        in_flight={},
        accepted_starts={},
        terminal_unreachable_failure=None,
        depth={},
        growth_owner={},
        wiring_overrides={start_id: canonical_wiring},
        run_id=envelope.run_id,
        flow_fingerprint=record.flow_fingerprint,
        next_sequence=envelope.seq + 1,
        num_workers=record.num_workers,
        input=canonical_input,
        effective_llm_by_static_path=canonical_effective,
        memory_snapshots=record.memory_snapshots,
        memories=record.memories,
        memory_stacks=record.memory_stacks,
        learnable=record.learnable,
        backward_state=backward_state,
        memory_authority=memory_authority,
        observers=(RunStarted(),),
    )


def _apply_run_started(state: RunState, apply: _RunStartedApply) -> None:
    """Install already-detached values using only total replacement assignments."""
    state.pool = apply.pool
    state.node_state = apply.node_state
    state.edge_state = apply.edge_state
    state.ready = apply.ready
    state.in_flight = apply.in_flight
    state.accepted_starts = apply.accepted_starts
    state.result = None
    state.terminal_unreachable_failure = apply.terminal_unreachable_failure
    state.depth = apply.depth
    state.growth_owner = apply.growth_owner
    state.wiring_overrides = apply.wiring_overrides
    state.run_id = apply.run_id
    state.flow_fingerprint = apply.flow_fingerprint
    state.num_workers = apply.num_workers
    state.input = apply.input
    state.effective_llm_by_static_path = apply.effective_llm_by_static_path
    state.memory_snapshots = apply.memory_snapshots
    state.memories = apply.memories
    state.memory_stacks = apply.memory_stacks
    state.learnable = apply.learnable
    state.backward_state = apply.backward_state
    state._memory_authority = apply.memory_authority
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
        if delta.failure is not None:
            raise ValueError("successful terminal delta cannot carry a failure")
        output = ANY_VALUE_ADAPTER.validate_json(
            delta.output_bytes,
            strict=True,
        ).to_object()
        return RunSucceeded(output=output)
    if delta.output_bytes is not None:
        raise ValueError("failed terminal delta cannot carry output bytes")
    if type(delta.failure) is not Failure:
        raise ValueError("failed terminal delta requires one Failure")
    return RunFailed(failure=delta.failure)


def _materialize_terminal(delta: _TerminalDelta) -> _TerminalApply:
    """Build independent state and callback terminal values before durable append."""
    return _TerminalApply(
        result=_terminal_observer(delta),
        observer=_terminal_observer(delta),
    )


def _failure_observers(failure: Failure) -> tuple[Any, ...]:
    """Construct ordered observers sharing one immutable Failure authority."""
    return (
        NodeFailed(failure=failure),
        RunFailed(failure=failure),
    )


def _materialize_failure(failure: Failure) -> _FailureApply:
    """Materialize detached state and callback values before durable append."""
    return _FailureApply(
        result=RunFailed(failure=failure),
        observers=_failure_observers(failure),
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
        accepted_starts=dict(topology.accepted_starts),
        depth=dict(topology.depth),
        growth_owner=dict(topology.growth_owner),
        deferred_prune=topology.deferred_prune,
        wiring_overrides={
            node_id: copy.deepcopy(wiring)
            for node_id, wiring in topology.wiring_overrides
        },
        observers=(_growth_observer(record),),
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

"""Accepted execution history and paired reads over immutable producer entries.

The state reducer prepares sparse deltas here before append and applies them only
after acceptance. Records own outcome data; private callable references live in a
separate index and never cross the wire. This module depends on runtime contracts
and codecs plus lower node/expression/value contracts, never the driver, public
run boundary, or compile expansion implementation.

Accepted growth metadata installs pending child seed routes. Child outcomes keep
their exact recorded capture while deriving an effective input route, with actual
second expression evaluations taking precedence. Historical boundary associations
survive consumption and live graph pruning.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping

from yian.diagnostics import Failure
from yian.expr.expressions import ExpressionError
from yian.expr.lineage import ResolvedValue, SourceRef, ValueRoute
from yian.memory import MemoryScope
from yian.nodes.base import BackwardContext, Node, _validate_declared_names
from yian.nodes.binding import BindingCapture
from yian.nodes.contracts import ChildFlowBoundary, ExecutionScope, SeedConnection
from yian.runtime.contracts import BackwardCapture, ForwardCapture, _validate_source
from yian.runtime.log import (
    NodeExpandedRecord, NodeFailedRecord, NodeRoutedRecord, NodeStartedRecord,
    NodeSucceededRecord, RecordEnvelope, RunResumedRecord, RunStartedRecord,
    _decode_typed, _evidence_payload, canonical_seed, decode_pool, decode_record, encode_record,
)
from yian.typesys.values import ANY_VALUE_ADAPTER, field_of


def capture_bindings(
    bindings: BindingCapture,
    backward: BackwardCapture,
) -> ForwardCapture:
    """Join detached binding and backward material through the existing seed codec.

    Args:
        bindings (`BindingCapture`): Actual delivered values, routes, and indexed
            item bindings frozen by the evaluator.
        backward (`BackwardCapture`): The same invocation's declared names, scope,
            and captured node data. Item seeds have no separate node inventory.

    Returns:
        `ForwardCapture`: Canonical material ready for accepted-record validation.

    Raises:
        `TypeError`: If either transient carrier has the wrong type.
        `ValueError`: If captured data cannot satisfy the canonical wire contract.
    """
    if type(bindings) is not BindingCapture:
        raise TypeError("retained evaluation requires a BindingCapture")
    if type(backward) is not BackwardCapture:
        raise TypeError("retained evaluation requires a BackwardCapture")
    return ForwardCapture(
        canonical_seed(bindings.bound_inputs)[1], bindings.route,
        bindings.evaluated_keys,
        tuple((index, capture_bindings(item, BackwardCapture())) for index, item in bindings.items),
        backward.scope, backward.node_data,
        input_names=backward.input_names, variable_names=backward.variable_names,
    )


@dataclass(frozen=True, slots=True)
class SourceEntry:
    """A typed accepted value and its producer identity read as one unit."""

    source: SourceRef | None
    outcome_seq: int
    _value_bytes: bytes = field(repr=False)

    @property
    def value(self) -> Any:
        """Decode a detached typed value with the same semantics as the pool."""
        return ANY_VALUE_ADAPTER.validate_json(self._value_bytes, strict=True).to_object()


@dataclass(frozen=True, slots=True)
class TraceTerminal:
    """The reducer's terminal result and its actual accepted causal identity."""

    succeeded: bool
    source: SourceRef | None
    output_bytes: bytes | None
    failure: Failure | None

    def __post_init__(self) -> None:
        """Require distinct success/null and failure shapes with canonical values."""
        if type(self.succeeded) is not bool:
            raise TypeError("terminal succeeded must be a bool")
        if self.source is not None:
            _validate_source(self.source)
            if self.source.path:
                raise ValueError("terminal source must identify a root value")
        if self.succeeded:
            if type(self.output_bytes) is not bytes or self.failure is not None:
                raise ValueError("successful terminal requires output bytes and no failure")
            ANY_VALUE_ADAPTER.validate_json(self.output_bytes, strict=True)
        else:
            if self.output_bytes is not None or type(self.failure) is not Failure:
                raise ValueError("failed terminal requires a failure and no output")
            Failure.__post_init__(self.failure)

    @property
    def output(self) -> Any:
        """Return detached canonical output, or None when the terminal failed."""
        if not self.succeeded:
            return None
        value = ANY_VALUE_ADAPTER.validate_json(self.output_bytes, strict=True).to_object()
        return canonical_seed(value)[1]


def _copy_memory_scope(scope: MemoryScope | None) -> MemoryScope | None:
    """Detach validated memory keys without deriving execution coordinates."""
    if scope is None:
        return None
    if type(scope) is not MemoryScope:
        raise TypeError("historical memory scope must be a MemoryScope or None")
    return deepcopy(MemoryScope(scope.static_path, scope.memories, scope.occurrence_path))


@dataclass(frozen=True, slots=True)
class TraceOccurrence:
    """One accepted outcome and its derived route, identified by its accepted start.

    Private memory ownership is detached from the physical node before pruning.
    Replay reconstructs it from that node; it never replaces execution scope or
    adds memory ownership to the accepted outcome payload.
    """

    source: SourceRef
    node_id: str
    outcome_seq: int
    _record_bytes: bytes = field(repr=False)
    input_route: ValueRoute
    _memory_scope: MemoryScope | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Keep the detached memory scope alongside the historical record."""
        object.__setattr__(self, "_memory_scope", _copy_memory_scope(self._memory_scope))

    @property
    def memory_scope(self) -> MemoryScope | None:
        """Return detached receiver memory-scope metadata without relabeling execution."""
        return _copy_memory_scope(self._memory_scope)

    @property
    def record(self):
        """Return a detached view of the authoritative accepted outcome record."""
        return decode_record(self._record_bytes).record


@dataclass(frozen=True, slots=True)
class TraceBoundary:
    """One accepted expansion's child flow and its actual accepted endpoints."""

    owner: SourceRef
    start_id: str
    end_id: str
    start_source: SourceRef | None = None
    end_source: SourceRef | None = None


@dataclass(frozen=True, slots=True)
class _SeedRoutes:
    """Candidate input routes owned by one accepted growth occurrence."""

    owner: SourceRef
    fields: tuple[tuple[str, ValueRoute], ...]


def select_seed_route(
    capture: ForwardCapture, connection: SeedConnection, input_route: ValueRoute | None = None,
) -> ValueRoute | None:
    """Select one seed field route by explicit item/name metadata, preserving nulls.

    Args:
        capture (`ForwardCapture`): The owner's accepted forward capture.
        connection (`SeedConnection`): The seed connection naming an item and/or field.
        input_route (`ValueRoute`, *optional*, defaults to `None`): The owner's effective
            input route; when given, a field-only path selects from it instead of the raw
            capture route.

    Returns:
        `ValueRoute | None`: The selected route, or `None` when the item or field was not
            captured.
    """
    route = capture.route if input_route is None else input_route
    if len(connection.seed_path) == 2:
        binding = dict(capture.items).get(connection.seed_path[0])
        if binding is None:
            return None
        route = binding.route
    return dict(zip(route.names, route.children)).get(connection.seed_path[-1])


class RetainedTrace:
    """Reducer-owned history with immutable entries and sparse mutable indexes.

    A worker reads one current entry at binding time. Pruning removes current
    names only; historical values, accepted occurrences and private handlers stay
    available independently of the live scheduling graph.
    """

    def __init__(
        self, run_id: str, start_id: str, end_id: str | None = None,
        root_input_names: tuple[str, ...] = (),
    ) -> None:
        self.run_id = run_id
        self.start_id = start_id
        self.end_id = end_id
        _validate_declared_names(root_input_names, "root_input_names")
        self.root_input_names = root_input_names
        self.root_start_source: SourceRef | None = None
        self.terminal: TraceTerminal | None = None
        self._root_writer: SourceEntry | None = None
        self._causal_route_source: SourceRef | None = None
        self.next_sequence = 0
        self._occurrences: list[TraceOccurrence] = []
        self._occurrence_by_source: dict[SourceRef, TraceOccurrence] = {}
        self._sources: dict[SourceRef, SourceEntry] = {}
        self._current_writers: dict[str, SourceEntry] = {}
        self._started: dict[str, int] = {}
        self._backward_handlers: dict[SourceRef, Callable[..., Any]] = {}
        self._system: dict[str, bytes] = {}
        self._pending_connections: dict[str, _SeedRoutes] = {}
        self._boundaries: list[TraceBoundary] = []
        self._current_boundaries: dict[str, tuple[int, str]] = {}

    @property
    def pending_connections(self) -> Mapping[str, _SeedRoutes]:
        """Expose unconsumed child seed candidates without a writable index."""
        return MappingProxyType(self._pending_connections)

    @property
    def boundaries(self) -> tuple[TraceBoundary, ...]:
        """Retain accepted child associations even after their live nodes are pruned."""
        return tuple(self._boundaries)

    @property
    def occurrences(self) -> tuple[TraceOccurrence, ...]:
        """Return accepted outcomes in accepted record order."""
        return tuple(self._occurrences)

    @property
    def sources(self) -> Mapping[SourceRef, SourceEntry]:
        """Expose immutable historical source entries without writable indexes."""
        return MappingProxyType(self._sources)

    @property
    def current_writers(self) -> Mapping[str, SourceEntry]:
        """Expose current logical pool names and their physical producers."""
        return MappingProxyType(self._current_writers)

    @property
    def started(self) -> Mapping[str, int]:
        """Expose accepted unfinished start identities."""
        return MappingProxyType(self._started)

    @property
    def backward_handlers(self) -> Mapping[SourceRef, Callable[..., Any]]:
        """Expose reconstructed private node callables separately from records."""
        return MappingProxyType(self._backward_handlers)

    def lookup_source(self, source: SourceRef) -> Any:
        """Return canonical source-shaped data for an accepted producer."""
        root = SourceRef(source.run_id, source.execution_seq)
        value = canonical_seed(self._sources[root].value)[1]
        for part in source.path:
            value = value[part]
        return value

    def occurrence(self, source: SourceRef) -> TraceOccurrence:
        """Look up an accepted occurrence by its physical start identity.

        Args:
            source (`SourceRef`): Physical start identity with an empty path.

        Returns:
            `TraceOccurrence`: The accepted outcome and derived input route.

        Raises:
            `KeyError`: If no accepted outcome has this identity, including source zero.
        """
        return self._occurrence_by_source[source]

    def backward_context(
        self, source: SourceRef, *, downstream_context: tuple[dict[str, Any], ...] = (),
    ) -> BackwardContext:
        """Build detached context from accepted invocation facts without calling hooks.

        Args:
            source (`SourceRef`): Physical accepted start identity of an executed
                occurrence.
            downstream_context (`tuple[dict[str, Any], ...]`, *optional*, defaults
                to `()`): Transient attribution materialized by a caller. It is
                copied with the other context values.

        Returns:
            `BackwardContext`: Recorded names, usable inputs, node data, and the
                actual output or canonical forward failure. The private reserved
                `_evaluation_evidence` entry contains only accepted record evidence.

        Raises:
            `KeyError`: If the occurrence or its accepted output source is absent.
            `TypeError`: If supplied context material violates the context contract.
            `ValueError`: If retained or supplied data violates canonical constraints.
        """
        occurrence = self.occurrence(source)
        record = occurrence.record
        capture = record.forward_capture
        data = capture.node_data
        if data is None:
            data = {}
        if "_evaluation_evidence" in data:
            raise ValueError("captured node data uses reserved _evaluation_evidence")
        data["_evaluation_evidence"] = _evidence_payload(record.evidence)
        failure = record.failure if type(record) is NodeFailedRecord else (
            record.terminal_failure if type(record) is NodeRoutedRecord else None
        )
        return BackwardContext(
            capture.bound_inputs, input_names=capture.input_names, variable_names=capture.variable_names,
            output=self.lookup_source(source) if type(record) is NodeSucceededRecord else None,
            failure=failure, scope=capture.scope, run_id=source.run_id,
            node_id=occurrence.node_id, node_started_seq=source.execution_seq,
            node_data=data, downstream_context=downstream_context,
        )

    def resolve(self, path: str) -> ResolvedValue:
        """Pair each actual lookup's value, presence and producer in one read."""
        parts = path.strip().split(".")
        if any(not part for part in parts):
            raise ExpressionError(f"invalid variable path: '${{{path}}}'")
        head, *rest = parts
        entry = self._current_writers.get(head)
        if entry is not None and rest and rest[0] == "output":
            projection = rest[1:]
        elif head == "input" and rest:
            entry = self._current_writers.get(self.start_id)
            projection = rest
        elif head == "system":
            encoded = self._system.get(rest[0]) if rest else None
            if encoded is None:
                return ResolvedValue(None, False)
            value = ANY_VALUE_ADAPTER.validate_json(encoded, strict=True).to_object()
            return ResolvedValue(value, True)
        else:
            return ResolvedValue(None, False)
        if entry is None:
            return ResolvedValue(None, False)
        value = entry.value
        for part in projection:
            present, value = field_of(value, part)
            if not present:
                return ResolvedValue(None, False)
        source = entry.source
        return ResolvedValue(value, True, None if source is None else SourceRef(
            source.run_id, source.execution_seq, tuple(projection),
        ))


@dataclass(frozen=True, slots=True)
class TraceDelta:
    """A fully prepared sparse change, applied only after its record is accepted."""

    originating_trace: RetainedTrace | None
    sequence: int
    initialized: RetainedTrace | None = field(default=None, repr=False)
    started: tuple[str, int] | None = None
    resumed: bool = False
    occurrence: TraceOccurrence | None = None
    handler: Callable[..., Any] | None = field(default=None, repr=False, compare=False)
    writer: tuple[str, SourceEntry] | None = None
    prune_ids: tuple[str, ...] = ()
    connections: tuple[tuple[str, _SeedRoutes], ...] = ()
    consumed_connection: tuple[str, _SeedRoutes] | None = None
    boundaries: tuple[TraceBoundary, ...] = ()
    boundary_update: tuple[int, TraceBoundary] | None = None
    root_start_source: SourceRef | None = None
    root_writer: SourceEntry | None = None
    causal_route_source: SourceRef | None = None
    terminal: TraceTerminal | None = None

    def validate_origin(self, trace: RetainedTrace | None) -> None:
        """Reject stale or replaced trace authority before durable append."""
        if trace is not self.originating_trace:
            raise ValueError("trace transition requires its originating trace")
        if trace is not None and trace.next_sequence != self.sequence:
            raise ValueError("trace transition has a stale sequence")
        if self.occurrence is not None:
            if type(self.occurrence) is not TraceOccurrence:
                raise TypeError("trace delta requires an exact occurrence")
            _copy_memory_scope(self.occurrence._memory_scope)
        if self.consumed_connection is not None:
            node_id, routes = self.consumed_connection
            if trace is None or trace._pending_connections.get(node_id) is not routes:
                raise ValueError("seed connection no longer has its prepared owner")

    def apply(self) -> RetainedTrace:
        """Install only prevalidated index updates after accepted reducer apply."""
        trace = self.initialized if self.initialized is not None else self.originating_trace
        if self.started is not None:
            node_id, sequence = self.started
            trace._started[node_id] = sequence
        if self.resumed:
            trace._started.clear()
        if self.occurrence is not None:
            trace._occurrences.append(self.occurrence)
            trace._occurrence_by_source[self.occurrence.source] = self.occurrence
            trace._backward_handlers[self.occurrence.source] = self.handler
            del trace._started[self.occurrence.node_id]
            trace._current_boundaries.pop(self.occurrence.node_id, None)
        if self.consumed_connection is not None:
            node_id, _routes = self.consumed_connection
            del trace._pending_connections[node_id]
        if self.boundary_update is not None:
            index, boundary = self.boundary_update
            trace._boundaries[index] = boundary
        for node_id in self.prune_ids:
            trace._current_writers.pop(node_id, None)
            trace._pending_connections.pop(node_id, None)
            trace._current_boundaries.pop(node_id, None)
        trace._pending_connections.update(self.connections)
        for boundary in self.boundaries:
            index = len(trace._boundaries)
            trace._boundaries.append(boundary)
            trace._current_boundaries[boundary.start_id] = (index, "start_source")
            trace._current_boundaries[boundary.end_id] = (index, "end_source")
        if self.writer is not None:
            target, entry = self.writer
            trace._current_writers[target] = entry
            trace._sources[entry.source] = entry
        if self.root_start_source is not None:
            trace.root_start_source = self.root_start_source
        if self.root_writer is not None:
            trace._root_writer = self.root_writer
        if self.causal_route_source is not None:
            trace._causal_route_source = self.causal_route_source
        if self.terminal is not None:
            trace.terminal = self.terminal
        trace.next_sequence = self.sequence + 1
        return trace


def prepare_trace(
    trace: RetainedTrace | None,
    envelope: RecordEnvelope,
    node: Node | None,
    *,
    input_connections: tuple[SeedConnection, ...] = (),
    child_boundaries: tuple[ChildFlowBoundary, ...] = (),
    retired_ids: tuple[str, ...] = (),
    root_end_id: str | None = None,
    root_input_names: tuple[str, ...] | None = None,
    terminal_succeeded: bool | None = None,
    terminal_output_bytes: bytes | None = None,
    terminal_failure: Failure | None = None,
) -> TraceDelta | None:
    """Validate capture authority and prepare a detached accepted-record delta.

    ``retired_ids`` are the settled prune IDs, earlier deferred ones, and the nodes grown under them.
    """
    record = envelope.record
    if type(record) is RunStartedRecord:
        if not record.learnable:
            return None
        pool = decode_pool(record.initial_pool)
        if type(root_end_id) is not str or root_input_names is None:
            raise ValueError("retained RunStarted requires verified root metadata")
        initial = RetainedTrace(envelope.run_id, pool.start_id, root_end_id, root_input_names)
        for target, typed in pool.store.items():
            source = SourceRef(envelope.run_id, 0) if target == pool.start_id else None
            entry = SourceEntry(source, 0, ANY_VALUE_ADAPTER.dump_json(typed, warnings="error"))
            initial._current_writers[target] = entry
            if source is not None:
                initial._sources[source] = entry
        initial._root_writer = initial._current_writers.get(root_end_id)
        initial._system = {key: ANY_VALUE_ADAPTER.dump_json(value, warnings="error")
                           for key, value in pool.system.items()}
        return TraceDelta(None, envelope.seq, initialized=initial)
    capture = getattr(record, "forward_capture", None)
    if trace is None:
        if capture is not None:
            raise ValueError("forward capture requires recorded learning")
        return None
    if trace.run_id != envelope.run_id or trace.next_sequence != envelope.seq:
        raise ValueError("trace disagrees with accepted record coordinates")
    if terminal_succeeded is None and (terminal_output_bytes is not None or terminal_failure is not None):
        raise ValueError("terminal payload requires a terminal disposition")
    if type(record) is NodeStartedRecord:
        return TraceDelta(trace, envelope.seq, started=(record.node_id, envelope.seq))
    if type(record) is RunResumedRecord:
        if tuple(trace._started) != record.retry_node_ids:
            raise ValueError("trace unfinished starts disagree with retry frontier")
        return TraceDelta(trace, envelope.seq, resumed=True)
    if type(record) not in (NodeSucceededRecord, NodeRoutedRecord, NodeExpandedRecord, NodeFailedRecord):
        return TraceDelta(trace, envelope.seq)
    if type(capture) is not ForwardCapture:
        raise ValueError("accepted outcome requires forward capture while learning")
    if node is None or node.id not in trace._started:
        raise ValueError("trace outcome requires its accepted start")
    if capture.scope != node.execution_scope:
        raise ValueError("forward capture scope does not match the executing node")
    pending = [capture]
    while pending:
        binding = pending.pop()
        _validate_sources(trace, binding.route, envelope.seq)
        pending.extend(item for _, item in binding.items)
    source = SourceRef(envelope.run_id, trace._started[node.id])
    candidates = trace._pending_connections.get(node.id)
    input_route = capture.route
    if candidates is not None:
        fields = dict(candidates.fields)
        input_route = ValueRoute.record(
            (name, route if name in capture.evaluated_keys else fields.get(name, route))
            for name, route in zip(capture.route.names, capture.route.children)
        )
    _validate_sources(trace, input_route, envelope.seq)
    occurrence = TraceOccurrence(source, node.id, envelope.seq, encode_record(envelope), input_route, node.memory_scope)
    boundary_update = None
    boundary_target = trace._current_boundaries.get(node.id)
    if boundary_target is not None:
        index, role = boundary_target
        boundary_update = (index, replace(trace._boundaries[index], **{role: source}))
    connections: dict[str, list[tuple[str, ValueRoute]]] = {}
    for connection in sorted(input_connections, key=lambda item: (item.child_start_id, item.input_name)):
        # An owner that is itself a connection target (a LOOP driver whose omitted field
        # binds a default) forwards its joined route rather than the raw default, so
        # lineage survives every iteration.
        route = select_seed_route(capture, connection, input_route)
        if route is None:
            raise ValueError("seed connection field or item was not captured")
        connections.setdefault(connection.child_start_id, []).append((connection.input_name, route))
    for node_id in connections:
        if node_id in trace._pending_connections:
            raise ValueError("seed connection target already has an accepted owner")
    boundaries = tuple(
        TraceBoundary(source, boundary.start_id, boundary.end_id)
        for boundary in sorted(child_boundaries, key=lambda item: (item.start_id, item.end_id))
    )
    if any(node_id in trace._current_boundaries for boundary in boundaries
           for node_id in (boundary.start_id, boundary.end_id)):
        raise ValueError("child boundary target already has an accepted owner")
    handler = node.backward
    if not callable(handler):
        raise TypeError("node backward handler must be callable")
    writer = None
    if type(record) is NodeSucceededRecord:
        typed = _decode_typed(record.value)
        writer = (record.commit_target, SourceEntry(
            source, envelope.seq, ANY_VALUE_ADAPTER.dump_json(typed, warnings="error"),
        ))
    root_writer = writer[1] if writer is not None and writer[0] == trace.end_id else None
    causal_route_source = (
        source if type(record) is NodeRoutedRecord and record.terminal_failure is not None
        and trace._causal_route_source is None else None
    )
    terminal = None
    if terminal_succeeded is not None:
        if terminal_succeeded:
            accepted_writer = root_writer or trace._root_writer
            if accepted_writer is None or ANY_VALUE_ADAPTER.validate_json(
                accepted_writer._value_bytes, strict=True,
            ) != ANY_VALUE_ADAPTER.validate_json(terminal_output_bytes, strict=True):
                raise ValueError("terminal output disagrees with the accepted root writer")
            terminal_source = accepted_writer.source
        else:
            terminal_source = (
                source if type(record) is NodeFailedRecord
                else causal_route_source or trace._causal_route_source
            )
            if terminal_source is None:
                raise ValueError("accepted failure requires an accepted causal occurrence")
        terminal = TraceTerminal(
            terminal_succeeded, terminal_source, terminal_output_bytes, terminal_failure,
        )
    return TraceDelta(
        trace, envelope.seq, occurrence=occurrence, handler=handler, writer=writer,
        prune_ids=retired_ids if type(record) is NodeExpandedRecord else (),
        connections=tuple((node_id, _SeedRoutes(source, tuple(fields)))
                          for node_id, fields in connections.items()),
        consumed_connection=None if candidates is None else (node.id, candidates),
        boundaries=boundaries, boundary_update=boundary_update,
        root_start_source=source if node.id == trace.start_id and trace.root_start_source is None else None,
        root_writer=root_writer, causal_route_source=causal_route_source, terminal=terminal,
    )


def _validate_sources(trace: RetainedTrace, route: ValueRoute, outcome_seq: int) -> None:
    """Require sources accepted before the consumer outcome, regardless of starts."""
    if route.source is not None:
        source = route.source
        root = SourceRef(source.run_id, source.execution_seq)
        entry = trace._sources.get(root)
        if entry is None or entry.outcome_seq >= outcome_seq:
            raise ValueError("route source must name an earlier accepted producer")
        try:
            trace.lookup_source(source)
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("route source path is absent from its accepted value") from exc
    for child in route.children:
        _validate_sources(trace, child, outcome_seq)


__all__ = ["RetainedTrace", "SourceEntry", "TraceBoundary", "TraceOccurrence", "TraceTerminal"]

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

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping

from agent_composer.expr.expressions import ExpressionError
from agent_composer.expr.lineage import ResolvedValue, SourceRef, ValueRoute
from agent_composer.nodes.base import Node
from agent_composer.nodes.binding import BindingCapture
from agent_composer.nodes.contracts import ChildFlowBoundary, ExecutionScope, SeedConnection
from agent_composer.runtime.contracts import ForwardCapture
from agent_composer.runtime.log import (
    NodeExpandedRecord, NodeFailedRecord, NodeRoutedRecord, NodeStartedRecord,
    NodeSucceededRecord, RecordEnvelope, RunResumedRecord, RunStartedRecord,
    _decode_typed, canonical_seed, decode_pool, decode_record, encode_record,
)
from agent_composer.typesys.values import ANY_VALUE_ADAPTER


def capture_bindings(
    bindings: BindingCapture,
    scope: ExecutionScope | None,
) -> ForwardCapture:
    """Canonicalize actual detached binder values through the existing seed codec."""
    if type(bindings) is not BindingCapture:
        raise TypeError("retained evaluation requires a BindingCapture")
    return ForwardCapture(
        canonical_seed(bindings.bound_inputs)[1], bindings.route,
        bindings.evaluated_keys,
        tuple((index, capture_bindings(item, None)) for index, item in bindings.items),
        scope,
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
class TraceOccurrence:
    """One accepted outcome and its derived route, identified by its accepted start."""

    source: SourceRef
    node_id: str
    outcome_seq: int
    _record_bytes: bytes = field(repr=False)
    input_route: ValueRoute

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


def select_seed_route(capture: ForwardCapture, connection: SeedConnection) -> ValueRoute | None:
    """Select one captured field by explicit item/name metadata, preserving nulls."""
    binding = capture
    if len(connection.seed_path) == 2:
        binding = dict(capture.items).get(connection.seed_path[0])
        if binding is None:
            return None
    return dict(zip(binding.route.names, binding.route.children)).get(connection.seed_path[-1])


class RetainedTrace:
    """Reducer-owned history with immutable entries and sparse mutable indexes.

    A worker reads one current entry at binding time. Pruning removes current
    names only; historical values, accepted occurrences and private handlers stay
    available independently of the live scheduling graph.
    """

    def __init__(self, run_id: str, start_id: str) -> None:
        self.run_id = run_id
        self.start_id = start_id
        self.next_sequence = 0
        self._occurrences: list[TraceOccurrence] = []
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
            if not isinstance(value, dict) or part not in value:
                return ResolvedValue(None, False)
            value = value[part]
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

    def validate_origin(self, trace: RetainedTrace | None) -> None:
        """Reject stale or replaced trace authority before durable append."""
        if trace is not self.originating_trace:
            raise ValueError("trace transition requires its originating trace")
        if trace is not None and trace.next_sequence != self.sequence:
            raise ValueError("trace transition has a stale sequence")
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
        trace.next_sequence = self.sequence + 1
        return trace


def prepare_trace(
    trace: RetainedTrace | None,
    envelope: RecordEnvelope,
    node: Node | None,
    *,
    input_connections: tuple[SeedConnection, ...] = (),
    child_boundaries: tuple[ChildFlowBoundary, ...] = (),
) -> TraceDelta | None:
    """Validate capture authority and prepare a detached accepted-record delta."""
    record = envelope.record
    if type(record) is RunStartedRecord:
        if not record.learn:
            return None
        pool = decode_pool(record.initial_pool)
        initial = RetainedTrace(envelope.run_id, pool.start_id)
        for target, typed in pool.store.items():
            source = SourceRef(envelope.run_id, 0) if target == pool.start_id else None
            entry = SourceEntry(source, 0, ANY_VALUE_ADAPTER.dump_json(typed, warnings="error"))
            initial._current_writers[target] = entry
            if source is not None:
                initial._sources[source] = entry
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
    occurrence = TraceOccurrence(source, node.id, envelope.seq, encode_record(envelope), input_route)
    boundary_update = None
    boundary_target = trace._current_boundaries.get(node.id)
    if boundary_target is not None:
        index, role = boundary_target
        boundary_update = (index, replace(trace._boundaries[index], **{role: source}))
    connections: dict[str, list[tuple[str, ValueRoute]]] = {}
    for connection in sorted(input_connections, key=lambda item: (item.child_start_id, item.input_name)):
        route = select_seed_route(capture, connection)
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
    return TraceDelta(
        trace, envelope.seq, occurrence=occurrence, handler=handler, writer=writer,
        prune_ids=record.prune_ids if type(record) is NodeExpandedRecord else (),
        connections=tuple((node_id, _SeedRoutes(source, tuple(fields)))
                          for node_id, fields in connections.items()),
        consumed_connection=None if candidates is None else (node.id, candidates),
        boundaries=boundaries, boundary_update=boundary_update,
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


__all__ = ["RetainedTrace", "SourceEntry", "TraceBoundary", "TraceOccurrence"]

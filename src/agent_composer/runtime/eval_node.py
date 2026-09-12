"""Node evaluation at the engine's pure read/execution boundary.

A node is a pure function of its bound input record; the ENGINE owns both boundaries —
the read (bind the inputs from the pool) and the write (the dispatcher stores `Output`).
``eval_node`` is the append-only dispatcher's evaluation seam: it binds the record,
pre-resolves reserved keys, builds trait-gated capabilities, runs the node through its
assert-owning ``run`` wrapper, forwards transient chunks through ``emit_stream``, and
returns an ``EvaluatedOutcome`` containing exactly
``Output | Route | Grow | EvaluationFailure`` plus optional evidence and input capture. It emits no
observer notifications; durable observer ordering belongs to the dispatcher.

Expected rejection is classified where its operation occurs: binding evaluation,
supplied-input contracts, wrapper assertions, node execution, or node-result validation.
Host control, stream callback exceptions, unexpected binder faults, and malformed
compiled/setup state propagate outside ``EvaluationFailure``.

The bind is PURE: it reads the node's sources ONLY from `flow.wiring[node.id]` joined to the
node's `params` — the node carries no source. Direct-drive tests supply a stub `flow` with
`wiring` (the test helpers derive it). Layer: runtime imports nodes.* freely.

Binding precedes node-owned setup validation. Thus an unreachable unbaked MAP with bad
``over``/input wiring reports that binding rejection first; with valid binding, its missing
compiled child propagates as malformed setup. The loader always bakes valid composites.
"""

from __future__ import annotations

import inspect
import threading
import traceback as _tb
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any, TypeAlias

from agent_composer.diagnostics import Failure, FailureCode, OriginRef
from agent_composer.events import StreamChunk
from agent_composer.expr.lineage import ResolvedValue, ValueRoute
from agent_composer.memory import (
    EvaluationEvidence,
    EvaluationTrace,
    MemoryRecallError,
    MemoryScope,
    MemorySection,
    RecallContext,
    VisibleInteraction,
)
from agent_composer.nodes.base import (
    Grow,
    Node,
    Output,
    Route,
    _AssertionStageError,
    _BindingStageError,
    _CapabilityFault,
    _DeferredBindingFault,
    _HostStop,
    _NodeBodyError,
)
from agent_composer.nodes.binding import (
    BindingCapture,
    BindingCollector,
    BindingError,
    SourceResolver,
    _resolve_source,
    bind_params,
)
from agent_composer.runtime.contracts import BackwardCapture
from agent_composer.runtime.log import canonical_seed
from agent_composer.typesys.pool import VariablePool


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    """One raw node-evaluation failure for durable dispatcher preparation.

    ``failure`` is the sole semantic and presentation authority.
    """

    failure: Failure


def _evaluation_failure(
    node,
    code: FailureCode,
    exception: Exception,
    *,
    origin: OriginRef | None = None,
    traceback_text: str | None = None,
) -> EvaluationFailure:
    """Build one semantic failure with its canonical opaque origin trace."""
    return EvaluationFailure(
        failure=Failure(
            code=code,
            message=str(exception),
            node_id=node.id,
            trace=node.failure_trace(origin),
            cause_type=type(exception).__name__,
            traceback=(traceback_text if traceback_text is not None else _tb.format_exc()),
        ),
    )


def _format_exception(exception: Exception) -> str:
    """Format one original exception without a private stage-marker tail."""
    return "".join(
        _tb.format_exception(
            type(exception),
            exception,
            exception.__traceback__,
        )
    )


WorkerOutcome: TypeAlias = Output | Route | Grow | EvaluationFailure
"""Closed raw worker result consumed by the append-only dispatcher."""

MemoryRecall: TypeAlias = Callable[
    [MemoryScope, RecallContext],
    tuple[MemorySection, ...],
]
"""Injected source-neutral recall boundary for one compiled node scope."""

NodeMemoryRecall: TypeAlias = Callable[
    [str, str],
    tuple[MemorySection, ...],
]
"""Narrow node capability after the evaluator captures scope and bound inputs."""


@dataclass(frozen=True, slots=True)
class EvaluatedOutcome:
    """One raw worker outcome with frozen evidence and optional backward capture.

    Binding values/routes and common inventories/static material travel together
    as separate lower-binding and runtime carriers. Neither is accepted history
    until the dispatcher records this outcome.
    """

    outcome: WorkerOutcome
    evidence: EvaluationEvidence | None
    bindings: BindingCapture | None = None
    backward_capture: BackwardCapture | None = None

    def __post_init__(self) -> None:
        """Require one closed raw outcome and exact optional evidence."""
        if not isinstance(self.outcome, (Output, Route, Grow, EvaluationFailure)):
            raise TypeError("outcome must be a WorkerOutcome")
        if self.evidence is not None and type(self.evidence) is not EvaluationEvidence:
            raise TypeError("evidence must be an EvaluationEvidence or None")
        if self.bindings is not None and type(self.bindings) is not BindingCapture:
            raise TypeError("bindings must be a BindingCapture or None")
        if self.backward_capture is not None:
            if type(self.backward_capture) is not BackwardCapture:
                raise TypeError("backward_capture must be a BackwardCapture or None")
            capture = self.backward_capture
            BackwardCapture(capture.input_names, capture.variable_names, capture.scope, capture.node_data)
        if (self.bindings is None) != (self.backward_capture is None):
            raise ValueError("binding capture and backward invocation metadata must be retained together")


class _EvaluationTraceCollector(EvaluationTrace):
    """Private mutable trace frozen once at the evaluator return boundary."""

    def __init__(self, scope: MemoryScope | None) -> None:
        self._scope = scope
        self._bound_inputs: dict[str, Any] | None = None
        self._base_system_prompt: str | None = None
        self._prompt: str | None = None
        self._memory_sections: tuple[MemorySection, ...] = ()
        self._interactions: list[VisibleInteraction] = []
        self._prompt_recorded = False
        self._frozen = False

    def _require_open(self) -> None:
        """Reject writes after the evaluator has frozen this invocation."""
        if self._frozen:
            raise RuntimeError("evaluation trace is already frozen")

    def _set_bound_inputs(self, bound_inputs: dict[str, Any]) -> None:
        """Retain the evaluator's canonical pre-execution input snapshot."""
        self._require_open()
        if type(bound_inputs) is not dict:
            raise TypeError("bound_inputs must be an exact dict")
        try:
            validated = EvaluationEvidence(self._scope, bound_inputs, None, None)
        except (TypeError, ValueError, RecursionError):
            # Evidence is observational. The run-log can carry deeper ordinary
            # values than the deliberately conservative memory JSON boundary.
            self._bound_inputs = None
            return
        self._bound_inputs = validated.bound_inputs

    def record_prompt(
        self,
        base_system_prompt: str,
        prompt: str,
        memory_sections: tuple[MemorySection, ...] = (),
    ) -> None:
        """Record one exact visible prompt and its ordered memory sections."""
        self._require_open()
        if self._prompt_recorded:
            raise RuntimeError("evaluation trace prompt was already recorded")
        self._prompt_recorded = True
        try:
            validated = EvaluationEvidence(
                self._scope,
                None,
                base_system_prompt,
                prompt,
                memory_sections,
            )
        except (TypeError, ValueError, RecursionError):
            # The model already received this prompt. Trace serialization must not
            # reclassify that ordinary provider execution as a node failure.
            return
        self._base_system_prompt = validated.base_system_prompt
        self._prompt = validated.prompt
        self._memory_sections = validated.memory_sections

    def record_interaction(self, interaction: VisibleInteraction) -> None:
        """Append one exact visible interaction without exposing trace reads."""
        self._require_open()
        validated = EvaluationEvidence(
            self._scope,
            None,
            None,
            None,
            interactions=(interaction,),
        )
        self._interactions.append(validated.interactions[0])

    def _freeze(self) -> EvaluationEvidence:
        """Close the collector and return its immutable durable carrier."""
        self._require_open()
        self._frozen = True
        return EvaluationEvidence(
            self._scope,
            self._bound_inputs,
            self._base_system_prompt,
            self._prompt,
            self._memory_sections,
            tuple(self._interactions),
        )


def _new_evaluation_trace(
    node: Node,
    *,
    capture_evidence: bool,
) -> _EvaluationTraceCollector | None:
    """Apply the exact kind-blind trait/scope capture gate."""
    if not capture_evidence or node.needs_evaluation_trace is not True:
        return None
    scope = node.memory_scope
    if scope is not None and type(scope) is not MemoryScope:
        raise TypeError("node memory_scope must be an exact MemoryScope or None")
    return _EvaluationTraceCollector(scope)


def _validated_memory_sections(
    scope: MemoryScope,
    sections: tuple[MemorySection, ...],
) -> tuple[MemorySection, ...]:
    """Validate one callback result as an ordered subsequence of the scope stack."""
    if type(sections) is not tuple:
        raise TypeError("memory recall must return an exact tuple")
    next_index = 0
    for section in sections:
        if type(section) is not MemorySection:
            raise TypeError("memory recall items must be MemorySection values")
        MemorySection(section.frame_ref, section.text)
        if not section.text:
            raise ValueError("memory recall sections must contain non-empty text")
        try:
            index = scope.frame_stack.index(section.frame_ref, next_index)
        except ValueError as exc:
            raise ValueError(
                "memory recall sections must follow the consumer frame stack"
            ) from exc
        next_index = index + 1
    return sections


def _memory_recall_capability(
    node: Node,
    record: dict[str, Any],
    memory_recall: MemoryRecall | None,
) -> NodeMemoryRecall | None:
    """Bind one injected recall seam to a node's immutable invocation context."""
    if node.needs_memory_recall is not True:
        return None
    scope = node.memory_scope
    if scope is None:
        return None
    if type(scope) is not MemoryScope:
        raise TypeError("node memory_scope must be an exact MemoryScope or None")
    MemoryScope(
        scope.root_flow_id,
        scope.static_path,
        scope.frame_stack,
        scope.occurrence_path,
        scope.boundary,
    )
    if not scope.frame_stack:
        return None
    if memory_recall is None:
        raise RuntimeError(
            f"scoped node {node.id!r} requires an injected memory recall callback"
        )

    invalid_inputs = False
    try:
        canonical_inputs = canonical_seed(record)[0]
        if type(canonical_inputs) is not dict:
            raise TypeError("bound node inputs did not canonicalize to an exact dict")
    except (TypeError, ValueError, RecursionError):
        canonical_inputs = {}
        invalid_inputs = True

    def recall_memory(
        base_system_prompt: str,
        rendered_prompt: str,
    ) -> tuple[MemorySection, ...]:
        """Construct one complete policy context and validate its ordered result."""
        if invalid_inputs:
            raise MemoryRecallError("memory recall context is invalid")
        try:
            context = RecallContext(
                base_system_prompt,
                rendered_prompt,
                canonical_inputs,
                scope.static_path,
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise MemoryRecallError("memory recall context is invalid") from exc

        try:
            sections = memory_recall(scope, context)
        except MemoryRecallError:
            raise
        except Exception as exc:
            raise _CapabilityFault(exc) from exc
        try:
            return _validated_memory_sections(scope, sections)
        except (TypeError, ValueError, RecursionError) as exc:
            raise MemoryRecallError("memory recall returned invalid sections") from exc

    return recall_memory


def _result_contract_failure(node, message: str) -> EvaluationFailure:
    """Create one synthetic result-contract exception with a useful traceback."""
    try:
        raise RuntimeError(message)
    except RuntimeError as exc:
        return _evaluation_failure(node, "contract.node_result", exc)


def _evaluate_node_stream(
    node,
    flow,
    pool: VariablePool,
    *,
    wiring_override=None,
    stop_event=None,
    evaluation_trace: _EvaluationTraceCollector | None = None,
    memory_recall: MemoryRecall | None = None,
    binding_collector: BindingCollector | None = None,
    capture_inputs: Callable[[dict[str, Any]], None] | None = None,
    source_resolver: SourceResolver | None = None,
) -> Generator[StreamChunk, None, WorkerOutcome]:
    """Bind and run one node, yielding chunks and returning one normalized result.

    `wiring_override` is a per-run wiring dict the engine may supply for THIS node instead of
    the compiled `flow.wiring[node.id]` — used to narrow top-level START's `${input.X}` sources
    to only the passed run args (so an OMITTED input binds absent and bind_params fills START's
    declared default) WITHOUT mutating the shared compiled IR. `None` falls back to flow.wiring.

    `stop_event` is the host-owned cooperative-stop signal. Every evaluation checks it before
    running, but only a node declaring `needs_stop_event` receives it as a capability. A direct
    opted-in evaluation that omits it receives a fresh unset private event.

    `memory_recall` is a host-injected, source-neutral recall boundary. A scoped node declaring
    `needs_memory_recall` receives only a closure over its canonical bound inputs and memory
    scope; every other node remains unaware of the capability.

    Returns:
        `WorkerOutcome`:
            One ``Output``, ``Route``, ``Grow``, or frozen ``EvaluationFailure``.
            Stream chunks are yielded before that return value.

    Raises:
        `_HostStop`:
            Host-control flow is outside the node failure sum and propagates unchanged.
    """
    # Host cancellation is control flow outside the node-result sum.
    if stop_event is not None and stop_event.is_set():
        raise _HostStop(primary=False)

    # The flow owns every source. A missing trait-declared reserved key is a
    # malformed compiled graph and deliberately escapes the expected binder catch.
    if wiring_override is not None:
        node_wiring = wiring_override
    else:
        node_wiring = {} if flow is None else flow.wiring.get(node.id, {})

    per_item = node.binds_per_item
    uses_base_template = type(node).run is Node.run
    reserved_binder = None
    reserved_sources: set[str] = set()
    if binding_collector is not None:
        def reserved_binder(name, source, normalize):
            """Capture one node-owned reserved value after its normalization."""
            value = _resolve_source(
                source, pool, collector=binding_collector, name=name,
                source_resolver=source_resolver,
            )
            value = normalize(value)
            reserved_sources.add(name)
            return value

    try:
        if per_item:
            record = {}
        else:
            params = node.params if node.params is not None else []
            record = bind_params(
                params, node_wiring, pool, collector=binding_collector,
                source_resolver=source_resolver, _finish_capture=False,
            )
        reserved = node.bind_reserved(node_wiring, pool, bind=reserved_binder)
        record.update(reserved)
        if binding_collector is not None:
            for name, value in reserved.items():
                if name not in reserved_sources:
                    binding_collector._record_source(name, ValueRoute.literal(value), False)
    except BindingError as exc:
        if binding_collector is not None:
            binding_collector.invalidate()
        return _evaluation_failure(
            node,
            exc.failure_code,
            exc,
            origin=exc.origin,
        )

    if capture_inputs is not None and not uses_base_template:
        capture_inputs(record)

    recall_memory = _memory_recall_capability(node, record, memory_recall)

    if evaluation_trace is not None:
        try:
            bound_inputs = canonical_seed(record)[0]
            if type(bound_inputs) is not dict:
                raise TypeError(
                    "bound node inputs did not canonicalize to an exact dict"
                )
        except (TypeError, ValueError, RecursionError):
            if recall_memory is None:
                raise
            # The recall capability owns the semantic rejection for its complete
            # query context. Evidence remains observational and simply omits the
            # same inputs when they cannot cross its JSON boundary.
        else:
            evaluation_trace._set_bound_inputs(bound_inputs)

    # Build only the node-owned, trait-gated capabilities. Deferred MAP binding
    # raises the same binder-owned marker from inside its body and is preserved below.
    caps: dict[str, Any] = {}
    if per_item:
        params = node.params if node.params is not None else []

        def bind_item(element, index):
            """Preserve only this evaluator-owned deferred binding operation."""
            try:
                item_collector = None
                if binding_collector is not None:
                    item_collector = binding_collector.for_item(index)
                    key = node.item_source_key
                    route = (
                        binding_collector.route_for(key).project((index,))
                        if key is not None else None
                    )
                    element = ResolvedValue(element, True, route)
                return bind_params(
                    params, node_wiring, pool, item=element,
                    collector=item_collector, source_resolver=source_resolver,
                )
            except BindingError as exc:
                if binding_collector is not None:
                    binding_collector.invalidate()
                raise _BindingStageError(exc) from exc
            except Exception as exc:
                raise _DeferredBindingFault(exc) from exc

        caps["bind_item"] = bind_item
    if node.needs_stop_event:
        if stop_event is None:
            stop_event = threading.Event()
        caps["stop_event"] = stop_event
    if evaluation_trace is not None:
        caps["evaluation_trace"] = evaluation_trace
    if recall_memory is not None:
        caps["recall_memory"] = recall_memory

    try:
        outcome = (
            node._run_for_evaluator(
                record,
                _capture_inputs=capture_inputs,
                **caps,
            )
            if uses_base_template
            else node.run(record, **caps)
        )
        streamed = inspect.isgenerator(outcome)
        result = (
            (yield from node._drain_node_generator(outcome))
            if streamed
            else outcome
        )
    except _HostStop:
        raise
    except _DeferredBindingFault as wrapped:
        original = wrapped.original
        raise original.with_traceback(original.__traceback__) from None
    except _CapabilityFault as wrapped:
        original = wrapped.original
        raise original.with_traceback(original.__traceback__) from None
    except _BindingStageError as wrapped:
        original = wrapped.original
        if not uses_base_template:
            return _evaluation_failure(
                node,
                "node.execute",
                original,
                origin=node.exception_origin(original),
                traceback_text=_format_exception(original),
            )
        if binding_collector is not None:
            binding_collector.invalidate()
        return _evaluation_failure(
            node,
            original.failure_code,
            original,
            origin=original.origin,
            traceback_text=_format_exception(original),
        )
    except _AssertionStageError as wrapped:
        original = wrapped.original
        if not uses_base_template:
            return _evaluation_failure(
                node,
                "node.execute",
                original,
                origin=node.exception_origin(original),
                traceback_text=_format_exception(original),
            )
        return _evaluation_failure(
            node,
            "assertion.failed",
            original,
            origin=original.assertion.origin,
            traceback_text=_format_exception(original),
        )
    except _NodeBodyError as wrapped:
        original = wrapped.original
        return _evaluation_failure(
            node,
            "node.execute",
            original,
            origin=node.exception_origin(original),
            traceback_text=_format_exception(original),
        )
    except Exception as exc:
        if uses_base_template:
            raise
        return _evaluation_failure(
            node,
            "node.execute",
            exc,
            origin=node.exception_origin(exc),
        )

    if not isinstance(result, (Output, Route, Grow)):
        if streamed:
            return _result_contract_failure(
                node,
                f"node {node.id!r} generator did not return a NodeResult",
            )
        return _result_contract_failure(
            node,
            f"node {node.id!r} run() returned {type(result).__name__}, not a NodeResult",
        )
    if isinstance(result, Grow) and not node.is_spawner:
        # `node.kind.value` is message-only; the gate remains the kind-blind trait.
        kind_value = node.kind.value
        return _result_contract_failure(
            node,
            f"node {node.id!r} (kind {kind_value}) returned a Grow but is not a "
            "spawner (CALL/MAP/LOOP); only spawner kinds may grow the graph",
        )
    return result


def eval_node(
    node,
    flow,
    pool: VariablePool,
    *,
    wiring_override=None,
    stop_event=None,
    emit_stream: Callable[[StreamChunk], None] | None = None,
    capture_evidence: bool = False,
    memory_recall: MemoryRecall | None = None,
    capture_bindings: bool = False,
    source_resolver: SourceResolver | None = None,
) -> EvaluatedOutcome:
    """Evaluate one node into a wrapped worker result and transient stream callback.

    ``emit_stream`` is called synchronously for each chunk yielded by the node and is
    deliberately invoked outside the node-failure funnel.  If it raises, that exact host
    exception escapes; it must never become a durable node failure.  When omitted, chunks
    are discarded. ``capture_evidence`` must be an exact bool; its default leaves tracing
    dormant. ``memory_recall`` reaches only a scoped, opted-in node through a narrower
    capability. ``_HostStop`` likewise propagates unchanged.
    ``capture_bindings`` independently retains input values and expression routes,
    with ``source_resolver`` supplying paired reads when available. It snapshots
    common input/variable inventories and scope before binding. The base run
    template captures node-owned static material after defaults and before
    assertions or body; custom run overrides retain the evaluator's
    completed-binding, pre-run capture boundary.
    Capture and serialization errors propagate as host faults. Its default
    leaves these hooks dormant and does not require a memory frame.
    """
    if type(capture_evidence) is not bool:
        raise TypeError("capture_evidence must be a bool")
    if type(capture_bindings) is not bool:
        raise TypeError("capture_bindings must be a bool")
    if source_resolver is not None and not callable(source_resolver):
        raise TypeError("source_resolver must be callable or None")
    if memory_recall is not None and not callable(memory_recall):
        raise TypeError("memory_recall must be callable or None")
    evaluation_trace = _new_evaluation_trace(
        node,
        capture_evidence=capture_evidence,
    )
    binding_collector = BindingCollector() if capture_bindings else None
    backward_capture = (
        BackwardCapture(
            node.backward_input_names(),
            tuple(parameter.name for parameter in node.declared_variables()),
            node.execution_scope,
        )
        if capture_bindings else None
    )

    def capture_inputs(inputs):
        """Snapshot completed inputs and static material before the body can mutate them."""
        nonlocal backward_capture
        binding_collector.finish(inputs)
        data = node.capture_backward_context()
        if type(data) is not dict:
            raise TypeError("capture_backward_context must return a dictionary")
        captured = BackwardCapture(
            backward_capture.input_names, backward_capture.variable_names,
            backward_capture.scope, data,
        )
        canonical_seed(captured.node_data)
        backward_capture = captured

    stream = _evaluate_node_stream(
        node,
        flow,
        pool,
        wiring_override=wiring_override,
        stop_event=stop_event,
        evaluation_trace=evaluation_trace,
        memory_recall=memory_recall,
        binding_collector=binding_collector,
        capture_inputs=capture_inputs if capture_bindings else None,
        source_resolver=source_resolver,
    )
    try:
        while True:
            try:
                chunk = next(stream)
            except StopIteration as stopped:
                result = stopped.value
                break
            if emit_stream is not None:
                # Host-owned transport/callback boundary: keep this OUTSIDE the evaluator's
                # exception funnel so callback failures propagate with identity intact.
                emit_stream(chunk)
    except BaseException:
        # A callback can abort while the node's streaming generator is suspended at a
        # yield.  Close the shared evaluator explicitly so ``yield from`` propagates
        # GeneratorExit into the node and releases its resources before this boundary
        # returns the initiating exception. Cleanup must not replace that primary error.
        try:
            stream.close()
        except BaseException:
            pass
        raise
    else:
        stream.close()

    evidence = evaluation_trace._freeze() if evaluation_trace is not None else None
    bindings = binding_collector.freeze() if binding_collector is not None else None
    return EvaluatedOutcome(result, evidence, bindings, backward_capture)

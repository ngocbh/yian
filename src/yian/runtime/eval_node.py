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
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

from yian.diagnostics import Failure, FailureCode, OriginRef
from yian.events import StreamChunk
from yian.expr.lineage import ResolvedValue, ValueRoute
from yian.memory import (
    CheckStopped,
    ConversationInvocation,
    EvaluationEvidence,
    EvaluationTrace,
    MemoryContext,
    MemoryContextRequest,
    MemoryRecallError,
    MemoryScope,
    MemorySection,
    RecallContext,
    VisibleInteraction,
)
from yian.nodes.base import (
    Grow,
    Node,
    Output,
    Route,
    Steering,
    _AssertionStageError,
    _BindingStageError,
    _CapabilityFault,
    _DeferredBindingFault,
    _HostStop,
    _NodeBodyError,
)
from yian.nodes.binding import (
    BindingCapture,
    BindingCollector,
    BindingError,
    SourceResolver,
    _resolve_source,
    bind_params,
)
from yian.nodes.contracts import NodeSteps
from yian.runtime.contracts import BackwardCapture
from yian.runtime.log import canonical_seed
from yian.tools import ToolSet
from yian.typesys.pool import VariablePool
from yian.typesys.values import rewrite_digit_limit_text


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
    """Build one semantic failure with its canonical opaque origin trace.

    The message gives Python's int/str digit-limit text as yian's; the traceback
    keeps Python's text.
    """
    return EvaluationFailure(
        failure=Failure(
            code=code,
            # Python's digit-limit text names a setting yian authors cannot change.
            message=rewrite_digit_limit_text(str(exception)),
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


def _observer_sink(sink: Callable[[Any], None]) -> Callable[[Any], None]:
    """Carry a host observer fault out of a node body as a capability fault.

    The evaluator unwraps the carrier and re-raises the original exception, so a
    failing ``on_stream``/``on_event`` stays a host fault, never a node failure.
    """

    def emit(item: Any) -> None:
        try:
            sink(item)
        except (_CapabilityFault, _HostStop):
            raise
        except Exception as exc:
            raise _CapabilityFault(exc) from exc

    return emit


def require_steering(steering: Any) -> Steering | None:
    """Return a host steering inbox after checking its shape, or `None`.

    Raises:
        `TypeError`: `steering` is not `None` and lacks callable `drain`/`interrupted`.
    """
    if steering is not None and not (
        callable(getattr(steering, "drain", None))
        and callable(getattr(steering, "interrupted", None))
    ):
        raise TypeError("steering must provide drain(node_id) and interrupted(node_id), or be None")
    return steering


class _SteeringCapability:
    """Carry host inbox faults and malformed answers out of a node body as capability faults.

    The evaluator unwraps the carrier and re-raises the original exception, so a
    failing inbox stays a host fault, never a node failure.
    """

    def __init__(self, inbox: Steering) -> None:
        self._inbox = inbox

    def drain(self, node_id: str) -> list[str]:
        """Return the node's queued steer messages as a fresh list of strings."""
        messages = self._call(self._inbox.drain, node_id)
        if type(messages) is not list or any(type(message) is not str for message in messages):
            raise _CapabilityFault(TypeError("steering.drain must return a list of str"))
        return list(messages)

    def interrupted(self, node_id: str) -> bool:
        """Return whether the host interrupted the node."""
        value = self._call(self._inbox.interrupted, node_id)
        if type(value) is not bool:
            raise _CapabilityFault(TypeError("steering.interrupted must return a bool"))
        return value

    @staticmethod
    def _call(method: Callable[[str], Any], node_id: str) -> Any:
        try:
            return method(node_id)
        except (_CapabilityFault, _HostStop):
            raise
        except Exception as exc:
            raise _CapabilityFault(exc) from exc


WorkerOutcome: TypeAlias = Output | Route | Grow | EvaluationFailure
"""Closed raw worker result consumed by the append-only dispatcher."""


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
    """Validate one callback result as an ordered subsequence of the scope's memory keys."""
    if type(sections) is not tuple:
        raise TypeError("memory recall must return an exact tuple")
    next_index = 0
    for section in sections:
        if type(section) is not MemorySection:
            raise TypeError("memory recall items must be MemorySection values")
        MemorySection(section.memory, section.text)
        if not section.text:
            raise ValueError("memory recall sections must contain non-empty text")
        try:
            index = scope.memories.index(section.memory, next_index)
        except ValueError as exc:
            raise ValueError(
                "memory recall sections must follow the consumer memory stack"
            ) from exc
        next_index = index + 1
    return sections


class MemoryContextOpener(Protocol):
    """Host-injected admission of notes and an optional structured transcript."""

    def __call__(
        self, scope: MemoryScope, request: MemoryContextRequest, *, check_stopped: CheckStopped,
    ) -> AbstractContextManager[MemoryContext]:
        """Select one memory context without entering it."""
        ...


def _note_memory_cleanup(primary: BaseException, cleanup: BaseException) -> None:
    """Retain secondary cleanup evidence without replacing the initiating error."""
    if cleanup is primary:
        return
    try:
        primary.add_note(f"Memory context cleanup failed: {type(cleanup).__name__}: {cleanup}")
    except BaseException:
        pass


def _copy_cleanup_notes(source: BaseException, target: BaseException) -> None:
    """Preserve backend notes when generator close is secondary to a host error."""
    if source is target:
        return
    try:
        for note in tuple(getattr(source, "__notes__", ())):
            target.add_note(note)
    except BaseException:
        pass


@contextmanager
def _memory_context_capability(
    scope, opener, invocation, stop_event, close_error, canonical_inputs, invalid_inputs,
):
    """Bind a lazy single-entry capability to this execution's resource lifetime."""
    stack = ExitStack()
    closed = used = False
    def check_stopped():
        if stop_event.is_set():
            raise _HostStop(primary=False)

    def open_memory_context(base_system_prompt, rendered_prompt, compatibility):
        nonlocal used
        if closed:
            raise RuntimeError("memory context capability is closed")
        try:
            if used:
                raise RuntimeError("memory context capability was already entered")
            used = True
            if scope is None or not scope.memories:
                return MemoryContext()
            if invalid_inputs:
                raise MemoryRecallError("memory recall context is invalid")
            try:
                recall = RecallContext(
                    base_system_prompt, rendered_prompt, canonical_inputs, scope.static_path,
                )
                request = MemoryContextRequest(recall, invocation, compatibility)
            except (TypeError, ValueError, RecursionError) as exc:
                raise MemoryRecallError("memory recall context is invalid") from exc
            context = stack.enter_context(opener(scope, request, check_stopped=check_stopped))
            try:
                if type(context) is not MemoryContext:
                    raise TypeError("memory context must be an exact MemoryContext")
                _validated_memory_sections(scope, context.sections)
            except (TypeError, ValueError, RecursionError) as exc:
                raise MemoryRecallError("memory recall returned invalid context") from exc
            return context
        except (_HostStop, _CapabilityFault, MemoryRecallError):
            raise
        except Exception as exc:
            raise _CapabilityFault(exc) from exc

    try:
        yield open_memory_context
    except BaseException as primary:
        closed = True
        if isinstance(primary, _HostStop) and primary.primary:
            stop_event.set()
        # Backend cleanup sees the real failure, not the evaluator's stage marker,
        # so any retained cleanup notes survive later failure conversion.
        original = (
            primary.original
            if isinstance(primary, (_CapabilityFault, _NodeBodyError, _AssertionStageError,
                                    _BindingStageError, _DeferredBindingFault, _ResultContractError))
            else primary
        )
        try:
            stack.__exit__(type(original), original, original.__traceback__)
        except BaseException as cleanup:
            _note_memory_cleanup(original, cleanup)
        finally:
            if close_error:
                _copy_cleanup_notes(original, close_error[0])
        raise
    else:
        closed = True
        try:
            stack.close()
        except (_HostStop, _CapabilityFault):
            raise
        except Exception as exc:
            raise _CapabilityFault(exc) from exc
    finally:
        closed = True


class _ResultContractError(Exception):
    """Carry the result-validation stage across memory context cleanup."""

    def __init__(self, message):
        try:
            raise RuntimeError(message)
        except RuntimeError as original:
            self.original = original
        super().__init__(message)


def _evaluate_node_stream(
    node,
    flow,
    pool: VariablePool,
    *,
    wiring_override=None,
    stop_event=None,
    evaluation_trace: _EvaluationTraceCollector | None = None,
    memory_context_opener: MemoryContextOpener | None = None,
    memory_invocation: ConversationInvocation | None = None,
    memory_scope: MemoryScope | None = None,
    skills: Mapping[str, Any] | None = None,
    close_error: list[BaseException] | None = None,
    binding_collector: BindingCollector | None = None,
    capture_inputs: Callable[[dict[str, Any]], None] | None = None,
    capture_render: Callable[[dict[str, Any]], None] | None = None,
    source_resolver: SourceResolver | None = None,
    tools: ToolSet | None = None,
    live_stream: Callable[[StreamChunk], None] | None = None,
    live_event: Callable[[Any], None] | None = None,
    steering: Steering | None = None,
    steps: NodeSteps | None = None,
) -> Generator[StreamChunk, None, WorkerOutcome]:
    """Bind and run one node, yielding chunks and returning one normalized result.

    `wiring_override` is a per-run wiring dict the engine may supply for THIS node instead of
    the compiled `flow.wiring[node.id]` — used to narrow top-level START's `${input.X}` sources
    to only the passed run args (so an OMITTED input binds absent and bind_params fills START's
    declared default) WITHOUT mutating the shared compiled IR. `None` falls back to flow.wiring.

    `stop_event` is the host-owned cooperative-stop signal. Every evaluation checks it before
    running, but only a node declaring `needs_stop_event` receives it as a capability. A direct
    opted-in evaluation that omits it receives a fresh unset private event.

    `memory_context_opener` is a source-neutral context boundary. A node declaring
    `needs_memory_context` receives only a closure over its canonical bound inputs
    and memory scope; other nodes remain unaware of the capability.

    `skills` is the run's host-supplied skill mapping, passed as ``caps['skills']`` only to
    a node declaring `needs_skills`.

    `steps` is this start's ``NodeSteps``, passed as ``caps['steps']`` only to a node
    declaring `needs_steps`.

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
                source_resolver=source_resolver, types=node.types,
                labels=node.binding_labels, _finish_capture=False,
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

    canonical_inputs = {}
    invalid_inputs = False
    if memory_scope is not None and memory_scope.memories:
        if memory_context_opener is None:
            raise RuntimeError(
                f"scoped node {node.id!r} requires an injected memory context callback"
            )
        try:
            canonical_inputs = canonical_seed(record)[0]
            if type(canonical_inputs) is not dict:
                raise TypeError("bound node inputs did not canonicalize to an exact dict")
        except (TypeError, ValueError, RecursionError):
            invalid_inputs = True

    if evaluation_trace is not None:
        try:
            bound_inputs = canonical_seed(record)[0]
            if type(bound_inputs) is not dict:
                raise TypeError(
                    "bound node inputs did not canonicalize to an exact dict"
                )
        except (TypeError, ValueError, RecursionError):
            if memory_scope is None or not memory_scope.memories:
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
                    types=node.types, labels=node.binding_labels,
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
    if capture_render is not None:
        caps["capture_render"] = capture_render
    if node.needs_tools:
        caps["tools"] = tools if tools is not None else ToolSet()
    if live_stream is not None:
        caps["emit_stream"] = live_stream
    if live_event is not None:
        caps["emit_event"] = live_event
    if skills is not None and node.needs_skills is True:
        caps["skills"] = skills
    if steering is not None and node.needs_steering is True:
        caps["steering"] = _SteeringCapability(steering)
    if steps is not None and node.needs_steps is True:
        caps["steps"] = steps

    try:
        with _memory_context_capability(
            memory_scope, memory_context_opener, memory_invocation, stop_event, close_error,
            canonical_inputs, invalid_inputs,
        ) as open_context:
            if node.needs_memory_context is True:
                caps["open_memory_context"] = open_context
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
            result = (yield from node._drain_node_generator(outcome)) if streamed else outcome
            if not isinstance(result, (Output, Route, Grow)):
                message = (
                    f"node {node.id!r} generator did not return a NodeResult" if streamed
                    else f"node {node.id!r} run() returned {type(result).__name__}, not a NodeResult"
                )
                raise _ResultContractError(message)
            if isinstance(result, Grow) and not node.is_spawner:
                # Kind is diagnostic text only; the spawner trait owns this gate.
                raise _ResultContractError(
                    f"node {node.id!r} (kind {node.kind.value}) returned a Grow but is not a "
                    "spawner (CALL/MAP/LOOP); only spawner kinds may grow the graph"
                )
    except _HostStop:
        raise
    except _DeferredBindingFault as wrapped:
        original = wrapped.original
        raise original.with_traceback(original.__traceback__) from None
    except _CapabilityFault as wrapped:
        original = wrapped.original
        raise original.with_traceback(original.__traceback__) from None
    except _ResultContractError as wrapped:
        original = wrapped.original
        return _evaluation_failure(
            node, "contract.node_result", original,
            traceback_text=_format_exception(original),
        )
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

    return result


def eval_node(
    node,
    flow,
    pool: VariablePool,
    *,
    wiring_override=None,
    stop_event=None,
    emit_stream: Callable[[StreamChunk], None] | None = None,
    emit_event: Callable[[Any], None] | None = None,
    capture_evidence: bool = False,
    memory_context_opener: MemoryContextOpener | None = None,
    memory_invocation: ConversationInvocation | None = None,
    skills: Mapping[str, Any] | None = None,
    capture_bindings: bool = False,
    source_resolver: SourceResolver | None = None,
    effective_llm_by_static_path: dict[str, dict[str, Any]] | None = None,
    tools: ToolSet | None = None,
    steering: Steering | None = None,
    steps: NodeSteps | None = None,
) -> EvaluatedOutcome:
    """Evaluate one node into a wrapped worker result and transient stream callback.

    ``emit_stream`` is called synchronously for each chunk yielded by the node and is
    deliberately invoked outside the node-failure funnel.  If it raises, that exact host
    exception escapes; it must never become a durable node failure.  When omitted, chunks
    are discarded. A node declaring ``needs_live_events`` also receives in-body
    ``emit_stream``/``emit_event`` capabilities for whichever of these host sinks
    were supplied; their exceptions likewise escape as host faults, never node
    failures. ``emit_event`` receives only transient in-loop observers.
    ``capture_evidence`` must be an exact bool; its default leaves tracing
    dormant. ``memory_context_opener`` reaches only an opted-in node through a
    narrow ``open_memory_context(base_system_prompt, rendered_prompt, compatibility)``
    capability over canonical bound inputs and the captured memory scope.
    Opening is lazy and single-use. An unscoped node receives an empty context;
    a scoped node requires an injected opener. ``memory_invocation`` is the
    accepted node-start identity, required only for transcript admission.
    ``skills`` is the run's host-supplied skill mapping, delivered only to nodes
    declaring ``needs_skills``; a non-mapping raises ``TypeError``.
    The evaluator owns context cleanup through execution, generator draining,
    postassertions, and result validation. Typed recall failures remain node
    failures; unexpected opening and cleanup faults remain host errors.
    ``_HostStop`` propagates unchanged.
    ``capture_bindings`` independently retains input values and expression routes,
    with ``source_resolver`` supplying paired reads when available. It snapshots
    common input/variable inventories and scope before binding. The base run
    template captures node-owned static material after defaults and before
    assertions or body; custom run overrides retain the evaluator's
    completed-binding, pre-run capture boundary.
    Retained nodes opting into render capture receive a single-write local sink;
    it freezes their actual render data and the selected recorded model config.
    ``effective_llm_by_static_path`` supplies accepted RunStarted configuration,
    selected only by the captured root-relative static scope, never node IDs.
    Capture and serialization errors propagate as host faults. Its default
    leaves these hooks dormant and does not require a memory scope.
    ``tools`` is the run's exact ``ToolSet``; only a node declaring ``needs_tools``
    receives it. ``None`` gives such a node the global registry.
    ``steering`` is the run's host inbox, delivered only to nodes declaring
    ``needs_steering``; its exceptions and malformed answers escape as host faults.
    ``steps`` is this start's ``NodeSteps`` from the engine, delivered only to nodes
    declaring ``needs_steps``; a non-``NodeSteps`` value raises ``TypeError``. Its
    recorder raises its faults as capability faults, so they escape as host faults.
    """
    require_steering(steering)
    if type(capture_evidence) is not bool:
        raise TypeError("capture_evidence must be a bool")
    if type(capture_bindings) is not bool:
        raise TypeError("capture_bindings must be a bool")
    if emit_stream is not None and not callable(emit_stream):
        raise TypeError("emit_stream must be callable or None")
    if emit_event is not None and not callable(emit_event):
        raise TypeError("emit_event must be callable or None")
    if source_resolver is not None and not callable(source_resolver):
        raise TypeError("source_resolver must be callable or None")
    if memory_context_opener is not None and not callable(memory_context_opener):
        raise TypeError("memory_context_opener must be callable or None")
    if tools is not None and type(tools) is not ToolSet:
        raise TypeError("tools must be a ToolSet or None")
    if skills is not None and not isinstance(skills, Mapping):
        raise TypeError("skills must be a mapping or None")
    if steps is not None and type(steps) is not NodeSteps:
        raise TypeError("steps must be a NodeSteps or None")
    if memory_invocation is not None:
        if type(memory_invocation) is not ConversationInvocation:
            raise TypeError("memory_invocation must be a ConversationInvocation or None")
        memory_invocation = ConversationInvocation(
            memory_invocation.run_id, memory_invocation.start_sequence,
        )
    if stop_event is None:
        stop_event = threading.Event()
    memory_scope = None
    if node.needs_memory_context is True and node.memory_scope is not None:
        if type(node.memory_scope) is not MemoryScope:
            raise TypeError("node memory_scope must be an exact MemoryScope or None")
        memory_scope = deepcopy(node.memory_scope)
        MemoryScope(
            memory_scope.static_path, memory_scope.memories, memory_scope.occurrence_path,
        )
    if effective_llm_by_static_path is not None and type(effective_llm_by_static_path) is not dict:
        raise TypeError("effective_llm_by_static_path must be a dict or None")
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
    render_enabled = capture_bindings and node.needs_render_capture is True
    render_closed = False
    render_recorded = False
    effective_config = None
    if render_enabled and backward_capture.scope is not None and effective_llm_by_static_path is not None:
        key = "/".join(backward_capture.scope.static_path)
        if key not in effective_llm_by_static_path:
            raise ValueError(f"recorded effective model configuration is missing static path {key!r}")
        config = effective_llm_by_static_path[key]
        if type(config) is not dict:
            raise TypeError("recorded effective model configuration must be a dictionary")
        effective_config = canonical_seed(config)[1]

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

    def capture_render(data):
        """Freeze one actual render publication independently from visible evidence."""
        nonlocal backward_capture, render_recorded
        if render_closed:
            raise RuntimeError("render capture is already closed")
        try:
            if render_recorded:
                raise RuntimeError("render capture was already recorded")
            if type(data) is not dict:
                raise TypeError("render capture must be a dictionary")
            if "effective_llm_config" in data:
                raise ValueError("effective_llm_config belongs to recorded runtime authority")
            snapshot = canonical_seed(data)[1]
            static_data = backward_capture.node_data
            if static_data.keys() & snapshot.keys():
                raise ValueError("render capture cannot replace static node data")
            backward_capture = BackwardCapture(
                backward_capture.input_names, backward_capture.variable_names, backward_capture.scope,
                {**static_data, **snapshot, "effective_llm_config": effective_config},
            )
            render_recorded = True
        except (_CapabilityFault, _HostStop):
            raise
        except Exception as exc:
            raise _CapabilityFault(exc) from exc

    live_stream = live_event = None
    if node.needs_live_events is True:
        live_stream = None if emit_stream is None else _observer_sink(emit_stream)
        live_event = None if emit_event is None else _observer_sink(emit_event)

    # A callback failure remains outside the user generator. This transient reference
    # routes cleanup notes back to it while close() still delivers GeneratorExit.
    close_error: list[BaseException] = []
    stream = _evaluate_node_stream(
        node,
        flow,
        pool,
        wiring_override=wiring_override,
        stop_event=stop_event,
        evaluation_trace=evaluation_trace,
        memory_context_opener=memory_context_opener,
        memory_invocation=memory_invocation,
        memory_scope=memory_scope,
        skills=skills,
        close_error=close_error,
        tools=tools,
        binding_collector=binding_collector,
        capture_inputs=capture_inputs if capture_bindings else None,
        capture_render=capture_render if render_enabled else None,
        source_resolver=source_resolver,
        live_stream=live_stream,
        live_event=live_event,
        steering=steering,
        steps=steps,
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
    except BaseException as primary:
        # A callback can abort while the node's streaming generator is suspended at a
        # yield.  Close the shared evaluator explicitly so ``yield from`` propagates
        # GeneratorExit into the node and releases its resources before this boundary
        # returns the initiating exception. Cleanup must not replace that primary error.
        if isinstance(primary, _HostStop) and primary.primary:
            stop_event.set()
        close_error.append(primary)
        try:
            stream.close()
        except BaseException:
            pass
        raise
    else:
        stream.close()
    finally:
        render_closed = True

    evidence = evaluation_trace._freeze() if evaluation_trace is not None else None
    bindings = binding_collector.freeze() if binding_collector is not None else None
    return EvaluatedOutcome(result, evidence, bindings, backward_capture)

"""Node evaluation at the engine's pure read/execution boundary.

A node is a pure function of its bound input record; the ENGINE owns both boundaries —
the read (bind the inputs from the pool) and the write (the dispatcher stores `Output`).
``eval_node`` is the append-only dispatcher's evaluation seam: it binds the record,
pre-resolves reserved keys, builds trait-gated capabilities, runs the node through its
assert-owning ``run`` wrapper, forwards transient chunks through ``emit_stream``, and
returns exactly ``Output | Route | Grow | EvaluationFailure``.  It emits no observer
notifications; durable observer ordering belongs to the dispatcher.

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
from agent_composer.nodes.base import (
    Grow,
    Node,
    Output,
    Route,
    _AssertionStageError,
    _BindingStageError,
    _DeferredBindingFault,
    _HostStop,
    _NodeBodyError,
)
from agent_composer.nodes.binding import BindingError, bind_params
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
) -> Generator[StreamChunk, None, WorkerOutcome]:
    """Bind and run one node, yielding chunks and returning one normalized result.

    `wiring_override` is a per-run wiring dict the engine may supply for THIS node instead of
    the compiled `flow.wiring[node.id]` — used to narrow top-level START's `${input.X}` sources
    to only the passed run args (so an OMITTED input binds absent and bind_params fills START's
    declared default) WITHOUT mutating the shared compiled IR. `None` falls back to flow.wiring.

    `stop_event` is the host-owned cooperative-stop signal. Every evaluation checks it before
    running, but only a node declaring `needs_stop_event` receives it as a capability. A direct
    opted-in evaluation that omits it receives a fresh unset private event.

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
    try:
        if per_item:
            record = {}
        else:
            params = node.params if node.params is not None else []
            record = bind_params(params, node_wiring, pool)
        record.update(node.bind_reserved(node_wiring, pool))
    except BindingError as exc:
        return _evaluation_failure(
            node,
            exc.failure_code,
            exc,
            origin=exc.origin,
        )

    # Build only the node-owned, trait-gated capabilities. Deferred MAP binding
    # raises the same binder-owned marker from inside its body and is preserved below.
    caps: dict[str, Any] = {}
    if per_item:
        params = node.params if node.params is not None else []

        def bind_item(element):
            """Preserve only this evaluator-owned deferred binding operation."""
            try:
                return bind_params(params, node_wiring, pool, item=element)
            except BindingError as exc:
                raise _BindingStageError(exc) from exc
            except Exception as exc:
                raise _DeferredBindingFault(exc) from exc

        caps["bind_item"] = bind_item
    if node.needs_stop_event:
        if stop_event is None:
            stop_event = threading.Event()
        caps["stop_event"] = stop_event

    uses_base_template = type(node).run is Node.run
    try:
        outcome = (
            node._run_for_evaluator(record, **caps)
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
) -> WorkerOutcome:
    """Evaluate one node into a raw worker result and transient stream callback.

    ``emit_stream`` is called synchronously for each chunk yielded by the node and is
    deliberately invoked outside the node-failure funnel.  If it raises, that exact host
    exception escapes; it must never become a durable node failure.  When omitted, chunks
    are discarded.  ``_HostStop`` likewise propagates unchanged.
    """
    stream = _evaluate_node_stream(
        node,
        flow,
        pool,
        wiring_override=wiring_override,
        stop_event=stop_event,
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

    return result

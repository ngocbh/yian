"""Node evaluation at the engine's pure read/execution boundary.

A node is a pure function of its bound input record; the ENGINE owns both boundaries —
the read (bind the inputs from the pool) and the write (the dispatcher stores `Output`).
``eval_node`` is the append-only dispatcher's evaluation seam: it binds the record,
pre-resolves reserved keys, builds trait-gated capabilities, runs the node through its
assert-owning ``run`` wrapper, forwards transient chunks through ``emit_stream``, and
returns exactly ``Output | Route | Grow | EvaluationFailure``.  It emits no observer
notifications; durable observer ordering belongs to the dispatcher.

Every node-side failure path funnels to ``EvaluationFailure``: a node ``raise``, a
``Grow`` returned by a non-spawner, and a non-``NodeResult`` return.  ``_HostStop`` and
an injected stream callback exception are deliberate host-side exceptions and propagate
unchanged.

The bind is PURE: it reads the node's sources ONLY from `flow.wiring[node.id]` joined to the
node's `params` — the node carries no source. Direct-drive tests supply a stub `flow` with
`wiring` (the test helpers derive it). Layer: runtime imports nodes.* freely.

Accepted bind ordering vs the old per-node `_run` on an unreachable loaded-flow state: a
`call` that is both unbaked and has a bad `over`/binding surfaces the bind error before the
not-baked guard; the loader always bakes calls, and either path terminates in `NodeFailed`.
"""

from __future__ import annotations

import inspect
import threading
import traceback as _tb
from collections.abc import Callable, Generator
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

from agent_composer.diagnostics import Failure, FailureCode, OriginRef
from agent_composer.events import SourceRange, SourceSpan, StreamChunk
from agent_composer.nodes.base import (
    Grow,
    NodeAssertFailed,
    Output,
    Route,
    _HostStop,
    _NodeBodyError,
)
from agent_composer.nodes.binding import BindingError, bind_params
from agent_composer.typesys.pool import VariablePool


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    """One raw node-evaluation failure for durable dispatcher preparation.

    ``failure`` is the sole semantic authority. ``locator`` and ``range`` retain
    only the legacy presentation bridge until compose resolves opaque origins.
    """

    failure: Failure
    locator: SourceSpan | None = None
    range: SourceRange | None = None

    @property
    def node_id(self) -> str | None:
        """Return the temporary source-compatible execution identity view."""
        return self.failure.node_id

    @property
    def error(self) -> str:
        """Return the temporary source-compatible message view."""
        return self.failure.message

    @property
    def error_type(self) -> str:
        """Return the temporary source-compatible exception-name view."""
        return self.failure.cause_type or ""

    @property
    def traceback(self) -> str | None:
        """Return the temporary source-compatible traceback view."""
        return self.failure.traceback


def _failure_trace(node, origin: OriginRef | None) -> tuple[OriginRef, ...]:
    """Return the available operation origin, then the authored node fallback."""
    selected = origin if origin is not None else node.origin
    return () if selected is None else (selected,)


def _evaluation_failure(
    node,
    code: FailureCode,
    exception: Exception,
    *,
    origin: OriginRef | None = None,
    traceback_text: str | None = None,
) -> EvaluationFailure:
    """Build one semantic failure plus the temporary range presentation bridge."""
    locator = getattr(exception, "locator", None)
    if locator is not None and locator.node is None:
        locator = replace(locator, node=node.id)
    range_ = getattr(exception, "range", None)
    if range_ is None and locator is not None:
        range_ = node.field_ranges.get((locator.kind, locator.key))
    return EvaluationFailure(
        failure=Failure(
            code=code,
            message=str(exception),
            node_id=node.id,
            trace=_failure_trace(node, origin),
            cause_type=type(exception).__name__,
            traceback=(traceback_text if traceback_text is not None else _tb.format_exc()),
        ),
        locator=locator,
        range=range_,
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
        caps["bind_item"] = lambda element: bind_params(
            params,
            node_wiring,
            pool,
            item=element,
        )
    if node.needs_stop_event:
        if stop_event is None:
            stop_event = threading.Event()
        caps["stop_event"] = stop_event

    try:
        outcome = node.run(record, **caps)
        streamed = inspect.isgenerator(outcome)
        result = (
            (yield from node._drain_node_generator(outcome))
            if streamed
            else outcome
        )
    except _HostStop:
        raise
    except BindingError as exc:
        return _evaluation_failure(
            node,
            exc.failure_code,
            exc,
            origin=exc.origin,
        )
    except NodeAssertFailed as exc:
        if exc.assertion is None:
            return _evaluation_failure(node, "node.execute", exc)
        return _evaluation_failure(
            node,
            "assertion.failed",
            exc,
            origin=exc.assertion.origin,
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

    try:
        if not isinstance(result, (Output, Route, Grow)):
            if streamed:
                raise RuntimeError(
                    f"node {node.id!r} generator did not return a NodeResult"
                )
            raise RuntimeError(
                f"node {node.id!r} run() returned {type(result).__name__}, "
                "not a NodeResult"
            )
        if isinstance(result, Grow) and not node.is_spawner:
            # `node.kind.value` is message-only; the gate remains the kind-blind trait.
            raise RuntimeError(
                f"node {node.id!r} (kind {node.kind.value}) returned a Grow but is "
                "not a spawner (CALL/MAP/LOOP); only spawner kinds may grow the graph"
            )
    except Exception as exc:  # synthetic result-contract boundary
        return _evaluation_failure(node, "contract.node_result", exc)
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
    finally:
        # A callback can abort while the node's streaming generator is suspended at a
        # yield.  Close the shared evaluator explicitly so ``yield from`` propagates
        # GeneratorExit into the node and releases its resources before this boundary
        # returns the initiating host exception.
        stream.close()

    return result

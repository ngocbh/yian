"""One effect-free reverse walk over accepted forward occurrences.

The fold consumes recorded routes and node-owned backward hooks. It does not
interpret kinds, evaluate expressions, drive child flows, or write logs/memory.
Small durable attribution references live in runtime.contracts below this module;
materialized interpretation data and ordering keys exist only during this fold.
"""

from __future__ import annotations

from typing import Any
import traceback

from agent_composer.expr.lineage import (
    FeedbackValue, RouteContext, SourceRef,
)
from agent_composer.nodes.base import (
    BackwardContext, BackwardResult, _CapabilityFault, _HostStop,
    _copy_backward_value, validate_backward_result,
)
from agent_composer.runtime.contracts import (
    AttributedBackwardResult, BackwardFailure,
)
from agent_composer.runtime.log import _route_payload
from agent_composer.runtime.trace import RetainedTrace
from agent_composer.runtime.learning import (
    _BackwardMessage as _Message, _project_root_backward, _seed_backward, _transfer_backward,
)


class _BackwardComputeError(Exception):
    """Attribute a callback or result-contract failure without changing forward state."""

    def __init__(self, failure: BackwardFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


def _source_data(source: SourceRef | None) -> dict[str, Any] | None:
    if source is None:
        return None
    return {"run_id": source.run_id, "execution_seq": source.execution_seq, "path": list(source.path)}


def _materialize_context(trace: RetainedTrace, message: _Message, index: int) -> dict[str, Any]:
    """Resolve only the accepted values needed by one transient callback view."""
    details = []
    for context in message.context:
        sources = [{"source": _source_data(source), "value": trace.lookup_source(source)}
                   for source in context.sources]
        result = context.result
        if context.result_json is None and context.result_type is None and len(sources) == 1:
            result = sources[0]["value"]
        details.append({
            "expression": context.expression, "result": result, "sources": sources,
            "role": context.role,
            "operand": list(context.operand) if type(context.operand) is tuple else context.operand,
            "positions": [list(position) for position in context.positions],
            "span": list(context.span) if context.span is not None else None,
            "result_type": context.result_type,
            "snapshot": _route_payload(context.snapshot) if context.snapshot is not None else None,
        })
    attribution = message.attribution
    return {"message_index": index, "sender": _source_data(attribution.sender),
            "input_name": attribution.input_name, "sender_message_index": attribution.message_index,
            "contribution_index": attribution.contribution_index, "parents": list(attribution.parents),
            "contexts": details}


def _failure(context: BackwardContext, code: str, exc: Exception) -> _BackwardComputeError:
    return _BackwardComputeError(BackwardFailure(
        stage="backward", code=code, message=str(exc), scope=context.scope,
        run_id=context.run_id, node_id=context.node_id, node_started_seq=context.node_started_seq,
        cause_type=type(exc).__name__, traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    ))


def _output_context(
    source: SourceRef, context: BackwardContext, name: str, has_output: bool,
) -> tuple[RouteContext, ...]:
    """Reference an accepted structural output without rebuilding or copying it."""
    if not has_output:
        return ()
    path = context.node_data.get("input_output_paths", {}).get(name)
    if path is None:
        return ()
    value = context.output
    for part in path:
        value = value[part]
    positions = ((path[0],),) if path and type(path[0]) is int else ()
    return (RouteContext("", None, (source,), "output", tuple(path), positions),)


def compute_backward(
    trace: RetainedTrace, root_feedback: FeedbackValue,
) -> tuple[BackwardResult, tuple[AttributedBackwardResult, ...]]:
    """Project root feedback through one accepted-outcome reverse walk.

    Args:
        trace (`RetainedTrace`): A retained terminal execution with private
            backward callbacks.
        root_feedback (`FeedbackValue`): One ordinary text, record, positional
            list, or None value selecting the actual terminal output. Failed
            terminals accept text or None.

    Returns:
        `tuple[BackwardResult, tuple[AttributedBackwardResult, ...]]`: Root input
            feedback and results in accepted-outcome order, including unreached
            occurrences whose callbacks were not invoked.

    Raises:
        `TypeError`: For malformed feedback or trace carriers.
        `ValueError`: For an unfinished trace, invalid feedback shape, or broken
            retained route/source authority.
        `_BackwardComputeError`: For an attributed callback or result-contract error.
            Host capability faults and cancellation propagate unchanged.
    """
    pending, direct_root = _seed_backward(trace, root_feedback)
    occurrences = trace.occurrences
    results = {item.source: AttributedBackwardResult(item.source, (), (), BackwardResult(), False)
               for item in occurrences}
    root_inputs: dict[str, tuple[FeedbackValue, ...]] = {}
    for occurrence in reversed(occurrences):
        messages = sorted(pending.pop(occurrence.source, ()), key=lambda item: item.order)
        if not messages:
            continue
        context = trace.backward_context(
            occurrence.source,
            downstream_context=tuple(_materialize_context(trace, message, i) for i, message in enumerate(messages)),
        )
        received = tuple(item.feedback for item in messages)
        feedback = tuple(_copy_backward_value(message, "feedback") for message in received)
        handler = trace.backward_handlers[occurrence.source]
        try:
            result = handler(context, feedback)
        except (_HostStop, MemoryError, RecursionError):
            raise
        except _CapabilityFault as fault:
            raise fault.original.with_traceback(fault.original.__traceback__) from None
        except Exception as exc:
            raise _failure(context, "backward.execute", exc) from exc
        try:
            validate_backward_result(result, context)
        except (TypeError, ValueError) as exc:
            raise _failure(context, "backward.result", exc) from exc
        results[occurrence.source] = AttributedBackwardResult(
            occurrence.source, received, tuple(item.attribution for item in messages), result, True,
        )
        input_feedback = result.input_feedback
        if occurrence.source == trace.root_start_source:
            root_inputs = input_feedback
        has_output = occurrence.source in trace.sources
        for source, transfer in _transfer_backward(trace, occurrence, context, messages, result):
            attribution = transfer.attribution
            inherited = tuple(detail for index in attribution.parents for detail in messages[index].context)
            output_context = _output_context(occurrence.source, context, attribution.input_name, has_output)
            pending.setdefault(source, []).append(_Message(
                transfer.feedback, attribution, transfer.order,
                (*inherited, *output_context, *transfer.context),
            ))
    root = _project_root_backward(trace, pending, root_inputs, direct_root)
    return root, tuple(results[item.source] for item in occurrences)


__all__ = ["compute_backward"]

"""One effect-free reverse walk over accepted forward occurrences.

The fold consumes recorded routes and node-owned backward hooks. It does not
interpret kinds, evaluate expressions, drive child flows, or write logs/memory.
Small durable attribution references live in runtime.contracts below this module;
materialized interpretation data and ordering keys exist only during this fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import traceback

from agent_composer.expr.lineage import (
    FeedbackValue, RouteContext, SourceRef, has_feedback, route_feedback,
    validate_feedback,
)
from agent_composer.nodes.base import (
    BackwardContext, BackwardResult, _CapabilityFault, _HostStop,
    _copy_backward_value, validate_backward_result,
)
from agent_composer.runtime.contracts import (
    AttributedBackwardResult, BackwardAttribution, BackwardFailure,
)
from agent_composer.runtime.log import _route_payload
from agent_composer.runtime.trace import RetainedTrace, TraceTerminal


class _BackwardComputeError(Exception):
    """Attribute a callback or result-contract failure without changing forward state."""

    def __init__(self, failure: BackwardFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


@dataclass(frozen=True)
class _Message:
    feedback: FeedbackValue
    attribution: BackwardAttribution
    order: tuple[int, ...]
    context: tuple[RouteContext, ...] = ()


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


def _structural_parents(
    context: BackwardContext, messages: list[_Message], result: BackwardResult,
    has_output: bool,
) -> dict[str, tuple[int, ...]] | None:
    """Verify a metadata-prescribed transfer before assigning incoming ordinals."""
    paths = context.node_data.get("input_output_paths")
    inputs = context.inputs
    if paths is None or inputs is None or (not has_output and context.failure is None):
        return None
    expected = {}
    parents = {}
    for name in context.input_names:
        if name not in paths or name not in inputs:
            continue
        if has_output:
            value = context.output
            for part in paths[name]:
                value = value[part]
        selected = []
        indices = []
        for index, message in enumerate(messages):
            feedback = message.feedback
            for part in paths[name]:
                if feedback is None or type(feedback) is str:
                    break
                feedback = feedback.get(part) if type(feedback) is dict else feedback[part]
            if has_feedback(feedback):
                selected.append(feedback)
                indices.append(index)
        if selected:
            expected[name] = tuple(selected)
            parents[name] = tuple(indices)
    return parents if result.input_feedback == expected else None


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
    if type(trace) is not RetainedTrace or type(trace.terminal) is not TraceTerminal:
        raise ValueError("backward requires a retained terminal trace")
    terminal = trace.terminal
    TraceTerminal.__post_init__(terminal)
    validate_feedback(root_feedback, terminal.output)
    root_feedback = _copy_backward_value(root_feedback, "root_feedback")
    occurrences = trace.occurrences
    results = {item.source: AttributedBackwardResult(item.source, (), (), BackwardResult(), False)
               for item in occurrences}
    if not has_feedback(root_feedback):
        return BackwardResult(), tuple(results[item.source] for item in occurrences)
    if terminal.source is None:
        root = BackwardResult(self_feedback=(root_feedback,)) if not terminal.succeeded else BackwardResult()
        return root, tuple(results[item.source] for item in occurrences)
    pending = {terminal.source: [_Message(root_feedback, BackwardAttribution(None, None, 0, None, ()), (0,))]}
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
        structural_parents = _structural_parents(
            context, messages, result, has_output,
        )
        routes = dict(zip(occurrence.input_route.names, occurrence.input_route.children))
        for input_index, name in enumerate(context.input_names):
            for message_index, feedback_value in enumerate(input_feedback.get(name, ())):
                parents = (structural_parents[name][message_index],) if structural_parents is not None else tuple(range(len(messages)))
                inherited = tuple(detail for index in parents for detail in messages[index].context)
                output_context = _output_context(occurrence.source, context, name, has_output)
                routed = route_feedback(
                    routes[name], feedback_value, trace.lookup_source,
                    recorded_value=context.inputs[name],
                )
                for contribution_index, contribution in enumerate(routed.contributions):
                    attribution = BackwardAttribution(
                        occurrence.source, name, message_index, contribution_index, parents,
                    )
                    order = (*messages[parents[0]].order, input_index, message_index, contribution_index)
                    pending.setdefault(contribution.source, []).append(_Message(
                        contribution.feedback, attribution, order, (*inherited, *output_context, *contribution.context),
                    ))
    source_zero = SourceRef(trace.run_id, 0)
    initial = trace.lookup_source(source_zero)
    for message in sorted(pending.pop(source_zero, ()), key=lambda item: item.order):
        if message.attribution.sender == trace.root_start_source and trace.root_start_source is not None:
            continue
        for name in trace.root_input_names:
            if name not in initial:
                continue
            selected = message.feedback if type(message.feedback) is str else message.feedback.get(name)
            if has_feedback(selected):
                root_inputs[name] = (*root_inputs.get(name, ()), selected)
    if pending:
        raise ValueError("feedback targets an unavailable or later accepted occurrence")
    root = BackwardResult(root_inputs)
    return root, tuple(results[item.source] for item in occurrences)


__all__ = ["compute_backward"]

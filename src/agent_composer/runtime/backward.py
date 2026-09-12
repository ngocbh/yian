"""One effect-free reverse walk over accepted forward occurrences.

The fold consumes recorded routes and node-owned backward hooks. It does not
interpret kinds, evaluate expressions, drive child flows, or write logs/memory.
Small durable attribution references live in runtime.contracts below this module;
materialized interpretation data and ordering keys remain transient. The shared
learning projection owns formatting without performing this callback fold.
"""

from __future__ import annotations

import traceback

from agent_composer.expr.lineage import FeedbackValue
from agent_composer.nodes.base import (
    BackwardContext, BackwardResult, _CapabilityFault, _HostStop,
    _copy_backward_value, validate_backward_result,
)
from agent_composer.runtime.contracts import (
    AttributedBackwardResult, BackwardFailure,
)
from agent_composer.runtime.trace import RetainedTrace
from agent_composer.runtime.learning import (
    _contextual_transfers, _materialize_context, _project_root_backward, _seed_backward,
)


class _BackwardComputeError(Exception):
    """Attribute a callback or result-contract failure without changing forward state."""

    def __init__(self, failure: BackwardFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


def _failure(context: BackwardContext, code: str, exc: Exception) -> _BackwardComputeError:
    return _BackwardComputeError(BackwardFailure(
        stage="backward", code=code, message=str(exc), scope=context.scope,
        run_id=context.run_id, node_id=context.node_id, node_started_seq=context.node_started_seq,
        cause_type=type(exc).__name__, traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    ))


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
        for source, transfer in _contextual_transfers(trace, occurrence, context, messages, result):
            pending.setdefault(source, []).append(transfer)
    root = _project_root_backward(trace, pending, root_inputs, direct_root)
    return root, tuple(results[item.source] for item in occurrences)


__all__ = ["compute_backward"]

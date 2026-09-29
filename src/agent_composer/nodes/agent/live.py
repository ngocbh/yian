"""Transient AGENT observers — model/tool call brackets and provider token usage.

Everything here is callback-only: events go to the evaluator-supplied sink and
nowhere else. Nothing is recorded, retained in memory, or replayed on resume,
and the node value is identical with or without a sink.

Sink exceptions arrive already wrapped as `_CapabilityFault` by the evaluator and
propagate unchanged; provider and tool exceptions are reported by type name only,
because their text can contain credentials, then re-raised.

Layer: nodes — imports events and nodes.base.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Optional

from agent_composer.events import (
    ModelCallFinished,
    ModelCallStarted,
    TokenUsage,
    ToolCallFinished,
    ToolCallStarted,
)
from agent_composer.nodes.base import _CapabilityFault, _HostStop


@dataclass(frozen=True)
class LiveSinks:
    """The transient sink of one AGENT invocation plus its effective model identity."""

    node_id: str
    provider: Optional[str]
    model: Optional[str]
    emit_event: Callable[[Any], None]


def live_sinks(ctx: Any) -> Optional[LiveSinks]:
    """Select the event sink of one mode context; `None` when there is none."""
    emit_event = getattr(ctx, "emit_event", None)
    if emit_event is None:
        return None
    config = ctx.llm_config or {}
    read = config.get if isinstance(config, dict) else (lambda key: getattr(config, key, None))
    provider, model = read("provider"), read("model")
    return LiveSinks(
        ctx.node_id,
        provider if isinstance(provider, str) else None,
        model if isinstance(model, str) else None,
        emit_event,
    )


def _count(value: Any) -> Optional[int]:
    return value if type(value) is int else None


def usage_of(message: Any) -> Optional[TokenUsage]:
    """Project a reply's provider-reported `usage_metadata`; `None` when absent."""
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return None
    counts = [_count(usage.get(key)) for key in ("input_tokens", "output_tokens", "total_tokens")]
    if any(count is None for count in counts):
        return None
    inputs = usage.get("input_token_details")
    outputs = usage.get("output_token_details")
    inputs = inputs if isinstance(inputs, dict) else {}
    outputs = outputs if isinstance(outputs, dict) else {}
    return TokenUsage(
        *counts,
        cache_read_tokens=_count(inputs.get("cache_read")),
        cache_creation_tokens=_count(inputs.get("cache_creation")),
        reasoning_tokens=_count(outputs.get("reasoning")),
    )


def observe_model_call(
    live: Optional[LiveSinks],
    invoke: Callable[[], Any],
    *,
    purpose: str,
    exchange_id: str,
    usage: Callable[[Any], Optional[TokenUsage]] = usage_of,
) -> Any:
    """Bracket one provider call with started/finished observers carrying its usage.

    `usage` maps the call's return value to its usage. After an exception it is
    given `None`, so the default reports no usage for a failed call.
    """
    if live is None:
        return invoke()
    identity = (live.node_id, exchange_id, purpose, live.provider, live.model)
    live.emit_event(ModelCallStarted(*identity))
    try:
        reply = invoke()
    except (_CapabilityFault, _HostStop):
        raise
    except Exception as exc:
        live.emit_event(ModelCallFinished(*identity, usage=usage(None), error=type(exc).__name__))
        raise
    live.emit_event(ModelCallFinished(*identity, usage=usage(reply)))
    return reply


def invoke_chat(
    live: Optional[LiveSinks], chat: Any, messages: list, *, exchange_id: str, purpose: str = "chat"
) -> Any:
    """Call a chat model once, bracketed by observers when an event sink exists."""
    call = lambda: chat.invoke(messages)  # noqa: E731
    return observe_model_call(live, call, purpose=purpose, exchange_id=exchange_id)


def _text(result: Any) -> Optional[str]:
    try:
        return str(result)
    except Exception:
        return None  # the caller's own representation boundary owns this error


def observe_tool_call(
    live: Optional[LiveSinks],
    invoke: Callable[[], Any],
    *,
    exchange_id: str,
    call_index: int,
    tool_call_id: str,
    name: str,
    arguments: Any,
) -> Any:
    """Bracket one granted tool call with started/finished observers."""
    if live is None:
        return invoke()
    try:
        shown = deepcopy(arguments)
    except Exception:
        shown = None
    identity = (live.node_id, exchange_id, call_index, tool_call_id, name)
    live.emit_event(ToolCallStarted(*identity, arguments=shown))
    try:
        result = invoke()
    except (_CapabilityFault, _HostStop):
        raise
    except Exception as exc:
        live.emit_event(ToolCallFinished(*identity, error=type(exc).__name__))
        raise
    live.emit_event(ToolCallFinished(*identity, result=_text(result)))
    return result

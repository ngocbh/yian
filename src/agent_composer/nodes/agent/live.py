"""Transient AGENT observers — streamed reply text, model/tool call brackets, usage.

Everything here is callback-only: chunks and events go to evaluator-supplied sinks
and nowhere else. Nothing is recorded, retained in memory, or replayed on resume,
and the node value is identical with or without sinks. A chat reply streams only
when a stream sink exists; otherwise the model is invoked exactly as before.

Sink exceptions arrive already wrapped as `_CapabilityFault` by the evaluator and
propagate unchanged; provider and tool exceptions are reported by type name only,
because their text can contain credentials, then re-raised.

Layer: nodes — imports events, nodes.base, peer AGENT modules, and LangChain.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, message_chunk_to_message
from langchain_core.runnables import Runnable

from agent_composer.events import (
    ModelCallFinished,
    ModelCallStarted,
    StreamChunk,
    TokenUsage,
    ToolCallFinished,
    ToolCallStarted,
)
from agent_composer.nodes.agent.interrupts import ReplyInterrupted
from agent_composer.nodes.agent.modes.utils import text_of
from agent_composer.nodes.base import _CapabilityFault, _HostStop

STREAM_KEY = "text"


@dataclass(frozen=True)
class LiveSinks:
    """The transient sinks of one AGENT invocation plus its effective model identity."""

    node_id: str
    provider: Optional[str]
    model: Optional[str]
    emit_event: Optional[Callable[[Any], None]]
    emit_stream: Optional[Callable[[Any], None]] = None


def live_sinks(ctx: Any) -> Optional[LiveSinks]:
    """Select the sinks of one mode context; `None` when neither sink exists."""
    emit_event = getattr(ctx, "emit_event", None)
    emit_stream = getattr(ctx, "emit_stream", None)
    if emit_event is None and emit_stream is None:
        return None
    config = ctx.llm_config or {}
    read = config.get if isinstance(config, dict) else (lambda key: getattr(config, key, None))
    provider, model = read("provider"), read("model")
    return LiveSinks(
        ctx.node_id,
        provider if isinstance(provider, str) else None,
        model if isinstance(model, str) else None,
        emit_event,
        emit_stream,
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


def _normalized(chat: Any, message: Any) -> Any:
    """Apply the client's own reply normalization, which `invoke` applies but `stream` skips."""
    normalize = getattr(getattr(chat, "bound", chat), "normalize_reply", None)
    return message if normalize is None else normalize(message)


def stream_reply(
    live: LiveSinks,
    chat: Any,
    messages: list,
    *,
    exchange_id: str,
    interrupted: Callable[[], bool] = lambda: False,
) -> Any:
    """Stream one chat reply, forwarding visible text, and return the aggregated message.

    The aggregate is the same message `invoke` returns: chunks are summed and
    converted like LangChain's own `generate_from_stream`, then normalized by the
    client. A model without native streaming, or a duck-typed client without
    `stream`, yields one complete message. The empty `final=True` chunk is sent
    only after the reply is complete and normalized, so a call that fails
    mid-stream ends without one; the host discards that exchange's partial text.

    `interrupted` is checked when each chunk after the first arrives, before it is
    shown. When it answers true the stream is closed, the `final=True` chunk is
    sent so the host keeps the text it showed, and `ReplyInterrupted` carries that
    text as the partial reply, with the usage reported so far. A single-chunk
    reply, or an interrupt that lands on the last chunk, returns the complete
    reply; the caller's tool loop then skips its calls.

    Raises:
        `ReplyInterrupted`: The host interrupted the reply mid-stream.
    """
    stream = getattr(chat, "stream", None)
    chunks = iter(stream(messages) if callable(stream) else (chat.invoke(messages),))
    aggregate = None
    shown: list[str] = []
    try:
        for chunk in chunks:
            # Checked before taking a later chunk, never after the last one, so a
            # stream that runs out always returns its complete reply.
            if aggregate is not None and interrupted():
                live.emit_stream(
                    StreamChunk(live.node_id, STREAM_KEY, "", final=True, exchange_id=exchange_id)
                )
                raise ReplyInterrupted(AIMessage(content="".join(shown)), usage_of(aggregate))
            aggregate = chunk if aggregate is None else aggregate + chunk
            text = text_of(chunk)
            if text:
                shown.append(text)
                live.emit_stream(StreamChunk(live.node_id, STREAM_KEY, text, exchange_id=exchange_id))
    finally:
        close = getattr(chunks, "close", None)
        if close is not None:
            close()  # release the provider connection when a sink or the stream fails
    if aggregate is None:
        raise ValueError("model stream produced no reply")
    message = _normalized(chat, message_chunk_to_message(aggregate))
    live.emit_stream(StreamChunk(live.node_id, STREAM_KEY, "", final=True, exchange_id=exchange_id))
    return message


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
    given `None`, so the default reports no usage for a failed call. A reply cut
    by a host interrupt reports the usage carried by `ReplyInterrupted`.
    """
    if live is None or live.emit_event is None:
        return invoke()
    identity = (live.node_id, exchange_id, purpose, live.provider, live.model)
    live.emit_event(ModelCallStarted(*identity))
    try:
        reply = invoke()
    except (_CapabilityFault, _HostStop):
        raise
    except ReplyInterrupted as cut:
        # A host interrupt ends the call without a provider error; the provider
        # still billed the chunks already received.
        live.emit_event(ModelCallFinished(*identity, usage=cut.usage))
        raise
    except Exception as exc:
        live.emit_event(ModelCallFinished(*identity, usage=usage(None), error=type(exc).__name__))
        raise
    live.emit_event(ModelCallFinished(*identity, usage=usage(reply)))
    return reply


def invoke_chat(
    live: Optional[LiveSinks],
    chat: Any,
    messages: list,
    *,
    exchange_id: str,
    purpose: str = "chat",
    interrupted: Callable[[], bool] = lambda: False,
) -> Any:
    """Call a chat model once: streamed when a stream sink exists, observed when an event sink does.

    `interrupted` is checked between streamed chunks; an unstreamed call always
    runs to completion.
    """
    if live is not None and live.emit_stream is not None and purpose == "chat":
        call = lambda: stream_reply(  # noqa: E731
            live, chat, messages, exchange_id=exchange_id, interrupted=interrupted
        )
    else:
        call = lambda: chat.invoke(messages)  # noqa: E731
    return observe_model_call(live, call, purpose=purpose, exchange_id=exchange_id)


class _UsageCapture(BaseCallbackHandler):
    """Keep the usage of the chat model reply inside one structured-output call."""

    def __init__(self) -> None:
        self.usage: Optional[TokenUsage] = None

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        generations = getattr(response, "generations", None) or [[]]
        for generation in generations[0]:
            self.usage = usage_of(getattr(generation, "message", None))


def invoke_structured(
    live: Optional[LiveSinks], bound: Any, messages: list, *, exchange_id: str
) -> Any:
    """Invoke a native structured-output binding, bracketed when an event sink exists.

    The result is returned exactly as `bound.invoke` produced it, so parsing and
    retries behave the same with or without a sink. Usage comes from a LangChain
    callback on the inner model call, which also sees a reply that later fails to
    parse. A binding that is not a LangChain `Runnable` reports no usage.
    """
    capture = _UsageCapture()
    if live is not None and live.emit_event is not None and isinstance(bound, Runnable):
        call = lambda: bound.invoke(messages, config={"callbacks": [capture]})  # noqa: E731
    else:
        call = lambda: bound.invoke(messages)  # noqa: E731
    return observe_model_call(
        live, call, purpose="structured_native", exchange_id=exchange_id,
        usage=lambda _reply: capture.usage,
    )


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
    if live is None or live.emit_event is None:
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

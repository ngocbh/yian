"""Provider-message encoding and effects over memory-owned conversation turns.

The adapter retains no history or lease. Every call reads or writes the supplied
turn; only the memory backend owns conversation state. Full provider input and
newly retained messages stay separate so retries need not repeat stored history.
"""

from __future__ import annotations

from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError as FutureCancelledError
from contextlib import contextmanager
from copy import deepcopy
import json
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage, ToolMessage, message_to_dict, messages_from_dict

from agent_composer._settings import default_llm_provider
from agent_composer.memory import (
    ConversationItem, ConversationTurn, JsonObject, MemoryOperationFailure,
)
from agent_composer.nodes.base import _CapabilityFault, _HostStop


MESSAGE_FORMAT = "agent-composer.langchain-message.v1"


def compatibility(config) -> JsonObject:
    """Select provider and message format, independent of model and current grants.

    Args:
        config (`dict | LLMConfig | None`): Effective model configuration. An
            absent provider uses the same environment default as the factory.

    Returns:
        `JsonObject`: Provider and versioned message format, without credentials.
    """
    provider = config.get("provider") if isinstance(config, dict) else getattr(config, "provider", None)
    return {"provider": (provider or default_llm_provider()).lower(), "message_format": MESSAGE_FORMAT}


def _canonical(value) -> str:
    """Compare finite JSON without Python's bool/integer equality coercion."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@contextmanager
def _memory_boundary():
    """Keep codec/backend faults distinct from errors eligible for model retries."""
    try:
        yield
    except (_CapabilityFault, _HostStop):
        raise
    except Exception as exc:
        raise _CapabilityFault(exc) from exc


def encode_message(message: BaseMessage, exchange_id: str) -> ConversationItem:
    """Detach one finite-JSON message under unique item and exchange identities.

    Args:
        message (`BaseMessage`): Actual provider input or surfaced output.
        exchange_id (`str`): Nonempty adapter-generated request identity, not a
            provider's tool-call identifier.

    Returns:
        `ConversationItem`: Immutable, detached versioned message payload.

    Raises:
        `TypeError`: The message or its payload is not supported finite JSON.
        `ValueError`: The message cannot round-trip without losing supplied data.
    """
    if not isinstance(message, BaseMessage):
        raise TypeError("conversation messages must be LangChain BaseMessage values")
    item = ConversationItem(uuid4().hex, MESSAGE_FORMAT,
                            {"exchange_id": exchange_id, "message": message_to_dict(message)})
    decode_message(item)
    return item


def decode_message(item: ConversationItem) -> BaseMessage:
    """Decode history, preserving every supplied field across compatible upgrades.

    Args:
        item (`ConversationItem`): Detached payload in the supported message format.

    Returns:
        `BaseMessage`: Restored role, content and metadata, with compatible defaults.

    Raises:
        `TypeError`: The carrier or payload is malformed.
        `ValueError`: The format is unknown, invalid, or loses supplied data.
    """
    if type(item) is not ConversationItem:
        raise TypeError("stored messages must be exact ConversationItem values")
    item = ConversationItem(item.item_id, item.format, item.payload)
    if item.format != MESSAGE_FORMAT:
        raise ValueError("unsupported conversation message format; reset is required")
    payload = item.payload
    if set(payload) != {"exchange_id", "message"}:
        raise ValueError("conversation message fields do not match the supported format")
    if type(payload["exchange_id"]) is not str or not payload["exchange_id"]:
        raise ValueError("conversation exchange_id must be a nonempty string")
    wire = payload["message"]
    if type(wire) is not dict or set(wire) != {"type", "data"} or type(wire["data"]) is not dict:
        raise ValueError("invalid stored LangChain message envelope")
    try:
        message, = messages_from_dict([deepcopy(wire)])
        decoded = message_to_dict(message)
        # New optional message fields may acquire defaults across compatible
        # library upgrades. Every field actually stored must survive unchanged.
        supplied = {"type": decoded["type"],
                    "data": {key: decoded["data"][key] for key in wire["data"]}}
        expected = deepcopy(wire)
        if wire["type"] in {"ai", "AIMessageChunk"}:
            # These are library-owned discriminators, never defaults for opaque
            # args, content blocks, additional_kwargs, or arbitrary added keys.
            for field, discriminator in (
                ("tool_calls", "tool_call"),
                ("invalid_tool_calls", "invalid_tool_call"),
                ("tool_call_chunks", "tool_call_chunk"),
            ):
                if field in type(message).model_fields:
                    for call in expected["data"].get(field, []):
                        call.setdefault("type", discriminator)
        if _canonical(supplied) != _canonical(expected):
            raise ValueError("stored message would be changed by decoding")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid stored conversation message") from exc
    return message


def history_messages(turn: ConversationTurn) -> list[BaseMessage]:
    """Project history into provider context without replaying effects or instructions.

    Args:
        turn (`ConversationTurn`): Admitted backend-owned turn providing prior items.

    Returns:
        `list[BaseMessage]`: Prior messages with context-only missing-result repairs,
            inert unpairable calls, and truthful failure/interruption annotations.
            Historical system/developer instructions are excluded. Repairs are not
            observations and must never be persisted as new input deltas.

    Raises:
        `_CapabilityFault`: Reading or decoding retained history fails.
        `_HostStop`: The backend reports cooperative cancellation.
    """
    messages = []
    with _memory_boundary():
        view = turn.view
        decoded = [(item, decode_message(item)) for item in view.items]
        purposes = {effect.request["exchange_id"]: effect.request.get("purpose", "chat")
                    for effect in view.effects if effect.kind == "model"}
        tool_effects = {(effect.request.get("model_exchange_id"), effect.request.get("call_index")): effect
                        for effect in view.effects if effect.kind == "tool"}
        consumed = set()
        for index, (item, message) in enumerate(decoded):
            if item.item_id in consumed:
                continue
            if isinstance(message, SystemMessage) or (
                isinstance(message, ChatMessage) and message.role in {"system", "developer"}
            ):
                continue
            exchange = item.payload["exchange_id"]
            if isinstance(message, ToolMessage):
                messages.append(_inert_message(message, "unpaired tool result"))
                continue
            if isinstance(message, AIMessage):
                if purposes.get(exchange) == "structured_native" or not _pairable_calls(message):
                    messages.append(_inert_message(message, "structured or unpairable model reply"))
                    continue
                messages.append(message)
                for call_index, call in enumerate(message.tool_calls):
                    matches = [(other, result) for other, result in decoded[index + 1:]
                               if other.payload["exchange_id"] == exchange
                               and isinstance(result, ToolMessage) and result.tool_call_id == call["id"]]
                    if len(matches) == 1:
                        other, result = matches[0]
                        consumed.add(other.item_id)
                        messages.append(result)
                    else:
                        effect = tool_effects.get((exchange, call_index))
                        status = None if effect is None else effect.status
                        if status == "completed":
                            note = "Tool execution completed; represented result unavailable."
                        elif status == "failed":
                            note = "Tool invocation failed; result unavailable."
                        else:
                            note = f"Tool result unavailable; execution outcome unknown (status: {status or 'unrecorded'})."
                        messages.append(ToolMessage(content=note, tool_call_id=call["id"], status="error"))
                continue
            messages.append(message)
        statuses = [{"invocation": entry.status} for entry in view.closed_invocations if entry.status != "completed"]
        closed = {entry.invocation for entry in view.closed_invocations}
        unclosed = dict.fromkeys(effect.invocation for effect in view.effects if effect.invocation not in closed)
        statuses += [{"invocation": "still open; completion unknown"} for _ in unclosed]
        statuses += [{"effect": effect.kind, "status": effect.status} for effect in view.effects if effect.status != "completed"]
        if statuses:
            messages.append(HumanMessage(content="Session history status (context only): " + _canonical(statuses)))
    return messages


def _inert_message(message: BaseMessage, reason: str) -> HumanMessage:
    """Render prior data as labelled text with no actionable message metadata."""
    return HumanMessage(content=f"Prior {reason} (context only, not an executable request): "
                        + _canonical(message_to_dict(message)))


def _pairable_calls(message: AIMessage) -> bool:
    """Require unambiguous tool correlations and no hidden provider call shape."""
    if message.invalid_tool_calls or "function_call" in message.additional_kwargs:
        return False
    if isinstance(message.content, list) and any(
        isinstance(block, dict) and isinstance(block.get("type"), str)
        and ("tool" in block["type"] or block["type"] == "function_call")
        for block in message.content
    ):
        return False
    calls = message.tool_calls
    if any(type(call.get("name")) is not str or not call["name"] or type(call.get("args")) is not dict
           for call in calls):
        return False
    ids = [call.get("id") for call in calls]
    if any(type(value) is not str or not value for value in ids) or len(set(ids)) != len(ids):
        return False
    raw = message.additional_kwargs.get("tool_calls")
    if raw is not None:
        if type(raw) is not list or len(raw) != len(calls):
            return False
        try:
            for encoded, call in zip(raw, calls):
                if encoded["type"] != "function" or encoded["id"] != call["id"]:
                    return False
                function = encoded["function"]
                if function["name"] != call["name"] or _canonical(json.loads(function["arguments"])) != _canonical(call["args"]):
                    return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def invoke_model(
    turn: ConversationTurn,
    invoke: Callable[[], Any],
    *,
    new_messages: Sequence[BaseMessage],
    check_stopped: Callable[[], None],
    exchange_id: str | None = None,
    purpose: str = "chat",
    raw_reply: Callable[[Any], BaseMessage] | None = None,
) -> Any:
    """Persist request/reply deltas around exactly one external model invocation.

    New messages are the actual input delta, not the full provider context.
    A surfaced reply is persisted before stop or caller-owned parsing. Request
    and successful observation failures remain outside the provider-error catch.

    Args:
        turn (`ConversationTurn`): Admitted backend-owned turn to update.
        invoke (`Callable`): One model call, invoked outside the memory boundary.
        new_messages (`Sequence[BaseMessage]`): Only newly submitted input messages.
        check_stopped (`Callable`): Cooperative stop check after reply publication.
        exchange_id (`str | None`): Optional caller-owned correlation identity;
            omission generates one unique ID for this model exchange.
        purpose (`str`): Request purpose; defaults to ordinary `chat` generation.
        raw_reply (`Callable | None`): Extract a raw message from a native structured
            carrier. Omission treats the returned value as the raw message.

    Returns:
        `Any`: The actual provider return, before caller-owned parsing.

    Raises:
        `_CapabilityFault`: Encoding or publishing messages fails.
        `_HostStop`: Cooperative cancellation was requested. Provider exceptions
            also propagate unchanged, including when failure publication fails.
    """
    exchange_id = exchange_id or uuid4().hex
    with _memory_boundary():
        items = tuple(encode_message(message, exchange_id) for message in new_messages)
        ordinal = turn.request("model", {"exchange_id": exchange_id, "purpose": purpose}, items=items)
    try:
        reply = invoke()
    except BaseException as primary:
        _observe_failure(turn, ordinal, "model", primary)
        raise
    with _memory_boundary():
        raw = reply if raw_reply is None else raw_reply(reply)
        turn.observe(ordinal, "completed", items=(encode_message(raw, exchange_id),))
    check_stopped()
    return reply


def _observe_preserving_error(turn, ordinal, status, primary, *, failure=None):
    """Preserve initiating identity while marking publication faults non-retryable.

    Ordinary errors use the existing evaluator-unwrapped capability carrier when
    publication also fails. BaseException cancellation and existing carriers retain
    their control identity; in particular `_HostStop.primary` must stay visible.
    """
    try:
        turn.observe(ordinal, status, failure=failure)
    except BaseException as secondary:
        original = primary.original if isinstance(primary, _CapabilityFault) else primary
        if secondary is not primary and secondary is not original:
            try:
                if hasattr(original, "add_note"):
                    original.add_note(f"Conversation observation failed: {type(secondary).__name__}: {secondary}")
            except BaseException:
                pass
        if isinstance(primary, Exception) and not isinstance(
            primary, (_CapabilityFault, _HostStop)
        ):
            raise _CapabilityFault(primary) from secondary


def _observe_failure(turn, ordinal, operation, primary):
    """Record a sanitized external-effect failure, preserving its exact exception."""
    interrupted = isinstance(primary, (_HostStop, AsyncCancelledError, FutureCancelledError)) or not isinstance(primary, Exception)
    status = "interrupted" if interrupted else "failed"
    # External exception text may contain credentials or request headers.
    failure = MemoryOperationFailure(operation, f"api.{operation}.{status}",
                                     f"{operation.capitalize()} invocation {status} ({type(primary).__name__}).", False)
    _observe_preserving_error(turn, ordinal, status, primary, failure=failure)


def invoke_tool_effect(
    turn: ConversationTurn,
    invoke: Callable[[], Any],
    *,
    name: str,
    arguments: JsonObject,
    model_exchange_id: str,
    call_index: int,
    tool_call_id: str,
    check_stopped: Callable[[], None],
) -> ToolMessage:
    """Persist one public tool invocation separately from result representation.

    Args:
        turn (`ConversationTurn`): Admitted backend-owned turn to update.
        invoke (`Callable`): Existing public tool validation and invocation boundary.
        name (`str`): Currently granted tool name.
        arguments (`JsonObject`): Actual requested public tool arguments.
        model_exchange_id (`str`): Unique model exchange that requested this call.
        call_index (`int`): Zero-based call position within that model reply.
        tool_call_id (`str`): Provider correlation ID, which may repeat across exchanges.
        check_stopped (`Callable`): Cooperative stop check after publishing the result.

    Returns:
        `ToolMessage`: Actual represented result, already retained in session memory.

    Raises:
        `_CapabilityFault`: Encoding or memory access fails.
        `_HostStop`: Cooperative cancellation occurs. Tool and representation errors
            propagate unchanged; a representation error does not mark execution failed.
    """
    with _memory_boundary():
        ordinal = turn.request("tool", {
            "name": name, "arguments": arguments, "model_exchange_id": model_exchange_id,
            "call_index": call_index, "tool_call_id": tool_call_id,
        })
    try:
        result = invoke()
    except BaseException as primary:
        _observe_failure(turn, ordinal, "tool", primary)
        raise
    try:
        text = str(result)
        with _memory_boundary():
            message = ToolMessage(content=text, tool_call_id=tool_call_id)
            item = encode_message(message, model_exchange_id)
    except BaseException as primary:
        _observe_preserving_error(turn, ordinal, "completed", primary)
        raise
    with _memory_boundary():
        turn.observe(ordinal, "completed", items=(item,))
    check_stopped()
    return message

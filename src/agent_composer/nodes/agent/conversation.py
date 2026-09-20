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
from uuid import uuid4

from langchain_core.messages import BaseMessage, ChatMessage, SystemMessage, message_to_dict, messages_from_dict

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
    """Restore actual messages without reactivating historical system instructions.

    Args:
        turn (`ConversationTurn`): Admitted backend-owned turn providing prior items.

    Returns:
        `list[BaseMessage]`: Ordered messages excluding prior system/developer roles.

    Raises:
        `_CapabilityFault`: Reading or decoding retained history fails.
        `_HostStop`: The backend reports cooperative cancellation.
    """
    messages = []
    with _memory_boundary():
        for item in turn.view.items:
            message = decode_message(item)
            if isinstance(message, SystemMessage) or (
                isinstance(message, ChatMessage) and message.role in {"system", "developer"}
            ):
                continue
            messages.append(message)
    return messages


def invoke_model(
    turn: ConversationTurn,
    invoke: Callable[[], BaseMessage],
    *,
    new_messages: Sequence[BaseMessage],
    check_stopped: Callable[[], None],
) -> BaseMessage:
    """Persist request/reply deltas around exactly one external model invocation.

    New messages are the actual input delta, not the full provider context.
    A surfaced reply is persisted before stop or caller-owned parsing. Request
    and successful observation failures remain outside the provider-error catch.

    Args:
        turn (`ConversationTurn`): Admitted backend-owned turn to update.
        invoke (`Callable`): One model call, invoked outside the memory boundary.
        new_messages (`Sequence[BaseMessage]`): Only newly submitted input messages.
        check_stopped (`Callable`): Cooperative stop check after reply publication.

    Returns:
        `BaseMessage`: The actual surfaced reply, before caller-owned parsing.

    Raises:
        `_CapabilityFault`: Encoding or publishing messages fails.
        `_HostStop`: Cooperative cancellation was requested. Provider exceptions
            also propagate unchanged, including when failure publication fails.
    """
    exchange_id = uuid4().hex
    with _memory_boundary():
        items = tuple(encode_message(message, exchange_id) for message in new_messages)
        ordinal = turn.request("model", {"exchange_id": exchange_id}, items=items)
    try:
        reply = invoke()
    except BaseException as primary:
        interrupted = isinstance(primary, (_HostStop, AsyncCancelledError, FutureCancelledError)) or not isinstance(primary, Exception)
        status = "interrupted" if interrupted else "failed"
        # Provider exception text may contain credentials or request headers.
        failure = MemoryOperationFailure("model", "api.model." + status,
                                         f"Model invocation {status} ({type(primary).__name__}).", False)
        try:
            turn.observe(ordinal, status, failure=failure)
        except BaseException as secondary:
            if secondary is not primary:
                try:
                    if hasattr(primary, "add_note"):
                        primary.add_note(f"Conversation observation failed: {type(secondary).__name__}: {secondary}")
                except BaseException:
                    pass
        raise
    with _memory_boundary():
        turn.observe(ordinal, "completed", items=(encode_message(reply, exchange_id),))
    check_stopped()
    return reply

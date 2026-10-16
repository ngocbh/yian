"""Provider-neutral messages, requests and replies, with a stable JSON shape.

A `Message` holds typed parts; `Message.to_dict` writes the ``yian.message.v1`` JSON
shape and `Message.from_dict` reads it back strictly. Every class checks its values
on construction, so an invalid message cannot exist: opaque provider data, tool-call
arguments and tool parameters must be plain JSON within yian's nesting and int-digit
limits. The check walks values iteratively before anything is copied, so a deep value
raises `MessageFormatError`, never `RecursionError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
import json
import math
import re
from typing import Any, Literal

from yian.defaults import current

MESSAGE_FORMAT = "yian.message.v1"

Role = Literal["system", "user", "assistant", "tool"]
_ROLES = frozenset({"system", "user", "assistant", "tool"})
_MESSAGE_KEYS = frozenset({"format", "role", "parts", "protocol", "provider_payload"})
_PLAIN_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# A Python str never pairs surrogates, so any code point in this range is unpaired and
# cannot be encoded as UTF-8 or sent as JSON.
_SURROGATE = re.compile("[\ud800-\udfff]")
_UNPAIRED_SURROGATE = "string has an unpaired surrogate"
_TOOL_CHOICES = frozenset({"auto", "required", "none"})


class MessageFormatError(ValueError):
    """A message, part or value does not fit the ``yian.message.v1`` shape.

    The text never echoes payload values, which may hold provider secrets.

    Args:
        path (`str`): JSON path of the offending value, e.g. ``parts[2].arguments``;
            empty for the value itself.
        reason (`str`): What is wrong with it.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"{path}: {reason}" if path else reason)
        self.path = path
        self.reason = reason

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild from path and reason, which `Exception`'s default reduction would merge."""
        return (type(self), (self.path, self.reason))

    def within(self, prefix: str) -> MessageFormatError:
        """Return the same error with its path placed under `prefix`.

        Args:
            prefix (`str`): Path of the value that contains this error's value.

        Returns:
            `MessageFormatError`: An error whose path is ``prefix`` joined with this path.
        """
        return MessageFormatError(_join(prefix, self.path), self.reason)


def _join(prefix: str, path: str) -> str:
    """Join two JSON paths, e.g. ``parts[1]`` and ``text`` into ``parts[1].text``."""
    if not prefix:
        return path
    if not path:
        return prefix
    return prefix + path if path.startswith("[") else f"{prefix}.{path}"


def _key_segment(key: str) -> str:
    """Return the path segment for a dict key: ``.key``, or ``["odd key"]``."""
    if _PLAIN_KEY.fullmatch(key):
        return "." + key
    return f"[{json.dumps(key)}]"


_INT_BOUNDS: dict[int, int] = {}


def _int_bound() -> int:
    """Return the smallest int with more digits than `current().max_int_digits` allows."""
    digits = current().max_int_digits
    bound = _INT_BOUNDS.get(digits)
    if bound is None:
        bound = _INT_BOUNDS[digits] = 10**digits
    return bound


# A path in `check_json_value`: the root path, or ``(parent, key or index)``.
_Where = str | tuple[Any, str | int]


def _format_path(where: _Where) -> str:
    """Return the JSON path text for a `check_json_value` parent chain."""
    segments = []
    while isinstance(where, tuple):
        where, segment = where
        segments.append(f"[{segment}]" if isinstance(segment, int) else _key_segment(segment))
    return where + "".join(reversed(segments))


def check_json_value(value: Any, path: str) -> None:
    """Raise unless `value` is plain JSON within yian's nesting and int-digit limits.

    Allowed: `dict` with `str` keys, `list`, `str`, `int`, finite `float`, `bool` and
    `None`; no string or key may hold an unpaired surrogate. Each `dict` and `list` is
    one level. The walk uses an explicit stack, so a value of any depth raises
    `MessageFormatError` rather than `RecursionError`.

    Args:
        value (`Any`): The value to check.
        path (`str`): JSON path of `value`, used in the error.

    Raises:
        `MessageFormatError`: Naming the path of the first offending value.
    """
    limit = current().max_value_nesting
    bound = _int_bound()
    # Each entry keeps its parent chain and the path text is built only on failure, so
    # a large valid value costs O(nodes) rather than O(nodes x depth).
    stack: list[tuple[Any, _Where, int]] = [(value, path, 1)]
    while stack:
        item, where, level = stack.pop()
        if isinstance(item, dict):
            if level > limit:
                raise MessageFormatError(
                    _format_path(where), f"value nests deeper than {limit} levels"
                )
            children = []
            for key, child in item.items():
                if not isinstance(key, str):
                    raise MessageFormatError(
                        _format_path(where), f"object key is {type(key).__name__}, not str"
                    )
                if _SURROGATE.search(key):
                    raise MessageFormatError(
                        _format_path(where), "object key has an unpaired surrogate"
                    )
                children.append((child, (where, key), level + 1))
            # Reversed so the first offending value in document order is reported.
            stack.extend(reversed(children))
        elif isinstance(item, list):
            if level > limit:
                raise MessageFormatError(
                    _format_path(where), f"value nests deeper than {limit} levels"
                )
            stack.extend(
                (item[i], (where, i), level + 1) for i in range(len(item) - 1, -1, -1)
            )
        elif isinstance(item, str):
            if _SURROGATE.search(item):
                raise MessageFormatError(_format_path(where), _UNPAIRED_SURROGATE)
        elif item is None or isinstance(item, bool):
            continue
        elif isinstance(item, int):
            if abs(item) >= bound:
                digits = current().max_int_digits
                raise MessageFormatError(
                    _format_path(where), f"int has more than {digits} digits"
                )
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise MessageFormatError(_format_path(where), "float is not finite")
        else:
            raise MessageFormatError(
                _format_path(where), f"{type(item).__name__} is not a JSON value"
            )


def _copy_json(value: Any) -> Any:
    """Deep-copy a value `check_json_value` accepted; its depth bounds the recursion."""
    if isinstance(value, dict):
        return {key: _copy_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_copy_json(child) for child in value]
    return value


def _check_type(value: Any, expected: type, path: str, *, optional: bool = False) -> None:
    """Raise `MessageFormatError` unless `value` is an `expected` (or `None` when `optional`).

    A `str` must also hold no unpaired surrogate, which could not be sent as JSON.
    """
    if value is None and optional:
        return
    if not isinstance(value, expected):
        raise MessageFormatError(path, f"expected {expected.__name__}, got {type(value).__name__}")
    if isinstance(value, str) and _SURROGATE.search(value):
        raise MessageFormatError(path, _UNPAIRED_SURROGATE)


def _set_object(instance: Any, name: str, *, optional: bool = True) -> None:
    """Check a JSON-object field of a frozen dataclass and replace it with a deep copy."""
    value = getattr(instance, name)
    if value is None and optional:
        return
    if not isinstance(value, dict):
        raise MessageFormatError(name, f"expected an object, got {type(value).__name__}")
    check_json_value(value, name)
    object.__setattr__(instance, name, _copy_json(value))


def _set_sequence(instance: Any, name: str, item_type: type) -> None:
    """Check a list-or-tuple field of `item_type` items and store it as a tuple."""
    value = getattr(instance, name)
    if not isinstance(value, (list, tuple)):
        raise MessageFormatError(name, f"expected a list, got {type(value).__name__}")
    for index, item in enumerate(value):
        if not isinstance(item, item_type):
            raise MessageFormatError(
                f"{name}[{index}]", f"expected a {item_type.__name__}, got {type(item).__name__}"
            )
    object.__setattr__(instance, name, tuple(value))


@dataclass(frozen=True)
class TextPart:
    """Text the model wrote or the user sent.

    Attributes:
        text (`str`): The text.
        provider_payload (`dict | None`): Opaque provider data, kept untouched.
    """

    text: str
    provider_payload: dict | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _check_type(self.text, str, "text")
        _set_object(self, "provider_payload")


@dataclass(frozen=True)
class MediaPart:
    """An image or file, given inline as base64 or by URL; exactly one of the two.

    Attributes:
        mime (`str`): The media type, e.g. ``image/png``.
        data (`str | None`): Base64 content.
        url (`str | None`): Where the content lives.
        name (`str | None`): A file name, if any.
        provider_payload (`dict | None`): Opaque provider data, e.g. OpenAI's image ``detail``.
    """

    mime: str
    data: str | None = None
    url: str | None = None
    name: str | None = None
    provider_payload: dict | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _check_type(self.mime, str, "mime")
        _check_type(self.data, str, "data", optional=True)
        _check_type(self.url, str, "url", optional=True)
        _check_type(self.name, str, "name", optional=True)
        if (self.data is None) == (self.url is None):
            raise MessageFormatError("", "media needs exactly one of data or url")
        _set_object(self, "provider_payload")


@dataclass(frozen=True)
class ToolCallPart:
    """A tool call the model asked for.

    By adapter convention (not enforced), `arguments` is `None` exactly when the raw
    text failed `parse_arguments`; both may be set when the adapter keeps the raw text.

    Attributes:
        id (`str`): The call id that the tool result answers.
        name (`str`): The tool name.
        arguments (`dict | None`): The parsed arguments.
        raw_arguments (`str | None`): The arguments text as the model sent it.
        provider_payload (`dict | None`): Opaque provider data, e.g. a Responses item id.
    """

    id: str
    name: str
    arguments: dict | None
    raw_arguments: str | None = None
    provider_payload: dict | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _check_type(self.id, str, "id")
        _check_type(self.name, str, "name")
        _check_type(self.raw_arguments, str, "raw_arguments", optional=True)
        if self.arguments is None and self.raw_arguments is None:
            raise MessageFormatError("arguments", "a tool call needs arguments or raw_arguments")
        _set_object(self, "arguments")
        _set_object(self, "provider_payload")


@dataclass(frozen=True)
class ToolResultPart:
    """The result of a tool call, sent back to the model.

    Attributes:
        call_id (`str`): The id of the call this answers.
        name (`str`): The tool name.
        content (`str`): The result text.
        is_error (`bool`): Whether the tool failed.
        provider_payload (`dict | None`): Opaque provider data, kept untouched.
    """

    call_id: str
    name: str
    content: str
    is_error: bool = False
    provider_payload: dict | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _check_type(self.call_id, str, "call_id")
        _check_type(self.name, str, "name")
        _check_type(self.content, str, "content")
        _check_type(self.is_error, bool, "is_error")
        _set_object(self, "provider_payload")


@dataclass(frozen=True)
class ReasoningPart:
    """The model's reasoning text.

    Attributes:
        text (`str`): The reasoning, or a summary of it; may be empty.
        provider_payload (`dict | None`): Opaque provider data, e.g. an Anthropic
            signature or Responses ``encrypted_content``.
    """

    text: str
    provider_payload: dict | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _check_type(self.text, str, "text")
        _set_object(self, "provider_payload")


Part = TextPart | MediaPart | ToolCallPart | ToolResultPart | ReasoningPart
_PART_CLASSES: dict[str, type] = {
    "text": TextPart,
    "media": MediaPart,
    "tool_call": ToolCallPart,
    "tool_result": ToolResultPart,
    "reasoning": ReasoningPart,
}
_PART_TYPES = {cls: name for name, cls in _PART_CLASSES.items()}
# Keys a part's JSON must hold; the rest are omitted when `None` or the default.
_REQUIRED_KEYS: dict[type, frozenset[str]] = {
    TextPart: frozenset({"text"}),
    MediaPart: frozenset({"mime"}),
    ToolCallPart: frozenset({"id", "name"}),
    ToolResultPart: frozenset({"call_id", "name", "content"}),
    ReasoningPart: frozenset({"text"}),
}


def _part_to_dict(part: Part) -> dict:
    """Return a part's JSON, omitting `None` fields and ``is_error: false``."""
    data: dict[str, Any] = {"type": _PART_TYPES[type(part)]}
    for field in fields(part):
        value = getattr(part, field.name)
        if value is None or (field.name == "is_error" and value is False):
            continue
        data[field.name] = _copy_json(value)
    return data


def _part_from_dict(data: Any, path: str) -> Part:
    """Read one part's JSON strictly; errors name paths under `path`."""
    if not isinstance(data, Mapping):
        raise MessageFormatError(path, f"expected an object, got {type(data).__name__}")
    if "type" not in data:
        raise MessageFormatError(_join(path, "type"), "missing")
    kind = data["type"]
    cls = _PART_CLASSES.get(kind) if isinstance(kind, str) else None
    if cls is None:
        raise MessageFormatError(_join(path, "type"), "unknown part type")
    names = {field.name for field in fields(cls)}
    for key in data:
        if key != "type" and key not in names:
            raise MessageFormatError(_join(path, _segment(key)), "unknown key")
    required = _REQUIRED_KEYS[cls]
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        if field.name in data:
            kwargs[field.name] = data[field.name]
        elif field.name in required:
            raise MessageFormatError(_join(path, field.name), "missing")
        elif field.default is MISSING:
            # `ToolCallPart.arguments` has no default but is omitted from JSON when `None`.
            kwargs[field.name] = None
    try:
        return cls(**kwargs)
    except MessageFormatError as error:
        raise error.within(path) from None


def _segment(key: Any) -> str:
    """Return a key's path segment without the leading dot, for any key type."""
    if isinstance(key, str):
        segment = _key_segment(key)
        return segment[1:] if segment.startswith(".") else segment
    return f"[{key!r}]"


@dataclass(frozen=True)
class Message:
    """One message of a conversation.

    Attributes:
        role (`Role`): ``system``, ``user``, ``assistant`` or ``tool``.
        parts (`tuple[Part, ...]`): The content; a list is accepted and stored as a tuple.
        protocol (`str | None`): Protocol id of the adapter that produced this message,
            e.g. ``openai_responses`` or ``catalog:<provider>``. Every provider payload in
            the message belongs to it, so an adapter can drop foreign payloads after a
            model switch.
        provider_payload (`dict | None`): Message-level opaque provider data.
    """

    role: Role
    parts: tuple[Part, ...]
    protocol: str | None = None
    provider_payload: dict | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or self.role not in _ROLES:
            raise MessageFormatError("role", "expected system, user, assistant or tool")
        if not isinstance(self.parts, (list, tuple)):
            raise MessageFormatError("parts", f"expected a list, got {type(self.parts).__name__}")
        for index, part in enumerate(self.parts):
            if not isinstance(part, Part):
                raise MessageFormatError(
                    f"parts[{index}]", f"expected a part, got {type(part).__name__}"
                )
        object.__setattr__(self, "parts", tuple(self.parts))
        _check_type(self.protocol, str, "protocol", optional=True)
        _set_object(self, "provider_payload")

    def to_dict(self) -> dict:
        """Return the message as ``yian.message.v1`` JSON.

        Returns:
            `dict`: ``{"format", "role", "parts", ...}``, omitting `None` fields; payloads
                are copies, so changing the result leaves the message unchanged.
        """
        data: dict[str, Any] = {
            "format": MESSAGE_FORMAT,
            "role": self.role,
            "parts": [_part_to_dict(part) for part in self.parts],
        }
        if self.protocol is not None:
            data["protocol"] = self.protocol
        if self.provider_payload is not None:
            data["provider_payload"] = _copy_json(self.provider_payload)
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> Message:
        """Read a message from ``yian.message.v1`` JSON, strictly.

        Args:
            data (`Mapping`): The JSON object, as `to_dict` writes it.

        Returns:
            `Message`: The message.

        Raises:
            `MessageFormatError`: Naming the JSON path, for a missing or wrong format, an
                unknown key or part type, a wrong type, a missing field, or a value that
                is not plain JSON within yian's limits.
        """
        if not isinstance(data, Mapping):
            raise MessageFormatError("", f"expected an object, got {type(data).__name__}")
        if "format" not in data:
            raise MessageFormatError("format", "missing")
        if data["format"] != MESSAGE_FORMAT:
            raise MessageFormatError("format", f"expected {MESSAGE_FORMAT!r}")
        for key in data:
            if key not in _MESSAGE_KEYS:
                raise MessageFormatError(_segment(key), "unknown key")
        for key in ("role", "parts"):
            if key not in data:
                raise MessageFormatError(key, "missing")
        raw_parts = data["parts"]
        if not isinstance(raw_parts, list):
            raise MessageFormatError("parts", f"expected a list, got {type(raw_parts).__name__}")
        parts = tuple(
            _part_from_dict(part, f"parts[{index}]") for index, part in enumerate(raw_parts)
        )
        return cls(
            role=data["role"],
            parts=parts,
            protocol=data.get("protocol"),
            provider_payload=data.get("provider_payload"),
        )

    def text(self) -> str:
        """Return the text parts' texts joined in order.

        Returns:
            `str`: The concatenated text; empty when there are no text parts.
        """
        return "".join(part.text for part in self.parts if isinstance(part, TextPart))

    def tool_calls(self) -> tuple[ToolCallPart, ...]:
        """Return the tool-call parts in order.

        Returns:
            `tuple[ToolCallPart, ...]`: The tool calls.
        """
        return tuple(part for part in self.parts if isinstance(part, ToolCallPart))


@dataclass(frozen=True)
class Usage:
    """Token counts for one model call.

    Attributes:
        input_tokens (`int`): Prompt tokens.
        output_tokens (`int`): Completion tokens.
        total_tokens (`int`): Their total, as the provider reports it.
        cache_read_tokens (`int | None`): Prompt tokens read from the provider's cache.
        cache_creation_tokens (`int | None`): Prompt tokens written to the cache.
        reasoning_tokens (`int | None`): Output tokens spent on reasoning.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call.

    Attributes:
        name (`str`): The tool name.
        description (`str`): What the tool does, for the model.
        parameters (`dict`): The JSON Schema of its arguments; checked and copied like a payload.
        strict (`bool | None`): Ask the provider to enforce the schema; `None` omits the field.
    """

    name: str
    description: str
    parameters: dict
    strict: bool | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _check_type(self.name, str, "name")
        _check_type(self.description, str, "description")
        _check_type(self.strict, bool, "strict", optional=True)
        _set_object(self, "parameters", optional=False)


@dataclass(frozen=True)
class NamedTool:
    """A tool choice that forces one named tool.

    Attributes:
        name (`str`): The tool the model must call.
    """

    name: str


ToolChoice = Literal["auto", "required", "none"] | NamedTool | None


@dataclass(frozen=True)
class Request:
    """One model call's input.

    Attributes:
        messages (`tuple[Message, ...]`): The conversation; a list is stored as a tuple.
        tools (`tuple[ToolSpec, ...]`): Tools the model may call; a list is stored as a tuple.
        tool_choice (`ToolChoice`): ``auto``, ``required``, ``none``, a `NamedTool`, or
            `None` to omit the field.
        stream (`bool`): `False` makes a non-streaming HTTP call; `ChatModel.stream`
            still emits the parsed reply as events.
    """

    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice = None
    stream: bool = True

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _set_sequence(self, "messages", Message)
        _set_sequence(self, "tools", ToolSpec)
        choice = self.tool_choice
        if not (
            choice is None
            or isinstance(choice, NamedTool)
            or (isinstance(choice, str) and choice in _TOOL_CHOICES)
        ):
            raise MessageFormatError(
                "tool_choice", "expected auto, required, none, a NamedTool or None"
            )


@dataclass(frozen=True)
class Response:
    """One model call's reply.

    Attributes:
        message (`Message`): The assistant message.
        usage (`Usage | None`): Token counts, when the provider reported them.
        finish_reason (`str | None`): ``stop``, ``length``, ``tool_calls``,
            ``content_filter`` or other provider text.
        response_id (`str | None`): The provider's id for the reply.
    """

    message: Message
    usage: Usage | None = None
    finish_reason: str | None = None
    response_id: str | None = None

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _check_type(self.message, Message, "message")
        if self.message.role != "assistant":
            raise MessageFormatError("message.role", "expected assistant")

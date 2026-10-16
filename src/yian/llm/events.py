"""Stream events a model emits, and how they add up to a `Response`.

An adapter turns a provider's stream into these events: deltas per part index, an
authoritative `PartDone` per part when the provider gives one, usage, and a `Finish`.
`collect` assembles them into a `Response`; `EventStream` is the iterator a model
returns, which owns and closes the underlying connection.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
import json
from typing import Any, Protocol

from yian.defaults import current
from yian.llm.errors import ErrorKind, ProviderError
from yian.llm.messages import (
    Message,
    Part,
    ReasoningPart,
    Request,
    Response,
    TextPart,
    ToolCallPart,
    Usage,
    _check_json,
)


@dataclass(frozen=True)
class TextDelta:
    """More text for the text part at `index`.

    Attributes:
        index (`int`): The part's position in the reply.
        text (`str`): The new text.
    """

    index: int
    text: str


@dataclass(frozen=True)
class ReasoningDelta:
    """More reasoning text for the reasoning part at `index`.

    Attributes:
        index (`int`): The part's position in the reply.
        text (`str`): The new text.
    """

    index: int
    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    """More of the tool call at `index`.

    Attributes:
        index (`int`): The part's position in the reply.
        id (`str | None`): The call id, when this delta carries it.
        name (`str | None`): The tool name, when this delta carries it.
        arguments (`str`): More arguments text.
    """

    index: int
    id: str | None
    name: str | None
    arguments: str


@dataclass(frozen=True)
class PartDone:
    """The final part at `index`; it replaces whatever its deltas built.

    Attributes:
        index (`int`): The part's position in the reply.
        part (`Part`): The complete part.
    """

    index: int
    part: Part

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class UsageReport:
    """Token counts so far; the last report wins.

    Attributes:
        usage (`Usage`): The counts.
    """

    usage: Usage


@dataclass(frozen=True)
class Finish:
    """The reply ended; the last `Finish` wins.

    Attributes:
        reason (`str | None`): The finish reason, e.g. ``stop`` or ``tool_calls``.
        response_id (`str | None`): The provider's id for the reply.
        protocol (`str | None`): The adapter's protocol id, for `Message.protocol`.
        provider_payload (`dict | None`): Message-level opaque data, for
            `Message.provider_payload`.
    """

    reason: str | None
    response_id: str | None = None
    protocol: str | None = None
    provider_payload: dict | None = None

    __hash__ = None  # type: ignore[assignment]


Event = TextDelta | ReasoningDelta | ToolCallDelta | PartDone | UsageReport | Finish


def _reject_constant(name: str) -> Any:
    """Refuse ``NaN`` and ``Infinity``, which `json.loads` accepts by default."""
    raise ValueError(f"{name} is not JSON")


def _parse_int(token: str) -> int:
    """Parse a JSON int token, refusing one past `current().max_int_digits` before `int()` runs."""
    # A JSON int token has no leading zeros, so its length less any sign is its digit count.
    if len(token) - token.startswith("-") > current().max_int_digits:
        raise ValueError("int has too many digits")
    return int(token)


def parse_arguments(text: str) -> tuple[dict | None, str | None]:
    """Parse a tool call's complete arguments text.

    An empty text is an empty object. Otherwise the text must be a JSON object with
    no ``NaN`` or ``Infinity``, no non-finite float, no int past
    `current().max_int_digits` digits, nested at most `current().max_value_nesting`
    levels.

    Args:
        text (`str`): The arguments text as the model sent it.

    Returns:
        `tuple[dict | None, str | None]`: ``(obj, None)`` on success, ``(None, text)``
            on any failure, so the caller can apply its own checks to the raw text.
    """
    if text == "":
        return {}, None
    try:
        value = json.loads(text, parse_constant=_reject_constant, parse_int=_parse_int)
        if not isinstance(value, dict):
            return None, text
        # Also refuses a float that overflowed to infinity, e.g. ``1e400``.
        _check_json(value, "")
    except (ValueError, RecursionError):
        return None, text
    return value, None


@dataclass
class _Assembly:
    """The deltas seen so far at one index."""

    kind: type
    chunks: list[str] = field(default_factory=list)
    id: str | None = None
    name: str | None = None

    def part(self, index: int) -> Part:
        """Build the part these deltas describe."""
        text = "".join(self.chunks)
        if self.kind is TextDelta:
            return TextPart(text)
        if self.kind is ReasoningDelta:
            return ReasoningPart(text)
        if self.id is None:
            raise ProviderError(ErrorKind.PROTOCOL, f"tool call {index} has no id")
        if self.name is None:
            raise ProviderError(ErrorKind.PROTOCOL, f"tool call {index} has no name")
        arguments, raw = parse_arguments(text)
        return ToolCallPart(self.id, self.name, arguments, raw_arguments=raw)


def collect(events: Iterable[Event]) -> Response:
    """Assemble a reply from its stream events.

    Parts are ordered by index. A `PartDone` part wins; otherwise the part is built
    from its deltas, with a tool call's id and name taken from the first non-`None`
    value and its arguments parsed by `parse_arguments`. The finish reason, response
    id, and the message's protocol and provider payload come from the last `Finish`;
    usage comes from the last `UsageReport`. A stream without `Finish` still collects.

    Args:
        events (`Iterable[Event]`): The events, in stream order.

    Returns:
        `Response`: The reply, with an assistant message.

    Raises:
        `ValueError`: One index has deltas of different kinds.
        `ProviderError`: A tool call built from deltas has no id or no name (kind
            ``PROTOCOL``).
        `TypeError`: An item is not an event.
    """
    done: dict[int, Part] = {}
    assemblies: dict[int, _Assembly] = {}
    usage: Usage | None = None
    finish: Finish | None = None
    for event in events:
        if isinstance(event, (TextDelta, ReasoningDelta, ToolCallDelta)):
            assembly = assemblies.get(event.index)
            if assembly is None:
                assembly = assemblies[event.index] = _Assembly(type(event))
            elif assembly.kind is not type(event):
                first, second = assembly.kind.__name__, type(event).__name__
                raise ValueError(f"index {event.index} mixes {first} and {second}")
            if isinstance(event, ToolCallDelta):
                assembly.chunks.append(event.arguments)
                if assembly.id is None:
                    assembly.id = event.id
                if assembly.name is None:
                    assembly.name = event.name
            else:
                assembly.chunks.append(event.text)
        elif isinstance(event, PartDone):
            done[event.index] = event.part
        elif isinstance(event, UsageReport):
            usage = event.usage
        elif isinstance(event, Finish):
            finish = event
        else:
            raise TypeError(f"not a stream event: {type(event).__name__}")
    parts = [
        done[index] if index in done else assemblies[index].part(index)
        for index in sorted(done.keys() | assemblies.keys())
    ]
    message = Message(
        "assistant",
        parts,
        protocol=finish.protocol if finish else None,
        provider_payload=finish.provider_payload if finish else None,
    )
    return Response(
        message,
        usage=usage,
        finish_reason=finish.reason if finish else None,
        response_id=finish.response_id if finish else None,
    )


class EventStream(Iterator[Event]):
    """Iterator over a model's events that owns the underlying connection.

    Closing is idempotent and also happens when the events run out or raise, so a
    connection is released however iteration ends.

    Args:
        events (`Iterator[Event]`): The events; closed with the stream when it has a
            ``close`` method, e.g. a generator.
        close (`Callable[[], None] | None`): Releases the connection; called once.
    """

    def __init__(self, events: Iterator[Event], close: Callable[[], None] | None = None) -> None:
        self._events = events
        self._close = close
        self._closed = False

    def __next__(self) -> Event:
        """Return the next event, closing the stream when the events end or fail."""
        if self._closed:
            raise StopIteration
        try:
            return next(self._events)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the events and release the connection; later calls do nothing."""
        if self._closed:
            return
        self._closed = True
        try:
            close_events = getattr(self._events, "close", None)
            if close_events is not None:
                close_events()
        finally:
            if self._close is not None:
                self._close()

    def __enter__(self) -> EventStream:
        """Return the stream itself."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the stream."""
        self.close()


class ChatModel(Protocol):
    """A model that answers requests, provider-neutral.

    Attributes:
        provider (`str`): The provider name, e.g. ``openai``.
        model (`str`): The model name.
    """

    provider: str
    model: str

    def stream(self, request: Request) -> EventStream:
        """Start a call and return its events.

        Args:
            request (`Request`): The call's input.

        Returns:
            `EventStream`: The reply's events; close it to release the connection.
        """
        ...

    def generate(self, request: Request) -> Response:
        """Make a call and return the whole reply, usually ``collect(self.stream(request))``.

        Args:
            request (`Request`): The call's input.

        Returns:
            `Response`: The reply.
        """
        ...

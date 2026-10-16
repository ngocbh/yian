"""Split a streamed HTTP body into server-sent events or NDJSON values.

Both readers take the body as byte chunks of any size and do work linear in its
length: a partial line is kept as a list of pieces and joined once, never rescanned.
"""

from __future__ import annotations

import codecs
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import json
import re
from typing import Any

from yian.llm.errors import ErrorKind, ProviderError

_LINE_END = re.compile(r"\r\n|\r|\n")


@dataclass(frozen=True)
class SSEEvent:
    """One dispatched server-sent event.

    Attributes:
        data (`str`): The event's ``data`` lines joined with ``\\n``.
        event (`str | None`): The ``event`` field, or `None` when absent or empty.
        id (`str | None`): The last event id seen in the stream, or `None` if none.
    """

    data: str
    event: str | None = None
    id: str | None = None


def iter_sse(chunks: Iterable[bytes]) -> Iterator[SSEEvent]:
    """Parse a server-sent event stream, following the WHATWG rules.

    The bytes are decoded as UTF-8 (invalid bytes become U+FFFD) with one leading BOM
    removed. Lines end at CRLF, CR or LF, even when a CRLF is split across chunks.
    Comment lines (starting with ``:``) and ``retry`` or unknown fields are ignored. A
    blank line dispatches the event if it had a ``data`` field. An event still open
    at the end of the stream is dropped.

    Args:
        chunks (`Iterable[bytes]`): The body as byte chunks of any size.

    Yields:
        `SSEEvent`: Each dispatched event, as soon as its blank line arrives.
    """
    # utf-8-sig removes one leading BOM, even one split across chunks.
    decoder = codecs.getincrementaldecoder("utf-8-sig")(errors="replace")
    data: list[str] = []
    event: str | None = None
    last_id: str | None = None
    for line in _lines(decoder.decode(chunk) for chunk in chunks):
        if not line:
            if data:
                yield SSEEvent(data="\n".join(data), event=event, id=last_id)
            data, event = [], None
            continue
        if line.startswith(":"):
            continue
        name, colon, value = line.partition(":")
        if colon and value.startswith(" "):
            value = value[1:]
        if name == "data":
            data.append(value)
        elif name == "event":
            event = value or None
        elif name == "id" and "\0" not in value:
            last_id = value


def _lines(texts: Iterable[str]) -> Iterator[str]:
    """Yield the complete lines of decoded text pieces, without their line ends.

    A trailing line with no line end is not yielded, since SSE drops it anyway.
    """
    pending: list[str] = []
    # A CR ending one piece may be the first half of a CRLF; its LF is then skipped.
    after_cr = False
    for text in texts:
        if after_cr and text.startswith("\n"):
            text = text[1:]
            after_cr = False
        if text:
            after_cr = text.endswith("\r")
        start = 0
        for match in _LINE_END.finditer(text):
            pending.append(text[start:match.start()])
            yield "".join(pending)
            pending = []
            start = match.end()
        if start < len(text):
            pending.append(text[start:])


def iter_ndjson(chunks: Iterable[bytes]) -> Iterator[Any]:
    """Parse newline-delimited JSON: one value per non-blank line.

    A final line without a newline still counts.

    Args:
        chunks (`Iterable[bytes]`): The body as byte chunks of any size.

    Yields:
        `Any`: Each line's JSON value.

    Raises:
        `ProviderError`: A line is not valid UTF-8 JSON (kind `PROTOCOL`); the message
            gives the line number, never the line.
    """
    pending: list[bytes] = []
    number = 0
    for chunk in chunks:
        *complete, rest = chunk.split(b"\n")
        for piece in complete:
            pending.append(piece)
            number += 1
            line = b"".join(pending)
            pending = []
            if line.strip():
                yield _json_line(line, number)
        if rest:
            pending.append(rest)
    line = b"".join(pending)
    if line.strip():
        yield _json_line(line, number + 1)


def _json_line(line: bytes, number: int) -> Any:
    """Parse one NDJSON line, raising a `ProviderError` that does not echo it."""
    try:
        return json.loads(line)
    except (ValueError, RecursionError):
        pass
    # Raised outside the except block so the decode error, whose `doc` holds the line,
    # is not chained.
    raise ProviderError(ErrorKind.PROTOCOL, f"invalid JSON line {number} in NDJSON stream")

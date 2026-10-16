"""Test helpers for model calls: cassettes, a redacting recorder, and a scripted model.

A cassette is a JSON file of HTTP exchanges. `CassetteTransport` replays one through
an httpx2 client and checks each request against the recording; `RecordingTransport`
writes one from real traffic with credentials replaced by `REDACTED`, which replay
treats as "present, any value". `find_secrets` flags anything shaped like a credential,
so a test can refuse to commit a cassette that still holds one. `ScriptedModel` is a
`ChatModel` that answers from a script, for tests above the HTTP layer.

Import this module as ``yian.llm.testing``; ``yian.llm`` does not import it.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qsl, quote, quote_plus, urlsplit

import httpx2

from yian.llm.events import (
    Event,
    EventStream,
    Finish,
    PartDone,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    UsageReport,
    collect,
)
from yian.llm.messages import ReasoningPart, Request, Response, TextPart, ToolCallPart

CASSETTE_FORMAT = "yian.cassette.v1"
REDACTED = "<redacted>"
CREDENTIAL_HEADERS = frozenset(
    {
        "authorization",
        "x-api-key",
        "api-key",
        "x-goog-api-key",
        "chatgpt-account-id",
        "cookie",
        "set-cookie",
        "proxy-authorization",
    }
)
IGNORED_HEADERS = frozenset(
    {
        "user-agent",
        "content-length",
        "host",
        "connection",
        "accept-encoding",
        "transfer-encoding",
        "content-encoding",
    }
)

_SOURCES = frozenset({"synthetic", "recorded"})
_CASSETTE_KEYS = frozenset({"format", "source", "description", "interactions"})
_REQUEST_KEYS = frozenset({"method", "url", "headers", "json"})
_RESPONSE_KEYS = frozenset({"status", "headers", "chunks", "json", "text"})
_BODY_KEYS = ("chunks", "json", "text")
# Headers whose saved value would describe the encoded body, which is not what is saved.
_DROPPED_RESPONSE_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})
# A chunk ends after a blank line, so a recorded SSE body replays one event per chunk.
_CHUNK = re.compile(r".*?(?:\r\n\r\n|\n\n|\r\r)|.+", re.DOTALL)
_MAX_DIFFS = 10

_UNICODE_ESCAPE = re.compile(r"\\u([0-9A-Fa-f]{4})")
_NOT_PRECEDED = r"(?<![A-Za-z0-9_\-])"
_SECRET_PATTERNS = (
    ("openai_key", re.compile(_NOT_PRECEDED + r"sk-[A-Za-z0-9_\-]{16,}")),
    ("anthropic_key", re.compile(_NOT_PRECEDED + r"sk-ant-[A-Za-z0-9_\-]{16,}")),
    ("google_api_key", re.compile(_NOT_PRECEDED + r"AIza[0-9A-Za-z_\-]{30,}")),
    (
        "jwt",
        re.compile(
            _NOT_PRECEDED + r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"
        ),
    ),
    (
        "bearer_token",
        re.compile(_NOT_PRECEDED + r"bearer\s+(?!<redacted>)[A-Za-z0-9._\-]{16,}", re.IGNORECASE),
    ),
)


def find_secrets(text: str) -> list[str]:
    """Find text shaped like an API key, a JWT or a bearer token.

    Each pattern must start at a word boundary, so text such as ``task-0123...`` is not
    read as an ``sk-`` key. ``Bearer <redacted>`` is not flagged. ``\\uXXXX`` escapes
    are decoded first, so a key a JSON encoder escaped (``\\u0073k-...``) is flagged.

    Args:
        text (`str`): The text to search, e.g. a cassette's JSON.

    Returns:
        `list[str]`: One kind name per match (``openai_key``, ``anthropic_key``,
            ``google_api_key``, ``jwt``, ``bearer_token``), never the matched values;
            empty when nothing matches.
    """
    text = _UNICODE_ESCAPE.sub(lambda match: chr(int(match[1], 16)), text)
    return [kind for kind, pattern in _SECRET_PATTERNS for _ in pattern.finditer(text)]


# --- cassettes -------------------------------------------------------------------------


def _check_keys(data: Any, where: str, required: Iterable[str], allowed: frozenset[str]) -> None:
    """Raise unless ``data`` is an object with every required key and no others."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected an object, got {type(data).__name__}")
    for key in required:
        if key not in data:
            raise ValueError(f"{where}: missing {key!r}")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}")


def _check_headers(headers: Any, where: str) -> None:
    """Raise unless ``headers`` maps strings to strings."""
    if not isinstance(headers, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in headers.items()
    ):
        raise ValueError(f"{where}.headers: expected an object of strings")


def _check_cassette(data: Any) -> dict:
    """Return ``data`` if it is a well-formed cassette, else raise naming the problem."""
    # The format is checked first, so a file of another kind is named as such.
    if not isinstance(data, dict) or data.get("format") != CASSETTE_FORMAT:
        raise ValueError(f"cassette: format must be {CASSETTE_FORMAT!r}")
    _check_keys(data, "cassette", _CASSETTE_KEYS, _CASSETTE_KEYS)
    if data["source"] not in _SOURCES:
        raise ValueError("cassette: source must be 'synthetic' or 'recorded'")
    if not isinstance(data["description"], str):
        raise ValueError("cassette: description must be a string")
    if not isinstance(data["interactions"], list):
        raise ValueError("cassette: interactions must be a list")
    for number, interaction in enumerate(data["interactions"]):
        where = f"interactions[{number}]"
        _check_keys(interaction, where, ("request", "response"), frozenset({"request", "response"}))
        request, response = interaction["request"], interaction["response"]
        _check_keys(request, f"{where}.request", ("method", "url"), _REQUEST_KEYS)
        if not isinstance(request["method"], str) or not isinstance(request["url"], str):
            raise ValueError(f"{where}.request: method and url must be strings")
        _check_headers(request.get("headers", {}), f"{where}.request")
        _check_keys(response, f"{where}.response", ("status",), _RESPONSE_KEYS)
        if not isinstance(response["status"], int) or isinstance(response["status"], bool):
            raise ValueError(f"{where}.response: status must be an int")
        _check_headers(response.get("headers", {}), f"{where}.response")
        if sum(key in response for key in _BODY_KEYS) != 1:
            raise ValueError(f"{where}.response: needs exactly one of chunks, json, text")
        chunks = response.get("chunks", [])
        if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
            raise ValueError(f"{where}.response: chunks must be a list of strings")
        if not isinstance(response.get("text", ""), str):
            raise ValueError(f"{where}.response: text must be a string")
    return data


def load_cassette(path: str | Path) -> dict:
    """Read a cassette file and check its shape.

    Args:
        path (`str | Path`): The cassette's JSON file.

    Returns:
        `dict`: The cassette.

    Raises:
        `ValueError`: If the file is not JSON or not a ``yian.cassette.v1`` cassette; the
            message names the offending part, e.g. ``interactions[0].response``.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"cassette {path} is not JSON: {exc}") from None
    return _check_cassette(data)


def _json_diffs(recorded: Any, actual: Any, path: str, out: list[str]) -> None:
    """Append the paths where ``actual`` differs from ``recorded``, without values.

    `REDACTED` in ``recorded`` matches any value at that path.
    """
    if recorded == REDACTED:
        return
    where = path or "(root)"
    if isinstance(recorded, dict) and isinstance(actual, dict):
        for key in recorded:
            child = f"{path}.{key}" if path else key
            if key not in actual:
                out.append(f"body {child}: missing")
            else:
                _json_diffs(recorded[key], actual[key], child, out)
        for key in actual.keys() - recorded.keys():
            out.append(f"body {path + '.' if path else ''}{key}: unexpected")
    elif isinstance(recorded, list) and isinstance(actual, list):
        if len(recorded) != len(actual):
            out.append(f"body {where}: length {len(actual)}, recorded {len(recorded)}")
        for index, (item, other) in enumerate(zip(recorded, actual)):
            _json_diffs(item, other, f"{path}[{index}]", out)
    elif type(recorded) is not type(actual) or recorded != actual:
        out.append(f"body {where}: differs")


def _query_diffs(recorded_url: str, actual_url: str) -> list[str]:
    """Compare query strings as multimaps, naming keys but never values."""
    recorded: dict[str, list[str]] = {}
    actual: dict[str, list[str]] = {}
    for url, into in ((recorded_url, recorded), (actual_url, actual)):
        for key, value in parse_qsl(urlsplit(url).query, keep_blank_values=True):
            into.setdefault(key, []).append(value)
    diffs = []
    for key in sorted(recorded.keys() | actual.keys()):
        expected, got = recorded.get(key, []), actual.get(key, [])
        if len(expected) != len(got):
            diffs.append(f"query parameter {key!r}: {len(got)} value(s), recorded {len(expected)}")
            continue
        remaining = Counter(got)
        for value in expected:
            if value == REDACTED:
                continue
            if remaining[value] == 0:
                diffs.append(f"query parameter {key!r}: value differs")
                break
            remaining[value] -= 1
    return diffs


def _request_diffs(recorded: Mapping[str, Any], request: httpx2.Request) -> list[str]:
    """List how ``request`` differs from a recorded one, never showing header or query values."""
    diffs = []
    if request.method.upper() != recorded["method"].upper():
        diffs.append(f"method {request.method}, recorded {recorded['method'].upper()}")
    expected_url = httpx2.URL(recorded["url"])
    for part in ("scheme", "host", "port", "path"):
        if getattr(request.url, part) != getattr(expected_url, part):
            diffs.append(f"url {part} differs")
    diffs.extend(_query_diffs(recorded["url"], str(request.url)))
    for name, value in recorded.get("headers", {}).items():
        lower = name.lower()
        if lower in IGNORED_HEADERS:
            continue
        if lower not in request.headers:
            diffs.append(f"header {lower!r}: missing")
        elif value != REDACTED and request.headers[lower] != value:
            diffs.append(f"header {lower!r}: value differs")
    content = request.content
    if "json" in recorded:
        try:
            body = json.loads(content)
        except ValueError:
            diffs.append("body: not JSON")
        else:
            _json_diffs(recorded["json"], body, "", diffs)
    elif content:
        diffs.append("body: present, recorded none")
    return diffs


class _Chunks(httpx2.SyncByteStream):
    """A response body that yields the recorded chunks one at a time."""

    def __init__(self, chunks: Sequence[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks


def _replay_response(recorded: Mapping[str, Any]) -> httpx2.Response:
    """Build the response a recorded interaction describes."""
    if "chunks" in recorded:
        body = [chunk.encode("utf-8") for chunk in recorded["chunks"]]
    elif "json" in recorded:
        body = [json.dumps(recorded["json"]).encode("utf-8")]
    else:
        body = [recorded["text"].encode("utf-8")]
    return httpx2.Response(
        recorded["status"], headers=recorded.get("headers", {}), stream=_Chunks(body)
    )


class CassetteTransport(httpx2.MockTransport):
    """Replay a cassette's interactions in order and record every request.

    Each request must match the next interaction: the same method; the same URL
    scheme, host, port and path; the same query parameters as a multimap; an equal
    JSON body; and every recorded header except `IGNORED_HEADERS`, by case-insensitive
    name. `REDACTED` in a recorded query value, header value or JSON value matches
    any value, as long as one is present. A mismatch or an extra request raises
    `AssertionError` naming the differing parts, never header or query values.

    Args:
        cassette (`dict | str | Path`): The cassette, or the path of its JSON file.

    Attributes:
        requests (`list[httpx2.Request]`): Every request received, in order.

    Raises:
        `ValueError`: If the cassette is malformed.
    """

    def __init__(self, cassette: dict | str | Path) -> None:
        data = _check_cassette(cassette) if isinstance(cassette, dict) else load_cassette(cassette)
        self._interactions: list[dict] = data["interactions"]
        self.requests: list[httpx2.Request] = []
        super().__init__(self._replay)

    def _replay(self, request: httpx2.Request) -> httpx2.Response:
        """Check ``request`` against the next interaction and return its response."""
        self.requests.append(request)
        number, total = len(self.requests), len(self._interactions)
        if number > total:
            raise AssertionError(
                f"unexpected request {number}: the cassette has {total} interaction(s)"
            )
        interaction = self._interactions[number - 1]
        diffs = _request_diffs(interaction["request"], request)
        if diffs:
            shown = diffs[:_MAX_DIFFS]
            if len(diffs) > _MAX_DIFFS:
                shown.append(f"... and {len(diffs) - _MAX_DIFFS} more")
            raise AssertionError(
                f"request {number} does not match the cassette:\n  " + "\n  ".join(shown)
            )
        return _replay_response(interaction["response"])

    def assert_done(self) -> None:
        """Check that every interaction was used.

        Raises:
            `AssertionError`: If some interactions were never requested, naming how many.
        """
        total = len(self._interactions)
        unused = total - len(self.requests)
        if unused > 0:
            raise AssertionError(f"{unused} of {total} interaction(s) were not used")


# --- recording -------------------------------------------------------------------------


class _Tee(httpx2.SyncByteStream):
    """A response body that passes chunks through and keeps a copy of each."""

    def __init__(self, inner: httpx2.SyncByteStream, sink: list[bytes]) -> None:
        self._inner = inner
        self._sink = sink

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._inner:
            self._sink.append(chunk)
            yield chunk

    def close(self) -> None:
        self._inner.close()


class RecordingTransport(httpx2.BaseTransport):
    """Pass requests to `inner` and keep a redacted copy of each exchange.

    Request headers are saved except `IGNORED_HEADERS`; the response is saved as its
    decoded text, without the content-encoding, content-length and transfer-encoding
    headers. The response body is teed as the client reads it, so a stream closed
    early still records what was read. Credential headers and every secret string are
    replaced with `REDACTED` when `cassette` is called.

    Replay treats `REDACTED` as "any value" only when it is a whole query value,
    header value or JSON value. A secret inside a larger value, such as a URL path
    segment or ``pre-<redacted>``, is still redacted but must match literally, so that
    cassette will not replay with a different key.

    Args:
        inner (`httpx2.BaseTransport`): The transport that really sends requests.
        secrets (`Iterable[str]`, *optional*, defaults to `()`): Values to replace with
            `REDACTED` in the URL, header values, JSON body and response text, e.g. the
            API key in use; also caught percent-encoded, plus-encoded and, in response
            text and request body strings, JSON-escaped (including ``\\/`` and
            ``\\uXXXX`` escapes).
    """

    def __init__(self, inner: httpx2.BaseTransport, *, secrets: Iterable[str] = ()) -> None:
        self._inner = inner
        # Longest first, so a secret that contains a shorter one is replaced whole.
        self._secrets = sorted({secret for secret in secrets if secret}, key=len, reverse=True)
        self._exchanges: list[tuple[httpx2.Request, httpx2.Response, list[bytes]]] = []

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        """Send ``request`` through the inner transport and tee its response body.

        Args:
            request (`httpx2.Request`): The request to send.

        Returns:
            `httpx2.Response`: The inner transport's response, with its body teed.
        """
        request.read()
        response = self._inner.handle_request(request)
        sink: list[bytes] = []
        self._exchanges.append((request, response, sink))
        return httpx2.Response(
            response.status_code,
            headers=response.headers,
            stream=_Tee(response.stream, sink),
            extensions=response.extensions,
            request=request,
        )

    def close(self) -> None:
        """Close the inner transport."""
        self._inner.close()

    def _redact(self, text: str, *, json_escaped: bool = False) -> str:
        """Replace each secret in ``text`` with `REDACTED`.

        Args:
            text (`str`): The text to redact.
            json_escaped (`bool`, *optional*, defaults to `False`): Also replace the
                forms a JSON string may hold: escaped as `json.dumps` writes it (with
                or without ASCII escapes, and with ``/`` as ``\\/``) and with every
                character as a ``\\uXXXX`` escape, its hex digits in either case.

        Returns:
            `str`: ``text`` with each secret replaced plain, percent-encoded (``%20``
            for a space, as in a path) and plus-encoded (``+``, as in a query).
        """
        for secret in self._secrets:
            forms = {secret, quote(secret, safe=""), quote_plus(secret, safe="")}
            if json_escaped:
                for escaped in (
                    json.dumps(secret)[1:-1],
                    json.dumps(secret, ensure_ascii=False)[1:-1],
                ):
                    forms.update((escaped, escaped.replace("/", "\\/")))
                units = secret.encode("utf-16-be", errors="surrogatepass")
                codes = [int.from_bytes(units[i : i + 2], "big") for i in range(0, len(units), 2)]
                # Every character \u-escaped, its hex digits in either case.
                escaped_all = "".join(f"\\\\u{code:04x}" for code in codes)
                text = re.sub(escaped_all, REDACTED, text, flags=re.IGNORECASE)
            # Longest first, so a form that contains a shorter one is replaced whole.
            for form in sorted(forms, key=len, reverse=True):
                text = text.replace(form, REDACTED)
        return text

    def _redact_json(self, value: Any) -> Any:
        """Return ``value`` with secrets replaced in every string and key.

        JSON-escaped forms are replaced too, since a string may itself hold JSON, such
        as a tool call's arguments.
        """
        if isinstance(value, dict):
            return {
                self._redact(key, json_escaped=True): self._redact_json(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._redact_json(item) for item in value]
        if isinstance(value, str):
            return self._redact(value, json_escaped=True)
        return value

    def _headers(self, headers: httpx2.Headers, dropped: frozenset[str]) -> dict[str, str]:
        """Save ``headers`` by lowercase name, without ``dropped`` and with credentials redacted."""
        saved: dict[str, str] = {}
        for name, value in headers.multi_items():
            name = name.lower()
            if name in dropped:
                continue
            value = REDACTED if name in CREDENTIAL_HEADERS else self._redact(value)
            # A repeated header keeps every value, joined as HTTP allows.
            saved[name] = f"{saved[name]}, {value}" if name in saved else value
        return saved

    def cassette(self, *, description: str) -> dict:
        """Return the exchanges so far as a redacted ``recorded`` cassette.

        Args:
            description (`str`): What the cassette exercises.

        Returns:
            `dict`: A ``yian.cassette.v1`` cassette with source ``recorded``.

        Raises:
            `ValueError`: If a request body is not JSON.
            `httpx2.DecodingError`: If a response body cannot be decoded.
        """
        interactions = []
        for request, response, sink in self._exchanges:
            saved_request: dict[str, Any] = {
                "method": request.method,
                "url": self._redact(str(request.url)),
                "headers": self._headers(request.headers, IGNORED_HEADERS),
            }
            if request.content:
                saved_request["json"] = self._redact_json(json.loads(request.content))
            encoding = response.headers.get("content-encoding")
            # A throwaway response undoes the content-encoding the way the client did.
            decoded = httpx2.Response(
                200,
                headers={"content-encoding": encoding} if encoding else None,
                content=b"".join(sink),
            ).content
            # Redacted after joining, so a secret split across chunks is still caught.
            text = self._redact(decoded.decode("utf-8", errors="replace"), json_escaped=True)
            interactions.append(
                {
                    "request": saved_request,
                    "response": {
                        "status": response.status_code,
                        "headers": self._headers(response.headers, _DROPPED_RESPONSE_HEADERS),
                        "chunks": _CHUNK.findall(text),
                    },
                }
            )
        return {
            "format": CASSETTE_FORMAT,
            "source": "recorded",
            "description": description,
            "interactions": interactions,
        }


# --- scripted model --------------------------------------------------------------------


def _response_events(response: Response) -> Iterator[Event]:
    """Yield the events a provider would stream for ``response``."""
    for index, part in enumerate(response.message.parts):
        if isinstance(part, TextPart):
            yield TextDelta(index, part.text)
        elif isinstance(part, ReasoningPart):
            yield ReasoningDelta(index, part.text)
        elif isinstance(part, ToolCallPart):
            arguments = part.raw_arguments if part.arguments is None else json.dumps(part.arguments)
            yield ToolCallDelta(index, part.id, part.name, arguments)
        yield PartDone(index, part)
    if response.usage is not None:
        yield UsageReport(response.usage)
    yield Finish(
        response.finish_reason,
        response.response_id,
        protocol=response.message.protocol,
        provider_payload=response.message.provider_payload,
    )


def _scripted_events(items: Sequence[Event | BaseException]) -> Iterator[Event]:
    """Yield ``items`` in order, raising any exception at its position."""
    for item in items:
        if isinstance(item, BaseException):
            raise item
        yield item


class ScriptedModel:
    """A `ChatModel` that answers each request with the next scripted reply.

    A `Response` reply streams as events: per part a delta (text, reasoning, or the
    tool call's arguments as JSON text, its raw text when the arguments did not parse)
    then its `PartDone`, then a `UsageReport` when there is usage, then a `Finish`
    carrying the message's protocol and provider payload, so `collect` of the events
    equals the `Response`. An event sequence reply streams as given, raising any
    exception in it at that position. An exception reply is raised by the call itself.

    Args:
        replies (`Iterable[Response | Sequence[Event | BaseException] | BaseException]`):
            The replies, one per request.
        provider (`str`): The provider name; defaults to ``scripted``.
        model (`str`): The model name; defaults to ``scripted``.

    Attributes:
        requests (`list[Request]`): Every request received, in order.
        calls (`list[str]`): ``stream`` or ``generate`` for each request.
        closed (`int`): How many streams returned by `stream` have been closed.
    """

    def __init__(
        self,
        replies: Iterable[Response | Sequence[Event | BaseException] | BaseException],
        *,
        provider: str = "scripted",
        model: str = "scripted",
    ) -> None:
        self.provider = provider
        self.model = model
        self._replies = deque(replies)
        self.requests: list[Request] = []
        self.calls: list[str] = []
        self.closed = 0

    def _events(self, request: Request, call: str) -> Iterator[Event]:
        """Record the call and return the next reply's events."""
        self.requests.append(request)
        self.calls.append(call)
        if not self._replies:
            number = len(self.requests)
            raise AssertionError(f"ScriptedModel has no reply left for request {number}")
        reply = self._replies.popleft()
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, Response):
            return _response_events(reply)
        return _scripted_events(list(reply))

    def _count_close(self) -> None:
        self.closed += 1

    def stream(self, request: Request) -> EventStream:
        """Answer ``request`` with the next reply's events.

        Args:
            request (`Request`): The call's input; recorded.

        Returns:
            `EventStream`: The reply's events.

        Raises:
            `AssertionError`: If no reply is left.
            `BaseException`: The reply itself, when it is an exception.
        """
        return EventStream(self._events(request, "stream"), close=self._count_close)

    def generate(self, request: Request) -> Response:
        """Answer ``request`` with the next reply, collected.

        Args:
            request (`Request`): The call's input; recorded.

        Returns:
            `Response`: `collect` of the reply's events.

        Raises:
            `AssertionError`: If no reply is left.
            `BaseException`: The reply itself when it is an exception, or an exception
                scripted inside its events.
        """
        return collect(self._events(request, "generate"))

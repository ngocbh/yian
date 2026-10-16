"""Send a model request over a route and read its reply as frames.

The executor retries a request only until the reply's first frame is parsed: status
errors, connection failures and timeouts before that point follow the openai SDK's
retry policy, and nothing after it is retried, since part of the reply has already
reached the caller. Every httpx2 failure becomes a `ProviderError`, with the exact
credentials the route sent replaced in its text.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
import json
import math
import random as _random
import time
from typing import Any, Literal

import httpx2

from yian.llm.errors import ErrorKind, ProviderError, error_from_response
from yian.llm.framing import iter_ndjson, iter_sse
from yian.llm.route import Route
from yian.llm.transport import check_client

Framing = Literal["sse", "ndjson", "json"]

# The same marker as `llm_clients.redact.REDACTED`, which this leaf may not import.
REDACTED = "[REDACTED]"
_FRAMINGS = ("sse", "ndjson", "json")
# The openai and anthropic SDKs send this Accept on every request, streamed ones too.
_DEFAULT_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
_MAX_ERROR_BODY = 64 * 1024
_WAIT_SLICE = 0.25
_NOTHING = object()


def _scrub(text: str, credentials: Iterable[str]) -> str:
    """Replace each credential in ``text`` with `REDACTED`.

    Longest first, so a credential that contains a shorter one is replaced whole.
    """
    for value in sorted(set(credentials), key=len, reverse=True):
        if value:
            text = text.replace(value, REDACTED)
    return text


def _scrub_error(error: ProviderError, credentials: Iterable[str]) -> ProviderError:
    """Replace each credential in ``error``'s text, which lives in ``message`` and ``args``.

    The body is scrubbed before parsing too, but a JSON-escaped echo (``\\/`` for ``/``)
    only matches a credential once the parse has undone the escape.
    """
    error.message = _scrub(error.message, credentials)
    error.args = (error.message,)
    return error


def _encode_body(body: Mapping) -> bytes:
    """Serialize ``body`` as UTF-8 JSON without NaN or infinity.

    Args:
        body (`Mapping`): The JSON request body.

    Returns:
        `bytes`: The encoded body.

    Raises:
        `ValueError`: If ``body`` is not JSON or holds an unpaired surrogate; unchained,
            so the serialized body the encoder's error holds is not kept.
    """
    try:
        return json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except UnicodeEncodeError as exc:
        problem = f"request body has an unpaired surrogate at position {exc.start}"
    except (TypeError, ValueError, RecursionError) as exc:
        problem = f"request body is not JSON: {exc}"
    # Raised outside the except blocks so the encoder's error is not the context.
    raise ValueError(problem)


def _http_error(
    exc: httpx2.HTTPError, route: Route, *, streaming: bool
) -> ProviderError:
    """Map an httpx2 failure to a `ProviderError`.

    Timeouts and other transport errors are retryable only before the first frame;
    decoding errors and any other httpx2 error never are.

    Args:
        exc (`httpx2.HTTPError`): The failure.
        route (`Route`): The route the request used, for the provider and credentials.
        streaming (`bool`): Whether a frame was already parsed.

    Returns:
        `ProviderError`: The error to raise, ``from exc``.
    """
    if isinstance(exc, httpx2.DecodingError):
        kind, retryable = ErrorKind.PROTOCOL, False
    elif isinstance(exc, httpx2.TimeoutException):
        kind, retryable = ErrorKind.TIMEOUT, not streaming
    elif isinstance(exc, httpx2.TransportError):
        kind, retryable = ErrorKind.CONNECTION, not streaming
    else:
        kind, retryable = ErrorKind.CONNECTION, False
    message = f"{route.provider} request failed ({kind.value}): {type(exc).__name__}: {exc}"
    return ProviderError(
        kind,
        _scrub(message, route.auth.credentials()),
        retryable=retryable,
        provider=route.provider,
    )


class FrameStream(Iterator[Any]):
    """The frames of one reply; owns the HTTP response and closes it when done.

    Yields `SSEEvent` values for ``sse`` framing, one JSON value per line for
    ``ndjson``, and the single parsed body for ``json``. The first frame is parsed
    before the stream is returned, so failures up to that point are raised by
    `Executor.open`. Closing is idempotent and also happens on exhaustion and on error.

    Args:
        response (`httpx2.Response`): The open, streaming 2xx response.
        framing (`Framing`): How to split the body.
        route (`Route`): The route the request used, for errors.
        check_stopped (`Callable[[], None]`): Called on every chunk read and before
            each frame is returned; raises to stop.
    """

    def __init__(
        self,
        response: httpx2.Response,
        framing: Framing,
        route: Route,
        check_stopped: Callable[[], None],
    ) -> None:
        self._response = response
        self._route = route
        self._check_stopped = check_stopped
        self._frames = self._parse(framing)
        self._pending: Any = _NOTHING
        self._streaming = False
        self._closed = False

    def _chunks(self) -> Iterator[bytes]:
        """Yield the decoded body chunks, checking for a stop on each."""
        # iter_bytes, not iter_raw, so content-encoding is undone and a bad encoding
        # raises DecodingError.
        for chunk in self._response.iter_bytes():
            self._check_stopped()
            yield chunk

    def _parse(self, framing: Framing) -> Iterator[Any]:
        """Yield the body's frames; a blank ``json`` body yields none."""
        if framing == "sse":
            yield from iter_sse(self._chunks())
        elif framing == "ndjson":
            yield from iter_ndjson(self._chunks())
        else:
            body = b"".join(self._chunks())
            if not body.strip():
                return
            try:
                value = json.loads(body)
            except (ValueError, RecursionError):
                value = _NOTHING
            # Raised outside the except block so the decode error, whose `doc` holds the
            # body, is not chained.
            if value is _NOTHING:
                raise ProviderError(
                    ErrorKind.PROTOCOL,
                    f"{self._route.provider} returned an invalid JSON body",
                    provider=self._route.provider,
                )
            yield value

    def _prefetch(self) -> None:
        """Parse the first frame and hold it for the first `__next__`."""
        try:
            self._pending = self._next_frame()
        except StopIteration:
            pass

    def _next_frame(self) -> Any:
        """Return the next frame, closing the stream and mapping the error on failure."""
        try:
            frame = next(self._frames)
        except ProviderError as error:
            self.close()
            if error.provider is None:
                error.provider = self._route.provider
            raise
        except httpx2.HTTPError as exc:
            self.close()
            raise _http_error(exc, self._route, streaming=self._streaming) from exc
        except BaseException:
            self.close()
            raise
        self._streaming = True
        return frame

    def __iter__(self) -> FrameStream:
        """Return the stream itself."""
        return self

    def __next__(self) -> Any:
        """Return the next frame.

        Returns:
            `Any`: An `SSEEvent`, an NDJSON value, or the parsed JSON body.

        Raises:
            `StopIteration`: The body ended or the stream was closed.
            `ProviderError`: The connection failed, timed out, or sent a body that could
                not be decoded or framed; never retried at this point.
        """
        if self._pending is not _NOTHING:
            frame, self._pending = self._pending, _NOTHING
        elif self._closed:
            raise StopIteration
        else:
            frame = self._next_frame()
        try:
            self._check_stopped()
        except BaseException:
            self.close()
            raise
        return frame

    def close(self) -> None:
        """Close the HTTP response; later calls do nothing."""
        if self._closed:
            return
        self._closed = True
        self._pending = _NOTHING
        try:
            self._frames.close()
        finally:
            self._response.close()

    def __enter__(self) -> FrameStream:
        """Return the stream itself."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the stream."""
        self.close()


class Executor:
    """Send requests over routes with a shared client, retrying before the first frame.

    Args:
        client (`httpx2.Client`): The client to send with; it must not follow redirects.
        check_stopped (`Callable[[], None] | None`): Called before each attempt, between
            wait slices, on every chunk read and before each frame; raises to stop.
            Defaults to None (never stops).
        sleep (`Callable[[float], None]`): Waits between attempts; defaults to
            `time.sleep`.
        random (`Callable[[], float]`): A value in [0, 1) for backoff jitter; defaults to
            `random.random`.

    Raises:
        `ValueError`: If ``client`` follows redirects.
    """

    def __init__(
        self,
        client: httpx2.Client,
        *,
        check_stopped: Callable[[], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random: Callable[[], float] = _random.random,
    ) -> None:
        check_client("client", client)
        self._client = client
        self._check_stopped = check_stopped or (lambda: None)
        self._sleep = sleep
        self._random = random

    def open(self, route: Route, body: Mapping, *, framing: Framing) -> FrameStream:
        """POST ``body`` as JSON over ``route`` and return the reply's frames.

        The request sends ``Content-Type`` and ``Accept`` of ``application/json`` for
        every framing, as the openai and anthropic SDKs do; ``route.headers`` override
        them, and the auth headers, resolved again on each attempt, override both. A
        retryable status, connection failure or timeout before the first frame is
        retried up to ``route.max_retries`` times, waiting the server's retry-after when it allows
        a retry, else an exponential backoff with jitter capped at 8 seconds.

        Args:
            route (`Route`): Where to send the request and how to retry it.
            body (`Mapping`): The JSON request body.
            framing (`Framing`): How to split the reply: ``"sse"``, ``"ndjson"`` or
                ``"json"``.

        Returns:
            `FrameStream`: The reply's frames, with the first one already parsed; empty
            when a 2xx body ends before any frame.

        Raises:
            `ValueError`: If ``framing`` is unknown, ``body`` is not JSON-serializable
                without NaN or infinity or holds an unpaired surrogate, or an auth header
                value is invalid. Raised before any request.
            `ProviderError`: If the request fails and is not retried (any more), the
                server redirects, or the reply cannot be decoded or framed.
        """
        if framing not in _FRAMINGS:
            raise ValueError(f"unknown framing {framing!r}; expected sse, ndjson or json")
        content = _encode_body(body)
        retries = 0
        while True:
            self._check_stopped()
            try:
                return self._attempt(route, content, framing)
            except ProviderError as error:
                if not error.retryable or retries >= route.max_retries:
                    raise
                self._wait(self._delay(error, route, retries))
                retries += 1

    def _attempt(self, route: Route, content: bytes, framing: Framing) -> FrameStream:
        """Send one request and parse up to the first frame."""
        headers = httpx2.Headers(_DEFAULT_HEADERS)
        # Each update replaces earlier names case-insensitively, so a route header
        # overrides a framing default and auth wins over a route header such as
        # "authorization".
        headers.update(route.headers)
        headers.update(route.auth.headers())
        extra: dict[str, Any] = {}
        if route.timeout is not None:
            extra["timeout"] = route.timeout
        request = self._client.build_request(
            "POST", route.url(), params=route.query, headers=headers, content=content, **extra
        )
        try:
            response = self._client.send(request, stream=True, follow_redirects=False)
        except httpx2.HTTPError as exc:
            raise _http_error(exc, route, streaming=False) from exc
        status = response.status_code
        if 300 <= status < 400:
            response.close()
            error = error_from_response(
                status, response.headers, "", provider=route.provider,
                max_retry_after=route.max_retry_after,
            )
            # A redirect is never retried, whatever x-should-retry says.
            error.retryable = False
            raise _scrub_error(error, route.auth.credentials())
        if not 200 <= status < 300:
            credentials = route.auth.credentials()
            text = _scrub(self._error_body(response), credentials)
            error = error_from_response(
                status, response.headers, text, provider=route.provider,
                max_retry_after=route.max_retry_after,
            )
            raise _scrub_error(error, credentials)
        stream = FrameStream(response, framing, route, self._check_stopped)
        stream._prefetch()
        return stream

    def _error_body(self, response: httpx2.Response) -> str:
        """Read at most 64 KiB of an error reply's body and close it."""
        parts: list[bytes] = []
        size = 0
        try:
            for chunk in response.iter_bytes():
                self._check_stopped()
                parts.append(chunk)
                size += len(chunk)
                if size >= _MAX_ERROR_BODY:
                    break
        except httpx2.HTTPError:
            # The status already names the failure; keep whatever body was read.
            pass
        finally:
            response.close()
        return b"".join(parts)[:_MAX_ERROR_BODY].decode("utf-8", errors="replace")

    def _delay(self, error: ProviderError, route: Route, retries: int) -> float:
        """Return the wait before the next attempt, as the openai SDK computes it."""
        retry_after = error.retry_after
        if (
            retry_after is not None
            and math.isfinite(retry_after)
            and 0 < retry_after
            and (route.max_retry_after is None or retry_after <= route.max_retry_after)
        ):
            return retry_after
        # The exponent is capped as openai does, so a huge retry count cannot overflow.
        return min(0.5 * 2.0 ** min(retries, 1000), 8.0) * (1 - 0.25 * self._random())

    def _wait(self, seconds: float) -> None:
        """Sleep ``seconds`` in short slices, checking for a stop before each."""
        remaining = seconds
        # The tolerance absorbs float residue so no near-zero extra slice is slept.
        while remaining > 1e-9:
            self._check_stopped()
            step = min(_WAIT_SLICE, remaining)
            self._sleep(step)
            remaining -= step

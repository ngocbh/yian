"""Provider errors: one exception type, a kind per failure, and openai's retry policy.

`error_from_response` turns a non-2xx HTTP reply into a `ProviderError` whose kind
comes from the status (and, for overload and context overflow, the body) and whose
`retryable` flag follows the openai SDK's `_should_retry`. `parse_retry_after` reads
the server's requested delay the way that SDK does.
"""

from __future__ import annotations

from collections.abc import Mapping
import email.utils
from enum import Enum
import json
import math
import time
from typing import Any
from urllib.parse import urlsplit

# openai's MAX_RETRY_AFTER_DELAY + 1: an overflowing delay stays finite, so the default
# 120 s refusal still applies to it.
_OVERFLOW_RETRY_AFTER = 121.0
_INFINITY_LITERALS = frozenset({"inf", "+inf", "infinity", "+infinity"})
_CONTEXT_PHRASES = ("context_length_exceeded", "prompt is too long", "maximum context length")
_MAX_DETAIL_CHARS = 2000


class ErrorKind(str, Enum):
    """What went wrong with a model call, independent of the provider."""

    AUTH = "auth"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    BAD_REQUEST = "bad_request"
    CONTEXT_LENGTH = "context_length"
    SERVER = "server"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    REDIRECT = "redirect"
    INCOMPLETE = "incomplete"
    PROTOCOL = "protocol"


# Statuses whose kind does not depend on the body.
_STATUS_KINDS = {
    401: ErrorKind.AUTH,
    403: ErrorKind.PERMISSION,
    404: ErrorKind.NOT_FOUND,
    408: ErrorKind.TIMEOUT,
    429: ErrorKind.RATE_LIMIT,
    529: ErrorKind.OVERLOADED,
}


class ProviderError(Exception):
    """A model call failed.

    The text lives only in ``args[0]`` and ``message``, which callers rewrite to scrub
    credentials; the error keeps no raw response body.

    Args:
        kind (`ErrorKind`): What went wrong.
        message (`str`): Human-readable text; ``str(error)`` returns it.
        status (`int | None`, *optional*, defaults to `None`): The HTTP status, when the
            provider replied.
        retry_after (`float | None`, *optional*, defaults to `None`): Seconds the
            provider asked to wait, if any.
        retryable (`bool`, *optional*, defaults to `False`): Whether retrying the same
            request may succeed.
        provider (`str | None`, *optional*, defaults to `None`): The provider name, for
            example ``"openai"``.
    """

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        retryable: bool = False,
        provider: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = ErrorKind(kind)
        self.message = message
        self.status = status
        self.retry_after = retry_after
        self.retryable = retryable
        self.provider = provider

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild from kind and message, then restore the other attributes.

        `Exception`'s default reduction passes only ``args`` to the constructor, which
        loses the keyword-only arguments; the attributes are restored as state.
        """
        return (type(self), (self.kind, self.message), dict(self.__dict__))


def parse_retry_after(headers: Mapping[str, str], *, now: float | None = None) -> float | None:
    """Return the delay in seconds a reply asks for before a retry.

    ``retry-after-ms`` (milliseconds) is preferred, then ``retry-after`` in seconds;
    either may be float text. A number too large for a float gives ``121.0``, just
    over the default 120 s limit, so a retry is refused by default; with
    ``max_retry_after=None`` that sentinel means a 121 s wait. Otherwise
    ``retry-after`` is read as an HTTP date, in seconds from ``now`` (negative when
    the date has passed). A literal ``inf`` or ``nan`` is returned as is.

    Args:
        headers (`Mapping[str, str]`): The reply headers; names match case-insensitively.
        now (`float | None`, *optional*, defaults to `None`): Current time in epoch
            seconds for the HTTP-date form; `None` uses `time.time()`.

    Returns:
        `float | None`: The delay, or `None` when the header is absent or unparsable.
    """
    for name, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = _header(headers, name)
        if value is None:
            continue
        try:
            delay = float(value)
        except ValueError:
            continue
        if delay == math.inf and value.strip().lower() not in _INFINITY_LITERALS:
            return _OVERFLOW_RETRY_AFTER
        return delay / divisor
    value = _header(headers, "retry-after")
    if value is None:
        return None
    try:
        parsed = email.utils.parsedate_tz(value)
        if parsed is None:
            return None
        when = email.utils.mktime_tz(parsed)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return float(when - (time.time() if now is None else now))


def error_from_response(
    status: int,
    headers: Mapping[str, str],
    body: str,
    *,
    provider: str,
    max_retry_after: float | None = 120.0,
) -> ProviderError:
    """Build the error for a non-2xx HTTP reply.

    The kind comes from the status:

    - 401 auth, 403 permission, 404 not found, 408 timeout, 429 rate limit.
    - 529, or a 503 whose body says "overloaded": overloaded.
    - Any other 5xx: server.
    - 3xx: redirect.
    - A 400, 413 or 422 whose body names a context overflow: context length.
    - Any other 4xx: bad request.

    ``retryable`` follows openai's `_should_retry`: a finite retry-after over
    ``max_retry_after`` refuses, then ``x-should-retry`` decides, then 408, 409, 429
    and 5xx retry.

    Args:
        status (`int`): The HTTP status.
        headers (`Mapping[str, str]`): The reply headers; names match case-insensitively.
        body (`str`): The reply body text, already stripped of credentials.
        provider (`str`): The provider name, used in the message.
        max_retry_after (`float | None`, *optional*, defaults to `120.0`): The longest
            retry-after that still allows a retry; `None` allows any.

    Returns:
        `ProviderError`: The error, with ``status``, ``retry_after``, ``retryable``
        and ``provider`` set.
    """
    kind = _kind(status, body)
    retry_after = parse_retry_after(headers)
    if kind is ErrorKind.REDIRECT:
        detail = _redirect_detail(_header(headers, "location"))
    else:
        detail = _detail(body)
    return ProviderError(
        kind,
        f"{provider} returned HTTP {status} ({kind.value}): {detail}",
        status=status,
        retry_after=retry_after,
        retryable=_retryable(status, headers, retry_after, max_retry_after),
        provider=provider,
    )


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Return the header ``name``, or `None` when absent.

    ``name`` must be lower case; header names match case-insensitively.
    """
    value = headers.get(name)
    if value is not None:
        return value
    for key, item in headers.items():
        if key.lower() == name:
            return item
    return None


def _kind(status: int, body: str) -> ErrorKind:
    """Return the error kind for ``status`` and, where it matters, ``body``."""
    if status in _STATUS_KINDS:
        return _STATUS_KINDS[status]
    lowered = body.lower()
    if status == 503 and "overloaded" in lowered:
        return ErrorKind.OVERLOADED
    if status >= 500:
        return ErrorKind.SERVER
    if 300 <= status < 400:
        return ErrorKind.REDIRECT
    if status in (400, 413, 422) and any(phrase in lowered for phrase in _CONTEXT_PHRASES):
        return ErrorKind.CONTEXT_LENGTH
    if 400 <= status < 500:
        return ErrorKind.BAD_REQUEST
    # A 1xx or 2xx never reaches here from the executor; treat it as a broken reply.
    return ErrorKind.PROTOCOL


def _retryable(
    status: int,
    headers: Mapping[str, str],
    retry_after: float | None,
    max_retry_after: float | None,
) -> bool:
    """Return whether the reply allows a retry, in the order openai checks."""
    if (
        max_retry_after is not None
        and retry_after is not None
        and math.isfinite(retry_after)
        and retry_after > max_retry_after
    ):
        return False
    should_retry = _header(headers, "x-should-retry")
    if should_retry == "true":
        return True
    if should_retry == "false":
        return False
    return status in (408, 409, 429) or status >= 500


def _detail(body: str) -> str:
    """Return the most useful short text of an error body."""
    detail = _json_detail(body) or " ".join(body.split()) or "no body"
    if len(detail) > _MAX_DETAIL_CHARS:
        detail = detail[:_MAX_DETAIL_CHARS] + "…"
    return detail


def _json_detail(body: str) -> str | None:
    """Return the body's JSON ``error.message``, ``message`` or ``detail`` string."""
    try:
        data = json.loads(body)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    candidates = [error.get("message") if isinstance(error, dict) else None,
                  data.get("message"), data.get("detail")]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _redirect_detail(location: str | None) -> str:
    """Name a redirect's target by scheme and host only.

    The path, query and user info can carry tokens, so they never reach the message.
    """
    if not location:
        return "redirect without a Location header"
    try:
        parts = urlsplit(location.strip())
        host = parts.hostname
        port = parts.port
    except ValueError:
        return "redirect to an unparsable Location"
    if not host:
        return "redirect to a relative Location"
    if not parts.scheme:
        return "redirect to a Location without a scheme"
    if ":" in host:
        host = f"[{host}]"
    return f"redirect to {parts.scheme}://{host}" + (f":{port}" if port is not None else "")

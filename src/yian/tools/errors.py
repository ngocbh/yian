"""Errors a host tool raises to steer a TOOL node's retries, and yian's own check error.

A TOOL node with `retries:` retries a failed call unless the tool raised
``PermanentToolError`` or a check yian runs failed (``ToolCheckError``). A tool raises
``TransientToolError(retry_after=...)`` to say how long to wait before the next attempt.

Uses only the Python standard library.
"""

from __future__ import annotations

import math


class PermanentToolError(Exception):
    """A tool call failed in a way another attempt can't fix; TOOL retries stop.

    Example:
        ```python
        def place(order: dict) -> str:
            if order["qty"] <= 0:
                raise PermanentToolError("quantity must be positive")
            ...
        ```
    """


class TransientToolError(Exception):
    """A tool call failed in a way a later attempt may not; TOOL retries apply.

    Args:
        *args (`object`):
            The exception's arguments, usually one message.
        retry_after (`float | None`, *optional*, defaults to `None`):
            Seconds to wait before the next attempt, finite and at least 0. It
            replaces the computed backoff, up to the retry delay cap. `None` uses
            the computed backoff.

    Raises:
        `TypeError`:
            Raised when `retry_after` is not an int, a float, or `None` (a bool is
            refused).
        `ValueError`:
            Raised when `retry_after` is negative or not finite.

    Example:
        ```python
        def quote(symbol: str) -> float:
            response = fetch(symbol)
            if response.status == 429:
                raise TransientToolError("rate limited", retry_after=5)
            ...
        ```
    """

    def __init__(self, *args: object, retry_after: float | None = None) -> None:
        if retry_after is not None:
            if isinstance(retry_after, bool) or not isinstance(retry_after, (int, float)):
                raise TypeError(
                    f"retry_after must be a number of seconds or None, "
                    f"not {type(retry_after).__name__}"
                )
            # An int is always finite, and math.isfinite overflows on a huge one.
            if isinstance(retry_after, float) and not math.isfinite(retry_after):
                raise ValueError(f"retry_after must be finite, not {retry_after!r}")
            if retry_after < 0:
                raise ValueError(f"retry_after must be at least 0, not {retry_after!r}")
        super().__init__(*args)
        self.retry_after = retry_after


class ToolCheckError(TypeError):
    """A check yian runs on a tool call's result failed; TOOL retries never retry it."""


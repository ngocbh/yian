"""Errors a host tool raises to report a failure or steer retries, and yian's own check error.

A tool raises `ReportedToolError` for a failure an AGENT's model should read and act
on; in a TOOL node it fails the node like any other error. A TOOL node with
`retries:` retries a failed call unless the tool raised `PermanentToolError` or a
check yian runs failed (`ToolCheckError`). A tool raises
`TransientToolError(retry_after=...)` to say how long to wait before the next attempt.

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


class ReportedToolError(Exception):
    """A tool call failed in a way the calling model should read and act on.

    In an AGENT's tool loop, the model reads `str(exc)` as the call's result (an
    error tool message, or the exception's type name when the text is empty) and the
    loop continues. In a TOOL node it is an ordinary failure: the node fails, and
    `retries:` retries it unless it is also a `PermanentToolError`. The message goes
    to the model provider and the session transcript, so it must not hold secrets.

    Example:
        ```python
        import os

        def read_file(path: str) -> str:
            if not os.path.exists(path):
                raise ReportedToolError(f"no such file: {path}")
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


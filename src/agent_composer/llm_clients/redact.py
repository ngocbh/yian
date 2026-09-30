"""Remove a host API key from provider errors before anything records them.

Providers can echo a rejected key in their error text, in full or masked with a
visible prefix and suffix (OpenAI's 401 says ``Incorrect API key provided:
sk-abc****wxyz``). Callers that hold a host key scrub the exceptions their model
calls raise, so failure messages and tracebacks never carry any part of it.
"""

from __future__ import annotations

import re
from typing import Optional, Union

from pydantic import SecretStr

REDACTED = "[REDACTED]"

# A masked echo: optional visible head, a run of mask characters, optional visible tail.
# The lookbehind starts a head only at the start of a run, which keeps matching linear.
_MASKED = re.compile(r"(?<![A-Za-z0-9_\-])([A-Za-z0-9_\-]*)(?:\*{2,}|\.{3}|…)([A-Za-z0-9_\-]*)")


def _plain(key: Union[str, SecretStr, None]) -> Optional[str]:
    """Return the key's plain text, unwrapping a `SecretStr`."""
    return key.get_secret_value() if isinstance(key, SecretStr) else key


def redact_key(text: str, key: Union[str, SecretStr, None]) -> str:
    """Replace ``key`` and masked echoes of it in ``text`` with ``[REDACTED]``.

    A masked echo counts when its visible head is a prefix of the key or its
    visible tail is a suffix of it; other masked text is left alone.

    Args:
        text (`str`): Text that may contain the key.
        key (`str | SecretStr | None`): The key; `None` or empty returns ``text``.

    Returns:
        `str`: ``text`` without the key or its masked fragments.
    """
    plain = _plain(key)
    if not plain or not text:
        return text
    text = text.replace(plain, REDACTED)

    def masked(match: re.Match[str]) -> str:
        head, tail = match.group(1), match.group(2)
        if (head and plain.startswith(head)) or (tail and plain.endswith(tail)):
            return REDACTED
        return match.group(0)

    return _MASKED.sub(masked, text)


def scrub_key(exc: BaseException, key: Union[str, SecretStr, None]) -> None:
    """Redact ``key`` in place from ``exc``, its chained causes, and group members.

    String ``args``, a writable string ``message`` attribute, and ``__notes__``
    are rewritten, which covers what ``str`` and tracebacks show for ordinary
    exceptions, including provider SDK errors. An exception group's own message
    is read-only and stays as raised; its members are scrubbed.

    Args:
        exc (`BaseException`): The raised exception.
        key (`str | SecretStr | None`): The key; `None` or empty does nothing.
    """
    plain = _plain(key)
    if not plain:
        return
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        # A rewrite that fails must neither stop the walk nor replace the provider error.
        try:
            _rewrite(current, plain)
        except Exception:
            pass
        pending.extend(_linked(current))


def _rewrite(exc: BaseException, plain: str) -> None:
    """Redact one exception's string ``args``, ``message``, and ``__notes__``."""
    try:
        exc.args = tuple(redact_key(arg, plain) if isinstance(arg, str) else arg for arg in exc.args)
    except Exception:
        pass
    try:
        message = getattr(exc, "message", None)
        if isinstance(message, str):
            exc.message = redact_key(message, plain)
    except Exception:
        pass
    notes = getattr(exc, "__notes__", None)
    if isinstance(notes, list):
        notes[:] = [redact_key(note, plain) if isinstance(note, str) else note for note in notes]


def _linked(exc: BaseException) -> list[BaseException]:
    """Return the exceptions chained to ``exc`` and, for a group, its members."""
    linked = [exc.__cause__, exc.__context__]
    try:
        linked.extend(getattr(exc, "exceptions", None) or ())
    except Exception:
        pass
    return [item for item in linked if isinstance(item, BaseException)]

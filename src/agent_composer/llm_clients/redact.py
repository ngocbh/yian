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
_MASKED = re.compile(r"([A-Za-z0-9_\-]*)(?:\*{2,}|\.{3}|…)([A-Za-z0-9_\-]*)")


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
    pending = [exc]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        try:
            current.args = tuple(
                redact_key(arg, plain) if isinstance(arg, str) else arg for arg in current.args
            )
        except (AttributeError, TypeError):
            pass
        message = getattr(current, "message", None)
        if isinstance(message, str):
            try:
                current.message = redact_key(message, plain)
            except (AttributeError, TypeError):
                pass
        notes = getattr(current, "__notes__", None)
        if isinstance(notes, list):
            notes[:] = [redact_key(note, plain) if isinstance(note, str) else note for note in notes]
        pending.extend((current.__cause__, current.__context__))
        pending.extend(getattr(current, "exceptions", None) or ())

"""Remove a host API key from provider errors before anything records them.

Providers can echo a rejected key in their error text, in full or masked with a
visible prefix and suffix (OpenAI's 401 says ``Incorrect API key provided:
sk-abc****wxyz``). Callers that hold a host key scrub the exceptions their model
calls raise, so failure messages and tracebacks never carry any part of it.
"""

from __future__ import annotations

import functools
import re
from typing import Optional, Union

from pydantic import SecretStr

REDACTED = "[REDACTED]"

# A masked echo counts only when it shows at least this many key characters on one side.
_MIN_VISIBLE = 3
_WORD = "A-Za-z0-9_\\-"
_MARK = re.compile(r"\*{2,}|\.{3}|…")


@functools.lru_cache(maxsize=32)
def _masked_pattern(extra: str) -> re.Pattern[str]:
    """Match a masked echo: visible head, mask marks, visible tail.

    Visible characters are word characters plus ``extra``, the key's other
    characters. The lookbehind starts a head only at the start of a run, which
    keeps matching linear.
    """
    visible = f"[{_WORD}{re.escape(extra)}]"
    return re.compile(f"(?<!{visible}){visible}*(?:{_MARK.pattern}){visible}*")


def _plain(key: Union[str, SecretStr, None]) -> Optional[str]:
    """Return the key's plain text, unwrapping a `SecretStr`."""
    return key.get_secret_value() if isinstance(key, SecretStr) else key


def _prefix_start(head: str, key: str) -> Optional[int]:
    """Return where the longest suffix of ``head`` that begins ``key`` starts, if long enough."""
    for size in range(min(len(head), len(key)), _MIN_VISIBLE - 1, -1):
        if key.startswith(head[-size:]):
            return len(head) - size
    return None


def _suffix_stop(tail: str, key: str) -> Optional[int]:
    """Return where the longest prefix of ``tail`` that ends ``key`` stops, if long enough."""
    for size in range(min(len(tail), len(key)), _MIN_VISIBLE - 1, -1):
        if key.endswith(tail[:size]):
            return size
    return None


def redact_key(text: str, key: Union[str, SecretStr, None]) -> str:
    """Replace ``key`` and masked echoes of it in ``text`` with ``[REDACTED]``.

    A masked echo is visible text around ``**``, ``...``, or ``…``. It counts
    when at least three visible characters before the mask begin the key or at
    least three after it end the key; the matching text and the mask are
    replaced, and other masked text is left alone. Visible characters are
    letters, digits, ``_``, ``-``, and any other character in the key, except a
    mask character. A key that contains ``..`` or ``**`` can leave fragments.

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
    extra = "".join(sorted({
        char for char in plain
        if not (char.isascii() and (char.isalnum() or char in "_-")) and char not in "*…"
    }))

    def masked(match: re.Match[str]) -> str:
        token = match.group(0)
        marks = list(_MARK.finditer(token))
        head, tail = token[:marks[0].start()], token[marks[-1].end():]
        start, stop = _prefix_start(head, plain), _suffix_stop(tail, plain)
        if start is None and stop is None:
            return token
        # Text on a side that does not match the key is kept only when the other side matched.
        left = "" if start is None else head[:start]
        right = tail if stop is None else tail[stop:]
        return left + REDACTED + right

    return _masked_pattern(extra).sub(masked, text)


def scrub_key(exc: BaseException, key: Union[str, SecretStr, None]) -> None:
    """Redact ``key`` in place from ``exc``, its chained causes, and group members.

    On ``exc``, every ``__cause__`` and ``__context__``, and every exception
    group member, this rewrites string ``args``, a writable string ``message``
    attribute, and ``__notes__``. Other attributes, such as an SDK error's
    response body, are not rewritten. An exception group's own message is
    read-only and stays as raised. An exception that cannot be rewritten is
    skipped.

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

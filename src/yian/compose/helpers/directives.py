"""Recognize a whole-value `call(...)` directive in binding text."""

from __future__ import annotations

import re

from yian.expr import ExpressionError
from yian.expr.template import _find_paren_end, _split_calls_aware
from yian.secret_names import is_secret_name

# An unquoted value whose trimmed text starts with this is a `call(...)` directive.
CALL_HEAD = re.compile(r"call\s*\(")
_FLOW_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
# A name directly before `=`, anywhere in a directive's raw text.
_NAME_BEFORE_EQUALS = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)\s*=")


def holds_secret_keyword(text: str) -> bool:
    """Return whether `text` writes a name under a secret name directly before `=`.

    A malformed `call(...)` whose text holds such a keyword may hold a credential in a
    part the parse never reached, so its error must not quote the text.

    Args:
        text (`str`): The raw directive text.

    Returns:
        `bool`: `True` when some `name=` in `text` has a secret name.
    """
    return any(is_secret_name(m.group(1)) for m in _NAME_BEFORE_EQUALS.finditer(text))


def quoted_unless_secret(part: str, text: str, lead: str = " ") -> str:
    """Return `part` quoted after `lead` for an error message, or `""` when `text`
    holds a secret keyword."""
    return "" if holds_secret_keyword(text) else f"{lead}{part!r}"


def match_call_directive(value: str) -> tuple[str, str] | None:
    """Recognize a whole-value `call(...)` directive.

    A value whose trimmed text starts with `call(` must be exactly one directive: its
    parentheses balance, nothing follows the closing `)`, and the first argument is a
    bare flow id (an identifier, hyphens allowed). A positional argument after the flow
    id is kept in the remaining text, so the keyword-argument parser rejects it with
    its own message.

    Args:
        value (`str`): The binding text.

    Returns:
        `tuple[str, str] | None`: The flow id and the raw argument text after the
            flow id's comma (empty for `call(f)`), or `None` when the trimmed text
            does not start with `call(`.

    Raises:
        `ExpressionError`: The value starts with `call(` but its parentheses are
            unbalanced, text follows the closing `)`, or the first argument is not a
            bare flow id. The message quotes no part of a value that
            `holds_secret_keyword`.
    """
    s = value.strip()
    m = CALL_HEAD.match(s)
    if m is None:
        return None
    open_idx = m.end() - 1
    close_idx = _find_paren_end(s, open_idx + 1)
    if close_idx is None:
        raise ExpressionError("`call(` is never closed (check its parentheses and quotes)")
    trailing = s[close_idx + 1 :].strip()
    if trailing:
        raise ExpressionError(
            f"text follows the closing `)`{quoted_unless_secret(trailing, s, ': ')}; "
            "a call(...) must be the whole value"
        )
    args_str = s[open_idx + 1 : close_idx]
    first = _split_calls_aware(args_str, ",")[0].strip()
    if not _FLOW_ID.fullmatch(first):
        raise ExpressionError(
            "a call(...) must start with a flow id (a bare name) and a comma"
            f"{quoted_unless_secret(first, s, ', got ')}"
        )
    # Everything after the first comma is the remaining (keyword) args; the loop over
    # this substring never re-sees the positional flow id, so a positional arg among
    # the remaining args (e.g. `call(f, 30)`) fails loudly there.
    comma = args_str.find(",")
    remaining = args_str[comma + 1 :] if comma != -1 else ""
    return first, remaining

"""Find a literal value written under a secret name, and word the load error for it.

Every authored value is part of a flow's identity, so a credential must not be written
in the flow file. Under a name `yian.secret_names.is_secret_name` accepts, a value must
be a binding (`${...}` text) or `null`. Containers under other names are walked, so a
secret key nested in a literal mapping or list is found too. The message builders name
the slot and the fix and never include the value, so every slot that applies the rule
reports it the same way.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from yian.expr import ExpressionError, scan_template
from yian.expr.template import Span
from yian.secret_names import is_secret_name, normalize_name

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NON_IDENTIFIER_CHAR = re.compile(r"[^A-Za-z0-9_]")


def is_binding_text(value: Any) -> bool:
    """Return whether `value` is template text holding at least one `${...}` span.

    Args:
        value (`Any`): An authored value.

    Returns:
        `bool`: `True` for a string with a well-formed `${...}` span. Plain text, a
            `$${...}` escape (literal text) and malformed template text are `False`.
    """
    if not isinstance(value, str):
        return False
    try:
        return any(isinstance(segment, Span) for segment in scan_template(value))
    except ExpressionError:
        return False


def secret_literal_path(
    name: Any,
    value: Any,
    *,
    is_binding: Callable[[Any], bool] = is_binding_text,
) -> tuple[str | int, ...] | None:
    """Locate the first literal value written under a secret name.

    Under a secret `name` the value must be `None` or pass `is_binding`. Under any
    other name, mappings and lists are walked in authored order: a string key is the
    name of its value, while a non-string key or a list index names nothing. A
    container already on the walk (a recursive YAML alias) is skipped.

    Args:
        name (`Any`): The slot's name, or `None` when the slot has no name.
        value (`Any`): The authored value.
        is_binding (`Callable[[Any], bool]`, *optional*, defaults to
            `is_binding_text`): Whether a value at the slot itself is a binding.
            Nested values always use `is_binding_text`.

    Returns:
        `tuple[str | int, ...] | None`: `()` when the slot itself holds the literal,
            the keys and indexes from the slot down to the offending secret key for a
            nested one, or `None` when there is none.
    """
    return _walk(name, value, is_binding, on_walk=set())


def _walk(
    name: Any,
    value: Any,
    is_binding: Callable[[Any], bool],
    *,
    on_walk: set[int],
) -> tuple[str | int, ...] | None:
    """Recursive step of `secret_literal_path`; `on_walk` holds the open containers."""
    if isinstance(name, str) and is_secret_name(name):
        return None if value is None or is_binding(value) else ()
    if isinstance(value, dict):
        children = list(value.items())
    elif isinstance(value, list):
        children = list(enumerate(value))
    else:
        return None
    if id(value) in on_walk:
        return None
    on_walk.add(id(value))
    try:
        for step, child in children:
            child_name = step if isinstance(value, dict) else None
            found = _walk(child_name, child, is_binding_text, on_walk=on_walk)
            if found is not None:
                return (step, *found)
    finally:
        on_walk.discard(id(value))
    return None


def input_ref(name: str) -> str:
    """Return the `${input.<name>}` reference suggested for a secret slot.

    Args:
        name (`str`): The slot's name as written.

    Returns:
        `str`: `${input.<name>}` when `name` is an identifier; otherwise the
            normalized name with every other character replaced by `_`.
    """
    if not _IDENTIFIER.fullmatch(name):
        name = _NON_IDENTIFIER_CHAR.sub("_", normalize_name(name))
        if not _IDENTIFIER.fullmatch(name):
            name = f"_{name}"
    return "${input." + name + "}"


def render_path(name: str, path: tuple[str | int, ...]) -> str:
    """Render a nested position as `name.key[0].key`.

    Args:
        name (`str`): The slot's name.
        path (`tuple[str | int, ...]`): Keys and list indexes below the slot.

    Returns:
        `str`: The dotted path, with list indexes in brackets.
    """
    text = name
    for step in path:
        text += f"[{step}]" if isinstance(step, int) else f".{step}"
    return text


def secret_literal_message(
    slot: str, name: str, path: tuple[str | int, ...], *, owner: str | None = None
) -> str:
    """Word the load error for a literal value under a secret name.

    Args:
        slot (`str`): The slot as the message names it, e.g. `"input 'password'"`.
        name (`str`): The slot's name, used for the suggested input reference and
            the nested path.
        path (`tuple[str | int, ...]`): The result of `secret_literal_path`: `()`
            for the slot itself, else the position of the nested secret key.
        owner (`str | None`, *optional*, defaults to `None`): The slot's owner, e.g.
            `"node 'fetch'"`, prefixed to the message.

    Returns:
        `str`: The message. It never includes the value.
    """
    prefix = f"{owner}: " if owner else ""
    ref = input_ref(name)
    if not path:
        return (
            f"{prefix}{slot} is a secret name, so its value can't be written in the "
            f"flow file; pass it as an input ({ref}) or from the host"
        )
    where = f" at {render_path(name, path)}" if len(path) > 1 else ""
    return (
        f"{prefix}{slot} has key {path[-1]!r} (a secret name){where} with a literal "
        f"value; pass the value as an input ({ref}) or from the host"
    )

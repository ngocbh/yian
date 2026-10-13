"""Recognize `Type = value` text in a binding by its YAML scalar style.

An unquoted (plain-style) binding value containing ` = ` is always a typed binding:
the text before the first separator must be a type, or loading fails with a
suggestion to quote the value. A quoted or block scalar is always plain text, so
quoting is the one way to write literal text that contains ` = `. A literal right
side is checked against the type here, where the authored text is still at hand.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Any

from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.source import SourceSyntaxError, parse_yaml_literal
from yian.diagnostics import SourceRange
from yian.typesys import (
    BUILTIN_TYPE_NAMES,
    TypeCheckError,
    TypeRegistry,
    build_value_as,
    type_for,
)
from yian.typesys.values import Type

_SEPARATOR = " = "
# YAML drops the trailing space of `qty: int = `, delivering "int =".
_TRAILING_SEPARATOR = " ="
_PLAIN_TEXT_STYLES = frozenset({"single", "double", "literal", "folded"})
_STYLES = _PLAIN_TEXT_STYLES | {"plain"}
# A right side starting with a quote or a bracket is read as YAML first.
_YAML_FIRST_STARTS = ("'", '"', "[", "{")
_HEAD_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CLOSE_MATCH_CUTOFF = 0.6
_UNION = "Union"


@dataclass(frozen=True)
class TypedBindingText:
    """The two halves of a recognized `Type = value` binding.

    Attributes:
        type_text (`str`): The stripped type text before the separator, e.g.
            `"Optional[int]"`; it names a valid type.
        value (`Any`): The binding value: template, expression, or directive text
            (unquoted when the right side was quoted), or otherwise the right side
            read as a YAML literal.
    """

    type_text: str
    value: Any


def split_typed_binding(
    value: Any,
    *,
    style: str | None,
    registry: TypeRegistry,
    binding_label: str,
    diagnostics: LoadDiagnostics,
    range: SourceRange | None = None,
) -> TypedBindingText | None:
    """Split a binding value into its declared type and value when it is typed.

    The value is typed when it is a string of plain or unknown style containing
    ` = ` (or ending in ` =`). A right side that starts with a quote or a bracket
    is read as YAML first: a quoted string containing `${` or starting with
    `call(` is then a template or directive, and a collection is a literal whose
    nested strings stay text. Any other right side containing `${` or starting
    with `call(` is kept as written, and the rest is read as YAML. The right side
    is checked against the type when it is a literal; a template, expression, or
    directive is checked later, once its sources are known.

    Args:
        value (`Any`): The authored binding value.
        style (`str | None`): The scalar's YAML style: `"plain"`, `"single"`,
            `"double"`, `"literal"`, `"folded"`, or `None` when unknown. Unknown
            follows the plain rule.
        registry (`TypeRegistry`): The flow's typedefs the type text resolves against.
        binding_label (`str`): The binding's key name, used in error messages.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.
        range (`SourceRange | None`, *optional*, defaults to `None`): The value's
            location, attached to any error.

    Returns:
        `TypedBindingText | None`: The split binding, or `None` when the value is
            plain text (non-string, quoted or block style, or no separator).

    Raises:
        `LoadError`: The left side is not a type, the right side is empty or
            malformed YAML, or a literal right side does not fit the type.
        `ValueError`: `style` is not a known style name.
    """
    if style is not None and style not in _STYLES:
        raise ValueError(f"unknown YAML scalar style {style!r}")
    if style in _PLAIN_TEXT_STYLES or not isinstance(value, str):
        return None
    if _SEPARATOR in value:
        left, right = value.split(_SEPARATOR, 1)
    elif value.endswith(_TRAILING_SEPARATOR):
        left, right = value[: -len(_TRAILING_SEPARATOR)], ""
    else:
        return None
    left, right = left.strip(), right.strip()

    try:
        declared = type_for(left, registry)
    except TypeCheckError as exc:
        raise diagnostics.error(
            _not_a_type_message(left, value.strip(), binding_label, registry, exc),
            range=range,
        ) from None
    if not right:
        raise diagnostics.error(
            f"binding {binding_label!r}: the value after `{left} =` is empty", range=range
        )
    # A quoted or bracketed right side is a YAML scalar or collection, so it is read
    # as YAML before the template check: quotes never become part of the value, and
    # strings nested in a literal collection stay literal text.
    if not right.startswith(_YAML_FIRST_STARTS) and _is_template(right):
        return TypedBindingText(type_text=left, value=right)
    try:
        parsed = parse_yaml_literal(right)
    except SourceSyntaxError as exc:
        raise diagnostics.error(
            f"binding {binding_label!r}: bad value {right!r}: {exc}", range=range
        ) from None
    if isinstance(parsed, str) and _is_template(parsed):
        return TypedBindingText(type_text=left, value=parsed)
    try:
        build_value_as(declared, parsed)
    except TypeCheckError as exc:
        raise diagnostics.error(
            _literal_mismatch_message(binding_label, left, right, parsed, declared, exc),
            range=range,
        ) from None
    return TypedBindingText(type_text=left, value=parsed)


def _is_template(text: str) -> bool:
    """Whether binding text is a template or directive, resolved once sources are known.

    Args:
        text (`str`): The binding text.

    Returns:
        `bool`: True when the text contains a `${` span or is a `call(...)` directive.
    """
    return "${" in text or text.startswith("call(")


def _literal_mismatch_message(
    label: str,
    type_text: str,
    right: str,
    parsed: Any,
    declared: Type,
    error: TypeCheckError,
) -> str:
    """Explain that a literal right side does not fit, suggesting quotes when they help.

    Args:
        label (`str`): The binding's key name.
        type_text (`str`): The authored type text.
        right (`str`): The authored right-side text.
        parsed (`Any`): The right side read as a YAML literal.
        declared (`Type`): The type `type_text` names.
        error (`TypeCheckError`): The failed check of `parsed` against `declared`.

    Returns:
        `str`: The error message.
    """
    message = f"binding {label!r}: `{right}` is not a valid {type_text}"
    if error.path:
        message += f" ({error})"
    # Quoting only helps when YAML read the text as a non-string and the text
    # itself would fit, e.g. `str = 30` or `date = 2026-01-01`.
    if isinstance(parsed, str):
        return message
    try:
        build_value_as(declared, right)
    except TypeCheckError:
        return message
    message += (
        f"; quote it to keep it as text: `{label}: {type_text} = {yaml_quoted(right)}`"
    )
    if isinstance(parsed, bool):
        message += " (YAML reads unquoted yes/no/on/off/true/false as booleans)"
    return message


def _not_a_type_message(
    left: str, text: str, label: str, registry: TypeRegistry, error: TypeCheckError
) -> str:
    """Explain that `left` is not a type and show how to quote `text` instead.

    Args:
        left (`str`): The authored text before the separator.
        text (`str`): The whole authored binding value.
        label (`str`): The binding's key name.
        registry (`TypeRegistry`): The flow's typedefs, offered as close matches.
        error (`TypeCheckError`): Why `left` failed to resolve.

    Returns:
        `str`: The error message.
    """
    head = _HEAD_IDENTIFIER.match(left)
    reason = ""
    # A `Union[...]` (or a `|` written for one) fails for reasons a close match can't
    # fix, such as a member that is not a record, so the resolver's own reason is the
    # useful part.
    if "|" in left or (head is not None and head.group() == _UNION and left != _UNION):
        reason = f" ({error})"
    message = (
        f"`{left}` in `{label}: {text}` is not a type{reason}. "
        f'If you meant the text "{text}", quote it: `{label}: {yaml_quoted(text)}`'
    )
    names = BUILTIN_TYPE_NAMES | registry.keys()
    if head is None:
        return message
    # A valid head means the error lies later in the expression, e.g. `list[Strr]`;
    # suggesting a sibling of the head would point at the wrong part.
    if head.group() in names:
        return message
    matches = difflib.get_close_matches(
        head.group(), sorted(names), n=1, cutoff=_CLOSE_MATCH_CUTOFF
    )
    if matches:
        message += f" Did you mean `{matches[0]}`?"
    return message


def yaml_quoted(text: str) -> str:
    """Quote `text` as a YAML scalar that reads back as exactly `text`."""
    # A single-quoted scalar folds line breaks and cannot escape, so text with a
    # control character needs a double-quoted scalar with escapes.
    if not text.isprintable():
        return '"' + "".join(_yaml_escaped(ch) for ch in text) + '"'
    # A double-quoted scalar treats backslashes as escapes; a single-quoted one
    # only needs its own quote doubled.
    if '"' in text or "\\" in text:
        return "'" + text.replace("'", "''") + "'"
    return f'"{text}"'


def _yaml_escaped(ch: str) -> str:
    """Escape one character for a YAML double-quoted scalar."""
    if ch.isprintable() and ch not in '"\\':
        return ch
    if ord(ch) > 0xFFFF:
        # JSON would write a surrogate pair, which YAML does not join.
        return f"\\U{ord(ch):08X}"
    # JSON's escapes for the basic plane are all valid YAML escapes.
    return json.dumps(ch)[1:-1]

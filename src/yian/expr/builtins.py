"""Builtins — pure functions an author may call inside a `${...}` expression.

A `${...}` span may be a plain reference (`${name}`) OR a builtin call
(`${ render_as_json(briefs, 4) }`). `TEMPLATE_FNS` maps a builtin name to a pure
value->value function. The unified evaluator (`expr.expressions.eval_expr`) dispatches
every call through it, so builtins work in prompts, bindings, and conditions; a prompt
calls them against the node's already-bound declared inputs.

This is a deliberate, bounded bend of the "all computation is a node" law: builtins are
read-only formatting and produce no graph node or edge. See `docs/expressions.md`.

Knows about: `expr.expressions` (peer — `ExpressionError`).
Never imports: `template`, `nodes`, `compile`, `runtime` (they sit above / beside it).
"""

from __future__ import annotations

from typing import Any, Callable

import yian.json_codec as json_codec

# name -> a pure value->value formatter, called as `fn(*pos, **kw)` at prompt-render time.
TEMPLATE_FNS: dict[str, Callable[..., Any]] = {}


def register_template_fn(name: "str | None" = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering the wrapped function as a prompt builtin.

    `@register_template_fn()` uses the function's own name; `@register_template_fn("alias")`
    registers under `alias`. Returns the function unchanged (last writer wins)."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        TEMPLATE_FNS[name or fn.__name__] = fn
        return fn

    return _decorator


_KIND_NAMES = {bool: "bool", int: "int", float: "float", str: "str", list: "list", dict: "record"}


def _kind(value: Any) -> str:
    """Name `value`'s type for an error message: `str`, `int`, `list`, `record`, ..."""
    return _KIND_NAMES.get(type(value), type(value).__name__)


def _reject_null(fn: str, value: Any, default: str = '""') -> None:
    """Raise when a builtin argument is null, so it never becomes text like "None".
    `default` is the example fallback the message suggests."""
    if value is None:
        raise TypeError(f"{fn}() got null; give a default, e.g. `{fn}(x :- {default})`")


def _text(fn: str, s: Any) -> str:
    """Return `s` if it is a str; raise on null or any other type."""
    _reject_null(fn, s)
    if not isinstance(s, str):
        raise TypeError(f"{fn}() takes str, got {_kind(s)}")
    return s


@register_template_fn("render_as_json")
def _render_as_json(value: Any, indent: int = 2) -> str:
    """Pretty-print `value` as JSON — the headline builtin (a list/dict input renders as a
    JSON block instead of its Python repr). `default=str` keeps non-JSON values printable,
    and ints of any size are written in full. A null `value` is an error; a null nested
    inside it renders as `null`."""
    _reject_null("render_as_json", value)
    return json_codec.dumps(value, indent=indent, default=str, ensure_ascii=False)


@register_template_fn("join")
def _join(value: Any, sep: str = "\n") -> str:
    """Join a list's elements (stringified) with `sep`. A null list or element is an
    error, and so is a non-list (a string is not split into characters)."""
    _reject_null("join", value, "[]")
    if not isinstance(value, list):
        raise TypeError(f"join() takes a list, got {_kind(value)}")
    if any(x is None for x in value):
        raise TypeError("join() got a null element; drop or default it before joining")
    return sep.join(json_codec.as_text(x) for x in value)


@register_template_fn("upper")
def _upper(s: Any) -> str:
    """Upper-case a str. Null or a non-str value is an error."""
    return _text("upper", s).upper()


@register_template_fn("lower")
def _lower(s: Any) -> str:
    """Lower-case a str. Null or a non-str value is an error."""
    return _text("lower", s).lower()

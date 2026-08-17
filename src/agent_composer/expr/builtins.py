"""Prompt builtins — functions an author may call INSIDE an AGENT/HUMAN_INPUT prompt.

A prompt `${...}` span may be a plain reference (`${name}`) OR a builtin call
(`${ render_as_json(${briefs}, 4) }`). `TEMPLATE_FNS` maps a builtin name to a pure
value->value function; the prompt renderer (`expr.template.render_template_record`)
invokes it at render time against the node's already-bound declared inputs.

This is a deliberate, bounded bend of the "all computation is a node" law: builtins are
read-only string formatting over a node's OWN inputs, produce no graph node or edge, and
are unavailable in `from:`/`when:`/bindings. See `docs/expressions.md`.

Knows about: `expr.expressions` (peer — `ExpressionError`).
Never imports: `template`, `nodes`, `compile`, `runtime` (they sit above / beside it).
"""

from __future__ import annotations

import json
from typing import Any, Callable

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


@register_template_fn("render_as_json")
def _render_as_json(value: Any, indent: int = 2) -> str:
    """Pretty-print `value` as JSON — the headline builtin (a list/dict input renders as a
    JSON block instead of its Python repr). `default=str` keeps non-JSON values printable."""
    return json.dumps(value, indent=indent, default=str, ensure_ascii=False)


@register_template_fn("join")
def _join(value: Any, sep: str = "\n") -> str:
    """Join an iterable's elements (stringified) with `sep`."""
    return sep.join(str(x) for x in value)


@register_template_fn("upper")
def _upper(s: Any) -> str:
    return str(s).upper()


@register_template_fn("lower")
def _lower(s: Any) -> str:
    return str(s).lower()

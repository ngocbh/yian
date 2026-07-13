"""Input binding — resolve a node's declared params into a typed input record (the read boundary).

The node/flow split: a node declares its signature as `ParamDecl`s (name + required/default/type,
NO source); the flow owns the sources in `CompiledFlow.wiring[node_id][param]`. `bind_params`
joins each `ParamDecl` to its `wiring[name]` source (an `expr.template` value — a whole `${...}`
resolved against the pool with type preserved, an embedded `${...}` stringified, or a literal),
building the `{name: value}` record the engine's `eval_node` hands the node instead of the pool.
Type-checking reuses the `type_for` / `build_value_as` helpers and stays lenient
on record / variant types that can't resolve against the empty registry.

A MAP body binds a per-element `item` scope: `${item}` / `${item.path}` resolve from the current
element passed via the `item=` kwarg (a body-local scope, NOT a pool head). The default (`item`
unset) is the ordinary non-MAP bind path.

Imports `expr` (eval_binding / resolve_reference) + `typesys` (type_for /
build_value_as); both sit below/beside `nodes` in the layer ladder.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

from agent_composer.events import SourceSpan
from agent_composer.expr import (
    ExpressionError,
    RequiredError,
    eval_binding,
    resolve_reference,
)
from agent_composer.typesys import TypeCheckError, Type, build_value_as, type_for
from agent_composer.typesys.pool import VariablePool


class BindingError(ValueError):
    """A node's declared input could not be bound (missing-required or type mismatch).

    Carries an optional `locator` — a `SourceSpan(None, "input", <param>)` naming the
    failing input. `bind_params` stamps it; eval_node's funnel fills in the node id.
    """

    locator: object = None  # Optional[SourceSpan]


@dataclass(frozen=True)
class ParamDecl:
    """
    One declared input on a node — the node-side half of the node/flow split.

    The node declares only its signature; the flow owns the wiring
    (`CompiledFlow.wiring[node_id][name] -> source`), so a `ParamDecl` holds NO source.

    Attributes:
        name (`str`):
            The input name (the key the bound record is keyed by).
        type_str (`str`, *optional*, defaults to `None`):
            The declared type name; `None` is an UNTYPED param (e.g. TOOL args).
        required (`bool`, *optional*, defaults to `False`):
            Whether an omitted input is an error. In practice only START params set this.
        default (`Any`, *optional*, defaults to `None`):
            The value filled for an OMITTED input. In practice only START params set this.
        type (`Type`, *optional*, defaults to `None`):
            The compile-stamped resolved type used to type-check the bound value.
    """

    name: str
    type_str: Optional[str] = None
    required: bool = False
    default: Any = None
    type: Optional[Type] = None


def _resolve_source(source: Any, pool: VariablePool, item: Any = None) -> Any:
    """Resolve a binding's `source` to a value via the `expr.template` language.

    A value that is exactly one `${...}` yields the **typed** value of that reference
    (type preserved); a `${...}` embedded in surrounding text is **stringified** into
    it; a value with no `${...}` is a plain literal (after `$$` -> `$`). The interior
    is a `|` coalesce, first-non-None, where an atom is a ref, a `:-` literal-or-
    one-nested-`${...}` default, a `:?` required, or a literal. `${item}` / `${item.path}`
    resolve from the MAP-body-local scope. A non-string `source` is a literal.

    Parse/eval errors (malformed ref, two-level nesting, an unbound `:?`) surface as
    `BindingError` — `expr`'s `ExpressionError`/`RequiredError` never leak across the
    `nodes <- expr` boundary."""
    if not isinstance(source, str):
        return source
    try:
        return eval_binding(source, lambda path: resolve_reference(path, pool), item)
    except (RequiredError, ExpressionError) as exc:
        raise BindingError(str(exc)) from exc


def bind_params(
    params: list[ParamDecl],
    wiring: dict[str, Any],
    pool: VariablePool,
    *,
    item: Any = None,
) -> dict[str, Any]:
    """Resolve each declared `ParamDecl` into a typed record — the engine's read boundary.

    Joins each param to its source from the flow-owned `wiring` dict (`wiring[name] -> source`):
    coalesce / default / required / type-check / deep-copy / `item` scope. An OMITTED input (no
    wiring edge) fills its declared default or fails when required; a caller's explicit null
    SHADOWS the child default (`f(x=None)` semantics).

    Args:
        params (`list[ParamDecl]`):
            The node-side declared signature to bind.
        wiring (`dict`):
            The flow-owned `name -> source` map for this node.
        pool (`VariablePool`):
            The pool each `${...}` source resolves against.
        item (`Any`, *optional*, defaults to `None`):
            The current element for a MAP body's `${item}` scope; `None` is the non-MAP path.

    Returns:
        `dict[str, Any]`: The bound `{name: value}` record handed to the node instead of the pool.

    Raises:
        BindingError: On a missing-required input or a type mismatch.
    """
    record: dict[str, Any] = {}
    for p in params:
        try:
            present = p.name in wiring  # an absent edge (caller OMITTED) vs a present one (incl. bound-null)
            source = wiring.get(p.name)
            value = _resolve_source(source, pool, item)
            if value is None:
                # Distinguish an OMITTED input (no wiring edge) from one BOUND-TO-NULL (an edge that
                # resolves to None): only an omitted input fills its declared default / fails when
                # required — a caller's explicit null SHADOWS the child default (`f(x=None)` semantics,
                # the apply_defaults contract). In practice only START_ID's params carry default/required
                # (every other kind builds bare ParamDecls), so this refinement is inert elsewhere; it
                # is what lets the spliced child START_ID own omitted-input defaulting.
                if not present and p.default is not None:
                    value = p.default
                elif not present and p.required:
                    raise BindingError(f"required input {p.name!r} is unbound (from={source!r})")
                elif not present:
                    # Truly OMITTED, no default, not required -> leave the key ABSENT so the
                    # wrapper's apply_default can fill it (or the render raises on unbound).
                    continue
                else:
                    # PRESENT but resolved to None -> a caller's explicit bound-null. Keep it
                    # None to SHADOW any default (`f(x=None)`); the wrapper must NOT re-fill it.
                    record[p.name] = None
                    continue
            typ = p.type
            if typ is None and p.type_str is not None:
                try:
                    typ = type_for(p.type_str, {})
                except TypeCheckError:
                    typ = None
            if typ is not None:
                try:
                    value = build_value_as(typ, value).to_object()
                except TypeCheckError as exc:
                    raise BindingError(f"input {p.name!r}: {exc}") from exc
            record[p.name] = copy.deepcopy(value)
        except BindingError as exc:
            # Name the failing input for precise error framing; node id is stamped by
            # eval_node's funnel (binding has no node identity here).
            if exc.locator is None:
                exc.locator = SourceSpan(node=None, kind="input", key=p.name)
            raise
    return record

"""Input binding — resolve a node's declared params into a typed input record (the read boundary).

The node/flow split: a node declares its signature as source-neutral `Parameter` values (name +
contract/default policy, NO source); the flow owns the sources in
`CompiledFlow.wiring[node_id][param]`. `bind_params` joins each parameter to its `wiring[name]`
source (an `expr.template` value — a whole `${...}`
resolved against the pool with type preserved, an embedded `${...}` stringified, or a literal),
building the `{name: value}` record the engine's `eval_node` hands the node instead of the pool.
Type-checking consumes an optional resolved `ValueContract`; untyped parameters
accept any resolved Python value.

A MAP body binds a per-element `item` scope: `${item}` / `${item.path}` resolve from the current
element passed via the `item=` kwarg (a body-local scope, NOT a pool head). The default (`item`
unset) is the ordinary non-MAP bind path.

Imports `expr` (eval_binding / resolve_reference) + `typesys` (type_for /
build_value_as); both sit below/beside `nodes` in the layer ladder.
"""

from __future__ import annotations

import copy
from typing import Any

from agent_composer.events import SourceSpan
from agent_composer.expr import (
    ExpressionError,
    RequiredError,
    eval_binding,
    resolve_reference,
)
from agent_composer.nodes.contracts import Parameter, binding_source
from agent_composer.typesys import TypeCheckError, build_value_as
from agent_composer.typesys.pool import VariablePool
from agent_composer.typesys.seeding import coerce_param


class BindingError(ValueError):
    """A node's declared input could not be bound (missing-required or type mismatch).

    Carries an optional `locator` — a `SourceSpan(None, "input", <param>)` naming the
    failing input. `bind_params` stamps it; eval_node's funnel fills in the node id.
    """

    locator: object = None  # Optional[SourceSpan]


def _resolve_source(source: Any, pool: VariablePool, item: Any = None) -> Any:
    """Resolve one raw or carrier-wrapped source via the template language.

    Exactly one semantic ``Binding`` layer is removed at this read boundary; a
    nested ``Binding`` remains an ordinary literal payload. A value that is
    exactly one `${...}` yields the **typed** value of that reference
    (type preserved); a `${...}` embedded in surrounding text is **stringified** into
    it; a value with no `${...}` is a plain literal (after `$$` -> `$`). The interior
    is a `|` coalesce, first-non-None, where an atom is a ref, a `:-` literal-or-
    one-nested-`${...}` default, a `:?` required, or a literal. `${item}` / `${item.path}`
    resolve from the MAP-body-local scope. A non-string `source` is a literal.

    Parse/eval errors (malformed ref, two-level nesting, an unbound `:?`) surface as
    `BindingError` — `expr`'s `ExpressionError`/`RequiredError` never leak across the
    `nodes <- expr` boundary."""
    source = binding_source(source)
    if not isinstance(source, str):
        return source
    try:
        return eval_binding(source, lambda path: resolve_reference(path, pool), item)
    except (RequiredError, ExpressionError) as exc:
        raise BindingError(str(exc)) from exc


def bind_params(
    params: list[Parameter],
    wiring: dict[str, Any],
    pool: VariablePool,
    *,
    item: Any = None,
) -> dict[str, Any]:
    """Resolve each declared `Parameter` into a typed record — the engine's read boundary.

    Joins each param to its source from the flow-owned `wiring` dict (`wiring[name] -> source`):
    coalesce / default / required / type-check / deep-copy / `item` scope. An OMITTED input (no
    wiring edge) fills its declared default or fails when required; a caller's explicit null
    SHADOWS the child default (`f(x=None)` semantics).

    Args:
        params (`list[Parameter]`):
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
    if type(params) is not list:
        raise TypeError(
            f"node params must be an exact list, got {type(params).__name__}"
        )
    record: dict[str, Any] = {}
    for p in params:
        if type(p) is not Parameter:
            raise TypeError(
                f"node params must contain Parameter values, got {type(p).__name__}"
            )
        try:
            present = p.name in wiring  # an absent edge (caller OMITTED) vs a present one (incl. bound-null)
            used_default = False
            source = wiring.get(p.name)
            value = _resolve_source(source, pool, item)
            if value is None:
                # Distinguish an OMITTED input (no wiring edge) from one BOUND-TO-NULL (an edge that
                # resolves to None): only an omitted input fills its declared default / fails when
                # required — a caller's explicit null SHADOWS the child default (`f(x=None)` semantics,
                # the apply_defaults contract). Loader-built body parameters carry no default;
                # START parameters do, so the spliced child START owns omitted-input defaulting.
                if not present and p.has_default:
                    value = p.default
                    used_default = True
                elif not present and p.required:
                    raise BindingError(f"required input {p.name!r} is unbound (from={source!r})")
                elif not present:
                    # Truly OMITTED, no default, not required -> leave the key ABSENT so the
                    # wrapper's apply_default can fill it (or the render raises on unbound).
                    continue
                else:
                    # PRESENT but resolved to None -> a caller's explicit bound-null. It
                    # SHADOWS any default, then still passes through the optional contract:
                    # nullable/untyped accepts it; a non-nullable contract rejects it.
                    value = None
            if p.contract is not None:
                if present and p.coerce_strings:
                    value = coerce_param(p, value)
                try:
                    checked = build_value_as(p.contract.type, value)
                except TypeCheckError as exc:
                    raise BindingError(f"input {p.name!r}: {exc}") from exc
                if not used_default:
                    value = checked.to_object()
            record[p.name] = copy.deepcopy(value)
        except BindingError as exc:
            # Name the failing input for precise error framing; node id is stamped by
            # eval_node's funnel (binding has no node identity here).
            if exc.locator is None:
                exc.locator = SourceSpan(node=None, kind="input", key=p.name)
            raise
    return record

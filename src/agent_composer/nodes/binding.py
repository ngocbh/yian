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

Imports `expr` for source evaluation and `typesys` for contract validation and
coercion; both sit below/beside `nodes` in the layer ladder.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypeAlias

from agent_composer.diagnostics import OriginRef
from agent_composer.expr import (
    ExpressionError,
    RequiredError,
    eval_binding,
    resolve_reference,
)
from agent_composer.expr.lineage import ResolvedValue, ValueRoute
from agent_composer.expr.template import Span, eval_template, scan_template
from agent_composer.nodes.contracts import Binding, Parameter, binding_source
from agent_composer.typesys import TypeCheckError, ValueKind, build_value_as
from agent_composer.typesys.pool import VariablePool


BindingFailureCode: TypeAlias = Literal["binding.evaluate", "contract.input"]
SourceResolver: TypeAlias = Callable[[str], ResolvedValue]
ReservedBinder: TypeAlias = Callable[[str, Any, Callable[[Any], Any]], Any]


@dataclass(frozen=True, slots=True)
class BindingCapture:
    """Detached delivered inputs and the routes evaluated while binding them.

    An incomplete bind has no usable ``bound_inputs``. Its route and evaluated
    keys retain only the safe context already collected before the failure.
    ``evaluated_keys`` distinguishes a template evaluated here from a materialized
    string passed through literally, even when evaluation produced a literal route.
    """

    _bound_inputs: dict[str, Any] | None = field(repr=False)
    route: ValueRoute
    evaluated_keys: frozenset[str]
    items: tuple[tuple[int, BindingCapture], ...] = ()

    @property
    def bound_inputs(self) -> dict[str, Any] | None:
        """Return an isolated copy of the complete pre-execution input record."""
        return copy.deepcopy(self._bound_inputs)


class BindingCollector:
    """Collect one parameter record at its existing read and copying boundary."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}
        self._routes: dict[str, ValueRoute] = {}
        self._evaluated: set[str] = set()
        self._items: dict[int, BindingCollector] = {}
        self._complete = False
        self._frozen = False

    def _require_open(self) -> None:
        if self._frozen:
            raise RuntimeError("binding capture is already frozen")

    def _record_source(self, name: str, route: ValueRoute, evaluated: bool) -> None:
        """Retain a route from the same scan and evaluation that produced a value."""
        self._require_open()
        self._routes[name] = route
        if evaluated:
            self._evaluated.add(name)
        else:
            self._evaluated.discard(name)

    def finish(self, record: dict[str, Any]) -> None:
        """Snapshot the complete delivered record before node execution.

        Args:
            record (`dict[str, Any]`): The exact record returned by binding,
                including any materialized values supplied by node-owned hooks.
        """
        self._require_open()
        self._values = copy.deepcopy(record)
        self._routes = {
            name: self._routes[name] if name in self._routes else ValueRoute.literal(value)
            for name, value in self._values.items()
        }
        self._evaluated.intersection_update(record)
        self._complete = True

    def route_for(self, name: str) -> ValueRoute:
        """Return an already evaluated input route for local-scope projection."""
        return self._routes[name]

    def for_item(self, index: int) -> BindingCollector:
        """Allocate an item record under the explicit index supplied by its node."""
        self._require_open()
        if type(index) is not int or index < 0:
            raise ValueError("binding item index must be a nonnegative integer")
        if index in self._items:
            raise ValueError(f"binding item index {index} was already recorded")
        collector = BindingCollector()
        self._items[index] = collector
        return collector

    def invalidate(self) -> None:
        """Keep partial context while discarding the usable input record."""
        self._require_open()
        self._complete = False
        for item in self._items.values():
            item.invalidate()

    def freeze(self) -> BindingCapture:
        """Close this collector and return its detached binding capture."""
        self._require_open()
        self._frozen = True
        return BindingCapture(
            self._values if self._complete else None,
            ValueRoute.record(self._routes.items()),
            frozenset(self._evaluated),
            tuple((index, item.freeze()) for index, item in self._items.items()),
        )


class BindingError(ValueError):
    """An expected binding operation failed with one semantic stage and origin.

    ``failure_code`` distinguishes expression/default/required evaluation from a
    supplied value violating its input contract. ``origin`` is the most precise
    available authored operation and feeds the canonical failure trace.
    """

    def __init__(
        self,
        message: str,
        *,
        failure_code: BindingFailureCode,
        origin: OriginRef | None = None,
    ) -> None:
        """Retain one validated binder-owned semantic classification."""
        super().__init__(message)
        if failure_code not in ("binding.evaluate", "contract.input"):
            raise ValueError(f"invalid binding failure code {failure_code!r}")
        if origin is not None:
            if not isinstance(origin, OriginRef):
                raise TypeError("binding failure origin must be an OriginRef or None")
            OriginRef(origin.key)
        self.failure_code = failure_code
        self.origin = origin


def _source_origin(source: Any) -> OriginRef | None:
    """Return one canonical binding carrier's origin, or ``None`` for raw input."""
    return source.origin if isinstance(source, Binding) else None


def _validate_parameters(params: list[Parameter]) -> None:
    """Require the canonical exact-list parameter carrier at every binding seam."""
    if type(params) is not list:
        raise TypeError(
            f"node params must be an exact list, got {type(params).__name__}"
        )
    for parameter in params:
        if type(parameter) is not Parameter:
            raise TypeError(
                "node params must contain Parameter values, "
                f"got {type(parameter).__name__}"
            )


def coerce_param(parameter: Parameter, value: Any) -> Any:
    """Coerce a supplied string under one parameter's explicit scalar policy."""
    if type(parameter) is not Parameter:
        raise TypeError(
            "parameter must be an exact Parameter, "
            f"got {type(parameter).__name__}"
        )
    if (
        value is None
        or not isinstance(value, str)
        or not parameter.coerce_strings
        or parameter.contract is None
    ):
        return value

    raw = value.strip()
    try:
        if parameter.contract.type.kind is ValueKind.INTEGER:
            return int(raw)
        if parameter.contract.type.kind is ValueKind.NUMBER:
            return float(raw)
        if parameter.contract.type.kind is ValueKind.BOOLEAN:
            return raw.lower() in ("1", "true", "yes", "y", "on")
    except ValueError:
        return value
    return value


def coerce_inputs(
    params: list[Parameter], raw: dict[str, Any]
) -> dict[str, Any]:
    """Coerce supplied top-level values while retaining undeclared keys unchanged."""
    _validate_parameters(params)
    by_name = {parameter.name: parameter for parameter in params}
    return {
        name: coerce_param(by_name[name], value) if name in by_name else value
        for name, value in raw.items()
    }


def apply_defaults(
    params: list[Parameter], inputs: dict[str, Any]
) -> dict[str, Any]:
    """Fill validated defaults for omitted parameters without coercing payloads.

    A caller-passed value, including ``None``, always shadows a default. Defaults
    are validated against any semantic contract but retain their original Python
    payload rather than the typed value system's normalized representation.
    """
    _validate_parameters(params)
    output = dict(inputs)
    for parameter in params:
        if parameter.name in output or not parameter.has_default:
            continue
        if parameter.contract is not None:
            try:
                build_value_as(parameter.contract.type, parameter.default)
            except TypeCheckError as exc:
                raise BindingError(
                    str(exc),
                    failure_code="binding.evaluate",
                    origin=parameter.default_origin or parameter.origin,
                ) from exc
        output[parameter.name] = parameter.default
    return output


def _resolve_source(
    source: Any,
    pool: VariablePool,
    item: Any = None,
    *,
    collector: BindingCollector | None = None,
    name: str = "",
    source_resolver: SourceResolver | None = None,
) -> Any:
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
    `nodes <- expr` boundary. ``collector`` optionally retains the selected route
    under ``name``; ``source_resolver`` pairs values with accepted producer identities
    in one read. Without that injected resolver, capture records no producer identity.
    """
    origin = _source_origin(source)
    source_value = binding_source(source)
    if not isinstance(source_value, str):
        if collector is not None:
            collector._record_source(name, ValueRoute.literal(source_value), False)
        return source_value
    try:
        if collector is not None:
            def resolve(path: str) -> ResolvedValue:
                if source_resolver is not None:
                    return source_resolver(path)
                value = resolve_reference(path, pool)
                return ResolvedValue(value, value is not None)

            segments = scan_template(source_value)
            result = eval_template(segments, resolve, item, trace=True)
            collector._record_source(
                name, result.route, any(isinstance(seg, Span) for seg in segments)
            )
            return result.value
        return eval_binding(
            source_value,
            lambda path: resolve_reference(path, pool),
            item,
        )
    except (RequiredError, ExpressionError) as exc:
        raise BindingError(
            str(exc),
            failure_code="binding.evaluate",
            origin=origin,
        ) from exc


def bind_params(
    params: list[Parameter],
    wiring: dict[str, Any],
    pool: VariablePool,
    *,
    item: Any = None,
    collector: BindingCollector | None = None,
    source_resolver: SourceResolver | None = None,
    _finish_capture: bool = True,
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
        collector (`BindingCollector`, *optional*, defaults to `None`):
            Retain delivered values and routes independently of node execution.
        source_resolver (`SourceResolver`, *optional*, defaults to `None`):
            One-read value/presence/source lookup used only when capturing bindings.
        _finish_capture (`bool`, *optional*, defaults to `True`):
            Internal evaluator seam. False retains routes while deferring the
            input snapshot until reserved values and wrapper defaults are complete.

    Returns:
        `dict[str, Any]`: The bound `{name: value}` record handed to the node instead of the pool.

    Raises:
        BindingError: On expression/default/required evaluation failure or supplied
            input contract rejection; ``failure_code`` identifies the stage.
    """
    _validate_parameters(params)
    if collector is not None:
        collector.invalidate()
    record: dict[str, Any] = {}
    for p in params:
        try:
            present = p.name in wiring  # an absent edge (caller OMITTED) vs a present one (incl. bound-null)
            used_default = False
            source = wiring.get(p.name)
            value = _resolve_source(
                source, pool, item, collector=collector if present else None,
                name=p.name, source_resolver=source_resolver,
            )
            if value is None:
                # Distinguish an OMITTED input (no wiring edge) from one BOUND-TO-NULL (an edge that
                # resolves to None): only an omitted input fills its declared default / fails when
                # required — a caller's explicit null SHADOWS the child default (`f(x=None)` semantics,
                # the apply_defaults contract). Loader-built body parameters carry no default;
                # START parameters do, so the spliced child START owns omitted-input defaulting.
                if not present and p.has_default:
                    value = p.default
                    used_default = True
                    if collector is not None:
                        collector._record_source(p.name, ValueRoute.literal(value), False)
                elif not present and p.required:
                    raise BindingError(
                        f"required input {p.name!r} is unbound (from={source!r})",
                        failure_code="binding.evaluate",
                        origin=p.origin,
                    )
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
                    if used_default:
                        failure_code: BindingFailureCode = "binding.evaluate"
                        origin = p.default_origin or p.origin
                    else:
                        failure_code = "contract.input"
                        origin = p.contract.origins.at(exc.path) or p.origin
                    raise BindingError(
                        f"input {p.name!r}: {exc}",
                        failure_code=failure_code,
                        origin=origin,
                    ) from exc
                if not used_default:
                    value = checked.to_object()
            record[p.name] = copy.deepcopy(value)
        except BindingError as exc:
            if exc.origin is None:
                exc.origin = p.origin
            raise
    if collector is not None and _finish_capture:
        collector.finish(record)
    return record

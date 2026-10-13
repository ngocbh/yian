"""Input binding — resolve a node's declared params into a typed input record (the read boundary).

The node/flow split: a node declares its signature as source-neutral `Parameter` values (name +
contract/default policy, NO source); the flow owns the sources in
`CompiledFlow.wiring[node_id][param]`. `bind_params` joins each parameter to its `wiring[name]`
source (an `expr.template` value — a whole `${...}`
resolved against the pool with type preserved, an embedded `${...}` stringified, or a literal),
building the `{name: value}` record the engine's `eval_node` hands the node instead of the pool.
Type-checking consumes an optional resolved `ValueContract`; untyped parameters
accept any resolved Python value. A typed `Binding` additionally checks its resolved
value against the binding's own declared contract, without converting it.

A MAP body binds a per-element `item` scope: `${item}` / `${item.path}` resolve from the current
element passed via the `item=` kwarg (a body-local scope, NOT a pool head). The default (`item`
unset) is the ordinary non-MAP bind path.

Binding never converts a string. Supplied text is converted only at the run boundary:
`coerce_inputs` applies each top-level parameter's `coerce_strings` policy before START
binds, so a value crossing a CALL/MAP/LOOP edge into a child START is validated as is.

Imports `expr` for source evaluation and `typesys` for contract validation; both sit
below/beside `nodes` in the layer ladder.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypeAlias

from yian.diagnostics import OriginRef
from yian.expr import (
    ExpressionError,
    RequiredError,
    eval_binding,
    resolve_reference,
)
from yian.expr.lineage import ResolvedValue, ValueRoute
from yian.expr.template import Span, eval_template, scan_template
from yian.nodes.contracts import Binding, Parameter, ValueContract, binding_source
from yian.typesys import TypeCheckError, ValueKind, build_value_as
from yian.typesys.pool import VariablePool
from yian.typesys.values import Type, reject_nonfinite


BindingFailureCode: TypeAlias = Literal[
    "binding.evaluate", "binding.type", "contract.input"
]
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

    ``failure_code`` distinguishes expression/default/required evaluation
    (``binding.evaluate``), a value violating its typed binding's declared type
    (``binding.type``), and a supplied value violating its input contract
    (``contract.input``). ``origin`` is the most precise available authored
    operation and feeds the canonical failure trace.
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
        if failure_code not in ("binding.evaluate", "binding.type", "contract.input"):
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


def _reject_nonfinite_binding(value: Any, where: str, origin: OriginRef | None) -> None:
    """Raise a ``binding.evaluate`` `BindingError` if ``value`` holds a NaN or infinity."""
    try:
        reject_nonfinite(value, where)
    except TypeCheckError as exc:
        raise BindingError(str(exc), failure_code="binding.evaluate", origin=origin) from exc


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


_TRUE_SPELLINGS = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_SPELLINGS = frozenset({"0", "false", "no", "n", "off"})


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
            spelling = raw.lower()
            if spelling in _TRUE_SPELLINGS:
                return True
            if spelling in _FALSE_SPELLINGS:
                return False
            # Any other text stays a string so the bool contract rejects it.
            return value
    except ValueError:
        return value
    return value


def check_input_names(params: list[Parameter], raw: Mapping[str, Any]) -> None:
    """Reject top-level inputs that are not a mapping or name an undeclared input.

    Raises:
        `TypeError`: ``raw`` is not a mapping.
        `ValueError`: A key names no declared input; the message lists the
            declared names.
    """
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"inputs must be a mapping of input names to values, got {type(raw).__name__}"
        )
    declared = [parameter.name for parameter in params]
    unknown = sorted(repr(name) for name in raw if name not in declared)
    if unknown:
        noun = "input" if len(unknown) == 1 else "inputs"
        raise ValueError(
            f"unknown {noun} {', '.join(unknown)}; "
            f"declared inputs: {', '.join(declared) or '(none)'}"
        )


def coerce_inputs(
    params: list[Parameter], raw: Mapping[str, Any]
) -> dict[str, Any]:
    """Coerce supplied top-level values, each keyed by a declared input name.

    Raises:
        `TypeError`: ``raw`` is not a mapping.
        `ValueError`: A key names no declared input.
    """
    _validate_parameters(params)
    check_input_names(params, raw)
    by_name = {parameter.name: parameter for parameter in params}
    return {name: coerce_param(by_name[name], value) for name, value in raw.items()}


def apply_defaults(
    params: list[Parameter], inputs: dict[str, Any]
) -> dict[str, Any]:
    """Fill validated defaults for omitted parameters without coercing payloads.

    A caller-passed value, including ``None``, always shadows a default. Defaults
    are validated against any semantic contract but retain their original Python
    payload rather than the typed value system's normalized representation.

    Raises:
        BindingError: With ``failure_code="binding.evaluate"`` when a default holds a
            NaN or infinity at any depth or violates its parameter's contract.
    """
    _validate_parameters(params)
    output = dict(inputs)
    for parameter in params:
        if parameter.name in output or not parameter.has_default:
            continue
        _reject_nonfinite_binding(
            parameter.default,
            f"default for {parameter.name!r}",
            parameter.default_origin or parameter.origin,
        )
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
    types: dict[str, Type] | None = None,
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
    ``types`` is the table an ``isinstance(x, T)`` looks ``T`` up in (``None`` means no types).
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
            result = eval_template(segments, resolve, item, trace=True, types=types)
            collector._record_source(
                name, result.route, any(isinstance(seg, Span) for seg in segments)
            )
            return result.value
        return eval_binding(
            source_value,
            lambda path: resolve_reference(path, pool),
            item,
            types=types,
        )
    except (RequiredError, ExpressionError) as exc:
        raise BindingError(
            str(exc),
            failure_code="binding.evaluate",
            origin=origin,
        ) from exc


def _check_binding_type(
    name: str, binding: Binding, contract: ValueContract, value: Any
) -> None:
    """Validate a resolved value against its typed binding's declared contract.

    The check runs before the parameter contract and discards the built value, so
    the original payload is delivered unchanged (an ``int`` declared ``float`` stays
    an ``int``) and a string never satisfies a non-string declared type.

    Args:
        name (`str`): The bound parameter name, used in the failure message.
        binding (`Binding`): The typed binding, whose origin is the fallback location.
        contract (`ValueContract`): The binding's declared contract.
        value (`Any`): The value the binding's source resolved to.

    Raises:
        BindingError: With ``failure_code="binding.type"`` when the value violates
            the declared type, located at the nearest contract origin for the
            rejected path, else the binding's origin.
    """
    try:
        build_value_as(contract.type, value)
    except TypeCheckError as exc:
        raise BindingError(
            f"binding {name!r}: {exc}",
            failure_code="binding.type",
            origin=contract.origins.at(exc.path) or binding.origin,
        ) from exc


def bind_params(
    params: list[Parameter],
    wiring: dict[str, Any],
    pool: VariablePool,
    *,
    item: Any = None,
    collector: BindingCollector | None = None,
    source_resolver: SourceResolver | None = None,
    types: dict[str, Type] | None = None,
    _finish_capture: bool = True,
) -> dict[str, Any]:
    """Resolve each declared `Parameter` into a typed record — the engine's read boundary.

    Joins each param to its source from the flow-owned `wiring` dict (`wiring[name] -> source`):
    coalesce / default / required / type-check / deep-copy / `item` scope. A present typed
    `Binding` first checks its resolved value against the binding's declared contract, leaving
    the value unconverted; the parameter's own contract then runs as usual. An OMITTED input (no
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
        types (`dict[str, Type]`, *optional*, defaults to `None`):
            The type table an `isinstance(x, T)` in a source looks `T` up in, keyed by type
            name (the node's `types`). `None` means no types.
        _finish_capture (`bool`, *optional*, defaults to `True`):
            Internal evaluator seam. False retains routes while deferring the
            input snapshot until reserved values and wrapper defaults are complete.

    Returns:
        `dict[str, Any]`: The bound `{name: value}` record handed to the node instead of the pool.

    Raises:
        BindingError: On expression/default/required evaluation failure, a resolved value
            or filled default holding a NaN or infinity at any depth, typed binding
            rejection, or supplied input contract rejection; ``failure_code`` identifies
            the stage.
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
                name=p.name, source_resolver=source_resolver, types=types,
            )
            _reject_nonfinite_binding(value, f"input {p.name!r}", _source_origin(source))
            if present and isinstance(source, Binding) and source.contract is not None:
                _check_binding_type(p.name, source, source.contract, value)
            if value is None:
                # Distinguish an OMITTED input (no wiring edge) from one BOUND-TO-NULL (an edge that
                # resolves to None): only an omitted input fills its declared default / fails when
                # required — a caller's explicit null SHADOWS the child default (`f(x=None)` semantics,
                # the apply_defaults contract). Loader-built body parameters carry no default;
                # START parameters do, so the spliced child START owns omitted-input defaulting.
                if not present and p.has_default:
                    value = p.default
                    used_default = True
                    _reject_nonfinite_binding(
                        value, f"input {p.name!r}", p.default_origin or p.origin
                    )
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
                # No string conversion here: a CALL/MAP/LOOP edge is not a flow
                # boundary. `run_flow` converts the top-level flow's supplied inputs
                # (`coerce_inputs`) before START binds them.
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

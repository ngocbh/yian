"""Flow-input default seeding — shared by the run boundary and REF/MAP children.

The Compose run boundary coerces + defaults the top-level run args; a REF/MAP child
runs via the engine (not the boundary), so its `_run` reuses `apply_defaults` to fill
the child's own declared defaults for omitted args. The clock reaches children through
the inherited `pool.system` (`${system.today}`/`${system.now}`) — there is no separate
`as_of` clock special-case.

Lives below nodes so the node layer can import it without an import-direction
inversion. It structurally supports both canonical node `Parameter` values and the
Compose boundary's retained `InputDecl` without importing either package.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

from agent_composer.typesys.values import ValueKind, build_value_as

if TYPE_CHECKING:  # pragma: no cover - typing only (keeps state import-clean)
    from agent_composer.typesys.pool import VariablePool


def coerce_param(field: Any, value: Any) -> Any:
    """Coerce a supplied string under a parameter's explicit scalar policy.

    A canonical node `Parameter` opts in with `coerce_strings=True` and supplies
    its resolved type through `contract`. A retained Compose `InputDecl` supplies
    `type_str` and keeps its existing boundary coercion. Only unambiguous scalar
    types coerce; other values pass through for the owning validator to reject.
    """
    if value is None or not isinstance(value, str):
        return value
    contract = getattr(field, "contract", None)
    if hasattr(field, "coerce_strings"):
        if not field.coerce_strings or contract is None:
            return value
        kind = contract.type.kind
        type_name = {
            ValueKind.INTEGER: "int",
            ValueKind.NUMBER: "float",
            ValueKind.BOOLEAN: "bool",
        }.get(kind)
    else:
        type_name = field.type_str
    raw = value.strip()
    try:
        if type_name == "int":
            return int(raw)
        if type_name == "float":
            return float(raw)
        if type_name == "bool":
            return raw.lower() in ("1", "true", "yes", "y", "on")
    except ValueError:
        return value
    return value


def coerce_inputs(inputs: List[Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    by_name = {io.name: io for io in inputs}
    return {k: coerce_param(by_name[k], v) if k in by_name else v for k, v in raw.items()}


def apply_defaults(inputs: List[Any], coerced: Dict[str, Any]) -> Dict[str, Any]:
    """Fill each declared input's default for any name the caller omitted.

    A canonical `Parameter.has_default` distinguishes an explicit null default from
    absence and retains its stored payload without coercion. The retained Compose
    `InputDecl` has no presence bit, so its historical non-null default rule and
    boundary coercion remain until that front-end carrier is cut over.

    A caller-passed value is never overridden — even a falsy one, and even ``None``.
    So a parent that BINDS ``null`` to a child input SHADOWS that child's default (the
    name is present in the record): null is a value, not absence (like Python
    ``f(x=None)``). The child default fills only when the parent omits the input. The
    cross-flow type check relies on this — a nullable source needs a binding-level
    non-null guarantee (`:-literal` / `:?`), not the child's default, to fill a
    non-nullable input.
    """
    out = dict(coerced)
    for io in inputs:
        if io.name in out:
            continue
        if hasattr(io, "has_default"):
            if io.has_default:
                contract = getattr(io, "contract", None)
                if contract is not None:
                    # Validate the authored payload without replacing it with the typed
                    # system's normalized object. Defaults are semantic source data; coercion
                    # applies only to supplied boundary strings.
                    build_value_as(contract.type, io.default)
                out[io.name] = io.default
        elif io.default is not None:
            out[io.name] = coerce_param(io, io.default)
    return out


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def now_utc() -> str:
    """The run's wall-clock 'now' as an ISO-8601 datetime (UTC)."""
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    """A fresh run identifier for when the host doesn't supply one.

    A run id is inherently unique per run, so the boundary mints one (uuid4 hex)
    unless the host injects a meaningful value — mirroring how the clock ambients
    are auto-seeded, so `${system.run_id}` always resolves.
    """
    return uuid.uuid4().hex


def seed_system_clock(pool: "VariablePool") -> None:
    """Seed the host-ambient clock into the pool's `system` namespace, once per run.

    `${system.today}` is a `date`; `${system.now}` a `datetime`. Computed once here and
    inherited by REF/MAP children (they copy `pool.system`), so every node, sub-flow,
    and MAP element observes one consistent clock. The clock is the only seam through
    which a flow may read "now" (the engine never resolves date words).
    """
    from agent_composer.typesys.values import DateValue, DateTimeValue

    pool.system["today"] = DateValue(value=today_utc())
    pool.system["now"] = DateTimeValue(value=now_utc())

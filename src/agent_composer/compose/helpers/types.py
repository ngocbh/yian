"""The type-node reader: a Compose `outputs:`/`inputs:` source node -> one runtime `Type`.

A node here is a leaf type string (`"float"`, `"list[str]"`, a registry name), a
map of field -> node (a record, nested natively), or a list of single-key maps
(the list-of-fields style). Resolution of leaf strings and nested records is the
type layer's job (`type_for` recurses over records/lists/Optional); this reader
is the thin composition that walks the YAML structure and wraps any type error loudly.
"""

from collections.abc import Mapping
from typing import Any, Optional

from agent_composer.nodes.contracts import Parameter, ValueContract
from agent_composer.typesys import TypeCheckError, Type, build_value_as, type_for
from agent_composer.typesys.values import ValueKind
from agent_composer.typesys.types import TypeRegistry

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.events import SourceRange
from agent_composer.compose.source import SourceSyntaxError, parse_yaml_literal


def read_type(node, registry: TypeRegistry) -> Type:
    """Read a Compose source node into one runtime `Type` (recursively)."""
    if isinstance(node, str):
        try:
            return type_for(node, registry)
        except TypeCheckError as exc:
            raise LoadError(f"bad type expression {node!r}: {exc}") from exc

    if isinstance(node, dict):
        fields = {k: read_type(v, registry) for k, v in node.items()}
        return Type(
            kind=ValueKind.OBJECT,
            fields=fields,
            required=frozenset(k for k, ty in fields.items() if not ty.nullable),
        )

    if isinstance(node, list):
        merged: dict = {}
        for elem in node:
            if not (isinstance(elem, dict) and len(elem) == 1):
                raise LoadError(
                    f"a list of fields must hold single-key maps, got {elem!r}"
                )
            merged.update(elem)
        return read_type(merged, registry)

    raise LoadError(f"cannot read type from {node!r} (type {type(node).__name__})")


# ---------- flow `inputs:` declarations ----------


def _split_default(spec: str) -> tuple[str, Optional[str]]:
    """Split a `TYPE = default` spec on the FIRST top-level `=` (outside []/{}/quotes).

    Returns `(type_part, default_part)`; `default_part` is None when there is no `=`
    (a bare type). A `==`/`>=`/`<=`/`!=` is not a default assignment and is skipped.
    """
    depth = 0
    quote: Optional[str] = None
    for i, ch in enumerate(spec):
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        elif ch == "=" and depth == 0:
            prev = spec[i - 1] if i else ""
            nxt = spec[i + 1] if i + 1 < len(spec) else ""
            if prev in "=<>!" or nxt == "=":
                continue  # part of ==/<=/>=/!=, not a default assignment
            return spec[:i].strip(), spec[i + 1 :].strip()
    return spec.strip(), None


def read_flow_inputs(
    mapping,
    registry: TypeRegistry,
    ranges: Mapping[str, SourceRange | None] | None = None,
) -> list[Parameter]:
    """Read Compose ``inputs:`` directly into canonical semantic parameters.

    A `str` value is the `TYPE [= default]` / `Optional[X]` form; a `dict` value is a
    record type (resolved via `read_type`, no default — the `{type:, default:}`
    escape-hatch map is intentionally not supported). Authored defaults must already
    satisfy their declared type; only supplied boundary strings are coercible.
    """
    parameters: list[Parameter] = []
    for name, value in (mapping or {}).items():
        source_range = None if ranges is None else ranges.get(name)
        try:
            if isinstance(value, str):
                type_part, default_part = _split_default(value)
                typ = read_type(type_part, registry)
                has_default = default_part is not None
                if has_default and default_part == "":
                    raise LoadError(f"input {name!r}: default cannot be empty")
                default: Any = None
                if has_default:
                    assert default_part is not None
                    try:
                        default = parse_yaml_literal(default_part)
                    except SourceSyntaxError as exc:
                        raise LoadError(
                            f"input {name!r}: bad default {default_part!r}: {exc}"
                        ) from exc
                    try:
                        build_value_as(typ, default)
                    except ValueError as exc:
                        raise LoadError(
                            f"input {name!r}: default {default!r} does not satisfy "
                            f"the declared type: {exc}"
                        ) from exc
                parameters.append(
                    Parameter(
                        name=name,
                        contract=ValueContract(typ),
                        required=not typ.nullable and not has_default,
                        has_default=has_default,
                        default=default,
                        coerce_strings=True,
                    )
                )
            elif isinstance(value, dict):
                typ = read_type(value, registry)
                parameters.append(
                    Parameter(
                        name=name,
                        contract=ValueContract(typ),
                        required=not typ.nullable,
                        coerce_strings=True,
                    )
                )
            else:
                raise LoadError(
                    f"input {name!r}: declaration must be a type string or a record map, "
                    f"got {type(value).__name__}"
                )
        except LoadError as exc:
            if exc.range is not None or source_range is None:
                raise
            raise LoadError(
                str(exc),
                lines=exc.lines,
                notes=exc.notes,
                range=source_range,
            ) from exc
    return parameters

"""Compose type declarations -> source-neutral runtime contracts.

A node here is a leaf type string (`"float"`, `"list[str]"`, a registry name), a
map of field -> node (a record, nested natively), or a list of single-key maps
(the list-of-fields style). Resolution of leaf strings and nested records is the
type layer's job (`type_for` recurses over aliases/lists/Optional); this reader
walks native record syntax and wraps any type error loudly. ``read_type`` is the
source-free semantic view. ``read_value_contract`` performs the same recursion
over a retained cursor and compiles authored root/nested sites into opaque
``ContractOrigins`` without leaking YAML concepts below Compose.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Optional

from agent_composer.nodes.contracts import (
    ContractOrigins,
    Parameter,
    ValueContract,
    ValuePath,
)
from agent_composer.typesys import TypeCheckError, Type, build_value_as, type_for
from agent_composer.typesys.values import ValueKind
from agent_composer.typesys.types import TypeRegistry

from agent_composer.compose.helpers.errors import LoadDiagnostics, LoadError
from agent_composer.events import SourceRange
from agent_composer.compose.source import (
    SourceAccessError,
    SourceCursor,
    SourceSyntaxError,
    parse_yaml_literal,
)
from agent_composer.diagnostics import OriginRef

if TYPE_CHECKING:
    from agent_composer.compose.source.catalog import _SourceCatalogBuilder


_ContractReader = Callable[[Any], ValueContract]


def _path_sort_key(path: ValuePath) -> tuple[tuple[int, str | int], ...]:
    """Return the source-neutral canonical order required by ``ContractOrigins``."""
    return tuple(
        (0, segment) if type(segment) is str else (1, segment) for segment in path
    )


def _read_record_type(
    entries: list[tuple[Any, Any, SourceCursor | None]],
    registry: TypeRegistry,
    *,
    path: ValuePath,
    diagnostics: LoadDiagnostics,
) -> tuple[Type, list[tuple[ValuePath, SourceCursor]]]:
    """Read normalized ``(field, value, cursor)`` entries into one record type."""
    fields: dict[str, Type] = {}
    sites: list[tuple[ValuePath, SourceCursor]] = []
    for key, value, child_cursor in entries:
        child_path = (*path, key)
        if child_cursor is not None:
            sites.append((child_path, child_cursor))
        field_type, child_sites = _read_type_and_sites(
            value,
            registry,
            cursor=child_cursor,
            path=child_path,
            diagnostics=diagnostics,
        )
        fields[key] = field_type
        sites.extend(child_sites)
    return (
        Type(
            kind=ValueKind.OBJECT,
            fields=fields,
            required=frozenset(
                key for key, field_type in fields.items() if not field_type.nullable
            ),
        ),
        sites,
    )


def _read_type_and_sites(
    node: Any,
    registry: TypeRegistry,
    *,
    cursor: SourceCursor | None,
    path: ValuePath,
    diagnostics: LoadDiagnostics,
) -> tuple[Type, list[tuple[ValuePath, SourceCursor]]]:
    """Read one type node and collect each physically authored record-field site."""
    if isinstance(node, str):
        try:
            return type_for(node, registry), []
        except TypeCheckError as exc:
            raise diagnostics.error(f"bad type expression {node!r}: {exc}") from exc

    if isinstance(node, dict):
        return _read_record_type(
            [
                (
                    key,
                    value,
                    cursor.child(key) if cursor is not None else None,
                )
                for key, value in node.items()
            ],
            registry,
            path=path,
            diagnostics=diagnostics,
        )

    if isinstance(node, list):
        # The list syntax is another spelling of one record. Preserve the same
        # first-key order / last-value semantics as the previous ``dict.update``
        # implementation while retaining the winning value occurrence.
        merged: dict[Any, tuple[Any, SourceCursor | None]] = {}
        for index, element in enumerate(node):
            if not (isinstance(element, dict) and len(element) == 1):
                raise diagnostics.error(
                    f"a list of fields must hold single-key maps, got {element!r}"
                )
            key, value = next(iter(element.items()))
            item_cursor = cursor.child(index) if cursor is not None else None
            child_cursor = item_cursor.child(key) if item_cursor is not None else None
            merged[key] = (value, child_cursor)

        return _read_record_type(
            [
                (key, value, child_cursor)
                for key, (value, child_cursor) in merged.items()
            ],
            registry,
            path=path,
            diagnostics=diagnostics,
        )

    raise diagnostics.error(
        f"cannot read type from {node!r} (type {type(node).__name__})"
    )


def read_type(
    node,
    registry: TypeRegistry,
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> Type:
    """Read a Compose source node into one runtime `Type` (recursively)."""
    diagnostics = diagnostics or LoadDiagnostics()
    typ, _sites = _read_type_and_sites(
        node,
        registry,
        cursor=None,
        path=(),
        diagnostics=diagnostics,
    )
    return typ


def read_value_contract(
    cursor: SourceCursor,
    registry: TypeRegistry,
    *,
    site_path: tuple[str | int, ...],
    catalog_builder: "_SourceCatalogBuilder",
    diagnostics: LoadDiagnostics | None = None,
) -> ValueContract:
    """Compile one authored type declaration and its nested source sites.

    ``site_path`` is the semantic prefix for this contract. The reader appends
    ``root`` for the whole declaration and ``path/<segments...>`` for each
    physically authored record field. A scalar list/alias has no interior YAML
    declaration, so element failures correctly fall back to the root.
    """
    diagnostics = diagnostics or LoadDiagnostics.from_cursor(cursor)
    try:
        typ, sites = _read_type_and_sites(
            cursor.value,
            registry,
            cursor=cursor,
            path=(),
            diagnostics=diagnostics,
        )
        default = catalog_builder.register((*site_path, "root"), cursor)
        entries: list[tuple[ValuePath, OriginRef]] = []
        for path, field_cursor in sorted(
            sites,
            key=lambda item: _path_sort_key(item[0]),
        ):
            origin = catalog_builder.register(
                (*site_path, "path", *path),
                field_cursor,
            )
            if origin is not None:
                entries.append((path, origin))
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc
    return ValueContract(
        type=typ,
        origins=ContractOrigins(default=default, entries=tuple(entries)),
    )


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


def _read_flow_parameter(
    name: str,
    value: Any,
    *,
    read_contract: _ContractReader,
    parameter_origin: OriginRef | None = None,
    default_origin: Callable[[], OriginRef | None] | None = None,
    diagnostics: LoadDiagnostics,
) -> Parameter:
    """Read one flow-input declaration using an injected contract compiler."""
    if isinstance(value, str):
        type_part, default_part = _split_default(value)
        contract = read_contract(type_part)
        typ = contract.type
        has_default = default_part is not None
        if has_default and default_part == "":
            raise diagnostics.error(f"input {name!r}: default cannot be empty")
        default: Any = None
        if has_default:
            assert default_part is not None
            try:
                default = parse_yaml_literal(default_part)
            except SourceSyntaxError as exc:
                raise diagnostics.error(
                    f"input {name!r}: bad default {default_part!r}: {exc}"
                ) from exc
            try:
                build_value_as(typ, default)
            except ValueError as exc:
                raise diagnostics.error(
                    f"input {name!r}: default {default!r} does not satisfy "
                    f"the declared type: {exc}"
                ) from exc
        return Parameter(
            name=name,
            contract=contract,
            required=not typ.nullable and not has_default,
            has_default=has_default,
            default=default,
            coerce_strings=True,
            origin=parameter_origin,
            default_origin=(
                default_origin() if has_default and default_origin is not None else None
            ),
        )

    if isinstance(value, (dict, list)):
        contract = read_contract(value)
        return Parameter(
            name=name,
            contract=contract,
            required=not contract.type.nullable,
            coerce_strings=True,
            origin=parameter_origin,
        )

    raise diagnostics.error(
        f"input {name!r}: declaration must be a type string or a record map/list, "
        f"got {type(value).__name__}"
    )


def read_flow_inputs(
    mapping,
    registry: TypeRegistry,
    ranges: Mapping[str, SourceRange | None] | None = None,
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> list[Parameter]:
    """Read Compose ``inputs:`` directly into canonical semantic parameters.

    A `str` value is the `TYPE [= default]` / `Optional[X]` form; a mapping or
    list-of-single-key-maps is a record type (resolved via `read_type`, no
    default — the `{type:, default:}` escape-hatch map is intentionally not
    supported). Authored defaults must already satisfy their declared type;
    only supplied boundary strings are coercible. This source-free entry point
    deliberately emits no origins.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    parameters: list[Parameter] = []
    for name, value in (mapping or {}).items():
        source_range = None if ranges is None else ranges.get(name)
        try:
            parameters.append(
                _read_flow_parameter(
                    name,
                    value,
                    read_contract=lambda type_node: ValueContract(
                        read_type(type_node, registry, diagnostics=diagnostics)
                    ),
                    diagnostics=diagnostics,
                )
            )
        except LoadError as exc:
            if exc.diagnostic.frames or source_range is None:
                raise
            raise diagnostics.prepend(
                exc,
                range=source_range,
            ) from exc
    return parameters


def read_source_flow_inputs(
    cursor: SourceCursor,
    registry: TypeRegistry,
    *,
    site_path: tuple[str | int, ...],
    catalog_builder: "_SourceCatalogBuilder",
    diagnostics: LoadDiagnostics | None = None,
) -> list[Parameter]:
    """Compile authored flow inputs into located semantic parameters.

    ``site_path`` identifies the containing flow. Each parameter receives
    stable child identities for its key declaration, contract tree, and packed
    default. The resulting carriers retain only ``OriginRef`` values.
    """
    diagnostics = diagnostics or LoadDiagnostics.from_cursor(cursor)
    mapping = cursor.value
    if not isinstance(mapping, Mapping):
        raise diagnostics.error(
            f"flow input must be a mapping, got {type(mapping).__name__}",
            range=cursor.range,
        )

    parameters: list[Parameter] = []
    try:
        for name, value in mapping.items():
            declaration_cursor = cursor.child(name)
            parameter_path = (*site_path, "input", name)
            parameter_origin = catalog_builder.register(
                (*parameter_path, "parameter"),
                declaration_cursor,
                anchor="key",
            )

            def contract_reader(type_node: Any) -> ValueContract:
                """Compile this declaration's type-only view at its value site."""
                contract_cursor = (
                    declaration_cursor.with_value(type_node)
                    if isinstance(value, str)
                    else declaration_cursor
                )
                return read_value_contract(
                    contract_cursor,
                    registry,
                    site_path=(*parameter_path, "contract"),
                    catalog_builder=catalog_builder,
                    diagnostics=diagnostics,
                )

            try:
                parameters.append(
                    _read_flow_parameter(
                        name,
                        value,
                        read_contract=contract_reader,
                        parameter_origin=parameter_origin,
                        default_origin=lambda: catalog_builder.register(
                            (*parameter_path, "default"),
                            declaration_cursor,
                        ),
                        diagnostics=diagnostics,
                    )
                )
            except LoadError as exc:
                if exc.diagnostic.frames or declaration_cursor.range is None:
                    raise
                raise diagnostics.prepend(
                    exc,
                    range=declaration_cursor.range,
                ) from exc
    except SourceAccessError as exc:
        raise diagnostics.error(str(exc), range=exc.range) from exc
    return parameters

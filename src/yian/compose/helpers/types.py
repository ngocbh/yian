"""Compose type declarations -> source-neutral runtime contracts.

A node here is a leaf type string (`"float"`, `"list(str)"`, a registry name), a
map of field -> node (a record, nested natively), or a list of single-key maps
(the list-of-fields style). Resolution of leaf strings and nested records is the
type layer's job (`type_for` recurses over aliases/lists/Optional); this reader
walks native record syntax and wraps any type error loudly. ``read_type`` is the
source-free semantic view. ``read_value_contract`` performs the same recursion
over a retained cursor and compiles authored root/nested sites into opaque
``ContractOrigins`` without leaking YAML concepts below Compose.
"""

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from yian.nodes.contracts import (
    ContractOrigins,
    Parameter,
    ValueContract,
    ValuePath,
)
from yian.typesys import TypeCheckError, Type, type_for
from yian.typesys.values import ValueKind
from yian.typesys.types import BUILTIN_TYPE_NAMES, TypeRegistry, encodes_as_utf8

from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.secret_literals import (
    input_ref,
    is_binding_text,
    secret_literal_message,
    secret_literal_path,
)
from yian.compose.helpers.tags import (
    close_type_name,
    old_form_suggestion,
    read_tagged_value,
    resolve_type_text,
    spell_type_text,
    split_typed_text,
    suggest_tag,
    yaml_quoted,
)
from yian.diagnostics import SourceRange
from yian.compose.source import (
    SourceAccessError,
    SourceCursor,
    SourceSyntaxError,
    parse_yaml_literal,
)
from yian.diagnostics import OriginRef
from yian.secret_names import is_secret_name

if TYPE_CHECKING:
    from yian.compose.source.catalog import _SourceCatalogBuilder

_HEAD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Text this plain is safe to show unquoted after a tag, once it reads back as text.
_PLAIN_TEXT = re.compile(r"[A-Za-z][A-Za-z0-9 _.,'-]*")


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
        if child_cursor is not None and (type(key) is not str or not key):
            raise diagnostics.error(
                "record field names must be non-empty strings",
                range=child_cursor.key_range or child_cursor.range,
            )
        if child_cursor is not None and not encodes_as_utf8(key):
            raise diagnostics.error(
                f"record field name {key!r}: field names must not contain a Unicode surrogate",
                range=child_cursor.key_range or child_cursor.range,
            )
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
            raise diagnostics.error(
                f"bad type expression {node!r}: {exc}",
                range=(cursor.range if cursor is not None else None),
            ) from exc

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
            item_cursor = cursor.child(index) if cursor is not None else None
            if not (isinstance(element, dict) and len(element) == 1):
                raise diagnostics.error(
                    f"a list of fields must hold single-key maps, got {element!r}",
                    range=(item_cursor.range if item_cursor is not None else None),
                )
            key, value = next(iter(element.items()))
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
        f"cannot read type from {node!r} (type {type(node).__name__})",
        range=(cursor.range if cursor is not None else None),
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


# ---------- flow `input:` declarations ----------


_SOURCE_FREE_DEFAULT = (
    "source-free declarations take a type only; write defaults in YAML with a tag"
)


def _read_type_only(
    name: str,
    value: Any,
    registry: TypeRegistry,
    *,
    diagnostics: LoadDiagnostics,
) -> Parameter:
    """Read one source-free input declaration, which is type text with no default.

    Raises:
        `LoadError`: The value is not text, or holds a default (`T = default`) or is
            not a type.
    """
    if not isinstance(value, str):
        raise diagnostics.error(
            f"input {name!r}: a source-free declaration is type text, "
            f"got {type(value).__name__}"
        )
    try:
        typ = type_for(value, registry)
    except TypeCheckError as exc:
        if "=" in value:
            raise diagnostics.error(f"input {name!r}: {_SOURCE_FREE_DEFAULT}") from None
        raise diagnostics.error(
            f"input {name!r}: bad type expression {value!r}: {exc}"
        ) from exc
    return Parameter(
        name=name,
        contract=ValueContract(typ),
        required=not typ.nullable,
        coerce_strings=True,
    )


def _secret_default_error(
    name: str, declaration_label: str, diagnostics: LoadDiagnostics
) -> LoadError:
    """Word the error for a literal default under a secret name; it omits the value.

    Args:
        name (`str`): The declaration's name.
        declaration_label (`str`): What the declaration is, `"input"` or `"variable"`.
        diagnostics (`LoadDiagnostics`): Factory for load errors.

    Returns:
        `LoadError`: The error to raise.
    """
    if declaration_label == "variable":
        return diagnostics.error(
            f"variable {name!r}: a secret name can't be a variable, whose value is "
            f"written in the flow file; pass it as a node input ({input_ref(name)}) "
            "instead"
        )
    return diagnostics.error(
        f"{declaration_label} {name!r}: a secret name can't have a default value in "
        "the flow file; drop the default and pass the value when the flow runs"
    )


def read_flow_inputs(
    mapping,
    registry: TypeRegistry,
    ranges: Mapping[str, SourceRange | None] | None = None,
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> list[Parameter]:
    """Read source-free `input:` declarations, each a type text such as `Optional(int)`.

    A source-free declaration has no default: defaults are written in YAML after a
    tag. This entry point deliberately emits no origins.

    Raises:
        `LoadError`: A declaration is not type text or holds a default.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    parameters: list[Parameter] = []
    for name, value in (mapping or {}).items():
        source_range = None if ranges is None else ranges.get(name)
        try:
            parameters.append(
                _read_type_only(name, value, registry, diagnostics=diagnostics)
            )
        except LoadError as exc:
            if exc.diagnostic.frames or source_range is None:
                raise
            raise diagnostics.prepend(
                exc,
                range=source_range,
            ) from exc
    return parameters


def read_variables(
    mapping: Any,
    registry: TypeRegistry,
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> list[Parameter]:
    """Read source-free local declarations: only an empty mapping succeeds.

    A variable needs a default, and a source-free declaration can't carry one;
    defaults are written in YAML after a tag (`n: !int 4`) and read by
    `read_source_parameters`.

    Args:
        mapping (`Any`): The `variables:` section.
        registry (`TypeRegistry`): The flow's typedefs; unused, since no declaration
            is read.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`): Factory
            for load errors.

    Returns:
        `list[Parameter]`: Always empty.

    Raises:
        `LoadError`: `mapping` is not a mapping, a name is not a non-empty string, or
            it declares any variable.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    if not isinstance(mapping, Mapping):
        raise diagnostics.error("variables must be a mapping")
    for name in mapping:
        if type(name) is not str or not name:
            raise diagnostics.error("variable names must be non-empty strings")
        raise diagnostics.error(
            f"variable {name!r}: a variable needs a default; {_SOURCE_FREE_DEFAULT}"
        )
    return []


def read_source_flow_inputs(
    cursor: SourceCursor,
    registry: TypeRegistry,
    *,
    site_path: tuple[str | int, ...],
    catalog_builder: "_SourceCatalogBuilder",
    diagnostics: LoadDiagnostics | None = None,
) -> list[Parameter]:
    """Compile authored flow inputs using the shared located declaration reader."""
    return read_source_parameters(
        cursor,
        registry,
        site_path=site_path,
        catalog_builder=catalog_builder,
        diagnostics=diagnostics,
    )


def _read_tagged_parameter(
    name: str,
    cursor: SourceCursor,
    registry: TypeRegistry,
    *,
    parameter_path: tuple[str | int, ...],
    parameter_origin: OriginRef | None,
    catalog_builder: "_SourceCatalogBuilder",
    diagnostics: LoadDiagnostics,
    declaration_label: str,
    require_default: bool,
) -> Parameter:
    """Read one tagged declaration: the tag is its type, a value after it its default.

    A declaration with no value after the tag (`qty: !int`) has no default and is
    required unless its type is nullable. A default is part of the flow file, so
    under a secret name it must be `null` or `${...}` text, and no default may hold
    a literal under a nested secret key.
    """
    if cursor.has_value:
        # Checked before the default is read or type-checked, since those errors
        # quote it.
        default = cursor.value
        if is_secret_name(name) and default is not None and not is_binding_text(default):
            raise _secret_default_error(name, declaration_label, diagnostics)
        path = secret_literal_path(None, default)
        if path is not None:
            raise diagnostics.error(
                secret_literal_message(
                    "default", name, path, owner=f"{declaration_label} {name!r}"
                )
            )
    slot = read_tagged_value(
        cursor,
        registry,
        label=name,
        # The same contract prefix an untagged declaration's type registers under.
        site_path=(*parameter_path, "contract"),
        catalog_builder=catalog_builder,
        diagnostics=diagnostics,
        ledger=catalog_builder.consumed_tags,
        literal_only=True,
    )
    contract = slot.contract
    assert contract is not None
    if require_default and not slot.has_value:
        spelled = suggest_tag(name, contract.type, "<default>", type_text=cursor.tag)
        raise diagnostics.error(
            f"{declaration_label} {name!r}: a {declaration_label} needs a default: "
            f"{spelled}",
            range=cursor.range,
        )
    return Parameter(
        name=name,
        contract=contract,
        required=not contract.type.nullable and not slot.has_value,
        has_default=slot.has_value,
        default=slot.value,
        coerce_strings=True,
        origin=parameter_origin,
        default_origin=(
            catalog_builder.register((*parameter_path, "default"), cursor)
            if slot.has_value
            else None
        ),
    )


def _untagged_declaration_error(
    name: str,
    value: Any,
    registry: TypeRegistry,
    *,
    declaration_label: str,
    require_default: bool,
) -> str:
    """Explain how to write an untagged input or variable declaration with a tag.

    Args:
        name (`str`): The declared name.
        value (`Any`): The untagged value: old `T` or `T = default` text, a record
            map, or anything else.
        registry (`TypeRegistry`): The flow's typedefs the old type text resolves
            against.
        declaration_label (`str`): `"input"` or `"variable"`.
        require_default (`bool`): Whether the declaration needs a default.

    Returns:
        `str`: The error message.
    """
    noun = "flow input" if declaration_label == "input" else declaration_label
    head = f"{noun} {name!r}"
    if isinstance(value, dict):
        record = "".join(part[:1].upper() + part[1:] for part in name.split("_"))
        return (
            f"{head}: name the record in `typedefs:` and tag the {declaration_label} "
            f"with it, e.g. `{name}: !{record or 'Record'}`"
        )
    if isinstance(value, str):
        # The whole text first: a type such as `Literal("a = b")` may hold ` = `.
        split = split_typed_text(value)
        candidates = [(value.strip(), "")] + ([split] if split is not None else [])
        for type_text, default in candidates:
            type_text = _type_text_with_known_head(type_text, registry)
            if type_text is None:
                continue  # value text, not a type
            resolved = resolve_type_text(type_text, registry)
            # Text whose leading name is a type but whose arguments name none still
            # gets its tag, so pasting it back reaches the tag reader's unknown-type
            # error.
            if resolved is not None:
                typ, spelled = resolved
            else:
                typ, spelled = None, spell_type_text(type_text)
            if spelled is not None:
                suggestion = old_form_suggestion(
                    name,
                    typ,
                    spelled,
                    default,
                    registry=registry,
                    placeholder="<default>" if require_default else None,
                )
                return f"{head}: write the type as a tag: {suggestion}"
    tag, example = _example_default(value)
    if require_default:
        return (
            f"{head}: a {declaration_label} is a tag with a default, "
            f"e.g. `{name}: !{tag} {example}`"
        )
    return (
        f"{head}: an {declaration_label} is a tag with an optional default, "
        f"e.g. `{name}: !{tag}` or `{name}: !{tag} {example}`"
    )


def _type_text_with_known_head(text: str, registry: TypeRegistry) -> str | None:
    """Return type text whose leading name is a type name, else `None`.

    A leading name close to a builtin or typedef name is replaced with that name
    (`strr` gives `str`); text whose leading name is neither is a value.

    Args:
        text (`str`): Authored text that may be a type.
        registry (`TypeRegistry`): The flow's typedefs, offered with the builtins.

    Returns:
        `str | None`: The text with a type name in front, or `None`.
    """
    text = text.strip()
    head = _HEAD_NAME.match(text)
    if head is None:
        return None
    if head.group() in BUILTIN_TYPE_NAMES or head.group() in registry:
        return text
    close = close_type_name(text, registry)
    return None if close is None else close + text[head.end():]


def _example_default(value: Any) -> tuple[str, str]:
    """Spell a tag and a default for an untagged value, as in `!str hello`.

    Args:
        value (`Any`): The untagged value.

    Returns:
        `tuple[str, str]`: The tag's type and the default written after it: the
            value itself when it is text or a scalar, else `int` and `30`.
    """
    if isinstance(value, bool):
        return "bool", "true" if value else "false"
    if isinstance(value, int):
        return "int", str(value)
    if isinstance(value, float):
        return "float", repr(value)
    if isinstance(value, str) and value.strip():
        return "str", value if _reads_as_text(value) else yaml_quoted(value)
    return "int", "30"


def _reads_as_text(text: str) -> bool:
    """Whether `text` written unquoted after a tag reads back as exactly that text."""
    if not _PLAIN_TEXT.fullmatch(text):
        return False
    try:
        return parse_yaml_literal(text) == text
    except SourceSyntaxError:
        return False


def read_source_parameters(
    cursor: SourceCursor,
    registry: TypeRegistry,
    *,
    site_path: tuple[str | int, ...],
    catalog_builder: "_SourceCatalogBuilder",
    diagnostics: LoadDiagnostics | None = None,
    field_name: str = "input",
    declaration_label: str = "input",
    require_default: bool = False,
) -> list[Parameter]:
    """Compile authored input or variable declarations into located parameters.

    A declaration takes its type from its tag and its default from the value after
    it (`n: !int(1..12) 4`); with no value (`qty: !int`) it has no default. An
    untagged declaration is an error that shows the tagged spelling.
    ``site_path`` identifies the containing owner; ``field_name`` its section.
    ``declaration_label`` names individual errors, and ``require_default``
    selects declarations whose values must include a typed literal default.
    Each parameter receives
    stable child identities for its key declaration, contract tree, and packed
    default. The resulting carriers retain only ``OriginRef`` values.
    """
    diagnostics = diagnostics or LoadDiagnostics.from_cursor(cursor)
    mapping = cursor.value
    section_label = "flow input" if field_name == "input" else field_name
    if not isinstance(mapping, Mapping):
        raise diagnostics.error(
            f"{section_label} must be a mapping, got {type(mapping).__name__}",
            range=cursor.range,
        )

    parameters: list[Parameter] = []
    try:
        for name, value in mapping.items():
            declaration_cursor = cursor.child(name)
            if type(name) is not str or not name:
                raise diagnostics.error(
                    f"{section_label} names must be non-empty strings",
                    range=declaration_cursor.key_range or declaration_cursor.range,
                )
            parameter_path = (*site_path, field_name, name)
            parameter_origin = catalog_builder.register(
                (*parameter_path, "parameter"),
                declaration_cursor,
                anchor="key",
            )

            try:
                if declaration_cursor.tag is None:
                    raise diagnostics.error(
                        _untagged_declaration_error(
                            name,
                            value,
                            registry,
                            declaration_label=declaration_label,
                            require_default=require_default,
                        ),
                        range=declaration_cursor.range,
                    )
                parameters.append(
                    _read_tagged_parameter(
                        name,
                        declaration_cursor,
                        registry,
                        parameter_path=parameter_path,
                        parameter_origin=parameter_origin,
                        catalog_builder=catalog_builder,
                        diagnostics=diagnostics,
                        declaration_label=declaration_label,
                        require_default=require_default,
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

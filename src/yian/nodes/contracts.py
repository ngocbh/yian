"""Source-neutral semantic IR shared by node and flow construction.

These frozen carriers attach opaque origins to executable meanings without
depending on Compose/YAML or runtime state. Compiled wiring, node assertions,
node parameters, and node outputs use their canonical carriers. Execution scopes
identify authored locations and structured occurrences independently of memory.
Seed connections and child boundary pairs describe growth topology without values
or runtime producer identities.
Compose output derivation preserves those carriers and projects exact field/value
types only inside type algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, TypeAlias, cast

from yian.diagnostics import OriginRef
from yian.typesys import (
    FieldAbsent,
    FieldPartial,
    Type,
    TypeCheckError,
    TypedValue,
    ValueKind,
    build_value_as,
    canonical_type_label,
    union_field,
    union_tag_fields,
)
from yian.typesys.values import (
    _sort_repr,
    file_field_type,
    has_file,
    shown_text,
    value_repr,
)


ValuePath: TypeAlias = tuple[str | int, ...]
"""A source-neutral path through record fields and list indices."""

_OriginMapper: TypeAlias = Callable[[OriginRef], OriginRef]


def _validate_optional_origin(origin: OriginRef | None, field_name: str) -> None:
    """Require an optional origin field to contain only the opaque leaf value."""
    if origin is not None and not isinstance(origin, OriginRef):
        raise TypeError(f"{field_name} must be an OriginRef or None")


def _map_origin(origin: OriginRef | None, mapper: _OriginMapper) -> OriginRef | None:
    """Transform one present origin and reject erasure or a non-origin result."""
    if origin is None:
        return None
    mapped = mapper(origin)
    if not isinstance(mapped, OriginRef):
        raise TypeError("origin mapper must return an OriginRef")
    return mapped


def _validate_path(path: ValuePath, *, field_name: str) -> None:
    """Validate one immutable value path using exact string/integer segments."""
    if type(path) is not tuple:
        raise TypeError(f"{field_name} path must be a tuple")
    for segment in path:
        if type(segment) not in (str, int):
            raise TypeError(
                f"{field_name} path segment must be an exact str or int, "
                f"got {type(segment).__name__}"
            )


def _path_sort_key(path: ValuePath) -> tuple[tuple[int, str | int], ...]:
    """Return a Python-3.10-safe total key with strings before integers."""
    return tuple(
        (0, segment) if type(segment) is str else (1, segment) for segment in path
    )


def _validate_authored_path(path: tuple[str, ...], name: str) -> None:
    """Require immutable authored names while allowing the empty root path."""
    if type(path) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if any(type(part) is not str or not part for part in path):
        raise ValueError(f"{name} components must be non-empty strings")


@dataclass(frozen=True, slots=True)
class ExecutionPosition:
    """One structured position in an instantiated child computation.

    Attributes:
        kind (`Literal["call", "map_item", "loop_body", "loop_driver"]`):
            The authored composition operation and, for LOOP, its execution role.
        static_path (`tuple[str, ...]`): Root-relative path of the owning spawner.
        index (`int | None`): Non-negative item or iteration index. CALL uses
            `None` because it instantiates exactly one child per execution.
    """

    kind: Literal["call", "map_item", "loop_body", "loop_driver"]
    static_path: tuple[str, ...]
    index: int | None = None

    def __post_init__(self) -> None:
        """Validate structured positions without interpreting runtime display IDs."""
        if type(self.kind) is not str or self.kind not in (
            "call",
            "map_item",
            "loop_body",
            "loop_driver",
        ):
            raise ValueError("unknown execution position kind")
        _validate_authored_path(self.static_path, "static_path")
        if self.kind == "call":
            if self.index is not None:
                raise ValueError("a call position cannot have an index")
        elif type(self.index) is not int:
            raise TypeError("execution position index must be an integer")
        elif self.index < 0:
            raise ValueError("execution position index must be non-negative")


@dataclass(frozen=True, slots=True)
class ExecutionScope:
    """An authored execution location independent of memory declarations.

    Attributes:
        static_path (`tuple[str, ...]`): Root-relative authored node or flow path.
            The empty tuple identifies the root itself, including unnamed flows.
        flow_path (`tuple[str, ...]`): Root-relative path of the containing flow;
            the empty tuple denotes the root flow.
        occurrence_path (`tuple[ExecutionPosition, ...]`): Outer-to-inner child
            positions. Exact dynamic identity additionally uses the run ID and
            accepted node-start sequence, supplied by the runtime.
        boundary (`Literal["start", "end"] | None`): This location's role as
            a flow boundary, when it has one.
    """

    static_path: tuple[str, ...]
    flow_path: tuple[str, ...] = ()
    occurrence_path: tuple[ExecutionPosition, ...] = ()
    boundary: Literal["start", "end"] | None = None

    def __post_init__(self) -> None:
        """Require immutable paths and explicit source-neutral occurrence data."""
        _validate_authored_path(self.static_path, "static_path")
        _validate_authored_path(self.flow_path, "flow_path")
        if type(self.occurrence_path) is not tuple:
            raise TypeError("occurrence_path must be a tuple")
        for position in self.occurrence_path:
            if type(position) is not ExecutionPosition:
                raise TypeError("occurrence_path items must be ExecutionPosition values")
            ExecutionPosition(position.kind, position.static_path, position.index)
        if self.boundary is not None and (
            type(self.boundary) is not str or self.boundary not in ("start", "end")
        ):
            raise ValueError("execution boundary must be 'start', 'end', or None")


@dataclass(frozen=True, slots=True)
class SeedConnection:
    """Connect a fragment input to a materialized parent seed route.

    The target is a child START parameter, or an input of another fragment node
    that is declared with a default and left unwired, such as a field a LOOP body
    omits and the next driver keeps from the carried record.

    ``child_start_id`` names the target node, which is a child START or a node with
    a defaulted unwired input.

    ``seed_path`` is a parameter name, or an explicit item index followed by a
    parameter name. Producer identities are supplied only by accepted runtime
    captures; this recipe contains no seed value or expression.
    """

    child_start_id: str
    input_name: str
    seed_path: tuple[str | int, ...]

    def __post_init__(self) -> None:
        """Require exact immutable names and a closed materialized-seed selector."""
        for name in (self.child_start_id, self.input_name):
            if type(name) is not str or not name:
                raise ValueError("seed connection identifiers must be non-empty strings")
        if type(self.seed_path) is not tuple or len(self.seed_path) not in (1, 2):
            raise ValueError("seed_path must be a name or an index/name tuple")
        if type(self.seed_path[-1]) is not str or not self.seed_path[-1]:
            raise ValueError("seed_path must end in a non-empty parameter name")
        if len(self.seed_path) == 2 and (
            type(self.seed_path[0]) is not int or self.seed_path[0] < 0
        ):
            raise ValueError("seed_path item index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ChildFlowBoundary:
    """The actual START and END of one instantiated authored child flow."""

    start_id: str
    end_id: str

    def __post_init__(self) -> None:
        """Require distinct exact identifiers without interpreting their spelling."""
        if any(type(name) is not str or not name for name in (self.start_id, self.end_id)):
            raise ValueError("child boundary identifiers must be non-empty strings")
        if self.start_id == self.end_id:
            raise ValueError("child START and END must be distinct")


@dataclass(frozen=True)
class Binding:
    """One flow-owned input expression or literal, its origin, and its declared type.

    The binding owns its declared contract: a typed binding states the type its
    value must have, independent of the receiving node's parameter contract.

    Attributes:
        source (`Any`): Semantic binding payload; retained by identity.
        origin (`OriginRef | None`): Authored binding site, when available.
        contract (`ValueContract | None`): Declared value contract, when typed.
    """

    source: Any
    origin: OriginRef | None = None
    contract: ValueContract | None = None

    def __post_init__(self) -> None:
        """Reject non-origin provenance values and non-contract declared types."""
        _validate_optional_origin(self.origin, "binding origin")
        if self.contract is not None and type(self.contract) is not ValueContract:
            raise TypeError("binding contract must be a ValueContract or None")

    def map_origins(self, mapper: _OriginMapper) -> Binding:
        """Return this binding with every origin transformed by ``mapper``.

        That covers its own origin, its contract's origins, and, for a
        ``ValueTemplate`` source, the template's origins and nested bindings.
        Any other source is retained by identity.
        """
        origin = _map_origin(self.origin, mapper)
        contract = (
            self.contract.map_origins(mapper) if self.contract is not None else None
        )
        source = (
            self.source.map_origins(mapper)
            if isinstance(self.source, ValueTemplate)
            else self.source
        )
        return Binding(source=source, origin=origin, contract=contract)

    def iter_origins(self) -> Iterator[OriginRef]:
        """Yield this binding's present origin, then its contract's origins.

        For a ``ValueTemplate`` source, the template's origins follow, then each
        nested binding's origin and contract origins, in tree order.
        """
        yield from _iter_origin_carrier(self, set())


@dataclass(frozen=True)
class ValueTemplate:
    """An authored mapping or list whose leaf strings are binding templates.

    The binder evaluates every leaf string as a template (a whole `${...}` keeps
    its type, an embedded one is stringified, `$$` is unescaped) and every nested
    typed ``Binding`` against its own contract. A plain ``dict`` or ``list``
    binding source, by contrast, is always a literal.

    A nested ``Binding`` sits at its position without adding a path segment; its
    source is part of the same tree (a template string or a nested container).

    Attributes:
        value (`Any`): The tree, a ``dict`` or ``list``; retained by identity.
        origins (`tuple[tuple[ValuePath, OriginRef], ...]`): Authored sites of
            leaf strings, keyed by their path in ``value``, for diagnostics.
    """

    value: Any
    origins: tuple[tuple[ValuePath, OriginRef], ...] = ()

    def __post_init__(self) -> None:
        """Require a container tree and well-formed origin entries."""
        if type(self.value) not in (dict, list):
            raise TypeError(
                "value template must hold a dict or list, "
                f"got {type(self.value).__name__}"
            )
        if type(self.origins) is not tuple:
            raise TypeError("value template origins must be a tuple")
        for entry in self.origins:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError("each value template origin must be a (path, origin) tuple")
            path, origin = entry
            _validate_path(path, field_name="value template origin")
            if not isinstance(origin, OriginRef):
                raise TypeError("value template origin must be an OriginRef")
        _validate_tree_keys(self.value)

    def origin_at(self, path: ValuePath) -> OriginRef | None:
        """Return the authored site registered for ``path``, if any."""
        for entry_path, origin in self.origins:
            if entry_path == path:
                return origin
        return None

    def map_origins(self, mapper: _OriginMapper) -> ValueTemplate:
        """Return this template with its origins and nested bindings' origins mapped."""
        return ValueTemplate(
            _map_tree(self.value, lambda text: text, mapper),
            tuple(
                (path, cast(OriginRef, _map_origin(origin, mapper)))
                for path, origin in self.origins
            ),
        )


def _validate_tree_keys(node: Any) -> None:
    """Require exact string mapping keys throughout a template tree.

    Keys become record route names, input-group path segments, and manifest
    mapping keys, all of which assume strings.
    """
    if isinstance(node, Binding):
        _validate_tree_keys(node.source)
    elif type(node) is dict:
        for key, item in node.items():
            if type(key) is not str:
                raise TypeError(
                    "value template mapping keys must be strings, "
                    f"got {type(key).__name__}"
                )
            _validate_tree_keys(item)
    elif type(node) is list:
        for item in node:
            _validate_tree_keys(item)


def _tree_bindings(node: Any) -> Iterator[Binding]:
    """Yield every binding nested in a template tree, each before those in its source."""
    if isinstance(node, Binding):
        yield node
        yield from _tree_bindings(node.source)
    elif type(node) is dict:
        for item in node.values():
            yield from _tree_bindings(item)
    elif type(node) is list:
        for item in node:
            yield from _tree_bindings(item)


def _tree_strings(node: Any, path: ValuePath) -> Iterator[tuple[ValuePath, str]]:
    """Yield a template tree's leaf strings, looking through nested bindings."""
    if isinstance(node, Binding):
        yield from _tree_strings(node.source, path)
    elif isinstance(node, str):
        yield path, node
    elif type(node) is dict:
        for key, item in node.items():
            yield from _tree_strings(item, (*path, key))
    elif type(node) is list:
        for index, item in enumerate(node):
            yield from _tree_strings(item, (*path, index))


def _map_tree(
    node: Any,
    fn: Callable[[str], Any],
    mapper: _OriginMapper | None = None,
) -> Any:
    """Rebuild a template tree with ``fn`` applied to each leaf string.

    Nested bindings keep their contracts; with ``mapper`` their origins are
    transformed too.
    """
    if isinstance(node, Binding):
        return Binding(
            source=_map_tree(node.source, fn, mapper),
            origin=node.origin if mapper is None else _map_origin(node.origin, mapper),
            contract=(
                node.contract
                if mapper is None or node.contract is None
                else node.contract.map_origins(mapper)
            ),
        )
    if isinstance(node, str):
        return fn(node)
    if type(node) is dict:
        return {key: _map_tree(item, fn, mapper) for key, item in node.items()}
    if type(node) is list:
        return [_map_tree(item, fn, mapper) for item in node]
    return node


def iter_source_strings(source: Any) -> Iterator[tuple[ValuePath, str]]:
    """Yield the template strings of one binding source with their paths.

    Exactly one ``Binding`` layer is unwrapped. A plain string yields itself at
    the empty path; a ``ValueTemplate`` yields each leaf string, looking through
    nested bindings. Any other literal, including a plain ``dict`` or ``list``,
    yields nothing.

    Args:
        source (`Any`): A binding or its raw source.

    Returns:
        `Iterator[tuple[ValuePath, str]]`: ``(path, text)`` pairs in tree order.
    """
    value = binding_source(source)
    if isinstance(value, str):
        yield (), value
    elif isinstance(value, ValueTemplate):
        yield from _tree_strings(value.value, ())


def map_source_strings(source: Any, fn: Callable[[str], Any]) -> Any:
    """Rebuild one binding source with ``fn`` applied to each template string.

    The strings are those ``iter_source_strings`` yields. Bindings, templates,
    origins, and contracts are rebuilt around the new strings; a source with no
    template strings is returned unchanged.

    Args:
        source (`Any`): A binding or its raw source.
        fn (`Callable[[str], Any]`): Maps one template string to its replacement.

    Returns:
        `Any`: The rebuilt source, of the same carrier shape.
    """
    if isinstance(source, Binding):
        value = source.source
        if not isinstance(value, (str, ValueTemplate)):
            return source
        return Binding(
            source=map_source_strings(value, fn),
            origin=source.origin,
            contract=source.contract,
        )
    if isinstance(source, str):
        return fn(source)
    if isinstance(source, ValueTemplate):
        return ValueTemplate(_map_tree(source.value, fn), source.origins)
    return source


def _escape_group_segment(segment: str | int) -> str:
    """Escape one path segment as a JSON-pointer reference token."""
    return str(segment).replace("~", "~0").replace("/", "~1")


def binding_input_groups(param: str, source: Any) -> list[tuple[str, str]]:
    """Return the input group and template string for each string in a source.

    A plain string source is one group named ``param``. Each leaf string of a
    ``ValueTemplate`` is its own group, named ``param`` followed by its path as
    a JSON pointer (``~`` as ``~0``, ``/`` as ``~1``, list indices in decimal),
    so group names never collide. A source with no template strings has none.

    Args:
        param (`str`): The bound parameter's name.
        source (`Any`): The parameter's binding or raw source.

    Returns:
        `list[tuple[str, str]]`: ``(group, text)`` pairs in tree order.
    """
    return [
        (
            param + "".join("/" + _escape_group_segment(part) for part in path),
            text,
        )
        for path, text in iter_source_strings(source)
    ]


@dataclass(frozen=True)
class Assertion:
    """An executable assertion and its stable author-facing presentation.

    Attributes:
        expression (`str`): Expression evaluated after any lowering rewrite.
        display (`str`): Original text reported to the author.
        origin (`OriginRef | None`): Authored assertion occurrence, when known.
    """

    expression: str
    display: str
    origin: OriginRef | None = None

    def __post_init__(self) -> None:
        """Reject non-origin provenance values."""
        _validate_optional_origin(self.origin, "assertion origin")

    def map_origins(self, mapper: _OriginMapper) -> Assertion:
        """Return this assertion with its present origin transformed."""
        origin = _map_origin(self.origin, mapper)
        return Assertion(
            expression=self.expression,
            display=self.display,
            origin=origin,
        )

    def iter_origins(self) -> Iterator[OriginRef]:
        """Yield this assertion's present origin."""
        yield from _iter_origin_carrier(self, set())


@dataclass(frozen=True)
class ContractOrigins:
    """Immutable origin index for one value contract and its nested paths.

    ``default`` exclusively owns the root/fallback location. ``entries`` is a
    strictly sorted tuple of unique, non-empty ``(ValuePath, OriginRef)`` pairs.
    String path segments sort before integer segments; a prefix sorts before its
    descendants.

    Attributes:
        default (`OriginRef | None`): Root/fallback contract site.
        entries (`tuple[tuple[ValuePath, OriginRef], ...]`): Nested sites in
            canonical path order.
    """

    default: OriginRef | None = None
    entries: tuple[tuple[ValuePath, OriginRef], ...] = ()

    def __post_init__(self) -> None:
        """Reject mutable, ambiguous, duplicate, or noncanonical entries."""
        _validate_optional_origin(self.default, "contract default origin")
        if type(self.entries) is not tuple:
            raise TypeError("contract origin entries must be a tuple")

        paths: list[ValuePath] = []
        seen: set[ValuePath] = set()
        for entry in self.entries:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError("each contract origin entry must be a (path, origin) tuple")
            path, origin = entry
            _validate_path(path, field_name="contract origin entry")
            if not path:
                raise ValueError(
                    "contract origin entry path cannot be empty; use default for the root"
                )
            if not isinstance(origin, OriginRef):
                raise TypeError("contract origin entry value must be an OriginRef")
            if path in seen:
                raise ValueError(f"duplicate contract origin path: {path!r}")
            seen.add(path)
            paths.append(path)

        if paths != sorted(paths, key=_path_sort_key):
            raise ValueError("contract origin entries must be sorted canonically")

    def at(self, path: ValuePath) -> OriginRef | None:
        """Select the exact, nearest ancestor, or default origin for ``path``."""
        _validate_path(path, field_name="contract lookup")
        indexed = dict(self.entries)
        for length in range(len(path), 0, -1):
            origin = indexed.get(path[:length])
            if origin is not None:
                return origin
        return self.default

    def map_origins(self, mapper: _OriginMapper) -> ContractOrigins:
        """Return this index with every present origin transformed in order."""
        default = _map_origin(self.default, mapper)
        entries = tuple((path, _map_origin(origin, mapper)) for path, origin in self.entries)
        return ContractOrigins(default=default, entries=entries)

    def iter_origins(self) -> Iterator[OriginRef]:
        """Yield the default then nested origins in canonical path order."""
        yield from _iter_origin_carrier(self, set())


@dataclass(frozen=True)
class ValueContract:
    """A semantic runtime type paired with source-neutral nested provenance.

    Attributes:
        type (`Type`): Provenance-free recursive runtime type.
        origins (`ContractOrigins`): Root and nested authored contract sites.
    """

    type: Type
    origins: ContractOrigins = field(default_factory=ContractOrigins)

    def __post_init__(self) -> None:
        """Require the dedicated semantic and provenance value types."""
        if not isinstance(self.type, Type):
            raise TypeError("value contract type must be a Type")
        if not isinstance(self.origins, ContractOrigins):
            raise TypeError("value contract origins must be ContractOrigins")

    def map_origins(self, mapper: _OriginMapper) -> ValueContract:
        """Return this contract with nested origins transformed, retaining its type."""
        return ValueContract(type=self.type, origins=self.origins.map_origins(mapper))

    def iter_origins(self) -> Iterator[OriginRef]:
        """Yield this contract's root and nested origins in canonical order."""
        yield from _iter_origin_carrier(self, set())


@dataclass(frozen=True)
class PartialRecordContract:
    """A closed record shape whose individual field types may be unproven.

    Every field name is part of the contract: runtime values must contain exactly
    this key set. A present ``ValueContract`` enforces that field recursively;
    ``None`` keeps only the field's structural presence. At least one field must
    be opaque, otherwise one exact record ``ValueContract`` is canonical.

    Attributes:
        fields (`tuple[tuple[str, ValueContract | None], ...]`):
            Ordered field names paired with optional exact value contracts.
    """

    fields: tuple[tuple[str, ValueContract | None], ...]

    def __post_init__(self) -> None:
        """Require one immutable, unique, genuinely partial record declaration."""
        self.validate_shape()

    def validate_shape(self) -> None:
        """Reject malformed or noncanonical field metadata, including after copying."""
        if type(self.fields) is not tuple:
            raise TypeError("partial record fields must be a tuple")
        if not self.fields:
            raise ValueError("partial record contract must declare at least one field")
        names: set[str] = set()
        has_opaque_field = False
        for entry in self.fields:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError(
                    "partial record fields must be (name, optional contract) tuples"
                )
            name, contract = entry
            if type(name) is not str or not name:
                raise TypeError("partial record field names must be non-empty strings")
            if name in names:
                raise ValueError(f"duplicate partial record field {name!r}")
            names.add(name)
            if contract is None:
                has_opaque_field = True
            elif type(contract) is not ValueContract:
                raise TypeError(
                    "partial record field contracts must be exact ValueContract "
                    f"values or None, got {type(contract).__name__} for {name!r}"
                )
        if not has_opaque_field:
            raise ValueError(
                "a fully typed record must use one exact ValueContract"
            )

    @property
    def names(self) -> tuple[str, ...]:
        """Return the closed field-name sequence in declaration order."""
        return tuple(name for name, _contract in self.fields)

    def map_origins(self, mapper: _OriginMapper) -> PartialRecordContract:
        """Return this contract with every proven field origin transformed."""
        return PartialRecordContract(
            fields=tuple(
                (
                    name,
                    None if contract is None else contract.map_origins(mapper),
                )
                for name, contract in self.fields
            )
        )

    def iter_origins(self) -> Iterator[OriginRef]:
        """Yield proven field-contract origins in declaration order."""
        yield from _iter_origin_carrier(self, set())


OutputContract: TypeAlias = ValueContract | PartialRecordContract
"""The closed compiled output-contract sum: exact value or partial record."""


def output_contract_type(contract: OutputContract | None) -> Type | None:
    """Return an exact whole-value type, validating either contract arm's shapes."""
    if contract is None:
        return None
    if type(contract) is ValueContract:
        if not isinstance(contract.type, Type):
            raise TypeError("value contract type must be a Type")
        contract.type.validate_shape()
        return contract.type
    if type(contract) is PartialRecordContract:
        contract.validate_shape()
        for _name, field_contract in contract.fields:
            if field_contract is not None:
                if not isinstance(field_contract.type, Type):
                    raise TypeError("value contract type must be a Type")
                field_contract.type.validate_shape()
        return None
    raise TypeError(
        "output contract must be an exact ValueContract, "
        "PartialRecordContract, or None, "
        f"got {type(contract).__name__}"
    )


def file_fields_type(fields: Iterable[tuple[str, Type]]) -> Type | None:
    """Return a record of only the `fields` whose type may hold a file, none required.

    Read through `normalize_files`, the record turns each such field's file JSON forms
    into `FileRef`s and leaves every other field of a value as given.

    Args:
        fields (`Iterable[tuple[str, Type]]`):
            Field names paired with their types.

    Returns:
        `Type | None`:
            The record type, or `None` when no field may hold a file.

    Example:
        `file_fields_type([("f", file), ("n", int)])` returns the record `{f: file}`
        with no required field.
    """
    typed = {name: typ for name, typ in fields if has_file(typ)}
    if not typed:
        return None
    return Type(kind=ValueKind.OBJECT, fields=typed, required=frozenset())


def output_contract_file_type(contract: OutputContract | None) -> Type | None:
    """Return the type a value written under `contract` is read through to restore its files.

    A stored or JSON-decoded value holds each nested file as its JSON form; reading it
    through this type with `normalize_files` gives a `FileRef` at each position the
    contract types as `file`. A whole-value contract gives its type; a partial record
    gives `file_fields_type` of its typed fields, so its opaque fields are read as given.

    Args:
        contract (`OutputContract`, *optional*):
            The contract the value was written under. `None` has no file position.

    Returns:
        `Type | None`:
            The reading type, or `None` when no position may hold a file.

    Example:
        For `ValueContract(list(file))` it returns `list(file)`; for
        `ValueContract(int)` it returns `None`.
    """
    if type(contract) is ValueContract:
        return contract.type if has_file(contract.type) else None
    if type(contract) is PartialRecordContract:
        return file_fields_type(
            (name, field.type) for name, field in contract.fields if field is not None
        )
    return None


# Kinds a field read is not checked on: `object` is opaque unless it declares fields, and
# an `Any` value may be an object.
_UNCHECKED_FIELD_KINDS = frozenset({ValueKind.OBJECT, ValueKind.ANY})


def _member_only_field_error(
    union: Type,
    having: tuple[Type, ...],
    field_name: str,
    path: str,
    union_path: str,
    isinstance_hint: bool,
) -> str:
    """Return the error for reading a field only some members of `union` have.

    `path` is the whole reference path and `union_path` the path of the union value.
    `isinstance_hint` points the fix at an `isinstance` conjunct instead of a case.
    """
    tag_fields = union_tag_fields(union)
    if tag_fields:
        tag = tag_fields[0]
        tags = ", ".join(next(iter(member.fields[tag].tags)) for member in having)
        which = f"only union members with {tag} {tags} have"
        fix = f"read it inside a case on ${{{union_path}.{tag}}}"
    else:
        which = f"only {len(having)} of {len(union.variants)} union members have"
        fix = f"check the member with isinstance({union_path}, <Type>) in a case first"
    if isinstance_hint:
        fix = f"check the member first with `isinstance({union_path}, <Type>) and ...`"
    return f"reference ${{{path}}} reads field {field_name!r}, which {which}; {fix}"


def file_field_error(path: str, field_name: str) -> str:
    """Return the error for reading field `field_name`, which no `file` has.

    `path` is the whole reference path without `${}`.
    """
    return (
        f"reference ${{{path}}} reads field {field_name!r} on a file, "
        "which has only uri, mime, and name"
    )


def output_contract_field_error(
    contract: OutputContract,
    fields: list[str] | tuple[str, ...],
    path: str,
    *,
    partial_union_fields: bool = False,
    isinstance_hint: bool = False,
    union_root: str | None = None,
) -> str | None:
    """Return why a reference reads a field its contract can't have, else `None`.

    An exact record is closed at every proven nested record, and a partial record is
    closed at its root; reading below an opaque partial field is not checked. A `file`
    has only `uri`, `mime` and `name`, typed by `file_field_type`. Every other type but
    `object` and `Any` (scalars, `Literal[...]`, lists, and their `Optional[...]`
    forms) has no fields, so reading one is an error. On a union, a
    field every member has can be read, and the walk continues with its shared type
    (or stops when the members' types differ); a field no member has is an unknown
    field, and a field only some members have is an error unless
    `partial_union_fields` allows it.

    Args:
        contract (`OutputContract`):
            The exact or partial contract of the referenced value.
        fields (`list[str] | tuple[str, ...]`):
            The field names the reference reads below that value. May be empty.
        path (`str`):
            The reference path without `${}`, used in the message. Its last
            `len(fields)` segments are `fields`.
        partial_union_fields (`bool`, *optional*, defaults to `False`):
            Whether a field only some union members have can be read. The read is
            then not checked further.
        isinstance_hint (`bool`, *optional*, defaults to `False`):
            Whether a member-only field error tells the author to check the member
            with an `isinstance(..., T) and ...` conjunct before the read rather than
            in a case, for a condition no case can route (a LOOP predicate).
        union_root (`str | None`, *optional*, defaults to `None`):
            The pool path the referenced value is bound from, when `path` reads a
            node-local name. A member-only field error's fix then names the union
            from this path, which a case can route on, instead of from the local name.

    Returns:
        `str | None`:
            The error message, or `None` when the path can be read.

    Raises:
        `TypeError`:
            A field name is not a non-empty string, or `contract` is not an
            `OutputContract`.
        `ValueError`:
            A contract's type metadata is malformed.
    """
    if not isinstance(fields, (list, tuple)) or any(
        type(field) is not str or not field for field in fields
    ):
        raise TypeError("output contract fields must be non-empty strings")
    if type(contract) is PartialRecordContract:
        contract.validate_shape()
        if not fields:
            return None
        field_name, *remaining = fields
        field_contracts = dict(contract.fields)
        if field_name not in field_contracts:
            return f"reference ${{{path}}} reads unknown field {field_name!r} on record"
        field_contract = field_contracts[field_name]
        if field_contract is None:
            return None
        return output_contract_field_error(
            field_contract,
            remaining,
            path,
            partial_union_fields=partial_union_fields,
            isinstance_hint=isinstance_hint,
            union_root=None if union_root is None else f"{union_root}.{field_name}",
        )
    if type(contract) is not ValueContract:
        output_contract_type(contract)
        return None  # pragma: no cover - the validator above always raises

    typ = output_contract_type(contract)
    assert typ is not None
    # The path segments walked so far, starting with the ones before `fields`.
    walked = (
        path.split(".")[: path.count(".") + 1 - len(fields)]
        if union_root is None
        else [union_root]
    )
    for field_name in fields:
        if typ.variants is not None:
            read = union_field(typ, field_name)
            if isinstance(read, FieldAbsent):
                return f"reference ${{{path}}} reads unknown field {field_name!r} on record"
            if isinstance(read, FieldPartial):
                if partial_union_fields:
                    return None
                return _member_only_field_error(
                    typ, read.members, field_name, path, ".".join(walked), isinstance_hint
                )
            if read.type is None:
                return None
            typ = read.type
            walked.append(field_name)
            continue
        if typ.fields is None:
            if typ.kind in _UNCHECKED_FIELD_KINDS:
                return None
            if typ.kind is ValueKind.FILE:
                read_type = file_field_type(field_name)
                if read_type is None:
                    return file_field_error(path, field_name)
                typ = read_type
                walked.append(field_name)
                continue
            return (
                f"reference ${{{path}}} reads field {field_name!r} on type "
                f"{canonical_type_label(typ)}, which has no fields"
            )
        if field_name not in typ.fields:
            return f"reference ${{{path}}} reads unknown field {field_name!r} on record"
        typ = typ.fields[field_name]
        walked.append(field_name)
    return None


def output_contract_origin(
    contract: OutputContract | None,
    path: ValuePath,
) -> OriginRef | None:
    """Select the most precise authored origin for one output-validation path."""
    _validate_path(path, field_name="output contract lookup")
    if contract is None:
        return None
    if type(contract) is ValueContract:
        return contract.origins.at(path)
    if type(contract) is not PartialRecordContract:
        output_contract_type(contract)
        return None  # pragma: no cover - the validator above always raises
    if not path or type(path[0]) is not str:
        return None
    field_contract = dict(contract.fields).get(path[0])
    if field_contract is None:
        return None
    return field_contract.origins.at(path[1:])


def validate_output_contract(
    contract: OutputContract | None,
    value: Any,
) -> Any:
    """Validate a plain or typed value against either output-contract arm.

    Args:
        contract (`OutputContract | None`): The contract to enforce; `None` checks nothing.
        value (`Any`): The plain or typed value to check.

    Returns:
        `Any`: The plain value with each typed part normalized (an int in a `float`
            slot widened); an untyped part or an unchecked value is returned as given.

    Raises:
        `TypeCheckError`: The value violates the contract.
    """
    exact_type = output_contract_type(contract)
    if exact_type is not None:
        return build_value_as(exact_type, value).to_object()
    if isinstance(value, TypedValue):
        value = value.to_object()
    if contract is None:
        return value
    assert type(contract) is PartialRecordContract
    if not isinstance(value, dict):
        raise TypeCheckError(
            f"{value_repr(value)} is not an object for a partial record contract"
        )

    expected = set(contract.names)
    missing = [name for name in contract.names if name not in value]
    if missing:
        raise TypeCheckError(
            f"record missing required fields: {missing!r}",
            path=(missing[0],),
        )
    unknown = value.keys() - expected
    if unknown:
        try:
            ordered_unknown = sorted(unknown)
        except TypeError:
            ordered_unknown = sorted(
                unknown,
                key=lambda name: (type(name).__qualname__, _sort_repr(name)),
            )
        path = (
            (ordered_unknown[0],)
            if all(type(name) is str for name in unknown)
            else ()
        )
        shown = shown_text(", ".join(value_repr(name) for name in ordered_unknown))
        raise TypeCheckError(
            f"record has unknown fields: [{shown}]",
            path=path,
        )
    checked = dict(value)
    for name, field_contract in contract.fields:
        if field_contract is None:
            continue
        try:
            checked[name] = build_value_as(field_contract.type, value[name]).to_object()
        except TypeCheckError as exc:
            raise TypeCheckError(
                *exc.args,
                path=(name, *exc.path),
            ) from exc
    return checked


@dataclass(frozen=True)
class HostToolCall:
    """One direct tool call a node makes, as the run boundary checks it.

    Args:
        tool_id (`str`):
            The id of the tool the node invokes.
        inputs (`tuple[str, ...]`):
            The names of the inputs the node passes, in declaration order.
        input_types (`Mapping[str, Type]`):
            The proven type of each input's binding; an input whose type is not
            proven is absent.
        output_type (`Type | None`):
            The node's declared whole-value output type, or `None` when it has none.
    """

    tool_id: str
    inputs: tuple[str, ...]
    input_types: Mapping[str, Type]
    output_type: Type | None


@dataclass(frozen=True, slots=True)
class NodeSteps:
    """The durable steps one start of a node reads and records, from the engine.

    The engine passes it as ``caps['steps']`` to a node declaring ``needs_steps``.
    A step is a JSON object the node gives meaning to; the engine appends each one
    to the run log and keeps it until the node settles, so a resumed node reads the
    steps its earlier starts recorded.

    Args:
        starts (`int`):
            Accepted starts of this node, including the current one; at least 1.
            A resume of an interrupted start counts as a new start.
        prior (`tuple[dict[str, Any], ...]`):
            Copies of the steps earlier starts recorded, in record order.
            Changing them does not change the run.
        record (`Callable[[dict[str, Any]], None]`):
            Appends one step durably and returns once the engine has applied it.
            Call it from the node's own evaluation, one call at a time, before the
            node returns; after the evaluation ends it raises. An exception from it
            does not prove the step was not recorded: a stop can land mid-commit.

    Raises:
        `TypeError`:
            A field has the wrong type.
        `ValueError`:
            ``starts`` is less than 1.

    Example:
        ```python
        def forward(self, inputs, *, steps=None, **caps):
            if steps is not None:
                attempt = steps.starts + len(steps.prior)
                steps.record({"event": "started", "attempt": attempt})
        ```
    """

    starts: int
    prior: tuple[dict[str, Any], ...]
    record: Callable[[dict[str, Any]], None]

    def __post_init__(self) -> None:
        """Check the field types and that the current start is counted."""
        if type(self.starts) is not int:
            raise TypeError("starts must be an int")
        if self.starts < 1:
            raise ValueError("starts must be at least 1")
        if type(self.prior) is not tuple or any(type(step) is not dict for step in self.prior):
            raise TypeError("prior must be a tuple of dicts")
        if not callable(self.record):
            raise TypeError("record must be callable")


@dataclass(frozen=True)
class Parameter:
    """One named declared value with an optional contract and default policy.

    Formal inputs and node-local variables share this carrier; their owning
    construct decides whether a contract or default is required.

    ``has_default`` distinguishes an explicit ``None`` default from no default.
    ``coerce_strings`` is run-boundary policy derived from ``contract.type`` by
    ``coerce_inputs``, which applies it only to the top-level flow's supplied inputs;
    there is deliberately no parallel textual type field.

    Attributes:
        name (`str`): Declared name in the owner's local record.
        contract (`ValueContract | None`): Optional semantic value contract.
        required (`bool`): Whether omission is invalid.
        has_default (`bool`): Whether ``default`` is semantically present.
        default (`Any`): Retained default payload; meaningful only when present.
        coerce_strings (`bool`): Whether the run boundary may convert supplied text.
        origin (`OriginRef | None`): Parameter declaration site.
        default_origin (`OriginRef | None`): Explicit default site, when present.
    """

    name: str
    contract: ValueContract | None = None
    required: bool = False
    has_default: bool = False
    default: Any = None
    coerce_strings: bool = False
    origin: OriginRef | None = None
    default_origin: OriginRef | None = None

    def __post_init__(self) -> None:
        """Enforce one unambiguous representation of required/default state."""
        if self.contract is not None and not isinstance(self.contract, ValueContract):
            raise TypeError("parameter contract must be a ValueContract or None")
        _validate_optional_origin(self.origin, "parameter origin")
        _validate_optional_origin(self.default_origin, "parameter default origin")
        if self.required and self.has_default:
            raise ValueError("a required parameter cannot have a default")
        if not self.has_default and self.default is not None:
            raise ValueError("parameter default requires has_default=True")
        if not self.has_default and self.default_origin is not None:
            raise ValueError("parameter default_origin requires has_default=True")

    def map_origins(self, mapper: _OriginMapper) -> Parameter:
        """Transform parameter, default, and nested contract origins without copying data."""
        contract = (
            self.contract.map_origins(mapper) if self.contract is not None else None
        )
        origin = _map_origin(self.origin, mapper)
        default_origin = _map_origin(self.default_origin, mapper)
        return Parameter(
            name=self.name,
            contract=contract,
            required=self.required,
            has_default=self.has_default,
            default=self.default,
            coerce_strings=self.coerce_strings,
            origin=origin,
            default_origin=default_origin,
        )

    def iter_origins(self) -> Iterator[OriginRef]:
        """Yield contract, declaration, then default origins in stable order."""
        yield from _iter_origin_carrier(self, set())


_OriginCarrier: TypeAlias = (
    Binding
    | ValueTemplate
    | Assertion
    | ContractOrigins
    | ValueContract
    | PartialRecordContract
    | Parameter
)


def _iter_origin_carrier(
    carrier: _OriginCarrier,
    seen: set[int],
) -> Iterator[OriginRef]:
    """Yield one frozen carrier graph once, preserving its semantic origin order."""
    identity = id(carrier)
    if identity in seen:
        return
    seen.add(identity)

    if isinstance(carrier, Binding):
        _validate_optional_origin(carrier.origin, "binding origin")
        if carrier.origin is not None:
            yield carrier.origin
        if carrier.contract is not None:
            yield from _iter_origin_carrier(carrier.contract, seen)
        if isinstance(carrier.source, ValueTemplate):
            yield from _iter_origin_carrier(carrier.source, seen)
        return
    if isinstance(carrier, ValueTemplate):
        for _, origin in carrier.origins:
            if not isinstance(origin, OriginRef):
                raise TypeError("value template origin must be an OriginRef")
            yield origin
        for binding in _tree_bindings(carrier.value):
            yield from _iter_origin_carrier(binding, seen)
        return
    if isinstance(carrier, Assertion):
        _validate_optional_origin(carrier.origin, "assertion origin")
        if carrier.origin is not None:
            yield carrier.origin
        return
    if isinstance(carrier, ContractOrigins):
        _validate_optional_origin(carrier.default, "contract default origin")
        if carrier.default is not None:
            yield carrier.default
        for _, origin in carrier.entries:
            if not isinstance(origin, OriginRef):
                raise TypeError("contract entry origin must be an OriginRef")
            yield origin
        return
    if isinstance(carrier, ValueContract):
        yield from _iter_origin_carrier(carrier.origins, seen)
        return
    if isinstance(carrier, PartialRecordContract):
        for _name, contract in carrier.fields:
            if contract is not None:
                yield from _iter_origin_carrier(contract, seen)
        return
    if isinstance(carrier, Parameter):
        if carrier.contract is not None:
            yield from _iter_origin_carrier(carrier.contract, seen)
        _validate_optional_origin(carrier.origin, "parameter origin")
        _validate_optional_origin(carrier.default_origin, "parameter default origin")
        if carrier.origin is not None:
            yield carrier.origin
        if carrier.default_origin is not None:
            yield carrier.default_origin
        return
    raise TypeError(f"unsupported origin carrier: {type(carrier).__name__}")


def _rewrite_origin_carrier(
    carrier: _OriginCarrier,
    mapper: _OriginMapper,
    seen: set[int],
) -> None:
    """Rewrite one carrier in a private copy without replacing its identity."""
    identity = id(carrier)
    if identity in seen:
        return
    seen.add(identity)

    # Public carriers remain frozen. Bypassing their generated setter only on the
    # unreachable private copy preserves payload back-references to these objects.
    if isinstance(carrier, Binding):
        object.__setattr__(
            carrier,
            "origin",
            _map_origin(carrier.origin, mapper),
        )
        if carrier.contract is not None:
            _rewrite_origin_carrier(carrier.contract, mapper, seen)
        if isinstance(carrier.source, ValueTemplate):
            _rewrite_origin_carrier(carrier.source, mapper, seen)
    elif isinstance(carrier, ValueTemplate):
        object.__setattr__(
            carrier,
            "origins",
            tuple(
                (path, cast(OriginRef, _map_origin(origin, mapper)))
                for path, origin in carrier.origins
            ),
        )
        for binding in _tree_bindings(carrier.value):
            _rewrite_origin_carrier(binding, mapper, seen)
    elif isinstance(carrier, Assertion):
        object.__setattr__(
            carrier,
            "origin",
            _map_origin(carrier.origin, mapper),
        )
    elif isinstance(carrier, ContractOrigins):
        object.__setattr__(
            carrier,
            "default",
            _map_origin(carrier.default, mapper),
        )
        object.__setattr__(
            carrier,
            "entries",
            tuple(
                (path, cast(OriginRef, _map_origin(origin, mapper)))
                for path, origin in carrier.entries
            ),
        )
    elif isinstance(carrier, ValueContract):
        _rewrite_origin_carrier(carrier.origins, mapper, seen)
    elif isinstance(carrier, PartialRecordContract):
        for _name, contract in carrier.fields:
            if contract is not None:
                _rewrite_origin_carrier(contract, mapper, seen)
    elif isinstance(carrier, Parameter):
        if carrier.contract is not None:
            _rewrite_origin_carrier(carrier.contract, mapper, seen)
        object.__setattr__(
            carrier,
            "origin",
            _map_origin(carrier.origin, mapper),
        )
        object.__setattr__(
            carrier,
            "default_origin",
            _map_origin(carrier.default_origin, mapper),
        )
    else:
        raise TypeError(f"unsupported origin carrier: {type(carrier).__name__}")


def as_binding(value: Any) -> Binding:
    """Return ``value`` unchanged when bound, otherwise wrap it origin-free."""
    return value if isinstance(value, Binding) else Binding(source=value)


def binding_source(value: Any) -> Any:
    """Unwrap exactly one ``Binding`` layer, or return a raw value unchanged."""
    return value.source if isinstance(value, Binding) else value


__all__ = [
    "Assertion",
    "Binding",
    "ContractOrigins",
    "ChildFlowBoundary",
    "ExecutionPosition",
    "ExecutionScope",
    "NodeSteps",
    "OutputContract",
    "Parameter",
    "PartialRecordContract",
    "SeedConnection",
    "ValueContract",
    "ValuePath",
    "ValueTemplate",
    "as_binding",
    "binding_input_groups",
    "binding_source",
    "file_field_error",
    "file_fields_type",
    "iter_source_strings",
    "map_source_strings",
    "output_contract_field_error",
    "output_contract_file_type",
    "output_contract_origin",
    "output_contract_type",
    "validate_output_contract",
]

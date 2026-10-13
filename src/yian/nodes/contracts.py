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
from typing import Any, Callable, Iterator, Literal, TypeAlias, cast

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
        """Return this binding with its origin and contract origins transformed by ``mapper``."""
        origin = _map_origin(self.origin, mapper)
        contract = (
            self.contract.map_origins(mapper) if self.contract is not None else None
        )
        return Binding(source=self.source, origin=origin, contract=contract)

    def iter_origins(self) -> Iterator[OriginRef]:
        """Yield this binding's present origin, then its contract's origins."""
        yield from _iter_origin_carrier(self, set())


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


# Kinds a field read is not checked on: `object` is opaque unless it declares fields, and
# `file` is the reserved file reference.
_UNCHECKED_FIELD_KINDS = frozenset({ValueKind.OBJECT, ValueKind.FILE})


def _member_only_field_error(
    union: Type,
    having: tuple[Type, ...],
    field_name: str,
    path: str,
    union_path: str,
) -> str:
    """Return the error for reading a field only some members of `union` have.

    `path` is the whole reference path and `union_path` the path of the union value.
    """
    tag_fields = union_tag_fields(union)
    if tag_fields:
        tag = tag_fields[0]
        tags = ", ".join(next(iter(member.fields[tag].tags)) for member in having)
        return (
            f"reference ${{{path}}} reads field {field_name!r}, which only union members "
            f"with {tag} {tags} have; read it inside a case on ${{{union_path}.{tag}}}"
        )
    return (
        f"reference ${{{path}}} reads field {field_name!r}, which only {len(having)} of "
        f"{len(union.variants)} union members have; check the member with "
        f"isinstance({union_path}, <Type>) in a case first"
    )


def output_contract_field_error(
    contract: OutputContract,
    fields: list[str] | tuple[str, ...],
    path: str,
    *,
    partial_union_fields: bool = False,
) -> str | None:
    """Return why a reference reads a field its contract can't have, else `None`.

    An exact record is closed at every proven nested record, and a partial record is
    closed at its root; reading below an opaque partial field is not checked. Every type
    other than `object` and `file` (scalars, `Literal[...]`, lists, and their
    `Optional[...]` forms) has no fields, so reading one is an error. On a union, a
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
        )
    if type(contract) is not ValueContract:
        output_contract_type(contract)
        return None  # pragma: no cover - the validator above always raises

    typ = output_contract_type(contract)
    assert typ is not None
    # The path segments walked so far, starting with the ones before `fields`.
    walked = path.split(".")[: path.count(".") + 1 - len(fields)]
    for field_name in fields:
        if typ.variants is not None:
            read = union_field(typ, field_name)
            if isinstance(read, FieldAbsent):
                return f"reference ${{{path}}} reads unknown field {field_name!r} on record"
            if isinstance(read, FieldPartial):
                if partial_union_fields:
                    return None
                return _member_only_field_error(
                    typ, read.members, field_name, path, ".".join(walked)
                )
            if read.type is None:
                return None
            typ = read.type
            walked.append(field_name)
            continue
        if typ.fields is None:
            if typ.kind in _UNCHECKED_FIELD_KINDS:
                return None
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
) -> None:
    """Validate a plain or typed value against either output-contract arm."""
    exact_type = output_contract_type(contract)
    if exact_type is not None:
        build_value_as(exact_type, value)
        return
    if contract is None:
        return
    assert type(contract) is PartialRecordContract
    if isinstance(value, TypedValue):
        value = value.to_object()
    if not isinstance(value, dict):
        raise TypeCheckError(
            f"{value!r} is not an object for a partial record contract"
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
                key=lambda name: (type(name).__qualname__, repr(name)),
            )
        path = (
            (ordered_unknown[0],)
            if all(type(name) is str for name in unknown)
            else ()
        )
        raise TypeCheckError(
            f"record has unknown fields: {ordered_unknown!r}",
            path=path,
        )
    for name, field_contract in contract.fields:
        if field_contract is None:
            continue
        try:
            build_value_as(field_contract.type, value[name])
        except TypeCheckError as exc:
            raise TypeCheckError(
                *exc.args,
                path=(name, *exc.path),
            ) from exc


@dataclass(frozen=True)
class Parameter:
    """One named declared value with an optional contract and default policy.

    Formal inputs and node-local variables share this carrier; their owning
    construct decides whether a contract or default is required.

    ``has_default`` distinguishes an explicit ``None`` default from no default.
    ``coerce_strings`` is boundary policy derived from ``contract.type`` by a
    later consumer; there is deliberately no parallel textual type field.

    Attributes:
        name (`str`): Declared name in the owner's local record.
        contract (`ValueContract | None`): Optional semantic value contract.
        required (`bool`): Whether omission is invalid.
        has_default (`bool`): Whether ``default`` is semantically present.
        default (`Any`): Retained default payload; meaningful only when present.
        coerce_strings (`bool`): Whether a boundary may coerce textual input.
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
    "OutputContract",
    "Parameter",
    "PartialRecordContract",
    "SeedConnection",
    "ValueContract",
    "ValuePath",
    "as_binding",
    "binding_source",
    "output_contract_field_error",
    "output_contract_origin",
    "output_contract_type",
    "validate_output_contract",
]

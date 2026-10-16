"""Narrowing facts: which part of a union a value can be, and what checks prove about it.

A fact says which part of its type a value at a base path can be: `Members` of the base's
union, or an `Exact` type it satisfies. Facts join by union (either holds) and meet by
intersection (both hold); losing a fact is always sound. The facts a check proves come
from `check_fact`: a true or false `isinstance`, or a true or false comparison of a union's
tag field with a text.

Knows about: `typesys` types and unions, `compile.validation`, and `isinstance` type
resolution.
Never imports: `compose.validate`, runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, TypeAlias

from yian.compile.validation import types_compatible
from yian.compose.helpers.condition_guards import Check, IsinstanceCheck
from yian.compose.helpers.isinstance_types import (
    isinstance_members,
    resolve_isinstance_type,
)
from yian.typesys.types import TypeRegistry
from yian.typesys.values import (
    Type,
    ValueKind,
    canonical_union_order,
    union_tag_fields,
)

__all__ = [
    "Env",
    "Exact",
    "Fact",
    "Members",
    "check_fact",
    "fact_type",
    "intersect_envs",
    "intersect_facts",
    "isinstance_fact",
    "members_type",
    "not_isinstance_fact",
    "tag_fact",
    "union_envs",
    "union_facts",
]


@dataclass(frozen=True)
class Members:
    """The value is one of `members`, all members of the base's union, or `None` if nullable.

    Members are deduplicated by `==` and kept in canonical union order. An empty set
    that is not nullable means no value satisfies the facts, so the node never runs.

    Args:
        members (`tuple[Type, ...]`):
            Non-nullable record members of the base's union.
        nullable (`bool`, *optional*, defaults to `False`):
            Whether the value may also be `None`.
    """

    members: tuple[Type, ...]
    nullable: bool = False

    def __post_init__(self) -> None:
        """Deduplicate the members and put them in canonical order."""
        unique: list[Type] = []
        for member in self.members:
            if not any(member == seen for seen in unique):
                unique.append(member)
        object.__setattr__(self, "members", canonical_union_order(unique))


@dataclass(frozen=True)
class Exact:
    """The value satisfies `type`.

    Args:
        type (`Type`):
            The type the value is known to satisfy.
    """

    type: Type


Fact: TypeAlias = Members | Exact
"""One narrowing fact for a base path."""

Env: TypeAlias = dict[str, Fact]
"""Facts keyed by dotted base path; an absent base is unconstrained."""


def _contains(members: Sequence[Type], typ: Type) -> bool:
    """Whether `typ` equals one of `members`."""
    return any(member == typ for member in members)


def union_facts(a: Optional[Fact], b: Optional[Fact]) -> Optional[Fact]:
    """Return what holds when either fact holds, or `None` for no fact.

    Args:
        a (`Fact | None`):
            One fact; `None` is unconstrained.
        b (`Fact | None`):
            The other fact; `None` is unconstrained.

    Returns:
        `Fact | None`:
            The set union of two `Members`, an `Exact` both sides share, or `None`.

    Example:
        `union_facts(Members((Buy,)), Members((Sell,)))` is `Members((Buy, Sell))`;
        `union_facts(Members((Buy,)), None)` is `None`.
    """
    if isinstance(a, Members) and isinstance(b, Members):
        return Members(a.members + b.members, a.nullable or b.nullable)
    if isinstance(a, Exact) and isinstance(b, Exact) and a.type == b.type:
        return a
    return None


def intersect_facts(a: Optional[Fact], b: Optional[Fact]) -> Optional[Fact]:
    """Return what holds when both facts hold, or `None` for no fact.

    Losing information is always sound, so a pair that cannot be combined gives `None`.

    Args:
        a (`Fact | None`):
            One fact; `None` is unconstrained.
        b (`Fact | None`):
            The other fact; `None` is unconstrained.

    Returns:
        `Fact | None`:
            The set intersection of two `Members`; `Members` of the one member an
            `Exact` names, else that `Exact`; an `Exact` both sides share; the other
            fact when one side is `None`; otherwise `None`.

    Example:
        `intersect_facts(Members((Buy, Sell)), Members((Sell, Hold)))` is
        `Members((Sell,))`; `intersect_facts(Exact(Buy), Members((Buy, Sell)))` is
        `Members((Buy,))`.
    """
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, Members) and isinstance(b, Members):
        kept = tuple(member for member in a.members if _contains(b.members, member))
        return Members(kept, a.nullable and b.nullable)
    if isinstance(a, Exact) and isinstance(b, Exact):
        return a if a.type == b.type else None
    exact, members = (a, b) if isinstance(a, Exact) else (b, a)
    if _contains(members.members, exact.type):
        return Members((exact.type,))
    return exact


def isinstance_fact(base_type: Type, typ: Type) -> Optional[Fact]:
    """Return the fact that a value of `base_type` satisfies `typ`, or `None` for no fact.

    Args:
        base_type (`Type`):
            The type of the value being tested.
        typ (`Type`):
            The type it satisfies. A nullable `typ` also admits `None`.

    Returns:
        `Fact | None`:
            `None` when every `base_type` value already satisfies `typ`, since `typ`
            would only widen it (`object`, say, would turn off field checks);
            `Members` of the base union's members that `typ` (ignoring nullability)
            shares, when no other member could pass it; otherwise `Exact(typ)`.

    Example:
        For `Union[Buy, Hold, Sell]`, `Buy` gives `Members((Buy,))`,
        `Union[Hold, Sell, Cancel]` gives `Members((Hold, Sell))`, an unrelated record
        gives `Exact` of that record, and `object` gives `None`.
    """
    if types_compatible(base_type, typ):
        return None
    kept = isinstance_members(base_type, typ)
    if kept is not None:
        return Members(kept, typ.nullable)
    return Exact(typ)


def not_isinstance_fact(base_type: Type, typ: Type) -> Optional[Fact]:
    """Return the fact that a value of `base_type` fails `isinstance(x, typ)`, or `None`.

    A base member equal to a member of `typ` (ignoring `typ`'s nullability) always
    passes the check, so it is excluded. A member that only overlaps `typ`, or whose
    values `typ` would accept with extra fields, may still fail, so it is kept.

    Args:
        base_type (`Type`):
            The type of the value being tested.
        typ (`Type`):
            The type the check names. A union names each of its members.

    Returns:
        `Fact | None`:
            `Members` of the base union's remaining members, nullable when the base is
            and `typ` is not; `None` when the base is no union or nothing is excluded.

    Example:
        For `Union[Buy, Hold, Sell]`, `Buy` gives `Members((Hold, Sell))` and an
        unrelated record gives `None`.
    """
    members = base_type.variants
    if members is None:
        return None
    strict = replace(typ, nullable=False)
    wanted = strict.variants or (strict,)
    kept = tuple(member for member in members if not _contains(wanted, member))
    nullable = base_type.nullable and not typ.nullable
    if len(kept) == len(members) and nullable == base_type.nullable:
        return None
    return Members(kept, nullable)


def tag_fact(base_type: Type, field: str, value: str, holds: bool) -> Optional[Fact]:
    """Return the fact that `<base>.<field> == value` is `holds`, or `None` for no fact.

    The members are chosen as an `on:` CASE's branches choose them.

    Args:
        base_type (`Type`):
            The type of the value whose field is compared.
        field (`str`):
            The compared field.
        value (`str`):
            The text it is compared with.
        holds (`bool`):
            Whether the field equals `value`.

    Returns:
        `Fact | None`:
            For a union whose `field` is a tag field: when `holds`, `Members` of the
            members whose tags contain `value`, not nullable (a null never equals a
            text); otherwise `Members` of the members whose tags are not exactly
            `{value}`, as nullable as the base. `None` for any other base or field.

    Example:
        For `Union[Buy, Hold, Sell]` tagged by `action`, `("action", "buy", True)`
        gives `Members((Buy,))` and `("action", "buy", False)` gives
        `Members((Hold, Sell))`.
    """
    members = base_type.variants
    if members is None or field not in union_tag_fields(base_type):
        return None
    if holds:
        return Members(tuple(m for m in members if value in m.fields[field].tags))
    kept = tuple(m for m in members if m.fields[field].tags != {value})
    return Members(kept, base_type.nullable)


def check_fact(
    check: Check,
    base_type: Optional[Type],
    holds: bool,
    registry: TypeRegistry,
) -> Optional[Fact]:
    """Return the fact a check proves about its base when it evaluates to `holds`.

    Args:
        check (`Check`):
            The check.
        base_type (`Type | None`):
            The base's declared type, or `None` when it is unproven.
        holds (`bool`):
            What the check evaluated to.
        registry (`TypeRegistry`):
            The flow's typedefs, for an `isinstance` type name.

    Returns:
        `Fact | None`:
            `isinstance_fact` or `not_isinstance_fact` for an `isinstance`, `tag_fact`
            for a tag comparison. On an unproven base a true `isinstance` gives
            `Exact` of its type and anything else gives `None`.

    Raises:
        `TypeCheckError`:
            Raised when an `isinstance` type name does not resolve.

    Example:
        `check_fact(IsinstanceCheck("v", "Buy"), None, True, registry)` is
        `Exact(Buy)`.
    """
    if isinstance(check, IsinstanceCheck):
        typ = resolve_isinstance_type(check.type_name, registry)
        if base_type is None:
            return Exact(typ) if holds else None
        if holds:
            return isinstance_fact(base_type, typ)
        return not_isinstance_fact(base_type, typ)
    if base_type is None:
        return None
    return tag_fact(base_type, check.field, check.value, holds)


def union_envs(envs: Sequence[Env]) -> Env:
    """Join environments by per-base union; a base missing from any is dropped.

    Args:
        envs (`Sequence[Env]`):
            The environments. `[]` gives `{}`.

    Returns:
        `Env`:
            What holds when any one of them holds.
    """
    if not envs:
        return {}
    joined: Env = {}
    for base, fact in envs[0].items():
        for env in envs[1:]:
            fact = union_facts(fact, env.get(base))
            if fact is None:
                break
        if fact is not None:
            joined[base] = fact
    return joined


def intersect_envs(envs: Sequence[Env]) -> Env:
    """Meet environments by per-base intersection.

    Args:
        envs (`Sequence[Env]`):
            The environments.

    Returns:
        `Env`:
            What holds when all of them hold.
    """
    met: Env = {}
    for env in envs:
        for base, fact in env.items():
            combined = intersect_facts(met.get(base), fact)
            if combined is None:
                met.pop(base, None)
            else:
                met[base] = combined
    return met


def members_type(members: Sequence[Type], nullable: bool) -> Optional[Type]:
    """Return the type of a value that is one of `members`, or `None` if it may be `None`.

    Args:
        members (`Sequence[Type]`):
            Record members of one union.
        nullable (`bool`):
            Whether the value may also be `None`.

    Returns:
        `Type | None`:
            The lone member, a union of the members, `None`'s type for a nullable empty
            set, or `None` for an empty set that is not nullable: no value fits.
    """
    if not members:
        return Type.scalar(ValueKind.NONE) if nullable else None
    if len(members) == 1:
        return replace(members[0], nullable=nullable)
    return Type(
        kind=ValueKind.OBJECT,
        variants=canonical_union_order(members),
        nullable=nullable,
    )


def fact_type(fact: Fact) -> Optional[Type]:
    """Return the type a value has under `fact`, or `None` when no value can.

    An empty member set on a nullable base can only be `None`; on a non-nullable base
    no value fits, so there is no type.

    Args:
        fact (`Fact`):
            The fact.

    Returns:
        `Type | None`:
            The type, or `None` for an empty member set that is not nullable.
    """
    if isinstance(fact, Exact):
        return fact.type
    return members_type(fact.members, fact.nullable)

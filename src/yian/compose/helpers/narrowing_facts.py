"""Narrowing facts: which part of a union a value can be, and what checks prove about it.

A fact says which part of its type a value at a base path can be: `Members` of the base's
union, or an `Exact` type it satisfies. Facts join by union (either holds) and meet by
intersection (both hold); losing a fact is always sound. The facts a check proves come
from `check_fact`: a true or false `isinstance`, or a true or false comparison of a union's
tag field with a text. `guard_env` evaluates a guard from `condition_guards` into the facts
that hold whenever it does, and `guarded_read_types` gives the types a field read under a
guard is checked against.

Knows about: `typesys` types and unions, `compile.validation`, `isinstance` type
resolution, condition guards, and output-contract projection.
Never imports: `compose.validate`, runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional, Sequence, TypeAlias

from yian.compile.validation import types_compatible
from yian.compose.helpers.condition_guards import (
    AllOf,
    Check,
    Guard,
    Holds,
    IsinstanceCheck,
    conjuncts,
    guard_checks,
)
from yian.compose.helpers.isinstance_types import (
    isinstance_members,
    resolve_isinstance_type,
)
from yian.compose.helpers.output_contracts import project_output_contract
from yian.nodes.contracts import ValueContract
from yian.typesys import TypeCheckError
from yian.typesys.types import TypeRegistry
from yian.typesys.values import (
    Type,
    ValueKind,
    canonical_union_order,
    union_tag_fields,
)

__all__ = [
    "NOT_NARROWABLE",
    "BaseTypeOf",
    "Env",
    "Exact",
    "Fact",
    "Members",
    "NotNarrowable",
    "check_fact",
    "fact_type",
    "guard_env",
    "guard_unsatisfiable",
    "guarded_read_types",
    "intersect_envs",
    "intersect_facts",
    "isinstance_fact",
    "members_type",
    "not_isinstance_fact",
    "tag_fact",
    "union_envs",
    "union_facts",
    "unresolved_type",
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


# --------------------------------------------------------------------------- #
# Guards into facts
# --------------------------------------------------------------------------- #


class NotNarrowable(Enum):
    """The answer of a `BaseTypeOf` for a base the site cannot narrow."""

    NOT_NARROWABLE = "not narrowable"


NOT_NARROWABLE = NotNarrowable.NOT_NARROWABLE
"""A base the site cannot narrow, such as a path that is no pool path at a CASE."""

BaseTypeOf: TypeAlias = Callable[[str], "Type | None | NotNarrowable"]
"""A site's declared type of a dotted base path: a `Type` when proven, `None` when
unproven, or `NOT_NARROWABLE` for a base the site cannot narrow."""


def _base_type(base: str, base_type_of: BaseTypeOf, outer: Env) -> Type | None | NotNarrowable:
    """The type a check on `base` is taken against.

    A base whose declared type is unproven, such as a field only some members of an
    outer union have, takes its type through the facts already proven for the longest
    outer base it is below.
    """
    declared = base_type_of(base)
    if declared is not None:
        return declared
    parts = base.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        fact = outer.get(".".join(parts[:cut]))
        if fact is None:
            continue
        typ = fact_type(fact)
        if typ is None:
            return None
        projected = project_output_contract(ValueContract(typ), parts[cut:])
        return projected.type if type(projected) is ValueContract else None
    return None


def _guard_env(
    guard: Guard,
    base_type_of: BaseTypeOf,
    registry: TypeRegistry,
    outer: Env,
) -> Env:
    """The facts `guard` proves, given that the facts in `outer` already hold."""
    if isinstance(guard, Holds):
        base = guard.check.base
        base_type = _base_type(base, base_type_of, outer)
        if base_type is NOT_NARROWABLE:
            return {}
        try:
            fact = check_fact(guard.check, base_type, guard.value, registry)
        except TypeCheckError:
            # The type-table pass reports the unknown name.
            return {}
        return {} if fact is None else {base: fact}
    if isinstance(guard, AllOf):
        env: Env = {}
        for part in guard.parts:
            part_env = _guard_env(
                part, base_type_of, registry, intersect_envs([outer, env])
            )
            env = intersect_envs([env, part_env])
        return env
    return union_envs(
        [_guard_env(part, base_type_of, registry, outer) for part in guard.parts]
    )


def guard_env(guard: Guard, base_type_of: BaseTypeOf, registry: TypeRegistry) -> Env:
    """Return the facts that hold whenever `guard` holds.

    `Holds` gives its check's fact (`check_fact`), `AllOf` meets its parts and `AnyOf`
    joins them. Facts are taken against each base's declared type, so facts from
    different checks compare by member. The parts of an `AllOf` are taken left to right:
    a base whose declared type is unproven because it sits below a union (a field only
    some members have) takes its type through the facts the parts before it proved for an
    outer base, in nested parts too. A check whose base `base_type_of` refuses, or whose
    type name does not resolve, proves nothing.

    Args:
        guard (`Guard`):
            A guard from `condition_guards`.
        base_type_of (`BaseTypeOf`):
            The site's declared type of a base path: a `Type`, `None` when unproven (a
            true `isinstance` then gives `Exact` of its type), or `NOT_NARROWABLE`.
        registry (`TypeRegistry`):
            The flow's typedefs, for `isinstance` type names.

    Returns:
        `Env`:
            The facts by base path.

    Example:
        With `d.output` a `Union[Buy, Hold, Sell]` and `leg` only on `Buy`, the guard of
        `d.output.action == "buy" and d.output.leg.type == "limit"` gives
        `{"d.output": Members((Buy,)), "d.output.leg": Members((Limit,))}`.
    """
    return _guard_env(guard, base_type_of, registry, {})


def guard_unsatisfiable(env: Env) -> bool:
    """Return whether no value can satisfy the facts in `env`.

    Args:
        env (`Env`):
            Facts by base path.

    Returns:
        `bool`:
            `True` when some base's fact is an empty member set that is not nullable.
    """
    return any(fact == Members(()) for fact in env.values())


def unresolved_type(guard: Guard, registry: TypeRegistry) -> bool:
    """Return whether an `isinstance` in `guard` names a type that does not resolve.

    Args:
        guard (`Guard`):
            A guard from `condition_guards`.
        registry (`TypeRegistry`):
            The flow's typedefs.

    Returns:
        `bool`:
            `True` when some `isinstance` type name is neither a typedef nor a builtin
            name. A site then leaves the read alone, so the type-table pass reports the
            name.
    """
    for check in guard_checks(guard):
        if not isinstance(check, IsinstanceCheck):
            continue
        try:
            resolve_isinstance_type(check.type_name, registry)
        except TypeCheckError:
            return True
    return False


def guarded_read_types(
    read: str,
    guard: Guard,
    base_type_of: BaseTypeOf,
    registry: TypeRegistry,
) -> Optional[list[tuple[Type, tuple[str, ...], str]]]:
    """Return the narrowed types a field read under `guard` is checked against.

    Each of the guard's own facts and the facts of each of its top-level conjuncts gives
    a candidate: the fact type of the longest base that is a strict prefix of `read`,
    with the field names below it. A conjunct's candidate keeps what the whole guard's
    meet can lose (two unrelated exact types). The read is fine when any candidate
    accepts it.

    Args:
        read (`str`):
            The dotted reference path read, such as `"v.qty"`.
        guard (`Guard`):
            The guard that holds whenever the read is evaluated.
        base_type_of (`BaseTypeOf`):
            The site's declared type of a base path, as for `guard_env`.
        registry (`TypeRegistry`):
            The flow's typedefs.

    Returns:
        `list[tuple[Type, tuple[str, ...], str]] | None`:
            `(narrowed type, field names below it, narrowed base path)` per distinct
            candidate, in order; `[]` when no fact applies to the read, so it is checked
            unnarrowed; `None` when the guard can never hold, so the read is never
            evaluated and is not checked (`impossible_checks` explains why).

    Example:
        With `v` a `Union[Buy, Hold, Sell]`, the read `v.qty` under the guard of
        `isinstance(v, Buy)` gives `[(Buy, ("qty",), "v")]`.
    """
    envs: list[Env] = []
    for part in (guard, *conjuncts(guard)):
        env = guard_env(part, base_type_of, registry)
        if env not in envs:
            envs.append(env)
    if any(guard_unsatisfiable(env) for env in envs):
        return None
    parts = read.split(".")
    candidates: list[tuple[Type, tuple[str, ...], str]] = []
    for env in envs:
        longest: Optional[list[str]] = None
        for base in env:
            base_parts = base.split(".")
            if (
                len(base_parts) < len(parts)
                and parts[: len(base_parts)] == base_parts
                and (longest is None or len(base_parts) > len(longest))
            ):
                longest = base_parts
        if longest is None:
            continue
        base = ".".join(longest)
        typ = fact_type(env[base])
        if typ is None:
            continue
        candidate = (typ, tuple(parts[len(longest) :]), base)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates

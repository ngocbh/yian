"""Narrowing facts: which part of a union a value can be, and what checks prove about it.

A fact says which part of its type a value at a base path can be: `Members` of the base's
union, or an `Exact` type it satisfies. Facts join by union (either holds) and meet by
intersection (both hold); losing a fact is always sound. The facts a check proves come
from `check_fact`: a true or false `isinstance`, or a true or false comparison of a union's
tag field with a text. `guard_env` evaluates a guard from `condition_guards` into the facts
that hold whenever it does, `checked_refs` gives the guard each read in a condition is
checked under, and `guarded_read_types` the types a field read under a guard is checked
against. `impossible_checks` finds the checks in a condition that are wrong
whatever the values, and the message helpers word those errors and a narrowed read's error.

Knows about: `typesys` types and unions, `compile.validation`, `isinstance` type
resolution, condition guards, output-contract projection, and type labels.
Never imports: `compose.validate`, runtime.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterable, Optional, Sequence, TypeAlias

from lark import Token, Tree

from yian.compile.validation import types_compatible
from yian.compose.helpers.condition_guards import (
    NEVER,
    NOTHING,
    AllOf,
    Check,
    CheckLiteral,
    Guard,
    Holds,
    IsinstanceCheck,
    TagCheck,
    all_of,
    any_of,
    check_text,
    conjuncts,
    falsified,
    guard_checks,
    guarded_refs,
    literal_text,
    literals_in_context,
    masked,
)
from yian.compose.helpers.isinstance_types import (
    isinstance_members,
    resolve_isinstance_type,
)
from yian.compose.helpers.output_contracts import project_output_contract
from yian.compose.helpers.tags import CLOSE_MATCH_CUTOFF, named_type_label
from yian.nodes.contracts import ValueContract
from yian.typesys import TypeCheckError
from yian.typesys.types import TypeRegistry
from yian.typesys.values import (
    Type,
    ValueKind,
    canonical_union_order,
    types_overlap,
    union_tag_fields,
)

__all__ = [
    "NOT_NARROWABLE",
    "BaseTypeOf",
    "Env",
    "Exact",
    "Fact",
    "ImpossibleCheck",
    "ImpossibleReason",
    "Members",
    "NotNarrowable",
    "check_fact",
    "checked_refs",
    "env_read_type",
    "env_unsatisfiable",
    "fact_type",
    "guard_env",
    "guarded_read_types",
    "impossible_check_message",
    "impossible_checks",
    "intersect_envs",
    "intersect_facts",
    "isinstance_fact",
    "members_type",
    "narrowed_read_error",
    "narrowed_read_message",
    "narrowed_type_label",
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


def env_unsatisfiable(env: Env) -> bool:
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


def _guard_envs(guard: Guard, base_type_of: BaseTypeOf, registry: TypeRegistry) -> list[Env]:
    """The distinct facts of `guard` and of each of its top-level conjuncts."""
    envs: list[Env] = []
    for part in (guard, *conjuncts(guard)):
        env = guard_env(part, base_type_of, registry)
        if env not in envs:
            envs.append(env)
    return envs


def _decided_by_type(
    atom: Holds,
    base_type_of: BaseTypeOf,
    registry: TypeRegistry,
    routing: Env,
) -> bool:
    """Whether the base's known type alone makes `atom` false, other than a non-tag text.

    The known type is the declared type met with `routing`. Such an atom comes from a
    check that type always decides, which is no error by itself. A tag comparison with a
    text that is not a tag is always reported, so its atom is not counted here.
    """
    base = atom.check.base
    base_type = base_type_of(base)
    if not isinstance(base_type, Type):
        return False
    check = atom.check
    if isinstance(check, TagCheck):
        members = base_type.variants
        if members is None or check.field not in union_tag_fields(base_type):
            return False
        if not any(check.value in member.fields[check.field].tags for member in members):
            return False
    try:
        fact = check_fact(check, base_type, atom.value, registry)
    except TypeCheckError:
        return False
    return intersect_facts(routing.get(base), fact) == Members(())


def _without_decided(
    guard: Guard,
    base_type_of: BaseTypeOf,
    registry: TypeRegistry,
    routing: Env,
) -> Guard:
    """`guard` with each atom the known type alone makes false proving nothing instead."""
    if isinstance(guard, Holds):
        decided = _decided_by_type(guard, base_type_of, registry, routing)
        return NOTHING if decided else guard
    parts = [_without_decided(part, base_type_of, registry, routing) for part in guard.parts]
    return all_of(*parts) if isinstance(guard, AllOf) else any_of(*parts)


def _satisfiable(
    guard: Guard,
    base_type_of: BaseTypeOf,
    registry: TypeRegistry,
    routing: Env,
) -> bool:
    """Whether some value satisfies `guard`'s facts and each top-level conjunct's, with `routing`."""
    envs = _guard_envs(guard, base_type_of, registry)
    return not any(env_unsatisfiable(intersect_envs([routing, env])) for env in envs)


def checked_refs(
    tree: Tree | Token,
    base_type_of: BaseTypeOf,
    registry: TypeRegistry,
    *,
    routing: Optional[Env] = None,
) -> list[tuple[str, Optional[Guard]]]:
    """Return each reference occurrence in a condition with the guard its field read is checked under.

    A read that runs only when a check `impossible_checks` reports has the value it can
    never have is not checked (`None`), whatever the report's reason: the report explains
    it, and errors on the code it makes dead would only repeat it. So is a read no value
    reaches because of a reported check. Otherwise a guard some value satisfies is kept. A
    read no value reaches only because a check the known type always decides, and that is
    not reported, went the other way (`isinstance(v, Decision) or v.nosuch > 0` with `v`
    declared `Decision`) is no error by itself: such checks prove nothing here, so the read
    is checked as without them (with `NOTHING` when the guard still can't hold).

    Args:
        tree (`Tree | Token`):
            A condition parsed with `parse_condition`, in any spelling.
        base_type_of (`BaseTypeOf`):
            The site's declared type of a base path, as for `guard_env`.
        registry (`TypeRegistry`):
            The flow's typedefs.
        routing (`Env | None`, *optional*, defaults to `None`):
            The facts that hold wherever the condition is evaluated, such as a CASE
            node's routing facts; with the declared type they make the known type. `None`,
            or facts no value satisfies, is none.

    Returns:
        `list[tuple[str, Guard | None]]`:
            One `(reference path, guard or None)` pair per occurrence, in `guarded_refs`
            order.

    Example:
        With `v` a `Union[Buy, Hold, Sell]`, the `v.qty` read in
        `isinstance(v, Decision) or v.qty > 0` gets `NOTHING`, and in
        `isinstance(v, Buy) and v.action == 'hold' and v.qty > 0` and
        `isinstance(v, Text) and v.qty > 0` gets `None`.
    """
    known = routing if routing and not env_unsatisfiable(routing) else {}
    refs = guarded_refs(tree)
    reported = _reported_literals(tree, base_type_of, registry, known)
    if not reported:
        unreported = needing = refs
    else:
        unreported = guarded_refs(masked(tree, [literal.node for literal, _ in reported]))
        never = [(literal.node, _never_value(literal, found)) for literal, found in reported]
        needing = guarded_refs(falsified(tree, never))
    checked: list[tuple[str, Optional[Guard]]] = []
    for (ref, guard), (_, without_reported), (_, live) in zip(
        refs, unreported, needing, strict=True
    ):
        if live == NEVER:
            checked.append((ref, None))
        elif _satisfiable(guard, base_type_of, registry, known):
            checked.append((ref, guard))
        elif _satisfiable(without_reported, base_type_of, registry, known):
            checked.append((ref, None))
        else:
            trimmed = _without_decided(without_reported, base_type_of, registry, known)
            satisfiable = _satisfiable(trimmed, base_type_of, registry, known)
            checked.append((ref, trimmed if satisfiable else NOTHING))
    return checked


def _never_value(literal: CheckLiteral, impossible: "ImpossibleCheck") -> bool:
    """The value a reported literal's check sub-tree (`v.action != 'x'`, without any
    `not` around it) never has.

    A text that is no tag never equals the tag field, whichever way the literal compares
    it; any other reported literal can never be true.
    """
    check_never = True if impossible.reason is ImpossibleReason.NOT_A_TAG else literal.value
    # `!=` spells the check negatively: the sub-tree is true when the check is false.
    positive = literal.value != literal.negated
    return check_never == positive


def env_read_type(read: str, env: Env) -> Optional[tuple[Type, tuple[str, ...], str]]:
    """Return the narrowed type a field read sees under `env`, or `None` when none applies.

    Args:
        read (`str`):
            The dotted reference path read, such as `"v.qty"`.
        env (`Env`):
            Facts by base path.

    Returns:
        `tuple[Type, tuple[str, ...], str] | None`:
            `(narrowed type, field names below it, narrowed base path)` for the longest
            base that is a strict prefix of `read`; `None` when there is no such base or
            no value can have its fact.

    Example:
        `env_read_type("v.leg.price", {"v": Members((Buy,))})` is
        `(Buy, ("leg", "price"), "v")`.
    """
    parts = read.split(".")
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
        return None
    base = ".".join(longest)
    typ = fact_type(env[base])
    if typ is None:
        return None
    return typ, tuple(parts[len(longest) :]), base


def guarded_read_types(
    read: str,
    guard: Guard,
    base_type_of: BaseTypeOf,
    registry: TypeRegistry,
) -> list[tuple[Type, tuple[str, ...], str]]:
    """Return the narrowed types a field read under `guard` is checked against.

    Each of the guard's own facts and the facts of each of its top-level conjuncts gives
    a candidate (`env_read_type`). A conjunct's candidate keeps what the whole guard's
    meet can lose (two unrelated exact types). The read is fine when any candidate
    accepts it. Take the guard from `checked_refs`, which also says which reads are not
    checked at all.

    Args:
        read (`str`):
            The dotted reference path read, such as `"v.qty"`.
        guard (`Guard`):
            The guard the read is checked under.
        base_type_of (`BaseTypeOf`):
            The site's declared type of a base path, as for `guard_env`.
        registry (`TypeRegistry`):
            The flow's typedefs.

    Returns:
        `list[tuple[Type, tuple[str, ...], str]]`:
            `(narrowed type, field names below it, narrowed base path)` per distinct
            candidate, in order; `[]` when no fact applies to the read, so it is checked
            unnarrowed.

    Example:
        With `v` a `Union[Buy, Hold, Sell]`, the read `v.qty` under the guard of
        `isinstance(v, Buy)` gives `[(Buy, ("qty",), "v")]`.
    """
    candidates: list[tuple[Type, tuple[str, ...], str]] = []
    for env in _guard_envs(guard, base_type_of, registry):
        candidate = env_read_type(read, env)
        if candidate is not None and candidate not in candidates:
            candidates.append(candidate)
    return candidates


# --------------------------------------------------------------------------- #
# Impossible checks and messages
# --------------------------------------------------------------------------- #


class ImpossibleReason(Enum):
    """Why `impossible_checks` reports a check."""

    CONTEXT = "context"
    """The checks evaluated before the literal rule it out."""
    ROUTING = "routing"
    """The case branch that routes the node rules the literal out on its own."""
    NO_OVERLAP = "no overlap"
    """An `isinstance` names a type no value of the base's type can be."""
    ALWAYS = "always"
    """A negated `isinstance` names a type every value of the base's type is."""
    NOT_A_TAG = "not a tag"
    """A tag comparison's text is not a tag of the base union's tag field: a misspelling,
    reported for `==` (never true) and `!=` (always true) alike."""


@dataclass(frozen=True)
class ImpossibleCheck:
    """A check in a condition that is wrong whatever the values.

    Args:
        text (`str`):
            The literal's canonical source, such as `not isinstance(v, Buy)`.
        base (`str`):
            The checked base path.
        reason (`ImpossibleReason`):
            Why it is reported.
        base_label (`str`, *optional*, defaults to `""`):
            The base's type: narrowed for `CONTEXT` and `ROUTING`, else the type the base
            is checked against.
        declared (`bool`, *optional*, defaults to `True`):
            For `NO_OVERLAP` and `ALWAYS`, whether the base's type is its declared type;
            `False` when it comes from the checks before the literal.
        type_label (`str`, *optional*, defaults to `""`):
            The type an `isinstance` names, for `NO_OVERLAP`.
        always_true (`str`, *optional*, defaults to `""`):
            The canonical source of the check that is always true where the literal needs
            it false, such as `isinstance(v, Buy)` for `not isinstance(v, Buy)`; `""` when
            the literal needs it true.
        value (`str`, *optional*, defaults to `""`):
            The compared text, for `NOT_A_TAG`.
        field (`str`, *optional*, defaults to `""`):
            The compared field, for `NOT_A_TAG`.
        tags (`tuple[str, ...]`, *optional*, defaults to `()`):
            The tag field's tags, sorted, for `NOT_A_TAG`.
    """

    text: str
    base: str
    reason: ImpossibleReason
    base_label: str = ""
    declared: bool = True
    type_label: str = ""
    always_true: str = ""
    value: str = ""
    field: str = ""
    tags: tuple[str, ...] = ()


def narrowed_type_label(typ: Type, registry: TypeRegistry) -> str:
    """Return a narrowed type's label for a message, naming a union member by member.

    Args:
        typ (`Type`):
            A narrowed type, such as a `fact_type` result.
        registry (`TypeRegistry`):
            The flow's typedefs.

    Returns:
        `str`:
            A union's members by typedef name joined by ` | ` (`Buy | Sell`), inside
            `Optional(...)` when it is nullable; any other type by `named_type_label`,
            which falls back to the structural label for an anonymous type.
    """
    if typ.variants is None:
        return named_type_label(typ, registry)
    label = " | ".join(named_type_label(member, registry) for member in typ.variants)
    return f"Optional({label})" if typ.nullable else label


def _fact_label(fact: Optional[Fact], fallback: Optional[Type], registry: TypeRegistry) -> str:
    """The label of the type a base has under `fact`, else of `fallback`."""
    typ = fact_type(fact) if fact is not None else None
    if typ is None:
        typ = fallback
    return narrowed_type_label(typ, registry) if typ is not None else "unknown"


def _is_open(typ: Type) -> bool:
    """Whether every value fits `typ`'s kind: `Any`, or a plain `object`."""
    if typ.kind is ValueKind.ANY:
        return True
    return typ.kind is ValueKind.OBJECT and typ.fields is None and typ.variants is None


def _always_true(literal: CheckLiteral) -> str:
    """The check that is always true when `literal` can never be true, if it needs it false."""
    return "" if literal.value else check_text(literal.check, True)


def _declared_impossible(
    literal: CheckLiteral,
    base_type: Type,
    declared: bool,
    registry: TypeRegistry,
) -> Optional[ImpossibleCheck]:
    """The report a literal gets against its base's type alone, else `None`."""
    check = literal.check
    text = literal_text(literal)
    label = named_type_label(base_type, registry)
    if isinstance(check, TagCheck):
        if base_type.variants is None or check.field not in union_tag_fields(base_type):
            return None
        tags = tuple(
            sorted(tag for member in base_type.variants for tag in member.fields[check.field].tags)
        )
        if check.value in tags:
            return None
        return ImpossibleCheck(
            text,
            check.base,
            ImpossibleReason.NOT_A_TAG,
            label,
            value=check.value,
            field=check.field,
            tags=tags,
        )
    if _is_open(base_type):
        return None
    try:
        typ = resolve_isinstance_type(check.type_name, registry)
    except TypeCheckError:
        return None
    if literal.value:
        # A missing value is checked as null, which a nullable type admits.
        if typ.nullable or types_overlap(base_type, typ):
            return None
        return ImpossibleCheck(
            text,
            check.base,
            ImpossibleReason.NO_OVERLAP,
            label,
            declared,
            type_label=named_type_label(typ, registry),
        )
    # A null base fails a non-nullable check, so its negation can hold.
    if base_type.nullable and not typ.nullable:
        return None
    strict, strict_base = replace(typ, nullable=False), replace(base_type, nullable=False)
    wanted = strict.variants or (strict,)
    if not all(_contains(wanted, member) for member in strict_base.variants or (strict_base,)):
        return None
    always_true = _always_true(literal)
    return ImpossibleCheck(
        text, check.base, ImpossibleReason.ALWAYS, label, declared, always_true=always_true
    )


def impossible_checks(
    tree: Tree | Token,
    base_type_of: BaseTypeOf,
    registry: TypeRegistry,
    *,
    routing: Optional[Env] = None,
) -> list[ImpossibleCheck]:
    """Return the checks in a condition that are wrong whatever the values.

    Each literal in a boolean position (a check, or `not` applied to one) is tested
    against its base's type and against what holds when it is evaluated:

    - Against the base's type alone, whatever the context: an `isinstance` naming a type
      no value of it can be (not a nullable type: a missing value is checked as null), a
      negated `isinstance` naming a type every value of it is, and a tag comparison whose
      text is not a tag of the union's tag field, reported as a misspelling for `==` and
      `!=` alike.
    - Against the context, when the context can hold: the checks before the literal, met
      with what it proves, leave its base no member and no null. With `routing`, a
      literal the routing rules out together with the context is reported too, worded as
      routing only when the routing rules it out on its own.

    A check the types always decide (always true in an `or`, always false in an `and`) is
    no error by itself, and neither is the code after it that no value reaches: sites check
    such a read as without that check, and do not check a read after a reported check
    (`checked_refs`), so the report does not cascade into errors on the code it makes dead.
    Unproven, `object` and `Any` bases, unknown type names, bases `base_type_of` refuses,
    and facts lost to an intersection prove nothing. Each literal text is reported once.

    Args:
        tree (`Tree | Token`):
            A condition parsed with `parse_condition`.
        base_type_of (`BaseTypeOf`):
            The site's declared type of a base path, as for `guard_env`.
        registry (`TypeRegistry`):
            The flow's typedefs.
        routing (`Env | None`, *optional*, defaults to `None`):
            The facts that hold wherever the condition is evaluated, such as a CASE
            node's routing facts. `None` is none. When no value satisfies them, the
            condition never runs and nothing is reported.

    Returns:
        `list[ImpossibleCheck]`:
            The reported literals in source order.

    Example:
        With `v` a `Union[Buy, Hold, Sell]` tagged by `action`,
        `isinstance(v, Buy) and v.action == 'hold'` gives one `CONTEXT` check on
        `v.action == "hold"`, whose base is narrowed to `Buy`.
    """
    found: list[ImpossibleCheck] = []
    for _, impossible in _reported_literals(tree, base_type_of, registry, routing or {}):
        if all(impossible.text != seen.text for seen in found):
            found.append(impossible)
    return found


def _reported_literals(
    tree: Tree | Token,
    base_type_of: BaseTypeOf,
    registry: TypeRegistry,
    routing: Env,
) -> list[tuple[CheckLiteral, ImpossibleCheck]]:
    """Each literal `impossible_checks` reports, with its report, before deduplication."""
    if env_unsatisfiable(routing):
        return []
    found: list[tuple[CheckLiteral, ImpossibleCheck]] = []
    for literal in literals_in_context(tree):
        context_env = guard_env(literal.context, base_type_of, registry)
        routed_env = intersect_envs([routing, context_env])
        base = literal.check.base
        base_type = _base_type(base, base_type_of, context_env)
        if base_type is NOT_NARROWABLE:
            continue
        impossible = None
        if base_type is not None:
            declared = base_type_of(base) is not None
            impossible = _declared_impossible(literal, base_type, declared, registry)
        if impossible is None and not (
            env_unsatisfiable(context_env) or env_unsatisfiable(routed_env)
        ):
            impossible = _context_impossible(
                literal, base_type, context_env, routed_env, base_type_of, registry, routing
            )
        if impossible is not None:
            found.append((literal, impossible))
    return found


def _context_impossible(
    literal: CheckLiteral,
    base_type: Optional[Type],
    context_env: Env,
    routed_env: Env,
    base_type_of: BaseTypeOf,
    registry: TypeRegistry,
    routing: Env,
) -> Optional[ImpossibleCheck]:
    """The report a literal gets from what holds when it is evaluated, else `None`."""
    base = literal.check.base
    own = Holds(literal.check, literal.value)
    text = literal_text(literal)
    met = intersect_envs([context_env, _guard_env(own, base_type_of, registry, context_env)])
    if met.get(base) == Members(()):
        label = _fact_label(context_env.get(base), base_type, registry)
        return ImpossibleCheck(
            text, base, ImpossibleReason.CONTEXT, label, always_true=_always_true(literal)
        )
    if intersect_envs([routing, met]).get(base) != Members(()):
        return None
    routed_own = intersect_envs([routing, _guard_env(own, base_type_of, registry, routing)])
    if routed_own.get(base) == Members(()):
        label = _fact_label(routing.get(base), base_type, registry)
        reason = ImpossibleReason.ROUTING
    else:
        label = _fact_label(routed_env.get(base), base_type, registry)
        reason = ImpossibleReason.CONTEXT
    return ImpossibleCheck(text, base, reason, label, always_true=_always_true(literal))


def _close_match(word: str, options: Iterable[str]) -> Optional[str]:
    """The option closest to `word`, if one is close enough."""
    matches = difflib.get_close_matches(word, list(options), n=1, cutoff=CLOSE_MATCH_CUTOFF)
    return matches[0] if matches else None


def impossible_check_message(check: ImpossibleCheck) -> str:
    """Return the load error text for an item of `impossible_checks`.

    Args:
        check (`ImpossibleCheck`):
            An item of `impossible_checks`.

    Returns:
        `str`:
            The message without a site prefix, such as: `` `isinstance(d.output, Text)`
            can never be true: d.output is declared Decision, and no Decision value fits
            Text ``.
    """
    text = f"`{check.text}`"
    always = f", so `{check.always_true}` is always true" if check.always_true else ""
    if check.reason in (ImpossibleReason.CONTEXT, ImpossibleReason.ROUTING):
        why = (
            "the checks before it"
            if check.reason is ImpossibleReason.CONTEXT
            else "the case branch that routes this node"
        )
        return (
            f"{text} can never be true here: {check.base} is {check.base_label}, "
            f"by {why}{always}"
        )
    if check.reason in (ImpossibleReason.NO_OVERLAP, ImpossibleReason.ALWAYS):
        if check.declared:
            base = f"{check.base} is declared {check.base_label}"
        else:
            base = f"{check.base} is {check.base_label} here, by the checks before it"
        if check.reason is ImpossibleReason.ALWAYS:
            return f"{text} can never be true: {base}{always}"
        return (
            f"{text} can never be true: {base}, and no {check.base_label} value fits "
            f"{check.type_label}"
        )
    message = (
        f"{text} compares {check.base}.{check.field} with {check.value!r}, which is not "
        f"one of its tags ({', '.join(check.tags)})"
    )
    close = _close_match(check.value, check.tags)
    return message if close is None else f"{message}; did you mean {close!r}?"


def narrowed_read_message(
    ref: str,
    field: str,
    narrowed_label: str,
    base: str,
    fields: Iterable[str],
) -> str:
    """Return the load error text for a read of a field its narrowed type doesn't have.

    Args:
        ref (`str`):
            The dotted reference path read, such as `"r.answr"`.
        field (`str`):
            The field name the narrowed type lacks.
        narrowed_label (`str`):
            The narrowed type's label, from `narrowed_type_label`.
        base (`str`):
            The narrowed base path.
        fields (`Iterable[str]`):
            The narrowed type's field names, for a close-match hint.

    Returns:
        `str`:
            Such as `reference ${r.answr} reads field 'answr', which Done doesn't have;
            r is Done here, by the checks before this read; did you mean 'answer'?`.
    """
    message = (
        f"reference ${{{ref}}} reads field {field!r}, which {narrowed_label} doesn't have; "
        f"{base} is {narrowed_label} here, by the checks before this read"
    )
    close = _close_match(field, fields)
    return message if close is None else f"{message}; did you mean {close!r}?"


def _field_names(typ: Type) -> Optional[set[str]]:
    """The fields a record has, or any member of a union has; `None` for other types."""
    if typ.variants is not None:
        names: set[str] = set()
        for member in typ.variants:
            if member.fields is None:
                return None
            names.update(member.fields)
        return names
    return None if typ.fields is None else set(typ.fields)


def narrowed_read_error(
    ref: str,
    candidate: tuple[Type, tuple[str, ...], str],
    error: str,
    registry: TypeRegistry,
) -> str:
    """Return a narrowed read's error, naming the narrowed type when it lacks the field.

    Args:
        ref (`str`):
            The dotted reference path read, such as `"r.answr"`.
        candidate (`tuple[Type, tuple[str, ...], str]`):
            The candidate the read failed on, from `guarded_read_types`.
        error (`str`):
            The site's own error for the read on the candidate's type.
        registry (`TypeRegistry`):
            The flow's typedefs, for the type label.

    Returns:
        `str`:
            `narrowed_read_message`'s text when no member of the narrowed type has the
            first field below the narrowed base; otherwise `error` unchanged (a field only
            some members have, or a field deeper down).

    Example:
        For `r.answr` with the candidate `(Done, ("answr",), "r")`, the text is
        `reference ${r.answr} reads field 'answr', which Done doesn't have; r is Done
        here, by the checks before this read; did you mean 'answer'?`.
    """
    typ, fields, base = candidate
    names = _field_names(typ)
    if names is None or fields[0] in names:
        return error
    label = narrowed_type_label(typ, registry)
    return narrowed_read_message(ref, fields[0], label, base, sorted(names))

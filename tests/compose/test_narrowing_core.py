"""Union narrowing from CASE branches: the fact domain, propagation, and node views."""

from dataclasses import replace

import pytest

from yian.compile.model import Edge
from yian.compose.helpers import narrowing_facts
from yian.compose.helpers.condition_guards import (
    NOTHING,
    Holds,
    IsinstanceCheck,
    TagCheck,
    proves,
)
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.helpers.isinstance_types import isinstance_members
from yian.compose.helpers.narrowing_facts import (
    NOT_NARROWABLE,
    ImpossibleReason,
    check_fact,
    guard_env,
    env_unsatisfiable,
    checked_refs,
    guarded_read_types,
    impossible_check_message,
    impossible_checks,
    narrowed_read_error,
    narrowed_read_message,
    narrowed_type_label,
    not_isinstance_fact,
    tag_fact,
    unresolved_type,
)
from yian.compose.helpers.types import read_type
from yian.compose.nodes.base import BuildContext, NodeDescriptor
from yian.compose.nodes.case import (
    CaseDescriptor,
    isinstance_conjuncts,
    searched_param_guards,
)
from yian.compose.validate.case_exhaustive import check_case_when_values
from yian.compose.validate.narrowing import (
    Exact,
    Members,
    compute_narrowing,
    intersect_facts,
    isinstance_fact,
    narrow_contract_at,
    union_facts,
)
from yian.diagnostics import OriginRef, SourceRange
from yian.expr import parse_condition
from yian.nodes.contracts import ContractOrigins, PartialRecordContract, ValueContract
from yian.typesys import Type, TypeCheckError, ValueKind
from yian.typesys.types import read_typedefs
from yian.typesys.values import canonical_type_label, types_overlap

_TYPEDEFS = {
    "Limit": {"type": "Literal(limit)", "price": "float"},
    "Market": {"type": "Literal(market)"},
    "Leg": "Union(Limit, Market)",
    "Buy": {"action": "Literal(buy)", "qty": "int", "leg": "Leg"},
    "Hold": {"action": "Literal(hold)"},
    "Sell": {"action": "Literal(sell)", "qty": "int", "price": "float"},
    "Decision": "Union(Buy, Hold, Sell)",
    "BuyOrSell": "Union(Buy, Sell)",
    "Text": {"text": "str"},
    "Cancel": {"action": "Literal(cancel)", "reason": "str"},
    "Rest": "Union(Hold, Sell, Cancel)",
    "Wide": {"action": "Literal(buy, cancel)", "qty": "int", "leg": "Leg"},
    "SellOrWide": "Union(Sell, Wide)",
    "TextOrWide": "Union(Text, Wide)",
}
_REGISTRY = read_typedefs(_TYPEDEFS)


def _t(name: str) -> Type:
    """Resolve one type expression against the test typedefs."""
    return read_type(name, _REGISTRY)


BUY, HOLD, SELL = _t("Buy"), _t("Hold"), _t("Sell")
DECISION = _t("Decision")
BUY_OR_SELL = _t("BuyOrSell")
LIMIT = _t("Limit")
TEXT = _t("Text")

_ROOT = OriginRef("/s:root/s:node/s:d/s:field/s:output")
_QTY = OriginRef("/s:root/s:node/s:d/s:field/s:output/s:qty")


def _node(node_id: str, *, depends_on=(), runs_after=()) -> NodeDescriptor:
    """A plain node descriptor with optional ordering lists."""
    return NodeDescriptor(
        id=node_id, depends_on=list(depends_on), runs_after=list(runs_after)
    )


def _case(
    case_id: str,
    on: str | None,
    cases: list[tuple[str, str]],
    else_: str | None = None,
    *,
    when_ranges=(),
) -> tuple[CaseDescriptor, list[Edge]]:
    """A CASE descriptor and the data and control edges its own build returns."""
    desc = CaseDescriptor(
        id=case_id,
        on=on,
        cases=[{"when": when, "then": then} for when, then in cases],
        else_=else_,
        when_ranges=list(when_ranges),
    )
    return desc, list(desc.build(BuildContext({})).edges)


def _data(src: str, dst: str, group: str = "x", *, optional: bool = False) -> Edge:
    """One data edge from `src`'s output into `dst`'s input group."""
    return Edge(
        id=f"{src}->{dst}:{group}",
        from_=src,
        to=dst,
        input_group=group,
        optional=optional,
    )


def _compute(nodes, edges, producers, inputs=None):
    """Run the narrowing pass over hand-built nodes and edges."""
    descriptors = {}
    all_edges = list(edges)
    for item in nodes:
        if isinstance(item, tuple):
            desc, case_edges = item
            all_edges.extend(case_edges)
        else:
            desc = item
        descriptors[desc.id] = desc
    return compute_narrowing(
        descriptors,
        all_edges,
        producers,
        inputs or {},
        registry=_REGISTRY,
    )


def _decision_contract(typ: Type = DECISION) -> ValueContract:
    """An exact output contract for `d` with a root and a nested origin."""
    return ValueContract(
        type=typ,
        origins=ContractOrigins(default=_ROOT, entries=((("qty",), _QTY),)),
    )


def _view_type(narrowing, node_id: str, producer: str = "d") -> Type:
    """The type node `node_id` sees for `producer`'s output."""
    producers, _inputs = narrowing.for_node(node_id)
    return producers[producer].type


def _branches():
    """`d` (a Decision) routed by `c` on its tag: buy -> b, sell -> s, else -> h."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "b"), ("sell", "s")], "h"),
        _node("b"),
        _node("s"),
        _node("h"),
    ]
    return nodes


# --------------------------------------------------------------------------- #
# The fact domain
# --------------------------------------------------------------------------- #


def test_union_of_members_is_the_set_union_and_or_of_nullability():
    """Two member sets join to their union; either nullable makes it nullable."""
    joined = union_facts(Members((BUY,)), Members((SELL, BUY), nullable=True))
    assert joined == Members((BUY, SELL), nullable=True)


def test_union_of_exact_facts_keeps_only_an_equal_type():
    """Equal exact types survive a union; anything else loses the fact."""
    assert union_facts(Exact(TEXT), Exact(TEXT)) == Exact(TEXT)
    assert union_facts(Exact(TEXT), Exact(BUY)) is None
    assert union_facts(Exact(BUY), Members((BUY,))) is None
    assert union_facts(Members((BUY,)), None) is None


def test_intersection_of_members_is_the_set_intersection():
    """Two member sets meet at their common members; both nullable stays nullable."""
    met = intersect_facts(
        Members((BUY, SELL), nullable=True), Members((SELL, HOLD), nullable=True)
    )
    assert met == Members((SELL,), nullable=True)
    assert intersect_facts(Members((BUY,)), Members((SELL,))) == Members(())
    assert intersect_facts(Members((BUY,)), None) == Members((BUY,))


def test_intersection_of_exact_with_members():
    """An exact member picks that member; an unrelated exact type stays exact."""
    assert intersect_facts(Exact(BUY), Members((BUY, SELL))) == Members((BUY,))
    assert intersect_facts(Members((BUY, SELL)), Exact(TEXT)) == Exact(TEXT)
    assert intersect_facts(Exact(TEXT), Exact(TEXT)) == Exact(TEXT)
    assert intersect_facts(Exact(TEXT), Exact(BUY)) is None


def test_isinstance_fact_against_the_base_union():
    """A member or a sub-union gives members; any other type gives an exact fact."""
    assert isinstance_fact(DECISION, BUY) == Members((BUY,))
    assert isinstance_fact(DECISION, BUY_OR_SELL) == Members((BUY, SELL))
    assert isinstance_fact(DECISION, TEXT) == Exact(TEXT)
    assert isinstance_fact(TEXT, TEXT) is None
    assert isinstance_fact(DECISION, Type.scalar(ValueKind.OBJECT)) is None
    assert isinstance_fact(DECISION, DECISION) is None
    assert isinstance_fact(replace(DECISION, nullable=True), BUY) == Members((BUY,))
    assert isinstance_fact(Type.scalar(ValueKind.OBJECT), TEXT) == Exact(TEXT)
    assert isinstance_fact(DECISION, replace(BUY, nullable=True)) == Members(
        (BUY,), nullable=True
    )


def test_isinstance_fact_against_a_partly_shared_union():
    """A union sharing some base members gives those members when no other member can pass."""
    assert isinstance_fact(DECISION, _t("Rest")) == Members((HOLD, SELL))
    assert isinstance_fact(DECISION, _t("Optional(Rest)")) == Members(
        (HOLD, SELL), nullable=True
    )
    assert isinstance_fact(replace(DECISION, nullable=True), _t("Rest")) == Members(
        (HOLD, SELL)
    )
    # `Buy` values can pass `Wide`, so keeping only `Sell` would be unsound.
    assert types_overlap(BUY, _t("Wide"))
    assert isinstance_fact(DECISION, _t("SellOrWide")) == Exact(_t("SellOrWide"))
    assert isinstance_fact(DECISION, _t("TextOrWide")) == Exact(_t("TextOrWide"))


def test_the_fact_algebra_lives_in_helpers_and_narrowing_re_exports_it():
    from yian.compose.validate import narrowing

    for name in ("Env", "Exact", "Fact", "Members", "intersect_facts", "isinstance_fact",
                 "union_facts"):
        assert getattr(narrowing, name) is getattr(narrowing_facts, name)
        assert name in narrowing.__all__


def test_not_isinstance_fact_keeps_the_members_the_check_cannot_have_passed():
    """A false check removes the base members equal to a member of the checked type."""
    assert not_isinstance_fact(DECISION, BUY) == Members((HOLD, SELL))
    assert not_isinstance_fact(DECISION, _t("Rest")) == Members((BUY,))
    # `Buy` only overlaps `Wide`, so a `Buy` value may fail the check: it is kept.
    assert not_isinstance_fact(DECISION, _t("SellOrWide")) == Members((BUY, HOLD))
    assert not_isinstance_fact(DECISION, DECISION) == Members(())


def test_not_isinstance_fact_nullability():
    """A null fails a non-nullable check and passes a nullable one."""
    nullable = replace(DECISION, nullable=True)
    assert not_isinstance_fact(nullable, BUY) == Members((HOLD, SELL), nullable=True)
    assert not_isinstance_fact(nullable, _t("Optional(Buy)")) == Members((HOLD, SELL))
    assert not_isinstance_fact(nullable, _t("Optional(Text)")) == Members(
        (BUY, HOLD, SELL)
    )


def test_not_isinstance_fact_gives_nothing_without_a_union_or_an_exclusion():
    assert not_isinstance_fact(BUY, BUY) is None
    assert not_isinstance_fact(Type.scalar(ValueKind.OBJECT), BUY) is None
    assert not_isinstance_fact(DECISION, TEXT) is None
    assert not_isinstance_fact(replace(DECISION, nullable=True), TEXT) is None


def test_tag_fact_selects_members_by_tag():
    assert tag_fact(DECISION, "action", "buy", True) == Members((BUY,))
    assert tag_fact(DECISION, "action", "buy", False) == Members((HOLD, SELL))
    assert tag_fact(DECISION, "action", "zzz", True) == Members(())
    assert tag_fact(DECISION, "action", "zzz", False) == Members((BUY, HOLD, SELL))


def test_tag_fact_nullability():
    """A null base never equals a tag, so only a true comparison drops the null."""
    nullable = replace(DECISION, nullable=True)
    assert tag_fact(nullable, "action", "buy", True) == Members((BUY,))
    assert tag_fact(nullable, "action", "buy", False) == Members(
        (HOLD, SELL), nullable=True
    )


def test_tag_fact_gives_nothing_off_a_tag_field_or_a_union():
    assert tag_fact(DECISION, "qty", "1", True) is None
    assert tag_fact(BUY, "action", "buy", True) is None
    assert tag_fact(TEXT, "text", "x", False) is None


def test_check_fact_dispatches_on_the_check():
    buy = IsinstanceCheck("v", "Buy")
    assert check_fact(buy, DECISION, True, _REGISTRY) == Members((BUY,))
    assert check_fact(buy, DECISION, False, _REGISTRY) == Members((HOLD, SELL))
    assert check_fact(buy, BUY, True, _REGISTRY) is None
    tag = TagCheck("v", "action", "sell")
    assert check_fact(tag, DECISION, True, _REGISTRY) == Members((SELL,))
    assert check_fact(tag, DECISION, False, _REGISTRY) == Members((BUY, HOLD))


def test_check_fact_on_an_unproven_base():
    """A true `isinstance` on an unproven base proves its type; nothing else proves anything."""
    assert check_fact(IsinstanceCheck("v", "Buy"), None, True, _REGISTRY) == Exact(BUY)
    assert check_fact(IsinstanceCheck("v", "Buy"), None, False, _REGISTRY) is None
    assert check_fact(TagCheck("v", "action", "buy"), None, True, _REGISTRY) is None


def test_check_fact_rejects_an_unknown_type_name():
    with pytest.raises(TypeCheckError):
        check_fact(IsinstanceCheck("v", "Nope"), DECISION, True, _REGISTRY)


# --------------------------------------------------------------------------- #
# Guards into facts
# --------------------------------------------------------------------------- #

MARKET = _t("Market")
OBJECT = Type.scalar(ValueKind.OBJECT)


def _g(condition: str):
    """What `condition` proves when it is true."""
    return proves(parse_condition(condition), True)


def _types(**declared):
    """A `base_type_of` over `declared` (dots spelled `__`); other bases are unproven."""
    by_base = {base.replace("__", "."): typ for base, typ in declared.items()}
    return lambda base: by_base.get(base)


def _env(condition: str, **declared):
    """The facts `condition` proves when true, with the bases typed by `declared`."""
    return guard_env(_g(condition), _types(**declared), _REGISTRY)


def test_guard_env_of_one_check():
    assert _env("isinstance(v, Buy)", v=DECISION) == {"v": Members((BUY,))}
    assert _env("not isinstance(v, Buy)", v=DECISION) == {"v": Members((HOLD, SELL))}
    assert _env("v.action == 'sell'", v=DECISION) == {"v": Members((SELL,))}
    assert _env("isinstance(v, Buy)", v=BUY) == {}
    assert _env("isinstance(v, Buy)") == {"v": Exact(BUY)}
    assert _env("v.action == 'sell'") == {}
    assert guard_env(NOTHING, _types(), _REGISTRY) == {}


def test_guard_env_meets_and_joins_on_one_base():
    assert _env("isinstance(v, BuyOrSell) and v.action != 'buy'", v=DECISION) == {
        "v": Members((SELL,))
    }
    assert _env("isinstance(v, Buy) or isinstance(v, Sell)", v=DECISION) == {
        "v": Members((BUY, SELL))
    }
    assert _env("not (isinstance(v, Hold) or v.action == 'sell')", v=DECISION) == {
        "v": Members((BUY,))
    }


def test_guard_env_over_two_bases():
    assert _env("isinstance(v, Buy) and isinstance(w, Sell)", v=DECISION, w=DECISION) == {
        "v": Members((BUY,)),
        "w": Members((SELL,)),
    }
    # `w` is unconstrained when `v` is a `Buy`, so only the joined facts on `v` remain.
    assert _env("isinstance(v, Buy) or isinstance(w, Sell)", v=DECISION, w=DECISION) == {
        "v": Members((BUY, HOLD, SELL))
    }
    # Each way through the `or` proves something about `v`.
    assert _env(
        "isinstance(v, Buy) and isinstance(w, Sell) or isinstance(v, Hold)",
        v=DECISION,
        w=DECISION,
    ) == {"v": Members((BUY, HOLD))}


def test_guard_env_skips_unknown_types_and_bases_it_cannot_narrow():
    assert _env("isinstance(v, Nope) and isinstance(w, Buy)", v=DECISION, w=DECISION) == {
        "w": Members((BUY,))
    }
    assert unresolved_type(_g("isinstance(v, Nope) and isinstance(w, Buy)"), _REGISTRY)
    assert not unresolved_type(_g("isinstance(w, Buy) and v.action == 'x'"), _REGISTRY)
    assert not unresolved_type(_g("isinstance(w, Buy) or isinstance(v, int)"), _REGISTRY)

    def base_type_of(base):
        return DECISION if base == "v" else NOT_NARROWABLE

    guard = _g("isinstance(v, Buy) and isinstance(x, Sell)")
    assert guard_env(guard, base_type_of, _REGISTRY) == {"v": Members((BUY,))}


def test_guard_env_types_a_member_only_base_through_the_checks_before_it():
    assert _env("v.action == 'buy' and v.leg.type == 'limit'", v=DECISION) == {
        "v": Members((BUY,)),
        "v.leg": Members((LIMIT,)),
    }
    assert _env("isinstance(v, Buy) and not isinstance(v.leg, Limit)", v=DECISION) == {
        "v": Members((BUY,)),
        "v.leg": Members((MARKET,)),
    }
    # Into the parts of a nested `or`, and through the `and` inside it.
    assert _env(
        "v.action == 'buy' and (v.leg.type == 'limit' or v.leg.type == 'market')",
        v=DECISION,
    ) == {"v": Members((BUY,)), "v.leg": Members((LIMIT, MARKET))}
    # With no check before it, the member-only base stays unproven.
    assert _env("v.leg.type == 'limit' and v.action == 'buy'", v=DECISION) == {
        "v": Members((BUY,))
    }
    assert _env("isinstance(v.leg, Limit) and v.action == 'buy'", v=DECISION) == {
        "v": Members((BUY,)),
        "v.leg": Exact(LIMIT),
    }


def test_env_unsatisfiable():
    assert env_unsatisfiable({"v": Members(())})
    assert env_unsatisfiable({"w": Exact(BUY), "v": Members(())})
    assert not env_unsatisfiable({"v": Members((), nullable=True)})
    assert not env_unsatisfiable({"v": Exact(BUY)})
    assert not env_unsatisfiable({})
    assert env_unsatisfiable(_env("isinstance(v, Buy) and v.action == 'hold'", v=DECISION))


def _reads(read: str, condition: str, **declared):
    """The candidate types of `read` under what `condition` proves."""
    return guarded_read_types(read, _g(condition), _types(**declared), _REGISTRY)


def test_guarded_read_types_reads_below_the_narrowed_base():
    assert _reads("v.qty", "isinstance(v, Buy)", v=DECISION) == [(BUY, ("qty",), "v")]
    assert _reads("v.leg.price", "isinstance(v, Buy)", v=DECISION) == [
        (BUY, ("leg", "price"), "v")
    ]
    assert _reads("v.qty", "isinstance(v, Buy) or isinstance(v, Sell)", v=DECISION) == [
        (BUY_OR_SELL, ("qty",), "v")
    ]


def test_guarded_read_types_takes_the_longest_base():
    condition = "isinstance(v, Buy) and isinstance(v.leg, Limit)"
    assert _reads("v.leg.price", condition, v=DECISION) == [
        (LIMIT, ("price",), "v.leg"),
        (BUY, ("leg", "price"), "v"),
    ]


def test_guarded_read_types_without_a_fact_for_the_read():
    assert _reads("v", "isinstance(v, Buy)", v=DECISION) == []
    assert _reads("w.qty", "isinstance(v, Buy)", v=DECISION) == []
    assert _reads("vv.qty", "isinstance(v, Buy)", v=DECISION) == []
    assert _reads("v.qty", "n > 1", v=DECISION) == []
    # `isinstance(v, Buy) and n > 1` may be false through `n > 1`, so `v` keeps every member.
    assert _reads("v.qty", "not (isinstance(v, Buy) and n > 1)", v=DECISION) == []


def test_guarded_read_types_keeps_each_conjunct_s_candidate():
    """Two unrelated exact types lose their meet, but each still narrows the read."""
    assert _reads("v.qty", "isinstance(v, Buy) and isinstance(v, Text)", v=OBJECT) == [
        (BUY, ("qty",), "v"),
        (TEXT, ("qty",), "v"),
    ]


def _checked(condition: str, routing=None, **declared):
    """The guard each read in `condition` is checked under, by reference path."""
    tree = parse_condition(condition)
    return checked_refs(tree, _types(**declared), _REGISTRY, routing=routing)


def test_checked_refs_keeps_a_guard_some_value_satisfies():
    assert _checked("isinstance(v, Buy) and v.qty > 0", v=DECISION) == [
        ("v", NOTHING),
        ("v.qty", Holds(IsinstanceCheck("v", "Buy"), True)),
    ]


def test_a_read_after_a_reported_check_is_not_checked():
    assert _checked("isinstance(v, Buy) and v.action == 'hold' and v.qty > 0", v=DECISION)[
        -1
    ] == ("v.qty", None)
    assert _checked("v.action != 'biy' or v.qty > 0", v=DECISION)[-1] == ("v.qty", None)
    assert _checked("v.action == 'zzz' and v.qty > 0", v=DECISION)[-1] == ("v.qty", None)
    # A negated check the declared type always passes is reported, so what follows is dead.
    assert _checked("not isinstance(v, Decision) and v.qty > 0", v=DECISION)[-1] == (
        "v.qty",
        None,
    )
    routing = {"v": Members((BUY,))}
    assert _checked("isinstance(v, Sell) and v.qty > 0", routing=routing, v=DECISION)[-1] == (
        "v.qty",
        None,
    )


def test_a_read_after_a_check_the_known_type_decides_is_checked_without_it():
    assert _checked("isinstance(v, Decision) or v.nosuch > 0", v=DECISION)[-1] == (
        "v.nosuch",
        NOTHING,
    )
    # A false check on a non-union base proves nothing, so its guard is kept.
    read, guard = _checked("isinstance(v, Buy) or v.qty > 0", v=BUY)[-1]
    assert guarded_read_types(read, guard, _types(v=BUY), _REGISTRY) == []
    # The other checks before the read still narrow it.
    assert _checked("v.action == 'buy' or isinstance(v, Decision) or v.qty > 0", v=DECISION)[
        -1
    ] == ("v.qty", Holds(TagCheck("v", "action", "buy"), False))
    routing = {"v": Members((BUY,))}
    assert _checked("isinstance(v, Buy) or v.qty > 0", routing=routing, v=DECISION)[-1] == (
        "v.qty",
        NOTHING,
    )
    assert _checked("isinstance(v, Buy) or v.qty > 0", v=DECISION)[-1] == (
        "v.qty",
        Holds(IsinstanceCheck("v", "Buy"), False),
    )
    # The checks before it decide the second check, not the type: nothing is reported, so
    # the read is checked unnarrowed.
    assert _checked("isinstance(v, Buy) and (isinstance(v, Buy) or v.qty > 0)", v=DECISION)[
        -1
    ] == ("v.qty", NOTHING)


# --------------------------------------------------------------------------- #
# Impossible checks and narrowed read messages
# --------------------------------------------------------------------------- #

_MORE_REGISTRY = read_typedefs(
    {**_TYPEDEFS, "MaybeText": "Optional(Text)", "OptBuy": "Optional(Buy)"}
)
BOOL = _t("bool")


def _impossible(condition: str, registry=_REGISTRY, routing=None, **declared) -> list[str]:
    """The impossible-check messages of `condition`, with the bases typed by `declared`."""
    found = impossible_checks(
        parse_condition(condition), _types(**declared), registry, routing=routing
    )
    return [impossible_check_message(check) for check in found]


def test_a_check_the_checks_before_it_rule_out_is_impossible():
    found = impossible_checks(
        parse_condition("isinstance(v, Buy) and v.action == 'hold'"),
        _types(v=DECISION),
        _REGISTRY,
    )
    assert [(c.text, c.base, c.reason) for c in found] == [
        ('v.action == "hold"', "v", ImpossibleReason.CONTEXT)
    ]
    assert impossible_check_message(found[0]) == (
        '`v.action == "hold"` can never be true here: v is Buy, by the checks before it'
    )
    assert _impossible("not isinstance(v, Buy) or v.action == 'hold'", v=DECISION) == [
        '`v.action == "hold"` can never be true here: v is Buy, by the checks before it'
    ]
    assert _impossible("v.action != 'buy' and isinstance(v, Buy)", v=DECISION) == [
        "`isinstance(v, Buy)` can never be true here: v is Hold | Sell, "
        "by the checks before it"
    ]


def test_a_check_the_checks_before_it_make_always_true_says_so():
    assert _impossible("isinstance(v, Buy) and not isinstance(v, Buy)", v=DECISION) == [
        "`not isinstance(v, Buy)` can never be true here: v is Buy, by the checks before it, "
        "so `isinstance(v, Buy)` is always true"
    ]
    assert _impossible("isinstance(v, Buy) and v.action != 'buy'", v=DECISION) == [
        '`v.action != "buy"` can never be true here: v is Buy, by the checks before it, '
        'so `v.action == "buy"` is always true'
    ]


def test_a_tag_value_no_member_has_is_reported():
    message = (
        "compares v.action with 'biy', which is not one of its tags (buy, hold, sell); "
        "did you mean 'buy'?"
    )
    assert _impossible("v.action == 'biy'", v=DECISION) == [f'`v.action == "biy"` {message}']
    assert _impossible("v.action != 'biy'", v=DECISION) == [f'`v.action != "biy"` {message}']
    assert _impossible("v.action == 'zzzz'", v=DECISION) == [
        "`v.action == \"zzzz\"` compares v.action with 'zzzz', which is not one of its tags "
        "(buy, hold, sell)"
    ]
    # A field only `Buy` has is typed through the check before it.
    assert _impossible("v.action == 'buy' and v.leg.type == 'lmit'", v=DECISION) == [
        "`v.leg.type == \"lmit\"` compares v.leg.type with 'lmit', which is not one of its "
        "tags (limit, market); did you mean 'limit'?"
    ]


def test_an_isinstance_the_declared_type_rules_out_is_impossible():
    message = (
        "`isinstance(v, Text)` can never be true: v is declared Decision, "
        "and no Decision value fits Text"
    )
    assert _impossible("isinstance(v, Text)", v=DECISION) == [message]
    assert _impossible("not not isinstance(v, Text)", v=DECISION) == [message]
    assert _impossible("not isinstance(v, Text)", v=DECISION) == []
    # A missing value is checked as null, which a nullable type admits.
    assert _impossible("isinstance(v, MaybeText)", registry=_MORE_REGISTRY, v=DECISION) == []
    assert _impossible("isinstance(v, int)", v=BOOL) == [
        "`isinstance(v, int)` can never be true: v is declared bool, and no bool value fits int"
    ]


def test_an_isinstance_on_a_nested_base_the_type_rules_out_is_impossible():
    assert _impossible("isinstance(v.leg, Text)", v=BUY, v__leg=_t("Leg")) == [
        "`isinstance(v.leg, Text)` can never be true: v.leg is declared Leg, "
        "and no Leg value fits Text"
    ]
    # `leg` is only on `Buy`, so its type comes from the check before it.
    assert _impossible("isinstance(v, Buy) and isinstance(v.leg, Text)", v=DECISION) == [
        "`isinstance(v.leg, Text)` can never be true: v.leg is Leg here, "
        "by the checks before it, and no Leg value fits Text"
    ]


def test_a_negated_isinstance_the_declared_type_always_passes_is_impossible():
    assert _impossible("not isinstance(v, Buy)", v=BUY) == [
        "`not isinstance(v, Buy)` can never be true: v is declared Buy, "
        "so `isinstance(v, Buy)` is always true"
    ]
    assert _impossible("not isinstance(v, BuyOrSell)", v=BUY) == [
        "`not isinstance(v, BuyOrSell)` can never be true: v is declared Buy, "
        "so `isinstance(v, BuyOrSell)` is always true"
    ]
    assert _impossible("not isinstance(v, Decision)", v=DECISION) == [
        "`not isinstance(v, Decision)` can never be true: v is declared Decision, "
        "so `isinstance(v, Decision)` is always true"
    ]
    assert _impossible("not isinstance(v, Buy)", v=replace(BUY, nullable=True)) == []
    assert _impossible("isinstance(v, Buy)", v=BUY) == []


def test_a_negated_isinstance_of_a_nullable_typedef():
    nullable_buy = replace(BUY, nullable=True)
    expected = "so `isinstance(v, OptBuy)` is always true"
    assert _impossible("not isinstance(v, OptBuy)", registry=_MORE_REGISTRY, v=nullable_buy) == [
        f"`not isinstance(v, OptBuy)` can never be true: v is declared Optional(Buy), {expected}"
    ]
    assert _impossible("not isinstance(v, OptBuy)", registry=_MORE_REGISTRY, v=BUY) == [
        f"`not isinstance(v, OptBuy)` can never be true: v is declared Buy, {expected}"
    ]


def test_possible_checks_are_not_reported():
    assert _impossible("isinstance(v, Buy) or isinstance(v, Sell)", v=DECISION) == []
    assert _impossible("isinstance(v, Buy) and v.action == 'buy'", v=DECISION) == []
    assert _impossible("(v.action == 'biy') == true", v=DECISION) == []
    assert _impossible("v.qty == 'x'", v=DECISION) == []
    # An always-true check is no error on its own.
    assert _impossible("isinstance(v, Decision)", v=DECISION) == []
    assert _impossible("isinstance(v, Decision) and v.qty > 0", v=DECISION) == []


def test_unproven_bases_and_lost_facts_prove_nothing_impossible():
    assert _impossible("isinstance(v, Buy) and v.action == 'hold'", v=OBJECT) == []
    assert _impossible("isinstance(v, Buy) and v.action == 'hold'") == []
    assert _impossible("isinstance(v, Buy) and isinstance(v, Text)", v=OBJECT) == []
    assert _impossible("isinstance(v, Buy) and isinstance(v, Text)") == []
    assert _impossible("isinstance(v, Nope)", v=DECISION) == []
    assert _impossible("isinstance(v, Nope) and not isinstance(v, Nope)", v=DECISION) == []

    def base_type_of(base):
        return NOT_NARROWABLE

    tree = parse_condition("isinstance(v, Text) and v.action == 'biy'")
    assert impossible_checks(tree, base_type_of, _REGISTRY) == []


def test_each_impossible_check_is_reported_once_and_only_first():
    assert _impossible("v.action == 'biy' or v.action == 'biy'", v=DECISION) == [
        '`v.action == "biy"` compares v.action with \'biy\', which is not one of its tags '
        "(buy, hold, sell); did you mean 'buy'?"
    ]
    # Once the checks before a literal can't all hold, it is not reported again.
    assert _impossible(
        "isinstance(v, Buy) and v.action == 'hold' and v.action == 'sell'", v=DECISION
    ) == ['`v.action == "hold"` can never be true here: v is Buy, by the checks before it']


def test_code_after_an_always_true_check_is_not_reported():
    # The not-a-tag text is wrong whatever runs before it.
    assert _impossible("isinstance(v, Decision) or v.action == 'biy'", v=DECISION) == [
        "`v.action == \"biy\"` compares v.action with 'biy', which is not one of its tags "
        "(buy, hold, sell); did you mean 'buy'?"
    ]
    # A read after a check the declared type decides is checked by its site, unnarrowed.
    assert _impossible("isinstance(v, Decision) or v.nosuch > 0", v=DECISION) == []
    assert _impossible("v.action == 'buy' or isinstance(v, Decision) or v.qty > w", v=DECISION) == []
    assert _impossible("isinstance(v, Text) or isinstance(v, Decision) or v.x", v=DECISION) == [
        "`isinstance(v, Text)` can never be true: v is declared Decision, "
        "and no Decision value fits Text"
    ]


def test_a_check_the_routing_rules_out_is_impossible():
    assert _impossible("v.action == 'hold'", routing={"v": Members((BUY,))}, v=DECISION) == [
        '`v.action == "hold"` can never be true here: v is Buy, '
        "by the case branch that routes this node"
    ]
    assert _impossible(
        "v.action == 'buy'", routing={"v": Members((BUY, SELL))}, v=DECISION
    ) == []
    assert _impossible(
        "not isinstance(v, Sell) and isinstance(v, Hold)",
        routing={"v": Members((BUY, SELL))},
        v=DECISION,
    ) == [
        "`isinstance(v, Hold)` can never be true here: v is Buy | Sell, "
        "by the case branch that routes this node"
    ]
    # A node the routing never runs gets no impossible checks.
    assert _impossible("v.action == 'hold'", routing={"v": Members(())}, v=DECISION) == []


def test_routing_and_the_checks_together_use_the_checks_wording():
    assert _impossible(
        "not isinstance(v, Buy) and v.action != 'sell'",
        routing={"v": Members((BUY, SELL))},
        v=DECISION,
    ) == [
        '`v.action != "sell"` can never be true here: v is Sell, by the checks before it, '
        'so `v.action == "sell"` is always true'
    ]
    assert _impossible(
        "isinstance(v, Buy) or v.qty > 0", routing={"v": Members((BUY,))}, v=DECISION
    ) == []


def test_narrowed_type_label_names_members():
    assert narrowed_type_label(BUY, _REGISTRY) == "Buy"
    assert narrowed_type_label(BUY_OR_SELL, _REGISTRY) == "Buy | Sell"
    assert narrowed_type_label(replace(BUY_OR_SELL, nullable=True), _REGISTRY) == (
        "Optional(Buy | Sell)"
    )
    assert narrowed_type_label(replace(BUY, nullable=True), _REGISTRY) == "Optional(Buy)"
    anonymous = replace(TEXT, fields={"note": Type.scalar(ValueKind.STRING)})
    assert narrowed_type_label(anonymous, _REGISTRY) == canonical_type_label(anonymous)
    assert narrowed_type_label(_t("SellOrWide"), _REGISTRY) == "Wide | Sell"


def test_narrowed_read_message():
    assert narrowed_read_message("r.answr", "answr", "Done", "r", ["answer"]) == (
        "reference ${r.answr} reads field 'answr', which Done doesn't have; r is Done here, "
        "by the checks before this read; did you mean 'answer'?"
    )
    assert narrowed_read_message("v.qty", "qty", "Hold", "v", ["action"]) == (
        "reference ${v.qty} reads field 'qty', which Hold doesn't have; v is Hold here, "
        "by the checks before this read"
    )


def test_isinstance_members_keeps_the_shared_members_only_when_nothing_else_can_pass():
    assert isinstance_members(DECISION, _t("Rest")) == (HOLD, SELL)
    assert isinstance_members(DECISION, BUY) == (BUY,)
    assert isinstance_members(DECISION, _t("SellOrWide")) is None
    assert isinstance_members(DECISION, TEXT) is None
    assert isinstance_members(TEXT, TEXT) is None


# --------------------------------------------------------------------------- #
# isinstance checks in a searched CASE
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("when", "checks"),
    [
        ("isinstance(d.output, Buy)", [("d.output", "Buy")]),
        ("x > 1 and isinstance(d.output.leg, Limit)", [("d.output.leg", "Limit")]),
        ("${isinstance(input.d, Buy) and isinstance(e.output, Text)}",
         [("input.d", "Buy"), ("e.output", "Text")]),
        ("isinstance(d.output, Buy) or x", []),
        ("not isinstance(d.output, Buy)", []),
        ("isinstance(upper(d.output), Buy)", []),
        ("isinstance(d.output.items[0], Buy)", []),
        ("isinstance(", []),
        (3, []),
    ],
)
def test_isinstance_conjuncts_reads_top_level_and_operands(when, checks):
    assert isinstance_conjuncts(when) == checks


def test_searched_param_guards_lists_the_checks_before_each_read():
    desc, _ = _case(
        "c",
        None,
        [
            ("isinstance(d.output, Buy) and d.output.qty > 5", "b"),
            ("d.output.qty > 1 and isinstance(d.output, Sell)", "s"),
        ],
    )
    assert searched_param_guards(desc) == {
        "__r0": [(0, ()), (1, ())],
        "__r1": [(0, (("d.output", "Buy"),)), (1, ())],
    }
    on_desc, _ = _case("o", "${d.output.action}", [("buy", "b")])
    assert searched_param_guards(on_desc) == {}


def _searched_view(cases, else_=None, typ: Type = DECISION, target: str = "b") -> Type:
    """The type `target` sees for `d` under a searched CASE `c` over `cases`."""
    nodes = [_node("d"), _case("c", None, cases, else_), _node("b"), _node("s")]
    narrowing = _compute(nodes, [], {"d": ValueContract(typ)})
    return _view_type(narrowing, target)


def test_an_isinstance_case_narrows_its_target_to_the_member_or_sub_union():
    assert _searched_view([("isinstance(d.output, Buy)", "b")]) == BUY
    assert _searched_view([("isinstance(d.output, BuyOrSell) and x", "b")]) == BUY_OR_SELL


def test_an_isinstance_case_on_a_non_union_base_gives_the_exact_type():
    obj = Type.scalar(ValueKind.OBJECT)
    nodes = [_node("d"), _case("c", None, [("isinstance(d.output, Text)", "b")]), _node("b")]
    narrowing = _compute(nodes, [], {"d": ValueContract(obj)})
    assert narrowing.environment("b") == {"d.output": Exact(TEXT)}
    assert _view_type(narrowing, "b") == TEXT


def test_isinstance_facts_use_the_original_base_union():
    """Under an outer `buy` branch, an inner `isinstance(d.output, Buy)` still names a member."""
    nodes = [
        _node("d"),
        _case("o", "${d.output.action}", [("buy", "c")], "h"),
        _case("c", None, [("isinstance(d.output, Buy)", "b")]),
        _node("b"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": ValueContract(DECISION)})
    assert narrowing.environment("b") == {"d.output": Members((BUY,))}


def test_an_isinstance_of_a_type_the_base_fits_gives_no_fact():
    nodes = [
        _node("d"),
        _case("c", None, [("isinstance(d.output, Decision) and isinstance(d.output, Buy)", "b")]),
        _node("b"),
    ]
    narrowing = _compute(nodes, [], {"d": ValueContract(DECISION)})
    assert narrowing.environment("b") == {"d.output": Members((BUY,))}


def test_two_isinstance_cases_into_one_target_join_by_union():
    cases = [("isinstance(d.output, Buy)", "b"), ("isinstance(d.output, Sell)", "b")]
    assert _searched_view(cases) == BUY_OR_SELL


def test_two_isinstance_cases_on_unrelated_types_into_one_target_drop_the_fact():
    obj = ValueContract(Type.scalar(ValueKind.OBJECT))
    cases = [("isinstance(d.output, Text)", "b"), ("isinstance(d.output, Buy)", "b")]
    nodes = [_node("d"), _case("c", None, cases), _node("b")]
    narrowing = _compute(nodes, [], {"d": obj})
    assert narrowing.environment("b") == {}
    assert narrowing.for_node("b")[0]["d"] is obj


def test_a_plain_case_or_else_into_the_target_drops_the_fact():
    nodes = [
        _node("d"),
        _case("c", None, [("isinstance(d.output, Buy)", "b"), ("x > 1", "b")], "s"),
        _node("b"),
        _node("s"),
    ]
    narrowing = _compute(nodes, [], {"d": ValueContract(DECISION)})
    assert narrowing.environment("b") == {}
    assert narrowing.environment("s") == {}


def test_an_unknown_isinstance_type_gives_no_fact():
    nodes = [_node("d"), _case("c", None, [("isinstance(d.output, Nope)", "b")]), _node("b")]
    narrowing = _compute(nodes, [], {"d": ValueContract(DECISION)})
    assert narrowing.environment("b") == {}


def test_a_guarded_case_param_gets_the_checked_view():
    nodes = [
        _node("d"),
        _case(
            "c",
            None,
            [
                ("isinstance(d.output, Buy) and d.output.qty > 5", "b"),
                ("isinstance(d.output, Sell) and d.output.qty > 1", "s"),
                ("d.output.action == 'hold' and isinstance(d.output, Hold)", "s"),
            ],
        ),
        _node("b"),
        _node("s"),
    ]
    narrowing = _compute(nodes, [], {"d": ValueContract(DECISION)})
    assert narrowing.for_param("c", "__r1")[0]["d"].type == BUY_OR_SELL
    # `d.output` and `d.output.action` are read unguarded: they see the CASE's own view.
    assert narrowing.for_param("c", "__r0")[0]["d"].type == DECISION
    assert narrowing.for_param("c", "__r2")[0]["d"].type == DECISION


# --------------------------------------------------------------------------- #
# Propagation
# --------------------------------------------------------------------------- #


def test_on_case_narrows_each_branch_and_else_to_the_unlisted_members():
    """Each `when:` branch sees its member; `else:` sees the rest."""
    producers = {"d": _decision_contract()}
    narrowing = _compute(_branches(), [], producers)

    assert _view_type(narrowing, "b") == BUY
    assert _view_type(narrowing, "s") == SELL
    assert _view_type(narrowing, "h") == HOLD
    assert narrowing.environment("b") == {"d.output": Members((BUY,))}


def test_views_keep_origins_and_unnarrowed_nodes_get_the_same_maps():
    """A narrowed view keeps the producer's origins; other nodes see the inputs as-is."""
    producers = {"d": _decision_contract()}
    inputs = {"s": ValueContract(Type.scalar(ValueKind.STRING))}
    narrowing = _compute(_branches(), [], producers, inputs)

    view, view_inputs = narrowing.for_node("b")
    assert view["d"].origins == producers["d"].origins
    assert view_inputs is inputs
    plain_producers, plain_inputs = narrowing.for_node("d")
    assert plain_producers is producers and plain_inputs is inputs
    assert narrowing.for_node("c") == (producers, inputs)


def test_else_keeps_the_base_nullability_and_when_branches_drop_it():
    """A nullable base stays nullable under `else:` only."""
    producers = {"d": _decision_contract(replace(DECISION, nullable=True))}
    narrowing = _compute(_branches(), [], producers)

    assert _view_type(narrowing, "b") == BUY
    assert _view_type(narrowing, "h") == replace(HOLD, nullable=True)


def test_a_when_branch_and_else_into_one_target_join_to_every_member():
    """Routing `buy` and `else:` to one node leaves every member possible."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "x")], "x"),
        _node("x"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert narrowing.environment("x") == {"d.output": Members((BUY, HOLD, SELL))}
    assert narrowing.for_node("x")[0]["d"].type == DECISION


def test_a_then_target_named_default_is_narrowed_like_any_node():
    """A `then:` node named `default` sees its member; the name is not reserved."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "b"), ("sell", "default")]),
        _node("b"),
        _node("default"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert _view_type(narrowing, "b") == BUY
    assert _view_type(narrowing, "default") == SELL


def test_a_then_target_named_default_leaves_the_else_target_the_unlisted_members():
    """With `then: default` and `else: h`, `h` still sees the unlisted member."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "b"), ("sell", "default")], "h"),
        _node("b"),
        _node("default"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert _view_type(narrowing, "b") == BUY
    assert _view_type(narrowing, "default") == SELL
    assert _view_type(narrowing, "h") == HOLD


def test_else_keeps_nullability_from_a_nullable_path_segment():
    """A nullable record on the way to the base makes the `else:` members nullable."""
    holder = Type(
        kind=ValueKind.OBJECT,
        fields={"dec": DECISION},
        required=frozenset({"dec"}),
        nullable=True,
    )
    nodes = [
        _node("d"),
        _case("c", "${d.output.dec.action}", [("buy", "b")], "h"),
        _node("b"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": ValueContract(holder)})

    assert narrowing.environment("h") == {
        "d.output.dec": Members((HOLD, SELL), nullable=True)
    }
    assert _view_type(narrowing, "h").fields["dec"] == Type(
        kind=ValueKind.OBJECT, variants=(HOLD, SELL), nullable=True
    )
    assert _view_type(narrowing, "b").fields["dec"] == BUY


def test_control_edges_join_by_union_and_an_edge_without_facts_unconstrains():
    """Two branches into one node give both members; a fact-free route removes the fact."""
    nodes = [
        _node("d"),
        _node("z"),
        _case("c", "${d.output.action}", [("buy", "x"), ("sell", "x"), ("hold", "y")]),
        _case("g", None, [("z.output > 0", "y")]),
        _node("x"),
        _node("y"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert _view_type(narrowing, "x") == BUY_OR_SELL
    assert narrowing.environment("y") == {}
    assert narrowing.for_node("y")[0]["d"].type == DECISION


def test_a_required_data_group_joins_its_producers_by_union():
    """A coalesce of the buy and sell branches sees both members."""
    edges = [
        _data("b", "m", "v"),
        _data("s", "m", "v"),
        _data("b", "o", "v", optional=True),
    ]
    nodes = _branches() + [_node("m"), _node("o")]
    narrowing = _compute(nodes, edges, {"d": _decision_contract()})

    assert _view_type(narrowing, "m") == BUY_OR_SELL
    assert narrowing.environment("o") == {}


def test_depends_on_carries_the_source_environment_and_runs_after_does_not():
    """A co-skipping ordering edge narrows; a pure-order edge does not."""
    nodes = _branches() + [_node("n", depends_on=["b"]), _node("r", runs_after=["b"])]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert _view_type(narrowing, "n") == BUY
    assert narrowing.environment("r") == {}


def test_contributions_intersect():
    """A data group from both branches and a dependency on one give that one."""
    edges = [_data("b", "m", "v"), _data("s", "m", "v")]
    nodes = _branches() + [_node("m", depends_on=["s"])]
    narrowing = _compute(nodes, edges, {"d": _decision_contract()})

    assert _view_type(narrowing, "m") == SELL


def test_two_required_groups_from_disjoint_branches_meet_empty():
    """An empty member set on a non-nullable base leaves the node unnarrowed."""
    edges = [_data("b", "m", "u"), _data("s", "m", "v")]
    nodes = _branches() + [_node("m")]
    narrowing = _compute(nodes, edges, {"d": _decision_contract()})

    assert narrowing.environment("m") == {"d.output": Members(())}
    assert _view_type(narrowing, "m") == DECISION


def test_nested_case_composes_with_its_own_environment():
    """A CASE inside a branch narrows within that branch's members."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "c2"), ("sell", "c2")], "h"),
        _case("c2", "${d.output.action}", [("buy", "p")], "q"),
        _node("p"),
        _node("q"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert _view_type(narrowing, "c2") == BUY_OR_SELL
    assert _view_type(narrowing, "p") == BUY
    assert _view_type(narrowing, "q") == SELL


def test_a_contradictory_nested_case_branch_is_empty():
    """A sell branch inside the buy branch never runs, so it stays unnarrowed."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "c2")], "h"),
        _case("c2", "${d.output.action}", [("sell", "z")]),
        _node("z"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert narrowing.environment("z") == {"d.output": Members(())}
    assert _view_type(narrowing, "z") == DECISION


def test_outer_bases_narrow_before_inner_ones():
    """Narrowing `d.output` to Buy lets a nested case narrow `d.output.leg`."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "c2")], "h"),
        _case("c2", "${d.output.leg.type}", [("limit", "l")]),
        _node("l"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    seen = _view_type(narrowing, "l")
    assert seen.fields["leg"] == LIMIT
    assert seen == replace(BUY, fields={**BUY.fields, "leg": LIMIT})
    assert narrowing.for_node("l")[0]["d"].origins == _decision_contract().origins


def test_flow_input_bases_narrow_the_input_view():
    """An `on:` over `input.<name>.<tag>` narrows the flow-input contract."""
    inputs = {"order": ValueContract(DECISION)}
    nodes = [_case("c", "${input.order.action}", [("hold", "w")]), _node("w")]
    narrowing = _compute(nodes, [], {}, inputs)

    producers, view = narrowing.for_node("w")
    assert producers == {}
    assert view["order"].type == HOLD


def test_partial_record_producer_narrows_a_proven_field():
    """A proven union field of a partial record narrows inside the record."""
    contract = PartialRecordContract(
        fields=(("dec", ValueContract(DECISION)), ("raw", None))
    )
    nodes = [
        _node("d"),
        _case("c", "${d.output.dec.action}", [("sell", "s")]),
        _node("s"),
    ]
    narrowing = _compute(nodes, [], {"d": contract})

    view = narrowing.for_node("s")[0]["d"]
    assert view == PartialRecordContract(
        fields=(("dec", ValueContract(SELL)), ("raw", None))
    )


def test_bases_that_are_not_tagged_unions_give_no_facts():
    """A non-union base, a non-tag field, or a computed `on:` narrows nothing."""
    producers = {
        "d": _decision_contract(),
        "t": ValueContract(TEXT),
    }
    nodes = [
        _node("d"),
        _node("t"),
        _case("c1", "${t.output.text}", [("hi", "a")]),
        _case("c2", "${d.output.qty}", [("1", "b")]),
        _case("c3", "${d.output.action:-hold}", [("buy", "e")]),
        _node("a"),
        _node("b"),
        _node("e"),
    ]
    narrowing = _compute(nodes, [], producers)

    for node_id in ("a", "b", "e"):
        assert narrowing.environment(node_id) == {}


# --------------------------------------------------------------------------- #
# Empty sets
# --------------------------------------------------------------------------- #


def test_defensive_else_after_every_tag_is_empty_and_unnarrowed():
    """`else:` after listing every tag never runs; a non-nullable base keeps its type."""
    nodes = [
        _node("d"),
        _case(
            "c",
            "${d.output.action}",
            [("buy", "b"), ("hold", "b"), ("sell", "b")],
            "e",
        ),
        _node("b"),
        _node("e"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert narrowing.environment("e") == {"d.output": Members(())}
    assert _view_type(narrowing, "e") == DECISION
    assert _view_type(narrowing, "b") == DECISION


def test_defensive_else_over_a_nullable_base_sees_none():
    """`else:` after every tag on a nullable base can only see `None`."""
    nodes = [
        _node("d"),
        _case(
            "c",
            "${d.output.action}",
            [("buy", "b"), ("hold", "b"), ("sell", "b")],
            "e",
        ),
        _node("b"),
        _node("e"),
    ]
    producers = {"d": _decision_contract(replace(DECISION, nullable=True))}
    narrowing = _compute(nodes, [], producers)

    assert narrowing.environment("e") == {"d.output": Members((), nullable=True)}
    assert _view_type(narrowing, "e").kind is ValueKind.NONE


def _descriptors(nodes) -> dict[str, NodeDescriptor]:
    """The descriptors `_compute` builds from `nodes`, by id."""
    descriptors = {}
    for item in nodes:
        desc = item[0] if isinstance(item, tuple) else item
        descriptors[desc.id] = desc
    return descriptors


def test_a_when_value_that_is_not_a_tag_narrows_its_branch_to_nothing():
    """Narrowing tolerates a typo'd tag: its branch gets an empty member set."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buyy", "b")], "h"),
        _node("b"),
        _node("h"),
    ]
    narrowing = _compute(nodes, [], {"d": _decision_contract()})

    assert narrowing.environment("b") == {"d.output": Members(())}


def test_a_when_value_that_is_not_a_tag_is_a_load_error():
    """A typo'd tag on a union base is rejected with the union's tags."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buyy", "b")], "h"),
        _node("b"),
        _node("h"),
    ]
    producers = {"d": _decision_contract()}
    narrowing = _compute(nodes, [], producers)
    with pytest.raises(LoadError) as caught:
        check_case_when_values(_descriptors(nodes), producers, {}, narrowing=narrowing)
    message = str(caught.value)
    assert message.startswith(
        "case node 'c': `when: buyy` is not a tag of `on: ${d.output.action}` ("
    )
    assert message.endswith("); use one of: buy, hold, sell")


def test_the_tag_error_points_at_the_when_value():
    """The error is located at the offending `when:` value's range."""
    ranges = [SourceRange(5, 15, 5, 18), SourceRange(7, 15, 7, 19)]
    nodes = [
        _node("d"),
        _case(
            "c",
            "${d.output.action}",
            [("buy", "b"), ("sel", "s")],
            when_ranges=ranges,
        ),
        _node("b"),
        _node("s"),
    ]
    producers = {"d": _decision_contract()}
    narrowing = _compute(nodes, [], producers)
    diagnostics = LoadDiagnostics(label="flow.yaml", text="line\n" * 10)
    with pytest.raises(LoadError) as caught:
        check_case_when_values(
            _descriptors(nodes),
            producers,
            {},
            narrowing=narrowing,
            diagnostics=diagnostics,
        )
    assert caught.value.diagnostic.frames[-1].range == ranges[1]


def test_a_tag_error_on_a_base_only_the_branch_view_has():
    """A member-only union field is checked against the union the branch sees."""
    nodes = [
        _node("d"),
        _case("c", "${d.output.action}", [("buy", "c2")], "h"),
        _case("c2", "${d.output.leg.type}", [("limt", "l")]),
        _node("l"),
        _node("h"),
    ]
    producers = {"d": _decision_contract()}
    narrowing = _compute(nodes, [], producers)
    with pytest.raises(LoadError) as caught:
        check_case_when_values(_descriptors(nodes), producers, {}, narrowing=narrowing)
    message = str(caught.value)
    assert message.startswith(
        "case node 'c2': `when: limt` is not a tag of `on: ${d.output.leg.type}` ("
    )
    assert message.endswith("); use one of: limit, market")


# --------------------------------------------------------------------------- #
# Cycles and unknown ids
# --------------------------------------------------------------------------- #


def test_nodes_kahn_cannot_order_stay_unnarrowed():
    """A branch target on a cycle, and what follows it, are reported unordered."""
    edges = [_data("a", "k", "v"), _data("k", "a", "w"), _data("k", "after", "v")]
    nodes = _branches() + [
        _node("a", depends_on=["b"]),
        _node("k"),
        _node("after"),
    ]
    narrowing = _compute(nodes, edges, {"d": _decision_contract()})

    assert narrowing.unordered == frozenset({"a", "k", "after"})
    assert _view_type(narrowing, "a") == DECISION
    assert _view_type(narrowing, "b") == BUY


def test_unknown_ids_are_ignored():
    """Edges and dependencies naming unknown nodes neither fail nor contribute."""
    edges = [_data("ghost", "b", "v"), _data("b", "nowhere", "v")]
    nodes = _branches() + [_node("n", depends_on=["b", "ghost"])]
    narrowing = _compute(nodes, edges, {"d": _decision_contract()})

    assert _view_type(narrowing, "b") == BUY
    assert _view_type(narrowing, "n") == BUY
    assert narrowing.unordered == frozenset()


# --------------------------------------------------------------------------- #
# narrow_contract_at
# --------------------------------------------------------------------------- #


def test_narrow_contract_at_the_root_and_a_nested_record_field():
    """The type at the path is replaced and every origin is kept."""
    contract = _decision_contract()
    assert narrow_contract_at(contract, (), BUY) == ValueContract(
        BUY, contract.origins
    )

    buy = ValueContract(BUY, contract.origins)
    narrowed = narrow_contract_at(buy, ("leg",), LIMIT)
    assert narrowed.type == replace(BUY, fields={**BUY.fields, "leg": LIMIT})
    assert narrowed.origins == contract.origins


def test_narrow_contract_at_does_not_cross_unions_or_missing_fields():
    """A path through a union, an absent field, or a scalar cannot be narrowed."""
    contract = _decision_contract()
    assert narrow_contract_at(contract, ("leg",), LIMIT) is None
    assert narrow_contract_at(ValueContract(BUY), ("nope",), LIMIT) is None
    assert narrow_contract_at(ValueContract(BUY), ("qty", "x"), LIMIT) is None


def test_narrow_contract_at_inside_a_partial_record():
    """A proven field narrows in place; an opaque one cannot be narrowed by a path."""
    inner = ValueContract(DECISION, ContractOrigins(default=_ROOT))
    contract = PartialRecordContract(fields=(("dec", inner), ("raw", None)))

    narrowed = narrow_contract_at(contract, ("dec",), BUY)
    assert narrowed == PartialRecordContract(
        fields=(("dec", ValueContract(BUY, inner.origins)), ("raw", None))
    )
    assert narrow_contract_at(contract, ("raw", "x"), BUY) is None
    assert narrow_contract_at(contract, ("missing",), BUY) is None


def test_narrow_contract_at_the_root_of_a_partial_record_rebases_field_origins():
    """Replacing a whole partial record keeps each field's origins under its name."""
    inner = ValueContract(
        DECISION,
        ContractOrigins(default=_ROOT, entries=((("qty",), _QTY),)),
    )
    contract = PartialRecordContract(fields=(("dec", inner), ("raw", None)))
    record = Type(
        kind=ValueKind.OBJECT,
        fields={"dec": BUY, "raw": Type.scalar(ValueKind.STRING)},
        required=frozenset({"dec", "raw"}),
    )

    narrowed = narrow_contract_at(contract, (), record)
    assert narrowed == ValueContract(
        record,
        ContractOrigins(entries=((("dec",), _ROOT), (("dec", "qty"), _QTY))),
    )


def test_narrowed_read_error_names_the_narrowed_type_only_for_a_field_it_lacks():
    assert narrowed_read_error("v.qtty", (BUY, ("qtty",), "v"), "plain", _REGISTRY) == (
        "reference ${v.qtty} reads field 'qtty', which Buy doesn't have; v is Buy here, "
        "by the checks before this read; did you mean 'qty'?"
    )
    assert narrowed_read_error("v.price", (BUY_OR_SELL, ("price",), "v"), "plain", _REGISTRY) == (
        "plain"
    )
    assert narrowed_read_error("v.nope", (BUY_OR_SELL, ("nope",), "v"), "plain", _REGISTRY) == (
        "reference ${v.nope} reads field 'nope', which Buy | Sell doesn't have; "
        "v is Buy | Sell here, by the checks before this read"
    )
    assert narrowed_read_error(
        "v.leg.prise", (BUY, ("leg", "prise"), "v"), "plain", _REGISTRY
    ) == "plain"
    assert narrowed_read_error("v.x", (BOOL, ("x",), "v"), "plain", _REGISTRY) == "plain"

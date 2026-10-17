"""Null checks narrow: `x != null` proves `x` is not null, as `isinstance` proves its type."""

import pytest

from yian.compose.helpers import narrowing_facts
from yian.compose.helpers.condition_guards import (
    NOTHING,
    Holds,
    NotNullCheck,
    check_in,
    check_text,
    guarded_refs,
    literals_in_context,
    proves,
)
from yian.compose.helpers.narrowing_facts import (
    Exact,
    ImpossibleReason,
    Members,
    check_fact,
    guard_env,
    impossible_check_message,
    impossible_checks,
    intersect_facts,
    unresolved_type,
)
from yian.compose.helpers.types import read_type
from yian.expr import parse_condition
from yian.typesys import Type, ValueKind
from yian.typesys.types import read_typedefs

_REGISTRY = read_typedefs(
    {
        "A": {"kind": "Literal(a)", "n": "int"},
        "B": {"kind": "Literal(b)", "s": "str"},
        "AB": "Union(A, B)",
        "Item": {"name": "str", "qty": "int"},
        "Wide": {"name": "str"},
        "Other": {"flag": "bool"},
        "MaybeA": "Optional(A)",
    }
)


def _t(name: str) -> Type:
    """Resolve one type expression against the test typedefs."""
    return read_type(name, _REGISTRY)


A, B, ITEM, WIDE, OTHER = _t("A"), _t("B"), _t("Item"), _t("Wide"), _t("Other")
STR = _t("str")
ANY = _t("Any")
NULL = Type.scalar(ValueKind.NONE)
X = NotNullCheck("x")


def _types(**declared):
    """A `base_type_of` over `declared`; other bases are unproven."""
    return lambda base: declared.get(base)


def _env(condition: str, **declared):
    """The facts `condition` proves when true, with the bases typed by `declared`."""
    return guard_env(proves(parse_condition(condition), True), _types(**declared), _REGISTRY)


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x != null", (X, True)),
        ("null != x", (X, True)),
        ("x != none", (X, True)),
        ("input.a.b != null", (NotNullCheck("input.a.b"), True)),
        ("${x != null}", (X, True)),
        ("x == null", (X, False)),
        ("null == x", (X, False)),
        ("x == none", (X, False)),
        ("x != 1", None),
        ("x.y != z", None),
        ("len(x) != null", None),
        ("x != None", None),
        ("null == null", None),
        ("x > null", None),
    ],
)
def test_check_in_recognizes_a_null_check(text, expected):
    assert check_in(parse_condition(text)) == expected


def test_a_null_check_proves_like_any_check():
    assert proves(parse_condition("not (x == null)"), True) == Holds(X, True)
    assert proves(parse_condition("x != null or y > 1"), True) == NOTHING
    reads = dict(guarded_refs(parse_condition("x == null or x.f > 0")))
    assert reads["x.f"] == Holds(X, True)


def test_check_text_spells_a_null_check_by_value():
    assert check_text(X) == "x != null"
    assert check_text(X, False) == "x == null"
    for text in ("x != null", "x == null"):
        check, value = check_in(parse_condition(text))
        assert check_text(check, value) == text


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("Optional(str)", Exact(STR)),
        ("Optional(AB)", Members((A, B), nullable=False)),
        ("Optional(Any)", Exact(ANY)),
        ("Optional(A)", Exact(A)),
        ("str", None),
        ("AB", None),
    ],
)
def test_a_true_null_check_proves_the_base_is_not_null(base, expected):
    assert check_fact(X, _t(base), True, _REGISTRY) == expected


def test_a_null_check_proves_nothing_else():
    assert check_fact(X, NULL, True, _REGISTRY) is None
    assert check_fact(X, None, True, _REGISTRY) is None
    assert check_fact(X, _t("Optional(str)"), False, _REGISTRY) is None
    assert check_fact(X, _t("Optional(AB)"), False, _REGISTRY) is None


def test_a_null_check_meets_an_isinstance_on_the_same_base():
    maybe_ab = _t("Optional(AB)")
    assert _env("x != null and isinstance(x, A)", x=maybe_ab) == {"x": Members((A,))}
    assert _env("isinstance(x, MaybeA) and x != null", x=maybe_ab) == {
        "x": Members((A,))
    }
    assert _env("x != null", x=maybe_ab) == {"x": Members((A, B))}
    assert _env("x == null", x=maybe_ab) == {}


@pytest.mark.parametrize("base", ["Optional(Any)", "Optional(Wide)"])
@pytest.mark.parametrize(
    "condition", ["x != null and isinstance(x, Item)", "isinstance(x, Item) and x != null"]
)
def test_a_null_check_keeps_an_isinstance_narrowing(base, condition):
    assert _env(condition, x=_t(base)) == {"x": Exact(ITEM)}


# --------------------------------------------------------------------------- #
# Meeting two exact facts
# --------------------------------------------------------------------------- #


def test_two_exact_facts_meet_to_the_narrower():
    assert intersect_facts(Exact(ANY), Exact(ITEM)) == Exact(ITEM)
    assert intersect_facts(Exact(ITEM), Exact(ANY)) == Exact(ITEM)
    assert intersect_facts(Exact(WIDE), Exact(ITEM)) == Exact(ITEM)
    assert intersect_facts(Exact(ITEM), Exact(WIDE)) == Exact(ITEM)
    assert intersect_facts(Exact(ITEM), Exact(ITEM)) == Exact(ITEM)


def test_two_unrelated_exact_facts_give_no_fact():
    assert intersect_facts(Exact(ITEM), Exact(OTHER)) is None
    assert intersect_facts(Exact(OTHER), Exact(ITEM)) is None


def test_two_isinstance_checks_on_one_base_meet_to_the_narrower():
    assert _env("isinstance(x, Wide) and isinstance(x, Item)", x=ANY) == {"x": Exact(ITEM)}


# --------------------------------------------------------------------------- #
# Impossible checks
# --------------------------------------------------------------------------- #


def test_a_null_check_names_no_type():
    assert unresolved_type(Holds(X, True), _REGISTRY) is False
    assert unresolved_type(Holds(X, False), _REGISTRY) is False


@pytest.mark.parametrize("text", ["x != null", "x == null", "not x != null"])
@pytest.mark.parametrize("base", ["str", "Optional(str)", "AB", "Any", "None"])
def test_a_null_check_is_never_impossible_by_its_declared_type(text, base):
    (literal,) = literals_in_context(parse_condition(text))
    assert narrowing_facts._declared_impossible(literal, _t(base), True, _REGISTRY) is None


@pytest.mark.parametrize(
    ("text", "never"), [("x != null", True), ("x == null", True), ("not x != null", False)]
)
def test_a_reported_null_check_literal_is_never_true(text, never):
    # The value is that of the comparison without the `not` around it.
    impossible = narrowing_facts.ImpossibleCheck("x", "x", ImpossibleReason.CONTEXT)
    (literal,) = literals_in_context(parse_condition(text))
    assert narrowing_facts._never_value(literal, impossible) is never


@pytest.mark.parametrize(
    "condition",
    [
        "x != null and x.name != ''",
        "x == null or x.name != ''",
        "x != null and x != null",
        "x == null and x == null",
    ],
)
@pytest.mark.parametrize("base", ["Item", "Optional(Item)"])
def test_a_null_check_on_any_value_is_not_reported(condition, base):
    tree = parse_condition(condition)
    assert impossible_checks(tree, _types(x=_t(base)), _REGISTRY) == []


def test_an_isinstance_check_is_still_reported_after_a_null_check():
    tree = parse_condition("x != null and isinstance(x, Other)")
    found = impossible_checks(tree, _types(x=_t("Optional(Item)")), _REGISTRY)
    assert [(check.text, check.reason) for check in found] == [
        ("isinstance(x, Other)", ImpossibleReason.NO_OVERLAP)
    ]



def _impossible(condition: str, routing=None) -> list[str]:
    """The impossible-check messages of `condition`, with `x` an `Optional(AB)`."""
    found = impossible_checks(
        parse_condition(condition), _types(x=_t("Optional(AB)")), _REGISTRY, routing=routing
    )
    return [impossible_check_message(check) for check in found]


def test_a_null_check_the_checks_before_it_prove_null_is_reported():
    assert _impossible("not isinstance(x, A) and not isinstance(x, B) and x != null") == [
        "`x != null` can never be true here: x is None, by the checks before it"
    ]


def test_a_check_a_null_check_before_it_makes_dead_is_reported():
    assert _impossible("x != null and not isinstance(x, A) and not isinstance(x, B)") == [
        "`not isinstance(x, B)` can never be true here: x is B, by the checks before it, "
        "so `isinstance(x, B)` is always true"
    ]


def test_a_null_check_the_routing_proves_null_is_reported():
    assert _impossible("x != null", routing={"x": Members((), nullable=True)}) == [
        "`x != null` can never be true here: x is None, by the case branch that routes "
        "this node"
    ]
    assert _impossible("x == null", routing={"x": Members((), nullable=True)}) == []

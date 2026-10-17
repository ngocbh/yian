"""Null checks narrow: `x != null` proves `x` is not null, as `isinstance` proves its type."""

import pytest

from yian.compose import LoadError, load_flow
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
from yian.compose.validate.narrowing import _narrow_type
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
        "MaybeItem": "Optional(Item)",
        "Holder": {"b": "Optional(str)"},
        "Outer": {"h": "Optional(Holder)"},
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


@pytest.mark.parametrize("base", ["Optional(Any)", "Optional(Wide)"])
@pytest.mark.parametrize(
    "condition",
    ["x != null and isinstance(x, MaybeItem)", "isinstance(x, MaybeItem) and x != null"],
)
def test_a_null_check_keeps_a_nullable_isinstance_narrowing(base, condition):
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


@pytest.mark.parametrize("other", [ANY, WIDE])
def test_a_non_null_exact_fact_meets_a_nullable_one_with_null_removed(other):
    maybe_item = _t("Optional(Item)")
    assert intersect_facts(Exact(other), Exact(maybe_item)) == Exact(ITEM)
    assert intersect_facts(Exact(maybe_item), Exact(other)) == Exact(ITEM)
    assert intersect_facts(Exact(ITEM), Exact(maybe_item)) == Exact(ITEM)
    assert intersect_facts(Exact(maybe_item), Exact(ITEM)) == Exact(ITEM)


def test_two_nullable_exact_facts_meet_to_the_narrower_still_nullable():
    maybe_item, maybe_wide = _t("Optional(Item)"), _t("Optional(Wide)")
    assert intersect_facts(Exact(maybe_wide), Exact(maybe_item)) == Exact(maybe_item)
    assert intersect_facts(Exact(maybe_item), Exact(maybe_wide)) == Exact(maybe_item)


def test_two_unrelated_exact_facts_give_no_fact():
    assert intersect_facts(Exact(ITEM), Exact(OTHER)) is None
    assert intersect_facts(Exact(OTHER), Exact(ITEM)) is None
    assert intersect_facts(Exact(OTHER), Exact(_t("Optional(Item)"))) is None


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


_AB_FLOW_HEAD = (
    "id: p\nname: p\n"
    "typedefs:\n"
    "  A:\n    kind: Literal(a)\n    n: int\n"
    "  B:\n    kind: Literal(b)\n    s: str\n"
    "  AB: Union(A, B)\n"
    "input:\n  x: !Optional(AB)\n"
    "nodes:\n"
)


def _code_node(node_id: str, extra: str = "") -> str:
    """A CODE node `node_id` returning `1`, with `extra` lines added to its body."""
    return f"  {node_id}:\n    kind: code\n    code: return 1\n{extra}"


def _searched(node_id: str, when: str, then: str, else_: str) -> str:
    """A searched CASE `node_id` routing to `then` under `when`, else to `else_`."""
    return (
        f"  {node_id}:\n    kind: case\n    cases:\n"
        f'      - when: "{when}"\n        then: {then}\n'
        f"    else: {else_}\n"
    )


def _load_error(text: str) -> str:
    """The message of the `LoadError` loading `text` raises."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return str(caught.value)


@pytest.mark.parametrize(
    ("when", "message"),
    [
        (
            "input.x != null and input.x.kind != 'a' and input.x.kind != 'b'",
            '`input.x.kind != "b"` can never be true here: input.x is B, by the checks '
            'before it, so `input.x.kind == "b"` is always true',
        ),
        (
            "isinstance(input.x, A) or isinstance(input.x, B) or input.x != null",
            "`input.x != null` can never be true here: input.x is None, by the checks before it",
        ),
        (
            "input.x.kind == 'a' or input.x.kind == 'b' or input.x != null",
            "`input.x != null` can never be true here: input.x is None, by the checks before it",
        ),
    ],
    ids=["tags-after-null-check", "null-check-after-isinstance", "null-check-after-tags"],
)
def test_a_when_null_check_impossible_with_the_checks_around_it_is_reported(when, message):
    text = (
        _AB_FLOW_HEAD
        + _searched("route", when, "t", "e")
        + _code_node("t")
        + _code_node("e")
        + "output: ${input.x}\n"
    )
    assert f"case node 'route': `cases[0].when`: {message}" in _load_error(text)


def test_a_node_assert_check_a_null_check_makes_impossible_is_reported():
    assertion = "x != null and not isinstance(x, A) and not isinstance(x, B)"
    body = f'    input:\n      x: ${{input.x}}\n    asserts:\n      - "{assertion}"\n'
    text = _AB_FLOW_HEAD + _code_node("c", body) + "output: ${c.output}\n"
    assert (
        "node 'c' assert: `not isinstance(x, B)` can never be true here: x is B, by the checks "
        "before it, so `isinstance(x, B)` is always true"
    ) in _load_error(text)


def test_a_null_check_case_under_a_branch_that_rules_out_every_member_is_reported():
    text = (
        _AB_FLOW_HEAD
        + _searched(
            "outer", "not isinstance(input.x, A) and not isinstance(input.x, B)", "inner", "e"
        )
        + _searched("inner", "input.x != null", "t", "e2")
        + _code_node("t")
        + _code_node("e")
        + _code_node("e2")
        + "output: ${input.x}\n"
    )
    assert (
        "case node 'inner': `cases[0].when`: `input.x != null` can never be true here: "
        "input.x is None, by the case branch that routes this node"
    ) in _load_error(text)


# --------------------------------------------------------------------------- #
# Nested optional fields
# --------------------------------------------------------------------------- #


def test_a_non_null_field_is_present_in_every_record_around_it():
    narrowed = _narrow_type(_t("Optional(Outer)"), ("h", "b"), STR)
    assert narrowed.nullable is False
    assert "h" in narrowed.required
    holder = narrowed.fields["h"]
    assert holder.nullable is False
    assert "b" in holder.required
    assert holder.fields["b"] == STR


def test_a_nullable_field_leaves_the_records_around_it_alone():
    outer = _t("Optional(Outer)")
    narrowed = _narrow_type(outer, ("h", "b"), _t("Optional(str)"))
    assert narrowed.nullable is True
    assert narrowed.required == outer.required
    assert narrowed.fields["h"].required == outer.fields["h"].required
    assert narrowed.fields["h"].nullable is True


def test_a_field_narrowed_to_null_leaves_the_records_around_it_alone():
    holder = _t("Optional(Holder)")
    narrowed = _narrow_type(holder, ("b",), NULL)
    assert narrowed.nullable is True
    assert narrowed.required == holder.required


def _nested_flow(when: str, a_type: str = "A", source: str = "${input.a.b}", sink: str = "str"):
    """A flow whose searched CASE passes `source` to a def taking `!<sink>` when `when`."""
    return (
        "id: p\nname: p\n"
        "typedefs:\n  A: {b: Optional(str)}\n"
        f"input:\n  a: !{a_type}\n"
        "nodes:\n"
        "  route:\n    kind: case\n    cases:\n"
        f'      - when: "{when}"\n'
        f"        then: call(need, v={source})\n"
        "    else: e\n"
        "  e:\n    kind: code\n    code: return 1\n"
        "output: ${route.output}\n"
        "defs:\n  need:\n"
        f"    input:\n      v: !{sink}\n"
        "    nodes:\n      p:\n        kind: code\n"
        "        input:\n          o: ${input.v}\n        code: return 1\n"
        "    output: ${p.output}\n"
    )


@pytest.mark.parametrize("when", ["isinstance(input.a.b, str)", "input.a.b != null"])
@pytest.mark.parametrize("a_type", ["A", "Optional(A)"])
def test_a_narrowed_optional_field_fills_a_required_input(when, a_type):
    load_flow(_nested_flow(when, a_type))


@pytest.mark.parametrize("when", ["isinstance(input.a.b, str)", "input.a.b != null"])
def test_a_narrowed_field_makes_its_parent_non_null(when):
    load_flow(_nested_flow(when, "Optional(A)", source="${input.a}", sink="A"))


def test_an_unnarrowed_optional_field_still_cannot_fill_a_required_input():
    with pytest.raises(LoadError, match="child expects 'str', source is 'Optional\\(str\\)'"):
        load_flow(_nested_flow("input.a.b == null"))


def _inner_on_flow(when: str) -> str:
    """A flow with an `on:` CASE over `input.a.b` inside the branch where it is not null."""
    return (
        "id: p\nname: p\n"
        "typedefs:\n  A: {b: Optional(str)}\n"
        "input:\n  a: !A\n"
        "nodes:\n"
        "  route:\n    kind: case\n    cases:\n"
        '      - when: "input.a.b != null"\n        then: inner\n'
        "    else: e\n"
        "  inner:\n    kind: case\n    on: ${input.a.b}\n    cases:\n"
        f"      - when: {when}\n        then: n\n"
        "    else: e2\n"
        "  n:\n    kind: code\n    code: return 1\n"
        "  e:\n    kind: code\n    code: return 1\n"
        "  e2:\n    kind: code\n    code: return 1\n"
        "output: ${route.output}\n"
    )


def test_a_null_when_is_no_on_value_even_on_a_narrowed_field():
    # YAML `null` is no `when:` value at all; the build rejects it before any typing.
    with pytest.raises(LoadError, match="each on:-form case needs a `when:` value"):
        load_flow(_inner_on_flow("null"))


def _exhaustive_on_flow(when: str) -> str:
    """A flow whose read CASE routes, under `when`, to an `on:` CASE over the tag
    `input.a.m` that lists every tag and has no `else:`; `input.a` is `Optional(A)`."""
    return (
        "id: p\nname: p\n"
        "typedefs:\n  A:\n    b: Optional(str)\n    m: Literal(p, q)\n"
        "input:\n  a: !Optional(A)\n"
        "nodes:\n"
        "  route:\n    kind: case\n    cases:\n"
        f'      - when: "{when}"\n        then: inner\n'
        "    else: e\n"
        "  inner:\n    kind: case\n    on: ${input.a.m}\n    cases:\n"
        "      - when: p\n        then: np\n"
        "      - when: q\n        then: nq\n"
        "  np:\n    kind: code\n    code: return 1\n"
        "  nq:\n    kind: code\n    code: return 2\n"
        "  e:\n    kind: code\n    code: return 3\n"
        "output: ${route.output}\n"
    )


@pytest.mark.parametrize("when", ["input.a.b != null", "input.a != null"])
def test_a_null_check_on_a_field_or_its_parent_makes_a_sibling_tag_exhaustive(when):
    # A non-null `input.a.b` means `input.a` is not null, so its required tag `m` is not.
    load_flow(_exhaustive_on_flow(when))


def test_a_field_of_an_unchecked_optional_parent_may_still_be_null():
    with pytest.raises(LoadError, match="\\$\\{input.a.m\\} may be null"):
        load_flow(_exhaustive_on_flow("input.a == null"))

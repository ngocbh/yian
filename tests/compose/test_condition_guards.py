"""Guards: what a condition proves, and what holds at each of its reads."""

import pytest

from yian.compose.helpers.condition_guards import (
    NEVER,
    NOTHING,
    AllOf,
    AnyOf,
    CheckLiteral,
    Holds,
    IsinstanceCheck,
    NotNullCheck,
    TagCheck,
    all_of,
    any_of,
    check_in,
    conjuncts,
    falsified,
    guard_checks,
    guarded_refs,
    literal_text,
    literals_in_context,
    masked,
    proves,
)
from yian.expr import condition_reads, expr_refs, parse_condition

A = IsinstanceCheck("v", "Buy")
B = TagCheck("v", "action", "sell")
C = IsinstanceCheck("w", "Done")

# The source of each of A, B and C as a condition.
_A, _B, _C = "isinstance(v, Buy)", "v.action == 'sell'", "isinstance(w, Done)"


def _p(text: str, value: bool = True):
    """What the condition `text` proves when it evaluates to `value`."""
    return proves(parse_condition(text), value)


def _guards(text: str) -> list:
    """Each reference occurrence of `text` with its guard."""
    return guarded_refs(parse_condition(text))


def _guard_of(text: str, ref: str):
    """The guard of the only occurrence of `ref` in `text`."""
    found = [guard for path, guard in _guards(text) if path == ref]
    assert len(found) == 1, found
    return found[0]


# --------------------------------------------------------------------------- #
# Guard constructors
# --------------------------------------------------------------------------- #


def test_constructors_flatten_and_drop_nothing():
    a, b, c = Holds(A, True), Holds(B, False), Holds(C, True)
    assert all_of(a, all_of(b, c)) == AllOf((a, b, c))
    assert all_of(a, NOTHING, b) == AllOf((a, b))
    assert all_of(a) == a
    assert all_of() == NOTHING
    assert any_of(a, any_of(b, c)) == AnyOf((a, b, c))
    assert any_of(a, NOTHING) == NOTHING
    assert any_of(a) == a


# --------------------------------------------------------------------------- #
# check_in
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("isinstance(d.output, Buy)", (IsinstanceCheck("d.output", "Buy"), True)),
        ("${isinstance(d.output, Buy)}", (IsinstanceCheck("d.output", "Buy"), True)),
        ('v.action == "buy"', (TagCheck("v", "action", "buy"), True)),
        ('"buy" == v.action', (TagCheck("v", "action", "buy"), True)),
        ("v.action != 'buy'", (TagCheck("v", "action", "buy"), False)),
        ("d.output.leg.type == 'limit'", (TagCheck("d.output.leg", "type", "limit"), True)),
        ("v.action == 'a\\b'", (TagCheck("v", "action", "a\\b"), True)),
        ("v.qty > 0", None),
        ('v == "x"', None),
        ("isinstance(f(x), Buy)", None),
        ("v.action == other.action", None),
        ("v.action == 1", None),
        ("v.action == null", (NotNullCheck("v.action"), False)),
        ("v.action == true", None),
        ("upper(v).action == 'buy'", None),
        ("not isinstance(v, Buy)", None),
        ("n > 1", None),
    ],
)
def test_check_in_recognizes_isinstance_tag_and_null_comparisons(text, expected):
    assert check_in(parse_condition(text)) == expected


# --------------------------------------------------------------------------- #
# proves
# --------------------------------------------------------------------------- #


def test_proves_a_check_and_its_negation():
    assert _p(_A) == Holds(A, True)
    assert _p(_A, False) == Holds(A, False)
    assert _p(f"not {_A}") == Holds(A, False)
    assert _p(f"not {_A}", False) == Holds(A, True)
    assert _p(f"not not {_A}") == Holds(A, True)
    assert _p("v.action != 'sell'") == Holds(B, False)
    assert _p("v.action != 'sell'", False) == Holds(B, True)


def test_proves_and():
    assert _p(f"{_A} and {_B}") == AllOf((Holds(A, True), Holds(B, True)))
    assert _p(f"{_A} and {_B}", False) == AnyOf(
        (Holds(A, False), AllOf((Holds(A, True), Holds(B, False))))
    )


def test_proves_or():
    assert _p(f"{_A} or {_B}") == AnyOf(
        (Holds(A, True), AllOf((Holds(A, False), Holds(B, True))))
    )
    assert _p(f"{_A} or {_B}", False) == AllOf((Holds(A, False), Holds(B, False)))


def test_proves_not_of_and_and_or():
    assert _p(f"not ({_A} and {_B})") == _p(f"{_A} and {_B}", False)
    assert _p(f"not ({_A} or {_B})") == AllOf((Holds(A, False), Holds(B, False)))


def test_proves_nested_groups():
    assert _p(f"({_A} or {_B}) and {_C}") == AllOf(
        (
            AnyOf((Holds(A, True), AllOf((Holds(A, False), Holds(B, True))))),
            Holds(C, True),
        )
    )
    assert _p(f"{_A} and ({_B} or {_C})") == AllOf(
        (
            Holds(A, True),
            AnyOf((Holds(B, True), AllOf((Holds(B, False), Holds(C, True))))),
        )
    )


def test_proves_a_chain():
    assert _p(f"{_A} and {_B} and {_C}") == AllOf(
        (Holds(A, True), Holds(B, True), Holds(C, True))
    )
    assert _p(f"{_A} and {_B} and {_C}", False) == AnyOf(
        (
            Holds(A, False),
            AllOf((Holds(A, True), Holds(B, False))),
            AllOf((Holds(A, True), Holds(B, True), Holds(C, False))),
        )
    )


def test_proves_nothing_for_other_expressions():
    assert _p("n > 1") == NOTHING
    assert _p("n > 1", False) == NOTHING
    assert _p(f"n > 1 and {_A}") == Holds(A, True)
    # `n > 1` may be what made the `or` true, so nothing is proven.
    assert _p(f"n > 1 or {_A}") == NOTHING
    assert _p(f"({_A}) :- false") == NOTHING
    assert _p("true") == NOTHING
    # A false `and` (a true `or`) may be decided by an operand that proves nothing.
    assert _p(f"{_A} and n > 1", False) == NOTHING
    assert _p(f"{_A} and n > 1 and {_B}", False) == NOTHING
    assert _p(f"not {_A} or n > 1") == NOTHING
    assert _p(f"{_A} and n > 1") == Holds(A, True)
    assert _p(f"not {_A} or n > 1", False) == Holds(A, True)


# --------------------------------------------------------------------------- #
# guarded_refs
# --------------------------------------------------------------------------- #


def test_a_read_after_an_and_check_is_guarded_by_it():
    assert _guard_of("isinstance(v, Buy) and v.qty > 0", "v.qty") == Holds(A, True)


def test_a_read_after_an_or_check_is_guarded_by_its_falsity():
    assert _guard_of("not isinstance(v, Buy) or v.qty > 0", "v.qty") == Holds(A, True)
    assert _guard_of('v.action != "sell" or v.qty > 0', "v.qty") == Holds(B, True)
    assert _guard_of("isinstance(v, Buy) or v.qty > 0", "v.qty") == Holds(A, False)


def test_a_read_after_an_and_with_a_non_check_is_unguarded():
    assert _guard_of("(isinstance(v, Buy) and n > 1) or v.qty > 0", "v.qty") == NOTHING
    assert _guard_of("(not isinstance(v, Buy) or n > 1) and v.qty > 0", "v.qty") == NOTHING


def test_a_check_after_the_read_does_not_guard_it():
    assert _guard_of("v.qty > 0 and isinstance(v, Buy)", "v.qty") == NOTHING


def test_a_non_check_before_the_read_proves_nothing():
    assert _guard_of("n > 5 or isinstance(r, Done) and r.answer == 'ok'", "r.answer") == (
        Holds(IsinstanceCheck("r", "Done"), True)
    )


def test_not_and_escapes_pass_the_context_down():
    assert _guard_of("not (isinstance(r, Done) and r.answer == 'ok')", "r.answer") == (
        Holds(IsinstanceCheck("r", "Done"), True)
    )
    assert _guard_of("(isinstance(v, Buy) and v.qty > 0) :- false", "v.qty") == Holds(A, True)


def test_every_read_inside_a_guarded_operand_is_guarded():
    pairs = _guards("isinstance(v, Buy) and (v.qty > 0 or v.qty < -1)")
    assert pairs == [
        ("v", NOTHING),
        ("v.qty", Holds(A, True)),
        ("v.qty", Holds(A, True)),
    ]


def test_a_builtin_argument_read_is_guarded():
    assert _guard_of('isinstance(v, Buy) and join(v.tags, ",") == "x"', "v.tags") == (
        Holds(A, True)
    )


def test_guards_of_a_nested_or_inside_an_and():
    pairs = _guards(f"{_A} and (v.action == 'sell' or v.qty > 0)")
    assert pairs[-1] == ("v.qty", AllOf((Holds(A, True), Holds(B, False))))


_ORDER_CASES = [
    "isinstance(v, Buy) and v.qty > 0",
    "not isinstance(v, Buy) or v.qty > 0",
    "n > 5 or isinstance(r, Done) and r.answer == 'ok'",
    "(isinstance(v, Buy) and v.qty > 0) :- false",
    "(a.output | b.output) == 2 and c.x > 1",
    "a.output :? 'need a' == 'x' or b",
    "v.action in [w.x, 'buy'] and not v.qty in [1, u.y]",
    'isinstance(v, Buy) and join(v.tags, ",") == upper(w.name)',
    "${isinstance(v, Buy) and lower(w.name) == v.action}",
    "not (isinstance(r, Done) and r.answer == 'ok') or f(k).z > m",
]


@pytest.mark.parametrize("text", _ORDER_CASES)
def test_guarded_refs_walks_the_reads_in_expr_refs_order(text):
    tree = parse_condition(text)
    paths = [path for path, _ in guarded_refs(tree)]
    assert paths == expr_refs(tree)
    assert paths == [read.path for read in condition_reads(tree)]


# --------------------------------------------------------------------------- #
# guard_checks / conjuncts
# --------------------------------------------------------------------------- #


def test_guard_checks_and_conjuncts():
    guard = _p(f"{_A} and ({_B} or {_C}) and {_A}")
    assert guard_checks(guard) == (A, B, C)
    assert guard_checks(NOTHING) == ()
    assert conjuncts(guard) == guard.parts
    assert conjuncts(Holds(A, True)) == (Holds(A, True),)
    assert conjuncts(NOTHING) == ()


# --------------------------------------------------------------------------- #
# literals_in_context
# --------------------------------------------------------------------------- #


def _literals(text: str) -> list[CheckLiteral]:
    """Each literal of `text` with its context."""
    return literals_in_context(parse_condition(text))


def test_a_literal_carries_the_context_before_it():
    literals = _literals("isinstance(v, Buy) and v.action == 'hold'")
    assert literals == [
        CheckLiteral(NOTHING, A, True, False),
        CheckLiteral(Holds(A, True), TagCheck("v", "action", "hold"), True, False),
    ]


def test_a_negated_check_is_one_literal():
    literals = _literals("not isinstance(v, Buy) or x")
    assert literals == [CheckLiteral(NOTHING, A, False, True)]
    assert _literals("not not isinstance(v, T)") == [
        CheckLiteral(NOTHING, IsinstanceCheck("v", "T"), True, False)
    ]
    assert _literals("v.action != 'buy'") == [
        CheckLiteral(NOTHING, TagCheck("v", "action", "buy"), False, False)
    ]


def test_literals_are_only_in_boolean_positions():
    assert _literals("(isinstance(v, Buy)) == true") == []
    assert _literals("upper(v.action == 'buy')") == []
    assert _literals("[isinstance(v, Buy)] == x") == []
    assert _literals("(v.action == 'buy') | false") == []
    assert _literals("(isinstance(v, Buy) and n > 1) :- false") == [
        CheckLiteral(NOTHING, A, True, False)
    ]
    assert _literals("isinstance(v, Buy) :? 'need v'") == [CheckLiteral(NOTHING, A, True, False)]


def test_literals_under_not_of_a_group_see_their_own_context():
    literals = _literals(f"not ({_A} and {_B})")
    assert literals == [
        CheckLiteral(NOTHING, A, True, False),
        CheckLiteral(Holds(A, True), B, True, False),
    ]


def test_literal_text_is_canonical():
    texts = [
        literal_text(literal)
        for literal in _literals(
            "not not isinstance( v , Buy ) and not v.action == 'hold' "
            "and v.action != 'a\"b' and 'sell' == v.action"
        )
    ]
    assert texts == [
        "isinstance(v, Buy)",
        'not v.action == "hold"',
        "v.action != 'a\"b'",
        'v.action == "sell"',
    ]


def test_masked_checks_prove_nothing_and_keep_the_reads():
    tree = parse_condition("isinstance(v, Buy) and not isinstance(w, Done) and v.qty > w.n")
    literal = literals_in_context(tree)[0]
    assert literal.node is not None
    copy = masked(tree, [literal.node])
    assert guarded_refs(copy) == [
        ("v", NOTHING),
        ("w", NOTHING),
        ("v.qty", Holds(C, False)),
        ("w.n", Holds(C, False)),
    ]
    assert [lit.check for lit in literals_in_context(copy)] == [C]
    # The original is unchanged.
    assert guarded_refs(tree)[2] == ("v.qty", all_of(Holds(A, True), Holds(C, False)))


def test_never_absorbs_an_all_of_and_drops_out_of_an_any_of():
    a, b = Holds(A, True), Holds(B, False)
    assert all_of(a, NEVER, b) == NEVER
    assert any_of(a, NEVER) == a
    assert any_of(NEVER, NEVER) == NEVER
    assert any_of(NEVER, NOTHING) == NOTHING


def test_falsified_checks_prove_never_for_the_value_they_never_have():
    tree = parse_condition("isinstance(v, Buy) and v.action != 'sell' and v.qty > 0")
    first, second = (literal.node for literal in literals_in_context(tree))
    reads = guarded_refs(falsified(tree, [(first, True)]))
    assert reads[-1] == ("v.qty", NEVER)
    # The other value proves nothing, and the reads keep their order.
    reads = guarded_refs(falsified(tree, [(first, False)]))
    assert [read for read, _ in reads] == [read for read, _ in guarded_refs(tree)]
    assert reads[-1] == ("v.qty", Holds(B, False))
    # A `!=` sub-tree that is never false: the read after it still runs.
    assert guarded_refs(falsified(tree, [(second, False)]))[-1] == ("v.qty", Holds(A, True))
    assert guarded_refs(falsified(tree, [(second, True)]))[-1] == ("v.qty", NEVER)

"""Operator chains are flat nodes that evaluate and trace like the old binary trees, at any
length."""

import io
import json
import logging
import random

import lark
import pytest
from lark import Lark, Token, Tree

from tests.engine.test_expr_eval import _eval, _from
from tests.engine.test_expr_lineage import Sources, messages
from yian.expr import grammar
from yian.expr.expressions import ExpressionError, ResolveMode
from yian.expr.grammar import parse_expr
from yian.expr.lineage import ValueRoute, _route_value, route_feedback
from yian.runtime.contracts import _validate_route
from yian.runtime.log import _route_from_payload, _route_payload


def _ref(name):
    return Tree("refcall", [Token("NAME", name)])


def _num(text):
    return Token("NUMBER", text)


def _add(symbol):
    return Token("ADD_OP", symbol)


def _mul(symbol):
    return Token("MUL_OP", symbol)


_POW = Token("POW_OP", "**")


# --------------------------------------------------------------------------- #
# Parse shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, expected",
    [
        ("a + b - c", Tree("sum", [_ref("a"), _add("+"), _ref("b"), _add("-"), _ref("c")])),
        (
            "a * b / c % d",
            Tree("product", [
                _ref("a"), _mul("*"), _ref("b"), _mul("/"), _ref("c"), _mul("%"), _ref("d"),
            ]),
        ),
        ("a ** b ** c", Tree("power", [_ref("a"), _POW, _ref("b"), _POW, _ref("c")])),
        (
            "a + b * c",
            Tree("sum", [_ref("a"), _add("+"), Tree("product", [_ref("b"), _mul("*"), _ref("c")])]),
        ),
        ("-a ** 2", Tree("neg", [Tree("power", [_ref("a"), _POW, _num("2")])])),
        (
            "2 ** -a ** b",
            Tree("power", [
                _num("2"), _POW, Tree("neg", [Tree("power", [_ref("a"), _POW, _ref("b")])]),
            ]),
        ),
        ("a - -b", Tree("sum", [_ref("a"), _add("-"), Tree("neg", [_ref("b")])])),
        ("a--b", Tree("sum", [_ref("a"), _add("-"), Tree("neg", [_ref("b")])])),
        ("1 -1", Tree("sum", [_num("1"), _add("-"), _num("1")])),
        ("a*-b", Tree("product", [_ref("a"), _mul("*"), Tree("neg", [_ref("b")])])),
        (
            "(a + b) + c",
            Tree("sum", [Tree("sum", [_ref("a"), _add("+"), _ref("b")]), _add("+"), _ref("c")]),
        ),
        (
            "a + (b + c)",
            Tree("sum", [_ref("a"), _add("+"), Tree("sum", [_ref("b"), _add("+"), _ref("c")])]),
        ),
        ("a/b", _ref("a/b")),
    ],
)
def test_chain_shapes(text, expected):
    assert parse_expr(text) == expected


def test_the_grammar_has_no_lalr_conflict():
    # Lark logs a resolved shift/reduce conflict at DEBUG while it builds the tables, so the
    # handler must be in place before a fresh parser is built (the module's is built at import).
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    level = lark.logger.level
    lark.logger.addHandler(handler)
    lark.logger.setLevel(logging.DEBUG)
    try:
        Lark(grammar._GRAMMAR, parser="lalr", maybe_placeholders=False)
    finally:
        lark.logger.removeHandler(handler)
        lark.logger.setLevel(level)
    assert "conflict" not in buffer.getvalue().lower()


# --------------------------------------------------------------------------- #
# Flat chains against their fully parenthesized (old binary) forms
# --------------------------------------------------------------------------- #

# A generated expression is a small tree: ("leaf", text, size), ("neg", child), or
# ("chain", level, operands, operators, grouped). `size` bounds a value's magnitude or length,
# so no generated repetition or power grows without limit.
_LEVELS = {"sum": ("+", "-"), "product": ("*", "/", "%"), "power": ("**",)}
_PRECEDENCE = {"sum": 1, "product": 2, "neg": 3, "power": 4, "leaf": 5}
_SIZE_LIMIT = 10_000

_POOL = {"x": 3, "s": "ab", "xs": [1, 2]}
_LEAVES = [
    ("0", 0), ("1", 1), ("2", 2), ("3", 3), ("0.5", 1), ("1.5", 2), ("2.0", 2),
    ("'a'", 1), ('"b"', 1), ("[]", 0), ("[1]", 1), ("['a', 2]", 2),
    ("x", 3), ("s", 2), ("xs", 2), ("m", 0),
]


def _leaf(text, size=1):
    return ("leaf", text, size)


def _neg(child):
    return ("neg", child)


def _chain(level, operands, operators, grouped=False):
    return ("chain", level, operands, operators, grouped)


def _precedence(node):
    if node[0] == "chain":
        return _PRECEDENCE["leaf"] if node[4] else _PRECEDENCE[node[1]]
    return _PRECEDENCE[node[0]]


def _flat(node):
    """The expression as an author writes it: one chain per precedence level."""
    if node[0] == "leaf":
        return node[1]
    if node[0] == "neg":
        child = node[1]
        text = _flat(child)
        return f"-({text})" if _precedence(child) < _PRECEDENCE["neg"] else f"-{text}"
    _, level, operands, operators, grouped = node
    parts = []
    for index, operand in enumerate(operands):
        text = _flat(operand)
        precedence = _precedence(operand)
        if level == "power":
            last_neg = index == len(operands) - 1 and operand[0] == "neg"
            needs = precedence < _PRECEDENCE["leaf"] and not last_neg
        else:
            needs = precedence <= _PRECEDENCE[level]
        parts.append(f"({text})" if needs else text)
    text = parts[0] + "".join(f" {op} {part}" for op, part in zip(operators, parts[1:]))
    return f"({text})" if grouped else text


def _binary(node):
    """The same expression with every operator step in its own parentheses."""
    if node[0] == "leaf":
        return node[1]
    if node[0] == "neg":
        return f"(-{_binary(node[1])})"
    _, level, operands, operators, _ = node
    if level == "power":
        text = _binary(operands[-1])
        for operand in reversed(operands[:-1]):
            text = f"({_binary(operand)} ** {text})"
        return text
    text = _binary(operands[0])
    for op, operand in zip(operators, operands[1:]):
        text = f"({text} {op} {_binary(operand)})"
    return text


def _size(node):
    """An upper bound on the magnitude or length of the node's value."""
    if node[0] == "leaf":
        return node[2]
    if node[0] == "neg":
        return _size(node[1])
    _, level, operands, _, _ = node
    sizes = [_size(operand) for operand in operands]
    if level == "sum":
        return sum(sizes)
    if level == "product":
        bound = 1
        for size in sizes:
            bound *= max(size, 2)
        return bound
    bound = max(sizes[-1], 1)
    for size in reversed(sizes[:-1]):
        if bound > 32:
            return float("inf")
        bound = max(size, 2) ** bound
    return bound


def _random_leaf(rng, leaves=_LEAVES):
    # Mostly numbers, so most chains evaluate; text, lists and the missing `m` make the errors.
    text, size = rng.choice(_LEAVES[:7] + [("x", 3)] if rng.random() < 0.93 else leaves)
    return _leaf(text, size)


def _random_node(rng, depth=0):
    if depth >= 2 or rng.random() < 0.3:
        leaf = _random_leaf(rng)
        return _neg(leaf) if rng.random() < 0.1 else leaf
    level = rng.choice(list(_LEVELS))
    if level == "power":
        count = rng.randint(2, 3)
        operands = [_random_leaf(rng, _LEAVES[:7] + _LEAVES[12:]) for _ in range(count)]
        if rng.random() < 0.3:
            operands[-1] = _neg(operands[-1])
    else:
        count = rng.randint(2, 8)
        operands = [_random_node(rng, depth + 1) for _ in range(count)]
    operators = [rng.choice(_LEVELS[level]) for _ in range(count - 1)]
    node = _chain(level, operands, operators, grouped=rng.random() < 0.15)
    return _neg(node) if rng.random() < 0.05 else node


def _random_expressions(count):
    rng = random.Random(610)
    nodes = []
    while len(nodes) < count:
        node = _random_node(rng)
        if node[0] == "chain" and _size(node) <= _SIZE_LIMIT:
            nodes.append(node)
    return nodes


_ZERO = _chain("sum", [_leaf("3"), _leaf("3")], ["-"])  # `3 - 3`
_DIVISION_BY_ZERO = [
    _chain("product", [_leaf("1"), _ZERO], ["/"]),
    _chain("product", [_leaf("x"), _leaf("2"), _ZERO], ["*", "%"]),
    _chain(
        "sum", [_leaf("1"), _leaf("2"), _chain("product", [_leaf("x"), _ZERO], ["/"])], ["+", "-"],
    ),
]


def _outcome(text, mode):
    resolve = _from(_POOL)
    try:
        return "value", _eval(text, resolve, mode=mode)
    except ExpressionError as exc:
        return "error", str(exc)


def test_chains_match_the_old_binary_steps():
    nodes = _random_expressions(400) + _DIVISION_BY_ZERO
    kinds = set()
    for node in nodes:
        flat, binary = _flat(node), _binary(node)
        for mode in (ResolveMode.BINDING_NONE, ResolveMode.STRICT_RAISE):
            outcome = _outcome(flat, mode)
            assert outcome == _outcome(binary, mode), (flat, binary, mode)
            kinds.add(outcome[0])
    assert kinds == {"value", "error"}
    assert _outcome(_flat(_DIVISION_BY_ZERO[0]), ResolveMode.STRICT_RAISE) == (
        "error", "division by zero: 1 / 0"
    )


# The traced pool: list content, scalar counts (negative, zero, bool), and text.
_TRACED = {"x": 3, "s": "ab", "xs": [1, 2], "ys": ["c"], "t": True, "q": 3, "n": -2}
_LIST_LEAVES = ["[x]", "xs", "ys", "[]", "[s, x]", "[1]"]
_COUNT_LEAVES = ["0", "1", "2", "3", "t", "q", "x", "n"]


def _random_count(rng):
    if rng.random() < 0.2:
        return _neg(_leaf(rng.choice(["1", "2"])))
    return _leaf(rng.choice(_COUNT_LEAVES))


def _random_repeat(rng):
    count = rng.randint(2, 6)
    at = rng.randrange(count)
    operands = [_random_count(rng) for _ in range(count)]
    operands[at] = _leaf(rng.choice(_LIST_LEAVES))
    operators = []
    for index in range(count - 1):
        if index < at - 1 and rng.random() < 0.5:
            operators.append(rng.choice(["%", "/"]))
        else:
            operators.append("*" if rng.random() < 0.95 else "%")
    return _chain("product", operands, operators)


def _random_concat(rng):
    count = rng.randint(2, 8)
    operands = [
        _random_repeat(rng) if rng.random() < 0.25 else _leaf(rng.choice(_LIST_LEAVES))
        for _ in range(count)
    ]
    return _chain("sum", operands, ["+"] * (count - 1))


def _traced_expressions():
    rng = random.Random(610)
    nodes = [_random_concat(rng) for _ in range(150)]
    nodes += [_random_repeat(rng) for _ in range(150)]
    nodes += [node for node in _random_expressions(400) if "m" not in _flat(node)][:100]
    return nodes


def _contributions(routed):
    return sorted(messages(routed), key=repr)


def _json_like(value):
    if isinstance(value, list):
        return all(_json_like(item) for item in value)
    return type(value) in (int, float, str, bool)


def test_traced_chains_match_the_old_binary_steps():
    sources = Sources(**_TRACED)
    shapes = set()
    for node in _traced_expressions():
        flat, binary = _flat(node), _binary(node)
        try:
            new = sources.evaluate(flat)
        except ExpressionError as exc:
            with pytest.raises(ExpressionError) as old_error:
                sources.evaluate(binary)
            assert str(old_error.value) == str(exc)
            continue
        old = sources.evaluate(binary)
        assert new.value == old.value, flat
        if not _json_like(new.value):
            continue
        shapes.add(new.route.kind)
        assert _route_value(new.route, sources.lookup) == new.value, flat
        feedbacks = ["Whole."]
        if isinstance(new.value, list):
            feedbacks.append([f"m{i}" for i in range(len(new.value))])
        for feedback in feedbacks:
            assert _contributions(sources.feedback(new, feedback)) == _contributions(
                sources.feedback(old, feedback)
            ), (flat, feedback)
        if new.value == []:
            assert sources.feedback(new, []).contributions == ()
            assert sources.feedback(old, []).contributions == ()
    assert {"concat", "repeat", "scalar"} <= shapes


# --------------------------------------------------------------------------- #
# Long chains, step errors and error precedence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "term, op, expected",
    [
        ("1", "+", 3000),
        ("1", "*", 1),
        ("2", "*", 2**3000),
        ("-1", "*", 1),
        ("1", "-", 2 - 3000),
        ("1", "**", 1),
        ("'a'", "+", "a" * 3000),
        ("[1]", "+", [1] * 3000),
        ("x", "+", 6000),
    ],
)
def test_long_chains_evaluate(term, op, expected):
    text = f" {op} ".join([term] * 3000)
    assert _eval(text, _from({"x": 2})) == expected


def test_a_long_power_chain_folds_from_the_right():
    # 2 ** (3 ** (1 ** ... ** 1)) is 8; a left fold or a dropped operand would differ.
    text = "2 ** 3 ** " + " ** ".join(["1"] * 2998)
    assert _eval(text, _from({})) == 8


@pytest.mark.parametrize(
    "text, message",
    [
        ("1 + 2 / 0", "division by zero: 2 / 0"),
        ("6 / 3 / 0", "division by zero: 2.0 / 0"),
        ("1 + 'a' + 2", "type error evaluating 1 + 'a'"),
        ("'a' + 'b' + 1", "type error evaluating 'ab' + 1"),
        ("2 ** 3 ** 1000.0", "cannot compute 3 ** 1000.0: Result too large"),
    ],
)
def test_step_errors_name_the_running_value(text, message):
    with pytest.raises(ExpressionError) as error:
        _eval(text)
    assert str(error.value).startswith(message)


@pytest.mark.parametrize(
    "text, message",
    [
        ("1 / 0 + c", "division by zero: 1 / 0"),
        ("c + 1 / 0", "unresolved reference ${c} in expression"),
        ("c ** (1 / 0)", "unresolved reference ${c} in expression"),
    ],
)
def test_error_precedence_keeps_left_to_right_order(text, message):
    with pytest.raises(ExpressionError) as error:
        _eval(text, mode=ResolveMode.STRICT_RAISE)
    assert str(error.value) == message


# --------------------------------------------------------------------------- #
# Traced routes keep the run-record format
# --------------------------------------------------------------------------- #


def _depth(route):
    return 1 + max((_depth(child) for child in route.children), default=-1)


def _round_trip(route):
    return _route_from_payload(json.loads(json.dumps(_route_payload(route))))


def test_a_list_chain_traces_as_balanced_two_child_concats():
    sources = Sources(left=["a"], right=["b", "c"])
    result = sources.evaluate("[] + right + []")
    route = result.route
    assert route.kind == "concat" and route.expression == "([] + right + [])"
    assert route.sizes == (0, 2)
    first, rest = route.children
    assert rest.kind == "concat" and rest.sizes == (2, 0)
    assert rest.expression == "(right + [])"
    assert [first, *rest.children] == [
        sources.evaluate("[]").route,
        sources.evaluate("right").route,
        sources.evaluate("[]").route,
    ]
    assert messages(sources.feedback(result, [None, "Bad."])) == [
        (sources.identities["right"], [None, "Bad."])
    ]
    assert messages(sources.feedback(result, "Too many.")) == [
        (sources.identities["right"], "Too many.")
    ]


def test_a_long_traced_list_chain_stays_shallow():
    sources = Sources(x="v")
    result = sources.evaluate(" + ".join(["[x]"] * 3000))
    route = result.route
    assert result.value == ["v"] * 3000
    assert _depth(route) <= 13
    assert _route_value(route, sources.lookup) == result.value
    _validate_route(route)
    assert _round_trip(route) == route
    feedback = [f"m{i}" for i in range(3000)]
    assert messages(sources.feedback(result, feedback)) == [
        (sources.identities["x"], f"m{i}") for i in range(3000)
    ]


@pytest.mark.parametrize("expression, content_index", [("[x] * 2 * 3", 0), ("2 * 3 * [x]", 1)])
def test_a_repeat_chain_traces_as_one_repeat(expression, content_index):
    sources = Sources(x="v")
    result = sources.evaluate(expression)
    route = result.route
    assert result.value == ["v"] * 6
    assert route.kind == "repeat"
    assert (route.content_index, route.count, route.sizes) == (content_index, 6, (1,))
    count = route.children[1 - content_index]
    assert count.kind == "scalar" and count.expression == "(2 * 3)"
    assert [child.value_json for child in count.children] == ["2", "3"]
    assert json.loads(count.value_json) == 6
    assert route.children[content_index].kind == "list"
    _validate_route(route)
    assert _round_trip(route) == route
    feedback = [None] * 6
    feedback[4] = "Bad."
    routed = sources.feedback(result, feedback)
    assert messages(routed) == [(sources.identities["x"], "Bad.")]
    context = routed.contributions[0].context
    assert next(c for c in context if c.role == "content").positions == ((4,),)
    assert next(c for c in context if c.role == "item").operand == 0


@pytest.mark.parametrize(
    "expression, value, count_text",
    [
        ("q % 2 * y", ["a", "b"], "(q % 2)"),
        ("t % 2 * x", ["a", "b"], "(t % 2)"),
        ("2 * 3 % 4 * x * 2", ["a", "b"] * 4, "(2 * 3 % 4 * 2)"),
        ("q % 2 * y * 3", ["a", "b"] * 3, "(q % 2 * 3)"),
    ],
)
def test_a_running_count_before_the_list_traces_as_one_repeat(expression, value, count_text):
    sources = Sources(q=3, t=True, x=["a", "b"], y=["a", "b"])
    result = sources.evaluate(expression)
    route = result.route
    assert result.value == value
    assert route.kind == "repeat" and route.content_index == 1
    assert route.children[0].expression == count_text
    assert route.children[1].kind == "source"
    assert _route_value(route, sources.lookup) == value
    feedback = [f"m{i}" for i in range(len(value))]
    assert len(sources.feedback(result, feedback).contributions) == len(value) // 2


@pytest.mark.parametrize(
    "expression, value, count",
    [
        ("[x] * -1 * -1", [], 0),
        ("-1 * [x] * -1", [], 0),
        ("-1 * -1 * [x]", ["v"], 1),
        ("[x] * 0 * 5", [], 0),
    ],
)
def test_repeat_counts_clamp_at_each_step(expression, value, count):
    sources = Sources(x="v")
    result = sources.evaluate(expression)
    assert result.value == value
    assert result.route.kind == "repeat" and result.route.count == count
    assert _route_value(result.route, sources.lookup) == value
    if value == []:
        assert route_feedback(result.route, [], sources.lookup).contributions == ()


def test_a_scalar_chain_traces_as_one_scalar():
    sources = Sources()
    route = sources.evaluate("1 + 2 - 3").route
    assert route.kind == "scalar" and len(route.children) == 3
    assert route.expression == "(1 + 2 - 3)"
    assert json.loads(route.value_json) == 0
    sources = Sources(left=3, right=4)
    assert sources.evaluate("left + right").route.expression == "(left + right)"


def test_an_old_nested_concat_route_still_validates_and_routes_feedback():
    sources = Sources(a=["a"], b=["b", "c"], c=["d"])

    def source(name):
        return ValueRoute("source", source=sources.identities[name], expression=name)

    inner = ValueRoute(
        "concat", children=(source("a"), source("b")), expression="(a + b)", sizes=(1, 2),
    )
    old = ValueRoute(
        "concat", children=(inner, source("c")), expression="((a + b) + c)", sizes=(3, 1),
    )
    _validate_route(old)
    assert _round_trip(old) == old
    assert _route_value(old, sources.lookup) == ["a", "b", "c", "d"]
    new = sources.evaluate("a + b + c")
    assert new.route != old
    for feedback in ("Whole.", ["1", None, "3", "4"]):
        assert _contributions(route_feedback(old, feedback, sources.lookup)) == _contributions(
            sources.feedback(new, feedback)
        )

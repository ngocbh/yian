"""`${T(...)}` constructors in the expression language.

`T` is a type name looked up in the node's type table. `T(x)` checks the value of `x`
against `T` and gives it back unchanged; `T(a=..., b=...)` builds a record from exactly
the given fields and checks it. A failed check raises `ConstructorTypeError`, which the
binder reports with failure code `binding.type`.
"""

import re

import pytest

from yian.expr.expressions import (
    ConstructorTypeError,
    ExpressionError,
    constructor_names,
    eval_expr,
    evaluate_when_record,
    expr_refs,
    infer_expr_type,
    rewrite_expr_refs,
)
from yian.expr.grammar import parse_expr
from yian.expr.lineage import ResolvedValue, SourceRef, route_feedback
from yian.expr.template import (
    binding_co_skips,
    eval_binding,
    flow_call_callees_in_spans,
    prompt_refs,
)
from yian.nodes.binding import BindingError, bind_params
from yian.nodes.contracts import Parameter, ValueTemplate
from yian.typesys.pool import VariablePool
from yian.typesys.values import Type, ValueKind


def _literal(*tags: str) -> Type:
    return Type(kind=ValueKind.STRING, tags=frozenset(tags))


BUY = Type(
    kind=ValueKind.OBJECT,
    fields={"action": _literal("buy"), "conviction": Type.scalar(ValueKind.NUMBER)},
    required=frozenset({"action", "conviction"}),
)
SELL = Type(
    kind=ValueKind.OBJECT,
    fields={"action": _literal("sell"), "price": Type.scalar(ValueKind.NUMBER)},
    required=frozenset({"action", "price"}),
)
TYPES = {
    "Buy": BUY,
    "Sell": SELL,
    "Decision": Type(kind=ValueKind.OBJECT, variants=(BUY, SELL)),
    "Size": _literal("s", "m", "l"),
}

A_BUY = {"action": "buy", "conviction": 0.5}
A_SELL = {"action": "sell", "price": 1.5}


def _resolver(**values):
    def resolve(path):
        head, *parts = path.split(".")
        value = values.get(head)
        for part in parts:
            value = value.get(part) if isinstance(value, dict) else None
        return value

    return resolve


def _eval(text, **values):
    return eval_expr(parse_expr(text), _resolver(**values), types=TYPES)


# --- call shape --- #


@pytest.mark.parametrize(
    "text",
    ["Buy(a, b)", "Buy(a, action=\"buy\")", "Buy(action=\"buy\", a)"],
)
def test_a_constructor_takes_keywords_or_one_value(text):
    parse_expr(text)  # the callee isn't known to be a constructor until it resolves
    with pytest.raises(
        ExpressionError,
        match=re.escape(
            "Buy(...) takes keyword fields, as in Buy(name=value, ...), or one value "
            "to check, as in Buy(x)"
        ),
    ):
        eval_expr(parse_expr(text), _resolver(a=A_BUY, b=A_BUY), types=TYPES)


@pytest.mark.parametrize("text", ["Buy(action=1, action=2)", "join(xs, sep=\",\", sep=\";\")"])
def test_a_call_repeating_a_keyword_is_rejected(text):
    with pytest.raises(ExpressionError, match=re.escape("repeats the keyword")):
        parse_expr(text)


def test_a_builtin_still_mixes_positional_and_keyword_arguments():
    assert eval_expr(parse_expr('join(xs, sep="-")'), _resolver(xs=["a", "b"])) == "a-b"


# --- positional form: check and narrow --- #


def test_a_value_that_fits_is_given_back_unchanged():
    value = _eval("Buy(d.output)", d={"output": A_BUY})
    assert value == A_BUY


def test_an_int_checked_as_a_float_field_stays_an_int():
    value = _eval("Buy(d.output)", d={"output": {"action": "buy", "conviction": 3}})
    assert type(value["conviction"]) is int


def test_a_value_missing_a_required_field_fails_naming_it():
    with pytest.raises(ConstructorTypeError, match=re.escape("Buy(...): record missing required fields: ['action']")):
        _eval("Buy(d.output)", d={"output": {"conviction": 0.5}})


def test_a_missing_value_fails_the_check():
    with pytest.raises(ConstructorTypeError, match=re.escape("Buy(...)")):
        _eval("Buy(d.output)")


def test_a_union_typedef_checks_against_every_member():
    assert _eval("Decision(d.output)", d={"output": A_SELL}) == A_SELL
    with pytest.raises(ConstructorTypeError):
        _eval("Decision(d.output)", d={"output": {"action": "hold"}})


def test_an_enum_alias_checks_a_scalar():
    assert _eval("Size(x)", x="m") == "m"
    with pytest.raises(ConstructorTypeError, match=re.escape("Size(...)")):
        _eval("Size(x)", x="xl")


def test_a_field_read_on_the_result():
    assert _eval("Buy(d.output).conviction", d={"output": A_BUY}) == 0.5


def test_a_constructor_error_is_an_expression_error():
    assert issubclass(ConstructorTypeError, ExpressionError)


# --- keyword form: build a record --- #


def test_keywords_build_a_record_from_exactly_the_given_fields():
    value = _eval('Buy(action="buy", conviction=s.output)', s={"output": 0.7})
    assert value == {"action": "buy", "conviction": 0.7}


def test_a_built_record_missing_a_field_fails_naming_it():
    with pytest.raises(ConstructorTypeError, match=re.escape("record missing required fields: ['action']")):
        _eval("Buy(conviction=s.output)", s={"output": 0.7})


def test_a_built_record_with_a_wrong_field_fails():
    with pytest.raises(ConstructorTypeError, match=re.escape("does not match declared type float")):
        _eval('Buy(action="buy", conviction="high")')


# --- inside larger expressions --- #


def test_constructors_in_a_list_literal():
    value = _eval("[Buy(a.output), Sell(b.output)]", a={"output": A_BUY}, b={"output": A_SELL})
    assert value == [A_BUY, A_SELL]


def test_a_constructor_in_a_condition():
    record = {"d": A_BUY}
    assert evaluate_when_record("Buy(d).conviction > 0.1", record, TYPES) is True
    with pytest.raises(ConstructorTypeError):
        evaluate_when_record("Sell(d).price > 0.1", record, TYPES)


def test_a_constructor_in_an_embedded_template_span():
    text = eval_binding(
        "conviction ${Buy(d.output).conviction}", _resolver(d={"output": A_BUY}), types=TYPES
    )
    assert text == "conviction 0.5"


# --- names that are not constructors --- #


@pytest.mark.parametrize("name", ["int", "str", "list", "Optional"])
def test_a_builtin_type_name_is_not_a_conversion(name):
    types = {**TYPES, name: Type.scalar(ValueKind.INTEGER)}
    with pytest.raises(
        ExpressionError,
        match=re.escape(
            f"unknown expression function {name!r}; types in expressions are typedef "
            f"names; `{name}(...)` is not a conversion"
        ),
    ):
        eval_expr(parse_expr(f"{name}(x)"), _resolver(x=1), types=types)


def test_a_name_absent_from_the_type_table_is_an_unknown_function():
    with pytest.raises(ExpressionError, match=re.escape("unknown expression function 'Hold'")):
        _eval("Hold(x)", x=1)


# --- walkers --- #


def test_reference_walks_read_every_argument_and_skip_the_names():
    tree = parse_expr('Buy(action="buy", conviction=s.output) == Buy(d.output)')
    assert expr_refs(tree) == ["s.output", "d.output"]


def test_rewrites_leave_the_type_and_field_names_alone():
    renamed = rewrite_expr_refs(
        "Buy(action=a, conviction=s.output).action", lambda path: "x." + path
    )
    assert renamed == "Buy(action=x.a, conviction=x.s.output).action"


def test_a_positional_constructor_co_skips_like_its_argument():
    assert binding_co_skips("${Buy(d.output)}") is True
    assert binding_co_skips("${Buy(d.output).conviction}") is True
    assert binding_co_skips('${Buy(action="buy", conviction=d.output)}') is False
    assert binding_co_skips("${upper(d.output)}") is False


def test_constructor_names_are_the_callees_that_are_no_builtin():
    tree = parse_expr("[Buy(d.output), upper(x), isinstance(y, Sell), Sell(price=p), Buy(z)]")
    assert constructor_names(tree) == ["Buy", "Sell"]
    assert constructor_names(parse_expr("x")) == []


def test_a_typedef_name_is_not_a_flow_callee():
    text = "${Buy(d.output)} ${review(d.output)}"
    assert flow_call_callees_in_spans(text) == ["Buy", "review"]
    assert flow_call_callees_in_spans(text, frozenset({"Buy"})) == ["review"]


def test_a_constructor_is_not_a_prompt_function():
    with pytest.raises(
        ExpressionError,
        match=re.escape(
            "unknown prompt function 'Buy'; prompts call only formatting builtins "
            "(join, lower, render_as_json, upper); build or check a typed value with a "
            "`typedefs:` constructor `T(...)` in an input binding"
        ),
    ):
        prompt_refs("Order: ${Buy(order)}")


def test_a_builtin_type_name_in_a_prompt_is_not_a_conversion():
    with pytest.raises(
        ExpressionError,
        match=re.escape(
            "unknown prompt function 'int'; types in expressions are typedef names; "
            "`int(...)` is not a conversion"
        ),
    ):
        prompt_refs("Count: ${int(order)}")


def test_a_prompt_call_with_several_arguments_is_an_unknown_prompt_function():
    with pytest.raises(ExpressionError, match=re.escape("unknown prompt function 'foo';")):
        prompt_refs("${foo(a, b)}")


# --- lineage --- #


def _traced_resolver(**values):
    def resolve(path):
        head, *parts = path.split(".")
        value = values.get(head)
        for part in parts:
            value = value.get(part) if isinstance(value, dict) else None
        return ResolvedValue(value, value is not None, SourceRef("run", 1, (head, *parts)))

    return resolve


def test_a_positional_constructor_routes_to_its_argument():
    traced = eval_binding(
        "${Buy(d.output)}", _traced_resolver(d={"output": A_BUY}), trace=True, types=TYPES
    )
    assert traced.value == A_BUY
    route = traced.route
    assert route.kind == "selected"
    assert route.expression == "Buy(d.output)"
    assert route.children[0].source == SourceRef("run", 1, ("d", "output"))


def test_a_keyword_constructor_routes_each_field():
    traced = eval_binding(
        '${Buy(action="buy", conviction=s.output)}',
        _traced_resolver(s={"output": 0.7}),
        trace=True,
        types=TYPES,
    )
    route = traced.route
    assert route.kind == "record"
    assert route.names == ("action", "conviction")
    assert route.expression == 'Buy(action="buy", conviction=s.output)'
    assert route.children[1].source == SourceRef("run", 1, ("s", "output"))


def test_a_field_read_on_a_constructor_projects_its_route():
    traced = eval_binding(
        "${Buy(d.output).conviction}",
        _traced_resolver(d={"output": A_BUY}),
        trace=True,
        types=TYPES,
    )
    assert traced.value == 0.5
    assert traced.route.kind == "project"
    assert traced.route.path == ("conviction",)


# --- the binder --- #


def _pool(**outputs):
    pool = VariablePool()
    for node_id, value in outputs.items():
        pool.set(node_id, value)
    return pool


def test_the_binder_reports_a_failed_constructor_as_binding_type():
    with pytest.raises(BindingError) as caught:
        bind_params(
            [Parameter("x")],
            {"x": "${Buy(d.output)}"},
            _pool(d={"conviction": 0.5}),
            types=TYPES,
        )
    assert caught.value.failure_code == "binding.type"
    assert "record missing required fields: ['action']" in str(caught.value)


def test_the_binder_reports_a_failed_nested_constructor_as_binding_type():
    with pytest.raises(BindingError) as caught:
        bind_params(
            [Parameter("x")],
            {"x": ValueTemplate({"d": "${Buy(d.output)}"})},
            _pool(d={"conviction": 0.5}),
            types=TYPES,
        )
    assert caught.value.failure_code == "binding.type"
    assert "binding 'x.d'" in str(caught.value)


def test_the_binder_delivers_a_constructed_record():
    record = bind_params(
        [Parameter("x")],
        {"x": '${Buy(action="buy", conviction=d.output)}'},
        _pool(d=0.25),
        types=TYPES,
    )
    assert record == {"x": {"action": "buy", "conviction": 0.25}}


# --- static inference --- #


def _type_fn(name):
    return TYPES.get(name)


def _no_refs(path):
    return None


def test_inference_gives_a_constructor_its_type():
    assert infer_expr_type(parse_expr("Buy(d.output)"), _no_refs, _type_fn) == BUY
    assert infer_expr_type(parse_expr('Buy(action="buy", conviction=1)'), _no_refs, _type_fn) == BUY


def test_inference_reads_fields_of_a_constructor_result():
    tree = parse_expr("Buy(d.output).conviction")
    assert infer_expr_type(tree, _no_refs, _type_fn) == Type.scalar(ValueKind.NUMBER)
    assert infer_expr_type(parse_expr("Buy(d.output).size"), _no_refs, _type_fn) is None


def test_inference_reads_a_field_every_union_member_has():
    inferred = infer_expr_type(parse_expr("Decision(d.output).action"), _no_refs, _type_fn)
    assert inferred == _literal("buy", "sell")


def test_inference_without_types_leaves_a_call_unknown():
    assert infer_expr_type(parse_expr("Buy(d.output)"), _no_refs) is None
    assert infer_expr_type(parse_expr("upper(x)"), _no_refs, _type_fn) is None


def test_feedback_on_a_built_field_reaches_its_argument():
    sources = {SourceRef("run", 1): {"output": 0.7}, SourceRef("run", 2): {"output": A_BUY}}

    def resolve(path):
        head, *parts = path.split(".")
        seq = {"s": 1, "d": 2}[head]
        value = sources[SourceRef("run", seq)]
        for part in parts:
            value = value[part]
        return ResolvedValue(value, True, SourceRef("run", seq, tuple(parts)))

    built = eval_binding(
        '${Buy(action="buy", conviction=s.output)}', resolve, trace=True, types=TYPES
    )
    routed = route_feedback(built.route, {"conviction": "too high"}, sources.__getitem__)
    assert [(c.source, c.feedback) for c in routed.contributions] == [
        (SourceRef("run", 1), {"output": "too high"})
    ]
    checked = eval_binding("${Buy(d.output)}", resolve, trace=True, types=TYPES)
    routed = route_feedback(checked.route, {"conviction": "too high"}, sources.__getitem__)
    assert [(c.source, c.feedback) for c in routed.contributions] == [
        (SourceRef("run", 2), {"output": {"conviction": "too high"}})
    ]

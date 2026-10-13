"""`isinstance(x, T)` in the expression language.

`isinstance` is a special form: the first argument is an expression, the second a
bare type name looked up in the node's type table (`types=`), never a reference.
"""

import json
from dataclasses import replace

import pytest

from yian.compose.loader import load_flow
from yian.expr.expressions import (
    ExpressionError,
    ResolveMode,
    condition_refs,
    eval_expr,
    evaluate_when,
    evaluate_when_record,
    expr_refs,
    infer_expr_type,
    rewrite_condition_refs,
    rewrite_expr_refs,
)
from yian.expr.grammar import parse_expr
from yian.expr.lineage import ResolvedValue, SourceRef
from yian.expr.template import (
    binding_co_skips,
    eval_binding,
    eval_template,
    expr_refs_of,
    flow_call_callees_in_spans,
    prompt_refs,
    rewrite_template_refs,
    scan_template,
)
from yian.nodes.case import Case, CaseNode
from yian.nodes.case.node import DEFAULT_HANDLE
from yian.run import resume_flow, run_flow
from yian.typesys.pool import VariablePool
from yian.typesys.values import Type, ValueKind


def _member(tag: str, **fields: ValueKind) -> Type:
    """Build a closed record tagged by a single-value `action` Literal."""
    record_fields = {"action": Type(kind=ValueKind.STRING, tags=frozenset({tag}))}
    record_fields.update({name: Type.scalar(kind) for name, kind in fields.items()})
    return Type(kind=ValueKind.OBJECT, fields=record_fields, required=frozenset(record_fields))


BUY = _member("buy", qty=ValueKind.INTEGER)
SELL = _member("sell", price=ValueKind.NUMBER)
TYPES = {"Buy": BUY, "Sell": SELL, "Order": Type(kind=ValueKind.OBJECT, variants=(BUY, SELL))}

A_BUY = {"action": "buy", "qty": 3}
A_SELL = {"action": "sell", "price": 1.5}


def _resolver(**values):
    def resolve(path):
        head, *parts = path.split(".")
        value = values.get(head)
        for part in parts:
            value = value.get(part) if isinstance(value, dict) else None
        return value

    return resolve


# --- evaluation --- #


def test_eval_expr_checks_the_value_against_the_named_type():
    resolve = _resolver(d={"output": A_BUY})
    assert eval_expr(parse_expr("isinstance(d.output, Buy)"), resolve, types=TYPES) is True
    assert eval_expr(parse_expr("isinstance(d.output, Sell)"), resolve, types=TYPES) is False
    assert eval_expr(parse_expr("isinstance(d.output, Order)"), resolve, types=TYPES) is True


def test_a_missing_value_is_not_an_instance_of_a_record():
    assert eval_expr(parse_expr("isinstance(d.output, Buy)"), _resolver(), types=TYPES) is False


def test_isinstance_composes_with_other_operators():
    resolve = _resolver(d={"output": A_BUY})
    tree = parse_expr("isinstance(d.output, Buy) and d.output.qty > 2")
    assert eval_expr(tree, resolve, types=TYPES) is True
    tree = parse_expr("not isinstance(d.output, Buy)")
    assert eval_expr(tree, resolve, types=TYPES) is False


def test_evaluate_when_record_uses_the_type_table():
    assert evaluate_when_record("isinstance(order, Sell)", {"order": A_SELL}, types=TYPES)
    assert not evaluate_when_record("${isinstance(order, Buy)}", {"order": A_SELL}, types=TYPES)


def test_evaluate_when_uses_the_type_table():
    pool = VariablePool()
    pool.set("d", A_SELL)
    assert evaluate_when("isinstance(d.output, Sell)", pool, types=TYPES)
    assert not evaluate_when("isinstance(d.output, Buy)", pool, types=TYPES)


def test_eval_binding_and_eval_template_use_the_type_table():
    resolve = _resolver(d={"output": A_BUY})
    assert eval_binding("${isinstance(d.output, Buy)}", resolve, types=TYPES) is True
    assert eval_binding("buy? ${isinstance(d.output, Sell)}", resolve, types=TYPES) == "buy? False"
    segments = scan_template("${isinstance(d.output, Buy)}")
    assert eval_template(segments, resolve, types=TYPES) is True


def test_scalar_type_names_check_scalars():
    types = {"int": Type.scalar(ValueKind.INTEGER), "str": Type.scalar(ValueKind.STRING)}
    resolve = _resolver(n=3)
    assert eval_expr(parse_expr("isinstance(n, int)"), resolve, types=types) is True
    assert eval_expr(parse_expr("isinstance(n, str)"), resolve, types=types) is False


@pytest.mark.parametrize("types", [None, {}, {"Sell": SELL}])
def test_an_unknown_type_name_is_an_error_naming_it(types):
    with pytest.raises(ExpressionError, match="'Buy'"):
        eval_expr(parse_expr("isinstance(d.output, Buy)"), _resolver(d={"output": A_BUY}), types=types)


def test_record_evaluation_without_types_names_the_type():
    with pytest.raises(ExpressionError, match="'Buy'"):
        evaluate_when_record("isinstance(order, Buy)", {"order": A_BUY})


def test_a_missing_value_raises_under_strict_raise():
    with pytest.raises(ExpressionError, match="unresolved reference"):
        eval_expr(
            parse_expr("isinstance(d.output, Buy)"), _resolver(),
            mode=ResolveMode.STRICT_RAISE, types=TYPES,
        )


def test_an_optional_alias_accepts_a_missing_value():
    types = {"MaybeBuy": replace(BUY, nullable=True)}
    tree = parse_expr("isinstance(d.output, MaybeBuy)")
    assert eval_expr(tree, _resolver(), types=types) is True
    assert eval_expr(tree, _resolver(d={"output": A_BUY}), types=types) is True
    assert eval_expr(tree, _resolver(d={"output": A_SELL}), types=types) is False


# --- parsing --- #


@pytest.mark.parametrize("text", [
    "isinstance(x)",
    "isinstance()",
    "isinstance(x, Buy, Sell)",
])
def test_isinstance_takes_exactly_two_arguments(text):
    with pytest.raises(ExpressionError, match="isinstance"):
        parse_expr(text)


@pytest.mark.parametrize("text", [
    'isinstance(x, "Buy")',
    "isinstance(x, a.b)",
    "isinstance(x, ${Buy})",
    "isinstance(x, f(Buy))",
    "isinstance(x, 3)",
    "isinstance(x, [Buy])",
    "isinstance(x, in)",
])
def test_the_second_argument_must_be_a_bare_type_name(text):
    with pytest.raises(ExpressionError, match="type name"):
        parse_expr(text)


@pytest.mark.parametrize("text", [
    "isinstance(value=x, type=Buy)",
    "isinstance(x, type=Buy)",
])
def test_isinstance_has_no_keyword_form(text):
    with pytest.raises(ExpressionError, match="keyword"):
        parse_expr(text)


def test_isinstance_result_has_no_fields_to_read():
    with pytest.raises(ExpressionError, match="gives a bool"):
        parse_expr("isinstance(x, Buy).ok")


def test_isinstance_nested_in_a_larger_expression_is_validated():
    with pytest.raises(ExpressionError, match="isinstance"):
        parse_expr("a > 1 and isinstance(x)")
    with pytest.raises(ExpressionError, match="isinstance"):
        parse_expr("upper(isinstance(x, a.b))")


# --- references and rewrites --- #


def test_reference_walks_skip_the_type_name():
    assert expr_refs(parse_expr("isinstance(d.output, Buy)")) == ["d.output"]
    assert condition_refs("isinstance(d.output, Buy) and d.output.qty > 1") == [
        "d.output", "d.output.qty",
    ]
    assert condition_refs("${isinstance(d.output, Buy)}") == ["d.output"]
    assert expr_refs_of("ok: ${isinstance(d.output, Buy)}") == ["d.output"]


def test_rewrites_leave_the_type_name_alone():
    def rename(path):
        return "x." + path

    assert rewrite_expr_refs("isinstance(d.output, Buy)", rename) == "isinstance(x.d.output, Buy)"
    assert rewrite_condition_refs("${isinstance(d.output, Buy)}", rename) == (
        "${isinstance(x.d.output, Buy)}"
    )
    assert rewrite_template_refs("ok ${isinstance(d.output, Buy)}", rename) == (
        "ok ${isinstance(x.d.output, Buy)}"
    )


# --- static checks --- #


def test_infer_expr_type_gives_bool_without_looking_up_the_type_name():
    seen = []

    def ref_type(path):
        seen.append(path)
        return None

    inferred = infer_expr_type(parse_expr("isinstance(d.output, Buy)"), ref_type)
    assert inferred == Type.scalar(ValueKind.BOOLEAN)
    assert "Buy" not in seen


def test_isinstance_binding_does_not_co_skip():
    assert binding_co_skips("${isinstance(d.output, Buy)}") is False
    assert binding_co_skips("${upper(d.output)}") is False


# --- callee and prompt handling --- #


def test_isinstance_is_not_a_flow_callee():
    assert flow_call_callees_in_spans("${isinstance(x, Buy)}") == []
    assert flow_call_callees_in_spans("${isinstance(x, Buy) and flow(a=1)}") == ["flow"]


def test_isinstance_is_not_a_prompt_function():
    with pytest.raises(ExpressionError, match="isinstance is not a prompt function"):
        prompt_refs("${isinstance(x, Buy)}")
    with pytest.raises(ExpressionError, match="compute it in an input binding"):
        prompt_refs("Order: ${render_as_json(x)} ${upper(isinstance(x, Buy))}")


# --- tracing --- #


def test_traced_isinstance_records_one_route_over_the_first_argument():
    def resolve(path):
        head, *parts = path.split(".")
        return ResolvedValue(A_BUY, True, SourceRef("run", 0, tuple(parts)))

    traced = eval_binding("${isinstance(d.output, Buy)}", resolve, trace=True, types=TYPES)
    assert traced.value is True
    route = traced.route
    assert route.kind == "opaque"
    assert route.expression == "isinstance(d.output, Buy)"
    assert len(route.children) == 1
    assert route.names == ("",)
    child = route.children[0]
    assert child.kind == "source"
    assert child.source == SourceRef("run", 0, ("output",))


# --- nodes carry the type table --- #

# Compose fills `Node.types` from the names each node's expressions use; these tests also
# set it by hand on loaded flows and directly built nodes.

ORDER_FLOW = """
id: orders
name: orders
typedefs:
  Buy: {action: 'Literal[buy]', qty: int}
  Sell: {action: 'Literal[sell]', price: float}
  Order: Union[Buy, Sell]
input:
  order: object
nodes:
  d:
    kind: code
    input:
      order: ${input.order}
    code: |
      return inputs["order"]
  check:
    kind: code
    input:
      ok: ${isinstance(d.output, Buy)}
    code: |
      return inputs["ok"]
  route:
    kind: case
    cases:
      - when: isinstance(d.output, Sell)
        then: s
    else: b
  s:
    kind: code
    code: |
      return "sell"
  b:
    kind: code
    code: |
      return "buy"
output:
  ok: ${check.output}
  side: ${s.output | b.output}
"""


def _orders_flow():
    flow = load_flow(ORDER_FLOW)
    flow.compiled.nodes["check"].types = dict(TYPES)
    flow.compiled.nodes["route"].types = dict(TYPES)
    return flow


def test_a_node_starts_with_an_empty_type_table():
    assert CaseNode("route", [Case("s", "isinstance(order, Sell)")]).types == {}
    assert load_flow(ORDER_FLOW).compiled.nodes["d"].types == {}


@pytest.mark.parametrize("order, expected", [
    (A_BUY, {"ok": True, "side": "buy"}),
    (A_SELL, {"ok": False, "side": "sell"}),
])
def test_bindings_and_case_conditions_use_the_node_type_table(order, expected):
    result = run_flow(_orders_flow(), {"order": order})
    assert result.status == "succeeded", result.failure
    assert result.output == expected


def test_a_node_without_the_type_name_fails_naming_it():
    flow = _orders_flow()
    flow.compiled.nodes["check"].types = {}
    result = run_flow(flow, {"order": A_BUY})
    assert result.status == "failed"
    assert result.failure.node_id == "check"
    assert "'Buy'" in result.failure.message


def test_case_forward_routes_with_its_type_table():
    node = CaseNode("route", [Case("s", "isinstance(order, Sell)")])
    node.types = dict(TYPES)
    assert node.forward({"order": A_SELL}).handle == "s"
    assert node.forward({"order": A_BUY}).handle == DEFAULT_HANDLE


ASSERT_FLOW = """
id: asserted
name: asserted
typedefs:
  Buy: {action: 'Literal[buy]', qty: int}
  Sell: {action: 'Literal[sell]', price: float}
  Order: Union[Buy, Sell]
input:
  order: object
nodes:
  step:
    kind: code
    input:
      order: ${input.order}
    code: |
      return inputs["order"]
    asserts:
      - isinstance(order, Buy)
      - isinstance(output, Order)
output: ${step.output}
"""


@pytest.mark.parametrize("order, status", [(A_BUY, "succeeded"), (A_SELL, "failed")])
def test_node_asserts_use_the_node_type_table(order, status):
    flow = load_flow(ASSERT_FLOW)
    flow.compiled.nodes["step"].types = dict(TYPES)
    result = run_flow(flow, {"order": order})
    assert result.status == status, result.failure
    if status == "failed":
        assert result.failure.code == "assertion.failed"


LOOP_FLOW = """
id: counter
name: counter
typedefs:
  Running: Literal[running]
defs:
  bump:
    input:
      n: int
      status: str
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
          status: ${input.status}
        code: |
          n = inputs["n"] + 1
          return {"n": n, "status": "done" if n >= 4 else "running"}
        output:
          n: int
          status: str
    output: ${step.output}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: 0
      status: running
    while: isinstance(status, Running)
    max: 10
output: ${loop.output}
"""

LOOP_TYPES = {"Running": Type(kind=ValueKind.STRING, tags=frozenset({"running"}))}


def _loop_flow():
    flow = load_flow(LOOP_FLOW)
    flow.compiled.nodes["loop"].types = dict(LOOP_TYPES)
    return flow


def test_loop_respawn_copies_the_type_table():
    driver = _loop_flow().compiled.nodes["loop"]
    assert driver.respawn(1).types == LOOP_TYPES


def test_loop_while_uses_the_type_table_across_iterations():
    result = run_flow(_loop_flow(), {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 4, "status": "done"}


def test_loop_with_a_type_table_resumes_mid_loop(tmp_path):
    log = tmp_path / "run.jsonl"
    first = run_flow(_loop_flow(), {}, log=log)
    assert first.status == "succeeded", first.failure
    # Cut after the second iteration, so resume replays two iterations and runs two live.
    kept = _cut_log_after(log, "loop#1/__end__")
    assert "loop~1" in kept
    assert not any(node_id and node_id.startswith(("loop~2", "loop#2/")) for node_id in kept)
    resumed = resume_flow(_loop_flow(), log=log)
    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == first.output


def _cut_log_after(log, node_id: str) -> list[str]:
    """Keep the log up to `node_id`'s success record; return the kept records' node ids."""
    lines = log.read_text().splitlines(keepends=True)
    records = [json.loads(line) for line in lines]
    end = next(
        index for index, record in enumerate(records)
        if record["kind"] == "node_succeeded" and record["payload"]["node_id"] == node_id
    )
    log.write_text("".join(lines[: end + 1]))
    return [record["payload"].get("node_id") for record in records[: end + 1]]

"""Ints of any size flow through lineage routes, expressions and input coercion."""

from __future__ import annotations

import sys

import pytest

import yian.json_codec as json_codec
from tests.engine._graph_builder import _engine
from yian import load_flow, run_flow
from yian.events import RunSucceeded
from yian.expr.expressions import _parse_literal
from yian.expr.lineage import SourceRef, ValueRoute, route_feedback
from yian.nodes.binding import coerce_param
from yian.nodes.contracts import Parameter, ValueContract
from yian.runtime.contracts import _validate_route
from yian.typesys.values import Type, ValueKind


BIG = 10**4999 + 12345  # 5000 digits, past Python's default 4300-digit limit
DIGITS = json_codec.int_text(BIG)
ONES = (10**5000 - 1) // 9  # the int spelled by 5000 ones


@pytest.fixture(autouse=True)
def _default_int_digit_limit():
    """Run each test under Python's default int/str digit limit."""
    get = getattr(sys, "get_int_max_str_digits", None)
    if get is None:
        yield  # this Python has no limit
        return
    before = get()
    sys.set_int_max_str_digits(4300)
    try:
        yield
        after = get()
    finally:
        sys.set_int_max_str_digits(before)
    assert after == 4300, "yian code changed the global digit limit"


_INCREMENT_FLOW = """
id: big-increment
name: big_increment
input:
  x: !int
nodes:
  inc:
    kind: code
    input:
      x: ${input.x + 1}
    code: |
      return inputs["x"]
output: ${inc.output}
"""

_RENDER_FLOW = """
id: big-render
name: big_render
input:
  xs: !list(int)
nodes:
  show:
    kind: code
    input:
      text: ${render_as_json(input.xs)}
    code: |
      return inputs["text"]
output: ${show.output}
"""

_LITERAL_FLOW = f"""
id: big-literal
name: big_literal
nodes:
  lit:
    kind: code
    input:
      n: ${{{"1" * 5000} + 1}}
    code: |
      return inputs["n"]
output: ${{lit.output}}
"""

_COERCE_FLOW = """
id: big-coerce
name: big_coerce
input:
  n: !int
nodes:
  echo:
    kind: code
    input:
      n: ${input.n}
    code: |
      return inputs["n"]
output: ${echo.output}
"""


def test_a_big_int_literal_route_keeps_its_value():
    route = ValueRoute.literal(BIG)

    assert route.kind == "literal"
    assert route.result_type is None
    assert route.value_json == DIGITS
    assert json_codec.loads(route.value_json) == BIG
    _validate_route(route)


def test_a_nested_big_int_literal_route_keeps_its_value():
    value = {"a": [BIG]}

    route = ValueRoute.literal(value)

    assert route.kind == "literal"
    assert route.result_type is None
    assert route.value_json == '{"a": [' + DIGITS + "]}"
    assert json_codec.loads(route.value_json) == value
    _validate_route(route)


def test_feedback_context_keeps_a_big_int_literal_result():
    route = ValueRoute(
        "list",
        children=(
            ValueRoute("source", source=SourceRef("r", 1), expression="${a.output}"),
            ValueRoute.literal(BIG),
        ),
        expression="[${a.output}, BIG]",
    )

    routed = route_feedback(route, ["bad", "too big"], lambda ref: "x")

    (contribution,) = routed.contributions
    assert contribution.source == SourceRef("r", 1)
    assert contribution.feedback == "bad"
    item = contribution.context[0]
    assert item.role == "item"
    assert item.result == ["x", BIG]


def test_a_learnable_run_records_the_big_int_binding_route():
    flow = load_flow(_INCREMENT_FLOW).compiled
    engine = _engine(flow, {"x": BIG}, learnable=True)

    list(engine.run())

    assert isinstance(engine.state.result, RunSucceeded), engine.state.result
    occurrences = {item.node_id: item for item in engine.state.trace.occurrences}
    capture = occurrences["inc"].record.forward_capture
    assert capture.bound_inputs == {"x": BIG + 1}
    route = dict(zip(capture.route.names, capture.route.children))["x"]
    assert route.kind in ("literal", "scalar")
    assert route.result_type is None
    assert json_codec.loads(route.value_json) == BIG + 1


def test_a_learnable_run_of_a_big_int_expression_succeeds():
    result = run_flow(load_flow(_INCREMENT_FLOW), {"x": BIG}, learnable=True)

    assert result.status == "succeeded", result.failure
    assert result.output == BIG + 1


def test_render_as_json_writes_a_big_int_in_full():
    result = run_flow(load_flow(_RENDER_FLOW), {"xs": [BIG]})

    assert result.status == "succeeded", result.failure
    assert DIGITS in result.output


def test_coerce_param_reads_a_big_int_string():
    parameter = Parameter(
        "n", contract=ValueContract(Type.scalar(ValueKind.INTEGER)), coerce_strings=True
    )

    assert coerce_param(parameter, DIGITS) == BIG


def test_a_supplied_big_int_string_is_coerced_to_the_int():
    result = run_flow(load_flow(_COERCE_FLOW), {"n": DIGITS})

    assert result.status == "succeeded", result.failure
    assert result.output == BIG


def test_a_big_int_expression_literal_evaluates_to_the_int():
    result = run_flow(load_flow(_LITERAL_FLOW), {})

    assert result.status == "succeeded", result.failure
    assert result.output == ONES + 1


def test_parse_literal_reads_a_big_int():
    assert _parse_literal("1" * 5000) == ONES

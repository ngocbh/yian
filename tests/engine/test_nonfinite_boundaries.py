"""NaN, infinity, and float overflow fail at the first boundary that sees them."""

import math

import pytest

import yian.nodes.agent.node as agent_module
from yian import load_flow, run_flow
from yian.events import NodeStarted
from yian.nodes.base import Output
from yian.nodes.binding import BindingError, apply_defaults
from yian.nodes.contracts import Parameter, ValueContract
from yian.typesys import Type, ValueKind


def _code_flow(output: str, value: str) -> str:
    """Return a flow whose `n` CODE node declares `output` and returns `value`."""
    return f"""
id: overflow
name: overflow
nodes:
  n:
    kind: code
    output: {output}
    code: |
      return {value}
output: ${{n.output}}
"""


def test_a_float_output_too_large_for_float_fails_the_node():
    result = run_flow(load_flow(_code_flow("float", "10**400")), {})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "n"
    assert "too large for float" in result.failure.message


def test_a_record_field_too_large_for_float_names_the_field():
    result = run_flow(load_flow(_code_flow("{stats: float}", '{"stats": 10**400}')), {})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.trace[0].key.endswith("/s:path/s:stats")


def test_an_undeclared_output_widening_past_float_fails_the_node():
    flow = load_flow("""
id: overflow
name: overflow
nodes:
  n:
    kind: code
    code: tests.engine._compose_codefns:huge_float_list
output: ${n.output}
""")
    result = run_flow(flow, {})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "n"


def test_a_run_input_too_large_for_float_fails_at_start():
    flow = load_flow("""
id: overflow
name: overflow
input:
  x: !float
nodes:
  n:
    kind: code
    input:
      x: ${input.x}
    code: |
      return inputs["x"]
output: ${n.output}
""")
    result = run_flow(flow, {"x": 10**400})

    assert result.status == "failed"
    assert result.failure.code == "contract.input"
    assert result.failure.node_id == "__start__"


def _reader_flow(output_line: str, value: str) -> str:
    """Return a flow whose `c` CODE node returns `value` and whose `d` node reads it."""
    return f"""
id: nonfinite
name: nonfinite
nodes:
  c:
    kind: code
{output_line}
    code: |
      return {value}
  d:
    kind: code
    input:
      seen: ${{c.output}}
    code: |
      return inputs["seen"]
output: ${{d.output}}
"""


def _assert_output_fails_before_reader(result, message: str) -> None:
    """Assert `c` failed its output contract with `message` and `d` never ran."""
    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "c"
    assert result.failure.message == message
    assert not [event for event in result.events if getattr(event, "node_id", None) == "d"]


def test_a_nonfinite_object_output_fails_the_node_at_the_field():
    result = run_flow(load_flow(_reader_flow("    output: object", '{"a": float("nan")}')), {})

    _assert_output_fails_before_reader(
        result, "output at a: non-finite number nan is not a yian value"
    )
    assert result.failure.trace[0].key.endswith("/s:output/s:contract/s:root")


def test_a_nonfinite_undeclared_output_fails_the_node_at_the_item():
    result = run_flow(load_flow(_reader_flow("", '[{"a": float("-inf")}]')), {})

    _assert_output_fails_before_reader(
        result, "output at [0].a: non-finite number -inf is not a yian value"
    )


def test_a_nonfinite_value_inside_a_record_field_names_the_path():
    flow = load_flow(_reader_flow("    output: {stats: object}", '{"stats": {"a": float("inf")}}'))
    result = run_flow(flow, {})

    _assert_output_fails_before_reader(
        result, "output at stats.a: non-finite number inf is not a yian value"
    )
    assert result.failure.trace[0].key.endswith("/s:output/s:contract/s:path/s:stats")


def test_a_nonfinite_tool_output_fails_the_node():
    flow = load_flow("""
id: nonfinite
name: nonfinite
nodes:
  t:
    kind: tool
    tool_id: measure
    input:
      qty: 5
output: ${t.output}
""")

    def measure(qty):
        return {"a": float("nan")}

    result = run_flow(flow, {}, tools={"measure": measure})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "t"
    assert result.failure.message == "output at a: non-finite number nan is not a yian value"


def test_a_nonfinite_float_output_fails_the_node():
    result = run_flow(load_flow(_code_flow("float", 'float("nan")')), {})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "n"
    assert result.failure.message == "output: non-finite number nan is not a yian value"


_SCALED_BINDING_FLOW = """
id: nonfinite
name: nonfinite
input:
  x: !float
nodes:
  c:
    kind: code
    input:
      y: ${input.x * 10}
    code: |
      return inputs["y"]
output: ${c.output}
"""


def _assert_binding_fails(result, node_id: str, message: str) -> None:
    """Assert `node_id` failed `binding.evaluate` with exactly `message`."""
    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == node_id
    assert result.failure.message == message


@pytest.mark.parametrize("learnable", [False, True])
def test_a_binding_that_evaluates_to_infinity_fails_its_node(learnable):
    result = run_flow(load_flow(_SCALED_BINDING_FLOW), {"x": 1e308}, learnable=learnable)

    _assert_binding_fails(result, "c", "input 'y': non-finite number inf is not a yian value")
    assert result.failure.trace[0].key == "/s:root/s:node/s:c/s:input/s:y/s:binding"


def test_a_learnable_literal_nan_binding_fails_its_node():
    flow = load_flow("""
id: nonfinite
name: nonfinite
nodes:
  c:
    kind: code
    input:
      y: .nan
    code: |
      return inputs["y"]
output: ${c.output}
""")
    result = run_flow(flow, {}, learnable=True)

    _assert_binding_fails(result, "c", "input 'y': non-finite number nan is not a yian value")


def _call_flow(declared: str) -> str:
    """Return a flow whose `k` CALL node binds `input.x * 10` to a def input `y: declared`."""
    return f"""
id: nonfinite
name: nonfinite
input:
  x: !float
defs:
  body:
    input:
      y: !{declared}
    nodes:
      inner:
        kind: code
        input:
          y: ${{input.y}}
        code: |
          return inputs["y"]
    output: ${{inner.output}}
nodes:
  k:
    kind: call
    call: body
    input:
      y: ${{input.x * 10}}
output: ${{k.output}}
"""


@pytest.mark.parametrize("declared", ["float", "int"])
def test_a_call_binding_that_evaluates_to_infinity_fails_the_call(declared):
    result = run_flow(load_flow(_call_flow(declared)), {"x": 1e308})

    _assert_binding_fails(result, "k", "input 'y': non-finite number inf is not a yian value")
    started = [event.node_id for event in result.events if isinstance(event, NodeStarted)]
    assert started == ["__start__", "k"]


def test_a_literal_binding_names_the_nonfinite_path():
    flow = load_flow("""
id: nonfinite
name: nonfinite
nodes:
  c:
    kind: code
    input:
      y: {a: [1, .nan]}
    code: |
      return inputs["y"]
output: ${c.output}
""")
    result = run_flow(flow, {})

    _assert_binding_fails(
        result, "c", "input 'y' at a[1]: non-finite number nan is not a yian value"
    )


def test_a_nonfinite_start_default_fails_start():
    flow = load_flow("""
id: nonfinite
name: nonfinite
input:
  x: !float .inf
nodes:
  c:
    kind: code
    input:
      y: ${input.x}
    code: |
      return inputs["y"]
output: ${c.output}
""")
    result = run_flow(flow, {})

    _assert_binding_fails(
        result, "__start__", "default for 'x': non-finite number inf is not a yian value"
    )


def _map_flow(over: str, item_binding: str) -> str:
    """Return a flow whose `m` MAP node iterates `over` and binds `y` to `item_binding`."""
    return f"""
id: nonfinite
name: nonfinite
input:
  x: !float
nodes:
  m:
    kind: map
    over: "{over}"
    input:
      y: !float {item_binding}
    nodes:
      work:
        kind: code
        input:
          y: ${{input.y}}
        code: |
          return inputs["y"]
    output: ${{work.output}}
output: ${{m.output}}
"""


def test_a_nonfinite_map_over_item_fails_the_map():
    result = run_flow(load_flow(_map_flow("${[input.x, input.x * 10]}", "${item}")), {"x": 1e308})

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == "m"
    assert "`over`" in result.failure.message
    assert "at [1]" in result.failure.message
    assert result.failure.trace[0].key == "/s:root/s:node/s:m/s:reserved/s:over/s:binding"


def test_a_nonfinite_map_item_binding_fails_the_map():
    result = run_flow(load_flow(_map_flow("${[input.x, 2.0]}", "${item * 10}")), {"x": 1e308})

    _assert_binding_fails(result, "m", "input 'y': non-finite number inf is not a yian value")


def test_a_nonfinite_flow_output_binding_fails_the_end_binding():
    flow = load_flow("""
id: nonfinite
name: nonfinite
input:
  x: !float
nodes:
  c:
    kind: code
    code: |
      return 1
output: ${input.x * 10}
""")
    result = run_flow(flow, {"x": 1e308})

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.message == "input 'result': non-finite number inf is not a yian value"


def test_a_nonfinite_temporary_inside_a_template_is_allowed():
    flow = load_flow("""
id: nonfinite
name: nonfinite
input:
  x: !float
nodes:
  c:
    kind: code
    input:
      y: v=${input.x * 10}
    code: |
      return inputs["y"]
output: ${c.output}
""")
    result = run_flow(flow, {"x": 1e308})

    assert result.status == "succeeded"
    assert result.output == "v=inf"


def test_apply_defaults_rejects_a_nonfinite_default():
    with pytest.raises(BindingError) as caught:
        apply_defaults([Parameter("p", has_default=True, default=math.inf)], {})

    assert caught.value.failure_code == "binding.evaluate"
    assert str(caught.value) == "default for 'p': non-finite number inf is not a yian value"


def test_apply_defaults_rejects_a_nonfinite_default_before_its_contract():
    parameter = Parameter(
        "p", ValueContract(Type(ValueKind.INTEGER)), has_default=True, default=math.nan
    )

    with pytest.raises(BindingError, match="^default for 'p': non-finite number nan"):
        apply_defaults([parameter], {})


@pytest.mark.parametrize("learnable", [False, True])
def test_a_nonfinite_agent_variable_default_fails_the_agent(monkeypatch, learnable):
    calls = []
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: calls.append(ctx) or Output("x"))
    flow = load_flow("""
id: nonfinite
name: nonfinite
llm_config: {provider: ollama, model: test}
nodes:
  writer:
    kind: agent
    variables:
      t: !float .inf
    prompt: 'go ${t}'
    output: str
output: ${writer.output}
""")
    result = run_flow(flow, {}, learnable=learnable)

    assert result.status == "failed"
    assert result.failure.code == "node.execute"
    assert result.failure.node_id == "writer"
    assert result.failure.message == "default for 't': non-finite number inf is not a yian value"
    assert result.failure.trace[0].key == "/s:root/s:node/s:writer/s:variables/s:t/s:default"
    assert calls == []


def _input_flow(declared: str) -> str:
    """Return a flow whose run input `x: declared` feeds one CODE node."""
    return f"""
id: nonfinite
name: nonfinite
input:
  x: !{declared}
nodes:
  c:
    kind: code
    input:
      y: ${{input.x}}
    code: |
      return inputs["y"]
output: ${{c.output}}
"""


@pytest.mark.parametrize(
    ("declared", "value", "message"),
    [
        ("float", math.nan, "input 'x': non-finite number nan is not a yian value"),
        ("float", "inf", "input 'x': non-finite number inf is not a yian value"),
        (
            "object",
            {"a": [1, math.inf]},
            "input 'x' at a[1]: non-finite number inf is not a yian value",
        ),
        (
            "object",
            [1.0, -math.inf],
            "input 'x' at [1]: non-finite number -inf is not a yian value",
        ),
    ],
    ids=["nan", "coerced-string", "nested-object", "list"],
)
def test_a_nonfinite_run_input_is_rejected_before_the_run_starts(
    tmp_path, declared, value, message
):
    log = tmp_path / "run.jsonl"

    with pytest.raises(ValueError) as caught:
        run_flow(load_flow(_input_flow(declared)), {"x": value}, log=log)

    assert str(caught.value) == message
    assert not log.exists()

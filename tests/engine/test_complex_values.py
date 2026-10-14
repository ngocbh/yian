"""A complex number fails at the first boundary that sees it, like NaN."""

import pytest

from yian import load_flow, run_flow
from yian.events import NodeStarted, NodeSucceeded

C = (0 - 8) ** 0.5


def _assert_binding_fails(result, node_id: str, message: str) -> None:
    """Assert `node_id` failed `binding.evaluate` with exactly `message`."""
    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == node_id
    assert result.failure.message == message


@pytest.mark.parametrize("learnable", [False, True])
def test_a_binding_that_evaluates_to_a_complex_fails_its_node(learnable):
    flow = load_flow("""
id: complex
name: complex
nodes:
  n:
    kind: code
    input:
      v: ${(0 - 8) ** 0.5}
    code: |
      return inputs["v"]
output: ${n.output}
""")
    result = run_flow(flow, {}, learnable=learnable)

    _assert_binding_fails(result, "n", f"input 'v': complex number {C!r} is not a yian value")
    assert result.failure.trace[0].key == "/s:root/s:node/s:n/s:input/s:v/s:binding"
    assert not [
        event for event in result.events
        if isinstance(event, NodeSucceeded) and event.node_id == "n"
    ]


def _call_flow(declared: str, binding: str) -> str:
    """Return a flow whose `k` CALL node binds `binding` to a def input `y: declared`."""
    return f"""
id: complex
name: complex
input:
  x: float
defs:
  body:
    input:
      y: {declared}
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
      y: {binding}
output: ${{k.output}}
"""


@pytest.mark.parametrize("declared", ["float", "int"])
def test_a_call_binding_that_evaluates_to_a_complex_fails_the_call(declared):
    result = run_flow(load_flow(_call_flow(declared, "${(0 - 8) ** 0.5}")), {"x": 1.0})

    _assert_binding_fails(result, "k", f"input 'y': complex number {C!r} is not a yian value")
    started = [event.node_id for event in result.events if isinstance(event, NodeStarted)]
    assert started == ["__start__", "k"]


def _reader_flow(output_line: str, value: str) -> str:
    """Return a flow whose `c` CODE node returns `value` and whose `d` node reads it."""
    return f"""
id: complex
name: complex
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


def test_a_complex_object_output_fails_the_node_at_the_field():
    result = run_flow(load_flow(_reader_flow("    output: object", '{"a": [1j]}')), {})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "c"
    assert result.failure.message == "output at a[0]: complex number 1j is not a yian value"
    assert not [event for event in result.events if getattr(event, "node_id", None) == "d"]


def test_a_complex_undeclared_output_fails_the_node():
    flow = load_flow("""
id: complex
name: complex
nodes:
  n:
    kind: code
    code: |
      return (-8) ** 0.5
output: ${n.output}
""")
    result = run_flow(flow, {})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "n"
    assert result.failure.message == f"output: complex number {C!r} is not a yian value"


def test_a_complex_flow_output_binding_fails_the_end_binding():
    flow = load_flow("""
id: complex
name: complex
nodes:
  c:
    kind: code
    code: |
      return 1
output: ${(0 - 8) ** 0.5}
""")
    result = run_flow(flow, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.message == f"input 'result': complex number {C!r} is not a yian value"


def test_a_complex_map_over_item_fails_the_map():
    flow = load_flow("""
id: complex
name: complex
nodes:
  m:
    kind: map
    over: "${[1.0, (0 - 8) ** 0.5]}"
    input:
      y: float = ${item}
    nodes:
      work:
        kind: code
        input:
          y: ${input.y}
        code: |
          return inputs["y"]
    output: ${work.output}
output: ${m.output}
""")
    result = run_flow(flow, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == "m"
    assert "`over`" in result.failure.message
    assert f"at [1]: complex number {C!r} is not a yian value" in result.failure.message
    assert result.failure.trace[0].key == "/s:root/s:node/s:m/s:reserved/s:over/s:binding"


def test_a_complex_run_input_is_rejected_before_the_run_starts(tmp_path):
    flow = load_flow("""
id: complex
name: complex
input:
  x: object
nodes:
  c:
    kind: code
    input:
      y: ${input.x}
    code: |
      return inputs["y"]
output: ${c.output}
""")
    log = tmp_path / "run.jsonl"

    with pytest.raises(ValueError) as caught:
        run_flow(flow, {"x": {"a": 1j}}, log=log)

    assert str(caught.value) == "input 'x' at a: complex number 1j is not a yian value"
    assert not log.exists()


def test_a_complex_temporary_inside_a_template_is_allowed():
    flow = load_flow("""
id: complex
name: complex
nodes:
  c:
    kind: code
    input:
      y: v=${(0 - 8) ** 0.5}
    code: |
      return inputs["y"]
output: ${c.output}
""")
    result = run_flow(flow, {})

    assert result.status == "succeeded"
    assert result.output == f"v={C}"


def test_a_complex_tool_output_fails_the_node():
    flow = load_flow("""
id: complex
name: complex
nodes:
  t:
    kind: tool
    tool_id: measure
    input:
      qty: 5
output: ${t.output}
""")

    def measure(qty):
        return {"a": 1j}

    result = run_flow(flow, {}, tools={"measure": measure})

    assert result.status == "failed"
    assert result.failure.code == "contract.output"
    assert result.failure.node_id == "t"
    assert result.failure.message == "output at a: complex number 1j is not a yian value"

"""NaN, infinity, and float overflow fail at the first boundary that sees them."""

from yian import load_flow, run_flow


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
  x: float
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

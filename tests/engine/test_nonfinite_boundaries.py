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

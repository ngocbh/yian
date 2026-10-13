"""Typed flow outputs checked when a run reaches the flow's END.

An output written `name: Type = value` declares the value's type. END's binder
checks the resolved value against it (code `binding.type`, located on the output),
and END commits through the declared contract.
"""

from yian.compose.loader import load_flow
from yian.run import run_flow


def _value_flow(value: object, output: str) -> str:
    """Return a flow whose `n` node returns `value` into `output:` written as `output`."""
    return f"""
id: typed_out
name: typed_out
nodes:
  n:
    kind: code
    code: |
      return {value!r}
output: {output}
"""


def test_mismatched_typed_output_fails_end_on_the_output_line():
    text = _value_flow("x", "int = ${n.output}")
    loaded = load_flow(text)

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "__end__"
    diagnostic = loaded.diagnose(result.failure)
    assert diagnostic.frames, diagnostic.notes
    line = text.splitlines().index("output: int = ${n.output}") + 1
    assert diagnostic.frames[-1].range.covered_lines(text) == [line]


def test_matching_typed_output_returns_the_value():
    result = run_flow(load_flow(_value_flow(5, "int = ${n.output}")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == 5


def test_float_output_widens_an_int_value():
    result = run_flow(load_flow(_value_flow(30, "float = ${n.output}")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == 30.0
    assert type(result.output) is float


def _record_flow(b: str) -> str:
    """Return a flow whose record output types `a: float` and writes `b` as `b`."""
    return f"""
id: typed_record
name: typed_record
nodes:
  n:
    kind: code
    code: |
      return 30
  m:
    kind: code
    code: |
      return "m"
output:
  a: float = ${{n.output}}
  b: {b}
"""


def test_fully_typed_record_output_widens_an_int_in_a_float_field():
    result = run_flow(load_flow(_record_flow("str = ${m.output}")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == {"a": 30.0, "b": "m"}
    assert type(result.output["a"]) is float


def test_partly_typed_record_output_widens_its_typed_float_field():
    result = run_flow(load_flow(_record_flow("${m.output}")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == {"a": 30.0, "b": "m"}
    assert type(result.output["a"]) is float


def test_partly_typed_record_output_keeps_its_untyped_field_as_given():
    text = _record_flow("${m.output}").replace('return "m"', "return 7")

    result = run_flow(load_flow(text), {})

    assert result.status == "succeeded", result.failure
    assert result.output == {"a": 30.0, "b": 7}
    assert type(result.output["b"]) is int


def test_record_node_output_widens_like_a_whole_float_output():
    text = """
id: recfloat
name: recfloat
typedefs:
  Score:
    value: float
input:
  n: int
nodes:
  top:
    kind: code
    input: {n: "${input.n}"}
    output: float
    code: |
      return inputs["n"]
  rec:
    kind: code
    input: {n: "${input.n}"}
    output: Score
    code: |
      return {"value": inputs["n"]}
output:
  top: ${top.output}
  rec: ${rec.output.value}
"""
    result = run_flow(load_flow(text), {"n": 1})

    assert result.status == "succeeded", result.failure
    assert result.output == {"top": 1.0, "rec": 1.0}
    assert type(result.output["top"]) is float
    assert type(result.output["rec"]) is float


def test_partly_typed_record_output_rejects_a_mistyped_typed_field():
    text = _record_flow("${m.output}").replace("return 30", 'return "x"')

    result = run_flow(load_flow(text), {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "__end__"


def _call_flow(child_value: object, child_output: str) -> str:
    """Return a flow that CALLs a child whose `n` returns `child_value`."""
    return f"""
id: typed_parent
name: typed_parent
defs:
  child:
    nodes:
      n:
        kind: code
        code: |
          return {child_value!r}
    output: {child_output}
nodes:
  use:
    kind: call
    call: child
output: ${{use.output}}
"""


def test_mismatched_typed_child_output_fails_the_run():
    result = run_flow(load_flow(_call_flow("x", "int = ${n.output}")), {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id.endswith("__end__")


def test_float_child_output_widens_through_the_call():
    result = run_flow(load_flow(_call_flow(30, "float = ${n.output}")), {})

    assert result.status == "succeeded", result.failure
    assert result.output == 30.0
    assert type(result.output) is float

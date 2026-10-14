"""Runtime binding errors name a desugared param the way load errors name it.

A CASE `on:` reads as `node '<id>' on:`, a value entry's one input reads as
`value entry '<id>'`, and a flow output reads as `flow output` (a record field as
`flow output '<name>'`); template paths add `at <path>`. An inline call's argument reads
as `<site> call(<callee>) argument '<p>'`. An ordinary node input keeps `input '<p>'`.
"""

import pytest

from yian import load_flow, run_flow
from yian.compose import LoadError


def test_a_case_on_failure_names_the_on_field():
    flow = load_flow("""
id: f
name: f
input:
  x: !float
nodes:
  g:
    kind: case
    on: ${input.x * 10}
    cases:
      - when: 1
        then: a
    else: a
  a: !str hi
output: ${g.output}
""")
    result = run_flow(flow, {"x": 1e308})

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == "g"
    assert result.failure.message == (
        "node 'g' on: non-finite number inf is not a yian value"
    )
    assert result.failure.trace[-1].key == "/s:root/s:node/s:g/s:field/s:on"


def test_a_case_on_failure_in_a_called_def_names_the_on_field():
    flow = load_flow("""
id: f
name: f
input:
  x: !float
defs:
  route:
    input:
      x: !float
    nodes:
      g:
        kind: case
        on: ${input.x * 10}
        cases:
          - when: 1
            then: a
        else: a
      a: !str hi
    output: ${g.output}
nodes:
  invoke:
    kind: call
    call: route
    input:
      x: ${input.x}
output: ${invoke.output}
""")
    result = run_flow(flow, {"x": 1e308})

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == "invoke/g"
    assert result.failure.message == (
        "node 'g' on: non-finite number inf is not a yian value"
    )


def test_a_value_entry_failure_names_the_entry():
    flow = load_flow("""
id: f
name: f
input:
  x: !float
nodes:
  v: !float ${input.x * 10}
output: ${v.output}
""")
    result = run_flow(flow, {"x": 1e308})

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == "v"
    assert result.failure.message == (
        "value entry 'v': non-finite number inf is not a yian value"
    )
    assert result.failure.trace[-1].key == "/s:root/s:node/s:v/s:input/s:v/s:binding"


def test_a_value_entry_template_failure_names_the_entry_and_path():
    flow = load_flow("""
id: f
name: f
input:
  x: !float
  z: !float
nodes:
  v: !object {a: "${input.x / input.z}"}
output: ${v.output}
""")
    result = run_flow(flow, {"x": 1.0, "z": 0.0})

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == "v"
    assert result.failure.message == "value entry 'v' at a: division by zero: 1.0 / 0.0"


def test_a_value_entry_type_failure_names_the_entry():
    flow = load_flow("""
id: f
name: f
nodes:
  p:
    kind: code
    code: |
      return 5
  x: !str ${p.output}
output: ${x.output}
""")
    result = run_flow(flow, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "x"
    assert result.failure.message == "value entry 'x': 5 does not match declared type str"


def test_a_routed_value_entry_failure_points_at_the_entry():
    flow = load_flow("""
id: f
name: f
input:
  x: !float
  flag: !bool
nodes:
  g:
    kind: case
    on: ${input.flag}
    cases:
      - when: true
        then: a
    else: a
  a: !float ${input.x * 10}
output: ${g.output}
""")
    result = run_flow(flow, {"x": 1e308, "flag": True})

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == "a"
    assert result.failure.message == (
        "value entry 'a': non-finite number inf is not a yian value"
    )
    assert result.failure.trace[-1].key == "/s:root/s:node/s:a/s:input/s:a/s:binding"


def test_a_code_node_input_failure_keeps_the_input_name():
    flow = load_flow("""
id: f
name: f
input:
  x: !float
nodes:
  n:
    kind: code
    input:
      y: ${input.x * 10}
    code: |
      return inputs["y"]
output: ${n.output}
""")
    result = run_flow(flow, {"x": 1e308})

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == "n"
    assert result.failure.message == "input 'y': non-finite number inf is not a yian value"


def _code_and(output: str, inputs: str = "  x: !float\n") -> str:
    """Return a flow with a trivial CODE node `c` and the given `output:` text."""
    return f"""
id: f
name: f
input:
{inputs}nodes:
  c:
    kind: code
    code: |
      return 1
output: {output}
"""


def _mismatch_flow(output: str) -> str:
    """Return a flow whose CODE `m` returns "x" and the given `output:` text."""
    return f"""
id: f
name: f
nodes:
  m:
    kind: code
    code: |
      return "x"
output: {output}
"""


def _assert_end_fails(result, code: str, message: str) -> None:
    """Assert `result` failed at END's binding with `code` and `message`."""
    assert result.status == "failed"
    assert result.failure.code == code
    assert result.failure.node_id == "__end__"
    assert result.failure.message == message


def test_a_single_flow_output_failure_names_the_flow_output():
    result = run_flow(load_flow(_code_and("${input.x * 10}")), {"x": 1e308})

    _assert_end_fails(
        result, "binding.evaluate", "flow output: non-finite number inf is not a yian value"
    )
    assert result.failure.trace[-1].key == "/s:root/s:output/i:0/s:binding"


def test_a_record_flow_output_failure_names_the_field():
    result = run_flow(load_flow(_code_and("\n  r: ${input.x * 10}")), {"x": 1e308})

    _assert_end_fails(
        result, "binding.evaluate", "flow output 'r': non-finite number inf is not a yian value"
    )


def test_a_single_flow_output_type_failure_names_the_flow_output():
    result = run_flow(load_flow(_mismatch_flow("!int ${m.output}")), {})

    _assert_end_fails(
        result, "binding.type", "flow output: 'x' does not match declared type int"
    )


def test_a_record_flow_output_type_failure_names_the_field():
    result = run_flow(load_flow(_mismatch_flow("\n  r: !int ${m.output}")), {})

    _assert_end_fails(
        result, "binding.type", "flow output 'r': 'x' does not match declared type int"
    )


def test_a_templated_flow_output_failure_names_the_path():
    flow = load_flow(_code_and('!object {a: "${input.x / input.z}"}', "  x: !float\n  z: !float\n"))
    result = run_flow(flow, {"x": 1.0, "z": 0.0})

    _assert_end_fails(
        result, "binding.evaluate", "flow output at a: division by zero: 1.0 / 0.0"
    )


def test_an_unknown_type_in_a_single_flow_output_names_the_flow_output():
    with pytest.raises(LoadError) as caught:
        load_flow(_code_and("${isinstance(input.s, Nope)}", "  s: !str\n"))

    assert "flow output: unknown type 'Nope'" in str(caught.value)


def test_an_unknown_type_in_a_record_flow_output_names_the_field():
    with pytest.raises(LoadError) as caught:
        load_flow(_code_and("\n  r: ${isinstance(input.s, Nope)}", "  s: !str\n"))

    assert "flow output 'r': unknown type 'Nope'" in str(caught.value)


def test_an_inline_call_argument_failure_names_the_argument():
    text = """
id: f
name: f
input:
  x: !float
defs:
  child:
    input:
      y: !float
    nodes:
      c:
        kind: code
        code: "return inputs['y']"
        input:
          y: ${input.y}
        output: float
    output: ${c.output}
nodes:
  n:
    kind: code
    code: "return inputs['a']"
    input:
      a: call(child, y=${input.x * 10})
    output: float
output: ${n.output}
"""
    result = run_flow(load_flow(text), {"x": 1e308})

    assert result.status == "failed"
    assert result.failure.node_id == "__call_0"
    assert result.failure.message == (
        "node 'n' input 'a' call(child) argument 'y': non-finite number inf is not a yian value"
    )


_CHILD_DEFS = """
defs:
  child:
    input:
      y: !float
    nodes:
      c:
        kind: code
        code: "return inputs['y']"
        input:
          y: ${input.y}
        output: float
    output: ${c.output}
  check:
    input:
      y: !float
    nodes:
      c:
        kind: code
        code: "return True"
        input:
          y: ${input.y}
        output: bool
    output: ${c.output}
  lister:
    input:
      n: !float
    nodes:
      c:
        kind: code
        code: "return [inputs['n']]"
        input:
          n: ${input.n}
        output: list(float)
    output: ${c.output}
"""

_BAD_ARG = "${input.x * 10}"


def _call_flow(body: str, inputs: str = "  x: !float\n") -> str:
    """Return a flow with the `child`, `check`, and `lister` defs and the given body."""
    header = "id: f\nname: f\n" + (f"input:\n{inputs}" if inputs else "")
    return header + _CHILD_DEFS + body


def _argument_failure_key(result) -> str:
    """Assert `result` failed evaluating a call argument and return its last trace key."""
    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.trace
    return result.failure.trace[-1].key


def test_a_node_input_call_argument_failure_points_at_the_input():
    text = _call_flow(f"""
nodes:
  n:
    kind: code
    code: "return inputs['a']"
    input:
      a: call(child, y={_BAD_ARG})
    output: float
output: ${{n.output}}
""")
    result = run_flow(load_flow(text), {"x": 1e308})

    assert _argument_failure_key(result) == "/s:root/s:node/s:n/s:input/s:a/s:binding"


def test_a_value_entry_call_argument_failure_points_at_the_entry():
    text = _call_flow(f"""
nodes:
  v: !float call(child, y={_BAD_ARG})
output: ${{v.output}}
""")
    result = run_flow(load_flow(text), {"x": 1e308})

    assert _argument_failure_key(result) == "/s:root/s:node/s:v/s:input/s:v/s:binding"


def test_a_single_flow_output_call_argument_failure_points_at_the_output():
    text = _call_flow(f"""
nodes:
  n: {{kind: code, code: "return 1"}}
output: call(child, y={_BAD_ARG})
""")
    result = run_flow(load_flow(text), {"x": 1e308})

    assert _argument_failure_key(result) == "/s:root/s:output/i:0/s:binding"


def test_a_record_flow_output_call_argument_failure_points_at_the_field():
    text = _call_flow(f"""
nodes:
  n: {{kind: code, code: "return 1"}}
output:
  q: ${{n.output}}
  r: call(child, y={_BAD_ARG})
""")
    result = run_flow(load_flow(text), {"x": 1e308})

    assert _argument_failure_key(result) == "/s:root/s:output/i:1/s:binding"


def test_a_nested_call_argument_failure_points_at_the_enclosing_binding():
    text = _call_flow(f"""
nodes:
  n:
    kind: code
    code: "return inputs['a']"
    input:
      a: call(child, y=call(child, y={_BAD_ARG}))
    output: float
output: ${{n.output}}
""")
    result = run_flow(load_flow(text), {"x": 1e308})

    assert _argument_failure_key(result) == "/s:root/s:node/s:n/s:input/s:a/s:binding"


def _case_call_flow(branches: str, readers: str = "") -> str:
    """Return a flow whose CASE `g` routes on `input.flag` with the given branches."""
    return _call_flow(
        f"""
nodes:
  g:
    kind: case
    on: ${{input.flag}}
{branches}
  b: !float 1.0
{readers}output: ${{g.output}}
""",
        inputs="  x: !float\n  flag: !bool\n",
    )


def test_a_case_else_call_argument_failure_points_at_the_else_field():
    text = _case_call_flow(f"""    cases:
      - when: true
        then: b
    else: call(child, y={_BAD_ARG})""")
    result = run_flow(load_flow(text), {"x": 1e308, "flag": False})

    assert _argument_failure_key(result).endswith("/s:node/s:g/s:field/s:else")


def test_a_case_then_call_argument_failure_points_at_the_then_field():
    text = _case_call_flow(f"""    cases:
      - when: true
        then: call(child, y={_BAD_ARG})
    else: b""")
    result = run_flow(load_flow(text), {"x": 1e308, "flag": True})

    assert _argument_failure_key(result).endswith("/s:node/s:g/s:case/i:0/s:then")


def test_a_read_case_else_call_argument_failure_points_at_the_else_field():
    readers = """  r:
    kind: code
    code: "return inputs['v']"
    input:
      v: ${g.output}
    output: float
"""
    text = _case_call_flow(
        f"""    cases:
      - when: true
        then: b
    else: call(child, y={_BAD_ARG})""",
        readers=readers,
    )
    result = run_flow(load_flow(text), {"x": 1e308, "flag": False})

    assert _argument_failure_key(result).endswith("/s:node/s:g/s:field/s:else")


def test_a_flow_assert_call_argument_failure_points_at_the_assert():
    text = _call_flow(f"""
nodes:
  n: {{kind: code, code: "return 1"}}
output: ${{n.output}}
asserts:
  - call(check, y={_BAD_ARG})
""")
    result = run_flow(load_flow(text), {"x": 1e308})

    assert _argument_failure_key(result) == "/s:root/s:assert/i:0"


def test_a_map_over_call_argument_failure_points_at_the_over_field():
    text = _call_flow(f"""
nodes:
  m:
    kind: map
    call: child
    over: call(lister, n={_BAD_ARG})
    input:
      y: ${{item}}
output: ${{m.output}}
""")
    result = run_flow(load_flow(text), {"x": 1e308})

    assert _argument_failure_key(result) == "/s:root/s:node/s:m/s:reserved/s:over/s:binding"


def test_a_child_rejection_of_a_call_argument_has_the_authored_outer_frame():
    text = _call_flow(
        """
nodes:
  m:
    kind: code
    code: return "x"
  n:
    kind: code
    code: "return inputs['a']"
    input:
      a: call(child, y=${m.output})
    output: float
output: ${n.output}
""",
        inputs="",
    )
    result = run_flow(load_flow(text), {})

    assert result.status == "failed"
    assert result.failure.code == "contract.input"
    assert result.failure.node_id == "__call_0/__start__"
    assert result.failure.trace[0].key == "/s:root/s:node/s:n/s:input/s:a/s:binding"
    assert result.failure.trace[-1].key == "/s:def/s:child/s:input/s:y/s:contract/s:type/s:root"

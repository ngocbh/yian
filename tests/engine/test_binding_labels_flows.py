"""Runtime binding errors name a desugared param the way load errors name it.

A CASE `on:` reads as `node '<id>' on:`, and a value entry's one input reads as
`value entry '<id>'` (template paths add `at <path>`). An ordinary node input keeps
`input '<p>'`.
"""

from yian import load_flow, run_flow


def _case_flow() -> str:
    """Return a flow whose CASE `g` routes on `input.x * 10`."""
    return """
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
"""


def test_a_case_on_failure_names_the_on_field():
    result = run_flow(load_flow(_case_flow()), {"x": 1e308})

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
  z: !float
nodes:
  g:
    kind: case
    on: ${input.x}
    cases:
      - when: 1.0
        then: a
    else: a
  a: !float ${input.x / input.z}
output: ${g.output}
""")
    result = run_flow(flow, {"x": 1.0, "z": 0.0})

    assert result.status == "failed"
    assert result.failure.code == "binding.evaluate"
    assert result.failure.node_id == "a"
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

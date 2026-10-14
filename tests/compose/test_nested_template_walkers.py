"""Load-time walkers see the strings inside a nested binding value.

A binding written as a mapping or list holding `${...}` spans
(`x: {a: ${p.output}}`) is a `ValueTemplate`: every leaf string is its own template.
Reference checks, producer edges, `isinstance` names, desugaring and runtime growth
all read those leaf strings.
"""

from __future__ import annotations

import re

import pytest

from yian.compose.desugar.inline_calls import desugar_inline_calls
from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.compose.nodes.code import CodeDescriptor
from yian.compose.nodes.map import MapDescriptor
from yian.nodes.contracts import Binding, ValueTemplate
from yian.run import run_flow


def _code_flow(binding: str) -> str:
    """Return a flow whose CODE node `c` binds `x` as `binding`."""
    return f"""
id: nested
name: nested
nodes:
  p:
    kind: code
    code: |
      return "hi"
  c:
    kind: code
    input:
      x: {binding}
    code: |
      return inputs["x"]
output: ${{c.output}}
"""


def _source(compiled, node_id: str, name: str):
    """Return one node's wiring binding source by parameter name."""
    binding = compiled.wiring[node_id][name]
    assert type(binding) is Binding
    return binding.source


def test_untagged_nested_binding_reads_as_a_template():
    compiled = load_flow(_code_flow("{a: '${p.output}', b: 1}")).compiled

    source = _source(compiled, "c", "x")
    assert type(source) is ValueTemplate
    assert source.value == {"a": "${p.output}", "b": 1}
    assert compiled.wiring["c"]["x"].contract is None


def test_flow_output_field_reads_as_a_template():
    text = _code_flow("${p.output}").replace(
        "output: ${c.output}", "output:\n  r: {a: '${c.output}'}"
    )
    compiled = load_flow(text).compiled

    (output,) = compiled.outputs
    assert output.name == "r"
    assert type(output.binding.source) is ValueTemplate


def _load_error(text: str) -> LoadError:
    """Load `text`, expecting a load error, and return it."""
    with pytest.raises(LoadError) as info:
        load_flow(text)
    return info.value


def _error_line(error: LoadError) -> int:
    """Return the 1-based line of the error's last diagnostic frame."""
    frames = error.diagnostic.frames
    assert frames, error.diagnostic.message
    return frames[-1].range.start_line


def _line_of(text: str, line: str) -> int:
    """Return the 1-based line number of `line` in `text`."""
    return text.splitlines().index(line) + 1


def test_dangling_nested_reference_fails_at_its_own_line():
    text = _code_flow("\n        a: ${p.output}\n        b: ${nope.output}")

    error = _load_error(text)

    assert (
        "node 'c' input 'x' from: reference ${nope.output} uses unknown namespace 'nope'"
        in error.diagnostic.message
    )
    assert _error_line(error) == _line_of(text, "        b: ${nope.output}")


def test_dangling_nested_output_reference_fails_at_its_own_line():
    text = _code_flow("${p.output}").replace(
        "output: ${c.output}",
        "output:\n  r:\n    a: ${c.output}\n    b:\n      - ${nope.output}",
    )

    error = _load_error(text)

    assert (
        "flow output from: reference ${nope.output} uses unknown namespace 'nope'"
        in error.diagnostic.message
    )
    assert _error_line(error) == _line_of(text, "      - ${nope.output}")


def test_nested_reference_field_is_checked():
    text = (
        "typedefs:\n  Rec: {a: str}\n"
        + _code_flow("{a: '${p.output.zz}'}").replace(
            "  p:\n    kind: code\n", "  p:\n    kind: code\n    output: Rec\n"
        )
    )

    error = _load_error(text)

    assert (
        "reference ${p.output.zz} reads unknown field 'zz' on record"
        in error.diagnostic.message
    )


def test_nested_reference_orders_its_consumer_after_the_producer():
    compiled = load_flow(_code_flow("{a: ['${p.output}']}")).compiled

    assert ("p", "c") in {(edge.from_, edge.to) for edge in compiled.edges}


def test_nested_isinstance_name_resolves():
    text = "typedefs:\n  Rec: {a: str}\n" + _code_flow(
        "{ok: '${isinstance(p.output, Rec)}'}"
    )

    compiled = load_flow(text).compiled

    assert set(compiled.nodes["c"].types) == {"Rec"}


def test_unknown_nested_isinstance_name_fails():
    error = _load_error(_code_flow("{ok: '${isinstance(p.output, Nope)}'}"))

    assert "unknown type 'Nope'" in error.diagnostic.message


_CASE_FLOW = """
id: nested_case
name: nested_case
input:
  go: !bool
nodes:
  route:
    kind: case
    cases:
      - when: ${input.go}
        then: yes_branch
    else: no_branch
  yes_branch:
    kind: code
    code: return "yes"
  no_branch:
    kind: code
    code: return "no"
  c:
    kind: code
    input:
      x:
        picked: ${route.output}
        note: took ${route.output}
    code: return inputs["x"]
output:
  c: ${c.output}
  direct:
    - ${route.output}
"""


@pytest.mark.parametrize(("go", "picked"), [(True, "yes"), (False, "no")])
def test_nested_case_output_expands_in_inputs_and_outputs(go, picked):
    result = run_flow(load_flow(_CASE_FLOW), {"go": go})

    assert result.status == "succeeded", result.failure
    assert result.output == {
        "c": {"picked": picked, "note": f"took {picked}"},
        "direct": [picked],
    }


def test_nested_case_output_in_a_default_fails():
    text = _CASE_FLOW.replace("note: took ${route.output}", "note: ${yes_branch.output :- route.output}")

    error = _load_error(text)

    assert "${route.output} — a case value can't be read in a coalesce operand" in (
        error.diagnostic.message
    )


def test_retired_inline_flow_call_in_a_nested_output_fails():
    text = _code_flow("${p.output}").replace(
        "output: ${c.output}", "output:\n  r:\n    a: ${ sub(x=c.output) }"
    )
    text = text.replace("nodes:\n", "defs:\n  sub:\n    nodes: {}\nnodes:\n", 1)

    error = _load_error(text)

    assert "a flow call 'sub' inside ${...} is not supported" in error.diagnostic.message


def test_nested_call_directive_is_not_lifted():
    template = ValueTemplate({"a": "call(sub, x=1)"})
    descriptors = {"c": CodeDescriptor(id="c", code="return 1", inputs={"x": template})}

    with pytest.raises(LoadError, match=re.escape(
        "c: a `call(...)` inside a value isn't supported; bind it at the top level"
    )):
        desugar_inline_calls(descriptors, None)


@pytest.mark.parametrize(
    "value", ["\n      - ${a.output}\n      - 2", "\n      x: ${a.output}"]
)
def test_map_over_container_with_a_span_fails_at_the_container(value):
    text = f"""
id: m
name: m
defs:
  echo:
    input:
      v: !int
    nodes: {{}}
    output: ${{input.v}}
nodes:
  a:
    kind: code
    code: return 1
  m:
    kind: map
    call: echo
    over: {value}
    input:
      v: ${{item}}
output: ${{m.output}}
"""
    error = _load_error(text)

    assert (
        "map node 'm' over: a value holding ${...} can't be a list or mapping; write "
        "the list as an expression, e.g. `${[a.output, b.output]}`"
    ) in error.diagnostic.message
    assert _error_line(error) == _line_of(text, "    over: ") + 1


def test_wait_until_container_with_a_span_fails_at_the_container():
    text = """
id: w
name: w
nodes:
  a:
    kind: code
    code: return '2030-01-01'
  w:
    kind: wait
    until:
      - ${a.output}
output: ${w.output}
"""
    error = _load_error(text)

    assert (
        "wait node 'w' until: a value holding ${...} can't be a list or mapping"
    ) in error.diagnostic.message
    assert _error_line(error) == _line_of(text, "      - ${a.output}")


def test_map_over_literal_list_without_spans_still_loads():
    text = """
id: m
name: m
defs:
  echo:
    input:
      v: !int
    nodes: {}
    output: ${input.v}
nodes:
  m:
    kind: map
    call: echo
    over: [1, 2]
    input:
      v: ${item}
output: ${m.output}
"""
    assert run_flow(load_flow(text), {}).output == [1, 2]


_MAP_CHILD_FLOW = """
id: nested_map
name: nested_map
defs:
  wrap:
    input:
      v: !int
    nodes:
      e:
        kind: code
        input:
          v: ${input.v}
        code: return inputs["v"] * 10
      w:
        kind: code
        input:
          x:
            v: ${input.v}
            e: ['${e.output}']
            note: v=${input.v}
        code: return inputs["x"]
    output: ${w.output}
nodes:
  m:
    kind: map
    call: wrap
    over: [1, 2]
    input:
      v: ${item}
output: ${m.output}
"""


def test_nested_references_in_a_map_child_run_after_growth():
    result = run_flow(load_flow(_MAP_CHILD_FLOW), {})

    assert result.status == "succeeded", result.failure
    assert result.output == [
        {"v": 1, "e": [10], "note": "v=1"},
        {"v": 2, "e": [20], "note": "v=2"},
    ]


_LOOP_CHILD_FLOW = """
id: nested_loop
name: nested_loop
defs:
  bump:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: ${input.n}
        code: return inputs["n"] + 1
      pack:
        kind: code
        input:
          x:
            n: ${step.output}
            seen:
              - ${input.n}
        code: return inputs["x"]
        output:
          n: int
          seen: list(int)
    output:
      n: ${pack.output.n}
nodes:
  loop:
    kind: loop
    call: bump
    input:
      n: 0
    while: n < 3
    max: 10
output: ${loop.output}
"""


def test_nested_references_in_a_loop_body_run_after_growth():
    result = run_flow(load_flow(_LOOP_CHILD_FLOW), {})

    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 3}

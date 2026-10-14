"""Load errors about an inline `call(...)` argument name the authored argument.

An inline `call(child, y=...)` desugars to a synth `__call_<n>` node. An error about one
of its arguments reads `<site> call(<callee>) argument '<name>'`, where the site is the
authored binding the call was written in, and never names the synth node.
"""

import pytest

from yian.compose import LoadError, load_flow


def _child_def(y_type: str = "str") -> str:
    """Return a `defs:` block holding `child`, which echoes its one input `y`."""
    return f"""
defs:
  child:
    input:
      y: !{y_type}
    nodes:
      x:
        kind: code
        code: "return inputs['y']"
        input:
          y: ${{input.y}}
        output: {y_type}
    output: ${{x.output}}
"""


def _node_input_flow(binding: str, y_type: str = "str") -> str:
    """Return a flow whose CODE node `n` binds input `a` to `binding`."""
    return f"""
id: f
name: f
input:
  s: !str
{_child_def(y_type)}
nodes:
  n:
    kind: code
    code: "return inputs['a']"
    input:
      a: {binding}
    output: {y_type}
output: ${{n.output}}
"""


def _load_error(text: str) -> str:
    """Load `text`, expecting a `LoadError`; return its message."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return str(caught.value)


def _assert_names(text: str, phrase: str) -> None:
    """Assert loading `text` fails with `phrase` and never names a synth node."""
    message = _load_error(text)
    assert phrase in message
    assert "__call_" not in message


_BAD_REF = "${nosuch.output}"


def test_a_bad_ref_in_a_node_input_call_names_the_argument():
    _assert_names(
        _node_input_flow(f"call(child, y={_BAD_REF})"),
        f"node 'n' input 'a' call(child) argument 'y' from: reference {_BAD_REF}",
    )


def test_a_bad_ref_in_a_value_entry_call_names_the_argument():
    text = f"""
id: f
name: f
{_child_def()}
nodes:
  v: !str call(child, y={_BAD_REF})
output: ${{v.output}}
"""
    _assert_names(text, "value entry 'v' call(child) argument 'y' from:")


def _case_flow(then: str, else_: str) -> str:
    """Return a flow whose CASE `g` routes to `then` or `else_`."""
    return f"""
id: f
name: f
input:
  s: !str
{_child_def()}
nodes:
  g:
    kind: case
    cases:
      - when: "input.s == 'a'"
        then: {then}
    else: {else_}
  a: {{kind: code, code: "return 'a'", output: str}}
output: ${{g.output}}
"""


def test_a_bad_ref_in_a_case_else_call_names_the_argument():
    _assert_names(
        _case_flow("a", f"call(child, y={_BAD_REF})"),
        "node 'g' else call(child) argument 'y' from:",
    )


def test_a_bad_ref_in_a_case_then_call_names_the_argument():
    _assert_names(
        _case_flow(f"call(child, y={_BAD_REF})", "a"),
        "node 'g' cases[0].then call(child) argument 'y' from:",
    )


def _output_flow(output: str) -> str:
    """Return a flow with one CODE node and the flow `output:` section `output`."""
    return f"""
id: f
name: f
{_child_def()}
nodes:
  n: {{kind: code, code: "return 'a'", output: str}}
output: {output}
"""


def test_a_bad_ref_in_a_single_flow_output_call_names_the_argument():
    _assert_names(
        _output_flow(f"call(child, y={_BAD_REF})"),
        "flow output call(child) argument 'y' from:",
    )


def test_a_bad_ref_in_a_record_flow_output_call_names_the_argument():
    _assert_names(
        _output_flow(f"\n  r: call(child, y={_BAD_REF})"),
        "flow output 'r' call(child) argument 'y' from:",
    )


def test_a_bad_ref_in_a_map_over_call_names_the_argument():
    text = f"""
id: f
name: f
defs:
  lister:
    input:
      n: !str
    nodes:
      x:
        kind: code
        code: "return [inputs['n']]"
        input:
          n: ${{input.n}}
        output: list(str)
    output: ${{x.output}}
  one:
    input:
      topic: !str
    nodes:
      x:
        kind: code
        code: "return inputs['topic']"
        input:
          topic: ${{input.topic}}
        output: str
    output: ${{x.output}}
nodes:
  m:
    kind: map
    call: one
    over: call(lister, n={_BAD_REF})
    input:
      topic: ${{item}}
output: ${{m.output}}
"""
    _assert_names(text, "node 'm' map over call(lister) argument 'n' from:")


def test_a_bad_ref_in_a_flow_assert_call_names_the_argument():
    text = f"""
id: f
name: f
defs:
  check:
    input:
      y: !str
    nodes:
      x:
        kind: code
        code: "return True"
        input:
          y: ${{input.y}}
        output: bool
    output: ${{x.output}}
nodes:
  n: {{kind: code, code: "return 'a'", output: str}}
output: ${{n.output}}
asserts:
  - call(check, y={_BAD_REF})
"""
    _assert_names(
        text,
        f"flow assert 'call(check, y={_BAD_REF})' call(check) argument 'y' from:",
    )


def test_a_nested_call_shares_its_enclosing_binding_site():
    _assert_names(
        _node_input_flow(f"call(child, y=call(child, y={_BAD_REF}))"),
        "node 'n' input 'a' call(child) argument 'y' from:",
    )


def test_a_mis_called_builtin_in_an_argument_names_the_argument():
    _assert_names(
        _node_input_flow("call(child, y=${upper(input.s, 1)})"),
        "node 'n' input 'a' call(child) argument 'y' from: expression function 'upper'",
    )


def test_an_undeclared_argument_names_the_argument():
    _assert_names(
        _node_input_flow("call(child, z=1, y=1)", "int"),
        "node 'n' input 'a' call(child) argument 'z' is not a declared input of "
        "callable 'child'",
    )


def test_a_missing_argument_names_the_call():
    _assert_names(
        _node_input_flow("call(child)", "int"),
        "node 'n' input 'a' call(child): required callable input 'y' is unbound",
    )


def test_an_argument_source_type_mismatch_names_the_argument():
    _assert_names(
        _node_input_flow("call(child, y=${input.s})", "int"),
        "node 'n' input 'a' call(child) argument 'y' — child expects 'int', source is 'str'",
    )


def test_an_argument_literal_type_mismatch_names_the_argument():
    _assert_names(
        _node_input_flow("call(child, y=abc)", "int"),
        "node 'n' input 'a' call(child) argument 'y' — child expects 'int', but the value "
        '"abc" does not fit',
    )


def test_an_undeclared_argument_in_a_value_entry_call_names_the_argument():
    text = f"""
id: f
name: f
{_child_def("int")}
nodes:
  v: !int call(child, z=1, y=1)
output: ${{v.output}}
"""
    _assert_names(
        text,
        "value entry 'v' call(child) argument 'z' is not a declared input of callable 'child'",
    )


def _call_node_flow(inputs: str) -> str:
    """Return a flow whose authored `kind: call` node `c` binds `inputs`."""
    return f"""
id: f
name: f
input:
  s: !str
{_child_def("int")}
nodes:
  c: {{kind: call, call: child, input: {inputs}}}
output: ${{c.output}}
"""


def test_an_authored_call_node_keeps_its_binding_wording():
    assert (
        "call node 'c': binding 'z' is not a declared input of callable 'child'"
        in _load_error(_call_node_flow("{z: 1, y: 1}"))
    )
    assert "call node 'c': binding 'y' — child expects 'int', source is 'str'" in (
        _load_error(_call_node_flow("{y: '${input.s}'}"))
    )


def test_a_cycle_through_an_inline_call_names_its_host_in_a_note():
    text = f"""
id: f
name: f
{_child_def()}
nodes:
  n:
    kind: code
    code: "return inputs['a']"
    input:
      a: call(child, y=${{n.output}})
    output: str
output: ${{n.output}}
"""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    diagnostic = caught.value.diagnostic
    assert diagnostic.message == (
        "flow has a cycle involving ['__call_0', 'n']; flows must be acyclic"
    )
    assert "__call_0 is generated from node 'n'" in diagnostic.notes

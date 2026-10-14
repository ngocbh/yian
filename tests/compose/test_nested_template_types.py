"""Types of nested binding values are inferred from their structure.

A mapping or list binding holding `${...}` spans is a `ValueTemplate`. Its type is
inferred field by field: a nested typed binding gives its declared type, a string
its template type, a literal its kind (its exact value only where a tag checks it).
A tag on the whole value, a CALL child's parameter, a TOOL's stamped input types,
a LOOP's carried record and an inline body's generated parameters all read that
structural type.
"""

from __future__ import annotations

import re

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.run import run_flow
from yian.typesys import canonical_type_label

_TYPEDEFS = """typedefs:
  Rec:
    a: str
    n: int
  Buy:
    conviction: float
  Obj:
    n: int
  Trade:
    action: Literal(buy)
    conviction: float
"""


def _code_flow(binding: str, *, p_output: str = "str", extra: str = "") -> str:
    """Return a flow whose CODE node `c` binds `x` as `binding`, reading `p`."""
    return (
        _TYPEDEFS
        + f"""id: nested_types
name: nested_types
nodes:
  p:
    kind: code
    output: {p_output}
    code: |
      return "hi"
  c:
    kind: code
    input:
      x: {binding}
    code: |
      return inputs["x"]
{extra}output: ${{c.output}}
"""
    )


def _load_error(text: str) -> LoadError:
    """Load `text`, expecting a load error, and return it."""
    with pytest.raises(LoadError) as info:
        load_flow(text)
    return info.value


def _error_line(error: LoadError) -> int:
    """Return the 1-based start line of a load error's innermost frame."""
    return error.diagnostic.frames[-1].range.start_line


def _line_of(text: str, line: str) -> int:
    """Return the 1-based number of the first line of `text` equal to `line`."""
    return text.splitlines().index(line) + 1


# A tag on the whole template.


def test_tagged_template_with_a_reference_field_loads():
    load_flow(_code_flow("!Rec {a: '${p.output}', n: 1}"))


def test_tagged_template_with_a_disjoint_field_fails():
    error = _load_error(_code_flow("!Rec {a: '${p.output}', n: x}"))

    assert error.diagnostic.message == (
        'node \'c\' input \'x\': declared Rec, but the value is '
        '{"a":str,"n":"x"}'
    )


def test_tagged_template_missing_a_required_field_names_it():
    error = _load_error(_code_flow("!Rec {a: '${p.output}'}"))

    assert error.diagnostic.message == (
        'node \'c\' input \'x\': declared Rec, but the value is '
        '{"a":str} (missing required `n`)'
    )


def test_tagged_template_with_a_str_reference_into_a_float_fails():
    error = _load_error(_code_flow("!Buy {conviction: '${p.output}'}"))

    assert error.diagnostic.message == (
        'node \'c\' input \'x\': declared Buy, but the value is '
        '{"conviction":str}'
    )


def test_tagged_template_with_an_optional_reference_fails_at_runtime():
    flow = load_flow(
        _code_flow("!Buy {conviction: '${p.output}'}", p_output="Optional(float)").replace(
            'return "hi"', "return None"
        )
    )

    result = run_flow(flow, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"


def test_nested_typed_binding_is_checked_at_its_own_line():
    text = _code_flow("\n        a: !int '${p.output}'\n        n: 1")

    error = _load_error(text)

    assert error.diagnostic.message == (
        "node 'c' input 'x.a': declared int, but the value is str"
    )
    assert _error_line(error) == _line_of(text, "        a: !int '${p.output}'")


def test_nested_typed_literal_fits_the_outer_tag():
    load_flow(_code_flow("!Rec {a: '${p.output}', n: !int 3}"))


# CALL children, TOOL stamps, and child outputs.


def _call_flow(binding: str, param: str, *, n_output: str = "str") -> str:
    """Return a flow whose CALL node `c` binds the child's `p` as `binding`."""
    return (
        _TYPEDEFS
        + f"""id: nested_call
name: nested_call
defs:
  child:
    input:
      p: {param}
    nodes: {{}}
    output: ${{input.p}}
nodes:
  n:
    kind: code
    output: {n_output}
    code: |
      return "hi"
  c:
    kind: call
    call: child
    input:
      p: {binding}
output: ${{c.output}}
"""
    )


def test_template_into_a_record_child_param_loads():
    load_flow(_call_flow("{a: '${n.output}', n: 1}", "!Rec"))


def test_template_with_a_disjoint_field_into_a_child_param_fails():
    with pytest.raises(LoadError, match=re.escape("call node 'c': binding 'p' — child expects")):
        load_flow(_call_flow("{a: '${n.output}', n: 1}", "!Rec", n_output="int"))


def test_tagged_empty_list_into_a_list_child_param_loads():
    text = _call_flow("{items: !list(str) []}", "!Items").replace(
        "typedefs:\n", "typedefs:\n  Items:\n    items: list(str)\n"
    )

    load_flow(text)


def _tool_flow(binding: str) -> str:
    """Return a flow whose TOOL node `t` binds `v` as `binding`."""
    return f"""id: stamped
name: stamped
input:
  n: !int
  a: !str
  o: !Optional(str)
nodes:
  t:
    kind: tool
    tool_id: place_order
    input:
      v: {binding}
output: ${{t.output}}
"""


@pytest.mark.parametrize(
    ("binding", "label"),
    [
        ("{a: '${input.a}', n: '${input.n}'}", '{"a":str,"n":int}'),
        ("['${input.a}', 'x ${input.n}']", "list(str)"),
        ("['${input.n}', 2]", "list(int)"),
        ("['${input.o}', '${input.a}']", "list(Optional(str))"),
        ("{a: ['${input.a}'], b: '${input.n}'}", '{"a":list(str),"b":int}'),
    ],
)
def test_tool_template_input_is_stamped_with_its_structural_type(binding, label):
    node = load_flow(_tool_flow(binding)).compiled.nodes["t"]

    assert {k: canonical_type_label(t) for k, t in node.input_types.items()} == {
        "v": label
    }


@pytest.mark.parametrize(
    "binding", ["['${input.a}', '${input.n}']", "{a: '${input.a}', b: []}"]
)
def test_tool_template_input_with_no_exact_type_is_absent(binding):
    node = load_flow(_tool_flow(binding)).compiled.nodes["t"]

    assert node.input_types == {}


_CHILD_OUTPUT = """id: nested_child_output
name: nested_child_output
defs:
  child:
    nodes:
      n:
        kind: code
        output: str
        code: |
          return "hi"
    output:
      a:
        b: ${n.output}
nodes:
  sub:
    kind: call
    call: child
  c:
    kind: code
    input:
      x: READ
    code: |
      return inputs["x"]
output: ${c.output}
"""


def test_child_template_output_types_the_parent_read():
    error = _load_error(_CHILD_OUTPUT.replace("READ", "!int ${sub.output.a.b}"))

    assert error.diagnostic.message == (
        "node 'c' input 'x': declared int, but the value is str"
    )


def test_child_template_output_rejects_an_unknown_field_read():
    error = _load_error(_CHILD_OUTPUT.replace("READ", "${sub.output.a.zz}"))

    assert error.diagnostic.message == (
        "flow has unresolved references:\n"
        "  node 'c' input 'x' from: reference ${sub.output.a.zz} reads unknown field "
        "'zz' on record"
    )


_SKIPPED_CHILD_OUTPUT = """id: nested_skipped_child
name: nested_skipped_child
defs:
  child:
    input:
      go: !bool
    nodes:
      gate:
        kind: case
        cases:
          - when: ${input.go}
            then: n
      n:
        kind: code
        output: str
        code: |
          return "hi"
    output:
      a:
        b: ${n.output}
nodes:
  sub:
    kind: call
    call: child
    input:
      go: true
  c:
    kind: code
    input:
      x: !str ${sub.output.a.b}
    code: |
      return inputs["x"]
output: ${c.output}
"""


def test_a_skippable_nested_end_read_is_optional():
    loaded = load_flow(_SKIPPED_CHILD_OUTPUT)

    contract = loaded.compiled.nodes["sub"].output_contract
    assert canonical_type_label(contract.type) == '{"a":{"b":Optional(str)}}'


# LOOP seeds.


_LOOP = """id: nested_loop
name: nested_loop
input:
  start: !int
defs:
  step:
    input:
      n: !int
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: ${bump.output}
nodes:
  repeat:
    kind: loop
    call: step
    times: 2
    input:
      n: ${input.start}
      meta: SEED
output: ${repeat.output}
"""


def test_loop_template_seed_types_the_carried_field():
    loaded = load_flow(_LOOP.replace("SEED", "{first: '${input.start}'}"))

    contract = loaded.compiled.nodes["repeat"].output_contract
    assert canonical_type_label(contract.type) == '{"meta":{"first":int},"n":int}'
    result = run_flow(loaded, {"start": 3})
    assert result.status == "succeeded", result.failure
    assert result.output == {"n": 5, "meta": {"first": 3}}


def _inline_loop(seed: str) -> str:
    """Return a flow whose inline LOOP body returns its carried `state` record."""
    return f"""id: inline_loop
name: inline_loop
input:
  start: !int
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      state: {seed}
    nodes:
      bump:
        kind: code
        input:
          state: ${{input.state}}
        output: STATE_TYPE
        code: |
          return {{"state": {{"n": inputs["state"]["n"] + 1}}}}
    output: ${{bump.output}}
output: ${{repeat.output}}
"""


def test_inline_loop_template_seed_generates_its_parameter():
    text = "typedefs:\n  State:\n    n: int\n" + _inline_loop(
        "{n: '${input.start}'}"
    ).replace("STATE_TYPE", "\n          state: State")

    result = run_flow(load_flow(text), {"start": 1})

    assert result.status == "succeeded", result.failure
    assert result.output == {"state": {"n": 3}}


def test_inline_loop_template_seed_mismatch_suggests_a_tag():
    text = "typedefs:\n  State:\n    n: float\n" + _inline_loop(
        "{n: '${input.start}'}"
    ).replace("STATE_TYPE", "\n          state: State")

    error = _load_error(text)

    assert (
        'the body returns \'state\' as {"n":float}, but the carried field is typed '
        '{"n":int} from its seed; declare the seed\'s type: `state: !State '
    ) in error.diagnostic.message


# Inline bodies.


def _inline_call(binding: str) -> str:
    """Return a flow whose inline CALL body reads `payload` bound as `binding`."""
    return f"""id: inline_call
name: inline_call
input:
  text: !str
nodes:
  c:
    kind: call
    input:
      payload: {binding}
    nodes:
      echo:
        kind: code
        input:
          x: ${{input.payload}}
        code: |
          return inputs["x"]
    output: ${{echo.output}}
output: ${{c.output}}
"""


def test_inline_call_template_arg_generates_its_parameter():
    loaded = load_flow(_inline_call("{text: '${input.text}', tags: ['a', 'b']}"))

    (param,) = loaded.compiled.nodes["c"].child_inputs
    assert canonical_type_label(param.contract.type) == '{"tags":list(str),"text":str}'
    result = run_flow(loaded, {"text": "hi"})
    assert result.status == "succeeded", result.failure
    assert result.output == {"text": "hi", "tags": ["a", "b"]}


def test_inline_call_template_arg_with_a_nested_tag_uses_it():
    loaded = load_flow(_inline_call("{text: '${input.text}', items: !list(str) []}"))

    (param,) = loaded.compiled.nodes["c"].child_inputs
    assert canonical_type_label(param.contract.type) == '{"items":list(str),"text":str}'


def test_inline_call_template_arg_with_no_proven_type_suggests_a_tag():
    error = _load_error(_inline_call("{text: '${input.text}', items: []}"))

    assert error.diagnostic.message == (
        "cannot infer type for inline binding 'c.input.payload': empty literal list "
        "at c.input.payload['items'] has no provable element type; declare its type "
        "with a tag (`payload: !<type> ...`) or use an explicitly typed named `defs:` body"
    )


def test_nested_typed_binding_in_a_flow_output_is_named_by_its_path():
    text = _code_flow("${p.output}").replace(
        "output: ${c.output}", "output:\n  r:\n    a: !int ${p.output}"
    )

    error = _load_error(text)

    assert error.diagnostic.message == (
        "output 'r.a': declared int, but the value is str"
    )
    assert _error_line(error) == _line_of(text, "    a: !int ${p.output}")


_MAP = _TYPEDEFS + """  Row:
    name: int
id: nested_map
name: nested_map
input:
  rows: !list(Row)
defs:
  child:
    input:
      v: !object
    nodes: {}
    output: ${input.v}
nodes:
  m:
    kind: map
    call: child
    over: ${input.rows}
    input:
      v: !Rec {a: '${item.name}', n: 1}
output: ${m.output}
"""


def test_nested_item_read_is_typed_from_the_map_element():
    error = _load_error(_MAP)

    assert error.diagnostic.message == (
        'node \'m\' input \'v\': declared Rec, but the value is '
        '{"a":int,"n":1}'
    )


def test_inline_call_template_list_of_mixed_types_fails():
    error = _load_error(_inline_call("['${input.text}', 1]"))

    assert (
        "list at c.input.payload has heterogeneous element types" in error.diagnostic.message
    )


def test_inline_call_template_list_joins_optional_items():
    text = _inline_call("['${input.text}', '${input.note}']").replace(
        "  text: !str\n", "  text: !str\n  note: !Optional(str)\n", 1
    )

    (param,) = load_flow(text).compiled.nodes["c"].child_inputs

    assert canonical_type_label(param.contract.type) == "list(Optional(str))"


# Nested typed containers with no strings.


@pytest.mark.parametrize(
    "binding",
    [
        "{k: '${p.output}', inner: !Obj {n: !int 3}}",
        "{k: '${p.output}', xs: !list(int) [!int 3]}",
        "!list(int) [!int 3]",
    ],
)
def test_nested_typed_container_holding_only_bindings_loads(binding):
    load_flow(_code_flow(binding))


def test_inline_call_nested_typed_container_holding_only_bindings_loads():
    text = _TYPEDEFS + _inline_call("{inner: !Obj {n: !int 3}}")

    result = run_flow(load_flow(text), {"text": "hi"})

    assert result.status == "succeeded", result.failure
    assert result.output == {"inner": {"n": 3}}


# Literal leaves.


_LITERAL_LOOP = """id: literal_loop
name: literal_loop
input:
  start: !int
defs:
  step:
    nodes:
      bump:
        kind: code
        output: int
        code: |
          return 7
    output:
      st: {phase: scan, n: '${bump.output}'}
nodes:
  repeat:
    kind: loop
    call: step
    times: 2
    input:
      st: {phase: init, n: '${input.start}'}
output: ${repeat.output}
"""


def test_loop_seed_differing_from_the_body_only_in_a_literal_runs():
    result = run_flow(load_flow(_LITERAL_LOOP), {"start": 1})

    assert result.status == "succeeded", result.failure
    assert result.output == {"st": {"phase": "scan", "n": 7}}


def test_wrong_literal_value_under_a_tag_fails_at_load():
    error = _load_error(
        _code_flow("!Trade {action: sell, conviction: '${p.output}'}", p_output="float")
    )

    assert error.diagnostic.message == (
        "node 'c' input 'x': declared Trade, "
        'but the value is {"action":"sell","conviction":float}'
    )


# Partial templates.


def _untyped_p(text: str) -> str:
    """Drop `p`'s declared output type and make it return an int."""
    return text.replace("    output: str\n", "", 1).replace('return "hi"', "return 5")


def test_partial_template_under_a_tag_is_checked_at_runtime():
    flow = load_flow(_untyped_p(_code_flow("!Rec {a: '${p.output}', n: 1}")))

    result = run_flow(flow, {})

    assert result.status == "failed"
    assert result.failure.code == "binding.type"


def test_partial_template_into_a_child_param_is_checked_at_runtime():
    text = _call_flow("{a: '${n.output}', n: 1}", "!Rec").replace(
        "    output: str\n", "", 1
    ).replace('return "hi"', "return 5")

    result = run_flow(load_flow(text), {})

    assert result.status == "failed"

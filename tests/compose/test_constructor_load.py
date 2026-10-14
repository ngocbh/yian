"""`${T(...)}` constructors resolve at load into each node's type table.

A constructor name is a `typedefs:` name; `T(x)` checks a value (exactly `!T ${x}`), and
`T(a=..., b=...)` builds a record from exactly the given fields and checks it.
"""

import re

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow
from yian.typesys.types import read_typedefs, type_for

_REGISTRY = read_typedefs(
    {
        "Buy": {"action": "Literal(buy)", "conviction": "float"},
        "Sell": {"action": "Literal(sell)", "price": "float"},
        "Decision": "Union(Buy, Sell)",
        "Size": "Literal(s, m, l)",
    }
)
BUY = type_for("Buy", _REGISTRY)

TYPEDEFS = """typedefs:
  Buy: {action: Literal(buy), conviction: float}
  Sell: {action: Literal(sell), price: float}
  Decision: Union(Buy, Sell)
  Size: Literal(s, m, l)
"""

A_BUY = {"action": "buy", "conviction": 0.5}
A_SELL = {"action": "sell", "price": 1.5}

# One CODE node `d` that returns the untyped `order` input, so only a check can narrow it.
_HEAD = (
    "id: p\nname: p\n"
    + TYPEDEFS
    + """input:
  order: !object
nodes:
  d:
    kind: code
    input:
      order: ${input.order}
    output: object
    code: |
      return inputs["order"]
"""
)


def _flow(rest: str) -> str:
    return _HEAD + rest


def _consumer(binding: str) -> str:
    """A flow whose node `c` binds `x: <binding>` and returns it."""
    return _flow(
        f"""  c:
    kind: code
    input:
      x: {binding}
    code: |
      return inputs["x"]
output: ${{c.output}}
"""
    )


def _line_of(text: str, marker: str) -> int:
    for number, line in enumerate(text.splitlines(), start=1):
        if marker in line:
            return number
    raise AssertionError(f"{marker!r} not in text")


def _load_error(text: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    return caught.value


def _error_line(error: LoadError) -> int:
    return error.diagnostic.frames[-1].range.start_line


# --- resolution and evaluation --- #


def test_a_constructor_resolves_into_the_node_type_table():
    flow = load_flow(_consumer("${Buy(d.output)}"))
    assert flow.compiled.nodes["c"].types == {"Buy": BUY}
    assert flow.compiled.nodes["d"].types == {}


@pytest.mark.parametrize(
    "binding, rewrite",
    [
        ("${Buy(${d.output})}", "${Buy(d.output)}"),
        (
            '${Buy(action="buy", conviction=${d.output.conviction})}',
            '${Buy(action="buy", conviction=d.output.conviction)}',
        ),
    ],
)
def test_a_span_inside_a_constructor_shows_the_rewrite(binding, rewrite):
    error = _load_error(_consumer(f"'{binding}'"))
    assert str(error.diagnostic.message) == (
        f"c: `${{...}}` can't appear inside an expression; write `{rewrite}`"
    )


def test_a_value_that_fits_passes_unchanged():
    result = run_flow(load_flow(_consumer("${Buy(d.output)}")), {"order": A_BUY})
    assert result.status == "succeeded", result.failure
    assert result.output == A_BUY


def test_a_value_lacking_a_field_fails_with_binding_type_naming_it():
    text = _consumer("${Buy(d.output)}")
    flow = load_flow(text)
    result = run_flow(flow, {"order": {"conviction": 0.5}})
    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "c"
    assert "record missing required fields: ['action']" in result.failure.message
    frame = flow.diagnose(result.failure).frames[-1]
    assert frame.range.start_line == _line_of(text, "x: ${Buy")


def test_keywords_build_the_record():
    text = _consumer('${Buy(action="buy", conviction=d.output.conviction)}')
    result = run_flow(load_flow(text), {"order": A_BUY})
    assert result.status == "succeeded", result.failure
    assert result.output == A_BUY


def test_a_bad_built_field_fails_with_binding_type():
    text = _consumer('${Buy(action="buy", conviction=d.output.price)}')
    result = run_flow(load_flow(text), {"order": {"price": "high"}})
    assert result.failure.code == "binding.type"
    assert "does not match declared type float" in result.failure.message


def test_a_union_typedef_and_an_enum_alias_check():
    text = _flow(
        """  c:
    kind: code
    input:
      x: ${Decision(d.output.decision)}
      size: ${Size(d.output.size)}
    code: |
      return [inputs["x"]["action"], inputs["size"]]
output: ${c.output}
"""
    )
    result = run_flow(load_flow(text), {"order": {"decision": A_SELL, "size": "m"}})
    assert result.status == "succeeded", result.failure
    assert result.output == ["sell", "m"]
    failed = run_flow(load_flow(text), {"order": {"decision": A_SELL, "size": "xl"}})
    assert failed.failure.code == "binding.type"


def test_isinstance_with_a_builtin_name_does_not_make_it_a_constructor():
    text = _consumer("${isinstance(d.output, int)}")
    assert load_flow(text).compiled.nodes["c"].types == {"int": type_for("int", {})}


# --- positions --- #


def test_constructors_in_a_list_literal_and_a_nested_value():
    text = _flow(
        """  c:
    kind: code
    input:
      xs: ${[Buy(d.output), Buy(d.output)]}
      nested:
        first: ${Buy(d.output).conviction}
        note: 'conviction ${Buy(d.output).conviction}'
    code: |
      return [inputs["xs"], inputs["nested"]]
output: ${c.output}
"""
    )
    result = run_flow(load_flow(text), {"order": A_BUY})
    assert result.status == "succeeded", result.failure
    assert result.output == [[A_BUY, A_BUY], {"first": 0.5, "note": "conviction 0.5"}]


def test_a_nested_constructor_failure_is_binding_type():
    text = _flow(
        """  c:
    kind: code
    input:
      nested:
        first: ${Buy(d.output)}
    code: |
      return inputs["nested"]
output: ${c.output}
"""
    )
    result = run_flow(load_flow(text), {"order": A_SELL})
    assert result.failure.code == "binding.type"
    assert "binding 'nested.first'" in result.failure.message


_CASE_FLOW = _flow(
    """  route:
    kind: case
    cases:
      - when: Buy(d.output).conviction > 0.4
        then: strong
    else: weak
  strong:
    kind: code
    code: return "strong"
  weak:
    kind: code
    code: return "weak"
output: ${strong.output | weak.output}
"""
)


@pytest.mark.parametrize(
    ("order", "expected"),
    [(A_BUY, "strong"), ({"action": "buy", "conviction": 0.1}, "weak")],
)
def test_a_constructor_in_a_case_when(order, expected):
    flow = load_flow(_CASE_FLOW)
    assert flow.compiled.nodes["route"].types == {"Buy": BUY}
    result = run_flow(flow, {"order": order})
    assert result.status == "succeeded", result.failure
    assert result.output == expected


def test_a_constructor_in_node_and_flow_asserts():
    text = _flow(
        """  c:
    kind: code
    input:
      x: ${d.output}
    asserts:
      - Buy(x).conviction > 0.1
    code: |
      return inputs["x"]
output: ${c.output}
asserts:
  - ${Buy(c.output).conviction < 0.9}
"""
    )
    flow = load_flow(text)
    assert run_flow(flow, {"order": A_BUY}).status == "succeeded"
    assert run_flow(flow, {"order": {"action": "buy", "conviction": 0.95}}).status == "failed"
    assert run_flow(flow, {"order": {"action": "buy", "conviction": 0.05}}).status == "failed"


def test_a_constructor_in_a_flow_output():
    text = _flow("output:\n  decision: ${Buy(d.output)}\n")
    result = run_flow(load_flow(text), {"order": A_BUY})
    assert result.status == "succeeded", result.failure
    assert result.output == {"decision": A_BUY}


def test_a_constructor_in_a_map_per_item_input():
    text = (
        "id: m\nname: m\n"
        + TYPEDEFS
        + """input:
  orders: !list(object)
defs:
  one:
    input:
      x: !object
    nodes:
      e:
        kind: code
        input:
          x: ${input.x}
        code: |
          return inputs["x"]["conviction"]
    output: ${e.output}
nodes:
  each:
    kind: map
    call: one
    over: ${input.orders}
    input:
      x: ${Buy(item)}
output: ${each.output}
"""
    )
    result = run_flow(load_flow(text), {"orders": [A_BUY, A_BUY]})
    assert result.status == "succeeded", result.failure
    assert result.output == [0.5, 0.5]


_USES_CHILD = """id: child
name: child
typedefs:
  Hold: {action: Literal(hold)}
input:
  order: !object
nodes:
  c:
    kind: code
    input:
      x: ${Hold(input.order)}
    code: |
      return inputs["x"]
output: ${c.output}
"""

_USES_PARENT = """id: parent
name: parent
input:
  order: !object
uses:
  child: child
nodes:
  run:
    kind: call
    call: child
    input:
      order: ${input.order}
output: ${run.output}
"""


def _uses_resolver():
    loaded = {"child": load_flow(_USES_CHILD)}
    return lambda flow_id, version=None: loaded[flow_id]


def test_a_uses_child_resolves_constructors_against_its_own_typedefs():
    flow = load_flow(_USES_PARENT, child_resolver=_uses_resolver())
    assert "Hold" in flow.compiled.nodes["run"].child.nodes["c"].types
    result = run_flow(flow, {"order": {"action": "hold"}})
    assert result.status == "succeeded", result.failure
    assert result.output == {"action": "hold"}


def test_a_flow_call_with_several_arguments_keeps_the_flow_call_message():
    error = _load_error(_consumer("${review(d.output, d.output)}"))
    assert "a flow call 'review' inside ${...} is not supported" in str(error)


@pytest.mark.parametrize("site", ["binding", "case when"])
def test_a_malformed_constructor_call_is_a_load_error(site):
    if site == "binding":
        text = _consumer('${Buy(d.output, action="buy")}')
    else:
        text = _CASE_FLOW.replace("Buy(d.output)", "Buy(d.output, d.output)")
    error = _load_error(text)
    assert (
        "Buy(...) takes keyword fields, as in Buy(name=value, ...), or one value to "
        "check, as in Buy(x)" in str(error)
    )


def test_a_parent_cannot_construct_a_childs_typedef():
    text = _USES_PARENT.replace(
        "      order: ${input.order}\n",
        "      order: ${input.order}\n  probe:\n    kind: code\n    input:\n"
        "      x: ${Hold(input.order)}\n    code: |\n      return 1\n",
    )
    with pytest.raises(LoadError) as caught:
        load_flow(text, child_resolver=_uses_resolver())
    assert "a flow call 'Hold' inside ${...} is not supported" in str(caught.value)


# --- names that are not constructors --- #


def test_a_builtin_type_name_in_a_binding_is_not_a_conversion():
    text = _consumer("${int(d.output)}")
    error = _load_error(text)
    assert (
        "unknown expression function 'int'; types in expressions are typedef names; "
        "`int(...)` is not a conversion" in str(error)
    )
    assert _error_line(error) == _line_of(text, "x: ${int")


def test_a_builtin_type_name_in_a_case_when_is_not_a_conversion():
    text = _CASE_FLOW.replace("Buy(d.output).conviction > 0.4", "int(d.output) > 0")
    error = _load_error(text)
    assert "case node 'route' cases[0].when: unknown expression function 'int'; types" in str(
        error
    )
    assert _error_line(error) == _line_of(text, "int(d.output)")


def test_an_unknown_name_in_a_case_when_is_an_unknown_function():
    text = _CASE_FLOW.replace("Buy(d.output)", "Hold(d.output)")
    error = _load_error(text)
    assert re.search(r"unknown expression function 'Hold'$", str(error).splitlines()[0])


def test_a_constructor_in_wait_until_is_a_located_load_error():
    text = (
        "id: w\nname: w\n"
        + TYPEDEFS
        + """input:
  when: !date
nodes:
  w:
    kind: wait
    until: ${Size(input.when)}
output: ${w.output}
"""
    )
    error = _load_error(text)
    assert "Size(...) is not supported in a wait node's until:" in str(error)
    assert _error_line(error) == _line_of(text, "until:")


def test_a_constructor_in_a_call_nodes_asserts_is_a_located_load_error():
    text = (
        "id: p\nname: p\n"
        + TYPEDEFS
        + """input:
  order: !object
defs:
  check:
    input:
      order: !object
    nodes:
      e:
        kind: code
        input:
          order: ${input.order}
        code: |
          return inputs["order"]
    output: ${e.output}
nodes:
  run:
    kind: call
    call: check
    input:
      order: ${input.order}
    asserts:
      - Buy(output).conviction > 0
output: ${run.output}
"""
    )
    error = _load_error(text)
    assert (
        "Buy(...) in a call node's asserts is not supported; assert inside the child flow"
        in str(error)
    )
    assert _error_line(error) == _line_of(text, "Buy(output)")


def test_a_constructor_in_an_agent_prompt_is_a_located_load_error():
    text = (
        "id: a\nname: a\n"
        + TYPEDEFS
        + """input:
  order: !object
nodes:
  ask:
    kind: agent
    input:
      order: ${input.order}
    prompt: "Is it a buy? ${Buy(order)}"
    output: str
output: ${ask.output}
"""
    )
    error = _load_error(text)
    assert "unknown prompt function 'Buy'" in str(error)
    assert (
        "build or check a typed value with a `typedefs:` constructor `T(...)` in an input "
        "binding" in str(error)
    )
    assert _error_line(error) == _line_of(text, "prompt:")


# --- static types --- #


def _typed_head(d_output: str) -> str:
    """Like `_HEAD`, with `d` declaring `d_output`."""
    return _HEAD.replace("    output: object\n", f"    output: {d_output}\n", 1)


def test_a_field_read_on_a_constructor_result_type_checks():
    load_flow(_consumer("${Buy(d.output).conviction}"))
    text = _consumer("${Buy(d.output).size}")
    error = _load_error(text)
    assert "reference ${Buy(...).size} reads unknown field 'size' on record" in str(error)
    assert _error_line(error) == _line_of(text, "x: ${Buy")


def test_a_field_read_on_a_constructor_in_a_case_when_type_checks():
    text = _CASE_FLOW.replace("Buy(d.output).conviction", "Buy(d.output).size")
    error = _load_error(text)
    assert "reads unknown field 'size' on record" in str(error)
    assert _error_line(error) == _line_of(text, "Buy(d.output).size")


def test_a_disjoint_argument_is_a_load_error():
    text = _typed_head("Sell") + """  c:
    kind: code
    input:
      x: ${Buy(d.output)}
    code: |
      return inputs["x"]
output: ${c.output}
"""
    error = _load_error(text)
    assert (
        'Buy(...): the value is {"action":Literal("sell"),"price":float}, which can never '
        "be a Buy" in str(error)
    )
    assert _error_line(error) == _line_of(text, "x: ${Buy")


def test_a_narrower_or_unknown_argument_loads():
    load_flow(_typed_head("Decision") + "output: ${Buy(d.output)}\n")
    load_flow(_typed_head("Optional(Buy)") + "output: ${Buy(d.output)}\n")


def test_keywords_missing_a_required_field_are_a_load_error():
    text = _consumer("${Buy(conviction=d.output.conviction)}")
    error = _load_error(text)
    assert "Buy(...) is missing required field `action`" in str(error)
    assert _error_line(error) == _line_of(text, "x: ${Buy")


def test_keywords_missing_a_field_in_a_case_when_are_a_load_error():
    text = _CASE_FLOW.replace("Buy(d.output)", "Buy(conviction=d.output.conviction)")
    assert "Buy(...) is missing required field `action`" in str(_load_error(text))


def test_an_unknown_keyword_field_is_a_load_error():
    text = _consumer('${Buy(action="buy", conviction=1, size="m")}')
    assert "Buy has no field 'size' (fields: action, conviction)" in str(_load_error(text))


@pytest.mark.parametrize(
    ("binding", "message"),
    [
        (
            '${Buy(action="sell", conviction=1)}',
            "Buy(...) field 'action': declared Literal(\"buy\"), but the value is Literal(\"sell\")",
        ),
        (
            '${Buy(action="buy", conviction=d.output)}',
            "Buy(...) field 'conviction': declared float, but the value is str",
        ),
    ],
)
def test_a_disjoint_keyword_field_is_a_load_error(binding, message):
    text = _typed_head("str") + f"""  c:
    kind: code
    input:
      x: {binding}
    code: |
      return inputs["x"]
output: ${{c.output}}
"""
    assert message in str(_load_error(text))


def test_keywords_cannot_build_a_union():
    error = _load_error(_consumer("${Decision(conviction=1)}"))
    assert (
        "Decision is a union; construct a member (Buy(...) or Sell(...)), or check a "
        "value with Decision(x)" in str(error)
    )


def test_keywords_cannot_build_a_scalar_type():
    error = _load_error(_consumer('${Size(name="m")}'))
    assert "Size is not a record, so Size(...) takes one value to check, as in Size(x)" in str(
        error
    )


def test_downstream_sees_the_constructed_type():
    # A typed binding over a constructor compares the declared type with the constructor's.
    text = _consumer("!Sell ${Buy(d.output)}")
    assert (
        'declared Sell, but the value is '
        '{"action":Literal("buy"),"conviction":float}' in str(_load_error(text))
    )



def test_a_call_argument_is_checked_with_the_constructed_type():
    text = _flow(
        """  run:
    kind: call
    call: sell
    input:
      order: ${Buy(d.output)}
output: ${run.output}
"""
    ).replace(
        "nodes:\n",
        """defs:
  sell:
    input:
      order: !Sell
    nodes:
      e:
        kind: code
        input:
          order: ${input.order}
        code: |
          return inputs["order"]
    output: ${e.output}
nodes:
""",
        1,
    )
    error = _load_error(text)
    assert "call node 'run': binding 'order'" in str(error)


_PARENT = (
    "id: p\nname: p\n"
    + TYPEDEFS
    + """input:
  order: !object
defs:
  decide:
    input:
      order: !object
    nodes:
      e:
        kind: code
        input:
          order: ${input.order}
        code: |
          return inputs["order"]
    output: ${Buy(e.output)}
nodes:
  run:
    kind: call
    call: decide
    input:
      order: ${input.order}
  use:
    kind: code
    input:
      conviction: ${run.output.conviction}
    code: |
      return inputs["conviction"]
output: ${use.output}
"""
)


def test_a_caller_sees_a_childs_constructed_output_type():
    result = run_flow(load_flow(_PARENT), {"order": A_BUY})
    assert result.status == "succeeded", result.failure
    assert result.output == 0.5
    text = _PARENT.replace("${run.output.conviction}", "${run.output.size}")
    assert "reads unknown field 'size' on record" in str(_load_error(text))


def test_an_inline_body_parameter_takes_the_constructed_type():
    text = _flow(
        """  inline:
    kind: call
    input:
      decision: ${Buy(d.output)}
    nodes:
      e:
        kind: code
        input:
          c: ${input.decision.conviction}
        code: |
          return inputs["c"]
    output: ${e.output}
output: ${inline.output}
"""
    )
    flow = load_flow(text)
    child = flow.compiled.nodes["inline"].child
    child = getattr(child, "compiled", child)
    assert child.nodes[child.start_id].params[0].contract.type == BUY
    result = run_flow(flow, {"order": A_BUY})
    assert result.status == "succeeded", result.failure
    assert result.output == 0.5

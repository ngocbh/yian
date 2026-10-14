"""Loading tagged bindings: a top-level tag types a node input or a flow output.

`qty: !int ${size.output.qty}` declares the binding's type with a YAML tag. The tag
only checks and narrows: a literal is checked at load, a reference is rejected when
its inferred type is disjoint from the tag, and downstream checks see the tagged
type. A tagged whole `output:` is one typed output, whatever its value.
"""

from __future__ import annotations

import re

import pytest

from yian.compile.manifest import flow_fingerprint
from yian.compile.model import START_ID
from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.diagnostics import Failure
from yian.nodes.contracts import Binding, ValueContract
from yian.typesys import TypeRegistry, type_for
from yian.typesys.types import read_typedefs

_INT = type_for("int", TypeRegistry())
_REGISTRY = read_typedefs(
    {
        "Size": {"qty": "int"},
        "T": {"a": "int"},
        "Buy": {"action": "Literal(buy)", "conviction": "float"},
    }
)


def _binding(compiled, node_id: str, name: str) -> Binding:
    """Return one node's exact wiring binding by parameter name."""
    binding = compiled.wiring[node_id][name]
    assert type(binding) is Binding
    return binding


def _outputs(compiled) -> dict:
    """Return the compiled flow's outputs keyed by name."""
    return {output.name: output for output in compiled.outputs}


def _child(compiled, node_id: str):
    """Return the compiled child flow baked onto a CALL, MAP, or LOOP node."""
    child = compiled.nodes[node_id].child
    return getattr(child, "compiled", child)


def _start_params(compiled, node_id: str) -> dict:
    """Return the START parameters of the child flow on `node_id`, keyed by name."""
    child = _child(compiled, node_id)
    return {param.name: param for param in child.nodes[START_ID].params}


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


def _code_flow(binding: str) -> str:
    """Return a flow whose CODE node `use` binds `qty` as `binding`."""
    return f"""
id: tagged
name: tagged
input:
  n: !int
nodes:
  size:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${{input.n}}
    output:
      qty: int
  note:
    kind: code
    code: |
      return "hi"
    output: str
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      qty: {binding}
    output: int
output: ${{use.output}}
"""


# --- node inputs ------------------------------------------------------------ #


def test_tagged_code_input_carries_its_contract():
    loaded = load_flow(_code_flow("!int ${size.output.qty}"))

    binding = _binding(loaded.compiled, "use", "qty")
    assert binding.source == "${size.output.qty}"
    assert binding.contract.type == _INT


def test_tagged_input_infers_the_same_edges_as_the_plain_binding():
    tagged = load_flow(_code_flow("!int ${size.output.qty}"))
    plain = load_flow(_code_flow("${size.output.qty}"))

    def edges_from_size(loaded):
        return [edge for edge in loaded.compiled.edges if edge.from_ == "size"]

    assert edges_from_size(tagged)
    assert edges_from_size(tagged) == edges_from_size(plain)
    assert flow_fingerprint(tagged.compiled) != flow_fingerprint(plain.compiled)


def test_tagged_contract_origin_points_at_the_binding_line():
    text = _code_flow("!int ${size.output.qty}")
    loaded = load_flow(text)
    contract = _binding(loaded.compiled, "use", "qty").contract

    diagnostic = loaded.diagnose(
        Failure(
            code="binding.type",
            message="x",
            node_id="use",
            trace=(contract.origins.default,),
        )
    )

    assert diagnostic.frames[-1].range.start_line == _line_of(
        text, "      qty: !int ${size.output.qty}"
    )


def test_tagged_literal_input_binds_its_value():
    binding = _binding(load_flow(_code_flow("!int 0")).compiled, "use", "qty")

    assert binding.source == 0
    assert type(binding.source) is int
    assert binding.contract.type == _INT


def test_tagged_literal_that_does_not_fit_fails_at_its_line():
    text = _code_flow("!int hello")

    error = _load_error(text)

    assert str(error.diagnostic.message) == "'qty': `hello` is not a valid int"
    assert _error_line(error) == _line_of(text, "      qty: !int hello")


def test_former_boolean_word_after_a_bool_tag_names_the_boolean_to_write():
    text = _code_flow("!bool yes")

    error = _load_error(text)

    assert str(error.diagnostic.message) == (
        "'qty': `yes` is not a valid bool; `yes` is text; write `true`"
    )
    assert _error_line(error) == _line_of(text, "      qty: !bool yes")


def test_malformed_unquoted_call_after_a_tag_is_a_load_error():
    text = _code_flow("!str call(f) and more")

    error = _load_error(text)

    assert str(error.diagnostic.message) == (
        "use: text follows the closing `)`: 'and more'; a call(...) must be the whole value"
    )


def test_quoted_call_text_after_a_tag_is_text_checked_against_the_tag():
    binding = _binding(load_flow(_code_flow('!str "call(f) and more"')).compiled, "use", "qty")

    assert binding.source == "call(f) and more"
    with pytest.raises(LoadError, match=re.escape("""'qty': `"call(f)"` is not a valid int""")):
        load_flow(_code_flow('!int "call(f)"'))


def test_tag_without_a_value_asks_for_one():
    with pytest.raises(
        LoadError,
        match=re.escape("'qty': a binding needs a value: `qty: !int ${...}`"),
    ):
        load_flow(_code_flow("!int"))


def test_disjoint_tagged_input_fails_at_its_line():
    text = _code_flow("!int ${note.output}")

    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "node 'use' input 'qty': declared int, but the value is str"
    )
    assert _error_line(error) == _line_of(text, "      qty: !int ${note.output}")


def test_unknown_tag_on_an_input_offers_a_close_match():
    with pytest.raises(LoadError, match=re.escape("Did you mean `int`?")):
        load_flow(_code_flow("!itn ${size.output.qty}"))


def test_recursive_alias_in_a_binding_is_a_located_load_error():
    text = _code_flow("&a [1, *a]")

    error = _load_error(text)

    assert str(error.diagnostic.message) == (
        "'qty': cannot descend through a recursive alias occurrence"
    )
    assert _error_line(error) == _line_of(text, "      qty: &a [1, *a]")


def test_input_name_read_as_a_bool_is_a_located_load_error():
    text = _code_flow("1").replace("      qty: 1", "      true: 1")

    error = _load_error(text)

    assert str(error.diagnostic.message) == (
        "the key `true` was read by YAML as a boolean, but names in a flow must be "
        'text. Quote it ("true") or choose another name.'
    )
    assert _error_line(error) == _line_of(text, "      true: 1")


def test_tagged_inline_call_binding_references_the_synthetic_call():
    loaded = load_flow(
        """
id: tagged_call
name: tagged_call
defs:
  f:
    input:
      a: !int
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.a}
        output: int
    output: ${x.output}
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      v: !int call(f, a=1)
    output: int
output: ${use.output}
"""
    )

    binding = _binding(loaded.compiled, "use", "v")
    assert binding.source == "${__call_0.output}"
    assert binding.contract.type == _INT


_CASE_FLOW = """
id: tagged_case
name: tagged_case
typedefs:
  Decision: Literal(go, stop)
input:
  flag: !bool
nodes:
  route:
    kind: case
    cases:
      - when: ${{input.flag}}
        then: go
    else: stop
  go:
    kind: code
    code: tests.engine._compose_codefns:pick_pro
    input:
      flag: ${{input.flag}}
    output: Decision
  stop:
    kind: code
    code: tests.engine._compose_codefns:pick_pro
    input:
      flag: ${{input.flag}}
    output: Decision
{use}output: {output}
"""

_CASE_USE = """  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      x: !Decision ${route.output}
    output: str
"""


def test_tagged_case_output_binding_expands_and_keeps_its_contract():
    loaded = load_flow(_CASE_FLOW.format(use=_CASE_USE, output="${use.output}"))

    binding = _binding(loaded.compiled, "use", "x")
    assert binding.source == "${go.output | stop.output}"
    assert binding.contract.type == loaded.compiled.nodes["go"].output_contract.type


def test_tagged_agent_input_carries_its_contract():
    loaded = load_flow(
        """
id: tagged_agent
name: tagged_agent
nodes:
  src:
    kind: code
    code: |
      return 3
  ask:
    kind: agent
    input:
      count: !int ${src.output}
    prompt: "Count ${count}"
    output: str
output: ${ask.output}
"""
    )

    assert _binding(loaded.compiled, "ask", "count").contract.type == _INT


def test_tagged_tool_input_carries_its_contract():
    loaded = load_flow(
        """
id: tagged_tool
name: tagged_tool
input:
  n: !int
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: !int ${input.n}
output: ${place.output}
"""
    )

    assert _binding(loaded.compiled, "place", "qty").contract.type == _INT


def test_tagged_human_input_context_carries_its_contract():
    loaded = load_flow(
        """
id: tagged_gate
name: tagged_gate
nodes:
  src:
    kind: code
    code: |
      return 3
  ask:
    kind: human_input
    input:
      count: !int ${src.output}
    prompt: "Proceed with ${count} items?"
output: ${ask.output}
"""
    )

    assert _binding(loaded.compiled, "ask", "count").contract.type == _INT


_STR_CHILD = """
defs:
  shout:
    input:
      y: !str
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${input.y}
    output: ${echo.output}
"""


def test_disjoint_tagged_call_arg_reports_the_binding():
    text = f"""
id: tagged_call_arg
name: tagged_call_arg
{_STR_CHILD}nodes:
  a:
    kind: code
    output: str
    code: |
      return "hi"
  run:
    kind: call
    call: shout
    input:
      y: !int ${{a.output}}
output: ${{run.output}}
"""
    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "node 'run' input 'y': declared int, but the value is str"
    )


def test_tagged_call_arg_narrows_a_nullable_source_into_a_strict_child():
    loaded = load_flow(
        """
id: tagged_narrow
name: tagged_narrow
defs:
  child:
    input:
      x: !int
    nodes:
      twice:
        kind: code
        input:
          x: ${input.x}
        output: int
        code: |
          return inputs["x"] * 2
    output: ${twice.output}
nodes:
  maybe:
    kind: code
    output: Optional(int)
    code: |
      return 3
  run:
    kind: call
    call: child
    input:
      x: !int ${maybe.output}
output: ${run.output}
"""
    )

    assert _binding(loaded.compiled, "run", "x").contract.type == _INT


def _map_flow(binding: str) -> str:
    """Return a flow that maps an inline body over `list(int)`, binding `t` as `binding`."""
    return f"""
id: tagged_map
name: tagged_map
input:
  words: !list(int)
nodes:
  m:
    kind: map
    over: ${{input.words}}
    input:
      t: {binding}
    nodes:
      echo:
        kind: code
        code: tests.engine._compose_codefns:echo_x
        input:
          x: ${{input.t}}
    output: ${{echo.output}}
output: ${{m.output}}
"""


def test_disjoint_tagged_map_item_binding_fails():
    error = _load_error(_map_flow("!str ${item}"))

    assert str(error.diagnostic.message).startswith(
        "node 'm' input 't': declared str, but the value is int"
    )


def test_fitting_tagged_map_item_binding_types_the_body_parameter():
    loaded = load_flow(_map_flow("!int ${item}"))

    assert _binding(loaded.compiled, "m", "t").contract.type == _INT
    assert _start_params(loaded.compiled, "m")["t"].contract.type == _INT


def _loop_flow(seed: str) -> str:
    """Return a flow whose LOOP seeds `n` as `seed` into a body returning `n: int`."""
    return f"""
id: tagged_loop
name: tagged_loop
defs:
  bump:
    input:
      n: !int
    nodes:
      step:
        kind: code
        input:
          n: ${{input.n}}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      n: ${{step.output}}
nodes:
  note:
    kind: code
    output: str
    code: |
      return "hi"
  loop:
    kind: loop
    times: 2
    input:
      n: {seed}
    call: bump
output: ${{loop.output}}
"""


def test_disjoint_tagged_loop_seed_fails():
    error = _load_error(_loop_flow("!int ${note.output}"))

    assert str(error.diagnostic.message).startswith(
        "node 'loop' input 'n': declared int, but the value is str"
    )


def test_tagged_literal_loop_seed_carries_its_contract():
    loaded = load_flow(_loop_flow("!int 0"))

    binding = _binding(loaded.compiled, "loop", "n")
    assert binding.source == 0
    assert binding.contract.type == _INT


def _inline_call_flow(binding: str) -> str:
    """Return a flow whose inline CALL body takes `n` bound as `binding`."""
    return f"""
id: tagged_inline
name: tagged_inline
nodes:
  a:
    kind: code
    code: |
      return 4
  note:
    kind: code
    output: str
    code: |
      return "hi"
  wrap:
    kind: call
    input:
      n: {binding}
    nodes:
      twice:
        kind: code
        input:
          n: ${{input.n}}
        output: int
        code: |
          return inputs["n"] * 2
    output: ${{twice.output}}
output: ${{wrap.output}}
"""


def test_tagged_inline_body_parameter_takes_the_declared_type():
    loaded = load_flow(_inline_call_flow("!int ${a.output + 1}"))

    param = _start_params(loaded.compiled, "wrap")["n"]
    assert param.contract == ValueContract(_INT)
    assert param.required is True


def test_disjoint_tagged_inline_body_arg_fails():
    error = _load_error(_inline_call_flow("!int ${note.output}"))

    assert str(error.diagnostic.message).startswith(
        "node 'wrap' input 'n': declared int, but the value is str"
    )


# --- no fill-in ------------------------------------------------------------- #


def _buy_flow(binding: str, *, producer_output: str = "Conv") -> str:
    """Return a flow whose `use` node binds `decision` as `binding` from producer `d`."""
    return f"""
id: tagged_buy
name: tagged_buy
typedefs:
  Buy:
    action: Literal(buy)
    conviction: float
  Conv:
    conviction: float
nodes:
  d:
    kind: code
    output: {producer_output}
    code: |
      return {{"conviction": 0.5}}
  use:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${{d.output}}
      decision: {binding}
output: ${{use.output}}
"""


def test_tag_over_a_producer_missing_a_required_field_names_the_field():
    with pytest.raises(LoadError, match=re.escape("(missing required `action`)")):
        load_flow(_buy_flow("!Buy ${d.output}"))


def test_tag_over_an_object_producer_loads():
    loaded = load_flow(_buy_flow("!Buy ${d.output}", producer_output="object"))

    assert _binding(loaded.compiled, "use", "decision").contract.type == type_for(
        "Buy", _REGISTRY
    )


def test_tagged_record_literal_with_every_field_loads():
    loaded = load_flow(_buy_flow("!Buy {action: buy, conviction: 0.5}"))

    binding = _binding(loaded.compiled, "use", "decision")
    assert binding.source == {"action": "buy", "conviction": 0.5}
    assert binding.contract.type == type_for("Buy", _REGISTRY)


def test_tagged_record_literal_missing_a_field_names_it():
    with pytest.raises(
        LoadError,
        match=re.escape(
            "'decision': the value is not a valid Buy: record missing required "
            "fields: ['action'] (at `decision.action`)"
        ),
    ):
        load_flow(_buy_flow("!Buy {conviction: 0.5}"))


def _nested_buy_flow(binding: str) -> str:
    """Return a flow whose `c` node binds `p` as the block `binding` over a float."""
    body = "\n".join("      " + line for line in binding.strip("\n").splitlines())
    return f"""
id: nested_buy
name: nested_buy
typedefs:
  Buy:
    action: Literal(buy)
    conviction: float
  Order:
    decision: Buy
    qty: int
nodes:
  score:
    kind: code
    output: float
    code: |
      return 0.5
  c:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${{score.output}}
{body}
output: ${{c.output}}
"""


def test_list_element_missing_a_required_field_names_it_and_its_path():
    flow = _nested_buy_flow(
        """
p: !list(Buy)
  - conviction: ${score.output}
"""
    )
    with pytest.raises(LoadError, match=re.escape("(missing required `action` at `p[0]`)")):
        load_flow(flow)


def test_nested_record_missing_a_required_field_names_it_and_its_path():
    flow = _nested_buy_flow(
        """
p: !Order
  qty: 1
  decision:
    conviction: ${score.output}
"""
    )
    with pytest.raises(
        LoadError, match=re.escape("(missing required `action` at `p.decision`)")
    ):
        load_flow(flow)


def test_optional_nested_record_missing_a_required_field_names_it():
    flow = _nested_buy_flow(
        """
p: !list(Optional(Buy))
  - conviction: ${score.output}
"""
    )
    with pytest.raises(LoadError, match=re.escape("(missing required `action` at `p[0]`)")):
        load_flow(flow)


# --- flow outputs ----------------------------------------------------------- #


def _size_flow(output: str) -> str:
    """Return a flow whose untyped `size` node feeds `output:` written as `output`."""
    return f"""
id: sized
name: sized
typedefs:
  Size:
    qty: int
  T:
    a: int
nodes:
  size:
    kind: code
    code: |
      return {{"qty": 1}}
  note:
    kind: code
    output: str
    code: |
      return "hi"
output: {output}
"""


def test_tagged_output_field_carries_its_contract():
    loaded = load_flow(_size_flow("\n  qty: !int ${size.output.qty}\n  raw: ${size.output}"))

    outputs = _outputs(loaded.compiled)
    assert outputs["qty"].binding.source == "${size.output.qty}"
    assert outputs["qty"].binding.contract.type == _INT
    assert outputs["raw"].binding.contract is None
    assert loaded.compiled.nodes["__end__"].returns_record


def test_disjoint_tagged_output_field_fails_at_its_line():
    text = _size_flow("\n  x: !int ${note.output}")

    error = _load_error(text)

    assert str(error.diagnostic.message).startswith(
        "output 'x': declared int, but the value is str"
    )
    assert _error_line(error) == _line_of(text, "  x: !int ${note.output}")


def test_tagged_single_output_is_one_typed_value():
    loaded = load_flow(_size_flow("!Size ${size.output}"))

    output = _outputs(loaded.compiled)["result"]
    assert output.binding.source == "${size.output}"
    assert output.binding.contract.type == type_for("Size", _REGISTRY)
    end = loaded.compiled.nodes["__end__"]
    assert not end.returns_record
    assert type(end.output_contract) is ValueContract
    assert end.output_contract.type == type_for("Size", _REGISTRY)


def test_disjoint_tagged_single_output_fails():
    with pytest.raises(
        LoadError, match=re.escape("output: declared int, but the value is str")
    ):
        load_flow(_size_flow("!int ${note.output}"))


def test_tagged_mapping_output_is_one_typed_value_not_fields():
    loaded = load_flow(_size_flow("!T {a: 1}"))

    outputs = _outputs(loaded.compiled)
    assert list(outputs) == ["result"]
    assert outputs["result"].binding.source == {"a": 1}
    assert outputs["result"].binding.contract.type == type_for("T", _REGISTRY)
    end = loaded.compiled.nodes["__end__"]
    assert not end.returns_record
    assert end.output_contract.type == type_for("T", _REGISTRY)


def test_tagged_block_output_is_one_typed_value():
    loaded = load_flow(_size_flow("!T\n  a: 1"))

    outputs = _outputs(loaded.compiled)
    assert list(outputs) == ["result"]
    assert outputs["result"].binding.source == {"a": 1}
    assert not loaded.compiled.nodes["__end__"].returns_record


def test_tagged_mapping_output_that_does_not_fit_fails():
    with pytest.raises(LoadError, match="'output'"):
        load_flow(_size_flow("!T {a: x}"))


def test_tagged_output_without_a_value_asks_for_one():
    with pytest.raises(
        LoadError,
        match=re.escape("'output': a binding needs a value: `output: !T ${...}`"),
    ):
        load_flow(_size_flow("!T"))


def test_tagged_single_output_desugars_an_inline_call():
    loaded = load_flow(
        """
id: tagged_out_call
name: tagged_out_call
defs:
  f:
    input:
      a: !int
    nodes:
      x:
        kind: code
        input:
          a: ${input.a}
        output: int
        code: |
          return inputs["a"]
    output: ${x.output}
nodes:
  seed:
    kind: code
    code: |
      return 1
output: !int call(f, a=1)
"""
    )

    output = _outputs(loaded.compiled)["result"]
    assert output.binding.source == "${__call_0.output}"
    assert output.binding.contract.type == _INT


def test_tagged_single_output_reads_a_case_value():
    loaded = load_flow(_CASE_FLOW.format(use="", output="!Decision ${route.output}"))

    output = _outputs(loaded.compiled)["result"]
    assert output.binding.source == "${go.output | stop.output}"
    assert output.binding.contract.type == loaded.compiled.nodes["go"].output_contract.type


def test_tagged_single_output_checks_its_references():
    with pytest.raises(LoadError, match="unresolved references"):
        load_flow(_size_flow("!Size ${missing.output}"))


def test_defs_caller_sees_one_value_of_the_tagged_output_type():
    loaded = load_flow(
        """
id: parent
name: parent
typedefs:
  T:
    a: int
defs:
  child:
    nodes:
      x:
        kind: code
        code: |
          return 1
    output: !T {a: 1}
nodes:
  use:
    kind: call
    call: child
output: ${use.output}
"""
    )

    contract = loaded.compiled.nodes["use"].output_contract
    assert type(contract) is ValueContract
    assert contract.type == type_for("T", _REGISTRY)


def test_defs_caller_sees_a_tagged_reference_output_type():
    loaded = load_flow(
        """
id: parent
name: parent
typedefs:
  Size:
    qty: int
defs:
  child:
    nodes:
      size:
        kind: code
        code: |
          return {"qty": 1}
    output: !Size ${size.output}
nodes:
  use:
    kind: call
    call: child
output: ${use.output}
"""
    )

    assert loaded.compiled.nodes["use"].output_contract.type == type_for("Size", _REGISTRY)


def test_a_tagged_binding_with_a_non_string_key_is_a_load_error():
    text = _code_flow("!object {1: 2}")
    error = _load_error(text)
    assert "the key `1` was read by YAML as an integer" in str(error.diagnostic.message)
    assert _error_line(error) == _line_of(text, "      qty: !object {1: 2}")


def test_a_tagged_whole_output_with_a_non_string_key_is_a_load_error():
    with pytest.raises(
        LoadError,
        match=re.escape("the key `1` was read by YAML as an integer"),
    ):
        load_flow(_size_flow("!object {1: 2}"))


def test_reference_whose_nested_record_misses_a_required_field_names_its_path():
    flow = """
id: nested_ref
name: nested_ref
typedefs:
  Buy:
    action: Literal(buy)
    conviction: float
  Order:
    decision: Buy
    qty: int
  Conv:
    conviction: float
  Loose:
    decision: Conv
    qty: int
nodes:
  d:
    kind: code
    output: Loose
    code: |
      return {"decision": {"conviction": 0.5}, "qty": 1}
  c:
    kind: code
    code: tests.engine._compose_codefns:echo_x
    input:
      x: ${d.output}
      p: !Order ${d.output}
output: ${c.output}
"""
    with pytest.raises(
        LoadError, match=re.escape("(missing required `action` at `p.decision`)")
    ):
        load_flow(flow)


def test_a_literal_in_a_template_shows_as_written_in_the_error():
    flow = _nested_buy_flow(
        """
p: !Order
  qty: 1
  decision:
    conviction: ${score.output}
"""
    )
    with pytest.raises(
        LoadError,
        match=re.escape('but the value is {"decision":{"conviction":float},"qty":1}'),
    ):
        load_flow(flow)


def test_a_list_template_shows_its_items_in_the_error():
    flow = _nested_buy_flow(
        """
p: !list(Buy)
  - conviction: ${score.output}
"""
    )
    with pytest.raises(
        LoadError, match=re.escape('but the value is [{"conviction":float}]')
    ):
        load_flow(flow)


def test_a_declared_typedef_is_named_in_the_error():
    flow = _nested_buy_flow(
        """
p: !Optional(Order)
  qty: 1
  decision:
    conviction: ${score.output}
"""
    )
    with pytest.raises(
        LoadError, match=re.escape("node 'c' input 'p': declared Optional(Order), but the value is")
    ):
        load_flow(flow)


def test_a_declared_list_of_a_typedef_is_named_in_the_error():
    flow = _nested_buy_flow(
        """
p: !list(Buy)
  - conviction: ${score.output}
"""
    )
    with pytest.raises(
        LoadError, match=re.escape("node 'c' input 'p': declared list(Buy), but the value is")
    ):
        load_flow(flow)

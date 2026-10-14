"""Loading typed flow outputs: `output: !Type value` and typed record fields."""

from __future__ import annotations


from yian.compile.manifest import flow_fingerprint
from yian.compose.loader import load_flow
from yian.nodes.contracts import PartialRecordContract, ValueContract
from yian.run import run_flow
from yian.typesys import TypeRegistry, type_for
from yian.typesys.types import read_typedefs

_REGISTRY = read_typedefs({"Size": {"qty": "int"}, "Decision": "Literal(go, stop)"})
_SIZE = type_for("Size", _REGISTRY)


def _outputs(compiled) -> dict:
    """Return the compiled flow's outputs keyed by name."""
    return {output.name: output for output in compiled.outputs}


def _child(compiled, node_id: str):
    """Return the compiled child flow baked onto a CALL, MAP, or LOOP node."""
    child = compiled.nodes[node_id].child
    return getattr(child, "compiled", child)


def _size_flow(output: str) -> str:
    """Return a flow whose untyped `size` node feeds `output:` written as `output`."""
    return f"""
id: sized
name: sized
typedefs:
  Size:
    qty: int
nodes:
  size:
    kind: code
    code: |
      return {{"qty": 1}}
output: {output}
"""


def test_typed_single_output_carries_its_contract():
    loaded = load_flow(_size_flow("!Size ${size.output}"))

    output = _outputs(loaded.compiled)["result"]
    assert output.binding.source == "${size.output}"
    assert output.binding.contract.type == _SIZE
    end = loaded.compiled.nodes["__end__"]
    assert not end.returns_record
    assert type(end.output_contract) is ValueContract
    assert end.output_contract.type == _SIZE


def test_untyped_single_output_leaves_end_untyped():
    loaded = load_flow(_size_flow("${size.output}"))

    assert _outputs(loaded.compiled)["result"].binding.contract is None
    assert loaded.compiled.nodes["__end__"].output_contract is None


def test_typed_output_changes_the_flow_fingerprint():
    typed = load_flow(_size_flow("!Size ${size.output}"))
    plain = load_flow(_size_flow("${size.output}"))

    assert flow_fingerprint(typed.compiled) != flow_fingerprint(plain.compiled)


def test_record_output_types_only_its_typed_fields():
    loaded = load_flow(
        """
id: decided
name: decided
typedefs:
  Decision: Literal(go, stop)
nodes:
  d:
    kind: code
    code: |
      return "go"
  n:
    kind: code
    code: |
      return "fine"
output:
  decision: !Decision ${d.output}
  note: ${n.output}
  label: "a = b"
"""
    )

    outputs = _outputs(loaded.compiled)
    decision = type_for("Decision", _REGISTRY)
    assert outputs["decision"].binding.source == "${d.output}"
    assert outputs["decision"].binding.contract.type == decision
    assert outputs["note"].binding.contract is None
    assert outputs["label"].binding.source == "a = b"
    assert outputs["label"].binding.contract is None
    end = loaded.compiled.nodes["__end__"]
    assert end.returns_record
    contract = end.output_contract
    assert type(contract) is PartialRecordContract
    fields = dict(contract.fields)
    assert fields["decision"].type == decision
    assert fields["note"] is None
    assert fields["label"] is None


def test_fully_typed_record_output_gives_end_an_exact_record():
    loaded = load_flow(
        """
id: exact
name: exact
nodes:
  n:
    kind: code
    code: |
      return 1
output:
  a: !int ${n.output}
  b: !str ${n.output}
"""
    )

    contract = loaded.compiled.nodes["__end__"].output_contract
    assert type(contract) is ValueContract
    assert contract.type.fields == {
        "a": type_for("int", TypeRegistry()),
        "b": type_for("str", TypeRegistry()),
    }


def test_untyped_record_output_leaves_end_untyped():
    loaded = load_flow(
        """
id: plain
name: plain
nodes:
  n:
    kind: code
    code: |
      return 1
output:
  a: ${n.output}
"""
    )

    assert loaded.compiled.nodes["__end__"].output_contract is None


def test_plain_output_whose_left_side_is_not_a_type_is_text():
    text = """
id: plain
name: plain
nodes:
  n:
    kind: code
    code: |
      return 1
output: a = b
"""
    result = run_flow(load_flow(text), {})

    assert result.status == "succeeded", result.failure
    assert result.output == "a = b"


def test_record_output_field_whose_left_side_is_not_a_type_is_text():
    text = """
id: plain
name: plain
nodes:
  n:
    kind: code
    code: |
      return 1
output:
  note: a = b
"""
    result = run_flow(load_flow(text), {})

    assert result.status == "succeeded", result.failure
    assert result.output == {"note": "a = b"}


def test_compact_flow_output_stays_the_node_output_type():
    loaded = load_flow(
        """
id: compact
name: compact
typedefs:
  Size:
    qty: int
kind: code
code: |
  return {"qty": 1}
output: Size
"""
    )

    assert loaded.compiled.nodes["compact"].output_contract.type == _SIZE
    output = _outputs(loaded.compiled)["result"]
    assert output.binding.source == "${compact.output}"
    assert output.binding.contract is None
    assert loaded.compiled.nodes["__end__"].output_contract is None


_CHILD_FLOW = """
id: parent
name: parent
typedefs:
  Size:
    qty: int
defs:
  sized:
    input:
      n: !int
    nodes:
      size:
        kind: code
        input:
          n: ${{input.n}}
        code: |
          return {{"qty": inputs["n"]}}
    output: {child_output}
nodes:
  use:
    kind: call
    call: sized
    input:
      n: 1
  each:
    kind: map
    over: [1, 2]
    call: sized
    input:
      n: ${{item}}
output:
  one: ${{use.output}}
  many: ${{each.output}}
"""


def test_call_parent_sees_the_child_declared_output_type():
    loaded = load_flow(_CHILD_FLOW.format(child_output="!Size ${size.output}"))

    assert loaded.compiled.nodes["use"].output_contract.type == _SIZE


def test_map_parent_sees_a_list_of_the_child_declared_output_type():
    loaded = load_flow(_CHILD_FLOW.format(child_output="!Size ${size.output}"))

    assert loaded.compiled.nodes["each"].output_contract.type == type_for(
        "list(Size)", _REGISTRY
    )


def test_call_parent_sees_a_partial_record_for_a_partly_typed_child():
    child_output = "\n      size: !Size ${size.output}\n      raw: ${size.output}"
    loaded = load_flow(_CHILD_FLOW.format(child_output=child_output))

    contract = loaded.compiled.nodes["use"].output_contract
    assert type(contract) is PartialRecordContract
    fields = dict(contract.fields)
    assert fields["size"].type == _SIZE
    assert fields["raw"] is None


def test_inline_call_body_output_field_carries_its_declared_type():
    loaded = load_flow(
        """
id: inline_call
name: inline_call
input:
  n: !int
nodes:
  wrapper:
    kind: call
    input:
      n: ${input.n}
    nodes:
      echo:
        kind: code
        input:
          n: ${input.n}
        code: |
          return inputs["n"]
    output:
      qty: !int ${echo.output}
output: ${wrapper.output}
"""
    )

    child = _child(loaded.compiled, "wrapper")
    assert _outputs(child)["qty"].binding.contract.type == type_for(
        "int", TypeRegistry()
    )
    contract = loaded.compiled.nodes["wrapper"].output_contract
    assert type(contract) is ValueContract
    assert contract.type.fields == {"qty": type_for("int", TypeRegistry())}


def _loop_flow(field: str) -> str:
    """Return a flow whose inline LOOP body returns `rounds` written as `field`."""
    return f"""
id: inline_loop
name: inline_loop
nodes:
  loop:
    kind: loop
    times: 2
    input:
      rounds: 0
    nodes:
      bump:
        kind: code
        input:
          n: ${{input.rounds}}
        code: |
          return inputs["n"] + 1
    output:
      rounds: {field}
output: ${{loop.output}}
"""


def test_inline_loop_body_output_field_types_the_carried_record():
    typed = load_flow(_loop_flow("!int ${bump.output}"))
    plain = load_flow(_loop_flow("${bump.output}"))

    contract = typed.compiled.nodes["loop"].output_contract
    assert type(contract) is ValueContract
    assert contract.type.fields == {"rounds": type_for("int", TypeRegistry())}
    assert plain.compiled.nodes["loop"].output_contract != contract

"""Loading stamps each TOOL node input's proven inferred type on the node."""

from __future__ import annotations

import pytest

from yian.compose.loader import load_flow
from yian.typesys import canonical_type_label


def _child(compiled, node_id: str):
    """Return the compiled child flow baked onto a CALL or MAP node."""
    child = compiled.nodes[node_id].child
    return getattr(child, "compiled", child)


def _labels(node) -> dict[str, str]:
    """Return a TOOL node's stamped input types as canonical labels."""
    return {name: canonical_type_label(t) for name, t in node.input_types.items()}


def _tool_flow(binding: str) -> str:
    """Return a flow whose TOOL node `t` binds `v` as `binding`."""
    return f"""
id: stamped
name: stamped
input:
  n: !int
  a: !str
  b: !str
nodes:
  c:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${{input.n}}
  t:
    kind: tool
    tool_id: place_order
    input:
      v: {binding}
output: ${{t.output}}
"""


@pytest.mark.parametrize(
    "binding, label",
    [
        ("${input.n}", "int"),
        ('"${input.a}-${input.b}"', "str"),
        ("5", "int(5..5)"),
        ("true", "bool"),
        ('"x"', 'Literal("x")'),
        ("[1, 2]", "list(int)"),
        ("!float ${input.n}", "float"),
        ('""', "str"),
        ('"$${x}"', 'Literal("${x}")'),
        ("1.5", "float(1.5..1.5)"),
        ("null", "None"),
    ],
)
def test_a_proven_input_is_stamped_with_its_inferred_type(binding, label):
    loaded = load_flow(_tool_flow(binding))

    assert _labels(loaded.compiled.nodes["t"]) == {"v": label}


def test_a_non_finite_number_literal_is_absent():
    loaded = load_flow(_tool_flow(".inf"))

    assert loaded.compiled.nodes["t"].input_types == {}


def test_an_unproven_input_is_absent():
    loaded = load_flow(_tool_flow("${c.output}"))

    assert loaded.compiled.nodes["t"].input_types == {}


def test_a_tool_in_a_child_flow_is_stamped():
    loaded = load_flow(
        """
id: outer
name: outer
input:
  n: !int
defs:
  sub:
    input:
      m: !int
    nodes:
      t:
        kind: tool
        tool_id: place_order
        input:
          v: ${input.m}
    output: ${t.output}
nodes:
  use:
    kind: call
    call: sub
    input:
      m: ${input.n}
output: ${use.output}
"""
    )

    child = _child(loaded.compiled, "use")
    assert _labels(child.nodes["t"]) == {"v": "int"}


def test_a_case_narrowed_input_is_stamped_with_the_member_record():
    loaded = load_flow(
        """
id: narrowed
name: narrowed
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Sell: {action: 'Literal(sell)', qty: int, price: float}
  Decision: Union(Buy, Sell)
input:
  decision: !object
nodes:
  d:
    kind: code
    input:
      decision: ${input.decision}
    code: |
      return inputs["decision"]
    output: Decision
  route:
    kind: case
    on: ${d.output.action}
    cases:
      - when: sell
        then: t
    else: hold
  t:
    kind: tool
    tool_id: place_order
    input:
      order: ${d.output}
  hold:
    kind: code
    depends_on: [d]
    code: |
      return "hold"
output: ${t.output | hold.output}
"""
    )

    stamped = loaded.compiled.nodes["t"].input_types["order"]
    assert stamped.variants is None
    assert set(stamped.fields) == {"action", "qty", "price"}


def test_a_map_body_tool_reads_the_element_type():
    loaded = load_flow(
        """
id: batch
name: batch
input:
  xs: !list(int)
nodes:
  each:
    kind: map
    over: ${input.xs}
    input:
      x: ${item}
    nodes:
      t:
        kind: tool
        tool_id: place_order
        input:
          x: ${input.x}
    output: ${t.output}
output: ${each.output}
"""
    )

    child = _child(loaded.compiled, "each")
    assert _labels(child.nodes["t"]) == {"x": "int"}

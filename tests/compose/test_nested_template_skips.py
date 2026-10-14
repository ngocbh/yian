"""Each string nested in a value template is its own input group.

Every leaf string of a `ValueTemplate` binding behaves as if it were its own
input: its data edges carry their own input group (the parameter name followed by
the string's JSON-pointer path) and their own co-skip stance, a flow output gets
null at a nested position whose producers may all be skipped, and arithmetic on a
nested value that may be skipped is rejected. A plain top-level string keeps the
parameter name as its group.
"""

from __future__ import annotations

import re

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.nodes.contracts import ValueTemplate

_HEADER = """
id: nested_skips
name: nested_skips
input:
  approve: !bool
nodes:
  gate:
    kind: case
    cases:
      - when: ${input.approve}
        then: order
    else: rejected
  order:
    kind: code
    output: str
    code: |
      return "filled"
  rejected:
    kind: code
    output: str
    code: |
      return "no"
"""


def _flow(consumer_input: str, output: str = "${c.output}") -> str:
    """Return a gated flow whose CODE node `c` binds `x` as `consumer_input`."""
    return (
        _HEADER
        + f"""  c:
    kind: code
    input:
      x: {consumer_input}
    code: |
      return inputs["x"]
output: {output}
"""
    )


def _edges_into(compiled, node_id: str) -> list[tuple[str, str | None, bool]]:
    """Return `(from, input_group, optional)` for each data edge into `node_id`."""
    return [
        (edge.from_, edge.input_group, edge.optional)
        for edge in compiled.edges
        if edge.to == node_id and edge.source_handle is None and not edge.ordering
        and edge.input_group is not None
    ]


def test_each_nested_string_has_its_own_group_and_stance():
    compiled = load_flow(
        _flow("{a: '${order.output}', b: ['note ${rejected.output}']}")
    ).compiled

    assert _edges_into(compiled, "c") == [
        ("order", "x/a", False),
        ("rejected", "x/b/0", True),
    ]


def test_nested_group_names_escape_pointer_characters():
    compiled = load_flow(_flow("{'a/b~c': '${order.output}'}")).compiled

    assert _edges_into(compiled, "c") == [("order", "x/a~1b~0c", False)]


def test_plain_string_keeps_the_parameter_as_its_group():
    compiled = load_flow(_flow("${order.output}")).compiled

    assert _edges_into(compiled, "c") == [("order", "x", False)]
    assert _edges_into(compiled, "__end__") == [("c", "result", True)]


def test_flow_output_has_a_group_per_nested_string():
    compiled = load_flow(
        _flow(
            "${order.output}",
            output="\n  r: {a: {b: '${order.output}'}, n: ['${rejected.output}']}",
        )
    ).compiled

    assert _edges_into(compiled, "__end__") == [
        ("order", "r/a/b", True),
        ("rejected", "r/n/0", True),
    ]


def test_skipped_nested_output_string_defaults_to_null():
    compiled = load_flow(
        _flow(
            "${order.output}",
            output=(
                "\n  r: {a: {b: '${order.output}'}, t: 'got ${order.output}',"
                " s: '${order.output | rejected.output}'}"
            ),
        )
    ).compiled

    by_name = {output.name: output.binding.source for output in compiled.outputs}
    source = by_name["r"]
    assert type(source) is ValueTemplate
    # Only the co-skipping string whose producers may all skip gets the default.
    assert source.value == {
        "a": {"b": "${order.output:-null}"},
        "t": "got ${order.output}",
        "s": "${order.output | rejected.output}",
    }
    assert [path for path, _ in source.origins] == [("a", "b"), ("t",), ("s",)]


def test_nested_arithmetic_on_a_skipped_value_is_rejected_at_its_line():
    text = _flow("\n        a: ok\n        b: ${order.output + 1}")

    with pytest.raises(LoadError) as info:
        load_flow(text)

    assert info.value.diagnostic.message == (
        "node 'c' input 'x': ${order.output + 1} does arithmetic on order.output, which "
        "is null when node 'order' is skipped; give it a default with :-, e.g. "
        "(order.output :- <default>)"
    )
    frames = info.value.diagnostic.frames
    assert frames[-1].range.start_line == text.splitlines().index(
        "        b: ${order.output + 1}"
    ) + 1


def test_nested_output_arithmetic_on_a_skipped_value_is_rejected():
    text = _flow("${order.output}", output="\n  r: {a: '${order.output + 1}'}")

    with pytest.raises(
        LoadError,
        match=re.escape(
            "output 'r': ${order.output + 1} does arithmetic on order.output"
        ),
    ):
        load_flow(text)

"""Skips through strings nested in a value template, end to end.

Each nested `${...}` string behaves as its own input: a whole-value read whose
producers are all skipped skips the reading node, an interpolation of a skipped
node renders empty text and the node runs, a flow output gets null at the nested
position, and these groups survive runtime growth.
"""

from __future__ import annotations

import pytest

from yian.compile.expand import clone_child
from yian.compose import load_flow
from yian.events import NodeSkipped
from yian.run import run_flow

_GATED_NODES = """
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


def _flow(extra_nodes: str, output: str) -> str:
    """Return a flow gated on `input.approve` with `extra_nodes` and `output`."""
    return (
        "id: nested_skips\nname: nested_skips\ninput:\n  approve: !bool\nnodes:"
        + _GATED_NODES
        + extra_nodes
        + f"output:{output}\n"
    )


def _code(node_id: str, binding: str) -> str:
    """Return a CODE node that returns its input `x`, bound as `binding`."""
    return (
        f"  {node_id}:\n    kind: code\n    input:\n      x: {binding}\n"
        '    code: |\n      return inputs["x"]\n'
    )


def _run(text: str, approve: bool):
    """Run `text` with `approve`; the run must succeed."""
    result = run_flow(load_flow(text), {"approve": approve})
    assert result.status == "succeeded", result.failure
    return result


def _skipped(result) -> set[str]:
    """Return the ids of the nodes the run skipped."""
    return {event.node_id for event in result.events if isinstance(event, NodeSkipped)}


def test_nested_whole_value_read_of_a_skipped_node_skips_the_reader():
    text = _flow(
        _code("c", "{a: '${order.output}', b: 1}"),
        "\n  c: ${c.output}\n  r: ${rejected.output}",
    )

    skipped = _run(text, approve=False)
    assert "c" in _skipped(skipped)
    assert skipped.output == {"c": None, "r": "no"}

    taken = _run(text, approve=True)
    assert "c" not in _skipped(taken)
    assert taken.output == {"c": {"a": "filled", "b": 1}, "r": None}


def test_nested_interpolation_of_a_skipped_node_renders_empty_text():
    text = _flow(_code("c", "{a: 'got ${order.output}!'}"), " ${c.output}")

    skipped = _run(text, approve=False)
    assert "c" not in _skipped(skipped)
    assert skipped.output == {"a": "got !"}

    assert _run(text, approve=True).output == {"a": "got filled!"}


def test_one_skipped_nested_string_skips_the_reader_even_beside_a_live_one():
    text = _flow(
        _code("c", "{a: '${rejected.output}', b: ['${order.output}']}"),
        "\n  c: ${c.output}",
    )

    assert "c" in _skipped(_run(text, approve=False))
    assert "c" in _skipped(_run(text, approve=True))


def test_nested_coalesce_over_both_branches_runs_the_reader():
    text = _flow(_code("c", "{a: '${order.output | rejected.output}'}"), " ${c.output}")

    assert _run(text, approve=False).output == {"a": "no"}
    assert _run(text, approve=True).output == {"a": "filled"}


def test_flow_output_gets_null_at_a_skipped_nested_position():
    text = _flow("", "\n  r: {a: {b: '${order.output}'}, n: 1}")

    assert _run(text, approve=False).output == {"r": {"a": {"b": None}, "n": 1}}
    assert _run(text, approve=True).output == {"r": {"a": {"b": "filled"}, "n": 1}}


def test_value_mode_output_gets_null_at_a_skipped_nested_position():
    text = _flow("", " {a: {b: '${order.output}'}}")

    assert _run(text, approve=False).output == {"a": {"b": None}}


def test_consumer_waits_for_a_nested_producer():
    text = (
        "id: wait\nname: wait\nnodes:\n"
        "  slow:\n    kind: code\n    code: |\n"
        "      import time\n      time.sleep(0.05)\n      return 7\n"
        + _code("c", "{deep: [{v: '${slow.output}'}]}")
        + "output: ${c.output}\n"
    )
    compiled = load_flow(text).compiled

    assert [
        (edge.from_, edge.input_group) for edge in compiled.edges if edge.to == "c"
    ] == [("slow", "x/deep/0/v")]
    result = run_flow(load_flow(text), {})
    assert result.status == "succeeded", result.failure
    assert result.output == {"deep": [{"v": 7}]}


_CHILD = """
id: child
name: child
input:
  approve: !bool
nodes:{nodes}  c:
    kind: code
    input:
      x: {{a: '${{order.output}}', t: 'got ${{order.output}}'}}
    code: |
      return inputs["x"]
output:
  c: ${{c.output}}
  r: {{b: '${{order.output}}'}}
"""


def test_cloned_child_keeps_the_nested_groups():
    child = load_flow(_CHILD.format(nodes=_GATED_NODES)).compiled

    clone = clone_child(child, "k", {"approve": True})

    by_group = sorted(
        (edge.to, edge.input_group, edge.optional)
        for edge in clone.edges
        if edge.from_ == "k/order" and edge.input_group is not None
    )
    assert by_group == [
        ("k/__end__", "r/b", True),
        ("k/c", "x/a", False),
        ("k/c", "x/t", True),
    ]


_PARENT = """
id: parent
name: parent
input:
  approve: !bool
defs:
  child:
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
      c:
        kind: code
        input:
          x: {a: '${order.output}', t: 'got ${order.output}'}
        code: |
          return inputs["x"]
    output:
      c: ${c.output}
      r: {b: '${order.output}'}
nodes:
  k:
    kind: call
    call: child
    input:
      approve: ${input.approve}
output: ${k.output}
"""


@pytest.mark.parametrize(
    ("approve", "expected", "c_skipped"),
    [
        (False, {"c": None, "r": {"b": None}}, True),
        (True, {"c": {"a": "filled", "t": "got filled"}, "r": {"b": "filled"}}, False),
    ],
)
def test_nested_skips_hold_inside_a_called_child(approve, expected, c_skipped):
    result = _run(_PARENT, approve=approve)

    assert result.output == expected
    assert ("k/c" in _skipped(result)) is c_skipped

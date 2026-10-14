"""Nested `${...}` in input and output values is evaluated, end to end.

A mapping or list binding value is a template at every depth: a nested whole
`${...}` keeps its value, a nested interpolation renders text, `$$` unescapes,
and a nested tag checks the value at its position without filling anything in.
"""

from __future__ import annotations

import re

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.run import run_flow

_PRODUCER = """  n:
    kind: code
    output: str
    code: |
      return "hi"
"""


def _flow(nodes: str, output: str, *, head: str = "") -> str:
    """Return a flow with the `n` producer, `nodes` and `output`."""
    return f"{head}id: nested\nname: nested\nnodes:\n{_PRODUCER}{nodes}output:{output}\n"


def _echo(binding: str) -> str:
    """Return a CODE node `c` that returns its input `p`, bound as `binding`."""
    return (
        f"  c:\n    kind: code\n    input:\n      p:{binding}\n"
        '    code: |\n      return inputs["p"]\n'
    )


def _run(text: str, inputs: dict | None = None):
    """Load and run `text`; the run must succeed."""
    result = run_flow(load_flow(text), inputs or {})
    assert result.status == "succeeded", result.failure
    return result


def test_nested_reference_in_a_flow_output_field_is_evaluated():
    text = _flow("", "\n  a:\n    b: ${n.output}")

    assert _run(text).output == {"a": {"b": "hi"}}


def test_reference_in_a_flow_output_list_is_evaluated():
    text = _flow("", "\n  - ${n.output}\n  - 1")

    assert _run(text).output == ["hi", 1]


def test_nested_reference_in_a_node_input_is_evaluated():
    text = _flow(_echo("\n        q: ${n.output}"), " ${c.output}")

    assert _run(text).output == {"q": "hi"}


def test_nested_escaped_span_unescapes_to_text():
    text = _flow(_echo("\n        q: $${x}\n        r: ${n.output}"), " ${c.output}")

    assert _run(text).output == {"q": "${x}", "r": "hi"}


def test_nested_escape_without_a_span_unescapes_to_text():
    text = _flow(_echo("\n        q: costs $$5"), " ${c.output}")

    assert _run(text).output == {"q": "costs $5"}


def test_nested_interpolation_renders_text():
    text = _flow(_echo("\n        q:\n          - got ${n.output}!"), " ${c.output}")

    assert _run(text).output == {"q": ["got hi!"]}


def test_nested_item_read_in_a_map_input_is_evaluated_per_item():
    text = """id: nested_map
name: nested_map
input:
  rows: !list(str)
nodes:
  m:
    kind: map
    over: ${input.rows}
    input:
      v:
        i: ${item}
    nodes:
      echo:
        kind: code
        input:
          x: ${input.v}
        code: |
          return inputs["x"]
    output: ${echo.output}
output: ${m.output}
"""

    assert _run(text, {"rows": ["a", "b"]}).output == [{"i": "a"}, {"i": "b"}]


def test_nested_reference_in_an_inline_body_argument_is_evaluated():
    text = _flow(
        """  c:
    kind: call
    input:
      payload:
        q: ${n.output}
        tags: [x, y]
    nodes:
      echo:
        kind: code
        input:
          x: ${input.payload}
        code: |
          return inputs["x"]
    output: ${echo.output}
""",
        " ${c.output}",
    )

    assert _run(text).output == {"q": "hi", "tags": ["x", "y"]}


_BUY_TYPEDEFS = """typedefs:
  Buy:
    action: Literal(buy)
    conviction: float
"""

_SCORE = """  score:
    kind: code
    output: float
    code: |
      return 0.75
"""


def _decision_flow(fields: str) -> str:
    """Return a flow whose `c` node binds `decision` as a `!Buy` block of `fields`."""
    return _flow(
        _SCORE
        + "  c:\n    kind: code\n    input:\n      decision: !Buy\n"
        + fields
        + '    code: |\n      return inputs["decision"]\n',
        " ${c.output}",
        head=_BUY_TYPEDEFS,
    )


def test_nested_tagged_block_checks_and_runs():
    text = _decision_flow("        action: buy\n        conviction: ${score.output}\n")

    assert _run(text).output == {"action": "buy", "conviction": 0.75}


def test_nested_tagged_block_without_a_required_field_names_it():
    text = _decision_flow("        conviction: ${score.output}\n")

    with pytest.raises(LoadError, match=re.escape("(missing required `action`)")):
        load_flow(text)


def test_nested_tag_inside_a_list_checks_its_item():
    text = _flow(
        _SCORE + _echo("\n        - !float ${score.output}\n        - ${n.output}"),
        " ${c.output}",
    )

    assert _run(text).output == [0.75, "hi"]


def test_nested_tag_inside_a_list_rejects_a_mismatched_item():
    text = _flow(_echo("\n        - !int ${n.output}"), " ${c.output}")

    with pytest.raises(LoadError, match=re.escape("declared int, but the value is str")):
        load_flow(text)

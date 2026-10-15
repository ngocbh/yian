"""Tests for the `when:` hint on a node body's unknown-key error.

No node kind takes a top-level `when:`; the only `when:` lives in a `case` node's `cases:`
entries. A node that authors `when:` gets the generic "field 'when' is not allowed" error plus
a hint pointing at the `case` node (or, on a `case` node, at its `cases:` entries).
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.nodes import KINDS
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceDocument

_NON_CASE_HINT = "route to it from a `case` node"
_CASE_HINT = "goes in each entry of `cases:`"

# call/loop/map check their `call:`-or-`nodes:` choice before body keys, so they need one.
_EXTRA_BODY = {"call": "call: sub", "loop": "call: sub", "map": "call: sub"}


@pytest.mark.parametrize("kind", sorted(set(KINDS) - {"case"}))
def test_when_on_non_case_node_suggests_a_case_node(kind):
    text = f"""\
id: f
name: f
nodes:
  x:
    kind: {kind}
    {_EXTRA_BODY.get(kind, "")}
    when: ${{x.output}}
output: ${{x.output}}
"""
    with pytest.raises(LoadError) as caught:
        load_flow(text)

    message = str(caught.value)
    assert "field 'when' is not allowed" in message
    assert "(allowed: " in message
    assert _NON_CASE_HINT in message


def test_when_on_case_node_points_at_cases_entries():
    text = """\
id: f
name: f
nodes:
  route:
    kind: case
    when: ${a.output}
    cases:
      - when: ${a.output}
        then: b
  a:
    kind: code
    code: "def run(): return True"
  b:
    kind: code
    code: "def run(): return 1"
output: ${b.output}
"""
    with pytest.raises(LoadError) as caught:
        load_flow(text)

    message = str(caught.value)
    assert "field 'when' is not allowed" in message
    assert _CASE_HINT in message
    assert "route to it from" not in message


def test_when_error_points_at_the_when_key():
    source = """\
nodes:
  x:
    kind: code
    code: "def run(): return 1"
    when: ${a.output}
"""
    nodes = SourceDocument.parse(source).root.child("nodes")
    node = nodes.child("x")

    with pytest.raises(LoadError, match="field 'when' is not allowed") as caught:
        parse_nodes(nodes)

    assert caught.value.diagnostic.frames[-1].range == node.range_of("when")


def test_other_unknown_key_gets_no_case_hint():
    text = """\
id: f
name: f
nodes:
  x:
    kind: code
    code: "def run(): return 1"
    bogus: 1
output: ${x.output}
"""
    with pytest.raises(LoadError) as caught:
        load_flow(text)

    message = str(caught.value)
    assert "field 'bogus' is not allowed" in message
    assert message.endswith(")")
    assert "route to it" not in message
    assert "`case`" not in message


def test_when_on_compact_flow_suggests_a_case_node():
    text = """\
id: x
name: x
kind: code
code: "def run(): return 1"
when: ${input.flag}
"""
    with pytest.raises(LoadError) as caught:
        load_flow(text)

    message = str(caught.value)
    assert "field 'when' is not allowed" in message
    assert _NON_CASE_HINT in message

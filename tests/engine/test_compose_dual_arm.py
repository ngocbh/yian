"""Parser rejects legacy section keywords with a bespoke LoadError.

`inputs:`/`outputs:` at the top level, in a node body, or in a `defs:` entry raises a
located LoadError naming the retirement; `input:`/`output:` is the only accepted spelling.
"""

from __future__ import annotations

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose import load_flow
from yian.compose.parse.file import parse_file
from yian.compose.source import SourceDocument


def _new_only() -> str:
    return """
id: f
name: f
input:
  x: !str
nodes:
  n:
    kind: code
    input:
      v: ${input.x}
    output: str
    code: tests.engine._fakes:passthrough
output:
  result: ${n.output}
"""


def _legacy_input() -> str:
    return """
id: f
name: f
inputs:
  x: !str
nodes: {}
"""


def _legacy_output() -> str:
    return """
id: f
name: f
input:
  x: !str
nodes: {}
outputs:
  r: ${n.output}
"""


def test_new_only_parses() -> None:
    cf = parse_file(_new_only())
    assert set(cf.inputs.keys()) == {"x"}
    assert "n" in cf.nodes


def test_legacy_inputs_rejected() -> None:
    with pytest.raises(LoadError, match="rename the section"):
        parse_file(_legacy_input())


def test_legacy_outputs_rejected() -> None:
    with pytest.raises(LoadError, match="rename the section"):
        parse_file(_legacy_output())


_NODE_PLURAL = """\
id: f
name: f
input:
  x: !int
nodes:
  n:
    kind: code
    {first}:
      x: ${{input.x}}
    {second}: int
    code: return inputs["x"]
output: ${{n.output}}
"""


@pytest.mark.parametrize(
    ("first", "second", "legacy"),
    [
        ("inputs", "output", "inputs"),
        ("input", "outputs", "outputs"),
    ],
)
def test_legacy_node_signature_key_rejected_at_its_key(first, second, legacy) -> None:
    text = _NODE_PLURAL.format(first=first, second=second)
    with pytest.raises(LoadError, match=f"node 'n'.*{legacy!r}.*rename the key") as exc:
        load_flow(text)
    key_range = SourceDocument.parse(text).root.child("nodes").child("n").key_range_of(
        legacy
    )
    assert exc.value.diagnostic.frames[-1].range == key_range


@pytest.mark.parametrize("order", [("input", "inputs"), ("inputs", "input")])
def test_node_with_both_signature_spellings_rejected(order) -> None:
    text = f"""\
id: f
name: f
input:
  x: !int
nodes:
  n:
    kind: code
    {order[0]}:
      x: ${{input.x}}
    {order[1]}:
      y: ${{input.x}}
    output: int
    code: return inputs["x"]
output: ${{n.output}}
"""
    with pytest.raises(LoadError, match="'inputs'.*rename the key"):
        load_flow(text)


def test_legacy_def_signature_key_rejected_at_its_key() -> None:
    text = """\
id: f
name: f
input:
  x: !int
defs:
  child:
    inputs:
      x: !int
    nodes:
      n:
        kind: code
        input:
          x: ${input.x}
        output: int
        code: return inputs["x"]
    output: ${n.output}
nodes:
  c:
    kind: call
    call: child
    input:
      x: ${input.x}
output: ${c.output}
"""
    with pytest.raises(LoadError, match="defs entry 'child'.*'inputs'.*rename the key") as exc:
        load_flow(text)
    key_range = SourceDocument.parse(text).root.child("defs").child("child").key_range_of(
        "inputs"
    )
    assert exc.value.diagnostic.frames[-1].range == key_range


def test_legacy_single_node_def_signature_key_rejected() -> None:
    text = """\
id: f
name: f
input:
  x: !int
defs:
  child:
    kind: code
    input:
      x: !int
    outputs: int
    code: return inputs["x"]
nodes:
  c:
    kind: call
    call: child
    input:
      x: ${input.x}
output: ${c.output}
"""
    with pytest.raises(LoadError, match="defs entry 'child'.*'outputs'.*rename the key"):
        load_flow(text)

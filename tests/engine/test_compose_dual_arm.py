"""Parser rejects legacy section keywords with a bespoke LoadError.

`inputs:`/`outputs:` at the top level raises a located LoadError naming the retirement;
`input:`/`output:` is the only accepted spelling.
"""

from __future__ import annotations

import pytest

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.parser import parse_file


def _new_only() -> str:
    return """
id: f
name: f
input:
  x: str
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
  x: str
nodes: {}
"""


def _legacy_output() -> str:
    return """
id: f
name: f
input:
  x: str
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

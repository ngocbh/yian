import pytest
from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError

_BAD_LITERAL = """
id: f
name: f
nodes:
  ask:
    kind: human_input
    questions:
      - {question: "dup", header: "H"}
      - {question: "dup2", header: "H"}
output: ${ask.output}
"""

_REF_UNDECLARED = """
id: f
name: f
nodes:
  ask: {kind: human_input, questions: "${qs}"}
output: ${ask.output}
"""

_POOL_REF_IN_FIELD = """
id: f
name: f
nodes:
  a: {kind: code, output: list[object], code: tests.seeds.fns:questions_seed, input: {seed: "${input.s}"}}
  ask: {kind: human_input, questions: "${a.output}"}
input: {s: str}
output: ${ask.output}
"""


def test_rejects_bad_literal():
    with pytest.raises(LoadError, match="header"):
        load_flow(_BAD_LITERAL)


def test_rejects_ref_to_undeclared_input():
    with pytest.raises(LoadError):
        load_flow(_REF_UNDECLARED)


def test_rejects_pool_ref_in_questions_field():
    with pytest.raises(LoadError):
        load_flow(_POOL_REF_IN_FIELD)  # a node-output ref belongs in input: wiring

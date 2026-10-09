from yian.compose import load_flow

_STATIC = """
id: f
name: f
nodes:
  ask:
    kind: human_input
    questions:
      - {question: "Which?", header: "Framework", options: [{label: React}, {label: Vue}]}
output: ${ask.output}
"""

_REF = """
id: f
name: f
input: {seed: str}
nodes:
  src:
    kind: code
    input: {seed: "${input.seed}"}
    output: list[object]
    code: tests.seeds.fns:questions_seed   # returns a questions list
  ask:
    kind: human_input
    input: {qs: "${src.output}"}
    questions: "${qs}"
output: ${ask.output}
"""


def test_static_questions_carry_literal_and_object_output():
    n = load_flow(_STATIC).compiled.nodes["ask"]
    assert n.questions and n.questions_input is None
    assert n.output_contract.type.kind.value == "object"


def test_ref_questions_record_input_param():
    n = load_flow(_REF).compiled.nodes["ask"]
    assert n.questions_input == "qs" and n.questions is None

from agent_composer.compose import load_flow

_AUTO = """
id: f
name: f
input: {ctx: str}
nodes:
  ask:
    kind: human_input
    input: {ctx: "${input.ctx}"}
    prompt: "Help me with ${ctx}."
    adaptive_questions:
      prompt: "Design 1-3 questions with options for: ${ctx}"
output: ${ask.output}
"""


def test_adaptive_questions_lowers_to_agent_plus_gate():
    nodes = load_flow(_AUTO).compiled.nodes
    assert "ask" in nodes and "ask__compose" in nodes
    gate = nodes["ask"]
    agent = nodes["ask__compose"]
    assert agent.kind.name == "AGENT"
    assert agent.output_contract.type.kind.value == "list[object]"
    assert gate.questions_input is not None
    assert gate.output_contract.type.kind.value == "object"


def test_adaptive_questions_scope_is_validated_like_an_agent():
    import pytest
    from agent_composer.compose.helpers.errors import LoadError
    bad = _AUTO.replace("for: ${ctx}", "for: ${undeclared}")
    with pytest.raises(LoadError):
        load_flow(bad)   # synth agent inherits agent prompt-scope checking

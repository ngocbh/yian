"""The `agent` kind handle: parse ranges and reject retired continuation fields."""

import pytest

from agent_composer.compose import LoadError
from agent_composer.compose.nodes.agent import AgentDescriptor
from agent_composer.compose.source import SourceDocument
from agent_composer.diagnostics import SourceRange


_FLOW = """\
nodes:
  writer:
    kind: agent
    prompt: "Write about ${topic}"
    inputs:
      topic: ${input.topic}
    outputs: string
"""


def _reader_for(text, node_id):
    return SourceDocument.parse(text).root.child("nodes").child(node_id)


def test_agent_parse_matches_today_descriptor_and_stamps_ranges():
    reader = _reader_for(_FLOW, "writer")
    desc = AgentDescriptor.parse("writer", reader)

    # The descriptor is exactly today's AgentDescriptor (field_ranges excluded from ==).
    assert desc == AgentDescriptor(
        id="writer",
        prompt="Write about ${topic}",
        inputs={"topic": "${input.topic}"},
        outputs="string",
    )
    # Ranges are recorded for the node body and its fields.
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges[("field", "prompt")], SourceRange)
    assert isinstance(desc.field_ranges[("field", "inputs")], SourceRange)
    # The prompt value sits on line 4 of the flow (1-based).
    assert desc.field_ranges[("field", "prompt")].start_line == 4


def test_agent_descriptor_rejects_controls_field():
    text = """\
nodes:
  writer:
    kind: agent
    prompt: hi
    controls: [ask_user]
"""
    with pytest.raises(LoadError, match="controls.*not allowed"):
        AgentDescriptor.parse("writer", _reader_for(text, "writer"))


def test_agent_variable_declarations_remain_separate_from_binding_fields():
    text = """\
nodes:
  writer:
    kind: agent
    variables:
      num_sentences: int[1,12] = 4
      temperature: float[0,1] = 0.4
      tone: Literal[concise,detailed,technical] = concise
      instruction: str = "Summarize clearly"
    prompt: "${instruction} in ${num_sentences} sentences."
"""
    descriptor = AgentDescriptor.parse("writer", _reader_for(text, "writer"))
    assert descriptor.variables == {
        "num_sentences": "int[1,12] = 4",
        "temperature": "float[0,1] = 0.4",
        "tone": "Literal[concise,detailed,technical] = concise",
        "instruction": 'str = "Summarize clearly"',
    }
    assert descriptor.inputs == {}
    assert descriptor.binding_fields == ("inputs",)

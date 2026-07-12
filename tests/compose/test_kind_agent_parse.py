"""6.1 — the `agent` kind handle: parse records field ranges; descriptor unchanged."""

from agent_composer.compose.kinds.agent import AgentDescriptor
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans
from agent_composer.events import SourceRange


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
    value, spans, root = load_with_spans(text, return_root=True)
    root_reader = NodeReader(value, spans, root)
    return root_reader.child("nodes").child(node_id)


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
    assert isinstance(desc.field_ranges["prompt"], SourceRange)
    assert isinstance(desc.field_ranges["inputs"], SourceRange)
    # The prompt value sits on line 4 of the flow (1-based).
    assert desc.field_ranges["prompt"].start_line == 4

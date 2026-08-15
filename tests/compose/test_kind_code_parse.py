"""6.2 — the `code` kind handle: parse records the code body's range; build reconciles the
inline `def main(inputs):` wrapper offset so an inline syntax slip frames at the YAML line."""

import pytest

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.nodes.base import BuildContext
from agent_composer.compose.nodes.code import CodeDescriptor
from agent_composer.compose.source import SourceDocument
from agent_composer.events import SourceRange


_FLOW = """\
nodes:
  verdict:
    kind: code
    inputs:
      rating: ${input.rating}
    outputs: str
    code: |
      x = inputs["rating"]
      return x
"""


def _reader_for(text, node_id):
    return SourceDocument.parse(text).root.child("nodes").child(node_id)


def test_code_parse_matches_descriptor_and_records_body_range():
    reader = _reader_for(_FLOW, "verdict")
    desc = CodeDescriptor.parse("verdict", reader)

    # The descriptor is exactly today's CodeDescriptor (field_ranges excluded from ==).
    assert desc == CodeDescriptor(
        id="verdict",
        code='x = inputs["rating"]\nreturn x\n',
        inputs={"rating": "${input.rating}"},
        outputs="str",
    )
    # Ranges are recorded for the node body and its fields (the code body carries its own).
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges[("field", "code")], SourceRange)
    assert isinstance(desc.field_ranges[("field", "inputs")], SourceRange)


def test_code_build_returns_code_node_and_stamps_ranges():
    reader = _reader_for(_FLOW, "verdict")
    desc = CodeDescriptor.parse("verdict", reader)
    result = desc.build(BuildContext({}, source_file="flow.yaml"))
    node, wiring = result.node, result.wiring

    from agent_composer.nodes.code import CodeNode

    assert isinstance(node, CodeNode)
    assert node.output_type is not None
    assert node.range is desc.range or node.range == desc.range
    assert wiring == {"rating": "${input.rating}"}


def test_code_build_frames_inline_syntax_error_at_yaml_line():
    # `code: |` indicator is on line 7; the body's first (and only) line is line 8.
    text = (
        "nodes:\n"
        "  broken:\n"
        "    kind: code\n"
        "    inputs:\n"
        "      rating: ${input.rating}\n"
        "    outputs: string\n"
        "    code: |\n"
        "      return (\n"
    )
    reader = _reader_for(text, "broken")
    desc = CodeDescriptor.parse("broken", reader)
    with pytest.raises(LoadError) as ei:
        desc.build(BuildContext({}, source_file="flow.yaml"))
    # The padded compile frames the slip at its absolute YAML line (8), not body-relative.
    assert ei.value.line == 8

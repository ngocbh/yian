"""6.4 — the `tool` kind handle: parse records field ranges into the descriptor; build
returns a runtime ToolNode with `inputs` wiring (the tool's sink surface) and stamped ranges."""

from agent_composer.compose.nodes.base import BuildContext
from agent_composer.compose.nodes.tool import ToolDescriptor
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans
from agent_composer.events import SourceRange


_FLOW = """\
nodes:
  fetch:
    kind: tool
    tool_id: http_get
    inputs:
      url: ${input.url}
"""


def _reader_for(text, node_id):
    value, spans, root = load_with_spans(text, return_root=True)
    root_reader = NodeReader(value, spans, root)
    return root_reader.child("nodes").child(node_id)


def test_tool_parse_matches_descriptor_and_records_ranges():
    reader = _reader_for(_FLOW, "fetch")
    desc = ToolDescriptor.parse("fetch", reader)

    assert desc == ToolDescriptor(
        id="fetch",
        tool_id="http_get",
        inputs={"url": "${input.url}"},
    )
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges[("field", "tool_id")], SourceRange)
    assert isinstance(desc.field_ranges[("field", "inputs")], SourceRange)


def test_tool_build_returns_tool_node_and_wires_inputs():
    reader = _reader_for(_FLOW, "fetch")
    desc = ToolDescriptor.parse("fetch", reader)
    result = desc.build(BuildContext({}))
    node, wiring = result.node, result.wiring

    from agent_composer.nodes.tool import ToolNode

    assert isinstance(node, ToolNode)
    assert node.tool_id == "http_get"
    # A tool's sink surface is the base's `inputs` map; the wiring carries those sources.
    assert wiring == {"url": "${input.url}"}
    assert node.range == desc.range

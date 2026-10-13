"""The `tool` kind handle: parsing records field ranges in the descriptor; build
returns a runtime ToolNode with `inputs` wiring (the tool's sink surface) and stamped ranges,
and turns an optional authored `output:` into the node's output contract."""

from yian.compose.nodes.base import BuildContext
from yian.compose.nodes.tool import ToolDescriptor
from yian.compose.source import SourceDocument
from yian.diagnostics import SourceRange
from yian.typesys import Type, ValueKind


_FLOW = """\
nodes:
  fetch:
    kind: tool
    tool_id: http_get
    inputs:
      url: ${input.url}
"""


def _reader_for(text, node_id):
    return SourceDocument.parse(text).root.child("nodes").child(node_id)


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

    from yian.nodes.tool import ToolNode

    assert isinstance(node, ToolNode)
    assert node.tool_id == "http_get"
    # A tool's sink surface is the base's `inputs` map; the wiring carries those sources.
    assert wiring == {"url": "${input.url}"}
    assert result.node_range == desc.range


def test_a_tool_node_with_an_authored_output_loads():
    from yian.compose import load_flow

    flow = load_flow(
        """\
id: count
name: count
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: 5
    output: int
output: ${place.output}
"""
    )

    assert flow.compiled.nodes["place"].output_contract.type == Type.scalar(ValueKind.INTEGER)


def _built_tool(output_yaml):
    text = f"""\
nodes:
  place:
    kind: tool
    tool_id: place_order
    inputs:
      qty: 5
    outputs: {output_yaml}
"""
    desc = ToolDescriptor.parse("place", _reader_for(text, "place"))
    return desc.build(BuildContext({})).node


def test_tool_build_sets_a_scalar_output_contract():
    node = _built_tool("int")

    assert node.output_contract is not None
    assert node.output_contract.type == Type.scalar(ValueKind.INTEGER)


def test_tool_build_sets_a_record_output_contract():
    node = _built_tool("{qty: int}")

    fields = node.output_contract.type.fields
    assert set(fields) == {"qty"}
    assert fields["qty"] == Type.scalar(ValueKind.INTEGER)

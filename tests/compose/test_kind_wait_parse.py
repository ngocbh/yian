"""The ``wait`` kind parses one required absolute ``until`` deadline."""

from yian.compose.nodes.base import BuildContext
from yian.compose.nodes.wait import WaitDescriptor
from yian.compose.source import SourceDocument
from yian.diagnostics import SourceRange


_FLOW = """\
nodes:
  hold:
    kind: wait
    until: ${input.deadline}
"""


def _reader_for(text, node_id):
    return SourceDocument.parse(text).root.child("nodes").child(node_id)


def test_wait_parse_matches_descriptor_and_records_ranges():
    reader = _reader_for(_FLOW, "hold")
    desc = WaitDescriptor.parse("hold", reader)

    assert desc == WaitDescriptor(id="hold", until="${input.deadline}")
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges[("field", "until")], SourceRange)


def test_wait_build_returns_blocking_node_with_reserved_until_wiring():
    reader = _reader_for(_FLOW, "hold")
    desc = WaitDescriptor.parse("hold", reader)
    result = desc.build(BuildContext({}))
    node, wiring = result.node, result.wiring

    from yian.nodes.wait import WaitNode

    assert isinstance(node, WaitNode)
    # The `until` source rides the reserved wiring key; the node carries no author params.
    assert node.params == []
    assert wiring == {"until": "${input.deadline}"}
    assert result.node_range == desc.range

"""6.5 — the `wait` kind handle: parse records the `until` field range; build returns a timed
WaitNode whose `until` SOURCE rides the reserved wiring key (no author params)."""

from agent_composer.compose.nodes.base import BuildContext
from agent_composer.compose.nodes.wait import WaitDescriptor
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans
from agent_composer.events import SourceRange


_FLOW = """\
nodes:
  hold:
    kind: wait
    until: ${input.deadline}
"""


def _reader_for(text, node_id):
    value, spans, root = load_with_spans(text, return_root=True)
    root_reader = NodeReader(value, spans, root)
    return root_reader.child("nodes").child(node_id)


def test_wait_parse_matches_descriptor_and_records_ranges():
    reader = _reader_for(_FLOW, "hold")
    desc = WaitDescriptor.parse("hold", reader)

    assert desc == WaitDescriptor(id="hold", until="${input.deadline}")
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges[("field", "until")], SourceRange)


def test_wait_build_returns_timed_node_with_reserved_until_wiring():
    reader = _reader_for(_FLOW, "hold")
    desc = WaitDescriptor.parse("hold", reader)
    result = desc.build(BuildContext({}))
    node, wiring = result.node, result.wiring

    from agent_composer.nodes.wait import WaitNode

    assert isinstance(node, WaitNode)
    assert node.is_timed is True
    # The `until` source rides the reserved wiring key; the node carries no author params.
    assert node.params == []
    assert wiring == {"until": "${input.deadline}"}
    assert node.range == desc.range

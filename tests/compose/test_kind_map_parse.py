"""6.9 — the `map` kind handle: `MapDescriptor` subclasses call's `CallDescriptor`, inheriting
the template parse; it records the `call`/`over`/`inputs` field ranges. `over` (the
iteration source) is required alongside `call`; build (`CallDescriptor.build`, `MapNode` branch on
`isinstance(self, MapDescriptor)`) is exercised end-to-end via the `load_flow` suite."""

import pytest

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.nodes.map import MapDescriptor
from agent_composer.compose.source import SourceDocument
from agent_composer.diagnostics import SourceRange


_FLOW = """\
nodes:
  fan:
    kind: map
    call: child_flow
    over: ${input.items}
    inputs:
      x: ${item}
"""


def _reader_for(text, node_id):
    return SourceDocument.parse(text).root.child("nodes").child(node_id)


def test_map_parse_matches_descriptor_and_records_ranges():
    reader = _reader_for(_FLOW, "fan")
    desc = MapDescriptor.parse("fan", reader)

    assert desc == MapDescriptor(
        id="fan",
        call="child_flow",
        over="${input.items}",
        inputs={"x": "${item}"},
    )
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges[("field", "call")], SourceRange)
    assert isinstance(desc.field_ranges[("field", "over")], SourceRange)
    assert isinstance(desc.field_ranges[("field", "inputs")], SourceRange)


def test_map_parse_requires_over():
    flow = "nodes:\n  fan:\n    kind: map\n    call: child_flow\n    inputs:\n      x: ${item}\n"
    reader = _reader_for(flow, "fan")
    with pytest.raises(LoadError, match="missing required field 'over'"):
        MapDescriptor.parse("fan", reader)


def test_map_parse_records_parallel():
    flow = (
        "nodes:\n"
        "  fan:\n"
        "    kind: map\n"
        "    call: child_flow\n"
        "    over: ${input.items}\n"
        "    parallel: true\n"
    )
    reader = _reader_for(flow, "fan")
    desc = MapDescriptor.parse("fan", reader)
    assert desc.parallel is True
    assert isinstance(desc.field_ranges[("field", "parallel")], SourceRange)

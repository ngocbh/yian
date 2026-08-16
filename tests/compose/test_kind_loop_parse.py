"""6.10 — the `loop` kind handle: parse records the `call`/`while`/`until`/`times`/`max` field
ranges and maps the Python-keyword surface keys `while:`/`until:` onto the `while_`/`until_`
fields; the required flat field is `call` (the body subflow). Build (drive of `build_loop_node`,
predicate-slice legality) is exercised end-to-end via the `load_flow` suite."""

import pytest

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.nodes.loop import LoopDescriptor
from agent_composer.compose.source import SourceDocument
from agent_composer.diagnostics import SourceRange


_FLOW = """\
nodes:
  drive:
    kind: loop
    call: body_flow
    inputs:
      acc: ${input.seed}
    while: ${carry.go}
    max: 10
"""


def _reader_for(text, node_id):
    return SourceDocument.parse(text).root.child("nodes").child(node_id)


def test_loop_parse_matches_descriptor_and_records_ranges():
    reader = _reader_for(_FLOW, "drive")
    desc = LoopDescriptor.parse("drive", reader)

    # `while:` (a Python keyword) maps onto the `while_` field.
    assert desc == LoopDescriptor(
        id="drive",
        call="body_flow",
        inputs={"acc": "${input.seed}"},
        while_="${carry.go}",
        max=10,
    )
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges[("field", "call")], SourceRange)
    assert isinstance(desc.field_ranges[("field", "inputs")], SourceRange)
    # Range recorded under the RAW surface key text (`while`), not the field name (`while_`).
    assert isinstance(desc.field_ranges[("field", "while")], SourceRange)
    assert isinstance(desc.field_ranges[("field", "max")], SourceRange)


def test_loop_parse_requires_call():
    flow = "nodes:\n  drive:\n    kind: loop\n    times: 3\n"
    reader = _reader_for(flow, "drive")
    with pytest.raises(LoadError, match="missing required field 'call'"):
        LoopDescriptor.parse("drive", reader)


def test_loop_parse_maps_until_keyword():
    flow = (
        "nodes:\n"
        "  drive:\n"
        "    kind: loop\n"
        "    call: body_flow\n"
        "    until: ${carry.done}\n"
        "    max: 5\n"
    )
    reader = _reader_for(flow, "drive")
    desc = LoopDescriptor.parse("drive", reader)
    assert desc.until_ == "${carry.done}"
    assert desc.while_ is None
    assert isinstance(desc.field_ranges[("field", "until")], SourceRange)

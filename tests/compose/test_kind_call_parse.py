"""6.8 — the `call` kind handle: parse stamps `kind="call"` onto the shared
`CallDescriptor` and records the `call`/`inputs` field ranges; the required flat field is
`call` (the callable). Build (drive of `build_call_node`) is exercised end-to-end via the
`load_flow` suite; this module covers the located parse."""

import pytest

from agent_composer.compose.errors import LoadError
from agent_composer.compose.kinds.call import CallDescriptor, parse
from agent_composer.compose.reader import NodeReader
from agent_composer.compose.spanloader import load_with_spans
from agent_composer.events import SourceRange


_FLOW = """\
nodes:
  step:
    kind: call
    call: child_flow
    inputs:
      x: ${input.x}
"""


def _reader_for(text, node_id):
    value, spans, root = load_with_spans(text, return_root=True)
    root_reader = NodeReader(value, spans, root)
    return root_reader.child("nodes").child(node_id)


def test_call_parse_matches_descriptor_and_records_ranges():
    reader = _reader_for(_FLOW, "step")
    desc = parse("step", reader)

    assert desc == CallDescriptor(
        id="step",
        call="child_flow",
        inputs={"x": "${input.x}"},
        kind="call",
    )
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges["call"], SourceRange)
    assert isinstance(desc.field_ranges["inputs"], SourceRange)


def test_call_parse_requires_call():
    flow = "nodes:\n  step:\n    kind: call\n    inputs:\n      x: ${input.x}\n"
    reader = _reader_for(flow, "step")
    with pytest.raises(LoadError, match="missing required field 'call'"):
        parse("step", reader)


def test_call_parse_rejects_non_string_asserts():
    flow = (
        "nodes:\n"
        "  step:\n"
        "    kind: call\n"
        "    call: child_flow\n"
        "    asserts:\n"
        "      - 5\n"
    )
    reader = _reader_for(flow, "step")
    with pytest.raises(LoadError, match="`asserts:` must be a list of strings"):
        parse("step", reader)

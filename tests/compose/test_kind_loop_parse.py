"""The `loop` kind handle: parsing records the `call`/`while`/`until`/`times`/`max` field
ranges and maps the Python-keyword surface keys `while:`/`until:` onto the `while_`/`until_`
fields; the required flat field is `call` (the body subflow). `LoopDescriptor.build`, including
predicate-reference validation, is exercised end-to-end via the `load_flow` suite."""

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.nodes.loop import LoopDescriptor
from yian.compose.source import SourceDocument
from yian.diagnostics import SourceRange


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


_PREDICATE_FLOW = """\
id: f
name: f
input:
  n: int
defs:
  body:
    input:
      x: int
    nodes:
      inc:
        kind: code
        code: return inputs["x"] + 1
        input:
          x: ${input.x}
        output: int
    output:
      x: ${inc.output}
nodes:
  lp:
    kind: loop
    call: body
    input:
      x: ${input.n}
    PREDICATE
    max: 3
output: ${lp.output}
"""


@pytest.mark.parametrize(
    ("predicate", "message"),
    [
        ("while: true", r"`while:` must be an expression string .* \(got True\)"),
        ("while: false", r"`while:` must be an expression string .* \(got False\)"),
        ("until: 1", r"`until:` must be an expression string .* \(got 1\)"),
        ("until: [x]", r"`until:` must be an expression string .* \(got \['x'\]\)"),
    ],
)
def test_loop_rejects_a_non_string_predicate_at_its_field(predicate, message):
    """A YAML bool/number/list predicate is a located load error, not a crash or a runtime failure."""
    from yian.compose import load_flow

    with pytest.raises(LoadError, match=message) as excinfo:
        load_flow(_PREDICATE_FLOW.replace("PREDICATE", predicate))
    assert excinfo.value.diagnostic.frames[-1].range.start_line == 24


def test_loop_rejects_an_empty_predicate_at_load():
    """`while: ''` is falsy but present: it fails to parse at load, not at runtime."""
    from yian.compose import load_flow

    with pytest.raises(LoadError, match=r"`while:` is not a valid boolean expression"):
        load_flow(_PREDICATE_FLOW.replace("PREDICATE", "while: ''"))

"""The CODE descriptor records source and resolves wrapped syntax lines by origin."""

from dataclasses import replace

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadDiagnostics, LoadError
from yian.compose.nodes.base import BuildContext
from yian.compose.nodes.code import CodeDescriptor
from yian.compose.source import SourceDocument
from yian.compose.source.catalog import _SourceCatalogBuilder
from yian.diagnostics import OriginRef
from yian.diagnostics import SourceRange


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


def test_code_build_returns_code_node_and_retains_compose_ranges():
    reader = _reader_for(_FLOW, "verdict")
    desc = CodeDescriptor.parse("verdict", reader)
    result = desc.build(BuildContext({}, source_file="flow.yaml"))
    node, wiring = result.node, result.wiring

    from yian.nodes.code import CodeNode

    assert isinstance(node, CodeNode)
    assert node.output_contract is not None
    assert result.node_range is desc.range or result.node_range == desc.range
    assert wiring == {"rating": "${input.rating}"}


def test_code_build_stamps_descriptor_origin_without_affecting_equality():
    """The central finish seam transfers opaque provenance to every built kind."""
    parsed = CodeDescriptor.parse("verdict", _reader_for(_FLOW, "verdict"))
    origin = OriginRef("root/node/verdict")
    located = replace(parsed, origin=origin)

    result = located.build(BuildContext({}, source_file="flow.yaml"))

    assert located == parsed
    assert result.node.origin is origin
    with pytest.raises(
        TypeError, match="node descriptor origin must be an OriginRef or None"
    ):
        replace(parsed, origin="root/node/verdict")  # type: ignore[arg-type]


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
    builder = _SourceCatalogBuilder()
    desc = CodeDescriptor.parse("broken", reader).with_source_origins(
        reader,
        site_path=("root", "node", "broken"),
        catalog_builder=builder,
    )
    with pytest.raises(LoadError) as ei:
        desc.build(
            BuildContext(
                {},
                source_file="flow.yaml",
                resolve_origin=builder.resolve_registered,
                diagnostics=LoadDiagnostics.from_cursor(reader),
            )
        )
    # Wrapper line 2 selects the first exact constructed CODE line (YAML line 8).
    assert len(ei.value.diagnostic.frames) == 1
    assert ei.value.diagnostic.frames[0].range.start_line == 8
    assert ei.value.diagnostic.frames[0].text == text
    assert ei.value.diagnostic.message == "node 'broken': '(' was never closed"


@pytest.mark.parametrize(
    "code_field",
    [
        '    code: "return (\\n"\n',
        "    code: >-\n      return (\n",
    ],
    ids=["quoted", "folded"],
)
def test_nonliteral_syntax_error_uses_whole_code_scalar(code_field):
    text = (
        "id: syntax-fallback\n"
        "name: syntax-fallback\n"
        "nodes:\n"
        "  broken:\n"
        "    kind: code\n"
        "    outputs: string\n"
        f"{code_field}"
        "output: ${broken.output}\n"
    )
    code_cursor = (
        SourceDocument.parse(text, label="flow.yaml")
        .root.child("nodes")
        .child("broken")
        .child("code")
    )

    with pytest.raises(LoadError) as caught:
        load_flow(text, source_label="flow.yaml")

    assert caught.value.diagnostic.message == "node 'broken': '(' was never closed"
    assert caught.value.diagnostic.frames[0].range == code_cursor.range

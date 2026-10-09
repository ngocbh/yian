"""The `model` kind handle: parsing records field ranges in the descriptor; build
returns a runtime ModelNode with an output contract, wiring, and stamped ranges."""

from yian.compose.nodes.base import BuildContext
from yian.compose.nodes.model import ModelDescriptor
from yian.compose.source import SourceDocument
from yian.diagnostics import SourceRange


_FLOW = """\
nodes:
  infer:
    kind: model
    model_id: sentiment-v2
    inputs:
      text: ${input.text}
    outputs: str
"""


def _reader_for(text, node_id):
    return SourceDocument.parse(text).root.child("nodes").child(node_id)


def test_model_parse_matches_descriptor_and_records_ranges():
    reader = _reader_for(_FLOW, "infer")
    desc = ModelDescriptor.parse("infer", reader)

    assert desc == ModelDescriptor(
        id="infer",
        model_id="sentiment-v2",
        inputs={"text": "${input.text}"},
        outputs="str",
    )
    assert isinstance(desc.range, SourceRange)
    assert isinstance(desc.field_ranges[("field", "model_id")], SourceRange)
    assert isinstance(desc.field_ranges[("field", "inputs")], SourceRange)


def test_model_build_returns_model_node_and_retains_compose_ranges():
    reader = _reader_for(_FLOW, "infer")
    desc = ModelDescriptor.parse("infer", reader)
    result = desc.build(BuildContext({}))
    node, wiring = result.node, result.wiring

    from yian.nodes.model import ModelNode

    assert isinstance(node, ModelNode)
    assert node.model_id == "sentiment-v2"
    assert node.output_contract is not None
    assert result.node_range == desc.range
    assert wiring == {"text": "${input.text}"}

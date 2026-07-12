"""6.3 — the `model` kind handle: parse records field ranges into the descriptor; build
returns a runtime ModelNode with output_type + wiring and stamped ranges."""

from agent_composer.compose.kinds.model import ModelDescriptor
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans
from agent_composer.events import SourceRange


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
    value, spans, root = load_with_spans(text, return_root=True)
    root_reader = NodeReader(value, spans, root)
    return root_reader.child("nodes").child(node_id)


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
    assert isinstance(desc.field_ranges["model_id"], SourceRange)
    assert isinstance(desc.field_ranges["inputs"], SourceRange)


def test_model_build_returns_model_node_and_stamps_ranges():
    reader = _reader_for(_FLOW, "infer")
    desc = ModelDescriptor.parse("infer", reader)
    node, wiring = desc.build(registry={}, resolver=None)

    from agent_composer.nodes.model import ModelNode

    assert isinstance(node, ModelNode)
    assert node.model_id == "sentiment-v2"
    assert node.output_type is not None
    assert node.range == desc.range
    assert wiring == {"text": "${input.text}"}

"""`AgentDescriptor.build(ctx)` honors `AgentDescriptor.output_contract_override`.

The adaptive_questions desugar pass synthesizes an AGENT whose structured
output is a code-built `list(Question)` Type (it has no surface type-string).
The override lets its canonical contract win over the surface-derived one.
"""

from yian.compose.parse.file import AgentDescriptor
from yian.compose.nodes.base import BuildContext
from yian.nodes.contracts import ValueContract
from yian.nodes.human_input.questions import question_list_type


def test_contract_override_wins_without_invoking_the_output_reader():
    """A semantic override bypasses an authored-declaration reader entirely."""
    contract = ValueContract(question_list_type())
    desc = AgentDescriptor(
        id="g__compose",
        prompt="x",
        inputs={},
        outputs="str",
        output_contract_override=contract,
    )

    def unexpected_reader(_node_id, _output_spec):
        raise AssertionError("an output override must bypass the Compose reader")

    node = desc.build(BuildContext({}, output_contract_reader=unexpected_reader)).node
    assert node.output_contract is contract

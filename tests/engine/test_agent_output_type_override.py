"""`AgentDescriptor.build(ctx)` honors `AgentDescriptor.output_contract_override`.

The adaptive_questions desugar pass synthesizes an AGENT whose structured
output is a code-built `list[Question]` Type (it has no surface type-string).
The override lets its canonical contract win over the surface-derived one.
"""

from agent_composer.compose.parse.file import AgentDescriptor
from agent_composer.compose.nodes.base import BuildContext
from agent_composer.nodes.contracts import ValueContract
from agent_composer.nodes.human_input.questions import question_list_type


def test_contract_override_wins_over_outputs_string():
    contract = ValueContract(question_list_type())
    desc = AgentDescriptor(
        id="g__compose",
        prompt="x",
        inputs={},
        outputs="str",
        output_contract_override=contract,
    )
    node = desc.build(BuildContext({})).node
    assert node.output_contract is contract

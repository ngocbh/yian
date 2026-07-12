"""build_leaf_node honors AgentDescriptor.output_type_override.

The adaptive_questions desugar pass synthesizes an AGENT whose structured
output is a code-built `list[Question]` Type (it has no surface type-string).
The override lets that code-built Type win over the type-string-derived one.
"""

from agent_composer.compose.parse.file import AgentDescriptor
from agent_composer.compose.build import build_leaf_node
from agent_composer.nodes.human_input.questions import question_list_type


def test_override_wins_over_outputs_string():
    desc = AgentDescriptor(
        id="g__compose",
        prompt="x",
        inputs={},
        output_type_override=question_list_type(),
    )
    node, _ = build_leaf_node(desc, {})
    assert node.output_type.kind.value == "list[object]"

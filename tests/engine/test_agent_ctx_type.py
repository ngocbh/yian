"""`AgentNode._ctx` threads the node's declared `output_type` into the mode context."""

from agent_composer.nodes.agent.node import AgentNode
from agent_composer.typesys.values import Type, ValueKind


def test_ctx_carries_output_type():
    n = AgentNode("a", prompt="hi")
    n.output_type = Type.scalar(ValueKind.INTEGER)
    ctx = n._ctx(prompt="hi")
    assert ctx.output_type == n.output_type

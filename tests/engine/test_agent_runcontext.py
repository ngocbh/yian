"""`AgentRunContext` carries one local loop's immutable configuration data."""

from agent_composer.nodes.agent.modes.common import AgentRunContext


def test_runcontext_carries_config_no_scratch():
    ctx = AgentRunContext(node_id="a", prompt="p", llm_config={"model": "test"})
    assert ctx.llm_config == {"model": "test"}
    assert not hasattr(ctx, "scratch")
    assert not hasattr(ctx, "model")

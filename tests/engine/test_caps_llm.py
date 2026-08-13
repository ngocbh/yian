"""P7: LLM client injected via caps["llm"], gated on the needs_llm trait."""
from agent_composer.nodes.base import Node
from agent_composer.nodes.agent.node import AgentNode


def test_base_node_does_not_need_llm():
    assert Node.needs_llm is False


def test_agent_node_needs_llm():
    assert AgentNode.needs_llm is True


from agent_composer.nodes.base import NodeKind, Output
from agent_composer.typesys.pool import VariablePool
from agent_composer.runtime.eval_node import eval_node, _default_llm


def test_eval_node_passes_llm_cap_to_needs_llm_node():
    seen = {}

    class Fake(AgentNode):
        def _run_impl(self, inputs, **caps):
            seen["llm"] = caps.get("llm")
            return Output(value="ok")

    node = Fake("a", prompt="hi")
    sentinel = lambda cfg: object()
    list(eval_node(node, None, VariablePool(), llm=sentinel))
    assert seen["llm"] is sentinel


def test_eval_node_omits_llm_cap_for_plain_node():
    # A non-needs_llm node must not receive an llm cap.
    seen = {}

    class Plain(Node):
        kind = NodeKind.CODE  # any concrete kind; needs_llm stays False (base default)

        def _run_impl(self, inputs, **caps):
            seen["caps"] = dict(caps)
            return Output(value="ok")

    node = Plain("p")
    list(eval_node(node, None, VariablePool(), llm=lambda cfg: object()))
    assert "llm" not in seen["caps"]


def test_flow_engine_default_llm_is_lazy_thunk():
    from agent_composer.compile.model import CompiledFlow
    from agent_composer.runtime.engine import FlowEngine
    from tests.engine._fakes import FuncNode

    flow = CompiledFlow(nodes={"n": FuncNode("n", lambda p: {})}, edges=[])
    eng = FlowEngine(flow)
    assert eng.llm is _default_llm


def test_flow_engine_uses_explicit_llm():
    from agent_composer.compile.model import CompiledFlow
    from agent_composer.runtime.engine import FlowEngine
    from tests.engine._fakes import FuncNode

    flow = CompiledFlow(nodes={"n": FuncNode("n", lambda p: {})}, edges=[])
    sentinel = lambda cfg: object()
    eng = FlowEngine(flow, llm=sentinel)
    assert eng.llm is sentinel


def test_agent_run_uses_injected_llm_over_default(monkeypatch):
    """When caps['llm'] is supplied, the AGENT node builds its model from the cap and NEVER
    calls the package `model_from_config`."""
    import pytest
    from langchain_core.messages import AIMessage
    import agent_composer.llm_clients as llm_mod
    from agent_composer.llm_clients import LLMConfig

    # If the cap were ignored, the node would fall back to the (monkeypatched) package factory.
    monkeypatch.setattr(
        llm_mod, "model_from_config",
        lambda cfg: pytest.fail("used default package factory, not caps['llm']"),
    )

    class _FakeChat:
        def __init__(self, reply):
            self._reply = reply
            self.calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.calls += 1
            return self._reply

    chat = _FakeChat(AIMessage(content="cap answer"))
    built = {}
    cap = lambda cfg: built.setdefault("chat", chat)

    node = AgentNode("a", prompt="hi", llm_config=LLMConfig(), mode="plain")
    term = list(eval_node(node, None, VariablePool(), llm=cap))[-1]
    assert term.output == "cap answer"
    assert built["chat"] is chat
    assert chat.calls == 1


def test_agent_direct_run_falls_back_to_lazy_package_factory(monkeypatch):
    """A no-cap direct driver resolves the lower llm_clients seam at call time."""
    from langchain_core.messages import AIMessage

    import agent_composer.llm_clients as llm_mod

    class _FakeChat:
        def invoke(self, messages):
            return AIMessage(content="direct answer")

    seen = []
    chat = _FakeChat()

    def factory(config):
        seen.append(config)
        return chat

    monkeypatch.setattr(llm_mod, "model_from_config", factory)
    config = {"provider": "openai", "model": "gpt-test"}
    node = AgentNode("direct", prompt="hi", llm_config=config, mode="plain")

    result = node._run_impl({})

    assert result == Output(value="direct answer")
    assert seen == [config]



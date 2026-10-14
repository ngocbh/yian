"""AGENT `retries` and tool-loop defaults come from `yian.defaults` when the node sets none."""

from langchain_core.messages import AIMessage

import yian.tools as tools
from yian.compile.manifest import flow_manifest
from yian.compose import load_flow
from yian.nodes.agent import AgentNode
from yian.run import run_flow
from tests.engine.test_agent import _FakeChat, _FakeTool, _ai_tool_call, _patch_model

# The sentinel model has no native structured output, so the reply is parsed from text
# and a bad reply costs one retry.
_STRUCTURED = """id: scored
name: scored
llm_config: {provider: vllm, model: no-structured-sentinel}
input:
  question: !str
nodes:
  agent:
    kind: agent
    prompt: ${question}
    input:
      question: ${input.question}
    output: {score: float}
output: ${agent.output}
"""

_LOOPING = """id: looping
name: looping
llm_config: {provider: openai, model: test-model}
input:
  question: !str
nodes:
  agent:
    kind: agent
    tools: [lookup]
    prompt: ${question}
    input:
      question: ${input.question}
    output: str
output: ${agent.output}
"""


def _with_retries(text, retries):
    return text.replace("    output:", f"    retries: {retries}\n    output:", 1)


def _bad_then_good(monkeypatch):
    model = _FakeChat([AIMessage(content='{"score": "bad"}'), AIMessage(content='{"score": 1.5}')])
    _patch_model(monkeypatch, model)
    return model


def _looping(monkeypatch, replies):
    monkeypatch.setitem(tools.TOOL_REGISTRY, "lookup", _FakeTool(lambda _: "ok"))
    model = _FakeChat(replies)
    _patch_model(monkeypatch, model)
    return model


def test_agent_retries_default_of_zero_fails_on_the_first_bad_reply(monkeypatch, set_defaults):
    set_defaults(AGENT_RETRIES=0)
    model = _bad_then_good(monkeypatch)
    result = run_flow(load_flow(_STRUCTURED), {"question": "q"})
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert model.calls == 1


def test_authored_retries_win_over_the_default(monkeypatch, set_defaults):
    set_defaults(AGENT_RETRIES=0)
    model = _bad_then_good(monkeypatch)
    result = run_flow(load_flow(_with_retries(_STRUCTURED, 1)), {"question": "q"})
    assert result.output == {"score": 1.5}
    assert model.calls == 2


def test_the_manifest_holds_two_retries_by_default():
    manifest = flow_manifest(load_flow(_STRUCTURED).compiled)
    assert manifest["nodes"]["agent"]["retries"] == 2
    assert AgentNode("n").retries == 2


def test_the_manifest_holds_the_agent_retries_default(set_defaults):
    set_defaults(AGENT_RETRIES=0)
    manifest = flow_manifest(load_flow(_STRUCTURED).compiled)
    assert manifest["nodes"]["agent"]["retries"] == 0
    assert AgentNode("n").retries == 0


def test_agent_max_tool_iterations_default_caps_the_loop(monkeypatch, set_defaults):
    set_defaults(AGENT_MAX_TOOL_ITERATIONS=1)
    model = _looping(monkeypatch, [_ai_tool_call("lookup", {}, str(i)) for i in range(5)])
    result = run_flow(load_flow(_LOOPING), {"question": "q"})
    assert result.status == "failed" and result.failure.code == "node.execute"
    assert "tool-iteration cap (1)" in result.failure.message
    assert model.calls == 1


def test_authored_max_tool_iterations_wins_over_the_default(monkeypatch, set_defaults):
    set_defaults(AGENT_MAX_TOOL_ITERATIONS=1)
    replies = [_ai_tool_call("lookup", {}, "0"), _ai_tool_call("lookup", {}, "1")]
    model = _looping(monkeypatch, [*replies, AIMessage(content="done")])
    text = _LOOPING.replace("    tools:", "    env: {max_tool_iterations: 3}\n    tools:", 1)
    assert run_flow(load_flow(text), {"question": "q"}).output == "done"
    assert model.calls == 3

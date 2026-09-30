"""Host steering of a running AGENT: queued steer messages and node-scoped interrupts.

A scripted LangChain chat model stands in for the provider (no API keys). The
host inbox is a plain object with `drain(node_id)` and `interrupted(node_id)`;
only nodes authored `steerable: true` ever receive it.
"""

import pytest

import agent_composer.llm_clients as llm_clients_mod
from agent_composer.compose import load_flow
from agent_composer.nodes.agent import MODES, AgentNode
from agent_composer.nodes.base import Output
from agent_composer.run import Steering, run_flow
from agent_composer.runtime.eval_node import EvaluationFailure, eval_node
from agent_composer.typesys.pool import VariablePool
from tests.engine._stream_fakes import ScriptedStreamChat, reply
from tests.engine.test_agent_live_events import CONFIG


class Inbox:
    """A host inbox: steer messages queued per node id, plus interrupt flags."""

    def __init__(self, steers=None, *, interrupt=()):
        self.queued = {node_id: list(messages) for node_id, messages in (steers or {}).items()}
        self.interrupt = set(interrupt)
        self.calls = []

    def drain(self, node_id):
        self.calls.append(("drain", node_id))
        return self.queued.pop(node_id, [])

    def interrupted(self, node_id):
        self.calls.append(("interrupted", node_id))
        return node_id in self.interrupt


def _flow(*, steerable, second="") -> str:
    return f"""\
id: steer
name: steer
llm_config: {{provider: anthropic, model: claude-test}}
nodes:
  chat:
    kind: agent
    mode: tool_calling
    steerable: {str(steerable).lower()}
    prompt: Help the user.
    output: str
{second}output: ${{chat.output}}
"""


def _chat(monkeypatch, script):
    chat = ScriptedStreamChat(script=list(script), mode=[])
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: chat)
    return chat


# --- the capability reaches only steerable nodes ---------------------------- #


def test_inbox_satisfies_the_public_protocol():
    assert isinstance(Inbox(), Steering)


def test_non_steerable_agent_never_calls_the_inbox(monkeypatch):
    _chat(monkeypatch, [reply("done")])
    inbox = Inbox({"chat": ["ignored"]}, interrupt={"chat"})
    result = run_flow(load_flow(_flow(steerable=False)), {}, steering=inbox)
    assert result.output == "done" and inbox.calls == []


def test_mode_context_receives_steering_only_when_steerable(monkeypatch):
    seen = []

    def probe(ctx):
        seen.append(ctx.steering is not None)
        return Output(value="ok")

    monkeypatch.setitem(MODES, "tool_calling", probe)
    for steerable in (True, False):
        node = AgentNode("chat", prompt="hi", llm_config=dict(CONFIG), mode="tool_calling",
                         steerable=steerable)
        eval_node(node, None, VariablePool(), steering=Inbox())
        eval_node(node, None, VariablePool())
    assert seen == [True, False, False, False]


@pytest.mark.parametrize("bad", [object(), type("OnlyDrain", (), {"drain": lambda self, n: []})()])
def test_run_boundary_rejects_an_object_without_the_inbox_methods(bad, tmp_path):
    path = tmp_path / "run.jsonl"
    with pytest.raises(TypeError, match="steering must provide drain"):
        run_flow(load_flow(_flow(steerable=True)), {}, steering=bad, log=path)
    assert not path.exists()

"""Host steering of a running AGENT: queued steer messages and node-scoped interrupts.

A scripted LangChain chat model stands in for the provider (no API keys). The
host inbox is a plain object with `drain(node_id)` and `interrupted(node_id)`;
only nodes authored `steerable: true` ever receive it.
"""

import pytest

import agent_composer.llm_clients as llm_clients_mod
from langchain_core.messages import HumanMessage, ToolMessage

from agent_composer.compose import load_flow
from agent_composer.memory.session import SessionMemory
from agent_composer.nodes.agent import MODES, AgentNode
from agent_composer.nodes.base import Output
from agent_composer.run import Steering, run_flow
from agent_composer.runtime.eval_node import EvaluationFailure, eval_node
from agent_composer.typesys.pool import VariablePool
from tests.engine._stream_fakes import ScriptedStreamChat, reply
from tests.engine.test_agent_live_events import CONFIG
from tests.engine.test_agent_session import _view


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


def _flow(*, steerable=True, tools=(), memory=False, after="") -> str:
    """One `chat` agent granted `tools`, optionally with session memory and a later node."""
    return f"""\
id: steer
name: steer
llm_config: {{provider: anthropic, model: claude-test}}
{"long_term_memory: session" if memory else ""}
nodes:
  chat:
    kind: agent
    mode: tool_calling
    steerable: {str(steerable).lower()}
    tools: [{", ".join(tools)}]
    prompt: Help the user.
    output: str
{after}output: ${{{"after" if after else "chat"}.output}}
"""


class Recorder:
    """A duck-typed chat model that records each request and serves scripted replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        answer = self.replies.pop(0)
        return answer(messages) if callable(answer) else answer


def _recorder(monkeypatch, replies):
    model = Recorder(replies)
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: model)
    return model


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
        run_flow(load_flow(_flow()), {}, steering=bad, log=path)
    assert not path.exists()


# --- steer messages ----------------------------------------------------------- #


def test_steers_are_sent_as_user_messages_at_the_top_of_each_iteration(monkeypatch):
    inbox = Inbox({"chat": ["use dataset B"]})

    def lookup(topic: str) -> str:
        """Look up a topic."""
        inbox.queued["chat"] = ["only the 2024 rows", "and cite them"]
        return f"found {topic}"

    model = _recorder(monkeypatch, [reply(calls=[("lookup", {"topic": "a"}, "call-1")]), reply("done")])
    result = run_flow(load_flow(_flow(tools=["lookup"])), {}, steering=inbox, tools={"lookup": lookup})
    assert result.output == "done"
    assert model.calls[0][-2:] == [HumanMessage("Help the user."), HumanMessage("use dataset B")]
    assert isinstance(model.calls[1][-3], ToolMessage) and model.calls[1][-3].content == "found a"
    assert model.calls[1][-2:] == [HumanMessage("only the 2024 rows"), HumanMessage("and cite them")]
    assert [call for call in inbox.calls if call[0] == "drain"] == [("drain", "chat")] * 2


def test_steers_are_recorded_with_the_next_model_request(monkeypatch):
    inbox = Inbox()

    def lookup(topic: str) -> str:
        """Look up a topic."""
        inbox.queued["chat"] = ["use dataset B"]
        return "found"

    model = _recorder(monkeypatch, [
        reply(calls=[("lookup", {"topic": "a"}, "call-1")]), reply("done"), reply("again"),
    ])
    memory = SessionMemory()
    flow = load_flow(_flow(tools=["lookup"], memory=True))
    run_flow(flow, {}, steering=inbox, tools={"lookup": lookup}, memory={"session": memory})
    view = _view(memory)
    requests = [effect.request["exchange_id"] for effect in view.effects if effect.kind == "model"]
    humans = [
        (item.payload["exchange_id"], item.payload["message"]["data"]["content"])
        for item in view.items if item.payload["message"]["type"] == "human"
    ]
    assert humans[-1] == (requests[1], "use dataset B")
    # The next run's history carries the steer as an ordinary user message.
    run_flow(flow, {}, steering=inbox, tools={"lookup": lookup}, memory={"session": memory})
    assert HumanMessage("use dataset B") in model.calls[-1]


def test_inbox_failure_escapes_as_a_host_fault(monkeypatch):
    class HostBroken(Exception):
        pass

    class Broken(Inbox):
        def drain(self, node_id):
            raise HostBroken("inbox down")

    _chat(monkeypatch, [reply("done")])
    with pytest.raises(HostBroken):
        run_flow(load_flow(_flow()), {}, steering=Broken())


@pytest.mark.parametrize(
    ("method", "value", "message"),
    [("drain", "one message", "drain must return a list of str"),
     ("drain", [1], "drain must return a list of str")],
)
def test_malformed_inbox_answers_escape_as_host_type_errors(monkeypatch, method, value, message):
    inbox = Inbox()
    setattr(inbox, method, lambda node_id: value)
    _chat(monkeypatch, [reply("done")])
    node = AgentNode("chat", prompt="hi", llm_config=dict(CONFIG), mode="tool_calling", steerable=True)
    with pytest.raises(TypeError, match=message):
        eval_node(node, None, VariablePool(), steering=inbox)

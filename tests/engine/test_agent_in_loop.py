"""A tool_calling agent inside a LOOP body runs to a value in-process (no mid-body pause).

Load-bearing for the `ac chat` REPL: chat uses a tool_calling reply agent inside a
per-turn LOOP. Ordinary (synchronous) tools return, never suspend, so the known
AGENT-in-loop stamping blocker (DEFER) must not apply. Fake LLM + fake tool, no keys.
"""
import agent_composer.llm_clients as llm_clients_mod
import agent_composer.tools as tools_mod
from langchain_core.messages import AIMessage
from agent_composer.compose.loader import load_flow
from agent_composer.run import run_flow


class _FakeChat:
    def __init__(self, replies):
        self._replies = list(replies)
    def bind_tools(self, tools):
        return self
    def invoke(self, messages):
        return self._replies.pop(0)


class _FakeTool:                       # real signature, mirrors tests/engine/test_agent.py
    def __init__(self, fn):
        self._fn = fn
        self.seen = []
    def invoke(self, args):
        self.seen.append(args)
        return self._fn(args)


def _tool_call(name, args, call_id="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


FLOW = """
id: t
name: t
input:
  seed: str
defs:
  body:
    input:
      transcript: str
      exited: bool
    nodes:
      reply:
        kind: agent
        mode: tool_calling
        tools: [ping]
        input:
          transcript: ${input.transcript}
        output: str
        prompt: "history ${transcript}"
      fold:
        kind: code
        code: tests.engine._compose_codefns:reply_fold
        input:
          transcript: ${input.transcript}
          reply:      ${reply.output}
          exited:     ${input.exited}
        output:
          transcript: str
          exited: bool
    output: ${fold.output}
nodes:
  loop:
    kind: loop
    call: body
    input:
      transcript: ${input.seed}
      exited: false
    times: 2
output: ${loop.output.transcript}
"""


def test_tool_calling_agent_in_loop_runs(monkeypatch):
    # Each turn: the model calls the tool once, then answers. Two turns -> 4 messages.
    chat = _FakeChat([
        _tool_call("ping", {}), AIMessage(content="turn1"),
        _tool_call("ping", {}), AIMessage(content="turn2"),
    ])
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "ping", _FakeTool(lambda a: "pong"))
    loaded = load_flow(FLOW)
    result = run_flow(loaded, {"seed": "s"})
    assert result.status == "succeeded", result.error
    assert result.output == "s|turn1|turn2"

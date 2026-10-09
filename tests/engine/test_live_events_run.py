"""End to end: a tool-calling AGENT publishes in-loop observers with token usage.

`run_flow` forwards in-loop observers to `on_event`, strictly between the node's
`NodeStarted` and `NodeSucceeded`, in serial and pooled execution. They never
reach `RunResult.events` or the run log, and usage totals equal what the
scripted provider reported.
"""

import threading

import pytest
from pydantic import BaseModel

import yian.llm_clients as llm_clients_mod
import yian.tools as tools_mod
from yian.compose.loader import load_flow
from yian.events import (
    IN_LOOP_EVENTS,
    ModelCallFinished,
    ModelCallStarted,
    NodeStarted,
    NodeSucceeded,
    ToolCallFinished,
    ToolCallStarted,
)
from yian.run import run_flow
from yian.runtime.log import JsonlRunLog
from tests.engine._stream_fakes import ScriptedStreamChat, reply, usage

FLOW = """
id: live
name: live
llm_config: {provider: anthropic, model: claude-test}
input:
  topic: str
nodes:
  writer:
    kind: agent
    mode: tool_calling
    tools: [lookup]
    input:
      topic: ${input.topic}
    output: str
    prompt: "research ${topic}"
output: ${writer.output}
"""

PAIR = """
id: pair
name: pair
llm_config: {provider: anthropic, model: claude-test}
input:
  topic: str
nodes:
  left:
    kind: agent
    mode: tool_calling
    tools: [lookup]
    input:
      topic: ${input.topic}
    output: str
    prompt: "left ${topic}"
  right:
    kind: agent
    mode: tool_calling
    tools: [lookup]
    input:
      topic: ${input.topic}
    output: str
    prompt: "right ${topic}"
output:
  left: ${left.output}
  right: ${right.output}
"""


class _Args(BaseModel):
    topic: str


class _Tool:
    args_schema = tool_call_schema = _Args
    name = "lookup"

    def __init__(self, barrier=None):
        self.barrier = barrier

    def invoke(self, args):
        if self.barrier is not None:
            self.barrier.wait()  # both AGENTs are inside a tool call at once
        return f"found:{args['topic']}"


def _script():
    return [
        reply("checking", calls=[("lookup", {"topic": "acme"}, "call-1")],
              tokens=usage(10, 4, cache_read=6)),
        reply("acme looks fine", tokens=usage(25, 3)),
    ]


@pytest.fixture
def chat(monkeypatch):
    model = ScriptedStreamChat(script=_script(), mode=[])
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: model)
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "lookup", _Tool())
    return model


@pytest.mark.parametrize("num_workers", [0, 2])
def test_tool_calling_agent_reports_ordered_events_and_usage(chat, tmp_path, num_workers):
    timeline = []
    log = tmp_path / "run.jsonl"
    result = run_flow(
        load_flow(FLOW), {"topic": "acme"}, log=log, num_workers=num_workers,
        on_event=timeline.append,
    )
    assert result.status == "succeeded" and result.output == "acme looks fine"
    assert chat.mode == ["invoke", "invoke"]

    writer = [item for item in timeline if getattr(item, "node_id", None) == "writer"]
    assert isinstance(writer[0], NodeStarted) and isinstance(writer[-1], NodeSucceeded)
    assert [type(item) for item in writer[1:-1]] == [
        ModelCallStarted, ModelCallFinished, ToolCallStarted, ToolCallFinished,
        ModelCallStarted, ModelCallFinished,
    ]
    first, second = (e.exchange_id for e in writer if isinstance(e, ModelCallStarted))
    assert first != second
    assert {e.exchange_id for e in writer if isinstance(e, ToolCallStarted)} == {first}

    finished = [e for e in writer if isinstance(e, ModelCallFinished)]
    assert sum(e.usage.input_tokens for e in finished) == 35
    assert sum(e.usage.output_tokens for e in finished) == 7
    assert sum(e.usage.total_tokens for e in finished) == 42
    assert finished[0].usage.cache_read_tokens == 6
    assert {(e.provider, e.model) for e in finished} == {("anthropic", "claude-test")}
    tool = next(e for e in writer if isinstance(e, ToolCallStarted))
    assert (tool.name, tool.arguments, tool.tool_call_id) == ("lookup", {"topic": "acme"}, "call-1")

    assert result.events == [e for e in timeline if not isinstance(e, IN_LOOP_EVENTS)]
    with JsonlRunLog.resume(log) as reopened:
        names = {type(envelope.record).__name__ for envelope in reopened}
    assert not names & {cls.__name__ for cls in IN_LOOP_EVENTS}


def test_run_without_callbacks_is_unchanged(chat):
    result = run_flow(load_flow(FLOW), {"topic": "acme"})
    assert result.output == "acme looks fine"
    assert chat.mode == ["invoke", "invoke"]
    assert not any(isinstance(event, IN_LOOP_EVENTS) for event in result.events)


def test_parallel_agents_keep_their_own_events_in_pooled_runs(monkeypatch):
    model = ScriptedStreamChat(
        script=[
            reply(calls=[("lookup", {"topic": "acme"}, "call-1")]),
            reply(calls=[("lookup", {"topic": "acme"}, "call-2")]),
            reply("done"),
            reply("done"),
        ],
        mode=[],
    )
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda _config: model)
    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY, "lookup", _Tool(threading.Barrier(2, timeout=10))
    )
    timeline = []
    result = run_flow(load_flow(PAIR), {"topic": "acme"}, num_workers=2, on_event=timeline.append)
    assert result.output == {"left": "done", "right": "done"}

    exchanges = {}
    for node_id in ("left", "right"):
        own = [i for i, item in enumerate(timeline) if getattr(item, "node_id", None) == node_id]
        started = next(i for i in own if isinstance(timeline[i], NodeStarted))
        done = next(i for i in own if isinstance(timeline[i], NodeSucceeded))
        in_loop = [i for i in own if isinstance(timeline[i], IN_LOOP_EVENTS)]
        assert len(in_loop) == 6 and started < in_loop[0] and in_loop[-1] < done
        exchanges[node_id] = {timeline[i].exchange_id for i in in_loop}
    assert not exchanges["left"] & exchanges["right"]
    # The barrier holds both tool calls open, so the two nodes' events interleave.
    tool_started = [i for i, item in enumerate(timeline) if isinstance(item, ToolCallStarted)]
    tool_finished = [i for i, item in enumerate(timeline) if isinstance(item, ToolCallFinished)]
    assert max(tool_started) < min(tool_finished)

"""Tool-calling AGENT behavior inside a dynamically grown LOOP body.

These tests cover both uninterrupted execution and durable whole-node recovery.
An interrupted AGENT does not serialize its model conversation: recovery retries
the namespaced node atomically, so already-performed tool effects are at-least-once.
Fake models and tools keep the proof deterministic and provider-free.
"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

import yian.llm_clients as llm_clients_mod
import yian.tools as tools_mod
from yian.compose.loader import load_flow
from yian.events import IN_LOOP_EVENTS
from yian.run import resume_flow, run_flow
from yian.runtime.log import (
    JsonlRunLog,
    NodeStartedRecord,
    NodeSucceededRecord,
    RunResumedRecord,
)


class _SimulatedCrash(BaseException):
    """Process-loss marker that bypasses ordinary node-failure conversion."""


class _FakeChat:
    """Return scripted model replies, including an optional host-level crash."""

    def __init__(self, replies):
        self._replies = list(replies)

    def bind_tools(self, tools):
        """Return this provider-free model after accepting the tool surface."""
        return self

    def invoke(self, messages):
        """Return the next scripted reply or raise its crash marker."""
        reply = self._replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class _NoToolArgs(BaseModel):
    """Schema for the fake ping tool, which accepts no arguments."""


class _FakeTool:
    """Minimal LangChain-compatible tool that records every external effect."""

    def __init__(self, fn):
        self._fn = fn
        self.seen = []
        self.args_schema = _NoToolArgs
        self.tool_call_schema = _NoToolArgs

    def invoke(self, args):
        """Record and execute one fake tool invocation."""
        self.seen.append(args)
        return self._fn(args)


def _tool_call(name, args, call_id="1"):
    """Build one provider-neutral model message requesting a tool call."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
        ],
    )


def _durable_records(path: Path) -> tuple[object, ...]:
    """Read one closed JSONL authority while proving its lock is available."""
    with JsonlRunLog.resume(path) as log:
        return tuple(envelope.record for envelope in log)


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
    """Two uninterrupted LOOP turns each complete one local model/tool loop."""
    # Each turn: the model calls the tool once, then answers. Two turns -> 4 messages.
    chat = _FakeChat(
        [
            _tool_call("ping", {}),
            AIMessage(content="turn1"),
            _tool_call("ping", {}),
            AIMessage(content="turn2"),
        ]
    )
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "ping", _FakeTool(lambda a: "pong"))
    loaded = load_flow(FLOW)
    result = run_flow(loaded, {"seed": "s"})
    assert result.status == "succeeded", result.failure
    assert result.output == "s|turn1|turn2"


def test_tool_calling_agent_in_loop_retries_whole_node_after_process_loss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A fresh flow/model context retries one namespaced AGENT at least once."""
    path = tmp_path / "agent-loop.jsonl"
    tool = _FakeTool(lambda _args: "pong")
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "ping", tool)

    interrupted_chat = _FakeChat(
        [
            _tool_call("ping", {}, "turn-1"),
            AIMessage(content="turn1"),
            _tool_call("ping", {}, "turn-2-before-crash"),
            _SimulatedCrash("process lost after the tool effect"),
        ]
    )
    monkeypatch.setattr(
        llm_clients_mod,
        "model_from_config",
        lambda _config: interrupted_chat,
    )

    with pytest.raises(_SimulatedCrash, match="process lost after the tool effect"):
        run_flow(
            load_flow(FLOW),
            {"seed": "s"},
            log=path,
            run_id="agent-loop-recovery",
        )

    prefix = _durable_records(path)
    assert isinstance(prefix[-1], NodeStartedRecord)
    retry_id = prefix[-1].node_id
    assert retry_id == "loop#1/reply"
    assert sum(
        isinstance(record, NodeSucceededRecord)
        and record.node_id.endswith("/reply")
        for record in prefix
    ) == 1
    assert tool.seen == [{}, {}]

    recovered_chat = _FakeChat(
        [
            _tool_call("ping", {}, "turn-2-retry"),
            AIMessage(content="turn2"),
        ]
    )
    monkeypatch.setattr(
        llm_clients_mod,
        "model_from_config",
        lambda _config: recovered_chat,
    )
    recovered_events: list[object] = []

    recovered = resume_flow(
        load_flow(FLOW),
        log=path,
        on_event=recovered_events.append,
    )

    assert recovered.status == "succeeded", recovered.failure
    assert recovered.output == "s|turn1|turn2"
    assert recovered.run_id == "agent-loop-recovery"
    # Only the re-executed AGENT publishes fresh in-loop observers; none are replayed.
    assert recovered.events == [
        event for event in recovered_events if not isinstance(event, IN_LOOP_EVENTS)
    ]
    assert tool.seen == [{}, {}, {}]
    records = _durable_records(path)
    assert [
        record for record in records if isinstance(record, RunResumedRecord)
    ] == [RunResumedRecord(retry_node_ids=(retry_id,), num_workers=8)]
    assert sum(
        isinstance(record, NodeStartedRecord) and record.node_id == retry_id
        for record in records
    ) == 2
    assert sum(
        isinstance(record, NodeSucceededRecord) and record.node_id == retry_id
        for record in records
    ) == 1

    def unexpected_model(_config):
        """Fail if terminal replay tries to execute model work again."""
        pytest.fail("terminal replay must not build a model")

    monkeypatch.setattr(llm_clients_mod, "model_from_config", unexpected_model)
    before = path.read_bytes()
    terminal_events: list[object] = []

    terminal = resume_flow(
        load_flow(FLOW),
        log=path,
        on_event=terminal_events.append,
    )

    assert terminal.status == "succeeded", terminal.failure
    assert terminal.output == recovered.output
    assert terminal.run_id == recovered.run_id
    assert terminal.events == []
    assert terminal_events == []
    assert path.read_bytes() == before
    assert tool.seen == [{}, {}, {}]

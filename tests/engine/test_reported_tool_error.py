"""`ReportedToolError`: an AGENT's model reads it as the call's result; a TOOL node fails."""

from __future__ import annotations

from threading import Event

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import yian.llm_clients as llm_clients
from yian.compose import load_flow
from yian.events import ToolCallFinished
from yian.memory.session import SessionMemory
from yian.nodes.agent.memory import decode_message
from yian.nodes.base import _HostStop
from yian.run import _run_flow_controlled, run_flow
from yian.tools import PermanentToolError, ReportedToolError
from tests.engine.test_agent_session import _view
from tests.engine.test_agent_stop_wins import _contexts
from tests.engine.test_agent_session_tools import _TOOL_FLOW as _SESSION_FLOW


class _ScriptedModel:
    """Return scripted replies and capture each request."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


_AGENT_FLOW = """
id: reported
name: reported
llm_config: {provider: openai, model: test-model}
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [read_file]
output: ${worker.output}
"""


def _tool_flow(retries: int | None = None):
    """Load a one-TOOL flow calling `read_file`, with `retries:` when given."""
    retries_line = "" if retries is None else f"    retries: {retries}\n"
    return load_flow(
        "id: t\nname: t\nnodes:\n  call:\n    kind: tool\n    tool_id: read_file\n"
        f"{retries_line}output: ${{call.output}}\n"
    )


def _raising(error: BaseException, calls: list[str]):
    """Return a `read_file` tool that records each call and raises `error`."""

    def read_file(path: str = "a.txt") -> str:
        """Read one file.

        Args:
            path: The file to read.
        """
        calls.append(path)
        raise error

    return read_file


def _call_then_answer(answer: str = "the file is missing") -> list[AIMessage]:
    return [
        AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"path": "a.txt"}, "id": "c0", "type": "tool_call"},
        ]),
        AIMessage(content=answer),
    ]


def _tool_messages(messages: list) -> list[ToolMessage]:
    return [message for message in messages if isinstance(message, ToolMessage)]


def test_an_agent_model_reads_a_reported_error_and_the_loop_continues(monkeypatch):
    model = _ScriptedModel(_call_then_answer())
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)
    calls: list[str] = []
    events: list = []

    result = run_flow(
        load_flow(_AGENT_FLOW), {},
        tools={"read_file": _raising(ReportedToolError("no such file"), calls)},
        on_event=events.append,
    )

    assert result.status == "succeeded", result.failure
    assert result.output == "the file is missing"
    assert calls == ["a.txt"]
    [message] = _tool_messages(model.calls[1])
    assert message.status == "error"
    assert message.content == "no such file"
    assert message.tool_call_id == "c0"
    [finished] = [event for event in events if isinstance(event, ToolCallFinished)]
    assert finished.error == "ReportedToolError"
    assert finished.result is None


def test_an_empty_reported_error_reads_as_its_type_name(monkeypatch):
    class NotFound(ReportedToolError):
        pass

    model = _ScriptedModel(_call_then_answer())
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)

    result = run_flow(
        load_flow(_AGENT_FLOW), {}, tools={"read_file": _raising(NotFound(), [])}
    )

    assert result.status == "succeeded", result.failure
    [message] = _tool_messages(model.calls[1])
    assert (message.status, message.content) == ("error", "NotFound")


def test_an_ordinary_tool_error_still_fails_the_agent_node(monkeypatch):
    model = _ScriptedModel(_call_then_answer())
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)

    result = run_flow(
        load_flow(_AGENT_FLOW), {}, tools={"read_file": _raising(RuntimeError("disk on fire"), [])}
    )

    assert result.status == "failed"
    assert result.failure.message == "disk on fire"
    assert result.failure.cause_type == "RuntimeError"
    assert len(model.calls) == 1


@pytest.mark.parametrize("session", [True, False])
def test_a_pending_stop_wins_over_a_reported_error(monkeypatch, session):
    stop = Event()
    model = _ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "lookup", "args": {"value": 1}, "id": "c0", "type": "tool_call"},
        ]),
        AIMessage(content="missing"),
    ])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)

    def lookup(value: int) -> str:
        """Look one thing up.

        Args:
            value: What to look up.
        """
        stop.set()
        raise ReportedToolError("no such file")

    memory = {"session": SessionMemory()} if session else None
    flow = _SESSION_FLOW if session else _AGENT_FLOW.replace("read_file", "lookup")
    question = {"question": "q"} if session else {}

    with pytest.raises(_HostStop) as caught:
        _run_flow_controlled(
            load_flow(flow), question, tools={"lookup": lookup}, memory=memory,
            stop_event=stop,
        )

    assert "no such file" in _contexts(caught.value)
    assert len(model.calls) == 1
    if session:
        view = _view(memory["session"])
        assert [effect.status for effect in view.effects] == ["completed", "interrupted"]
        assert view.effects[1].failure.code == "api.tool.interrupted"


def test_a_session_records_the_reported_call_completed_with_its_message(monkeypatch):
    model = _ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "lookup", "args": {"value": 1}, "id": "c0", "type": "tool_call"},
        ]),
        AIMessage(content="missing"),
        AIMessage(content="still missing"),
    ])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _: model)

    def lookup(value: int) -> str:
        """Look one thing up.

        Args:
            value: What to look up.
        """
        raise ReportedToolError(f"no record {value}")

    memory = SessionMemory()
    result = run_flow(
        load_flow(_SESSION_FLOW), {"question": "q"}, tools={"lookup": lookup},
        memory={"session": memory},
    )

    assert result.status == "succeeded", result.failure
    assert result.output == "missing"
    view = _view(memory)
    assert [(effect.kind, effect.status) for effect in view.effects] == [
        ("model", "completed"), ("tool", "completed"), ("model", "completed"),
    ]
    assert view.effects[1].failure is None
    stored = _tool_messages([decode_message(item) for item in view.items])
    assert [(message.status, message.content, message.tool_call_id) for message in stored] == [
        ("error", "no record 1", "c0"),
    ]

    # A later turn on the same session replays the reported error to the model.
    second = run_flow(
        load_flow(_SESSION_FLOW), {"question": "again"}, tools={"lookup": lookup},
        memory={"session": memory},
    )
    assert second.status == "succeeded", second.failure
    assert second.output == "still missing"
    replayed = _tool_messages(model.calls[2])
    assert [(message.status, message.content, message.tool_call_id) for message in replayed] == [
        ("error", "no record 1", "c0"),
    ]


def test_a_tool_node_fails_on_a_reported_error():
    calls: list[str] = []

    result = run_flow(
        _tool_flow(), {}, tools={"read_file": _raising(ReportedToolError("no such file"), calls)}
    )

    assert result.status == "failed"
    assert result.failure.message == "no such file"
    assert result.failure.cause_type == "ReportedToolError"
    assert len(calls) == 1


def test_tool_retries_retry_a_reported_error(set_defaults):
    set_defaults(TOOL_RETRY_DELAY_SECONDS=0)
    calls: list[str] = []

    result = run_flow(
        _tool_flow(1), {}, tools={"read_file": _raising(ReportedToolError("no such file"), calls)}
    )

    assert result.status == "failed"
    assert len(calls) == 2


def test_tool_retries_stop_on_a_reported_error_that_is_also_permanent(set_defaults):
    set_defaults(TOOL_RETRY_DELAY_SECONDS=0)

    class Gone(ReportedToolError, PermanentToolError):
        pass

    calls: list[str] = []

    result = run_flow(
        _tool_flow(2), {}, tools={"read_file": _raising(Gone("gone"), calls)}
    )

    assert result.status == "failed"
    assert result.failure.message == "gone"
    assert result.failure.cause_type == "Gone"
    assert len(calls) == 1

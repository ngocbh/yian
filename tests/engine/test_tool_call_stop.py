"""Each tool call sees the stop event of the run that made it."""

import threading

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

import yian.llm_clients as llm_clients
from yian.compose import load_flow
from yian.run import _run_flow_controlled, run_flow
from yian.tools import ToolSet
from yian.tools.context import _current_stop

_TOOL_FLOW = """
id: stop_tool
name: stop_tool
nodes:
  probe:
    kind: tool
    tool_id: probe
output: ${probe.output}
"""

_AGENT_FLOW = """
id: stop_agent
name: stop_agent
llm_config: {provider: openai, model: test-model}
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [probe]
output: ${worker.output}
"""


def _recorder(seen: list):
    """Return a tool that records the stop event it sees during its call."""

    def probe() -> str:
        """Record the current stop event."""
        seen.append(_current_stop())
        return "ok"

    return probe


def test_for_run_binds_the_stop_event_seen_during_a_call():
    ev = threading.Event()
    seen: list = []
    bound = ToolSet({"probe": _recorder(seen)}).for_run("r", stop_event=ev)

    assert bound.run_id == "r"
    assert bound.stop_event is ev
    assert _current_stop() is None
    bound.invoke("probe", {})
    assert seen == [ev]
    assert _current_stop() is None


def test_a_langchain_tool_sees_the_stop_event():
    ev = threading.Event()
    seen: list = []
    tool = StructuredTool.from_function(_recorder(seen), name="probe")
    ToolSet({"probe": tool}).for_run("r", stop_event=ev).invoke("probe", {})

    assert seen == [ev]
    assert _current_stop() is None


def test_an_unbound_set_and_for_run_without_stop_event_give_none():
    seen: list = []
    tools = ToolSet({"probe": _recorder(seen)})
    tools.invoke("probe", {})
    tools.for_run("r").invoke("probe", {})

    assert tools.stop_event is None
    assert seen == [None, None]


def test_for_run_sets_exactly_the_stop_event_passed():
    seen: list = []
    bound = ToolSet({"probe": _recorder(seen)}).for_run("r", stop_event=threading.Event())
    rebound = bound.for_run("r2")

    assert rebound.stop_event is None
    rebound.invoke("probe", {})
    assert seen == [None]


def test_the_stop_event_is_cleared_after_a_tool_raises():
    ev = threading.Event()
    seen: list = []

    def boom() -> str:
        """Fail."""
        seen.append(_current_stop())
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        ToolSet({"boom": boom}).for_run("r", stop_event=ev).invoke("boom", {})
    assert seen == [ev]
    assert _current_stop() is None


@pytest.mark.parametrize("num_workers", [0, 8])
def test_a_tool_node_sees_the_engine_stop_event(num_workers: int):
    seen: list = []
    result = run_flow(load_flow(_TOOL_FLOW), {}, tools={"probe": _recorder(seen)}, num_workers=num_workers)

    assert result.status == "succeeded", result.failure
    [event] = seen
    assert isinstance(event, threading.Event)


@pytest.mark.parametrize("num_workers", [0, 8])
def test_a_tool_node_sees_its_run_stop_event(num_workers: int):
    ev = threading.Event()
    seen: list = []
    result = _run_flow_controlled(
        load_flow(_TOOL_FLOW), {}, tools={"probe": _recorder(seen)},
        num_workers=num_workers, stop_event=ev,
    )

    assert result.status == "succeeded", result.failure
    assert seen == [ev]


def test_concurrent_runs_see_their_own_stop_events():
    barrier = threading.Barrier(2, timeout=10)
    events = {"a": threading.Event(), "b": threading.Event()}
    seen: dict[str, object] = {}
    results: dict[str, object] = {}

    def make_probe(name: str):
        def probe() -> str:
            """Record the current stop event once both runs are inside a call."""
            barrier.wait()
            seen[name] = _current_stop()
            barrier.wait()
            return "ok"

        return probe

    def drive(name: str) -> None:
        results[name] = _run_flow_controlled(
            load_flow(_TOOL_FLOW), {}, tools={"probe": make_probe(name)},
            stop_event=events[name],
        )

    threads = [threading.Thread(target=drive, args=(name,)) for name in events]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(result.status == "succeeded" for result in results.values()), results
    assert seen["a"] is events["a"]
    assert seen["b"] is events["b"]


class _ScriptedModel:
    """Return scripted replies."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.replies.pop(0)


def test_an_agent_tool_call_sees_its_run_stop_event(monkeypatch):
    ev = threading.Event()
    seen: list = []
    model = _ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "probe", "args": {}, "id": "c0", "type": "tool_call"}]),
        AIMessage(content="done"),
    ])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)

    result = _run_flow_controlled(
        load_flow(_AGENT_FLOW), {}, tools={"probe": _recorder(seen)}, stop_event=ev,
    )

    assert result.status == "succeeded", result.failure
    assert seen == [ev]

"""Stopping a run, or Ctrl-C in the host, ends its waiting MCP calls and leaves nothing running."""

import os
import signal
import threading
import time

import pytest

pytest.importorskip("mcp")

from langchain_core.messages import AIMessage

import yian.llm_clients as llm_clients
from yian.compose import load_flow
from yian.nodes.base import _HostStop
from yian.run import _run_flow_controlled, resume_flow, run_flow
from yian.tools.mcp import mcp_server

from tests.engine.mcp_helpers import child_alive, mcp_threads, stdio_params

# Bounds that keep a hang from stalling the suite; a clean stop takes well under a second.
_JOIN_SECONDS = 15
_STOP_WITHIN = 3

_SLOW_FLOW = """
id: mcp_stop
name: mcp_stop
input:
  seconds: !float
nodes:
  call:
    kind: tool
    tool_id: slow
    input: {seconds: "${input.seconds}"}
output: ${call.output}
"""

_AGENT_FLOW = """
id: mcp_stop_agent
name: mcp_stop_agent
llm_config: {provider: openai, model: test-model}
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [fake]
output: ${worker.output}
"""


class _ScriptedModel:
    """Return scripted replies and bind any tools."""

    def __init__(self, replies):
        self.replies = list(replies)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.replies.pop(0)


def _wait_for_call(server, timeout=30):
    """Wait until `server` has a tool call in flight; return whether one appeared in time."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server._calls:
            return True
        time.sleep(0.02)
    return False


def _run_in_thread(target):
    """Run `target` in a thread; return the thread and a dict that gets its result or error."""
    outcome = {}

    def run():
        try:
            outcome["result"] = target()
        except BaseException as exc:
            outcome["error"] = exc
        outcome["ended"] = time.monotonic()

    thread = threading.Thread(target=run)
    thread.start()
    return thread, outcome


def _stop_after_call(server, stop, thread):
    """Set `stop` once `server`'s call has waited 0.5 s; return when it was set."""
    assert _wait_for_call(server), "the MCP call never started"
    time.sleep(0.5)
    stop.set()
    stopped_at = time.monotonic()
    thread.join(_JOIN_SECONDS)
    assert not thread.is_alive(), "the run did not end after its stop"
    return stopped_at


def _pids(marker):
    """Return the pids the fake server wrote to `marker`, one per start."""
    return [int(line) for line in marker.read_text().split()]


def _assert_closed(server, marker=None):
    """Assert the server is closed, no MCP thread runs, and its child (if any) is gone."""
    assert not server.is_open
    assert mcp_threads() == []
    if marker is not None:
        for pid in _pids(marker):
            assert not child_alive(pid)


@pytest.mark.parametrize("num_workers", [0, 8])
def test_a_stop_ends_a_run_owned_server_call_and_the_log_resumes(tmp_path, num_workers):
    marker = tmp_path / "pid"
    log = tmp_path / "run.jsonl"
    flow = load_flow(_SLOW_FLOW)
    server = mcp_server(**stdio_params("--marker", str(marker)))
    stop = threading.Event()

    thread, outcome = _run_in_thread(lambda: _run_flow_controlled(
        flow, {"seconds": 30}, log=log, num_workers=num_workers, tools={"fake": server}, stop_event=stop,
    ))
    stopped_at = _stop_after_call(server, stop, thread)

    # A host stop is control flow, not a run result: the run raises it and its
    # started node stays unfinished in the log.
    assert isinstance(outcome.get("error"), _HostStop), outcome
    assert outcome["ended"] - stopped_at < _STOP_WITHIN
    _assert_closed(server, marker)

    def slow(seconds: float) -> str:
        """Answer at once."""
        return "done"

    resumed = resume_flow(flow, log=log, tools={"fake": [slow]})
    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == "done"


@pytest.mark.parametrize("num_workers", [0, 8])
def test_ctrl_c_in_the_host_ends_a_run_owned_server_call(tmp_path, num_workers):
    marker = tmp_path / "pid"
    server = mcp_server(**stdio_params("--marker", str(marker)))
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    sent = []

    def interrupt():
        if _wait_for_call(server):
            time.sleep(0.5)
            sent.append(time.monotonic())
            os.kill(os.getpid(), signal.SIGINT)

    interrupter = threading.Thread(target=interrupt, name="test-sigint")
    interrupter.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            run_flow(load_flow(_SLOW_FLOW), {"seconds": 30}, num_workers=num_workers, tools={"fake": server})
        ended = time.monotonic()
    finally:
        interrupter.join(_JOIN_SECONDS)
        signal.signal(signal.SIGINT, previous)

    assert not interrupter.is_alive()
    [sent_at] = sent
    assert ended - sent_at < _STOP_WITHIN
    _assert_closed(server, marker)


@pytest.mark.parametrize("num_workers", [0, 8])
def test_a_stop_cancels_only_its_own_run_on_a_shared_host_server(num_workers):
    flow = load_flow(_SLOW_FLOW)
    stop_a, stop_b = threading.Event(), threading.Event()
    with mcp_server(**stdio_params()) as server:
        thread_a, outcome_a = _run_in_thread(lambda: _run_flow_controlled(
            flow, {"seconds": 30}, num_workers=num_workers, tools={"fake": server}, stop_event=stop_a,
        ))
        assert _wait_for_call(server), "run A's call never started"
        thread_b, outcome_b = _run_in_thread(lambda: _run_flow_controlled(
            flow, {"seconds": 1.5}, num_workers=num_workers, tools={"fake": server}, stop_event=stop_b,
        ))
        deadline = time.monotonic() + 30
        while len(server._calls) < 2 and time.monotonic() < deadline and thread_b.is_alive():
            time.sleep(0.02)
        stopped_at = _stop_after_call(server, stop_a, thread_a)
        thread_b.join(_JOIN_SECONDS)

        assert not thread_b.is_alive()
        assert isinstance(outcome_a.get("error"), _HostStop), outcome_a
        assert outcome_a["ended"] - stopped_at < _STOP_WITHIN
        assert "error" not in outcome_b, outcome_b
        assert outcome_b["result"].status == "succeeded", outcome_b["result"].failure
        assert outcome_b["result"].output == "done"
        assert not stop_b.is_set()
        assert server.is_open
        third = run_flow(flow, {"seconds": 0}, num_workers=num_workers, tools={"fake": server})
        assert third.status == "succeeded", third.failure
    _assert_closed(server)


@pytest.mark.parametrize("num_workers", [0, 8])
def test_a_stop_ends_an_agent_tool_call(monkeypatch, tmp_path, num_workers):
    marker = tmp_path / "pid"
    calls = [{"name": "slow", "args": {"seconds": 30}, "id": "c0", "type": "tool_call"}]
    model = _ScriptedModel([AIMessage(content="", tool_calls=calls), AIMessage(content="done")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    server = mcp_server(**stdio_params("--marker", str(marker)))
    stop = threading.Event()

    thread, outcome = _run_in_thread(lambda: _run_flow_controlled(
        load_flow(_AGENT_FLOW), {}, num_workers=num_workers, tools={"fake": server}, stop_event=stop,
    ))
    stopped_at = _stop_after_call(server, stop, thread)

    assert isinstance(outcome.get("error"), _HostStop), outcome
    assert outcome["ended"] - stopped_at < _STOP_WITHIN
    _assert_closed(server, marker)

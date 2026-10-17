"""`resume_flow` opens the MCP servers passed to it unopened only for an unfinished run."""

import threading

import pytest

pytest.importorskip("mcp")

from yian.compose import load_flow
from yian.events import RunStarted
from yian.memory.contracts import MemorySetupError
from yian.nodes.base import _HostStop
from yian.run import _resume_flow_controlled, resume_flow, run_flow
from yian.tools.context import ToolContext
from yian.tools.mcp import McpConnectError, mcp_server

from tests.engine._memory_fakes import _Memory
from tests.engine.mcp_helpers import child_alive, mcp_threads, stdio_params

_WHOAMI_FLOW = """
id: mcp_resume
name: mcp_resume
nodes:
  call:
    kind: tool
    tool_id: whoami
output: ${call.output}
"""


class _SimulatedCrash(BaseException):
    """Process-loss marker that bypasses node-failure conversion."""


def _crashed_run(log) -> ToolContext:
    """Crash a run of `_WHOAMI_FLOW` inside its TOOL node and return the call's context."""
    seen: list[ToolContext] = []

    def whoami(ctx: ToolContext) -> str:
        """Crash the process on the call."""
        seen.append(ctx)
        raise _SimulatedCrash()

    with pytest.raises(_SimulatedCrash):
        run_flow(load_flow(_WHOAMI_FLOW), {}, log=log, tools={"whoami": whoami}, run_id="run-5c2e")
    [ctx] = seen
    return ctx


def _assert_closed(server, marker=None):
    """Assert the server is closed, no MCP thread runs, and its child (if any) is gone."""
    assert not server.is_open
    assert mcp_threads() == []
    if marker is not None:
        for pid in marker.read_text().split():
            assert not child_alive(int(pid))


def test_a_resumed_tool_node_calls_an_unopened_server_which_the_resume_closes(tmp_path):
    log = tmp_path / "run.jsonl"
    marker = tmp_path / "pid"
    first = _crashed_run(log)
    server = mcp_server(**stdio_params("--marker", str(marker)))

    resumed = resume_flow(load_flow(_WHOAMI_FLOW), log=log, tools={"fake": server})

    assert resumed.status == "succeeded", resumed.failure
    assert first.call_id == "run-5c2e/call"
    assert resumed.output == f"run-5c2e/call {first.attempt + 1}"
    _assert_closed(server, marker)


def test_resuming_a_finished_run_starts_no_server(tmp_path):
    log = tmp_path / "run.jsonl"
    marker = tmp_path / "pid"
    flow = load_flow(_WHOAMI_FLOW)

    def whoami() -> str:
        """Answer without a server."""
        return "local"

    finished = run_flow(flow, {}, log=log, tools={"whoami": whoami})
    assert finished.status == "succeeded", finished.failure
    server = mcp_server(**stdio_params("--marker", str(marker)))

    resumed = resume_flow(flow, log=log, tools={"fake": server})

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == "local"
    assert not marker.exists()
    _assert_closed(server)


def test_a_bad_entry_fails_before_the_log_is_touched(tmp_path):
    log = tmp_path / "run.jsonl"
    marker = tmp_path / "pid"
    flow = load_flow(_WHOAMI_FLOW)
    _crashed_run(log)
    before = (log.read_bytes(), log.stat().st_mtime_ns)
    server = mcp_server(**stdio_params("--marker", str(marker)))

    with pytest.raises(TypeError, match=r"tools\['g'\]\[0\] is a lambda"):
        resume_flow(flow, log=log, tools={"fake": server, "g": [lambda text: text]})

    assert (log.read_bytes(), log.stat().st_mtime_ns) == before
    assert not marker.exists()
    _assert_closed(server)
    resumed = resume_flow(flow, log=log, tools={"fake": server})
    assert resumed.status == "succeeded", resumed.failure


def test_a_server_that_fails_to_open_on_resume_closes_the_log(tmp_path):
    log = tmp_path / "run.jsonl"
    flow = load_flow(_WHOAMI_FLOW)
    _crashed_run(log)
    bad = mcp_server(command="/nonexistent/x")

    with pytest.raises(McpConnectError):
        resume_flow(flow, log=log, tools={"fake": bad})

    assert not bad.is_open
    server = mcp_server(**stdio_params())
    resumed = resume_flow(flow, log=log, tools={"fake": server})
    assert resumed.status == "succeeded", resumed.failure
    _assert_closed(server)


def test_a_check_that_needs_no_tools_fails_before_the_server_starts(tmp_path):
    log = tmp_path / "run.jsonl"
    marker = tmp_path / "pid"
    flow = load_flow(_WHOAMI_FLOW)
    _crashed_run(log)
    server = mcp_server(**stdio_params("--marker", str(marker)))

    with pytest.raises(TypeError, match="skills"):
        resume_flow(flow, log=log, tools={"fake": server}, skills={"x": object()})

    assert not marker.exists()
    _assert_closed(server)
    resumed = resume_flow(flow, log=log, tools={"fake": server})
    assert resumed.status == "succeeded", resumed.failure


_MEMORY_FLOW = """
id: mcp_memory
name: mcp_memory
long_term_memory: test
nodes:
  work:
    kind: code
    code: return 7
    output: int
output: ${work.output}
"""


def test_resuming_without_the_runs_memory_fails_before_any_server_starts(tmp_path):
    log = tmp_path / "run.jsonl"
    marker = tmp_path / "pid"
    flow = load_flow(_MEMORY_FLOW)

    def stop_after_start(event: object) -> None:
        if isinstance(event, RunStarted):
            raise RuntimeError("host stop")

    with pytest.raises(RuntimeError, match="host stop"):
        run_flow(flow, {}, log=log, memory={"test": _Memory()}, on_event=stop_after_start)
    server = mcp_server(**stdio_params("--marker", str(marker)))

    with pytest.raises(MemorySetupError, match="requires its enabled memory mapping"):
        resume_flow(flow, log=log, tools={"fake": server})

    assert not marker.exists()
    _assert_closed(server)


def test_a_stop_before_the_servers_open_on_resume_starts_none(tmp_path):
    log = tmp_path / "run.jsonl"
    flow = load_flow(_WHOAMI_FLOW)
    _crashed_run(log)
    markers = [tmp_path / "a.pid", tmp_path / "b.pid"]
    servers = [mcp_server(**stdio_params("--marker", str(marker))) for marker in markers]
    stop = threading.Event()
    stop.set()

    with pytest.raises(_HostStop):
        _resume_flow_controlled(flow, log=log, tools={"a": servers[0], "b": servers[1]}, stop_event=stop)

    assert not any(marker.exists() for marker in markers)
    for server in servers:
        _assert_closed(server)
    resumed = resume_flow(flow, log=log, tools={"fake": mcp_server(**stdio_params())})
    assert resumed.status == "succeeded", resumed.failure

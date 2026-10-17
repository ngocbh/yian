"""`run_flow` opens the MCP servers passed to it unopened and closes them when the run ends."""

import threading
import time

import pytest

pytest.importorskip("mcp")

from langchain_core.messages import AIMessage, ToolMessage

import yian.llm_clients as llm_clients
from yian.compose import load_flow
from yian.nodes.base import _HostStop
from yian.run import _run_flow_controlled, run_flow
from yian.tools.mcp import McpConnectError, mcp_server

from tests.engine.mcp_helpers import child_alive, mcp_threads, run_http_server, stdio_params


class _ScriptedModel:
    """Return scripted replies, capture each request and the tools bound."""

    def __init__(self, replies: list) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []
        self.bound: list = []

    def bind_tools(self, tools):
        self.bound = list(tools)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def _tool_flow(tool_id: str, inputs: str = "") -> str:
    """Return a flow whose one TOOL node calls `tool_id` with the YAML mapping `inputs`."""
    line = f"    input: {inputs}\n" if inputs else ""
    return (
        f"id: mcp_tool\nname: mcp_tool\nnodes:\n  call:\n    kind: tool\n    tool_id: {tool_id}\n"
        f"{line}output: ${{call.output}}\n"
    )


_AGENT_FLOW = """
id: mcp_agent
name: mcp_agent
llm_config: {provider: openai, model: test-model}
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [GRANT]
output: ${worker.output}
"""


def _call_then_answer(name: str, args: dict) -> list[AIMessage]:
    """One tool call, then a final answer."""
    return [
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "c0", "type": "tool_call"}]),
        AIMessage(content="done"),
    ]


def _pids(marker) -> list[int]:
    """Return the pids the fake server wrote to `marker`, one per start."""
    return [int(line) for line in marker.read_text().split()]


def _assert_closed(server, marker=None):
    """Assert the server is closed, no MCP thread runs, and its child (if any) is gone."""
    assert not server.is_open
    assert mcp_threads() == []
    if marker is not None:
        for pid in _pids(marker):
            assert not child_alive(pid)


def test_a_tool_node_calls_an_unopened_server_which_the_run_closes(tmp_path):
    marker = tmp_path / "pid"
    server = mcp_server(**stdio_params("--marker", str(marker)))

    result = run_flow(load_flow(_tool_flow("echo", '{text: "hi"}')), {}, tools={"fake": server})

    assert result.status == "succeeded", result.failure
    assert result.output == "hi"
    _assert_closed(server, marker)


def test_an_agent_gets_the_server_tools_and_calls_one(monkeypatch):
    model = _ScriptedModel(_call_then_answer("add", {"a": 2, "b": 3}))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    server = mcp_server(**stdio_params())

    result = run_flow(load_flow(_AGENT_FLOW.replace("GRANT", "fake")), {}, tools={"fake": server})

    assert result.status == "succeeded", result.failure
    by_name = {tool.name: tool for tool in model.bound}
    assert {"echo", "add", "fail"} <= set(by_name)
    assert set(by_name["add"].args) == {"a", "b"}
    [reply] = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    # `add` returns structured content, which the call's result is.
    assert reply.content == "{'result': 5}"
    _assert_closed(server)


def test_an_agent_calls_an_unopened_http_server(monkeypatch):
    model = _ScriptedModel(_call_then_answer("echo", {"text": "over http"}))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)

    with run_http_server() as (url, _headers):
        server = mcp_server(url=url)
        result = run_flow(load_flow(_AGENT_FLOW.replace("GRANT", "web")), {}, tools={"web": server})

    assert result.status == "succeeded", result.failure
    [reply] = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert reply.content == "over http"
    _assert_closed(server)


def test_an_agent_reads_a_tool_error_and_carries_on(monkeypatch):
    model = _ScriptedModel(_call_then_answer("fail", {"message": "boom"}))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    server = mcp_server(**stdio_params())

    result = run_flow(load_flow(_AGENT_FLOW.replace("GRANT", "fake")), {}, tools={"fake": server})

    assert result.status == "succeeded", result.failure
    assert result.output == "done"
    [reply] = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert "boom" in reply.content
    _assert_closed(server)


def test_a_failed_run_start_check_closes_the_server(tmp_path):
    marker = tmp_path / "pid"
    log = tmp_path / "run.jsonl"
    server = mcp_server(**stdio_params("--marker", str(marker)))

    with pytest.raises(ValueError, match=r"TOOL node 'call' input 'a': tool 'add' takes int"):
        run_flow(load_flow(_tool_flow("add", '{a: "x", b: 1}')), {}, log=log, tools={"fake": server})

    assert not log.exists()
    _assert_closed(server, marker)


def test_a_host_opened_server_stays_open_across_runs():
    flow = load_flow(_tool_flow("echo", '{text: "hi"}'))
    with mcp_server(**stdio_params()) as server:
        for _ in range(2):
            result = run_flow(flow, {}, tools={"fake": server})
            assert result.status == "succeeded", result.failure
            assert server.is_open
    _assert_closed(server)


def test_one_server_under_two_keys_starts_once(tmp_path):
    marker = tmp_path / "pid"
    server = mcp_server(**stdio_params("--marker", str(marker)))

    result = run_flow(load_flow(_tool_flow("echo", '{text: "hi"}')), {}, tools={"a": server, "b": server})

    assert result.status == "succeeded", result.failure
    assert len(_pids(marker)) == 1
    _assert_closed(server, marker)


def test_a_server_another_run_opened_is_rejected():
    server = mcp_server(**stdio_params())
    flow = load_flow(_tool_flow("slow", "{seconds: 2}"))
    results = []

    def run_a():
        try:
            results.append(run_flow(flow, {}, tools={"fake": server}))
        except BaseException as exc:
            results.append(exc)

    first = threading.Thread(target=run_a)
    first.start()
    try:
        deadline = time.monotonic() + 30
        while not server.is_open:
            assert first.is_alive() and time.monotonic() < deadline
            time.sleep(0.02)
        with pytest.raises(ValueError) as caught:
            run_flow(flow, {}, tools={"shared": server})
    finally:
        first.join(30)

    assert str(caught.value) == (
        "tools['shared']: this MCP server was opened by another run; "
        'open it with "with server:" to share it between runs'
    )
    [result] = results
    assert not isinstance(result, BaseException), repr(result)
    assert result.status == "succeeded", result.failure
    _assert_closed(server)


def test_two_runs_racing_for_one_unopened_server_start_it_once(tmp_path):
    marker = tmp_path / "pid"
    server = mcp_server(**stdio_params("--marker", str(marker)))
    # Long enough that the losing run finds the server still held by the winner.
    flow = load_flow(_tool_flow("slow", "{seconds: 5}"))
    barrier = threading.Barrier(2)
    outcomes = {}

    def run(key):
        barrier.wait()
        try:
            outcomes[key] = run_flow(flow, {}, tools={key: server})
        except BaseException as exc:
            outcomes[key] = exc

    threads = [threading.Thread(target=run, args=(key,)) for key in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    errors = {key: out for key, out in outcomes.items() if isinstance(out, BaseException)}
    results = {key: out for key, out in outcomes.items() if not isinstance(out, BaseException)}
    assert len(results) == 1 and len(errors) == 1, outcomes
    [(winner, result)] = results.items()
    [(loser, error)] = errors.items()
    assert result.status == "succeeded", result.failure
    assert type(error) is ValueError
    assert str(error) == (
        f"tools[{loser!r}]: this MCP server was opened by another run; "
        'open it with "with server:" to share it between runs'
    )
    assert len(_pids(marker)) == 1
    _assert_closed(server, marker)


@pytest.mark.parametrize(
    ("key", "match"),
    [
        ("ask_user", r"tools\['ask_user'\] must be a plain callable handler"),
        (1, "tool ids in tools must be non-empty strings"),
    ],
)
def test_a_server_under_a_bad_key_fails_before_it_starts(tmp_path, key, match):
    marker = tmp_path / "pid"
    server = mcp_server(**stdio_params("--marker", str(marker)))

    with pytest.raises(TypeError, match=match):
        run_flow(load_flow(_tool_flow("echo", '{text: "hi"}')), {}, tools={key: server})

    assert not marker.exists()
    _assert_closed(server)


def test_a_check_that_needs_no_tools_fails_before_any_server_starts(tmp_path):
    marker = tmp_path / "pid"
    server = mcp_server(**stdio_params("--marker", str(marker)))

    with pytest.raises(TypeError, match="skills"):
        run_flow(
            load_flow(_tool_flow("echo", '{text: "hi"}')), {},
            tools={"fake": server}, skills={"x": object()},
        )

    assert not marker.exists()
    _assert_closed(server)


def test_a_server_that_fails_to_open_closes_the_ones_already_open(tmp_path):
    marker = tmp_path / "pid"
    log = tmp_path / "run.jsonl"
    ok = mcp_server(**stdio_params("--marker", str(marker)))
    bad = mcp_server(command="/nonexistent/x")

    with pytest.raises(McpConnectError):
        run_flow(load_flow(_tool_flow("echo", '{text: "hi"}')), {}, log=log, tools={"ok": ok, "bad": bad})

    assert not log.exists()
    assert len(_pids(marker)) == 1
    _assert_closed(ok, marker)
    assert not bad.is_open


def test_a_bad_entry_fails_before_any_server_starts(tmp_path):
    marker = tmp_path / "pid"
    server = mcp_server(**stdio_params("--marker", str(marker)))

    with pytest.raises(TypeError, match=r"tools\['g'\]\[0\]"):
        run_flow(
            load_flow(_tool_flow("echo", '{text: "hi"}')), {},
            tools={"fake": server, "g": [lambda text: text]},
        )

    assert not marker.exists()
    _assert_closed(server)


def test_a_failing_node_still_closes_the_server(tmp_path):
    marker = tmp_path / "pid"
    server = mcp_server(**stdio_params("--marker", str(marker)))

    result = run_flow(load_flow(_tool_flow("bad_request")), {}, tools={"fake": server})

    assert result.status == "failed"
    _assert_closed(server, marker)


def test_a_stop_before_the_servers_open_starts_none(tmp_path):
    markers = [tmp_path / "a.pid", tmp_path / "b.pid"]
    servers = [mcp_server(**stdio_params("--marker", str(marker))) for marker in markers]
    stop = threading.Event()
    stop.set()

    with pytest.raises(_HostStop):
        _run_flow_controlled(
            load_flow(_tool_flow("echo", '{text: "hi"}')), {},
            tools={"a": servers[0], "b": servers[1]}, stop_event=stop,
        )

    assert not any(marker.exists() for marker in markers)
    for server in servers:
        _assert_closed(server)


def test_a_stop_while_the_servers_open_closes_the_opened_one_and_opens_no_more(monkeypatch, tmp_path):
    markers = [tmp_path / "a.pid", tmp_path / "b.pid"]
    first, second = (mcp_server(**stdio_params("--marker", str(marker))) for marker in markers)
    stop = threading.Event()
    open_for_run = type(first)._open_for_run

    def open_then_stop(self, token):
        """Open the server, then set the stop as if Ctrl-C came during the open."""
        opened = open_for_run(self, token)
        stop.set()
        return opened

    monkeypatch.setattr(type(first), "_open_for_run", open_then_stop)

    with pytest.raises(_HostStop):
        _run_flow_controlled(
            load_flow(_tool_flow("echo", '{text: "hi"}')), {},
            tools={"a": first, "b": second}, stop_event=stop,
        )

    assert len(_pids(markers[0])) == 1
    assert not markers[1].exists()
    _assert_closed(first, markers[0])
    _assert_closed(second)

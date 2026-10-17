"""MCP call failures map to yian's error and retry classes; stops and `close()` end waiting calls."""

import asyncio
import concurrent.futures
import pickle
import threading
import time

import pytest

pytest.importorskip("mcp")

from langchain_core.messages import AIMessage, ToolMessage

import yian.llm_clients as llm_clients
import yian.tools.mcp as mcp_module
from yian.compose import load_flow
from yian.run import run_flow
from yian.tools import ToolSet
from yian.tools.errors import PermanentToolError, ReportedToolError, TransientToolError
from yian.tools.mcp import (
    McpCallCancelled,
    McpInvalidRequestError,
    McpRequestError,
    McpServerClosed,
    McpTimeoutError,
    McpToolError,
    McpTransportError,
    _proxy_for,
    mcp_server,
)

from tests.engine.mcp_helpers import BAD_OUTPUT, HttpServer, stdio_params


def _stdio(mode="auto", script=None, **options):
    params = stdio_params() if script is None else stdio_params(script=script)
    server = mcp_server(**params, **options)
    server._mode = mode
    return server


@pytest.fixture(scope="module", params=["auto", "legacy"])
def server(request):
    with _stdio(request.param) as opened:
        yield opened


def _tools(server):
    return ToolSet({"fake": list(server.tools)})


def _call_in_thread(target):
    """Run `target` in a thread; return the thread and a dict that gets its result or error."""
    outcome = {}

    def run():
        start = time.monotonic()
        try:
            outcome["result"] = target()
        except BaseException as exc:
            outcome["error"] = exc
        outcome["ended"] = time.monotonic() - start

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, outcome


def test_an_error_result_is_a_reported_but_not_permanent_tool_error(server):
    with pytest.raises(McpToolError, match="no luck") as caught:
        _tools(server).invoke("fail", {"message": "no luck"})
    assert isinstance(caught.value, ReportedToolError)
    assert not isinstance(caught.value, PermanentToolError)


def test_an_invalid_params_answer_is_reported_and_permanent(server):
    with pytest.raises(McpInvalidRequestError) as caught:
        _tools(server).invoke("bad_request", {})
    error = caught.value
    assert error.code == -32602
    assert isinstance(error, McpRequestError)
    assert isinstance(error, ReportedToolError)
    assert isinstance(error, PermanentToolError)
    assert str(error) == f"MCP server '{server.label}' tool 'bad_request': bad params (code -32602)"


def test_a_busy_answer_is_a_request_error_and_the_server_still_works(server):
    with pytest.raises(McpRequestError, match="busy") as caught:
        _tools(server).invoke("busy", {})
    assert caught.value.code == -32000
    assert not isinstance(caught.value, (McpServerClosed, PermanentToolError))
    assert not server._dead
    assert _tools(server).invoke("echo", {"text": "still here"}) == "still here"


def test_an_internal_error_answer_is_a_request_error(server):
    with pytest.raises(McpRequestError, match="internal") as caught:
        _tools(server).invoke("crash", {})
    assert caught.value.code == -32603
    assert not isinstance(caught.value, (McpTransportError, PermanentToolError))


def test_structured_content_failing_the_output_schema_is_a_tool_error():
    with _stdio(script=BAD_OUTPUT, label="bad") as server:
        with pytest.raises(McpToolError) as caught:
            _tools(server).invoke("bad_output", {})
    assert str(caught.value) == f"MCP server 'bad' tool 'bad_output' failed ({type(caught.value.__cause__).__name__})"
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_a_call_past_its_timeout_is_a_transient_timeout_error(mode):
    with _stdio(mode, timeout=0.5, label="slowpoke") as server:
        tools = _tools(server)
        start = time.monotonic()
        with pytest.raises(McpTimeoutError) as caught:
            tools.invoke("slow", {"seconds": 5})
        assert time.monotonic() - start < 3
        assert isinstance(caught.value, ReportedToolError)
        assert isinstance(caught.value, TransientToolError)
        assert str(caught.value) == "MCP server 'slowpoke' tool 'slow' timed out after 0.5 s"
        assert tools.invoke("echo", {"text": "after"}) == "after"


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_a_server_that_exits_is_closed_and_later_calls_fail_at_once(mode):
    with _stdio(mode, label="quitter") as server:
        tools = _tools(server)
        with pytest.raises(McpServerClosed) as caught:
            tools.invoke("exit_now", {})
        assert isinstance(caught.value, PermanentToolError)
        assert not isinstance(caught.value, ReportedToolError)
        assert str(caught.value) == "MCP server 'quitter' is closed"
        assert server._dead
        assert server.is_open
        start = time.monotonic()
        with pytest.raises(McpServerClosed):
            tools.invoke("echo", {"text": "x"})
        assert time.monotonic() - start < 0.5


def test_a_stopped_http_server_is_a_transport_error_and_a_restart_recovers():
    http = HttpServer()
    http.start()
    try:
        server = mcp_server(url=f"{http.url}?token=SECRET", label="remote")
        with server:
            tools = _tools(server)
            assert tools.invoke("echo", {"text": "up"}) == "up"
            port = http.port
            http.stop()
            with pytest.raises(McpTransportError) as caught:
                tools.invoke("echo", {"text": "down"})
            assert isinstance(caught.value, ReportedToolError)
            assert isinstance(caught.value, TransientToolError)
            assert str(caught.value) == "MCP server 'remote' transport error (connection failed)"
            assert "SECRET" not in str(caught.value)
            assert server.is_open
            assert not server._dead
            http.start(port=port)
            assert tools.invoke("echo", {"text": "back"}) == "back"
    finally:
        http.stop()


def test_a_run_stop_cancels_a_waiting_call_and_the_server_still_works(server):
    stop = threading.Event()
    bound = _tools(server).for_run("r", stop_event=stop)
    thread, outcome = _call_in_thread(lambda: bound.invoke("slow", {"seconds": 30}))
    time.sleep(0.3)
    stop.set()
    thread.join(5)
    assert isinstance(outcome.get("error"), McpCallCancelled), outcome
    assert outcome["ended"] < 0.3 + 2
    assert isinstance(outcome["error"], PermanentToolError)
    assert str(outcome["error"]) == "MCP call to 'slow' cancelled by a run stop"
    assert _tools(server).invoke("echo", {"text": "after stop"}) == "after stop"


def test_a_stop_cancels_only_the_call_of_its_own_run(server):
    first_stop, second_stop = threading.Event(), threading.Event()
    first = _tools(server).for_run("r1", stop_event=first_stop)
    second = _tools(server).for_run("r2", stop_event=second_stop)
    first_thread, first_outcome = _call_in_thread(lambda: first.invoke("slow", {"seconds": 30}))
    second_thread, second_outcome = _call_in_thread(lambda: second.invoke("slow", {"seconds": 1.5}))
    time.sleep(0.3)
    first_stop.set()
    first_thread.join(5)
    second_thread.join(20)
    assert isinstance(first_outcome.get("error"), McpCallCancelled), first_outcome
    assert second_outcome == {"result": "done", "ended": second_outcome["ended"]}


def test_close_from_another_thread_ends_a_waiting_call():
    server = _stdio(label="closing").open()
    try:
        tools = _tools(server)
        thread, outcome = _call_in_thread(lambda: tools.invoke("slow", {"seconds": 30}))
        time.sleep(0.3)
        closer = threading.Thread(target=server.close, daemon=True)
        closer.start()
        thread.join(10)
        assert isinstance(outcome.get("error"), McpServerClosed), outcome
        assert outcome["ended"] < 0.3 + 3
        closer.join(15)
        assert not closer.is_alive()
    finally:
        server.close()
    with pytest.raises(McpServerClosed):
        tools.invoke("echo", {"text": "x"})


_RETRY_FLOW = """
id: retry
name: retry
nodes:
  call:
    kind: tool
    tool_id: TOOL
    retries: 2
    input: INPUT
output: ${call.output}
"""


@pytest.mark.parametrize(("tool", "arguments", "attempts"), [("fail", "{message: nope}", 3), ("bad_request", "{}", 1)])
def test_tool_node_retries_follow_the_error_class(set_defaults, tool, arguments, attempts):
    set_defaults(TOOL_RETRY_DELAY_SECONDS=0)
    with _stdio() as server:
        tools = _tools(server)
        before = tools.invoke("calls", {})
        flow = load_flow(_RETRY_FLOW.replace("TOOL", tool).replace("INPUT", arguments))
        result = run_flow(flow, {}, tools={"fake": list(server.tools)})
        assert result.status == "failed"
        assert tools.invoke("calls", {})["result"] - before["result"] == attempts


_AGENT_FLOW = """
id: mcp_agent
name: mcp_agent
llm_config: {provider: openai, model: test-model}
nodes:
  worker:
    kind: agent
    prompt: go
    tools: [fake]
output: ${worker.output}
"""


class _ScriptedModel:
    """Return scripted replies and capture each request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def test_an_agent_model_reads_an_mcp_error_result_and_the_node_succeeds(monkeypatch):
    model = _ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "fail", "args": {"message": "no luck"}, "id": "c0", "type": "tool_call"},
        ]),
        AIMessage(content="gave up"),
    ])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: model)
    with _stdio() as server:
        result = run_flow(load_flow(_AGENT_FLOW), {}, tools={"fake": list(server.tools)})
    assert result.status == "succeeded", result.failure
    assert result.output == "gave up"
    [message] = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
    assert message.status == "error"
    assert "no luck" in message.content


def test_a_late_failure_from_an_earlier_connection_leaves_the_reopened_server_alive(server):
    # A call close() cancelled can resolve after a quick re-open; its connection
    # generation (`_ready`) is then stale.
    stale = threading.Event()
    error = server._call_error("echo", concurrent.futures.CancelledError(), time.monotonic(), 0, stale)
    assert isinstance(error, McpServerClosed)
    assert not server._dead
    assert _tools(server).invoke("echo", {"text": "alive"}) == "alive"


def test_a_call_with_the_stop_already_set_is_cancelled_before_it_is_sent(server):
    tools = _tools(server)
    before = tools.invoke("calls", {})["result"]
    stop = threading.Event()
    stop.set()
    with pytest.raises(McpCallCancelled):
        tools.for_run("r", stop_event=stop).invoke("echo", {"text": "x"})
    assert tools.invoke("calls", {})["result"] == before


@pytest.mark.parametrize("kind", [McpRequestError, McpInvalidRequestError])
def test_a_request_error_survives_a_pickle_round_trip(kind):
    copy = pickle.loads(pickle.dumps(kind("MCP server 's' tool 't': nope (code -32602)", -32602)))
    assert type(copy) is kind
    assert str(copy) == "MCP server 's' tool 't': nope (code -32602)"
    assert copy.code == -32602


def test_a_restarted_session_http_server_is_closed():
    http = HttpServer()
    http.start()
    try:
        server = mcp_server(url=http.url, label="restarted")
        server._mode = "legacy"
        with server:
            tools = _tools(server)
            assert tools.invoke("echo", {"text": "up"}) == "up"
            port = http.port
            http.stop()
            http.start(port=port)
            with pytest.raises(McpServerClosed, match="'restarted' is closed"):
                tools.invoke("echo", {"text": "lost session"})
            assert server._dead
    finally:
        http.stop()


def _error(server, exc):
    return server._call_error("echo", exc, time.monotonic(), 0, server._ready)


def test_a_closed_stream_after_a_transport_failure_is_a_transport_error(server, monkeypatch):
    from mcp import MCPError

    def probe():
        # The soft-fail transport counted a failure while the probe ran, and the
        # probe's own 502 still counts as an answer.
        server._count_failure()
        return True

    monkeypatch.setattr(server, "_probe_alive", probe)
    monkeypatch.setattr(server, "_transport_failures", 0)
    error = _error(server, MCPError(-32000, "SSE stream ended without a response"))
    assert type(error) is McpTransportError
    assert str(error) == f"MCP server '{server.label}' transport error (connection failed)"
    assert not server._dead


def test_an_sdk_redirect_answer_hides_the_location(server):
    from mcp import MCPError

    message = "Redirect to https://elsewhere.example/mcp?token=SECRET not followed; use that URL"
    error = _error(server, MCPError(-32600, message))
    assert isinstance(error, McpInvalidRequestError)
    assert error.code == -32600
    assert str(error) == (
        f"MCP server '{server.label}' redirected the request; yian doesn't follow redirects to another origin"
    )


def test_a_raw_timeout_is_a_transport_error(server):
    error = _error(server, TimeoutError())
    assert type(error) is McpTransportError
    assert str(error) == f"MCP server '{server.label}' transport error (TimeoutError)"


def test_a_call_that_never_resolves_times_out_on_its_own_deadline(server, monkeypatch):
    class Stuck:
        """A client whose call ignores its timeout and never resolves."""

        async def call_tool(self, *args, **kwargs):
            await asyncio.Event().wait()

    monkeypatch.setattr(mcp_module, "_CALL_GRACE_SECONDS", 0.3)
    monkeypatch.setattr(server, "_call_timeout", 0.2)
    monkeypatch.setattr(server, "_client", Stuck())
    start = time.monotonic()
    with pytest.raises(McpTimeoutError, match="timed out after 0.2 s"):
        _tools(server).invoke("echo", {"text": "x"})
    assert time.monotonic() - start < 2
    assert not server._calls


def test_all_proxy_is_the_fallback_proxy(monkeypatch):
    for name in ("http_proxy", "https_proxy", "no_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("ALL_PROXY", "http://proxy.example:3128")
    assert _proxy_for("https://tools.example.com/mcp") == "http://proxy.example:3128"
    monkeypatch.setenv("HTTPS_PROXY", "http://https-proxy.example:3128")
    assert _proxy_for("https://tools.example.com/mcp") == "http://https-proxy.example:3128"

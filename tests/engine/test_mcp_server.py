"""`McpServer` opens a server in a background thread, lists its tools once, and closes."""

import os
import signal
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

import anyio
from mcp import Client, StdioServerParameters

from yian.tools.context import ToolContext
from yian.tools.function_tools import FunctionTool
from yian.tools.mcp import McpConnectError, McpError, McpServer, mcp_server

from tests.engine.mcp_helpers import (
    PAGED,
    child_alive,
    child_env,
    free_port,
    mcp_threads,
    run_http_server,
    stdio_params,
)


def _ctx(name):
    return ToolContext(call_id="call-1", attempt=1, tool_name=name, run_id="", node="", context=None)


def _invoke(server, name, args=None):
    tool = next(tool for tool in server.tools if tool.name == name)
    return tool.on_invoke(_ctx(name), args or {})


def _server(*flags, mode="auto", script=None, **options):
    params = stdio_params(*flags) if script is None else stdio_params(*flags, script=script)
    server = mcp_server(**params, **options)
    server._mode = mode
    return server


def _bare_listing(*flags):
    async def main():
        async with Client(StdioServerParameters(**stdio_params(*flags))) as client:
            return (await client.list_tools()).tools

    return anyio.run(main)


@pytest.mark.parametrize(
    ("options", "error", "field"),
    [
        ({}, ValueError, "command"),
        ({"command": "x", "url": "http://h/mcp"}, ValueError, "url"),
        ({"url": "ftp://x"}, ValueError, "url"),
        ({"command": "x", "args": "x"}, TypeError, "args"),
        ({"command": "x", "args": ["a", 1]}, TypeError, "args"),
        ({"command": "x", "env": {"A": 1}}, TypeError, "env"),
        ({"url": "http://h/mcp", "headers": {"x-test": 1}}, TypeError, "headers"),
        ({"command": "x", "timeout": 0}, ValueError, "timeout"),
        ({"command": "x", "connect_timeout": -1}, ValueError, "connect_timeout"),
        ({"command": "x", "timeout": "5"}, TypeError, "timeout"),
        ({"command": "x", "cwd": 3}, TypeError, "cwd"),
        ({"command": 3}, TypeError, "command"),
    ],
)
def test_bad_arguments_raise_naming_the_field_and_start_nothing(options, error, field):
    with pytest.raises(error, match=field):
        mcp_server(**options)
    assert mcp_threads() == []


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_with_block_opens_lists_tools_and_closes(mode):
    listed = _bare_listing()
    with _server(mode=mode) as server:
        assert server.is_open
        assert server._owner == "host"
        assert server.label == os.path.basename(sys.executable)
        assert [tool.name for tool in server.tools] == [tool.name for tool in listed]
        for tool, expected in zip(server.tools, listed):
            assert isinstance(tool, FunctionTool)
            assert tool.params_json_schema == expected.input_schema
            assert tool.description == expected.description
        pid = int(_invoke(server, "pid"))
        assert child_alive(pid)
    assert not server.is_open
    assert server._owner is None
    assert mcp_threads() == []
    assert not child_alive(pid)


def test_label_option_names_the_thread_and_url_label_is_the_host():
    server = _server(label="tools")
    assert server.label == "tools"
    with server:
        assert mcp_threads() == ["yian-mcp-tools"]
    assert mcp_server(url="https://example.com:8443/mcp").label == "example.com"


def test_open_for_run_records_the_run_token():
    server = _server("--only", "echo")
    try:
        assert server._open_for_run("run-token") is server
        assert server._owner == "run-token"
        assert [tool.name for tool in server.tools] == ["echo"]
    finally:
        server.close()
    assert server._owner is None


def test_paging_lists_every_page():
    with _server(script=PAGED) as server:
        assert [tool.name for tool in server.tools] == ["page1", "page2"]
    with _server("--pages", "3", script=PAGED) as server:
        assert [tool.name for tool in server.tools] == ["page1", "page2", "page3"]


def test_more_than_100_pages_fails_to_open():
    server = _server("--pages", "101", script=PAGED)
    with pytest.raises(McpConnectError, match="100 pages"):
        server.open()
    assert not server.is_open
    assert mcp_threads() == []


def test_tool_list_is_a_snapshot():
    with _server(mode="legacy") as server:
        names = [tool.name for tool in server.tools]
        assert _invoke(server, "add_tool") == "added"
        time.sleep(0.3)
        assert [tool.name for tool in server.tools] == names
        assert "late" not in names


_LISTING_SERVER = textwrap.dedent(
    """
    import anyio
    import mcp_types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    async def list_tools(ctx, params):
        schema = {"type": "object", "properties": {}}
        tools = [types.Tool(name="twin", input_schema=schema), types.Tool(name="twin", input_schema=schema)]
        return types.ListToolsResult(tools=tools)

    async def serve():
        server = Server("listing", on_list_tools=list_tools)
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(serve)
    """
)


def test_a_tool_listed_twice_fails_to_open_naming_it(tmp_path):
    script = tmp_path / "listing_server.py"
    script.write_text(_LISTING_SERVER)
    server = mcp_server(command=sys.executable, args=[str(script)], env=child_env(), label="listing")
    with pytest.raises(McpConnectError, match="twin") as raised:
        server.open()
    assert "'listing'" in str(raised.value)
    assert mcp_threads() == []


def test_a_tool_whose_schema_is_not_an_object_is_rejected():
    # The SDK refuses such a listing on both ends, so a stand-in client serves it.
    class Client:
        async def list_tools(self, cursor=None):
            tool = SimpleNamespace(name="loose", title=None, description=None, input_schema={"properties": {}})
            return SimpleNamespace(tools=[tool], next_cursor=None)

    server = mcp_server(command="x", label="listing")
    with pytest.raises(McpConnectError, match="'listing' tool 'loose'"):
        anyio.run(server._list_all, Client())


def test_missing_command_fails_to_open_with_the_os_error():
    server = mcp_server(command="/nonexistent/yian-no-such-server")
    with pytest.raises(McpConnectError, match="yian-no-such-server") as raised:
        server.open()
    assert isinstance(raised.value, McpError)
    assert isinstance(raised.value, RuntimeError)
    assert isinstance(raised.value.__cause__, OSError)
    assert not server.is_open
    assert mcp_threads() == []


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_server_stderr_is_in_the_connect_error(mode, capfd):
    server = _server("--stderr", "boom", mode=mode, label="noisy")
    with pytest.raises(McpConnectError, match="noisy") as raised:
        server.open()
    assert "boom" in str(raised.value)
    assert "boom" not in capfd.readouterr().err
    assert mcp_threads() == []


def test_hung_server_times_out_and_is_killed(tmp_path):
    marker = tmp_path / "pid"
    server = _server("--hang", "--marker", str(marker), connect_timeout=1, label="hung")
    start = time.monotonic()
    with pytest.raises(McpConnectError, match="hung") as raised:
        server.open()
    assert time.monotonic() - start < 1 + 4
    assert "1 s" in str(raised.value)
    assert mcp_threads() == []
    assert not child_alive(int(marker.read_text()))


def test_open_twice_and_tools_when_closed_raise_and_close_is_idempotent():
    server = _server("--only", "echo")
    server.close()
    with pytest.raises(RuntimeError, match="not open"):
        server.tools
    server.open()
    try:
        with pytest.raises(RuntimeError, match="already open"):
            server.open()
        assert server.is_open
    finally:
        server.close()
    server.close()
    with pytest.raises(RuntimeError, match="not open"):
        server.tools


def test_keyboard_interrupt_during_open_closes_the_server(tmp_path):
    marker = tmp_path / "pid"
    server = _server("--hang", "--marker", str(marker), connect_timeout=20)
    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT))
    start = time.monotonic()
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            server.open()
    finally:
        timer.cancel()
    assert time.monotonic() - start < 2 + 10
    assert not server.is_open
    assert mcp_threads() == []
    assert not child_alive(int(marker.read_text()))


def test_close_from_another_thread_during_open_fails_the_open(tmp_path):
    marker = tmp_path / "pid"
    server = _server("--hang", "--marker", str(marker), connect_timeout=20, label="starting")
    timer = threading.Timer(0.5, server.close)
    start = time.monotonic()
    timer.start()
    try:
        with pytest.raises(McpConnectError, match="'starting' was closed while starting"):
            server.open()
    finally:
        timer.cancel()
        timer.join(15)
    assert time.monotonic() - start < 0.5 + 10
    assert not server.is_open
    assert mcp_threads() == []
    assert not child_alive(int(marker.read_text()))


def test_open_without_the_sdk_names_the_extra(monkeypatch):
    import importlib

    monkeypatch.setitem(sys.modules, "mcp", None)
    # A fresh copy of the module, so the import itself runs without the SDK; monkeypatch
    # puts the original back afterwards.
    monkeypatch.delitem(sys.modules, "yian.tools.mcp")
    module = importlib.import_module("yian.tools.mcp")
    with pytest.raises(ImportError, match=r"yian\[mcp\]"):
        module.mcp_server(command="x").open()
    assert mcp_threads() == []


def test_http_server_lists_tools_and_sends_headers():
    with run_http_server() as (url, recorded):
        with mcp_server(url=url, headers={"x-test": "v"}) as server:
            names = [tool.name for tool in server.tools]
            assert _invoke(server, "headers") == "v"
        assert any(headers.get("x-test") == "v" for headers in recorded)
    assert names == [tool.name for tool in _bare_listing()]
    assert server.label == "localhost"
    assert mcp_threads() == []


def test_http_server_that_is_down_fails_without_leaking_the_url():
    port = free_port()
    server = mcp_server(url=f"http://localhost:{port}/mcp?token=SECRET", connect_timeout=5)
    with pytest.raises(McpConnectError, match="localhost") as raised:
        server.open()
    assert "SECRET" not in str(raised.value)
    assert str(port) not in str(raised.value)
    assert not server.is_open
    assert mcp_threads() == []

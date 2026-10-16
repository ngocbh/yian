"""`ToolSet` takes an opened MCP server as a tool group, and `yian.tools` exports the MCP API."""

import dataclasses
import importlib
import sys

import pytest

pytest.importorskip("mcp")

import yian
from yian.tools import ToolSet
from yian.tools.mcp import McpServerClosed, mcp_server

from tests.engine.mcp_helpers import stdio_params


@pytest.fixture(scope="module")
def server():
    with mcp_server(**stdio_params()) as opened:
        yield opened


def test_an_open_server_is_a_group_of_its_tools(server):
    tools = ToolSet({"fake": server})
    assert tools.groups == ("fake",)
    assert tools.expand(["fake"]) == [tool.name for tool in server.tools]
    assert tools.invoke("echo", {"text": "hi"}) == "hi"


def test_an_unopened_server_is_rejected():
    unopened = mcp_server(**stdio_params())
    with pytest.raises(TypeError) as caught:
        ToolSet({"fake": unopened})
    assert str(caught.value) == (
        "tools['fake'] is an MCP server that isn't open; open it with \"with server:\" "
        "or pass it to run_flow/resume_flow, which open it"
    )


def test_a_server_inside_a_list_group_is_rejected(server):
    with pytest.raises(TypeError) as caught:
        ToolSet({"g": [server]})
    assert str(caught.value) == (
        "tools['g'][0] is an MCP server; pass it as its own tools= entry, which becomes its group"
    )


def test_two_servers_with_one_tool_name_clash():
    with mcp_server(**stdio_params("--only", "echo", "--name", "a")) as first:
        with mcp_server(**stdio_params("--only", "echo", "--name", "b")) as second:
            with pytest.raises(ValueError, match="tool 'echo' is in both groups 'a' and 'b' as different tools"):
                ToolSet({"a": first, "b": second})


def test_a_server_tool_named_like_a_top_level_key_clashes(server):
    def echo(text: str) -> str:
        """Return the text."""
        return text

    with pytest.raises(ValueError, match=r"tool 'echo' in group 'fake' clashes with tools\['echo'\]"):
        ToolSet({"echo": echo, "fake": server})


def test_one_server_under_two_keys_lists_each_tool_once(server):
    tools = ToolSet({"a": server, "b": server})
    names = [tool.name for tool in server.tools]
    assert tools.groups == ("a", "b")
    assert tools.expand(["a", "b"]) == names


def test_a_renamed_tool_in_a_list_group_calls_the_servers_tool(server):
    echo = next(tool for tool in server.tools if tool.name == "echo")
    tools = ToolSet({"renamed": [dataclasses.replace(echo, name="other")]})

    assert tools.expand(["renamed"]) == ["other"]
    assert tools.invoke("other", {"text": "hi"}) == "hi"


def test_a_dead_server_is_rejected():
    with mcp_server(**stdio_params(), label="quitter") as quitter:
        with pytest.raises(McpServerClosed):
            ToolSet({"fake": quitter}).invoke("exit_now", {})
        with pytest.raises(ValueError) as caught:
            ToolSet({"fake": quitter})
    assert str(caught.value) == "tools['fake']: MCP server 'quitter' is closed"


def test_yian_tools_exports_the_mcp_api():
    from yian.tools import (  # noqa: F401
        McpCallCancelled,
        McpConnectError,
        McpError,
        McpInvalidRequestError,
        McpRequestError,
        McpServer,
        McpServerClosed,
        McpTimeoutError,
        McpToolError,
        McpTransportError,
        mcp_server,
    )

    assert hasattr(yian, "McpServer") is False
    assert hasattr(yian, "mcp_server") is False


def test_yian_tools_imports_without_the_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp", None)
    # Fresh copies of the tools package, so its import runs without the SDK; monkeypatch
    # puts the originals back afterwards.
    monkeypatch.setattr(yian, "tools", yian.tools)
    for name in [name for name in sys.modules if name == "yian.tools" or name.startswith("yian.tools.")]:
        monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module("yian.tools")
    assert module.McpServer.__name__ == "McpServer"
    with pytest.raises(ImportError, match=r"yian\[mcp\]"):
        module.mcp_server(command="x").open()

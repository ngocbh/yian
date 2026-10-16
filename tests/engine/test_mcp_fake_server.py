"""The fake MCP servers the MCP client tests use answer a bare SDK client."""

import pytest

pytest.importorskip("mcp")

import anyio
from mcp import Client, StdioServerParameters

from tests.engine.mcp_helpers import (
    BAD_OUTPUT,
    FAKE,
    PAGED,
    HttpServer,
    child_env,
    run_http_server,
    stdio_params,
)

_FAKE_TOOLS = {
    "echo",
    "add",
    "fail",
    "bad_request",
    "busy",
    "crash",
    "picture",
    "resource",
    "whoami",
    "slow",
    "pid",
    "exit_now",
    "gate",
    "peak",
    "calls",
    "nested",
    "add_tool",
    "headers",
}


def _params(*flags, script=FAKE):
    return StdioServerParameters(**stdio_params(*flags, script=script))


async def _tool_names(client):
    return {tool.name for tool in (await client.list_tools()).tools}


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_stdio_lists_tools_and_echo_returns_text(mode):
    async def main():
        async with Client(_params(), mode=mode) as client:
            assert await _tool_names(client) == _FAKE_TOOLS
            result = await client.call_tool("echo", {"text": "hi"})
            assert result.content[0].text == "hi"
            assert result.structured_content is None
            assert not result.is_error

    anyio.run(main)


def test_whoami_reads_request_meta_and_add_is_structured():
    async def main():
        async with Client(_params()) as client:
            who = await client.call_tool("whoami", {}, meta={"yian/call_id": "c", "yian/attempt": 2})
            assert who.content[0].text == "c 2"
            added = await client.call_tool("add", {"a": 1, "b": 2})
            assert added.structured_content == {"result": 3}

    anyio.run(main)


def test_only_keeps_a_subset_and_name_renames_the_server():
    async def main():
        async with Client(_params("--name", "other", "--only", "echo,pid"), mode="legacy") as client:
            assert await _tool_names(client) == {"echo", "pid"}
            assert client.server_info.name == "other"

    anyio.run(main)


def test_fail_is_an_error_result_carrying_the_message():
    async def main():
        async with Client(_params()) as client:
            result = await client.call_tool("fail", {"message": "boom"})
            assert result.is_error
            assert "boom" in result.content[0].text

    anyio.run(main)


def test_marker_records_the_pid_of_the_started_server(tmp_path):
    marker = tmp_path / "started"

    async def main():
        async with Client(_params("--marker", str(marker))) as client:
            pid = (await client.call_tool("pid", {})).content[0].text
        assert marker.read_text() == pid

    anyio.run(main)


def test_http_lists_the_same_tools_and_records_headers():
    async def main(url):
        async with Client(url) as client:
            assert await _tool_names(client) == _FAKE_TOOLS
            assert (await client.call_tool("echo", {"text": "hi"})).content[0].text == "hi"

    with run_http_server() as (url, recorded):
        anyio.run(main, url)
    assert recorded


def test_http_server_restarts_on_the_same_port():
    async def main(url):
        async with Client(url) as client:
            return (await client.call_tool("echo", {"text": "again"})).content[0].text

    server = HttpServer()
    server.start()
    try:
        url = server.url
        assert anyio.run(main, url) == "again"
        server.stop()
        server.start(port=server.port)
        assert server.url == url
        assert anyio.run(main, url) == "again"
    finally:
        server.stop()


def test_paged_server_lists_both_pages():
    async def main():
        async with Client(_params(script=PAGED)) as client:
            first = await client.list_tools()
            assert first.next_cursor
            second = await client.list_tools(cursor=first.next_cursor)
            assert second.next_cursor is None
            return [tool.name for tool in first.tools + second.tools]

    assert anyio.run(main) == ["page1", "page2"]


def test_paged_server_pages_flag_sets_the_page_count():
    async def main():
        names = []
        cursor = None
        async with Client(_params("--pages", "3", script=PAGED)) as client:
            while True:
                page = await client.list_tools(cursor=cursor)
                names += [tool.name for tool in page.tools]
                cursor = page.next_cursor
                if cursor is None:
                    return names

    assert anyio.run(main) == ["page1", "page2", "page3"]


def test_bad_output_server_runs():
    async def main():
        async with Client(_params(script=BAD_OUTPUT)) as client:
            tools = (await client.list_tools()).tools
            assert [tool.name for tool in tools] == ["bad_output"]
            assert tools[0].output_schema["required"] == ["n"]

    anyio.run(main)


def test_stderr_flag_makes_the_child_exit_before_the_handshake():
    async def main():
        params = StdioServerParameters(**stdio_params("--stderr", "no good"))
        with pytest.raises(Exception):
            with anyio.fail_after(20):
                async with Client(params):
                    pass

    anyio.run(main)


def test_child_env_lets_the_child_import_mcp():
    assert "PYTHONPATH" in child_env()

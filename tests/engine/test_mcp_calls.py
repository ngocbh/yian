"""`McpServer` tools call the server and map its results to yian values."""

import base64
import contextlib
import threading

import pytest

pytest.importorskip("mcp")

from mcp_types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

import tests.fixtures.mcp.fake_server as fake_server
from yian.compose import load_flow
from yian.run import run_flow
from yian.tools import ToolSet
from yian.tools.errors import ReportedToolError
from yian.tools.mcp import McpError, McpToolError, _result_value, mcp_server
from yian.typesys.values import TypeCheckError

from tests.engine.mcp_helpers import run_http_server, stdio_params

_ECHO_FLOW = """
id: mcp_echo
name: mcp_echo
nodes:
  say:
    kind: tool
    tool_id: echo
    input:
      text: hi
output: ${say.output}
"""


def _open(kind, stack):
    """Open a fake server of `kind` ("auto", "legacy" or "http") and register its cleanup."""
    if kind == "http":
        url, _ = stack.enter_context(run_http_server())
        server = mcp_server(url=url)
    else:
        server = mcp_server(**stdio_params())
        server._mode = kind
    return stack.enter_context(server)


@pytest.fixture(scope="module", params=["auto", "legacy", "http"])
def server(request):
    with contextlib.ExitStack() as stack:
        yield _open(request.param, stack)


def _tools(server):
    return ToolSet({"fake": list(server.tools)})


def test_echo_returns_the_text(server):
    assert _tools(server).invoke("echo", {"text": "hi"}) == "hi"


def test_structured_content_is_the_value(server):
    assert _tools(server).invoke("add", {"a": 1, "b": 2}) == {"result": 3}


def test_the_call_meta_carries_the_call_id_and_attempt(server):
    tools = _tools(server).for_run("run-1")
    assert tools.invoke("whoami", {}, node="n", attempt=2) == "run-1/n 2"


def test_media_blocks_become_one_line_markers(server):
    size = len(base64.b64decode(fake_server.PNG_BASE64))
    tools = _tools(server)
    assert tools.invoke("picture", {}) == f"[image image/png, {size} B]\n\ncaption"
    assert tools.invoke("resource", {}) == "note body\n\n[resource file:///a.png image/png]"


def test_bad_arguments_fail_before_the_server_is_called(server):
    tools = _tools(server)
    before = tools.invoke("calls", {})
    with pytest.raises(TypeCheckError):
        tools.invoke("add", {"a": "x", "b": 1})
    assert tools.invoke("calls", {}) == before


def test_nested_arguments_reach_the_server(server):
    item = {"name": "p", "point": {"x": 1, "y": 2}}
    assert _tools(server).invoke("nested", {"item": item}) == '{"name":"p","point":{"x":1,"y":2}}'


def test_an_error_result_raises_a_reported_mcp_tool_error(server):
    with pytest.raises(McpToolError, match="no luck") as caught:
        _tools(server).invoke("fail", {"message": "no luck"})
    assert isinstance(caught.value, ReportedToolError)
    assert isinstance(caught.value, McpError)


@pytest.mark.parametrize("kind", ["auto", "legacy", "http"])
def test_calls_from_many_threads_run_at_once(kind):
    # The HTTP server runs in this process, so its counters outlive earlier tests.
    fake_server._peak = 0
    with contextlib.ExitStack() as stack:
        tools = _tools(_open(kind, stack))
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(tools.invoke("gate", {"seconds": 0.3})))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert results == ["done"] * 8
        assert tools.invoke("peak", {})["result"] > 1


def test_a_tool_node_calls_a_host_opened_server():
    with mcp_server(**stdio_params("--only", "echo")) as server:
        result = run_flow(load_flow(_ECHO_FLOW), {}, tools={"fake": list(server.tools)})
        assert server.is_open
    assert result.status == "succeeded", result.failure
    assert result.output == "hi"


def _b64(size):
    return base64.b64encode(b"\0" * size).decode()


def test_result_value_renders_each_kind_of_content():
    result = CallToolResult(
        content=[
            TextContent(text="one"),
            AudioContent(data=_b64(1023), mime_type="audio/wav"),
            ImageContent(data=_b64(2048), mime_type="image/png"),
            EmbeddedResource(
                resource=BlobResourceContents(
                    uri="file:///b.bin", blob=_b64(3 * 1024 * 1024 // 2), mime_type="application/octet-stream"
                )
            ),
            EmbeddedResource(resource=BlobResourceContents(uri="file:///c.bin", blob=_b64(10))),
            EmbeddedResource(resource=TextResourceContents(uri="text://t", text="two")),
            ResourceLink(name="d", uri="file:///d"),
        ]
    )
    assert _result_value(result) == "\n\n".join(
        [
            "one",
            "[audio audio/wav, 1023 B]",
            "[image image/png, 2.0 KB]",
            "[resource file:///b.bin application/octet-stream, 1.5 MB]",
            "[resource file:///c.bin, 10 B]",
            "two",
            "[resource file:///d]",
        ]
    )


def test_result_value_prefers_structured_content_and_renders_empty_content_as_empty():
    assert _result_value(CallToolResult(content=[], structured_content=[1, 2])) == [1, 2]
    assert _result_value(CallToolResult(content=[TextContent(text="x")], structured_content=0)) == 0
    assert _result_value(CallToolResult(content=[])) == ""

"""A fake MCP server with one tool per behaviour the MCP client tests need.

Import `build_server` to serve it in-process, or run this file as a script to serve it
over stdio (or streamable HTTP with `--http PORT`).
"""

import argparse
import os
import sys
import threading
import time

import anyio
from mcp import MCPError
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import EmbeddedResource, ImageContent, ResourceLink, TextContent, TextResourceContents
from pydantic import BaseModel

# A 1x1 transparent PNG.
PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

# Shared by every server in this process: `gate` and `peak` measure concurrency, `calls`
# counts served calls.
_lock = threading.Lock()
_running = 0
_peak = 0
_calls = 0


class Point(BaseModel):
    """A nested model, so `Item`'s input schema carries `$defs` and `$ref`."""

    x: int
    y: int


class Item(BaseModel):
    """The argument of `nested`."""

    name: str
    point: Point


class _CountingServer(MCPServer):
    """An `MCPServer` that counts every tool call except `calls` itself."""

    async def call_tool(self, name, arguments, context=None):
        """Count the call, then run the tool as `MCPServer` does.

        Args:
            name (`str`):
                The tool name.
            arguments (`dict[str, Any]`):
                The tool arguments.
            context (`Context`, *optional*, defaults to `None`):
                The request context. `None` builds one outside any request.

        Returns:
            `CallToolResult | InputRequiredResult`:
                The tool's result.
        """
        global _calls
        if name != "calls":
            with _lock:
                _calls += 1
        return await super().call_tool(name, arguments, context)


def build_server(name="fake", only=None):
    """Build the fake server.

    Args:
        name (`str`, *optional*, defaults to `"fake"`):
            The server name it reports.
        only (`set[str]`, *optional*, defaults to `None`):
            Tool names to keep. `None` keeps every tool.

    Returns:
        `MCPServer`:
            The server, not yet serving.
    """
    srv = _CountingServer(name)

    def tool(**options):
        def register(fn):
            if only is None or fn.__name__ in only:
                srv.add_tool(fn, **options)
            return fn

        return register

    text = tool(structured_output=False)

    @text
    def echo(text: str) -> str:
        """Return `text`."""
        return text

    @tool()
    def add(a: int, b: int) -> int:
        """Return `a + b`."""
        return a + b

    @text
    def fail(message: str) -> str:
        """Fail with `message`, which the server reports as an error result."""
        # A plain exception would reach the client as "Error executing tool fail" alone;
        # `ToolError` keeps the message.
        raise ToolError(message)

    @text
    def bad_request() -> str:
        """Fail with a JSON-RPC invalid-params error."""
        raise MCPError(-32602, "bad params")

    @text
    def busy() -> str:
        """Fail with a server-defined JSON-RPC error."""
        raise MCPError(-32000, "busy")

    @text
    def crash() -> str:
        """Fail with a JSON-RPC internal error."""
        raise MCPError(-32603, "internal")

    @text
    def picture() -> list:
        """Return an image block, then a text caption."""
        return [ImageContent(data=PNG_BASE64, mime_type="image/png"), TextContent(text="caption")]

    @text
    def resource() -> list:
        """Return an embedded text resource and a resource link."""
        return [
            EmbeddedResource(
                resource=TextResourceContents(uri="text://note", text="note body", mime_type="text/plain")
            ),
            ResourceLink(name="a.png", uri="file:///a.png", mime_type="image/png"),
        ]

    @text
    def whoami(ctx: Context) -> str:
        """Return the call id and attempt yian put in the request `_meta`."""
        meta = ctx.request_context.meta or {}
        return f"{meta['yian/call_id']} {meta['yian/attempt']}"

    @text
    async def slow(seconds: float) -> str:
        """Sleep `seconds` without blocking the server, then return "done"."""
        await anyio.sleep(seconds)
        return "done"

    @text
    def pid() -> str:
        """Return the server's process id."""
        return str(os.getpid())

    @text
    def exit_now() -> str:
        """Kill the server process mid-call."""
        os._exit(3)

    @text
    def gate(seconds: float) -> str:
        """Block a worker thread for `seconds`, tracking how many calls overlap."""
        global _running, _peak
        with _lock:
            _running += 1
            _peak = max(_peak, _running)
        try:
            time.sleep(seconds)
        finally:
            with _lock:
                _running -= 1
        return "done"

    @tool()
    def peak() -> int:
        """Return the most `gate` calls that ran at once."""
        return _peak

    @tool()
    def calls() -> int:
        """Return how many tool calls, other than `calls`, the server has served."""
        return _calls

    @text
    def nested(item: Item) -> str:
        """Return `item` as JSON."""
        return item.model_dump_json()

    @text
    async def add_tool(ctx: Context) -> str:
        """Register the tool `late` and tell the client the tool list changed."""

        def late() -> str:
            """A tool added while the server runs."""
            return "late"

        srv.add_tool(late, structured_output=False)
        await ctx.session.send_tool_list_changed()
        return "added"

    @text
    def headers(ctx: Context) -> str:
        """Return the request's `x-test` HTTP header, or "" when there is none."""
        return (ctx.headers or {}).get("x-test", "")

    return srv


def serve_http(app, port):
    """Serve an ASGI app on 127.0.0.1 until the process is stopped.

    Args:
        app (`Callable`):
            The ASGI app.
        port (`int`):
            The port to listen on.
    """
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, lifespan="on", log_level="warning")


def main(argv=None):
    """Run the fake server as a script.

    Args:
        argv (`list[str]`, *optional*, defaults to `None`):
            Command-line arguments. `None` reads `sys.argv`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="fake")
    parser.add_argument("--only", help="comma-separated tool names to keep")
    parser.add_argument("--marker", help="append the server's pid to this file at start, one line per start")
    parser.add_argument("--hang", action="store_true", help="never answer")
    parser.add_argument("--http", type=int, metavar="PORT", help="serve streamable HTTP on 127.0.0.1:PORT")
    parser.add_argument("--stderr", metavar="TEXT", help="write TEXT to stderr and exit 1 before serving")
    args = parser.parse_args(argv)

    if args.marker:
        # Appended, so a test can count the starts; a single start writes the bare pid.
        with open(args.marker, "a") as marker:
            marker.write(("\n" if marker.tell() else "") + str(os.getpid()))
    if args.stderr is not None:
        sys.stderr.write(args.stderr + "\n")
        sys.exit(1)
    if args.hang:
        while True:
            time.sleep(3600)
    srv = build_server(args.name, set(args.only.split(",")) if args.only else None)
    if args.http is not None:
        serve_http(srv.streamable_http_app(), args.http)
    else:
        srv.run("stdio")


if __name__ == "__main__":
    main()

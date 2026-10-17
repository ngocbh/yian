"""Serve a lowlevel MCP server over stdio."""

import anyio
from mcp.server.stdio import stdio_server


def run_stdio(server):
    """Serve `server` over this process's stdin and stdout until the client closes.

    Args:
        server (`mcp.server.Server`):
            The lowlevel server.
    """

    async def serve():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    anyio.run(serve)

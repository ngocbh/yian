"""A lowlevel MCP server that lists its tools one per page, following a cursor.

Run as a script with `--pages N` to serve N pages (default 2); tool `pageK` answers "K".
"""

import argparse

import mcp_types as types
from mcp.server import Server


def build_server(pages=2):
    """Build the server.

    Args:
        pages (`int`, *optional*, defaults to `2`):
            How many pages `tools/list` returns, one tool each.

    Returns:
        `mcp.server.Server`:
            The server, not yet serving.
    """

    async def list_tools(ctx, params):
        index = int(params.cursor) if params is not None and params.cursor else 1
        tool = types.Tool(name=f"page{index}", input_schema={"type": "object", "properties": {}})
        return types.ListToolsResult(tools=[tool], next_cursor=str(index + 1) if index < pages else None)

    async def call_tool(ctx, params):
        return types.CallToolResult(content=[types.TextContent(text=params.name.removeprefix("page"))])

    return Server("paged", on_list_tools=list_tools, on_call_tool=call_tool)


if __name__ == "__main__":
    from _stdio import run_stdio

    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=2)
    run_stdio(build_server(parser.parse_args().pages))

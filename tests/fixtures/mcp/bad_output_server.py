"""A lowlevel MCP server whose one tool returns structured content its `outputSchema` rejects."""

import mcp_types as types
from mcp.server import Server

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
}


async def _list_tools(ctx, params):
    return types.ListToolsResult(
        tools=[
            types.Tool(
                name="bad_output",
                description="Return structured content that violates the advertised schema.",
                input_schema={"type": "object", "properties": {}},
                output_schema=OUTPUT_SCHEMA,
            )
        ]
    )


async def _call_tool(ctx, params):
    return types.CallToolResult(
        content=[types.TextContent(text='{"n": "x"}')],
        structured_content={"n": "x"},
    )


def build_server():
    """Build the server.

    Returns:
        `mcp.server.Server`:
            The server, not yet serving.
    """
    return Server("bad-output", on_list_tools=_list_tools, on_call_tool=_call_tool)


if __name__ == "__main__":
    from _stdio import run_stdio

    run_stdio(build_server())

"""Tool registry seam for TOOL, AGENT, and host-owned human interaction.

The core ships no domain tools; a host registers its own via ``register_tool``.
Model-requested calls use the strict public schema. Host-owned interactions use
the trusted path for injected metadata. ``ToolSet`` holds one run's tools: the
global registry, or a host mapping (with its ``ask_user`` handler) that replaces it.
``tools.types`` reads a tool's declared argument and return types as yian types.
``tools.schema`` builds a tool's argument model and description from a plain
function, whose optional first ``ToolContext`` parameter (``tools.context``)
receives the call's identity and the host's app object. ``tools.function_tools``
holds ``FunctionTool`` (a tool given as a name, description, JSON Schema and
``on_invoke``) and ``function_tool``, which builds one from a function with an
optional name and description override. ``tools.errors`` holds
``ReportedToolError``, which a tool raises for a failure an AGENT's model reads as
the call's result, ``PermanentToolError`` and ``TransientToolError``, which a tool
raises to stop or steer a TOOL node's retries, and ``ToolCheckError``, raised when a
check yian runs on a call's result fails. ``tools.mcp`` holds ``mcp_server``, which
describes a Model Context Protocol server, the ``McpServer`` it returns, whose opened
tools a ``ToolSet`` takes as one group, and the ``McpError`` classes its calls raise;
the MCP SDK is imported only when a server opens.

Knows about: langchain-core, pydantic, typesys, the skills leaf, the stdlib-only
yian.defaults, the MCP SDK (lazily, in ``tools.mcp``), and the Python standard library.
Never imports: nodes, compile, compose, runtime, the run boundary, or CLI.
"""

from __future__ import annotations

from yian.tools.context import ToolContext
from yian.tools.errors import (
    PermanentToolError,
    ReportedToolError,
    ToolCheckError,
    TransientToolError,
)
from yian.tools.function_tools import FunctionTool, function_tool
from yian.tools.mcp import (
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
from yian.tools.registry import (
    TOOL_REGISTRY,
    ToolSet,
    invoke_tool,
    invoke_tool_trusted,
    register_tool,
    registered_tools,
    resolve_tools,
)

__all__ = [
    "TOOL_REGISTRY",
    "FunctionTool",
    "McpCallCancelled",
    "McpConnectError",
    "McpError",
    "McpInvalidRequestError",
    "McpRequestError",
    "McpServer",
    "McpServerClosed",
    "McpTimeoutError",
    "McpToolError",
    "McpTransportError",
    "PermanentToolError",
    "ReportedToolError",
    "ToolCheckError",
    "ToolContext",
    "ToolSet",
    "TransientToolError",
    "function_tool",
    "invoke_tool",
    "invoke_tool_trusted",
    "mcp_server",
    "register_tool",
    "registered_tools",
    "resolve_tools",
]

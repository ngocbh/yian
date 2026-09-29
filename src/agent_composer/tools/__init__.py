"""Tool registry seam for TOOL, AGENT, and host-owned human interaction.

The core ships no domain tools; a host registers its own via ``register_tool``.
Model-requested calls use the strict public schema. Host-owned interactions use
the trusted path for injected metadata. ``ToolSet`` holds one run's tools: the
global registry, or a host mapping (with its ``ask_user`` handler) that replaces it.

Knows about: langchain-core and the Python standard library.
Never imports: nodes, compile, compose, runtime, the run boundary, or CLI.
"""

from __future__ import annotations

from agent_composer.tools.registry import (
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
    "ToolSet",
    "invoke_tool",
    "invoke_tool_trusted",
    "register_tool",
    "registered_tools",
    "resolve_tools",
]

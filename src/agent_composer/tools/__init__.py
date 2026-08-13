"""Tool registry seam — the engine's TOOL node and AGENT tool-calling resolve here.

The core ships no domain tools; a host registers its own via ``register_tool``.
"""

from __future__ import annotations

from agent_composer.tools.registry import (
    TOOL_REGISTRY,
    install_ask_user,
    invoke_tool,
    invoke_tool_trusted,
    register_tool,
    resolve_tools,
)

__all__ = [
    "TOOL_REGISTRY",
    "install_ask_user",
    "invoke_tool",
    "invoke_tool_trusted",
    "register_tool",
    "resolve_tools",
]

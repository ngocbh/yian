"""Tool registry seam for TOOL, AGENT, and host-owned human interaction.

The core ships no domain tools; a host registers its own via ``register_tool``.
Model-requested calls use the strict public schema. HUMAN_INPUT alone uses the
trusted path for host-injected metadata. ``install_ask_user`` gives the reserved
entry one process-wide owner for a complete host lifecycle.
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

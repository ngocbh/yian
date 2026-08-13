"""AGENT package — one authored leaf with a process-local model/tool loop.

``modes/`` holds the prompting methods. Every granted capability, including
``ask_user``, is an ordinary tool executed inside the live loop.
"""

from agent_composer.nodes.agent.node import AgentNode, DEFAULT_MODE
from agent_composer.nodes.agent.modes import (
    MODES,
    AgentLoopError,
    AgentRunContext,
    register_mode,
)

__all__ = [
    "AgentNode",
    "DEFAULT_MODE",
    "MODES",
    "AgentLoopError",
    "AgentRunContext",
    "register_mode",
]

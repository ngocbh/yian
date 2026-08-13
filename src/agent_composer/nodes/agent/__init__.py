"""AGENT package — one authored leaf with a process-local model/tool loop.

``modes/`` holds the prompting methods. Every granted capability, including the
host-installed blocking ``ask_user``, is an ordinary authored tool executed
through the public invocation path inside the live loop. No agent continuation
or pause topology lives here.
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

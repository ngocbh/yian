"""AGENT package — one authored leaf with a process-local model/tool loop.

``modes/`` holds the prompting methods. Every granted capability, including the
host-installed blocking ``ask_user``, is an ordinary authored tool executed
through the public invocation path inside the live loop. The package owns no
durable run state.

Knows about: nodes.base, expr, typesys, tools, llm_clients, peer AGENT modules,
             LangChain, and Pydantic.
Never imports: compile, compose, runtime, the run boundary, or CLI.
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

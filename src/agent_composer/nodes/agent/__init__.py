"""AGENT package — one authored leaf with shared model/tool execution.

``modes/`` holds the prompting methods. Explicitly granted Composer tools,
including the host-installed blocking ``ask_user``, execute through the public
invocation path inside the live loop. The package owns no durable run state.
``conversation`` adapts provider messages to optional memory-owned turns; it
owns neither retained history nor turn admission. Modes persist actual exchanges
through this adapter, independently of evaluation evidence.
The adapter also projects safe historical context and injects the current
unbound model for memory-owned summary preparation; summary state stays in memory.
``backward`` builds one tool-free model request from accepted invocation material;
it neither rerenders the forward prompt nor updates local variables.

Knows about: nodes.base, expr, typesys, tools, llm_clients, memory contracts, peer AGENT modules,
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

"""AGENT package — one authored leaf with shared model/tool execution.

``loop`` holds the one AGENT body: one model call without tools, the tool loop
with them. ``context`` holds the per-invocation context, and ``text`` the
visible-text and trace helpers. Explicitly granted Yian tools, including the
run's blocking ``ask_user``, execute through the public invocation path inside
the live loop. The package owns no durable run state.
``memory`` adapts the shared memory context to provider messages and effects; it
owns neither retained history nor turn admission. The loop persists actual exchanges
through this adapter, independently of evaluation evidence.
The adapter also projects safe historical context and injects the current
unbound model for memory-owned summary preparation; summary state stays in memory.
``backward`` builds one tool-free model request from accepted invocation material;
it neither rerenders the forward prompt nor updates local variables.
``live`` publishes transient streamed reply text, model/tool call observers, and
provider token usage to evaluator-supplied sinks; it records and retains nothing.
``interrupts`` holds the signal a host interrupt raises out of a streamed reply.

Knows about: nodes.base, events, expr, typesys, tools, llm_clients, memory contracts, peer AGENT modules,
             LangChain, and Pydantic.
Never imports: compile, compose, runtime, the run boundary, or CLI.
"""

from yian.nodes.agent.context import AgentLoopError, AgentRunContext
from yian.nodes.agent.node import AgentNode

__all__ = [
    "AgentNode",
    "AgentLoopError",
    "AgentRunContext",
]

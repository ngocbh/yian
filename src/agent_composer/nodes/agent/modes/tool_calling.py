"""Tool-calling mode — run one local model/tool loop until a final answer."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

import agent_composer.llm_clients as llm_clients
from agent_composer.nodes.agent.modes.common import (
    DEFAULT_SYSTEM,
    AgentLoopError,
    AgentRunContext,
    register_mode,
)
from agent_composer.nodes.agent.modes.utils import text_of
from agent_composer.nodes.base import Output
from agent_composer.tools import invoke_tool, resolve_tools

# -1 means no bound; a positive per-node env value bounds model turns.
MAX_TOOL_ITERATIONS = -1


@register_mode("tool_calling")
def tool_calling(ctx: AgentRunContext) -> Output:
    """Execute model-returned tool calls in order until the model answers."""
    messages: list[BaseMessage] = [
        SystemMessage(content=DEFAULT_SYSTEM),
        HumanMessage(content=ctx.prompt),
    ]
    model = llm_clients.model_from_config(ctx.llm_config or {})
    bound = resolve_tools(list(ctx.tools))
    chat = model.bind_tools(bound) if bound else model
    iterations = 0

    while ctx.max_tool_iterations < 0 or iterations < ctx.max_tool_iterations:
        reply = chat.invoke(messages)
        iterations += 1
        messages.append(reply)
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            from agent_composer.nodes.agent.structured import generate_structured, type_to_schema

            if ctx.output_type is not None and type_to_schema(ctx.output_type) is not None:
                return Output(
                    value=generate_structured(
                        model,
                        messages,
                        ctx.output_type,
                        max_retries=ctx.retries,
                        llm_config=ctx.llm_config,
                    )
                )
            return Output(value=text_of(reply))

        for call in calls:
            name = call.get("name") or ""
            if name not in ctx.tools:
                raise AgentLoopError(
                    f"agent node {ctx.node_id!r} requested tool {name!r}, but it is not "
                    f"granted; granted tools: {ctx.tools}"
                )
            result = invoke_tool(name, call.get("args") or {})
            messages.append(
                ToolMessage(content=str(result), tool_call_id=call.get("id") or "")
            )

    raise AgentLoopError(
        f"agent node {ctx.node_id!r} hit the tool-iteration cap "
        f"({ctx.max_tool_iterations}) without a final answer"
    )

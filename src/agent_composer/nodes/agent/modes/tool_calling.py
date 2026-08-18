"""Tool-calling mode — run one local model/tool loop until a final answer."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

import agent_composer.llm_clients as llm_clients
from agent_composer.memory.contracts import (
    ModelTextInteraction,
    ToolCallInteraction,
    ToolResultInteraction,
)
from agent_composer.nodes.agent.modes.common import (
    DEFAULT_SYSTEM,
    AgentLoopError,
    AgentRunContext,
    register_mode,
)
from agent_composer.nodes.agent.modes.utils import text_of
from agent_composer.nodes.agent.structured import generate_structured, type_to_schema
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
    ctx.check_stopped()
    model = llm_clients.model_from_config(ctx.llm_config or {})
    ctx.check_stopped()
    if ctx.evaluation_trace is not None:
        ctx.evaluation_trace.record_prompt(DEFAULT_SYSTEM, ctx.prompt)
    schema = type_to_schema(ctx.output_type) if ctx.output_type is not None else None
    if not ctx.tools and schema is not None:
        return Output(
            value=generate_structured(
                model,
                messages,
                ctx.output_type,
                max_retries=ctx.retries,
                llm_config=ctx.llm_config,
                check_stopped=ctx.check_stopped,
                evaluation_trace=ctx.evaluation_trace,
            )
        )

    bound = resolve_tools(list(ctx.tools))
    ctx.check_stopped()
    chat = model.bind_tools(bound) if bound else model
    iterations = 0

    while ctx.max_tool_iterations < 0 or iterations < ctx.max_tool_iterations:
        ctx.check_stopped()
        reply = chat.invoke(messages)
        ctx.check_stopped()
        iterations += 1
        messages.append(reply)
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            text = text_of(reply)
            if ctx.evaluation_trace is not None:
                ctx.evaluation_trace.record_interaction(ModelTextInteraction(text))
            if ctx.output_type is not None and schema is not None:
                return Output(
                    value=generate_structured(
                        model,
                        messages,
                        ctx.output_type,
                        max_retries=ctx.retries,
                        llm_config=ctx.llm_config,
                        check_stopped=ctx.check_stopped,
                        evaluation_trace=ctx.evaluation_trace,
                    )
                )
            return Output(value=text)

        for call in calls:
            ctx.check_stopped()
            raw_name = call.get("name") or ""
            name = str(raw_name) if isinstance(raw_name, str) else raw_name
            raw_arguments = call.get("args")
            arguments = raw_arguments or {}
            if ctx.evaluation_trace is not None:
                try:
                    call_interaction = ToolCallInteraction(name, arguments)
                except (TypeError, ValueError, RecursionError):
                    # Evidence remains observational for malformed arguments; the
                    # ordinary tool boundary below retains its established error.
                    pass
                else:
                    ctx.evaluation_trace.record_interaction(call_interaction)
            if name not in ctx.tools:
                raise AgentLoopError(
                    f"agent node {ctx.node_id!r} requested tool {name!r}, but it is not "
                    f"granted; granted tools: {ctx.tools}"
                )
            result = invoke_tool(name, arguments)
            ctx.check_stopped()
            result_text = str(result)
            if ctx.evaluation_trace is not None:
                ctx.evaluation_trace.record_interaction(
                    ToolResultInteraction(name, result_text)
                )
            messages.append(
                ToolMessage(content=result_text, tool_call_id=call.get("id") or "")
            )

    raise AgentLoopError(
        f"agent node {ctx.node_id!r} hit the tool-iteration cap "
        f"({ctx.max_tool_iterations}) without a final answer"
    )

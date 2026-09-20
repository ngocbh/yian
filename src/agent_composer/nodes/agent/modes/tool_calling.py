"""Tool-calling mode — drive only newly returned model calls using current grants.

An optional memory-owned turn retains actual exchanges. Restored provider context
never drives tool dispatch, and its synthetic missing-result repairs are not saved.
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

import agent_composer.llm_clients as llm_clients
from agent_composer.memory.contracts import (
    ToolCallInteraction,
)
from agent_composer.nodes.agent.conversation import (
    current_history_messages,
    prepare_history,
    invoke_model,
    invoke_tool_effect,
)
from agent_composer.nodes.agent.modes.common import (
    DEFAULT_SYSTEM,
    AgentLoopError,
    AgentRunContext,
    register_mode,
)
from agent_composer.nodes.agent.modes.utils import (
    record_model_text,
    record_tool_result,
    system_prompt_with_memory,
    text_of,
)
from agent_composer.nodes.agent.structured import generate_structured, type_to_schema
from agent_composer.nodes.base import Output
from agent_composer.tools import invoke_tool, resolve_tools

# -1 means no bound; a positive per-node env value bounds model turns.
MAX_TOOL_ITERATIONS = -1


@register_mode("tool_calling")
def tool_calling(ctx: AgentRunContext) -> Output:
    """Execute model-returned tool calls in order until the model answers."""
    system_prompt = system_prompt_with_memory(DEFAULT_SYSTEM, ctx.memory_sections)
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=ctx.prompt),
    ]
    new_messages = list(messages)
    ctx.check_stopped()
    model = llm_clients.model_from_config(ctx.llm_config or {})
    ctx.check_stopped()
    prior = []
    if ctx.conversation_turn is not None:
        prior = prepare_history(
            ctx.conversation_turn, model, check_stopped=ctx.check_stopped
        )
        messages = [messages[0], *prior, *messages[1:]]
    if ctx.evaluation_trace is not None:
        ctx.evaluation_trace.record_prompt(
            DEFAULT_SYSTEM,
            ctx.prompt,
            ctx.memory_sections,
        )
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
                conversation_turn=ctx.conversation_turn,
                new_messages=new_messages,
            )
        )

    bound = resolve_tools(list(ctx.tools))
    ctx.check_stopped()
    chat = model.bind_tools(bound) if bound else model
    iterations = 0

    while ctx.max_tool_iterations < 0 or iterations < ctx.max_tool_iterations:
        ctx.check_stopped()
        exchange_id = uuid4().hex
        if ctx.conversation_turn is None:
            reply = chat.invoke(messages)
            ctx.check_stopped()
        else:
            reply = invoke_model(
                ctx.conversation_turn,
                lambda: chat.invoke(messages),
                new_messages=new_messages,
                check_stopped=ctx.check_stopped,
                exchange_id=exchange_id,
            )
            new_messages = []
        iterations += 1
        messages.append(reply)
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            text = text_of(reply)
            record_model_text(ctx.evaluation_trace, text)
            if ctx.output_type is not None and schema is not None:
                if ctx.conversation_turn is not None:
                    messages = [
                        messages[0],
                        *prior,
                        *current_history_messages(ctx.conversation_turn),
                    ]
                return Output(
                    value=generate_structured(
                        model,
                        messages,
                        ctx.output_type,
                        max_retries=ctx.retries,
                        llm_config=ctx.llm_config,
                        check_stopped=ctx.check_stopped,
                        evaluation_trace=ctx.evaluation_trace,
                        conversation_turn=ctx.conversation_turn,
                        new_messages=[],
                    )
                )
            return Output(value=text)

        for call_index, call in enumerate(calls):
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
            if ctx.conversation_turn is None:
                result = invoke_tool(name, arguments)
                ctx.check_stopped()
                result_text = str(result)
                record_tool_result(ctx.evaluation_trace, name, result_text)
                result_message = ToolMessage(
                    content=result_text, tool_call_id=call.get("id") or ""
                )
            else:
                result_message = invoke_tool_effect(
                    ctx.conversation_turn,
                    lambda: invoke_tool(name, arguments),
                    name=name,
                    arguments=arguments,
                    model_exchange_id=exchange_id,
                    call_index=call_index,
                    tool_call_id=call.get("id") or "",
                    check_stopped=ctx.check_stopped,
                )
                result_text = result_message.content
                record_tool_result(ctx.evaluation_trace, name, result_text)
            messages.append(result_message)

    raise AgentLoopError(
        f"agent node {ctx.node_id!r} hit the tool-iteration cap "
        f"({ctx.max_tool_iterations}) without a final answer"
    )

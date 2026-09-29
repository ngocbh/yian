"""Tool-calling mode — drive only newly returned model calls using current grants.

An optional memory-owned turn retains actual exchanges. Restored provider context
never drives tool dispatch, and its synthetic missing-result repairs are not saved.
"""

from __future__ import annotations

from functools import partial
from uuid import uuid4

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

import agent_composer.llm_clients as llm_clients
from agent_composer.memory.contracts import (
    ToolCallInteraction,
)
from agent_composer.nodes.agent.context import (
    current_history_messages,
    system_prompt_with_memory,
    prepare_history,
    invoke_model,
    invoke_tool_effect,
)
from agent_composer.nodes.agent.live import (
    invoke_chat,
    live_sinks,
    observe_model_call,
    observe_tool_call,
)
from agent_composer.nodes.agent.modes.common import (
    AgentLoopError,
    AgentRunContext,
    register_mode,
)
from agent_composer.nodes.agent.modes.utils import (
    record_model_text,
    record_tool_result,
    text_of,
)
from agent_composer.nodes.agent.structured import generate_structured, type_to_schema
from agent_composer.nodes.base import Output
from agent_composer.skills.tool import LOAD_SKILL, invoke_load_skill, load_skill_tool

# -1 means no bound; a positive per-node env value bounds model turns.
MAX_TOOL_ITERATIONS = -1


@register_mode("tool_calling")
def tool_calling(ctx: AgentRunContext) -> Output:
    """Execute model-returned tool calls in order until the model answers."""
    system_prompt = system_prompt_with_memory(ctx.base_system_prompt, ctx.memory_context.sections)
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=ctx.prompt),
    ]
    new_messages = list(messages)
    live = live_sinks(ctx)
    ctx.check_stopped()
    model = llm_clients.model_from_config(ctx.llm_config or {})
    ctx.check_stopped()
    prior = prepare_history(
        ctx.memory_context.transcript,
        model,
        check_stopped=ctx.check_stopped,
        observe_summary=lambda invoke: observe_model_call(
            live, invoke, purpose="history_summary", exchange_id=uuid4().hex
        ),
    )
    messages = [messages[0], *prior, *messages[1:]]
    if ctx.evaluation_trace is not None:
        ctx.evaluation_trace.record_prompt(
            ctx.base_system_prompt,
            ctx.prompt,
            ctx.memory_context.sections,
        )
    schema = type_to_schema(ctx.output_type) if ctx.output_type is not None else None
    if not ctx.tools and not ctx.skills and schema is not None:
        return Output(
            value=generate_structured(
                model,
                messages,
                ctx.output_type,
                max_retries=ctx.retries,
                llm_config=ctx.llm_config,
                check_stopped=ctx.check_stopped,
                evaluation_trace=ctx.evaluation_trace,
                memory_context=ctx.memory_context,
                new_messages=new_messages,
                live=live,
            )
        )

    bound = ctx.toolset.resolve(list(ctx.tools))
    skills = {skill.name: skill for skill in ctx.skills}
    if skills:
        bound.append(load_skill_tool())
    ctx.check_stopped()
    chat = model.bind_tools(bound) if bound else model
    iterations = 0

    while ctx.max_tool_iterations < 0 or iterations < ctx.max_tool_iterations:
        ctx.check_stopped()
        exchange_id = uuid4().hex
        reply = invoke_model(
            ctx.memory_context.transcript,
            lambda: invoke_chat(live, chat, messages, exchange_id=exchange_id),
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
                messages = [
                    messages[0],
                    *prior,
                    *current_history_messages(
                        ctx.memory_context.transcript, fallback=messages[1:]
                    ),
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
                        memory_context=ctx.memory_context,
                        new_messages=[],
                        live=live,
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
            if skills and name == LOAD_SKILL:
                invoke = partial(invoke_load_skill, skills, arguments)
            elif name in ctx.tools:
                invoke = partial(ctx.toolset.invoke, name, arguments)
            else:
                raise AgentLoopError(
                    f"agent node {ctx.node_id!r} requested tool {name!r}, but it is not "
                    f"granted; granted tools: {ctx.tools}"
                )
            tool_call_id = call.get("id") or ""
            result_message = invoke_tool_effect(
                ctx.memory_context.transcript,
                lambda: observe_tool_call(
                    live, invoke,
                    exchange_id=exchange_id, call_index=call_index,
                    tool_call_id=tool_call_id, name=name, arguments=arguments,
                ),
                name=name,
                arguments=arguments,
                model_exchange_id=exchange_id,
                call_index=call_index,
                tool_call_id=tool_call_id,
                check_stopped=ctx.check_stopped,
            )
            record_tool_result(ctx.evaluation_trace, name, result_message.content)
            messages.append(result_message)

    raise AgentLoopError(
        f"agent node {ctx.node_id!r} hit the tool-iteration cap "
        f"({ctx.max_tool_iterations}) without a final answer"
    )

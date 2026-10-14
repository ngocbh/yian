"""The AGENT loop — drive only newly returned model calls using current grants.

Without tools or skills the loop makes a single model call (structured when the
output type needs it); with them it runs the tool loop until the model answers.

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

import yian.llm_clients as llm_clients
from yian.memory.contracts import (
    ToolCallInteraction,
)
from yian.nodes.agent.memory import (
    current_history_messages,
    system_prompt_with_memory,
    prepare_history,
    interrupt_turn,
    invoke_model,
    invoke_tool_effect,
    record_skipped_tool,
)
from yian.nodes.agent.interrupts import ReplyInterrupted
from yian.nodes.agent.live import (
    invoke_chat,
    live_sinks,
    observe_model_call,
    observe_skipped_tool_call,
    observe_tool_call,
)
from yian.nodes.agent.context import AgentLoopError, AgentRunContext
from yian.nodes.agent.text import (
    record_model_text,
    record_tool_result,
    text_of,
)
from yian.nodes.agent.structured import generate_structured, type_to_schema
from yian.nodes.base import Output
from yian.skills.tool import LOAD_SKILL, invoke_load_skill, load_skill_tool
from yian.typesys.values import reject_deep_nesting, reject_nonfinite

# -1 means no bound; a positive per-node env value bounds model turns.
MAX_TOOL_ITERATIONS = -1


def run_agent(ctx: AgentRunContext) -> Output:
    """Run one AGENT invocation.

    Makes one model call without tools or skills, otherwise runs the tool loop until the
    model answers.
    """
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
    # A granted id may name a host-filled group; grants are checked by tool name.
    granted = ctx.toolset.expand(list(ctx.tools))
    if not granted and not ctx.skills and schema is not None:
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

    bound = ctx.toolset.resolve(granted)
    skills = {skill.name: skill for skill in ctx.skills}
    if skills:
        bound.append(load_skill_tool())
    ctx.check_stopped()
    chat = model.bind_tools(bound) if bound else model
    iterations = 0
    reply = None

    while True:
        ctx.check_stopped()
        if ctx.interrupted():
            return _interrupted(ctx, reply)
        # Steers join only here, never between a reply's tool results, and ride in
        # new_messages so the next model request records them.
        steers = [HumanMessage(content=steer) for steer in ctx.drain_steers()]
        messages.extend(steers)
        new_messages.extend(steers)
        exchange_id = uuid4().hex
        try:
            reply = invoke_model(
                ctx.memory_context.transcript,
                lambda: invoke_chat(
                    live, chat, messages, exchange_id=exchange_id,
                    interrupted=ctx.interrupted,
                ),
                new_messages=new_messages,
                check_stopped=ctx.check_stopped,
                exchange_id=exchange_id,
                check_reply=_reject_invalid_arguments,
            )
        except ReplyInterrupted as cut:
            return _interrupted(ctx, cut.reply)
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

        if iterations == ctx.max_tool_iterations:
            # No later model call could read these results, so running them would only
            # add side effects to a node that is already failing.
            raise AgentLoopError(
                f"agent node {ctx.node_id!r} hit the tool-iteration cap "
                f"({ctx.max_tool_iterations}) without a final answer; the last reply's "
                f"{len(calls)} tool call(s) were not run"
            )

        for call_index, call in enumerate(calls):
            ctx.check_stopped()
            if ctx.interrupted():
                for skipped_index, skipped in enumerate(calls[call_index:], call_index):
                    skipped_name = str(skipped.get("name") or "")
                    skipped_arguments = skipped.get("args") or {}
                    skipped_id = skipped.get("id") or ""
                    record_skipped_tool(
                        ctx.memory_context.transcript,
                        name=skipped_name,
                        arguments=skipped_arguments,
                        model_exchange_id=exchange_id,
                        call_index=skipped_index,
                        tool_call_id=skipped_id,
                    )
                    observe_skipped_tool_call(
                        live,
                        exchange_id=exchange_id,
                        call_index=skipped_index,
                        tool_call_id=skipped_id,
                        name=skipped_name,
                        arguments=skipped_arguments,
                    )
                return _interrupted(ctx, reply)
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
            elif name in granted:
                invoke = partial(ctx.toolset.invoke, name, arguments)
            else:
                raise AgentLoopError(
                    f"agent node {ctx.node_id!r} requested tool {name!r}, but it is not "
                    f"granted; granted tools: {granted}"
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


def _reject_invalid_arguments(reply: BaseMessage) -> None:
    """Raise `TypeCheckError` if a tool call's arguments nest too deep or hold a NaN or infinity."""
    for call in getattr(reply, "tool_calls", None) or []:
        where = f"tool call {call.get('name')!r} arguments"
        # Depth first: the non-finite walk recurses, so it must not see an unbounded value.
        reject_deep_nesting(call.get("args"), where)
        reject_nonfinite(call.get("args"), where)


def _interrupted(ctx: AgentRunContext, reply: BaseMessage | None) -> Output:
    """End an interrupted loop successfully with the latest reply's visible text."""
    interrupt_turn(ctx.memory_context.transcript)
    return Output(value="" if reply is None else text_of(reply))

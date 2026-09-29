"""plain mode — direct generation with no tools or tool loop.

The simplest agent: render the prompt and return unconstrained text, or use structured
generation (including its configured correction attempts) for a constrained output type.
Any authored tools are ignored because there is no tool loop.
"""

from __future__ import annotations

from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

import agent_composer.llm_clients as llm_clients
from agent_composer.nodes.agent.memory import prepare_history, invoke_model, system_prompt_with_memory
from agent_composer.nodes.agent.live import invoke_chat, live_sinks, observe_model_call
from agent_composer.nodes.agent.modes.common import (
    DEFAULT_SYSTEM,
    AgentRunContext,
    register_mode,
)
from agent_composer.nodes.agent.modes.utils import (
    record_model_text,
    text_of,
)
from agent_composer.nodes.agent.structured import generate_structured, type_to_schema
from agent_composer.nodes.base import Output


@register_mode("plain")
def plain(ctx: AgentRunContext) -> Output:
    ctx.check_stopped()
    model = llm_clients.model_from_config(ctx.llm_config or {})
    ctx.check_stopped()
    system_prompt = system_prompt_with_memory(DEFAULT_SYSTEM, ctx.memory_context.sections)
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=ctx.prompt)]
    new_messages = list(msgs)
    live = live_sinks(ctx)
    msgs = [
        msgs[0],
        *prepare_history(
            ctx.memory_context.transcript,
            model,
            check_stopped=ctx.check_stopped,
            observe_summary=lambda invoke: observe_model_call(
                live, invoke, purpose="history_summary", exchange_id=uuid4().hex
            ),
        ),
        *msgs[1:],
    ]
    if ctx.evaluation_trace is not None:
        ctx.evaluation_trace.record_prompt(
            DEFAULT_SYSTEM,
            ctx.prompt,
            ctx.memory_context.sections,
        )
    # A constrained type -> structured generation; only a bare str keeps the text path.
    schema = type_to_schema(ctx.output_type) if ctx.output_type is not None else None
    if schema is None:
        exchange_id = uuid4().hex
        reply = invoke_model(
            ctx.memory_context.transcript,
            lambda: invoke_chat(live, model, msgs, exchange_id=exchange_id),
            new_messages=new_messages,
            check_stopped=ctx.check_stopped,
            exchange_id=exchange_id,
        )
        text = text_of(reply)
        record_model_text(ctx.evaluation_trace, text)
        return Output(value=text)
    return Output(
        value=generate_structured(
            model,
            msgs,
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

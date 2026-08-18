"""plain mode — direct generation with no tools or tool loop.

The simplest agent: render the prompt and return unconstrained text, or use structured
generation (including its configured correction attempts) for a constrained output type.
Any authored tools are ignored because there is no tool loop.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

import agent_composer.llm_clients as llm_clients
from agent_composer.nodes.agent.modes.common import (
    DEFAULT_SYSTEM,
    AgentRunContext,
    register_mode,
)
from agent_composer.nodes.agent.modes.utils import record_model_text, text_of
from agent_composer.nodes.agent.structured import generate_structured, type_to_schema
from agent_composer.nodes.base import Output


@register_mode("plain")
def plain(ctx: AgentRunContext) -> Output:
    ctx.check_stopped()
    model = llm_clients.model_from_config(ctx.llm_config or {})
    ctx.check_stopped()
    msgs = [SystemMessage(content=DEFAULT_SYSTEM), HumanMessage(content=ctx.prompt)]
    if ctx.evaluation_trace is not None:
        ctx.evaluation_trace.record_prompt(DEFAULT_SYSTEM, ctx.prompt)
    # A constrained type -> structured generation; only a bare str keeps the text path.
    schema = type_to_schema(ctx.output_type) if ctx.output_type is not None else None
    if schema is None:
        reply = model.invoke(msgs)
        ctx.check_stopped()
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
        )
    )

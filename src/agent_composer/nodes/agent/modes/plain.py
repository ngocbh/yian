"""plain mode — a single LLM call. No tools, no loop.

The simplest agent: render the prompt, ask the model once, return its text. Any
skills on the node are ignored (there is no loop to invoke them). Good for pure
generation/synthesis nodes.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

import agent_composer.llm_clients as llm_clients
from agent_composer.nodes.agent.modes.common import DEFAULT_SYSTEM, AgentRunContext, register_mode
from agent_composer.nodes.agent.modes.utils import text_of
from agent_composer.nodes.agent.structured import generate_structured, type_to_schema
from agent_composer.nodes.base import Output


@register_mode("plain")
def plain(ctx: AgentRunContext) -> Output:
    model = llm_clients.model_from_config(ctx.llm_config or {})
    msgs = [SystemMessage(content=DEFAULT_SYSTEM), HumanMessage(content=ctx.prompt)]
    # A declared non-text type -> structured generation; a bare str/Literal -> text path.
    schema = type_to_schema(ctx.output_type) if ctx.output_type is not None else None
    if schema is None:
        return Output(value=text_of(model.invoke(msgs)))
    return Output(
        value=generate_structured(
            model,
            msgs,
            ctx.output_type,
            max_retries=ctx.retries,
            llm_config=ctx.llm_config,
        )
    )

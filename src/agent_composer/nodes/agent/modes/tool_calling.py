"""tool_calling mode — ask, run requested tools, repeat until answered.

Bind the node's tools, then loop: ask the model; if it requests ordinary tools,
run them via `TOOL_REGISTRY` and feed results back; if it requests a *control tool*
(e.g. `ask_user`), lower the pause to a seed-first continuation `Grow`. The compile-owned
builder constructs the human-input + resume-agent fragment and the engine splices it.

The agent loop body is the pure, self-contained `agent_step(messages, pending,
iterations, ctx) -> Output | Grow`: the re-entry frame rides as
arguments/return, never a private namespace. On a final answer it returns `Output`;
on a control call it returns the descriptor pair on `Grow.seed` for durable construction and
replay. The injected answer is delivered to
the human_input leaf as its `Output` and read by the resume_agent via the bare
`${<hi>.output}` forward-ref edge.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_to_dict,
)

from agent_composer.nodes.agent.controls import CONTROL_TOOLS
from agent_composer.nodes.agent.modes.common import (
    DEFAULT_SYSTEM,
    AgentLoopError,
    AgentRunContext,
    register_mode,
)
from agent_composer.nodes.agent.modes.utils import text_of
from agent_composer.nodes.base import Grow, Output

# Default tool-iteration policy: -1 means NO CAP — the loop runs until the model returns a
# final answer. A node bounds it per-flow with a POSITIVE `env: {max_tool_iterations: N}`
# (threaded as ctx.max_tool_iterations), which raises AgentLoopError after N turns. -1 is the
# sentinel for unbounded; 0 and other negatives are rejected as invalid caps.
MAX_TOOL_ITERATIONS = -1


def _slug_call_id(call_id: str) -> str:
    """Path-safe slug for a tool-call id embedded in a node id / `${...}` ref.
    The reference-path grammar allows `_ # /` but not `-`; provider call ids may be
    uuids (Ollama) containing dashes, which break ref parsing on resume. Map every
    non-[A-Za-z0-9_] char to `_`. The REAL id is kept verbatim in `pending["call_id"]`
    and the human-input `slot` for the ToolMessage / resume-id match."""
    return re.sub(r"[^A-Za-z0-9_]", "_", call_id)


def run_tool(name: str, args: dict[str, Any]) -> str:
    """Execute one registered ordinary tool (same `TOOL_REGISTRY` path as a TOOL node).
    Errors come back as text for the model to see rather than crashing the node."""
    from agent_composer.tools import TOOL_REGISTRY

    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return f"Tool {name!r} is not available to this node."
    try:
        return str(tool.invoke(args))
    except Exception as exc:  # noqa: BLE001
        return f"Tool {name!r} failed: {exc}"


def agent_step(
    messages: list[BaseMessage],
    pending: Optional[dict],
    iterations: int,
    ctx: AgentRunContext,
):
    """Run ONE segment of the agent loop — the re-entry frame rides as args/return.

    On ENTRY this ALWAYS invokes the model on the passed-in `messages` — there is no
    resume-replay branch (the answer for a `pending` control call is appended as a
    `ToolMessage` by the agent's `Resume` entry — `AgentNode.run` — *before* this is called,
    so `pending` is `None` here). On a final answer -> `Output`. On a control call -> a seed-first
    `Grow` carrying a `human_input` descriptor + a `resume_agent` descriptor with the re-entry
    frame as DATA (memo / iterations / config-as-data / pending). The compile-owned builder turns
    that pair into the continuation fragment, whose resume node reads `answer` via the BARE
    forward-ref `${<hi>.output}` (node-first). Data-tool calls in the turn are flushed into
    `messages` before the `Grow`.
    """
    from agent_composer.tools import resolve_tools

    control_set = set(ctx.controls)
    bound = resolve_tools(list(ctx.tools)) + [CONTROL_TOOLS[n].tool for n in ctx.controls]
    chat = ctx.model.bind_tools(bound) if bound else ctx.model

    # A negative cap (the -1 default) means NO bound — loop until a final answer; a positive
    # cap bounds the turns and trips AgentLoopError below.
    while ctx.max_tool_iterations < 0 or iterations < ctx.max_tool_iterations:
        reply = chat.invoke(messages)
        iterations += 1
        messages.append(reply)
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            # Final answer. A declared non-text `output:` forces a structured emit turn
            # (native or prompt-injection, with capped self-correction); a bare-str/Literal
            # type keeps the text path. The structured value is enforced at the engine write
            # boundary on BOTH the primary path (`pool.set(..., declared=output_type)`) and a
            # resumed agent's alias-filler path, so either re-entry validates it.
            from agent_composer.nodes.agent.structured import generate_structured, type_to_schema

            if ctx.output_type is not None and type_to_schema(ctx.output_type) is not None:
                return Output(
                    value=generate_structured(
                        ctx.model,
                        messages,
                        ctx.output_type,
                        max_retries=ctx.retries,
                        llm_config=ctx.llm_config,
                    )
                )
            return Output(value=text_of(reply))

        # Run all data-tool calls first so every call in the turn gets answered.
        for call in (c for c in calls if c.get("name") not in control_set):
            messages.append(
                ToolMessage(
                    content=run_tool(call.get("name") or "", call.get("args") or {}),
                    tool_call_id=call.get("id") or "",
                )
            )

        control_calls = [c for c in calls if c.get("name") in control_set]
        if control_calls:
            if len(control_calls) > 1:
                raise AgentLoopError(
                    f"agent node {ctx.node_id!r}: multiple control-tool calls in one turn "
                    f"are not supported"
                )
            call = control_calls[0]
            call_id = call.get("id") or ""
            slug = _slug_call_id(call_id)
            pending = {
                "name": call["name"],
                "call_id": call_id,  # REAL id: matched to the ToolMessage tool_call_id on resume
                "args": call.get("args") or {},
            }
            hi_id = f"__ask#{slug}"  # slug keeps the node id / answer ref path-safe
            human_input = {
                "kind": "human_input",
                "node_id": hi_id,
                "prompt": str(call.get("args", {}).get("question", "")),
                "slot": call_id,  # REAL id: mints the resume-node id, never parsed as a ref path
            }
            resume = {
                "kind": "resume_agent",
                "memo": messages_to_dict(messages),
                "iterations": iterations,
                "pending": pending,
                "answer": f"${{{hi_id}.output}}",  # node-first ref
                "llm_config": ctx.llm_config,
                "tools": list(ctx.tools),
                "controls": list(ctx.controls),
                "mode": "tool_calling",
            }
            return Grow(seed={"hi_desc": human_input, "resume_desc": resume})

    raise AgentLoopError(
        f"agent node {ctx.node_id!r} hit the tool-iteration cap "
        f"({ctx.max_tool_iterations}) without a final answer"
    )


@register_mode("tool_calling")
def tool_calling(ctx: AgentRunContext):
    messages: list[BaseMessage] = [
        SystemMessage(content=DEFAULT_SYSTEM),
        HumanMessage(content=ctx.prompt),
    ]
    return agent_step(messages, None, 0, ctx)

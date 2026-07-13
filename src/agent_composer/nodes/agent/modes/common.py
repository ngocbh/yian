"""Shared definitions for agent modes — the mode contract.

Common (shared across the mode modules), kept out of `modes/__init__.py` so the
concrete modes import from here and `__init__` only re-exports + registers — no
`__init__` <-> submodule cycle. Pure helper functions live in `utils.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent_composer.nodes.base import NodeResult

DEFAULT_SYSTEM = (
    "You are an analysis agent inside a larger workflow. Follow the instructions "
    "precisely, use the available tools to gather any facts you need, and return "
    "a clear, self-contained answer. Do not ask the user questions."
)


class AgentLoopError(RuntimeError):
    """A mode could not produce a final answer (e.g. hit an iteration cap)."""


@dataclass
class AgentRunContext:
    """What a mode receives to run one AGENT node."""

    node_id: str
    prompt: str  # already rendered (${...} substituted)
    tools: list[str] = field(default_factory=list)  # ordinary tools
    controls: list[str] = field(default_factory=list)  # control tools (e.g. ask_user)
    model: Any = None  # a ready langchain chat model
    llm_config: dict | None = None  # plain dict; a continuation carries it forward
    output_type: Any = None  # the node's declared output Type; None = text passthrough
    retries: int = 2  # structured-generation self-correction cap (extra attempts)
    max_tool_iterations: int = -1  # model turns before AgentLoopError; -1 = no cap (default)
    # The node's node-level POST `asserts:` reading `${output}` and their display texts. A pause
    # defers the agent's committed output to the resume terminal, so these ride that terminal (via
    # `agent_segment_subgraph`) and fire through its `run` wrapper — see `clone_continuation_pair`.
    post_asserts: list[str] = field(default_factory=list)
    assert_texts: dict = field(default_factory=dict)


# A mode: a pure function of its context returning a NodeResult (Output | Grow) — the
# agent pauses via the continuation `Grow`, never a direct `Pause`.
AgentMode = Callable[[AgentRunContext], NodeResult]

MODES: dict[str, AgentMode] = {}


def register_mode(name: str) -> Callable[[AgentMode], AgentMode]:
    """Register an agent mode under `name` (referenced by AgentBody.mode)."""

    def deco(fn: AgentMode) -> AgentMode:
        MODES[name] = fn
        return fn

    return deco

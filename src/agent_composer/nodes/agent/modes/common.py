"""Shared definitions for agent modes — the mode contract.

Common (shared across the mode modules), kept out of `modes/__init__.py` so the
concrete modes import from here and `__init__` only re-exports + registers — no
`__init__` <-> submodule cycle. Shared mode helpers live in `utils.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable

from agent_composer.memory.contracts import EvaluationTrace, MemorySection
from agent_composer.nodes.base import Output, _HostStop

DEFAULT_SYSTEM = (
    "You are an analysis agent inside a larger workflow. Follow the instructions "
    "precisely, use the available tools to gather any facts you need, and return "
    "a clear, self-contained answer. Ask the user only through an explicitly granted tool."
)


class AgentLoopError(RuntimeError):
    """A mode could not produce a final answer (e.g. hit an iteration cap)."""


@dataclass
class AgentRunContext:
    """What a mode receives to run one AGENT node."""

    node_id: str
    prompt: str  # already rendered (${...} substituted)
    tools: list[str] = field(default_factory=list)  # ordinary tools
    llm_config: dict | None = None  # effective plain-dict model config
    output_type: Any = None  # the node's declared output Type; None = text passthrough
    retries: int = 2  # structured-generation self-correction cap (extra attempts)
    max_tool_iterations: int = -1  # model turns before AgentLoopError; -1 = no cap (default)
    stop_event: Event | None = None  # transient host lifecycle; absent for direct mode tests
    evaluation_trace: EvaluationTrace | None = None  # write-only visible evidence sink
    memory_sections: tuple[MemorySection, ...] = ()  # ordered outer-flow to local-agent recall

    def check_stopped(self) -> None:
        """Unwind before starting another external effect after host cancellation."""
        if self.stop_event is not None and self.stop_event.is_set():
            raise _HostStop(primary=False)


# A mode is a pure node-body function of its process-local context.
AgentMode = Callable[[AgentRunContext], Output]

MODES: dict[str, AgentMode] = {}


def register_mode(name: str) -> Callable[[AgentMode], AgentMode]:
    """Register an agent mode under `name` (referenced by AgentBody.mode)."""

    def deco(fn: AgentMode) -> AgentMode:
        MODES[name] = fn
        return fn

    return deco

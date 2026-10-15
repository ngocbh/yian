"""Process-local context for one AGENT invocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable

from yian.defaults import current
from yian.memory.contracts import EvaluationTrace, MemoryContext
from yian.nodes.base import Steering, _HostStop
from yian.skills.catalog import Skill
from yian.tools import ToolSet

DEFAULT_SYSTEM = (
    "You are an analysis agent inside a larger workflow. Follow the instructions "
    "precisely, use the available tools to gather any facts you need, and return "
    "a clear, self-contained answer. Ask the user only through an explicitly granted tool."
)


class AgentLoopError(RuntimeError):
    """The AGENT tool loop could not produce a final answer.

    For example, it hit its iteration cap.
    """


@dataclass
class AgentRunContext:
    """What the AGENT loop receives to run one invocation."""

    node_id: str
    prompt: str  # already rendered (${...} substituted)
    tools: list[str] = field(default_factory=list)  # granted tool and group ids
    llm_config: dict | None = None  # effective plain-dict model config
    output_type: Any = None  # the node's declared output Type; None = text passthrough
    # structured-generation self-correction cap (extra attempts)
    retries: int = field(default_factory=lambda: current().agent_retries)
    # model turns before AgentLoopError; -1 = no cap
    max_tool_iterations: int = field(default_factory=lambda: current().agent_max_tool_iterations)
    stop_event: Event | None = None  # transient host lifecycle; absent for direct loop tests
    evaluation_trace: EvaluationTrace | None = None  # write-only visible evidence sink
    memory_context: MemoryContext = field(default_factory=MemoryContext)
    toolset: ToolSet = field(default_factory=ToolSet)  # the run's tools and groups; `tools` grants from it
    emit_stream: Callable[[Any], None] | None = None  # transient text sink; absent = no streaming
    emit_event: Callable[[Any], None] | None = None  # transient in-loop observer sink
    base_system_prompt: str = DEFAULT_SYSTEM  # authored or default system text plus skills; memory follows
    skills: tuple[Skill, ...] = ()  # granted skills served by `load_skill`, sorted by name
    steering: Steering | None = None  # host inbox; present only for a steerable node

    def check_stopped(self) -> None:
        """Unwind before starting another external effect after host cancellation."""
        if self.stop_event is not None and self.stop_event.is_set():
            raise _HostStop(primary=False)

    def drain_steers(self) -> list[str]:
        """Take the host's queued steer messages for this node; none without an inbox."""
        return [] if self.steering is None else self.steering.drain(self.node_id)

    def interrupted(self) -> bool:
        """Whether the host asked this node to stop its loop and return its text so far."""
        return self.steering is not None and self.steering.interrupted(self.node_id)


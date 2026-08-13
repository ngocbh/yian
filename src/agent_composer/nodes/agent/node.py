"""AGENT node — one authored node running one local model/tool loop.

The selected mode owns the conversation for the duration of ``run``. Tools,
including the host-installed ``ask_user``, are ordinary explicitly authored
tool grants. A crash or host interruption unwinds the node; no conversation or
continuation state is stored in the graph or variable pool.
"""

from __future__ import annotations

from typing import Any, Optional

from agent_composer.expr.template import render_template_record
from agent_composer.nodes.agent.modes import MODES, AgentRunContext
from agent_composer.nodes.agent.modes.tool_calling import MAX_TOOL_ITERATIONS
from agent_composer.nodes.base import Node, NodeKind, Output

DEFAULT_MODE = "plain"


class AgentNode(Node):
    """An LLM-backed leaf configured by a prompt, mode, and authored tools.

    ``llm_config`` is normalized by ``model_from_config`` when the node runs.
    The package attribute is resolved at call time so tests and hosts may
    monkeypatch the model factory without an engine-owned capability.
    """

    kind = NodeKind.AGENT
    needs_stop_event = True

    def __init__(
        self,
        node_id: str,
        *,
        prompt: str = "",
        tools: Optional[list[str]] = None,
        llm_config: Optional[dict[str, Any]] = None,
        llm_inherit: bool = True,
        mode: str = DEFAULT_MODE,
        retries: int = 2,
        title: Optional[str] = None,
    ) -> None:
        super().__init__(node_id, title=title)
        self.prompt = prompt or ""
        self.tools = tools or []
        # own_llm_config is the authored source re-resolved by each cascade pass;
        # llm_config is the effective config baked for this run.
        self.own_llm_config: dict[str, Any] = dict(llm_config) if llm_config else {}
        self.llm_config = llm_config
        self.llm_inherit = llm_inherit
        self.mode = mode or DEFAULT_MODE
        self.retries = retries
        if self.mode not in MODES:
            raise ValueError(
                f"AGENT node {node_id!r}: unknown mode {self.mode!r}; known: {sorted(MODES)}"
            )

    def _max_tool_iterations(self) -> int:
        """Return the positive authored loop bound or ``-1`` for no bound."""
        raw = self.env.get("max_tool_iterations", MAX_TOOL_ITERATIONS)
        if isinstance(raw, bool) or not isinstance(raw, int) or (raw < 1 and raw != -1):
            raise ValueError(
                f"AGENT node {self.id!r}: env max_tool_iterations must be a positive int "
                f"or -1 (no cap), got {raw!r}"
            )
        return raw

    def _ctx(self, prompt: str, *, stop_event=None) -> AgentRunContext:
        """Build the complete process-local context for one mode invocation."""
        return AgentRunContext(
            node_id=self.id,
            prompt=prompt,
            tools=list(self.tools),
            llm_config=self.llm_config,
            output_type=self.output_type,
            retries=self.retries,
            max_tool_iterations=self._max_tool_iterations(),
            stop_event=stop_event,
        )

    def _run_impl(self, inputs: dict, *, stop_event=None) -> Output:
        """Render declared inputs and run the selected local mode to completion."""
        prompt = render_template_record(self.prompt, inputs)
        return MODES[self.mode](self._ctx(prompt, stop_event=stop_event))

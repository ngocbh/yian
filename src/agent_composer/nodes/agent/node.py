"""AGENT node — one authored node running one local model/tool loop.

The selected mode owns the conversation for the duration of ``run``. Tools,
including the host-installed ``ask_user``, are ordinary explicitly authored
tool grants. A crash or host interruption unwinds the node; no conversation or
continuation state is stored in the graph or variable pool.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

from agent_composer.diagnostics import OriginRef
from agent_composer.expr.template import render_template_record
from agent_composer.memory.contracts import (
    EvaluationTrace,
    MemoryPolicySpec,
    _iter_memory_policy_origin,
    _rewrite_memory_policy_origin,
)
from agent_composer.nodes.agent.modes import MODES, AgentRunContext
from agent_composer.nodes.agent.modes.tool_calling import MAX_TOOL_ITERATIONS
from agent_composer.nodes.agent.structured import _structured_validation_path
from agent_composer.nodes.base import Node, NodeKind, Output
from agent_composer.nodes.contracts import ValueContract

DEFAULT_MODE = "plain"


class AgentNode(Node):
    """An LLM-backed leaf configured by a prompt, mode, and authored tools.

    ``llm_config`` is normalized by ``model_from_config`` when the node runs.
    The package attribute is resolved at call time so tests and hosts may
    monkeypatch the model factory without an engine-owned capability. An optional
    ``memory_policy`` is authored identity only until compile derives runtime scope.
    """

    kind = NodeKind.AGENT
    needs_stop_event = True
    needs_evaluation_trace = True

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
        memory_policy: MemoryPolicySpec | None = None,
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
        if memory_policy is not None and type(memory_policy) is not MemoryPolicySpec:
            raise TypeError(
                "memory_policy must be an exact MemoryPolicySpec or None"
            )
        self.memory_policy = memory_policy
        if self.mode not in MODES:
            raise ValueError(
                f"AGENT node {node_id!r}: unknown mode {self.mode!r}; known: {sorted(MODES)}"
            )

    def iter_local_origins(
        self,
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> Iterator[OriginRef]:
        """Yield common AGENT origins followed by its optional policy site."""
        seen = _seen_carriers if _seen_carriers is not None else set()
        yield from super().iter_local_origins(_seen_carriers=seen)
        if self.memory_policy is not None:
            yield from _iter_memory_policy_origin(self.memory_policy, seen)

    def rewrite_local_origins(
        self,
        mapper: Callable[[OriginRef], OriginRef],
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> None:
        """Rewrite common AGENT origins and its policy while retaining aliases."""
        seen = _seen_carriers if _seen_carriers is not None else set()
        super().rewrite_local_origins(mapper, _seen_carriers=seen)
        if self.memory_policy is not None:
            _rewrite_memory_policy_origin(self.memory_policy, mapper, seen)

    def _max_tool_iterations(self) -> int:
        """Return the positive authored loop bound or ``-1`` for no bound."""
        raw = self.env.get("max_tool_iterations", MAX_TOOL_ITERATIONS)
        if isinstance(raw, bool) or not isinstance(raw, int) or (raw < 1 and raw != -1):
            raise ValueError(
                f"AGENT node {self.id!r}: env max_tool_iterations must be a positive int "
                f"or -1 (no cap), got {raw!r}"
            )
        return raw

    def _ctx(
        self,
        prompt: str,
        *,
        stop_event=None,
        evaluation_trace: EvaluationTrace | None = None,
    ) -> AgentRunContext:
        """Build the complete process-local context for one mode invocation."""
        output_type = (
            self.output_contract.type
            if type(self.output_contract) is ValueContract
            else None
        )
        return AgentRunContext(
            node_id=self.id,
            prompt=prompt,
            tools=list(self.tools),
            llm_config=self.llm_config,
            output_type=output_type,
            retries=self.retries,
            max_tool_iterations=self._max_tool_iterations(),
            stop_event=stop_event,
            evaluation_trace=evaluation_trace,
        )

    def exception_origin(self, exception: Exception) -> OriginRef | None:
        """Select a generated structured-output contract origin when proven.

        Unrelated Pydantic errors, non-validation failures, and contracts without
        authored provenance retain the common node-origin fallback.
        """
        contract = self.output_contract
        if type(contract) is not ValueContract:
            return super().exception_origin(exception)
        path = _structured_validation_path(exception, contract.type)
        if path is None:
            return super().exception_origin(exception)
        origin = contract.origins.at(path)
        return origin if origin is not None else super().exception_origin(exception)

    def forward(
        self,
        inputs: dict,
        *,
        stop_event=None,
        evaluation_trace: EvaluationTrace | None = None,
    ) -> Output:
        """Render declared inputs and run the selected local mode to completion."""
        prompt = render_template_record(self.prompt, inputs)
        return MODES[self.mode](
            self._ctx(
                prompt,
                stop_event=stop_event,
                evaluation_trace=evaluation_trace,
            )
        )
